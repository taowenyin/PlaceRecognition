import torch
import numpy as np
import argparse
import configparser
import h5py

from dataset.mapillary_sls.MSLS import ImagesFromList, MSLS
from torch.nn import Module
from configparser import ConfigParser
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from os.path import join, exists
from os import makedirs
from tools import ROOT_DIR
from datetime import datetime


def validation(validation_dataset: MSLS, model: Module, encoding_dim, device,
               opt, config: ConfigParser, epoch, writer: SummaryWriter):
    """
    模型的验证

    :param validation_dataset: 验证的数据集
    :param model: 验证的模型
    :param encoding_dim: Backbone模型的输出维度
    :param device: 训练使用的设备
    :param opt: 传入的参数
    :param config: 训练的配置参数
    :param epoch: 当前所在的训练周期数
    :param writer: Tensorboard的写入对象
    """

    eval_set_queries = ImagesFromList(validation_dataset.q_images_key, validation_dataset.img_transform)
    eval_set_dbs = ImagesFromList(validation_dataset.db_images_key, transform=validation_dataset.img_transform)

    eval_data_loader_queries = DataLoader(dataset=eval_set_queries, batch_size=config['train'].getint('batch_size'))
    eval_data_loader_dbs = DataLoader(dataset=eval_set_dbs, batch_size=config['train'].getint('batch_size'))

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')
    batch_size = config['train'].getint('batch_size')

    model.eval()

    with torch.no_grad():
        print('===> 提验证集取图像特征中...')

        if config['train']['pooling'].lower() == 'netvlad' or config['train']['pooling'].lower() == 'patchnetvlad':
            pooling_dim = encoding_dim * config[dataset_name].getint('num_clusters')
        else:
            pooling_dim = encoding_dim

        q_feature = torch.zeros(len(eval_set_queries), pooling_dim).to(device)
        db_feature = torch.zeros(len(eval_set_dbs), pooling_dim).to(device)

        # 获取验证集Query的VLAD特征
        eval_q_data_bar = tqdm(enumerate(eval_data_loader_queries),
                               leave=True, total=len(eval_set_queries) // batch_size)
        for i, (data, idx) in eval_q_data_bar:
            eval_q_data_bar.set_description('[{}/{}]计算验证集Query的特征...'.format(i, eval_q_data_bar.total))
            image_descriptors = model.encoder(data.to(device))
            vlad_descriptors = model.pool(image_descriptors)
            # 如果是PatchNetVLAD那么只是用Global VLAD
            if config['train'].get('pooling') == 'patchnetvlad':
                vlad_descriptors = vlad_descriptors[1]
            q_feature[i * batch_size: (i + 1) * batch_size, :] = vlad_descriptors

        # 获取验证集Query的VLAD特征
        eval_db_data_bar = tqdm(enumerate(eval_data_loader_dbs), leave=True, total=len(eval_set_dbs) // batch_size)
        for i, (data, idx) in eval_db_data_bar:
            eval_db_data_bar.set_description('[{}/{}]计算验证集Database的特征...'.format(i, eval_db_data_bar.total))
            image_descriptors = model.encoder(data.to(device))
            vlad_descriptors = model.pool(image_descriptors)
            # 如果是PatchNetVLAD那么只是用Global VLAD
            if config['train'].get('pooling') == 'patchnetvlad':
                vlad_descriptors = vlad_descriptors[1]
            db_feature[i * batch_size: (i + 1) * batch_size, :] = vlad_descriptors

    del eval_data_loader_queries, eval_data_loader_dbs

    print('===> 构建验证集的最近邻')

    # 对Database进行拟合
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(db_feature.cpu().numpy())

    print('====> 计算召回率 @ N')
    n_values = [1, 5, 10, 20, 50, 100]

    # 计算最近邻
    predictions = np.square(knn.kneighbors(q_feature.cpu().numpy(), max(n_values))[1])

    # 得到所有正例
    gt = validation_dataset.all_positive_indices

    correct_at_n = np.zeros(len(n_values))

    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_n[i:] += 1
                break

    recall_at_n = correct_at_n / len(validation_dataset.q_seq_idx)

    # 保存所有召回率
    all_recalls = {}
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        print("====> 召回率@{}: {:.4f}".format(n, recall_at_n[i]))
        writer.add_scalar('验证集的召回率@{}'.format(str(n)), recall_at_n[i], epoch)

    return all_recalls


if __name__ == '__main__':
    from models.models_generic import get_backbone, get_model

    parser = argparse.ArgumentParser(description='Validation')

    parser.add_argument('--dataset_root_dir', type=str, default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')
    parser.add_argument('--config_path', type=str, default=join(ROOT_DIR, 'configs'), help='模型训练的配置文件的目录。')
    parser.add_argument('--no_cuda', action='store_true', help='如果使用该参数表示只使用CPU，否则使用GPU。')

    opt = parser.parse_args()

    config_file = join(opt.config_path, 'train.ini')
    config = configparser.ConfigParser()
    config.read(config_file)

    dataset_name = config['dataset'].get('name')

    cuda = not opt.no_cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("没有找到GPU，运行时添加参数 --no_cuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoding_model, encoding_dim = get_backbone(config)
    model = get_model(encoding_model, encoding_dim, config,
                      append_pca_layer=config['train'].getboolean('wpca'))

    init_cache_file = join(join(ROOT_DIR, 'desired', 'centroids'),
                           config['model'].get('backbone') + '_' +
                           dataset_name + '_' +
                           str(config[dataset_name].getint('num_clusters')) + '_desc_cen.hdf5')
    # 打开保存的聚类文件
    with h5py.File(init_cache_file, mode='r') as h5:
        # 获取图像聚类信息
        image_clusters = h5.get('centroids')[:]
        # 获取图像特征信息
        image_descriptors = h5.get('descriptors')[:]

        # 初始化模型参数
        model.pool.init_params(image_clusters, image_descriptors)

        del image_clusters, image_descriptors

    # 保存可视化结果的路径
    opt.result_dir = join(ROOT_DIR, 'result',
                          '{}_{}_{}'.format(config['model'].get('backbone'), dataset_name,
                                            config[dataset_name].get('num_clusters')),
                          datetime.now().strftime('%Y_%m_%d'))
    if not exists(opt.result_dir):
        makedirs(opt.result_dir)

    # 创建TensorBoard的写入对象
    writer = SummaryWriter(log_dir=join(opt.result_dir, datetime.now().strftime('%H:%M:%S')))

    validation_dataset = MSLS(opt.dataset_root_dir, mode='val', device=device, config=config, cities_list='cph',
                              img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                              positive_distance_threshold=config['train'].getint('positive_distance_threshold'),
                              batch_size=config['train'].getint('batch_size'),
                              exclude_panos=config['train'].getboolean('exclude_panos'))

    model = model.to(device)

    all_recalls = validation(validation_dataset, model, encoding_dim, device, opt, config, 0, writer)

    print('所有的召回率')
    print(all_recalls)
