import argparse
import configparser

import h5py
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn

from os.path import join, isfile, exists
from os import makedirs
from models.models_generic import get_backbone, get_model, create_image_clusters
from shutil import copyfile
from dataset.mapillary_sls.MSLS import MSLS
from tools import ROOT_DIR
from tqdm import trange
from time import sleep
from training.train_epoch import train_epoch
from model_validation.validation import validation
from datetime import datetime
from tools.common import save_checkpoint
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='移动机器人位置识别模型')

    parser.add_argument('--dataset_root_dir', type=str, default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='数据集的根目录')
    parser.add_argument('--config_path', type=str, default=join(ROOT_DIR, 'configs'), help='模型训练的配置文件的目录')
    parser.add_argument('--save_checkpoint_path', type=str, default=join(ROOT_DIR, 'desired', 'checkpoint'),
                        help='模型checkpoint的保存目录')
    parser.add_argument('--no_cuda', action='store_true', help='如果使用该参数表示只使用CPU，否则使用GPU')
    parser.add_argument('--resume_file', type=str, help='checkpoint文件的保存路径，用于从checkpoint载入训练参数，再次恢复训练')
    parser.add_argument('--cluster_file', type=str, help='聚类数据的保存路径，恢复训练')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='手动设置迭代开始位置，用于重新开始的训练')
    parser.add_argument('--epochs_count', default=30, type=int, help='模型训练的周期数')
    parser.add_argument('--save_every_epoch', action='store_true', help='是否开启每个EPOCH都进行checkpoint保存')

    opt = parser.parse_args()

    # 配置文件的地址
    config_file = join(opt.config_path, 'train.ini')
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read(config_file)

    cuda = not opt.no_cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("没有找到GPU，运行时添加参数 --no_cuda")

    device = torch.device("cuda" if cuda else "cpu")

    # 固定随机种子
    random.seed(config['train'].getint('seed'))
    np.random.seed(config['train'].getint('seed'))
    torch.manual_seed(config['train'].getint('seed'))
    if cuda:
        torch.cuda.manual_seed(config['train'].getint('seed'))

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')

    print('===> 构建网络模型')

    print('===> 构建基础BackBone模型')
    encoding_model, encoding_dim = get_backbone(config)

    if opt.resume_file:
        opt.resume_file = join(join(ROOT_DIR, 'desired', 'checkpoint'), opt.resume_file)

        if isfile(opt.resume_file):
            print('===> 载入checkpoint "{}"中...'.format(opt.resume_file))
            checkpoint = torch.load(opt.resume_file, map_location=lambda storage, loc: storage)
            # 保存载入的聚类中心点的
            config[dataset_name]['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

            model = get_model(encoding_model, encoding_dim, config,
                              append_pca_layer=config['train'].getboolean('wpca'))

            # 载入模型参数
            model.load_state_dict(checkpoint['state_dict'])
            # 设置迭代开始位置
            opt.start_epoch = checkpoint['epoch']

            print('===> 载入checkpoint "{}"完毕'.format(opt.resume_file))
        else:
            raise FileNotFoundError("=> 在'{}'中没有找到checkpoint文件".format(opt.resume_file))
    else:
        print('===> 载入模型')

        model = get_model(encoding_model, encoding_dim, config,
                          append_pca_layer=config['train'].getboolean('wpca'))

        # 保存的图像特征
        init_cache_file = join(join(ROOT_DIR, 'desired', 'centroids'),
                               config['model'].get('backbone') + '_' +
                               dataset_name + '_' +
                               str(config[dataset_name].getint('num_clusters')) + '_desc_cen.hdf5')

        if opt.cluster_file:
            opt.cluster_file = join(join(ROOT_DIR, 'desired', 'centroids'), opt.cluster_file)

            if isfile(opt.cluster_file):
                if opt.cluster_file != init_cache_file:
                    copyfile(opt.cluster_file, init_cache_file)
            else:
                raise FileNotFoundError("=> 在'{}'中没有找到聚类数据".format(opt.cluster_file))
        else:
            print('===> 寻找聚类中心点')

            print('===> 载入聚类数据集')
            train_dataset = MSLS(opt.dataset_root_dir, device=device, config=config, mode='test', cities_list='train',
                                 img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                                 batch_size=config['train'].getint('cache_batch_size'))

            print('===> 聚类数据集中的数据数量为: {}'.format(len(train_dataset.db_images_key)))

            model = model.to(device)

            print('===> 计算图像特征并创建聚类文件')
            create_image_clusters(train_dataset, model, encoding_dim, device, config, init_cache_file)

            # 把模型转为CPU模式，用于载入参数
            model = model.to(device='cpu')

        # 打开保存的聚类文件
        with h5py.File(init_cache_file, mode='r') as h5:
            # 获取图像聚类信息
            image_clusters = h5.get('centroids')[:]
            # 获取图像特征信息
            image_descriptors = h5.get('descriptors')[:]

            # 初始化模型参数
            model.pool.init_params(image_clusters, image_descriptors)

            del image_clusters, image_descriptors

            # 回头GPU内存
            torch.cuda.empty_cache()

    if config['train'].get('optim') == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad, model.parameters()),
                               lr=config['train'].getfloat('lr'))
    else:
        raise ValueError('未知的优化器: ' + config['train'].get('optim'))

    # 使用三元损失函数，并使用欧氏距离作为距离函数
    criterion = nn.TripletMarginLoss(margin=config['train'].getfloat('margin') ** 0.5,
                                     p=2, reduction='sum').to(device)

    model = model.to(device)

    # 载入优化器的参数
    if opt.resume_file:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('===> 载入训练和验证数据集')

    train_dataset = MSLS(opt.dataset_root_dir, mode='train', device=device, config=config, cities_list='trondheim',
                         img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                         negative_size=config['train'].getint('negative_size'),
                         batch_size=config['train'].getint('cache_batch_size'),
                         exclude_panos=config['train'].getboolean('exclude_panos'))

    validation_dataset = MSLS(opt.dataset_root_dir, mode='val', device=device, config=config, cities_list='cph',
                              img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                              positive_distance_threshold=config['train'].getint('positive_distance_threshold'),
                              batch_size=config['train'].getint('cache_batch_size'),
                              exclude_panos=config['train'].getboolean('exclude_panos'))

    print('===> 训练集中Query的数量为: {}'.format(len(train_dataset.q_seq_idx)))
    print('===> 验证集中Query的数量为: {}'.format(len(validation_dataset.q_seq_idx)))

    print('===> 开始训练...')

    # 保存训练参数的路径
    opt.resume_dir = join(ROOT_DIR, 'desired', 'checkpoint')
    # 如果目录不存在就创建目录
    if not exists(opt.resume_dir):
        makedirs(opt.resume_dir)

    # 保存可视化结果的路径
    opt.result_dir = join(ROOT_DIR, 'result',
                          '{}_{}_{}'.format(config['model'].get('backbone'), dataset_name,
                                            config[dataset_name].get('num_clusters')),
                          datetime.now().strftime('%Y_%m_%d'))
    if not exists(opt.result_dir):
        makedirs(opt.result_dir)

    # 创建TensorBoard的写入对象
    writer = SummaryWriter(log_dir=join(opt.result_dir, datetime.now().strftime('%H:%M:%S')))

    # 保存训练结果没有改善的次数
    not_improved = 0
    # 最好结果的分数
    best_score = 0
    # 如果有保存的参数，那么从参数中读取not_improved和best_score
    if opt.resume_file:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    # 开始训练，从opt.start_epoch + 1次开始，到opt.epochs_count次结束
    train_epoch_bar = trange(opt.start_epoch + 1, opt.epochs_count + 1)
    for epoch in train_epoch_bar:
        train_epoch_bar.set_description('第{}/{}次训练周期'.format(epoch, opt.epochs_count + 1 - opt.start_epoch + 1))

        # 执行一个训练周期
        train_epoch(train_dataset, model, optimizer, criterion, encoding_dim, device, epoch, opt, config, writer)

        # 每训练eval_every次，进行一次验证
        if(epoch % config['train'].getint('eval_every')) == 0:
            recalls = validation(validation_dataset, model, encoding_dim, device, opt, config, epoch, writer)

            # 如果在验证集上结果大于保存的结果，那么就把not_improved清零，并且保存当前最好结果，否not_improved加1
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1

            # 保存模型参数
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
            }, opt, is_best)

            # 假如patience大于0，且not_improved大于分配到每个验证周期的patience次数，那么就认为模型已经无法再改进了
            if config['train'].getint('patience') > 0 and \
                    not_improved > (config['train'].getint('patience') / config['train'].getint('eval_every')):
                print('经过{}个训练周期，模型的性能已经不能再改进，停止训练...'.format(config['train'].getint('patience')))
                break

    # 显示最好的结果
    print('===> 最好的结果 Recalls@5: {:.4f}'.format(best_score), flush=True)
    writer.close()

    # 清空CUDA缓存
    torch.cuda.empty_cache()

    print('训练结束...')
