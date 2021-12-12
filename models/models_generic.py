import torch
import torch.nn as nn
import numpy as np
import h5py
import torch.nn.functional as F

from torchvision.models import vgg16
from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset.mapillary_sls.MSLS import ImagesFromList, MSLS
from os.path import join, exists
from os import makedirs, remove
from tools import ROOT_DIR
from tqdm import tqdm
from sklearn.cluster import KMeans


def get_backbone():
    """
    获取BackBone的模型

    :return:
    encoding_model: BackBone的模型
    encoding_dim: BackBone的模型输出维度
    """
    # 模型的输出维度
    encoding_dim = 512
    # 图像编码模型为VGG-16，并且采用ImageNet的预训练参数
    encoding_model = vgg16(pretrained=True)

    # 获取所有的网络层
    layers = list(encoding_model.children())[:-2]
    # 只训练conv5_1, conv5_2, and conv5_3的参数，冻结前面所有曾的参数
    for layer in layers[:-5]:
        for p in layer.parameters():
            p.requires_grad = False

    # 重新构建BackBone模型
    encoding_model = nn.Sequential(*layers)

    return encoding_model, encoding_dim


def get_model(encoding_model, encoding_dim, append_pca_layer=True):
    """
    获取训练模型

    :param encoding_model: BackBone的模型
    :param encoding_dim: BackBone的模型输出维度
    :param append_pca_layer: 是否添加PCA层
    :return:
    """
    nn_model = nn.Module()
    nn_model.add_module('encoder', encoding_model)

    return nn_model


def get_clusters(cluster_set, model, encoding_dim, device, config):
    # 获取图像Resize大小
    resize = tuple(map(int, str.split(config['train'].get('resize'), ',')))

    # 一共要保存的图像特征数
    descriptors_size = 50000
    # 每个图像采集不同位置的特征数
    per_image_sample_count = 100
    # 向上取整后，计算一共要采集多少数据
    image_sample_count = ceil(descriptors_size / per_image_sample_count)

    # 聚类采样的索引
    cluster_sampler = SubsetRandomSampler(np.random.choice(len(cluster_set.db_images_key),
                                                           image_sample_count, replace=False))

    # 创建聚类数据集载入器
    cluster_data_loader = DataLoader(dataset=ImagesFromList(cluster_set.db_images_key,
                                                            transform=MSLS.input_transform(resize)),
                                     batch_size=config['train'].getint('batch_size'), shuffle=False,
                                     sampler=cluster_sampler)

    # 创建保存中心点的文件
    if not exists(join(ROOT_DIR, 'desired/centroids')):
        makedirs(join(ROOT_DIR, 'desired/centroids'))

    # 定义保存的聚类文件名
    init_cache_clusters = join(join(ROOT_DIR, 'desired/centroids'),
                               'vgg16_' + 'mapillary_' + config['train'].get('num_clusters') + '_desc_cen.hdf5')

    # 如果文件存在就删除该文件
    if exists(init_cache_clusters):
        remove(init_cache_clusters)

    with h5py.File(init_cache_clusters, mode='w') as h5_file:
        with torch.no_grad():
            model.eval()
            print('===> Extracting Descriptors')

            # 在H5文件中创建图像描述
            db_feature = h5_file.create_dataset("descriptors", [descriptors_size, encoding_dim], dtype=np.float32)

            for iteration, (input_data, indices) in enumerate(tqdm(cluster_data_loader, desc='Iter'), 1):
                input_data = input_data.to(device)
                # 使用BackBone提取图像特征，并且形状为
                # (B, C, H, W)->(B, encoding_dim, H, W)->(B, encoding_dim, HxW)->(B, HxW, encoding_dim)，
                # HxW表示不同位置，encoding_dim表示不同位置特征的维度
                image_descriptors = model.encoder(input_data).view(input_data.size(0), encoding_dim, -1).permute(0, 2, 1)
                # 对encoding_dim的图像特征进行L2正则化
                image_descriptors = F.normalize(image_descriptors, p=2, dim=2)

                # 每个图像per_image_sample_count个特征，一共有batch_size个图像，计算有多少个特征作为索引偏移
                batch_index = (iteration - 1) * config['train'].getint('batch_size') * per_image_sample_count
                for ix in range(image_descriptors.size(0)):
                    # 对Batch中的每个图像进行随机位置的采样
                    sample = np.random.choice(image_descriptors.size(1), per_image_sample_count, False)
                    # 设置在H5中保存的索引
                    start_ix = batch_index + ix * per_image_sample_count
                    # 保存每个图像提取到的per_image_sample_count个特征
                    db_feature[start_ix:start_ix + per_image_sample_count, :] = \
                        image_descriptors[ix, sample, :].detach().cpu().numpy()

                # 清空内存
                del input_data, image_descriptors

                if iteration == 125:
                    pass

        print('====> 开始进行聚类..')
        # 定义聚类方法KMeans
        kmeans = KMeans(n_clusters=config['train'].getint('num_clusters'), max_iter=100)
        # 拟合图像特征数据
        kmeans.fit(db_feature[...])

        print('====> 保存聚类的中心点 {}'.format(kmeans.cluster_centers_.shape))
        h5_file.create_dataset('centroids', data=kmeans.cluster_centers_)

        print('====> 聚类完成')
