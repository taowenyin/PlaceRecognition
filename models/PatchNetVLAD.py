import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors


class PatchNetVLAD(nn.Module):
    def __init__(self, num_clusters, encoding_dim, patch_sizes='4', strides='1', vlad_v2=False):
        """
        PatchNetVLAD模型

        :param num_clusters: 聚类的数量
        :param encoding_dim: 图像特征的编码
        :param patch_sizes: 小块的数量
        :param vlad_v2: True时表示VLAD V2，否则为V1
        """
        super(PatchNetVLAD, self).__init__()

        self.__num_clusters = num_clusters
        self.__encoding_dim = encoding_dim
        self.__alpha = 0
        self.__patch_sizes = patch_sizes
        self.__vlad_v2 = vlad_v2

        self.__conv = nn.Conv2d(encoding_dim, num_clusters, kernel_size=(1, 1), bias=vlad_v2)
        # 初始化中心点的维度是(num_clusters, encoding_dim)
        self.__centroids = nn.Parameter(torch.rand(num_clusters, encoding_dim))

        # 保存不同尺度Patch Size和Stride
        patch_sizes = patch_sizes.split(",")
        strides = strides.split(",")
        self.__patch_sizes = []
        self.__strides = []
        for patch_size, stride in zip(patch_sizes, strides):
            self.__patch_sizes.append(int(patch_size))
            self.__strides.append(int(stride))

    def init_params(self, clusters, descriptors):
        """
        初始化PatchNetVLAD的参数

        :param clusters: 图像聚类后的中心点
        :param descriptors: 经过BackBone后的图像描述
        """

        if not self.__vlad_v2: # 不是VLAD V2的参数初始化
            # 执行L2范数，并对原始数据进行正则化操作
            clusters_assignment = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
            # 求余弦距离
            cos_dis = np.dot(clusters_assignment, descriptors.T)
            cos_dis.sort(0)
            # 排序，降序
            cos_dis = cos_dis[::-1, :]
            self.__alpha = (np.log(0.01) / np.mean(cos_dis[0, :] - cos_dis[1, :])).item()
            # 使用聚类后的中心点来初始化中心点参数
            self.__centroids = nn.Parameter(torch.from_numpy(clusters))

            # 初始化卷积权重和偏置
            self.__conv.weight = nn.Parameter(
                torch.from_numpy(self.__alpha * clusters_assignment).unsqueeze(2).unsqueeze(3))
            self.__conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(descriptors)
            del descriptors
            distance_square = np.square(knn.kneighbors(clusters, 2)[1])
            del knn

            self.__alpha = (-1 * np.log(0.01) / np.mean(distance_square[:, 1] - distance_square[:, 0])).item()
            # 使用聚类后的中心点来初始化中心点参数
            self.__centroids = nn.Parameter(torch.from_numpy(clusters))
            del clusters, distance_square

            # 初始化卷积权重和偏置
            self.__conv.weight = nn.Parameter((2.0 * self.__alpha * self.__centroids).unsqueeze(-1).unsqueeze(-1))
            self.__conv.bias = nn.Parameter(-1 * self.__alpha * self.__centroids.norm(dim=1))

        pass
