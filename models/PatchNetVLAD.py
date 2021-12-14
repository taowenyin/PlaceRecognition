import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
        self.__patch_sizes = patch_sizes
        self.__vlad_v2 = vlad_v2

        self.__conv = nn.Conv2d(encoding_dim, num_clusters, kernel_size=(1, 1), bias=vlad_v2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, encoding_dim))

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

        if not self.__vlad_v2:
            # 执行L2范数，并对原始数据进行正则化操作
            clusters_assignment = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
            pass
        else:
            pass

        pass
