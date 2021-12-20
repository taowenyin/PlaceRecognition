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
            # 计算中心点特征和图像特征之间的余弦距离
            cos_dis = np.dot(clusters_assignment, descriptors.T)
            cos_dis.sort(0)
            # 对余弦距离进行降序排列
            cos_dis = cos_dis[::-1, :]
            #
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

            # 通过KNN算法得到与中心点最近的2个图像的索引
            distance_square = np.square(knn.kneighbors(clusters, 2)[1])
            del knn

            self.__alpha = (-1 * np.log(0.01) / np.mean(distance_square[:, 1] - distance_square[:, 0])).item()
            # 使用聚类后的中心点来初始化中心点参数
            self.__centroids = nn.Parameter(torch.from_numpy(clusters))
            del clusters, distance_square

            # 初始化卷积权重和偏置
            self.__conv.weight = nn.Parameter((2.0 * self.__alpha * self.__centroids).unsqueeze(-1).unsqueeze(-1))
            self.__conv.bias = nn.Parameter(-1 * self.__alpha * self.__centroids.norm(dim=1))

    def forward(self, x):
        B, C, H, W = x.shape

        # ========NetVLAD的soft-assignment部分==========
        # 经过一个1x1的卷积，从（B, C, H, W）->(B, K, H, W)
        soft_assign = self.__conv(x)
        # 经过Softmax得到soft-assignment
        soft_assign = F.softmax(soft_assign, dim=1)
        # =============================================

        # 创建用于保存每个聚类残差值的Tensor
        store_residual = torch.zeros([B, self.__num_clusters, C, H, W],
                                     dtype=x.dtype, layout=x.layout, device=x.device)

        # 循环计算X与每个重点之间的残差，并保存在store_residual中
        for i in range(self.__num_clusters):
            # =====================================NetVLAD的sVLAD core部分====================================
            # 把 (B, C, H, W)的X变为(B, 1, C, H, W)，用于与后续的num_clusters个聚类中心点进行残差计算，其中的1表示就是就是聚类个数
            input_x = x.unsqueeze(0).permute(1, 0, 2, 3, 4)
            # 取出每个聚类中心，形状为(1, encoding_dim)，把该中心点的形状变为(1, encoding_dim, H, W)，
            # 再变为(1, 1, encoding_dim, H, W)，与X的形状保持一致
            centroids = self.__centroids[0:1, :].expand(x.size(2), x.size(3), -1, -1).permute(2, 3, 0, 1).unsqueeze(0)

            # 计算X与每个中心点的残差
            residual = input_x - centroids
            # ===============================================================================================

            # soft-assignment作为α与残差相乘，并且把形状为(B, 1, H, W)的soft_assign变为(B, 1, 1, H, W)，
            # 第1个表示聚类中的一个，第2个1是增加的维度
            soft_assign_ = soft_assign[:, i:i + 1, :].unsqueeze(2)
            residual *= soft_assign_

            # 保存残差
            store_residual[:, i:i + 1, :, :, :] = residual

        return None


if __name__ == '__main__':
    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    device = torch.device("cuda" if cuda else "cpu")

    data = torch.rand(2, 512, 120, 160).to(device)

    image_clusters = np.random.rand(20, 512).astype(np.float32)
    image_descriptors = np.random.rand(50000, 512).astype(np.float32)

    model = PatchNetVLAD(20, 512)
    model.init_params(image_clusters, image_descriptors)

    model = model.to(device)

    output = model(data)
