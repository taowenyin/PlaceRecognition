import torch
import numpy as np

from configparser import ConfigParser


def calc_receptive_boxes(height, width):
    """
    计算每个特征点的感受域

    :param height: 特征图的宽度
    :param width: 特征图的高度
    :return:
    """

    # VGG-16 conv5_3固定的感受野、Stide和Padding，感受野的计算公式为 RF_i = (RF_(i-1) - 1) * Stride + K
    rf, stride, padding = [196.0, 16.0, 90.0]  # hardcoded for vgg-16 conv5_3

    # 根据高和宽形成网格，x，y的形状都为(height, width)
    x, y = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    # x、y合并为(height, width, 2)，然后再reshape为(height x width, 2)
    coordinates = torch.reshape(torch.stack([y, x], dim=2), [-1, 2])

    # [y,x,y,x]
    point_boxes = torch.cat([coordinates, coordinates], 1)
    # 偏移量
    bias = [-padding, -padding, -padding + rf - 1, -padding + rf - 1]

    # 返回(N, 4)的感受野Box的偏移量，其中N为height x width，每个Box为[x_min, y_min, x_max, y_max]
    rf_boxes = stride * point_boxes + torch.FloatTensor(bias)

    return rf_boxes


def calc_keypoint_centers_from_patches(config: ConfigParser, patch_size_h, patch_size_w, stride_h, stride_w):
    """
    计算Patch在图像中的位置

    :param config: 配置文件
    :param patch_size_h: Patch的高度
    :param patch_size_w: Patch的宽度
    :param stride_h: Stride的高度
    :param stride_w: Stride的宽度
    :return:
    """

    H, W = [int(s) for s in config['train'].get('resize').split(",")]

    padding_size = [0, 0]
    patch_size = (patch_size_h, patch_size_w)
    stride = (stride_h, stride_w)

    # 输出的图像大小，参考卷积输出的计算方法：((input - K + 2 * Padding) / S) + 1
    Hout = int(((H + (2 * padding_size[0]) - patch_size[0]) / stride[0]) + 1)
    Wout = int(((W + (2 * padding_size[1]) - patch_size[1]) / stride[1]) + 1)

    # 获取每个特征的感受野空间
    rf_boxes = calc_receptive_boxes(Hout, Wout)

    # 有多少个区域
    num_regions = Hout * Wout

    # Keypoint的索引
    k = 0
    indices = np.zeros((2, num_regions), dtype=int)
    keypoints = np.zeros((2, num_regions), dtype=int)

    for i in range(0, Hout, stride_h):
        for j in range(0, Wout, stride_w):
            # 记录Patch后的特征点索引
            indices[0, k] = j
            indices[1, k] = i

            # 把Patch后的特征点还原到原图中，并取原图中每个感受野的中心点作为Keypoint，并保存这个Keypoint在原图中的坐标
            # 求Keypoint的X轴的坐标
            keypoints[0, k] = ((rf_boxes[j + (i * W), 0] + rf_boxes[(j + (patch_size[1] - 1)) + (i * W), 2]) / 2)
            # 求Keypoint的Y轴的坐标
            keypoints[1, k] = ((rf_boxes[j + (i * W), 1] + rf_boxes[j + ((i + (patch_size[0] - 1)) * W), 3]) / 2)

            k += 1

    return keypoints, indices


if __name__ == '__main__':

    x, y = torch.meshgrid(torch.arange(0, 3), torch.arange(0, 4))

    a = torch.stack([y, x], dim=2)

    coordinates = torch.reshape(torch.stack([y, x], dim=2), [-1, 2])

    x_arr = x.numpy()
    y_arr = y.numpy()
    a_arr = a.numpy()
    coordinates_arr = coordinates.numpy()

    print(x.numpy())
    print(y.numpy())
    print(a.numpy())
    print(coordinates.numpy())

    print('xxx')
