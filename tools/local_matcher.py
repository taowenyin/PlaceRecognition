from configparser import ConfigParser


def calc_receptive_boxes(height, width):
    """
    计算每个特征点的感受域

    :param height: 特征图的宽度
    :param width: 特征图的高度
    :return:
    """
    print('xxx')


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

    keypoints = None
    indices = None

    return keypoints, indices
