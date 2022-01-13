import numpy as np
import torch


class PatchMatch():
    def __init__(self, matcher: str, patch_size: list, stride_size: list, keypoints: list, indices: list):
        """
        对Patch进行匹配

        :param matcher: 匹配的方法（RANSAC，spatialApproximator）
        :param patch_size: Patch的大小
        :param stride_size: Stride的大小
        :param keypoints: Keypoint在原图像中的坐标
        :param indices: Keypoint在输出图的索引
        """
        self.__indices = indices
        self.__keypoints = keypoints
        self.__stride_size = stride_size
        self.__patch_size = patch_size
        self.__matcher = matcher

    def match(self, q_features, db_features):
        if self.__matcher == 'RANSAC':
            return self.__match_with_ransac(q_features, db_features)
        elif self.__matcher == 'spatialApproximator':
            return self.__match_with_spatial(q_features, db_features)
        else:
            raise ValueError('未知的Patch匹配方法')

    def __match_with_ransac(self, q_features, db_features):
        scores = []
        all_inlier_index_keypoints = []
        all_inlier_query_keypoints = []

        for q_feat, db_feat, keypoint, stride in zip(q_features, db_features, self.__keypoints, self.__stride_size):
            fw_inds, bw_inds = self.__torch_nn(q_feat, db_feat)

            fw_inds = fw_inds.cpu().numpy()
            bw_inds = bw_inds.cpu().numpy()

            mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

            print('xxx')


        return None

    def __match_with_spatial(self, q_features, db_features):
        print('xxx')
        return None

    def __torch_nn(self, x, y):
        mul = torch.matmul(x, y)

        dist = 2 - 2 * mul + 1e-9
        dist = torch.sqrt(dist)

        _, fw_inds = torch.min(dist, 0)
        bw_inds = torch.argmin(dist, 1)

        return fw_inds, bw_inds


if __name__ == '__main__':


    tools = PatchMatch('RANSAC', patch_size=[2,5,8], stride_size=[1,1,1])

    print('xxx')
