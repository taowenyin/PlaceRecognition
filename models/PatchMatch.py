import numpy as np
import torch
import cv2


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

            # todo 获取至少4个点用于计算变换矩阵（不懂什么意思）
            mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

            if len(mutuals) > 3:
                # 获取关键点
                index_keypoints = keypoint[:, mutuals]
                query_keypoints = keypoint[:, fw_inds[mutuals]]

                index_keypoints = np.transpose(index_keypoints)
                query_keypoints = np.transpose(query_keypoints)

                # todo 找到对应的转换矩阵，由于采用VGG，因此使用[stride*1.5]计算（其实不懂）
                _, mask = cv2.findHomography(index_keypoints, query_keypoints, cv2.FM_RANSAC,
                                             ransacReprojThreshold=16 * stride * 1.5)

                # todo 没太懂
                inlier_index_keypoints = index_keypoints[mask.ravel() == 1]
                all_inlier_query_keypoints.append(query_keypoints[mask.ravel() == 1])
                inlier_count = inlier_index_keypoints.shape[0]
                scores.append(-1 * inlier_count / q_feat.shape[0])
                all_inlier_index_keypoints.append(inlier_index_keypoints)
            else:
                scores.append(0)

        return scores, all_inlier_query_keypoints, all_inlier_index_keypoints

    def __match_with_spatial(self, q_features, db_features):
        # todo 未完成
        return None, None, None

    def __torch_nn(self, x, y):
        mul = torch.matmul(x, y)

        dist = 2 - 2 * mul + 1e-9
        dist = torch.sqrt(dist)

        _, fw_inds = torch.min(dist, dim=0)
        bw_inds = torch.argmin(dist, dim=1)

        return fw_inds, bw_inds


if __name__ == '__main__':

    print('xxx')
