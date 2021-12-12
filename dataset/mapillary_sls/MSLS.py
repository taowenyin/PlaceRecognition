import math
import random
import pandas as pd
import numpy as np
import sys
import torchvision.transforms as transforms
import torch.utils.data as data
import itertools

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from os.path import join
from sklearn.neighbors import NearestNeighbors
from PIL import Image

default_cities = {
    'train': ['trondheim', 'london', 'boston', 'melbourne', 'amsterdam', 'helsinki',
              'tokyo', 'toronto', 'saopaulo', 'moscow', 'zurich', 'paris', 'bangkok',
              'budapest', 'austin', 'berlin', 'ottawa', 'phoenix', 'goa', 'amman', 'nairobi', 'manila'],
    'val': ['cph', 'sf'],
    'test': ['miami', 'athens', 'buenosaires', 'stockholm', 'bengaluru', 'kampala']
}


class ImagesFromList(Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = [Image.open(im) for im in self.images[idx].split(",")]
        except:
            img = [Image.open(self.images[0])]
        img = [self.transform(im) for im in img]

        if len(img) == 1:
            img = img[0]

        return img, idx


class MSLS(Dataset):
    def __init__(self, root_dir, mode='train', cities_list=None, img_resize=(480, 640), negative_size=5,
                 positive_distance_threshold=10, negative_distance_threshold=25, cached_queries=1000,
                 batch_size=24, task='im2im', sub_task='all', seq_length=1, exclude_panos=True, positive_sampling=True):
        """
        Mapillary Street-level Sequences数据集的读取

        task（任务）：im2im（图像到图像）, seq2seq（图像序列到图像序列）, seq2im（图像序列到图像）, im2seq（图像到图像序列）

        sub_task（子任务）：all，s2w（summer2winter），w2s（winter2summer），o2n（old2new），n2o（new2old），d2n（day2night），n2d（night2day）

        :param root_dir: 数据集的路径
        :param mode: 数据集的模式[train, val, test]
        :param cities_list: 城市列表
        :param img_resize: 图像大小
        :param negative_size: 每个正例对应的反例个数
        :param positive_distance_threshold: 正例的距离阈值
        :param negative_distance_threshold: 反例的距离阈值，在该距离之内认为是非反例，之外才属于反例，同时正例要在正例阈值内才算正例，正例阈值和负例阈值之间属于非负例
        :param cached_queries: 每次缓存的Query总数，即每个完整的EPOCH中，数据的总量，和Batch Size不同
        :param batch_size: 每批数据的大小
        :param task: 任务类型 [im2im, seq2seq, seq2im, im2seq]
        :param sub_task: 任务类型 [all, s2w, w2s, o2n, n2o, d2n, n2d]
        :param seq_length: 不同任务的序列长度
        :param exclude_panos: 是否排除全景图像
        :param positive_sampling: 是否进行正采样
        """
        super().__init__()

        if cities_list in default_cities:
            self.__cities_list = default_cities[cities_list]
        elif cities_list == '':
            self.__cities_list = default_cities[mode]
        else:
            self.__cities_list = cities_list.split(',')

        # 筛选后的Query图像
        self.__q_images_key = []
        # 筛选后的Database图像
        self.__db_images_key = []
        # Query的序列索引
        self.__q_seq_idx = []
        # positive的序列索引
        self.__p_seq_idx = []
        # 不是负例的索引
        self.__non_negative_indices = []
        # 路边的数据
        self.__sideways = []
        # 晚上的数据
        self.__night = []

        # 三元数据
        self.__triplets_data = []

        self.__mode = mode
        self.__sub_task = sub_task
        self.__exclude_panos = exclude_panos
        self.__negative_num = negative_size
        self.__positive_distance_threshold = positive_distance_threshold
        self.__negative_distance_threshold = negative_distance_threshold
        self.__cached_queries = cached_queries

        # 记录当前EPOCH调用数据集自己的次数，也就是多少个cached_queries数据
        self.__current_subset = 0

        # 得到图像转换对象
        self.__img_transform = MSLS.input_transform(img_resize)

        # 把所有数据分为若干批，每批数据的集合
        self.__cached_subset_idx = []

        # 所有Query对应的正例索引
        self.all_positive_indices = []
        # 每批cached_queries个数据，提供有多少批数据
        self.cached_subset_size = 0

        # 根据任务类型得到序列长度
        if task == 'im2im': # 图像到图像
            seq_length_q, seq_length_db = 1, 1
        elif task == 'seq2seq': # 图像序列到图像序列
            seq_length_q, seq_length_db = seq_length, seq_length
        elif task == 'seq2im': # 图像序列到图像
            seq_length_q, seq_length_db = seq_length, 1
        else:  # im2seq 图像到图像序列
            seq_length_q, seq_length_db = 1, seq_length

        # 载入数据
        load_data_bar = tqdm(self.__cities_list)
        for city in load_data_bar:
            load_data_bar.set_description('=====> 载入{}数据'.format(city))

            # 根据城市获得数据文件夹名称
            subdir = 'test' if city in default_cities['test'] else 'train_val'

            # 保存没有正例的图像数
            non_positive_q_seq_keys_count = 0
            # 保存有正例的图像数
            has_positive_q_seq_keys_count = 0
            # 保存数据集的个数
            q_seq_keys_count = 0

            # 获取到目前为止用于索引的城市图像的长度
            _lenQ = len(self.__q_images_key)
            _lenDb = len(self.__db_images_key)

            # 读取训练集或验证集数据集
            if self.__mode in ['train', 'val']:
                # 载入Query数据
                q_data = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col=0)
                q_data_raw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col=0)

                # 读取数据集数据
                db_data = pd.read_csv(join(root_dir, subdir, city, 'database', 'postprocessed.csv'), index_col=0)
                db_data_raw = pd.read_csv(join(root_dir, subdir, city, 'database', 'raw.csv'), index_col=0)

                # 根据任务把数据变成序列
                q_seq_keys, q_seq_idxs = self.__rang_to_sequence(q_data, join(root_dir, subdir, city, 'query'),
                                                                 seq_length_q)
                db_seq_keys, db_seq_idxs = self.__rang_to_sequence(db_data, join(root_dir, subdir, city, 'database'),
                                                                   seq_length_db)
                q_seq_keys_count = len(q_seq_keys)

                # 如果验证集，那么需要确定子任务的类型
                if self.__mode in ['val']:
                    q_idx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                    db_idx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                    # 从所有序列数据中根据符合子任务的中心索引找到序列数据帧
                    val_frames = np.where(q_idx[self.__sub_task])[0]
                    q_seq_keys, q_seq_idxs = self.__data_filter(q_seq_keys, q_seq_idxs, val_frames)

                    val_frames = np.where(db_idx[self.__sub_task])[0]
                    db_seq_keys, db_seq_idxs = self.__data_filter(db_seq_keys, db_seq_idxs, val_frames)

                # 筛选出不同全景的数据
                if self.__exclude_panos:
                    panos_frames = np.where((q_data_raw['pano'] == False).values)[0]
                    # 从Query数据中筛选出不是全景的数据
                    q_seq_keys, q_seq_idxs = self.__data_filter(q_seq_keys, q_seq_idxs, panos_frames)

                    panos_frames = np.where((db_data_raw['pano'] == False).values)[0]
                    # 从Query数据中筛选出不是全景的数据
                    db_seq_keys, db_seq_idxs = self.__data_filter(db_seq_keys, db_seq_idxs, panos_frames)

                # 删除重复的idx
                unique_q_seq_idxs = np.unique(q_seq_idxs)
                unique_db_seq_idxs = np.unique(db_seq_idxs)

                # 如果排除重复后没有数据，那么就下一个城市
                if len(unique_q_seq_idxs) == 0 or len(unique_db_seq_idxs) == 0:
                    continue

                # 保存筛选后的图像
                self.__q_images_key.extend(q_seq_keys)
                self.__db_images_key.extend(db_seq_keys)

                # 从原数据中筛选后数据
                q_data = q_data.loc[unique_q_seq_idxs]
                db_data = db_data.loc[unique_db_seq_idxs]

                # 获取图像的UTM坐标
                utm_q = q_data[['easting', 'northing']].values.reshape(-1, 2)
                utm_db = db_data[['easting', 'northing']].values.reshape(-1, 2)

                # 获取Query图像的Night状态、否是Sideways，以及图像索引
                night, sideways, index = q_data['night'].values, \
                                         (q_data['view_direction'] == 'Sideways').values, \
                                         q_data.index

                # 创建最近邻算法，使用暴力搜索法
                neigh = NearestNeighbors(algorithm='brute')
                # 对数据集进行拟合
                neigh.fit(utm_db)
                # 在Database中找到符合positive_distance_threshold要求的Query数据的最近邻数据的索引
                positive_distance, positive_indices = neigh.radius_neighbors(utm_q, self.__positive_distance_threshold)
                # 保存所有正例索引
                self.all_positive_indices.extend(positive_indices)

                # 训练模式下，获取负例索引
                if self.__mode == 'train':
                    negative_distance, negative_indices = neigh.radius_neighbors(
                        utm_q, self.__negative_distance_threshold)

                # 查看每个Seq的正例
                for q_seq_key_idx in range(len(q_seq_keys)):
                    # 返回每个序列的帧集合
                    q_frame_idxs = self.__seq_idx_2_frame_idx(q_seq_key_idx, q_seq_idxs)
                    # 返回q_frame_idxs在unique_q_seq_idxs中的索引集合
                    q_uniq_frame_idx = self.__frame_idx_2_uniq_frame_idx(q_frame_idxs, unique_q_seq_idxs)
                    # 返回序列Query中序列对应的正例索引
                    positive_uniq_frame_idxs = np.unique([p for pos in positive_indices[q_uniq_frame_idx] for p in pos])

                    # 查询的序列Query至少要有一个正例
                    if len(positive_uniq_frame_idxs) > 0:
                        # 获取正例所在的序列索引，并去除重复的索引
                        positive_seq_idx = np.unique(self.__uniq_frame_idx_2_seq_idx(
                            unique_db_seq_idxs[positive_uniq_frame_idxs], db_seq_idxs))

                        # todo 不知道是什么意思
                        self.__p_seq_idx.append(positive_seq_idx + _lenDb)
                        self.__q_seq_idx.append(q_seq_key_idx + _lenQ)

                        # 在训练的时候需要根据两个阈值找到正例和负例
                        if self.__mode == 'train':
                            # 找到不是负例的数据
                            n_uniq_frame_idxs = np.unique(
                                [n for nonNeg in negative_indices[q_uniq_frame_idx] for n in nonNeg])
                            # 找到不是负例所在的序列索引，并去除重复的索引
                            n_seq_idx = np.unique(
                                self.__uniq_frame_idx_2_seq_idx(unique_db_seq_idxs[n_uniq_frame_idxs], db_seq_idxs))

                            # 保存数据
                            self.__non_negative_indices.append(n_seq_idx + _lenDb)

                            # todo 不知道是什么意思
                            if sum(night[np.in1d(index, q_frame_idxs)]) > 0:
                                self.__night.append(len(self.__q_seq_idx) - 1)
                            if sum(sideways[np.in1d(index, q_frame_idxs)]) > 0:
                                self.__sideways.append(len(self.__q_seq_idx) - 1)

                        has_positive_q_seq_keys_count += 1
                    else:
                        non_positive_q_seq_keys_count += 1

                print('\n=====> {}训练数据中，有正例的[{}/{}]个，无正例的[{}/{}]个'.format(
                    city,
                    has_positive_q_seq_keys_count,
                    q_seq_keys_count,
                    non_positive_q_seq_keys_count,
                    q_seq_keys_count))

            # 读取测试集数据集，GPS/UTM/Pano都不可用
            elif self.__mode in ['test']:
                # 载入对应子任务的图像索引
                q_idx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                db_idx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                # 根据任务把数据变成序列
                q_seq_keys, q_seq_idxs = self.__rang_to_sequence(q_idx, join(root_dir, subdir, city, 'query'),
                                                                 seq_length_q)
                db_seq_keys, db_seq_idxs = self.__rang_to_sequence(db_idx, join(root_dir, subdir, city, 'database'),
                                                                   seq_length_db)

                # 从所有序列数据中根据符合子任务的中心索引找到序列数据帧
                val_frames = np.where(q_idx[self.__sub_task])[0]
                q_seq_keys, q_seq_idxs = self.__data_filter(q_seq_keys, q_seq_idxs, val_frames)

                val_frames = np.where(db_idx[self.__sub_task])[0]
                db_seq_keys, db_seq_idxs = self.__data_filter(db_seq_keys, db_seq_idxs, val_frames)

                # 保存筛选后的图像
                self.__q_images_key.extend(q_seq_keys)
                self.__db_images_key.extend(db_seq_keys)

                # 添加Query索引
                self.__q_seq_idx.extend(list(range(_lenQ, len(q_seq_keys) + _lenQ)))

        # 如果选择了城市、任务和子任务的组合，其中没有Query和Database图像，则退出。
        if len(self.__q_images_key) == 0 or len(self.__db_images_key) == 0:
            print('退出...')
            print('如果选择了城市、任务和子任务的组合，其中没有Query和Database图像，则退出')
            print('如果选择了城市、任务和子任务的组合，其中没有Query和Database图像，则退出')
            print('尝试选择不同的子任务或其他城市')
            sys.exit()

        self.__q_seq_idx = np.asarray(self.__q_seq_idx)
        self.__q_images_key = np.asarray(self.__q_images_key)
        self.__p_seq_idx = np.asarray(self.__p_seq_idx, dtype=object)
        self.__non_negative_indices = np.asarray(self.__non_negative_indices, dtype=object)
        self.__db_images_key= np.asarray(self.__db_images_key)
        self.__sideways = np.asarray(self.__sideways)
        self.__night = np.asarray(self.__night)

        if self.__mode in ['train']:
            # 计算Query采样时的权重，即晚上和路边权重高，容易被采到
            if positive_sampling:
                self.__calc_sampling_weights()
            else:
                self.__weights = np.ones(len(self.__q_seq_idx)) / float(len(self.__q_seq_idx))

    def __getitem__(self, index):
        # 获取对应的数据和标签
        triplet, target = self.__triplets_data[index]

        # 获取Query、Positive和Negative的索引
        q_idx = triplet[0]
        p_idx = triplet[1]
        n_idx = triplet[2:]

        # 返回图像信息
        query = self.__img_transform(Image.open(self.__q_images_key[q_idx]))
        positive = self.__img_transform(Image.open(self.__db_images_key[p_idx]))
        negatives = torch.stack([self.__img_transform(Image.open(self.__db_images_key[idx])) for idx in n_idx], 0)

        return query, positive, negatives, [q_idx, p_idx] + n_idx

    def __len__(self):
        return len(self.__triplets_data)

    def __calc_sampling_weights(self):
        """
        计算数据权重
        """
        # 计算Query大小
        N = len(self.__q_seq_idx)

        # 初始化权重都为1
        self.__weights = np.ones(N)

        # 夜间或侧面时权重更高
        if len(self.__night) != 0:
            self.__weights[self.__night] += N / len(self.__night)
        if len(self.__sideways) != 0:
            self.__weights[self.__sideways] += N / len(self.__sideways)

        # 打印权重信息
        tqdm.write("#侧面 [{}/{}]; #夜间; [{}/{}]".format(len(self.__sideways), N, len(self.__night), N))
        tqdm.write("正面和白天的权重为{:.4f}".format(1))
        if len(self.__night) != 0:
            tqdm.write("正面且夜间的权重为{:.4f}".format(1 + N / len(self.__night)))
        if len(self.__sideways) != 0:
            tqdm.write("侧面且白天的权重为{:.4f}".format(1 + N / len(self.__sideways)))
        if len(self.__sideways) != 0 and len(self.__night) != 0:
            tqdm.write("侧面且夜间的权重为{:.4f}".format(1 + N / len(self.__night) + N / len(self.__sideways)))

    def __seq_idx_2_frame_idx(self, q_seq_key, q_seq_keys):
        """
        把序列索引转化为帧索引

        :param q_seq_key: 序列索引
        :param q_seq_keys: 序列集合
        :return: 索引对应的序列集合
        """
        return q_seq_keys[q_seq_key]

    def __frame_idx_2_uniq_frame_idx(self, frame_idx, uniq_frame_idx):
        """
        获取frame_idx在uniq_frame_idx中的索引列表

        :param frame_idx: 一个序列的帧ID
        :param uniq_frame_idx: 所有帧ID
        :return: 获取frame_idx在uniq_frame_idx中的索引列表
        """

        # 在不重复的数据帧列表uniq_frame_idx中找到要找的数据帧frame_idx，并产生对应的Mask
        frame_mask = np.in1d(uniq_frame_idx, frame_idx)

        # 返回frame_idx在uniq_frame_idx中的索引
        return np.where(frame_mask)[0]

    def __uniq_frame_idx_2_seq_idx(self, frame_idxs, seq_idxs):
        """
        返回图像帧对应的序列索引

        :param frame_idxs: 图像帧
        :param seq_idxs: 序列索引
        :return: 图像正所在的序列索引
        """

        # 在序列索引列表seq_idxs中找到要找的数据帧frame_idxs，并产生对应的Mask
        mask = np.in1d(seq_idxs, frame_idxs)
        # 把Mask重新组织成seq_idxs的形状
        mask = mask.reshape(seq_idxs.shape)

        # 得到序列的索引
        return np.where(mask)[0]

    def __rang_to_sequence(self, data, path, seq_length):
        """
        把数组变为序列

        :param data: 表型数据
        :param path: 数据地址
        :param seq_length: 序列长度
        """
        # 去读序列信息
        seq_info = pd.read_csv(join(path, 'seq_info.csv'), index_col=0)

        # 图像序列的名称和图像序列的索引
        seq_keys, seq_idxs = [], []

        for idx in data.index:
            # 边界的情况
            if idx < (seq_length // 2) or idx >= (len(seq_info) - seq_length // 2):
                continue

            # 计算当前序列数据帧的周边帧
            seq_idx = np.arange(-seq_length // 2, seq_length // 2) + 1 + idx
            # 获取一个序列帧
            seq = seq_info.iloc[seq_idx]

            # 一个序列必须是具有相同的序列键值（即sequence_key相同），以及连续的帧（即frame_number之间的差值为1）
            if len(np.unique(seq['sequence_key'])) == 1 and (seq['frame_number'].diff()[1:] == 1).all():
                seq_key = ','.join([join(path, 'images', key + '.jpg') for key in seq['key']])

                # 保存图像序列的名称
                seq_keys.append(seq_key)
                # 保存图像序列的索引
                seq_idxs.append(seq_idx)

        return seq_keys, np.asarray(seq_idxs)

    def __data_filter(self, seq_keys, seq_idxs, center_frame_condition):
        """
        根据序列中心点索引筛选序列

        :param seq_keys: 序列Key值
        :param seq_idxs: 序列索引
        :param center_frame_condition: 条件筛选的中心帧
        :return: 返回筛选后的Key和Idx
        """
        keys, idxs = [], []
        for key, idx in zip(seq_keys, seq_idxs):
            # 如果序列的中间索引在中心帧中，那么就把Key和Idx放入数组中
            if idx[len(idx) // 2] in center_frame_condition:
                keys.append(key)
                idxs.append(idx)
        return keys, np.asarray(idxs)

    @staticmethod
    def input_transform(resize):
        """
        对图像进行转换

        :param resize: 转换后的图像大小
        :return: 返回转换对象
        """

        if resize[0] > 0 and resize[1] > 0:
            return transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    @staticmethod
    def collate_fn(batch):
        """
        从三元数据列表中创建mini-batch

        :param batch: batch数据
        :return: query: 数据的形状为(batch_size, 3, h, w); positive: 数据的形状为(batch_size, 3, h, w); negatives: 数据的形状为(batch_size, n, 3, h, w)，n表示反例的个数
        """
        # 对Batch中所有的数据进行检查
        batch = list(filter(lambda x: x is not None, batch))

        if len(batch) == 0:
            return None, None, None, None, None

        query, positive, negatives, indices = zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        negative_counts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat([torch.unsqueeze(x, 0) for x in negatives], 0)
        indices = torch.from_numpy(np.asarray(indices))

        return query, positive, negatives, negative_counts, indices

    @property
    def db_images_key(self):
        return self.__db_images_key

    def new_epoch(self):
        """
        每一个EPOCH都需要运行改程序，主要作用是把数据分为若干批，每一批再通过循环输出模型
        """

        # 通过向上取整后，计算一共有多少批Query数据
        self.cached_subset_size = math.ceil(len(self.__q_seq_idx) / self.__cached_queries)

        ##################### 在验证机或测试集上使用
        # 构建所有数据集的索引数组
        q_seq_idx_array = np.arange(len(self.__q_seq_idx))

        # 使用采样方式对Query数据集进行采样
        q_seq_idx_array = random.choices(q_seq_idx_array, self.__weights, k=len(q_seq_idx_array))

        # 把随机采样的Query数据集分为cached_subset_size份
        self.__cached_subset_idx = np.array_split(q_seq_idx_array, self.cached_subset_size)
        #######################

        # 重置子集的计数
        self.__current_subset = 0

    def refresh_data(self, model=None, output_dim=None):
        """
        刷新数据，原因是每个EPOCH都不是取全部数据，而是一部分数据，即cached_queries多的数据，所以要刷新数据，来获取新数据

        :param model: 如果网络已经存在，那么使用该网络对图像进行特征提取，用于验证集或测试集
        :param output_dim: 网络输出的维度
        """
        # 清空数据
        self.__triplets_data.clear()

        if model is None:
            # 随机从q_seq_idx中采样cached_queries长度的数据索引
            q_choice_idxs = np.random.choice(len(self.__q_seq_idx), self.__cached_queries, replace=False)

            for q_choice_idx in q_choice_idxs:
                # 读取随机采样的Query索引
                q_idx = self.__q_seq_idx[q_choice_idx]
                # 读取随机采样的Query的正例索引，并随机从Query的正例中选取1个正例
                p_idx = np.random.choice(self.__p_seq_idx[q_choice_idx], size=1)[0]

                while True:
                    # 从数据库中随机读取negative_num个反例
                    n_idxs = np.random.choice(len(self.__db_images_key), self.__negative_num)

                    # Query的negative_distance_threshold距离外才被认为是负例，而negative_distance_threshold内认为是正例或非负例，
                    # 下面的判断是为了保证选择负例不在negative_distance_threshold范围内
                    if sum(np.in1d(n_idxs, self.__non_negative_indices[q_choice_idx])) == 0:
                        break

                # 创建三元数据和对应的标签
                triplet = [q_idx, p_idx, *n_idxs]
                target = [-1, 1] + [0] * len(n_idxs)

                self.__triplets_data.append((triplet, target))

            # 子数据集调用次数+1
            self.__current_subset += 1

            return

        # todo 如果model存在，那么就需要下面对图像进行特征提取
        pass

