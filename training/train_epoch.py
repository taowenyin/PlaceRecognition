import torch

from dataset.mapillary_sls.MSLS import MSLS
from configparser import ConfigParser
from tqdm import trange
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm


def train_epoch(train_dataset: MSLS, model: Module, optimizer, criterion, encoding_dim, device, opt, config: ConfigParser):
    """
    一次训练的过程

    :param train_dataset: 训练的数据集
    :param model: 训练的模型
    :param optimizer: 训练的优化器
    :param criterion: 训练的损失函数
    :param encoding_dim: Backbone模型的输出维度
    :param device: 训练使用的设备
    :param opt: 传入的参数
    :param config: 训练的配置参数
    """

    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False

    train_dataset.new_epoch()

    # 每个训练周期的损失
    epoch_loss = 0

    # 每个训练周期中，Step的起始索引
    start_iter = 1

    # 计算一种有多少个Batch
    batch_count = len(train_dataset.q_seq_idx) // config['train'].getint('batch_size')

    # 迭代每一批Query
    cached_q_count_bar = trange(train_dataset.cached_subset_size)
    for sub_cached_q_iter in cached_q_count_bar:
        cached_q_count_bar.set_description(
            '第{}/{}批Query数据'.format(sub_cached_q_iter, train_dataset.cached_subset_size))

        # 刷新数据
        train_dataset.refresh_data(model, encoding_dim)

        # 训练数据集的载入器
        training_data_loader = DataLoader(dataset=train_dataset, batch_size=config['train'].getint('batch_size'),
                                          shuffle=True, collate_fn=MSLS.collate_fn)

        # 进入训练模式
        model.train()

        training_data_bar = tqdm(training_data_loader, leave=False)
        for iteration, (query, positives, negatives, neg_counts, indices) in enumerate(training_data_bar, start_iter):
            training_data_bar.set_description('第{}批的第{}组训练数据'.format(sub_cached_q_iter, iteration))

            if query is None:
                continue

            # 获取Query的B、C、H、W
            B, C, H, W = query.shape
            # 计算所有Query对应的反例数量和
            neg_size = torch.sum(neg_counts)
            # 把Query、Positives和Negatives进行拼接，合并成一个Tensor
            data_input = torch.cat([query, positives, negatives])
            # 把数据放到GPU中
            data_input = data_input.to(device)

            # 对数据使用BackBone提取图像特征
            data_encoding = model.encoder(data_input)
            # 经过池化后的数据
            pooling_data = model.pool(data_encoding)

            # 把Pooling的数据分为Query、正例和负例
            pooling_Q, pooling_P, pooling_N = torch.split(pooling_data, [B, B, neg_size])

            optimizer.zero_grad()

            # todo 训练还未结束

            pass

    pass