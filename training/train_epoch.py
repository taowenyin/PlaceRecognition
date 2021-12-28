import torch
import math

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

    # 计算有多少个Batch
    batch_count = math.ceil(len(train_dataset.q_seq_idx) / config['train'].getint('batch_size'))

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')

    # 迭代每一批Query
    cached_q_count_bar = trange(train_dataset.cached_subset_size)
    for sub_cached_q_iter in cached_q_count_bar:
        cached_q_count_bar.set_description(
            '第{}/{}批Query数据'.format(sub_cached_q_iter, train_dataset.cached_subset_size))

        if config['train']['pooling'].lower() == 'netvlad' or config['train']['pooling'].lower() == 'patchnetvlad':
            encoding_dim *= config[dataset_name].getint('num_clusters')

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
            pooling_q, pooling_p, pooling_n = torch.split(pooling_data, [B, B, neg_size])

            optimizer.zero_grad()

            # 对每个Query、Positive、Negative组成的三元对象进行Loss计算，由于每个Query对应的Negative数量不同，所以需要这样计算
            loss = 0
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    neg_ix = (torch.sum(neg_counts[:i]) + n).item()
                    loss += criterion(pooling_q[i: i + 1], pooling_p[i: i + 1], pooling_n[neg_ix:neg_ix + 1])

            # 对损失求平均
            loss /= neg_size.float().to(device)

            loss.backward()
            optimizer.step()
            del data_input, data_encoding, pooling_data, pooling_q, pooling_p, pooling_n
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or batch_count <= 10:
                # todo 记录损失
                print('xxx')

        start_iter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    # 计算平均损失
    avg_loss = epoch_loss / batch_count

    # todo 记录损失
    print('xxx')
