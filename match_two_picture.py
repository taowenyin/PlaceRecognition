import argparse
import configparser
import torch
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from dataset.mapillary_sls.MSLS import MSLS
from tools import ROOT_DIR
from os.path import join, isfile
from models.models_generic import get_backbone, get_model


def match_image(model, device, config, image_1, image_2, result_save_path):
    """
    对两张图片进行匹配

    :param model: 特征提取模型
    :param device: 运行设置
    :param config: 配置
    :param image_1: 图片1
    :param image_2: 图片2
    :param result_save_path: 结果保存的地址
    """

    pool_size = int(config['train']['num_pcas'])

    model.eval()

    input_transform = MSLS.input_transform(resize=tuple(map(int, str.split(config['train'].get('resize'), ','))))

    image_1 = Image.fromarray(image_1)
    image_2 = Image.fromarray(image_2)

    image_1 = input_transform(image_1).unsqueeze(0)
    image_2 = input_transform(image_2).unsqueeze(0)

    input_data = torch.cat((image_1, image_2), dim=0).to(device)

    print('====> 提取图像特征')
    with torch.no_grad():
        image_encoding = model.encoder(input_data)

        # 得到VLAD的局部特征
        vlad_local, _ = model.pool(image_encoding)

        image_1_local_feature = []
        image_2_local_feature = []

        for this_iter, this_local in enumerate(vlad_local):
            # this_local -> vlad的Shape变化为(B, C, F) -> (F, B, C) -> (B x F, C)
            vlad = this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))
            # vlad再变化为(B x F, C, 1, ,1)送入WPCA，再经过Flatten变为（B x F, num_pcas）
            pca_encoding = model.WPCA(vlad.unsqueeze(-1).unsqueeze(-1))
            # （B x F, num_pcas）->(F, B, num_pcas)->(B, num_pcas, F)
            pca_encoding = pca_encoding.reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)

            a = pca_encoding[0, :, :]
            b = torch.transpose(pca_encoding[0, :, :], 0, 1)

            image_1_local_feature.append(torch.transpose(pca_encoding[0, :, :], 0, 1))
            image_2_local_feature.append(pca_encoding[1, :, :])

            print('xxx')


    print('xxx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patch-NetVLAD的两张图象匹配')
    parser.add_argument('--config_path', type=str, default=join(ROOT_DIR, 'configs'),
                        help='模型提取特征后执行的配置文件目录')
    parser.add_argument('--im_path', type=str, default=join(ROOT_DIR, 'example_images'), help='用于图像匹配的图像目录')
    parser.add_argument('--plot_save_path', type=str, default=join(ROOT_DIR, 'results'), help='匹配结果的保存路径')
    parser.add_argument('--nocuda', action='store_true', help='如果使用该参数表示只使用CPU，否则使用GPU')

    opt = parser.parse_args()

    configfile = join(opt.config_path, 'performance.ini')
    config = configparser.ConfigParser()
    config.read(configfile, encoding='utf-8')

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoding_model, encoding_dim = get_backbone(config)

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')

    resume_ckpt = join(ROOT_DIR, 'pretrained_models',
                       dataset_name + '_WPCA' + config['train']['num_pcas'] + '.pth.tar')

    db_img_file = join(opt.im_path, 'db.png')
    query_img_file = join(opt.im_path, 'query.png')

    if isfile(resume_ckpt):
        print("=> 载入训练好的模型参数 '{}'".format(resume_ckpt))
        check_point = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)

        a = check_point['state_dict']['pool.centroids']
        config[dataset_name]['num_clusters'] = str(check_point['state_dict']['pool.centroids'].shape[0])

        model = get_model(encoding_model, encoding_dim, config, append_pca_layer=True)

        # todo 预训练模型与写的模型不匹配
        # model.load_state_dict(check_point['state_dict'])
        model = model.to(device)

        print("=> 载入训练好的模型完毕 '{}'".format(resume_ckpt))
    else:
        raise FileNotFoundError('=> 在{}没有找到预训练的模型'.format(resume_ckpt))

    db_img = cv2.cvtColor(cv2.imread(db_img_file), cv2.COLOR_BGR2RGB)
    query_img = cv2.cvtColor(cv2.imread(query_img_file), cv2.COLOR_BGR2RGB)

    if db_img is None:
        raise FileNotFoundError(db_img_file + ' 不存在')
    if query_img is None:
        raise FileNotFoundError(query_img_file + ' 不存在')

    match_image(model, device, config, db_img, query_img, opt.plot_save_path)

    torch.cuda.empty_cache()

    print('匹配结束')
