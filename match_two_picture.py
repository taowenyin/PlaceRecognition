import argparse
import configparser
import torch

from tools import ROOT_DIR
from os.path import join
from models.models_generic import get_backbone, get_model, create_image_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patch-NetVLAD的两张图象匹配')
    parser.add_argument('--config_path', type=str, default=join(ROOT_DIR, 'configs'),
                        help='模型提取特征后执行的配置文件目录')
    parser.add_argument('--im_path', type=str, default=join(ROOT_DIR, 'example_images'), help='用于图像匹配的图像目录')
    parser.add_argument('--plot_save_path', type=str, default=join(ROOT_DIR, 'results'), help='匹配结果的保存路径')
    parser.add_argument('--nocuda', action='store_true', help='如果使用该参数表示只使用CPU，否则使用GPU')

    opt = parser.parse_args()

    configfile = opt.config_path
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoding_model, encoding_dim = get_backbone(config)

    resume_ckpt = join('pretrained_models',
                       config['dataset'].get('name') + '_WPCA' + config['performance']['num_pcas'] + '.pth.tar')
