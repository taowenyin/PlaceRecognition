import urllib.request

from tools import ROOT_DIR
from os.path import join, exists, isfile
from os import makedirs


if __name__ == '__main__':
    dest_dir = join(ROOT_DIR, 'pretrained_models')

    if not exists(dest_dir):
        makedirs(dest_dir)

    print('自动下载预训练模型到{}，需要2GB空间。'.format(dest_dir))

    if not isfile(join(dest_dir, "mapillary_WPCA128.pth.tar")):
        print('Downloading mapillary_WPCA128.pth.tar')
        urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/vvr0jizjti0z2LR/download",
                                   join(dest_dir, "mapillary_WPCA128.pth.tar"))
    if not isfile(join(dest_dir, "mapillary_WPCA512.pth.tar")):
        print('Downloading mapillary_WPCA512.pth.tar')
        urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/DFxbGgFwh1y1wAz/download",
                                   join(dest_dir, "mapillary_WPCA512.pth.tar"))
    if not isfile(join(dest_dir, "mapillary_WPCA4096.pth.tar")):
        print('Downloading mapillary_WPCA4096.pth.tar')
        urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/ZgW7DMEpeS47ELI/download",
                                   join(dest_dir, "mapillary_WPCA4096.pth.tar"))
    if not isfile(join(dest_dir, "pittsburgh_WPCA128.pth.tar")):
        print('Downloading pittsburgh_WPCA128.pth.tar')
        urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/2ORvaCckitjz4Sd/download",
                                   join(dest_dir, "pittsburgh_WPCA128.pth.tar"))
    if not isfile(join(dest_dir, "pittsburgh_WPCA512.pth.tar")):
        print('Downloading pittsburgh_WPCA512.pth.tar')
        urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/WKl45MoboSyB4SH/download",
                                   join(dest_dir, "pittsburgh_WPCA512.pth.tar"))
    if not isfile(join(dest_dir, "pittsburgh_WPCA4096.pth.tar")):
        print('Downloading pittsburgh_WPCA4096.pth.tar')
        urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/1aoTGbFjsekeKlB/download",
                                   join(dest_dir, "pittsburgh_WPCA4096.pth.tar"))

    print('完成所有预训练模型的下载。')
