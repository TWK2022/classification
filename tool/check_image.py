# 数据需准备成以下格式
# ├── 数据集路径：data_path
#     └── image：存放所有图片
#     └── train.txt：训练图片的绝对路径(或相对data_path下路径)和类别号，如-->image/mask/0.jpg 0 2<--表示该图片类别为0和2，空类别图片无类别号
#     └── val.txt：验证图片的绝对路径(或相对data_path下路径)和类别
#     └── class.txt：所有的类别名称
import os
import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='检查标签train.txt和val.txt中的图片是否存在')
parser.add_argument('--data_path', default=r'D:\dataset\classification\mask', type=str, help='|数据集根目录|')
args = parser.parse_args()
args.train = args.data_path + '/' + 'train.txt'
args.val = args.data_path + '/' + 'val.txt'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def _check_image(image_path):
    if not os.path.exists(image_path):
        print(f'没有找到图片:{image_path}')
        args.record += 1
    args.tqdm_show.update(1)


def check_image(txt_path):
    with open(txt_path)as f:
        image_path_list = [_.strip().split(' ')[0] for _ in f.readlines()]
    args.record = 0
    args.tqdm_show = tqdm.tqdm(total=len(image_path_list))
    with ThreadPoolExecutor() as executer:
        executer.map(_check_image, image_path_list)
    args.tqdm_show.close()
    print(f'| {txt_path}找到图片数:{len(image_path_list) - args.record} 缺失图片数:{args.record} |')


if __name__ == '__main__':
    check_image(args.train)
    check_image(args.val)
