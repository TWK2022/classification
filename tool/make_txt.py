import os
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='将文件夹中的图片和类别号按比例添加到train.txt和val.txt中')
parser.add_argument('--image_path', default=r'D:\dataset\classification\mask\image\mask', type=str, help='|图片所在目录|')
parser.add_argument('--add', default=' 0', type=str, help='|标签内容为[图片绝对路径+add]|')
parser.add_argument('--divide', default='9,1', type=str, help='|图片划分到train.txt和val.txt的比例|')
args = parser.parse_args()

# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    if not os.path.exists('train.txt'):
        with open('train.txt', 'w')as f:
            pass
    if not os.path.exists('val.txt'):
        with open('val.txt', 'w')as f:
            pass
    image_dir = sorted(os.listdir(args.image_path))
    args.divide = list(map(int, args.divide.split(',')))
    boundary = int(len(image_dir) * args.divide[0] / (args.divide[0] + args.divide[1]))
    with open('train.txt', 'a')as f:
        write_line = []
        for i in range(boundary):
            label = args.image_path + '/' + image_dir[i] + args.add
            write_line.append(label + '\n')
        f.writelines(write_line)
    with open('val.txt', 'a')as f:
        write_line = []
        for i in range(boundary, len(image_dir)):
            label = args.image_path + '/' + image_dir[i] + args.add
            write_line.append(label + '\n')
        f.writelines(write_line)
