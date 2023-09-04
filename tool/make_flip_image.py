# 制作翻转的图片，同时创建它们的标签，用于检测图片是否翻转的4分类任务
import os
import cv2
import tqdm
import random
import argparse
from scipy import ndimage

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default=r'D:\dataset\classification\flip\image\000', type=str)
parser.add_argument('--save_path', default=r'D:\dataset\classification\flip\image', type=str)
parser.add_argument('--file_path', default=r'D:\dataset\classification\flip', type=str)
parser.add_argument('--add0', default=True, type=bool, help='|增加色彩变换|')
parser.add_argument('--add1', default=True, type=bool, help='|增加角度倾斜变换|')
parser.add_argument('--divide', default=r'9,1', type=str)
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def resize(image, max_h=1000):  # 用于缩小图片大小，max_h为最大高度
    h, w, _ = image.shape
    h1 = max_h
    w1 = int(h1 / h * w)
    if h > h1:
        image = cv2.resize(image, (w1, h1))
    return image


def left(image):  # 逆时针转90度
    image = cv2.transpose(image)
    image = cv2.flip(image, 0)
    return image


def right(image):  # 顺时针转90度
    image = cv2.transpose(image)
    image = cv2.flip(image, 1)
    return image


def flip(image):  # 顺时针转180度
    image = cv2.flip(image, -1)
    return image


def rotate(image):
    image = ndimage.rotate(image, random.randint(-2, 2))  # 逆时针旋转几度
    return image


if __name__ == '__main__':
    if not os.path.exists(args.save_path + '/270'):
        os.makedirs(args.save_path + '/270')
    if not os.path.exists(args.save_path + '/090'):
        os.makedirs(args.save_path + '/090')
    if not os.path.exists(args.save_path + '/180'):
        os.makedirs(args.save_path + '/180')
    path_list = os.listdir(args.image_path)
    path_list = [f'{args.image_path}/{_}' for _ in path_list]
    A_list = []
    B_list = []
    C_list = []
    D_list = []
    for i, image_path in enumerate(tqdm.tqdm(path_list)):
        image = cv2.imread(image_path)
        image = resize(image)
        image_left = left(image)
        image_right = right(image)
        image_flip = flip(image)
        index = str(i).rjust(3, '0')
        save_left = args.save_path + f'/270/{index}_left.jpg'
        save_right = args.save_path + f'/090/{index}_right.jpg'
        save_flip = args.save_path + f'/180/{index}_flip.jpg'
        cv2.imwrite(save_left, image_left)
        cv2.imwrite(save_right, image_right)
        cv2.imwrite(save_flip, image_flip)
        A_list.append(image_path + ' 0\n')
        B_list.append(save_left + ' 3\n')
        C_list.append(save_right + ' 1\n')
        D_list.append(save_flip + ' 2\n')
        # 色彩变换
        if args.add0:
            A_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            B_rgb = cv2.cvtColor(image_left, cv2.COLOR_RGB2BGR)
            C_rgb = cv2.cvtColor(image_right, cv2.COLOR_RGB2BGR)
            D_rgb = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)
            A_rgb_path = image_path.split('.')[0] + '_bgr.jpg'
            B_rgb_path = args.save_path + f'/270/{index}_left_bgr.jpg'
            C_rgb_path = args.save_path + f'/090/{index}_right_bgr.jpg'
            D_rgb_path = args.save_path + f'/180/{index}_flip_bgr.jpg'
            cv2.imwrite(A_rgb_path, A_rgb)
            cv2.imwrite(B_rgb_path, B_rgb)
            cv2.imwrite(C_rgb_path, C_rgb)
            cv2.imwrite(D_rgb_path, D_rgb)
            A_list.append(A_rgb_path + ' 0\n')
            B_list.append(B_rgb_path + ' 3\n')
            C_list.append(C_rgb_path + ' 1\n')
            D_list.append(D_rgb_path + ' 2\n')
        # 角度变换
        if args.add1:
            A_rotate = rotate(image)
            B_rotate = rotate(image_left)
            C_rotate = rotate(image_right)
            D_rotate = rotate(image_flip)
            A_rotate_path = image_path.split('.')[0] + '_rotate.jpg'
            B_rotate_path = args.save_path + f'/270/{index}_left_rotate.jpg'
            C_rotate_path = args.save_path + f'/090/{index}_right_rotate.jpg'
            D_rotate_path = args.save_path + f'/180/{index}_flip_rotate.jpg'
            cv2.imwrite(A_rotate_path, A_rotate)
            cv2.imwrite(B_rotate_path, B_rotate)
            cv2.imwrite(C_rotate_path, C_rotate)
            cv2.imwrite(D_rotate_path, D_rotate)
            A_list.append(A_rotate_path + ' 0\n')
            B_list.append(B_rotate_path + ' 3\n')
            C_list.append(C_rotate_path + ' 1\n')
            D_list.append(D_rotate_path + ' 2\n')
    a, b = list(map(int, args.divide.split(',')))
    data_len = len(A_list)
    random.shuffle(A_list)
    random.shuffle(B_list)
    random.shuffle(C_list)
    random.shuffle(D_list)
    train_number = int(data_len * a / (a + b))
    val_number = int(data_len * b / (a + b))
    with open(args.file_path + '/train.txt', 'w', encoding='utf-8') as f:
        f.writelines(A_list[0:train_number])
        f.writelines(B_list[0:train_number])
        f.writelines(C_list[0:train_number])
        f.writelines(D_list[0:train_number])
    with open(args.file_path + '/val.txt', 'w', encoding='utf-8') as f:
        f.writelines(A_list[0:val_number])
        f.writelines(B_list[0:val_number])
        f.writelines(C_list[0:val_number])
        f.writelines(D_list[0:val_number])
