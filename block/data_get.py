import os
import cv2
import numpy as np
import pandas as pd


def data_get(args):
    dataset_dict = data_prepare(args)._load()
    return dataset_dict


class data_prepare(object):
    def __init__(self, args):
        self.args = args

    def _load(self):
        dataset_dict = {}
        dataset_dict['train'] = self._load_train()
        dataset_dict['val'] = self._load_val()
        dataset_dict['class'] = self._load_class()
        return dataset_dict

    def _load_train(self):
        train_values = pd.read_csv(self.args.data_path + '/' + 'train.csv', header=None).values  # 读取train.csv
        train_list = []
        for i in range(len(train_values)):
            if not os.path.exists(self.args.data_path + '/' + train_values[i, 0]):
                print('| {}不存在train.csv中图片:{} |'.format(self.args.data_path, train_values[i, 0]))
                continue
            image = cv2.imread(self.args.data_path + '/' + train_values[i, 0])  # 读取图片
            image = self._resize(image)  # 变为输入形状
            image = self._processing(image)  # 归一化和减均值
            class_onehot = np.zeros(self.args.output_class, dtype=np.float32)  # 标签转为独热编码
            class_onehot[train_values[i][1]] = 1
            train_list.append([image, class_onehot])
        return train_list

    def _load_val(self):
        val_values = pd.read_csv(self.args.data_path + '/' + 'val.csv', header=None).values  # 读取val.csv
        val_list = []
        for i in range(len(val_values)):
            if not os.path.exists(self.args.data_path + '/' + val_values[i, 0]):
                print('| {}不存在val.csv中图片:{} |'.format(self.args.data_path, val_values[i, 0]))
                continue
            image = cv2.imread(self.args.data_path + '/' + val_values[i, 0])  # 读取图片
            image = self._resize(image)  # 变为输入形状
            image = self._processing(image)  # 归一化和减均值
            class_onehot = np.zeros(self.args.output_class, dtype=np.float32)  # 标签转为独热编码
            class_onehot[val_values[i][1]] = 1
            val_list.append([image, class_onehot])
        return val_list

    def _load_class(self):
        class_ = pd.read_csv(self.args.data_path + '/' + 'class.csv', header=None).values
        return class_

    def _resize(self, image):
        args = self.args
        w0 = image.shape[1]
        h0 = image.shape[0]
        if w0 == h0:
            image = cv2.resize(image, (args.input_size, args.input_size))
        elif w0 > h0:  # 宽大于高
            w = args.input_size
            h = int(w / w0 * h0)
            image = cv2.resize(image, (w, h))
            add_y = (w - h) // 2
            image = cv2.copyMakeBorder(image, add_y, w - h - add_y, 0, 0, cv2.BORDER_CONSTANT, value=(126, 126, 126))
        else:  # 宽小于高
            h = self.args.input_size
            w = int(h / h0 * w0)
            image = cv2.resize(image, (w, h))
            add_x = (h - w) // 2
            image = cv2.copyMakeBorder(image, 0, 0, add_x, h - w - add_x, cv2.BORDER_CONSTANT, value=(126, 126, 126))
        return image

    def _processing(self, image):
        image = (image / 255).transpose(2, 0, 1).astype(np.float32)
        image[0] = image[0] - self.args.bgr_mean[0]
        image[1] = image[1] - self.args.bgr_mean[1]
        image[2] = image[2] - self.args.bgr_mean[2]
        return image


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', default='../dataset/classification/mask', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    args = parser.parse_args()
    dataset_dict = data_get(args)
