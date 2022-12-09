import os
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
            image_path = self.args.data_path + '/' + train_values[i, 0]
            if not os.path.exists(image_path):
                print('| {}不存在train.csv中图片:{} |'.format(self.args.data_path, train_values[i, 0]))
                continue
            class_onehot = np.zeros(self.args.output_class, dtype=np.float32)  # 标签转为独热编码
            class_onehot[train_values[i][1]] = 1
            train_list.append([image_path, class_onehot])
        return train_list

    def _load_val(self):
        val_values = pd.read_csv(self.args.data_path + '/' + 'val.csv', header=None).values  # 读取val.csv
        val_list = []
        for i in range(len(val_values)):
            image_path = self.args.data_path + '/' + val_values[i, 0]
            if not os.path.exists(image_path):
                print('| {}不存在val.csv中图片:{} |'.format(self.args.data_path, val_values[i, 0]))
                continue
            class_onehot = np.zeros(self.args.output_class, dtype=np.float32)  # 标签转为独热编码
            class_onehot[val_values[i][1]] = 1
            val_list.append([image_path, class_onehot])
        return val_list

    def _load_class(self):
        cls = pd.read_csv(self.args.data_path + '/' + 'class.csv', header=None).values.tolist()
        return cls


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', default='../dataset/classification/mask', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    args = parser.parse_args()
    dataset_dict = data_get(args)
