import numpy as np


def data_get(args):
    data_dict = data_prepare(args)._load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):
        self.args = args

    def _load(self):
        data_dict = {}
        data_dict['train'] = self._load_label('train.txt')
        data_dict['val'] = self._load_label('val.txt')
        data_dict['class'] = self._load_class()
        return data_dict

    def _load_label(self, txt_name):
        with open(self.args.data_path + '/' + txt_name, encoding='utf-8')as f:
            txt = [_.strip().split(' ') for _ in f.readlines()]  # 读取所有图片路径和类别号
        data_list = [[0, 0] for _ in range(len(txt))]  # [图片路径,类别独热编码]
        for i in range(len(txt)):
            image_path = self.args.data_path + '/image' + txt[i][0].split('image')[-1]
            data_list[i][0] = image_path
            data_list[i][1] = np.zeros(self.args.output_class, dtype=np.float32)
            for j in txt[i][1:]:
                data_list[i][1][int(j)] = 1
        return data_list

    def _load_class(self):
        with open(self.args.data_path + '/' + 'class.txt', encoding='utf-8')as f:
            txt = [_.strip() for _ in f.readlines()]
        return txt


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', default='../dataset/classification/mask', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    args = parser.parse_args()
    data_dict = data_get(args)
