import os
import torch


def model_get(args):
    if os.path.exists(args.weight):
        model_dict = torch.load(args.weight, map_location='cpu')
    else:
        if args.timm:
            model = model_prepare(args)._timm_model()
        else:
            choice_dict = {'yolov7_cls': 'model_prepare(args)._yolov7_cls()'}
            model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['epoch'] = -1  # 已训练的轮次
        model_dict['optimizer_state_dict'] = None  # 学习率参数
        model_dict['lr_adjust_item'] = 0  # 学习率调整参数
        model_dict['ema_updates'] = 0  # ema参数
        model_dict['standard'] = 0  # 评价指标
    return model_dict


class model_prepare(object):
    def __init__(self, args):
        self.args = args

    def _timm_model(self):
        from model.timm_model import timm_model
        model = timm_model(self.args)
        return model

    def _yolov7_cls(self):
        from model.yolov7_cls import yolov7_cls
        model = yolov7_cls(self.args)
        return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='分类任务')
    parser.add_argument('--model', default='', type=str, help='|模型选择|')
    parser.add_argument('--weight', default='', type=str, help='|模型位置，如果没找到模型则创建新模型|')
    args = parser.parse_args()
    model_get(args)
