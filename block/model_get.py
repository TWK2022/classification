import os
import torch


def model_get(args):
    if os.path.exists(args.weight):
        model_dict = torch.load(args.weight, map_location='cpu')
    else:
        if args.timm:
            model = model_prepare(args)._timm_model()
        else:
            choice_dict = {'cls': 'model_prepare(args)._cls()'}
            model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['val_loss'] = 999
        model_dict['val_m_ap'] = 0
    model_dict['model'].to(args.device)
    model_dict['model'](torch.rand(args.batch, args.input_size, args.input_size, 3).to(args.device))  # 检查
    return model_dict


class model_prepare(object):
    def __init__(self, args):
        self.args = args

    def _timm_model(self):
        from model.timm_model import timm_model
        model = timm_model(self.args)
        return model

    def _cls(self):
        from model.cls import cls
        model = cls(self.args)
        return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='分类任务')
    parser.add_argument('--model', default='', type=str, help='|模型选择|')
    parser.add_argument('--weight', default='', type=str, help='|模型位置，如果没找到模型则创建新模型|')
    args = parser.parse_args()
    model_get(args)
