import os
import torch

choice_dict = {'yolov7_cls': 'model_prepare(args).yolov7_cls()'}


def model_get(args):
    if os.path.exists(args.weight):  # 优先加载已有模型继续训练
        model_dict = torch.load(args.weight, map_location='cpu')
    else:  # 新建模型
        if args.prune:  # 模型剪枝
            model_dict = torch.load(args.prune_weight, map_location='cpu')
            model = model_dict['model']
            model = prune(args, model)
        elif args.timm:
            model = model_prepare(args).timm_model()
        else:
            model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['epoch'] = 0  # 已训练的轮次
        model_dict['optimizer_state_dict'] = None  # 学习率参数
        model_dict['lr_adjust_index'] = 0  # 学习率调整次数
        model_dict['ema_updates'] = 0  # ema参数
        model_dict['standard'] = 0  # 评价指标
    return model_dict


def prune(args, model):
    # 记录BN层权重
    BatchNorm2d_weight = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            BatchNorm2d_weight.append(module.weight.data.clone())
    BatchNorm2d_weight_abs = torch.concat(BatchNorm2d_weight, dim=0).abs()
    weight_len = len(BatchNorm2d_weight)
    # 记录权重与BN层编号的关系
    BatchNorm2d_id = []
    for i in range(weight_len):
        BatchNorm2d_id.extend([i for _ in range(len(BatchNorm2d_weight[i]))])
    id_all = torch.tensor(BatchNorm2d_id)
    # 筛选
    value, index = torch.sort(BatchNorm2d_weight_abs, dim=0, descending=True)
    boundary = int(len(index) * args.prune_ratio)
    prune_index = index[0:boundary]  # 保留参数的下标
    prune_index, _ = torch.sort(prune_index, dim=0, descending=False)
    prune_id = id_all[prune_index]
    # 将保留参数的下标放到每层中
    index_list = [[] for _ in range(weight_len)]
    for i in range(len(prune_index)):
        index_list[prune_id[i]].append(prune_index[i])
    # 将每层保留参数的下标换算成相对下标
    record_len = 0
    for i in range(weight_len):
        index_list[i] = torch.tensor(index_list[i])
        index_list[i] -= record_len
        if len(index_list[i]) == 0:  # 存在整层都被减去的情况，至少保留一层
            index_list[i] = torch.argmax(BatchNorm2d_weight[i], dim=0).unsqueeze(0)
        record_len += len(BatchNorm2d_weight[i])
    # 创建剪枝后的模型
    args.prune_num = [len(_) for _ in index_list]
    prune_model = eval(choice_dict[args.model])
    # BN层权重赋值和部分conv权重赋值
    index = 0
    for module, prune_module in zip(model.modules(), prune_model.modules()):
        if isinstance(module, torch.nn.Conv2d):  # 更新部分Conv2d层权重
            if index == 0:
                weight = module.weight.data.clone()[index_list[index]]
            elif index == weight_len:
                weight = module.weight.data.clone()
            else:
                weight = module.weight.data.clone()[index_list[index]]
                weight = weight[:, index_list[index - 1], :, :]
            if prune_module.weight.data.shape == weight.shape:
                prune_module.weight.data = weight
        if isinstance(module, torch.nn.BatchNorm2d):  # 更新BatchNorm2d层权重
            prune_module.weight.data = module.weight.data.clone()[index_list[index]]
            prune_module.bias.data = module.bias.data.clone()[index_list[index]]
            prune_module.running_mean = module.running_mean.clone()[index_list[index]]
            prune_module.running_var = module.running_var.clone()[index_list[index]]
            index += 1
    return prune_model


class model_prepare(object):
    def __init__(self, args):
        self.args = args

    def timm_model(self):
        from model.timm_model import timm_model
        model = timm_model(self.args)
        return model

    def yolov7_cls(self):
        from model.yolov7_cls import yolov7_cls
        model = yolov7_cls(self.args)
        return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='分类任务')
    parser.add_argument('--model', default='', type=str, help='|模型选择|')
    parser.add_argument('--weight', default='', type=str, help='|模型位置，如果没找到模型则创建新模型|')
    args = parser.parse_args()
