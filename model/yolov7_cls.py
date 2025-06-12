# 根据yolov7改编: https://github.com/WongKinYiu/yolov7
import torch
from model.layer import cbs, elan, mp, sppcspc, cls_head


class yolov7_cls(torch.nn.Module):
    def __init__(self, args, config=None):
        super().__init__()
        dim_dict = {'s': 16, 'm': 32, 'l': 64}
        n_dict = {'s': 1, 'm': 2, 'l': 3}
        dim = dim_dict[args.model_type]
        n = n_dict[args.model_type]
        output_class = args.output_class
        # 网络结构
        if config is None:  # 正常版本
            self.cbs0 = cbs(3, dim, 1, 1)
            self.cbs1 = cbs(dim, 2 * dim, 3, 2)  # input_size/2
            self.cbs2 = cbs(2 * dim, 2 * dim, 1, 1)
            self.cbs3 = cbs(2 * dim, 4 * dim, 3, 2)  # input_size/4
            self.elan4 = elan(4 * dim, 8 * dim, n)
            self.mp5 = mp(8 * dim, 8 * dim)  # input_size/8
            self.elan6 = elan(8 * dim, 16 * dim, n)
            self.mp7 = mp(16 * dim, 16 * dim)  # input_size/16
            self.elan8 = elan(16 * dim, 32 * dim, n)
            self.mp9 = mp(32 * dim, 32 * dim)  # input_size/32
            self.elan10 = elan(32 * dim, 32 * dim, n)
            self.sppcspc11 = sppcspc(32 * dim, 16 * dim)
            self.cbs12 = cbs(16 * dim, 8 * dim, 1, 1)
            self.cls_head13 = cls_head(8 * dim, output_class)
        else:  # 剪枝版本
            self.cbs0 = cbs(3, config[0], 1, 1)
            self.cbs1 = cbs(config[0], config[1], 3, 2)  # input_size/2
            self.cbs2 = cbs(config[1], config[2], 1, 1)
            self.cbs3 = cbs(config[2], config[3], 3, 2)  # input_size/4
            index = 4
            self.elan4 = elan(config[index - 1], n=n, config=config[index:])
            index += self.elan4.config_len
            self.mp5 = mp(self.elan4.last_layer, config=config[index:])
            index += self.mp5.config_len
            self.elan6 = elan(self.mp5.last_layer, n=n, config=config[index:])
            index += self.elan6.config_len
            self.mp7 = mp(self.elan6.last_layer, config=config[index:])
            index += self.mp7.config_len
            self.elan8 = elan(self.mp7.last_layer, n=n, config=config[index:])
            index += self.elan8.config_len
            self.mp9 = mp(self.elan8.last_layer, config=config[index:])
            index += self.mp9.config_len
            self.elan10 = elan(self.mp9.last_layer, n=n, config=config[index:])
            index += self.elan10.config_len
            self.sppcspc11 = sppcspc(self.elan10.last_layer, config=config[index:])
            index += self.sppcspc11.config_len
            self.cbs12 = cbs(config[index - 1], config[index], 1, 1)
            self.cls_head13 = cls_head(config[index], output_class)

    def forward(self, x):
        x = self.cbs0(x)
        x = self.cbs1(x)
        x = self.cbs2(x)
        x = self.cbs3(x)
        x = self.elan4(x)
        x = self.mp5(x)
        x = self.elan6(x)
        x = self.mp7(x)
        x = self.elan8(x)
        x = self.mp9(x)
        x = self.elan10(x)
        x = self.sppcspc11(x)
        x = self.cbs12(x)
        x = self.cls_head13(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', default='s', type=str)
    parser.add_argument('--input_size', default=320, type=int)
    parser.add_argument('--output_class', default=1, type=int)
    args = parser.parse_args()
    model = yolov7_cls(args)
    tensor = torch.rand(2, 3, args.input_size, args.input_size, dtype=torch.float32)
    pred = model(tensor)
    print(model)
    print(pred.shape)
