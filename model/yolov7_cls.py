# 根据yolov7改编: https://github.com/WongKinYiu/yolov7
import torch
from model.layer import cbs, elan, mp, sppcspc, linear_head


class yolov7_cls(torch.nn.Module):
    def __init__(self, args, config=None):
        super().__init__()
        dim_dict = {'n': 8, 's': 16, 'm': 32, 'l': 64}
        n_dict = {'n': 1, 's': 1, 'm': 2, 'l': 3}
        dim = dim_dict[args.model_type]
        n = n_dict[args.model_type]
        output_class = args.output_class
        # 网络结构
        if config is None:  # 正常版本
            self.l0 = cbs(3, dim, 1, 1)
            self.l1 = cbs(dim, 2 * dim, 3, 2)  # input_size/2
            self.l2 = cbs(2 * dim, 2 * dim, 1, 1)
            self.l3 = cbs(2 * dim, 4 * dim, 3, 2)  # input_size/4
            self.l4 = elan(4 * dim, 8 * dim, n)
            self.l5 = mp(8 * dim, 8 * dim)  # input_size/8
            self.l6 = elan(8 * dim, 16 * dim, n)
            self.l7 = mp(16 * dim, 16 * dim)  # input_size/16
            self.l8 = elan(16 * dim, 32 * dim, n)
            self.l9 = mp(32 * dim, 32 * dim)  # input_size/32
            self.l10 = elan(32 * dim, 32 * dim, n)
            self.l11 = sppcspc(32 * dim, 16 * dim)
            self.l12 = cbs(16 * dim, 8 * dim, 1, 1)
            self.linear_head = linear_head(8 * dim, output_class)
        else:  # 剪枝版本
            self.l0 = cbs(3, config[0], 1, 1)
            self.l1 = cbs(config[0], config[1], 3, 2)  # input_size/2
            self.l2 = cbs(config[1], config[2], 1, 1)
            self.l3 = cbs(config[2], config[3], 3, 2)  # input_size/4
            self.l4 = elan(config[3], None, n, config[4:7 + 2 * n])
            self.l5 = mp(config[6 + 2 * n], None, config[7 + 2 * n:10 + 2 * n])  # input_size/8
            self.l6 = elan(config[7 + 2 * n] + config[9 + 2 * n], None, n, config[10 + 2 * n:13 + 4 * n])
            self.l7 = mp(config[12 + 4 * n], None, config[13 + 4 * n:16 + 4 * n])  # input_size/16
            self.l8 = elan(config[13 + 4 * n] + config[15 + 4 * n], None, n, config[16 + 4 * n:19 + 6 * n])
            self.l9 = mp(config[18 + 6 * n], None, config[19 + 6 * n:22 + 6 * n])  # input_size/32
            self.l10 = elan(config[19 + 6 * n] + config[21 + 6 * n], None, n, config[22 + 6 * n:25 + 8 * n])
            self.l11 = sppcspc(config[24 + 8 * n], None, config[25 + 8 * n:32 + 8 * n])
            self.l12 = cbs(config[31 + 8 * n], config[32 + 8 * n], 1, 1)
            self.linear_head = linear_head(config[32 + 8 * n], output_class)

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.linear_head(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', default='n', type=str)
    parser.add_argument('--input_size', default=320, type=int)
    parser.add_argument('--output_class', default=1, type=int)
    args = parser.parse_args()
    model = yolov7_cls(args, False)
    tensor = torch.rand(2, 3, args.input_size, args.input_size, dtype=torch.float32)
    pred = model(tensor)
    print(model)
    print(pred.shape)
