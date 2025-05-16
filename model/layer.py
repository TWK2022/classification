import torch


class cbs(torch.nn.Module):
    def __init__(self, in_, out_, kernel_size, stride):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                                    bias=False)
        self.bn = torch.nn.BatchNorm2d(out_, eps=0.001, momentum=0.03)
        self.silu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class concat(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.concat = torch.concat
        self.dim = dim

    def forward(self, x):
        x = self.concat(x, dim=self.dim)
        return x


class residual(torch.nn.Module):  # in_->in_，len->len
    def __init__(self, in_, config=None):
        super().__init__()
        if config is None:  # 正常版本
            self.cbs0 = cbs(in_, in_, kernel_size=1, stride=1)
            self.cbs1 = cbs(in_, in_, kernel_size=3, stride=1)
        else:  # 剪枝版本: len(config) = 2
            self.cbs0 = cbs(in_, config[0], kernel_size=1, stride=1)
            self.cbs1 = cbs(config[0], in_, kernel_size=3, stride=1)
            self.config_len = 2  # 参数层数
            self.last_layer = in_  # 最后一层参数

    def forward(self, x):
        x0 = self.cbs0(x)
        x0 = self.cbs1(x0)
        x = x + x0
        return x + x0


class elan(torch.nn.Module):  # in_->out_，len->len
    def __init__(self, in_, out_=None, n=1, config=None):
        super().__init__()
        if config is None:  # 正常版本
            self.cbs0 = cbs(in_, out_ // 4, kernel_size=1, stride=1)
            self.cbs1 = cbs(in_, out_ // 4, kernel_size=1, stride=1)
            self.sequential2 = torch.nn.Sequential(
                *(cbs(out_ // 4, out_ // 4, kernel_size=3, stride=1) for _ in range(n)))
            self.sequential3 = torch.nn.Sequential(
                *(cbs(out_ // 4, out_ // 4, kernel_size=3, stride=1) for _ in range(n)))
            self.concat4 = concat()
            self.cbs5 = cbs(out_, out_, kernel_size=1, stride=1)
        else:  # 剪枝版本: len(config) = 4 + 2 * n
            self.cbs0 = cbs(in_, config[0], kernel_size=1, stride=1)
            self.cbs1 = cbs(in_, config[1], kernel_size=1, stride=1)
            self.sequential2 = torch.nn.Sequential(
                *(cbs(config[1 + _], config[1 + _ + 1], kernel_size=3, stride=1) for _ in range(n)))
            self.sequential3 = torch.nn.Sequential(
                *(cbs(config[1 + n + _], config[1 + n + _ + 1], kernel_size=3, stride=1) for _ in range(n)))
            self.concat4 = concat()
            self.cbs5 = cbs(config[0] + config[1] + config[1 + n] + config[1 + 2 * n], config[2 + 2 * n],
                            kernel_size=1, stride=1)
            self.config_len = 3 + 2 * n  # 参数层数
            self.last_layer = config[2 + 2 * n]  # 最后一层参数

    def forward(self, x):
        x0 = self.cbs0(x)
        x1 = self.cbs1(x)
        x2 = self.sequential2(x1)
        x3 = self.sequential3(x2)
        x = self.concat4([x0, x1, x2, x3])
        x = self.cbs5(x)
        return x


class elan_h(torch.nn.Module):  # in_->out_，len->len
    def __init__(self, in_, out_=None, config=None):
        super().__init__()
        if config is None:  # 正常版本
            self.cbs0 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
            self.cbs1 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
            self.cbs2 = cbs(in_ // 2, in_ // 4, kernel_size=3, stride=1)
            self.cbs3 = cbs(in_ // 4, in_ // 4, kernel_size=3, stride=1)
            self.cbs4 = cbs(in_ // 4, in_ // 4, kernel_size=3, stride=1)
            self.cbs5 = cbs(in_ // 4, in_ // 4, kernel_size=3, stride=1)
            self.concat6 = concat()
            self.cbs7 = cbs(2 * in_, out_, kernel_size=1, stride=1)
        else:  # 剪枝版本: len(config) = 7
            self.cbs0 = cbs(in_, config[0], kernel_size=1, stride=1)
            self.cbs1 = cbs(in_, config[1], kernel_size=1, stride=1)
            self.cbs2 = cbs(config[1], config[2], kernel_size=3, stride=1)
            self.cbs3 = cbs(config[2], config[3], kernel_size=3, stride=1)
            self.cbs4 = cbs(config[3], config[4], kernel_size=3, stride=1)
            self.cbs5 = cbs(config[4], config[5], kernel_size=3, stride=1)
            self.concat6 = concat()
            self.cbs7 = cbs(config[0] + config[1] + config[2] + config[3] + config[4] + config[5], config[6],
                            kernel_size=1, stride=1)
            self.config_len = 7  # 参数层数
            self.last_layer = config[6]  # 最后一层参数

    def forward(self, x):
        x0 = self.cbs0(x)
        x1 = self.cbs1(x)
        x2 = self.cbs2(x1)
        x3 = self.cbs3(x2)
        x4 = self.cbs4(x3)
        x5 = self.cbs5(x4)
        x = self.concat6([x0, x1, x2, x3, x4, x5])
        x = self.cbs7(x)
        return x


class mp(torch.nn.Module):  # in_->out_，len->len//2
    def __init__(self, in_, out_=None, config=None):
        super().__init__()
        if config is None:  # 正常版本
            self.maxpool0 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
            self.cbs1 = cbs(in_, out_ // 2, 1, 1)
            self.cbs2 = cbs(in_, out_ // 2, 1, 1)
            self.cbs3 = cbs(out_ // 2, out_ // 2, 3, 2)
            self.concat4 = concat(dim=1)
        else:  # 剪枝版本: len(config) = 3
            self.maxpool0 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
            self.cbs1 = cbs(in_, config[0], 1, 1)
            self.cbs2 = cbs(in_, config[1], 1, 1)
            self.cbs3 = cbs(config[1], config[2], 3, 2)
            self.concat4 = concat(dim=1)
            self.config_len = 3  # 参数层数
            self.last_layer = config[0] + config[2]  # 最后一层参数

    def forward(self, x):
        x0 = self.maxpool0(x)
        x0 = self.cbs1(x0)
        x1 = self.cbs2(x)
        x1 = self.cbs3(x1)
        x = self.concat4([x0, x1])
        return x


class sppf(torch.nn.Module):  # in_->out_，len->len
    def __init__(self, in_, out_=None, config=None):
        super().__init__()
        if config is None:  # 正常版本
            self.cbs0 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
            self.MaxPool2d1 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1)
            self.MaxPool2d2 = torch.nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1)
            self.MaxPool2d3 = torch.nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1)
            self.concat4 = concat(dim=1)
            self.cbs5 = cbs(2 * in_, out_, kernel_size=1, stride=1)
        else:  # 剪枝版本: len(config) = 2
            self.cbs0 = cbs(in_, config[0], kernel_size=1, stride=1)
            self.MaxPool2d1 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1)
            self.MaxPool2d2 = torch.nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1)
            self.MaxPool2d3 = torch.nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1)
            self.concat4 = concat(dim=1)
            self.cbs5 = cbs(4 * config[0], config[1], kernel_size=1, stride=1)
            self.config_len = 2  # 参数层数
            self.last_layer = config[1]  # 最后一层参数

    def forward(self, x):
        x = self.cbs0(x)
        x0 = self.MaxPool2d1(x)
        x1 = self.MaxPool2d2(x0)
        x2 = self.MaxPool2d3(x1)
        x = self.concat4([x, x0, x1, x2])
        x = self.cbs5(x)
        return x


class sppcspc(torch.nn.Module):  # in_->out_，len->len
    def __init__(self, in_, out_=None, config=None):
        super().__init__()
        if config is None:  # 正常版本
            self.cbs0 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
            self.cbs1 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
            self.cbs2 = cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1)
            self.cbs3 = cbs(in_ // 2, in_ // 2, kernel_size=1, stride=1)
            self.MaxPool2d4 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1)
            self.MaxPool2d5 = torch.nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1)
            self.MaxPool2d6 = torch.nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1)
            self.concat7 = concat(dim=1)
            self.cbs8 = cbs(2 * in_, in_ // 2, kernel_size=1, stride=1)
            self.cbs9 = cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1)
            self.concat10 = concat(dim=1)
            self.cbs11 = cbs(in_, out_, kernel_size=1, stride=1)
        else:  # 剪枝版本: len(config) = 7
            self.cbs0 = cbs(in_, config[0], kernel_size=1, stride=1)
            self.cbs1 = cbs(in_, config[1], kernel_size=1, stride=1)
            self.cbs2 = cbs(config[1], config[2], kernel_size=3, stride=1)
            self.cbs3 = cbs(config[2], config[3], kernel_size=1, stride=1)
            self.MaxPool2d4 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1)
            self.MaxPool2d5 = torch.nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1)
            self.MaxPool2d6 = torch.nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1)
            self.concat7 = concat(dim=1)
            self.cbs8 = cbs(4 * config[3], config[4], kernel_size=1, stride=1)
            self.cbs9 = cbs(config[4], config[5], kernel_size=3, stride=1)
            self.concat10 = concat(dim=1)
            self.cbs11 = cbs(config[0] + config[5], config[6], kernel_size=1, stride=1)
            self.config_len = 7  # 参数层数
            self.last_layer = config[6]  # 最后一层参数

    def forward(self, x):
        x0 = self.cbs0(x)
        x1 = self.cbs1(x)
        x1 = self.cbs2(x1)
        x1 = self.cbs3(x1)
        x4 = self.MaxPool2d4(x1)
        x5 = self.MaxPool2d5(x1)
        x6 = self.MaxPool2d6(x1)
        x = self.concat7([x1, x4, x5, x6])
        x = self.cbs8(x)
        x = self.cbs9(x)
        x = self.concat10([x, x0])
        x = self.cbs11(x)
        return x


class head(torch.nn.Module):  # in_->(batch, 3, output_size, output_size, 5+output_class))，len->len
    def __init__(self, in_, output_size, output_class, layer=3):
        super().__init__()
        self.layer = layer
        self.output_size = output_size
        self.output_class = output_class
        self.output = torch.nn.Conv2d(in_, layer * (5 + output_class), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.output(x).reshape(x.shape[0], self.layer, self.output_size, self.output_size, 5 + self.output_class)
        return x


class cls_head(torch.nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.avgpool0 = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten1 = torch.nn.Flatten()
        self.dropout2 = torch.nn.Dropout(0.2)
        self.linear3 = torch.nn.Linear(in_, in_ // 2)
        self.silu4 = torch.nn.SiLU()
        self.dropout5 = torch.nn.Dropout(0.2)
        self.linear6 = torch.nn.Linear(in_ // 2, out_)
        self.sigmoid7 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool0(x)
        x = self.flatten1(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.silu4(x)
        x = self.dropout5(x)
        x = self.linear6(x)
        x = self.sigmoid7(x)
        return x


class decode(torch.nn.Module):  # (cx,cy,w,h,confidence...)原始输出->(cx,cy,w,h,confidence...)真实坐标
    def __init__(self, args):
        super().__init__()
        self.anchor = args.input_size * torch.tensor(args.anchor)
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.output_layer = args.output_layer
        self.stride = [self.input_size / _ for _ in self.output_size]
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        grid = [torch.arange(_, device=x[0].device) for _ in self.output_size]
        # 遍历每一个大层
        x_decode = []
        for index, layer in enumerate(x):
            # 中心坐标[0-1]->[-0.5-1.5]->[-0.5*stride-1.5*stride]
            layer = self.sigmoid(layer)  # 归一化
            new_layer = layer.clone()  # 防止inplace丢失梯度
            new_layer[..., 0] = (2 * layer[..., 0] - 0.5 + grid[index]) * self.stride[index]
            new_layer[..., 1] = (2 * layer[..., 1] - 0.5 + grid[index].unsqueeze(1)) * self.stride[index]
            # 遍历每一个大层中的小层
            for index_ in range(self.output_layer[index]):  # [0-1]->[0-4*anchor]
                new_layer[:, index_, ..., 2] = 4 * layer[:, index_, ..., 2] ** 2 * self.anchor[index][index_][0]
                new_layer[:, index_, ..., 3] = 4 * layer[:, index_, ..., 3] ** 2 * self.anchor[index][index_][1]
            x_decode.append(new_layer)
        x_list = []
        batch, _, _, _, number = x_decode[0].shape
        for index in range(batch):  # 每个批量单独合并在一起
            x_list.append(torch.concat([_[index].reshape(-1, number) for _ in x_decode], dim=0))
        x = torch.stack(x_list, dim=0)
        return x


class deploy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x
