import torch


class image_deal(torch.nn.Module):  # 归一化、减均值、除以方差
    def __init__(self):
        super().__init__()
        self.rgb_mean = (0.406, 0.456, 0.485)
        self.rgb_std = (0.225, 0.224, 0.229)

    def forward(self, x):
        x[..., 0] = (x[..., 0] / 255 - self.rgb_mean[0]) / self.rgb_std[0]
        x[..., 1] = (x[..., 1] / 255 - self.rgb_mean[1]) / self.rgb_std[1]
        x[..., 2] = (x[..., 2] / 255 - self.rgb_mean[2]) / self.rgb_std[2]
        x = x.permute(0, 3, 1, 2)
        return x


class concat(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.concat = torch.concat
        self.dim = dim

    def forward(self, x):
        x = self.concat(x, dim=self.dim)
        return x


class cbs(torch.nn.Module):
    def __init__(self, in_, out_, kernel_size, stride):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.bn = torch.nn.BatchNorm2d(out_, eps=0.001, momentum=0.03)
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class elan(torch.nn.Module):  # in_->out_，len->len
    def __init__(self, in_, out_, n):
        super().__init__()
        self.cbs0 = cbs(in_, out_ // 4, kernel_size=1, stride=1)
        self.cbs1 = cbs(in_, out_ // 4, kernel_size=1, stride=1)
        self.sequential2 = torch.nn.Sequential(*(cbs(out_ // 4, out_ // 4, kernel_size=3, stride=1) for _ in range(n)))
        self.sequential3 = torch.nn.Sequential(*(cbs(out_ // 4, out_ // 4, kernel_size=3, stride=1) for _ in range(n)))
        self.concat4 = concat()
        self.cbs5 = cbs(out_, out_, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.cbs0(x)
        x1 = self.cbs1(x)
        x2 = self.sequential2(x1)
        x3 = self.sequential3(x2)
        x = self.concat4([x0, x1, x2, x3])
        x = self.cbs5(x)
        return x


class mp(torch.nn.Module):  # in_->out_，len->len//2
    def __init__(self, in_, out_):
        super().__init__()
        self.maxpool0 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.cbs1 = cbs(in_, out_ // 2, 1, 1)
        self.cbs2 = cbs(in_, out_ // 2, 1, 1)
        self.cbs3 = cbs(out_ // 2, out_ // 2, 3, 2)
        self.concat4 = concat(dim=1)

    def forward(self, x):
        x0 = self.maxpool0(x)
        x0 = self.cbs1(x0)
        x1 = self.cbs2(x)
        x1 = self.cbs3(x1)
        x = self.concat4([x0, x1])
        return x


class sppcspc(torch.nn.Module):  # in_->out_，len->len
    def __init__(self, in_, out_):
        super().__init__()
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



class linear_head(torch.nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.avgpool0 = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten1 = torch.nn.Flatten()
        self.Dropout2 = torch.nn.Dropout(0.2)
        self.linear3 = torch.nn.Linear(in_, in_ // 2)
        self.silu4 = torch.nn.SiLU()
        self.Dropout5 = torch.nn.Dropout(0.2)
        self.linear6 = torch.nn.Linear(in_ // 2, out_)
        if out_ == 1:
            self.normalization7 = torch.nn.Sigmoid()
        else:
            self.normalization7 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avgpool0(x)
        x = self.flatten1(x)
        x = self.Dropout2(x)
        x = self.linear3(x)
        x = self.silu4(x)
        x = self.Dropout5(x)
        x = self.linear6(x)
        x = self.normalization7(x)
        return x
