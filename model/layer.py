import torch


class cbs(torch.nn.Module):
    def __init__(self, in_, out_, kernel_size, stride):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.bn = torch.nn.BatchNorm2d(out_)
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class elan(torch.nn.Module):
    def __init__(self, in_, n):
        super().__init__()
        self.cbs0 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
        self.cbs1 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
        self.sequential2 = torch.nn.Sequential(*(cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1) for _ in range(n)))
        self.sequential3 = torch.nn.Sequential(*(cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1) for _ in range(n)))
        self.concat4 = torch.concat
        self.cbs5 = cbs(2 * in_, 2 * in_, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.cbs0(x)
        x1 = self.cbs1(x)
        x2 = self.sequential2(x1)
        x3 = self.sequential3(x2)
        x = self.concat4([x0, x1, x2, x3], dim=1)
        x = self.cbs5(x)
        return x


class mp1(torch.nn.Module):
    def __init__(self, in_):
        super().__init__()
        self.maxpool0 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.cbs1 = cbs(in_, in_ // 2, 1, 1)
        self.cbs2 = cbs(in_, in_ // 2, 1, 1)
        self.cbs3 = cbs(in_ // 2, in_ // 2, 3, 2)
        self.concat4 = torch.concat

    def forward(self, x):
        x0 = self.maxpool0(x)
        x0 = self.cbs1(x0)
        x1 = self.cbs2(x)
        x1 = self.cbs3(x1)
        x = self.concat4([x0, x1], dim=1)
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
