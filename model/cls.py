import torch
from model.layer import cbs, elan, mp1, linear_head


class cls(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dim_dict = {'s': 8, 'm': 16, 'l': 32}
        n_dict = {'s': 1, 'm': 2, 'l': 3}
        dim = dim_dict[args.model_type]
        n = n_dict[args.model_type]
        self.l0 = cbs(args.input_dim, dim, 1, 1)
        self.l1 = cbs(dim, 2 * dim, 3, 2)  # input_size/2
        self.l2 = cbs(2 * dim, 2 * dim, 1, 1)
        self.l3 = cbs(2 * dim, 4 * dim, 3, 2)  # input_size/4
        self.l4 = elan(4 * dim, n)
        self.l5 = mp1(8 * dim)  # input_size/8
        self.l6 = elan(8 * dim, n)
        self.l7 = mp1(16 * dim)  # input_size/16
        self.l8 = elan(16 * dim, n)
        self.l9 = mp1(32 * dim)  # input_size/32
        self.l10 = linear_head(32 * dim, args.output_class)

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
        return x


if __name__ == '__main__':
    import argparse
    from layer import cbs, elan, mp1, linear_head

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', default='s', type=str)
    parser.add_argument('--batch', default=4, type=int)
    parser.add_argument('--input_size', default=640, type=int)
    parser.add_argument('--input_dim', default=3, type=int)
    args = parser.parse_args()
    model = cls(args)
    print(model)
    tensor = torch.rand(2, args.input_dim, args.input_size, args.input_size, dtype=torch.float32)
    pred = model(tensor)
    print(pred.shape)
