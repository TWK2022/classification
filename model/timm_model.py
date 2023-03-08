import timm
import torch
from model.layer import image_deal, linear_head


class timm_model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_deal = image_deal()
        self.backbone = timm.create_model(args.model, in_chans=3, features_only=True, exportable=True)
        out_dim = self.backbone.feature_info.channels()[-1]  # backbone输出有多个，接最后一个输出，并得到其通道数
        self.linear_head = linear_head(out_dim, args.output_class)

    def forward(self, x):
        x = self.image_deal(x)
        x = self.backbone(x)
        x = self.linear_head(x[-1])
        return x


if __name__ == '__main__':
    import argparse
    from layer import image_deal, linear_head

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='efficientnetv2_s', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    parser.add_argument('--output_class', default=2, type=int)
    args = parser.parse_args()
    model = timm_model(args)
    print(model)
    tensor = torch.rand(2, args.input_size, args.input_size, 3, dtype=torch.float32)
    pred = model(tensor)
    print(pred.shape)
