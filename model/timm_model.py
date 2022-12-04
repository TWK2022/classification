import timm
import torch
from model.layer import linear_head


class timm_model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = timm.create_model(args.model, in_chans=args.input_dim, features_only=True, exportable=True)
        out_dim = self.backbone.feature_info.channels()[-1]  # backbone输出有多个，接最后一个输出，并得到其通道数
        self.head = linear_head(out_dim, args.output_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x[-1])
        return x
