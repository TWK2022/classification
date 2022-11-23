import torch
import torch.nn as nn
import timm
from thop import profile
from torch.nn.modules.linear import Identity


class ClassificationModel(nn.Module):
    def __init__(self, categories, scale="b7") -> None:
        super(ClassificationModel, self).__init__()
        # use efficient net b0 scale as baseline
        self.backbone = timm.create_model(
            f"efficientnet_{scale}",
            pretrained=False,
            features_only=True,
            exportable=True)
        self.main_channel = self.backbone.feature_info.channels()[-1]
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.main_channel, self.main_channel // 2),
            nn.ReLU(),
            nn.Linear(self.main_channel // 2, categories),
            nn.Softmax(dim=1)
        )
        self.proj_head = nn.Sequential(
            nn.Linear(self.main_channel, self.main_channel),
            nn.BatchNorm1d(self.main_channel),
            nn.ReLU(),
            nn.Linear(self.main_channel, self.main_channel),
            nn.BatchNorm1d(self.main_channel),
            nn.ReLU(),
            nn.Linear(self.main_channel, self.main_channel),
            nn.BatchNorm1d(self.main_channel),
            nn.ReLU(),
            nn.Linear(self.main_channel, self.main_channel // 2),
        )

    def forward(self, x, cls_job=True):
        if cls_job:
            return self.forward_cls(x)
        else:
            return self.forward_rl(x)

    def forward_cls(self, x):
        """
        Implement the supervised classificaiton learning job
        """
        feature = self.backbone(x)
        feature = self.flatten(feature[-1])
        logits = self.classifier(feature)

        return logits

    def forward_rl(self, x):
        """
        Implement the representation learning job
        """
        feature = self.backbone(x)
        feature = self.flatten(feature[-1])
        representations = self.proj_head(feature)

        return representations

    def extract(self, x):
        """
        Implement feature extractor job
        """
        feature = self.backbone(x)
        feature = self.flatten(feature[-1])

        return feature

    def _profile_model(self):
        # profiling the model with size and FLOPs
        # and the shape of output
        x = torch.randn(2, 3, 288, 288)
        flops, params = profile(self, inputs=(x,), verbose=True)
        output = self.forward(x)
        print("\n" + f"[INFO] This model has {params / (1000 ** 2)} M parameters" + "\n")
        print("\n" + f"[INFO] with {flops / (1000 ** 3)} Billion FLOPs" + "\n")
        print("\n" + f"[INFO] the shape of output is {output.shape}" + "\n")


class MultiLabelModel(ClassificationModel):
    def __init__(self, categories, scale="b7") -> None:
        super().__init__(categories, scale=scale)
        self.proj_head = Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.main_channel, categories),
            # nn.ReLU(),
            # nn.Linear(self.main_channel // 2, categories),
        )


def main():
    model = ClassificationModel(categories=3)
    print(model)
    model._profile_model()


if __name__ == "__main__":
    main()
