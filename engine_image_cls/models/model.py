#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rockstar He
# Date: 2021-09-14
# Description:


import torch
import torch.nn as nn
import timm
from thop import profile
from torch.nn.modules.linear import Identity


# class ClassificationModel(nn.Module):
#     def __init__(self, model, categories) -> None:
#         super(ClassificationModel, self).__init__()
#         self.backbone = timm.create_model(
#             model,
#             pretrained=False,
#             features_only=True,
#             exportable=True)
#         self.main_channel = self.backbone.feature_info.channels()[-1]
#         self.flatten = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten()
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(self.main_channel, self.main_channel // 2),
#             nn.ReLU(),
#             nn.Linear(self.main_channel // 2, categories),
#             nn.Softmax(dim=1)
#         )
#         self.proj_head = nn.Sequential(
#             nn.Linear(self.main_channel, self.main_channel),
#             nn.BatchNorm1d(self.main_channel),
#             nn.ReLU(),
#             nn.Linear(self.main_channel, self.main_channel),
#             nn.BatchNorm1d(self.main_channel),
#             nn.ReLU(),
#             nn.Linear(self.main_channel, self.main_channel),
#             nn.BatchNorm1d(self.main_channel),
#             nn.ReLU(),
#             nn.Linear(self.main_channel, self.main_channel // 2),
#         )
#
#     def forward(self, x, cls_job=True):
#         if cls_job:
#             return self.forward_cls(x)
#         else:
#             return self.forward_rl(x)
#
#     def forward_cls(self, x):
#         """
#         Implement the supervised classificaiton learning job
#         """
#         feature = self.backbone(x)
#         feature = self.flatten(feature[-1])
#         logits = self.classifier(feature)
#         return logits
#
#     def forward_rl(self, x):
#         """
#         Implement the representation learning job
#         """
#         feature = self.backbone(x)
#         feature = self.flatten(feature[-1])
#         representations = self.proj_head(feature)
#         return representations
#
#     def extract(self, x):
#         """
#         Implement feature extractor job
#         """
#         feature = self.backbone(x)
#         feature = self.flatten(feature[-1])
#         return feature
#
#     def _profile_model(self):
#         # profiling the model with size and FLOPs
#         # and the shape of output
#         x = torch.randn(1, 3, 288, 288)
#         flops, params = profile(self, inputs=(x,), verbose=True)
#         output = self.forward(x)
#         print("\n" + f"[INFO] This model has {params / (1000 ** 2)} M parameters" + "\n")
#         print("\n" + f"[INFO] with {flops / (1000 ** 3)} Billion FLOPs" + "\n")
#         print("\n" + f"[INFO] the shape of output is {output.shape}" + "\n")


class ClassificationModel(nn.Module):
    def __init__(self, model, categories) -> None:
        super(ClassificationModel, self).__init__()
        self.backbone = timm.create_model(
            model,
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x[-1])
        x = self.classifier(x)
        return x

    def _profile_model(self):
        x = torch.randn(1, 3, 288, 288)
        flops, params = profile(self, inputs=(x,), verbose=True)
        output = self.forward(x)
        print("\n" + f"[INFO] This model has {params / (1000 ** 2)} M parameters" + "\n")
        print("\n" + f"[INFO] with {flops / (1000 ** 3)} Billion FLOPs" + "\n")
        print("\n" + f"[INFO] the shape of output is {output.shape}" + "\n")


def main():
    model = ClassificationModel(categories=2)
    model._profile_model()


if __name__ == "__main__":
    main()
