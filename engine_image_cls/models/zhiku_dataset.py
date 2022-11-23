# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Rockmastar He
Email: 42338705@qq.com
Date: 2021-11-23
Description:
zhiku dataset, support urls and lists of base64
"""
import logging

logger = logging.getLogger(__name__)
import requests
import cv2
import numpy as np
import torch
import base64
from torch.utils.data import Dataset


def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


class UrlDataset(Dataset):
    def __init__(self, urls, transform) -> None:
        self.urls = urls.strip().split(",")
        self.transform = transform

    def __getitem__(self, index):
        url_id = self.urls[index]
        status = "200"
        try:
            resp = requests.get(self.urls[index], timeout=1)
            image = np.asarray(bytearray(resp.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shape = image.shape[:2]
        except Exception as err:
            img = np.zeros((3, 42, 42))
            status = err
            shape = img.shape[:2]
        img = self.transform(image=img)["image"]
        return img, url_id, shape, status

    def __len__(self):
        return len(self.urls)


class Base64Dataset(Dataset):
    def __init__(self, data_list, transform, image_size) -> None:
        self.data_list = data_list
        self.transform = transform
        self.image_size = image_size

    def __getitem__(self, index):
        status = 200
        id = self.data_list[index].get("content_id")
        try:
            image = base64_to_image(self.data_list[index].get("content"))
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as err:
            logger.error(f"INVAILD IMAGE {id}, Exception: {str(err)}")
            # fabricate a blank image
            img = np.zeros((42, 42, 3))
            status = 501
        shape = img.shape
        img = self.transform(image=img)["image"]
        return img, id, shape, status

    def __len__(self):
        return len(self.data_list)


def main():
    image_size = 640
    urls = ",".join(
        [
            "http://spider.nosdn.127.net/e97e3efefcd882e16bd79e5" "643232ded.jpeg"
            for _ in range(32)
        ]
    )
    dataset = UrlDataset(640, urls)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=3,
    )
    for tensor, id, shape, status in dataloader:
        print(id, shape, status)


if __name__ == "__main__":
    main()
