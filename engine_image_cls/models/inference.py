# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Rockmastar He
Email: 42338705@qq.com
Date: 2022-02-16
Description:
分类模型推理代码
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import requests
import os
import logging
from io import BytesIO
from threading import Thread

import torch
import numpy as np
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import utils
import zhiku_dataset

logger = logging.getLogger(__name__)


def assemble_tags(tags, white_list=None):
    result = []
    for tag in tags:
        tag_name = str(tag[0].strip())
        if white_list and (tag_name not in white_list):
            # get rid of those tags which are not in white list
            continue
        else:
            result.append({"tag_id": tag_name, "confidence": f"{tag[1]:.3f}"})
    return result


class ImageMultiLabelPredictor:
    """
    图像多标签模型分类器
    """

    def __init__(self, args, task=None, filter=None) -> None:
        super().__init__()
        if args.device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.ckpt = torch.load(args.weight, map_location=torch.device("cpu"))
        self.model = self.ckpt.get("model")
        self.model.eval()
        self.model.to(self.device, non_blocking=args.latch)
        if self.device != "cpu":
            self.model.half()
        # preprocess
        self.input_size = args.input_size
        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size=self.input_size),
                A.PadIfNeeded(min_height=self.input_size, min_width=self.input_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.class_names = np.array(self.ckpt.get("class_names"))
        self.thresh_hold = args.threshold
        self.batch_size = args.batch
        self.non_blocking = args.latch
        self.pin_memory = args.latch
        self.num_worker = args.num_worker
        self.white_list = None
        self.save_image = args.save_image

    def get_machine_result_from_batch(self, contents, timeout=1):  # {"content_id":1,"content":base64}
        if self.save_image:
            # save request image to local disk
            t = Thread(target=utils.save_image, args=(contents,))
            t.start()
        dataset = zhiku_dataset.Base64Dataset(contents, self.transform, self.input_size)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=self.batch_size,
                                                 pin_memory=self.pin_memory)
        ret = self.predict_batches(dataloader)
        ret = {
            "results": ret,
            "total": len(contents),
        }
        return ret

    @torch.no_grad()
    def predict_batches(self, dataloader, export_dataframe=False):
        """批量预测图片"""
        dt = [0.0 for _ in range(4)]
        ret = []
        seen = 0
        for batch_i, (img, unique_ids, shapes, status) in tqdm(enumerate(dataloader)):
            # inference
            t1 = utils.time_sync()
            img = img.to(self.device, non_blocking=self.non_blocking)
            img = img.half() if self.device == 'cuda' else img.float()  # uint8 to fp16/32
            # nb, _, height, width = img.shape  # batch size, channels, height, width
            t2 = utils.time_sync()
            dt[0] += t2 - t1
            # Run model
            logits = self.model(img).sigmoid_().float().detach().to("cpu").numpy()
            dt[1] += utils.time_sync() - t2
            # post process
            t3 = utils.time_sync()
            # by per image
            for i, logit in enumerate(logits):
                seen += 1
                raw_imgsz = (shapes[0][i].item(), shapes[1][i].item())
                result = {}
                result.update(
                    {
                        "id": unique_ids[i],
                        "size": "%g,%g" % raw_imgsz,
                        "status": status[i].item(),
                    }
                )
                label = self.class_names[logit >= self.thresh_hold]
                score = logit[logit >= self.thresh_hold]
                result["tags"] = assemble_tags(
                    list(zip(*(label, score))), white_list=self.white_list
                )
                ret.append(result)
            dt[2] += utils.time_sync() - t3
        # logger.info(f"{time.time() - start} seconds for all the batch")
        # t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
        # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms '
        # 'decode per image' % t)
        if export_dataframe:
            import pandas as pd
            dataframe = pd.DataFrame(columns=["ids", "tags"])
            dataframe["ids"] = [res["id"] for res in ret]
            dataframe["tags"] = [[tag["tag"] for tag in res["tags"]] for res in ret]
            dataframe.to_excel("output.xlsx", engine="openpyxl")

        return ret

    def initialize(self):
        self.batch_size = int(os.getenv("BATCH_SIZE"))
        CPU_NUM = int(os.getenv("CPU_NUM"))  # 这里设置成你想运行的CPU个数
        self.USE_WHITELIST = bool(os.getenv("WHITE_LIST"))
        os.environ["OMP_NUM_THREADS"] = str(CPU_NUM)
        os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_NUM)
        os.environ["MKL_NUM_THREADS"] = str(CPU_NUM)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_NUM)
        os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_NUM)
        self.save_image = os.getenv("SAVE_IMAGE", False)
        torch.set_num_threads(CPU_NUM)

    def inference_with_streamer(self, contents):
        """为了使用streamer输入和输出都应该是严格对应关系的数组"""
        dataset = zhiku_dataset.Base64Dataset(contents, self.transform, self.input_size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
        )
        ret = self.predict_batches(dataloader)
        return ret

    def predict_single_image(self, image):
        """
        正向推理，获得图片类别
        Args:
            image(RGB Array): 图像数组，注意需要RGB
        Returns:
            list of (tag name, confidece)
        """
        image_t = self.transform(image=image)["image"]
        image_t = torch.unsqueeze(image_t, 0)
        with torch.no_grad():
            logits = (
                self.model(image_t)
                    .sigmoid_()
                    .float()
                    .detach()
                    .to("cpu")
                    .numpy()
                    .squeeze()
            )
            labels = self.class_names[logits >= self.thresh_hold]
            scores = logits[logits >= self.thresh_hold]
        return list(zip(*(labels, scores)))

    def predict_from_url(self, url):
        """
        读取URL，推理图片，返回类别
        """
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = np.array(image)
        result = self.predict_single_image(image)
        return result

    def predict_from_pth(self, path):
        """
        读取本地路径，推理图片，返回类别
        """
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        result = self.predict_single_image(image)
        return result
