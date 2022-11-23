# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rockmastar He
Email: 42338705@qq.com
Date: 2022-02-16
Description:
some utils for image inference
'''
from pathlib import Path
import torch
import time
import random
import cv2
import logging
from zhiku_dataset import base64_to_image

logger = logging.getLogger(__name__)
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def save_image(data_list):
    try:
        saving_dir = Path('/image_data')
        saving_dir.mkdir(parents=True, exist_ok=True)
        for image_data in data_list:
            id = image_data.get("content_id", f"{str(random.randint(0, 1e8))}")
            image = base64_to_image(image_data.get("content"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            saving_path = saving_dir / (f"{id}" + ".jpg")
            cv2.imwrite(str(saving_path), image)
        return

    except Exception as err:
        logger.error(str(err))
        return
        