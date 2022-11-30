import base64
import cv2
import time
import torch
import argparse
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import inference, zhiku_dataset

import os
# os.environ["OMP_NUM_THREADS"] = str(2)
# os.environ["MKL_NUM_THREADS"] = str(2)
# torch.set_num_threads(2)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--weight', default='terror.pt', type=str, help='|模型位置|')
parser.add_argument('--threshold', default=0.5,
                    type=float, help='|判断为目标类别的阈值|')
parser.add_argument('--input_size', default=640, type=int, help='|输入图片大小|')
parser.add_argument('--save_image', default=False, type=bool, help='|是否保存结果|')
parser.add_argument('--device', default='cuda', type=str, help='|设备:cpu/cuda|')
parser.add_argument('--latch', default=False, type=bool,
                    help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int,
                    help='|有多少个进程处理数据，0为全加入到主进程|')
args = parser.parse_args()
args.n = 100


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
max_pixel_value = 255.0
transform = A.Compose(
    [
        A.LongestMaxSize(max_size=args.input_size),
        A.PadIfNeeded(min_height=args.input_size, min_width=args.input_size),
        A.Normalize(),
        ToTensorV2(),
    ]
)


def resize(args, image):
    w0 = image.shape[1]
    h0 = image.shape[0]
    if w0 == h0:
        image = cv2.resize(image, (args.input_size, args.input_size))
    elif w0 > h0:  # 宽大于高
        w = args.input_size
        h = int(w / w0 * h0)
        image = cv2.resize(image, (w, h))
        add_y = (w - h) // 2
        image = cv2.copyMakeBorder(image, add_y, w - h - add_y, 0, 0, cv2.BORDER_CONSTANT, value=(126, 126, 126))
    else:  # 宽小于高
        h = args.input_size
        w = int(h / h0 * w0)
        image = cv2.resize(image, (w, h))
        add_x = (h - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, add_x, h - w - add_x, cv2.BORDER_CONSTANT, value=(126, 126, 126))
    return image


def test(args, data_list, model, batchsize):
    with torch.no_grad():
        test_dataloader = torch.utils.data.DataLoader(
            zhiku_dataset.Base64Dataset(data_list=data_list, transform=transform, image_size=args.input_size),
            batch_size=batchsize, shuffle=False, drop_last=False, pin_memory=args.latch)
        for i, (tensor, id, shape, status) in enumerate(test_dataloader):
            tensor = tensor.to(args.device, non_blocking=args.latch)


def test1(args, data_list, model, batchsize):
    with torch.no_grad():
        for i in range(0, len(data_list), batchsize):
            images = [resize(args, zhiku_dataset.base64_to_image(data['content']))
                      for data in data_list[i: i + batchsize]]
            images = np.array(images)
            tensor = (torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255) \
                .to(args.device, non_blocking=args.latch)

# 模拟服务原始输出
image = cv2.imread('001.jpg')
image_array = cv2.imencode('.jpg', image)[1]
image_bytes = image_array.tostring()
image_base64 = base64.b64encode(image_bytes)
model = 1
# # 模型加载
# print('| 使用{} | 模型加载中... |'.format(args.device))
# model_dict = torch.load(args.weight)
# model = model_dict['model'].to(args.device)
# print('| 模型加载完毕! |')
# model.eval()
# # 模型预热
# a = base64.b64decode(image_base64)
# a = np.fromstring(a, dtype=np.uint8)
# a = cv2.imdecode(a, cv2.IMREAD_COLOR)
# a = resize(args, a)
# a = torch.tensor(a, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(args.device)
# for i in range(3):
#     model(a)

for batch_size in [1, 16]:
    image_list = [
        {
            "content_id": 'xxxx',
            "content": image_base64
        } for _ in range(args.n * batch_size)
    ]
    # tensor = torch.rand((8, 3, args.input_size, args.input_size), dtype=torch.float32).to('cuda', non_blocking=args.latch)
    # 原始数据
    start_time = time.time()
    test(args, image_list, model, batchsize=batch_size)
    end_time = time.time()
    print(f"原始数据读取方式测试结果:")
    print('| 批量{} | 平均每张耗时{:.5f} |'.format(
        batch_size, (end_time - start_time) / args.n / batch_size))
    # 修改后数据
    start_time = time.time()
    pred = test1(args, image_list, model, batch_size)
    end_time = time.time()
    print(f"修改后的批量数据处理读取方式结果:")
    print('| 批量{} | 平均每张耗时{:.5f} |'.format(
        batch_size, (end_time - start_time) / args.n / batch_size))

start_time = time.time()
for i in range(100):
    image_bytes = base64.b64decode(image_base64)
    image_array = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
end_time = time.time()
print('base64转换时间:{:.4f}'.format((end_time - start_time) / 100))

start_time = time.time()
for i in range(100):
    image = np.array(image)
    img = transform(image=image)["image"]
end_time = time.time()
print('transform转换时间:{:.4f}'.format((end_time - start_time) / 100))

start_time = time.time()
for i in range(100):
    img = torch.tensor(resize(args, image))
    img = (img - mean * max_pixel_value) / (std * max_pixel_value)
end_time = time.time()
print('resize等转换时间:{:.4f}'.format((end_time - start_time) / 100))

tensor = torch.rand((1, 3, args.input_size, args.input_size), dtype=torch.float32)
start_time = time.time()
for i in range(100):
    tensor1 = tensor
    tensor1.to('cuda')
end_time = time.time()
print('1批量张量从cpu到gpu的时间:{:.4f}'.format((end_time - start_time) / 100))

tensor = torch.rand((16, 3, args.input_size, args.input_size), dtype=torch.float32)
start_time = time.time()
for i in range(100):
    tensor1 = tensor
    tensor1.to('cuda')
end_time = time.time()
print('16批量张量从cpu到gpu的时间:{:.4f}'.format((end_time - start_time) / 1600))
