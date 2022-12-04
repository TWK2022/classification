import os
import cv2
import time
import torch
import argparse
import onnxruntime
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_path', default='best.pt', type=str, help='|onnx模型位置位置|')
parser.add_argument('--image_path', default='image', type=str, help='|测试图片/图片文件夹位置|')
parser.add_argument('--input_size', default=160, type=int, help='|onnx模型输入图片大小|')
parser.add_argument('--batch', default=1, type=int, help='|预测批量|')
parser.add_argument('--device', default='cpu', type=str, help='|onnx模型用CPU/GPU推理|')
parser.add_argument('--bgr_mean', default=(0.485, 0.456, 0.406), type=tuple, help='|图片预处理时BGR通道减去的均值|')
parser.add_argument('--float16', default=False, type=bool, help='|推理数据类型，要与模型相对应，False时为float32|')
args = parser.parse_args()
args.model_path = args.model_path.split('.')[0] + '.pt'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), '没有找到模型{}'.format(args.model_path)
assert os.path.exists(args.image_path), '没有找到图片/图片文件夹{}'.format(args.image_path)
if args.float16:
    assert torch.cuda.is_available(), 'cuda不可用，因此无法转为float16'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def resize(image):
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


def processing(image):
    image = (image / 255).transpose(2, 0, 1).astype(np.float16 if args.float16 else np.float32)
    image[0] = image[0] - args.bgr_mean[0]
    image[1] = image[1] - args.bgr_mean[1]
    image[2] = image[2] - args.bgr_mean[2]
    return image


def test_pt():
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)
    print('| 模型加载成功:{} |'.format(args.model_path))
    # 加载数据
    start_time = time.time()
    if os.path.isfile(args.image_path):
        image = cv2.imread(args.image_path)
        image = resize(image)  # 变为输入形状
        image = processing(image)  # 归一化和减均值
        image_all = image.unsqueeze(0)
    elif os.path.isdir(args.image_path):
        image_dir = sorted(os.listdir(args.image_path))
        image_all = np.zeros((len(image_dir), 3, args.input_size, args.input_size)).astype(
            np.float16 if args.float16 else np.float32)
        for i in range(len(image_dir)):
            image = cv2.imread(args.image_path + '/' + image_dir[i])
            image = resize(image)  # 变为输入形状
            image = processing(image)  # 归一化和减均值
            image_all[i] = image
    image_all = torch.tensor(image_all).to(args.device)
    end_time = time.time()
    print('| 数据加载成功:{} 每张耗时:{:.4f} |'.format(len(image_all), (end_time - start_time) / len(image_all)))
    # 推理
    start_time = time.time()
    with torch.no_grad():
        n = len(image_all) // args.batch
        pred_all = []
        if n != 0:
            for i in range(n):
                batch = image_all[i * args.batch:(i + 1) * args.batch]
                pred = model(batch)
                pred_all.extend(pred)
            if len(image_all) % args.batch > 0:
                batch = image_all[(i + 1) * args.batch:]
                pred = model(batch)
                pred_all.extend(pred)
        else:
            batch = image_all
            pred = model(batch)
            pred_all.extend(pred)
        pred_all = [torch.argmax(i).item() for i in pred_all]
        print(pred_all)
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_all), args.batch, (end_time - start_time) / len(image_all)))


if __name__ == '__main__':
    test_pt()
