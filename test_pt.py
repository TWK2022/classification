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
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--image_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=160, type=int, help='|模型输入图片大小|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量|')
parser.add_argument('--device', default='cpu', type=str, help='|用CPU/GPU推理|')
parser.add_argument('--bgr_mean', default=(0.485, 0.456, 0.406), type=tuple, help='|图片预处理时BGR通道减去的均值|')
parser.add_argument('--float16', default=False, type=bool, help='|推理数据类型，要与模型相对应，False时为float32|')
args = parser.parse_args()
args.model_path = args.model_path.split('.')[0] + '.pt'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'没有找到模型{args.model_path}'
assert os.path.exists(args.image_path), f'没有找到图片文件夹{args.image_path}'
if args.float16:
    assert torch.cuda.is_available(), 'cuda不可用，因此无法转为float16'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def test_pt():
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)
    cls = model_dict['class']
    print('| 模型加载成功:{} |'.format(args.model_path))
    # 推理
    image_dir = os.listdir(args.image_path)
    start_time = time.time()
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(torch_dataset(args, image_dir), batch_size=args.batch,
                                                 shuffle=False, drop_last=False, pin_memory=False)
        pred = []
        for item, batch in enumerate(dataloader):
            batch = batch.to(args.device)
            pred.extend(model(batch).detach().cpu())
        pred = torch.stack(pred, axis=0)
        result = [cls[torch.argmax(i)] for i in pred]
        print(f'| 预测结果:{result} |')
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_dir), args.batch, (end_time - start_time) / len(image_dir)))


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = cv2.imread(args.image_path + '/' + self.dataset[index])  # 读取图片
        image = self._resize(image)  # 变为输入形状
        image = torch.tensor(image, dtype=torch.float32)  # 转换为tensor(比np计算更快)
        image = self._processing(image)  # 归一化和减均值
        return image

    def _resize(self, image):
        args = self.args
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
            h = self.args.input_size
            w = int(h / h0 * w0)
            image = cv2.resize(image, (w, h))
            add_x = (h - w) // 2
            image = cv2.copyMakeBorder(image, 0, 0, add_x, h - w - add_x, cv2.BORDER_CONSTANT, value=(126, 126, 126))
        return image

    def _processing(self, image):
        image = (image / 255).permute(2, 0, 1)
        image[0] = image[0] - self.args.bgr_mean[0]
        image[1] = image[1] - self.args.bgr_mean[1]
        image[2] = image[2] - self.args.bgr_mean[2]
        return image


if __name__ == '__main__':
    test_pt()
