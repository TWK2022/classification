import cv2
import time
import torch
import argparse
import numpy as np
from models import inference


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
        image = cv2.copyMakeBorder(image, add_y, w - h - add_y, 0, 0, cv2.BORDER_CONSTANT,
                                   value=(126, 126, 126))
    else:  # 宽小于高
        h = args.input_size
        w = int(h / h0 * w0)
        image = cv2.resize(image, (w, h))
        add_x = (h - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, add_x, h - w - add_x, cv2.BORDER_CONSTANT,
                                   value=(126, 126, 126))
    return image


def test(args, dataset, model):
    with torch.no_grad():
        model.eval()
        test_dataloader = torch.utils.data.DataLoader(dataloader(args, dataset), batch_size=args.batch,
                                                      shuffle=False, drop_last=False, pin_memory=args.latch)
        for item, (test_batch) in enumerate(test_dataloader):
            test_batch = test_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(test_batch).detach().cpu()
    return pred_batch


class dataloader(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test = torch.tensor(self.dataset[index] / 255, dtype=torch.float32).permute(2, 0, 1)
        return test


def test0(args, dataset, model):
    with torch.no_grad():
        model.eval()
        dataset = np.array(dataset) / 255
        dataset = torch.tensor(dataset, dtype=torch.float32).permute(0, 3, 1, 2)
        for item in range(args.n):
            test_batch = dataset[item * args.batch:(item + 1) * args.batch].to(args.device, non_blocking=args.latch)
            pred_batch = model(test_batch).detach().cpu()
    return pred_batch


# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='')
parser.add_argument('--weight', default='terror.pt', type=str, help='|模型位置|')
parser.add_argument('--threshold', default=0.5, type=float, help='|判断为目标类别的阈值|')
parser.add_argument('--input_size', default=320, type=int, help='|输入图片大小|')
parser.add_argument('--save_image', default=False, type=bool, help='|是否保存结果|')
parser.add_argument('--device', default='cuda', type=str, help='|设备:cpu/cuda|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|有多少个进程处理数据，0为全加入到主进程|')
args = parser.parse_args()
args.n = 100
args.batch = 1

image = cv2.imread('001.jpg')
image = resize(args, image)
image_list = [0 for _ in range(args.n * args.batch)]
for i in range(args.n * args.batch):
    image_list[i] = image
print('| 使用{} | 模型加载中... |'.format(args.device))
model_dict = torch.load(args.weight)
model = model_dict['model']
print('| 模型加载完毕! |')
start_time = time.time()
pred = test0(args, image_list, model)
end_time = time.time()
print('| 轮次:{} | 批量{} | 平均每张耗时{:.3f} |'.format(args.n, args.batch, (end_time - start_time) / args.n / args.batch))
input('>>>按回车结束(此时可以查看GPU占用量)<<<')
