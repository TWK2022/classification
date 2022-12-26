import os
import cv2
import time
import torch
import argparse
import albumentations

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='pt模型推理')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--image_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=160, type=int, help='|模型输入图片大小|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量|')
parser.add_argument('--device', default='cuda', type=str, help='|用CPU/GPU推理|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要支持float16的GPU，False时为float32|')
args = parser.parse_args()
args.model_path = args.model_path.split('.')[0] + '.pt'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'没有找到模型{args.model_path}'
assert os.path.exists(args.image_path), f'没有找到图片文件夹{args.image_path}'
if args.float16:
    assert torch.cuda.is_available(), 'cuda不可用，因此无法使用float16'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def test_pt():
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)
    args.rgb_mean = model_dict['rgb_mean']
    args.rgb_std = model_dict['rgb_std']
    print('| 模型加载成功:{} |'.format(args.model_path))
    # 推理
    image_dir = sorted(os.listdir(args.image_path))
    start_time = time.time()
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(torch_dataset(image_dir), batch_size=args.batch,
                                                 shuffle=False, drop_last=False, pin_memory=False)
        pred_list = []
        for item, batch in enumerate(dataloader):
            batch = batch.to(args.device)
            pred_list.extend(model(batch).detach().cpu().tolist())
        result = pred_list
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_dir), args.batch, (end_time - start_time) / len(image_dir)))
    print(f'| 预测结果:{result} |')


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.Normalize(max_pixel_value=255, mean=args.rgb_mean, std=args.rgb_std),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))])

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image = cv2.imread(args.image_path + '/' + self.image_dir[index])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = self.transform(image=image)['image']  # 归一化、减均值、除以方差
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 转换为tensor
        return image


if __name__ == '__main__':
    test_pt()
