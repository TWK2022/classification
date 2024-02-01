import os
import cv2
import time
import torch
import argparse
import albumentations
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|pt模型推理|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=320, type=int, help='|模型输入图片大小|')
parser.add_argument('--normalization', default='sigmoid', type=str, help='|选择sigmoid或softmax归一化，单类别一定要选sigmoid|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量|')
parser.add_argument('--device', default='cuda', type=str, help='|推理设备|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据的进程数，0只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要支持float16的GPU，False时为float32|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'
if args.float16:
    assert torch.cuda.is_available(), 'cuda不可用，因此无法使用float16'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def test_pt(args):
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model = deploy(model, args.normalization)
    model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)
    epoch = model_dict['epoch']
    m_ap = round(model_dict['standard'], 3)
    print(f'| 模型加载成功:{args.model_path} | epoch:{epoch} | m_ap:{m_ap}|')
    # 推理
    image_dir = sorted(os.listdir(args.data_path))
    start_time = time.time()
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(torch_dataset(image_dir), batch_size=args.batch,
                                                 shuffle=False, drop_last=False, pin_memory=False,
                                                 num_workers=args.num_worker)
        result = []
        for item, batch in enumerate(dataloader):
            batch = batch.to(args.device)
            pred_batch = model(batch).detach().cpu()
            result.extend(pred_batch.tolist())
        for i in range(len(result)):
            result[i] = [round(result[i][_], 2) for _ in range(len(result[i]))]
            print(f'| {image_dir[i]}:{result[i]} |')
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_dir), args.batch, (end_time - start_time) / len(image_dir)))


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128))])

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image = cv2.imread(args.data_path + '/' + self.image_dir[index])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = self.transform(image=image)['image']  # 缩放和填充图片(归一化、调维度在模型中完成)
        image = torch.tensor(image, dtype=torch.float16 if args.float16 else torch.float32)
        return image


if __name__ == '__main__':
    test_pt(args)
