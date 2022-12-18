import cv2
import tqdm
import torch
import albumentations
from block.metric_get import metric


def val_get(args, data_dict, model, loss):
    with torch.no_grad():
        model.eval().to(args.device, non_blocking=args.latch)
        val_dataloader = torch.utils.data.DataLoader(torch_dataset(args, data_dict['val']), batch_size=args.batch,
                                                     shuffle=False, drop_last=False, pin_memory=args.latch)
        val_pred = []
        val_true = []
        for item, (val_batch, true_batch) in enumerate(tqdm.tqdm(val_dataloader)):
            val_batch = val_batch.to(args.device, non_blocking=args.latch)
            val_pred.extend(model(val_batch).detach().cpu())
            val_true.extend(true_batch.detach().cpu())
        val_pred = torch.stack(val_pred, dim=0)
        val_true = torch.stack(val_true, dim=0)
        val_loss = loss(val_pred, val_true) / len(val_pred)
        accuracy, precision, recall, m_ap = metric(val_pred, val_true)
        print('\n| 验证集:{} | val_loss:{:.4f} | accuracy:{:.4f} | precision:{:.4f} | recall:{:.4f} | m_ap:{:.4f} |\n'
              .format(len(data_dict['val']), val_loss, accuracy, precision, recall, m_ap))
    return val_loss, accuracy, precision, recall, m_ap


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.Normalize(max_pixel_value=255, mean=args.rgb_mean, std=args.rgb_std),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = cv2.imread(self.data[index][0])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = self.transform(image=image)['image']  # 归一化、减均值、除以方差
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 转换为tensor
        label = torch.tensor(self.data[index][1], dtype=torch.float32)  # 转换为tensor
        return image, label
