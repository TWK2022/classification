import cv2
import tqdm
import torch
import albumentations
from block.metric_get import metric


def val_get(args, data_dict, model, loss):
    with torch.no_grad():
        model.eval().to(args.device, non_blocking=args.latch)
        dataloader = torch.utils.data.DataLoader(torch_dataset(args, data_dict['val']), batch_size=args.batch,
                                                 shuffle=False, drop_last=False, pin_memory=args.latch)
        pred_all = []  # 记录所有预测
        true_all = []  # 记录所有标签
        for item, (image_batch, true_batch) in enumerate(tqdm.tqdm(dataloader)):
            image_batch = image_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(image_batch).detach().cpu()
            pred_all.extend(pred_batch)
            true_all.extend(true_batch)
        # 计算指标
        pred_all = torch.stack(pred_all, dim=0)
        true_all = torch.stack(true_all, dim=0)
        loss_all = loss(pred_all, true_all)
        accuracy, precision, recall, m_ap = metric(pred_all, true_all, args.class_threshold)
        print('\n| 验证集:{} | val_loss:{:.4f} | 阈值:{:.2f} | val_accuracy:{:.4f} | val_precision:{:.4f} |'
              ' val_recall:{:.4f} | val_m_ap:{:.4f} |\n'
              .format(len(data_dict['val']), loss_all, args.class_threshold, accuracy, precision, recall, m_ap))
    return loss_all, accuracy, precision, recall, m_ap


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = cv2.imread(self.data[index][0])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = self.transform(image=image)['image']  # 缩放和填充图片
        image = torch.tensor(image, dtype=torch.float32)  # 转换为tensor(归一化、减均值、除以方差、调维度等在模型中完成)
        label = torch.tensor(self.data[index][1], dtype=torch.float32)  # 转换为tensor
        return image, label
