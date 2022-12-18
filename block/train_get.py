import cv2
import tqdm
import torch
import albumentations
from block.val_get import val_get


def train_get(args, data_dict, model_dict, loss):
    model = model_dict['model']
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        # 训练
        print('\n-----------------------------------------------')
        print('| 第{}轮 | 训练集:{} | 批量:{} | 学习率:{} |\n'
              .format(epoch + 1, len(data_dict['train']), args.batch, optimizer.defaults['lr']))
        model.train().to(args.device, non_blocking=args.latch)
        train_loss = 0  # 记录训练损失
        train_dataloader = torch.utils.data.DataLoader(torch_dataset(args, data_dict['train']),
                                                       batch_size=args.batch, shuffle=True, drop_last=True,
                                                       pin_memory=args.latch)
        for item, (train_batch, true_batch) in enumerate(tqdm.tqdm(train_dataloader)):
            train_batch = train_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(train_batch)
            loss_batch = loss(pred_batch, true_batch)
            train_loss += loss_batch.item()
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        train_loss = train_loss / (item + 1) / args.batch
        print('\n| 训练集:{} | train_loss:{:.4f} |\n'.format(len(data_dict['train']), train_loss))
        # 清理显存空间
        del train_batch, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        val_loss, accuracy, precision, recall, m_ap = val_get(args, data_dict, model, loss)
        # 保存
        if m_ap > 0.8:
            if m_ap > model_dict['val_m_ap'] or m_ap == model_dict['val_m_ap'] and val_loss < model_dict['val_loss']:
                model_dict['model'] = model
                model_dict['class'] = data_dict['class']
                model_dict['rgb_mean'] = args.rgb_mean
                model_dict['rgb_std'] = args.rgb_std
                model_dict['epoch'] = epoch
                model_dict['train_loss'] = train_loss
                model_dict['val_loss'] = val_loss
                model_dict['val_m_ap'] = m_ap
                model_dict['val_accuracy'] = accuracy
                model_dict['val_precision'] = precision
                model_dict['val_recall'] = recall
                torch.save(model_dict, args.save_name)
                print('\n| 保存模型:{} | val_loss:{:.4f} | m_ap:{:.4f} |\n'
                      .format(args.save_name, val_loss, m_ap))
        # wandb
        if args.wandb:
            args.wandb_run.log({'metric/train_loss': train_loss, 'metric/val_loss': val_loss, 'metric/val_m_ap': m_ap,
                                'metric/val_accuracy': accuracy, 'metric/val_precision': precision,
                                'metric/val_recall': recall})
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.noise = albumentations.Compose([
            albumentations.GaussianBlur(blur_limit=(5, 5), p=0.2),
            albumentations.GaussNoise(var_limit=(10.0, 30.0), p=0.2)])
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
        if self.args.noise:  # 使用数据加噪
            image = self.noise(image=image)['image']
        image = self.transform(image=image)['image']  # 归一化、减均值、除以方差
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 转换为tensor
        label = torch.tensor(self.data[index][1], dtype=torch.float32)  # 转换为tensor
        return image, label
