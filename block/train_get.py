import cv2
import tqdm
import wandb
import torch
import numpy as np
import albumentations
from block.val_get import val_get


def train_get(args, data_dict, model_dict, loss):
    model = model_dict['model']
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = torch.utils.data.DataLoader(torch_dataset(args, 'train', data_dict['train'], data_dict['class']),
                                                   batch_size=args.batch, shuffle=True, drop_last=True,
                                                   pin_memory=args.latch, num_workers=args.num_worker)
    val_dataloader = torch.utils.data.DataLoader(torch_dataset(args, 'val', data_dict['val'], data_dict['class']),
                                                 batch_size=args.batch, shuffle=False, drop_last=False,
                                                 pin_memory=args.latch, num_workers=args.num_worker)
    for epoch in range(args.epoch):
        # 训练
        print(f'\n-----------------------第{epoch + 1}轮-----------------------')
        model.train()
        train_loss = 0  # 记录训练损失
        for item, (image_batch, true_batch) in enumerate(tqdm.tqdm(train_dataloader)):
            image_batch = image_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            if args.scaler:
                with torch.cuda.amp.autocast():
                    pred_batch = model(image_batch)
                    loss_batch = loss(pred_batch, true_batch)
                    optimizer.zero_grad()
                    args.scaler.scale(loss_batch).backward()
                    args.scaler.step(optimizer)
                    args.scaler.update()
            else:
                pred_batch = model(image_batch)
                loss_batch = loss(pred_batch, true_batch)
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
            # 记录损失
            train_loss += loss_batch.item()
        train_loss = train_loss / (item + 1)
        print('\n| 训练集:{} | train_loss:{:.4f} |\n'.format(len(data_dict['train']), train_loss))
        # 清理显存空间
        del image_batch, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        val_loss, accuracy, precision, recall, m_ap = val_get(args, val_dataloader, model, loss)
        # 保存
        if m_ap > 0.8:
            if m_ap > model_dict['val_m_ap'] or m_ap == model_dict['val_m_ap'] and val_loss < model_dict['val_loss']:
                model_dict['model'] = model
                model_dict['class'] = data_dict['class']
                model_dict['epoch'] = epoch
                model_dict['train_loss'] = train_loss
                model_dict['val_loss'] = val_loss
                model_dict['val_accuracy'] = accuracy
                model_dict['val_precision'] = precision
                model_dict['val_recall'] = recall
                model_dict['val_m_ap'] = m_ap
                torch.save(model_dict, args.save_name)
                print('\n| 保存模型:{} | m_ap:{:.4f} |\n'.format(args.save_name, m_ap))
        # wandb
        if args.wandb:
            args.wandb_run.log({'metric/train_loss': train_loss,
                                'metric/val_loss': val_loss,
                                'metric/val_m_ap': m_ap,
                                'metric/val_accuracy': accuracy,
                                'metric/val_precision': precision,
                                'metric/val_recall': recall})
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, tag, data, class_name):
        self.tag = tag
        self.data = data
        self.class_name = class_name
        self.use_noise = args.noise
        self.noise = albumentations.Compose([
            albumentations.GaussianBlur(blur_limit=(5, 5), p=0.2),
            albumentations.GaussNoise(var_limit=(10.0, 30.0), p=0.2)])
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127))])
        # wandb可视化部分
        self.wandb = args.wandb
        if self.wandb:
            self.wandb_run = args.wandb_run
            self.wandb_count = 0  # 用于限制添加的图片数量(最多添加args.wandb_image_num张)
            self.wandb_image_num = args.wandb_image_num
            self.wandb_image = []
            self.class_name = class_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = cv2.imread(self.data[index][0])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        if self.tag == 'train' and self.use_noise:  # 使用数据加噪
            image = self.noise(image=image)['image']
        image = self.transform(image=image)['image']  # 缩放和填充图片
        image = torch.tensor(image, dtype=torch.float32)  # 转换为tensor(归一化、减均值、除以方差、调维度等在模型中完成)
        label = torch.tensor(self.data[index][1], dtype=torch.float32)  # 转换为tensor
        # 使用wandb添加图片
        if self.wandb and self.wandb_count < self.wandb_image_num:
            self.wandb_count += 1
            text = ''
            for i in range(len(label)):
                text += str(int(label[i].item())) + '-'
            text = text[:-1]
            wandb_image = np.array(image, dtype=np.uint8)
            cv2.putText(wandb_image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            wandb_image = wandb.Image(wandb_image)
            self.wandb_image.append(wandb_image)
            if self.wandb_count == self.wandb_image_num:
                self.wandb_run.log({f'image/{self.tag}_image': self.wandb_image})
        return image, label
