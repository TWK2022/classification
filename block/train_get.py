import cv2
import tqdm
import wandb
import torch
import numpy as np
import albumentations
from block.val_get import val_get
from block.model_ema import model_ema
from block.lr_get import adam, lr_adjust


def train_get(args, data_dict, model_dict, loss):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 学习率
    optimizer = adam(args.regularization, args.r_value, model.parameters(), lr=args.lr_start, betas=(0.937, 0.999))
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    step_epoch = len(data_dict['train']) // args.batch // args.device_number * args.device_number  # 每轮的步数
    optimizer_adjust = lr_adjust(args, step_epoch, model_dict['epoch_finished'])  # 学习率调整函数
    optimizer = optimizer_adjust(optimizer)  # 学习率初始化
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = ModelEMA(model) if args.ema else None
    if args.ema:
        ema.updates = model_dict['ema_updates']
    # 数据集
    train_dataset = torch_dataset(args, 'train', data_dict['train'], data_dict['class'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                   drop_last=True, pin_memory=args.latch, num_workers=args.num_worker,
                                                   sampler=train_sampler)
    val_dataset = torch_dataset(args, 'val', data_dict['val'], data_dict['class'])
    val_sampler = None  # 分布式时数据合在主GPU上进行验证
    val_batch = args.batch // args.device_number  # 分布式验证时batch要减少为一个GPU的量
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                                                 drop_last=False, pin_memory=args.latch, num_workers=args.num_worker,
                                                 sampler=val_sampler)
    # 分布式初始化
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank) if args.distributed else model
    # wandb
    if args.wandb and args.local_rank == 0:
        wandb_image_list = []  # 记录所有的wandb_image最后一起添加(最多添加args.wandb_image_num张)
    epoch_base = model_dict['epoch_finished'] + 1  # 新的一轮要+1
    for epoch in range(epoch_base, args.epoch + 1):  # 训练
        print(f'\n-----------------------第{epoch}轮-----------------------') if args.local_rank == 0 else None
        model.train()
        train_loss = 0  # 记录损失
        if args.local_rank == 0:  # tqdm
            tqdm_show = tqdm.tqdm(total=step_epoch, mininterval=0.2)
        for index, (image_batch, true_batch) in enumerate(train_dataloader):
            if args.wandb and args.local_rank == 0 and len(wandb_image_list) < args.wandb_image_num:
                wandb_image_batch = (image_batch * 255).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            image_batch = image_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            if args.amp:
                with torch.cuda.amp.autocast():
                    pred_batch = model(image_batch)
                    loss_batch = loss(pred_batch, true_batch)
                args.amp.scale(loss_batch).backward()
                args.amp.step(optimizer)
                args.amp.update()
                optimizer.zero_grad()
            else:
                pred_batch = model(image_batch)
                loss_batch = loss(pred_batch, true_batch)
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
            # 调整参数，ema.updates会自动+1
            ema.update(model) if args.ema else None
            # 记录损失
            train_loss += loss_batch.item()
            # 调整学习率
            optimizer = optimizer_adjust(optimizer)
            # tqdm
            if args.local_rank == 0:
                tqdm_show.set_postfix({'train_loss': loss_batch.item(),
                                       'lr': optimizer.param_groups[0]['lr']})  # 添加显示
                tqdm_show.update(args.device_number)  # 更新进度条
            # wandb
            if args.wandb and args.local_rank == 0 and epoch == 0 and len(wandb_image_list) < args.wandb_image_num:
                cls = true_batch.cpu().numpy().tolist()
                for i in range(len(wandb_image_batch)):  # 遍历每一张图片
                    image = wandb_image_batch[i]
                    text = ['{:.0f}'.format(_) for _ in cls[i]]
                    text = text[0] if len(text) == 1 else '--'.join(text)
                    image = np.ascontiguousarray(image)  # 将数组的内存变为连续存储(cv2画图的要求)
                    cv2.putText(image, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    wandb_image = wandb.Image(image)
                    wandb_image_list.append(wandb_image)
                    if len(wandb_image_list) == args.wandb_image_num:
                        break
        # tqdm
        if args.local_rank == 0:
            tqdm_show.close()
        # 计算平均损失
        train_loss /= index + 1
        if args.local_rank == 0:
            print(f'\n| 训练 | train_loss:{train_loss:.4f} | lr:{optimizer.param_groups[0]["lr"]:.6f} |\n')
        # 清理显存空间
        del image_batch, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        if args.local_rank == 0:  # 分布式时只验证一次
            val_loss, accuracy, precision, recall, m_ap = val_get(args, val_dataloader, model, loss, ema)
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            model_dict['model'] = model.module if args.distributed else model
            model_dict['epoch_finished'] = epoch
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['ema_updates'] = ema.updates if args.ema else model_dict['ema_updates']
            model_dict['class'] = data_dict['class']
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            model_dict['val_accuracy'] = accuracy
            model_dict['val_precision'] = precision
            model_dict['val_recall'] = recall
            model_dict['val_m_ap'] = m_ap
            torch.save(model_dict, 'last.pt' if not args.prune else 'prune_last.pt')  # 保存最后一次训练的模型
            if m_ap > 0.5 and m_ap > model_dict['standard']:
                model_dict['standard'] = m_ap
                save_path = args.save_path if not args.prune else args.prune_save
                torch.save(model_dict, save_path)  # 保存最佳模型
                print(f'\n| 保存最佳模型:{save_path} | val_m_ap:{m_ap:.4f} |\n')
            # wandb
            if args.wandb:
                wandb_log = {}
                if epoch == 0:
                    wandb_log.update({f'image/train_image': wandb_image_list})
                wandb_log.update({'metric/train_loss': train_loss,
                                  'metric/val_loss': val_loss,
                                  'metric/val_m_ap': m_ap,
                                  'metric/val_accuracy': accuracy,
                                  'metric/val_precision': precision,
                                  'metric/val_recall': recall})
                args.wandb_run.log(wandb_log)
        torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, tag, data, class_name):
        self.tag = tag
        self.data = data
        self.class_name = class_name
        self.noise_probability = args.noise
        self.noise = albumentations.Compose([
            albumentations.GaussianBlur(blur_limit=(5, 5), p=0.2),
            albumentations.GaussNoise(var_limit=(10.0, 30.0), p=0.2)])
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128))])
        self.rgb_mean = (0.406, 0.456, 0.485)
        self.rgb_std = (0.225, 0.224, 0.229)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = cv2.imread(self.data[index][0])  # 读取图片
        if self.tag == 'train' and torch.rand(1) < self.noise_probability:  # 使用数据加噪
            image = self.noise(image=image)['image']
        image = self.transform(image=image)['image']  # 缩放和填充图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = self._image_deal(image)  # 归一化、转换为tensor、调维度
        label = torch.tensor(self.data[index][1], dtype=torch.float32)  # 转换为tensor
        return image, label

    def _image_deal(self, image):  # 归一化、转换为tensor、调维度
        image = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)
        return image
