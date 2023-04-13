import cv2
import tqdm
import wandb
import torch
import numpy as np
import albumentations
from block.val_get import val_get
from block.ModelEMA import ModelEMA
from block.lr_adjust import lr_adjust


def train_get(args, data_dict, model_dict, loss):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 分布式初始化
    torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                              output_device=args.local_rank) if args.distributed else None
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = ModelEMA(model) if args.ema else None
    if args.ema:
        ema.updates = model_dict['ema_updates']
    # 学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start, betas=(0.937, 0.999), weight_decay=0.0005)
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    optimizer_adjust = lr_adjust(args, model_dict['lr_adjust_item'])
    optimizer = optimizer_adjust(optimizer, model_dict['epoch'] + 1, 0)  # 初始化学习率
    # 数据集
    train_dataset = torch_dataset(args, 'train', data_dict['train'], data_dict['class'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                   drop_last=True, pin_memory=args.latch, num_workers=args.num_worker,
                                                   sampler=train_sampler)
    val_dataset = torch_dataset(args, 'val', data_dict['val'], data_dict['class'])
    val_sampler = None  # 分布式时数据合在主GPU上进行验证
    val_batch = args.batch // args.gpu_number  # 分布式验证时batch要减少为一个GPU的量
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                                                 drop_last=False, pin_memory=args.latch, num_workers=args.num_worker,
                                                 sampler=val_sampler)
    # wandb
    if args.wandb and args.local_rank == 0:
        wandb_image_list = []  # 记录所有的wandb_image最后一起添加(最多添加args.wandb_image_num张)
    epoch_base = model_dict['epoch'] + 1  # 新的一轮要+1
    for epoch in range(epoch_base, epoch_base + args.epoch):
        # 训练
        print(f'\n-----------------------第{epoch}轮-----------------------') if args.local_rank == 0 else None
        model.train()
        train_loss = 0  # 记录训练损失
        tqdm_show = tqdm.tqdm(total=len(data_dict['train']) // args.batch // args.gpu_number * args.gpu_number,
                              postfix=dict, mininterval=0.2) if args.local_rank == 0 else None  # tqdm
        for item, (image_batch, true_batch) in enumerate(train_dataloader):
            if args.wandb and args.local_rank == 0 and len(wandb_image_list) < args.wandb_image_num:
                wandb_image_batch = (image_batch * 255).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            image_batch = image_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            if args.amp:
                with torch.cuda.amp.autocast():
                    pred_batch = model(image_batch)
                    loss_batch = loss(pred_batch, true_batch)
                optimizer.zero_grad()
                args.amp.scale(loss_batch).backward()
                args.amp.step(optimizer)
                args.amp.update()
            else:
                pred_batch = model(image_batch)
                loss_batch = loss(pred_batch, true_batch)
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
            # 调整参数，ema.updates会自动+1
            ema.update(model) if args.ema else None
            # 记录损失
            train_loss += loss_batch.item()
            # tqdm
            if args.local_rank == 0:
                tqdm_show.set_postfix({'当前loss': loss_batch.item()})  # 添加loss显示
                tqdm_show.update(args.gpu_number)  # 更新进度条
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
        tqdm_show.close() if args.local_rank == 0 else None
        # 计算平均损失
        train_loss = train_loss / (item + 1)
        print('\n| train_loss:{:.4f} | lr:{:.6f} |\n'.format(train_loss, optimizer.param_groups[0]['lr']))
        # 调整学习率
        optimizer = optimizer_adjust(optimizer, epoch + 1, train_loss)
        # 清理显存空间
        del image_batch, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        if args.local_rank == 0:  # 分布式时只验证一次
            val_loss, accuracy, precision, recall, m_ap = val_get(args, val_dataloader, model, loss, ema)
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            model_dict['model'] = model.module.eval() if args.distributed else model.eval()
            model_dict['epoch'] = epoch
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['lr_adjust_item'] = optimizer_adjust.lr_adjust_item
            model_dict['ema_updates'] = ema.updates if args.ema else model_dict['ema_updates']
            model_dict['class'] = data_dict['class']
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            model_dict['val_accuracy'] = accuracy
            model_dict['val_precision'] = precision
            model_dict['val_recall'] = recall
            model_dict['val_m_ap'] = m_ap
            torch.save(model_dict, 'last.pt')  # 保存最后一次训练的模型
            if m_ap > 0.5 and m_ap > model_dict['standard']:
                model_dict['standard'] = m_ap
                torch.save(model_dict, args.save_name)  # 保存最佳模型
                print('\n| 保存最佳模型:{} | val_m_ap:{:.4f} |\n'.format(args.save_name, m_ap))
            if m_ap == 1:
                print('| 模型m_ap已达100%，暂停训练 |')
                break
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
                                       border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127))])
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
