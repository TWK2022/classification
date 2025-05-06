import os
import cv2
import math
import copy
import wandb
import torch
import logging
import numpy as np
import albumentations


class train_class:
    '''
        model_load: 加载模型
        data_load: 加载数据
        dataloader_load: 加载数据处理器
        optimizer_load: 加载学习率
        loss_load: 训练损失
        train: 训练模型
        validation: 训练时的模型验证
    '''

    def __init__(self, args):
        self.args = args
        self.model_dict = self.model_load()  # 模型
        self.model_dict['model'] = self.model_dict['model'].to(args.device, non_blocking=args.latch)  # 设备
        self.data_dict = self.data_load()  # 数据
        self.train_dataloader, self.val_dataloader, self.train_dataset = self.dataloader_load()  # 数据处理器
        self.optimizer, self.optimizer_adjust = self.optimizer_load()  # 学习率、学习率调整
        self.loss = self.loss_load()  # 损失函数
        if args.local_rank == 0 and args.ema:  # 平均指数移动(EMA)，创建ema模型
            self.ema = model_ema(self.model_dict['model'])
            self.ema.update_total = self.model_dict['ema_update']
        if args.distributed:  # 分布式初始化
            self.model_dict['model'] = torch.nn.parallel.DistributedDataParallel(self.model_dict['model'],
                                                                                 device_ids=[args.local_rank],
                                                                                 output_device=args.local_rank)
        if args.log:  # 日志
            log_path = os.path.dirname(__file__) + '/log.log'
            logging.basicConfig(filename=log_path, level=logging.INFO,
                                format='%(asctime)s | %(levelname)s | %(message)s')
            logging.info('-------------------- log --------------------')

    @staticmethod
    def metric(pred, label, threshold):  # 指标
        pred = torch.softmax(pred, dim=1)
        P_index = (label == 1)  # 正类
        N_index = (label == 0)  # 负类
        TP = (pred[P_index] >= threshold).sum()
        TN = (pred[N_index] < threshold).sum()
        FP = (pred[N_index] >= threshold).sum()
        FN = (pred[P_index] < threshold).sum()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        m_ap = precision * recall
        return accuracy, precision, recall, m_ap

    @staticmethod
    def weight_assignment(model, prune_model):  # 剪枝模型权重赋值
        for module, prune_module in zip(model.modules(), prune_model.modules()):
            if not hasattr(module, 'weight'):  # 对权重层赋值
                continue
            weight = module.weight.data
            prune_weight = prune_module.weight.data
            if len(weight.shape) == 1:  # 单维权重(如bn层)
                prune_module.weight.data = weight[:prune_weight.shape[0]]
            else:  # 两维权重(如conv层)
                prune_module.weight.data = weight[:prune_weight.shape[0], :prune_weight.shape[1]]
        return prune_model

    def model_load(self):
        args = self.args
        if os.path.exists(args.weight_path):
            model_dict = torch.load(args.weight_path, map_location='cpu', weights_only=False)
            for param in model_dict['model'].parameters():
                param.requires_grad_(True)  # 打开梯度(保存的ema模型为关闭)
            if args.weight_again:
                model_dict['epoch_finished'] = 0  # 已训练的轮数
                model_dict['optimizer_state_dict'] = None  # 学习率参数
                model_dict['ema_update'] = 0  # ema参数
                model_dict['standard'] = 0  # 评价指标
        else:  # 创建新模型
            if os.path.exists(args.prune_weight_path):
                model_dict = torch.load(args.prune_weight_path, map_location='cpu', weights_only=False)
                model = model_dict['model']  # 原模型
                exec(f'from model.{args.model} import {args.model}')
                config = self._bn_prune(model)  # 剪枝参数
                prune_model = eval(f'{args.model}(self.args, config=config)')  # 剪枝模型
                model = self.weight_assignment(model, prune_model)  # 剪枝模型赋值
            else:
                exec(f'from model.{args.model} import {args.model}')
                model = eval(f'{args.model}(self.args)')
            model_dict = {
                'model': model,
                'epoch_finished': 0,  # 已训练的轮数
                'optimizer_state_dict': None,  # 学习率参数
                'ema_update': 0,  # ema参数
                'standard': 0,  # 评价指标
            }
        return model_dict

    def data_load(self):
        args = self.args
        # 训练集
        with open(f'{args.data_path}/train.txt', encoding='utf-8') as f:
            train_list = [_.strip().split(' ') for _ in f.readlines()]  # 读取数据[[图片路径,类别],...]
        train_list = [[f'{args.data_path}/image/{os.path.basename(_[0])}', list(map(int, _[1:]))] for _ in train_list]
        # 验证集
        with open(f'{args.data_path}/val.txt', encoding='utf-8') as f:
            val_list = [_.strip().split(' ') for _ in f.readlines()]  # 读取数据[[图片路径,类别],...]
        val_list = [[f'{args.data_path}/image/{os.path.basename(_[0])}', list(map(int, _[1:]))] for _ in val_list]
        # 类别
        with open(f'{args.data_path}/class.txt', encoding='utf-8') as f:
            class_list = [_.strip() for _ in f.readlines()]
        data_dict = {
            'train': train_list,
            'val': val_list,
            'class': class_list,
        }
        return data_dict

    def dataloader_load(self):
        args = self.args
        # 数据集
        train_dataset = torch_dataset(args, 'train', self.data_dict['train'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                       drop_last=True, pin_memory=args.latch,
                                                       num_workers=args.num_worker,
                                                       sampler=train_sampler)
        val_dataset = torch_dataset(args, 'val', self.data_dict['val'])
        val_sampler = None  # 分布式时数据合在主GPU上进行验证
        val_batch = args.batch // args.device_number  # 分布式验证时batch要减少为一个GPU的量
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                                                     drop_last=False, pin_memory=args.latch,
                                                     num_workers=args.num_worker,
                                                     sampler=val_sampler)
        return train_dataloader, val_dataloader, train_dataset

    def optimizer_load(self):
        args = self.args
        if args.regularization == 'L2':
            optimizer = torch.optim.Adam(self.model_dict['model'].parameters(),
                                         lr=args.lr_start, betas=(0.937, 0.999), weight_decay=args.r_value)
        else:
            optimizer = torch.optim.Adam(self.model_dict['model'].parameters(),
                                         lr=args.lr_start, betas=(0.937, 0.999))
        if self.model_dict['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(self.model_dict['optimizer_state_dict'])
        step_epoch = len(self.data_dict['train']) // args.batch // args.device_number * args.device_number  # 每轮步数
        optimizer_adjust = lr_adjust(args, step_epoch, self.model_dict['epoch_finished'])  # 学习率调整函数
        optimizer = optimizer_adjust(optimizer)  # 学习率初始化
        return optimizer, optimizer_adjust

    def loss_load(self):
        loss = eval(f'torch.nn.{self.args.loss}()')
        return loss

    def train(self):
        args = self.args
        model = self.model_dict['model']
        epoch_base = self.model_dict['epoch_finished'] + 1  # 新的一轮要+1
        # wandb
        if args.wandb and args.local_rank == 0:
            wandb_image_list = []  # 记录所有的wandb_image最后一起添加(最多添加args.wandb_image_num张)
        for epoch in range(epoch_base, args.epoch + 1):
            if args.local_rank == 0:
                info = f'-----------------------epoch:{epoch}-----------------------'
                if args.print_info:
                    print(info)
            model.train()
            train_loss = 0  # 记录损失
            self.train_dataset.epoch_update(epoch)
            for index, (image_batch, label_batch) in enumerate(self.train_dataloader):
                if args.local_rank == 0 and args.wandb and len(wandb_image_list) < args.wandb_image_num:
                    wandb_image_batch = (image_batch * 255).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                image_batch = image_batch.to(args.device, non_blocking=args.latch)
                label_batch = label_batch.to(args.device, non_blocking=args.latch)
                if args.amp:
                    with torch.cuda.amp.autocast():
                        pred_batch = model(image_batch)
                        loss_batch = self.loss(pred_batch, label_batch)
                    args.amp.scale(loss_batch).backward()
                    args.amp.step(self.optimizer)
                    args.amp.update()
                    self.optimizer.zero_grad()
                else:
                    pred_batch = model(image_batch)
                    loss_batch = self.loss(pred_batch, label_batch)
                    loss_batch.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.ema.update(model) if args.local_rank == 0 and args.ema else None  # 更新ema模型参数
                train_loss += loss_batch.item()  # 记录损失
                self.optimizer = self.optimizer_adjust(self.optimizer)  # 调整学习率
                # wandb
                if args.wandb and args.local_rank == 0 and epoch == 0 and len(wandb_image_list) < 16:
                    cls = label_batch.cpu().numpy().tolist()
                    for i in range(len(wandb_image_batch)):  # 遍历每一张图片
                        image = wandb_image_batch[i]
                        text = [f'{_:.0f}' for _ in cls[i]]
                        text = text[0] if len(text) == 1 else '---'.join(text)
                        image = np.ascontiguousarray(image)  # 将数组的内存变为连续存储(cv2画图的要求)
                        cv2.putText(image, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        wandb_image = wandb.Image(image)
                        wandb_image_list.append(wandb_image)
                        if len(wandb_image_list) == 16:
                            break
            # 日志
            if args.local_rank == 0:
                train_loss /= index + 1  # 计算平均损失
                info = f'| train | train_loss:{train_loss:.4f} | lr:{self.optimizer.param_groups[0]["lr"]:.6f} |'
                if args.print_info:
                    print(info)
            # 清理显存空间
            del image_batch, label_batch, pred_batch, loss_batch
            torch.cuda.empty_cache()
            # 验证
            if args.local_rank == 0:  # 分布式时只验证一次
                val_loss, accuracy, precision, recall, m_ap = self.validation()
            # 保存
            if args.local_rank == 0:  # 分布式时只保存一次
                self.model_dict['model'] = self.ema.ema_model if args.ema else (
                    model.module if args.distributed else model)
                self.model_dict['epoch_finished'] = epoch
                self.model_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                self.model_dict['ema_update'] = self.ema.update_total if args.ema else self.model_dict['ema_update']
                self.model_dict['class'] = self.data_dict['class']
                self.model_dict['train_loss'] = train_loss
                self.model_dict['val_loss'] = val_loss
                self.model_dict['val_accuracy'] = accuracy
                self.model_dict['val_precision'] = precision
                self.model_dict['val_recall'] = recall
                self.model_dict['val_m_ap'] = m_ap
                if epoch % args.save_epoch == 0 or epoch == args.epoch:
                    torch.save(self.model_dict, args.save_path)  # 保存模型
                if val_loss < 1 and m_ap >= self.model_dict['standard']:
                    self.model_dict['standard'] = m_ap
                    torch.save(self.model_dict, args.save_best)  # 保存最佳模型
                    if args.local_rank == 0:  # 日志
                        info = (f'| best_model | val_loss:{val_loss:.4f} | threshold:{args.class_threshold:.2f} |'
                                f' val_accuracy:{accuracy:.4f} | val_precision:{precision:.4f} |'
                                f' val_recall:{recall:.4f} | val_m_ap:{m_ap:.4f} |')
                        if args.print_info:
                            print(info)
                        if args.log:
                            logging.info(info)
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

    def validation(self):
        args = self.args
        with torch.no_grad():
            model = self.ema.ema_model.eval() if args.ema else self.model_dict['model'].eval()
            pred_all = []
            label_all = []
            val_loss = 0
            for index, (image_batch, label_batch) in enumerate(self.val_dataloader):
                image_batch = image_batch.to(args.device, non_blocking=args.latch)
                pred_batch = model(image_batch).detach().cpu()
                loss_batch = self.loss(pred_batch, label_batch)
                val_loss += loss_batch.item()
                pred_all.extend(pred_batch)
                label_all.extend(label_batch)
            # 计算指标
            val_loss /= (index + 1)
            pred_all = torch.stack(pred_all, dim=0)
            label_all = torch.stack(label_all, dim=0)
            accuracy, precision, recall, m_ap = self.metric(pred_all, label_all, args.class_threshold)
            # 日志
            info = (f'| val | val_loss:{val_loss:.4f} | threshold:{args.class_threshold:.2f} |'
                    f' val_accuracy:{accuracy:.4f} | val_precision:{precision:.4f} |'
                    f' val_recall:{recall:.4f} | val_m_ap:{m_ap:.4f} |')
            if args.print_info:
                print(info)
        return val_loss, accuracy, precision, recall, m_ap

    def _bn_prune(self, model):  # 通过bn层裁剪模型
        args = self.args
        weight = []  # 权重
        weight_layer = []  # 每个权重所在的层
        layer = 0  # 层数记录
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                weight.append(module.weight.data.clone())
                weight_layer.append(np.full((len(module.weight.data),), layer))
                layer += 1
        weight_abs = torch.concatenate(weight, dim=0).abs()
        weight_index = np.concatenate(weight_layer, axis=0)
        # 剪枝
        boundary = int(len(weight_abs) * args.prune_ratio)
        weight_index_keep = weight_index[np.argsort(weight_abs)[-boundary:]]  # 保留的参数所在的层数
        config = []  # 裁剪结果
        for layer, weight_one in enumerate(weight):
            layer_number = max(np.sum(weight_index_keep == layer).item(), 1)  # 剪枝后该层的参数个数，至少1个
            config.append(layer_number)
        return config


class model_ema:
    def __init__(self, model, decay=0.9999, tau=2000, update_total=0):
        self.ema_model = copy.deepcopy(self._get_model(model)).eval()  # FP32 EMA
        self.update_total = update_total
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for param in self.ema_model.parameters():
            param.requires_grad_(False)  # 关闭梯度

    def update(self, model):
        with torch.no_grad():
            self.update_total += 1
            d = self.decay(self.update_total)
            state_dict = self._get_model(model).state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * state_dict[k].detach()

    def _get_model(self, model):
        if type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel):
            return model.module
        else:
            return model


class lr_adjust:
    def __init__(self, args, step_epoch, epoch_finished):
        self.lr_start = args.lr_start  # 初始学习率
        self.lr_end = args.lr_end_ratio * args.lr_start  # 最终学习率
        self.lr_end_epoch = args.lr_end_epoch  # 最终学习率达到的轮数
        self.step_all = self.lr_end_epoch * step_epoch  # 总调整步数
        self.step_finished = epoch_finished * step_epoch  # 已调整步数
        self.warmup_step = max(5, int(args.warmup_ratio * self.step_all))  # 预热训练步数

    def __call__(self, optimizer):
        self.step_finished += 1
        step_now = self.step_finished
        decay = step_now / self.step_all
        lr = self.lr_end + (self.lr_start - self.lr_end) * math.cos(math.pi / 2 * decay)
        if step_now <= self.warmup_step:
            lr = lr * (0.1 + 0.9 * step_now / self.warmup_step)
        lr = max(lr, 0.000001)
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        return optimizer


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, tag, data):
        self.tag = tag
        self.data = data
        self.noise = args.noise
        self.noise_probability = 0
        self.epoch_total = args.epoch
        self.output_class = args.output_class
        self.noise_function = albumentations.Compose([
            albumentations.GaussianBlur(blur_limit=(5, 5), p=0.2),
            albumentations.GaussNoise(var_limit=(10.0, 30.0), p=0.2)])
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = cv2.imdecode(np.fromfile(self.data[index][0], dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        if self.tag == 'train' and torch.rand(1) < self.noise_probability:  # 数据加噪
            image = self.noise_function(image=image)['image']
        image = self.transform(image=image)['image']  # 缩放、填充图片
        image = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)  # 归一化、转换为tensor、调维度
        label = torch.zeros(self.output_class, dtype=torch.float32)  # 标签
        label[self.data[index][1]] = 1
        return image, label

    def epoch_update(self, epoch_now):  # 根据轮数进行调整
        if 0.1 * self.epoch_total < epoch_now < 0.9 * self.epoch_total:  # 开始和末尾不加噪
            self.noise_probability = self.noise
        else:
            self.noise_probability = 0
