import cv2
import tqdm
import torch
from block.val_get import val_get


def train_get(args, dataset_dict, model_dict, loss):
    model = model_dict['model']
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        # 训练
        print('\n-----------------------------------------------')
        print('| 第{}轮 | 训练集:{} | 批量:{} | 学习率:{} |\n'
              .format(epoch + 1, len(dataset_dict['train']), args.batch, optimizer.defaults['lr']))
        model.train().to(args.device, non_blocking=args.latch)
        train_loss = 0  # 记录训练损失
        train_dataloader = torch.utils.data.DataLoader(torch_dataset(args, dataset_dict['train']),
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
        # 验证
        val_loss, accuracy, precision, recall, m_ap = val_get(args, dataset_dict, model, loss)
        # 保存
        if m_ap > 0.8:
            if m_ap > model_dict['val_m_ap'] or m_ap == model_dict['val_m_ap'] and val_loss < model_dict['val_loss']:
                model_dict['model'] = model
                model_dict['class'] = dataset_dict['class']
                model_dict['epoch'] = epoch
                model_dict['train_loss'] = train_loss
                model_dict['val_loss'] = val_loss
                model_dict['val_m_ap'] = m_ap
                model_dict['val_accuracy'] = accuracy
                model_dict['val_precision'] = precision
                model_dict['val_recall'] = recall
                model_dict['bgr_mean'] = args.bgr_mean
                torch.save(model_dict, args.save_name.split('.')[0] + '.pt')
                print('\n| 保存模型:{} | val_loss:{:.4f} | m_ap:{:.4f} |\n'
                      .format(args.save_name.split('.')[0] + '.pt', val_loss, m_ap))
        # wandb
        if args.wandb:
            args.wandb_run.log({'metric/train_loss': train_loss, 'metric/val_loss': val_loss, 'metric/val_m_ap': m_ap,
                                'metric/val_accuracy': accuracy, 'metric/val_precision': precision,
                                'metric/val_recall': recall})
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = cv2.imread(self.dataset[index][0])  # 读取图片
        image = self._resize(image)  # 变为输入形状
        image = torch.tensor(image, dtype=torch.float32)  # 转换为tensor(比np计算更快)
        image = self._processing(image)  # 归一化和减均值
        label = torch.tensor(self.dataset[index][1], dtype=torch.float32)
        return image, label

    def _resize(self, image):
        args = self.args
        w0 = image.shape[1]
        h0 = image.shape[0]
        if w0 == h0:
            image = cv2.resize(image, (args.input_size, args.input_size))
        elif w0 > h0:  # 宽大于高
            w = args.input_size
            h = int(w / w0 * h0)
            image = cv2.resize(image, (w, h))
            add_y = (w - h) // 2
            image = cv2.copyMakeBorder(image, add_y, w - h - add_y, 0, 0, cv2.BORDER_CONSTANT, value=(126, 126, 126))
        else:  # 宽小于高
            h = self.args.input_size
            w = int(h / h0 * w0)
            image = cv2.resize(image, (w, h))
            add_x = (h - w) // 2
            image = cv2.copyMakeBorder(image, 0, 0, add_x, h - w - add_x, cv2.BORDER_CONSTANT, value=(126, 126, 126))
        return image

    def _processing(self, image):
        image = (image / 255).permute(2, 0, 1)
        image[0] = image[0] - self.args.bgr_mean[0]
        image[1] = image[1] - self.args.bgr_mean[1]
        image[2] = image[2] - self.args.bgr_mean[2]
        return image