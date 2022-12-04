import tqdm
import torch
from block.val_get import val_get


def train_get(args, dataset_dict, model_dict, loss):
    model = model_dict['model']
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        print('\n-----------------------------------------------')
        print('| 第{}轮 | 训练集:{} | 批量:{} | 学习率:{} |\n'
              .format(epoch + 1, len(dataset_dict['train']), args.batch, optimizer.defaults['lr']))
        model.train().to(args.device)
        train_dataloader = torch.utils.data.DataLoader(torch_dataset(args, dataset_dict['train']),
                                                       batch_size=args.batch, shuffle=True, drop_last=True,
                                                       pin_memory=args.latch)
        for item, (train_batch, true_batch) in enumerate(tqdm.tqdm(train_dataloader)):
            train_batch = train_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(train_batch)
            loss_batch = loss(pred_batch, true_batch)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        # 验证
        accuracy, precision, recall, m_ap, val_loss = val_get(args, dataset_dict, model, loss)
        # 保存
        if m_ap > 0.8:
            if m_ap > model_dict['m_ap'] or m_ap == model_dict['m_ap'] and val_loss < model_dict['val_loss']:
                model_dict['model'] = model
                model_dict['m_ap'] = m_ap
                model_dict['val_loss'] = val_loss
                model_dict['accuracy'] = accuracy
                model_dict['precision'] = precision
                model_dict['recall'] = recall
                model_dict['bgr_mean'] = args.bgr_mean
                torch.save(model_dict, args.save_name.split('.')[0] + '.pt')
                print('\n| 保存模型:{} | m_ap:{:.4f} | val_loss:{:.4f} |\n'
                      .format(args.save_name.split('.')[0] + '.pt', m_ap, val_loss))
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        train = torch.tensor(self.dataset[index][0], dtype=torch.float32)
        true = torch.tensor(self.dataset[index][1], dtype=torch.float32)
        return train, true
