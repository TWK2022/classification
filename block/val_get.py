import tqdm
import torch
from block.metric_get import metric


def val_get(args, dataset_dict, model, loss):
    with torch.no_grad():
        model.eval().to(args.device)
        torch.cuda.empty_cache()
        val_dataloader = torch.utils.data.DataLoader(torch_dataset(args, dataset_dict['val']), batch_size=args.batch,
                                                     shuffle=False, drop_last=False, pin_memory=args.latch)
        pred_all = []
        true_all = []
        for item, (val_batch, true_batch) in enumerate(tqdm.tqdm(val_dataloader)):
            val_batch = val_batch.to(args.device, non_blocking=args.latch)
            pred_all.extend(model(val_batch).detach().cpu())
            true_all.extend(true_batch.detach().cpu())
        pred_all = torch.stack(pred_all, axis=0)
        true_all = torch.stack(true_all, axis=0)
        loss_all = loss(pred_all, true_all)
        accuracy, precision, recall, m_ap = metric(pred_all, true_all)
        print('\n| 验证集:{} | loss:{:.4f} | accuracy:{:.4f} | precision:{:.4f} | recall:{:.4f} | m_ap:{:.4f} |\n'
              .format(len(dataset_dict['val']), loss_all, accuracy, precision, recall, m_ap))
    return accuracy, precision, recall, m_ap, loss_all


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        val = torch.tensor(self.dataset[index][0], dtype=torch.float32)
        true = torch.tensor(self.dataset[index][1], dtype=torch.float32)
        return val, true
