import tqdm
import torch
from block.metric_get import metric


def val_get(args, val_dataloader, model, loss, ema):
    with torch.no_grad():
        model = ema.ema if args.ema else model.eval()
        pred_all = []  # 记录所有预测
        true_all = []  # 记录所有标签
        for item, (image_batch, true_batch) in enumerate(tqdm.tqdm(val_dataloader)):
            image_batch = image_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(image_batch).detach().cpu()
            pred_all.extend(pred_batch)
            true_all.extend(true_batch)
        # 计算指标
        pred_all = torch.stack(pred_all, dim=0)
        true_all = torch.stack(true_all, dim=0)
        loss_all = loss(pred_all, true_all).item()
        accuracy, precision, recall, m_ap = metric(pred_all, true_all, args.class_threshold)
        print(f'\n| val_loss:{loss_all:.4f} | 阈值:{args.class_threshold:.2f} | val_accuracy:{accuracy:.4f} |'
              f' val_precision:{precision:.4f} | val_recall:{recall:.4f} | val_m_ap:{m_ap:.4f} |')
    return loss_all, accuracy, precision, recall, m_ap
