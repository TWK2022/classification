import cv2
import tqdm
import torch
from block.metric_get import metric


def val_get(args, dataset_dict, model, loss):
    with torch.no_grad():
        model.eval().to(args.device, non_blocking=args.latch)
        torch.cuda.empty_cache()
        val_dataloader = torch.utils.data.DataLoader(torch_dataset(args, dataset_dict['val']), batch_size=args.batch,
                                                     shuffle=False, drop_last=False, pin_memory=args.latch)
        val_pred = []
        val_true = []
        for item, (val_batch, true_batch) in enumerate(tqdm.tqdm(val_dataloader)):
            val_batch = val_batch.to(args.device, non_blocking=args.latch)
            val_pred.extend(model(val_batch).detach().cpu())
            val_true.extend(true_batch.detach().cpu())
        val_pred = torch.stack(val_pred, axis=0)
        val_true = torch.stack(val_true, axis=0)
        val_loss = loss(val_pred, val_true) / len(val_pred)
        accuracy, precision, recall, m_ap = metric(val_pred, val_true)
        print('\n| 验证集:{} | loss:{:.4f} | accuracy:{:.4f} | precision:{:.4f} | recall:{:.4f} | m_ap:{:.4f} |\n'
              .format(len(dataset_dict['val']), val_loss, accuracy, precision, recall, m_ap))
    return val_loss, accuracy, precision, recall, m_ap


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
