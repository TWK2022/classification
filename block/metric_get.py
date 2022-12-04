import torch


def metric(pred, true):  # 0类别为非目标
    pred_argmax = torch.zeros(len(pred), dtype=torch.int8)
    true_argmax = torch.zeros(len(true), dtype=torch.int8)
    for i in range(len(pred)):
        pred_argmax[i] = torch.argmax(pred[i])
        true_argmax[i] = torch.argmax(true[i])
    TP = len(pred[torch.where((pred_argmax != 0) & (pred_argmax == true_argmax), True, False)])
    TN = len(pred[torch.where((pred_argmax == 0) & (true_argmax == 0), True, False)])
    FP = len(pred[torch.where((pred_argmax != 0) & (true_argmax == 0), True, False)])
    FN = len(pred[torch.where((pred_argmax == 0) & (true_argmax != 0), True, False)])
    accuracy = (TP + TN) / (TP + TN + FP + FN + 0.00001)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    m_ap = precision * recall
    return accuracy, precision, recall, m_ap
