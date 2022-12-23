import torch


def metric(pred, true):  # 所有类别输出在0.5以下为空标签
    TP = len(pred[torch.where((true == 1) & (pred > 0.5), True, False)])
    TN = len(pred[torch.where((true == 0) & (pred <= 0.5), True, False)])
    FP = len(pred[torch.where((true == 0) & (pred > 0.5), True, False)])
    FN = len(pred[torch.where((true == 1) & (pred <= 0.5), True, False)])
    accuracy = (TP + TN) / (TP + TN + FP + FN + 0.00001)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    m_ap = precision * recall
    return accuracy, precision, recall, m_ap
