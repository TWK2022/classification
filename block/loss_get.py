import torch


def loss_get(args):
    with torch.no_grad():
        choice_dict = {'bce': 'torch.nn.BCEWithLogitsLoss()'}
        loss = eval(choice_dict[args.loss])
    return loss
