import torch


def loss_get(args):
    choice_dict = {'bce': 'torch.nn.BCELoss()'}
    loss = eval(choice_dict[args.loss])
    return loss
