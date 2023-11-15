def loss_get(args):
    choice_dict = {'bce': 'torch.nn.BCEWithLogitsLoss()'}
    loss = eval(choice_dict[args.loss])
    return loss
