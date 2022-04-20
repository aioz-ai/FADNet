import torch.optim as optim
import numpy as np
import torch_optimizer as upgrade_optim


def get_optimizer(optimizer_name, net, lr_initial=1e-3):
    """

    :param optimizer_name:
    :param net:
    :param lr_initial:
    :return:
    """
    if optimizer_name == "adam":
        return optim.Adam([param for param in net.parameters() if param.requires_grad], lr=lr_initial, weight_decay=1e-5)

    elif optimizer_name == "sgd":
        return optim.SGD([param for param in net.parameters() if param.requires_grad], lr=lr_initial)

    elif optimizer_name == "radam":
        return upgrade_optim.RAdam([param for param in net.parameters() if param.requires_grad], lr=lr_initial, weight_decay=1e-5)
    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, epoch_size):
    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1/np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cyclic":
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=0.1)

    elif scheduler_name == "custom":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30*int(epoch_size), gamma=0.1)
    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")

