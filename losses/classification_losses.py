import torch.nn.functional as F
from confs_functions import get_func_args


def bceLoss(output, target, **kwargs):
    lossFunc = F.binary_cross_entropy
    funcArgs = get_func_args(kwargs, lossFunc)
    if output.shape[1] > 1:
        output = output[:, 1:2]
    loss = lossFunc(output, target, **funcArgs)

    return loss


def crossEntropy(output, target, **kwargs):
    lossFunc = F.cross_entropy
    funcArgs = get_func_args(kwargs, lossFunc)
    target = target[:, 0].long()
    loss = lossFunc(output, target, **funcArgs)

    return loss

