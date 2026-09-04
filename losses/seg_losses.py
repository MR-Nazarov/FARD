import torch
import losses.unified_focal_loss as unified_focal_loss
import losses.multi_class_focal_loss as multi_class_focal_loss
from torch.nn import functional as F
from torchmetrics import Dice


def focalLoss(self, output, target):
    loss = 0
    if not hasattr(self, 'focal_loss'):
        focal_type = self.kwargs.get('focal_loss_type',
                                     'focal_loss')
        self.focal_loss = getattr(unified_focal_loss, focal_type, None)
        if self.focal_loss is None:
            self.focal_loss = getattr(multi_class_focal_loss, focal_type)
        fArgs = self.kwargs.get('focal_loss_args', {})
        if not isinstance(fArgs, dict):
            fArgs = {k.split('|')[0]: float(k.split('|')[1])
                     for k in fArgs.split(',')}
        self.focal_loss = self.focal_loss(**fArgs)
    loss = self.focal_loss(y_pred=output,
                           y_true=target.long())
    return loss


def diceLoss(output, target, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        target: a tensor of shape [B, 1, H, W].
        output: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """

    if output.shape[1] == 2:
        output = output[:, 1:]
    num_classes = output.shape[1]
    imDims = range(1, output.ndim - 1, 1)
    true_1_hot = torch.eye(num_classes + int(num_classes == 1)) \
        .to(target.device)[target.long().squeeze(1)]
    true_1_hot = true_1_hot.permute(0, -1, *imDims).float()
    if num_classes == 1:
        true_1_hot_f = true_1_hot[:, :1]
        true_1_hot_s = true_1_hot[:, 1:2]
        true_1_hot = torch.cat([true_1_hot_s,
                                true_1_hot_f],
                               dim=1)
        pos_prob = torch.sigmoid(output)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob,
                            neg_prob],
                           dim=1)
    else:
        probas = F.softmax(output, dim=1)
    true_1_hot = true_1_hot.type(output.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()

    return 1 - dice_loss
