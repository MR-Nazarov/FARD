import torch
import copy
import numpy as np
from losses.fsim import FSIM
import lpips
from pytorch_ssim import SSIM3D
from PSNR_3D import PSNR3D
from sklearn.metrics import (roc_curve,
                             auc,
                             average_precision_score,
                             precision_recall_curve)
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from einops import rearrange
import torch.nn.functional as F

def fix_dimensions(x):
    if x.ndim > 1:
        x = x.flatten()

    return x


def calc_conf_mat(y_true, y_pred, N):
    y_pred = fix_dimensions(y_pred)
    y_true = fix_dimensions(y_true)
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat((y, torch.zeros(N * N - len(y),
                                      dtype=torch.long).to(y.device)))
    y = y.reshape(N, N)

    return y


def AP(confMat, batch=None, checkZeroExamples=True):
    if checkZeroExamples:
        _any = check_zero_examples(batch)
        if not torch.all(_any) and torch.any(_any):
            _batch = {}
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    _batch[k] = batch[k][_any]
                else:
                    _batch[k] = batch[k]
            return AP(confMat,
                      _batch,
                      checkZeroExamples=False)
        elif not torch.any(_any):
            # arbitrary number - in case all targets are empty slices
            return torch.tensor([0.5])
    xx, yy, thresholds = precision_recall_curve(y_true=batch['target'].cpu().data.flatten(),
                                                probas_pred=batch['output'].cpu().data.flatten())
    if xx[0] > 0.2:
        xx, yy, thresholds = precision_recall_curve(y_true=batch['target'].cpu().data.flatten(),
                                                    probas_pred=torch.nn.Softmax(dim=0)(batch['output'])
                                                    .cpu().data.flatten())
    _dict = {}
    _dict['x'] = xx
    _dict['y'] = yy
    _dict['ths'] = thresholds
    batch['precision_recall_curve'] = _dict
    ap = auc(np.sort(xx), yy)

    return torch.tensor([ap])


def check_zero_examples(batch):
    _any = batch['target'].any(dim=-1)
    for _ in range(2, batch['target'].ndim):
        _any = _any.any(dim=-1)
    return _any


def TP(confMat, batch=None):
    tp = torch.diag(confMat)
    return tp


def FN(confMat, batch=None):
    _conf = copy.deepcopy(confMat)
    _conf.fill_diagonal_(0)
    fn = torch.sum(_conf, dim=1)
    return fn


def TN(confMat, batch=None):
    fp = FP(confMat)
    fn = FN(confMat)
    tp = TP(confMat)
    tn_not = fp + fn + tp
    all = torch.sum(confMat)
    tn = all - tn_not
    return tn


def FP(confMat):
    _conf = copy.deepcopy(confMat)
    _conf.fill_diagonal_(0)
    fp = torch.sum(_conf, dim=0)
    return fp


def Recall(confMat, batch=None):
    eps = 1e-10
    tp = TP(confMat)
    fn = FN(confMat)
    return tp / (tp + fn + eps)


def Accuracy(confMat, batch=None):
    eps = 1e-10
    tn = TN(confMat)
    tp = TP(confMat)
    fp = FP(confMat)
    fn = FN(confMat)
    return (tp + tn) / (tp + tn + fp + fn + eps)


def Precision(confMat, batch=None):
    eps = 1e-10
    tp = TP(confMat)
    fp = FP(confMat)

    return tp / (tp + fp + eps)


def DiceCoefficient(confMat, batch=None):
    eps = 1e-10
    precision = Precision(confMat)
    recall = Recall(confMat)
    return (2 * precision * recall) / (precision + recall + eps)


def IoU(confMat, batch=None):
    eps = 1e-10
    tp = TP(confMat)
    fn = FN(confMat)
    fp = FP(confMat)

    return tp / (tp + fn + fp + eps)


def PSNR(batch):
    output = batch.output[0] if isinstance(batch.output, list) else batch.output
    if output.ndim > 4:
        psnr = PSNR3D().to(output.device)
        psnrOut = psnr(output.float(), batch.target.float())
    else:
        psnr = PeakSignalNoiseRatio().to(output.device)
        psnrOut = psnr(output, batch.target)
    return psnrOut


def SSIM(batch):
    output = batch.output[0] if isinstance(batch.output, list) else batch.output
    if output.ndim > 4:
        ssim = SSIM3D().to(output.device)
        ssimOut = ssim(output.float(), batch.target.float())
    else:
        ssim = StructuralSimilarityIndexMeasure().to(output.device)
        ssimOut = ssim(output, batch.target)
    return ssimOut

def similarity_measure(x, y, c):
    return (2 * x * y + c) / (x**2 + y**2 + c)

def conv2d_flexible(input, kernel):
    spatial_dims = input.dim() - 2
    if spatial_dims == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        padding = (1, 1)
    elif spatial_dims == 3:
        kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        padding = (1, 1, 1)
    else:
        raise ValueError(f"Unsupported number of spatial dimensions: {spatial_dims}")
    return F.conv2d(input, kernel.to(input.device), padding=padding)

def fsim_single_channel(output, target, T1=0.85, T2=160):
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float()
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float()

    output_grad_x = conv2d_flexible(output, kernel_x)
    output_grad_y = conv2d_flexible(output, kernel_y)
    target_grad_x = conv2d_flexible(target, kernel_x)
    target_grad_y = conv2d_flexible(target, kernel_y)

    output_grad_mag = torch.sqrt(output_grad_x**2 + output_grad_y**2)
    target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2)

    output_pc = output_grad_mag
    target_pc = target_grad_mag

    S_PC = similarity_measure(output_pc, target_pc, T1)
    S_G = similarity_measure(output_grad_mag, target_grad_mag, T2)

    FSIM = torch.sum(S_PC * S_G) / torch.sum(torch.ones_like(S_PC))

    return FSIM

def fsim(output, target, T1=0.85, T2=160):
    assert output.shape == target.shape
    assert output.dim() >= 3

    num_channels = output.shape[1]
    fsim_values = []

    for i in range(num_channels):
        fsim_value = fsim_single_channel(output[:, i:i+1, ...], target[:, i:i+1, ...], T1, T2)
        fsim_values.append(fsim_value)

    return torch.mean(torch.stack(fsim_values))  # Return tensor instead of float

class MetricsFSIM:
    def __init__(self):
        pass

    def FSIM(self, x):
        return fsim(x.output, x.target)

# Create a global instance of MetricsFSIM
metrics_fsim = MetricsFSIM()

# Top-level FSIM function
def FSIM(x):
    return metrics_fsim.FSIM(x)

class MetricsLPIPS:
    def __init__(self):
        self.lpips_model = None

    def initialize_lpips(self):
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net='vgg').cuda()

    def LPIPS(self, x):
        self.initialize_lpips()
        sum_loss = 0.0
        for channel in range(x.output.shape[1]):
            sum_loss += self.lpips_model(
                x.output[:, channel:channel + 1, ...].expand(-1, 3, -1, -1),
                x.target[:, channel:channel + 1, ...].expand(-1, 3, -1, -1)
            )

        return sum_loss / x.output.shape[1]

# Create a global instance of MetricsLPIPS
metrics_lpips = MetricsLPIPS()

# Other metric functions...

def LPIPS(x):
    return metrics_lpips.LPIPS(x)


def MAE(batch):
    """Mean Absolute Error between output and target."""
    mae = torch.mean(torch.abs(batch.output - batch.target))
    return mae
