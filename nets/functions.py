import torch
import functools
import operator
import numpy as np
import kornia
from skimage.measure import label
from torch.nn import functional as F


def cc_with_scipy(im):
    cc = label(im)
    return cc[None, None]


def cc_with_torch(im):
    cc = kornia.contrib.connected_components(im)
    return cc.long()


def combine_batch_and_channel(x):
    x = x.permute((0, 2, 1, 3, 4))
    shape = x.shape
    return x.reshape((-1, *shape[2:]))


def replace_batch_and_channel(x):
    dims = list(range(x.ndim))
    return x.permute((1, 0, *dims[2:]))


def split_batch_and_channel(x, sizeZ):
    shape = x.shape
    x = x.reshape((-1, sizeZ, *shape[1:]))
    x = x.permute((0, 2, 1, 3, 4))
    return x


def threshold_contrast_2d(im, B, maxNumOfPixels, th, useTorch=False):
    if not useTorch:
        cc = cc_with_scipy(im)
    else:
        cc = cc_with_torch(im)

    if useTorch:
        ccs = torch.unique(cc)
    else:
        ccs = np.unique(cc)

    for ccVal in ccs:

        indsCC = cc == ccVal
        if isinstance(indsCC, np.ndarray):
            indsCC = torch.tensor(indsCC)
        valToCheck = torch.max(B[indsCC])
        if valToCheck < th or torch.sum(indsCC) > maxNumOfPixels:
            B[indsCC] = 0

    return B


def threshold_contrast_3d(im, B, maxNumOfPixels, th, useTorch=False):
    for Slice in range(B.shape[2]):
        if useTorch:
            _im = im[:, :, Slice]
        else:
            _im = im[Slice]

        _B = threshold_contrast_2d(im=_im,
                                   B=B[:, :, Slice],
                                   th=th,
                                   maxNumOfPixels=maxNumOfPixels,
                                   useTorch=useTorch)
        B[:, :, Slice] = _B

    return B


def calc_contrast_map_with_th(A, kernel_size=7, multiplyContrast=1):
    if A.ndim == 4:
        I1 = torch.nn.AvgPool2d(kernel_size=kernel_size,
                                stride=1,
                                padding=int(torch.floor(torch.Tensor([kernel_size / 2]))[0]))(A)
        I2 = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(A)
    else:
        I1 = torch.nn.AvgPool3d(kernel_size=kernel_size,
                                stride=1,
                                padding=int(torch.floor(torch.Tensor([kernel_size / 2]))[0]))(A)
        I2 = torch.nn.AvgPool3d(kernel_size=3, stride=1, padding=1)(A)

    B = I2 - I1
    B[B < 0] = 0
    B = B * multiplyContrast
    maxNumOfPixels = 100
    th = 0
    im = copy.deepcopy(B)
    pre_cc_th = 0.001
    im[im < pre_cc_th] = 0
    im[im > 0] = 1

    if im.ndim > 4:
        B = threshold_contrast_3d(im=im,
                                  B=B,
                                  maxNumOfPixels=maxNumOfPixels,
                                  useTorch=True,
                                  th=th)
    else:
        B = threshold_contrast_2d(im=im,
                                  B=B,
                                  maxNumOfPixels=maxNumOfPixels,
                                  useTorch=True,
                                  th=th)

    return B


def calc_contrast_map(A, kernel_size=7, multiplyContrast=1):
    if A.ndim == 4:
        I1 = torch.nn.AvgPool2d(kernel_size=kernel_size,
                                stride=1,
                                padding=int(torch.floor(torch.Tensor([kernel_size / 2]))[0]))(A)
        I2 = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(A)
    else:
        I1 = torch.nn.AvgPool3d(kernel_size=kernel_size,
                                stride=1,
                                padding=int(torch.floor(torch.Tensor([kernel_size / 2]))[0]))(A)
        I2 = torch.nn.AvgPool3d(kernel_size=3, stride=1, padding=1)(A)

    B = I2 - I1
    B[B < 0] = 0
    B = B * multiplyContrast

    return B


def calc_fc_shape(model,
                  im_size,
                  multi_scale=False,
                  input_channels=1):
    out = model(torch.rand(1, input_channels, *im_size))
    if isinstance(out, list):
        if not multi_scale:
            out = out[-1]
        else:
            num_features_before_fc = \
                [functools.reduce(operator.mul,
                                  list(o.shape)) for o in out]
    if not multi_scale:
        num_features_before_fc = functools.reduce(operator.mul,
                                                  list(out.shape))

    return num_features_before_fc


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(torch.nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def convert_fig_to_ar(fig):
    img = np.frombuffer(fig.canvas.tostring_rgb(),
                        dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()
                      [::-1] + (3,))

    return img


def hypercolumn(x):
    shapes = torch.cat([torch.tensor(xx.shape)[None] for xx in x])
    maxShape = shapes.max(axis=0).values
    maxShape = tuple(maxShape.tolist())
    for idx, xx in enumerate(x):
        x[idx] = F.interpolate(xx, size=maxShape[-3:])
    x = torch.cat(x, axis=1)

    return x


def extract_features(x, model, normalize, layerNames):
    out = []
    if normalize:
        x = (x - x.min()) / (x.max() - x.min())
        x = x * 255
    if x.shape[1] == 1:
        x = torch.cat((x, x, x), dim=1)
    if '-1' in layerNames:
        out.append(x)

    for name, module in model._modules.items():
        if len(out) >= len(layerNames):
            continue
        try:
            x = module(x)
        except RuntimeError:
            print(f'Module {module}\n'
                  f'cannot process tensor with shape {x.shape}')
        if name in layerNames:
            out.append(x)

    out = out[0] if len(out) == 1 else out

    return out
