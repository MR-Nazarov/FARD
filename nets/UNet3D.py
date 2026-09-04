# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.
#
# 3D U-Net baseline: a thin wrapper over MONAI's UNet at the paper's
# configuration. No MONAI source is vendored; MONAI is a pinned dependency.
#
#   Cardoso, M.J., et al., 2022. MONAI: An open-source framework for deep
#   learning in healthcare. https://monai.io

import torch
from torch import nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 channels=(64, 128, 256, 512),
                 strides=(2, 2, 2),
                 num_res_units=2,
                 **kwargs):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

    def forward(self, x):
        return [self.model(x)]
