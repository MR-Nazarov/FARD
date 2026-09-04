# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.
#
# MMMD-Net baseline. The original is a two-input, acceleration-aware image-domain
# model with directionally elongated kernels; this work extends it to three
# contrasts. The extension is this work's, the architecture is not:
#
#   Hod, G., Green, M., Waserman, M., Konen, E., Shrot, S., Nelkenbaum, I.,
#   Kiryati, N., Mayer, A., 2023. Complementary phase encoding for pair-wise
#   neural deblurring of accelerated brain MRI. In: Computer Vision - ECCV 2022
#   Workshops, Springer Nature Switzerland, Cham, pp. 268-280.
#
# The DnCNN-1 / DnCNN-2 / DnCNN-3 columns of the paper's Figure 3 are the
# single-, two- and three-contrast variants of this model.
#
# In this repository: the network for the MNI ablation (Experiment A, cloned vs
# genuine channels). Configs inherit `cnnModule: DnCNN` from confs/defaults.yaml.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union




def init_weights(init_type='xavier'):
    """Weight initialiser.

    Adapted from RDUNet (https://github.com/JavierGurrola/RDUNet, MIT). Inlined so
    the RDUNet comparison model itself need not be redistributed -- it was used
    for nothing else here.
    """
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer


class DnCNN(nn.Module):
    def __init__(self, kernels: Tuple[int, int] = (3, 7), batch_norm: bool = False,
                 num_layers: int = 10,in_ch=1, dropout_rate: float=0.0):
        super().__init__()
        self.first_branch = DnCNNCore(in_channels=in_ch, kernel=kernels, batch_norm=batch_norm, dropout_rate=dropout_rate ,num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.first_branch(x)]


class DnCNNCore(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            hidden_channels: int = 64,
            out_channels: int = 1,
            kernel: Union[Tuple[int, int], int] = (3, 3),
            num_layers: int = 10,
            batch_norm: bool = False,
            dropout_rate: float = 0.0
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        # Standardize kernel input
        if isinstance(kernel, int):
            kernel = (kernel, kernel)

        padding = tuple(k // 2 for k in kernel)

        # Create layers dynamically
        self.layers = nn.ModuleList()

        # First layer: input to hidden
        self.layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel, padding=padding))

        # Middle layers: hidden to hidden
        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel, padding=padding))

        # Last layer: hidden to output
        self.layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel, padding=padding))

        # Batch norm layers if enabled (for all except last layer)
        if batch_norm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm2d(hidden_channels)
                for _ in range(num_layers - 1)  # No batch norm for final layer
            ])
        self.batch_norm = batch_norm

        init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through all but last layer
        for i, conv in enumerate(self.layers[:-1]):
            x_conv = conv(x if i == 0 else x_prev)
            if self.batch_norm:
                x_conv = self.bn_layers[i](x_conv)
            x_prev = F.relu(x_conv)

        # Final layer with residual connection
        out = x[:, 0:1] + self.layers[-1](x_prev)
        return out


if __name__ == "__main__":
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Instantiate Model
    # Using 3 input channels (RGB) and kernel size (3, 7)
    model = DnCNN(in_ch=3, num_layers=10, kernels=(3,7)).to(device)

    # 3. Calculate Parameters (Capacity)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # 4. Calculate FLOPs (Computational Cost)
    print("-" * 30)
    try:
        from thop import profile

        # Standard convention: Calculate FLOPs for 1 image
        # Shape: (Batch=1, Channels=3, Height=256, Width=256)
        flops_input = torch.randn(1, 3, 256, 256).to(device)

        # Calculate MACs
        macs, _ = profile(model, inputs=(flops_input,), verbose=False)

        # Convert to GFLOPs
        print(f"GFLOPs (per 256x256 image): {macs / 1e9:.4f} G")

    except ImportError:
        print("Error: 'thop' library not installed. Run 'pip install thop' to see FLOPs.")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
    print("-" * 30)

    # 5. Forward Pass with Batch Size 4
    # Shape: (Batch=4, Channels=3, Height=256, Width=256)
    dummy_x = torch.randn(4, 3, 256, 256).to(device)

    output_list = model(dummy_x)
    output_tensor = output_list[0]

    print(f"Input shape:  {dummy_x.shape}")
    print(f"Output shape: {output_tensor.shape}")