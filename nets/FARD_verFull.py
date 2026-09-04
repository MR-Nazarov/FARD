# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 the FARD authors.
#
# The FARD architecture is licensed CC BY-NC 4.0 (non-commercial), NOT MIT like
# the rest of this repository. See LICENSE-FARD.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

from monai.networks.nets.vit import ViT
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.blocks import TransformerBlock
from monai.networks.layers import Conv
from enhanced_attention.enhanced_attention import EnhancedAttention_block






class Frequency_module(nn.Module):
    """Enhanced frequency domain processing with multi-scale processing"""

    def __init__(self, channels, k_space_size=32):
        super().__init__()
        # Process frequencies at multiple scales
        reduced_channels = channels // 2
        self.k_space_size = k_space_size

        # Channel processing components
        self.channel_reduction = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.channel_expansion = nn.Conv2d(reduced_channels, channels, kernel_size=1)

        self.scales = [k_space_size]
        self.freq_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(reduced_channels * 2, reduced_channels * 2, kernel_size=1),
                #nn.BatchNorm2d(reduced_channels * 2),# do i need this?
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_channels * 2, reduced_channels * 2, kernel_size=1)
            ) for _ in self.scales
        ])

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1),
            #nn.BatchNorm2d(reduced_channels),# do i need this?
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Channel reduction
        x_reduced = self.channel_reduction(x)
        height, width = x_reduced.shape[2], x_reduced.shape[3]


        multi_scale_outputs = []

        for i, scale in enumerate(self.scales):
            # Use appropriate FFT size for this scale
            fft_h, fft_w = min(height, scale), min(width, scale)

            if height > fft_h or width > fft_w:
                x_downsampled = F.adaptive_avg_pool2d(x_reduced, (fft_h, fft_w))
            else:
                x_downsampled = x_reduced

            # Convert to frequency domain
            x_freq = torch.fft.rfft2(x_downsampled, norm='ortho')

            # Convert complex to channels
            x_freq_real = x_freq.real
            x_freq_imag = x_freq.imag
            x_freq_combined = torch.cat([x_freq_real, x_freq_imag], dim=1)

            # Process in frequency domain with current scale processor
            x_freq_processed = self.freq_convs[i](x_freq_combined)

            # Split back to real and imaginary
            channels = x_downsampled.size(1)
            x_freq_real_proc = x_freq_processed[:, :channels]
            x_freq_imag_proc = x_freq_processed[:, channels:]

            # Reconstruct complex tensor
            x_freq_proc_complex = torch.complex(x_freq_real_proc, x_freq_imag_proc)

            # Convert back to spatial domain
            x_proc = torch.fft.irfft2(x_freq_proc_complex, s=(fft_h, fft_w), norm='ortho')

            # Upsample to original size if needed
            if height > fft_h or width > fft_w:
                x_proc = F.interpolate(x_proc, size=(height, width), mode='bilinear', align_corners=False)

            multi_scale_outputs.append(x_proc)

        # Fuse multi-scale frequency outputs
        concat_features = torch.cat(multi_scale_outputs, dim=1)
        fused_features = self.fusion(concat_features)

        # Channel expansion and residual connection
        out = self.channel_expansion(fused_features)
        return x + out


class ConvBnReluBlock(nn.Module):
    """Enhanced convolution block with squeeze-excitation"""

    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)

        # Main convolution path
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

        # Squeeze-excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Main path
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # Apply SE attention
        x = x * self.se(x)

        # Residual connection
        x = x + residual
        return self.relu(x)


class FARD(nn.Module):
    def __init__(
            self,
            total_channels: int = 3,
            hidden_channels: int = 64,
            out_channels: int = 1,
            kernel: Union[Tuple[int, int], int] = (3, 3),
            batch_norm: bool = False,
            dropout_rate: float = 0.0,
            use_attention: bool = True,
            use_frequency: bool = True,
            split_channels: bool = False,
            gate_type: str = 'simple',  # Add this parameter
            k_space_size: int = 32,
            attention_type: str = 'standard',  # Add this parameter
            window_size: int = 7,
            reduction: int = 8,
            pool_factor: int = 4
    ):
        super().__init__()
        num_layers = 10
        if hidden_channels % 8 != 0:
            raise ValueError("hidden_channels must be divisible by 8")
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.use_attention = use_attention
        self.use_frequency = use_frequency
        self.split_channels = split_channels
        self.gate_type = gate_type
        self.reduction = reduction

        if isinstance(kernel, int):
            kernel = (kernel, kernel)

        padding = tuple(k // 2 for k in kernel)
        if self.split_channels:
            reduced_channels = hidden_channels // 2
            self.main_conv = nn.Conv2d(1, reduced_channels, kernel_size=kernel, padding=padding)
            self.aux_conv = nn.Conv2d(total_channels - 1, reduced_channels , kernel_size=kernel, padding=padding)

            # Fusion layer to combine main and auxiliary features
            self.fusion = nn.Sequential(
                nn.Conv2d(reduced_channels*2, reduced_channels*2, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        else:
            expanded_channels = 96  # 36 channels (clean fraction)
            self.aux_conv = nn.Conv2d(total_channels, expanded_channels, kernel_size=kernel, padding=padding)
            self.fusion = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=False),  # Changed
                nn.Conv2d(expanded_channels, hidden_channels, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=False)  # Changed
            )

        self.layers = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.frequency_modules = nn.ModuleList()

        start_channels = hidden_channels

        for i in range(num_layers):

            # Add convolutional layer
            self.layers.append(
                nn.Conv2d(start_channels, start_channels, kernel_size=kernel, padding=padding)
            )

            # Add attention modules at layers 2, 5, 8 if enabled
            # If not enabled, add a Conv+BN+ReLU block instead
            if use_attention and i in [1, 4, 7]:
                self.attention_modules.append(EnhancedAttention_block(
                    start_channels,
                    reduction=1,
                    attention_type=attention_type,
                    window_size=window_size,
                    gate_type = gate_type,
                    pool_factor=pool_factor
                ))
            else:
                # Add replacement Conv+BN+ReLU block for attention
                if i in [1, 4, 7]:
                    self.attention_modules.append(ConvBnReluBlock(start_channels, kernel))
                else:
                    self.attention_modules.append(None)

            # Add frequency modules layers 3, 6, 9 if enabled
            # If not enabled, add a Conv+BN+ReLU block instead
            if use_frequency and i in [2, 5, 8]:
                self.frequency_modules.append(Frequency_module(start_channels, k_space_size=k_space_size))
            else:
                # Add replacement Conv+BN+ReLU block for frequency
                if i in [2, 5, 8]:
                    self.frequency_modules.append(ConvBnReluBlock(start_channels, kernel))
                else:
                    self.frequency_modules.append(None)

            in_channels = start_channels

        # Final layer to produce output
        self.final_layer = nn.Conv2d(start_channels, out_channels, kernel_size=kernel, padding=padding)

        # Batch norm layers if enabled
        if batch_norm:
            self.bn_layers = nn.ModuleList([nn.BatchNorm2d(layer.out_channels) for layer in self.layers])
        self.batch_norm = batch_norm

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.split_channels:
            main_input = x[:, 0:1]
            aux_input = x[:, 1:]

            main_features = self.main_conv(main_input)
            aux_features = self.aux_conv(aux_input)
            combined = torch.cat([main_features, aux_features], dim=1)
        else:
            original_input = x
            combined = self.aux_conv(x)
            combined = torch.nn.functional.relu(combined)

        x = self.fusion(combined)
        #x = self.fusion(main_features, aux_features)
        for i, layer in enumerate(self.layers):
            x_prev = x
            x = layer(x)

            if self.batch_norm:
                x = self.bn_layers[i](x)

            x = F.relu(x)

            if self.dropout is not None:
                x = self.dropout(x)

            if self.attention_modules[i] is not None:
                x = self.attention_modules[i](x)

            if self.frequency_modules[i] is not None:
                x = self.frequency_modules[i](x)

            if i > 0 and i % 3 == 0 and x.shape == x_prev.shape:
                x = x + x_prev

        out = self.final_layer(x)

        # Residual connection from input
        if self.split_channels:
            out = main_input - out
        else:
            out = original_input[:,0:1] - out
        #out = torch.clamp(out, 0, 1)
        return out