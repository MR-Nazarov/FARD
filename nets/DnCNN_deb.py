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
# In this repository: `UpgradedDnCNN`, the network for the reg-B ablation
# (Experiment B, three-contrast) -- e.g. confs/examples/sheba_3ch_t1.yaml.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class Attention_block(nn.Module):
    """Memory-efficient self-attention module"""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Use more memory-efficient implementation with spatial dimensions flattened
        batch_size, C, H, W = x.size()

        # Reduce spatial dimensions first
        x_pooled = F.adaptive_avg_pool2d(x, output_size=(H // 4, W // 4))

        # Process with reduced size
        query = self.query(x_pooled).flatten(2).permute(0, 2, 1)
        key = self.key(x_pooled).flatten(2)
        value = self.value(x_pooled).flatten(2).permute(0, 2, 1)

        # Compute attention with reduced size
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(attention, value)

        # Reshape back
        out = out.permute(0, 2, 1).view(batch_size, C, H // 4, W // 4)

        # Upsample to original size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # Apply residual connection with learnable gamma
        return self.gamma * out + x



class Frequency_module(nn.Module):
    """Memory-efficient frequency domain processing"""

    def __init__(self, channels,k_space_size=32):
        super().__init__()
        # Process only a subset of frequencies to save memory
        reduced_channels = channels // 2
        self.k_space_size = k_space_size
        self.channel_reduction = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.freq_conv = nn.Conv2d(reduced_channels * 2, reduced_channels * 2, kernel_size=1)
        self.channel_expansion = nn.Conv2d(reduced_channels, channels, kernel_size=1)

    def forward(self, x):
        # Channel reduction
        x_reduced = self.channel_reduction(x)

        # Process at reduced size in frequency domain
        height, width = x_reduced.shape[2], x_reduced.shape[3]

        # Use smaller FFT size
        fft_h, fft_w = min(height, self.k_space_size), min(width, self.k_space_size)
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

        # Process in frequency domain
        x_freq_processed = self.freq_conv(x_freq_combined)

        # Split back to real and imaginary
        channels = x_downsampled.size(1)
        x_freq_real_proc = x_freq_processed[:, :channels]
        x_freq_imag_proc = x_freq_processed[:, channels:]

        # Reconstruct complex tensor
        x_freq_proc_complex = torch.complex(x_freq_real_proc, x_freq_imag_proc)

        # Convert back to spatial domain
        x_proc = torch.fft.irfft2(x_freq_proc_complex, s=(fft_h, fft_w), norm='ortho')

        # Upsample if necessary
        if height > fft_h or width > fft_w:
            x_proc = F.interpolate(x_proc, size=(height, width), mode='bilinear', align_corners=False)

        # Channel expansion
        x_proc = self.channel_expansion(x_proc)

        # Residual connection
        return x + x_proc


class UpgradedDnCNN(nn.Module):
    def __init__(
            self,
            total_channels: int = 3,  # Total channels in concatenated input
            hidden_channels: int = 64,
            out_channels: int = 1,
            kernel: Union[Tuple[int, int], int] = (3, 3),
            num_layers: int = 10,
            batch_norm: bool = False,
            dropout_rate: float = 0.0,
            use_attention: bool = True,
            use_frequency: bool = True,
            growth_rate: int = 16,  # For memory efficiency
            k_space_size: int = 32
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.use_attention = use_attention
        self.use_frequency = use_frequency

        # Standardize kernel input
        if isinstance(kernel, int):
            kernel = (kernel, kernel)

        padding = tuple(k // 2 for k in kernel)

        # Split the concatenated input and process - using fewer channels initially
        reduced_channels = hidden_channels // 2
        self.main_conv = nn.Conv2d(1, reduced_channels // 2, kernel_size=kernel, padding=padding)
        self.aux_conv = nn.Conv2d(total_channels - 1, reduced_channels // 2, kernel_size=kernel, padding=padding)

        # Fusion layer to combine main and auxiliary features
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Create layers dynamically with gradually increasing channels to save memory
        self.layers = nn.ModuleList()
        current_channels = reduced_channels

        # First half of the network - increasing channels
        expand_layers = (num_layers - 2) // 2
        for i in range(expand_layers):
            # Gradually increase channels
            next_channels = min(current_channels + growth_rate, hidden_channels)

            # Add convolution layer
            self.layers.append(
                nn.Conv2d(current_channels, next_channels, kernel_size=kernel, padding=padding)
            )
            current_channels = next_channels

            # Add attention or frequency modules sparingly (to save memory)
            if use_attention and i % 3 == 0:  # Less frequent attention
                self.layers.append(Attention_block(current_channels))

            if use_frequency and i % 4 == 0:  # Less frequent frequency processing
                self.layers.append(Frequency_module(current_channels,k_space_size=k_space_size))

        # Second half - decreasing channels
        for i in range(num_layers - 2 - expand_layers):
            # Gradually decrease channels
            next_channels = max(current_channels - growth_rate, reduced_channels)

            # Add convolution layer
            self.layers.append(
                nn.Conv2d(current_channels, next_channels, kernel_size=kernel, padding=padding)
            )
            current_channels = next_channels

            # Add attention or frequency modules sparingly (to save memory)
            if use_attention and i % 3 == 1:  # Less frequent attention
                self.layers.append(Attention_block(current_channels))

        # Last layer: hidden to output
        self.final_layer = nn.Conv2d(current_channels, out_channels, kernel_size=kernel, padding=padding)

        # Batch norm layers if enabled
        if batch_norm:
            # Count the number of convolutional layers (excluding attention and frequency modules)
            conv_layers = [layer for layer in self.layers if isinstance(layer, nn.Conv2d)]
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm2d(layer.out_channels) for layer in conv_layers
            ])
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

        main_input = x[:, 0:1]
        aux_input = x[:, 1:]

        main_features = self.main_conv(main_input)
        aux_features = self.aux_conv(aux_input)
        combined = torch.cat([main_features, aux_features], dim=1)
        x = self.fusion(combined)

        bn_idx = 0

        # Process through all layers
        for i, layer in enumerate(self.layers):
            # Apply the current layer
            x_prev = x
            x = layer(x)

            # Apply batch norm for conv layers
            if self.batch_norm and isinstance(layer, nn.Conv2d):
                x = self.bn_layers[bn_idx](x)
                bn_idx += 1

            # Apply ReLU for conv layers
            if isinstance(layer, nn.Conv2d):
                x = F.relu(x)

            # Apply dropout if enabled
            if self.dropout is not None and isinstance(layer, nn.Conv2d):
                x = self.dropout(x)

            # Add residual connections every few layers to help gradient flow
            if isinstance(layer, nn.Conv2d) and i > 0 and i % 3 == 0 and x.shape == x_prev.shape:
                x = x + x_prev

        # Final layer
        out = self.final_layer(x)

        # Residual connection from input
        out = main_input + out

        return out


# Memory-efficient training function
def train_with_memory_efficiency(model, train_loader, criterion, optimizer, device, gradient_accumulation_steps=4):
    model.train()
    total_loss = 0

    # Process batches with gradient accumulation
    optimizer.zero_grad()
    for i, (inputs, targets) in enumerate(train_loader):
        # Move to device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():  # Use mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, targets) / gradient_accumulation_steps

        # Backward pass with gradient accumulation
        loss.backward()
        total_loss += loss.item() * gradient_accumulation_steps

        # Update weights after accumulating gradients
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Clean up to save memory
        del inputs, targets, outputs, loss
        torch.cuda.empty_cache()

    return total_loss / len(train_loader)


# Example usage
if __name__ == "__main__":
    # Set memory management
    torch.cuda.empty_cache()

    # Use smaller batch size and enable memory-efficient operations
    model = UpgradedDnCNN(
        total_channels=3,
        hidden_channels=48,  # Reduced from 64
        num_layers=8,  # Reduced from 10
        use_attention=True,
        use_frequency=True,
        growth_rate=8  # Control channel growth
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test with a smaller input for memory efficiency
    batch_size = 2  # Reduced batch size
    x = torch.randn(batch_size, 3, 256, 256, device=device)

    # Forward pass with memory tracking
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():  # Disable gradient tracking for testing
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")