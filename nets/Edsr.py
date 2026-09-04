# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.
#
# EDSR+MMCHA baseline, adapted to the post-reconstruction fusion setting of this
# paper. The adaptation is this work's; the components it builds on are not.
#
# Multi-head channel attention (MMCHA):
#   Georgescu, M.I., Ionescu, R.T., Miron, A.I., Savencu, O., Ristea, N.C.,
#   Verga, N., Khan, F.S., 2022. Multimodal multi-head convolutional attention
#   with various kernel sizes for medical image super-resolution.
#   arXiv:2204.04218.  https://arxiv.org/abs/2204.04218
#
# EDSR backbone, forked from (MIT):
#   Lim, B., Son, S., Kim, H., Nah, S., Lee, K.M., 2017. Enhanced deep residual
#   networks for single image super-resolution. CVPRW.
#   https://github.com/sanghyun-son/EDSR-PyTorch

"""
Clean and readable implementation of EDSR with Multi-Head Channel Attention (MHCA)

Forked from https://github.com/sanghyun-son/EDSR-PyTorch.
MHCA code has the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) licence.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration class for EDSR-MHCA model"""
    # Input/Output configuration
    n_colors: int = 3  # Number of input channels
    n_colors_out: int = 1  # Number of output channels
    scale: int = 1  # Super-resolution scale factor

    # Architecture configuration
    n_resblocks: int = 16  # Number of residual blocks
    n_feats: int = 64  # Number of feature channels
    kernel_size: int = 3  # Convolution kernel size

    # Attention configuration
    use_mhca: bool = True  # Enable MHCA attention
    attention_heads: int = 3  # Number of attention heads (2 or 3)
    attention_ratio: float = 8.0  # Channel reduction ratio for attention

    # Training configuration
    res_scale: float = 1.0  # Residual connection scaling
    rgb_range: float = 255.0  # RGB value range for normalization

    @classmethod
    def create_multimodal_config(cls) -> 'ModelConfig':
        """Configuration for multi-modal super-resolution (paper's approach)"""
        return cls(
            n_colors=3,
            n_colors_out=1,
            scale=1,
            n_resblocks=16,
            n_feats=64,
            use_mhca=True,
            attention_heads=3
        )

    @classmethod
    def create_enhancement_config(cls) -> 'ModelConfig':
        """Configuration for single modality enhancement (no upsampling)"""
        return cls(
            n_colors=1,
            n_colors_out=1,
            scale=1,
            n_resblocks=16,
            n_feats=64,
            use_mhca=True,
            attention_heads=3
        )

    @classmethod
    def create_large_config(cls) -> 'ModelConfig':
        """Configuration for large model"""
        return cls(
            n_colors=3,
            n_colors_out=1,
            scale=4,
            n_resblocks=32,
            n_feats=256,
            use_mhca=True,
            attention_heads=3
        )


class MeanShift(nn.Conv2d):
    """Mean shift layer for input normalization"""

    def __init__(self, rgb_range: float, n_colors: int = 3, sign: int = -1):
        super().__init__(n_colors, n_colors, kernel_size=1)

        # Default RGB statistics
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        # Handle different channel configurations
        if n_colors == 1:
            mean = torch.tensor([0.449])
            std = torch.tensor([1.0])
        elif n_colors == 3:
            mean = torch.tensor(rgb_mean)
            std = torch.tensor(rgb_std)
        else:
            # Repeat RGB values for multi-channel inputs
            repeats = (n_colors // 3) + 1
            mean = torch.tensor(rgb_mean * repeats)[:n_colors]
            std = torch.tensor(rgb_std * repeats)[:n_colors]

        # Set up normalization parameters
        self.weight.data = torch.eye(n_colors).view(n_colors, n_colors, 1, 1) / std.view(n_colors, 1, 1, 1)
        self.bias.data = sign * rgb_range * mean / std

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    """Default convolution with automatic padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class MultiHeadChannelAttention(nn.Module):
    """Multi-Head Channel Attention (MHCA) module"""

    def __init__(self, n_feats: int, ratio: float = 8.0, num_heads: int = 3):
        super().__init__()

        if num_heads not in [2, 3]:
            raise ValueError("num_heads must be 2 or 3")

        self.num_heads = num_heads
        out_channels = int(n_feats // ratio)

        # Channel attention (1x1 convolutions)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(n_feats, out_channels, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(out_channels, n_feats, kernel_size=1, bias=True)
        )

        # Spatial attention 1 (3x3 convolutions)
        self.spatial_attention_1 = nn.Sequential(
            nn.Conv2d(n_feats, out_channels, kernel_size=(3,3), padding=0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels, n_feats, kernel_size=(3,3), padding=0, bias=True)
        )

        # Spatial attention 2 (5x5 convolutions) - only for 3-head version
        if num_heads == 3:
            self.spatial_attention_2 = nn.Sequential(
                nn.Conv2d(n_feats, out_channels, kernel_size=(5,5), padding=0, bias=True),
                nn.ReLU(True),
                nn.ConvTranspose2d(out_channels, n_feats, kernel_size=(5,5), padding=0, bias=True)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply channel attention
        channel_att = self.channel_attention(x)

        # Apply spatial attention 1
        spatial_att_1 = self.spatial_attention_1(x)

        # Combine attentions
        combined_att = channel_att + spatial_att_1

        # Add spatial attention 2 for 3-head version
        if self.num_heads == 3:
            spatial_att_2 = self.spatial_attention_2(x)
            combined_att = combined_att + spatial_att_2

        # Apply sigmoid and multiply with input
        attention_mask = self.sigmoid(combined_att)
        return x * attention_mask


class ResidualBlock(nn.Module):
    """Residual block with optional MHCA attention"""

    def __init__(self,
                 n_feats: int,
                 kernel_size: int = 3,
                 use_attention: bool = False,
                 attention_heads: int = 3,
                 attention_ratio: float = 8.0,
                 res_scale: float = 1.0,
                 bias: bool = True):
        super().__init__()

        self.res_scale = res_scale

        # Main residual path
        self.conv1 = default_conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu = nn.ReLU(True)
        self.conv2 = default_conv(n_feats, n_feats, kernel_size, bias=bias)

        # Optional attention
        if use_attention:
            self.attention = MultiHeadChannelAttention(n_feats, attention_ratio, attention_heads)
        else:
            self.attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main residual path
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = residual * self.res_scale

        # Add skip connection
        out = x + residual

        # Apply attention if enabled
        if self.attention is not None:
            out = self.attention(out)

        return out


class Upsampler(nn.Module):
    """Upsampling module using sub-pixel convolution"""

    def __init__(self, n_feats: int, scale: int, bias: bool = True):
        super().__init__()

        modules = []

        if (scale & (scale - 1)) == 0:  # Check if scale is power of 2
            # Handle powers of 2 (2, 4, 8, 16, ...)
            for _ in range(int(math.log(scale, 2))):
                modules.extend([
                    default_conv(n_feats, 4 * n_feats, 3, bias),
                    nn.PixelShuffle(2)
                ])
        elif scale == 3:
            # Handle scale factor 3
            modules.extend([
                default_conv(n_feats, 9 * n_feats, 3, bias),
                nn.PixelShuffle(3)
            ])
        else:
            raise NotImplementedError(f"Scale factor {scale} is not supported")

        self.upsampler = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(x)


class EDSR_MHCA(nn.Module):
    """Enhanced Deep Super-Resolution with Multi-Head Channel Attention"""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        # If no config is provided, use the default multimodal one
        if config is None:
            config = ModelConfig.create_multimodal_config()

        self.config = config
        self.use_mean_shift = False

        # ... (The rest of your __init__ code remains exactly the same) ...
        # Mean shift layers (optional)
        if self.use_mean_shift:
            self.sub_mean = MeanShift(config.rgb_range, config.n_colors, sign=-1)
            self.add_mean = MeanShift(config.rgb_range, config.n_colors_out, sign=1)

        # Head: Initial feature extraction
        self.head = default_conv(config.n_colors, config.n_feats, config.kernel_size)

        # Body: Residual blocks
        body_modules = []
        for _ in range(config.n_resblocks):
            body_modules.append(
                ResidualBlock(
                    n_feats=config.n_feats,
                    kernel_size=config.kernel_size,
                    use_attention=config.use_mhca,
                    attention_heads=config.attention_heads,
                    attention_ratio=config.attention_ratio,
                    res_scale=config.res_scale
                )
            )

        # Final convolution in body
        body_modules.append(default_conv(config.n_feats, config.n_feats, config.kernel_size))
        self.body = nn.Sequential(*body_modules)

        # Tail: Upsampling and final output
        tail_modules = []
        if config.scale > 1:
            tail_modules.append(Upsampler(config.n_feats, config.scale))
        tail_modules.append(default_conv(config.n_feats, config.n_colors_out, config.kernel_size))
        self.tail = nn.Sequential(*tail_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional mean shift
        if self.use_mean_shift:
            x = self.sub_mean(x)

        # Feature extraction
        x = self.head(x)

        # Residual learning with global skip connection
        residual = self.body(x)
        x = x + residual

        # Upsampling and final output
        x = self.tail(x)

        # Optional mean shift back
        if self.use_mean_shift:
            x = self.add_mean(x)

        return x

    def get_parameter_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """Get model information summary"""
        return {
            'total_parameters': self.get_parameter_count(),
            'input_channels': self.config.n_colors,
            'output_channels': self.config.n_colors_out,
            'scale_factor': self.config.scale,
            'residual_blocks': self.config.n_resblocks,
            'feature_channels': self.config.n_feats,
            'attention_enabled': self.config.use_mhca,
            'attention_heads': self.config.attention_heads if self.config.use_mhca else 0
        }


# Factory functions for easy model creation
def create_edsr_mhca_multimodal() -> EDSR_MHCA:
    """Create EDSR-MHCA model for multi-modal super-resolution"""
    config = ModelConfig.create_multimodal_config()
    return EDSR_MHCA(config)


def create_edsr_mhca_enhancement() -> EDSR_MHCA:
    """Create EDSR-MHCA model for single modality enhancement"""
    config = ModelConfig.create_enhancement_config()
    return EDSR_MHCA(config)


def create_edsr_mhca_large() -> EDSR_MHCA:
    """Create large EDSR-MHCA model"""
    config = ModelConfig.create_large_config()
    return EDSR_MHCA(config)


if __name__ == "__main__":
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ==========================================
    # Part 1: Model Comparisons (Parameters)
    # ==========================================
    print("=== EDSR-MHCA Model Comparison (Capacity) ===\n")

    # Multi-modal model
    model_multimodal = create_edsr_mhca_multimodal().to(device)
    print(f"Multi-modal Model Params:   {model_multimodal.get_parameter_count():,}")

    # Enhancement model (Single Channel)
    model_enhancement = create_edsr_mhca_enhancement().to(device)
    print(f"Enhancement Model Params:   {model_enhancement.get_parameter_count():,}")

    # Large model
    model_large = create_edsr_mhca_large().to(device)
    print(f"Large Model Params:         {model_large.get_parameter_count():,}")
    print()

    # ==========================================
    # Part 2: FLOPs Calculation (Computational Cost)
    # ==========================================
    print("=== FLOPs Calculation (Computational Cost) ===")
    try:
        from thop import profile

        # We calculate FLOPs for the Enhancement Model (1 channel)
        # using a standard 256x256 image resolution.
        input_res = (1, 256, 256)
        input_tensor = torch.randn(1, *input_res).to(device)

        print(f"Input Shape: {input_tensor.shape}")

        # Calculate MACs (Multiply-Accumulate operations)
        macs, params = profile(model_enhancement, inputs=(input_tensor,), verbose=False)

        # Convert to GFLOPs (1 GFLOP = 10^9 FLOPs)
        gflops = macs / 1e9

        print(f"GFLOPs:      {gflops:.4f} G")
        print(f"Params:      {params / 1e6:.4f} M")
        print("Note: FLOPs are calculated for a single 256x256 image.")

    except ImportError:
        print("Error: 'thop' library not found. Please run 'pip install thop' to calculate FLOPs.")
    except Exception as e:
        print(f"An error occurred during FLOPs calculation: {e}")

    # ==========================================
    # Part 3: Forward Pass Test
    # ==========================================
    print("\n=== Forward Pass Tests ===")
    with torch.no_grad():
        # Single modality test (Batch size 4)
        single_input = torch.randn(4, 1, 256, 256).to(device)
        enhanced_output = model_enhancement(single_input)
        print(f"Enhancement Pass: {single_input.shape} → {enhanced_output.shape}")

