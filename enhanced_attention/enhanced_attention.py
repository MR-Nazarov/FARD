# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 the FARD authors.
#
# The FARD architecture is licensed CC BY-NC 4.0 (non-commercial), NOT MIT like
# the rest of this repository. See LICENSE-FARD.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from monai.networks.blocks import SABlock

class SimpleDeformableAttention(nn.Module):
    """
    Very simplified deformable attention that's robust to various input dimensions
    Focuses on the core concept of deformed sampling without complex operations
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction

        # Projection layers
        self.query_proj = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Simple offset prediction - just one set of offsets for simplicity
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),  # 2 channels: x and y offsets
            nn.Tanh()  # Bound offsets to [-1, 1]
        )

        # Learnable weight for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Generate queries, keys, values
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Predict offsets - will be in range [-1, 1] due to tanh
        offsets = self.offset(x) * 0.2  # Scale to limit deformation

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Add predicted offsets to grid
        offset_x = offsets[:, 0:1].permute(0, 2, 3, 1)  # [B, H, W, 1]
        offset_y = offsets[:, 1:2].permute(0, 2, 3, 1)  # [B, H, W, 1]
        grid_with_offset = torch.cat([
            grid[:, :, :, 0:1] + offset_x,
            grid[:, :, :, 1:2] + offset_y
        ], dim=3)

        # Sample deformed features
        sampled_k = F.grid_sample(
            k, grid_with_offset, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        sampled_v = F.grid_sample(
            v, grid_with_offset, mode='bilinear', padding_mode='zeros', align_corners=True
        )

        # Compute attention weights
        attn = q * sampled_k
        attn = F.softmax(attn, dim=1)

        # Apply attention weights
        out = attn * sampled_v

        # Residual connection
        return x + self.gamma * out


class MONAIAttention(nn.Module):
    """
    MONAI Self-Attention Block for ablation comparison
    """

    def __init__(self, in_channels, num_heads=8, pool_factor=4, dropout_rate=0.0):
        super().__init__()
        self.pool_factor = pool_factor

        self.pool_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=pool_factor,
            stride=pool_factor,
            groups=in_channels
        )

        self.sa_block = SABlock(
            hidden_size=in_channels,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        self.norm = nn.LayerNorm(in_channels)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        pf = self.pool_factor

        # Padding
        pad_h = (pf - H % pf) % pf
        pad_w = (pf - W % pf) % pf
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        # Pool
        x_pooled = self.pool_conv(x)

        # Self-attention
        x_flat = x_pooled.flatten(2).transpose(1, 2)
        attn_out = self.sa_block(x_flat)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H // pf, W // pf)

        # Block-wise application
        x_blocks = x.view(B, C, H // pf, pf, W // pf, pf)
        att_blocks = attn_out.unsqueeze(3).unsqueeze(5)
        x_attended = x_blocks * att_blocks
        out = x_attended.view(B, C, H, W)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H - pad_h, :W - pad_w]
            x = x[:, :, :H - pad_h, :W - pad_w]

        return x + self.gamma * out


class EfficientAxialAttention(nn.Module):
    """
    Fixed axial attention that works with various input dimensions
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction

        # Height attention
        self.q_h = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.k_h = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.v_h = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Width attention
        self.q_w = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.k_w = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.v_w = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Gamma parameters
        self.gamma_h = nn.Parameter(torch.zeros(1))
        self.gamma_w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Height attention - operate on each column independently
        q_h = self.q_h(x)  # [B, C_r, H, W]
        k_h = self.k_h(x)  # [B, C_r, H, W]
        v_h = self.v_h(x)  # [B, C, H, W]

        # Reshape for height-wise operations: [B*W, H, C_r]
        q_h = q_h.permute(0, 3, 2, 1).reshape(batch_size * W, H, self.reduced_channels)
        k_h = k_h.permute(0, 3, 2, 1).reshape(batch_size * W, H, self.reduced_channels)
        v_h = v_h.permute(0, 3, 2, 1).reshape(batch_size * W, H, C)

        # Compute attention along height dimension
        attn_h = torch.bmm(q_h, k_h.transpose(1, 2)) / math.sqrt(self.reduced_channels)  # [B*W, H, H]
        attn_h = F.softmax(attn_h, dim=2)
        out_h = torch.bmm(attn_h, v_h)  # [B*W, H, C]

        # Reshape back: [B, C, H, W]
        out_h = out_h.reshape(batch_size, W, H, C).permute(0, 3, 2, 1)

        # Width attention - operate on each row independently
        q_w = self.q_w(x)  # [B, C_r, H, W]
        k_w = self.k_w(x)  # [B, C_r, H, W]
        v_w = self.v_w(x)  # [B, C, H, W]

        # Reshape for width-wise operations: [B*H, W, C_r]
        q_w = q_w.permute(0, 2, 3, 1).reshape(batch_size * H, W, self.reduced_channels)
        k_w = k_w.permute(0, 2, 3, 1).reshape(batch_size * H, W, self.reduced_channels)
        v_w = v_w.permute(0, 2, 3, 1).reshape(batch_size * H, W, C)

        # Compute attention along width dimension
        attn_w = torch.bmm(q_w, k_w.transpose(1, 2)) / math.sqrt(self.reduced_channels)  # [B*H, W, W]
        attn_w = F.softmax(attn_w, dim=2)
        out_w = torch.bmm(attn_w, v_w)  # [B*H, W, C]

        # Reshape back: [B, C, H, W]
        out_w = out_w.reshape(batch_size, H, W, C).permute(0, 3, 1, 2)

        # Combine with weighted residual connections
        return x + self.gamma_h * out_h + self.gamma_w * out_w


class SimpleGatedAttention(nn.Module):
    """
    Simplified gated attention that works with various input dimensions
    """

    def __init__(self, in_channels, reduction=8, pool_factor=4):
        super().__init__()
        self.in_channels = in_channels
        self.pool_factor = pool_factor  # Factor to reduce spatial dimensions
        self.reduced_channels = in_channels // reduction

        # Standard attention components
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Simple gate generator
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.pool_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=pool_factor,
            stride=pool_factor,
            groups=in_channels  # Depthwise to keep parameters low
        )
        self.noise_estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output between 0-1 (noise confidence)
        )

        # Gamma parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, C, H, W = x.shape
        pf = self.pool_factor
        # Ensure dimensions are divisible by pool_factor
        pad_h = (pf - H % pf) % pf
        pad_w = (pf - W % pf) % pf

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        # Compute attention at reduced resolution
        # x_pooled = F.adaptive_avg_pool2d(x, (H // 4, W // 4))
        x_pooled = self.pool_conv(x)
        #x_pooled = self.noise_aware_pool(x)  # Use noise-aware pooling
        # Standard attention computation
        query = self.query(x_pooled).flatten(2).permute(0, 2, 1)
        key = self.key(x_pooled).flatten(2)
        value = self.value(x_pooled).flatten(2).permute(0, 2, 1)

        attention = F.softmax(torch.bmm(query, key), dim=-1)
        attended = torch.bmm(attention, value)  # B, (H//4)*(W//4), C

        # Reshape to spatial attention map
        attended = attended.permute(0, 2, 1).view(B, C, H // pf, W // pf)

        # Reshape input into blocks and apply attention
        x_blocks = x.view(B, C, H // pf, pf, W // pf, pf)
        att_blocks = attended.unsqueeze(3).unsqueeze(5)  # Add block dimensions

        # Apply attention to each block
        x_attended = x_blocks * att_blocks

        # Reshape back to original size
        out = x_attended.view(B, C, H, W)

        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H - pad_h, :W - pad_w]
            x = x[:, :, :H - pad_h, :W - pad_w]  # Also remove padding from x for gate

        # Apply gate and residual
        gate = self.gate(x)
        return x + self.gamma * gate * out



class SimpleFlashAttention(nn.Module):
    """
    Simplified version of flash attention that works with various input dimensions
    Uses pooling and chunking for efficiency
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction

        # QKV projections
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Gamma parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Aggressive pooling for efficiency
        pool_size = max(1, min(H, W) // 16)
        x_pooled = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))

        # QKV projections on pooled features
        q = self.query(x_pooled)
        k = self.key(x_pooled)
        v = self.value(x_pooled)

        # Reshape for attention
        q_flat = q.flatten(2).transpose(1, 2)  # B, pool_size², C//r
        k_flat = k.flatten(2)  # B, C//r, pool_size²
        v_flat = v.flatten(2).transpose(1, 2)  # B, pool_size², C

        # Compute attention efficiently
        scale = 1.0 / math.sqrt(self.reduced_channels)
        attn = torch.bmm(q_flat, k_flat) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out_flat = torch.bmm(attn, v_flat)  # B, pool_size², C

        # Reshape and upsample
        out = out_flat.transpose(1, 2).reshape(batch_size, C, pool_size, pool_size)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # Residual connection
        return x + self.gamma * out


class EnhancedAttention_block(nn.Module):
    """Enhanced attention module with multiple high-performance options"""

    def __init__(
            self,
            in_channels,
            reduction=8,
            attention_type='standard',  # 'standard', 'flash', 'deformable', 'axial', 'gated'
            window_size=7,
            gate_type= 'gated',
            pool_factor=8,
            num_heads=8,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.in_channels = in_channels
        self.reduction = reduction
        self.reduced_channels = in_channels // reduction
        self.gate_type = gate_type
        self.window_size = window_size
        self.pool_factor = pool_factor

        # Common parameter for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))

        if attention_type == 'standard':
            # Standard attention layers (from original implementation)
            self.query = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=1)
            self.key = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=1)
            self.value = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        elif attention_type == 'flash':
            # Flash Attention for efficient computation
            self.flash_attn = SimpleFlashAttention(self.in_channels, reduction=self.reduction)

        elif attention_type == 'deformable':
            # Deformable Attention for adaptive receptive fields
            self.deform_attn = SimpleDeformableAttention(self.in_channels, reduction=self.reduction)

        elif attention_type == 'vit':
            from monai.networks.blocks import SABlock
            self.attn_module = MONAIAttention(in_channels, num_heads=num_heads, pool_factor=self.pool_factor)

        elif attention_type == 'axial':
            # Axial Attention for efficient 2D attention
            self.axial_attn = EfficientAxialAttention(self.in_channels, reduction=self.reduction)

        elif attention_type == 'gated':
            # Gated Attention for selective feature emphasis
            self.gated_attn = SimpleGatedAttention(self.in_channels, reduction=self.reduction,pool_factor=self.pool_factor)

        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

    def _standard_attention(self, x):
        batch_size, C, H, W = x.size()

        # Apply pooling to reduce spatial dimensions
        x_pooled = F.adaptive_avg_pool2d(x, output_size=(H // 4, W // 4))
        h_pooled, w_pooled = H // 4, W // 4

        query = self.query(x_pooled).flatten(2).permute(0, 2, 1)  # B, (H//4)*(W//4), C//r
        key = self.key(x_pooled).flatten(2)  # B, C//r, (H//4)*(W//4)
        value = self.value(x_pooled).flatten(2).permute(0, 2, 1)  # B, (H//4)*(W//4), C

        # Compute attention
        attention = F.softmax(torch.bmm(query, key)/ math.sqrt(self.reduced_channels),dim=-1)  # B, (H//4)*(W//4), (H//4)*(W//4)
        out = torch.bmm(attention, value)  # B, (H//4)*(W//4), C

        # Reshape back
        out = out.permute(0, 2, 1).view(batch_size, C, h_pooled, w_pooled)

        # Upsample to original size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out

    def forward(self, x):
        if self.attention_type == 'standard':
            out = self._standard_attention(x)
            return self.gamma * out + x
        elif self.attention_type == 'flash':
            return self.flash_attn(x)
        elif self.attention_type == 'deformable':
            return self.deform_attn(x)
        elif self.attention_type == 'axial':
            return self.axial_attn(x)
        elif self.attention_type == 'gated':
            return self.gated_attn(x)
        elif self.attention_type == 'vit':
            return self.attn_module(x)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")