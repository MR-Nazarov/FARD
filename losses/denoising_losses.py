import torch.nn.functional as F
import pytorch_ssim
from confs_functions import get_func_args
import torch
import torch.nn as nn

def get_func_args(kwargs, func):
    """Extract arguments that match the function parameters."""
    import inspect
    func_args = inspect.getfullargspec(func).args
    return {k: v for k, v in kwargs.items() if k in func_args}

class L1Loss(nn.Module):
    def __init__(self, **kwargs):
        super(L1Loss, self).__init__()
        self.lossFunc = F.l1_loss
        self.funcArgs = get_func_args(kwargs, self.lossFunc)

    def forward(self, output, target):
        return self.lossFunc(output, target, **self.funcArgs)

#def L1Loss(output, target, **kwargs):
#    lossFunc = F.l1_loss
#    funcArgs = get_func_args(kwargs, lossFunc)
#    loss = lossFunc(output, target, **funcArgs)

#    return loss
class MultiViewL1Loss(nn.Module):
    """Assumes output and target have shape (B, D, H, W)"""
    def __init__(self,loss_func= F.l1_loss, **kwargs):
        super(MultiViewL1Loss, self).__init__()
        self.lossFunc = loss_func
        self.funcArgs = get_func_args(kwargs, self.lossFunc)

    def forward(self, output, target):
        assert output.shape[1] == 3, f"Expected 3 views, got {output.shape[1]}"
        assert output.shape == target.shape, "Output and target shapes must match"
        loss1 = self.lossFunc(output[:, 0], target[:, 0], **self.funcArgs)
        loss2 = self.lossFunc(output[:, 1], target[:, 1], **self.funcArgs)
        loss3 = self.lossFunc(output[:, 2], target[:, 2], **self.funcArgs)

        return (loss1 + loss2 + loss3) / 3
class MRILoss(nn.Module):
    def __init__(self, alpha_loss=1, beta_loss=0.02, dc_cut_radius=2,
                 full_kspace=False, **kwargs):
        super(MRILoss, self).__init__()
        self.alpha = alpha_loss
        self.beta = beta_loss
        self.dc_cut_radius = dc_cut_radius  # Renamed for clarity
        self.full_kspace = full_kspace
        self.kwargs = kwargs

    def k_space_no_dc_loss(self, pred, target):
        pred_kspace = torch.fft.fft2(pred, norm='ortho')
        target_kspace = torch.fft.fft2(target, norm='ortho')

        h, w = pred_kspace.shape[-2:]

        if not self.full_kspace and self.dc_cut_radius > 0:
            # Create mask
            mask = torch.ones_like(pred_kspace, dtype=torch.float32)

            # Find center
            h_center, w_center = h // 2, w // 2

            if self.dc_cut_radius == 1:
                # Remove only center point
                mask[..., h_center, w_center] = 0
            else:
                # Remove square region around center
                r = self.dc_cut_radius
                h_start = max(0, h_center - r)
                h_end = min(h, h_center + r + 1)
                w_start = max(0, w_center - r)
                w_end = min(w, w_center + r + 1)
                mask[..., h_start:h_end, w_start:w_end] = 0

            pred_kspace = pred_kspace * mask
            target_kspace = target_kspace * mask

        # If dc_cut_radius is 0, use full k-space (no mask applied)
        return F.l1_loss(torch.abs(pred_kspace), torch.abs(target_kspace))

    def forward(self, pred, target, **kwargs):
        img_loss = F.l1_loss(pred, target, **self.kwargs)
        kspace_loss = self.k_space_no_dc_loss(pred, target)

        return self.alpha * img_loss + self.beta * kspace_loss

class MRILoss2(nn.Module):
    def __init__(self, alpha_loss=1, beta_loss=0.02,h_cut=4,w_cut=4,full_kspace= False, **kwargs):
        super(MRILoss2, self).__init__()
        self.alpha = alpha_loss
        self.beta = beta_loss
        self.kwargs = kwargs
        self.h_cut = h_cut
        self.w_cut = w_cut
        self.full_kspace = full_kspace

    def k_space_no_dc_loss(self, pred, target):
        pred_kspace = torch.fft.fft2(pred)
        target_kspace = torch.fft.fft2(target)
        h, w = pred_kspace.shape[-2:]
        # Create frequency weighting mask (protect low frequencies)
        mask = self.create_high_freq_mask(h, w, pred_kspace.device)
        if not self.full_kspace:
            #h, w = pred_kspace.shape[-2:]
            dc_mask = torch.ones_like(mask)
            dc_mask[..., h // 2 - self.h_cut:h // 2 + self.h_cut,
            w // 2 - self.w_cut:w // 2 + self.w_cut] = 0
            mask = mask * dc_mask
            #pred_kspace[..., h // self.h_cut, w // self.w_cut] = 0
            #target_kspace[..., h // self.h_cut, w // self.w_cut] = 0
            pred_kspace_masked = pred_kspace * mask
            target_kspace_masked = target_kspace * mask

        return F.l1_loss(torch.abs(pred_kspace_masked), torch.abs(target_kspace_masked))

    def create_high_freq_mask(self, h, w, device):
        """Weight high frequencies more, protect low frequencies"""
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.to(device), x.to(device)

        # Radial distance from center
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_norm = r / r.max()

        # Inverse Gaussian - emphasize high frequencies, protect low
        # This is KEY: we want to protect contrast (low freq)
        mask = 1 - torch.exp(-10 * r_norm ** 2)  # Near 0 at center, 1 at edges
        mask = mask * 0.8 + 0.2  # Ensure minimum weight of 0.2

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target, **kwargs):
        img_loss = F.l1_loss(pred, target, **self.kwargs)
        kspace_loss = self.k_space_no_dc_loss(pred, target)

        return self.alpha * img_loss + self.beta * kspace_loss

class FSMNetLoss(nn.Module):
    def __init__(self, fft_weight=0.01, **kwargs):
        super(FSMNetLoss, self).__init__()
        self.fft_weight = fft_weight
        self.eps = 1e-8  # For numerical stability

    def AmplitudeLoss(self, output, target, **kwargs):
        """Loss between amplitude components in Fourier domain."""
        # Convert to frequency domain with ortho normalization for stability
        x = torch.fft.rfft2(output, norm='ortho')
        y = torch.fft.rfft2(target, norm='ortho')

        # Get amplitude components
        x_mag = torch.abs(x)
        y_mag = torch.abs(y)

        # Calculate L1 loss between amplitudes
        loss = F.l1_loss(x_mag, y_mag)

        # Check for NaN and return 0 if found
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=output.device, dtype=output.dtype)
        return loss

    def PhaseLoss(self, output, target, **kwargs):
        """Loss between phase components in Fourier domain."""
        # Convert to frequency domain with ortho normalization for stability
        x = torch.fft.rfft2(output, norm='ortho')
        y = torch.fft.rfft2(target, norm='ortho')

        # Only compute phase where magnitude is significant (avoid instability)
        x_mag = torch.abs(x)
        y_mag = torch.abs(y)

        # Create mask for significant magnitudes
        mask = (x_mag > self.eps) & (y_mag > self.eps)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, dtype=output.dtype)

        # Get phase components only where mask is valid
        x_phase = torch.angle(x)
        y_phase = torch.angle(y)

        # Calculate L1 loss between phases (only for significant components)
        loss = F.l1_loss(x_phase[mask], y_phase[mask])

        # Check for NaN and return 0 if found
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=output.device, dtype=output.dtype)
        return loss

    def forward(self, output, target, **kwargs):
        # Handle both list and tensor inputs
        if isinstance(output, list):
            spatial_out = output[0]  # First element for spatial output
            freq_out = output[1]     # Second element for frequency output
        else:
            # If output is a tensor with 2 channels, split it
            spatial_out = output[:, 0:1, ...]
            freq_out = output[:, 1:2, ...]

        # Check for NaN/Inf in inputs and clamp if needed
        if torch.isnan(spatial_out).any() or torch.isinf(spatial_out).any():
            print(f"Warning: NaN/Inf in spatial_out, clamping values")
            spatial_out = torch.nan_to_num(spatial_out, nan=0.0, posinf=1.0, neginf=0.0)

        if torch.isnan(freq_out).any() or torch.isinf(freq_out).any():
            print(f"Warning: NaN/Inf in freq_out, clamping values")
            freq_out = torch.nan_to_num(freq_out, nan=0.0, posinf=1.0, neginf=0.0)

        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"Warning: NaN/Inf in target, clamping values")
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

        # Spatial loss
        spatial_loss = F.l1_loss(spatial_out, target)

        # Frequency branch loss
        freq_loss = F.l1_loss(freq_out, target)

        # Frequency domain losses only applied to frequency branch output
        amp_loss = self.AmplitudeLoss(freq_out, target, **kwargs)
        phase_loss = self.PhaseLoss(freq_out, target, **kwargs)

        # Combine losses
        total_loss = spatial_loss + freq_loss + self.fft_weight * (amp_loss + phase_loss)

        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf detected in FSMNetLoss. spatial_loss={spatial_loss.item()}, freq_loss={freq_loss.item()}, amp_loss={amp_loss.item()}, phase_loss={phase_loss.item()}")
            # Return a small valid loss to keep training going
            return torch.tensor(0.001, device=target.device, dtype=target.dtype, requires_grad=True)

        return total_loss
class charbonnier_loss(nn.Module):
    def __init__(self, eps=1e-12):
        super(charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        return torch.sqrt((output - target)**2 + self.eps).mean()




def normalized_L1Loss(output, target, **kwargs):
    # Calculate the L1 loss using the provided function and arguments
    lossFunc = F.l1_loss
    funcArgs = get_func_args(kwargs, lossFunc)
    loss = lossFunc(output, target, **funcArgs)

    # Normalize the L1 loss by the sum of absolute target values
    # Avoid division by zero by adding a small epsilon if the denominator is zero
    epsilon = 1e-4
    norm_factor = (1e-8)*torch.sum(torch.abs(target)) + epsilon
    normalized_loss = loss / norm_factor

    return normalized_loss

def L1Loss3d(output, target, **kwargs):
    lossFunc = F.l1_loss
    funcArgs = get_func_args(kwargs, lossFunc)
    loss1 = lossFunc(output[:, 0], target[:, 0], **funcArgs)
    loss2 = lossFunc(output[:, 1], target[:, 1], **funcArgs)

    return (loss1 + loss2) / 2

class SSIMLoss(nn.Module):
    def __init__(self, **kwargs):
        super(SSIMLoss, self).__init__()
        self.lossFunc = pytorch_ssim.SSIM(**kwargs)

    def __call__(self,output,target, **kwargs):
        ssim_value = self.lossFunc(output, target)
        return -ssim_value  # Return negative SSIM for loss


#def SSIMLoss(output, target, **kwargs):
#    lossFunc = pytorch_ssim.SSIM()
#    lossFunc = lossFunc(output, target)
#    return -lossFunc


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
        self.lossFunc = F.mse_loss
        self.funcArgs = get_func_args(kwargs, self.lossFunc)

    def forward(self, output, target):
        return self.lossFunc(output, target, reduction='mean', **self.funcArgs)
