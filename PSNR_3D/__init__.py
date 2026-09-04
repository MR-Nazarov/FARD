import torch
from typing import Optional, Union, Tuple, Any
from torch import Tensor
from torch.nn import Module


class PSNR3D(Module):
    """Peak Signal-to-Noise Ratio (PSNR) metric for 3D data.

    Calculates PSNR between two 3D tensors. Typically used for comparing original and
    reconstructed 3D volumes, such as in medical imaging or 3D reconstruction tasks.

    Args:
        data_range: Range of the data. If None, it's computed from the target tensor
        base: Base of logarithm to use (default: 10.0)
        reduction: Specifies the reduction to apply to the output
        dim: Dimensions along which PSNR will be computed
    """

    def __init__(
            self,
            data_range: Optional[Union[float, Tuple[float, float]]] = None,
            base: float = 10.0,
            reduction: str = "elementwise_mean",
            dim: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.base = base
        self.reduction = reduction
        self.dim = dim if dim is not None else (1, 2, 3)  # Default to all spatial dimensions

    def get_device(self, tensor: Tensor) -> torch.device:
        """Get the device of the tensor.

        Args:
            tensor: Input tensor

        Returns:
            torch.device: Device of the tensor
        """
        if tensor.is_cuda:
            return tensor.device
        return torch.device('cpu')

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Compute PSNR between prediction and target tensors.

        Args:
            preds: Predicted tensor (B, D, H, W) or (D, H, W)
            target: Target tensor (B, D, H, W) or (D, H, W)

        Returns:
            PSNR value as a tensor
        """
        if not torch.is_tensor(preds) or not torch.is_tensor(target):
            raise TypeError("Predictions and target must be tensors")

        if preds.shape != target.shape:
            raise ValueError(f"Predictions and target must have the same shape. Got {preds.shape} and {target.shape}")

        # Get device from input tensors
        device = self.get_device(preds)

        # Determine data range if not provided
        if self.data_range is None:
            data_range = target.max() - target.min()
        else:
            data_range = torch.tensor(self.data_range, device=device)

        # Calculate MSE along specified dimensions
        mse = torch.mean((preds - target) ** 2, dim=self.dim)

        # Calculate PSNR
        psnr = 10 * torch.log10(data_range ** 2 / mse)

        # Apply reduction if specified
        if self.reduction == "elementwise_mean":
            psnr = torch.mean(psnr)
        elif self.reduction == "sum":
            psnr = torch.sum(psnr)

        return psnr