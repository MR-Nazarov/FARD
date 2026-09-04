import torch
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
import matplotlib.pyplot as plt
from pathlib import Path
from munch import Munch


class Transform:
    def __call__(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[
        torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class CenterCrop(Transform):
    def __init__(self, size: Tuple[int, int]):
        """
        Center crop transform
        Args:
            size: (height, width) of the crop
        """
        self.size = size
        self.output = None
        self.enable_visualization = False  # Can be overridden

    def visualize_crop(self, original, cropped, crop_coords, slice_idx=None):
        """
        Visualize the center crop effect
        Args:
            original: Original tensor
            cropped: Cropped tensor
            crop_coords: (start_h, start_w, crop_h, crop_w)
            slice_idx: Which slice to visualize (defaults to middle slice)
        """
        import matplotlib.pyplot as plt
        from pathlib import Path

        # Select middle slice if not specified
        if slice_idx is None:
            slice_idx = original.shape[0] // 2

        # Convert tensors to numpy for visualization
        orig_slice = original[slice_idx].cpu().numpy()
        crop_slice = cropped[slice_idx].cpu().numpy()
        start_h, start_w, crop_h, crop_w = crop_coords

        # Create figure
        plt.figure(figsize=(15, 5))

        # Original image with crop region
        plt.subplot(131)
        plt.imshow(orig_slice, cmap='gray')
        plt.title(f'Original (slice {slice_idx})')
        # Draw crop region
        plt.gca().add_patch(plt.Rectangle((start_w, start_h),
                                        crop_w,
                                        crop_h,
                                        fill=False,
                                        color='red',
                                        linewidth=2))
        plt.colorbar()
        plt.axis('on')

        # Cropped region overlay
        plt.subplot(132)
        plt.imshow(orig_slice, cmap='gray', alpha=0.5)
        # Create a mask for the crop region
        mask = np.zeros_like(orig_slice)
        mask[start_h:start_h+crop_h, start_w:start_w+crop_w] = 1
        plt.imshow(mask, cmap='Reds', alpha=0.3)
        plt.title('Crop Region Overlay')
        plt.colorbar()
        plt.axis('on')

        # Cropped result
        plt.subplot(133)
        plt.imshow(crop_slice, cmap='gray')
        plt.title(f'Cropped Result ({crop_h}x{crop_w})')
        plt.colorbar()
        plt.axis('on')

        # Save visualization
        save_dir = Path('crop_visualizations')
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / f'center_crop_slice_{slice_idx}.png')
        plt.close()

    def crop_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply center crop to a single tensor"""
        if tensor.dim() == 3:
            _, h, w = tensor.shape
        else:
            raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")

        crop_h, crop_w = self.size

        if h < crop_h or w < crop_w:
            raise ValueError(f"Crop size {self.size} larger than input size ({h}, {w})")

        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2

        return tensor[:, start_h:start_h + crop_h, start_w:start_w + crop_w], (start_h, start_w, crop_h, crop_w)

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: Dictionary containing 'output' and 'target' tensors, optionally 'input'
        Returns:
            Dictionary with cropped tensors
        """
        target = data['target']
        self.output = data['output']

        print(f"Applying center crop {self.size} to tensors with shapes - Target: {target.shape}, Output: {self.output.shape}")

        # Apply crop to both tensors
        target_cropped, crop_coords = self.crop_tensor(target)
        self.output, _ = self.crop_tensor(self.output)

        print(f"Cropped shapes - Target: {target_cropped.shape}, Output: {self.output.shape}")

        # Prepare result dictionary
        result = {
            'output': self.output,
            'target': target_cropped
        }

        # Handle input if present
        if 'input' in data:
            input_tensor = data['input']
            print(f"  Cropping 'input' - Input shape: {input_tensor.shape}")
            input_cropped, _ = self.crop_tensor(input_tensor)
            result['input'] = input_cropped
            print(f"  Cropped 'input' - Output shape: {input_cropped.shape}")

        # Visualize the cropping if enabled
        if self.enable_visualization:
            try:
                self.visualize_crop(target, target_cropped, crop_coords)
            except Exception as e:
                print(f"Warning: Visualization failed: {str(e)}")

        return result


class TransposeTransform(Transform):
    def __init__(self,
                 dims: Union[List[int], Tuple[int, ...]],
                 keys: Optional[Union[List[str], Tuple[str, ...]]] = None):
        """
        Transform to transpose dimensions
        Args:
            dims: Tuple/List of dimensions to transpose to, e.g., (2, 0, 1) or (0, 2, 1)
            keys: Tuple/List of keys to apply transform to. If None, defaults to ['output', 'target']
        """
        self.dims = tuple(dims)
        self.keys = tuple(keys) if keys is not None else ('output', 'target')
        self.output = None
        self.enable_visualization = False  # Can be overridden

        if len(self.dims) != 3 or not all(0 <= d <= 2 for d in self.dims):
            raise ValueError(f"dims must be a permutation of (0,1,2), got {dims}")

        # Check if it's a valid permutation
        if sorted(self.dims) != [0, 1, 2]:
            raise ValueError(f"dims must be a permutation of (0,1,2), got {dims}")

        print(f"TransposeTransform initialized: dims={self.dims}, keys={self.keys}")

    def visualize_transpose(self, original, transposed, key_name=""):
        """Visualize the transpose effect"""
        try:
            # Create figure
            plt.figure(figsize=(15, 5))

            # Get middle slices for both orientations
            orig_middle = original.shape[0] // 2
            trans_middle = transposed.shape[0] // 2

            # Show middle slice of original volume
            plt.subplot(131)
            orig_slice = original[orig_middle].cpu().numpy()
            plt.imshow(orig_slice, cmap='gray')
            plt.title(f'Original ({key_name})\nShape: {original.shape}\nShowing slice {orig_middle}')
            plt.colorbar()
            plt.axis('on')

            # Show middle slice of transposed volume
            plt.subplot(132)
            trans_slice = transposed[trans_middle].cpu().numpy()
            plt.imshow(trans_slice, cmap='gray')
            plt.title(f'Transposed {self.dims} ({key_name})\nShape: {transposed.shape}\nShowing slice {trans_middle}')
            plt.colorbar()
            plt.axis('on')

            # Show histograms of both slices
            plt.subplot(133)
            plt.hist(orig_slice.ravel(), bins=50, alpha=0.5, label='Original', density=True)
            plt.hist(trans_slice.ravel(), bins=50, alpha=0.5, label='Transposed', density=True)
            plt.title(f'Intensity Distributions ({key_name})')
            plt.legend()
            plt.xlabel('Intensity')
            plt.ylabel('Density')

            # Save visualization
            save_dir = Path('transpose_visualizations')
            save_dir.mkdir(exist_ok=True)
            filename = f'transpose_{self.dims}_{key_name}.png' if key_name else f'transpose_{self.dims}.png'
            plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Warning: Visualization failed for {key_name}: {str(e)}")

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: Dictionary containing tensors
        Returns:
            Dictionary with transposed tensors for specified keys
        """
        result = data.copy()  # Start with all original data

        print(f"Applying transpose {self.dims} to keys {self.keys}")

        # Apply transpose only to specified keys
        for key in self.keys:
            if key in data:
                original_tensor = data[key]
                print(f"  Transposing '{key}' - Input shape: {original_tensor.shape}")

                # Apply transpose
                transposed_tensor = original_tensor.permute(*self.dims)
                result[key] = transposed_tensor

                print(f"  Transposed '{key}' - Output shape: {transposed_tensor.shape}")

                # Visualize the transpose if enabled
                if self.enable_visualization:
                    try:
                        self.visualize_transpose(original_tensor, transposed_tensor, key)
                    except Exception as e:
                        print(f"Warning: Visualization failed for {key}: {str(e)}")

            else:
                print(f"Warning: Key '{key}' not found in data dictionary. Available keys: {list(data.keys())}")

        # Store reference to output for backward compatibility
        if 'output' in result:
            self.output = result['output']

        return result


class FlipTransform(Transform):
    def __init__(self,
                 dims: Union[List[int], Tuple[int, ...]],
                 keys: Optional[Union[List[str], Tuple[str, ...]]] = None):
        """
        Transform to flip dimensions
        Args:
            dims: Tuple/List of dimensions to flip, e.g., (0,1), (2,), etc.
            keys: Tuple/List of keys to apply transform to. If None, defaults to ['output', 'target']
        """
        self.dims = tuple(dims)
        self.keys = tuple(keys) if keys is not None else ('output', 'target')
        self.output = None
        self.enable_visualization = False  # Can be overridden

        if not all(0 <= d <= 2 for d in self.dims):
            raise ValueError(f"dims must be in range [0,2], got {dims}")

        print(f"FlipTransform initialized: dims={self.dims}, keys={self.keys}")

    def visualize_flip(self, original, flipped, key_name=""):
        """Visualize the flip effect"""
        try:
            # Create figure
            plt.figure(figsize=(15, 5))

            # Get middle slice
            middle_slice = original.shape[0] // 2

            # Show original
            plt.subplot(131)
            orig_slice = original[middle_slice].cpu().numpy()
            plt.imshow(orig_slice, cmap='gray')
            plt.title(f'Original ({key_name})\nShape: {original.shape}\nSlice {middle_slice}')
            plt.colorbar()
            plt.axis('on')

            # Show flipped
            plt.subplot(132)
            flip_slice = flipped[middle_slice].cpu().numpy()
            plt.imshow(flip_slice, cmap='gray')
            plt.title(f'Flipped dims {self.dims} ({key_name})\nShape: {flipped.shape}\nSlice {middle_slice}')
            plt.colorbar()
            plt.axis('on')

            # Show difference
            plt.subplot(133)
            diff = np.abs(orig_slice - flip_slice)
            plt.imshow(diff, cmap='viridis')
            plt.title(f'Difference Map ({key_name})')
            plt.colorbar()
            plt.axis('on')

            # Save visualization
            save_dir = Path('flip_visualizations')
            save_dir.mkdir(exist_ok=True)
            filename = f'flip_dims_{self.dims}_{key_name}.png' if key_name else f'flip_dims_{self.dims}.png'
            plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Warning: Visualization failed for {key_name}: {str(e)}")

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: Dictionary containing tensors
        Returns:
            Dictionary with flipped tensors for specified keys
        """
        result = data.copy()  # Start with all original data

        print(f"Applying flip on dims {self.dims} to keys {self.keys}")

        # Apply flip only to specified keys
        for key in self.keys:
            if key in data:
                original_tensor = data[key]
                print(f"  Flipping '{key}' - Input shape: {original_tensor.shape}")

                # Apply flip
                flipped_tensor = original_tensor.flip(self.dims)
                result[key] = flipped_tensor

                print(f"  Flipped '{key}' - Output shape: {flipped_tensor.shape}")

                # Visualize the flip if enabled
                if self.enable_visualization:
                    try:
                        self.visualize_flip(original_tensor, flipped_tensor, key)
                    except Exception as e:
                        print(f"Warning: Visualization failed for {key}: {str(e)}")

            else:
                print(f"Warning: Key '{key}' not found in data dictionary. Available keys: {list(data.keys())}")

        # Store reference to output for backward compatibility
        if 'output' in result:
            self.output = result['output']

        return result
class BrainBoundingBoxTransform(Transform):
    def __init__(self, threshold: float = 0.01, padding: int = 1):
        self.threshold = threshold
        self.padding = padding
        self.output = None
        self.enable_visualization = False  # Can be overridden

    def visualize_bbox(self, original, cropped, bbox, slice_idx=None):
        """
        Visualize the bounding box and cropping effect
        Args:
            original: Original tensor
            cropped: Cropped tensor
            bbox: (y_min, y_max, x_min, x_max)
            slice_idx: Which slice to visualize (defaults to middle slice)
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np

        # Select middle slice if not specified
        if slice_idx is None:
            slice_idx = original.shape[0] // 2

        # Convert tensors to numpy for visualization
        orig_slice = original[slice_idx].cpu().numpy()
        crop_slice = cropped[slice_idx].cpu().numpy()
        y_min, y_max, x_min, x_max = bbox

        # Create figure
        plt.figure(figsize=(15, 5))

        # Original image with bounding box
        plt.subplot(131)
        plt.imshow(orig_slice, cmap='gray')
        plt.title(f'Original (slice {slice_idx})')
        # Draw bounding box
        plt.gca().add_patch(plt.Rectangle((x_min, y_min),
                                          x_max - x_min,
                                          y_max - y_min,
                                          fill=False,
                                          color='red',
                                          linewidth=2))
        plt.colorbar()
        plt.axis('on')

        # Mask visualization
        plt.subplot(132)
        mask = (orig_slice > self.threshold).astype(np.float32)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Threshold Mask ({self.threshold})')
        plt.gca().add_patch(plt.Rectangle((x_min, y_min),
                                          x_max - x_min,
                                          y_max - y_min,
                                          fill=False,
                                          color='red',
                                          linewidth=2))
        plt.colorbar()
        plt.axis('on')

        # Cropped result
        plt.subplot(133)
        plt.imshow(crop_slice, cmap='gray')
        plt.title('Cropped Result')
        plt.colorbar()
        plt.axis('on')

        # Save visualization
        save_dir = Path('bbox_visualizations')
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / f'bbox_slice_{slice_idx}.png')
        plt.close()

    def find_bounding_box(self, tensor: torch.Tensor):
        """Find the bounding box coordinates of the brain region"""
        mask = tensor > self.threshold
        print(f"Mask shape during bounding box: {mask.shape}")

        indices = torch.where(mask)

        if len(indices[1]) == 0:
            raise ValueError("No brain region found with current threshold")

        y_min = max(indices[1].min().item() - self.padding, 0)
        y_max = min(indices[1].max().item() + self.padding, tensor.shape[1])
        x_min = max(indices[2].min().item() - self.padding, 0)
        x_max = min(indices[2].max().item() + self.padding, tensor.shape[2])

        print(f"Found bounding box: Y:[{y_min}-{y_max}], X:[{x_min}-{x_max}]")
        return y_min, y_max, x_min, x_max

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target = data['target']
        self.output = data['output']

        print(f"Processing tensors with shapes - Target: {target.shape}, Output: {self.output.shape}")

        # Find bounding box from target
        y_min, y_max, x_min, x_max = self.find_bounding_box(target)
        bbox = (y_min, y_max, x_min, x_max)

        # Crop both tensors
        output_cropped = self.output[:, y_min:y_max, x_min:x_max]
        target_cropped = target[:, y_min:y_max, x_min:x_max]

        print(f"Cropped shapes - Target: {target_cropped.shape}, Output: {output_cropped.shape}")

        # Visualize middle slice if enabled
        if self.enable_visualization:
            self.visualize_bbox(target, target_cropped, bbox)

        self.output = output_cropped

        return {
            'output': self.output,
            'target': target_cropped
        }
class ClipTransform(Transform):
    def __init__(self,
                 min_val: float = 0.0,
                 max_val: float = 1.0,
                 keys: Optional[Union[List[str], Tuple[str, ...]]] = None):
        """
        Clip tensor values to a specified range
        Args:
            min_val: Minimum value to clip to
            max_val: Maximum value to clip to
            keys: Keys to apply clipping to. If None, applies to all tensors
        """
        self.min_val = min_val
        self.max_val = max_val
        self.keys = tuple(keys) if keys is not None else None

        print(f"ClipTransform initialized: clipping to [{min_val}, {max_val}], keys={keys}")

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip tensor values"""
        result = data.copy()

        # Determine which keys to process
        keys_to_process = self.keys if self.keys is not None else data.keys()

        for key in keys_to_process:
            if key in data:
                tensor = data[key]
                clipped = torch.clamp(tensor, min=self.min_val, max=self.max_val)

                # Report clipping stats
                num_clipped_low = (tensor < self.min_val).sum().item()
                num_clipped_high = (tensor > self.max_val).sum().item()

                if num_clipped_low > 0 or num_clipped_high > 0:
                    print(f"  Clipped '{key}': {num_clipped_low} values below {self.min_val}, {num_clipped_high} values above {self.max_val}")

                result[key] = clipped

        return result


class CropToMatchTransform(Transform):
    def __init__(self,
                 reference_key: str = 'target',
                 target_key: str = 'output',
                 mode: str = 'center'):
        """
        Crop target tensor to match reference tensor dimensions
        Args:
            reference_key: Key of the reference tensor (usually 'target')
            target_key: Key of the tensor to crop (usually 'output')
            mode: Cropping mode ('center' for center crop)
        """
        self.reference_key = reference_key
        self.target_key = target_key
        self.mode = mode
        self.enable_visualization = False

        print(f"CropToMatchTransform initialized: crop '{target_key}' to match '{reference_key}', mode={mode}")

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Crop target to match reference dimensions"""
        result = data.copy()

        if self.reference_key not in data:
            print(f"Warning: Reference key '{self.reference_key}' not found. Available: {list(data.keys())}")
            return result

        if self.target_key not in data:
            print(f"Warning: Target key '{self.target_key}' not found. Available: {list(data.keys())}")
            return result

        reference = data[self.reference_key]
        target = data[self.target_key]

        print(f"\nCropping '{self.target_key}' to match '{self.reference_key}'")
        print(f"  Reference shape: {tuple(reference.shape)}")
        print(f"  Target shape:    {tuple(target.shape)}")

        # Calculate crop needed for each dimension
        if self.mode == 'center':
            # Center crop
            crop_slices = []
            for ref_size, tgt_size in zip(reference.shape, target.shape):
                if tgt_size < ref_size:
                    raise ValueError(f"Target dimension ({tgt_size}) is smaller than reference ({ref_size}). Cannot crop.")

                # Calculate center crop
                diff = tgt_size - ref_size
                start = diff // 2
                end = start + ref_size
                crop_slices.append(slice(start, end))

            # Apply crop
            cropped = target[crop_slices[0], crop_slices[1], crop_slices[2]]

            result[self.target_key] = cropped
            print(f"  Cropped shape:   {tuple(cropped.shape)}")

        return result


class PadToMatchTransform(Transform):
    def __init__(self,
                 reference_key: str = 'output',
                 target_key: str = 'target',
                 mode: str = 'constant',
                 value: float = 0.0):
        """
        Pad target tensor to match reference tensor dimensions
        Args:
            reference_key: Key of the reference tensor (usually 'output')
            target_key: Key of the tensor to pad (usually 'target')
            mode: Padding mode ('constant', 'reflect', 'replicate', 'circular')
            value: Fill value for constant padding
        """
        self.reference_key = reference_key
        self.target_key = target_key
        self.mode = mode
        self.value = value
        self.enable_visualization = False

        print(f"PadToMatchTransform initialized: pad '{target_key}' to match '{reference_key}', mode={mode}")

    def visualize_padding(self, original, padded, pad_amounts, key_name=""):
        """Visualize the padding effect"""
        try:
            # Create figure
            plt.figure(figsize=(15, 5))

            # Get middle slices
            orig_middle = original.shape[0] // 2
            padded_middle = padded.shape[0] // 2

            # Show original
            plt.subplot(131)
            orig_slice = original[orig_middle].cpu().numpy()
            plt.imshow(orig_slice, cmap='gray')
            plt.title(f'Original {key_name}\nShape: {tuple(original.shape)}')
            plt.colorbar()
            plt.axis('on')

            # Show padded
            plt.subplot(132)
            padded_slice = padded[padded_middle].cpu().numpy()
            plt.imshow(padded_slice, cmap='gray')
            plt.title(f'Padded {key_name}\nShape: {tuple(padded.shape)}\nPad: {pad_amounts}')
            plt.colorbar()
            plt.axis('on')

            # Show overlay highlighting padded regions
            plt.subplot(133)
            plt.imshow(padded_slice, cmap='gray')
            # Create a mask for padded regions
            pad_d, pad_h, pad_w = pad_amounts
            mask = np.zeros_like(padded_slice)
            h, w = padded_slice.shape
            if pad_h[0] > 0: mask[:pad_h[0], :] = 1  # Top
            if pad_h[1] > 0: mask[h-pad_h[1]:, :] = 1  # Bottom
            if pad_w[0] > 0: mask[:, :pad_w[0]] = 1  # Left
            if pad_w[1] > 0: mask[:, w-pad_w[1]:] = 1  # Right
            plt.imshow(mask, cmap='Reds', alpha=0.3)
            plt.title(f'Padded Regions Overlay {key_name}')
            plt.colorbar()
            plt.axis('on')

            # Save
            save_dir = Path('pad_visualizations')
            save_dir.mkdir(exist_ok=True)
            filename = f'pad_to_match_{key_name}.png' if key_name else 'pad_to_match.png'
            plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Warning: Padding visualization failed for {key_name}: {str(e)}")

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pad target to match reference dimensions"""
        result = data.copy()

        if self.reference_key not in data:
            print(f"Warning: Reference key '{self.reference_key}' not found. Available: {list(data.keys())}")
            return result

        if self.target_key not in data:
            print(f"Warning: Target key '{self.target_key}' not found. Available: {list(data.keys())}")
            return result

        reference = data[self.reference_key]
        target = data[self.target_key]

        print(f"\nPadding '{self.target_key}' to match '{self.reference_key}'")
        print(f"  Reference shape: {tuple(reference.shape)}")
        print(f"  Target shape:    {tuple(target.shape)}")

        # Calculate padding needed for each dimension
        pad_amounts = []
        for ref_size, tgt_size in zip(reference.shape, target.shape):
            diff = ref_size - tgt_size
            if diff < 0:
                raise ValueError(f"Target dimension ({tgt_size}) is larger than reference ({ref_size}). Cannot pad.")

            # Split padding between before and after
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_amounts.append((pad_before, pad_after))

        # PyTorch pad format is (left, right, top, bottom, front, back) - reversed order!
        # For 3D tensor [D, H, W], pad is: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        pad_list = []
        for pad_before, pad_after in reversed(pad_amounts):
            pad_list.extend([pad_before, pad_after])

        print(f"  Padding amounts (D, H, W): {pad_amounts}")

        # Apply padding
        if self.mode == 'constant':
            padded = torch.nn.functional.pad(target, pad_list, mode='constant', value=self.value)
        else:
            padded = torch.nn.functional.pad(target, pad_list, mode=self.mode)

        result[self.target_key] = padded
        print(f"  Padded shape:    {tuple(padded.shape)}")

        # Visualize if enabled
        if self.enable_visualization:
            try:
                self.visualize_padding(target, padded, pad_amounts, self.target_key)
            except Exception as e:
                print(f"Warning: Visualization failed: {str(e)}")

        return result


class PadOrCropTransform(Transform):
    def __init__(self,
                 size: Tuple[int, int],
                 keys: Optional[List[str]] = None,
                 pad_value: float = 0.0):
        """
        Resize each key to exactly (H, W) by center-cropping if larger or
        zero-padding if smaller along each spatial dimension independently.

        Args:
            size: Target (height, width)
            keys: List of keys to apply to. Defaults to ['output', 'target']
            pad_value: Fill value for zero-padding
        """
        self.size = tuple(size)
        self.keys = list(keys) if keys is not None else ['output', 'target']
        self.pad_value = pad_value

    def _pad_or_crop_1d(self, tensor: torch.Tensor, dim: int, target: int) -> torch.Tensor:
        """Pad or crop tensor along a single spatial dimension (dim 1=H, 2=W)."""
        current = tensor.shape[dim]
        if current == target:
            return tensor
        if current > target:
            # Center crop
            start = (current - target) // 2
            idx = [slice(None)] * tensor.dim()
            idx[dim] = slice(start, start + target)
            return tensor[tuple(idx)]
        else:
            # Center pad
            diff = target - current
            pad_before = diff // 2
            pad_after = diff - pad_before
            # torch.nn.functional.pad uses reversed dimension order: (W_left, W_right, H_top, H_bottom, ...)
            if dim == 2:  # W
                pad_spec = [pad_before, pad_after, 0, 0, 0, 0]
            elif dim == 1:  # H
                pad_spec = [0, 0, pad_before, pad_after, 0, 0]
            else:  # D (dim 0) - shouldn't be needed but handled
                pad_spec = [0, 0, 0, 0, pad_before, pad_after]
            return torch.nn.functional.pad(tensor, pad_spec[:tensor.dim() * 2], mode='constant', value=self.pad_value)

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = data.copy()
        target_h, target_w = self.size
        for key in self.keys:
            if key not in data:
                continue
            t = data[key]
            orig_shape = tuple(t.shape)
            t = self._pad_or_crop_1d(t, dim=1, target=target_h)
            t = self._pad_or_crop_1d(t, dim=2, target=target_w)
            result[key] = t
            if tuple(t.shape) != orig_shape:
                print(f"  PadOrCrop '{key}': {orig_shape} -> {tuple(t.shape)}")
        return result


class ComposeTransforms:
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[
        torch.Tensor, Dict[str, torch.Tensor]]:
        for t in self.transforms:
            data = t(data)
        return data


TRANSFORM_MAP = {
    'crop': CenterCrop,
    'brain_bbox': BrainBoundingBoxTransform,
    'transpose': TransposeTransform,
    'flip': FlipTransform,
    'pad_to_match': PadToMatchTransform,
    'crop_to_match': CropToMatchTransform,
    'clip': ClipTransform,
    'pad_or_crop': PadOrCropTransform,
}


def create_transform_from_config(transform_name: str, params):
    # Allow suffixed names like 'transpose_training', 'flip_sagittal', 'pad_to_match_target'
    # Check if transform_name starts with any key in TRANSFORM_MAP
    actual_name = None

    # First try exact match
    if transform_name in TRANSFORM_MAP:
        actual_name = transform_name
    else:
        # Try to find a matching prefix from TRANSFORM_MAP
        # Sort by length descending to match longest first (e.g., 'pad_to_match' before 'pad')
        for map_key in sorted(TRANSFORM_MAP.keys(), key=len, reverse=True):
            if transform_name.startswith(map_key):
                actual_name = map_key
                break

    if actual_name is None:
        raise ValueError(f"Unknown transform: {transform_name}. Available: {list(TRANSFORM_MAP.keys())}")

    transform_class = TRANSFORM_MAP[actual_name]

    print(f"Creating transform '{transform_name}' using base '{actual_name}'")

    if actual_name == 'flip':
        if isinstance(params, (dict, Munch)):
            # Just unpack the dictionary - it already has the right keys
            return transform_class(**params)
        elif isinstance(params, (list, tuple)):
            return transform_class(dims=params)
    elif actual_name == 'transpose':
        if isinstance(params, (dict, Munch)):
            # Just unpack the dictionary - it already has the right keys
            return transform_class(**params)
        elif isinstance(params, (list, tuple)):
            return transform_class(dims=params)
        else:
            raise ValueError(f"Transpose transform expects dict or list/tuple of dimensions, got {params}")
    elif actual_name == 'brain_bbox':
        if isinstance(params, dict):
            threshold = params.get('threshold', 0.01)
            padding = params.get('padding', 10)
            return transform_class(threshold=threshold, padding=padding)
        elif isinstance(params, list):
            threshold, padding = params
            return transform_class(threshold=threshold, padding=padding)
        else:
            return transform_class(threshold=float(params))

    elif actual_name == 'crop':
        if not isinstance(params, list) or len(params) != 2:
            raise ValueError(f"Crop transform expects [height, width], got {params}")
        return transform_class(tuple(params))

    elif actual_name == 'pad_to_match':
        if isinstance(params, (dict, Munch)):
            # Extract parameters with defaults
            reference_key = params.get('reference_key', 'output')
            target_key = params.get('target_key', 'target')
            mode = params.get('mode', 'constant')
            value = params.get('value', 0.0)
            return transform_class(reference_key=reference_key, target_key=target_key,
                                 mode=mode, value=value)
        elif isinstance(params, bool) and params:
            # Simple boolean flag - use defaults
            return transform_class()
        else:
            raise ValueError(f"pad_to_match expects dict or boolean, got {params}")

    elif actual_name == 'crop_to_match':
        if isinstance(params, (dict, Munch)):
            # Extract parameters with defaults
            reference_key = params.get('reference_key', 'target')
            target_key = params.get('target_key', 'output')
            mode = params.get('mode', 'center')
            return transform_class(reference_key=reference_key, target_key=target_key, mode=mode)
        elif isinstance(params, bool) and params:
            # Simple boolean flag - use defaults
            return transform_class()
        else:
            raise ValueError(f"crop_to_match expects dict or boolean, got {params}")

    elif actual_name == 'clip':
        if isinstance(params, (dict, Munch)):
            # Extract parameters with defaults
            min_val = params.get('min_val', 0.0)
            max_val = params.get('max_val', 1.0)
            keys = params.get('keys', None)
            return transform_class(min_val=min_val, max_val=max_val, keys=keys)
        elif isinstance(params, bool) and params:
            # Simple boolean flag - use defaults [0, 1]
            return transform_class()
        else:
            raise ValueError(f"clip expects dict or boolean, got {params}")

    elif actual_name == 'pad_or_crop':
        if isinstance(params, (dict, Munch)):
            size = params.get('size', None)
            if size is None:
                raise ValueError("pad_or_crop requires 'size: [H, W]'")
            keys = params.get('keys', None)
            pad_value = params.get('pad_value', 0.0)
            return transform_class(size=tuple(size), keys=keys, pad_value=pad_value)
        elif isinstance(params, list) and len(params) == 2:
            return transform_class(size=tuple(params))
        else:
            raise ValueError(f"pad_or_crop expects dict with 'size' or [H, W] list, got {params}")

    raise ValueError(f"Transform {transform_name} not implemented")