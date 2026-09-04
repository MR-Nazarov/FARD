import os as _os, sys as _sys
# evaluation/ imports the shared confs_functions and losses from the repo root
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "src"), _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import munch
import torch
import numpy as np
import warnings
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import binary_dilation
import nibabel as nib

from munch import Munch
from transforms import ComposeTransforms, CenterCrop, create_transform_from_config
import losses.metrics as Metrics
from losses.metrics import metrics_lpips
import os
import pydicom
import glob
import copy


class MetricsCalculator:
    def __init__(self, args):
        self.args = args
        self.config_device()
        self.initialize_metrics()
        self.initialize_transforms()
        self.initialize_data_paths()
        self.initialize_image_saving()
        self.metricStats = {'Test': {}}
        self.args.epoch = 1  # For stats tracking
        self.metricStats['Test'][self.args.epoch] = {metric: [] for metric in self.args.metrics}

    def config_device(self):
        if torch.cuda.is_available() and self.args.device != 'cpu':
            self.args.device = torch.device("cuda:0")
            torch.cuda.set_device(self.args.device)
            self.scaler = torch.cuda.amp.GradScaler() if self.args.amp else None
        else:
            self.args.device = torch.device("cpu")
            self.scaler = None
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def initialize_image_saving(self):
        """Initialize image saving configurations"""
        self.save_images = getattr(self.args, 'save_images', False)
        if not self.save_images:
            return

        # Image saving configurations
        self.save_predictions = getattr(self.args, 'save_predictions', True)
        self.save_targets = getattr(self.args, 'save_targets', True)
        self.save_inputs = getattr(self.args, 'save_inputs', False)  # New: save input images
        self.save_differences = getattr(self.args, 'save_differences', False)
        self.save_overlays = getattr(self.args, 'save_overlays', False)
        self.save_input_differences = getattr(self.args, 'save_input_differences', False)  # New: input vs target diff

        # Colorbar options for difference images
        self.add_colorbar = getattr(self.args, 'add_colorbar', True)  # Add colorbar to diff images
        self.colorbar_size = getattr(self.args, 'colorbar_size', 0.05)  # Fraction of image width
        self.colorbar_pad = getattr(self.args, 'colorbar_pad', 0.02)  # Padding between image and colorbar
        self.save_colorbar_separately = getattr(self.args, 'save_colorbar_separately', False)  # Save colorbar as separate image

        # Difference image normalization options
        self.normalize_differences = getattr(self.args, 'normalize_differences', True)  # Normalize diff images
        self.diff_normalization_method = getattr(self.args, 'diff_normalization_method',
                                                 'minmax')  # 'minmax', 'percentile', 'relative', 'global', 'nad'
        self.diff_percentile_range = getattr(self.args, 'diff_percentile_range',
                                             (1, 99))  # For percentile normalization
        self.diff_global_max = getattr(self.args, 'diff_global_max', None)  # For global normalization across cases

        # NAD (Normalized Absolute Difference) method parameters
        self.nad_foreground_percentile = getattr(self.args, 'nad_foreground_percentile', 95)  # Percentile for foreground detection
        self.nad_foreground_threshold = getattr(self.args, 'nad_foreground_threshold', 0.02)  # Fraction of p95 for mask
        self.nad_normalization_percentile = getattr(self.args, 'nad_normalization_percentile', 99)  # Percentile for S constant
        self.nad_colorbar_max = getattr(self.args, 'nad_colorbar_max', None)  # Fixed vmax for NAD colorbar (e.g., 0.20)

        # Image format and quality
        self.image_format = getattr(self.args, 'image_format', 'png')  # 'png', 'jpg', 'tiff'
        self.image_quality = getattr(self.args, 'image_quality', 95)  # For JPEG

        # Normalization options
        self.normalize_images = getattr(self.args, 'normalize_images', True)
        self.normalization_method = getattr(self.args, 'normalization_method',
                                            'minmax')  # 'minmax', 'zscore', 'percentile'
        self.percentile_range = getattr(self.args, 'percentile_range', (1, 99))

        # Color mapping
        self.colormap = getattr(self.args, 'colormap', 'gray')  # matplotlib colormap name

        # Output directory
        self.images_output_dir = getattr(self.args, 'images_output_dir', 'saved_images')
        self.images_output_path = Path(self.images_output_dir)
        self.images_output_path.mkdir(parents=True, exist_ok=True)

        print(f"Image saving initialized: {self.images_output_path}")

    def initialize_data_paths(self):
        """Initialize paths for predictions, targets, and inputs"""
        self.output = {}
        self.targets = {}
        self.inputs = {}
        self.masks = {}

        # Check if input metrics are requested
        self.calculate_input_metrics = (
            (hasattr(self.args, 'input') and self.args.input is not None) or
            (hasattr(self.args, 'input_per_conf') and self.args.input_per_conf)
        )

        # Check if mask-based calculation is requested
        self.calc_in_mask = getattr(self.args, 'calc_in_mask', False)
        self.mask_name = getattr(self.args, 'mask_name', 'seg')
        self.mask_classes = getattr(self.args, 'mask_classes', [1, 2, 3, 4])

        # Build per-conf target lookup (falls back to global target)
        target_per_conf = getattr(self.args, 'target_per_conf', {})
        base_data_dir_per_conf = getattr(self.args, 'base_data_dir_per_conf', {})

        # Get all conf directories to process
        for conf_num in self.args.confs_to_process:
            conf_dir = Path(self.args.base_result_dir) / f"conf_{conf_num}"
            if not conf_dir.exists():
                print(f"Warning: {conf_dir} does not exist")
                continue

            # Process each case in the conf directory
            for case_dir in conf_dir.iterdir():
                if case_dir.is_dir():
                    case_num = case_dir.name

                    # Initialize nested dictionaries if needed
                    if conf_num not in self.output:
                        self.output[conf_num] = {}
                    if conf_num not in self.targets:
                        self.targets[conf_num] = {}
                    if self.calculate_input_metrics and conf_num not in self.inputs:
                        self.inputs[conf_num] = {}

                    # Get prediction path - find the first directory in case_dir.
                    # Optional: pred_subdir_per_conf selects a specific subdir by
                    # name substring when a conf has multiple prediction runs.
                    pred_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
                    pred_subdir_per_conf = getattr(self.args, 'pred_subdir_per_conf', {})
                    wanted_subdir = pred_subdir_per_conf.get(conf_num)
                    if pred_dirs:
                        if wanted_subdir is not None:
                            matches = [d for d in pred_dirs if wanted_subdir in d.name]
                            if not matches:
                                raise ValueError(
                                    f"pred_subdir_per_conf['{conf_num}']='{wanted_subdir}' "
                                    f"matched no subdir in {case_dir} (have: {[d.name for d in pred_dirs]})")
                            pred_path = matches[0]
                        else:
                            pred_path = pred_dirs[0]  # Take the first directory found
                        if pred_path.exists():
                            self.output[conf_num][case_num] = pred_path
                            print(f"Found prediction directory: {pred_path}")

                    # Get corresponding target path (dir of npy slices or a NIfTI file)
                    target_name = target_per_conf.get(conf_num, self.args.target)
                    target_suffix = getattr(self.args, 'target_suffix', '')
                    target_path = Path(self.args.base_data_dir) / case_num / (target_name + target_suffix)
                    if target_path.exists():
                        self.targets[conf_num][case_num] = target_path
                        print(f"Found target: {target_path}")

                    # Get corresponding input path if requested
                    if self.calculate_input_metrics:
                        input_per_conf = getattr(self.args, 'input_per_conf', {})
                        input_name = input_per_conf.get(conf_num, getattr(self.args, 'input', None))
                        input_path = Path(self.args.base_data_dir) / case_num / input_name
                        if input_path.exists():
                            self.inputs[conf_num][case_num] = input_path
                            print(f"Found input directory: {input_path}")
                        else:
                            print(f"Warning: Input directory not found: {input_path}")

                    # Get corresponding mask path if mask-based calculation is requested
                    if self.calc_in_mask:
                        if conf_num not in self.masks:
                            self.masks[conf_num] = {}
                        mask_path = Path(self.args.base_data_dir) / case_num / self.mask_name
                        if mask_path.exists():
                            self.masks[conf_num][case_num] = mask_path
                            print(f"Found mask directory: {mask_path}")
                        else:
                            print(f"Warning: Mask directory not found: {mask_path}")

    def initialize_metrics(self):
        """Initialize metric functions"""
        self.metricFuncs = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            for metric in self.args.metrics:
                if 'monai' not in metric:
                    if metric == 'LPIPS':
                        metrics_lpips.initialize_lpips()
                    self.metricFuncs[metric] = getattr(Metrics, metric)

    def initialize_transforms(self):
        """Initialize transforms from configuration"""
        transforms = []

        # Get visualization settings from config
        self.enable_transform_visualization = getattr(self.args, 'enable_transform_visualization', False)
        self.enable_slice_preview = getattr(self.args, 'enable_slice_preview', True)

        if hasattr(self.args, 'transforms'):
            for transform_name, params in self.args.transforms.items():
                try:
                    transform = create_transform_from_config(transform_name, params)

                    # Disable visualization in transforms if configured
                    if hasattr(transform, 'enable_visualization'):
                        transform.enable_visualization = self.enable_transform_visualization

                    transforms.append(transform)
                    print(f"Added {transform_name} transform with parameters {params}")
                except Exception as e:
                    print(f"Warning: Failed to initialize {transform_name} transform: {str(e)}")

        self.transforms = ComposeTransforms(transforms) if transforms else None

        if not self.enable_transform_visualization:
            print("Transform visualizations are DISABLED")
        if not self.enable_slice_preview:
            print("Middle slice preview is DISABLED")

    def show_middle_slice_preview(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor,
                                  conf_num: str, case_num: str, input_tensor: torch.Tensor = None):
        """Display the middle slice of prediction and target images before calculating metrics"""
        # Check if preview is enabled
        if not self.enable_slice_preview:
            return

        try:
            # Get middle slice index
            num_slices = pred_tensor.shape[0]
            middle_idx = num_slices // 2

            print(f"\nDisplaying middle slice ({middle_idx}/{num_slices - 1}) for Conf {conf_num}, Case {case_num}")

            # Extract middle slices and convert to numpy
            pred_slice = pred_tensor[middle_idx].cpu().numpy()
            target_slice = target_tensor[middle_idx].cpu().numpy()

            # Normalize images for display (0-1 range)
            pred_normalized = self.normalize_image_for_display(pred_slice)
            target_normalized = self.normalize_image_for_display(target_slice)

            # Create figure
            if input_tensor is not None:
                input_slice = input_tensor[middle_idx].cpu().numpy()
                input_normalized = self.normalize_image_for_display(input_slice)
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                titles = ['Input', 'Target', 'Prediction', 'Absolute Difference']

                # Calculate difference
                diff_slice = np.abs(pred_normalized - target_normalized)

                images = [input_normalized, target_normalized, pred_normalized, diff_slice]
                cmaps = ['gray', 'gray', 'gray', 'magma']

            else:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                titles = ['Target', 'Prediction', 'Absolute Difference']

                # Calculate difference
                diff_slice = np.abs(pred_normalized - target_normalized)

                images = [target_normalized, pred_normalized, diff_slice]
                cmaps = ['gray', 'gray', 'magma']

            # Plot images
            for i, (ax, img, title, cmap) in enumerate(zip(axes, images, titles, cmaps)):
                im = ax.imshow(img, cmap=cmap, aspect='equal')
                ax.set_title(f'{title}\nSlice {middle_idx}')
                ax.axis('off')

                # Add colorbar for difference image
                if 'Difference' in title:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Add overall title
            fig.suptitle(f'Configuration {conf_num} - Case {case_num} - Middle Slice Preview',
                         fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.show()

            # Print some statistics
            print(
                f"  Target slice - Min: {target_slice.min():.4f}, Max: {target_slice.max():.4f}, Mean: {target_slice.mean():.4f}")
            print(
                f"  Prediction slice - Min: {pred_slice.min():.4f}, Max: {pred_slice.max():.4f}, Mean: {pred_slice.mean():.4f}")
            if input_tensor is not None:
                print(
                    f"  Input slice - Min: {input_slice.min():.4f}, Max: {input_slice.max():.4f}, Mean: {input_slice.mean():.4f}")
            print(
                f"  Absolute difference - Min: {diff_slice.min():.4f}, Max: {diff_slice.max():.4f}, Mean: {diff_slice.mean():.4f}")

        except Exception as e:
            print(f"Error displaying preview for case {case_num}: {str(e)}")

    def normalize_image_for_display(self, image_array: np.ndarray) -> np.ndarray:
        """Normalize image for display (simple min-max to 0-1 range)"""
        min_val = np.min(image_array)
        max_val = np.max(image_array)
        if max_val > min_val:
            return (image_array - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(image_array)

    def normalize_difference_image(self, diff_array: np.ndarray, target_array: np.ndarray = None) -> tuple:
        """Normalize difference image for better visualization

        Returns:
            (normalized_diff, raw_stats, norm_info) - normalized array, raw statistics, normalization info
        """
        raw_stats = {
            'mean': np.mean(diff_array),
            'max': np.max(diff_array),
            'std': np.std(diff_array),
            'min': np.min(diff_array)
        }

        if not self.normalize_differences:
            return diff_array, raw_stats, "Raw values (no normalization)"

        if self.diff_normalization_method == 'minmax':
            # Standard min-max normalization to [0,1]
            min_val = np.min(diff_array)
            max_val = np.max(diff_array)
            if max_val > min_val:
                normalized = (diff_array - min_val) / (max_val - min_val)
                norm_info = f"MinMax normalized [0,1] (raw range: {min_val:.4f}-{max_val:.4f})"
            else:
                normalized = np.zeros_like(diff_array)
                norm_info = f"Constant difference: {max_val:.4f}"

        elif self.diff_normalization_method == 'percentile':
            # Percentile-based normalization to handle outliers
            p_low, p_high = self.diff_percentile_range
            low_val = np.percentile(diff_array, p_low)
            high_val = np.percentile(diff_array, p_high)
            if high_val > low_val:
                normalized = np.clip((diff_array - low_val) / (high_val - low_val), 0, 1)
                norm_info = f"Percentile normalized {p_low}-{p_high}% → [0,1] (raw: {low_val:.4f}-{high_val:.4f})"
            else:
                normalized = np.zeros_like(diff_array)
                norm_info = f"Constant in percentile range: {high_val:.4f}"

        elif self.diff_normalization_method == 'relative' and target_array is not None:
            # Normalize by target image range (shows differences relative to signal strength)
            target_range = np.max(target_array) - np.min(target_array)
            if target_range > 0:
                normalized = diff_array / target_range
                norm_info = f"Target-relative normalized (target range: {target_range:.4f})"
            else:
                normalized = diff_array
                norm_info = "Target range is zero, no normalization applied"

        elif self.diff_normalization_method == 'global' and self.diff_global_max is not None:
            # Use global maximum for consistent scale across all cases
            normalized = diff_array / self.diff_global_max
            norm_info = f"Global normalized (max: {self.diff_global_max:.4f})"

        elif self.diff_normalization_method == 'nad' and target_array is not None:
            # NAD (Normalized Absolute Difference) method
            # Compute foreground mask from GT
            a = np.abs(target_array)
            p95 = np.percentile(a, self.nad_foreground_percentile)
            threshold = self.nad_foreground_threshold * p95
            mask = a > threshold

            # Compute normalization constant S from GT within mask
            if np.any(mask):
                a_masked = a[mask]
                S = np.percentile(a_masked, self.nad_normalization_percentile)
                # Fallback if S is zero
                if S == 0:
                    S = np.max(a_masked) if np.max(a_masked) > 0 else 1.0
            else:
                # If no foreground detected, use whole image
                S = np.percentile(a, self.nad_normalization_percentile)
                if S == 0:
                    S = np.max(a) if np.max(a) > 0 else 1.0

            # Normalize error by S
            normalized = diff_array / S
            norm_info = f"NAD normalized (S={S:.4f}, mask_pixels={np.sum(mask)}/{mask.size}, threshold={threshold:.4f})"

        else:
            # Fallback to minmax if method not recognized or missing parameters
            min_val = np.min(diff_array)
            max_val = np.max(diff_array)
            if max_val > min_val:
                normalized = (diff_array - min_val) / (max_val - min_val)
                norm_info = f"Fallback MinMax [0,1] (raw: {min_val:.4f}-{max_val:.4f})"
            else:
                normalized = np.zeros_like(diff_array)
                norm_info = f"Constant difference: {max_val:.4f}"

        return normalized, raw_stats, norm_info

    def normalize_image(self, image_array: np.ndarray) -> np.ndarray:
        """Normalize image for saving"""
        if not self.normalize_images:
            return image_array

        if self.normalization_method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(image_array)
            max_val = np.max(image_array)
            if max_val > min_val:
                return (image_array - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(image_array)

        elif self.normalization_method == 'percentile':
            # Percentile-based normalization
            p_low, p_high = self.percentile_range
            low_val = np.percentile(image_array, p_low)
            high_val = np.percentile(image_array, p_high)
            if high_val > low_val:
                normalized = np.clip((image_array - low_val) / (high_val - low_val), 0, 1)
                return normalized
            else:
                return np.zeros_like(image_array)

        elif self.normalization_method == 'zscore':
            # Z-score normalization, then scale to [0, 1]
            mean_val = np.mean(image_array)
            std_val = np.std(image_array)
            if std_val > 0:
                z_normalized = (image_array - mean_val) / std_val
                # Scale to [0, 1] using 3-sigma rule
                normalized = np.clip((z_normalized + 3) / 6, 0, 1)
                return normalized
            else:
                return np.zeros_like(image_array)

        return image_array

    def save_single_image(self, image_tensor: torch.Tensor, filepath: Path,
                          image_type: str = "prediction"):
        """Save a single image tensor to file"""
        try:
            # Convert to numpy and squeeze unnecessary dimensions
            if image_tensor.dim() > 2:
                # Remove batch and channel dimensions
                while image_tensor.dim() > 2:
                    image_tensor = image_tensor.squeeze(0)

            image_array = image_tensor.cpu().numpy()

            # Normalize the image
            normalized_array = self.normalize_image(image_array)

            # Apply colormap if not grayscale
            if self.colormap != 'gray':
                cmap = cm.get_cmap(self.colormap)
                colored_array = cmap(normalized_array)
                # Convert to RGB (remove alpha channel if present)
                if colored_array.shape[-1] == 4:
                    colored_array = colored_array[..., :3]
                image_to_save = (colored_array * 255).astype(np.uint8)
            else:
                image_to_save = (normalized_array * 255).astype(np.uint8)

            # Create PIL Image
            if len(image_to_save.shape) == 2:
                pil_image = Image.fromarray(image_to_save, mode='L')
            else:
                pil_image = Image.fromarray(image_to_save, mode='RGB')

            # Save with appropriate format
            if self.image_format.lower() == 'jpg' or self.image_format.lower() == 'jpeg':
                pil_image.save(filepath, format='JPEG', quality=self.image_quality)
            elif self.image_format.lower() == 'png':
                pil_image.save(filepath, format='PNG')
            elif self.image_format.lower() == 'tiff':
                pil_image.save(filepath, format='TIFF')
            else:
                pil_image.save(filepath)

        except Exception as e:
            print(f"Error saving image {filepath}: {str(e)}")

    def save_difference_image(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor,
                              filepath: Path, title_suffix: str = ""):
        """Save difference/error map between prediction and target with optional colorbar and normalization"""
        try:
            # Ensure same dimensions
            if pred_tensor.dim() > 2:
                while pred_tensor.dim() > 2:
                    pred_tensor = pred_tensor.squeeze(0)
            if target_tensor.dim() > 2:
                while target_tensor.dim() > 2:
                    target_tensor = target_tensor.squeeze(0)

            pred_array = pred_tensor.cpu().numpy()
            target_array = target_tensor.cpu().numpy()

            # Calculate absolute difference
            diff_array = np.abs(pred_array - target_array)

            # Normalize difference image
            normalized_diff, raw_stats, norm_info = self.normalize_difference_image(diff_array, target_array)

            if self.save_colorbar_separately:
                # Save difference image without colorbar or title
                cmap = cm.get_cmap('magma')

                # Clip and normalize if using fixed vmax
                if self.diff_normalization_method == 'nad' and self.nad_colorbar_max is not None:
                    # Clip to [0, vmax] and normalize to [0, 1] for colormap
                    display_diff = np.clip(normalized_diff, 0, self.nad_colorbar_max) / self.nad_colorbar_max
                else:
                    # Use normalized_diff as-is
                    display_diff = normalized_diff

                colored_diff = cmap(display_diff)
                image_to_save = (colored_diff[..., :3] * 255).astype(np.uint8)

                pil_image = Image.fromarray(image_to_save, mode='RGB')
                pil_image.save(filepath, format='PNG')
                print(f"Saved difference image (no colorbar): {filepath}")

                # Save colorbar separately
                colorbar_path = filepath.parent / f"{filepath.stem}_colorbar.png"
                self.save_colorbar_only(normalized_diff, colorbar_path, raw_stats, norm_info, title_suffix)
                print(f"Saved colorbar: {colorbar_path}")

            elif self.add_colorbar:
                # Use matplotlib to create image with colorbar
                fig, ax = plt.subplots(figsize=(12, 8))

                # Display the normalized difference image
                im = ax.imshow(normalized_diff, cmap='magma', aspect='equal')

                # Create comprehensive title with both raw and normalized stats
                title = f'Absolute Difference{title_suffix}'
                raw_stats_text = f'Raw - Mean: {raw_stats["mean"]:.4f}, Max: {raw_stats["max"]:.4f}, Std: {raw_stats["std"]:.4f}'
                norm_stats_text = f'Displayed - {norm_info}'

                ax.set_title(f'{title}\n{raw_stats_text}\n{norm_stats_text}',
                             fontsize=11, pad=25)
                ax.axis('off')

                # Add colorbar with proper labeling
                cbar = plt.colorbar(im, ax=ax, fraction=self.colorbar_size,
                                    pad=self.colorbar_pad, shrink=0.8)

                if self.normalize_differences:
                    cbar.set_label('Normalized Absolute Difference', rotation=270, labelpad=20)
                else:
                    cbar.set_label('Raw Absolute Difference', rotation=270, labelpad=20)

                # Format colorbar ticks
                if not self.normalize_differences or self.diff_normalization_method == 'relative':
                    # Use scientific notation for raw values or relative values
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.formatter.set_scientific(True)
                    cbar.formatter.set_useMathText(True)
                cbar.update_ticks()

                # Save with matplotlib
                plt.tight_layout()
                plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

            else:
                # Original method without colorbar - use normalized difference
                cmap = cm.get_cmap('magma')
                colored_diff = cmap(normalized_diff)
                image_to_save = (colored_diff[..., :3] * 255).astype(np.uint8)

                pil_image = Image.fromarray(image_to_save, mode='RGB')
                pil_image.save(filepath, format='PNG')

        except Exception as e:
            print(f"Error saving difference image {filepath}: {str(e)}")

    def save_colorbar_only(self, normalized_diff, filepath: Path, raw_stats, norm_info, title_suffix: str = ""):
        """Save a standalone colorbar image with statistics"""
        try:
            # Create a figure with just the colorbar
            fig = plt.figure(figsize=(2, 8))
            ax = fig.add_axes([0.05, 0.05, 0.2, 0.9])

            # Create a dummy mappable for the colorbar
            import matplotlib.colors as mcolors

            # Determine vmin and vmax
            vmin = 0  # Always start at 0 for difference maps
            if self.diff_normalization_method == 'nad' and self.nad_colorbar_max is not None:
                # Use fixed vmax for NAD method
                vmax = self.nad_colorbar_max
            else:
                # Auto-scale to data
                vmax = normalized_diff.max()

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(cmap='magma', norm=norm)
            sm.set_array([])

            # Add colorbar
            cbar = plt.colorbar(sm, cax=ax)

            # Add label
            if self.normalize_differences:
                cbar.set_label('Normalized Absolute Difference', rotation=270, labelpad=20, fontsize=10)
            else:
                cbar.set_label('Raw Absolute Difference', rotation=270, labelpad=20, fontsize=10)

            # Format colorbar ticks
            if not self.normalize_differences or self.diff_normalization_method == 'relative':
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_useMathText(True)
            cbar.update_ticks()

            # Save the colorbar
            plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

            # Also save a text file with statistics
            stats_path = filepath.parent / f"{filepath.stem}_stats.txt"
            with open(stats_path, 'w') as f:
                f.write(f"Absolute Difference{title_suffix}\n")
                f.write("=" * 50 + "\n\n")
                f.write("Raw Statistics:\n")
                f.write(f"  Mean: {raw_stats['mean']:.6f}\n")
                f.write(f"  Max:  {raw_stats['max']:.6f}\n")
                f.write(f"  Min:  {raw_stats['min']:.6f}\n")
                f.write(f"  Std:  {raw_stats['std']:.6f}\n\n")
                f.write(f"Normalization: {norm_info}\n")

        except Exception as e:
            print(f"Error saving colorbar {filepath}: {str(e)}")

    def save_overlay_image(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor,
                           filepath: Path, alpha: float = 0.5):
        """Save overlay of prediction and target"""
        try:
            # Ensure same dimensions
            if pred_tensor.dim() > 2:
                while pred_tensor.dim() > 2:
                    pred_tensor = pred_tensor.squeeze(0)
            if target_tensor.dim() > 2:
                while target_tensor.dim() > 2:
                    target_tensor = target_tensor.squeeze(0)

            pred_array = self.normalize_image(pred_tensor.cpu().numpy())
            target_array = self.normalize_image(target_tensor.cpu().numpy())

            # Create RGB overlay: prediction in red channel, target in green channel
            overlay = np.zeros((*pred_array.shape, 3))
            overlay[..., 0] = pred_array  # Red channel for prediction
            overlay[..., 1] = target_array  # Green channel for target
            overlay[..., 2] = 0  # Blue channel empty

            # Apply alpha blending with grayscale background
            gray_bg = (pred_array + target_array) / 2
            gray_bg_rgb = np.stack([gray_bg] * 3, axis=-1)

            blended = alpha * overlay + (1 - alpha) * gray_bg_rgb
            image_to_save = (np.clip(blended, 0, 1) * 255).astype(np.uint8)

            pil_image = Image.fromarray(image_to_save, mode='RGB')
            pil_image.save(filepath, format='PNG')

        except Exception as e:
            print(f"Error saving overlay image {filepath}: {str(e)}")

    def save_slice_images(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor,
                          conf_num: str, case_num: str, slice_idx: int,
                          pred_metrics: Dict = None, input_tensor: torch.Tensor = None,
                          input_metrics: Dict = None):
        """Save images for a specific slice"""
        if not self.save_images:
            return

        # Create directory structure
        case_dir = self.images_output_path / f"conf_{conf_num}" / case_num
        case_dir.mkdir(parents=True, exist_ok=True)

        # Base filename
        base_name = f"slice_{slice_idx:03d}"

        # Add prediction metrics to filename if available
        if pred_metrics:
            pred_metrics_str = "_".join([f"{k}_{v:.3f}" for k, v in pred_metrics.items()
                                         if v is not None and k in ['PSNR', 'SSIM', 'MSE']])
            if pred_metrics_str:
                base_name += f"_pred_{pred_metrics_str}"

        # Save prediction
        if self.save_predictions:
            pred_path = case_dir / f"{base_name}_pred.{self.image_format}"
            self.save_single_image(pred_tensor, pred_path, "prediction")

        # Save target
        if self.save_targets:
            target_path = case_dir / f"{base_name}_target.{self.image_format}"
            self.save_single_image(target_tensor, target_path, "target")

        # Save input if available
        if input_tensor is not None and self.save_inputs:
            input_name = base_name
            if input_metrics:
                input_metrics_str = "_".join([f"{k}_{v:.3f}" for k, v in input_metrics.items()
                                              if v is not None and k in ['PSNR', 'SSIM', 'MSE']])
                if input_metrics_str:
                    input_name += f"_input_{input_metrics_str}"
            input_path = case_dir / f"{input_name}_input.{self.image_format}"
            self.save_single_image(input_tensor, input_path, "input")

        # Save prediction difference map
        if self.save_differences:
            diff_path = case_dir / f"{base_name}_pred_diff.png"
            print(f"Saving prediction diff to: {diff_path}")
            self.save_difference_image(pred_tensor, target_tensor, diff_path, " (Prediction vs Target)")
        else:
            print(f"Skipping prediction diff (save_differences={self.save_differences})")

        # Save input difference map if available
        if input_tensor is not None and self.save_input_differences:
            input_diff_path = case_dir / f"{base_name}_input_diff.png"
            print(f"Saving input diff to: {input_diff_path}")
            self.save_difference_image(input_tensor, target_tensor, input_diff_path, " (Input vs Target)")
        elif input_tensor is not None:
            print(f"Skipping input diff (save_input_differences={self.save_input_differences})")
        else:
            print("No input tensor - skipping input diff")

        # Save overlay
        if self.save_overlays:
            overlay_path = case_dir / f"{base_name}_pred_overlay.png"
            self.save_overlay_image(pred_tensor, target_tensor, overlay_path)

    def create_binary_mask(self, mask_tensor: torch.Tensor, classes: List[int]) -> torch.Tensor:
        """Create a binary mask from specified class labels.

        Args:
            mask_tensor: Tensor containing class labels (e.g., 0, 1, 2, 3, 4)
            classes: List of class labels to include in the mask (e.g., [1, 2])

        Returns:
            Binary mask tensor where 1 = inside mask, 0 = outside
        """
        binary_mask = torch.zeros_like(mask_tensor, dtype=torch.bool)
        for cls in classes:
            binary_mask = binary_mask | (mask_tensor == cls)
        return binary_mask.float()

    def apply_mask_to_tensor(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply binary mask to tensor, setting values outside mask to 0.

        Args:
            tensor: Input tensor to mask
            mask: Binary mask tensor (1 = keep, 0 = discard)

        Returns:
            Masked tensor
        """
        return tensor * mask

    def create_wt_mask(self, seg_tensor: torch.Tensor) -> torch.Tensor:
        """Create Whole Tumor (WT) mask from segmentation labels 1, 2, 3 only.

        BRATS labels:
        - 1 = NETC (non-enhancing tumor core)
        - 2 = SNFH (peritumoral edematous/invaded tissue)
        - 3 = ET (enhancing tumor)
        - 4 = RC (resection cavity) - EXCLUDED

        Args:
            seg_tensor: Segmentation tensor with class labels

        Returns:
            Binary mask where 1 = tumor tissue (labels 1-3), 0 = other
        """
        wt_mask = ((seg_tensor == 1) | (seg_tensor == 2) | (seg_tensor == 3))
        return wt_mask

    def create_brain_mask(self, target_tensor: torch.Tensor) -> torch.Tensor:
        """Create brain mask from target (HR) image.

        Since BraTS is skull-stripped, brain_mask = (HR != 0)

        Args:
            target_tensor: Target/HR image tensor

        Returns:
            Binary mask where 1 = brain tissue, 0 = background
        """
        brain_mask = (target_tensor != 0)
        return brain_mask

    def create_ring_mask(self, wt_mask: torch.Tensor, brain_mask: torch.Tensor,
                         dilation_iterations: int = 3) -> torch.Tensor:
        """Create perilesional ring mask by dilating WT and excluding WT itself.

        ring = dilated_WT & brain_mask & (~WT)

        Args:
            wt_mask: Whole Tumor binary mask
            brain_mask: Brain tissue binary mask
            dilation_iterations: Number of dilation iterations (default 3 voxels)

        Returns:
            Binary mask for the perilesional ring region
        """
        # Convert to numpy for scipy binary_dilation
        wt_np = wt_mask.cpu().numpy().astype(bool)

        # Dilate WT mask
        dilated_wt_np = binary_dilation(wt_np, iterations=dilation_iterations)

        # Convert back to torch tensor
        dilated_wt = torch.from_numpy(dilated_wt_np).to(wt_mask.device)

        # Compute ring: dilated_WT & brain_mask & (~WT)
        ring_mask = dilated_wt & brain_mask & (~wt_mask)

        return ring_mask

    def calculate_masked_mae(self, pred: torch.Tensor, target: torch.Tensor,
                             mask: torch.Tensor) -> tuple:
        """Calculate MAE within a mask region.

        Args:
            pred: Prediction tensor
            target: Target tensor
            mask: Binary mask tensor

        Returns:
            (mae_value, n_voxels) - MAE value and number of voxels in mask
        """
        mask_bool = mask.bool()
        n_voxels = mask_bool.sum().item()

        if n_voxels == 0:
            return None, 0

        pred_masked = pred[mask_bool]
        target_masked = target[mask_bool]

        mae = torch.mean(torch.abs(pred_masked - target_masked)).item()
        return mae, n_voxels

    def load_volume(self, path: Path) -> torch.Tensor:
        """
        Load a 3D volume either from:
          (1) a single .npy file,
          (2) a NIfTI file (.nii or .nii.gz),
          (3) a directory of per‐slice .npy files, or
          (4) a folder of DICOMs.
        """
        # (1) direct .npy file
        if path.suffix == '.npy':
            arr = np.load(str(path))

        # (2) NIfTI file
        elif path.suffix in ('.nii', '.gz'):
            arr = nib.load(str(path)).get_fdata().astype(np.float32)

        # (2) directory of .npy (BRATS mode)
        elif path.is_dir() and self.args.use_npy:
            npy_files = sorted(path.glob('*.npy'))
            if not npy_files:
                raise ValueError(f"No .npy files found in {path}")

            # if there's exactly one .npy and it's already a 3D array, load it directly
            if len(npy_files) == 1:
                arr = np.load(str(npy_files[0]))
            else:
                # assume each file is a 2D slice → stack into a 3D volume
                slices = [np.load(str(f)) for f in npy_files]
                arr = np.stack(slices, axis=0)

        # (3) fallback to DICOM series
        else:
            dicom_files = sorted(glob.glob(str(path / '*.dcm')))
            if not dicom_files:
                raise ValueError(f"No DICOM files found in {path}")
            first = pydicom.dcmread(dicom_files[0])
            arr = np.zeros((len(dicom_files), first.Rows, first.Columns),
                           dtype=np.float32)
            for i, dcm in enumerate(dicom_files):
                arr[i] = pydicom.dcmread(dcm).pixel_array.astype(np.float32)

        volume = torch.from_numpy(arr).float().to(self.args.device)
        return volume

    def calculate_metrics(self, data_dict):
        """Calculate metrics for a single pair"""
        metrics_values = {}
        data_dict = Munch(data_dict)
        # Normalize before metrics.  Two modes:
        #
        # normalize_by_target (preferred): normalize both pred and target using
        #   the TARGET's min/max.  Target → [0,1]; pred scaled by the same factor.
        #   Preserves contrast errors in the prediction.
        #
        # normalize_for_metrics (legacy default True): each tensor normalized
        #   independently by its own min/max.  Hides global gain errors.
        #
        # If normalize_by_target is True it takes precedence regardless of
        # normalize_for_metrics.
        if getattr(self.args, 'normalize_by_target', False):
            _tgt = data_dict.get('target')
            if _tgt is not None:
                _tgt = _tgt.float()
                _mn  = _tgt.min()
                _rng = _tgt.max() - _mn
                if _rng > 0:
                    data_dict['target'] = (_tgt - _mn) / _rng
                    _out = data_dict.get('output')
                    if _out is not None:
                        data_dict['output'] = (_out.float() - _mn) / _rng
        elif getattr(self.args, 'normalize_for_metrics', True):
            for _k in ('output', 'target'):
                _t = data_dict.get(_k)
                if _t is not None:
                    _t = _t.float()
                    _mn = _t.min()
                    _rng = _t.max() - _mn
                    if _rng > 0:
                        data_dict[_k] = (_t - _mn) / _rng
        for metric in self.args.metrics:
            if 'monai' not in metric:
                try:
                    value = self.metricFuncs[metric](data_dict)
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().detach().item()
                    metrics_values[metric] = value
                except Exception as e:
                    print(f"Error calculating {metric}: {str(e)}")
                    metrics_values[metric] = None

        return metrics_values

    def print_specific_slice_metrics(self, pred_slice_metrics: List[Dict], case_num: str, conf_num: str,
                                     slice_indices: List[int], start_offset: int = 0,
                                     input_slice_metrics: List[Dict] = None):
        """Print metrics for specific slices in a formatted table

        Args:
            pred_slice_metrics: Prediction vs target metrics
            input_slice_metrics: Input vs target metrics (optional)
            slice_indices: List of slice numbers to display
            start_offset: Offset to add to slice_indices for display (usually 0 for absolute indices)
        """
        print(f"\n=== Specific Slice Metrics for Conf {conf_num}, Case {case_num} ===")

        # Header
        header = f"{'Slice':<6}"

        # Prediction metrics columns
        for metric in self.args.metrics:
            if 'monai' not in metric:
                header += f"{'Pred_' + metric:<10}"

        # Input metrics columns (if available)
        if input_slice_metrics is not None:
            for metric in self.args.metrics:
                if 'monai' not in metric:
                    header += f"{'Input_' + metric:<10}"

        # Improvement columns (if both available)
        if input_slice_metrics is not None:
            for metric in self.args.metrics:
                if 'monai' not in metric:
                    if metric in ['PSNR', 'SSIM']:  # Higher is better
                        header += f"{'Δ_' + metric:<10}"
                    elif metric in ['MSE', 'MAE', 'LPIPS']:  # Lower is better
                        header += f"{'Δ_' + metric:<10}"

        print(header)
        print("-" * len(header))

        # Print metrics for each requested slice
        for i, slice_idx in enumerate(slice_indices):
            if i < len(pred_slice_metrics) and pred_slice_metrics[i] is not None:
                display_idx = start_offset + slice_idx
                row = f"{display_idx:<6}"

                # Prediction metrics
                pred_metrics = pred_slice_metrics[i]
                for metric in self.args.metrics:
                    if 'monai' not in metric:
                        value = pred_metrics.get(metric, None)
                        if value is not None:
                            row += f"{value:<10.4f}"
                        else:
                            row += f"{'N/A':<10}"

                # Input metrics (if available)
                if input_slice_metrics is not None and i < len(input_slice_metrics):
                    input_metrics = input_slice_metrics[i] if input_slice_metrics[i] is not None else {}
                    for metric in self.args.metrics:
                        if 'monai' not in metric:
                            value = input_metrics.get(metric, None)
                            if value is not None:
                                row += f"{value:<10.4f}"
                            else:
                                row += f"{'N/A':<10}"

                    # Improvement calculations
                    for metric in self.args.metrics:
                        if 'monai' not in metric:
                            pred_val = pred_metrics.get(metric, None)
                            input_val = input_metrics.get(metric, None)

                            if pred_val is not None and input_val is not None:
                                if metric in ['PSNR', 'SSIM']:  # Higher is better
                                    improvement = pred_val - input_val
                                    row += f"{improvement:<10.4f}"
                                elif metric in ['MSE', 'MAE', 'LPIPS']:  # Lower is better
                                    improvement = input_val - pred_val  # Positive = improvement
                                    row += f"{improvement:<10.4f}"
                                else:
                                    row += f"{'N/A':<10}"
                            else:
                                row += f"{'N/A':<10}"

                print(row)
            else:
                display_idx = start_offset + slice_idx
                print(f"{display_idx:<6}{'ERROR':<10}")
        print()

    def format_specific_slice_results_for_excel(self, all_slice_results: Dict) -> str:
        """Format specific slice results for Excel copy-paste"""
        output = "Configuration\tCase\tSlice\t"
        metrics = [m for m in self.args.metrics if 'monai' not in m]

        # Prediction metrics columns
        for metric in metrics:
            output += f"Pred_{metric}\t"

        # Input metrics columns (if available)
        if self.calculate_input_metrics:
            for metric in metrics:
                output += f"Input_{metric}\t"

        output = output.rstrip('\t') + '\n'

        for conf_num in sorted(all_slice_results.keys()):
            for case_num in sorted(all_slice_results[conf_num].keys()):
                for slice_data in all_slice_results[conf_num][case_num]:
                    slice_idx = slice_data['slice_idx']
                    pred_metrics = slice_data['pred_metrics']
                    input_metrics = slice_data.get('input_metrics', None)

                    output += f"{conf_num}\t{case_num}\t{slice_idx}\t"

                    # Prediction metrics
                    for metric in metrics:
                        value = pred_metrics.get(metric, None)
                        if value is not None:
                            output += f"{value:.4f}\t"
                        else:
                            output += "\t"

                    # Input metrics (if available)
                    if self.calculate_input_metrics:
                        for metric in metrics:
                            if input_metrics is not None:
                                value = input_metrics.get(metric, None)
                                if value is not None:
                                    output += f"{value:.4f}\t"
                                else:
                                    output += "\t"
                            else:
                                output += "\t"

                    output = output.rstrip('\t') + '\n'

        return output

    def format_results(self, results: Dict) -> str:
        """Format results in a tabular form that's easy to copy to Excel"""
        # Check if ring metrics were calculated
        calc_ring_metrics = getattr(self.args, 'calc_ring_metrics', False)

        # Header
        output = "Configuration\tCase\t"
        metrics = list(self.metricFuncs.keys())

        # Prediction metrics columns
        for metric in metrics:
            output += f"Pred_{metric}_Mean\tPred_{metric}_Std\t"

        # Regional metrics columns (if ring metrics enabled)
        if calc_ring_metrics:
            output += "Pred_MAE_WT_Mean\tPred_MAE_WT_Std\tPred_MAE_Ring_Mean\tPred_MAE_Ring_Std\t"
            output += "n_wt\tn_ring\t"

        # Input metrics columns (if input metrics were calculated)
        if self.calculate_input_metrics:
            for metric in metrics:
                output += f"Input_{metric}_Mean\tInput_{metric}_Std\t"
            # Input regional metrics (if ring metrics enabled)
            if calc_ring_metrics:
                output += "Input_MAE_WT_Mean\tInput_MAE_WT_Std\tInput_MAE_Ring_Mean\tInput_MAE_Ring_Std\t"

        output = output.rstrip('\t') + '\n'

        # Data rows
        for conf_num in sorted(results.keys()):
            for case_num in sorted(results[conf_num].keys()):
                if case_num == 'statistics':  # Skip the statistics entry
                    continue

                output += f"{conf_num}\t{case_num}\t"
                if results[conf_num][case_num] is not None:
                    # Prediction metrics
                    for metric in metrics:
                        if 'pred_mean' in results[conf_num][case_num]:
                            pred_mean = results[conf_num][case_num]['pred_mean'].get(metric, None)
                            pred_std = results[conf_num][case_num]['pred_std'].get(metric, None)
                            output += f"{pred_mean:.4f}\t{pred_std:.4f}\t" if pred_mean is not None else "\t\t"
                        else:
                            # Backward compatibility with old format
                            mean = results[conf_num][case_num].get('mean', {}).get(metric, None)
                            std = results[conf_num][case_num].get('std', {}).get(metric, None)
                            output += f"{mean:.4f}\t{std:.4f}\t" if mean is not None else "\t\t"

                    # Regional metrics (if ring metrics enabled)
                    if calc_ring_metrics and 'pred_mean' in results[conf_num][case_num]:
                        # MAE_WT
                        mae_wt_mean = results[conf_num][case_num]['pred_mean'].get('MAE_WT', None)
                        mae_wt_std = results[conf_num][case_num]['pred_std'].get('MAE_WT', None)
                        output += f"{mae_wt_mean:.4f}\t{mae_wt_std:.4f}\t" if mae_wt_mean is not None else "\t\t"
                        # MAE_Ring
                        mae_ring_mean = results[conf_num][case_num]['pred_mean'].get('MAE_Ring', None)
                        mae_ring_std = results[conf_num][case_num]['pred_std'].get('MAE_Ring', None)
                        output += f"{mae_ring_mean:.4f}\t{mae_ring_std:.4f}\t" if mae_ring_mean is not None else "\t\t"
                        # Voxel counts
                        n_wt = results[conf_num][case_num].get('total_n_wt', 0)
                        n_ring = results[conf_num][case_num].get('total_n_ring', 0)
                        output += f"{n_wt}\t{n_ring}\t"

                    # Input metrics (if available)
                    if self.calculate_input_metrics:
                        for metric in metrics:
                            if 'input_mean' in results[conf_num][case_num]:
                                input_mean = results[conf_num][case_num]['input_mean'].get(metric, None)
                                input_std = results[conf_num][case_num]['input_std'].get(metric, None)
                                output += f"{input_mean:.4f}\t{input_std:.4f}\t" if input_mean is not None else "\t\t"
                            else:
                                output += "\t\t"
                        # Input regional metrics (if ring metrics enabled)
                        if calc_ring_metrics and 'input_mean' in results[conf_num][case_num]:
                            input_mae_wt_mean = results[conf_num][case_num]['input_mean'].get('MAE_WT', None)
                            input_mae_wt_std = results[conf_num][case_num]['input_std'].get('MAE_WT', None)
                            output += f"{input_mae_wt_mean:.4f}\t{input_mae_wt_std:.4f}\t" if input_mae_wt_mean is not None else "\t\t"
                            input_mae_ring_mean = results[conf_num][case_num]['input_mean'].get('MAE_Ring', None)
                            input_mae_ring_std = results[conf_num][case_num]['input_std'].get('MAE_Ring', None)
                            output += f"{input_mae_ring_mean:.4f}\t{input_mae_ring_std:.4f}\t" if input_mae_ring_mean is not None else "\t\t"
                else:
                    # Fill with empty values
                    num_cols = len(metrics) * 2  # pred metrics
                    if calc_ring_metrics:
                        num_cols += 6  # MAE_WT, MAE_Ring (mean+std each) + n_wt + n_ring
                    if self.calculate_input_metrics:
                        num_cols += len(metrics) * 2  # input metrics
                        if calc_ring_metrics:
                            num_cols += 4  # input MAE_WT, MAE_Ring
                    output += '\t' * num_cols
                output = output.rstrip('\t') + '\n'

            # Add configuration averages
            output += f"{conf_num}\tAVG\t"

            # Prediction statistics
            for metric in metrics:
                key = f'pred_{metric}'
                if key in results[conf_num]['statistics']:
                    stats = results[conf_num]['statistics'][key]
                    mean = stats['mean']
                    std = stats['std']
                    if mean is not None and std is not None:
                        output += f"{mean:.4f}\t{std:.4f}\t"
                    else:
                        output += '\t\t'
                else:
                    # Backward compatibility
                    if metric in results[conf_num]['statistics']:
                        stats = results[conf_num]['statistics'][metric]
                        mean = stats['mean']
                        std = stats['std']
                        if mean is not None and std is not None:
                            output += f"{mean:.4f}\t{std:.4f}\t"
                        else:
                            output += '\t\t'
                    else:
                        output += '\t\t'

            # Regional metrics statistics (if ring metrics enabled)
            if calc_ring_metrics:
                # MAE_WT stats
                key = 'pred_MAE_WT'
                if key in results[conf_num]['statistics']:
                    stats = results[conf_num]['statistics'][key]
                    mean = stats['mean']
                    std = stats['std']
                    output += f"{mean:.4f}\t{std:.4f}\t" if mean is not None else "\t\t"
                else:
                    output += '\t\t'
                # MAE_Ring stats
                key = 'pred_MAE_Ring'
                if key in results[conf_num]['statistics']:
                    stats = results[conf_num]['statistics'][key]
                    mean = stats['mean']
                    std = stats['std']
                    output += f"{mean:.4f}\t{std:.4f}\t" if mean is not None else "\t\t"
                else:
                    output += '\t\t'
                # Total voxel counts
                n_wt_total = results[conf_num]['statistics'].get('total_n_wt', 0)
                n_ring_total = results[conf_num]['statistics'].get('total_n_ring', 0)
                output += f"{n_wt_total}\t{n_ring_total}\t"

            # Input statistics (if available)
            if self.calculate_input_metrics:
                for metric in metrics:
                    key = f'input_{metric}'
                    if key in results[conf_num]['statistics']:
                        stats = results[conf_num]['statistics'][key]
                        mean = stats['mean']
                        std = stats['std']
                        if mean is not None and std is not None:
                            output += f"{mean:.4f}\t{std:.4f}\t"
                        else:
                            output += '\t\t'
                    else:
                        output += '\t\t'
                # Input regional metrics stats (if ring metrics enabled)
                if calc_ring_metrics:
                    key = 'input_MAE_WT'
                    if key in results[conf_num]['statistics']:
                        stats = results[conf_num]['statistics'][key]
                        mean = stats['mean']
                        std = stats['std']
                        output += f"{mean:.4f}\t{std:.4f}\t" if mean is not None else "\t\t"
                    else:
                        output += '\t\t'
                    key = 'input_MAE_Ring'
                    if key in results[conf_num]['statistics']:
                        stats = results[conf_num]['statistics'][key]
                        mean = stats['mean']
                        std = stats['std']
                        output += f"{mean:.4f}\t{std:.4f}\t" if mean is not None else "\t\t"
                    else:
                        output += '\t\t'

            output = output.rstrip('\t') + '\n\n'

        return output

    def calculate(self):
        """Calculate metrics for all configurations and cases"""
        results = {}
        all_slice_results = {}  # Store individual slice results if requested

        # Check if specific slices are requested
        print_specific_slices = getattr(self.args, 'print_specific_slices', False)
        specific_slice_indices = getattr(self.args, 'specific_slice_indices', None)
        export_slice_metrics = getattr(self.args, 'export_slice_metrics', False)

        # Check if specific cases are requested
        specific_cases = getattr(self.args, 'specific_cases', None)
        if specific_cases is not None:
            if isinstance(specific_cases, (int, str)):
                specific_cases = [str(specific_cases)]
            else:
                specific_cases = [str(c) for c in specific_cases]
            print(f"Processing only specific cases: {specific_cases}")

        for conf_num in self.output:
            results[conf_num] = {}
            if export_slice_metrics:
                all_slice_results[conf_num] = {}
            print(f"\nProcessing configuration {conf_num}")

            for case_num in self.output[conf_num]:
                # Skip if specific cases are requested and this case is not in the list
                if specific_cases is not None and str(case_num) not in specific_cases:
                    print(f"Skipping case {case_num} (not in specific_cases list)")
                    continue
                print(f"Processing case {case_num}")

                try:
                    # Load prediction and target volumes
                    pred_tensor = self.load_volume(self.output[conf_num][case_num])
                    target_tensor = self.load_volume(self.targets[conf_num][case_num])

                    # Load input volume if input metrics are requested
                    input_tensor = None
                    if self.calculate_input_metrics and conf_num in self.inputs and case_num in self.inputs[conf_num]:
                        input_tensor = self.load_volume(self.inputs[conf_num][case_num])

                        print(
                            f"Loaded tensor shapes - Pred: {pred_tensor.shape}, Target: {target_tensor.shape}, Input: {input_tensor.shape}")
                    else:
                        print(f"Loaded tensor shapes - Pred: {pred_tensor.shape}, Target: {target_tensor.shape}")

                    # Load and apply mask if mask-based calculation is requested
                    mask_tensor = None
                    if self.calc_in_mask and conf_num in self.masks and case_num in self.masks[conf_num]:
                        mask_tensor = self.load_volume(self.masks[conf_num][case_num])
                        print(f"Loaded mask shape: {mask_tensor.shape}")
                        print(f"Mask unique values: {torch.unique(mask_tensor).cpu().numpy()}")
                        print(f"Using mask classes: {self.mask_classes}")

                    # Optional axis permutation (e.g. 3D volume to axial-first)
                    permute_pred = getattr(self.args, 'permute_pred', None)
                    if permute_pred is not None:
                        pred_tensor = pred_tensor.permute(*permute_pred)
                        print(f"Permuted pred axes {permute_pred} → {pred_tensor.shape}")

                    if self.args.use_npy and not getattr(self.args, 'skip_volume_alignment', False):
                        # 1) detect an H/W swap for prediction
                        Hp, Wp = pred_tensor.shape[1], pred_tensor.shape[2]
                        Ht, Wt = target_tensor.shape[1], target_tensor.shape[2]
                        if input_tensor is not None:
                            Hi, Wi = input_tensor.shape[1], input_tensor.shape[2]

                        if Hp == Wt and Wp == Ht:
                            # permute target so that (Ht,Wt)->(Hp,Wp)
                            target_tensor = target_tensor.permute(0, 2, 1)
                            if input_tensor is not None:
                                input_tensor = input_tensor.permute(0, 2, 1)
                                print(f"BRATS mode: permuted input from ({Hi},{Wi}) to {input_tensor.shape[1:]}")
                            print(f"BRATS mode: permuted target from ({Ht},{Wt}) to {target_tensor.shape[1:]}")

                        # 2) (optional) align slice counts
                        Sp, St = pred_tensor.shape[0], target_tensor.shape[0]
                        if input_tensor is not None:
                            Si = input_tensor.shape[0]
                        else:
                            Si = Sp

                        # Find minimum slice count across all tensors
                        min_slices = min(Sp, St, Si) if input_tensor is not None else min(Sp, St)

                        if Sp != St or (input_tensor is not None and Si != min_slices):
                            print(f"BRATS mode: truncating volumes to {min_slices} slices (pred:{Sp}, target:{St}, input:{Si if input_tensor is not None else 'N/A'})")
                            pred_tensor = pred_tensor[:min_slices]
                            target_tensor = target_tensor[:min_slices]
                            if input_tensor is not None:
                                input_tensor = input_tensor[:min_slices]

                        # Apply flip to target AND input to keep them aligned.
                        # FARD's pipeline is transpose+flip, so 256-frame preds need this
                        # flip; DnCNN is transpose-only (no flip) and sets brats_align_flip:false.
                        if getattr(self.args, 'brats_align_flip', True):
                            target_tensor = torch.flip(target_tensor, dims=[1])
                            if input_tensor is not None:
                                input_tensor = torch.flip(input_tensor, dims=[1])
                                print(f"BRATS mode: flipped target and input tensors along dim 1")

                    if input_tensor is not None:
                        print(
                            f"After alignment → Pred: {pred_tensor.shape}, Target: {target_tensor.shape}, Input: {input_tensor.shape}")
                    else:
                        print(f"After alignment → Pred: {pred_tensor.shape}, Target: {target_tensor.shape}")

                    if self.args.calc_3d:
                        # For 3D metrics, pass the whole volume
                        if self.transforms is not None:
                            data = {
                                'output': pred_tensor,  # [slices, H, W]
                                'target': target_tensor  # [slices, H, W]
                            }
                            transformed_data = self.transforms(data)
                            pred_tensor = transformed_data['output']
                            target_tensor = transformed_data['target']

                        # Add batch and channel dimensions for metrics
                        pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
                        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)

                        data_dict = {
                            'output': pred_tensor,
                            'target': target_tensor
                        }
                        metrics = self.calculate_metrics(data_dict)
                        results[conf_num][case_num] = metrics
                    else:
                        # Calculate metrics slice by slice
                        pred_slice_metrics = []
                        input_slice_metrics = [] if input_tensor is not None else None

                        # Apply transformations
                        if self.transforms is not None:
                            # Important: Save original target before any transforms
                            original_target_for_transforms = target_tensor.clone()

                            # Save original input before transforms (if exists)
                            original_input_for_transforms = input_tensor.clone() if input_tensor is not None else None

                            print("\n" + "="*80)
                            print("APPLYING TRANSFORMS TO ALL TENSORS")
                            print("="*80)
                            # Transform prediction, target, and input together
                            pred_data = {
                                'output': pred_tensor,
                                'target': target_tensor,
                            }
                            # Add input if available
                            if input_tensor is not None:
                                pred_data['input'] = input_tensor
                                print(f"Added input to transform pipeline: shape {input_tensor.shape}")
                            else:
                                print("No input tensor available - not adding to transform pipeline")

                            print(f"Keys in data dict before transforms: {list(pred_data.keys())}")
                            transformed_pred_data = self.transforms(pred_data)
                            pred_tensor = transformed_pred_data['output']
                            target_tensor = transformed_pred_data['target']
                            # Get transformed input if it was included
                            if 'input' in transformed_pred_data:
                                input_tensor = transformed_pred_data['input']

                            print(f"\nAfter transforms → Pred: {pred_tensor.shape}, Target: {target_tensor.shape}")
                            if input_tensor is not None:
                                print(f"                   Input: {input_tensor.shape}")

                            # Handle input tensor transformation (NOW HANDLED ABOVE IN MAIN TRANSFORM)
                            if False and original_input_for_transforms is not None:
                                try:
                                    print("\n" + "="*80)
                                    print("HANDLING INPUT TENSOR")
                                    print("="*80)

                                    # Check if input has the same shape as the original target
                                    # If so, it's already aligned and should NOT be transformed with geometric ops
                                    input_matches_target_shape = (
                                        original_input_for_transforms.shape[1:] == original_target_for_transforms.shape[1:]
                                    )

                                    if input_matches_target_shape:
                                        print(f"✓ Input shape {original_input_for_transforms.shape} matches target shape {original_target_for_transforms.shape}")
                                        print("  Input is already aligned with target - skipping geometric transforms on input")

                                        # Input already has same orientation as target, so we DON'T apply geometric transforms
                                        # Instead, we only apply transforms that affect BOTH output and target (like crop)
                                        # We do this by applying transforms to target and using the same crop on input

                                        # Apply full transform to get the final target
                                        target_only_data = {
                                            'output': original_target_for_transforms.clone(),  # Use target as dummy output
                                            'target': original_target_for_transforms.clone()
                                        }
                                        transformed_target_only = self.transforms(target_only_data)
                                        input_transformed_target = transformed_target_only['target']

                                        # Now manually apply only the crop transform to input based on target dimensions
                                        # The target tells us what the final dimensions should be
                                        input_tensor = original_input_for_transforms

                                        # Apply cropping to match final target dimensions
                                        if input_tensor.shape != input_transformed_target.shape:
                                            print(f"  Cropping input from {input_tensor.shape} to match target {input_transformed_target.shape}")

                                            # Center crop input to match transformed target dimensions
                                            for dim_idx in [1, 2]:  # Height and Width dimensions
                                                if input_tensor.shape[dim_idx] > input_transformed_target.shape[dim_idx]:
                                                    diff = input_tensor.shape[dim_idx] - input_transformed_target.shape[dim_idx]
                                                    start = diff // 2
                                                    end = start + input_transformed_target.shape[dim_idx]

                                                    if dim_idx == 1:
                                                        input_tensor = input_tensor[:, start:end, :]
                                                    else:  # dim_idx == 2
                                                        input_tensor = input_tensor[:, :, start:end]
                                                elif input_tensor.shape[dim_idx] < input_transformed_target.shape[dim_idx]:
                                                    print(f"ERROR: Input dim {dim_idx} ({input_tensor.shape[dim_idx]}) < target dim ({input_transformed_target.shape[dim_idx]})")
                                                    input_tensor = None
                                                    input_slice_metrics = None
                                                    break

                                        if input_tensor is not None:
                                            print(f"After spatial transforms → Input: {input_tensor.shape}")
                                    else:
                                        print(f"⚠️  Input shape {original_input_for_transforms.shape} differs from target shape {original_target_for_transforms.shape}")
                                        print("  Applying same transforms as prediction")

                                        input_data = {
                                            'output': original_input_for_transforms,
                                            'target': original_target_for_transforms.clone()
                                        }

                                        print(f"Before transforms → Input: {original_input_for_transforms.shape}")
                                        transformed_input_data = self.transforms(input_data)
                                        input_tensor = transformed_input_data['output']
                                        input_transformed_target = transformed_input_data['target']

                                        print(f"After transforms → Input: {input_tensor.shape}")

                                    # Verify both transforms produced the same target
                                    if not torch.allclose(target_tensor, input_transformed_target, rtol=1e-5, atol=1e-7):
                                        print("\n⚠️  WARNING: Targets don't match after transforming with pred vs input!")
                                        print(f"  Target from pred transform: shape={target_tensor.shape}, mean={target_tensor.mean():.6f}, std={target_tensor.std():.6f}")
                                        print(f"  Target from input transform: shape={input_transformed_target.shape}, mean={input_transformed_target.mean():.6f}, std={input_transformed_target.std():.6f}")
                                        print(f"  Max absolute difference: {(target_tensor - input_transformed_target).abs().max():.6f}")
                                    else:
                                        print("✓ Targets match after both transformations")

                                    # Verify shapes match
                                    if input_tensor.shape != pred_tensor.shape:
                                        print(f"\n⚠️  WARNING: Input shape {input_tensor.shape} doesn't match pred shape {pred_tensor.shape}!")
                                        print("This suggests the transforms were not applied consistently")
                                        input_tensor = None
                                        input_slice_metrics = None
                                    else:
                                        print(f"✓ Input and prediction shapes match: {input_tensor.shape}")

                                    print("="*80 + "\n")

                                except Exception as e:
                                    print(f"\n⚠️  ERROR: Failed to handle input tensor: {e}")
                                    print("Proceeding without input metrics for this case")
                                    import traceback
                                    traceback.print_exc()
                                    input_tensor = None
                                    input_slice_metrics = None

                        # *** Crop mask to match transformed target dimensions ***
                        if self.calc_in_mask and mask_tensor is not None:
                            original_mask_shape = mask_tensor.shape
                            # Crop mask to match target spatial dimensions
                            for dim_idx in [1, 2]:  # Height and Width dimensions
                                if mask_tensor.shape[dim_idx] > target_tensor.shape[dim_idx]:
                                    diff = mask_tensor.shape[dim_idx] - target_tensor.shape[dim_idx]
                                    start = diff // 2
                                    end = start + target_tensor.shape[dim_idx]
                                    if dim_idx == 1:
                                        mask_tensor = mask_tensor[:, start:end, :]
                                    else:  # dim_idx == 2
                                        mask_tensor = mask_tensor[:, :, start:end]
                            print(f"Cropped mask from {original_mask_shape} to {mask_tensor.shape}")

                        # *** Show middle slice preview AFTER applying all transforms ***
                        print("=" * 80)
                        print("MIDDLE SLICE PREVIEW AFTER TRANSFORMS (This is what will be compared)")
                        print("=" * 80)
                        self.show_middle_slice_preview(pred_tensor, target_tensor, conf_num, case_num, input_tensor)
                        print("=" * 80)

                        # Get number of slices AFTER transformations
                        # Use minimum to handle different slice counts after transpose
                        num_slices = min(pred_tensor.shape[0], target_tensor.shape[0])
                        print(f"Number of slices after transformations: {num_slices} (pred: {pred_tensor.shape[0]}, target: {target_tensor.shape[0]})")

                        start_slice = self.args.slices[0]
                        end_slice = num_slices - self.args.slices[1]

                        # Add bounds checking
                        start_slice = max(0, start_slice)
                        end_slice = min(end_slice, num_slices)

                        if start_slice >= end_slice:
                            raise ValueError(f"Invalid slice range: start ({start_slice}) >= end ({end_slice})")

                        print(f"Processing slices {start_slice} to {end_slice - 1} out of {num_slices} total slices")

                        # Determine which slices to process
                        if specific_slice_indices is not None:
                            # Process only specific slice indices (absolute slice numbers)
                            slice_indices_to_process = []
                            for abs_idx in specific_slice_indices:
                                if start_slice <= abs_idx < end_slice:
                                    relative_idx = abs_idx - start_slice
                                    slice_indices_to_process.append(relative_idx)
                                    print(f"Will process absolute slice {abs_idx} (relative index {relative_idx})")
                                else:
                                    print(
                                        f"Warning: Absolute slice index {abs_idx} is out of valid range [{start_slice}, {end_slice - 1}]")
                        else:
                            # Process all slices in range
                            slice_indices_to_process = list(range(end_slice - start_slice))

                        # Store slice results for export if requested
                        if export_slice_metrics:
                            all_slice_results[conf_num][case_num] = []

                        # Check if perilesional ring metrics are requested
                        calc_ring_metrics = getattr(self.args, 'calc_ring_metrics', False)
                        ring_dilation = getattr(self.args, 'ring_dilation', 3)

                        for relative_idx in slice_indices_to_process:
                            actual_slice_idx = start_slice + relative_idx

                            # Process prediction slice with added channel dimension
                            pred_slice = pred_tensor[actual_slice_idx].unsqueeze(0).unsqueeze(0)
                            target_slice = target_tensor[actual_slice_idx].unsqueeze(0).unsqueeze(0)

                            # Initialize metrics dict for this slice
                            pred_metrics = {}

                            # Apply mask if mask-based calculation is requested
                            if self.calc_in_mask and mask_tensor is not None:
                                mask_slice = mask_tensor[actual_slice_idx]

                                # *** Perilesional Ring Metrics ***
                                if calc_ring_metrics:
                                    # Get 2D slices without batch/channel dims for mask operations
                                    pred_2d = pred_tensor[actual_slice_idx]
                                    target_2d = target_tensor[actual_slice_idx]

                                    # Create WT mask (labels 1, 2, 3 only - excluding label 4 resection cavity)
                                    wt_mask = self.create_wt_mask(mask_slice)

                                    # Create brain mask (target != 0)
                                    brain_mask = self.create_brain_mask(target_2d)

                                    # Create ring mask (dilated_WT & brain_mask & ~WT)
                                    ring_mask = self.create_ring_mask(wt_mask, brain_mask, ring_dilation)

                                    # Calculate MAE in WT region
                                    mae_wt, n_wt = self.calculate_masked_mae(pred_2d, target_2d, wt_mask)

                                    # Calculate MAE in Ring region
                                    mae_ring, n_ring = self.calculate_masked_mae(pred_2d, target_2d, ring_mask)

                                    # Store regional metrics
                                    pred_metrics['MAE_WT'] = mae_wt
                                    pred_metrics['MAE_Ring'] = mae_ring
                                    pred_metrics['n_wt'] = n_wt
                                    pred_metrics['n_ring'] = n_ring

                                    # Also calculate overall metrics using standard mask (if requested)
                                    binary_mask = self.create_binary_mask(mask_slice, self.mask_classes)
                                    binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)
                                    pred_slice_masked = self.apply_mask_to_tensor(pred_slice, binary_mask)
                                    target_slice_masked = self.apply_mask_to_tensor(target_slice, binary_mask)

                                    pred_data_dict = {
                                        'output': pred_slice_masked,
                                        'target': target_slice_masked
                                    }
                                    standard_metrics = self.calculate_metrics(pred_data_dict)
                                    pred_metrics.update(standard_metrics)
                                else:
                                    # Standard mask-based calculation (no ring metrics)
                                    binary_mask = self.create_binary_mask(mask_slice, self.mask_classes)
                                    binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)

                                    # Apply mask to prediction and target
                                    pred_slice_masked = self.apply_mask_to_tensor(pred_slice, binary_mask)
                                    target_slice_masked = self.apply_mask_to_tensor(target_slice, binary_mask)

                                    # Calculate prediction → target metrics using masked slices
                                    pred_data_dict = {
                                        'output': pred_slice_masked,
                                        'target': target_slice_masked
                                    }
                                    pred_metrics = self.calculate_metrics(pred_data_dict)
                            else:
                                # Calculate prediction → target metrics without mask
                                pred_data_dict = {
                                    'output': pred_slice,
                                    'target': target_slice
                                }
                                pred_metrics = self.calculate_metrics(pred_data_dict)

                            # Calculate input → target metrics if input is available
                            input_metrics = None
                            input_slice = None
                            if input_tensor is not None:
                                input_slice = input_tensor[actual_slice_idx].unsqueeze(0).unsqueeze(0)

                                # Apply mask if mask-based calculation is requested
                                if self.calc_in_mask and mask_tensor is not None:
                                    if calc_ring_metrics:
                                        # Calculate input metrics for WT and Ring regions
                                        input_2d = input_tensor[actual_slice_idx]

                                        # Calculate MAE in WT region for input
                                        input_mae_wt, _ = self.calculate_masked_mae(input_2d, target_2d, wt_mask)

                                        # Calculate MAE in Ring region for input
                                        input_mae_ring, _ = self.calculate_masked_mae(input_2d, target_2d, ring_mask)

                                        input_metrics = {
                                            'MAE_WT': input_mae_wt,
                                            'MAE_Ring': input_mae_ring,
                                        }

                                        # Also calculate standard metrics
                                        input_slice_masked = self.apply_mask_to_tensor(input_slice, binary_mask)
                                        input_data_dict = {
                                            'output': input_slice_masked,
                                            'target': target_slice_masked
                                        }
                                        standard_input_metrics = self.calculate_metrics(input_data_dict)
                                        input_metrics.update(standard_input_metrics)
                                    else:
                                        input_slice_masked = self.apply_mask_to_tensor(input_slice, binary_mask)
                                        input_data_dict = {
                                            'output': input_slice_masked,
                                            'target': target_slice_masked
                                        }
                                        input_metrics = self.calculate_metrics(input_data_dict)
                                else:
                                    input_data_dict = {
                                        'output': input_slice,
                                        'target': target_slice
                                    }
                                    input_metrics = self.calculate_metrics(input_data_dict)

                            # Store metrics
                            if pred_metrics is not None:
                                pred_slice_metrics.append(pred_metrics)
                                if input_metrics is not None:
                                    input_slice_metrics.append(input_metrics)

                                # Store for export if requested
                                if export_slice_metrics:
                                    slice_result = {
                                        'slice_idx': actual_slice_idx,
                                        'pred_metrics': pred_metrics
                                    }
                                    if input_metrics is not None:
                                        slice_result['input_metrics'] = input_metrics
                                    all_slice_results[conf_num][case_num].append(slice_result)

                                # Save images for this slice only if print_specific_slices is enabled
                                if print_specific_slices:
                                    self.save_slice_images(
                                        pred_slice, target_slice,
                                        conf_num, case_num, actual_slice_idx,
                                        pred_metrics, input_slice, input_metrics
                                    )

                        # Print specific slice metrics if requested
                        if print_specific_slices and pred_slice_metrics:
                            if specific_slice_indices is not None:
                                # Convert back to absolute slice numbers for display
                                abs_slice_indices = [start_slice + rel_idx for rel_idx in slice_indices_to_process]
                                self.print_specific_slice_metrics(
                                    pred_slice_metrics, case_num, conf_num,
                                    abs_slice_indices, 0,  # start_slice is 0 since we're passing absolute indices
                                    input_slice_metrics
                                )
                            else:
                                # Print all processed slices
                                abs_slice_indices = [start_slice + rel_idx for rel_idx in
                                                     range(len(pred_slice_metrics))]
                                self.print_specific_slice_metrics(
                                    pred_slice_metrics, case_num, conf_num,
                                    abs_slice_indices, 0,  # start_slice is 0 since we're passing absolute indices
                                    input_slice_metrics
                                )

                        # Calculate mean and std across slices for prediction metrics, handling NaN values
                        avg_pred_metrics = {}
                        std_pred_metrics = {}

                        # Determine all metrics to aggregate (including regional metrics if calc_ring_metrics)
                        metrics_to_aggregate = [m for m in self.args.metrics if 'monai' not in m]
                        if calc_ring_metrics:
                            metrics_to_aggregate.extend(['MAE_WT', 'MAE_Ring'])

                        # Find slices valid for ALL metrics simultaneously
                        common_valid_pred_indices = []
                        for i, m in enumerate(pred_slice_metrics):
                            if m is None:
                                continue
                            all_valid = True
                            for metric in metrics_to_aggregate:
                                if metric not in m:
                                    all_valid = False
                                    break
                                value = m[metric]
                                if value is None:
                                    all_valid = False
                                    break
                                if isinstance(value, torch.Tensor):
                                    if torch.isnan(value).any() or torch.isinf(value).any():
                                        all_valid = False
                                        break
                                    value = value.item()
                                if np.isnan(value) or np.isinf(value):
                                    all_valid = False
                                    break
                            if all_valid:
                                common_valid_pred_indices.append(i)

                        n_common_pred = len(common_valid_pred_indices)
                        print(f"Case {case_num}: Using {n_common_pred}/{len(slice_indices_to_process)} common valid slices for pred metrics")

                        for metric in metrics_to_aggregate:
                            values = []
                            for i in common_valid_pred_indices:
                                m = pred_slice_metrics[i]
                                value = m[metric]
                                if isinstance(value, torch.Tensor):
                                    value = value.item()
                                values.append(value)

                            if values:
                                avg_pred_metrics[metric] = np.mean(values)
                                std_pred_metrics[metric] = np.std(values)
                                print(
                                    f"Case {case_num}, Pred {metric}: mean={avg_pred_metrics[metric]:.4f} ± {std_pred_metrics[metric]:.4f} "
                                    f"({len(values)}/{len(slice_indices_to_process)} valid slices)")
                            else:
                                print(f"Warning: No valid values for prediction {metric} in case {case_num}")
                                avg_pred_metrics[metric] = None
                                std_pred_metrics[metric] = None

                        # print(f"Case {case_num}: Pred valid slice indices: {common_valid_pred_indices}")

                        # Aggregate voxel counts if calc_ring_metrics is enabled
                        total_n_wt = 0
                        total_n_ring = 0
                        if calc_ring_metrics:
                            for m in pred_slice_metrics:
                                if m is not None:
                                    total_n_wt += m.get('n_wt', 0)
                                    total_n_ring += m.get('n_ring', 0)
                            print(f"Case {case_num}, Total voxel counts: n_wt={total_n_wt}, n_ring={total_n_ring}")

                        # Calculate mean and std across slices for input metrics if available
                        avg_input_metrics = {}
                        std_input_metrics = {}
                        if input_slice_metrics is not None:
                            # Find slices valid for ALL metrics simultaneously
                            common_valid_input_indices = []
                            for i, m in enumerate(input_slice_metrics):
                                if m is None:
                                    continue
                                all_valid = True
                                for metric in metrics_to_aggregate:
                                    if metric not in m:
                                        all_valid = False
                                        break
                                    value = m[metric]
                                    if value is None:
                                        all_valid = False
                                        break
                                    if isinstance(value, torch.Tensor):
                                        if torch.isnan(value).any() or torch.isinf(value).any():
                                            all_valid = False
                                            break
                                        value = value.item()
                                    if np.isnan(value) or np.isinf(value):
                                        all_valid = False
                                        break
                                if all_valid:
                                    common_valid_input_indices.append(i)

                            n_common_input = len(common_valid_input_indices)
                            print(f"Case {case_num}: Using {n_common_input}/{len(slice_indices_to_process)} common valid slices for input metrics")

                            for metric in metrics_to_aggregate:
                                values = []
                                for i in common_valid_input_indices:
                                    m = input_slice_metrics[i]
                                    value = m[metric]
                                    if isinstance(value, torch.Tensor):
                                        value = value.item()
                                    values.append(value)

                                if values:
                                    avg_input_metrics[metric] = np.mean(values)
                                    std_input_metrics[metric] = np.std(values)
                                    print(
                                        f"Case {case_num}, Input {metric}: mean={avg_input_metrics[metric]:.4f} ± {std_input_metrics[metric]:.4f} "
                                        f"({len(values)}/{len(slice_indices_to_process)} valid slices)")
                                else:
                                    print(f"Warning: No valid values for input {metric} in case {case_num}")
                                    avg_input_metrics[metric] = None
                                    std_input_metrics[metric] = None

                            # print(f"Case {case_num}: Input valid slice indices: {common_valid_input_indices}")

                        # Store results
                        case_results = {
                            'pred_mean': avg_pred_metrics,
                            'pred_std': std_pred_metrics,
                            'slices_used': f"{start_slice}-{end_slice}" if specific_slice_indices is None else f"absolute slices: {specific_slice_indices}"
                        }

                        # Add voxel counts if calc_ring_metrics is enabled
                        if calc_ring_metrics:
                            case_results['total_n_wt'] = total_n_wt
                            case_results['total_n_ring'] = total_n_ring

                        if input_slice_metrics is not None:
                            case_results['input_mean'] = avg_input_metrics
                            case_results['input_std'] = std_input_metrics

                        results[conf_num][case_num] = case_results

                except Exception as e:
                    print(f"Error processing case {case_num}: {str(e)}")
                    results[conf_num][case_num] = None

            # Calculate configuration averages and std
            print(f"\nConfiguration {conf_num} statistics:")
            config_stats = {}

            # Check if ring metrics were calculated
            calc_ring_metrics = getattr(self.args, 'calc_ring_metrics', False)

            # Determine all metrics to aggregate
            metrics_to_aggregate = [m for m in self.args.metrics if 'monai' not in m]
            if calc_ring_metrics:
                metrics_to_aggregate.extend(['MAE_WT', 'MAE_Ring'])

            # Calculate prediction metrics statistics
            for metric in metrics_to_aggregate:
                valid_values = []
                for case_num in results[conf_num]:
                    if case_num == 'statistics':
                        continue
                    if (results[conf_num][case_num] is not None and
                            isinstance(results[conf_num][case_num], dict) and
                            'pred_mean' in results[conf_num][case_num] and
                            metric in results[conf_num][case_num]['pred_mean'] and
                            results[conf_num][case_num]['pred_mean'][metric] is not None):
                        value = results[conf_num][case_num]['pred_mean'][metric]
                        if not (np.isnan(value) or np.isinf(value)):
                            valid_values.append(value)

                if valid_values:
                    mean_value = np.mean(valid_values)
                    std_value = np.std(valid_values)
                    config_stats[f'pred_{metric}'] = {
                        'mean': mean_value,
                        'std': std_value
                    }
                    print(f"Pred {metric}: {mean_value:.4f} ± {std_value:.4f} (across {len(valid_values)} cases)")
                else:
                    print(f"Warning: No valid values for prediction {metric} in configuration {conf_num}")
                    config_stats[f'pred_{metric}'] = {
                        'mean': None,
                        'std': None
                    }

            # Aggregate voxel counts if ring metrics were calculated
            if calc_ring_metrics:
                total_n_wt_all_cases = 0
                total_n_ring_all_cases = 0
                for case_num in results[conf_num]:
                    if case_num == 'statistics':
                        continue
                    if (results[conf_num][case_num] is not None and
                            isinstance(results[conf_num][case_num], dict)):
                        total_n_wt_all_cases += results[conf_num][case_num].get('total_n_wt', 0)
                        total_n_ring_all_cases += results[conf_num][case_num].get('total_n_ring', 0)
                config_stats['total_n_wt'] = total_n_wt_all_cases
                config_stats['total_n_ring'] = total_n_ring_all_cases
                print(f"Total voxel counts (all cases): n_wt={total_n_wt_all_cases}, n_ring={total_n_ring_all_cases}")

            # Calculate input metrics statistics if available
            if self.calculate_input_metrics:
                for metric in metrics_to_aggregate:
                    valid_values = []
                    for case_num in results[conf_num]:
                        if case_num == 'statistics':
                            continue
                        if (results[conf_num][case_num] is not None and
                                isinstance(results[conf_num][case_num], dict) and
                                'input_mean' in results[conf_num][case_num] and
                                metric in results[conf_num][case_num]['input_mean'] and
                                results[conf_num][case_num]['input_mean'][metric] is not None):
                            value = results[conf_num][case_num]['input_mean'][metric]
                            if not (np.isnan(value) or np.isinf(value)):
                                valid_values.append(value)

                    if valid_values:
                        mean_value = np.mean(valid_values)
                        std_value = np.std(valid_values)
                        config_stats[f'input_{metric}'] = {
                            'mean': mean_value,
                            'std': std_value
                        }
                        print(
                            f"Input {metric}: {mean_value:.4f} ± {std_value:.4f} (across {len(valid_values)} cases)")
                    else:
                        print(f"Warning: No valid values for input {metric} in configuration {conf_num}")
                        config_stats[f'input_{metric}'] = {
                            'mean': None,
                            'std': None
                        }

            results[conf_num]['statistics'] = config_stats

        print("\nResults (Tab-separated, copy-paste to Excel):")
        print("-" * 80)
        print(self.format_results(results))

        # Export specific slice metrics if requested
        if export_slice_metrics and all_slice_results:
            print("\nSpecific Slice Metrics (Tab-separated, copy-paste to Excel):")
            print("-" * 80)
            print(self.format_specific_slice_results_for_excel(all_slice_results))

        # Print image saving summary only if images were actually saved
        if self.save_images and print_specific_slices:
            print(f"\nImages saved to: {self.images_output_path}")

        return results