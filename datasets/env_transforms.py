from __future__ import annotations

import hashlib
import monai
import numpy as np
from monai.transforms.transform import MapTransform
from monai.transforms import ConcatItemsd, RandScaleCrop, RandSpatialCrop, SpatialPad, CenterSpatialCrop
from  monai import  transforms
import munch
from collections.abc import Sequence
from monai.config import DtypeLike, KeysCollection


class Munchify(MapTransform):

    def __init__(self, keys='input', allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        return munch.Munch(d)


class Slice3DTo2D(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            volume = d[key]
            slices = [volume[:, :, : , i] for i in range(volume.shape[3])]
            d[key] = slices
        return d


class ConditionalCropOrPad(MapTransform):
    def __init__(self, keys, desired_size, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.desired_size = tuple(desired_size)  # Ensure desired_size is a tuple

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            current_size = image.shape
            if all(current_size[i] >= self.desired_size[i] for i in range(len(self.desired_size))):
                # If the image is larger than the desired size in all dimensions, crop it
                crop_transform = CenterSpatialCrop(roi_size=self.desired_size)
                d[key] = crop_transform(image)
            else:
                # If the image is smaller than the desired size in any dimension, pad it
                pad_transform = SpatialPad(spatial_size=self.desired_size, mode='constant')
                d[key] = pad_transform(image)
        return d


class ConcatItemsd1(ConcatItemsd):

    def __init__(self, keys: KeysCollection, name: str, dim: int = 0, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, name, dim, allow_missing_keys)

# class RandCropforMRI(MapTransform):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         roi_scale: Sequence[float] | float,
#         max_roi_scale: Sequence[float] | float | None = None,
#         random_center: bool = True,
#         random_size: bool = True,
#         allow_missing_keys: bool = False,
#         lazy: bool = False,
#     ) -> None:
#         cropper = RandScaleCrop(roi_scale, max_roi_scale, random_center, random_size, lazy=lazy)
#         super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)


class SliceWithMaxNumLabelsd(MapTransform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            _slice = data[key].shape[-1]
            d[key] = d[key][..., _slice - 1]
        return d


class JointCrop(MapTransform):
    """
    Apply a crop operation on multiple related images (specified by keys) based on the bounding
    box computed from a single significant image. This is useful in cases where multiple
    modalities or channels of an image need to be cropped identically based on the region of
    interest found in one of the modalities.

    Attributes:
        keys (list of str): List of keys from the input data dictionary that should be cropped.
        significant_key (str): Key in the data dictionary that is used to compute the bounding box.
        allow_missing_keys (bool): If True, missing keys in the input data do not raise an error.
    """

    def __init__(self, keys, significant_key, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.significant_key = significant_key

    def __call__(self, data):
        data = dict(data)
        transform = monai.transforms.CropForeground(allow_smaller=False)

        # Check if data[self.significant_key] is a batch
        if len(data[self.significant_key].shape) > 3:  # Assuming shape is [B, C, H, W] or similar
            batch_size = data[self.significant_key].shape[0]
            for i in range(batch_size):
                bb = transform.compute_bounding_box(img=data[self.significant_key][i])
                for key in self.keys:
                    if key in data:
                        data[key][i] = transform.crop_pad(data[key][i], *bb)
                    elif not self.allow_missing_keys:
                        raise KeyError(f"Key '{key}' missing from input data and 'allow_missing_keys' is False.")
        else:
            bb = transform.compute_bounding_box(img=data[self.significant_key])
            for key in self.keys:
                if key in data:
                    data[key] = transform.crop_pad(data[key], *bb)
                elif not self.allow_missing_keys:
                    raise KeyError(f"Key '{key}' missing from input data and 'allow_missing_keys' is False.")

        return data

# class JointCrop(MapTransform):
#     """
#     Apply a crop operation on multiple related images (specified by keys) based on the bounding
#     box computed from a single significant image. This is useful in cases where multiple
#     modalities or channels of an image need to be cropped identically based on the region of
#     interest found in one of the modalities.
#
#     Attributes:
#         keys (list of str): List of keys from the input data dictionary that should be cropped.
#         significant_key (str): Key in the data dictionary that is used to compute the bounding box.
#         allow_missing_keys (bool): If True, missing keys in the input data do not raise an error.
#     """
#
#     def __init__(self, keys, significant_key, allow_missing_keys=False):
#         super().__init__(keys, allow_missing_keys=allow_missing_keys)
#         self.significant_key = significant_key
#
#     def __call__(self, data):
#         data = dict(data)
#         transform = monai.transforms.CropForeground(allow_smaller=False)
#         bb = transform.compute_bounding_box(img=data[self.significant_key])
#         for key in self.keys:
#             if key in data:
#                 data[key] = transform.crop_pad(data[key], *bb)
#             elif not self.allow_missing_keys:
#                 raise KeyError(f"Key '{key}' missing from input data and 'allow_missing_keys' is False.")
#
#         return data



class SelectDim(MapTransform):

    def __init__(self, dim, index, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.dim = dim
        self.index = index

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._slice(d[key])
        return d

    def _slice(self, x):
        x = x.select(dim=self.dim,
                     index=self.index)
        return x


class ConvertToArray(MapTransform):

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = np.array([d[key]])
        return d


class ChangeKeys(MapTransform):

    def __init__(self, newKeys, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.newKeys = newKeys

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.newKeys):
            d[key] = d[self.keys[idx]]
            del d[self.keys[idx]]
        return d


class GhostingCorruptiond(MapTransform):
    """In-plane N/2 Nyquist ghosting stressor.

    Phase-shifts every odd k-space row by delta_phi radians, creating a ghost
    copy of the image shifted by H/2 pixels in the row direction.
    Ghost amplitude ≈ sin(delta_phi/2) relative to the primary image.

    Fully deterministic: same input tensor → same output.
    Inserted after ResizeWithPadOrCropd, before ConcatItemsd.
    Only the listed keys are modified; all other dict entries are untouched.
    """

    SEVERITY_TO_DPHI = {
        'mild':     0.30,   # sin(0.15) ≈ 14.9 % ghost
        'moderate': 0.70,   # sin(0.35) ≈ 34.3 % ghost
        'severe':   1.20,   # sin(0.60) ≈ 56.5 % ghost
    }

    def __init__(self, keys, severity='mild', allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        if severity not in self.SEVERITY_TO_DPHI:
            raise ValueError(f"GhostingCorruptiond: severity must be one of "
                             f"{list(self.SEVERITY_TO_DPHI)}, got '{severity}'")
        self.severity = severity
        self.delta_phi = self.SEVERITY_TO_DPHI[severity]
        self._phase = np.exp(1j * self.delta_phi)   # cached complex scalar

    def _apply_ghost(self, arr):
        """arr: (C, H, W) numpy float32 → same shape, ghosted."""
        out = np.empty_like(arr)
        for c in range(arr.shape[0]):
            k = np.fft.fft2(arr[c])
            k[1::2] *= self._phase      # phase-shift odd rows → N/2 ghost
            ghosted = np.abs(np.fft.ifft2(k)).astype(np.float32)
            # Clip to original data range so the model sees in-distribution values.
            # Real MRI ghosting stays bounded by scanner normalisation.
            out[c] = np.clip(ghosted, 0.0, arr[c].max())
        return out

    def __call__(self, data):
        import torch
        d = dict(data)
        for key in self.key_iterator(d):
            arr = d[key]
            if isinstance(arr, np.ndarray):
                d[key] = self._apply_ghost(arr.astype(np.float32))
            else:
                # torch.Tensor or MONAI MetaTensor — clone to preserve metadata
                out = arr.clone()
                np_in = arr.float().detach().cpu().numpy()
                np_out = self._apply_ghost(np_in)
                out.copy_(torch.from_numpy(np_out))
                d[key] = out
        return d


class MotionCorruptiond(MapTransform):
    """In-plane rigid-body motion stressor.

    Simulates bulk head motion between k-space shot groups: k-space is split
    into n_shots contiguous segments along the phase-encode axis; each segment
    receives a random in-plane translation applied as a phase ramp.

    Seed is derived deterministically from the file path stored in the MONAI
    meta dict (sha256 of path + severity), so re-runs are bit-identical and
    different slices / cases get independent but reproducible corruptions.
    """

    SEVERITY_PARAMS = {
        'mild':     dict(n_shots=4, max_shift_px=1.5),
        'moderate': dict(n_shots=6, max_shift_px=4.0),
        'severe':   dict(n_shots=8, max_shift_px=8.0),
    }

    def __init__(self, keys, severity='mild', allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        if severity not in self.SEVERITY_PARAMS:
            raise ValueError(f"MotionCorruptiond: severity must be one of "
                             f"{list(self.SEVERITY_PARAMS)}, got '{severity}'")
        self.severity = severity
        p = self.SEVERITY_PARAMS[severity]
        self.n_shots = p['n_shots']
        self.max_shift_px = p['max_shift_px']

    @staticmethod
    def _make_seed(filepath, severity):
        h = hashlib.sha256(f"{filepath}|{severity}|motion".encode()).hexdigest()
        return int(h[:16], 16) % (2 ** 31)

    def _corrupt(self, img_2d, seed):
        H, W = img_2d.shape
        rng = np.random.default_rng(seed)
        k = np.fft.fftshift(np.fft.fft2(img_2d.astype(np.float32)))

        fy = np.fft.fftfreq(H)[:, None]
        fx = np.fft.fftfreq(W)[None, :]
        phase_grid = np.fft.fftshift(
            np.exp(-2j * np.pi * (fy * 0 + fx * 0))   # template, updated per shot
        )

        lines_per_shot = max(1, H // self.n_shots)
        k_out = k.copy()
        for i in range(self.n_shots):
            dy = rng.uniform(-self.max_shift_px, self.max_shift_px)
            dx = rng.uniform(-self.max_shift_px, self.max_shift_px)
            ramp = np.fft.fftshift(np.exp(-2j * np.pi * (dy * fy + dx * fx)))
            y0 = i * lines_per_shot
            y1 = y0 + lines_per_shot if i < self.n_shots - 1 else H
            k_out[y0:y1] = k[y0:y1] * ramp[y0:y1]

        return np.abs(np.fft.ifft2(np.fft.ifftshift(k_out))).astype(np.float32)

    def _apply_motion(self, arr, filepath):
        seed = self._make_seed(str(filepath), self.severity)
        out = np.empty_like(arr)
        for c in range(arr.shape[0]):
            out[c] = self._corrupt(arr[c], seed + c)
        return out

    def __call__(self, data):
        import torch
        d = dict(data)
        for key in self.key_iterator(d):
            arr = d[key]
            meta_key = f'{key}_meta_dict'
            filepath = d.get(meta_key, {}).get('filename_or_obj', key)
            if isinstance(arr, np.ndarray):
                d[key] = self._apply_motion(arr.astype(np.float32), filepath)
            else:
                out = arr.clone()
                np_out = self._apply_motion(arr.float().detach().cpu().numpy(), filepath)
                out.copy_(torch.from_numpy(np_out))
                d[key] = out
        return d


class MisregistrationCorruptiond(MapTransform):
    """In-plane rigid MISREGISTRATION stressor (known, deterministic offset).

    Simulates a residual registration error on ONE input contrast: a KNOWN
    in-plane rigid transform (rotation about the slice centre + translation
    along one axis) is applied to the listed key(s) before ConcatItemsd. The
    other inputs and the target are untouched.

    Unlike MotionCorruptiond this is NOT random: the offset is fixed per
    severity, so it is identical for every case/slice and re-runs are
    bit-identical. Rotation is applied first (about the array centre), then the
    translation, both with bilinear interpolation (order=1) and zero fill.
    Inserted after ResizeWithPadOrCropd, before ConcatItemsd.
    """

    # severity -> (translation in px along axis 0, rotation in degrees about centre)
    SEVERITY_PARAMS = {
        'mild':     dict(shift_px=2.0, rot_deg=2.0),
        'moderate': dict(shift_px=4.0, rot_deg=3.0),
        'severe':   dict(shift_px=8.0, rot_deg=5.0),
    }

    def __init__(self, keys, severity='mild', allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        if severity not in self.SEVERITY_PARAMS:
            raise ValueError(f"MisregistrationCorruptiond: severity must be one of "
                             f"{list(self.SEVERITY_PARAMS)}, got '{severity}'")
        self.severity = severity
        p = self.SEVERITY_PARAMS[severity]
        self.shift_px = p['shift_px']
        self.rot_deg = p['rot_deg']

    def _misreg_2d(self, img_2d):
        """img_2d: (H, W) float32 -> same shape, rotated about centre then shifted."""
        from scipy import ndimage
        out = ndimage.rotate(img_2d.astype(np.float32), self.rot_deg,
                             reshape=False, order=1, mode='constant', cval=0.0)
        out = ndimage.shift(out, (self.shift_px, 0.0),
                            order=1, mode='constant', cval=0.0)
        return out.astype(np.float32)

    def _apply_misreg(self, arr):
        """arr: (C, H, W) numpy float32 -> same shape, misregistered per channel."""
        out = np.empty_like(arr)
        for c in range(arr.shape[0]):
            out[c] = self._misreg_2d(arr[c])
        return out

    def __call__(self, data):
        import torch
        d = dict(data)
        for key in self.key_iterator(d):
            arr = d[key]
            if isinstance(arr, np.ndarray):
                d[key] = self._apply_misreg(arr.astype(np.float32))
            else:
                out = arr.clone()
                np_out = self._apply_misreg(arr.float().detach().cpu().numpy())
                out.copy_(torch.from_numpy(np_out))
                d[key] = out
        return d
