# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""Register each case to the MNI atlas and resample every scan into atlas space.

Rewrite of ``data/register_to_MNI.py``. The registration itself -- Mattes mutual
information, Similarity3D initialised by MOMENTS, scale forced to 1 so the applied
transform is rigid, BSpline resampling onto an identity-direction grid at atlas
spacing -- is unchanged. What changed is the driver: paths, cases, scan names and
the high-quality/low-quality reference choice come from a config file instead of
literals in ``if __name__``.

Usage:
    python -m data_prep.register_to_mni --config data_prep/configs/sheba_mni_train.yaml
    python -m data_prep.register_to_mni --config <cfg> --folds 1 --cases 1 2
"""

import argparse
import os
import sys
from pathlib import Path

import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_dicom.my_dicom import myDicom  # noqa: E402

from data_prep.config import RegistrationConfig  # noqa: E402
from data_prep.dicom_io import (  # noqa: E402
    calculate_transformed_image_size,
    read_dicom_dir,
    sitk_write_dicom,
)

TRANSFORM_NAME = 'hq2mni_transform.tfm'


def fit_atlas_transform(atlas_img, moving_img, sampling_seed=42):
    """Fit the atlas transform, then strip its scale so only rotation+translation
    are applied.

    The scale strip (``GetParameters()[:-1] + (1,)``) is what makes the saved
    ``.tfm`` a rigid transform, which is why it can later be replayed on other
    volumes sharing the same anchor geometry.

    ``sampling_seed=None`` reproduces the original wall-clock seeding, which made
    every run of this registration produce a different transform.
    """
    initial_transform = sitk.CenteredTransformInitializer(
        atlas_img, moving_img,
        sitk.Similarity3DTransform(),  # isotropic scaling
        sitk.CenteredTransformInitializerFilter.MOMENTS)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    if sampling_seed is None:
        registration_method.SetMetricSamplingPercentage(0.01)  # seed = wall clock
    else:
        registration_method.SetMetricSamplingPercentage(0.01, sampling_seed)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(atlas_img, moving_img)

    # Avoid scaling (only a rigid transform is applied).
    final_transform.SetParameters(final_transform.GetParameters()[:-1] + (1,))

    print(f'  final metric value: {registration_method.GetMetricValue():.6f}')
    print(f'  stopping condition: '
          f'{registration_method.GetOptimizerStopConditionDescription()}')
    return final_transform


def _resample_and_write(dicom, in_dicom, out_dicom, transform, output_spacing,
                        output_direction, atlas_dir, output_origin=None,
                        output_size=None):
    """Resample one scan through ``transform`` and write it as a DICOM series."""
    moving_img = sitk.Cast(read_dicom_dir(in_dicom), sitk.sitkFloat32)

    if output_origin is None or output_size is None:
        output_origin, output_size = calculate_transformed_image_size(
            moving_img, transform, output_spacing)

    result_image = sitk.Resample(moving_img, output_size, transform, sitk.sitkBSpline,
                                 output_origin, output_spacing, output_direction)

    _, meta = dicom.readDicomSeriesWithMeta(in_dicom)
    series_uid = meta[0][0x20, 0xE].value
    series_desc = meta[0].SeriesDescription

    os.makedirs(out_dicom, exist_ok=True)
    sitk_write_dicom(result_image, series_uid, series_desc, out_dicom, atlas_dir)


def register_case(cfg: RegistrationConfig, fold: int, case: str) -> bool:
    """Register one case to the atlas. Returns False if skipped."""
    in_case_dir = cfg.input_dir(fold) / case
    ref_dir = in_case_dir / cfg.reference_scan

    if not ref_dir.is_dir():
        msg = (f'fold {fold} case {case}: reference scan not found: {ref_dir}')
        if cfg.skip_missing:
            print(f'  SKIP {msg}')
            return False
        raise FileNotFoundError(
            f'{msg}\nPass --skip-missing to skip cases with no reference scan.')

    out_case_dir = cfg.output_dir(fold) / case
    os.makedirs(out_case_dir, exist_ok=True)

    dicom = myDicom()
    atlas_dir = str(cfg.atlas)

    atlas_img = sitk.Cast(read_dicom_dir(atlas_dir), sitk.sitkFloat32)
    moving_img = sitk.Cast(read_dicom_dir(str(ref_dir)), sitk.sitkFloat32)

    final_transform = fit_atlas_transform(atlas_img, moving_img, cfg.sampling_seed)

    output_spacing = atlas_img.GetSpacing()
    output_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    output_origin, output_size = calculate_transformed_image_size(
        moving_img, final_transform, output_spacing)

    # The reference scan itself, then the transform, then every moving scan.
    _resample_and_write(dicom, str(ref_dir), str(out_case_dir / cfg.reference_scan),
                        final_transform, output_spacing, output_direction, atlas_dir,
                        output_origin, output_size)

    sitk.WriteTransform(final_transform, str(out_case_dir / TRANSFORM_NAME))

    for scan in cfg.moving_scans:
        in_dicom = in_case_dir / scan
        if not in_dicom.is_dir():
            msg = f'fold {fold} case {case}: moving scan not found: {in_dicom}'
            if cfg.skip_missing:
                print(f'  SKIP {msg}')
                continue
            raise FileNotFoundError(msg)
        print(f'  resampling {scan}')
        _resample_and_write(dicom, str(in_dicom), str(out_case_dir / scan),
                            final_transform, output_spacing, output_direction, atlas_dir)

    if cfg.anonymize:
        n = anonymize_tree(dicom, out_case_dir)
        print(f'  anonymized {n} dcm files in {out_case_dir}')

    return True


def anonymize_tree(dicom, root) -> int:
    """De-identify every ``.dcm`` under ``root`` in place. Returns the file count.

    The registration writer copies Patient Name/ID/Birth Date and Accession Number
    from the source series, so this is what makes an output tree shareable. Errors
    are raised, not counted -- a partially de-identified tree must not look like a
    success.
    """
    import pydicom

    count = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.lower().endswith('.dcm'):
                continue
            p = os.path.join(dirpath, f)
            ds = pydicom.dcmread(p, force=True)
            pydicom.dcmwrite(p, dicom.anonymize(ds))
            count += 1
    return count


def run(cfg: RegistrationConfig) -> None:
    done = skipped = 0
    for fold in cfg.folds:
        cases = cfg.case_list(fold)
        print(f'\nfold {fold}: {len(cases)} case(s) '
              f'{cfg.input_stage} -> {cfg.output_stage}')
        for case in cases:
            print(f'fold {fold} case {case}: fitting {cfg.reference_scan} to atlas')
            if register_case(cfg, fold, case):
                done += 1
            else:
                skipped += 1
    print(f'\nDONE  registered={done}  skipped={skipped}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True,
                        help='YAML config (see data_prep/configs/)')
    parser.add_argument('--folds', nargs='+', type=int,
                        help='override the folds in the config')
    parser.add_argument('--cases', nargs='+',
                        help='override the cases in the config')
    parser.add_argument('--output-stage', help='override the output stage directory')
    parser.add_argument('--output-root',
                        help='write the output stage under this root instead of the '
                             'dataset root (leaves the source dataset untouched)')
    parser.add_argument('--skip-missing', action='store_true',
                        help='skip cases/scans that are absent instead of raising')
    args = parser.parse_args(argv)

    cfg = RegistrationConfig.from_yaml(args.config)
    if args.folds:
        cfg.folds = args.folds
    if args.cases:
        cfg.cases = args.cases
    if args.output_stage:
        cfg.output_stage = args.output_stage
    if args.output_root:
        cfg.output_root = Path(args.output_root)
    if args.skip_missing:
        cfg.skip_missing = True

    run(cfg)


if __name__ == '__main__':
    main()
