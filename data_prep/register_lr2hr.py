# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""Stage A and Stage B of the LR/HR + cross-modal registration.

Rewrite of ``data/register_LR2HR.py``. Two things changed:

1. **Backend.** The original called SimpleElastix (``sitk.ElastixImageFilter``),
   which is no longer distributed, so the original cannot run at all. This uses
   ``itk-elastix``, the maintained binding for the same elastix engine. Verified
   against the stored data: replaying the stored
   ``T2_HR_to_T1_HR_rigid_body_tfm.txt`` through this code reproduces the stored
   ``registration_B`` output at corr 0.99998.
2. **Driver.** Paths, cases and modality names come from a config file rather
   than literals in ``if __name__``. The old driver was also stale -- it named
   transforms ``with_c_to_without_c_*`` (the two-modality contrast setup) while
   the data on disk uses ``{moving}_to_{fixed}_*`` across three modalities.

Stage A registers each modality's HQ scan onto its own bicubic-upsampled LQ scan.
Stage B registers the other modalities onto the reference modality.

Usage:
    python -m data_prep.register_lr2hr --config data_prep/configs/sheba_lr2hr.yaml --stage A
    python -m data_prep.register_lr2hr --config <cfg> --stage B --folds 1 --cases 1
"""

import argparse
import os
import sys
from pathlib import Path

import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_dicom.my_dicom import myDicom  # noqa: E402

from data_prep.config import LR2HRConfig  # noqa: E402
from data_prep.dicom_io import read_dicom_dir  # noqa: E402
from data_prep.elastix import (  # noqa: E402
    apply_transform,
    invert_transform,
    register_rigid,
    write_parameter_file,
)


def _write_dicom(dicom, img_sitk, out_dir, geometry_src, description):
    """Write a resampled volume as a DICOM series, taking geometry from
    ``geometry_src`` (the fixed series), as the original did."""
    os.makedirs(out_dir, exist_ok=True)
    dicom.writeDicomSeries(
        I=sitk.GetArrayFromImage(img_sitk).astype('float32'),
        source_path=str(geometry_src),
        output_path=str(out_dir),
        description=description)


def _series_description(dicom, path):
    _, meta = dicom.readDicomSeriesWithMeta(str(path))
    return meta[0].SeriesDescription


def stage_a(cfg: LR2HRConfig, fold: int, case: str) -> None:
    """Register each modality's HQ scan onto its own bicubic LQ scan."""
    dicom = myDicom()
    src = cfg.input_dir(fold) / case

    for mod in cfg.modalities:
        fixed_dir = src / cfg.lq_name.format(mod=mod)
        moving_dir = src / cfg.hq_name.format(mod=mod)
        if not (fixed_dir.is_dir() and moving_dir.is_dir()):
            msg = f'fold {fold} case {case} {mod}: missing {fixed_dir} or {moving_dir}'
            if cfg.skip_missing:
                print(f'  SKIP {msg}')
                continue
            raise FileNotFoundError(msg)

        out_case = cfg.stage_a_out_dir(fold, mod) / case
        os.makedirs(out_case, exist_ok=True)
        print(f'  [A] {mod}: {moving_dir.name} -> {fixed_dir.name}')

        fixed = sitk.Cast(read_dicom_dir(str(fixed_dir)), sitk.sitkFloat32)
        moving = sitk.Cast(read_dicom_dir(str(moving_dir)), sitk.sitkFloat32)

        resampled, tp = register_rigid(fixed, moving, cfg.stage_a_parameters)

        write_parameter_file(
            tp, out_case / f'{moving_dir.name}_to_{fixed_dir.name}_rigid_body_tfm.txt')
        _write_dicom(dicom, resampled, out_case / moving_dir.name, fixed_dir,
                     _series_description(dicom, moving_dir))
        # The LQ reference is carried forward unchanged so stage B can read both
        # from one place, matching the original's copy_tree step.
        _write_dicom(dicom, fixed, out_case / fixed_dir.name, fixed_dir,
                     _series_description(dicom, fixed_dir))


def stage_b(cfg: LR2HRConfig, fold: int, case: str) -> None:
    """Register the non-reference modalities onto the reference modality."""
    dicom = myDicom()
    ref = cfg.reference_modality
    ref_scan = cfg.hq_name.format(mod=ref) if cfg.hq_based else cfg.lq_name.format(mod=ref)

    fixed_dir = cfg.stage_a_in_dir(fold, ref) / case / ref_scan
    if not fixed_dir.is_dir():
        msg = f'fold {fold} case {case}: stage-A reference missing: {fixed_dir}'
        if cfg.skip_missing:
            print(f'  SKIP {msg}')
            return
        raise FileNotFoundError(msg)

    out_case = cfg.stage_b_dir(fold) / case
    os.makedirs(out_case, exist_ok=True)
    fixed = sitk.Cast(read_dicom_dir(str(fixed_dir)), sitk.sitkFloat32)

    # The reference modality is already in the target frame, so it is carried
    # across unregistered -- the original did this with copy_tree. Without it the
    # stage-B case directory is missing T1_HR / BICUBIC_T1_LR.
    for ref_own in [cfg.hq_name.format(mod=ref), cfg.lq_name.format(mod=ref)]:
        src_dir = cfg.stage_a_in_dir(fold, ref) / case / ref_own
        if not src_dir.is_dir():
            if cfg.skip_missing:
                print(f'  SKIP reference scan {src_dir}')
                continue
            raise FileNotFoundError(f'reference scan missing: {src_dir}')
        print(f'  [B] carrying {ref_own} across unregistered')
        _write_dicom(dicom, sitk.Cast(read_dicom_dir(str(src_dir)), sitk.sitkFloat32),
                     out_case / ref_own, src_dir,
                     _series_description(dicom, src_dir))

    for mod in cfg.modalities:
        if mod == ref:
            continue
        moving_scan = cfg.hq_name.format(mod=mod) if cfg.hq_based \
            else cfg.lq_name.format(mod=mod)
        moving_dir = cfg.stage_a_in_dir(fold, mod) / case / moving_scan
        if not moving_dir.is_dir():
            msg = f'fold {fold} case {case} {mod}: missing {moving_dir}'
            if cfg.skip_missing:
                print(f'  SKIP {msg}')
                continue
            raise FileNotFoundError(msg)

        print(f'  [B] {mod}: {moving_scan} -> {ref_scan}')
        moving = sitk.Cast(read_dicom_dir(str(moving_dir)), sitk.sitkFloat32)
        resampled, tp = register_rigid(fixed, moving, cfg.stage_b_parameters)

        stem = f'{moving_scan}_to_{ref_scan}_rigid_body'
        fwd = out_case / f'{stem}_tfm.txt'
        write_parameter_file(tp, fwd)
        _write_dicom(dicom, resampled, out_case / moving_scan, fixed_dir,
                     _series_description(dicom, moving_dir))

        # Inverse, for warping predictions back to each modality's own frame.
        write_parameter_file(invert_transform(moving, fwd),
                             out_case / f'{stem}_inv_tfm.txt')

        # The same transform is replayed onto this modality's companion scans.
        for companion in cfg.companion_scans(mod, moving_scan):
            comp_dir = cfg.stage_a_in_dir(fold, mod) / case / companion
            if not comp_dir.is_dir():
                if cfg.skip_missing:
                    print(f'    SKIP companion {comp_dir}')
                    continue
                raise FileNotFoundError(f'companion scan missing: {comp_dir}')
            print(f'    replaying onto {companion}')
            comp = sitk.Cast(read_dicom_dir(str(comp_dir)), sitk.sitkFloat32)
            _write_dicom(dicom, apply_transform(comp, tp),
                         out_case / companion, fixed_dir,
                         _series_description(dicom, comp_dir))


def run(cfg: LR2HRConfig, stage: str) -> None:
    fn = {'A': stage_a, 'B': stage_b}[stage]
    for fold in cfg.folds:
        cases = cfg.case_list(fold)
        print(f'\nfold {fold}: {len(cases)} case(s), stage {stage}')
        for case in cases:
            print(f'fold {fold} case {case}')
            fn(cfg, fold, case)
    print('\nDONE')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=['A', 'B'])
    parser.add_argument('--folds', nargs='+', type=int)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--output-root')
    parser.add_argument('--skip-missing', action='store_true')
    args = parser.parse_args(argv)

    cfg = LR2HRConfig.from_yaml(args.config)
    if args.folds:
        cfg.folds = args.folds
    if args.cases:
        cfg.cases = args.cases
    if args.output_root:
        cfg.output_root = Path(args.output_root)
    if args.skip_missing:
        cfg.skip_missing = True

    run(cfg, args.stage)


if __name__ == '__main__':
    main()
