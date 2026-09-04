# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""Rigid registration via elastix, on top of ``itk-elastix``.

The original pipeline used SimpleElastix (``sitk.ElastixImageFilter`` /
``sitk.TransformixImageFilter``), a separate SimpleITK build that is no longer
distributed -- neither the pinned ``simpleitk==2.5.2`` nor any current release
provides it, so ``data/register_LR2HR.py`` cannot run at all.

``itk-elastix`` wraps the same elastix engine and is maintained, so the
registration method is unchanged; only the Python binding differs. The parameter
files it writes are format-compatible with the stored
``*_rigid_body_tfm.txt`` files, so existing transforms can still be replayed.

DICOM IO stays on SimpleITK, as before; images are converted at the elastix
boundary only.
"""

import numpy as np
import itk
import SimpleITK as sitk


def sitk_to_itk(img_sitk):
    """SimpleITK image -> ITK image, carrying spacing, origin and direction."""
    out = itk.GetImageFromArray(
        sitk.GetArrayFromImage(img_sitk).astype(np.float32))
    out.SetSpacing([float(x) for x in img_sitk.GetSpacing()])
    out.SetOrigin([float(x) for x in img_sitk.GetOrigin()])
    out.SetDirection(
        itk.matrix_from_array(np.array(img_sitk.GetDirection()).reshape(3, 3)))
    return out


def itk_to_sitk(img_itk):
    """ITK image -> SimpleITK image, carrying spacing, origin and direction."""
    out = sitk.GetImageFromArray(
        itk.GetArrayFromImage(img_itk).astype(np.float32))
    out.SetSpacing([float(x) for x in img_itk.GetSpacing()])
    out.SetOrigin([float(x) for x in img_itk.GetOrigin()])
    out.SetDirection(
        itk.array_from_matrix(img_itk.GetDirection()).flatten().tolist())
    return out


def rigid_parameter_object(overrides=None):
    """elastix's stock rigid parameter map, as the original used via
    ``sitk.GetDefaultParameterMap('rigid')``, with optional overrides.

    Stage A of the original pipeline did not use the stock map unmodified: its
    stored parameter files set ``MaximumNumberOfIterations`` to 512 rather than
    elastix's default 256. Stage B did use the stock map. Pass ``overrides`` to
    reproduce a stage's own settings.
    """
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap('rigid')
    for key, value in (overrides or {}).items():
        pm[key] = [str(value)]
    po.AddParameterMap(pm)
    return po


def register_rigid(fixed_sitk, moving_sitk, overrides=None, log_to_console=False):
    """Rigidly register ``moving_sitk`` onto ``fixed_sitk``.

    Returns ``(resampled_moving_sitk, transform_parameter_object)``.
    """
    reg = itk.ElastixRegistrationMethod.New(
        sitk_to_itk(fixed_sitk), sitk_to_itk(moving_sitk))
    reg.SetParameterObject(rigid_parameter_object(overrides))
    reg.SetLogToConsole(log_to_console)
    reg.Update()
    return itk_to_sitk(reg.GetOutput()), reg.GetTransformParameterObject()


def apply_transform(moving_sitk, transform_parameter_object):
    """Resample ``moving_sitk`` through an existing transform (transformix)."""
    return itk_to_sitk(
        itk.transformix_filter(sitk_to_itk(moving_sitk),
                               transform_parameter_object))


def invert_transform(moving_sitk, forward_tfm_path, log_to_console=False):
    """Compute the inverse of a saved forward transform.

    Reproduces the original's trick: register the moving image to *itself*
    starting from the forward transform, with a DisplacementMagnitudePenalty
    metric and composed transforms, so the optimiser solves for the displacement
    that undoes it. The resulting map's ``InitialTransformParametersFileName`` is
    cleared so the inverse stands alone.
    """
    po = rigid_parameter_object()
    po.SetParameter('HowToCombineTransforms', 'Compose')
    po.SetParameter('Metric', 'DisplacementMagnitudePenalty')

    moving_itk = sitk_to_itk(moving_sitk)
    reg = itk.ElastixRegistrationMethod.New(moving_itk, moving_itk)
    reg.SetParameterObject(po)
    reg.SetInitialTransformParameterFileName(str(forward_tfm_path))
    reg.SetLogToConsole(log_to_console)
    reg.Update()

    inverse = reg.GetTransformParameterObject()
    inverse.SetParameter(0, 'InitialTransformParametersFileName',
                         'NoInitialTransform')
    return inverse


def write_parameter_file(transform_parameter_object, path, index=0):
    """Write one parameter map to ``path`` in elastix's text format."""
    itk.ParameterObject.WriteParameterFile(
        transform_parameter_object.GetParameterMap(index), str(path))


def read_parameter_file(path):
    """Read an elastix parameter file (including the stored ones) back in."""
    po = itk.ParameterObject.New()
    po.AddParameterFile(str(path))
    return po
