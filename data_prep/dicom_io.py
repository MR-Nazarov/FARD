# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""DICOM read/write helpers for the registration pipeline.

Lifted verbatim from ``data/register_to_MNI.py`` so the numerics and the written
DICOM tags are unchanged. The only edits are import paths and docstrings; no
behaviour in these functions differs from the originals.
"""

import os
import time

import numpy as np
import SimpleITK as sitk
from pydicom.uid import generate_uid


def get_dicom_rescale(img_dicom):
    try:
        scale = img_dicom[0x2005, 0x140A].value
        intercept = img_dicom[0x2005, 0x1409].value
    except KeyError:
        scale = img_dicom.RescaleSlope
        intercept = img_dicom.RescaleIntercept
        print('WARNING: unable to extract scale / intercept from tags')
    return scale, intercept


def read_dicom_dir(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)

    reader.SetFileNames(dicom_names)
    return reader.Execute()


def sitk_write_dicom(img_sitk, series_uid, series_desc, out_dicom_dir, ref_dicom_dir):
    """Write ``img_sitk`` as an axial DICOM series, plus a sagittal series when the
    grid is isotropic and axis-aligned.

    NOTE: ``tags_to_copy`` below deliberately carries Patient Name, Patient ID,
    Patient Birth Date and Accession Number over from the reference series, so the
    output of this function is NOT de-identified. The pipeline's ``anonymize``
    stage (see ``register_to_mni.py``) is what produces the shareable
    ``*_anonymized`` trees.
    """

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(ref_dicom_dir)
    if not series_IDs:
        raise FileNotFoundError(
            f'reference directory contains no DICOM series: {ref_dicom_dir}')
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        ref_dicom_dir, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    # Configure the reader to load all of the DICOM tags (public+private).
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    series_reader.Execute()

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame-of-reference information from the meta-data
    # dictionary rather than the automatically generated information from file IO.
    writer.KeepOriginalImageUIDOn()

    tags_to_copy = ["0010|0010",  # Patient Name
                    "0010|0020",  # Patient ID
                    "0008|0050",  # Accession Number
                    "0010|0030",  # Patient Birth Date
                    "0020|000D",  # Study Instance UID, for machine consumption
                    "0020|0010",  # Study ID, for human consumption
                    "0008|0020",  # Study Date
                    "0008|0030",  # Study Time
                    "0008|0060",  # Modality
                    "0028|0030",  # Pixel Spacing
                    "0028|0100",  # Bits Allocated
                    "0028|0101",  # Bits Stored
                    "0028|0102",  # High Bit
                    "0028|0103",  # Pixel Representation
                    "0018|0050",  # Slice Thickness
                    "0020|0011",  # Series Number
                    "0020|0020",  # Patient Orientation
                    "0020|000e",  # Series Instance UID
                    "0020|0052",  # Frame of Reference UID
                    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    direction = img_sitk.GetDirection()
    series_tag_values = [(k, series_reader.GetMetaData(0, k)) for k in tags_to_copy if
                         series_reader.HasMetaDataKey(0, k)] + \
                        [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                         ("0020|0037",
                          '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                              direction[1], direction[4], direction[7])))),
                         ("0008|103e", series_desc + '_reg_to_atlas'),
                         ("0018|0088", "0.5")]  # Spacing between slices

    num_slices = img_sitk.GetDepth()
    out_dicom_dir_axial = out_dicom_dir + '_axial'
    os.makedirs(out_dicom_dir_axial, exist_ok=True)
    for i in range(num_slices):
        image_slice = img_sitk[:, :, i]
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)

        image_slice.SetMetaData("0020|000D", "")
        image_slice.SetMetaData("0020|000E", series_uid)
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, img_sitk.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i + 1))  # Instance Number

        writer.SetFileName(os.path.join(out_dicom_dir_axial, f'MNIICBM152HR_{i + 1:04}.dcm'))
        writer.Execute(image_slice)

    _write_sagittal(img_sitk, series_desc, out_dicom_dir, writer,
                    series_tag_values, direction)


def _write_sagittal(img_sitk, series_desc, out_dicom_dir, writer,
                    series_tag_values, direction):
    """Sagittal reformat, written only for an isotropic, axis-aligned grid.

    Kept as a separate function purely for readability; the guard and the body are
    unchanged from the original inline block.
    """
    isotropic = (img_sitk.GetSpacing()[0] == img_sitk.GetSpacing()[1]
                 and img_sitk.GetSpacing()[0] == img_sitk.GetSpacing()[-1])
    if direction != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) or not isotropic:
        return

    img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # (D, H, W)
    sagittal_img = np.zeros((img.shape[2], img.shape[0], img.shape[1]), dtype=np.float32)
    for slice_num in range(img.shape[2]):
        sagittal_img[slice_num, :, :] = img[:, :, slice_num]
    sagittal_img = np.flip(sagittal_img, axis=1)
    sagittal_img = sitk.GetImageFromArray(sagittal_img)
    sagittal_img.SetSpacing(img_sitk.GetSpacing())

    # sagittal_img.Get* return defaults, so the geometry tags are set explicitly.
    sagittal_direction = [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]

    series_uid = generate_uid()
    series_number = int(np.round(np.abs(np.random.randn(1)[0] * 1234567)))
    os.makedirs(out_dicom_dir, exist_ok=True)
    for i in range(sagittal_img.GetDepth()):
        image_slice = sagittal_img[:, :, i]
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)

        image_slice.SetMetaData("0020|0037", '\\'.join(
            map(str, (sagittal_direction[0], sagittal_direction[3], sagittal_direction[6],
                      sagittal_direction[1], sagittal_direction[4], sagittal_direction[7]))))

        image_slice.SetMetaData("0028|0010", str(image_slice.GetHeight()))  # rows
        image_slice.SetMetaData("0028|0011", str(image_slice.GetWidth()))   # cols
        image_slice.SetMetaData("0028|0030",
                                '\\'.join(map(str, img_sitk.GetSpacing()[:-1])))  # Pixel Spacing
        image_slice.SetMetaData("0018|0050", str(img_sitk.GetSpacing()[-1]))      # Slice Thickness
        image_slice.SetMetaData("0008|103e", series_desc + '_reg_to_atlas_' + '_sagittal')

        image_slice.SetMetaData("0020|000D", "")
        image_slice.SetMetaData("0020|000E", series_uid)
        image_slice.SetMetaData("0020|0011", str(series_number))
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, img_sitk.TransformIndexToPhysicalPoint((0, 0, i)))))
        image_slice.SetMetaData("0020|0013", str(i + 1))

        writer.SetFileName(os.path.join(out_dicom_dir, f'I_{i + 1:04}.dcm'))
        writer.Execute(image_slice)


def calculate_transformed_image_size(moving_img, transform, output_spacing):
    """Bounding box of ``moving_img`` after ``transform``, on a grid of
    ``output_spacing`` with an identity direction cosine matrix."""
    extreme_points = _corner_points(moving_img)

    inv_final_transform = transform.GetInverse()
    extreme_points_transformed = [inv_final_transform.TransformPoint(p) for p in extreme_points]

    min_x = min(extreme_points_transformed)[0]
    min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
    min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]
    max_x = max(extreme_points_transformed)[0]
    max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
    max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

    output_origin = [min_x, min_y, min_z]
    output_size = [int((max_x - min_x) / output_spacing[0]),
                   int((max_y - min_y) / output_spacing[1]),
                   int((max_z - min_z) / output_spacing[2])]

    return output_origin, output_size


def _corner_points(img):
    """The eight index-space corners of ``img`` in physical coordinates.

    Note this uses ``GetWidth()/GetHeight()/GetDepth()`` rather than size-1, which
    is what the original code did; kept as-is so the computed bounding box is
    identical.
    """
    w, h, d = img.GetWidth(), img.GetHeight(), img.GetDepth()
    return [img.TransformIndexToPhysicalPoint(p) for p in
            [(0, 0, 0), (w, 0, 0), (0, 0, d), (w, h, 0),
             (w, 0, d), (0, h, 0), (0, h, d), (w, h, d)]]
