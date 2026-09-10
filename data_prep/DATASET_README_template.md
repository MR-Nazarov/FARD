# Prospectively accelerated brain MRI — complementary phase encoding

Paired accelerated and high-resolution brain MRI from 10 subjects, acquired on a
1.5T Philips Ingenia Ambition S. Supporting data for *Multi-Contrast MRI
Acceleration via Post-Reconstruction Fusion* (Medical Image Analysis, 2026).

Each subject was scanned with three routine contrasts — T1-weighted,
T2-weighted and T2-FLAIR — each accelerated by reducing the phase-encoding
matrix along a **different** orthogonal axis. The resulting volumes lose
resolution along different directions, so their spatial-frequency content is
complementary: what one contrast cannot resolve, another can. A matched
high-resolution acquisition of each contrast is included as reference.

This is what makes the dataset unusual. Most accelerated-MRI data is
retrospectively undersampled from fully sampled k-space, which does not
reproduce vendor reconstruction or real prospective acquisition. These scans
were **prospectively acquired at the accelerated settings** on a clinical
scanner and are released as vendor-reconstructed magnitude images.

## Contents

```
FARD_dataset/
├── README.md
├── LICENSE
└── subjects/
    ├── sub-01/
    │   ├── T1_HR/     T1_LR/     BICUBIC_T1_LR/
    │   ├── T2_HR/     T2_LR/     BICUBIC_T2_LR/
    │   └── FLAIR_HR/  FLAIR_LR/  BICUBIC_FLAIR_LR/
    ├── sub-02/
    ...
    └── sub-10/
```

10 subjects · 9 DICOM series each · 15,720 files · 6.3 GB.

| Series | What it is |
|---|---|
| `<contrast>_HR` | high-resolution reference acquisition |
| `<contrast>_LR` | prospectively accelerated acquisition |
| `BICUBIC_<contrast>_LR` | the `_LR` series, bicubic-upsampled onto the `_HR` grid |

Series names match those the code expects, so the pipeline runs on this download
without renaming anything. The `BICUBIC_*` series are derived from `_LR` by
interpolation. They are included
so the model inputs used in the paper are reproducible exactly rather than
approximately — the upsampling is part of the method, not a preprocessing
convenience.

Subject numbering is arbitrary and carries no clinical ordering.

## Acquisition

| Series | Slices | Matrix | In-plane (mm) | Slice (mm) | TR (ms) | TE (ms) | Flip |
|---|---|---|---|---|---|---|---|
| `T1_HR` | 180 | 384×384 | 0.625×0.625 | 1.00 | 7.46–7.55 | 3.39–3.43 | 8° |
| `T2_HR` | 180 | 720×720 | 0.333×0.333 | 1.00 | 3000 | 260 | 90° |
| `FLAIR_HR` | 197 | 384×384 | 0.625×0.625 | 1.00 | 4800 | 300 | 90° |
| `T1_LR` | 180 | 240×240 | 1.000×1.000 | 1.00 | 7.46–7.54 | 3.39–3.43 | 8° |
| `T2_LR` | 180 | 240×240 | 1.000×1.000 | 1.00 | 3000 | 260 | 90° |
| `FLAIR_LR` | 98 | 384×384 | 0.625×0.625 | 4.00 | 4800 | 300 | 90° |

The `BICUBIC_*` series match their `*_HR` counterpart's grid by construction.

Field strength 1.5T throughout. Intensity rescale slope and intercept are in the
standard `RescaleSlope` / `RescaleIntercept` elements (see de-identification
below).

## De-identification

Every file was de-identified before release, and every file was then re-read and
checked. The release contains no file that failed that check.

- DICOM PS3.15 Annex E basic profile applied via `dicognito`: patient name,
  identifier, accession number, referring physician, institution, address and
  station replaced with surrogate values.
- Study, Series and SOP Instance UIDs and the Frame of Reference UID
  regenerated. Remapping is consistent, so series and study relationships are
  preserved — slices of a series still share a `SeriesInstanceUID`.
- Study and series dates shifted consistently. Relative timing within a study is
  preserved; absolute dates are not real.
- All private tags removed. **Note:** on this scanner the intensity rescale
  slope and intercept were stored *only* in Philips private tags
  `(2005,140A)`/`(2005,1409)`; the standard elements were absent. They were
  promoted to `RescaleSlope`/`RescaleIntercept` before the private blocks were
  stripped, so pixel values remain convertible to real intensities.
- `DeviceSerialNumber` and other device and free-text identifiers removed.
- No patient attributes are retained: sex, age, weight, size, ethnic group and
  patient history were all removed. None is needed for image restoration, and
  with a cohort this small their combination would be a quasi-identifier.
  `PatientSex` is a Type 2 element, so it remains present but empty as the
  standard requires.

Pixel data is unmodified. Acquisition parameters (TR, TE, flip angle, pixel
spacing, slice thickness, field strength) are preserved.

The de-identification is reproducible: `data_prep/deidentify.py` in the code
repository below.

## Ethics

<!-- REPLACE the bracketed fields with the actual approval details. -->

This study was approved by the Institutional Review Board of [INSTITUTION]
(approval number [NUMBER], [DATE]). It was conducted in accordance with the
Declaration of Helsinki. [Written informed consent was obtained from all
participants / The requirement for informed consent was waived by the IRB],
including consent for the release of de-identified imaging data for research
use.

All data in this record has been de-identified as described above. Users must
not attempt to re-identify participants, and must not link this data to other
datasets for that purpose.

## Reproducing the paper

- Code: https://github.com/MR-Nazarov/FARD
- Model weights: https://huggingface.co/Lexer1/FARD

The registration pipeline that turns this raw data into the model's inputs is
`data_prep/` in the code repository: intra-contrast alignment of accelerated to
reference, inter-contrast alignment to an anchor, then anchor-to-MNI.

A config matching this dataset's layout ships with the code. Point
`FARD_DATASET` at wherever you extracted the download, then:

```bash
python -m data_prep.register_lr2hr \
    --config data_prep/configs/fard_dataset_lr2hr.yaml --stage A
python -m data_prep.register_lr2hr \
    --config data_prep/configs/fard_dataset_lr2hr.yaml --stage B
```

(`sheba_lr2hr.yaml` addresses the authors' internal tree, which is organised by
fold rather than as a flat subjects directory. Series names are the same in
both.)

## Citation

```bibtex
@article{nazarov2026multicontrast,
  title   = {Multi-Contrast MRI Acceleration via Post-Reconstruction Fusion},
  author  = {Nazarov, Alexander and Kiryati, Nahum and Roizen, Dani and
             Kerpel, Ariel and Hoffmann, Chen and Greenberg, Gahl and
             Mayer, Arnaldo},
  journal = {Medical Image Analysis},
  year    = {2026}
}
```

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
Non-commercial use, with attribution. This matches the license on the FARD model
weights.

https://creativecommons.org/licenses/by-nc/4.0/
