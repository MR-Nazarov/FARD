# Prospectively accelerated brain MRI — complementary phase encoding

Paired accelerated and high-resolution brain MRI acquired on a 1.5T Philips
Ingenia Ambition S, supporting *Multi-Contrast MRI Acceleration via
Post-Reconstruction Fusion* (Medical Image Analysis, 2026).

Each subject was scanned with three routine contrasts — T1-weighted, T2-weighted
and T2-FLAIR — each accelerated by reducing the phase-encoding matrix along a
**different** orthogonal axis. The resulting volumes lose resolution along
different directions, so their spatial-frequency content is complementary. A
matched high-resolution acquisition of each contrast is included as reference.

## Contents

```
<case>/
  T1_HR/  T2_HR/  FLAIR_HR/                 high-resolution reference
  T1_LR/  T2_LR/  FLAIR_LR/                 accelerated acquisition
  BICUBIC_T1_LR/ BICUBIC_T2_LR/ BICUBIC_FLAIR_LR/
                                            accelerated, bicubic-upsampled to
                                            the reference grid
```

DICOM throughout. The bicubic series are derived from the `*_LR` series and are
included so the model inputs used in the paper are reproducible exactly.

TODO: subjects, total size, per-contrast matrix sizes, and the acquisition
parameter table (TR/TE/flip angle/FOV per contrast).

## De-identification

Every file was de-identified before release:

- DICOM PS3.15 Annex E basic profile applied via `dicognito`, replacing patient
  name, identifier, accession number, referring physician, institution and
  station with surrogate values.
- Study, series and SOP Instance UIDs, and the Frame of Reference UID, were
  regenerated. Remapping is consistent, so series and study relationships are
  preserved.
- Study and series dates were shifted consistently; relative timing within a
  study is preserved, absolute dates are not real.
- All private tags removed. **Note:** on this scanner the intensity rescale
  slope and intercept are stored only in Philips private tags
  `(2005,140A)`/`(2005,1409)`. They were promoted to the standard
  `RescaleSlope`/`RescaleIntercept` before the private blocks were stripped, so
  pixel values remain convertible to real intensities.
- `DeviceSerialNumber` and other device and free-text identifiers removed.
- Every written file was re-read and checked for residual identifiers; the
  release contains no file that failed that check.

Pixel data is unmodified. Acquisition parameters (TR, TE, flip angle, pixel
spacing, slice thickness, field strength) are preserved.

No patient attributes are retained. Sex, age, weight, size, ethnic group and
patient history were removed — none is needed for image restoration, and with a
cohort this small their combination would be a quasi-identifier. `PatientSex` is
a Type 2 element, so it remains present but empty as the standard requires.

TODO: state the ethics approval and the consent basis for public release.

## Reproducing the paper

Code: https://github.com/MR-Nazarov/FARD
Weights: https://huggingface.co/Lexer1/FARD

The registration pipeline that turns this raw data into the model's inputs is
`data_prep/` in the code repository — intra-contrast alignment, inter-contrast
alignment to an anchor, then anchor-to-MNI.

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

TODO: choose. CC BY-NC 4.0 matches the FARD weights; CC BY 4.0 is more common for
Zenodo datasets and more permissive for secondary research.
