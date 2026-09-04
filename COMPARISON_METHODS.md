# Comparison methods

The baselines in Table 2 are **not vendored here**. Each was run from its authors'
own implementation, linked below, so that the comparison reflects the published
method rather than a re-implementation.

| Method | Table 2 class | Implementation | License |
|---|---|---|---|
| MTrans (Feng et al., 2023) | reference-guided | [chunmeifeng/MTrans](https://github.com/chunmeifeng/MTrans) | none declared |
| FSMNet (Chen et al., 2024) | reference-guided | [qic999/FSMNet](https://github.com/qic999/FSMNet) | Apache-2.0 |
| DCAMSR (Huang et al., 2023) | reference-guided, and LR-guided | [Solor-pikachu/DCAMSR](https://github.com/Solor-pikachu/DCAMSR) | none declared |
| EDSR+MMCHA (Georgescu et al., 2022) | accelerated inputs only | `nets/Edsr.py` — **included**, see below | MIT (EDSR backbone) |
| 3D U-Net (Cardoso et al., 2022) | accelerated inputs only | `monai.networks.nets.UNet`, wrapped in `nets/UNet3D.py` | Apache-2.0 |
| MMMD-Net (Hod et al., 2023) | accelerated inputs only | `nets/DnCNN_deb.py` — **included**, see below | — |

Licenses were checked against the GitHub API on 2026-09-03. MTrans and DCAMSR
publish no `LICENSE`, `COPYING` or `NOTICE` file, which under default copyright
means all rights reserved — so redistributing their code here was not an option
regardless of preference.

## What is included, and why

Two baselines are included because this work modified them. The modifications are
ours; the architectures are not, and each file names its origin in its header.

**MMMD-Net** — `nets/DnCNN.py` and `nets/DnCNN_deb.py`. The original
is a two-input, acceleration-aware image-domain model with directionally
elongated kernels; this work extends it to three contrasts. The `DnCNN-1` /
`DnCNN-2` / `DnCNN-3` columns of Figure 3 are its single-, two- and
three-contrast variants, which differ by input count rather than by architecture.

Two files, because the SHEBA ablations use two configurations of it: `DnCNN.py`
for the MNI ablation (Experiment A, cloned vs genuine channels — configs inherit
`cnnModule: DnCNN` from `confs/defaults.yaml`) and `DnCNN_deb.py`'s
`UpgradedDnCNN` for the reg-B ablation (Experiment B, three-contrast).

> Hod, G., Green, M., Waserman, M., Konen, E., Shrot, S., Nelkenbaum, I.,
> Kiryati, N., Mayer, A., 2023. Complementary phase encoding for pair-wise neural
> deblurring of accelerated brain MRI. In: Computer Vision – ECCV 2022 Workshops,
> Springer Nature Switzerland, Cham, pp. 268–280.

**EDSR+MMCHA** — `nets/Edsr.py`. Adapted to the post-reconstruction fusion
setting; the architecture is not designed for acceleration-induced
direction-dependent resolution loss, which is what makes it useful for separating
generic multimodal fusion capacity from acceleration-aware modelling.

> Georgescu, M.I., Ionescu, R.T., Miron, A.I., Savencu, O., Ristea, N.C., Verga,
> N., Khan, F.S., 2022. Multimodal multi-head convolutional attention with various
> kernel sizes for medical image super-resolution. arXiv:2204.04218.

built on the EDSR backbone (MIT):

> Lim, B., Son, S., Kim, H., Nah, S., Lee, K.M., 2017. Enhanced deep residual
> networks for single image super-resolution. CVPRW.
> https://github.com/sanghyun-son/EDSR-PyTorch

**3D U-Net** — `nets/UNet3D.py`. Twenty-five lines wrapping MONAI's `UNet` at the
paper's configuration (channels 32/64/128/256, strides 2, two residual units per
stage). MONAI is a pinned dependency, so nothing is vendored.

## Reproducing a comparison

Fetch the implementation from the table, drop the file into `nets/`, and add it to
`nets/__init__.py`. No edits are needed: `MainModel.config_model` inspects a
model's `__init__` signature and passes only the config keys it accepts, so the
pipeline adapts to the model rather than the reverse.

This applies to MTrans, FSMNet and DCAMSR, which were used as published. The two
adapted baselines are already here.
