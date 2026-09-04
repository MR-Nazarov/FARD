# Writing a config

Every run is described by a YAML file under `confs/<project>/`, plus a twin under
`confs_test/<project>/` for inference. Two worked examples ship in
`confs/examples/`; this explains how to build others from them.

    python cli.py train --project examples --conf brats_fard_t1n
    python cli.py test  --project examples --conf brats_fard_t1n

A conf may be referred to by its full name (`brats_fard_t1n`) or, for the
historical `conf_<N>.yaml` style, by its number (`--conf 81`).

## How a config is assembled

Four layers, later ones winning:

1. `confs/defaults.yaml` — every field the trainer reads, with a default.
2. `inheritFrom:` — a test conf names its training conf and inherits everything
   not overridden. This is what keeps the pair in step.
3. The conf's own fields.
4. `--set key=value` on the command line.

Inference adds `confs/defaults_test.yaml` between 1 and 3.

> Overrides are applied *last*, after defaults are merged. The predecessor merged
> defaults afterwards, which silently discarded any override of an inherited key —
> `--set dontLog=true` did nothing at all on a conf using `inheritFrom`.

## The fields that decide an experiment

| Field | What it does |
|---|---|
| `modelName` / `cnnModule` | Class and module, resolved as `getattr(nets.<cnnModule>, <modelName>)`. The class must be imported in `nets/__init__.py`. |
| `jsonData` | Dataset index under `json_datasets/`. Its `base_dir` is remapped per machine via `local.yaml`. |
| `dataset` | Dataset class, resolved as `getattr(datasets, <dataset>)`. |
| `in_ch` / `mod` | Input channel count and the modality the network predicts. |
| `transforms` | MONAI transform names and arguments, applied in order. Names resolve against `monai.transforms` first, then `datasets/env_transforms.py`. |
| `loss` / `loss_coeff` | Loss terms and their weights, positionally matched. |
| `output_mode` | `clamp` (data normalised to [0,1]), `round` (raw DICOM intensities), or `none`. |
| `project_name` | The wandb project label. It no longer affects numerics, but see the caveat below. |

`project_name` deserves a caveat. In the predecessor it was matched as a string
to decide whether to clamp or round the model output, so renaming a project
silently changed the arithmetic — and the training and evaluation sites disagreed
with each other. That choice is now `output_mode`.

One such match remains, at `trainer.py`'s `dicom_results`:

```python
is_brats = 'BRATS' in self.args.project_name or self.args.use_npy
```

It selects the output *format* — `.npy` slices versus a DICOM series — not the
arithmetic. So a project whose name contains `BRATS` writes `.npy` even without
`use_npy`, which is why the BRATS example does not set that field. Set `use_npy`
explicitly on any new conf rather than relying on the name; the substring test is
kept only so existing confs keep working.

## BRATS — `confs/examples/brats_fard_t1n.yaml`

FARD predicting T1n from three bicubic-upsampled contrasts, in-plane degradation.

```yaml
modelName: FARD
cnnModule: FARD_verFull
in_ch: 3
mod: ['t1n_LR_4x']              # the contrast being predicted
jsonData: Ax_Brats_inplane_fold_1.json
dataset: denoising_dataset_monai

transforms:
  LoadImaged:      {keys: ['input','input1','input2','target'], ensure_channel_first: true}
  Transposed:      {keys: [...], indices: [0,2,1]}
  Flipd:           {keys: [...], spatial_axis: 0}
  ResizeWithPadOrCropd: {keys: [...], spatial_size: [240,240]}
  ConcatItemsd:    {keys: ['input','input1','input2'], name: 'input', dim: 0}

loss: ['vggL1','ssim','criterion']
loss_coeff: [1, 0.07, 0.85]
```

**To predict a different contrast**, change `mod` and reorder `ConcatItemsd` so
the target's own low-resolution input comes first. The channel order is the
convention the trained weights expect:

| Output | `ConcatItemsd` order |
|---|---|
| T1n | `input, input1, input2` |
| T2w | `input1, input2, input` |
| FLAIR | `input2, input, input1` |

Every key list — `LoadImaged`, `Transposed`, `Flipd`, `ResizeWithPadOrCropd` —
must name all three channels. Listing only two silently trains on a subset; this
caused a real mistake in the lesion-leakage baselines.

**To add input corruption** (the robustness sweeps), insert a transform from
`datasets/env_transforms.py` before `ConcatItemsd`, and set `no_clamp: true` so
the output is not clipped:

```yaml
  GhostingCorruptiond: {keys: ['input'], severity: 'mild'}
```

`GhostingCorruptiond`, `MotionCorruptiond` and `MisregistrationCorruptiond` all
take `keys` and `severity` (`mild` / `moderate` / `severe`).

## SHEBA — `confs/examples/sheba_3ch_t1.yaml`

Clinical DICOM rather than BRATS's `.npy` slices, so the pipeline differs in
three ways worth knowing.

```yaml
modelName: UpgradedDnCNN
cnnModule: DnCNN_deb
in_ch: 3
mod: ['BICUBIC_T1_LR']
jsonData: Saggital_regB_k_fold_1.json
```

1. **Input names are scan directories**, not contrast codes — `BICUBIC_T1_LR`,
   `T1_HR` — matching the layout `data_prep/` produces.
2. **No `Flipd` or `ResizeWithPadOrCropd`.** SHEBA confs load, transpose and
   concatenate only. Adding BRATS's geometry steps misaligns prediction from
   ground truth.
3. **Output is DICOM**, written via `myDicom`, so `output_mode: round` — the data
   is raw intensity, not normalised to [0,1].

The five folds are `Saggital_regB_k_fold_{1..5}.json`; the last digit of the old
`conf_44N` numbering was the fold.

> Do not score SHEBA with the trainer's built-in `write_quantitative`. The
> predictions come out at DICOM intensity (~[1250, 12000]) while the ground truth
> is normalised (~[0, 365]), so the metrics are meaningless — PSNR around −30. Use
> `evaluation/` with `normalize_for_metrics`, which min-max normalises both before
> comparing.

## Dataset indices

Confs name a `jsonData` index that is **not** committed — they are large derived
files listing every slice path, and they embed a `base_dir` from the machine that
built them. `local.yaml` remaps that root locally (copy `local.yaml.example`).
Regenerate an index with the generators under `data_prep/`, or copy one alongside.

The two examples reference `Ax_Brats_inplane_fold_1.json` and
`Saggital_regB_k_fold_1.json`; you will need those, or a `--set jsonData=...`
pointing at your own.

## The test twin

A test conf is short, because `inheritFrom` supplies the rest:

```yaml
inheritFrom: confs/examples/brats_fard_t1n.yaml
test: true
batchSize: 1
metrics: [SSIM, PSNR]
apply_inverse_transforms: true
use_npy: true        # BRATS writes .npy; omit for SHEBA, which writes DICOM
no_clamp: true       # only for the corruption sweeps
```

Checkpoints are looked up as `models/<project_name>/<conf>.pth`.

## Checklist for a new experiment

- [ ] Training conf in `confs/<project>/`, test twin in `confs_test/<project>/`
- [ ] `cnnModule` imported in `nets/__init__.py`
- [ ] Every transform key list names all input channels
- [ ] `ConcatItemsd` order matches the output contrast
- [ ] `output_mode` set deliberately, not inherited by accident
- [ ] `use_npy` set explicitly if the data is `.npy` — do not rely on
      `project_name` containing "BRATS", which is what the fallback tests
- [ ] Weights at `models/<project_name>/<conf>.pth`
- [ ] Smoke test first: `--set jsonData=<something>_smoke.json numEpochs=1`
