# FARD — Multi-Contrast MRI Acceleration via Post-Reconstruction Fusion

Reference implementation of **FARD** (Frequency Attention Residual Denoising), a
lightweight reference-free multi-contrast fusion network that refines
vendor-reconstructed magnitude images from complementary orthogonal
phase-encoding acquisitions — no raw k-space, no high-resolution reference
contrast.

This repository covers training, inference, metric evaluation, the three-stage
registration pipeline, and the robustness experiments (motion, ghosting,
misregistration, lesion-leakage).

Two datasets are supported. **BRATS** (public) is stored as `.npy` slices;
**SHEBA** is clinical DICOM and is not redistributable, so the SHEBA path is
provided as code and configuration only.

## Install

```bash
conda create -n fard python=3.11
conda activate fard
pip install -r requirements.txt
```

Verified against Python 3.11.11, PyTorch 2.9.1, CUDA 12.8, on an RTX 3090.
A GPU is needed for training and inference; the data-preparation and scoring
paths run on CPU.

## Point it at your data

Dataset indices (`json_datasets/*.json`) list every slice path and embed the
`base_dir` of the machine that built them. They are not committed — they are
large and machine-specific. `local.yaml` remaps that root:

```bash
cp local.yaml.example local.yaml     # then edit the right-hand sides
```

`_resolve_base_dir()` in `datasets/monai_dataset.py` does the lookup. If data
isn't found, fix `local.yaml` — never edit the JSONs.

## Run

A single entry point for both phases:

```bash
python cli.py train --project examples --conf brats_fard_t1n
python cli.py test  --project examples --conf brats_fard_t1n
python cli.py test  --project examples --conf brats_fard_t1n --set numEpochs=1 dontLog=true
```

`--project` is a directory under `confs/`; `--conf` is a file in it, by name or by
number. `--set key=value` overrides any field.

Two worked configurations ship in `confs/examples/` — one BRATS, one SHEBA. They
are meant to be copied and edited; **[docs/configs.md](docs/configs.md)** explains
every field that matters and the conventions that are easy to get wrong (channel
ordering per output contrast, which transforms SHEBA must *not* have).

Weights are looked up at `models/<project_name>/<conf>.pth`. They are published
separately — see *Weights* below.

## Evaluate

Do not use the trainer's built-in metrics for SHEBA: predictions come out at DICOM
intensity (~[1250, 12000]) against normalised ground truth (~[0, 365]), which
yields nonsense (PSNR ≈ −30). Score through `evaluation/` instead:

```bash
cd evaluation
python -c "from tester import main; main(project='examples', conf='brats_ghosting_t1n')"
```

Two worked scoring configurations ship in `evaluation/confs/examples/`, mirroring
the two training examples — one BRATS, one SHEBA. They name their data and result
roots symbolically:

```yaml
base_data_dir:   ${FARD_DATA}/BRATS/GLI/data/slices_axial/test
base_result_dir: ${FARD_RESULTS}/motion_corrupt
```

`${FARD_DATA}` and `${FARD_RESULTS}` resolve from the environment, or from the
`paths:` block of `local.yaml`. An unresolved name raises immediately rather than
producing a path with a literal `${...}` in it.

The `score_*.py` scripts drive the tester across a whole sweep and emit the
metrics CSVs; adapt one of the examples to the configs you want scored.

## Prepare data from scratch

Only needed to rebuild the registered dataset; both stages are config-driven.

```bash
# LR↔HR and cross-modal rigid registration (elastix)
python -m data_prep.register_lr2hr --config data_prep/configs/sheba_lr2hr.yaml --stage A
python -m data_prep.register_lr2hr --config data_prep/configs/sheba_lr2hr.yaml --stage B

# resample into MNI atlas space
python -m data_prep.register_to_mni --config data_prep/configs/sheba_mni_train.yaml
```

Use `sheba_mni_train.yaml` for training data (fits the atlas transform to the
high-quality scan) and `sheba_mni_test.yaml` for test data (fits to the
low-quality scan). That is the only difference between the two files.

> **The MNI stage is not reproducible run-to-run.** Its optimiser halts as soon as
> the convergence window fills, so repeated fits of the same scan scatter by up to
> ~12 mm in x-translation. The elastix stages (A and B) are exact. The saved
> `hq2mni_transform.tfm` files are therefore the durable record of the MNI
> alignment; replay those rather than re-fitting when comparability matters.

## Layout

```
cli.py              single entry point for train and test
trainer.py          training and inference loop
confs/              training configs        confs_test/  inference twins
nets/               architectures, resolved by the conf's cnnModule
datasets/           MONAI pipeline; env_transforms.py holds the corruption transforms
losses/             losses and metrics
data_prep/          registration pipeline (elastix + MNI)
evaluation/         metric scoring engine and its configs
src/fard/           output_mode: how raw model output is post-processed
docs/configs.md     how to write a config
```

## Weights

Weights live on the Hugging Face Hub rather than in this repository:

**https://huggingface.co/Lexer1/FARD**

```bash
hf download Lexer1/FARD --local-dir weights/
```

Three files, one per output contrast, ~2 MB each — weights only, no optimizer or
scheduler state. Place them at `models/<project_name>/<conf>.pth`, or load
directly with `safetensors`.

The whole robustness study needs **three files**, not one per config.
[docs/WEIGHTS.md](docs/WEIGHTS.md) has the checksums, the config-to-weights
mapping, and the input channel order each checkpoint expects.

## Reproducibility notes

Three things are worth knowing before trusting a number you regenerate.

**The FARD forward pass matters.** An earlier revision of `nets/FARD_verFull.py`
rewrote the forward pass; running the published weights through it degrades T1n
correlation from 0.986 to 0.69–0.88. This repository ships the forward pass the
published results were produced with, and only that one.

**Output post-processing is configuration, not a name.** Whether model output is
clamped to [0,1], rounded to integers, or left alone is the `output_mode` field.
In the predecessor it was decided by string-matching the wandb project name, so
renaming a project silently changed the arithmetic — and the training and
evaluation code disagreed about the rule. See `src/fard/output_mode.py`.

**Experiment logging is off by default.** `dontLog: true` in
`confs/defaults.yaml`; enable it per-config. Credentials belong in the
environment (`WANDB_API_KEY`), never in a config file — wandb uploads whatever
config it is given. `python scripts/check_secrets.py` checks for slips.

## Citation

```bibtex
@article{nazarov2026multicontrast,
  title   = {Multi-Contrast MRI Acceleration via Post-Reconstruction Fusion},
  author  = {Nazarov, Alexander and Kiryati, Nahum and Roizen, Dani and
             Kerpel, Ariel and Hoffmann, Chen and Greenberg, Gahl and
             Mayer, Arnaldo},
  journal = {Medical Image Analysis},
  year    = {2026},
  note    = {In press}
}
```

See [CITATION.cff](CITATION.cff). The complementary phase-encoding idea for
paired pre-/post-contrast T1 originates in Hod et al. (2023); this work extends it
to protocol-wide acceleration across T1, T2 and T2-FLAIR.

## License

**MIT by default** — see [LICENSE](LICENSE). Two exceptions:

| What | License | Why |
|---|---|---|
| The pipeline: trainer, datasets, evaluation, data_prep, CLI | MIT | generic infrastructure, freely reusable |
| The FARD architecture — `nets/FARD*.py` and `enhanced_attention/` | **CC BY-NC 4.0** | the novel contribution, non-commercial only. See [LICENSE-FARD](LICENSE-FARD) |
| Comparison baselines | not included — run from their authors' code | see [COMPARISON_METHODS.md](COMPARISON_METHODS.md) |

Every source file carries an `SPDX-License-Identifier` header, so the boundary is
visible in the code rather than only in this table.

Practical effect: the pipeline can be reused for anything, including commercially.
Running or redistributing the FARD model itself is non-commercial only.

Note that Creative Commons licenses are not designed for software — no patent
grant, not compatible with OSI-approved licenses. CC BY-NC is applied narrowly and
deliberately, to the architecture only.
