# Model weights

Weights are hosted on the Hugging Face Hub rather than committed here:

**https://huggingface.co/Lexer1/FARD**

Each file is a bare `state_dict` for the FARD architecture — weights only, no
optimizer or scheduler state. 0.504 M parameters, about 2 MB per file.

## Download

```bash
pip install huggingface_hub
hf download Lexer1/FARD --local-dir weights/
```

or in Python:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

path = hf_hub_download("Lexer1/FARD", "fard_t1n.safetensors")
model.load_state_dict(load_file(path))     # strict=True
```

## Files

| Hub file | Output contrast | Params | Source `.pth` MD5 |
|---|---|---|---|
| `fard_t1n.safetensors` | T1-weighted | 0.504 M | `d6f4dc53afb2` |
| `fard_t2w.safetensors` | T2-weighted | 0.504 M | `0858db9365fa` |
| `fard_flair.safetensors` | T2-FLAIR | 0.504 M | `60ce09bdb54d` |

`checksums.json` on the Hub carries each file's SHA-256 alongside the MD5 of the
original `.pth` it was converted from. The conversion is lossless — the tensors
are bit-identical, verified by reloading and comparing against the source.

The Hub files have the `model.` prefix stripped, so they load straight into the
`FARD` module. The trainer wraps it in `MainModel`, which adds that prefix back;
if you load into `MainModel` rather than the bare network, re-add it.

## Using them with this repository

The trainer looks for `models/<project_name>/<conf>.pth`, matching the
`project_name` of the config being run. Either point it at a converted file or
keep the original `.pth` layout.

## One set, many configs

The corruption studies — motion, ghosting, misregistration and lesion-leakage —
share these three checkpoints. The original tree held 99 per-config copies; they
deduplicate to three, one per output contrast:

| Config prefix | Weights |
|---|---|
| `1XX`, `4XX` (T1n output) | T1n |
| `2XX`, `5XX` (T2w output) | T2w |
| `3XX`, `6XX` (FLAIR output) | FLAIR |

## Channel order

Each checkpoint expects its own target contrast's accelerated input first:

| Output | Input channel order |
|---|---|
| T1n | `t1n, t2w, flair` |
| T2w | `t2w, flair, t1n` |
| FLAIR | `flair, t1n, t2w` |

Nothing in the network enforces this, so a wrong order yields plausible but
incorrect output. See `docs/configs.md` for how `ConcatItemsd` sets it.

## License

The weights are CC BY-NC 4.0, matching the FARD architecture. See
[LICENSE-FARD](../LICENSE-FARD).
