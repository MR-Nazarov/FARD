# Model weights

Weights are published separately from the source. Each file is a bare
`state_dict` for the FARD architecture (0.28M parameters).

Place them under `models/<project_name>/<conf>.pth`, matching the `project_name`
of the config you are running.

## Files

| Weights | Contrast | Size | MD5 |
|---|---|---|---|
| `conf_41.pth` | FARD, T1n output | 2.1 MB | `d6f4dc53afb2` |
| `conf_42.pth` | FARD, T2w output | 2.1 MB | `0858db9365fa` |
| `conf_49.pth` | FARD, FLAIR output | 2.1 MB | `60ce09bdb54d` |

## One set, many configs

The corruption studies — motion, ghosting, misregistration and lesion-leakage —
share these three checkpoints. The original tree held 99 per-config copies of
them; they deduplicate to the three above, one per output contrast:

| Config prefix | Weights |
|---|---|
| `1XX`, `4XX` (T1n output) | T1n |
| `2XX`, `5XX` (T2w output) | T2w |
| `3XX`, `6XX` (FLAIR output) | FLAIR |

Verify a download with `md5sum` against the table above.
