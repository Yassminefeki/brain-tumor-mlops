# data/

Everything in this folder except this file is produced by the DVC pipeline and is
**never committed to Git**.

| Path | Produced by | Content |
|---|---|---|
| `raw/` | `download` | Figshare zips, `cvind.mat`, README, `manifest.json` |
| `interim/` | `extract` | 3,064 `.mat` files |
| `metadata.csv` | `metadata` | one row per image |
| `processed/images/` | `preprocess` | 224 × 224 float16 `.npy` arrays |
| `processed/masks/` | `preprocess` | resized tumour masks (Grad-CAM checks only) |
| `splits.csv` | `split` | patient-level split, CV folds, leaky comparison split |

Get it with `dvc pull` (team copy) or `dvc repro` (rebuild from Figshare).
See [docs/data_pipeline.md](../docs/data_pipeline.md).
