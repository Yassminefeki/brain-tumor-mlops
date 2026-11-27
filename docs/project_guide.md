# Project guide — everything about the project in one place

A complete reference to prepare the presentation and answer questions. Every number
comes from the code, `params.yaml`, the data files or the MLflow runs of the Colab
T4 run of 2026-11-27.

---

## 1. The project in one paragraph

We classify **T1-weighted contrast-enhanced brain MRI slices** into three tumour types —
**meningioma, glioma, pituitary tumour** — and compare CNNs trained from scratch with
ImageNet-pretrained CNNs under identical, **leakage-free** conditions (split by patient).
Around the models, the repository is a full MLOps project: versioned data pipeline (DVC),
experiment tracking and model registry (MLflow), hyperparameter search (Optuna),
tests + CI (pytest, GitHub Actions, pre-commit), explainability (Grad-CAM) and serving
(FastAPI, Gradio, Docker). **Best model: ResNet-18, test macro-F1 0.916 (0.921 ± 0.017
over 5 folds).** Educational project, not a medical device.

---

## 2. The dataset

| Item | Value |
|---|---|
| Source | Figshare *brain tumor dataset* (article 1512427), Cheng et al., PLoS ONE 2015 and 2016 |
| Modality | T1-weighted contrast-enhanced MRI, 2-D slices, from hospitals in China |
| Format | MATLAB v7.3 (`.mat` = HDF5) files, struct `cjdata` with `image`, `label`, `PID`, `tumorBorder`, `tumorMask` |
| Size | **3,064 slices from 233 patients** (4 zip files, ~880 MB) |
| Classes | glioma 1,426 images / 89 patients · pituitary 930 / 62 · meningioma 708 / 82 |
| Image sizes | 3,049 at 512 × 512, 15 at 256 × 256 |
| Intensities | int16, range 0 – 12,728 (not standardised between scans) |
| Slices per patient | mean 13.2, median 13, min 1, max 38 |
| Tumour size | median 3,365 pixels = **1.3 % of the image** (small target) |
| Masks | expert tumour masks, used **only** to sanity-check Grad-CAM, never for training |

**Class imbalance:** glioma is 46.5 % of images, meningioma only 23.1 %. That is why we use
macro-F1 (not accuracy) as the main metric, and class weights in the loss.

**Labels in the file:** 1 = meningioma, 2 = glioma, 3 = pituitary, mapped to 0, 1, 2.
HDF5 stores MATLAB arrays column-major, so images are transposed back when read
(`data/mat_io.py`).

---

## 3. Repository map

```
brain-tumor-mlops/
├── params.yaml            single source of truth: every path and hyperparameter
├── dvc.yaml / dvc.lock    data pipeline definition / exact data versions (md5)
├── src/brain_tumor/
│   ├── config.py          loads params.yaml, dotted overrides (train.lr=0.001)
│   ├── data/              ingest, mat_io, metadata, preprocess, split, validate,
│   │                      dataset, datamodule, transforms
│   ├── models/            custom_cnn, pretrained (ResNet/EfficientNet/DenseNet), factory
│   ├── training/          engine (fit / evaluate / predict), metrics
│   ├── explain/gradcam.py Grad-CAM
│   ├── train.py           train + evaluate + log one model
│   ├── tune.py            Optuna search
│   ├── evaluate.py        test evaluation, TTA, latency, Grad-CAM grid
│   ├── baseline.py        majority-class baseline
│   ├── inference.py       Predictor used by the API and the demo
│   └── tracking.py        MLflow connection, lineage tags (git commit, data md5)
├── api/main.py            FastAPI service    api/gradio_app.py  Gradio demo
├── notebooks/             01 EDA · 02 Colab runner (template) · 03 results analysis
│                          runs/ = executed Colab run with all training logs
├── tests/                 39 pytest tests on synthetic data (no download needed)
├── reports/               data_validation.json, baseline.json, comparison.csv/.md, figures/
├── configs/best_params/   Optuna results (custom_cnn.yaml)
├── docs/                  architecture, data pipeline, models, training, experiments,
│                          MLOps setup, dataset card, model card, this guide
├── Dockerfile             inference image (CPU)
└── .github/workflows/ci.yml   lint → tests → pipeline check
```

---

## 4. The data pipeline (DVC, 6 stages)

Run everything with `dvc repro`; DVC re-runs only the stages whose code, inputs or
parameters changed, and records the md5 of every output in `dvc.lock`.

| # | Stage | Command | What it does | Output |
|---|---|---|---|---|
| 1 | download | `python -m brain_tumor.data.ingest download` | Lists the files through the Figshare API, downloads them with **retries and resume** (HTTP Range), checks each **md5** | `data/raw/` (persisted: a re-run keeps finished files) |
| 2 | extract | `… ingest extract` | Unzips the 4 archives, keeps only the `.mat` files, checks there are exactly 3,064 | `data/interim/*.mat` |
| 3 | metadata | `python -m brain_tumor.data.metadata` | One row per image: id, label, patient id, size, intensity range, tumour area | `data/metadata.csv` |
| 4 | preprocess | `python -m brain_tumor.data.preprocess` | Normalise + resize every image (see §6), resize masks | `data/processed/images/*.npy`, `masks/*.npy` |
| 5 | split | `python -m brain_tumor.data.split` | Patient-level split, CV folds, leaky image-level split (see §5) | `data/splits.csv` |
| 6 | validate | `python -m brain_tumor.data.validate` | 9 automatic checks (below); fails the pipeline if one fails | `reports/data_validation.json` |

**The 9 validation checks (all pass):** expected image/patient counts (3,064 / 233) ·
expected class counts · one label per patient · unique image ids · every image assigned
to a split · **no patient shared between train, val and test** · all 3 classes in each split ·
CV folds valid (test excluded, each patient in exactly one fold) · processed arrays
readable, right shape, finite, in [0, 1].

---

## 5. Splitting and data leakage (the key methodological point)

**Problem:** 233 patients but 3,064 slices — about 13 slices per patient, and neighbouring
slices of the same tumour look almost identical. A random split by image puts slices of
the same patient in train and test: the model is then tested on patients it has seen,
and the score measures memorisation, not generalisation.

**Our split (`split` column, used for everything):** patients are split 70 / 15 / 15,
**stratified by tumour class** (scikit-learn `train_test_split` on patients, seed 42);
each image inherits its patient's set.

| Set | Images | Patients | meningioma / glioma / pituitary |
|---|---|---|---|
| train | 2,150 | 163 | 468 / 1,010 / 672 |
| val | 494 | 35 | 128 / 228 / 138 |
| test | 420 | 35 | 112 / 188 / 120 |

**Cross-validation (`fold` column):** the 198 train+val patients are divided into 5
patient-level folds with `StratifiedGroupKFold` (39–40 patients each). The **test set
stays fixed** and excluded, so the 5 fold models are all scored on the same 420 test images.

**Leaky split (`image_split` column, for the comparison only):** a random 70/15/15 split
by image. Result: **182 of its 184 test patients also appear in its training set, i.e.
99.6 % of its test images are leaked.** Training the same ResNet-18 on it gives
0.963 macro-F1 instead of 0.916 → leakage inflates the score by ~5 points.

---

## 6. Preprocessing and augmentation

**Offline (preprocess stage, identical at inference time — same function):**
1. orientation fix (transpose of the HDF5 array);
2. **robust intensity clipping at the 0.5th and 99.5th percentiles** of each image
   (removes outlier pixels, makes scans with different intensity ranges comparable);
3. min-max scaling to [0, 1];
4. resize to **224 × 224** (area interpolation when shrinking, the best quality for
   downscaling; masks use nearest-neighbour so they stay binary);
5. saved as **float16** `.npy` (halves disk use: 300 MB of images instead of ~600 MB in
   float32; the masks add 156 MB).

**Online augmentation (training only, `data/transforms.py`)** — light and
anatomy-preserving: horizontal flip (p = 0.5; brains are roughly left-right symmetric),
rotation ±15°, translation ±5 %, zoom 0.9–1.1, brightness and contrast ±10 %.
No vertical flips or strong distortions (unrealistic anatomy).

**For pretrained models:** the grayscale channel is repeated to 3 channels and
normalised with the ImageNet mean/std, because their weights expect RGB ImageNet-like input.
The custom CNN takes the single channel in [0, 1].

**Data loading:** all 3,064 arrays are preloaded in RAM as float16 (~300 MB), so training
has no disk I/O; shuffled training loader with a seeded generator and seeded workers.

---

## 7. The models

### 7.1 Custom CNN (designed and trained from scratch) — 2.39 M parameters

```
Input 224 × 224 × 1
Block 1  2 × [Conv 3×3, 32 filters  → BatchNorm → ReLU] → MaxPool 2   224 → 112
Block 2  2 × [Conv 3×3, 64          → BN → ReLU] → MaxPool            112 → 56
Block 3  2 × [Conv 3×3, 128         → BN → ReLU] → MaxPool             56 → 28
Block 4  2 × [Conv 3×3, 256         → BN → ReLU] → MaxPool             28 → 14
Block 5  2 × [Conv 3×3, 256         → BN → ReLU] → MaxPool             14 → 7
Global average pooling → 256
Dropout 0.5 → Dense 128 → ReLU → Dropout 0.3 → Dense 3
```

Design choices, and why:
- **stacked 3×3 convolutions** (VGG idea): two 3×3 layers see as far as one 5×5 with fewer
  parameters and an extra non-linearity;
- **filters doubled when resolution halves**: keeps the computation per block balanced;
- **BatchNorm**: faster, more stable training, allows a higher learning rate (conv bias
  removed since BN has its own);
- **global average pooling instead of Flatten**: 256 features instead of 256×7×7 = 12,544 →
  far fewer parameters, less overfitting on 2 k images;
- **receptive field 156 px** (of 224): large enough to see a whole tumour and its context;
- **two dropouts** in the head against overfitting; **Kaiming (He) initialisation** for ReLU.

Optuna-tuned variant: 4 blocks (32-64-128-256), dense 256, dropout 0.32 → **1.24 M parameters**.

### 7.2 Pretrained models (transfer learning from ImageNet)

| Model | Parameters | Why it is in the study |
|---|---|---|
| **ResNet-18** | 11.2 M | Classic residual network; skip connections make deep nets trainable; strong, simple baseline for transfer |
| **EfficientNet-B0** | 4.0 M | Modern architecture designed for accuracy per parameter (compound scaling, depthwise convolutions, squeeze-and-excitation) |
| ResNet-50, DenseNet-121 | 23.5 M / 7.0 M | Supported by the code, not run (optional experiment) |

The ImageNet classifier is replaced by **Dropout(0.4) → Linear(3)**. The layers are split into
three groups for **discriminative learning rates**: *early* (generic edges/textures —
ResNet conv1…layer2), *late* (high-level features — layer3, layer4), *head* (new classifier).

---

## 8. Training

**Common to every model:**

| Setting | Value | Why |
|---|---|---|
| Loss | cross-entropy with **class weights** (train: meningioma 1.53, glioma 0.71, pituitary 1.07) and **label smoothing 0.1** | weights counter the imbalance; smoothing avoids over-confident predictions |
| Optimiser | **AdamW**, weight decay 1e-4 | Adam with decoupled weight decay (proper L2 regularisation) |
| Batch size | 32 | |
| Scheduler | **cosine annealing** per stage, 1 epoch linear warm-up | smooth decay of the learning rate |
| Model selection | best epoch by **validation macro-F1**; its weights are restored at the end | |
| Early stopping | stop when validation macro-F1 has not improved for *patience* epochs | avoids wasting time and overfitting |
| Mixed precision | AMP on GPU (float16 compute) | faster training and less GPU memory on the T4 |
| Seed | 42 everywhere (Python, NumPy, PyTorch, data loader) | reproducibility |

**Custom CNN:** one stage, up to 50 epochs, learning rate 1e-3, patience 8.

**Pretrained models — two-stage fine-tuning** (up to 20 epochs, patience 5):
1. **Stage "head"** (3 epochs): backbone frozen, *including its BatchNorm statistics*
   (otherwise small batches would overwrite the ImageNet statistics); only the new head
   learns, at 1e-3. Reason: a randomly initialised head sends large, noisy gradients that
   would damage the pretrained features.
2. **Stage "finetune"**: everything unfrozen with discriminative learning rates —
   early layers **1e-5**, late layers **1e-4**, head **1e-3**. Early layers hold generic
   features that transfer well and need little change; late layers must adapt more to MRI.
   Early stopping only counts epochs of this stage.

**Hardware:** training on Google Colab, NVIDIA T4 GPU (2–3.5 min per model). On the
laptop CPU one custom-CNN epoch takes 5–30 minutes, so training there is impractical.

**Saved per run** (`models/<model>/<run>/`): `best.pt` checkpoint (weights + full config +
class names + preprocessing settings, so it can be served alone), `training_curves.png`,
`confusion_matrix.png`, `gradcam.png`, `test_predictions.csv`, `history.json`,
`summary.json` — all also logged to MLflow.

---

## 9. Evaluation

- **Test set used once per final run**, never for choosing anything.
- **Main metric: macro-F1** = mean of the three per-class F1 scores. Each class counts
  equally, so the minority class (meningioma) cannot be ignored — unlike accuracy, which
  a model can raise by favouring glioma (the majority-class baseline already gets 0.448
  accuracy but only 0.206 macro-F1).
- Also: accuracy, per-class precision / recall / F1, confusion matrix, parameters,
  model size, **inference time** (batch size 1, 64 images after warm-up).
- **Test-time augmentation (TTA):** the prediction is the average of the softmax of the
  image and of its horizontal flip. Reported scores use TTA; the score without TTA is also
  logged (ResNet-18: 0.916 with, 0.925 without — TTA does not always help). The API and
  demo do **not** use TTA (simpler, twice as fast).
- **Baseline:** always predict the majority class (glioma) → defines "learnt nothing".

---

## 10. Hyperparameter search (Optuna)

| Setting | Value |
|---|---|
| Sampler | **TPE** (Tree-structured Parzen Estimator), seed 42 — Bayesian-style: proposes values similar to the best trials so far |
| Pruner | **Median pruner**: after 3 warm-up epochs, stops a trial whose validation macro-F1 is below the median of earlier trials at the same epoch (first 3 trials never pruned) |
| Budget | 20 trials × 10 epochs, objective = best validation macro-F1 |
| Custom CNN space | lr 1e-4–3e-3 (log), weight decay 1e-5–1e-2 (log), dropout 0.2–0.6, dense units {64, 128, 256}, base filters {16, 32}, blocks {4, 5} |
| Pretrained space | backbone lr 1e-5–5e-4 (log), weight decay, dropout 0.2–0.6, freeze epochs {0, 2, 4} |
| Storage | SQLite (`models/optuna_<model>.db`), every trial is a nested MLflow run |

**Result (custom CNN):** 3 trials completed, **17 pruned** (shown as `FAILED` with tag
`pruned=true` in MLflow — that is the pruner working, not a crash). Best trial:
lr 8.0e-4, weight decay 2.6e-5, dropout 0.32, dense 256, 4 blocks of 32-64-128-256 →
validation macro-F1 0.881 in a 10-epoch trial. Retrained fully with these values: test
macro-F1 **0.883** (vs 0.862 untuned) with **half the parameters**. Its full-training
validation score (0.822) is lower than the untuned model's (0.933), although its test score
is higher — a sign that 35 validation patients give noisy estimates.

---

## 11. Results

### 11.1 Main table (test set: 420 images, 35 patients, patient-level split)

| Model | Val macro-F1 | Test acc. | **Test macro-F1** | Params | ms / image (T4) | Train min |
|---|---|---|---|---|---|---|
| Majority class (baseline) | — | 0.448 | 0.206 | — | — | — |
| Custom CNN, no augmentation | 0.920 | 0.850 | 0.837 | 2.39 M | 1.7 | 2.8 |
| Custom CNN | 0.933 | 0.871 | 0.862 | 2.39 M | 1.9 | 3.3 |
| Custom CNN, Optuna-tuned | 0.822 | 0.886 | 0.883 | 1.24 M | 1.5 | 1.8 |
| **ResNet-18 (served)** | **0.957** | **0.921** | **0.916** | 11.2 M | 2.5 | 2.3 |
| EfficientNet-B0 | 0.946 | 0.917 | 0.909 | 4.0 M | 8.7 | 3.2 |
| ResNet-18, 5-fold CV (mean ± std) | 0.932 ± 0.041 | 0.926 ± 0.016 | **0.921 ± 0.017** | 11.2 M | — | ~2.4 / fold |
| ResNet-18, second run (registered) | 0.950 | 0.895 | 0.888 | 11.2 M | 2.7 | 1.6 |
| ResNet-18, **leaky image-level split** | 0.973 | 0.967 | 0.963 | 11.2 M | 2.6 | — |

### 11.2 Per-class recall (test)

| Model | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| Baseline | 0.000 | 1.000 | 0.000 |
| Custom CNN, no aug | 0.741 | 0.947 | 0.800 |
| Custom CNN | 0.777 | 0.936 | 0.858 |
| Custom CNN, tuned | 0.991 | 0.862 | 0.825 |
| **ResNet-18 (served)** | 0.929 | 0.947 | 0.875 |
| EfficientNet-B0 | 0.920 | 0.936 | 0.883 |
| ResNet-18, 5-fold CV | 0.925 ± 0.023 | 0.950 ± 0.008 | 0.890 ± 0.048 |
| ResNet-18, second run | 0.946 | 0.952 | 0.758 |

### 11.3 What the results say

1. **Transfer learning wins**: pretrained models beat the custom CNN by 2.6–5.4 points of
   macro-F1. With only ~2,000 training images, features learnt on 1.2 M ImageNet images
   (edges, textures, shapes) transfer even to grayscale MRI.
2. **Augmentation helps** the scratch model (+0.025), tuning helps more (+0.021 on top).
3. **ResNet-18 vs EfficientNet-B0**: nearly equal accuracy; EfficientNet has 2.8× fewer
   parameters but is **3.5× slower per image** here (depthwise convolutions are not
   efficient on GPU at batch size 1) → parameter count ≠ speed.
4. **Leakage inflates the score** by ~5 points (0.916 → 0.963).
5. **Variance is large**: two ResNet-18 runs with identical settings scored 0.916 and 0.888
   (different random augmentations, shuffling and early-stopping point); the 5 folds range
   0.891–0.937. With 35 test patients, **the 5-fold mean ± std is the reliable figure**.
6. **Errors per class**: the scratch CNN mostly misses meningiomas (the minority class);
   for ResNet-18 the weakest class becomes pituitary tumour.
7. **Overfitting signs**: ResNet-18 reaches 0.99+ training accuracy vs ~0.95 validation —
   regularisation (dropout, weight decay, augmentation, label smoothing, early stopping)
   keeps the gap moderate.

### 11.4 Which model is served, and why

Selection rule: **best validation macro-F1** (never the test score). Among the two ResNet-18
runs with the same settings, `resnet18-aug-20261127-202800` (val 0.957, best epoch 16) beats
`resnet18-final` (val 0.950) → it is the served `models/best.pt`. `resnet18-final` is the one
registered in the MLflow Model Registry (version 1). Sent through the API on CPU, the served
model gets 390/420 test images right (accuracy 0.929, no TTA).

---

## 12. Explainability: Grad-CAM

**What it shows:** which regions of the image drove the prediction.
**How it works:** take the feature maps of the last convolutional layer (ResNet-18:
`layer4[-1]`, 7 × 7; custom CNN: last ReLU of block 5); back-propagate the score of the
predicted class to them; average the gradients of each channel → one weight per channel;
weighted sum of the feature maps → ReLU (keep only positive evidence) → upsample to
224 × 224 → normalise to [0, 1] → overlay on the image as a jet colour map.
**Use here:** a sanity check that the model looks at the tumour, not at the skull or
image borders; the expert masks are saved to compare with. Available in the demo and at
`POST /explain`. Limitation: 7 × 7 resolution → coarse heat maps.

---

## 13. MLOps

### 13.1 Configuration
One file, `params.yaml`, holds every path and hyperparameter. Any value can be overridden
from the command line with dotted keys (`train.model=resnet18 train.epochs=20`); typos
fail loudly (unknown key error). Every run logs the full resolved configuration.

### 13.2 Data versioning — DVC
Large data is never in Git. `dvc.yaml` describes the pipeline; `dvc.lock` stores the md5
of every input and output, so Git versions *which* data each commit used. `dvc repro`
rebuilds only what changed. A DVC remote (DagsHub) can store the data for `dvc pull`;
without it, `dvc repro` rebuilds the same version from Figshare (md5-checked).

### 13.3 Experiment tracking — MLflow
Each run logs: all parameters, per-epoch metrics (loss, accuracy, macro-F1, learning rate),
test metrics, artifacts (curves, confusion matrix, Grad-CAM, predictions, checkpoint) and
**lineage tags**: git commit, branch, dirty flag, and the DVC md5 of the processed data and
the splits. → any number can be traced to the exact code and data that produced it.
Experiments are named `brain-tumor/<model>`; Optuna trials are nested runs. The final
model is registered as `brain-tumor-classifier`. Backend: DagsHub server, or a local
SQLite file `mlflow.db` (what we used; 34 runs). View: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### 13.4 Code quality and CI
- **39 tests** (pytest) on a **synthetic dataset** written in the exact `.mat` format (blobs
  whose position depends on the class): config parsing and overrides; data (orientation of
  the `.mat` files, patient-id decoding, normalisation, metadata, **no patient leakage in the
  split**, the image-level split does leak, folds, a patient with two labels is rejected,
  validation stage); models (custom CNN matches its documented design, pretrained outputs and
  layer groups, Grad-CAM, transforms, class weights, two-stage plan); and an end-to-end run —
  training, evaluation, Optuna, baseline, CV fold, leaky split and the API. CI never needs
  the real data.
- **Ruff** for linting and formatting.
- **pre-commit hooks** on every commit: whitespace, end of file, YAML, large files (> 2 MB
  refused — data belongs in DVC), private keys, Ruff, **nbstripout** (removes notebook
  outputs, except notebooks marked `keep_output`).
- **GitHub Actions** (`ci.yml`) on every push / pull request: lint → tests (CPU PyTorch) →
  `dvc stage list` (pipeline file valid).

### 13.5 Reproducibility, summarised
Same code (git commit) + same data (`dvc.lock` md5) + same parameters (`params.yaml`, logged)
+ same seed (42) → same split and comparable results. GPU non-determinism and random
augmentation still cause run-to-run variance, which is why we report 5-fold mean ± std.

### 13.6 Versions (Git tags)
`v1.0.0` results · `v1.0.1` per-class recall · `v1.0.2` served model = best validation run ·
`v1.0.3` results analysis and experiment report · `v1.0.4` demo examples, faster Docker builds.

---

## 14. Serving

### 14.1 FastAPI service (`api/main.py`, `make api`, http://localhost:8000/docs)

| Endpoint | Returns |
|---|---|
| `GET /health` | status, whether the model is loaded (`degraded` + error message if not) |
| `GET /model` | model name, best epoch, validation macro-F1, image size, classes, device |
| `POST /predict` | upload `.mat`, `.png` or `.jpg` → predicted class, confidence, 3 probabilities, disclaimer |
| `POST /explain` | upload an image → Grad-CAM overlay as PNG |

Errors: empty file 400 · file > 20 MB 413 · unreadable file 415 · no file 422 ·
model missing 503 (the service stays up and reports the problem on `/health`).
The same preprocessing function as training is applied to uploads.

### 14.2 Gradio demo (`api/gradio_app.py`, `make demo`, http://localhost:7860)
Upload a slice (or click one of the three test-set examples: meningioma, glioma,
pituitary) → probability bars + Grad-CAM. ~40 s to start (Gradio import + model load).

### 14.3 Docker
`python:3.11-slim`, CPU-only PyTorch, only inference dependencies (no DVC / MLflow /
Optuna), runs as a non-root user, health check on `/health`, model mounted as a volume:
`docker run -p 8000:8000 -v $(pwd)/models:/app/models brain-tumor-api`. Image 2.6 GB
(mostly PyTorch). Layers ordered so that a code or README change does not reinstall packages.

---

## 15. How to reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-dev                 # package, dev tools, API/demo deps, git hooks
make test                        # 39 tests, synthetic data
dvc repro                        # data: download → validate (~10 min on a laptop)
make baseline                    # majority-class reference
# training: on Colab (notebooks/02_colab_runner.ipynb, T4 GPU), or with a GPU:
python -m brain_tumor.train train.model=resnet18 --promote
python -m brain_tumor.tune train.model=custom_cnn
make api        # or: make demo / make docker
```

---

## 16. Limitations and ethics

- One MRI sequence (T1 contrast-enhanced), one dataset from hospitals in China: performance
  on other scanners, protocols or populations is unknown.
- 2-D slices only, not full 3-D volumes; one slice at a time.
- Always answers one of 3 tumour types: no "no tumour" or "other" class.
- Small test set (35 patients): ±2–3 points of uncertainty; CV mitigates it.
- Grad-CAM is coarse (7 × 7) and shows correlation, not causation.
- Not validated clinically, not a medical device: every output carries the disclaimer.
- The public dataset is anonymised; identifiable patient images must not be uploaded.

---

## 17. Likely questions and answers

**Why split by patient?** Slices of one patient are nearly identical; an image-level split
leaks 99.6 % of test images and inflates macro-F1 from 0.916 to 0.963.

**Why macro-F1 and not accuracy?** Classes are imbalanced (glioma 46.5 %). Predicting only
glioma gives 0.448 accuracy but 0.206 macro-F1; macro-F1 weighs every class equally.

**Why does ImageNet transfer to MRI?** Early layers learn generic edges, textures and blobs
that exist in any image; only the later layers need to adapt. Our results: +2.6 to +5.4
points over the scratch CNN.

**Why two-stage fine-tuning?** A new, random head produces large gradients that would
destroy pretrained features; we first train the head alone, then unfreeze with small
learning rates for early layers.

**Why freeze BatchNorm statistics in stage 1?** With batches of 32 grayscale MRIs, the
running mean/variance would drift away from ImageNet's, corrupting the frozen features.

**Why grayscale repeated to 3 channels?** Pretrained weights expect 3-channel input;
repeating keeps the pretrained first layer usable (alternative: average its filters).

**Why class weights and label smoothing?** Weights (inverse class frequency) make
mistakes on meningioma cost more; smoothing (0.1) prevents over-confidence and improves
calibration.

**Why AdamW and cosine schedule?** AdamW applies weight decay correctly with Adam;
cosine annealing lowers the learning rate smoothly, helping convergence at the end.

**Why global average pooling?** Cuts the classifier input from 12,544 to 256 values →
fewer parameters, less overfitting, and a natural fit for Grad-CAM.

**How do you know it is not overfitting badly?** It does overfit somewhat: training accuracy
reaches 0.99 vs ~0.95 on validation, and test (0.916) is below validation (0.957, which is
optimistic because it was used to choose the epoch). But the test score on 35 unseen
patients stays high, 5-fold CV gives a consistent 0.921 ± 0.017, and Grad-CAM focuses on
the tumour region. Dropout, weight decay, augmentation, label smoothing and early stopping
limit the gap.

**Why is the served model not the registered one?** Two identical-settings ResNet-18
runs; the rule is best validation macro-F1 → `resnet18-aug` (0.957 vs 0.950).

**Why did 17 Optuna trials "fail"?** They were pruned by the median pruner (tag
`pruned=true`); MLflow labels any interrupted run as FAILED.

**Why is the tuned CNN's validation score lower than the untuned one?** Validation has
only 35 patients; scores fluctuate by several points. Its test score is higher (0.883 vs
0.862). This is also why we rely on cross-validation.

**What is TTA and why is the test score without TTA higher for ResNet-18?** Averaging the
prediction of the image and its mirror; it usually stabilises predictions but is not
guaranteed to help (0.916 with vs 0.925 without) — differences of this size are noise
on 420 images.

**Why is EfficientNet slower despite fewer parameters?** Depthwise-separable convolutions
and squeeze-and-excitation have low arithmetic intensity: fewer operations, but more
memory-bound steps and more layers; at batch size 1 on GPU this costs time.

**What does DVC bring over Git?** Git cannot store 1 GB of data; DVC stores data
elsewhere and puts small md5 pointers in Git (`dvc.lock`), plus a pipeline that only
re-runs changed stages.

**What does MLflow bring?** Every run's parameters, metrics, artifacts, git commit and
data version in one searchable place; comparison across 34 runs; model registry.

**How would you improve the model?** More data or another dataset (external validation);
use neighbouring slices or 3-D volumes; tumour segmentation as an auxiliary task (masks are
available); stronger augmentation (elastic, intensity); ensembles of the 5 CV models;
calibration analysis; larger backbones (ResNet-50, DenseNet-121, already supported).

**How long does everything take?** On the Colab T4: data pipeline ~2.5 min; each model
2–3.5 min; Optuna (20 trials × 10 epochs, 17 pruned) ~11 min; 5-fold CV ~13 min; the whole
notebook under 1 hour. (On the laptop the data download alone takes ~10 min.)

---

## 18. Glossary

| Term | Meaning |
|---|---|
| Macro-F1 | Mean over classes of F1 = 2 · precision · recall / (precision + recall) |
| Recall (sensitivity) | Of the real cases of a class, the fraction found |
| Data leakage | Information from the test set reaching training, inflating scores |
| Transfer learning | Starting from weights learnt on another task (ImageNet) |
| Fine-tuning | Continuing to train pretrained weights on the new task |
| Discriminative learning rates | Different learning rates for different layer groups |
| Early stopping | Stop training when the validation metric stops improving |
| Label smoothing | Train towards 0.9/0.05/0.05 instead of 1/0/0 targets |
| TTA | Test-time augmentation: average predictions over transformed copies |
| Grad-CAM | Gradient-weighted class activation map: where the CNN looked |
| TPE | Optuna's default Bayesian-style sampler |
| Pruning | Stopping unpromising hyperparameter trials early |
| AMP | Automatic mixed precision (float16 compute on GPU) |
| DVC / dvc.lock | Data version control / file with the md5 of every pipeline output |
| MLflow run / registry | One tracked experiment / versioned store of production models |
