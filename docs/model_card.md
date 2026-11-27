# Model card — brain tumour MRI classifier

> Performance from the Colab T4 run of 2026-11-27 (logs in
> [`notebooks/runs/02_colab_runner_2026-11-27.ipynb`](../notebooks/runs/02_colab_runner_2026-11-27.ipynb)).

## Model details

| | |
|---|---|
| **Task** | 3-class classification of a single MRI slice: meningioma / glioma / pituitary tumour |
| **Architectures compared** | Custom CNN (2.39 M params), ResNet-18, EfficientNet-B0 (+ optional ResNet-50, DenseNet-121) |
| **Selected model** | ResNet-18, two-stage fine-tuning from ImageNet (11.2 M params); run `resnet18-final-20261127-210836`, registered as `brain-tumor-classifier` version 1 |
| **Framework** | PyTorch, torchvision |
| **Input** | one grayscale slice, any size; clipped to 0.5–99.5 percentiles, scaled to [0, 1], resized to 224 × 224 |
| **Output** | probabilities for the 3 classes |
| **Authors** | Yassmine Feki and Nour Al Houda Ben Slimene, Faculté des Sciences de Sfax (2026–2027) |
| **Licence** | MIT (code) |

## Intended use

- **Intended:** coursework, teaching and research on CNN comparison and MLOps practices.
- **Not intended:** clinical diagnosis, triage or any decision about a patient. The model
  has not been validated clinically or approved by any regulator.

## Training data

Figshare brain tumour dataset (3,064 slices, 233 patients) — see
[dataset_card.md](dataset_card.md). Patient-level split 70 / 15 / 15, stratified by class.

## Evaluation

- Selection on validation macro-F1; test set used once.
- Metrics: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix,
  latency and size. Test-time augmentation (horizontal flip) applied.
- Explainability: Grad-CAM heatmaps compared with the expert tumour masks.

### Performance (test set, patient-level split)

| Metric | Served model (final run) | ResNet-18, 5-fold CV |
|---|---|---|
| Accuracy | 0.895 | 0.926 ± 0.015 |
| Macro-F1 | 0.888 | 0.920 ± 0.018 |
| Recall meningioma / glioma / pituitary | 0.946 / 0.952 / 0.758 | 0.925 ± 0.023 / 0.950 ± 0.008 / 0.890 ± 0.048 |
| Inference time (ms / image, device) | 2.7 ms, T4 GPU, batch size 1 | — |

Reference points on the same test set: majority-class baseline 0.206 macro-F1, custom CNN
0.862 (0.883 after Optuna tuning), EfficientNet-B0 0.909. Trained and tested on a leaky
image-level split instead (a different test set), ResNet-18 reaches 0.963: scores reported
that way on this dataset are inflated. See the README for the full comparison.

## Limitations and risks

- Trained on one MRI sequence (T1 contrast-enhanced) from two Chinese hospitals:
  performance on other scanners, protocols or populations is unknown.
- Always predicts one of three tumour types; it cannot say "no tumour" or "other".
- Works on single 2-D slices, not on full 3-D volumes.
- The small test set (~35 patients) gives wide confidence intervals; cross-validation
  results are more reliable for ranking models.
- Meningioma, the minority class, is the hardest class for the custom CNN. For the served
  ResNet-18 the weakest class is pituitary tumour (recall 0.758): about 1 pituitary case in 4
  is classified as another tumour type.

## Ethical considerations

Medical images are sensitive data. The public dataset is anonymised; do not upload
identifiable patient images to the demo. Outputs must always be presented with the
"not a medical diagnosis" disclaimer, as the API does.
