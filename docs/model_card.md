# Model card — brain tumour MRI classifier

> Complete the *Performance* section with the final MLflow run before submission.

## Model details

| | |
|---|---|
| **Task** | 3-class classification of a single MRI slice: meningioma / glioma / pituitary tumour |
| **Architectures compared** | Custom CNN (2.39 M params), ResNet-18, EfficientNet-B0 (+ optional ResNet-50, DenseNet-121) |
| **Selected model** | *to fill: name, MLflow run ID, registry version* |
| **Framework** | PyTorch, torchvision |
| **Input** | one grayscale slice, any size; clipped to 0.5–99.5 percentiles, scaled to [0, 1], resized to 224 × 224 |
| **Output** | probabilities for the 3 classes |
| **Authors** | Yassmine and teammate, Faculté des Sciences de Sfax (2026–2027) |
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

| Metric | Value |
|---|---|
| Accuracy | *to fill* |
| Macro-F1 | *to fill* |
| Recall meningioma / glioma / pituitary | *to fill* |
| Inference time (ms / image, device) | *to fill* |

## Limitations and risks

- Trained on one MRI sequence (T1 contrast-enhanced) from two Chinese hospitals:
  performance on other scanners, protocols or populations is unknown.
- Always predicts one of three tumour types; it cannot say "no tumour" or "other".
- Works on single 2-D slices, not on full 3-D volumes.
- The small test set (~35 patients) gives wide confidence intervals; cross-validation
  results are more reliable for ranking models.
- Meningioma, the minority class, is the most frequently confused with glioma.

## Ethical considerations

Medical images are sensitive data. The public dataset is anonymised; do not upload
identifiable patient images to the demo. Outputs must always be presented with the
"not a medical diagnosis" disclaimer, as the API does.
