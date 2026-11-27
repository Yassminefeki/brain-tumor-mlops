# Experiment plan

Each experiment changes **one thing** so its effect can be attributed. All runs use the
patient-level split unless stated, and are compared on the same test set.

## Core experiments

| # | Run | Command | Question answered |
|---|---|---|---|
| 0 | Majority-class baseline | `python -m brain_tumor.baseline` | What score means "learnt nothing"? |
| 1 | Custom CNN, no augmentation | `train.model=custom_cnn train.augment=false` | Plain scratch baseline |
| 2 | Custom CNN + augmentation | `train.model=custom_cnn` | Effect of augmentation on a small model |
| 3 | ResNet-18 + augmentation | `train.model=resnet18` | Effect of transfer learning |
| 4 | EfficientNet-B0 + augmentation | `train.model=efficientnet_b0` | Modern, efficient architecture: accuracy vs. cost |
| 5 | Best model, image-level split | `train.model=<best> train.split_column=image_split` | How much does leakage inflate the score? |

## Optional experiments

| # | Run | Command | Question answered |
|---|---|---|---|
| 6 | Optuna, same budget per model | `python -m brain_tumor.tune train.model=<m>` then `--best-params` | Gain from automated tuning |
| 7 | 5-fold patient-level CV | `train.fold=0` … `train.fold=4` | Mean ± std: is the ranking stable? |
| 8 | Feature extraction only | `models.pretrained.freeze_epochs=20 models.pretrained.epochs=20` | How different is MRI from ImageNet? |
| 9 | Custom CNN ablations | see [models.md](models.md#configurable-variants-ablations) | Role of depth, BatchNorm, GAP, dropout |
| 10 | DenseNet-121 / ResNet-50 | `train.model=densenet121` | Does a bigger backbone help? |

## Run naming

`<model>-<variant>-<initials>`, e.g. `resnet18-aug-ys`, `custom_cnn-noaug-ab`.
`train.py` builds a default name from the model, augmentation, split and fold.

## Reporting results

1. Run `notebooks/03_results_analysis.ipynb`: it queries MLflow and writes
   `reports/comparison.csv`, `reports/comparison.md` and the accuracy-vs-size figure.
2. Copy the table into the README *Results* section and the presentation.
3. For each model, take `confusion_matrix.png`, `training_curves.png` and `gradcam.png`
   from its MLflow run.

### Comparison template

| Model | Val macro-F1 | Test acc. | Test macro-F1 | Recall men. | Recall gli. | Recall pit. | Params | ms/img | Train min |
|---|---|---|---|---|---|---|---|---|---|
| Majority class | | | | | | | — | — | — |
| Custom CNN (no aug) | | | | | | | 2.39 M | | |
| Custom CNN | | | | | | | 2.39 M | | |
| ResNet-18 | | | | | | | 11.2 M | | |
| EfficientNet-B0 | | | | | | | 4.0 M | | |

### Strengths / weaknesses template

Expected trends, to confirm or correct with your own results:

| Model | Strengths | Weaknesses |
|---|---|---|
| Custom CNN | Small, fast, fully understood design | Learns everything from 2 k images; lower accuracy |
| ResNet-18 | Strong transfer, simple to explain, good Grad-CAM | 5× larger than the custom CNN, overfits fast |
| EfficientNet-B0 | Best accuracy per parameter | More sensitive to learning rate; slower per image than its size suggests |
