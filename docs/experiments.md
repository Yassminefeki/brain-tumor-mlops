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

### Results (Colab T4, 2026-11-27)

Test set of 420 images, patient-level split; ms/img at batch size 1 on the T4. Logs:
[`notebooks/runs/02_colab_runner_2026-11-27.ipynb`](../notebooks/runs/02_colab_runner_2026-11-27.ipynb).

| # | Model | Val macro-F1 | Test acc. | Test macro-F1 | Recall men. | Recall gli. | Recall pit. | Params | ms/img | Train min |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Majority class | — | 0.448 | 0.206 | 0.000 | 1.000 | 0.000 | — | — | — |
| 1 | Custom CNN (no aug) | 0.920 | 0.850 | 0.837 | 0.741 | 0.947 | 0.800 | 2.39 M | 1.7 | 2.8 |
| 2 | Custom CNN | 0.933 | 0.871 | 0.862 | 0.777 | 0.936 | 0.858 | 2.39 M | 1.9 | 3.3 |
| 3 | ResNet-18 (served) | 0.957 | 0.921 | 0.916 | 0.929 | 0.947 | 0.875 | 11.2 M | 2.5 | 2.3 |
| 4 | EfficientNet-B0 | 0.946 | 0.917 | 0.909 | 0.920 | 0.936 | 0.883 | 4.0 M | 8.7 | 3.2 |
| 5 | ResNet-18, image-level split (leaky test set) | 0.973 | 0.967 | 0.963 | 0.953 | 0.972 | 0.971 | 11.2 M | 2.6 | — |
| 6 | Custom CNN, Optuna-tuned | 0.822 | 0.886 | 0.883 | 0.991 | 0.862 | 0.825 | 1.24 M | 1.5 | 1.8 |
| 7 | ResNet-18, 5-fold CV (mean ± std) | 0.932 ± 0.041 | 0.926 ± 0.016 | 0.921 ± 0.017 | 0.925 ± 0.023 | 0.950 ± 0.008 | 0.890 ± 0.048 | 11.2 M | — | ~2.4 per fold |

Experiments 8–10 were not run. The second ResNet-18 run (`resnet18-final`, registered in
MLflow: val 0.950, test 0.888) shows the run-to-run variance; see the README.

### Strengths / weaknesses

Confirmed or corrected by the results above:

| Model | Strengths | Weaknesses |
|---|---|---|
| Custom CNN | Small, fast (1.5–1.9 ms/img), fully understood design; tuning halves its size and gains 0.021 | Learns everything from 2 k images: 2.6–5.4 points below the pretrained models, misses meningiomas until tuned |
| ResNet-18 | Best test macro-F1 (0.916; 0.921 ± 0.017 over 5 folds), fast, simple to explain | 5× larger than the custom CNN; reaches 0.99 train accuracy, and scores vary by ~0.03 between runs |
| EfficientNet-B0 | Close to ResNet-18 (0.909) with 2.8× fewer parameters | 3.5× slower per image than ResNet-18 here (8.7 ms): parameter count does not predict speed |
