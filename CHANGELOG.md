# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Dataset built with `dvc repro` (`dvc.lock`), all 9 validation checks pass; majority-class baseline result.
- Results of the Colab T4 run of 2026-11-27 in the README and the model card: ResNet-18 is the best model (test macro-F1 0.916; 0.920 ± 0.018 over 5 folds), custom CNN 0.862 (0.883 tuned), EfficientNet-B0 0.909.
- Executed Colab notebooks kept with their outputs (`notebooks/01_eda.ipynb`, `notebooks/runs/`), EDA figures in `reports/figures/`, tuned custom CNN parameters in `configs/best_params/`.
- Colab runner section 3b restores the MLflow runs of earlier sessions from Google Drive.
- Per-class test recall for every model in the README and the model card (read from MLflow).

### Changed
- Served model is the ResNet-18 run with the best validation macro-F1 (`resnet18-aug`, test macro-F1 0.916) instead of the registered second run (0.888), following the project's selection rule.
- Colab runner works without DagsHub (rebuilds data with `dvc repro`, saves results to Google Drive) and clones the private repository with a `GITHUB_TOKEN` secret.
- `make install-dev` and `bootstrap.sh` also install the serving dependencies (FastAPI, Gradio).

### Fixed
- Figshare downloads retry and resume after network errors; DVC keeps finished downloads (`persist`).
- Docker image installs `pandas`, `scikit-learn` and `tqdm`, needed by the inference code.
- `.env.example` defaults to local MLflow tracking instead of a placeholder DagsHub URL.
- Plain `pytest` can import the `api` package.
- JSON reports end with a newline, as expected by the pre-commit hook.
- `.gitignore` covers every `mlflow*.db`, not only `mlflow.db`.

## [0.1.0] - 2026-11-27

### Added
- Reproducible DVC data pipeline: download, extract, metadata, preprocess, patient-level split, validation.
- Models: custom CNN (configurable depth, BatchNorm, GAP) and pretrained ResNet-18/50, EfficientNet-B0, DenseNet-121.
- Training engine with two-stage fine-tuning, discriminative learning rates, AMP, early stopping on validation macro-F1.
- MLflow tracking (params, git commit, data version, per-epoch metrics, artifacts) and model registry support.
- Optuna hyperparameter search with median pruning and nested MLflow runs.
- Evaluation with test-time augmentation, confusion matrix and Grad-CAM.
- FastAPI inference service, Gradio demo and Docker image.
- Unit tests on synthetic data, GitHub Actions CI, pre-commit hooks.
- `scripts/bootstrap.sh` for one-command local setup.
- Documentation: architecture, data pipeline, models, training, MLOps setup, experiments, dataset and model cards.
