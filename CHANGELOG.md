# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Dataset built with `dvc repro` (`dvc.lock`), all 9 validation checks pass; majority-class baseline result.

### Changed
- Colab runner works without DagsHub (rebuilds data with `dvc repro`, saves results to Google Drive) and clones the private repository with a `GITHUB_TOKEN` secret.
- `make install-dev` and `bootstrap.sh` also install the serving dependencies (FastAPI, Gradio).

### Fixed
- Figshare downloads retry and resume after network errors; DVC keeps finished downloads (`persist`).
- Docker image installs `pandas`, `scikit-learn` and `tqdm`, needed by the inference code.
- `.env.example` defaults to local MLflow tracking instead of a placeholder DagsHub URL.
- Plain `pytest` can import the `api` package.
- JSON reports end with a newline, as expected by the pre-commit hook.

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
