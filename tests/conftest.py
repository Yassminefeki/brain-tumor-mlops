"""Shared fixtures: a small synthetic copy of the dataset, in the original .mat format.

The real data is never needed to run the tests (CI has no access to it). The synthetic
images contain a bright blob whose position depends on the class, so a tiny model can
learn it in a few steps.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from brain_tumor.config import apply_overrides, load_params

REPO_ROOT = Path(__file__).resolve().parents[1]
PATIENTS_PER_CLASS = 10
IMAGES_PER_PATIENT = 3
N_CLASSES = 3


def write_fake_mat(
    path: Path, image: np.ndarray, label_raw: int, pid: str, mask: np.ndarray
) -> None:
    """Write a file with the same HDF5 layout as the Figshare .mat files."""
    with h5py.File(path, "w") as f:
        g = f.create_group("cjdata")
        g.create_dataset("image", data=image.T.astype(np.int16))  # MATLAB column-major
        g.create_dataset("label", data=np.array([[label_raw]], dtype=np.float64))
        g.create_dataset("PID", data=np.array([[ord(c)] for c in pid], dtype=np.uint16))
        g.create_dataset("tumorMask", data=mask.T.astype(np.uint8))
        g.create_dataset("tumorBorder", data=np.zeros((1, 4)))


def fake_image(label: int, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    centres = {0: (0.3, 0.3), 1: (0.5, 0.5), 2: (0.7, 0.7)}
    cy, cx = (np.array(centres[label]) * size).astype(int)
    yy, xx = np.mgrid[:size, :size]
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < (size // 8) ** 2
    image = rng.normal(300, 60, (size, size)) + mask * 1500
    image[0, 0] = 9000  # outlier pixel, should be removed by percentile clipping
    return np.clip(image, 0, None), mask


@pytest.fixture(scope="session")
def synthetic_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project folder with params.yaml and data/interim/*.mat (synthetic)."""
    root = tmp_path_factory.mktemp("project")
    interim = root / "data" / "interim"
    interim.mkdir(parents=True)
    rng = np.random.default_rng(0)
    image_id = 1
    for label in range(N_CLASSES):
        for p in range(PATIENTS_PER_CLASS):
            pid = f"{label}{p:04d}"
            for _ in range(IMAGES_PER_PATIENT):
                size = 64 if image_id % 10 else 48  # a few images have another size
                image, mask = fake_image(label, size, rng)
                write_fake_mat(interim / f"{image_id}.mat", image, label + 1, pid, mask)
                image_id += 1

    n_images = N_CLASSES * PATIENTS_PER_CLASS * IMAGES_PER_PATIENT
    per_class = PATIENTS_PER_CLASS * IMAGES_PER_PATIENT
    cfg = load_params(REPO_ROOT / "params.yaml")
    cfg = apply_overrides(
        cfg,
        {
            "data.expected_images": n_images,
            "data.expected_patients": N_CLASSES * PATIENTS_PER_CLASS,
            "data.expected_class_counts": {
                "meningioma": per_class,
                "glioma": per_class,
                "pituitary": per_class,
            },
            "preprocess.image_size": 32,
            "split.n_folds": 3,
            "train.epochs": 2,
            "train.batch_size": 16,
            "train.num_workers": 0,
            "train.patience": 5,
            "train.warmup_epochs": 0,
            "models.custom_cnn.filters": [8, 16],
            "models.custom_cnn.dense_units": 16,
            "models.pretrained.weights": "none",
            "models.pretrained.epochs": 2,
            "models.pretrained.freeze_epochs": 1,
            "evaluate.gradcam_samples": 4,
            "tune.n_trials": 2,
            "tune.epochs": 1,
        },
    )
    cfg.pop("_root")
    with (root / "params.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return root


@pytest.fixture(scope="session")
def prepared_project(synthetic_project: Path) -> Path:
    """Synthetic project after the metadata, preprocess and split stages."""
    from brain_tumor.data import metadata, preprocess, split

    params = str(synthetic_project / "params.yaml")
    os.environ["BRAIN_TUMOR_PARAMS"] = params
    try:
        metadata.main([])
        preprocess.main([])
        split.main([])
    finally:
        os.environ.pop("BRAIN_TUMOR_PARAMS", None)
    return synthetic_project


@pytest.fixture
def project_cfg(prepared_project: Path) -> dict:
    return load_params(prepared_project / "params.yaml")


@pytest.fixture(scope="session", autouse=True)
def local_mlflow(tmp_path_factory: pytest.TempPathFactory):
    """Never log tests to a remote MLflow server: use a throw-away SQLite store.

    The working directory is moved to a temporary folder too, because MLflow writes
    artifacts to ``./mlruns`` relative to it.
    """
    workdir = tmp_path_factory.mktemp("mlflow")
    previous = os.environ.get("MLFLOW_TRACKING_URI")
    previous_cwd = os.getcwd()
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{workdir / 'mlflow.db'}"
    os.chdir(workdir)
    yield
    os.chdir(previous_cwd)
    if previous is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = previous
