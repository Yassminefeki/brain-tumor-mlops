"""Data pipeline: .mat reading, preprocessing, patient-level split, validation."""

import numpy as np
import pandas as pd
import pytest

from brain_tumor.config import project_path
from brain_tumor.data.mat_io import decode_pid, read_mat
from brain_tumor.data.preprocess import normalize_intensity, preprocess_image
from brain_tumor.data.split import image_split, make_splits, patient_split
from brain_tumor.data.validate import check_no_patient_overlap, run_checks


def test_read_mat_restores_orientation_and_decodes(synthetic_project):
    rec = read_mat(synthetic_project / "data" / "interim" / "1.mat")
    assert rec.label_raw == 1 and rec.label == 0
    assert rec.patient_id == "00000"
    assert rec.image.shape == rec.mask.shape
    # blob of class 0 is near (0.3, 0.3): the mask must be there after transposition
    ys, xs = np.nonzero(rec.mask)
    assert abs(ys.mean() - xs.mean()) < 2


def test_decode_pid_ignores_padding():
    assert decode_pid(np.array([[49], [50], [0]])) == "12"


def test_normalize_removes_outliers_and_scales():
    img = np.full((50, 50), 100.0)
    img[10:20, 10:20] = 500
    img[0, 0] = 1e6  # outlier
    out = normalize_intensity(img, (0.5, 99.5))
    assert out.min() == 0 and out.max() == 1
    assert out[15, 15] == 1  # tumour stays bright despite the outlier


def test_preprocess_image_shape_and_range():
    out = preprocess_image(np.random.default_rng(0).integers(0, 5000, (512, 512)), 224)
    assert out.shape == (224, 224) and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1
    assert np.all(preprocess_image(np.ones((64, 64)), 32) == 0)  # constant image is safe


def test_metadata_and_processed_outputs(project_cfg):
    meta = pd.read_csv(project_path(project_cfg, "metadata"), dtype={"patient_id": str})
    assert len(meta) == project_cfg["data"]["expected_images"]
    assert set(meta["label"]) == {0, 1, 2}
    images = project_path(project_cfg, "processed_dir") / "images"
    arr = np.load(images / f"{meta['image_id'].iloc[0]}.npy")
    assert arr.shape == (32, 32) and arr.dtype == np.float16


def test_patient_split_has_no_leakage_and_is_deterministic(project_cfg):
    meta = pd.read_csv(project_path(project_cfg, "metadata"), dtype={"patient_id": str})
    a = patient_split(meta, 0.7, 0.15, 0.15, seed=1)
    b = patient_split(meta, 0.7, 0.15, 0.15, seed=1)
    assert a.equals(b)
    frame = meta.assign(split=a)
    assert check_no_patient_overlap(frame)[0]
    assert frame.groupby("split")["label"].nunique().min() == 3


def test_image_split_leaks_patients(project_cfg):
    meta = pd.read_csv(project_path(project_cfg, "metadata"), dtype={"patient_id": str})
    leaky = meta.assign(split=image_split(meta, 0.7, 0.15, 0.15, seed=0))
    assert not check_no_patient_overlap(leaky)[0]


def test_folds_exclude_test_and_group_patients(project_cfg):
    meta = pd.read_csv(project_path(project_cfg, "metadata"), dtype={"patient_id": str})
    splits = make_splits(meta, project_cfg)
    assert (splits.loc[splits["split"] == "test", "fold"] == -1).all()
    pool = splits[splits["split"] != "test"]
    assert pool.groupby("patient_id")["fold"].nunique().max() == 1
    assert set(pool["fold"]) == set(range(project_cfg["split"]["n_folds"]))


def test_patient_with_two_labels_is_rejected():
    meta = pd.DataFrame({"patient_id": ["a", "a", "b"], "label": [0, 1, 0]})
    with pytest.raises(ValueError):
        patient_split(meta, 0.7, 0.15, 0.15, 0)


def test_validation_stage_passes_on_clean_data(project_cfg):
    results = run_checks(project_cfg)
    failed = {k: v["detail"] for k, v in results.items() if not v["passed"]}
    assert not failed, failed
