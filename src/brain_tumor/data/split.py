"""Stage ``split``: assign every image to train / val / test **by patient**.

The dataset has 3064 slices from only 233 patients (~13 slices each). Splitting
images at random puts slices of the same patient in both training and test sets,
which inflates test scores (data leakage). Here patients are split, stratified by
tumour class, and each image inherits its patient's set.

Output ``data/splits.csv`` columns:

=========== ================================================================
image_id    image identifier (matches metadata.csv and processed/*.npy)
label       zero-based class index
patient_id  patient identifier
split       train | val | test — patient-level split (use this one)
fold        0..k-1 for train+val patients (patient-level CV), -1 for test
image_split train | val | test — image-level split, leaky, for comparison only
=========== ================================================================
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from brain_tumor.config import load_params, project_path
from brain_tumor.utils import get_logger

log = get_logger("split")


def patient_labels(meta: pd.DataFrame) -> pd.Series:
    """One label per patient; fails if a patient has several tumour types."""
    n_labels = meta.groupby("patient_id")["label"].nunique()
    if (n_labels > 1).any():
        bad = n_labels[n_labels > 1].index.tolist()
        raise ValueError(f"Patients with several labels (cannot stratify): {bad[:5]}")
    return meta.groupby("patient_id")["label"].first().sort_index()


def _check_fractions(train: float, val: float, test: float) -> None:
    if not np.isclose(train + val + test, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {train + val + test}")


def patient_split(
    meta: pd.DataFrame, train: float, val: float, test: float, seed: int
) -> pd.Series:
    """Return a Series (aligned with ``meta``) of train/val/test assigned per patient."""
    _check_fractions(train, val, test)
    labels = patient_labels(meta)
    train_p, rest_p = train_test_split(
        labels.index.to_numpy(), test_size=val + test, stratify=labels.values, random_state=seed
    )
    val_p, test_p = train_test_split(
        rest_p, test_size=test / (val + test), stratify=labels.loc[rest_p].values, random_state=seed
    )
    mapping = {
        **dict.fromkeys(train_p, "train"),
        **dict.fromkeys(val_p, "val"),
        **dict.fromkeys(test_p, "test"),
    }
    return meta["patient_id"].map(mapping)


def patient_folds(meta: pd.DataFrame, split: pd.Series, n_folds: int, seed: int) -> pd.Series:
    """Patient-level stratified K-fold over train+val images; test images get -1."""
    folds = pd.Series(-1, index=meta.index, dtype=int)
    pool = meta[split != "test"]
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(cv.split(pool, pool["label"], groups=pool["patient_id"])):
        folds.loc[pool.index[val_idx]] = k
    return folds


def image_split(meta: pd.DataFrame, train: float, val: float, test: float, seed: int) -> pd.Series:
    """Image-level stratified split that ignores patients (leaky, for comparison)."""
    _check_fractions(train, val, test)
    idx = meta.index.to_numpy()
    train_i, rest_i = train_test_split(
        idx, test_size=val + test, stratify=meta["label"], random_state=seed
    )
    val_i, test_i = train_test_split(
        rest_i,
        test_size=test / (val + test),
        stratify=meta.loc[rest_i, "label"],
        random_state=seed,
    )
    out = pd.Series("", index=meta.index, dtype=object)
    out.loc[train_i], out.loc[val_i], out.loc[test_i] = "train", "val", "test"
    return out


def make_splits(meta: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    s, seed = cfg["split"], cfg["seed"]
    splits = meta[["image_id", "label", "patient_id"]].copy()
    splits["split"] = patient_split(meta, s["train"], s["val"], s["test"], seed)
    splits["fold"] = patient_folds(meta, splits["split"], s["n_folds"], seed)
    splits["image_split"] = image_split(meta, s["train"], s["val"], s["test"], seed)
    return splits


def summarise(splits: pd.DataFrame, class_names: list[str]) -> None:
    for name in ("train", "val", "test"):
        part = splits[splits["split"] == name]
        counts = part["label"].value_counts().sort_index()
        pct = ", ".join(
            f"{class_names[i]} {counts.get(i, 0)} ({counts.get(i, 0) / len(part):.0%})"
            for i in range(len(class_names))
        )
        log.info(
            "%-5s %4d images, %3d patients | %s", name, len(part), part["patient_id"].nunique(), pct
        )
    test_patients = set(splits.loc[splits["image_split"] == "test", "patient_id"])
    train_patients = set(splits.loc[splits["image_split"] == "train", "patient_id"])
    leaked = splits[(splits["image_split"] == "test") & splits["patient_id"].isin(train_patients)]
    log.info(
        "Image-level split (comparison only): %d/%d test patients also in train, "
        "%.1f%% of test images leaked",
        len(test_patients & train_patients),
        len(test_patients),
        100 * len(leaked) / max(1, (splits["image_split"] == "test").sum()),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create data/splits.csv")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_intermixed_args(argv)
    cfg = load_params(overrides=args.overrides)
    meta = pd.read_csv(project_path(cfg, "metadata"), dtype={"patient_id": str})
    splits = make_splits(meta, cfg)
    out = project_path(cfg, "splits")
    splits.to_csv(out, index=False)
    summarise(splits, cfg["data"]["class_names"])
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
