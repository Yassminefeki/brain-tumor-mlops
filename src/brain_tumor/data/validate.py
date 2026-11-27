"""Stage ``validate``: data quality gate before any training.

Every check returns ``(passed, detail)``. Results are written to
``reports/data_validation.json`` (tracked as a DVC metric) and the stage exits with
an error if any check fails, which stops ``dvc repro`` before GPU time is spent.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brain_tumor.config import load_params, project_path
from brain_tumor.utils import get_logger, save_json

log = get_logger("validate")

Check = tuple[bool, str]


def check_counts(meta: pd.DataFrame, data_cfg: dict[str, Any]) -> Check:
    n_img, n_pat = len(meta), meta["patient_id"].nunique()
    ok = n_img == data_cfg["expected_images"] and n_pat == data_cfg["expected_patients"]
    return (
        ok,
        f"{n_img} images (expected {data_cfg['expected_images']}), "
        f"{n_pat} patients (expected {data_cfg['expected_patients']})",
    )


def check_class_counts(meta: pd.DataFrame, data_cfg: dict[str, Any]) -> Check:
    actual = meta["class_name"].value_counts().to_dict()
    expected = data_cfg["expected_class_counts"]
    return actual == expected, f"actual {actual}, expected {expected}"


def check_one_label_per_patient(meta: pd.DataFrame) -> Check:
    bad = int((meta.groupby("patient_id")["label"].nunique() > 1).sum())
    return bad == 0, f"{bad} patients with more than one label"


def check_unique_ids(splits: pd.DataFrame) -> Check:
    dup = int(splits["image_id"].duplicated().sum())
    return dup == 0, f"{dup} duplicated image ids"


def check_all_assigned(splits: pd.DataFrame) -> Check:
    missing = int((~splits["split"].isin(["train", "val", "test"])).sum())
    return missing == 0, f"{missing} images without a valid split"


def check_no_patient_overlap(splits: pd.DataFrame, column: str = "split") -> Check:
    sets = {s: set(splits.loc[splits[column] == s, "patient_id"]) for s in ("train", "val", "test")}
    overlaps = {
        f"{a}/{b}": len(sets[a] & sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return sum(overlaps.values()) == 0, f"shared patients {overlaps}"


def check_all_classes_in_each_split(splits: pd.DataFrame, n_classes: int) -> Check:
    per_split = splits.groupby("split")["label"].nunique().to_dict()
    return all(v == n_classes for v in per_split.values()), f"classes per split {per_split}"


def check_folds(splits: pd.DataFrame) -> Check:
    pool = splits[splits["split"] != "test"]
    test_ok = bool((splits.loc[splits["split"] == "test", "fold"] == -1).all())
    patients_per_fold = pool.groupby("patient_id")["fold"].nunique()
    ok = test_ok and bool((pool["fold"] >= 0).all()) and bool((patients_per_fold == 1).all())
    return (
        ok,
        f"{pool['fold'].nunique()} folds, test excluded: {test_ok}, "
        f"each patient in exactly one fold: {bool((patients_per_fold == 1).all())}",
    )


def check_processed_arrays(splits: pd.DataFrame, images_dir: Path, image_size: int) -> Check:
    problems: list[str] = []
    for image_id in splits["image_id"]:
        path = images_dir / f"{image_id}.npy"
        if not path.exists():
            problems.append(f"{image_id}: missing")
            continue
        arr = np.load(path)
        if arr.shape != (image_size, image_size):
            problems.append(f"{image_id}: shape {arr.shape}")
        elif not np.isfinite(arr).all():
            problems.append(f"{image_id}: non-finite values")
        elif arr.min() < 0 or arr.max() > 1:
            problems.append(f"{image_id}: range [{arr.min():.3f}, {arr.max():.3f}]")
        if len(problems) >= 10:
            break
    return not problems, "all arrays OK" if not problems else "; ".join(problems)


def run_checks(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    meta = pd.read_csv(project_path(cfg, "metadata"), dtype={"patient_id": str})
    splits = pd.read_csv(project_path(cfg, "splits"), dtype={"patient_id": str})
    images_dir = project_path(cfg, "processed_dir") / "images"
    n_classes = len(cfg["data"]["class_names"])

    checks: dict[str, Callable[[], Check]] = {
        "counts": lambda: check_counts(meta, cfg["data"]),
        "class_counts": lambda: check_class_counts(meta, cfg["data"]),
        "one_label_per_patient": lambda: check_one_label_per_patient(meta),
        "unique_image_ids": lambda: check_unique_ids(splits),
        "all_images_assigned": lambda: check_all_assigned(splits),
        "no_patient_overlap": lambda: check_no_patient_overlap(splits),
        "all_classes_in_each_split": lambda: check_all_classes_in_each_split(splits, n_classes),
        "patient_level_folds": lambda: check_folds(splits),
        "processed_arrays": lambda: check_processed_arrays(
            splits, images_dir, cfg["preprocess"]["image_size"]
        ),
    }
    results = {}
    for name, fn in checks.items():
        passed, detail = fn()
        results[name] = {"passed": bool(passed), "detail": detail}
        (log.info if passed else log.error)(
            "%-26s %s  %s", name, "PASS" if passed else "FAIL", detail
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the prepared dataset")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_intermixed_args(argv)
    cfg = load_params(overrides=args.overrides)
    results = run_checks(cfg)
    n_failed = sum(not r["passed"] for r in results.values())
    report = {
        "passed": n_failed == 0,
        "n_checks": len(results),
        "n_failed": n_failed,
        "checks": results,
    }
    save_json(report, project_path(cfg, "reports_dir") / "data_validation.json")
    if n_failed:
        log.error("%d check(s) failed: fix the data before training", n_failed)
        return 1
    log.info("All %d checks passed", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
