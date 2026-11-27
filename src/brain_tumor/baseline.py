"""Majority-class baseline: always predict the most frequent training class.

Any useful model must clearly beat it, especially on macro-F1.
"""

from __future__ import annotations

import argparse
from collections import Counter

import mlflow
import numpy as np
import pandas as pd

from brain_tumor.config import load_params, project_path
from brain_tumor.data.datamodule import select_partitions
from brain_tumor.tracking import lineage_tags, log_metrics, setup_mlflow
from brain_tumor.training.metrics import compute_metrics
from brain_tumor.utils import get_logger, save_json

log = get_logger("baseline")


def majority_baseline(cfg: dict) -> dict:
    splits = pd.read_csv(project_path(cfg, "splits"), dtype={"patient_id": str})
    frames = select_partitions(splits, cfg["train"]["split_column"], cfg["train"].get("fold"))
    majority = Counter(frames["train"]["label"]).most_common(1)[0][0]
    y_true = frames["test"]["label"].to_numpy()
    metrics = compute_metrics(y_true, np.full_like(y_true, majority), cfg["data"]["class_names"])
    metrics["majority_class"] = cfg["data"]["class_names"][majority]
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_intermixed_args(argv)
    cfg = load_params(overrides=args.overrides)
    metrics = majority_baseline(cfg)
    save_json(metrics, project_path(cfg, "reports_dir") / "baseline.json")
    log.info(
        "Majority class: %s | test accuracy %.3f | test macro-F1 %.3f",
        metrics["majority_class"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    if not args.no_mlflow:
        setup_mlflow(cfg, "baseline")
        with mlflow.start_run(run_name="majority-class"):
            mlflow.set_tags({**lineage_tags(cfg), "model": "majority_class"})
            log_metrics(metrics, prefix="test_")


if __name__ == "__main__":
    main()
