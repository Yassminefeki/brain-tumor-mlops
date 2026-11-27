"""Build train / val / test DataLoaders from ``splits.csv`` and the configuration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from brain_tumor.config import project_path
from brain_tumor.data.dataset import BrainTumorDataset
from brain_tumor.data.transforms import build_transform
from brain_tumor.utils import get_logger, seed_worker

log = get_logger("datamodule")


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_weights: torch.Tensor | None
    frames: dict[str, pd.DataFrame]


def select_partitions(
    splits: pd.DataFrame, split_column: str = "split", fold: int | None = None
) -> dict[str, pd.DataFrame]:
    """Return the train/val/test rows.

    * ``fold=None``: use ``split_column`` (``split`` = patient-level, ``image_split`` = leaky).
    * ``fold=k``: patient-level CV — val is fold k, train is every other non-test fold,
      test stays the fixed held-out test set.
    """
    if fold is None:
        if split_column not in splits.columns:
            raise KeyError(f"Column {split_column!r} not in splits.csv")
        return {name: splits[splits[split_column] == name] for name in ("train", "val", "test")}
    pool = splits[splits["split"] != "test"]
    if fold not in set(pool["fold"]):
        raise ValueError(f"Fold {fold} does not exist (available: {sorted(set(pool['fold']))})")
    return {
        "train": pool[pool["fold"] != fold],
        "val": pool[pool["fold"] == fold],
        "test": splits[splits["split"] == "test"],
    }


def compute_class_weights(labels: list[int] | pd.Series, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights ``N / (K * n_k)`` (balanced classes -> all ones)."""
    counts = Counter(int(y) for y in labels)
    total = sum(counts.values())
    return torch.tensor(
        [total / (n_classes * max(counts.get(k, 0), 1)) for k in range(n_classes)],
        dtype=torch.float32,
    )


def build_dataloaders(cfg: dict[str, Any], pretrained: bool, preload: bool = True) -> DataBundle:
    t = cfg["train"]
    splits = pd.read_csv(project_path(cfg, "splits"), dtype={"patient_id": str})
    frames = select_partitions(splits, t["split_column"], t.get("fold"))
    images_dir = project_path(cfg, "processed_dir") / "images"

    def make(name: str, train: bool) -> BrainTumorDataset:
        df = frames[name]
        return BrainTumorDataset(
            df["image_id"].tolist(),
            df["label"].tolist(),
            images_dir,
            build_transform(cfg, train=train, pretrained=pretrained),
            preload,
        )

    generator = torch.Generator().manual_seed(cfg["seed"])
    common = {
        "batch_size": t["batch_size"],
        "num_workers": t["num_workers"],
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": t["num_workers"] > 0,
    }
    train_loader = DataLoader(
        make("train", True),
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(make("val", False), shuffle=False, **common)
    test_loader = DataLoader(make("test", False), shuffle=False, **common)

    n_classes = len(cfg["data"]["class_names"])
    weights = (
        compute_class_weights(frames["train"]["label"], n_classes) if t["class_weights"] else None
    )
    log.info(
        "Data: train %d | val %d | test %d images (split=%s, fold=%s)",
        len(frames["train"]),
        len(frames["val"]),
        len(frames["test"]),
        t["split_column"],
        t.get("fold"),
    )
    return DataBundle(train_loader, val_loader, test_loader, weights, frames)
