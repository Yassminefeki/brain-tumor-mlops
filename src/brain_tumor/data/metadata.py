"""Stage ``metadata``: build one table describing every image.

Output ``data/metadata.csv`` columns:

========== ==========================================================
image_id   integer file stem (``123.mat`` -> 123)
file       path of the .mat file relative to the interim directory
label      zero-based class index (0 meningioma, 1 glioma, 2 pituitary)
class_name human-readable class
patient_id decoded patient identifier (used for the patient-level split)
height     original image height
width      original image width
min / max  original intensity range
mean       original mean intensity
tumor_area number of tumour pixels in the mask (original resolution)
========== ==========================================================
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from brain_tumor.config import load_params, project_path
from brain_tumor.data.mat_io import read_mat
from brain_tumor.utils import get_logger

log = get_logger("metadata")


def build_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    interim_dir = project_path(cfg, "interim_dir")
    class_names = cfg["data"]["class_names"]
    files = sorted(interim_dir.glob("*.mat"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"No .mat files in {interim_dir}; run the extract stage first")

    rows = []
    for path in tqdm(files, desc="metadata", leave=False):
        rec = read_mat(path)
        rows.append(
            {
                "image_id": int(path.stem),
                "file": path.name,
                "label": rec.label,
                "class_name": class_names[rec.label],
                "patient_id": rec.patient_id,
                "height": rec.image.shape[0],
                "width": rec.image.shape[1],
                "min": float(rec.image.min()),
                "max": float(rec.image.max()),
                "mean": float(np.mean(rec.image)),
                "tumor_area": int(rec.mask.sum()) if rec.mask is not None else -1,
            }
        )
    return pd.DataFrame(rows)


def summarise(meta: pd.DataFrame) -> None:
    log.info("Images: %d | patients: %d", len(meta), meta["patient_id"].nunique())
    for name, group in meta.groupby("class_name"):
        log.info(
            "  %-11s %5d images  %4d patients", name, len(group), group["patient_id"].nunique()
        )
    sizes = meta.groupby(["height", "width"]).size().to_dict()
    log.info("Image sizes: %s", sizes)
    log.info("Intensity range: %.0f .. %.0f", meta["min"].min(), meta["max"].max())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build data/metadata.csv")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_intermixed_args(argv)
    cfg = load_params(overrides=args.overrides)
    meta = build_metadata(cfg)
    out = project_path(cfg, "metadata")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    summarise(meta)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
