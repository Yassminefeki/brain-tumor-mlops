"""Stage ``preprocess``: turn raw MRI slices into model-ready arrays.

For every image, in this order:

1. orientation fix (done in :func:`brain_tumor.data.mat_io.read_mat`);
2. robust intensity clipping at the configured percentiles (default 0.5 / 99.5);
3. min-max scaling to [0, 1];
4. resizing to ``image_size`` x ``image_size`` (area interpolation when shrinking);
5. saving as ``.npy`` (float16 by default) in ``data/processed/images``.

Tumour masks are resized with nearest-neighbour interpolation and saved in
``data/processed/masks``; they are used only to sanity-check Grad-CAM, never for training.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from brain_tumor.config import load_params, project_path
from brain_tumor.data.mat_io import read_mat
from brain_tumor.utils import get_logger, save_json

log = get_logger("preprocess")


def normalize_intensity(
    image: np.ndarray, clip_percentiles: Sequence[float] = (0.5, 99.5)
) -> np.ndarray:
    """Clip to percentiles then min-max scale to [0, 1] (float32)."""
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, clip_percentiles)
    if hi <= lo:  # constant image: avoid division by zero
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo)


def resize(image: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    if image.shape == (size, size):
        return image
    if is_mask:
        interpolation = cv2.INTER_NEAREST
    elif image.shape[0] > size:
        interpolation = cv2.INTER_AREA  # best quality when shrinking
    else:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def preprocess_image(
    image: np.ndarray, image_size: int, clip_percentiles: Sequence[float] = (0.5, 99.5)
) -> np.ndarray:
    """Full image transform used both by the pipeline and at inference time."""
    out = resize(normalize_intensity(image, clip_percentiles), image_size)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def preprocess_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    return resize(mask.astype(np.uint8), image_size, is_mask=True)


def run(cfg: dict[str, Any]) -> None:
    p = cfg["preprocess"]
    interim_dir = project_path(cfg, "interim_dir")
    out_dir = project_path(cfg, "processed_dir")
    images_dir, masks_dir = out_dir / "images", out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    if p["save_masks"]:
        masks_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(project_path(cfg, "metadata"))
    dtype = np.dtype(p["dtype"])
    for row in tqdm(meta.itertuples(), total=len(meta), desc="preprocess", leave=False):
        rec = read_mat(interim_dir / row.file, load_mask=p["save_masks"])
        image = preprocess_image(rec.image, p["image_size"], p["clip_percentiles"])
        np.save(images_dir / f"{row.image_id}.npy", image.astype(dtype))
        if p["save_masks"] and rec.mask is not None:
            np.save(masks_dir / f"{row.image_id}.npy", preprocess_mask(rec.mask, p["image_size"]))

    save_json(
        {
            "n_images": len(meta),
            "image_size": p["image_size"],
            "dtype": str(dtype),
            "clip_percentiles": list(p["clip_percentiles"]),
            "masks": bool(p["save_masks"]),
        },
        out_dir / "manifest.json",
    )
    log.info("Preprocessed %d images -> %s", len(meta), out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess images into data/processed")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_intermixed_args(argv)
    run(load_params(overrides=args.overrides))


if __name__ == "__main__":
    main()
