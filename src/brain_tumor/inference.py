"""Prediction on new images, shared by the API and the demo.

Accepts the original Figshare ``.mat`` files or ordinary PNG/JPEG images, and applies
exactly the preprocessing used for training (stored in the checkpoint).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from brain_tumor.data.mat_io import read_mat
from brain_tumor.data.preprocess import preprocess_image
from brain_tumor.data.transforms import build_transform
from brain_tumor.evaluate import load_checkpoint
from brain_tumor.explain.gradcam import GradCAM, overlay
from brain_tumor.models.factory import gradcam_target
from brain_tumor.utils import get_device


def load_image_bytes(data: bytes, filename: str = "") -> np.ndarray:
    """Decode an uploaded file into a 2-D grayscale array of raw intensities."""
    if filename.lower().endswith(".mat"):
        return read_mat(data, load_mask=False).image
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32)
    except Exception as exc:  # PIL raises several exception types
        raise ValueError(f"Unsupported or corrupted image file {filename!r}") from exc


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        self.device = get_device(device)
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.model_name: str = ckpt["model_name"]
        self.class_names: list[str] = list(ckpt["class_names"])
        self.image_size: int = ckpt["image_size"]
        self.clip_percentiles = ckpt.get("clip_percentiles", (0.5, 99.5))
        self.transform = build_transform(ckpt["config"], train=False, pretrained=ckpt["pretrained"])
        self.info = {
            k: ckpt.get(k) for k in ("model_name", "best_epoch", "best_val_macro_f1", "image_size")
        }

    def prepare(self, image: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """Raw 2-D image -> (preprocessed [0, 1] array, model input tensor (1, C, H, W))."""
        if image.ndim != 2:
            raise ValueError(f"Expected a 2-D grayscale image, got shape {image.shape}")
        clean = preprocess_image(image, self.image_size, self.clip_percentiles)
        x = self.transform(torch.from_numpy(clean).unsqueeze(0)).unsqueeze(0).to(self.device)
        return clean, x

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict[str, Any]:
        _, x = self.prepare(image)
        probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu().numpy()
        best = int(probs.argmax())
        return {
            "prediction": self.class_names[best],
            "confidence": float(probs[best]),
            "probabilities": {n: float(p) for n, p in zip(self.class_names, probs, strict=True)},
            "model": self.model_name,
        }

    def explain(self, image: np.ndarray) -> np.ndarray:
        """RGB Grad-CAM overlay (H, W, 3) in [0, 1] for the predicted class."""
        clean, x = self.prepare(image)
        with GradCAM(self.model, gradcam_target(self.model, self.model_name)) as cam_fn:
            cam, _ = cam_fn(x)
        return overlay(clean, cam)
