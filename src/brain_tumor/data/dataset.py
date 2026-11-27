"""PyTorch dataset reading the preprocessed ``.npy`` arrays."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    """Returns ``(image, label)`` with image a float tensor of shape (C, H, W).

    With ``preload=True`` (default) all arrays are loaded once into RAM as float16
    (~300 MB for the full dataset at 224 px), which removes disk I/O from training.
    """

    def __init__(
        self,
        image_ids: Sequence[int],
        labels: Sequence[int],
        images_dir: str | Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        preload: bool = True,
    ) -> None:
        if len(image_ids) != len(labels):
            raise ValueError("image_ids and labels must have the same length")
        self.image_ids = [int(i) for i in image_ids]
        self.labels = [int(y) for y in labels]
        self.images_dir = Path(images_dir)
        self.transform = transform
        self._cache: np.ndarray | None = None
        if preload and self.image_ids:
            self._cache = np.stack([self._load(i) for i in self.image_ids]).astype(np.float16)

    def _load(self, image_id: int) -> np.ndarray:
        path = self.images_dir / f"{image_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `dvc pull` or `dvc repro preprocess`")
        return np.load(path)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        arr = self._cache[index] if self._cache is not None else self._load(self.image_ids[index])
        image = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
