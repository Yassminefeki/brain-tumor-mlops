"""Grad-CAM (Selvaraju et al., 2017): where does the network look?

The gradients of the predicted class score with respect to a convolutional layer's
feature maps are averaged into channel weights; the weighted sum of the maps,
passed through ReLU and upsampled, gives a heatmap over the input image.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()
        if output.requires_grad:
            output.register_hook(lambda grad: setattr(self, "_gradients", grad.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """``x``: (1, C, H, W). Returns (heatmap (H, W) in [0, 1], explained class)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire: check the target layer")
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), class_idx

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a grayscale image (H, W) in [0, 1] with a jet-coloured heatmap -> RGB."""
    heat = plt.get_cmap("jet")(cam)[..., :3]
    gray = np.repeat(image[..., None], 3, axis=-1)
    return np.clip((1 - alpha) * gray + alpha * heat, 0, 1)


def gradcam_grid(
    images: Sequence[np.ndarray],
    cams: Sequence[np.ndarray],
    titles: Sequence[str],
    path: str | Path,
    masks: Sequence[np.ndarray | None] | None = None,
) -> Path:
    """Two rows: original (with tumour contour if a mask is given) / Grad-CAM overlay."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(images[i], cmap="gray")
        if masks is not None and masks[i] is not None and masks[i].any():
            axes[0, i].contour(masks[i], levels=[0.5], colors="lime", linewidths=0.8)
        axes[0, i].set_title(titles[i], fontsize=8)
        axes[1, i].imshow(overlay(images[i], cams[i]))
        for ax in axes[:, i]:
            ax.axis("off")
    axes[0, 0].set_ylabel("image + mask")
    axes[1, 0].set_ylabel("Grad-CAM")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
