"""Image transforms applied on the fly to tensors of shape ``(1, H, W)`` in [0, 1].

* Training: light, anatomy-preserving augmentation (flip, small rotation/zoom/shift,
  slight brightness/contrast). No vertical flips or strong distortions.
* Evaluation: no augmentation.
* Pretrained models: grayscale is repeated to 3 channels and normalised with the
  ImageNet statistics their weights expect.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RepeatChannels(nn.Module):
    """(1, H, W) -> (n, H, W). A module (not a lambda) so it pickles for DataLoader workers."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(self.n, *x.shape[1:]).clone() if x.shape[0] == 1 else x


class ClampUnit(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp_(0.0, 1.0)


def build_transform(cfg: dict[str, Any], train: bool, pretrained: bool) -> nn.Module:
    ops: list[nn.Module] = []
    if train and cfg["train"]["augment"]:
        a = cfg["augment"]
        ops += [
            v2.RandomHorizontalFlip(p=a["hflip_p"]),
            v2.RandomAffine(
                degrees=a["rotation_degrees"],
                translate=(a["translate"], a["translate"]),
                scale=tuple(a["scale"]),
            ),
            v2.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
            ClampUnit(),
        ]
    if pretrained:
        ops += [RepeatChannels(3), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(ops) if ops else nn.Identity()


def denormalize(x: torch.Tensor, pretrained: bool) -> torch.Tensor:
    """Back to a displayable (H, W) image in [0, 1]."""
    if pretrained:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
        x = x * std + mean
    return x[0].clamp(0, 1)
