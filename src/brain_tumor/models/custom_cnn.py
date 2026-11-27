"""Model A: a CNN designed and trained from scratch.

Default architecture (224x224x1 input, ~2.39M parameters)::

    Block 1  2 x [Conv3x3(32)  -> BN -> ReLU] -> MaxPool2   224 -> 112
    Block 2  2 x [Conv3x3(64)  -> BN -> ReLU] -> MaxPool2   112 -> 56
    Block 3  2 x [Conv3x3(128) -> BN -> ReLU] -> MaxPool2    56 -> 28
    Block 4  2 x [Conv3x3(256) -> BN -> ReLU] -> MaxPool2    28 -> 14
    Block 5  2 x [Conv3x3(256) -> BN -> ReLU] -> MaxPool2    14 -> 7
    Global average pooling -> 256
    Dropout(0.5) -> Dense(128) -> ReLU -> Dropout(0.3) -> Dense(3)

Design choices (see docs/models.md): stacked 3x3 convolutions, filters doubled when
resolution halves, BatchNorm for stable training, GAP instead of Flatten to limit
parameters, and a receptive field (~156 px) large enough to cover a whole tumour.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_convs: int = 2,
        kernel_size: int = 3,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_convs):
            layers.append(
                nn.Conv2d(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=not batch_norm,
                )
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=False))  # not in-place: Grad-CAM hooks need it
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)

    @property
    def last_activation(self) -> nn.Module:
        return self.convs[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.convs(x))


class CustomCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        filters: Sequence[int] = (32, 64, 128, 256, 256),
        convs_per_block: int = 2,
        kernel_size: int = 3,
        batch_norm: bool = True,
        global_pool: str = "gap",
        dense_units: int = 128,
        dropout_before_dense: float = 0.5,
        dropout_after_dense: float = 0.3,
        input_size: int = 224,
    ) -> None:
        super().__init__()
        if global_pool not in ("gap", "flatten"):
            raise ValueError("global_pool must be 'gap' or 'flatten'")
        blocks, ch = [], in_channels
        for out_ch in filters:
            blocks.append(ConvBlock(ch, out_ch, convs_per_block, kernel_size, batch_norm))
            ch = out_ch
        self.features = nn.Sequential(*blocks)

        if global_pool == "gap":
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            n_features = ch
        else:
            spatial = input_size // 2 ** len(filters)
            if spatial < 1:
                raise ValueError("Too many blocks for this input size")
            self.pool = nn.Flatten()
            n_features = ch * spatial * spatial

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_before_dense),
            nn.Linear(n_features, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_after_dense),
            nn.Linear(dense_units, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last ReLU of the last block (7x7 feature maps by default)."""
        return self.features[-1].last_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def receptive_field(n_blocks: int, convs_per_block: int = 2, kernel_size: int = 3) -> int:
    """Theoretical receptive field (pixels) of the last conv layer's output."""
    rf, jump = 1, 1
    for _ in range(n_blocks):
        for _ in range(convs_per_block):
            rf += (kernel_size - 1) * jump
        rf += jump  # 2x2 max-pool
        jump *= 2
    return rf
