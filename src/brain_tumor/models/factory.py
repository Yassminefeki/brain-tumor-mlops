"""Single entry point to build any model of the study from the configuration."""

from __future__ import annotations

from typing import Any

from torch import nn

from brain_tumor.models import pretrained as pt
from brain_tumor.models.custom_cnn import CustomCNN

CUSTOM_MODELS = ("custom_cnn",)
AVAILABLE_MODELS = CUSTOM_MODELS + pt.SUPPORTED


def is_pretrained(name: str) -> bool:
    return name in pt.SUPPORTED


def build_model(
    cfg: dict[str, Any], name: str | None = None, weights: str | None = None
) -> nn.Module:
    """Build ``name`` (default ``cfg['train']['model']``).

    ``weights`` overrides the configured pretrained weights, e.g. ``"none"`` when the
    weights are about to be loaded from a checkpoint anyway.
    """
    name = name or cfg["train"]["model"]
    n_classes = len(cfg["data"]["class_names"])
    if name == "custom_cnn":
        c = cfg["models"]["custom_cnn"]
        model = CustomCNN(
            in_channels=1,
            num_classes=n_classes,
            filters=c["filters"],
            convs_per_block=c["convs_per_block"],
            kernel_size=c["kernel_size"],
            batch_norm=c["batch_norm"],
            global_pool=c["global_pool"],
            dense_units=c["dense_units"],
            dropout_before_dense=c["dropout_before_dense"],
            dropout_after_dense=c["dropout_after_dense"],
            input_size=cfg["preprocess"]["image_size"],
        )
    elif is_pretrained(name):
        p = cfg["models"]["pretrained"]
        model = pt.build_pretrained(name, n_classes, weights or p["weights"], p["dropout"])
    else:
        raise ValueError(f"Unknown model {name!r}; available: {AVAILABLE_MODELS}")
    model.model_name = name  # type: ignore[attr-defined]
    return model


def gradcam_target(model: nn.Module, name: str) -> nn.Module:
    if name == "custom_cnn":
        return model.gradcam_layer  # type: ignore[union-attr]
    return pt.gradcam_layer(model, name)
