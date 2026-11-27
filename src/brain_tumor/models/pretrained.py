"""Models B/C: ImageNet-pretrained backbones adapted to 3-class MRI classification.

For every backbone this module knows three parameter groups, used for two-stage
fine-tuning with discriminative learning rates:

* ``early`` — generic low-level layers (edges, textures): smallest learning rate;
* ``late``  — high-level layers: medium learning rate;
* ``head``  — the new classifier (Dropout -> Linear(3)): largest learning rate.
"""

from __future__ import annotations

from torch import nn
from torchvision import models as tvm

SUPPORTED = ("resnet18", "resnet50", "efficientnet_b0", "densenet121")

_BUILDERS = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "densenet121": (tvm.densenet121, tvm.DenseNet121_Weights.IMAGENET1K_V1),
}


def build_pretrained(
    name: str, num_classes: int = 3, weights: str = "imagenet", dropout: float = 0.4
) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown pretrained model {name!r}; choose from {SUPPORTED}")
    builder, imagenet_weights = _BUILDERS[name]
    model = builder(weights=imagenet_weights if weights == "imagenet" else None)

    if name.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def param_groups(model: nn.Module, name: str) -> dict[str, list[nn.Module]]:
    """Split the model's modules into early / late / head groups."""
    if name.startswith("resnet"):
        return {
            "early": [model.conv1, model.bn1, model.layer1, model.layer2],
            "late": [model.layer3, model.layer4],
            "head": [model.fc],
        }
    if name.startswith("efficientnet"):
        feats = list(model.features.children())
        return {"early": feats[:4], "late": feats[4:], "head": [model.classifier]}
    if name.startswith("densenet"):
        feats = dict(model.features.named_children())
        early_names = [
            "conv0",
            "norm0",
            "relu0",
            "pool0",
            "denseblock1",
            "transition1",
            "denseblock2",
            "transition2",
        ]
        return {
            "early": [feats[n] for n in early_names],
            "late": [m for n, m in feats.items() if n not in early_names],
            "head": [model.classifier],
        }
    raise ValueError(f"No parameter groups defined for {name!r}")


def backbone_modules(model: nn.Module, name: str) -> list[nn.Module]:
    groups = param_groups(model, name)
    return groups["early"] + groups["late"]


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    for module in backbone_modules(model, name):
        for p in module.parameters():
            p.requires_grad = trainable


def freeze_backbone_batchnorm(model: nn.Module, name: str) -> None:
    """Keep pretrained BatchNorm statistics fixed while the backbone is frozen.

    Call after ``model.train()``: small batches would otherwise overwrite the
    ImageNet running statistics.
    """
    for module in backbone_modules(model, name):
        for m in module.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("efficientnet"):
        return model.features[-1]
    if name.startswith("densenet"):
        return model.features.denseblock4
    raise ValueError(f"No Grad-CAM layer defined for {name!r}")
