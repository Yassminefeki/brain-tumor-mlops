"""Model shapes, parameter counts, Grad-CAM and the training plan."""

from pathlib import Path

import pytest
import torch

from brain_tumor.config import apply_overrides, load_params
from brain_tumor.data.datamodule import compute_class_weights
from brain_tumor.data.transforms import build_transform
from brain_tumor.explain.gradcam import GradCAM
from brain_tumor.models.custom_cnn import CustomCNN, receptive_field
from brain_tumor.models.factory import build_model, gradcam_target
from brain_tumor.models.pretrained import SUPPORTED, param_groups
from brain_tumor.training.engine import training_plan
from brain_tumor.utils import count_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return apply_overrides(
        load_params(REPO_ROOT / "params.yaml"), {"models.pretrained.weights": "none"}
    )


def test_default_custom_cnn_matches_documented_design(cfg):
    model = build_model(cfg, "custom_cnn")
    assert model(torch.randn(2, 1, 224, 224)).shape == (2, 3)
    assert count_parameters(model) == 2_386_595  # documented in docs/models.md
    assert receptive_field(5) == 156


def test_flatten_variant_is_much_larger():
    gap = CustomCNN(global_pool="gap")
    flat = CustomCNN(global_pool="flatten", input_size=224)
    assert flat(torch.randn(1, 1, 224, 224)).shape == (1, 3)
    assert count_parameters(flat) > count_parameters(gap) + 1_000_000


@pytest.mark.parametrize("name", SUPPORTED)
def test_pretrained_models_output_and_groups(cfg, name):
    model = build_model(cfg, name)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 3)
    groups = param_groups(model, name)
    grouped = {id(p) for mods in groups.values() for m in mods for p in m.parameters()}
    assert grouped == {id(p) for p in model.parameters()}, "every parameter in exactly one group"


@pytest.mark.parametrize("name", ["custom_cnn", *SUPPORTED])
def test_gradcam_heatmap(cfg, name):
    model = build_model(cfg, name).eval()
    x = torch.randn(1, 1 if name == "custom_cnn" else 3, 64, 64)
    with GradCAM(model, gradcam_target(model, name)) as cam_fn:
        cam, cls = cam_fn(x)
    assert cam.shape == (64, 64) and cam.min() >= 0 and cam.max() <= 1 and 0 <= cls < 3


def test_transforms_channels(cfg):
    x = torch.rand(1, 32, 32)
    assert build_transform(cfg, train=True, pretrained=False)(x).shape == (1, 32, 32)
    out = build_transform(cfg, train=False, pretrained=True)(x)
    assert out.shape == (3, 32, 32)
    assert torch.equal(build_transform(cfg, train=False, pretrained=False)(x), x)


def test_class_weights_inverse_frequency():
    w = compute_class_weights([0, 0, 0, 1], n_classes=3)
    assert w[1] > w[0] and torch.isclose(w[0], torch.tensor(4 / 9))


def test_training_plan_two_stages_for_pretrained(cfg):
    stages, _ = training_plan(cfg, "resnet18")
    assert [s.name for s in stages] == ["head", "finetune"]
    assert stages[0].freeze_backbone and set(stages[1].lrs) == {"early", "late", "head"}
    assert len(training_plan(cfg, "custom_cnn")[0]) == 1
