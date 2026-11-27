from pathlib import Path

import pytest

from brain_tumor.config import (
    ConfigError,
    apply_overrides,
    get_by_path,
    load_params,
    parse_override,
    project_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repository_params_load():
    cfg = load_params(REPO_ROOT / "params.yaml")
    assert cfg["data"]["class_names"] == ["meningioma", "glioma", "pituitary"]
    assert sum(cfg["data"]["expected_class_counts"].values()) == cfg["data"]["expected_images"]
    assert abs(cfg["split"]["train"] + cfg["split"]["val"] + cfg["split"]["test"] - 1) < 1e-9


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("train.epochs=30", ("train.epochs", 30)),
        ("train.lr=1e-4", ("train.lr", 1e-4)),
        ("train.augment=false", ("train.augment", False)),
        ("train.fold=null", ("train.fold", None)),
        ("models.custom_cnn.filters=[16,32]", ("models.custom_cnn.filters", [16, 32])),
        ("train.model=resnet18", ("train.model", "resnet18")),
    ],
)
def test_parse_override_types(raw, expected):
    assert parse_override(raw) == expected


def test_overrides_do_not_mutate_and_reject_typos():
    cfg = load_params(REPO_ROOT / "params.yaml")
    new = apply_overrides(cfg, ["train.epochs=3"])
    assert new["train"]["epochs"] == 3 and cfg["train"]["epochs"] != 3
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["train.epoch=3"])  # typo
    with pytest.raises(ConfigError):
        parse_override("train.epochs")


def test_paths_resolve_relative_to_params_file():
    cfg = load_params(REPO_ROOT / "params.yaml")
    assert project_path(cfg, "splits") == REPO_ROOT / "data" / "splits.csv"
    assert get_by_path(cfg, "missing.key", default=None) is None
