"""End-to-end on synthetic data: train -> evaluate -> tune -> serve (a few seconds on CPU)."""

from pathlib import Path

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from brain_tumor.baseline import majority_baseline
from brain_tumor.config import apply_overrides, load_best_params
from brain_tumor.train import run_training
from brain_tumor.tune import run_study

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained(prepared_project) -> dict:
    from brain_tumor.config import load_params

    cfg = load_params(prepared_project / "params.yaml")
    return run_training(
        cfg,
        run_name="test-cnn",
        use_mlflow=True,
        register=True,
        promote=True,
        device=torch.device("cpu"),
    )


def test_training_produces_checkpoint_and_metrics(trained):
    ckpt = Path(trained["checkpoint"])
    assert ckpt.exists()
    for name in (
        "confusion_matrix.png",
        "gradcam.png",
        "test_predictions.csv",
        "training_curves.png",
        "summary.json",
    ):
        assert (ckpt.parent / name).exists(), name
    assert 0 <= trained["test_macro_f1"] <= 1
    assert trained["epochs_run"] == 2
    assert "mlflow_run_id" in trained


def test_pretrained_two_stage_training(project_cfg):
    cfg = apply_overrides(project_cfg, {"train.model": "resnet18", "evaluate.gradcam_samples": 2})
    summary = run_training(cfg, use_mlflow=False, device=torch.device("cpu"))
    assert summary["epochs_run"] == 2


def test_cv_fold_and_leaky_split_runs(project_cfg):
    for override in ({"train.fold": 0}, {"train.split_column": "image_split"}):
        cfg = apply_overrides(project_cfg, {**override, "train.epochs": 1})
        assert run_training(cfg, use_mlflow=False, device=torch.device("cpu"))["epochs_run"] == 1


def test_baseline(project_cfg):
    metrics = majority_baseline(project_cfg)
    assert metrics["recall_" + metrics["majority_class"]] == 1.0


def test_optuna_writes_best_params(project_cfg, prepared_project):
    study = run_study(project_cfg, use_mlflow=True)
    assert len(study.trials) >= 2
    best_file = prepared_project / "configs" / "best_params" / "custom_cnn.yaml"
    overrides = load_best_params(best_file)
    assert "train.lr" in overrides
    apply_overrides(project_cfg, overrides)  # must be valid config keys
    assert yaml.safe_load(best_file.read_text())["model"] == "custom_cnn"


def test_api_predict_and_explain(trained, prepared_project, monkeypatch):
    monkeypatch.setenv("MODEL_CHECKPOINT", str(prepared_project / "models" / "best.pt"))
    from api.main import app

    mat_file = prepared_project / "data" / "interim" / "1.mat"
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        r = client.post("/predict", files={"file": ("1.mat", mat_file.read_bytes())})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["prediction"] in body["probabilities"]
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
        r = client.post("/explain", files={"file": ("1.mat", mat_file.read_bytes())})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        bad = client.post("/predict", files={"file": ("x.png", b"not an image")})
        assert bad.status_code == 415
