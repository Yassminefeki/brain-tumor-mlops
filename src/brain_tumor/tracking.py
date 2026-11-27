"""MLflow helpers: connection, configuration logging and lineage tags.

Every run records *what* was trained (resolved params), on *which code*
(git commit, dirty flag) and on *which data* (DVC hashes of ``data/processed``
and ``data/splits.csv`` read from ``dvc.lock``). That triple makes any result
traceable and reproducible.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import yaml

from brain_tumor.config import public_config
from brain_tumor.utils import get_logger, git_info

log = get_logger("tracking")


def setup_mlflow(cfg: dict[str, Any], experiment_suffix: str) -> str:
    """Connect to the tracking server and select ``<prefix>/<suffix>`` as experiment.

    Reads ``.env`` if present. Without ``MLFLOW_TRACKING_URI`` runs are logged to a local
    SQLite database ``./mlflow.db`` (handy offline; view with
    ``mlflow ui --backend-store-uri sqlite:///mlflow.db``).
    """
    root = Path(cfg.get("_root", "."))
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:  # python-dotenv is optional
        pass
    uri = os.environ.get("MLFLOW_TRACKING_URI") or local_tracking_uri(root)
    mlflow.set_tracking_uri(uri)
    experiment = f"{cfg['mlflow']['experiment_prefix']}/{experiment_suffix}"
    mlflow.set_experiment(experiment)
    log.info("MLflow tracking: %s | experiment: %s", uri, experiment)
    return experiment


def local_tracking_uri(root: str | Path) -> str:
    """Local SQLite backend (MLflow 3 no longer supports the ./mlruns file store by default)."""
    return f"sqlite:///{(Path(root) / 'mlflow.db').resolve()}"


def flatten(d: Mapping[str, Any], parent: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{parent}{sep}{key}" if parent else str(key)
        if isinstance(value, Mapping):
            out.update(flatten(value, name, sep))
        else:
            out[name] = value
    return out


def loggable_params(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Parameters relevant to a training run (other models' settings are skipped)."""
    pub = public_config(cfg)
    pub.pop("tune", None)
    models = pub.pop("models", {})
    key = "custom_cnn" if model_name == "custom_cnn" else "pretrained"
    pub["model"] = models.get(key, {})
    return {k: str(v)[:500] for k, v in flatten(pub).items()}


def data_version(root: str | Path) -> dict[str, str]:
    """DVC md5 of the processed data and of the splits, from dvc.lock."""
    lock = Path(root) / "dvc.lock"
    if not lock.exists():
        return {}
    with lock.open() as f:
        stages = (yaml.safe_load(f) or {}).get("stages", {})
    versions = {}
    for stage, tag in (
        ("preprocess", "data.processed_md5"),
        ("split", "data.splits_md5"),
        ("download", "data.raw_md5"),
    ):
        outs = stages.get(stage, {}).get("outs", [])
        if outs and "md5" in outs[0]:
            versions[tag] = outs[0]["md5"]
    return versions


def lineage_tags(cfg: dict[str, Any]) -> dict[str, str]:
    root = cfg.get("_root", ".")
    git = git_info(root)
    tags = {"git.commit": git["commit"], "git.branch": git["branch"], "git.dirty": git["dirty"]}
    tags.update(data_version(root))
    return {k: v for k, v in tags.items() if v}


def log_metrics(metrics: Mapping[str, Any], prefix: str = "", step: int | None = None) -> None:
    """Log numeric metrics, skipping private keys (``_probs``...) and non-numbers."""
    clean = {
        f"{prefix}{k}": float(v)
        for k, v in metrics.items()
        if not k.startswith("_") and isinstance(v, int | float)
    }
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_pytorch_model(model: Any, input_example: Any, registered_name: str | None = None) -> None:
    """Log the model in MLflow format (and optionally register it).

    ``input_example`` (a numpy array of shape (1, C, H, W)) documents the signature and is
    required by recent MLflow versions. Handles the MLflow 2.x (``artifact_path``) /
    3.x (``name``) API difference.
    """
    import mlflow.pytorch

    kwargs: dict[str, Any] = {"input_example": input_example}
    if registered_name:
        kwargs["registered_model_name"] = registered_name
    try:
        mlflow.pytorch.log_model(model, name="model", **kwargs)
    except TypeError:
        mlflow.pytorch.log_model(model, artifact_path="model", **kwargs)
