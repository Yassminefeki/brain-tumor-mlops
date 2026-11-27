"""Train one model, evaluate it once on the test set, and track everything in MLflow.

Examples::

    python -m brain_tumor.train                                  # model from params.yaml
    python -m brain_tumor.train train.model=resnet18
    python -m brain_tumor.train train.model=custom_cnn train.augment=false --run-name cnn-noaug
    python -m brain_tumor.train train.model=resnet18 train.split_column=image_split  # leakage demo
    python -m brain_tumor.train train.model=custom_cnn train.fold=0                 # CV fold
    python -m brain_tumor.train train.model=custom_cnn \\
        --best-params configs/best_params/custom_cnn.yaml                       # after Optuna
    python -m brain_tumor.train train.model=resnet18 --register --promote    # final model
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import torch

from brain_tumor.config import (
    apply_overrides,
    load_best_params,
    load_params,
    project_path,
    public_config,
)
from brain_tumor.data.datamodule import build_dataloaders
from brain_tumor.evaluate import full_evaluation
from brain_tumor.models.factory import build_model, is_pretrained
from brain_tumor.tracking import (
    lineage_tags,
    log_metrics,
    log_pytorch_model,
    loggable_params,
    setup_mlflow,
)
from brain_tumor.training.engine import fit
from brain_tumor.training.metrics import plot_history
from brain_tumor.utils import count_parameters, get_device, get_logger, save_json, set_seed

log = get_logger("train")


def default_run_name(cfg: dict[str, Any]) -> str:
    t = cfg["train"]
    parts = [t["model"], "aug" if t["augment"] else "noaug"]
    if t["split_column"] != "split":
        parts.append(t["split_column"])
    if t.get("fold") is not None:
        parts.append(f"fold{t['fold']}")
    return "-".join(parts)


def save_checkpoint(
    path: Path, model: torch.nn.Module, cfg: dict[str, Any], extra: dict[str, Any]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": cfg["train"]["model"],
            "state_dict": model.state_dict(),
            "config": public_config(cfg),
            "class_names": cfg["data"]["class_names"],
            "image_size": cfg["preprocess"]["image_size"],
            "clip_percentiles": cfg["preprocess"]["clip_percentiles"],
            "pretrained": is_pretrained(cfg["train"]["model"]),
            **extra,
        },
        path,
    )
    return path


def run_training(
    cfg: dict[str, Any],
    run_name: str | None = None,
    use_mlflow: bool = True,
    register: bool = False,
    promote: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Complete training run; returns a summary dict (also written to disk)."""
    set_seed(cfg["seed"])
    device = device or get_device()
    name = cfg["train"]["model"]
    run_name = run_name or default_run_name(cfg)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = project_path(cfg, "models_dir") / name / f"{run_name}-{stamp}"
    log.info("Training %s on %s -> %s", name, device, out_dir)

    bundle = build_dataloaders(cfg, pretrained=is_pretrained(name))
    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    log.info("%s: %.2fM parameters", name, n_params / 1e6)

    if use_mlflow:
        setup_mlflow(cfg, name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(loggable_params(cfg, name))
        mlflow.set_tags(
            {
                **lineage_tags(cfg),
                "model": name,
                "device": str(device),
                "split_column": cfg["train"]["split_column"],
            }
        )
        mlflow.log_metric("n_parameters", n_params)

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
        if use_mlflow:
            log_metrics(metrics, step=epoch)

    try:
        result = fit(
            model,
            name,
            bundle.train_loader,
            bundle.val_loader,
            cfg,
            device,
            bundle.class_weights,
            callback=on_epoch,
        )
        ckpt = save_checkpoint(
            out_dir / "best.pt",
            model,
            cfg,
            {"best_epoch": result.best_epoch, "best_val_macro_f1": result.best_val_macro_f1},
        )
        plot_history(result.history, out_dir / "training_curves.png", f"{name} ({run_name})")

        test_metrics, artifacts = full_evaluation(
            model, name, bundle.test_loader, bundle.frames["test"], cfg, device, out_dir
        )
        summary = {
            "model": name,
            "run_name": run_name,
            "checkpoint": str(ckpt),
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "best_val_macro_f1": result.best_val_macro_f1,
            "train_minutes": result.train_seconds / 60,
            "stopped_early": result.stopped_early,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        save_json(summary, out_dir / "summary.json")
        save_json(result.history, out_dir / "history.json")

        if use_mlflow:
            log_metrics(
                {
                    "best_val_macro_f1": result.best_val_macro_f1,
                    "best_epoch": result.best_epoch,
                    "train_minutes": summary["train_minutes"],
                }
            )
            log_metrics(test_metrics, prefix="test_")
            for path in [
                *artifacts,
                out_dir / "training_curves.png",
                out_dir / "history.json",
                out_dir / "summary.json",
            ]:
                mlflow.log_artifact(str(path))
            mlflow.log_artifact(str(ckpt), artifact_path="checkpoint")
            if register:
                example = next(iter(bundle.test_loader))[0][:1].numpy()
                log_pytorch_model(
                    model.cpu().eval(), example, cfg["mlflow"]["registered_model_name"]
                )
            summary["mlflow_run_id"] = mlflow.active_run().info.run_id
    finally:
        if use_mlflow and mlflow.active_run():
            mlflow.end_run()

    if promote:
        target = project_path(cfg, "models_dir") / "best.pt"
        shutil.copy2(ckpt, target)
        log.info("Promoted %s -> %s (used by the API)", ckpt, target)
    log.info(
        "Done: val macro-F1 %.3f | test macro-F1 %.3f | test accuracy %.3f",
        summary["best_val_macro_f1"],
        summary["test_macro_f1"],
        summary["test_accuracy"],
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.epochs=30")
    parser.add_argument("--params", default=None, help="path to params.yaml")
    parser.add_argument("--best-params", default=None, help="YAML written by brain_tumor.tune")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    parser.add_argument("--register", action="store_true", help="register in the MLflow registry")
    parser.add_argument("--promote", action="store_true", help="copy checkpoint to models/best.pt")
    args = parser.parse_intermixed_args(argv)

    cfg = load_params(args.params)
    if args.best_params:
        cfg = apply_overrides(cfg, load_best_params(args.best_params))
    cfg = apply_overrides(cfg, args.overrides)  # CLI wins over tuned values
    run_training(cfg, args.run_name, not args.no_mlflow, args.register, args.promote)


if __name__ == "__main__":
    main()
