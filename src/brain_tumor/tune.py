"""Optuna hyperparameter search (validation macro-F1 only — never the test set).

* TPE sampler (seeded) explores the search space from ``params.yaml > tune.search_space``.
* Median pruner stops unpromising trials after ``pruner_warmup_epochs`` epochs.
* Each trial is a nested MLflow run under one parent run.
* The study is stored in ``optuna_<model>.db`` so an interrupted search can resume.
* Best values are written to ``configs/best_params/<model>.yaml`` as dotted overrides,
  consumed by ``python -m brain_tumor.train --best-params <file>``.

Example::

    python -m brain_tumor.tune train.model=custom_cnn tune.n_trials=20
"""

from __future__ import annotations

import argparse
import copy
from typing import Any

import mlflow
import optuna
import yaml

from brain_tumor.config import apply_overrides, load_params, project_path
from brain_tumor.data.datamodule import build_dataloaders
from brain_tumor.models.factory import build_model, is_pretrained
from brain_tumor.tracking import lineage_tags, local_tracking_uri, log_metrics, setup_mlflow
from brain_tumor.training.engine import fit
from brain_tumor.utils import get_device, get_logger, save_json, set_seed

log = get_logger("tune")


def suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    low, high, log_scale = spec["low"], spec["high"], spec.get("log", False)
    if isinstance(low, int) and isinstance(high, int) and not log_scale:
        return trial.suggest_int(name, low, high)
    return trial.suggest_float(name, float(low), float(high), log=log_scale)


def trial_overrides(trial: optuna.Trial, cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Map sampled values onto dotted config keys."""
    if is_pretrained(model_name):
        space = cfg["tune"]["search_space"]["pretrained"]
        values = {k: suggest(trial, k, v) for k, v in space.items()}
        mapping = {
            "backbone_lr": "models.pretrained.backbone_lr",
            "weight_decay": "train.weight_decay",
            "dropout": "models.pretrained.dropout",
            "freeze_epochs": "models.pretrained.freeze_epochs",
            "head_lr": "models.pretrained.head_lr",
        }
        return {mapping[k]: v for k, v in values.items()}

    space = cfg["tune"]["search_space"]["custom_cnn"]
    values = {k: suggest(trial, k, v) for k, v in space.items()}
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key in ("lr", "weight_decay"):
            out[f"train.{key}"] = value
        elif key in ("base_filters", "n_blocks"):
            continue
        else:
            out[f"models.custom_cnn.{key}"] = value
    if "base_filters" in values or "n_blocks" in values:
        b = values.get("base_filters", cfg["models"]["custom_cnn"]["filters"][0])
        n = values.get("n_blocks", len(cfg["models"]["custom_cnn"]["filters"]))
        out["models.custom_cnn.filters"] = [b, 2 * b, 4 * b, 8 * b, 8 * b][:n]
    return out


def make_objective(cfg: dict[str, Any], bundle, device) -> Any:
    name = cfg["train"]["model"]
    epochs = cfg["tune"]["epochs"]

    def objective(trial: optuna.Trial) -> float:
        overrides = trial_overrides(trial, cfg, name)
        trial_cfg = apply_overrides(copy.deepcopy(cfg), overrides)
        # Short trials: same budget for every model.
        trial_cfg["train"]["epochs"] = epochs
        trial_cfg["models"]["pretrained"]["epochs"] = epochs
        trial_cfg["train"]["patience"] = epochs
        trial_cfg["models"]["pretrained"]["patience"] = epochs
        set_seed(cfg["seed"] + trial.number)
        model = build_model(trial_cfg).to(device)

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params({k: str(v) for k, v in overrides.items()})

            def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
                log_metrics(metrics, step=epoch)
                trial.report(metrics["val_macro_f1"], step=epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    raise optuna.TrialPruned(f"pruned at epoch {epoch}")

            result = fit(
                model,
                name,
                bundle.train_loader,
                bundle.val_loader,
                trial_cfg,
                device,
                bundle.class_weights,
                callback=on_epoch,
            )
            mlflow.log_metric("best_val_macro_f1", result.best_val_macro_f1)
        return result.best_val_macro_f1

    return objective


def run_study(cfg: dict[str, Any], use_mlflow: bool = True) -> optuna.Study:
    name = cfg["train"]["model"]
    tcfg = cfg["tune"]
    device = get_device()
    bundle = build_dataloaders(cfg, pretrained=is_pretrained(name))
    root = project_path(cfg, "models_dir").parent
    study = optuna.create_study(
        study_name=f"{name}-tune",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=tcfg["pruner_warmup_epochs"]
        ),
        storage=f"sqlite:///{root / f'optuna_{name}.db'}",
        load_if_exists=True,
    )
    if not use_mlflow:
        mlflow.set_tracking_uri(local_tracking_uri(root))
    else:
        setup_mlflow(cfg, f"{name}-tuning")
    timeout = tcfg["timeout_minutes"] * 60 if tcfg.get("timeout_minutes") else None
    with mlflow.start_run(run_name=f"{name}-optuna"):
        mlflow.set_tags({**lineage_tags(cfg), "model": name, "stage": "tuning"})
        mlflow.log_params({"n_trials": tcfg["n_trials"], "trial_epochs": tcfg["epochs"]})
        study.optimize(
            make_objective(cfg, bundle, device),
            n_trials=tcfg["n_trials"],
            timeout=timeout,
            gc_after_trial=True,
        )
        best = study.best_trial
        mlflow.log_metric("best_val_macro_f1", best.value)
        overrides = trial_overrides(optuna.trial.FixedTrial(best.params), cfg, name)
        mlflow.log_params({f"best.{k}": str(v) for k, v in overrides.items()})
        save_outputs(cfg, study, overrides)
    return study


def save_outputs(cfg: dict[str, Any], study: optuna.Study, overrides: dict[str, Any]) -> None:
    name = cfg["train"]["model"]
    out_dir = project_path(cfg, "best_params_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_file = out_dir / f"{name}.yaml"
    with best_file.open("w") as f:
        yaml.safe_dump(
            {
                "model": name,
                "best_val_macro_f1": float(study.best_value),
                "n_trials": len(study.trials),
                "overrides": overrides,
            },
            f,
            sort_keys=False,
        )
    report_dir = project_path(cfg, "reports_dir") / "tuning" / name
    report_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(report_dir / "trials.csv", index=False)
    counts = {s.name: sum(t.state == s for t in study.trials) for s in optuna.trial.TrialState}
    save_json(
        {"best_value": study.best_value, "best_params": study.best_params, "trial_states": counts},
        report_dir / "summary.json",
    )
    try:  # optional plots (matplotlib backend of optuna)
        import matplotlib.pyplot as plt
        from optuna.visualization import matplotlib as ovm

        for fn, fname in (
            (ovm.plot_optimization_history, "history.png"),
            (ovm.plot_param_importances, "importance.png"),
        ):
            ax = fn(study)
            ax.figure.savefig(report_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(ax.figure)
            if mlflow.active_run():
                mlflow.log_artifact(str(report_dir / fname))
    except Exception as exc:  # importance needs >1 completed trial, etc.
        log.warning("Could not draw Optuna plots: %s", exc)
    log.info("Best val macro-F1 %.3f -> %s", study.best_value, best_file)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="log to a local ./mlflow.db instead of the remote server",
    )
    args = parser.parse_intermixed_args(argv)
    run_study(load_params(overrides=args.overrides), use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
