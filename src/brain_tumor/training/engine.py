"""Training engine shared by every model and by the Optuna search.

Custom CNN: one stage, AdamW at ``train.lr``.

Pretrained models: two stages.

1. ``freeze_epochs`` epochs with the backbone frozen (BatchNorm statistics fixed),
   training only the new head at ``head_lr``;
2. everything unfrozen with discriminative learning rates
   (early layers < late layers < head).

Every epoch: train -> validate -> callback (MLflow logging, Optuna pruning) ->
keep the best weights by validation macro-F1 -> early stopping.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from brain_tumor.models import pretrained as pt
from brain_tumor.models.factory import is_pretrained
from brain_tumor.training.metrics import compute_metrics
from brain_tumor.utils import get_logger

log = get_logger("engine")

EpochCallback = Callable[[int, dict[str, float]], None]


@dataclass
class FitResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_macro_f1: float = -1.0
    epochs_run: int = 0
    train_seconds: float = 0.0
    stopped_early: bool = False


@dataclass
class Stage:
    name: str
    epochs: int
    freeze_backbone: bool
    lrs: dict[str, float]  # group name -> learning rate


def training_plan(cfg: dict[str, Any], model_name: str) -> tuple[list[Stage], int]:
    """Stages and early-stopping patience for ``model_name``."""
    t = cfg["train"]
    if not is_pretrained(model_name):
        return [Stage("train", t["epochs"], False, {"all": t["lr"]})], t["patience"]
    p = cfg["models"]["pretrained"]
    freeze = min(p["freeze_epochs"], p["epochs"])
    stages = []
    if freeze > 0:
        stages.append(Stage("head", freeze, True, {"head": p["head_lr"]}))
    stages.append(
        Stage(
            "finetune",
            p["epochs"] - freeze,
            False,
            {"early": p["early_layers_lr"], "late": p["backbone_lr"], "head": p["head_lr"]},
        )
    )
    return [s for s in stages if s.epochs > 0], p["patience"]


def build_optimizer(
    model: nn.Module, model_name: str, stage: Stage, weight_decay: float
) -> torch.optim.Optimizer:
    if "all" in stage.lrs:
        groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": stage.lrs["all"],
                "name": "all",
            }
        ]
    else:
        modules = pt.param_groups(model, model_name)
        groups = []
        for group_name, lr in stage.lrs.items():
            params = [p for m in modules[group_name] for p in m.parameters() if p.requires_grad]
            if params:
                groups.append({"params": params, "lr": lr, "name": group_name})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, kind: str, epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Per-epoch scheduler. ``cosine`` includes a linear warm-up."""
    if kind == "cosine":
        warmup = min(warmup_epochs, max(epochs - 1, 0))

        def factor(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / (warmup + 1)
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
    if kind == "none":
        return None
    raise ValueError(f"Unknown scheduler {kind!r}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp: bool,
    max_batches: int | None = None,
    after_train_mode: Callable[[], None] | None = None,
) -> dict[str, float]:
    model.train()
    if after_train_mode:
        after_train_mode()
    total_loss, n_seen, preds, labels = 0.0, 0, [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        preds.append(logits.argmax(1).detach().cpu())
        labels.append(y.cpu())
    y_true, y_pred = torch.cat(labels).numpy(), torch.cat(preds).numpy()
    return {
        "loss": total_loss / max(n_seen, 1),
        "accuracy": float((y_true == y_pred).mean()) if n_seen else 0.0,
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
    max_batches: int | None = None,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (probabilities [N, K], labels [N], mean loss).

    TTA averages the softmax of the image and of its horizontal flip. The loss (if a
    criterion is given) is computed on the un-flipped logits.
    """
    model.eval()
    probs, labels, total_loss, n_seen = [], [], 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y_dev = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x).float()
        if criterion is not None:
            total_loss += criterion(logits, y_dev).item() * x.size(0)
            n_seen += x.size(0)
        p = torch.softmax(logits, dim=1)
        if tta:
            p = (p + torch.softmax(model(torch.flip(x, dims=[-1])).float(), dim=1)) / 2
        probs.append(p.cpu())
        labels.append(y)
    loss = total_loss / n_seen if n_seen else float("nan")
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    criterion: nn.Module | None = None,
    tta: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Metrics on a loader. Keys starting with '_' hold raw arrays (not loggable)."""
    probs, y_true, loss = predict(model, loader, device, tta, max_batches, criterion)
    y_pred = probs.argmax(1)
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["loss"] = loss
    metrics["_probs"], metrics["_y_true"], metrics["_y_pred"] = probs, y_true, y_pred
    return metrics


def fit(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    callback: EpochCallback | None = None,
) -> FitResult:
    """Train with the plan from :func:`training_plan`; restores the best weights at the end."""
    t = cfg["train"]
    class_names = cfg["data"]["class_names"]
    amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=t["label_smoothing"],
    )
    stages, patience = training_plan(cfg, model_name)
    result = FitResult(
        history={
            k: []
            for k in (
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "lr",
            )
        }
    )
    best_state, epochs_without_improvement, global_epoch = None, 0, 0
    start = time.perf_counter()

    pretrained_model = is_pretrained(model_name)
    for stage in stages:
        if pretrained_model:
            pt.set_backbone_trainable(model, model_name, not stage.freeze_backbone)
        after_train_mode = (
            (lambda: pt.freeze_backbone_batchnorm(model, model_name))
            if pretrained_model and stage.freeze_backbone
            else None
        )
        optimizer = build_optimizer(model, model_name, stage, t["weight_decay"])
        scheduler = build_scheduler(optimizer, t["scheduler"], stage.epochs, t["warmup_epochs"])
        log.info("Stage '%s': %d epochs, lrs=%s", stage.name, stage.epochs, stage.lrs)

        for _ in range(stage.epochs):
            global_epoch += 1
            epoch_start = time.perf_counter()
            tr = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                amp,
                t.get("max_batches"),
                after_train_mode,
            )
            va = evaluate(
                model, val_loader, device, class_names, criterion, max_batches=t.get("max_batches")
            )
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va["macro_f1"])
            elif scheduler is not None:
                scheduler.step()

            epoch_metrics = {
                "train_loss": tr["loss"],
                "train_accuracy": tr["accuracy"],
                "val_loss": va["loss"],
                "val_accuracy": va["accuracy"],
                "val_macro_f1": va["macro_f1"],
                "lr": optimizer.param_groups[-1]["lr"],
            }
            for k, v in epoch_metrics.items():
                result.history[k].append(v)
            log.info(
                "[%s] epoch %3d | train loss %.3f acc %.3f | val loss %.3f acc %.3f "
                "F1 %.3f | %.0fs",
                model_name,
                global_epoch,
                tr["loss"],
                tr["accuracy"],
                epoch_metrics["val_loss"],
                va["accuracy"],
                va["macro_f1"],
                time.perf_counter() - epoch_start,
            )

            improved = va["macro_f1"] > result.best_val_macro_f1
            if improved:
                result.best_val_macro_f1, result.best_epoch = va["macro_f1"], global_epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            elif not stage.freeze_backbone:
                epochs_without_improvement += 1
            if callback is not None:
                callback(
                    global_epoch, {**epoch_metrics, "best_val_macro_f1": result.best_val_macro_f1}
                )
            if epochs_without_improvement >= patience:
                result.stopped_early = True
                log.info("Early stopping at epoch %d (best %d)", global_epoch, result.best_epoch)
                break
        if result.stopped_early:
            break

    if is_pretrained(model_name):
        pt.set_backbone_trainable(model, model_name, True)
    if best_state is not None:
        model.load_state_dict(best_state)
    result.epochs_run = global_epoch
    result.train_seconds = time.perf_counter() - start
    return result
