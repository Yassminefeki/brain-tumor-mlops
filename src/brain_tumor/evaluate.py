"""Test-set evaluation: metrics, confusion matrix, predictions, Grad-CAM, cost.

Used automatically at the end of ``train.py``, or standalone on a checkpoint::

    python -m brain_tumor.evaluate --checkpoint models/resnet18/<run>/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from brain_tumor.config import load_params, project_path
from brain_tumor.data.datamodule import build_dataloaders
from brain_tumor.data.transforms import build_transform
from brain_tumor.explain.gradcam import GradCAM, gradcam_grid
from brain_tumor.models.factory import build_model, gradcam_target, is_pretrained
from brain_tumor.training.engine import evaluate
from brain_tumor.training.metrics import plot_confusion_matrix, report_text
from brain_tumor.utils import count_parameters, get_device, get_logger, model_size_mb, save_json

log = get_logger("evaluate")


@torch.no_grad()
def inference_time_ms(
    model: nn.Module, loader: DataLoader, device: torch.device, n_images: int = 64
) -> float:
    """Average single-image latency (batch size 1), after a short warm-up."""
    model.eval()
    x = next(iter(loader))[0][:1].to(device)
    for _ in range(3):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_images):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / n_images


def gradcam_examples(
    model: nn.Module,
    model_name: str,
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    path: Path,
) -> Path | None:
    """Grad-CAM on a few test images: correct predictions first, then errors."""
    n = cfg["evaluate"]["gradcam_samples"]
    if n <= 0 or frame.empty:
        return None
    class_names = cfg["data"]["class_names"]
    processed = project_path(cfg, "processed_dir")
    frame = frame.reset_index(drop=True).assign(pred=y_pred)
    correct = frame[frame["label"] == frame["pred"]]
    wrong = frame[frame["label"] != frame["pred"]]
    n_wrong = min(len(wrong), n // 2)
    chosen = pd.concat(
        [
            correct.sample(min(len(correct), n - n_wrong), random_state=0),
            wrong.sample(n_wrong, random_state=0),
        ]
    )
    transform = build_transform(cfg, train=False, pretrained=is_pretrained(model_name))

    images, cams, titles, masks = [], [], [], []
    with GradCAM(model, gradcam_target(model, model_name)) as cam_fn:
        for row in chosen.itertuples():
            raw = np.load(processed / "images" / f"{row.image_id}.npy").astype(np.float32)
            x = transform(torch.from_numpy(raw).unsqueeze(0)).unsqueeze(0).to(device)
            cam, _ = cam_fn(x, class_idx=int(row.pred))
            mask_path = processed / "masks" / f"{row.image_id}.npy"
            images.append(raw)
            cams.append(cam)
            masks.append(np.load(mask_path) if mask_path.exists() else None)
            titles.append(f"true {class_names[row.label]}\npred {class_names[row.pred]}")
    return gradcam_grid(images, cams, titles, path, masks)


def full_evaluation(
    model: nn.Module,
    model_name: str,
    test_loader: DataLoader,
    test_frame: pd.DataFrame,
    cfg: dict[str, Any],
    device: torch.device,
    out_dir: Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Evaluate on the test set and write every artifact into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = cfg["data"]["class_names"]
    tta = bool(cfg["evaluate"]["tta"])
    model.eval()

    plain = evaluate(model, test_loader, device, class_names)
    final = evaluate(model, test_loader, device, class_names, tta=True) if tta else plain
    metrics = {k: v for k, v in final.items() if not k.startswith("_")}
    metrics["macro_f1_no_tta"] = plain["macro_f1"]
    metrics["accuracy_no_tta"] = plain["accuracy"]
    metrics["inference_ms_per_image"] = inference_time_ms(model, test_loader, device)
    metrics["n_parameters"] = count_parameters(model)
    metrics["model_size_mb"] = model_size_mb(model)
    metrics.pop("loss", None)

    y_true, y_pred, probs = final["_y_true"], final["_y_pred"], final["_probs"]
    artifacts = [
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            out_dir / "confusion_matrix.png",
            f"{model_name} — test set",
        ),
    ]
    (out_dir / "classification_report.txt").write_text(report_text(y_true, y_pred, class_names))
    artifacts.append(out_dir / "classification_report.txt")

    preds = test_frame[["image_id", "patient_id", "label"]].reset_index(drop=True).copy()
    preds["pred"] = y_pred
    for i, name in enumerate(class_names):
        preds[f"prob_{name}"] = probs[:, i]
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    artifacts.append(out_dir / "test_predictions.csv")

    gc = gradcam_examples(
        model, model_name, test_frame, y_pred, cfg, device, out_dir / "gradcam.png"
    )
    if gc:
        artifacts.append(gc)
    artifacts.append(save_json(metrics, out_dir / "test_metrics.json"))
    log.info(
        "TEST %s | accuracy %.3f | macro-F1 %.3f (no TTA %.3f) | %.1f ms/image",
        model_name,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["macro_f1_no_tta"],
        metrics["inference_ms_per_image"],
    )
    return metrics, artifacts


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from a checkpoint written by ``train.py``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"], ckpt["model_name"], weights="none")
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None, help="output dir (default reports/eval/<model>)")
    parser.add_argument("overrides", nargs="*", help="e.g. evaluate.tta=false")
    args = parser.parse_intermixed_args(argv)

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    name = ckpt["model_name"]
    # Paths come from the current params.yaml; model settings from the checkpoint.
    cfg = load_params(overrides=args.overrides)
    cfg["train"]["model"] = name
    bundle = build_dataloaders(cfg, pretrained=is_pretrained(name))
    out = Path(args.out) if args.out else project_path(cfg, "reports_dir") / "eval" / name
    full_evaluation(model, name, bundle.test_loader, bundle.frames["test"], cfg, device, out)


if __name__ == "__main__":
    main()
