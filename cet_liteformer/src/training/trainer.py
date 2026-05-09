from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.io import ensure_dir, save_json, save_yaml
from ..utils.logger import CSVLogger, print_section
from ..utils.model_stats import (
    count_parameters,
    count_trainable_parameters,
    estimate_flops,
    estimate_model_size_mb,
)
from .early_stopping import EarlyStopping
from .metrics import compute_classification_metrics


@dataclass
class TrainResult:
    best_epoch: int
    best_val_macro_f1: float
    best_model_path: str
    last_model_path: str


def _to_device(batch, device: torch.device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    loss_fn,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    losses = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        logits = out["logits"]
        if loss_fn is not None:
            ld = loss_fn(out, y)
            losses.append(float(ld["loss"].detach().item()))
        prob = torch.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        y_prob.append(prob.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    y_prob_np = np.concatenate(y_prob, axis=0)
    metrics = compute_classification_metrics(y_true_np, y_pred_np, y_prob_np)
    if losses:
        metrics["val_loss"] = float(np.mean(losses))
    return metrics


def train(
    model: torch.nn.Module,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    grad_clip: float,
    patience: int,
    output_dir: str | Path,
    config_used: Dict[str, Any],
) -> TrainResult:
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    save_yaml(config_used, out_dir / "config_used.yaml")

    print_section("MODEL SUMMARY")
    sample_x, _ = next(iter(train_loader))
    # FLOPs profiling runs on a CPU copy to avoid THOP hook leakage into GPU training.
    flops = estimate_flops(model, sample_x[:1])
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model)}")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
    print(f"Estimated size: {estimate_model_size_mb(model):.3f} MB")
    if flops is not None:
        print(f"Estimated FLOPs (approx): {flops:.3e}")

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"

    early = EarlyStopping(patience=patience, mode="max")

    logger = CSVLogger(
        out_dir / "training_log.csv",
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_precision_macro",
            "val_recall_macro",
            "val_f1_macro",
            "val_f1_weighted",
            "lr",
        ],
    )

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x, y = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss_dict = loss_fn(out, y)
            loss = loss_dict["loss"]
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running += float(loss.detach().item())
            n_batches += 1
            pbar.set_postfix({"loss": running / max(n_batches, 1)})

        train_loss = running / max(n_batches, 1)

        val_metrics = evaluate_epoch(model, loss_fn, val_loader, device=device)
        val_f1 = float(val_metrics["f1_macro"])

        lr = float(optimizer.param_groups[0].get("lr", 0.0))
        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics.get("val_loss"),
                "val_accuracy": val_metrics.get("accuracy"),
                "val_precision_macro": val_metrics.get("precision_macro"),
                "val_recall_macro": val_metrics.get("recall_macro"),
                "val_f1_macro": val_metrics.get("f1_macro"),
                "val_f1_weighted": val_metrics.get("f1_weighted"),
                "lr": lr,
            }
        )

        # scheduler step
        if scheduler is not None:
            if scheduler.__class__.__name__.lower().endswith("reducelronplateau"):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        # checkpointing best by macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1_macro": best_val_f1}, best_path)

        # always save last
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1_macro": val_f1}, last_path)

        # early stopping
        if early.step(val_f1):
            print(f"Early stopping at epoch {epoch}. Best epoch {best_epoch} macro-F1 {best_val_f1:.4f}")
            break

    return TrainResult(
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1,
        best_model_path=str(best_path),
        last_model_path=str(last_path),
    )

