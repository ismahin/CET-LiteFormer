from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import FlowTabularDataset
from .data.preprocessing import build_preprocessed_splits, compute_mi_prior
from .models.cet_liteformer import CETLiteFormer
from .training.losses import CETLiteFormerLoss, compute_class_weights_from_labels
from .training.scheduler import build_scheduler
from .training.trainer import train as train_loop
from .utils.io import ensure_dir, load_yaml, save_yaml
from .utils.logger import print_section
from .utils.plots import plot_training_curves
from .utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to configs/default.yaml")
    ap.add_argument("--csv_path", type=str, default=None, help="CSV/ARFF dataset path (overrides config.data.csv_path)")
    ap.add_argument("--label_col", type=str, default=None, help="Label column name (overrides config.data.label_col)")
    ap.add_argument("--experiment_name", type=str, default=None, help="Experiment name (overrides config.experiment.name)")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu / auto")
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()


def _resolve_device(arg: Optional[str]) -> torch.device:
    if arg is None or arg == "" or arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.csv_path is not None:
        cfg["data"]["csv_path"] = args.csv_path
    if args.label_col is not None:
        cfg["data"]["label_col"] = args.label_col
    if args.experiment_name is not None:
        cfg["experiment"]["name"] = args.experiment_name

    exp_name = cfg["experiment"]["name"]
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    out_root = Path(cfg["experiment"]["output_dir"])
    exp_dir = out_root / exp_name
    ensure_dir(exp_dir)

    print_section("TRAINING CET-LITEFORMER")
    print(f"Experiment: {exp_name}")
    print(f"Output dir: {exp_dir}")
    print(f"Seed: {seed}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    dataset_path = data_cfg["csv_path"]
    if not dataset_path:
        raise ValueError("data.csv_path is empty. Pass --csv_path or set in config.")

    # preprocessing (train-only fit) + persistence
    corr_cfg = data_cfg.get("correlation_selection", {}) or {}
    prep = build_preprocessed_splits(
        dataset_path=dataset_path,
        label_col=str(data_cfg.get("label_col", "") or ""),
        drop_cols=data_cfg.get("drop_cols", []) or [],
        test_size=float(data_cfg["test_size"]),
        val_size=float(data_cfg["val_size"]),
        stratify=bool(data_cfg.get("stratify", True)),
        seed=seed,
        max_rows=data_cfg.get("max_rows", None),
        missing_strategy=str(data_cfg.get("missing_strategy", "median")),
        normalize=str(data_cfg.get("normalize", "log_iqr")),
        remove_constant_features=bool(data_cfg.get("remove_constant_features", True)),
        corr_enabled=bool(corr_cfg.get("enabled", True)),
        corr_top_k=(None if corr_cfg.get("top_k", None) in (None, "null") else int(corr_cfg.get("top_k"))),
        corr_min_abs=float(corr_cfg.get("min_abs_corr", 0.0)),
        output_dir=exp_dir,
    )

    X_train = prep["X_train"]
    y_train = prep["y_train"]
    X_val = prep["X_val"]
    y_val = prep["y_val"]
    feature_names = prep["feature_names"]
    group_ids = prep["group_ids"]
    group_names = prep["group_names"]

    num_features = int(X_train.shape[1])
    num_classes = int(len(prep["label_mapping"]))

    # MI prior (train-only) aligned to final feature order
    mi_path = exp_dir / "feature_mi_scores.csv"
    mi_norm = compute_mi_prior(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        group_names=group_names,
        output_csv_path=mi_path,
    )
    mi_prior = torch.tensor(mi_norm, dtype=torch.float32)

    # datasets/loaders
    train_ds = FlowTabularDataset(X_train, y_train)
    val_ds = FlowTabularDataset(X_val, y_val)

    bs = int(train_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(args.num_workers), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=int(args.num_workers), pin_memory=True)

    # model
    model = CETLiteFormer(
        num_features=num_features,
        num_classes=num_classes,
        group_ids=group_ids,
        mi_prior=mi_prior if bool(model_cfg.get("use_entropy_gate", True)) else None,
        embed_dim=int(model_cfg["embed_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        rff_dim=int(model_cfg["rff_dim"]),
        sigma=float(model_cfg["sigma"]),
        dropout=float(model_cfg["dropout"]),
        use_cls_token=bool(model_cfg.get("use_cls_token", True)),
        use_entropy_gate=bool(model_cfg.get("use_entropy_gate", True)),
        use_correntropy_attention=bool(model_cfg.get("use_correntropy_attention", True)),
        attention_type=model_cfg.get("attention_type"),
        learnable_sigma=bool(model_cfg.get("learnable_sigma", False)),
        use_early_exit=bool(model_cfg.get("use_early_exit", True)),
        early_exit_threshold=float(model_cfg.get("early_exit_threshold", 0.90)),
        ffn_bottleneck_ratio=float(model_cfg.get("ffn_bottleneck_ratio", 0.5)),
        gate_prior_strength=float(model_cfg.get("gate_prior_strength", 1.0)),
    )

    # loss
    class_weights = compute_class_weights_from_labels(y_train, num_classes=num_classes)
    loss_fn = CETLiteFormerLoss(
        num_classes=num_classes,
        use_focal_loss=bool(train_cfg.get("use_focal_loss", True)),
        focal_gamma=float(train_cfg.get("focal_gamma", 2.0)),
        class_weights=class_weights,
        gate_l1_lambda=float(train_cfg.get("gate_l1_lambda", 1e-4)),
        exit_loss_lambda=float(train_cfg.get("exit_loss_lambda", 0.3)),
    )

    # optimizer/scheduler
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if opt_name != "adamw":
        raise ValueError("Only AdamW is supported in this implementation.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = build_scheduler(str(train_cfg.get("scheduler", "cosine")), optimizer, epochs=int(train_cfg["epochs"]))

    device = _resolve_device(args.device)
    result = train_loop(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(train_cfg["epochs"]),
        grad_clip=float(train_cfg.get("grad_clip", 0.0)),
        patience=int(train_cfg.get("patience", 15)),
        output_dir=exp_dir,
        config_used=cfg,
    )

    # persist a tiny summary
    summary = {
        "experiment": exp_name,
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "best_model_path": result.best_model_path,
        "last_model_path": result.last_model_path,
        "num_features": num_features,
        "num_classes": num_classes,
    }
    (exp_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # plots
    log_path = exp_dir / "training_log.csv"
    if log_path.exists():
        plot_training_curves(log_path, exp_dir)


if __name__ == "__main__":
    main()

