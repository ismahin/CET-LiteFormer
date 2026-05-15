from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .ablation_suites import get_variant_specs
from .data.dataset import FlowTabularDataset
from .data.preprocessing import build_preprocessed_splits, compute_mi_prior
from .models.baselines import MLPBaseline, StandardTransformerBaseline
from .models.cet_liteformer import CETLiteFormer
from .training.losses import CETLiteFormerLoss, compute_class_weights_from_labels
from .training.scheduler import build_scheduler
from .training.trainer import train as train_loop
from .utils.io import ensure_dir, load_yaml, save_json
from .utils.logger import print_section
from .utils.plots import (
    plot_ablation_accuracy_latency,
    plot_ablation_incremental_ladder,
    plot_ablation_multi_metric,
)
from .utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run ablation suites: incremental model components, training objectives, or legacy variants."
    )
    ap.add_argument("--config", type=str, required=True, help="Base YAML (e.g. configs/default.yaml)")
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--label_col", type=str, default="")
    ap.add_argument("--experiment_name", type=str, required=True, help="Run name; outputs go under output root")
    ap.add_argument(
        "--suite",
        type=str,
        default="model_components",
        help="model_components | training_objectives | legacy | all",
    )
    ap.add_argument(
        "--ablation_root",
        type=str,
        default=None,
        help="If set (e.g. ablation_study/runs), results go to <ablation_root>/<experiment_name>/...",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()


def _device(arg: str) -> torch.device:
    if arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _prep_splits(cfg: Dict[str, Any], variant_dir: Path, seed: int) -> Dict[str, Any]:
    data_cfg = cfg["data"]
    corr_cfg = data_cfg.get("correlation_selection", {}) or {}
    return build_preprocessed_splits(
        dataset_path=str(data_cfg["csv_path"]),
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
        output_dir=variant_dir,
    )


def _roc_value(metrics: Dict[str, Any]) -> Any:
    if "roc_auc" in metrics:
        return metrics.get("roc_auc")
    return metrics.get("roc_auc_ovr_macro")


def _compute_step_deltas(summary_df: pd.DataFrame, suite: str) -> pd.DataFrame:
    """Per-suite ordered deltas vs previous step (same suite block only)."""
    suite = (suite or "").lower()
    df = summary_df.copy()
    if suite == "all":
        blocks = [
            df[df["variant"].str.match(r"^\d{2}_", na=False)].sort_values("variant"),
            df[df["variant"].str.match(r"^T\d+_", na=False)].sort_values("variant"),
        ]
        parts = []
        for sub in blocks:
            if sub.empty:
                continue
            parts.append(_delta_block(sub))
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    if suite == "model_components":
        sub = df[df["variant"].str.match(r"^\d{2}_", na=False)].sort_values("variant")
    elif suite == "training_objectives":
        sub = df[df["variant"].str.match(r"^T\d+_", na=False)].sort_values("variant")
    else:
        return pd.DataFrame()
    if sub.empty:
        return pd.DataFrame()
    return _delta_block(sub)


def _delta_block(sub: pd.DataFrame) -> pd.DataFrame:
    cols = ["accuracy", "macro_f1", "weighted_f1", "mcc"]
    rows = []
    prev = None
    for _, row in sub.iterrows():
        r = {"variant": row["variant"], "description": row.get("description", "")}
        if prev is not None:
            for c in cols:
                if c in row and c in prev and pd.notna(row[c]) and pd.notna(prev[c]):
                    r[f"delta_{c}"] = float(row[c]) - float(prev[c])
        else:
            for c in cols:
                r[f"delta_{c}"] = None
        rows.append(r)
        prev = row
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    base_cfg["data"]["csv_path"] = args.csv_path
    base_cfg["data"]["label_col"] = args.label_col
    base_cfg["experiment"]["name"] = args.experiment_name

    if args.ablation_root:
        root_dir = Path(args.ablation_root) / args.experiment_name
    else:
        root_dir = Path(base_cfg["experiment"]["output_dir"]) / args.experiment_name

    ensure_dir(root_dir)

    print_section("ABLATION STUDY")
    print(f"Suite: {args.suite}")
    print(f"Root: {root_dir}")

    seed = int(base_cfg["experiment"]["seed"])
    set_seed(seed)
    device = _device(args.device)

    variant_specs = get_variant_specs(base_cfg, args.suite)
    manifest: List[Dict[str, Any]] = []

    summary_rows: List[Dict[str, Any]] = []
    for variant_name, description, cfg in variant_specs:
        cfg["experiment"]["name"] = variant_name
        variant_dir = root_dir / variant_name
        ensure_dir(variant_dir)
        (variant_dir / "variant_description.txt").write_text(description.strip() + "\n", encoding="utf-8")

        prep = _prep_splits(cfg, variant_dir, seed)

        X_train, y_train = prep["X_train"], prep["y_train"]
        X_val, y_val = prep["X_val"], prep["y_val"]
        X_test, y_test = prep["X_test"], prep["y_test"]

        feature_names = prep["feature_names"]
        group_ids = prep["group_ids"]
        group_names = prep["group_names"]
        num_features = int(X_train.shape[1])
        num_classes = int(len(prep["label_mapping"]))

        mi_norm = compute_mi_prior(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            group_names=group_names,
            output_csv_path=variant_dir / "feature_mi_scores.csv",
        )
        mi_prior = torch.tensor(mi_norm, dtype=torch.float32)

        bs = int(cfg["training"]["batch_size"])
        train_loader = DataLoader(
            FlowTabularDataset(X_train, y_train),
            batch_size=bs,
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=True,
        )
        val_loader = DataLoader(
            FlowTabularDataset(X_val, y_val),
            batch_size=bs,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
        )

        if cfg["model"].get("name") == "MLPBaseline":
            model = MLPBaseline(
                num_features=num_features,
                num_classes=num_classes,
                dropout=float(cfg["model"].get("dropout", 0.15)),
            )
        elif cfg["model"].get("name") == "StandardTransformerBaseline":
            model = StandardTransformerBaseline(
                num_features=num_features,
                num_classes=num_classes,
                group_ids=group_ids,
                embed_dim=int(cfg["model"]["embed_dim"]),
                num_layers=int(cfg["model"]["num_layers"]),
                num_heads=int(cfg["model"]["num_heads"]),
                dropout=float(cfg["model"]["dropout"]),
                use_cls_token=bool(cfg["model"].get("use_cls_token", True)),
            )
        else:
            model = CETLiteFormer(
                num_features=num_features,
                num_classes=num_classes,
                group_ids=group_ids,
                mi_prior=mi_prior if bool(cfg["model"].get("use_entropy_gate", True)) else None,
                embed_dim=int(cfg["model"]["embed_dim"]),
                num_layers=int(cfg["model"]["num_layers"]),
                num_heads=int(cfg["model"]["num_heads"]),
                rff_dim=int(cfg["model"]["rff_dim"]),
                sigma=float(cfg["model"]["sigma"]),
                dropout=float(cfg["model"]["dropout"]),
                use_cls_token=bool(cfg["model"].get("use_cls_token", True)),
                use_entropy_gate=bool(cfg["model"].get("use_entropy_gate", True)),
                use_correntropy_attention=bool(cfg["model"].get("use_correntropy_attention", True)),
                attention_type=cfg["model"].get("attention_type"),
                learnable_sigma=bool(cfg["model"].get("learnable_sigma", False)),
                use_early_exit=bool(cfg["model"].get("use_early_exit", True)),
                early_exit_threshold=float(cfg["model"].get("early_exit_threshold", 0.90)),
                ffn_bottleneck_ratio=float(cfg["model"].get("ffn_bottleneck_ratio", 0.5)),
                gate_prior_strength=float(cfg["model"].get("gate_prior_strength", 1.0)),
            )

        class_weights = compute_class_weights_from_labels(y_train, num_classes=num_classes)
        loss_fn = CETLiteFormerLoss(
            num_classes=num_classes,
            use_focal_loss=bool(cfg["training"].get("use_focal_loss", True)),
            focal_gamma=float(cfg["training"].get("focal_gamma", 2.0)),
            class_weights=class_weights,
            gate_l1_lambda=float(cfg["training"].get("gate_l1_lambda", 1e-4)),
            exit_loss_lambda=float(cfg["training"].get("exit_loss_lambda", 0.3)),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
        )
        scheduler = build_scheduler(
            str(cfg["training"].get("scheduler", "cosine")),
            optimizer,
            epochs=int(cfg["training"]["epochs"]),
            warmup_epochs=int(cfg["training"].get("warmup_epochs", 0)),
        )

        res = train_loop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=int(cfg["training"]["epochs"]),
            grad_clip=float(cfg["training"].get("grad_clip", 0.0)),
            patience=int(cfg["training"].get("patience", 15)),
            output_dir=variant_dir,
            config_used=cfg,
        )

        model.eval()
        model.to(device)
        y_pred = []
        y_prob = []
        for i in range(0, len(X_test), bs):
            xb = torch.from_numpy(X_test[i : i + bs]).to(device)
            out = model(xb)
            prob = torch.softmax(out["logits"], dim=-1).detach().cpu().numpy()
            y_prob.append(prob)
            y_pred.append(prob.argmax(axis=1))
        y_prob_np = np.concatenate(y_prob, axis=0)
        y_pred_np = np.concatenate(y_pred, axis=0)

        from .training.metrics import compute_classification_metrics

        metrics = compute_classification_metrics(y_test, y_pred_np, y_prob_np)
        save_json(metrics, variant_dir / "test_metrics.json")

        import time

        x1 = torch.from_numpy(X_test[:1]).float().to(device)
        warmup = int(cfg["evaluation"].get("latency_warmup", 50))
        repeats = int(cfg["evaluation"].get("latency_repeats", 200))
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(warmup):
            _ = model(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        cpu_latency_ms = 1000.0 * (t1 - t0) / repeats

        from .utils.model_stats import count_parameters, estimate_model_size_mb, estimate_flops

        params = count_parameters(model)
        size_mb = estimate_model_size_mb(model)
        flops = estimate_flops(model, x1)

        row = {
            "suite": args.suite,
            "variant": variant_name,
            "description": description,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("f1_macro"),
            "weighted_f1": metrics.get("f1_weighted"),
            "mcc": metrics.get("mcc"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "roc_auc": _roc_value(metrics),
            "params": params,
            "model_size_mb": size_mb,
            "flops": flops if flops is not None else "",
            "latency_ms_per_forward": cpu_latency_ms,
            "best_epoch": res.best_epoch,
            "best_val_macro_f1": res.best_val_macro_f1,
        }
        summary_rows.append(row)
        manifest.append(
            {
                "variant": variant_name,
                "description": description,
                "output_dir": str(variant_dir),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = root_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    save_json(
        {
            "suite": args.suite,
            "experiment_name": args.experiment_name,
            "config_path": args.config,
            "csv_path": args.csv_path,
            "label_col": args.label_col,
            "variants": manifest,
        },
        root_dir / "ablation_manifest.json",
    )

    delta_df = _compute_step_deltas(summary_df, args.suite)
    if not delta_df.empty:
        delta_df.to_csv(root_dir / "ablation_step_deltas.csv", index=False)

    plot_ablation_accuracy_latency(summary_path, root_dir / "ablation_accuracy_vs_latency.png")
    plot_ablation_incremental_ladder(
        summary_path,
        root_dir / "ablation_macro_f1_ladder.png",
        value_col="macro_f1",
        title=f"Test macro-F1 ladder ({args.suite})",
    )
    plot_ablation_multi_metric(
        summary_path,
        root_dir / "ablation_metrics_grouped.png",
        metric_cols=["accuracy", "macro_f1", "weighted_f1", "mcc"],
        title=f"Test metrics by variant ({args.suite})",
    )

    report = {
        "suite": args.suite,
        "rows": summary_rows,
        "deltas_csv": "ablation_step_deltas.csv" if not delta_df.empty else None,
    }
    (root_dir / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_section("ABLATION COMPLETE")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
