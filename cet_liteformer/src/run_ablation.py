from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import FlowTabularDataset
from .data.preprocessing import build_preprocessed_splits, compute_mi_prior
from .models.baselines import MLPBaseline, StandardTransformerBaseline
from .models.cet_liteformer import CETLiteFormer
from .training.losses import CETLiteFormerLoss, compute_class_weights_from_labels
from .training.scheduler import build_scheduler
from .training.trainer import train as train_loop
from .utils.io import ensure_dir, load_yaml, save_json
from .utils.logger import print_section
from .utils.plots import plot_ablation_accuracy_latency
from .utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--label_col", type=str, default="")
    ap.add_argument("--experiment_name", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()


def _device(arg: str) -> torch.device:
    if arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _variant_cfgs(base_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    cfgs: List[Tuple[str, Dict[str, Any]]] = []

    def cpy():
        return copy.deepcopy(base_cfg)

    # 1) full
    cfgs.append(("full_cet_liteformer", cpy()))

    # 2) no entropy gate
    v = cpy()
    v["model"]["use_entropy_gate"] = False
    cfgs.append(("no_entropy_gate", v))

    # 3) standard attention
    v = cpy()
    v["model"]["use_correntropy_attention"] = False
    cfgs.append(("standard_attention_instead_of_correntropy", v))

    # 4) no early exit
    v = cpy()
    v["model"]["use_early_exit"] = False
    cfgs.append(("no_early_exit", v))

    # 5) no focal loss
    v = cpy()
    v["training"]["use_focal_loss"] = False
    cfgs.append(("no_focal_loss", v))

    # 6) no gate sparsity
    v = cpy()
    v["training"]["gate_l1_lambda"] = 0.0
    cfgs.append(("no_gate_sparsity", v))

    # 7) shallow mlp baseline
    v = cpy()
    v["model"]["name"] = "MLPBaseline"
    cfgs.append(("shallow_mlp_baseline", v))

    # 8) standard transformer baseline
    v = cpy()
    v["model"]["name"] = "StandardTransformerBaseline"
    cfgs.append(("standard_transformer_baseline", v))

    return cfgs


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    base_cfg["data"]["csv_path"] = args.csv_path
    base_cfg["data"]["label_col"] = args.label_col
    base_cfg["experiment"]["name"] = args.experiment_name

    out_root = Path(base_cfg["experiment"]["output_dir"])
    root_dir = out_root / args.experiment_name
    ensure_dir(root_dir)

    print_section("RUNNING ABLATION STUDY")
    print(f"Ablation root: {root_dir}")

    seed = int(base_cfg["experiment"]["seed"])
    set_seed(seed)
    device = _device(args.device)

    summary_rows = []
    for variant_name, cfg in _variant_cfgs(base_cfg):
        variant_dir = root_dir / variant_name
        ensure_dir(variant_dir)

        # preprocessing (persist per variant for full reproducibility)
        prep = build_preprocessed_splits(
            dataset_path=cfg["data"]["csv_path"],
            label_col=str(cfg["data"].get("label_col", "") or ""),
            drop_cols=cfg["data"].get("drop_cols", []) or [],
            test_size=float(cfg["data"]["test_size"]),
            val_size=float(cfg["data"]["val_size"]),
            stratify=bool(cfg["data"].get("stratify", True)),
            seed=seed,
            max_rows=cfg["data"].get("max_rows", None),
            missing_strategy=str(cfg["data"].get("missing_strategy", "median")),
            normalize=str(cfg["data"].get("normalize", "log_iqr")),
            remove_constant_features=bool(cfg["data"].get("remove_constant_features", True)),
            output_dir=variant_dir,
        )

        X_train, y_train = prep["X_train"], prep["y_train"]
        X_val, y_val = prep["X_val"], prep["y_val"]
        X_test, y_test = prep["X_test"], prep["y_test"]

        feature_names = prep["feature_names"]
        group_ids = prep["group_ids"]
        group_names = prep["group_names"]
        num_features = int(X_train.shape[1])
        num_classes = int(len(prep["label_mapping"]))

        # MI prior (only used if entropy gate enabled)
        mi_norm = compute_mi_prior(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            group_names=group_names,
            output_csv_path=variant_dir / "feature_mi_scores.csv",
        )
        mi_prior = torch.tensor(mi_norm, dtype=torch.float32)

        # loaders
        bs = int(cfg["training"]["batch_size"])
        train_loader = DataLoader(FlowTabularDataset(X_train, y_train), batch_size=bs, shuffle=True, num_workers=int(args.num_workers), pin_memory=True)
        val_loader = DataLoader(FlowTabularDataset(X_val, y_val), batch_size=bs, shuffle=False, num_workers=int(args.num_workers), pin_memory=True)

        # model selection
        if cfg["model"].get("name") == "MLPBaseline":
            model = MLPBaseline(num_features=num_features, num_classes=num_classes, dropout=float(cfg["model"].get("dropout", 0.15)))
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
                use_early_exit=bool(cfg["model"].get("use_early_exit", True)),
                early_exit_threshold=float(cfg["model"].get("early_exit_threshold", 0.90)),
                ffn_bottleneck_ratio=float(cfg["model"].get("ffn_bottleneck_ratio", 0.5)),
                gate_prior_strength=float(cfg["model"].get("gate_prior_strength", 1.0)),
            )

        # loss/optim
        class_weights = compute_class_weights_from_labels(y_train, num_classes=num_classes)
        loss_fn = CETLiteFormerLoss(
            num_classes=num_classes,
            use_focal_loss=bool(cfg["training"].get("use_focal_loss", True)),
            focal_gamma=float(cfg["training"].get("focal_gamma", 2.0)),
            class_weights=class_weights,
            gate_l1_lambda=float(cfg["training"].get("gate_l1_lambda", 1e-4)),
            exit_loss_lambda=float(cfg["training"].get("exit_loss_lambda", 0.3)),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
        scheduler = build_scheduler(str(cfg["training"].get("scheduler", "cosine")), optimizer, epochs=int(cfg["training"]["epochs"]))

        # train
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

        # quick test eval (final logits only)
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

        # latency (simple: bs1 CPU time on chosen device)
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

        # params/size
        from .utils.model_stats import count_parameters, estimate_model_size_mb, estimate_flops

        params = count_parameters(model)
        size_mb = estimate_model_size_mb(model)
        flops = estimate_flops(model, x1)

        summary_rows.append(
            {
                "variant": variant_name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("f1_macro"),
                "weighted_f1": metrics.get("f1_weighted"),
                "params": params,
                "model_size_mb": size_mb,
                "flops": flops if flops is not None else "",
                "cpu_latency_ms": cpu_latency_ms,
                "gpu_latency_ms": cpu_latency_ms if device.type == "cuda" else "",
                "throughput_flows_per_sec": "",
                "early_exit_rate": "",
                "best_epoch": res.best_epoch,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(root_dir / "ablation_summary.csv", index=False)
    plot_ablation_accuracy_latency(root_dir / "ablation_summary.csv", root_dir / "ablation_accuracy_latency.png")


if __name__ == "__main__":
    main()

