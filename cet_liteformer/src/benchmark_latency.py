from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .evaluate import _load_df
from .data.preprocessing import (
    RobustLogIQRScaler,
    apply_imputer,
    clean_dataframe,
    load_imputer_state,
)
from .models.cet_liteformer import CETLiteFormer
from .utils.io import ensure_dir, load_json, load_yaml, save_json
from .utils.model_stats import count_parameters, estimate_flops, estimate_model_size_mb, get_memory_usage_mb


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    return ap.parse_args()


def _device(arg: str) -> torch.device:
    if arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _load_test_matrix(exp_dir: Path):
    cfg = load_yaml(exp_dir / "config_used.yaml")
    meta = load_json(exp_dir / "feature_metadata.json")
    splits = np.load(str(exp_dir / "splits.npz"))
    test_idx = splits["test_idx"].astype(np.int64)

    dataset_path = Path(meta["dataset_path"])
    label_col = str(meta["label_col"])
    drop_cols = meta.get("drop_cols", [])
    feat_orig = list(meta["feature_names_original"])
    keep_mask = np.asarray(meta["constant_keep_mask"], dtype=np.int64).astype(bool)

    df = _load_df(dataset_path, max_rows=cfg["data"].get("max_rows", None))
    df = clean_dataframe(df, drop_cols=drop_cols)
    X_all = df[feat_orig].to_numpy(dtype=np.float32, copy=True)

    imputer = load_imputer_state(exp_dir / "imputer_state.json")
    X_all = apply_imputer(X_all, imputer)
    scaler = RobustLogIQRScaler.load(exp_dir / "scaler.joblib")
    X_all = scaler.transform(X_all)
    X_all = X_all[:, keep_mask]

    corr_mask = meta.get("correlation_keep_mask")
    if corr_mask is not None:
        corr_mask = np.asarray(corr_mask, dtype=np.int64).astype(bool)
        if corr_mask.shape[0] != X_all.shape[1]:
            raise ValueError(
                "correlation_keep_mask length does not match feature count after constant removal."
            )
        X_all = X_all[:, corr_mask]

    X_test = X_all[test_idx]
    return X_test


@torch.no_grad()
def _time_forward(model: torch.nn.Module, x: torch.Tensor, device: torch.device, warmup: int, repeats: int) -> float:
    model.eval()
    x = x.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    cfg = load_yaml(exp_dir / "config_used.yaml")
    model_cfg = cfg["model"]

    warmup = int(args.warmup) if args.warmup is not None else int(cfg["evaluation"].get("latency_warmup", 100))
    repeats = int(args.repeats) if args.repeats is not None else int(cfg["evaluation"].get("latency_repeats", 1000))

    device = _device(args.device)

    X_test = _load_test_matrix(exp_dir)
    num_features = int(X_test.shape[1])
    num_classes = int(len(load_json(exp_dir / "label_mapping.json")))

    feature_groups_path = exp_dir / "feature_groups.json"
    if feature_groups_path.exists():
        fg = load_json(feature_groups_path)
        group_ids = [int(f["group_id"]) for f in fg.get("features", [])]
        if len(group_ids) != num_features:
            raise ValueError(
                f"feature_groups.json has {len(group_ids)} features but X has {num_features}."
            )
    else:
        group_ids = [0] * num_features

    mi_prior = torch.zeros(num_features, dtype=torch.float32)
    mi_csv = exp_dir / "feature_mi_scores.csv"
    if mi_csv.exists():
        try:
            mi_df = pd.read_csv(mi_csv)
            mi_prior = torch.tensor(mi_df["mi_normalized"].to_numpy(dtype=np.float32))
        except Exception:
            pass

    # instantiate and load model (architecture must match checkpoint)
    model = CETLiteFormer(
        num_features=num_features,
        num_classes=num_classes,
        group_ids=group_ids,
        mi_prior=mi_prior,
        embed_dim=int(model_cfg["embed_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        rff_dim=int(model_cfg["rff_dim"]),
        sigma=float(model_cfg["sigma"]),
        dropout=float(model_cfg["dropout"]),
        use_cls_token=bool(model_cfg.get("use_cls_token", True)),
        use_entropy_gate=bool(model_cfg.get("use_entropy_gate", True)),
        use_correntropy_attention=bool(model_cfg.get("use_correntropy_attention", True)),
        use_early_exit=bool(model_cfg.get("use_early_exit", True)),
        early_exit_threshold=float(model_cfg.get("early_exit_threshold", 0.90)),
        ffn_bottleneck_ratio=float(model_cfg.get("ffn_bottleneck_ratio", 0.5)),
        gate_prior_strength=float(model_cfg.get("gate_prior_strength", 1.0)),
    )
    ckpt = torch.load(str(exp_dir / "best_model.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    # stats
    sample = torch.from_numpy(X_test[:1]).float()
    flops = estimate_flops(model, sample.to(device))
    params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)

    # measure latency: early exit off vs on
    results = {"params": params, "model_size_mb": size_mb, "flops": flops, "device": str(device)}

    def bench_variant(enable_early_exit: bool) -> Dict[str, Any]:
        model.use_early_exit = bool(enable_early_exit)
        model.eval()
        # latency batch=1
        x1 = torch.from_numpy(X_test[:1]).float()
        sec = _time_forward(model, x1, device=device, warmup=warmup, repeats=repeats)
        latency_ms = 1000.0 * sec

        # throughput batch sizes
        through = {}
        for bs in [32, 128, 256]:
            xb = torch.from_numpy(X_test[:bs]).float()
            sec_b = _time_forward(model, xb, device=device, warmup=max(10, warmup // 10), repeats=max(50, repeats // 10))
            through[f"bs{bs}"] = float(bs / sec_b)

        return {"latency_ms_bs1": float(latency_ms), "throughput_flows_per_sec": through}

    results["early_exit_disabled"] = bench_variant(False)
    results["early_exit_enabled"] = bench_variant(True)
    results["memory_rss_mb"] = float(get_memory_usage_mb())

    ensure_dir(exp_dir)
    save_json(results, exp_dir / "latency_report.json")

    # csv summary
    rows = []
    for k in ["early_exit_disabled", "early_exit_enabled"]:
        rows.append(
            {
                "variant": k,
                "latency_ms_bs1": results[k]["latency_ms_bs1"],
                "throughput_bs32": results[k]["throughput_flows_per_sec"]["bs32"],
                "throughput_bs128": results[k]["throughput_flows_per_sec"]["bs128"],
                "throughput_bs256": results[k]["throughput_flows_per_sec"]["bs256"],
                "params": params,
                "model_size_mb": size_mb,
                "flops": flops if flops is not None else "",
                "device": str(device),
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(exp_dir / "latency_report.csv", index=False)


if __name__ == "__main__":
    main()

