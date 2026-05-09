from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from .data.preprocessing import (
    RobustLogIQRScaler,
    auto_detect_numeric_features,
    clean_dataframe,
    load_arff_dataset,
    load_csv_dataset,
    load_imputer_state,
)
from .models.cet_liteformer import CETLiteFormer
from .training.metrics import compute_classification_metrics, sklearn_classification_report_df
from .utils.io import ensure_dir, load_json, load_yaml, save_json
from .utils.logger import print_section
from .utils.plots import plot_confusion_matrix, plot_gate_importance_topk, plot_roc_curve

import time

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", type=str, required=True, help="outputs/<experiment_name>")
    ap.add_argument("--checkpoint", type=str, default="best_model.pt", help="best_model.pt or last_model.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=512)
    return ap.parse_args()


def _device(arg: str) -> torch.device:
    if arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _load_df(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".arff":
        return load_arff_dataset(str(path), max_rows=max_rows)
    return load_csv_dataset(str(path), max_rows=max_rows)


@torch.no_grad()
def _measure_latency(
    model: CETLiteFormer,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
    enable_early_exit: bool,
) -> Dict[str, Any]:
    """
    Measures average forward latency (seconds) for a fixed batch size.
    Also reports early-exit usage stats (exit layer chosen) when enabled.
    """
    model.use_early_exit = bool(enable_early_exit)
    model.eval()

    # Prepare a fixed batch tensor (repeat from X if smaller)
    if len(X) == 0:
        raise ValueError("Empty X for latency measurement.")
    idx = np.arange(min(len(X), batch_size))
    xb_np = X[idx]
    if xb_np.shape[0] < batch_size:
        reps = int(np.ceil(batch_size / xb_np.shape[0]))
        xb_np = np.tile(xb_np, (reps, 1))[:batch_size]

    xb = torch.from_numpy(xb_np).float().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup
    for _ in range(int(warmup)):
        _ = model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed repeats
    exit_used_counts: Dict[str, int] = {}
    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        out = model(xb)
        eu = out.get("exit_used", None)
        key = str(int(eu)) if eu is not None else "-1"
        exit_used_counts[key] = exit_used_counts.get(key, 0) + 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    sec = (t1 - t0) / max(int(repeats), 1)
    return {
        "batch_size": int(batch_size),
        "avg_latency_ms": float(1000.0 * sec),
        "throughput_flows_per_sec": float(batch_size / sec) if sec > 0 else None,
        "early_exit_enabled": bool(enable_early_exit),
        "exit_used_counts": exit_used_counts,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"experiment_dir not found: {exp_dir}")

    print_section("EVALUATING CET-LITEFORMER")

    cfg = load_yaml(exp_dir / "config_used.yaml")
    feature_meta = load_json(exp_dir / "feature_metadata.json")
    label_mapping = load_json(exp_dir / "label_mapping.json")
    le = joblib.load(str(exp_dir / "label_encoder.joblib"))
    imputer = load_imputer_state(exp_dir / "imputer_state.json")
    splits = np.load(str(exp_dir / "splits.npz"))
    feature_groups = load_json(exp_dir / "feature_groups.json") if (exp_dir / "feature_groups.json").exists() else None

    dataset_path = Path(feature_meta["dataset_path"])
    label_col = str(feature_meta["label_col"])
    drop_cols = feature_meta.get("drop_cols", [])

    df = _load_df(dataset_path, max_rows=cfg["data"].get("max_rows", None))
    df = clean_dataframe(df, drop_cols=drop_cols)

    # features: enforce the exact original feature list, then apply keep mask
    feature_names_original = list(feature_meta["feature_names_original"])
    keep_mask = np.asarray(feature_meta["constant_keep_mask"], dtype=np.int64).astype(bool)
    feature_names_final = list(feature_meta["feature_names_final"])

    missing_cols = [c for c in feature_names_original + [label_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset columns missing compared to training metadata: {missing_cols[:20]}")

    X_all = df[feature_names_original].to_numpy(dtype=np.float32, copy=True)
    y_all_raw = df[label_col].astype(str).to_numpy()

    # impute, scale, constant-remove
    from .data.preprocessing import apply_imputer

    X_all = apply_imputer(X_all, imputer)
    scaler = RobustLogIQRScaler.load(exp_dir / "scaler.joblib")
    X_all = scaler.transform(X_all)
    X_all = X_all[:, keep_mask]

    corr_mask = feature_meta.get("correlation_keep_mask")
    if corr_mask is not None:
        corr_mask = np.asarray(corr_mask, dtype=np.int64).astype(bool)
        if corr_mask.shape[0] != X_all.shape[1]:
            raise ValueError(
                "correlation_keep_mask length does not match feature count after constant removal."
            )
        X_all = X_all[:, corr_mask]

    # label encoding using saved encoder
    y_all = le.transform(y_all_raw)

    train_idx = splits["train_idx"].astype(np.int64)
    val_idx = splits["val_idx"].astype(np.int64)
    test_idx = splits["test_idx"].astype(np.int64)

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    num_features = int(X_test.shape[1])
    num_classes = int(len(label_mapping))

    # load group_ids aligned to final feature order
    if feature_groups is None:
        group_ids = [0] * num_features
    else:
        feats = feature_groups.get("features", [])
        group_ids = [int(f["group_id"]) for f in feats]
        if len(group_ids) != num_features:
            raise ValueError(
                f"feature_groups.json group_ids length {len(group_ids)} does not match num_features {num_features}."
            )

    # load MI prior if available (normalized)
    mi_prior = torch.zeros(num_features, dtype=torch.float32)
    mi_path = exp_dir / "feature_mi_scores.csv"
    if mi_path.exists():
        try:
            mi_df = pd.read_csv(mi_path)
            mi_prior = torch.tensor(mi_df["mi_normalized"].to_numpy(dtype=np.float32), dtype=torch.float32)
        except Exception:
            pass

    # build model with correct shapes; buffers (group_ids, mi_prior) load from state_dict
    model = CETLiteFormer(
        num_features=num_features,
        num_classes=num_classes,
        group_ids=group_ids,
        mi_prior=mi_prior,
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

    ckpt = torch.load(str(exp_dir / args.checkpoint), map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    device = _device(args.device)
    model.to(device)

    target_names = [k for k, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]

    def eval_split(split_name: str, X: np.ndarray, y: np.ndarray, save_predictions: bool = False):
        bs = int(args.batch_size)
        y_true, y_pred = [], []
        y_prob = []
        exit_used_all = []
        conf_all = []
        gate_scores_all = []

        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i : i + bs]).to(device)
            out = model(xb)
            logits = out["logits"]
            prob = torch.softmax(logits, dim=-1)
            pred = prob.argmax(dim=-1)

            y_true.append(y[i : i + bs])
            y_pred.append(pred.detach().cpu().numpy())
            y_prob.append(prob.detach().cpu().numpy())

            eu = out.get("exit_used", None)
            if eu is None:
                exit_used_all.extend([-1] * len(pred))
            else:
                exit_used_all.extend([int(eu)] * len(pred))

            conf = prob.max(dim=-1).values.detach().cpu().numpy()
            conf_all.append(conf)

            if out.get("gate_scores", None) is not None:
                gate_scores_all.append(out["gate_scores"].detach().cpu().numpy())

        y_true_np = np.concatenate(y_true, axis=0)
        y_pred_np = np.concatenate(y_pred, axis=0)
        y_prob_np = np.concatenate(y_prob, axis=0)
        conf_np = np.concatenate(conf_all, axis=0)

        metrics = compute_classification_metrics(y_true_np, y_pred_np, y_prob_np)
        save_json(metrics, exp_dir / f"{split_name}_metrics.json")

        rep_df = sklearn_classification_report_df(y_true_np, y_pred_np, target_names=target_names)
        rep_df.to_csv(exp_dir / f"{split_name}_classification_report.csv", index=True)

        cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))
        np.save(exp_dir / f"{split_name}_confusion_matrix.npy", cm)
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(exp_dir / f"{split_name}_confusion_matrix.csv")

        if split_name == "test":
            plot_confusion_matrix(cm, target_names, exp_dir / "confusion_matrix.png")
            # ROC curve figure for test
            try:
                plot_roc_curve(y_true_np, y_prob_np, target_names, exp_dir / "roc_curve.png")
            except Exception:
                pass

        if save_predictions:
            pred_df = pd.DataFrame(
                {
                    "y_true": y_true_np,
                    "y_pred": y_pred_np,
                    "confidence": conf_np,
                    "exit_used": np.asarray(exit_used_all, dtype=np.int64),
                }
            )
            for ci, name in enumerate(target_names):
                pred_df[f"prob_{name}"] = y_prob_np[:, ci]
            pred_df.to_csv(exp_dir / f"{split_name}_predictions.csv", index=False)

        return {"y_true": y_true_np, "y_pred": y_pred_np, "y_prob": y_prob_np, "gate_scores_all": gate_scores_all}

    # Evaluate all splits; save detailed predictions for test only by default
    eval_split("train", X_train, y_train, save_predictions=False)
    eval_split("val", X_val, y_val, save_predictions=False)
    test_out = eval_split("test", X_test, y_test, save_predictions=True)

    # gate importance
    gate_scores_all = test_out.get("gate_scores_all", [])
    if gate_scores_all:
        gate_arr = np.concatenate(gate_scores_all, axis=0)  # [N,F]
        mean_gate = gate_arr.mean(axis=0)
        std_gate = gate_arr.std(axis=0)

        # MI normalized if available
        mi_path = exp_dir / "feature_mi_scores.csv"
        if mi_path.exists():
            mi_df = pd.read_csv(mi_path)
            mi_norm = mi_df["mi_normalized"].to_numpy(dtype=np.float32)
            group_name = mi_df["group_name"].astype(str).tolist()
        else:
            mi_norm = np.ones_like(mean_gate, dtype=np.float32)
            group_name = ["unknown"] * len(mean_gate)

        gate_imp = pd.DataFrame(
            {
                "feature_name": feature_names_final,
                "group_name": group_name,
                "mi_score": mi_norm,
                "mean_gate_score": mean_gate,
                "std_gate_score": std_gate,
                "combined_importance": mi_norm * mean_gate,
            }
        ).sort_values("combined_importance", ascending=False)
        gate_imp.to_csv(exp_dir / "gate_feature_importance.csv", index=False)
        plot_gate_importance_topk(gate_imp, topk=30, out_path=exp_dir / "gate_importance_top30.png")

    # latency evaluation (prediction-time)
    eval_cfg = cfg.get("evaluation", {}) or {}
    if bool(eval_cfg.get("measure_latency", False)):
        bs_lat = int(eval_cfg.get("latency_batch_size", 1))
        warmup = int(eval_cfg.get("latency_warmup", 100))
        repeats = int(eval_cfg.get("latency_repeats", 1000))

        # Use test split for latency by default (can be changed later)
        lat_disabled = _measure_latency(
            model=model,
            X=X_test,
            device=device,
            batch_size=bs_lat,
            warmup=warmup,
            repeats=repeats,
            enable_early_exit=False,
        )
        lat_enabled = _measure_latency(
            model=model,
            X=X_test,
            device=device,
            batch_size=bs_lat,
            warmup=warmup,
            repeats=repeats,
            enable_early_exit=True,
        )

        lat_report = {
            "device": str(device),
            "latency_batch_size": bs_lat,
            "latency_warmup": warmup,
            "latency_repeats": repeats,
            "early_exit_disabled": lat_disabled,
            "early_exit_enabled": lat_enabled,
        }
        save_json(lat_report, exp_dir / "latency_eval.json")

        # compact CSV
        pd.DataFrame(
            [
                {
                    "variant": "early_exit_disabled",
                    "avg_latency_ms": lat_disabled["avg_latency_ms"],
                    "throughput_flows_per_sec": lat_disabled["throughput_flows_per_sec"],
                },
                {
                    "variant": "early_exit_enabled",
                    "avg_latency_ms": lat_enabled["avg_latency_ms"],
                    "throughput_flows_per_sec": lat_enabled["throughput_flows_per_sec"],
                },
            ]
        ).to_csv(exp_dir / "latency_eval.csv", index=False)


if __name__ == "__main__":
    main()

