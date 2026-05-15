from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _savefig(path: str | Path, dpi: int = 200) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    plt.close()


def plot_training_curves(training_log_csv: str | Path, out_dir: str | Path) -> None:
    import pandas as pd

    out = Path(out_dir)
    df = pd.read_csv(training_log_csv)
    if df.empty:
        return

    # loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _savefig(out / "loss_curve.png")

    # macro f1 curve
    if "val_f1_macro" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["epoch"], df["val_f1_macro"], label="val_macro_f1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.grid(True, alpha=0.3)
        plt.legend()
        _savefig(out / "macro_f1_curve.png")

    # accuracy curve
    if "val_accuracy" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["epoch"], df["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        _savefig(out / "accuracy_curve.png")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str | Path) -> None:
    import seaborn as sns

    cm = np.asarray(cm)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _savefig(out_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
) -> None:
    """
    ROC curve plot.
    - Binary: plot single ROC with AUC.
    - Multiclass: One-vs-Rest ROC per class (lightweight, can be many curves).
    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    c = y_prob.shape[1]

    plt.figure(figsize=(6, 5))
    if c == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    else:
        Y = label_binarize(y_true, classes=list(range(c)))
        # plot at most first 10 curves for readability
        max_curves = min(c, 10)
        for i in range(max_curves):
            fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})", linewidth=1.2)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    _savefig(out_path)


def plot_gate_importance_topk(gate_df, topk: int, out_path: str | Path) -> None:
    # gate_df: dataframe sorted by combined_importance
    df = gate_df.head(int(topk)).copy()
    df = df.iloc[::-1]  # plot bottom-up

    plt.figure(figsize=(8, 10))
    plt.barh(df["feature_name"], df["combined_importance"])
    plt.xlabel("combined_importance (mi_normalized * mean_gate_score)")
    plt.ylabel("feature")
    plt.grid(True, axis="x", alpha=0.3)
    _savefig(out_path)


def plot_ablation_accuracy_latency(ablation_csv: str | Path, out_path: str | Path) -> None:
    import pandas as pd

    df = pd.read_csv(ablation_csv)
    if df.empty:
        return

    # scatter: macro_f1 vs cpu_latency_ms
    plt.figure(figsize=(7, 5))
    x = df.get("cpu_latency_ms", None)
    y = df.get("macro_f1", None)
    if x is None or y is None:
        return
    plt.scatter(df["cpu_latency_ms"], df["macro_f1"])
    for _, row in df.iterrows():
        plt.text(row["cpu_latency_ms"], row["macro_f1"], str(row["variant"]), fontsize=8)
    plt.xlabel("CPU latency (ms)")
    plt.ylabel("Macro F1")
    plt.grid(True, alpha=0.3)
    _savefig(out_path)


def plot_ablation_incremental_ladder(
    ablation_csv: str | Path,
    out_path: str | Path,
    value_col: str = "macro_f1",
    title: str = "Ablation ladder (test set)",
) -> None:
    """
    Bar + line plot for ordered ablation steps (e.g. 01_..., 02_..., or T1_..., T2_...).
    """
    import pandas as pd

    df = pd.read_csv(ablation_csv)
    if df.empty or value_col not in df.columns:
        return

    df = df.sort_values("variant").reset_index(drop=True)
    labels = df["variant"].tolist()
    y = df[value_col].astype(float).to_numpy()

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    ax1.bar(x, y, color="#4C72B0", alpha=0.85, label=value_col)
    ax1.plot(x, y, color="#DD8452", marker="o", linewidth=2, markersize=6, label="trend")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel(value_col.replace("_", " ").title())
    ax1.set_title(title)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="lower right")
    _savefig(out_path)


def plot_ablation_multi_metric(
    ablation_csv: str | Path,
    out_path: str | Path,
    metric_cols: Optional[List[str]] = None,
    title: str = "Ablation: test metrics",
) -> None:
    """Grouped bars for several metrics (each normalized to [0,1] column-wise for shape only)."""
    import pandas as pd

    if metric_cols is None:
        metric_cols = ["accuracy", "macro_f1", "weighted_f1", "mcc"]

    df = pd.read_csv(ablation_csv)
    if df.empty:
        return
    df = df.sort_values("variant").reset_index(drop=True)
    labels = df["variant"].tolist()
    present = [c for c in metric_cols if c in df.columns]
    if not present:
        return

    x = np.arange(len(labels))
    width = 0.8 / max(len(present), 1)
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.15), 5))
    for i, col in enumerate(present):
        offset = (i - (len(present) - 1) / 2) * width
        ax.bar(x + offset, df[col].astype(float), width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(out_path)


def plot_feature_correlations(
    corr_df,
    out_path: str | Path,
    topk: int = 60,
    selected_mask_col: str = "selected",
    corr_col: str = "abs_corr",
) -> None:
    """
    Bar plot of top correlated features (by abs correlation).
    Highlights selected features if `selected` column exists.
    """
    df = corr_df.sort_values(corr_col, ascending=False).head(int(topk)).copy()
    df = df.iloc[::-1]

    selected = df[selected_mask_col].astype(bool).to_numpy() if selected_mask_col in df.columns else None
    colors = None
    if selected is not None:
        colors = ["#1f77b4" if s else "#bbbbbb" for s in selected]  # selected blue, others gray

    plt.figure(figsize=(9, 12))
    plt.barh(df["feature_name"], df[corr_col], color=colors)
    plt.xlabel("|Spearman correlation| (train-only)")
    plt.ylabel("feature")
    plt.grid(True, axis="x", alpha=0.3)
    _savefig(out_path)

