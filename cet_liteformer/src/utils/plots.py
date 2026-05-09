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

