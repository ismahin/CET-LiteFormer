from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    out: Dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
    }

    if y_prob is not None:
        try:
            num_classes = y_prob.shape[1]
            if num_classes == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            pass

    return out


def sklearn_classification_report_df(y_true: np.ndarray, y_pred: np.ndarray, target_names: Optional[list[str]] = None):
    rep = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    import pandas as pd

    df = pd.DataFrame(rep).T
    return df

