from __future__ import annotations

from typing import Optional

import numpy as np


def combined_importance(mi_normalized: np.ndarray, mean_gate_score: np.ndarray) -> np.ndarray:
    """
    Combined prior-weighted gate importance:
      combined = mi_normalized * mean_gate_score
    """
    mi = np.asarray(mi_normalized, dtype=np.float32)
    g = np.asarray(mean_gate_score, dtype=np.float32)
    if mi.shape != g.shape:
        raise ValueError("mi_normalized and mean_gate_score must have same shape.")
    return mi * g

