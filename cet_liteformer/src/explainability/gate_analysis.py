from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class GateStats:
    mean: np.ndarray
    std: np.ndarray


def compute_gate_stats(gate_scores: np.ndarray) -> GateStats:
    """
    gate_scores: [N,F]
    """
    gate_scores = np.asarray(gate_scores, dtype=np.float32)
    return GateStats(mean=gate_scores.mean(axis=0), std=gate_scores.std(axis=0))

