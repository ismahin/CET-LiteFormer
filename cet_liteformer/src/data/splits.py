from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_train_val_test_split(
    n_samples: int,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
    stratify: bool = True,
) -> SplitIndices:
    if test_size <= 0.0 or val_size <= 0.0 or (test_size + val_size) >= 1.0:
        raise ValueError("test_size and val_size must be >0 and sum to <1.")

    idx_all = np.arange(n_samples)
    strat = y if stratify else None

    idx_train_val, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    # val split is relative to remaining (train+val)
    val_rel = val_size / (1.0 - test_size)
    strat2 = y[idx_train_val] if stratify else None

    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_rel,
        random_state=seed,
        stratify=strat2,
    )

    return SplitIndices(train_idx=idx_train, val_idx=idx_val, test_idx=idx_test)

