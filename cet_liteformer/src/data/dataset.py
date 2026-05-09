from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FlowTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D [N,F], got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D [N], got {y.shape}")
        if len(X) != len(y):
            raise ValueError("X and y must have same length.")
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

