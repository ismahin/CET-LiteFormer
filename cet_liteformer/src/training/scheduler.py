from __future__ import annotations

from typing import Optional

import torch


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
):
    name = (name or "").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    if name in ("none", ""):
        return None
    raise ValueError(f"Unknown scheduler: {name}")

