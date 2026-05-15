from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 0,
):
    """
    Learning-rate schedulers stepped once per epoch (after each training epoch).

    cosine_warmup: linear warmup (start_factor -> 1.0), then cosine decay to 0
    over the remaining epochs. Helps larger / deeper models stabilize early training.
    """
    name = (name or "").lower()
    epochs = int(epochs)
    warmup_epochs = max(0, int(warmup_epochs))

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)

    if name in ("cosine_warmup", "cosine+warmup", "warmup_cosine"):
        if warmup_epochs <= 0:
            warmup_epochs = max(1, min(15, epochs // 10))
        if warmup_epochs >= epochs:
            return CosineAnnealingLR(optimizer, T_max=epochs)

        # Warmup: lr goes from start_factor * base_lr to base_lr.
        warm = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_steps = epochs - warmup_epochs
        cos = CosineAnnealingLR(optimizer, T_max=max(1, cosine_steps))
        return SequentialLR(optimizer, schedulers=[warm, cos], milestones=[warmup_epochs])

    if name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    if name in ("none", ""):
        return None
    raise ValueError(f"Unknown scheduler: {name}")

