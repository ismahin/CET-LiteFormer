from __future__ import annotations

import warnings
from typing import Optional

import torch


def resolve_device(arg: Optional[str] = None) -> torch.device:
    """Resolve cuda/cpu from CLI. Falls back to CPU if CUDA is requested but unavailable."""
    req = (arg or "auto").strip().lower()
    if req in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device((arg or "cuda").strip())
        warnings.warn(
            "CUDA requested but not available (CPU-only PyTorch or no GPU). Using CPU.",
            UserWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    return torch.device((arg or "cpu").strip())
