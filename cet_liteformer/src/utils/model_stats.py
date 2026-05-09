from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import psutil
import torch


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    # rough: parameter bytes only
    n_bytes = 0
    for p in model.parameters():
        n_bytes += p.numel() * p.element_size()
    return float(n_bytes) / (1024.0 * 1024.0)


def estimate_flops(model: torch.nn.Module, sample_input: torch.Tensor) -> Optional[float]:
    """
    Best-effort FLOPs estimate via THOP.

    Important: THOP uses forward hooks; if profiling fails mid-run, hooks can leak and
    interfere with training. To avoid contaminating the live model (esp. on GPU),
    we profile a detached CPU copy.
    """
    try:
        import copy

        from thop import profile  # type: ignore

        m = copy.deepcopy(model).cpu().eval()
        x = sample_input.detach().cpu()
        macs, _params = profile(m, inputs=(x,), verbose=False)
        return float(2.0 * macs)  # FLOPs ≈ 2 * MACs
    except Exception:
        return None


def get_memory_usage_mb() -> float:
    proc = psutil.Process()
    rss = float(proc.memory_info().rss)
    return rss / (1024.0 * 1024.0)

