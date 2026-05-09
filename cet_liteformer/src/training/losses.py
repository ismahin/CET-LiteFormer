from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights_from_labels(y: np.ndarray, num_classes: int, eps: float = 1e-8) -> torch.Tensor:
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = total / (num_classes * (counts + eps))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Multi-class focal loss:
      FL = -alpha_y (1 - p_y)^gamma log(p_y)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=True)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean/sum/none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B,C], target: [B]
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        idx = target.view(-1, 1)
        logp_y = logp.gather(1, idx).squeeze(1)
        p_y = p.gather(1, idx).squeeze(1)

        if self.alpha is not None:
            # Ensure alpha lives on the same device as logits/targets
            alpha = self.alpha.to(device=logits.device)
            alpha_y = alpha.gather(0, target)
        else:
            alpha_y = torch.ones_like(p_y)

        loss = -alpha_y * ((1.0 - p_y).clamp(min=0.0) ** self.gamma) * logp_y
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@dataclass
class LossOutput:
    loss: torch.Tensor
    main_loss: torch.Tensor
    exit_loss: torch.Tensor
    gate_loss: torch.Tensor


class CETLiteFormerLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_focal_loss: bool,
        focal_gamma: float,
        class_weights: Optional[torch.Tensor],
        gate_l1_lambda: float,
        exit_loss_lambda: float,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_focal_loss = bool(use_focal_loss)
        self.gate_l1_lambda = float(gate_l1_lambda)
        self.exit_loss_lambda = float(exit_loss_lambda)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float(), persistent=True)
        else:
            self.class_weights = None

        if self.use_focal_loss:
            self.main_criterion = FocalLoss(gamma=focal_gamma, alpha=self.class_weights, reduction="mean")
            self.exit_criterion = FocalLoss(gamma=focal_gamma, alpha=self.class_weights, reduction="mean")
        else:
            self.main_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            self.exit_criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, model_out: Dict[str, Any], y: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        exit_logits: List[torch.Tensor] = model_out.get("exit_logits", [])
        gate_scores = model_out.get("gate_scores", None)

        main_loss = self.main_criterion(logits, y)
        if exit_logits:
            exit_losses = [self.exit_criterion(z, y) for z in exit_logits]
            exit_loss = torch.stack(exit_losses).mean()
        else:
            exit_loss = torch.tensor(0.0, device=logits.device)

        if gate_scores is not None:
            gate_loss = gate_scores.abs().mean()
        else:
            gate_loss = torch.tensor(0.0, device=logits.device)

        total = main_loss + self.exit_loss_lambda * exit_loss + self.gate_l1_lambda * gate_loss
        return {
            "loss": total,
            "main_loss": main_loss.detach(),
            "exit_loss": exit_loss.detach(),
            "gate_loss": gate_loss.detach(),
        }

