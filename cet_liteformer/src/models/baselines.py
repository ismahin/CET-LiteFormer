from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import StandardMultiHeadSelfAttention
from .layers import AttentionPooling, ClassifierHead, FeatureTokenizer


class MLPBaseline(nn.Module):
    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.fc2(x)))
        logits = self.fc3(x)
        return {"logits": logits, "exit_logits": [], "gate_scores": None, "exit_used": None}


class StandardTransformerBaseline(nn.Module):
    """
    Fair baseline:
    - same FeatureTokenizer (feature + group embeddings)
    - standard MHSA
    - no entropy gate, no correntropy attention, no early exit
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        group_ids: List[int],
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.15,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.use_cls_token = bool(use_cls_token)
        self.tokenizer = FeatureTokenizer(num_features=num_features, embed_dim=embed_dim, group_ids=group_ids, use_cls_token=self.use_cls_token)

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(embed_dim),
                        "attn": StandardMultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout),
                        "norm2": nn.LayerNorm(embed_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(embed_dim, 4 * embed_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(4 * embed_dim, embed_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.pool: Optional[AttentionPooling] = None
        if not self.use_cls_token:
            self.pool = AttentionPooling(embed_dim)

        self.head = ClassifierHead(embed_dim, num_classes, dropout=dropout)

    def _represent(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            return tokens[:, 0, :]
        assert self.pool is not None
        return self.pool(tokens)

    def forward(self, x: torch.Tensor):
        t = self.tokenizer(x)
        for blk in self.blocks:
            t = t + blk["attn"](blk["norm1"](t))
            t = t + blk["ffn"](blk["norm2"](t))
        rep = self._represent(t)
        logits = self.head(rep)
        return {"logits": logits, "exit_logits": [], "gate_scores": None, "exit_used": None}

