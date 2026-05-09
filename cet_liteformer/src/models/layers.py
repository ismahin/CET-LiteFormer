from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    r"""
    Tokenizes tabular flow features into a sequence of learnable feature tokens.

    For feature j:
      e_j = W_j * x_j + b_j + p_j + g_{group(j)}

    - W_j, b_j are feature-specific parameters
    - p_j is feature-id embedding
    - g is group embedding
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        group_ids: List[int],
        num_groups: Optional[int] = None,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.embed_dim = int(embed_dim)
        self.use_cls_token = bool(use_cls_token)

        if len(group_ids) != self.num_features:
            raise ValueError("group_ids length must match num_features.")
        self.register_buffer("group_ids", torch.tensor(group_ids, dtype=torch.long), persistent=True)

        if num_groups is None:
            num_groups = int(max(group_ids) + 1) if group_ids else 1
        self.num_groups = int(num_groups)

        self.weight = nn.Parameter(torch.randn(self.num_features, self.embed_dim) * 0.02)  # [F,D]
        self.bias = nn.Parameter(torch.zeros(self.num_features, self.embed_dim))  # [F,D]
        self.feature_embedding = nn.Embedding(self.num_features, self.embed_dim)
        self.group_embedding = nn.Embedding(self.num_groups, self.embed_dim)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F]
        returns tokens: [B, F(+1), D]
        """
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"Expected x [B,{self.num_features}], got {tuple(x.shape)}")
        b = x.shape[0]

        # feature-specific linear expansion
        # x.unsqueeze(-1): [B,F,1]
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)  # [B,F,D]

        feat_ids = torch.arange(self.num_features, device=x.device, dtype=torch.long)
        tokens = tokens + self.feature_embedding(feat_ids).unsqueeze(0)  # [B,F,D]

        gemb = self.group_embedding(self.group_ids)  # [F,D]
        tokens = tokens + gemb.unsqueeze(0)  # [B,F,D]

        if self.use_cls_token:
            cls = self.cls_token.expand(b, -1, -1)  # [B,1,D]
            tokens = torch.cat([cls, tokens], dim=1)  # [B,F+1,D]
        return tokens


class EntropyFeatureGate(nn.Module):
    r"""
    Entropy-guided feature token gating with mutual-information prior (train-only computed).

    For each feature token e_j:
      a_j = sigmoid(W_g(e_j) + beta * m_j)
      e_hat_j = a_j * e_j

    Note: CLS token (if present) is not gated.
    """

    def __init__(
        self,
        embed_dim: int,
        num_features: int,
        mi_prior: Optional[torch.Tensor],
        gate_prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_features = int(num_features)
        self.gate = nn.Linear(self.embed_dim, 1, bias=True)

        self.beta = nn.Parameter(torch.tensor(float(gate_prior_strength)))

        if mi_prior is None:
            mi = torch.zeros(self.num_features, dtype=torch.float32)
        else:
            if mi_prior.numel() != self.num_features:
                raise ValueError("mi_prior must have shape [F].")
            mi = mi_prior.detach().float().view(-1)
        self.register_buffer("mi_prior", mi, persistent=True)

    def forward(self, tokens: torch.Tensor, has_cls: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: [B, N, D] where N = F (+1 if CLS)
        returns:
          gated_tokens: same shape as tokens
          gate_scores: [B, F] (feature tokens only, no CLS)
        """
        if has_cls:
            cls_tok = tokens[:, :1, :]
            feat_tok = tokens[:, 1:, :]
        else:
            cls_tok = None
            feat_tok = tokens

        if feat_tok.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} feature tokens, got {feat_tok.shape[1]}")

        logits = self.gate(feat_tok).squeeze(-1)  # [B,F]
        logits = logits + self.beta * self.mi_prior.unsqueeze(0)  # [B,F]
        gate_scores = torch.sigmoid(logits)
        gated = feat_tok * gate_scores.unsqueeze(-1)  # [B,F,D]

        if has_cls:
            out = torch.cat([cls_tok, gated], dim=1)
        else:
            out = gated
        return out, gate_scores


class BottleneckGatedFFN(nn.Module):
    """
    Lightweight gated bottleneck FFN:
      u = GELU(W1(x))
      g = sigmoid(W_gate(x))
      out = W2(u * g)
    """

    def __init__(self, embed_dim: int, bottleneck_ratio: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(embed_dim)
        hidden = max(8, int(d * float(bottleneck_ratio)))
        self.fc1 = nn.Linear(d, hidden)
        self.fc_gate = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.gelu(self.fc1(x))
        g = torch.sigmoid(self.fc_gate(x))
        out = self.fc2(u * g)
        out = self.drop(out)
        return out


class AttentionPooling(nn.Module):
    """
    Attention pooling over tokens when CLS token is not used.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        d = int(embed_dim)
        self.proj = nn.Linear(d, d)
        self.score = nn.Linear(d, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B,N,D] -> pooled: [B,D]
        h = torch.tanh(self.proj(tokens))
        s = self.score(h).squeeze(-1)  # [B,N]
        a = torch.softmax(s, dim=1)
        return (tokens * a.unsqueeze(-1)).sum(dim=1)


class ClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(embed_dim)
        c = int(num_classes)
        hidden = max(8, d // 2)
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, c)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

