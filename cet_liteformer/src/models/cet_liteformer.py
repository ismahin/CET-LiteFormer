from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    CorrentropyRBFAttention,
    ExperimentalCorrentropyLinearAttention,
    StandardMultiHeadSelfAttention,
)


def resolve_attention_type(model_cfg: Dict[str, Any]) -> str:
    """
    Prefer explicit model.attention_type; otherwise derive from use_correntropy_attention.
    Valid: correntropy_rbf (default when correntropy on), standard, correntropy_linear_experimental.
    """
    at = model_cfg.get("attention_type")
    if isinstance(at, str) and at.strip():
        return at.strip().lower()
    return "correntropy_rbf" if bool(model_cfg.get("use_correntropy_attention", True)) else "standard"
from .layers import (
    AttentionPooling,
    BottleneckGatedFFN,
    ClassifierHead,
    EntropyFeatureGate,
    FeatureTokenizer,
)


class CETLiteFormerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rff_dim: int,
        sigma: float,
        dropout: float,
        attention_type: str,
        learnable_sigma: bool,
        ffn_bottleneck_ratio: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        at = str(attention_type).strip().lower()
        if at == "correntropy_linear_experimental":
            self.attn = ExperimentalCorrentropyLinearAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rff_dim=rff_dim,
                sigma=sigma,
                dropout=dropout,
            )
        elif at == "standard":
            self.attn = StandardMultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        else:
            # Default: exact Gaussian RBF / correntropy (O(N^2) in tokens)
            self.attn = CorrentropyRBFAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sigma=sigma,
                dropout=dropout,
                learnable_sigma=learnable_sigma,
            )

        self.ffn = BottleneckGatedFFN(embed_dim=embed_dim, bottleneck_ratio=ffn_bottleneck_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class CETLiteFormer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        group_ids: List[int],
        mi_prior: Optional[torch.Tensor],
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        rff_dim: int = 32,
        sigma: float = 1.0,
        dropout: float = 0.15,
        use_cls_token: bool = True,
        use_entropy_gate: bool = True,
        use_correntropy_attention: bool = True,
        attention_type: Optional[str] = None,
        learnable_sigma: bool = False,
        use_early_exit: bool = True,
        early_exit_threshold: float = 0.90,
        ffn_bottleneck_ratio: float = 0.5,
        gate_prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.rff_dim = int(rff_dim)
        self.sigma = float(sigma)
        self.dropout = float(dropout)

        self.use_cls_token = bool(use_cls_token)
        self.use_entropy_gate = bool(use_entropy_gate)
        self.use_correntropy_attention = bool(use_correntropy_attention)
        self.attention_type = resolve_attention_type(
            {"attention_type": attention_type, "use_correntropy_attention": self.use_correntropy_attention}
        )
        self.learnable_sigma = bool(learnable_sigma)
        self.use_early_exit = bool(use_early_exit)
        self.early_exit_threshold = float(early_exit_threshold)

        self.tokenizer = FeatureTokenizer(
            num_features=self.num_features,
            embed_dim=self.embed_dim,
            group_ids=group_ids,
            use_cls_token=self.use_cls_token,
        )

        self.gate_layer: Optional[EntropyFeatureGate]
        if self.use_entropy_gate:
            self.gate_layer = EntropyFeatureGate(
                embed_dim=self.embed_dim,
                num_features=self.num_features,
                mi_prior=mi_prior,
                gate_prior_strength=gate_prior_strength,
            )
        else:
            self.gate_layer = None

        self.blocks = nn.ModuleList(
            [
                CETLiteFormerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    rff_dim=self.rff_dim,
                    sigma=self.sigma,
                    dropout=self.dropout,
                    attention_type=self.attention_type,
                    learnable_sigma=self.learnable_sigma,
                    ffn_bottleneck_ratio=ffn_bottleneck_ratio,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.pool: Optional[AttentionPooling] = None
        if not self.use_cls_token:
            self.pool = AttentionPooling(self.embed_dim)

        # early-exit heads after each block
        self.exit_heads = nn.ModuleList([ClassifierHead(self.embed_dim, self.num_classes, dropout=self.dropout) for _ in range(self.num_layers)])
        self.final_head = ClassifierHead(self.embed_dim, self.num_classes, dropout=self.dropout)

    @staticmethod
    def _confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B,C] -> confidence: [B]
        p = torch.softmax(logits, dim=-1)
        eps = 1e-12
        ent = -(p * (p + eps).log()).sum(dim=-1)  # [B]
        c = p.shape[-1]
        conf = 1.0 - ent / math.log(max(c, 2))
        return conf

    def _represent(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            return tokens[:, 0, :]
        assert self.pool is not None
        return self.pool(tokens)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        x: [B, F]
        Returns training dict:
          {
            "logits": [B,C],
            "exit_logits": List[[B,C]],
            "gate_scores": [B,F] or None,
            "exit_used": None
          }
        In eval() mode with early exit:
          exit_used is int in [0..L-1] or L for final.
        """
        tokens = self.tokenizer(x)  # [B,N,D]
        has_cls = self.use_cls_token

        gate_scores = None
        if self.gate_layer is not None:
            tokens, gate_scores = self.gate_layer(tokens, has_cls=has_cls)

        exit_logits: List[torch.Tensor] = []
        exit_used: Optional[int] = None

        # inference early exit: only when model is in eval mode and enabled
        do_early_exit = (self.use_early_exit and (not self.training))

        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            rep = self._represent(tokens)
            logits_i = self.exit_heads[i](rep)
            exit_logits.append(logits_i)

            if do_early_exit:
                conf = self._confidence_from_logits(logits_i)
                # if all samples confident, exit; otherwise keep going (batch-safe)
                if bool((conf >= self.early_exit_threshold).all()):
                    exit_used = i
                    return {
                        "logits": logits_i,
                        "exit_logits": exit_logits,
                        "gate_scores": gate_scores,
                        "exit_used": exit_used,
                    }

        final_rep = self._represent(tokens)
        final_logits = self.final_head(final_rep)
        if do_early_exit:
            exit_used = self.num_layers

        return {
            "logits": final_logits,
            "exit_logits": exit_logits,
            "gate_scores": gate_scores,
            "exit_used": exit_used,
        }


def _sanity_check() -> None:
    torch.manual_seed(0)
    b, f, c = 8, 32, 5
    group_ids = [0] * f
    mi = torch.rand(f)
    model = CETLiteFormer(
        num_features=f,
        num_classes=c,
        group_ids=group_ids,
        mi_prior=mi,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        rff_dim=32,
        sigma=1.0,
        dropout=0.1,
        use_cls_token=True,
        use_entropy_gate=True,
        use_correntropy_attention=True,
        attention_type=None,
        learnable_sigma=False,
        use_early_exit=True,
        early_exit_threshold=0.9,
        ffn_bottleneck_ratio=0.5,
        gate_prior_strength=1.0,
    )

    x = torch.randn(b, f)
    y = torch.randint(0, c, (b,))

    model.train()
    out = model(x)
    assert out["logits"].shape == (b, c)
    assert isinstance(out["exit_logits"], list) and len(out["exit_logits"]) == model.num_layers
    assert out["gate_scores"] is not None and out["gate_scores"].shape == (b, f)

    # quick loss check (CE) over final logits + exit logits + gate sparsity
    ce = F.cross_entropy(out["logits"], y)
    exit_ce = sum(F.cross_entropy(z, y) for z in out["exit_logits"]) / len(out["exit_logits"])
    gate_l1 = out["gate_scores"].abs().mean()
    loss = ce + 0.3 * exit_ce + 1e-4 * gate_l1
    loss.backward()

    # one optimizer step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    print("Sanity check passed.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity_check", action="store_true", help="Run unit-test-like forward/loss/one-step checks.")
    args = ap.parse_args()
    if args.sanity_check:
        _sanity_check()


if __name__ == "__main__":
    main()

