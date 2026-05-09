from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # x: [B, N, D] -> [B, H, N, Dh]
    b, n, d = x.shape
    if d % num_heads != 0:
        raise ValueError(f"embed_dim {d} must be divisible by num_heads {num_heads}")
    dh = d // num_heads
    return x.view(b, n, num_heads, dh).transpose(1, 2).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # x: [B, H, N, Dh] -> [B, N, D]
    b, h, n, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * dh)


class StandardMultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention (for ablations/baselines).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_drop = nn.Dropout(self.dropout)
        self.proj_drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        b, n, d = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = _split_heads(q, self.num_heads)  # [B,H,N,Dh]
        k = _split_heads(k, self.num_heads)
        v = _split_heads(v, self.num_heads)

        dh = q.shape[-1]
        scale = 1.0 / math.sqrt(dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,N,N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [B,H,N,Dh]
        out = _merge_heads(out)  # [B,N,D]
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class CorrentropyKernelLinearAttention(nn.Module):
    r"""
    Correntropy / Gaussian-RBF inspired linear attention using positive random features.

    Target kernel (per head):
      kappa(q,k) = exp(-||q-k||^2 / (2 sigma^2))

    Positive random feature map approximation:
      phi(x) = exp((x @ omega)/sigma - ||x||^2/(2 sigma^2)) / sqrt(R)

    Linear attention (no NxN attention matrix):
      KV = phi(K)^T @ V
      normalizer = phi(Q) @ sum(phi(K))
      out = phi(Q) @ KV / (normalizer + eps)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rff_dim: int = 32,
        sigma: float = 1.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.rff_dim = int(rff_dim)
        self.sigma = float(sigma)
        self.dropout = float(dropout)
        self.eps = float(eps)

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.proj_drop = nn.Dropout(self.dropout)

        # omega: [H, Dh, R], Gaussian random matrix (buffer)
        omega = torch.randn(self.num_heads, self.head_dim, self.rff_dim)
        self.register_buffer("omega", omega, persistent=True)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, N, Dh]
        returns phi(x): [B, H, N, R]
        """
        # (x @ omega) / sigma
        # x: [B,H,N,Dh], omega: [H,Dh,R] => proj: [B,H,N,R]
        proj = torch.einsum("bhn d, hdr -> bhnr", x, self.omega) / self.sigma
        # -||x||^2 / (2 sigma^2)
        x2 = (x * x).sum(dim=-1, keepdim=True)  # [B,H,N,1]
        exp_arg = proj - x2 / (2.0 * (self.sigma ** 2))

        # numerical stability: shift by max over R, clamp exponent
        exp_arg = exp_arg - exp_arg.max(dim=-1, keepdim=True).values
        exp_arg = exp_arg.clamp(min=-30.0, max=30.0)

        phi = torch.exp(exp_arg) / math.sqrt(self.rff_dim)
        return phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        b, n, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = _split_heads(q, self.num_heads)  # [B,H,N,Dh]
        k = _split_heads(k, self.num_heads)
        v = _split_heads(v, self.num_heads)

        phi_q = self._phi(q)  # [B,H,N,R]
        phi_k = self._phi(k)  # [B,H,N,R]

        # KV = phi(K)^T @ V
        # phi_k: [B,H,N,R], v:[B,H,N,Dh] => KV:[B,H,R,Dh]
        kv = torch.einsum("bhnr, bhnd -> bhrd", phi_k, v)

        # k_sum = sum_n phi_k
        k_sum = phi_k.sum(dim=2)  # [B,H,R]

        # numerator: phi_q @ KV => [B,H,N,Dh]
        num = torch.einsum("bhnr, bhrd -> bhnd", phi_q, kv)

        # denominator: phi_q @ k_sum => [B,H,N]
        denom = torch.einsum("bhnr, bhr -> bhn", phi_q, k_sum)
        denom = denom.unsqueeze(-1)  # [B,H,N,1]

        out = num / (denom + self.eps)
        out = _merge_heads(out)  # [B,N,D]
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

