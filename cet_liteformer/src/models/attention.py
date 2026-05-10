from __future__ import annotations

import argparse
import math
from typing import Optional, Tuple, Union

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

    attn = softmax((Q K^T) / sqrt(Dh), dim=-1)
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


class CorrentropyRBFAttention(nn.Module):
    """
    Exact Gaussian RBF / correntropy self-attention.

    Correntropy (Gaussian) kernel between query q_i and key k_j:

        kappa(q_i, k_j) = exp(-||q_i - k_j||^2 / (2 sigma^2))

    Attention weights (normalized over keys j):

        A_ij = softmax_j( -||q_i - k_j||^2 / (2 sigma^2) )

    Output:

        out_i = sum_j A_ij v_j

    This exact RBF/correntropy attention is mathematically faithful to the Gaussian
    correntropy kernel. It is used as the default for correctness. Linear approximation
    can be added as a separate experimental ablation (see ExperimentalCorrentropyLinearAttention).

    Complexity is O(N^2 d) in the number of tokens N (here: flow-feature tokens, typically
    tens to low hundreds), not packet-sequence length. The kernel is local and bounded, which
    can help with noisy, heavy-tailed, nonlinear flow statistics common in darknet traffic.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sigma: float = 1.0,
        dropout: float = 0.1,
        learnable_sigma: bool = False,
        eps: float = 1e-8,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.eps = float(eps)
        self.return_attn = bool(return_attn)
        self.learnable_sigma = bool(learnable_sigma)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_drop = nn.Dropout(float(dropout))
        self.out_drop = nn.Dropout(float(dropout))

        if self.learnable_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma), dtype=torch.float32)))
        else:
            self.register_buffer("sigma_buffer", torch.tensor(float(sigma), dtype=torch.float32))

    def get_sigma(self) -> torch.Tensor:
        if self.learnable_sigma:
            return torch.exp(self.log_sigma).clamp(min=1e-3, max=100.0)
        return self.sigma_buffer.clamp(min=1e-3, max=100.0)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, N, D]
        B, N, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        q_norm = (q**2).sum(dim=-1, keepdim=True)  # [B,H,N,1]
        k_norm = (k**2).sum(dim=-1).unsqueeze(-2)  # [B,H,1,N]
        dist2 = q_norm + k_norm - 2.0 * torch.matmul(q, k.transpose(-2, -1))
        dist2 = torch.clamp(dist2, min=0.0)

        sigma = self.get_sigma()
        denom = 2.0 * sigma * sigma + self.eps
        logits = -dist2 / denom
        logits = logits - logits.max(dim=-1, keepdim=True).values

        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.out_drop(out)

        want_attn = self.return_attn if return_attn is None else return_attn
        if want_attn:
            return out, attn
        return out


class ExperimentalCorrentropyLinearAttention(nn.Module):
    """
    Experimental approximate kernel attention via positive random features (RFF).

    This is NOT the default correntropy implementation and does NOT reproduce the exact
    Gaussian RBF kernel kappa(q,k)=exp(-||q-k||^2/(2 sigma^2)) in finite R. Use
    CorrentropyRBFAttention for mathematically exact Gaussian/RBF correntropy attention.

    Kept for ablation / backward compatibility only.
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

        omega = torch.randn(self.num_heads, self.head_dim, self.rff_dim)
        self.register_buffer("omega", omega, persistent=True)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, N, Dh]
        returns phi(x): [B, H, N, R]
        """
        proj = torch.einsum("bhnd,hdr->bhnr", x, self.omega) / self.sigma
        x2 = (x * x).sum(dim=-1, keepdim=True)  # [B,H,N,1]
        exp_arg = proj - x2 / (2.0 * (self.sigma**2))

        exp_arg = exp_arg - exp_arg.max(dim=-1, keepdim=True).values
        exp_arg = exp_arg.clamp(min=-30.0, max=30.0)

        phi = torch.exp(exp_arg) / math.sqrt(self.rff_dim)
        return phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = _split_heads(q, self.num_heads)
        k = _split_heads(k, self.num_heads)
        v = _split_heads(v, self.num_heads)

        phi_q = self._phi(q)
        phi_k = self._phi(k)

        kv = torch.einsum("bhnr,bhnd->bhrd", phi_k, v)
        k_sum = phi_k.sum(dim=2)

        num = torch.einsum("bhnr,bhrd->bhnd", phi_q, kv)
        denom = torch.einsum("bhnr,bhr->bhn", phi_q, k_sum).unsqueeze(-1)

        out = num / (denom + self.eps)
        out = _merge_heads(out)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# Backward compatibility for imports / old checkpoints referencing the old name.
CorrentropyKernelLinearAttention = ExperimentalCorrentropyLinearAttention


def _sanity_check() -> None:
    torch.manual_seed(0)

    # 1) Shape
    x = torch.randn(4, 81, 64)
    attn = CorrentropyRBFAttention(embed_dim=64, num_heads=4, dropout=0.0)
    y = attn(x)
    assert y.shape == x.shape, y.shape

    # 2) Finite
    assert torch.isfinite(y).all()

    # 3) Backprop
    loss = y.mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn.parameters() if p.requires_grad)

    attn.zero_grad(set_to_none=True)

    # 4) Attention normalization + map shape
    attn2 = CorrentropyRBFAttention(embed_dim=64, num_heads=4, dropout=0.0, return_attn=True)
    y2, amap = attn2(x, return_attn=True)
    assert amap.shape == (4, 4, 81, 81)
    sums = amap.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-4)

    # 5) Learnable sigma positive
    attn_ls = CorrentropyRBFAttention(embed_dim=64, num_heads=4, sigma=0.5, learnable_sigma=True, dropout=0.0)
    s0 = float(attn_ls.get_sigma().detach())
    assert s0 >= 1e-3
    (attn_ls(x).mean()).backward()
    assert float(attn_ls.get_sigma().detach()) >= 1e-3

    # 6) Manual agreement (identity projections, single head)
    Dh, Ntok = 4, 3
    mod = CorrentropyRBFAttention(embed_dim=Dh, num_heads=1, sigma=1.25, dropout=0.0, return_attn=True)
    with torch.no_grad():
        nn.init.eye_(mod.q_proj.weight)
        nn.init.zeros_(mod.q_proj.bias)
        nn.init.eye_(mod.k_proj.weight)
        nn.init.zeros_(mod.k_proj.bias)
        nn.init.eye_(mod.v_proj.weight)
        nn.init.zeros_(mod.v_proj.bias)
        nn.init.eye_(mod.out_proj.weight)
        nn.init.zeros_(mod.out_proj.bias)

    x3 = torch.randn(1, Ntok, Dh)
    _, am = mod(x3, return_attn=True)
    # Identity projections => q,k equal x3 reshaped as [B,H,N,Dh]
    q = x3.unsqueeze(1)
    k = x3.unsqueeze(1)
    qn = (q**2).sum(dim=-1, keepdim=True)
    kn = (k**2).sum(dim=-1).unsqueeze(-2)
    d2 = torch.clamp(qn + kn - 2.0 * torch.matmul(q, k.transpose(-2, -1)), min=0.0)
    sig = mod.get_sigma()
    logits = -d2 / (2.0 * sig * sig + mod.eps)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    am_manual = torch.softmax(logits, dim=-1)
    assert am_manual.shape == am.shape == (1, 1, Ntok, Ntok)
    assert torch.allclose(am, am_manual, atol=1e-5, rtol=1e-4)

    # Experimental module still runs
    ex = ExperimentalCorrentropyLinearAttention(embed_dim=32, num_heads=4, rff_dim=16)
    xe = torch.randn(2, 10, 32)
    ye = ex(xe)
    assert ye.shape == xe.shape and torch.isfinite(ye).all()

    print("attention.py sanity_check: OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity_check", action="store_true")
    args = ap.parse_args()
    if args.sanity_check:
        _sanity_check()


if __name__ == "__main__":
    main()
