"""Toy modules for the library scout's tests (issue #227): the op patterns real models use,
under names that say nothing about them (detection reads what they call, never the name)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Mixer(nn.Module):
    """A GQA attention block: q / k / v projections, a manual RMSNorm on q, rotate-half
    RoPE, causal SDPA with GQA and an output projection."""

    def __init__(self, hidden: int = 32, heads: int = 4, kv_heads: int = 1) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.dim = heads, kv_heads, hidden // heads
        self.a = nn.Linear(hidden, heads * self.dim, bias=False)
        self.b = nn.Linear(hidden, kv_heads * self.dim, bias=False)
        self.c = nn.Linear(hidden, kv_heads * self.dim, bias=False)
        self.d = nn.Linear(heads * self.dim, hidden, bias=False)
        self.n = Scale(self.dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.n(self.a(x).view(b, s, self.heads, self.dim)).transpose(1, 2)
        k = self.b(x).view(b, s, self.kv_heads, self.dim).transpose(1, 2)
        v = self.c(x).view(b, s, self.kv_heads, self.dim).transpose(1, 2)
        half = self.dim // 2
        turned = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
        q = q * cos + turned * sin
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.d(o.transpose(1, 2).reshape(b, s, -1))


class Scale(nn.Module):
    """RMSNorm written out (as transformers' Llama / Qwen ``*RMSNorm`` modules)."""

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class Promoted(Scale):
    """RMSNorm as MiniCPM writes it: the variance in fp32, the input itself times the fp32
    rsqrt (type promotion), the weight last."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = (x * torch.rsqrt(variance + self.eps)).to(dtype)
        return x * self.weight


class Plain(nn.Module):
    """``F.rms_norm`` directly."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, 1e-5)


class Feed(nn.Module):
    """A gated MLP (SwiGLU): ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, hidden: int = 16, inner: int = 32) -> None:
        super().__init__()
        self.p = nn.Linear(hidden, inner, bias=False)
        self.q = nn.Linear(hidden, inner, bias=False)
        self.r = nn.Linear(inner, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.r(F.silu(self.p(x)) * self.q(x))


class Head(nn.Module):
    """Logits, a softmax and a draw (sampling); ``topk`` alone would not be sampling."""

    def __init__(self, hidden: int = 16, vocab: int = 64) -> None:
        super().__init__()
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor, draw: bool = True) -> torch.Tensor:
        probs = torch.softmax(self.out(x), dim=-1)
        if draw:
            return torch.multinomial(probs, 1)
        return torch.topk(probs, 4).indices


class Residual(nn.Module):
    """A linear with a residual add and one with a tanh GELU (cuBLASLt's epilogues)."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.a = nn.Linear(hidden, hidden)
        self.b = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.a(x)
        return F.gelu(self.b(y), approximate="tanh")


class Block(nn.Module):
    """A pre-norm block: Mixer then a residual and a norm (for candidates on the CPU)."""

    def __init__(self) -> None:
        super().__init__()
        self.mix = Mixer()
        self.norm = Scale(32)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.mix(x, cos, sin))


class Core(nn.Module):
    """SDPA alone (q, k, v given): the attention kernel is all of its GPU time."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
