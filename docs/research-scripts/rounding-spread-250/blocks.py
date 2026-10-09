"""Rerounding on decoder-like blocks (SDPA, SiLU MLP, RMSNorm, in-place KV cache writes):
honest torch.compile candidates before/after, and the previous draw's output after (#250);
1 and 4 layers, decode (one token, 512 cached) and prefill (64 tokens).

    python blocks.py 200 2> blocks.txt
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

from kernel_agent.kernels import bench, compare, verify
from kernel_agent.selftest import RMSNorm

REAL = verify.rerounded_call


class Block(nn.Module):
    def __init__(self, d: int, heads: int, inner: int) -> None:
        super().__init__()
        self.h = heads
        self.n1, self.n2 = RMSNorm(d, 1e-6), RMSNorm(d, 1e-6)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, inner, bias=False)
        self.up = nn.Linear(d, inner, bias=False)
        self.down = nn.Linear(inner, d, bias=False)

    def forward(self, x, k_cache, v_cache, pos):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.h, d // self.h).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        k_cache.index_copy_(2, pos, k)
        v_cache.index_copy_(2, pos, v)
        n = int(pos[-1]) + 1
        a = F.scaled_dot_product_attention(q, k_cache[:, :, :n], v_cache[:, :, :n], is_causal=t > 1)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Stack(nn.Module):
    def __init__(self, layers: int, d=1024, heads=8, inner=2816) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, heads, inner) for _ in range(layers))

    def forward(self, x, k_caches, v_caches, pos):
        for blk, kc, vc in zip(self.blocks, k_caches, v_caches, strict=True):
            x = blk(x, kc, vc, pos)
        return x


def judge(ref, kept, plain):
    verify.rerounded_call = (lambda *a, **k: None) if plain else REAL
    try:
        return bench.check_timed_output(ref, kept)["failures"]
    finally:
        verify.rerounded_call = REAL


def main():
    draws = int(sys.argv[1])
    torch.manual_seed(0)
    for layers, t, ctx in ((1, 1, 512), (1, 64, 64), (4, 1, 512), (4, 64, 64)):
        ref = Stack(layers).cuda().to(torch.bfloat16).eval()
        with torch.no_grad():
            for name, p in ref.named_parameters():
                if "n1" in name or "n2" in name:
                    p.normal_(1.0, 0.1)
                else:
                    p.normal_(0.0, p.shape[-1] ** -0.5)
        comp = torch.compile(ref, mode="max-autotune-no-cudagraphs", dynamic=False)
        d, h = 1024, 8
        kc = [torch.randn(1, h, ctx, d // h, device="cuda", dtype=torch.bfloat16) for _ in range(layers)]
        vc = [torch.randn(1, h, ctx, d // h, device="cuda", dtype=torch.bfloat16) for _ in range(layers)]
        pos = torch.arange(ctx - t, ctx, device="cuda")
        x = torch.randn(1, t, d, device="cuda", dtype=torch.bfloat16)
        base = ((x, kc, vc, pos), {})
        before = after = 0
        cheat = {"cached": 0, "n": 0}
        spreads = []
        prev = None
        t0 = time.time()
        for i in range(draws):
            args, kwargs = bench._perturbed_copy(*base)
            pre = bench._snapshot((args, kwargs))
            with torch.inference_mode():
                out = comp(*args).clone()
            torch.cuda.synchronize()
            kept = {"iteration": i, "pre": pre, "post": (args, kwargs), "output": out}
            b = judge(ref, kept, plain=True)
            before += bool(b)
            if b:
                a = judge(ref, kept, plain=False)
                after += bool(a)
                if a:
                    print("  after:", a[0].get("name"), a[0].get("error"), file=sys.stderr)
            if i < 3:
                alt = verify.rerounded_call(ref, *pre)
                r_args = bench._snapshot(pre)
                with torch.inference_mode():
                    r = ref(*r_args[0])
                c = compare.compare_tensors("o", r, out, perturbed=True, rerounded=alt[0])
                spreads.append((c.get("rounding_spread"), c.get("rel_l2")))
            if prev is not None and i % 5 == 0:
                for name, o in (("cached", prev),):
                    k2 = {"iteration": i, "pre": pre, "post": (args, kwargs), "output": o}
                    cheat[name] += bool(judge(ref, k2, plain=False))
                cheat["n"] += 1
            prev = out
        print(
            f"layers={layers} t={t} ctx={ctx}: {draws} draws in {time.time() - t0:.0f}s "
            f"before={before} after={after} cheats={cheat} spread/rel_l2={spreads}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
