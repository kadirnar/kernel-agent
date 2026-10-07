"""Decode attention (one query token, GQA 16 q heads / 2 kv heads, head dim 128) over a
bf16 vs an e4m3 KV cache (per-tensor scales), Triton, one program per (sequence, kv head).
VoxCPM2 LM shapes; L = cached tokens. CUDA graph over 36 layer-sized caches (L2-cold when
the caches exceed L2), min of replays."""

import math
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gemm_bench import graph_us  # noqa: E402

from kernel_agent.kernels.bench import ensure_clocks, warm_gpu


@triton.jit
def _decode(q, k, v, o, L, ks, vs, sm_scale, G: tl.constexpr, D: tl.constexpr,
            BL: tl.constexpr, FP8: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    H = tl.num_programs(1)
    rg = tl.arange(0, G)
    rd = tl.arange(0, D)
    qv = tl.load(q + ((b * H + h) * G + rg)[:, None] * D + rd[None, :]).to(tl.float32)
    qv = (qv * sm_scale).to(tl.bfloat16)
    m = tl.full((G,), -float("inf"), tl.float32)
    l_ = tl.zeros((G,), tl.float32)
    acc = tl.zeros((G, D), tl.float32)
    base = (b * H + h) * L * D
    for start in range(0, L, BL):
        rl = start + tl.arange(0, BL)
        ok = rl < L
        kt = tl.load(k + base + rl[:, None] * D + rd[None, :], mask=ok[:, None], other=0.0)
        vt = tl.load(v + base + rl[:, None] * D + rd[None, :], mask=ok[:, None], other=0.0)
        if FP8:
            kt = (kt.to(tl.float32) * ks).to(tl.bfloat16)
            vt = (vt.to(tl.float32) * vs).to(tl.bfloat16)
        s = tl.dot(qv, tl.trans(kt))
        s = tl.where(ok[None, :], s, -float("inf"))
        mn = tl.maximum(m, tl.max(s, 1))
        p = tl.exp(s - mn[:, None])
        alpha = tl.exp(m - mn)
        l_ = l_ * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
        m = mn
    tl.store(o + ((b * H + h) * G + rg)[:, None] * D + rd[None, :], (acc / l_[:, None]).to(tl.bfloat16))


warm_gpu()
print("clock state", ensure_clocks())
B, HKV, G, D, LAYERS = 16, 2, 8, 128, 36
for L in (77, 512, 2048, 8192):
    q = torch.randn(B, HKV, G, D, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    kb = [torch.randn(B, HKV, L, D, device="cuda", dtype=torch.bfloat16) for _ in range(LAYERS)]
    vb = [torch.randn(B, HKV, L, D, device="cuda", dtype=torch.bfloat16) for _ in range(LAYERS)]
    kq = [t.to(torch.float8_e4m3fn) for t in kb]
    vq = [t.to(torch.float8_e4m3fn) for t in vb]
    res = {}
    for fp8 in (False, True):
        k_, v_ = (kq, vq) if fp8 else (kb, vb)
        calls = [lambda i=i, k_=k_, v_=v_, fp8=fp8: _decode[(B, HKV)](
            q, k_[i], v_[i], o, L, 1.0, 1.0, D**-0.5, G=G, D=D, BL=64, FP8=fp8, num_warps=4,
            num_stages=3) for i in range(LAYERS)]
        res[fp8] = graph_us(calls)
    # check FP8 vs bf16 output on layer 0
    _decode[(B, HKV)](q, kb[0], vb[0], o, L, 1.0, 1.0, D**-0.5, G=G, D=D, BL=64, FP8=False)
    ob = o.clone()
    _decode[(B, HKV)](q, kq[0], vq[0], o, L, 1.0, 1.0, D**-0.5, G=G, D=D, BL=64, FP8=True)
    err = float((o.float() - ob.float()).norm() / ob.float().norm())
    kv_mb = 2 * B * HKV * L * D * 2 / 1e6
    print(f"L={L:5d}: KV {kv_mb:6.2f} MB/layer bf16 | bf16 {res[False]:6.2f} us "
          f"({kv_mb / res[False] * 1e3:5.0f} GB/s) | e4m3 {res[True]:6.2f} us "
          f"({kv_mb / 2 / res[True] * 1e3:5.0f} GB/s) | {res[False] / res[True]:4.2f}x | "
          f"out rel err (random K/V) {err:.3f}")
    del kb, vb, kq, vq
    torch.cuda.empty_cache()
_ = math
