"""FP8 activations through fused ops (VoxCPM2 LocDiT, M = 352):
(1) CUTLASS SM120 dense FP8 GEMM (an epilogue-capable library kernel) vs cuBLASLt nvjet;
(2) quantising an activation in a separate pass vs fused into its producer (RMSNorm,
    silu(gate) * up) with per-token, static per-tensor and MXFP8 (1x32) scales;
(3) the gate|up -> silu*up -> e4m3 chain: cuBLASLt + a fused glue kernel vs one Triton GEMM
    with silu*up + 1x128-group quantisation in its epilogue.
Every time: CUDA graph, min of replays; GEMM weights rotated past L2."""

import math
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.functional import ScalingType as ST
from torch.nn.functional import SwizzleType as SW

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cutlass_bw import load as load_cutlass  # noqa: E402
from gemm_bench import graph_us, to_blocked  # noqa: E402
from lt_ext import load  # noqa: E402

from kernel_agent.kernels.bench import ensure_clocks, warm_gpu

E4M3 = torch.float8_e4m3fn
ext = load()
cut = load_cutlass()
warm_gpu()
print("clock state", ensure_clocks())


@triton.jit
def _norm_k(x, w, out, q, s, K, eps, inv_static, MODE: tl.constexpr, BLOCK: tl.constexpr):
    # MODE 0: bf16 out; 1: e4m3 + per-token scale; 2: e4m3 with a static scale;
    # 3: MXFP8 (e4m3 + e8m0 per 32, ceil rule)
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    v = tl.load(x + row * K + cols).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 0) / K + eps)
    h = (v * r).to(tl.bfloat16).to(tl.float32) * tl.load(w + cols).to(tl.float32)
    if MODE == 0:
        tl.store(out + row * K + cols, h.to(tl.bfloat16))
    elif MODE == 1:
        sc = tl.maximum(tl.max(tl.abs(h), 0) / 448.0, 1e-12)
        tl.store(q + row * K + cols, tl.clamp(h / sc, -448.0, 448.0).to(tl.float8e4nv))
        tl.store(s + row, sc)
    elif MODE == 2:
        tl.store(q + row * K + cols, tl.clamp(h * inv_static, -448.0, 448.0).to(tl.float8e4nv))
    else:
        g = tl.reshape(h, (BLOCK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(g), 1), 1e-30)
        e = tl.math.ceil(tl.math.log2(amax / 448.0))
        sc = tl.math.exp2(e)
        codes = tl.clamp(g / sc[:, None], -448.0, 448.0).to(tl.float8e4nv)
        tl.store(q + row * K + tl.reshape(cols, (BLOCK // 32, 32)), codes)
        tl.store(s.to(tl.pointer_type(tl.uint8)) + row * (K // 32) + tl.arange(0, BLOCK // 32),
                 (e + 127).to(tl.uint8))


@triton.jit
def _quant_tok(x, q, s, K, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    h = tl.load(x + row * K + cols).to(tl.float32)
    sc = tl.maximum(tl.max(tl.abs(h), 0) / 448.0, 1e-12)
    tl.store(q + row * K + cols, tl.clamp(h / sc, -448.0, 448.0).to(tl.float8e4nv))
    tl.store(s + row, sc)


@triton.jit
def _silu_mul_k(gu, sx, sw, out, q, s, F_, inv_static, SCALED: tl.constexpr,
                MODE: tl.constexpr, BLOCK: tl.constexpr):
    # gu [M, 2F] (gate | up) -> h = silu(gate) * up; SCALED: gu is an unscaled FP8 GEMM
    # output (tensorwise kernel): multiply by sx[row] * sw[col] first. MODE as _norm_k.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    g = tl.load(gu + row * 2 * F_ + cols).to(tl.float32)
    u = tl.load(gu + row * 2 * F_ + F_ + cols).to(tl.float32)
    if SCALED:
        a = tl.load(sx + row)
        g = g * a * tl.load(sw + cols)
        u = u * a * tl.load(sw + F_ + cols)
    h = g / (1.0 + tl.exp(-g)) * u
    if MODE == 0:
        tl.store(out + row * F_ + cols, h.to(tl.bfloat16))
    elif MODE == 1:
        sc = tl.maximum(tl.max(tl.abs(h), 0) / 448.0, 1e-12)
        tl.store(q + row * F_ + cols, tl.clamp(h / sc, -448.0, 448.0).to(tl.float8e4nv))
        tl.store(s + row, sc)
    elif MODE == 2:
        tl.store(q + row * F_ + cols, tl.clamp(h * inv_static, -448.0, 448.0).to(tl.float8e4nv))
    else:
        g2 = tl.reshape(h, (BLOCK // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(g2), 1), 1e-30)
        e = tl.math.ceil(tl.math.log2(amax / 448.0))
        codes = tl.clamp(g2 / tl.math.exp2(e)[:, None], -448.0, 448.0).to(tl.float8e4nv)
        tl.store(q + row * F_ + tl.reshape(cols, (BLOCK // 32, 32)), codes)
        tl.store(s.to(tl.pointer_type(tl.uint8)) + row * (F_ // 32) + tl.arange(0, BLOCK // 32),
                 (e + 127).to(tl.uint8))


@triton.jit
def _gu_fused(a, b, sa, sb, out, q, qs, M, F_, K, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, GM: tl.constexpr, QUANT: tl.constexpr):
    # gate|up GEMM on a column-interleaved weight (rows g0, u0, g1, u1, ...: [2F, K] e4m3,
    # per-row scales sb), per-token activation scales sa. Epilogue: h = silu(g) * u for the
    # tile's BN / 2 columns; QUANT: e4m3 codes + one scale per (row, BN/2-column group)
    # (qs [F / (BN/2), M], M-major: the 1x128 activation layout of a blockwise GEMM);
    # else bf16 h.
    N = 2 * F_
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    group = GM * npn
    first_m = (pid // group) * GM
    group_m = min(npm - first_m, GM)
    pm = first_m + (pid % group) % group_m
    pn = (pid % group) // group_m
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = a + rm[:, None] * K + rk[None, :]
    b_ptr = b + rn[None, :] * K + rk[:, None]
    row_ok = rm[:, None] < M
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a_ptr, mask=row_ok, other=0.0), tl.load(b_ptr), acc)
        a_ptr += BK
        b_ptr += BK
    acc = acc * tl.load(sa + rm, mask=rm < M, other=0.0)[:, None] * tl.load(sb + rn)[None, :]
    g, u = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
    h = g / (1.0 + tl.exp(-g)) * u
    hc = pn * (BN // 2) + tl.arange(0, BN // 2)
    if QUANT:
        sc = tl.maximum(tl.max(tl.abs(h), 1) / 448.0, 1e-12)
        codes = tl.clamp(h / sc[:, None], -448.0, 448.0).to(tl.float8e4nv)
        tl.store(q + rm[:, None] * F_ + hc[None, :], codes, mask=row_ok)
        tl.store(qs + pn * M + rm, sc, mask=rm < M)
    else:
        tl.store(out + rm[:, None] * F_ + hc[None, :], h.to(tl.bfloat16), mask=row_ok)


M = 352
# ---- (1) CUTLASS dense FP8 vs cuBLASLt nvjet ----
print("\n## (1) CUTLASS SM120 dense FP8 (TMA warp-specialised, scalar scale) vs cuBLASLt")
one = torch.ones((), device="cuda")
for N, K, label in ((2560, 1024, "q|k|v"), (1024, 2048, "o_proj"), (8192, 1024, "gate|up"),
                    (1024, 4096, "down"), (8192, 8192, "8192^2 x K=8192, M=8192")):
    m = 8192 if N == 8192 and K == 8192 else M
    copies = max(3, math.ceil(160e6 / (N * K)))
    x = torch.randn(m, K, device="cuda").to(E4M3)
    ws = [(torch.randn(N, K, device="cuda") * 0.05).to(E4M3) for _ in range(copies)]
    y = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ ws[0].float().T)
    ext.tune(ws[0], x, one, one, y, 0, 0, 1, 10, 3)
    row = [f"  {label:24s}"]
    for name, fn in (("cuBLASLt", lambda i: ext.run(ws[i], x, one, one, y, 0, 0, 1)),
                     ("CUTLASS coop 128x128", lambda i: cut.dense_coop128(x, ws[i], y)),
                     ("CUTLASS pingpong 64x128", lambda i: cut.dense_ping64(x, ws[i], y)),
                     ("CUTLASS pingpong 64x64", lambda i: cut.dense_ping64n64(x, ws[i], y))):
        try:
            fn(0)
            torch.cuda.synchronize()
            err = float((y.float() - ref).norm() / ref.norm())
            us = graph_us([lambda i=i, fn=fn: fn(i) for i in range(copies)])
            row.append(f"{name} {us:6.2f} us ({2 * m * N * K / us / 1e6:5.1f} TF, err {err:.0e})")
        except Exception as exc:  # noqa: BLE001
            row.append(f"{name} {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
    print(" | ".join(row))
    del ws
    torch.cuda.empty_cache()

# ---- (2) producer-side quantisation ----
print("\n## (2) quantising activations: separate pass vs fused into the producer (M = 352)")
H, F_ = 1024, 4096
x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
wn = torch.ones(H, device="cuda", dtype=torch.bfloat16)
hb = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
hq = torch.empty(M, H, device="cuda", dtype=E4M3)
hs = torch.empty(M, device="cuda")
hs_mx = torch.empty(M, H // 32, device="cuda", dtype=torch.uint8)
gu = torch.randn(M, 2 * F_, device="cuda", dtype=torch.bfloat16)
sx, sw = torch.rand(M, device="cuda") + 0.5, torch.rand(2 * F_, device="cuda") + 0.5
mb = torch.empty(M, F_, device="cuda", dtype=torch.bfloat16)
mq = torch.empty(M, F_, device="cuda", dtype=E4M3)
ms = torch.empty(M, device="cuda")
ms_mx = torch.empty(M, F_ // 32, device="cuda", dtype=torch.uint8)
nw = dict(num_warps=4)
cases = {
    "RMSNorm [352,1024] -> bf16 (no quant)": [lambda: _norm_k[(M,)](x, wn, hb, hq, hs, H, 1e-5, 1.0, MODE=0, BLOCK=H, **nw)],
    "RMSNorm -> bf16, then per-token quant pass": [
        lambda: _norm_k[(M,)](x, wn, hb, hq, hs, H, 1e-5, 1.0, MODE=0, BLOCK=H, **nw),
        lambda: _quant_tok[(M,)](hb, hq, hs, H, BLOCK=H, **nw)],
    "RMSNorm + per-token e4m3 (fused)": [lambda: _norm_k[(M,)](x, wn, hb, hq, hs, H, 1e-5, 1.0, MODE=1, BLOCK=H, **nw)],
    "RMSNorm + static-scale e4m3 (fused)": [lambda: _norm_k[(M,)](x, wn, hb, hq, hs, H, 1e-5, 0.5, MODE=2, BLOCK=H, **nw)],
    "RMSNorm + MXFP8 1x32 (fused)": [lambda: _norm_k[(M,)](x, wn, hb, hq, hs_mx, H, 1e-5, 1.0, MODE=3, BLOCK=H, **nw)],
    "silu*up [352,8192]->[352,4096] bf16 (no quant)": [lambda: _silu_mul_k[(M,)](gu, sx, sw, mb, mq, ms, F_, 1.0, SCALED=False, MODE=0, BLOCK=F_, num_warps=8)],
    "silu*up -> bf16, then per-token quant pass": [
        lambda: _silu_mul_k[(M,)](gu, sx, sw, mb, mq, ms, F_, 1.0, SCALED=False, MODE=0, BLOCK=F_, num_warps=8),
        lambda: _quant_tok[(M,)](mb, mq, ms, F_, BLOCK=F_, num_warps=8)],
    "silu*up + per-token e4m3 (fused)": [lambda: _silu_mul_k[(M,)](gu, sx, sw, mb, mq, ms, F_, 1.0, SCALED=False, MODE=1, BLOCK=F_, num_warps=8)],
    "scale(sx*sw) + silu*up + per-token e4m3 (fused)": [lambda: _silu_mul_k[(M,)](gu, sx, sw, mb, mq, ms, F_, 1.0, SCALED=True, MODE=1, BLOCK=F_, num_warps=8)],
    "silu*up + static-scale e4m3 (fused)": [lambda: _silu_mul_k[(M,)](gu, sx, sw, mb, mq, ms, F_, 0.5, SCALED=False, MODE=2, BLOCK=F_, num_warps=8)],
    "silu*up + MXFP8 1x32 (fused)": [lambda: _silu_mul_k[(M,)](gu, sx, sw, mb, mq, ms_mx, F_, 1.0, SCALED=False, MODE=3, BLOCK=F_, num_warps=8)],
}
for name, calls in cases.items():
    us = graph_us([lambda calls=calls: [c() for c in calls]], reps=50)
    print(f"  {name:52s} {us:6.2f} us")

# ---- (3) gate|up -> silu*up -> e4m3: library GEMM + glue vs fused epilogue ----
print("\n## (3) gate|up GEMM + silu*up + e4m3 for down_proj (M = 352, [1024 -> 2 x 4096])")
K, N2 = 1024, 2 * F_
copies = 20
xq = torch.randn(M, K, device="cuda").to(E4M3)
xs = torch.rand(M, device="cuda") + 0.5
ws = [(torch.randn(N2, K, device="cuda") * 0.05).to(E4M3) for _ in range(copies)]
wsc = torch.rand(N2, device="cuda") + 0.5
y = torch.empty(M, N2, device="cuda", dtype=torch.bfloat16)
qs128 = torch.empty(F_ // 128, M, device="cuda")
ext.tune(ws[0], xq, one, one, y, 0, 0, 1, 10, 3)
sxm = to_blocked(torch.ones(M, K // 32, device="cuda").to(torch.float8_e8m0fnu))
swm = to_blocked(torch.ones(N2, K // 32, device="cuda").to(torch.float8_e8m0fnu))


def gu_fused(i, quant, bm=64, bn=256, bk=128, warps=8, stages=3):
    grid = (triton.cdiv(M, bm) * (N2 // bn),)
    _gu_fused[grid](xq, ws[i], xs, wsc, mb, mq, qs128, M, F_, K, BM=bm, BN=bn, BK=bk, GM=8,
                    QUANT=quant, num_warps=warps, num_stages=stages)


chains = {
    "cuBLASLt tensorwise GEMM (bf16 out) + scale/silu*up/per-token quant kernel": lambda i: (
        ext.run(ws[i], xq, one, one, y, 0, 0, 1),
        _silu_mul_k[(M,)](y, xs, wsc, mb, mq, ms, F_, 1.0, SCALED=True, MODE=1, BLOCK=F_, num_warps=8)),
    "cuBLASLt tensorwise GEMM alone": lambda i: ext.run(ws[i], xq, one, one, y, 0, 0, 1),
    "cuBLASLt MXFP8 GEMM (bf16 out) + silu*up/MXFP8 quant kernel": lambda i: (
        F.scaled_mm(xq, ws[i].t(), scale_a=sxm, scale_recipe_a=ST.BlockWise1x32, scale_b=swm,
                    scale_recipe_b=ST.BlockWise1x32, swizzle_a=SW.SWIZZLE_32_4_4,
                    swizzle_b=SW.SWIZZLE_32_4_4, output_dtype=torch.bfloat16, ),
        _silu_mul_k[(M,)](y, xs, wsc, mb, mq, ms_mx, F_, 1.0, SCALED=False, MODE=3, BLOCK=F_, num_warps=8)),
    "Triton GEMM + silu*up epilogue (bf16 h out)": lambda i: gu_fused(i, False),
    "Triton GEMM + silu*up + 1x128 e4m3 epilogue (one kernel)": lambda i: gu_fused(i, True),
}
for name, fn in chains.items():
    us = graph_us([lambda i=i, fn=fn: fn(i) for i in range(copies)])
    print(f"  {name:78s} {us:6.2f} us")
best = None
for cfg in [(64, 256, 128, 8, 3), (64, 256, 64, 8, 4), (128, 256, 64, 8, 3), (32, 256, 128, 4, 3),
            (64, 256, 128, 4, 3), (128, 256, 128, 8, 2)]:
    try:
        us = graph_us([lambda i=i, c=cfg: gu_fused(i, True, *c) for i in range(copies)], reps=1)
        best = min(best or (math.inf, None), (us, cfg))
    except Exception:  # noqa: BLE001
        pass
print(f"  {'  best fused config ' + str(best[1]):78s} {best[0]:6.2f} us")
