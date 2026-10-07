"""FP8 GEMM speed at VoxCPM2 shapes on the RTX 5070 Ti, by scaling recipe and backend.

Every variant computes Y[M, N] = X[M, K] @ W[N, K]^T (bf16 out, fp32 accumulation).
Timing: a CUDA graph of back-to-back calls that rotate over enough weight copies
(>= 160 MB) to keep them out of the 48 MB L2, as in the model; min over replays.
Activations are already quantised (the GEMM alone); quantisation costs are in fusion_bench.
"""

import math
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.functional import ScalingType as ST
from torch.nn.functional import SwizzleType as SW

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lt_ext import load  # noqa: E402

from kernel_agent.agent.examples.triton_fp8_w8a8_gemm import CONFIGS, DEFAULT, _gemm_kernel

E4M3 = torch.float8_e4m3fn
dev = "cuda"
ext = load()


def to_blocked(m):
    rows, cols = m.shape
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    p = torch.zeros(nrb * 128, ncb * 4, device=m.device, dtype=m.dtype)
    p[:rows, :cols] = m
    blocks = p.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@triton.jit
def _bw_gemm(a, b, sa, sb, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, GM: tl.constexpr):
    # Blockwise (DeepSeek-V3): a [M, K] e4m3 with sa [K/128, M] fp32 (one scale per 1x128
    # group, M-major); b [N, K] e4m3 with sb [N/128, K/128] fp32 (one per 128x128 block).
    # Each 128-wide K step: a fresh fp32 tl.dot, promoted into acc with both scales.
    BK: tl.constexpr = 128
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
    nkb = K // BK
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(nkb):
        part = tl.dot(tl.load(a_ptr, mask=row_ok, other=0.0), tl.load(b_ptr))
        s_a = tl.load(sa + kb * M + rm, mask=rm < M, other=0.0)
        s_b = tl.load(sb + (rn // 128) * nkb + kb)
        acc += part * s_a[:, None] * s_b[None, :]
        a_ptr += BK
        b_ptr += BK
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=row_ok)


@triton.jit
def _mx_gemm(a, b, sa, sb, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             GM: tl.constexpr):
    # MXFP8 via tl.dot_scaled: a [M, K] e4m3 + sa [M, K/32] e8m0 (uint8); b [N, K] e4m3 +
    # sb [N, K/32] e8m0.
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
    rs = tl.arange(0, BK // 32)
    rm_c = tl.minimum(rm, M - 1)
    a_ptr = a + rm_c[:, None] * K + rk[None, :]
    b_ptr = b + rn[None, :] * K + rk[:, None]
    sa_ptr = sa + rm_c[:, None] * (K // 32) + rs[None, :]
    sb_ptr = sb + rn[:, None] * (K // 32) + rs[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot_scaled(tl.load(a_ptr), tl.load(sa_ptr), "e4m3", tl.load(b_ptr),
                            tl.load(sb_ptr), "e4m3", acc)
        a_ptr += BK
        b_ptr += BK
        sa_ptr += BK // 32
        sb_ptr += BK // 32
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=rm[:, None] < M)


def graph_us(calls, reps=3, replays=7):
    """Per-call time of `calls` (a list of zero-arg closures, one per weight copy) in a
    CUDA graph that runs the list `reps` times; min over `replays` replays."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for f in calls:
                f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for f in calls:
                f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    best = math.inf
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(replays):
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / (reps * len(calls)))
    del g
    return best


def bench_shape(M, N, K, label, variants):
    copies = max(3, math.ceil(160e6 / (N * K)))
    w = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wq = [t.to(E4M3) for t in w]
    xq = x.to(E4M3)
    one = torch.ones((), device=dev)
    y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    y2 = torch.empty(2, M, N, device=dev, dtype=torch.bfloat16)
    sx_row = torch.rand(M, 1, device=dev) + 0.5
    sw_row = torch.rand(1, N, device=dev) + 0.5
    sx_mx = to_blocked(torch.full((M, K // 32), 1.0, device=dev).to(torch.float8_e8m0fnu))
    sw_mx = to_blocked(torch.full((N, K // 32), 1.0, device=dev).to(torch.float8_e8m0fnu))
    sx_mx_u8 = torch.full((M, K // 32), 127, device=dev, dtype=torch.uint8)
    sw_mx_u8 = torch.full((N, K // 32), 127, device=dev, dtype=torch.uint8)
    sa_bw = torch.rand(K // 128, M, device=dev) + 0.5
    sb_bw = torch.rand(N // 128, K // 128, device=dev) + 0.5
    flops = 2 * M * N * K
    out = {}

    def tri_rw(i):
        bm, bn, bk, warps, stages = CONFIGS.get((N, K), DEFAULT)
        grid = (triton.cdiv(M, bm) * (N // bn),)
        _gemm_kernel[grid](xq, wq[i], sx_row, sw_row, sx_row, y, M, N, K, HAS_BIAS=False,
                           BM=bm, BN=bn, BK=bk, GM=8, num_warps=warps, num_stages=stages)

    def tri_rw_cfg(i, cfg):
        bm, bn, warps, stages = cfg
        grid = (triton.cdiv(M, bm) * (N // bn),)
        _gemm_kernel[grid](xq, wq[i], sx_row, sw_row, sx_row, y, M, N, K, HAS_BIAS=False,
                           BM=bm, BN=bn, BK=128, GM=8, num_warps=warps, num_stages=stages)

    def tri_bw(i, cfg):
        bm, bn, warps, stages = cfg
        grid = (triton.cdiv(M, bm) * (N // bn),)
        _bw_gemm[grid](xq, wq[i], sa_bw, sb_bw, y, M, N, K, BM=bm, BN=bn, GM=8,
                       num_warps=warps, num_stages=stages)

    def tri_mx(i, cfg):
        bm, bn, bk, warps, stages = cfg
        grid = (triton.cdiv(M, bm) * (N // bn),)
        _mx_gemm[grid](xq, wq[i], sx_mx_u8, sw_mx_u8, y, M, N, K, BM=bm, BN=bn, BK=bk, GM=8,
                       num_warps=warps, num_stages=stages)

    for v in variants:
        try:
            if v == "bf16":
                calls = [lambda i=i: torch.mm(x, w[i].t(), out=y) for i in range(copies)]
            elif v == "torch tensorwise":
                calls = [lambda i=i: torch._scaled_mm(xq, wq[i].t(), scale_a=one, scale_b=one,
                                                      out_dtype=torch.bfloat16)
                         for i in range(copies)]
            elif v == "torch rowwise":
                calls = [lambda i=i: torch._scaled_mm(xq, wq[i].t(), scale_a=sx_row,
                                                      scale_b=sw_row, out_dtype=torch.bfloat16)
                         for i in range(copies)]
            elif v == "torch MXFP8":
                calls = [lambda i=i: F.scaled_mm(
                    xq, wq[i].t(), scale_a=sx_mx, scale_recipe_a=ST.BlockWise1x32,
                    scale_b=sw_mx, scale_recipe_b=ST.BlockWise1x32,
                    swizzle_a=SW.SWIZZLE_32_4_4, swizzle_b=SW.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16) for i in range(copies)]
            elif v == "Lt tensorwise":
                ext.tune(wq[0], xq, one, one, y, 0, 0, 1, 20, 3)
                calls = [lambda i=i: ext.run(wq[i], xq, one, one, y, 0, 0, 1)
                         for i in range(copies)]
            elif v == "Lt tensorwise split2":
                ext.tune(wq[0], xq, one, one, y2, 0, 0, 2, 20, 3)
                calls = [lambda i=i: ext.run(wq[i], xq, one, one, y2, 0, 0, 2)
                         for i in range(copies)]
            elif v == "Lt MXFP8":
                ext.tune(wq[0], xq, sw_mx, sx_mx, y, 2, 2, 1, 20, 3)
                calls = [lambda i=i: ext.run(wq[i], xq, sw_mx, sx_mx, y, 2, 2, 1)
                         for i in range(copies)]
            elif v == "Triton rowwise (example)":
                calls = [lambda i=i: tri_rw(i) for i in range(copies)]
            elif v.startswith("Triton rowwise tuned") or v.startswith("Triton blockwise"):
                fn = tri_rw_cfg if v.startswith("Triton rowwise") else tri_bw
                best = (math.inf, None)
                for cfg in [(64, 64, 4, 3), (64, 128, 4, 3), (128, 64, 8, 3), (128, 128, 8, 3),
                            (64, 64, 4, 4), (32, 64, 4, 4), (16, 64, 4, 4), (16, 128, 4, 3)]:
                    if M <= 32 and cfg[0] > 64:
                        continue
                    try:
                        t = graph_us([lambda i=i, c=cfg, fn=fn: fn(i, c) for i in range(copies)],
                                     reps=1, replays=3)
                    except Exception:  # noqa: BLE001
                        continue
                    best = min(best, (t, cfg))
                cfg = best[1]
                calls = [lambda i=i, c=cfg, fn=fn: fn(i, c) for i in range(copies)]
                v = f"{v} {cfg}"
            elif v.startswith("Triton MXFP8"):
                best = (math.inf, None)
                for cfg in [(64, 64, 128, 4, 3), (128, 64, 128, 8, 3), (64, 128, 128, 4, 3),
                            (128, 128, 128, 8, 3), (32, 64, 128, 4, 3)]:
                    try:
                        t = graph_us([lambda i=i, c=cfg: tri_mx(i, c) for i in range(copies)],
                                     reps=1, replays=3)
                    except Exception as exc:  # noqa: BLE001
                        err = exc
                        continue
                    best = min(best, (t, cfg))
                if best[1] is None:
                    raise err
                cfg = best[1]
                calls = [lambda i=i, c=cfg: tri_mx(i, c) for i in range(copies)]
                v = f"{v} {cfg}"
            else:
                raise ValueError(v)
            us = graph_us(calls)
            out[v] = us
        except Exception as exc:  # noqa: BLE001
            out[v] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
    print(f"\n### {label}: M={M} N={N} K={K} ({copies} weight copies, L2-cold)")
    base = out.get("bf16")
    for v, us in out.items():
        if isinstance(us, str):
            print(f"  {v:44s} {us}")
            continue
        tf = flops / us / 1e6
        gbs = (N * K * (2 if v == "bf16" else 1)) / us / 1e3
        sp = f"{base / us:5.2f}x" if isinstance(base, float) else ""
        print(f"  {v:44s} {us:8.2f} us  {tf:6.1f} TFLOP/s  {gbs:6.0f} GB/s(W)  {sp}")
    del w, wq
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dit352"
    ALL = ["bf16", "torch tensorwise", "torch rowwise", "torch MXFP8", "Lt tensorwise",
           "Lt MXFP8", "Triton rowwise (example)", "Triton rowwise tuned BK128", "Triton blockwise 1x128/128x128",
           "Triton MXFP8 dot_scaled"]
    SETS = {
        "dit352": [(352, 2560, 1024, "LocDiT q|k|v"), (352, 1024, 2048, "LocDiT o_proj"),
                   (352, 8192, 1024, "LocDiT gate|up"), (352, 1024, 4096, "LocDiT down")],
        "dit352n1024": [(352, 1024, 2048, "LocDiT o_proj"), (352, 1024, 4096, "LocDiT down")],
        "dit22": [(22, 2560, 1024, "LocDiT q|k|v b1"), (22, 1024, 2048, "LocDiT o_proj b1"),
                  (22, 8192, 1024, "LocDiT gate|up b1"), (22, 1024, 4096, "LocDiT down b1")],
        "enc80": [(80, 2560, 1024, "LocEnc q|k|v b16"), (80, 8192, 1024, "LocEnc gate|up b16"),
                  (80, 1024, 4096, "LocEnc down b16")],
        "lm16": [(16, 2560, 2048, "LM q|k|v b16"), (16, 2048, 2048, "LM o_proj b16"),
                 (16, 12288, 2048, "LM gate|up b16"), (16, 2048, 6144, "LM down b16")],
        "lm1": [(1, 12288, 2048, "LM gate|up b1"), (1, 2048, 6144, "LM down b1")],
        "lm272": [(272, 12288, 2048, "LM prefill gate|up b16"), (272, 2048, 6144, "LM prefill down b16")],
    }
    variants = ALL
    if which in ("dit352", "dit352n1024"):
        variants = ALL[:6] + ["Lt tensorwise split2"] + ALL[6:]
    if which == "lm1":
        variants = ["bf16", "torch tensorwise", "torch rowwise", "torch MXFP8", "Lt tensorwise", "Lt MXFP8"]
    if len(sys.argv) > 2:
        keep = sys.argv[2].split(",")
        variants = [v for v in variants if any(k in v for k in keep)]
    from kernel_agent.kernels.bench import ensure_clocks, warm_gpu

    warm_gpu()
    print("clock state", ensure_clocks())
    torch.manual_seed(0)
    for M, N, K, label in SETS[which]:
        bench_shape(M, N, K, label, variants if not (which.startswith("dit352") and N != 1024) else [v for v in variants if v != "Lt tensorwise split2"])
