"""FP8 GEMM on sm_120 (RTX 5070 Ti): Triton tl.dot (pointer / TMA descriptor), tl.dot_scaled
(MXFP8) vs cuBLASLt (torch._scaled_mm tensor-wise and row-wise) vs bf16 cuBLAS, issue #135.

Timing: R distinct weight copies (L2-cold weights, 48 MB L2) called back to back inside one
CUDA graph; the graph is replayed and the median per-call time reported. Usage:
python -I bench_fp8.py [--quick]
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

FP8 = torch.float8_e4m3fn
R = 12  # distinct weights per graph
REPLAYS = 15

SHAPES = [  # (label, M, N, K)
    ("LocDiT gate|up", 352, 8192, 1024),
    ("LocDiT qkv", 352, 2560, 1024),
    ("LocDiT o_proj", 352, 1024, 2048),
    ("LocDiT down", 352, 1024, 4096),
    ("LM decode gate|up M=16", 16, 12288, 2048),
    ("square 4096", 4096, 4096, 4096),
]

CONFIGS = [  # BM, BN, BK, warps, stages
    (64, 64, 64, 4, 3),
    (64, 64, 128, 4, 3),
    (64, 128, 64, 4, 3),
    (64, 128, 128, 4, 3),
    (128, 64, 64, 4, 4),
    (128, 64, 128, 8, 3),
    (128, 64, 128, 8, 4),
    (128, 128, 64, 8, 3),
    (128, 128, 128, 8, 3),
    (128, 128, 64, 4, 4),
    (128, 256, 64, 8, 3),
    (64, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (16, 128, 128, 4, 4),
    (16, 64, 256, 4, 3),
]


@triton.jit
def fp8_ptr(a, b, sa, sb, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
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
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=row_ok)


@triton.jit
def fp8_tma(a_desc, b_desc, sa, sb, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    group = GM * npn
    first_m = (pid // group) * GM
    group_m = min(npm - first_m, GM)
    pm = first_m + (pid % group) % group_m
    pn = (pid % group) // group_m
    off_m = pm * BM
    off_n = pn * BN
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = a_desc.load([off_m, k])
        b = b_desc.load([off_n, k])
        acc = tl.dot(a, b.T, acc)
    rm = off_m + tl.arange(0, BM)
    rn = off_n + tl.arange(0, BN)
    acc = acc * tl.load(sa + rm, mask=rm < M, other=0.0)[:, None] * tl.load(sb + rn)[None, :]
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=rm[:, None] < M)


@triton.jit
def mxfp8_ptr(a, b, sa, sb, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    # a [M, K] e4m3, sa [M, K // 32] e8m0; b [N, K] e4m3, sb [N, K // 32] e8m0
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
    KS = K // 32
    a_ptr = a + rm[:, None] * K + rk[None, :]
    b_ptr = b + rn[None, :] * K + rk[:, None]
    sa_ptr = sa + rm[:, None] * KS + rs[None, :]
    sb_ptr = sb + rn[:, None] * KS + rs[None, :]
    row_ok = rm[:, None] < M
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot_scaled(
            tl.load(a_ptr, mask=row_ok, other=0.0), tl.load(sa_ptr, mask=row_ok, other=0), "e4m3",
            tl.load(b_ptr), tl.load(sb_ptr), "e4m3", acc,
        )
        a_ptr += BK
        b_ptr += BK
        sa_ptr += BK // 32
        sb_ptr += BK // 32
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=row_ok)


def graph_time(fn_list) -> float:
    """Median us per call of the callables in fn_list, run back to back in one CUDA graph."""
    for f in fn_list:  # compile / warm
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list:
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPLAYS):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) * 1000 / len(fn_list))
    return statistics.median(times)


def ptx_features(kernel) -> dict:
    ptx = kernel.asm.get("ptx", "")
    feats = {
        "mma_e4m3": bool(re.search(r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f32\.e4m3\.e4m3\.f32", ptx)),
        "mma_block_scale": "block_scale" in ptx,
        "mma_bf16": bool(re.search(r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32", ptx)),
        "tma_load": "cp.async.bulk.tensor" in ptx,
        "cp_async": "cp.async.cg" in ptx or "cp.async.ca" in ptx,
        "wgmma": "wgmma" in ptx,
        "tcgen05": "tcgen05" in ptx,
        "ldmatrix": "ldmatrix" in ptx,
    }
    tgt = re.search(r"\.target (\S+)", ptx)
    feats["target"] = tgt[1] if tgt else "?"
    feats["shared_bytes"] = kernel.metadata.shared
    feats["regs"] = getattr(kernel, "n_regs", None)
    return feats


def main(quick: bool) -> None:
    torch.manual_seed(0)
    dev = "cuda"
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "triton", triton.__version__,
          "torch", torch.__version__, flush=True)
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=dev))
    warm = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    for _ in range(200):  # clocks up before the first measurement
        warm @ warm
    torch.cuda.synchronize()
    results = []
    shapes = SHAPES[:1] + SHAPES[-1:] if quick else SHAPES
    for label, M, N, K in shapes:
        t0 = time.time()
        flops = 2 * M * N * K
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05 for _ in range(R)]
        xs = (x.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
        xq = (x.float() / xs).to(FP8)
        wss = [(w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12) for w in ws]
        wqs = [(w.float() / s).to(FP8) for w, s in zip(ws, wss)]
        wst = [s.reshape(1, N).contiguous() for s in wss]
        one = torch.ones((), device=dev, dtype=torch.float32)
        row = {"shape": label, "M": M, "N": N, "K": K}
        outs = [torch.empty(M, N, device=dev, dtype=torch.bfloat16) for _ in range(R)]

        final = {
            "bf16_cublas": [lambda w=w, o=o: torch.mm(x, w.t(), out=o) for w, o in zip(ws, outs)],
            "scaled_mm_tensorwise": [
                lambda w=w: torch._scaled_mm(xq, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
                for w in wqs
            ],
            "scaled_mm_rowwise": [
                lambda w=w, s=s: torch._scaled_mm(xq, w.t(), scale_a=xs, scale_b=s, out_dtype=torch.bfloat16)
                for w, s in zip(wqs, wst)
            ],
        }
        ref = torch._scaled_mm(xq, wqs[0].t(), scale_a=xs, scale_b=wst[0], out_dtype=torch.bfloat16).float()
        xs_flat = xs.reshape(M).contiguous()
        wst_flat = [s.reshape(N).contiguous() for s in wst]

        for variant in ("ptr", "tma"):
            best = None
            errors = []
            for cfg in CONFIGS:
                BM, BN, BK, warps, stages = cfg
                if N % BN or K % BK or (BM > 64 and M <= 16) or (BM == 16 and M > 64):
                    continue
                grid = (triton.cdiv(M, BM) * (N // BN),)
                try:
                    if variant == "ptr":
                        def mk(w, s, o, cfg=cfg, grid=grid):
                            BM, BN, BK, warps, stages = cfg
                            return lambda: fp8_ptr[grid](xq, w, xs_flat, s, o, M, N, K, BM=BM, BN=BN, BK=BK, GM=8,
                                                        num_warps=warps, num_stages=stages)
                    else:
                        a_desc = TensorDescriptor.from_tensor(xq, [BM, BK])

                        def mk(w, s, o, cfg=cfg, grid=grid, a_desc=a_desc):
                            BM, BN, BK, warps, stages = cfg
                            b_desc = TensorDescriptor.from_tensor(w, [BN, BK])
                            return lambda: fp8_tma[grid](a_desc, b_desc, xs_flat, s, o, M, N, K, BM=BM, BN=BN, BK=BK,
                                                        GM=8, num_warps=warps, num_stages=stages)
                    fns = [mk(w, s, o) for w, s, o in zip(wqs, wst_flat, outs)]
                    kern = fns[0]()
                    torch.cuda.synchronize()
                    err = ((outs[0].float() - ref).norm() / ref.norm()).item()
                    if err > 1e-2:
                        errors.append(f"{cfg}: rel err {err:.3g}")
                        continue
                    us = graph_time(fns)
                    if best is None or us < best[0]:
                        best = (us, cfg, ptx_features(kern), err)
                        final[f"triton_{variant}"] = fns
                except Exception as e:  # out of resources, unsupported
                    errors.append(f"{cfg}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
            row[f"triton_{variant}_us"] = best[0] if best else None
            row[f"triton_{variant}_cfg"] = best[1] if best else None
            row[f"triton_{variant}_ptx"] = best[2] if best else None
            row[f"triton_{variant}_relerr"] = best[3] if best else None
            row[f"triton_{variant}_errors"] = errors[:6]
            row[f"triton_{variant}_n_errors"] = len(errors)

        # MXFP8 (tl.dot_scaled e4m3 x e4m3, ue8m0 scale per 32 along K) at two configs
        if M >= 64:
            sa = torch.full((M, K // 32), 127, device=dev, dtype=torch.uint8)  # 2^0
            sbs = [torch.full((N, K // 32), 127, device=dev, dtype=torch.uint8) for _ in range(R)]
            ref1 = torch._scaled_mm(xq, wqs[0].t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16).float()
            best = None
            errs = []
            for cfg in [(128, 64, 128, 8, 3), (64, 64, 128, 4, 3), (128, 128, 128, 8, 3), (128, 128, 64, 8, 3)]:
                BM, BN, BK, warps, stages = cfg
                if N % BN or K % BK:
                    continue
                grid = (triton.cdiv(M, BM) * (N // BN),)
                try:
                    fns = [
                        (lambda w=w, s=s, o=o, cfg=cfg, grid=grid: mxfp8_ptr[grid](
                            xq, w, sa, s, o, M, N, K, BM=cfg[0], BN=cfg[1], BK=cfg[2], GM=8,
                            num_warps=cfg[3], num_stages=cfg[4]))
                        for w, s, o in zip(wqs, sbs, outs)
                    ]
                    kern = fns[0]()
                    torch.cuda.synchronize()
                    err = ((outs[0].float() - ref1).norm() / ref1.norm()).item()
                    us = graph_time(fns)
                    if best is None or us < best[0]:
                        best = (us, cfg, ptx_features(kern), err)
                        final["triton_mxfp8"] = fns
                except Exception as e:
                    errs.append(f"{cfg}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
            row["triton_mxfp8_us"] = best[0] if best else None
            row["triton_mxfp8_cfg"] = best[1] if best else None
            row["triton_mxfp8_ptx"] = best[2] if best else None
            row["triton_mxfp8_relerr"] = best[3] if best else None
            row["triton_mxfp8_errors"] = errs[:4]

        # interleaved re-time of the winners: min over rounds of the per-round median
        rounds: dict[str, list[float]] = {k: [] for k in final}
        for _ in range(3):
            for k, fns in final.items():
                rounds[k].append(graph_time(fns))
        for k, v in rounds.items():
            row[f"{k}_us"] = min(v)
            row[f"{k}_spread"] = round((max(v) - min(v)) / min(v), 3)
        for k in list(row):
            if k.endswith("_us") and row[k]:
                row[k.replace("_us", "_tflops")] = round(flops / (row[k] * 1e-6) / 1e12, 1)
                row[k] = round(row[k], 2)
        row["wall_s"] = round(time.time() - t0, 1)
        print(json.dumps(row), flush=True)
        results.append(row)
        del ws, wqs, outs
        torch.cuda.empty_cache()
    with open(sys.argv[-1] if sys.argv[-1].endswith(".json") else "bench_fp8.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main("--quick" in sys.argv)
