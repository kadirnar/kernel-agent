"""sm_120 follow-up (#135): tl.dot_scaled (MXFP8, native block-scaled QMMA.SF) with TMA
descriptors and warp specialisation vs plain e4m3 tl.dot and cuBLASLt tensor-wise.
python -I bench_mx.py out.json
"""

from __future__ import annotations

import json
import sys

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_fp8 import FP8, R, graph_time, ptx_features  # noqa: E402


@triton.jit
def mx_tma(a_desc, b_desc, sa, sb, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GM: tl.constexpr, SCALED: tl.constexpr, WS: tl.constexpr, STAGES: tl.constexpr):
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
    rm = off_m + tl.arange(0, BM)
    rn = off_n + tl.arange(0, BN)
    rs = tl.arange(0, BK // 32)
    KS = K // 32
    acc = tl.zeros((BM, BN), tl.float32)
    for k in tl.range(0, K, BK, num_stages=STAGES, warp_specialize=WS):
        a = a_desc.load([off_m, k])
        b = b_desc.load([off_n, k])
        if SCALED:
            s_a = tl.load(sa + rm[:, None] * KS + (k // 32 + rs)[None, :], mask=rm[:, None] < M, other=127)
            s_b = tl.load(sb + rn[:, None] * KS + (k // 32 + rs)[None, :])
            acc = tl.dot_scaled(a, s_a, "e4m3", b.T, s_b, "e4m3", acc)
        else:
            acc = tl.dot(a, b.T, acc)
    tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=rm[:, None] < M)


SHAPES = [("LocDiT gate|up", 352, 8192, 1024), ("LocDiT down", 352, 1024, 4096), ("square 4096", 4096, 4096, 4096)]
CONFIGS = [(64, 64, 128, 4, 3), (128, 64, 128, 4, 3), (128, 64, 128, 8, 3), (128, 128, 64, 8, 4), (128, 128, 128, 8, 2),
           (128, 128, 128, 8, 3), (128, 128, 128, 4, 3), (128, 256, 64, 8, 3), (256, 128, 64, 8, 3)]


def main() -> None:
    dev = "cuda"
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=dev))
    torch.manual_seed(0)
    warm = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    for _ in range(200):  # clocks up
        warm @ warm
    torch.cuda.synchronize()
    rows = []
    for label, M, N, K in SHAPES:
        flops = 2 * M * N * K
        xq = torch.randn(M, K, device=dev).to(FP8)
        wqs = [torch.randn(N, K, device=dev).to(FP8) for _ in range(R)]
        outs = [torch.empty(M, N, device=dev, dtype=torch.bfloat16) for _ in range(R)]
        sa = torch.full((M, K // 32), 127, device=dev, dtype=torch.uint8)
        sb = torch.full((N, K // 32), 127, device=dev, dtype=torch.uint8)
        one = torch.ones((), device=dev)
        ref = torch._scaled_mm(xq, wqs[0].t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16).float()
        row = {"shape": label, "M": M, "N": N, "K": K}
        row["cublaslt_tensorwise_us"] = graph_time(
            [lambda w=w: torch._scaled_mm(xq, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16) for w in wqs])
        for scaled in (False, True):
            for ws in (False, True):
                key = f"{'mx' if scaled else 'e4m3'}_tma{'_ws' if ws else ''}"
                best, errs = None, []
                for BM, BN, BK, warps, stages in CONFIGS:
                    if N % BN or K % BK:
                        continue
                    grid = (triton.cdiv(M, BM) * (N // BN),)
                    try:
                        a_desc = TensorDescriptor.from_tensor(xq, [BM, BK])
                        fns = []
                        for w, o in zip(wqs, outs):
                            b_desc = TensorDescriptor.from_tensor(w, [BN, BK])
                            fns.append(lambda a_desc=a_desc, b_desc=b_desc, o=o, cfg=(BM, BN, BK, warps, stages), grid=grid:
                                       mx_tma[grid](a_desc, b_desc, sa, sb, o, M, N, K, BM=cfg[0], BN=cfg[1], BK=cfg[2],
                                                    GM=8, SCALED=scaled, WS=ws, STAGES=cfg[4], num_warps=cfg[3]))
                        kern = fns[0]()
                        torch.cuda.synchronize()
                        err = ((outs[0].float() - ref).norm() / ref.norm()).item()
                        if err > 1e-3:
                            errs.append(f"{(BM, BN, BK, warps, stages)} err {err:.3g}")
                            continue
                        us = graph_time(fns)
                        if best is None or us < best[0]:
                            best = (us, (BM, BN, BK, warps, stages), ptx_features(kern), err)
                    except Exception as e:
                        errs.append(f"{(BM, BN, BK, warps, stages)}: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
                row[key] = None if best is None else {
                    "us": round(best[0], 2), "tflops": round(flops / best[0] / 1e6, 1), "cfg": best[1],
                    "ptx": best[2], "relerr_vs_tensorwise": best[3]}
                row[key + "_errors"] = errs[:3]
                row[key + "_n_errors"] = len(errs)
        row["cublaslt_tensorwise_tflops"] = round(flops / row["cublaslt_tensorwise_us"] / 1e6, 1)
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump(rows, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
