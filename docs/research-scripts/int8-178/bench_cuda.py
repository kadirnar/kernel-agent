"""The CUDA INT8 examples streamed (CUDA graph over enough weight copies to exceed the 48 MB
L2): `cuda_int8_skinny_gemm.py` (W8A8, bit for bit against `quant.int8_w8a8_linear`) and
`cuda_int8_gemv.py` (weight-only, against `quant.int8_weights_linear`) next to cuBLAS bf16 and
the FP8 skinny GEMM / GEMV examples. Usage: bench_cuda.py [rows ...] (default 1 16 32 64);
`sweep`: the skinny kernel's rows / warps / unroll per shape at 1, 8, 16, 32 rows."""

import itertools
import sys

import torch
from int8lib import example, streamed_us
from torch import nn

from kernel_agent.kernels import quant
from kernel_agent.kernels.bench import ensure_clocks, warm_gpu

skinny8 = example("cuda_int8_skinny_gemm.py")
gemv8 = example("cuda_int8_gemv.py")
skinnyf8 = example("cuda_fp8_skinny_gemm.py")
gemvf8 = example("cuda_fp8_gemv.py")
L2 = 48 * 2**20
SHAPES = [(2048, 6144), (2048, 12288), (6144, 2048), (1024, 4096), (4096, 1024), (2048, 2048)]


def linears(k, n):
    copies = max(2, int(2 * L2 // (n * k)) + 1)  # int8 weights: > 2x L2 in total
    out = []
    for _ in range(copies):
        lin = nn.Linear(k, n, bias=False).cuda().bfloat16()
        with torch.no_grad():
            lin.weight.normal_(0, k**-0.5)
        out.append(lin)
    return out


def bench(rows_list):
    for k, n in SHAPES:
        lins = linears(k, n)
        sk = [skinny8.build(lin) for lin in lins]
        gv = [gemv8.build(lin) for lin in lins]
        skf = [skinnyf8.build(lin) for lin in lins]
        gvf = [gemvf8.build(lin) for lin in lins]
        for m in rows_list:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                y = sk[0](x)
                ref = quant.int8_w8a8_linear(x, sk[0].weight_int8, sk[0].weight_scale)
                exact = torch.equal(y, ref)
                yb = lins[0](x).float()
                rel = float((y.float() - yb).norm() / yb.norm())
                g = gv[0](x).float()
                gref = quant.int8_weights_linear(x, gv[0].weight_int8, gv[0].weight_scale)
                grel = float((g - gref.float()).norm() / gref.float().norm())
                grel_bf = float((g - yb).norm() / yb.norm())
            line = (
                f"[{m}, {k}] x [{k}, {n}]: bf16 {streamed_us(lins, x):.1f} us | int8 W8A8 "
                f"skinny {streamed_us(sk, x):.1f} (exact {exact}, rel L2 {rel:.4f}) | fp8 "
                f"skinny {streamed_us(skf, x):.1f}"
            )
            if m <= 4:
                t_gv = streamed_us(gv, x)
                line += (
                    f" | int8 gemv {t_gv:.1f} ({n * k / t_gv / 1e3:.0f} GB/s, vs ref "
                    f"{grel:.1e}, rel L2 {grel_bf:.4f}) | fp8 gemv {streamed_us(gvf, x):.1f}"
                )
            print(line, flush=True)
        del lins, sk, gv, skf, gvf
        torch.cuda.empty_cache()


def sweep():
    for k, n in [(2048, 6144), (2048, 12288), (6144, 2048), (4096, 1024)]:
        lins = linears(k, n)
        for m in (1, 8, 16, 32):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            res = []
            for rows, warps, unroll in itertools.product((1, 2), (4, 8), (1, 2)):
                mods = [skinny8.build(lin, rows, warps, unroll) for lin in lins]
                res.append((round(streamed_us(mods, x), 1), (rows, warps, unroll)))
            auto = round(streamed_us([skinny8.build(lin) for lin in lins], x), 1)
            res.sort()
            print(f"[{m}, {k}] x [{k}, {n}] auto {auto}: {res[:3]}", flush=True)
        del lins
        torch.cuda.empty_cache()


warm_gpu()
ensure_clocks()
torch.manual_seed(0)
if "sweep" in sys.argv:
    sweep()
else:
    bench([int(v) for v in sys.argv[1:]] or [1, 16, 32, 64])
