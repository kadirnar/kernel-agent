"""`examples/triton_int8_w8a8_gemm.py` at the VoxCPM2 LocDiT GEMM shapes (M = 352 / 704): bit
for bit against `quant.int8_w8a8_linear` (`torch._int_mm`), CUDA-graph time (warm L2) against
cuBLAS bf16, the `_int_mm` reference path and `examples/triton_fp8_w8a8_gemm.py`; with
`sweep`, the best tile configs (BM, BN, BK, warps, stages)."""

import itertools
import sys

import torch
from int8lib import example, graph_us
from torch import nn

from kernel_agent.kernels import quant
from kernel_agent.kernels.bench import ensure_clocks, warm_gpu

int8 = example("triton_int8_w8a8_gemm.py")
fp8 = example("triton_fp8_w8a8_gemm.py")
warm_gpu()
ensure_clocks()
torch.manual_seed(0)
shapes = [(8192, 1024), (1024, 4096), (2560, 1024), (1024, 2048)]
sweep = "sweep" in sys.argv
for (n, k), m in itertools.product(shapes, (352, 704)):
    lin = nn.Linear(k, n, bias=False).cuda().bfloat16()
    with torch.no_grad():
        lin.weight.normal_(0, k**-0.5)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    mod = int8.build(lin)
    with torch.inference_mode():
        y = mod(x)
        ref = quant.int8_w8a8_linear(x, mod.weight_int8, mod.weight_scale)
        exact = torch.equal(y, ref)
        rel = float((y.float() - lin(x).float()).norm() / lin(x).float().norm())
        t_int8 = graph_us(lambda: mod(x))
        t_ref = graph_us(lambda: quant.int8_w8a8_linear(x, mod.weight_int8, mod.weight_scale))
        t_bf16 = graph_us(lambda: lin(x))
        f = fp8.build(lin)
        t_fp8 = graph_us(lambda: f(x))
    print(
        f"N={n} K={k} M={m}: int8 example {t_int8:.1f} us (bit-exact vs _int_mm ref: {exact}, "
        f"rel L2 vs bf16 {rel:.4f}) | _int_mm path {t_ref:.1f} | bf16 {t_bf16:.1f} | "
        f"fp8 example {t_fp8:.1f}",
        flush=True,
    )
    if sweep:
        best = []
        grid = itertools.product((64, 128), (64, 128, 256), (64, 128, 256), (4, 8), (2, 3, 4))
        for bm, bn, bk, w, st in grid:
            if n % bn or k % bk:
                continue
            try:
                c = int8.build(lin, bm, bn, bk, w, st)
                with torch.inference_mode():
                    if not torch.equal(c(x), y):
                        print("  mismatch", (bm, bn, bk, w, st))
                        continue
                    best.append((graph_us(lambda c=c: c(x), 10), (bm, bn, bk, w, st)))
            except Exception:  # out of shared memory etc.
                pass
        best.sort()
        print("   best:", [(round(t, 1), c) for t, c in best[:4]], flush=True)
