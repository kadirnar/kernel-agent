"""CUTLASS SM120 blockwise FP8 GEMM (example-87-style) at VoxCPM2 shapes: correctness
against the fp32 math of the block-dequantised operands, and L2-cold CUDA-graph timing."""

import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cutlass_bw import load  # noqa: E402
from fp8lib import q8  # noqa: E402
from gemm_bench import graph_us  # noqa: E402

from kernel_agent.kernels.bench import ensure_clocks, warm_gpu

ext = load()
E4M3 = torch.float8_e4m3fn
warm_gpu()
print("clock state", ensure_clocks())
SHAPES = [(352, 2560, 1024, "LocDiT q|k|v"), (352, 1024, 2048, "LocDiT o_proj"),
          (352, 8192, 1024, "LocDiT gate|up"), (352, 1024, 4096, "LocDiT down"),
          (22, 8192, 1024, "LocDiT gate|up b1"), (80, 8192, 1024, "LocEnc gate|up b16"),
          (272, 12288, 2048, "LM prefill gate|up b16"), (272, 2048, 6144, "LM prefill down b16"),
          (16, 12288, 2048, "LM gate|up b16")]
torch.manual_seed(0)
for M, N, K, label in SHAPES:
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(N, K, device="cuda") * 0.02
    g = x.view(M, K // 128, 128)
    sx = g.abs().amax(-1) / 448  # [M, K/128]
    xq = (g / sx[..., None]).clamp(-448, 448).to(E4M3).view(M, K)
    gw = w.view(N // 128, 128, K // 128, 128)
    sw = gw.abs().amax(dim=(1, 3)) / 448  # [N/128, K/128]
    wq = (gw / sw[:, None, :, None]).clamp(-448, 448).to(E4M3).view(N, K)
    xd = (xq.float().view(M, K // 128, 128) * sx[..., None]).view(M, K)
    wd = (wq.float().view(N // 128, 128, K // 128, 128) * sw[:, None, :, None]).view(N, K)
    ref = (xd.double() @ wd.double().T).float()
    sx_mn = sx.t().contiguous()  # [K/128, M]: M-major
    sw_mn = sw.t().contiguous()  # [K/128, N/128]: N-major
    y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    copies = max(3, math.ceil(160e6 / (N * K)))
    wqs = [wq] + [wq.clone() for _ in range(copies - 1)]
    print(f"\n### {label}: M={M} N={N} K={K}")
    for name in ("coop128", "ping64"):
        fn = getattr(ext, name)
        try:
            fn(xq, wq, sx_mn, sw_mn, y)
            torch.cuda.synchronize()
            err = float((y.float() - ref).norm() / ref.norm())
            us = graph_us([lambda i=i: fn(xq, wqs[i], sx_mn, sw_mn, y) for i in range(copies)])
            print(f"  CUTLASS blockwise {name:8s} {us:7.2f} us {2 * M * N * K / us / 1e6:6.1f} "
                  f"TFLOP/s  rel err vs fp32 math {err:.1e}")
        except Exception as exc:  # noqa: BLE001
            print(f"  CUTLASS blockwise {name:8s} {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
    del wqs
    torch.cuda.empty_cache()
_ = q8
