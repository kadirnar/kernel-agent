"""(1) Host time per call of each FP8 GEMM API (eager, as the module evaluator times it), and
(2) the dense FP8 peak of each scaling recipe at 8192^3 on this GPU."""

import math
import sys
import time

import torch
import torch.nn.functional as F
import triton
from torch.nn.functional import ScalingType as ST
from torch.nn.functional import SwizzleType as SW

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cutlass_bw import load as load_cutlass  # noqa: E402
from gemm_bench import _bw_gemm, _mx_gemm, graph_us, to_blocked  # noqa: E402
from lt_ext import load  # noqa: E402

from kernel_agent.agent.examples import triton_fp8_w8a8_gemm as tex
from kernel_agent.kernels.bench import ensure_clocks, warm_gpu

E4M3 = torch.float8_e4m3fn
ext = load()
cut = load_cutlass()
warm_gpu()
print("clock state", ensure_clocks())


def host_us(fn, n=3000):
    """Wall time per call of n back-to-back calls with no sync inside (the GPU work is
    tiny, so the host is the bottleneck), median of 5 rounds; and GPU time per call."""
    for _ in range(200):
        fn()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        rounds.append((time.perf_counter() - t) / n * 1e6)
        torch.cuda.synchronize()
    rounds.sort()
    return rounds[2]


# ---- (1) host overhead, small GEMM (GPU time a few us) ----
M, N, K = 16, 256, 256
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
xq, wq = x.to(E4M3), w.to(E4M3)
one = torch.ones((), device="cuda")
sx, sw = torch.ones(M, 1, device="cuda"), torch.ones(1, N, device="cuda")
sxm = to_blocked(torch.ones(M, K // 32, device="cuda").to(torch.float8_e8m0fnu))
swm = to_blocked(torch.ones(N, K // 32, device="cuda").to(torch.float8_e8m0fnu))
y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
ext.tune(wq, xq, one, one, y, 0, 0, 1, 10, 2)
qs, ss = tex.quantize_fp8(w)
cfg = [16, 64, 128, 4, 3]
sxb = torch.ones(K // 128, M, device="cuda")
swb = torch.ones(K // 128, N // 128, device="cuda")
cases = {
    "F.linear bf16 (cuBLAS)": lambda: F.linear(x, w),
    "torch._scaled_mm tensorwise": lambda: torch._scaled_mm(
        xq, wq.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
    "torch._scaled_mm rowwise": lambda: torch._scaled_mm(
        xq, wq.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16),
    "F.scaled_mm MXFP8": lambda: F.scaled_mm(
        xq, wq.t(), scale_a=sxm, scale_recipe_a=ST.BlockWise1x32, scale_b=swm,
        scale_recipe_b=ST.BlockWise1x32, swizzle_a=SW.SWIZZLE_32_4_4,
        swizzle_b=SW.SWIZZLE_32_4_4, output_dtype=torch.bfloat16),
    "at::_scaled_mm from C++ (pybind11)": lambda: ext.aten_scaled_mm(xq, wq.t(), one, one),
    "cuBLASLt direct, cached plan (pybind11)": lambda: ext.run(wq, xq, one, one, y, 0, 0, 1),
    "pybind11 call, empty body": lambda: ext.noop(wq, xq, one, one, y, 0, 0, 1),
    "CUTLASS sm120 blockwise (load_inline, per-call init)": lambda: cut.ping64(xq, wq, sxb, swb, y),
    "Triton e4m3 GEMM launch (prequantised x)": lambda: tex._gemm_kernel[(1 * (N // 64),)](
        xq, wq, sx, ss, sx, y, M, N, K, HAS_BIAS=False, BM=16, BN=64, BK=128, GM=8,
        num_warps=4, num_stages=3),
    "Triton quant + GEMM (example launcher)": lambda: tex._w8a8_linear(x, qs, ss, None, cfg),
    "Triton quant + GEMM via torch.library custom_op": lambda: tex.w8a8_linear(x, qs, ss, None, cfg),
}
print(f"\n## host time per call, eager, [{M}, {K}] x [{K}, {N}] (GPU work ~ a few us)")
results = {}
for name, fn in cases.items():
    try:
        results[name] = host_us(fn)
        print(f"  {name:55s} {results[name]:6.1f} us/call")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:55s} {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
n4 = host_us(lambda: ext.run_n(wq, xq, one, one, y, 4), n=2000)
print(f"  {'4 cuBLASLt GEMMs behind one pybind11 call':55s} {n4:6.1f} us/call ({n4 / 4:.1f} per GEMM)")

# CUDA graph replay of one GEMM
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    ext.run(wq, xq, one, one, y, 0, 0, 1)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    for _ in range(4):
        ext.run(wq, xq, one, one, y, 0, 0, 1)
print(f"  {'CUDA graph replay (4 cuBLASLt GEMMs inside)':55s} {host_us(g.replay, n=2000):6.1f} us/replay")

# CuTe DSL (TVM-FFI) vs CUDA C++ launch floor: the bundled RMSNorm examples
from kernel_agent.agent.examples import cuda_rmsnorm, cute_rmsnorm  # noqa: E402
from voxcpm.modules.minicpm4.model import MiniCPMRMSNorm  # noqa: E402

ref = MiniCPMRMSNorm(1024).cuda().to(torch.bfloat16)
h = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
for label, mod in (("CuTe DSL RMSNorm (TVM-FFI launch)", cute_rmsnorm.build(ref)),
                   ("CUDA C++ RMSNorm (load_inline launch)", cuda_rmsnorm.build(ref)),
                   ("torch RMSNorm module (eager ops)", ref)):
    print(f"  {label:55s} {host_us(lambda m=mod: m(h)):6.1f} us/call")

# ---- (2) dense FP8 peaks at 8192^3 ----
print("\n## dense peak, 8192^3, CUDA graph (min of replays)")
M = N = K = 8192
x = (torch.randn(M, K, device="cuda") * 0.5).to(E4M3)
w = (torch.randn(N, K, device="cuda") * 0.5).to(E4M3)
xb = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
wb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
sx, sw = torch.ones(M, 1, device="cuda"), torch.ones(1, N, device="cuda")
sxm = to_blocked(torch.ones(M, K // 32, device="cuda").to(torch.float8_e8m0fnu))
swm = to_blocked(torch.ones(N, K // 32, device="cuda").to(torch.float8_e8m0fnu))
sxm_u8 = torch.full((M, K // 32), 127, device="cuda", dtype=torch.uint8)
swm_u8 = torch.full((N, K // 32), 127, device="cuda", dtype=torch.uint8)
sxb = torch.ones(K // 128, M, device="cuda")
swb = torch.ones(K // 128, N // 128, device="cuda")
swb_t = torch.ones(N // 128, K // 128, device="cuda")
ext.tune(w, x, one, one, y, 0, 0, 1, 5, 2)
peaks = {
    "bf16 cuBLAS": lambda: torch.mm(xb, wb.t(), out=y),
    "tensorwise cuBLASLt (direct)": lambda: ext.run(w, x, one, one, y, 0, 0, 1),
    "rowwise torch._scaled_mm": lambda: torch._scaled_mm(x, w.t(), scale_a=sx, scale_b=sw,
                                                         out_dtype=torch.bfloat16),
    "MXFP8 cuBLASLt (F.scaled_mm)": lambda: F.scaled_mm(
        x, w.t(), scale_a=sxm, scale_recipe_a=ST.BlockWise1x32, scale_b=swm,
        scale_recipe_b=ST.BlockWise1x32, swizzle_a=SW.SWIZZLE_32_4_4,
        swizzle_b=SW.SWIZZLE_32_4_4, output_dtype=torch.bfloat16),
    "blockwise CUTLASS sm120 coop 128x128x128": lambda: cut.coop128(x, w, sxb, swb, y),
    "blockwise CUTLASS sm120 pingpong 64x128x128": lambda: cut.ping64(x, w, sxb, swb, y),
    "blockwise Triton (128x128, 8 warps)": lambda: _bw_gemm[(64 * 64,)](
        x, w, sxb, swb_t, y, M, N, K, BM=128, BN=128, GM=8, num_warps=8, num_stages=3),
    "MXFP8 Triton tl.dot_scaled (128x128x128)": lambda: _mx_gemm[(64 * 64,)](
        x, w, sxm_u8, swm_u8, y, M, N, K, BM=128, BN=128, BK=128, GM=8, num_warps=8,
        num_stages=3),
    "rowwise Triton tl.dot (128x128x128)": lambda: tex._gemm_kernel[(64 * 64,)](
        x, w, sx, sw, sx, y, M, N, K, HAS_BIAS=False, BM=128, BN=128, BK=128, GM=8,
        num_warps=8, num_stages=3),
}
for name, fn in peaks.items():
    try:
        us = graph_us([fn], reps=5, replays=5)
        print(f"  {name:45s} {2 * M * N * K / us / 1e6:6.1f} TFLOP/s")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:45s} {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
_ = (math, triton)
