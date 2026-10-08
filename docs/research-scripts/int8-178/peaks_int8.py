"""INT8 peak (torch._int_mm) and the s8 IMMA mma.sync rate on this GPU, next to FP8 / bf16."""

import torch

from kernel_agent.kernels import mma_peaks, quant, roofline
from kernel_agent.kernels.bench import ensure_clocks, warm_gpu

warm_gpu()
ensure_clocks()
free, _ = torch.cuda.mem_get_info()
big = [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 8192, 8192)]
print("int8 _int_mm TOPS per shape:")
for shape in big:
    print(" ", shape, roofline._int8_tops([shape], free))
print("fp8 _scaled_mm TFLOP/s:", roofline._fp8_tflops(big[:2], free))
print("bf16 TFLOP/s:", roofline._matmul_tflops(torch.bfloat16, big[:2], free))
rates, missing = mma_peaks.measure()
print("mma rates:", rates, missing)
print(mma_peaks.describe(rates))

# exactness: _int_mm vs the exact fp32-chunk fallback, odd M
g = torch.Generator(device="cuda").manual_seed(0)
for m, k, n in ((1, 4096, 1024), (5, 2048, 6144), (352, 1024, 8192), (17, 8192, 1024)):
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8, generator=g)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8, generator=g)
    lt = quant.int8_matmul(a, b)
    ref = quant.int8_matmul(a.cpu(), b.cpu())
    print("exact", (m, k, n), torch.equal(lt.cpu(), ref))
# GEMM shapes of the VoxCPM2 LocDiT at M=352 / 704: _int_mm vs bf16 cuBLAS vs fp8 tensorwise
from kernel_agent.kernels.roofline import _best_ms

for m, n, k in ((352, 8192, 1024), (704, 8192, 1024), (352, 1024, 4096), (352, 2560, 1024), (4096, 4096, 4096)):
    a8 = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    b8 = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8).t()
    abf = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bbf = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
    af8, bf8 = abf.to(torch.float8_e4m3fn), torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).t()
    one = torch.ones((), device="cuda")
    ti = _best_ms(lambda: torch._int_mm(a8, b8), 50)
    tb = _best_ms(lambda: abf @ bbf, 50)
    tf = _best_ms(lambda: torch._scaled_mm(af8, bf8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16), 50)
    ops = 2 * m * n * k / 1e9
    print(f"M={m} N={n} K={k}: int8 {ti*1000:.1f} us ({ops/ti:.0f} TOPS) bf16 {tb*1000:.1f} us ({ops/tb:.0f}) fp8 {tf*1000:.1f} us ({ops/tf:.0f})")
