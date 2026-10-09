"""Where the Triton W4A4 example's eager time goes (#233): its quantiser launches and
``F.scaled_mm``'s host dispatch, each alone.

    python host_parts.py   (results/host_time.out)
"""

import importlib.util
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from kernel_agent.kernels import quant

base = str(Path(__file__).resolve().parents[3] / "src/kernel_agent/agent/examples") + "/"
spec = importlib.util.spec_from_file_location("ex", base + "triton_nvfp4_w4a4_gemm.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def eager_us(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


torch.manual_seed(0)
lin = torch.nn.Linear(1024, 8192, bias=False, device="cuda", dtype=torch.bfloat16)
x = torch.randn(704, 1024, device="cuda", dtype=torch.bfloat16)
mod = ex.build(lin)
m, k = x.shape
cols = k // 16
found = torch.zeros(2, device="cuda")
codes = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
scales = torch.zeros(triton.cdiv(m, 128) * 128 * cols, device="cuda", dtype=torch.uint8)
xq, xs, outer = ex._quantize(x, swizzle=True)
fp4 = torch.float4_e2m1fn_x2
bw, tw = F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise
sw, ns = F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE
with torch.no_grad():
    print("forward", eager_us(lambda: mod(x)))
    print("_quantize", eager_us(lambda: ex._quantize(x, swizzle=True)))
    print("amax kernel", eager_us(lambda: ex._amax_kernel[(m,)](x, found, k, x.stride(0), CHUNK=512, num_warps=4)))
    print("quant kernel", eager_us(lambda: ex._nvfp4_quant_kernel[(m, triton.cdiv(cols, ex.BLOCKS))](
        x, found, codes, scales, k, x.stride(0), cols, quant.NVFP4_OUTER_STEP, SWIZZLE=True, BLOCKS=ex.BLOCKS,
        num_warps=ex.QUANT_WARPS)))
    print("zeros(2)", eager_us(lambda: torch.zeros(2, device="cuda")))
    print("scaled_mm 2-level", eager_us(lambda: F.scaled_mm(
        xq.view(fp4), mod.weight_fp4.view(fp4).t(), scale_a=[xs, outer], scale_recipe_a=[bw, tw],
        scale_b=[mod.weight_scales_swizzled, mod.tensor_scale], scale_recipe_b=[bw, tw],
        swizzle_a=[sw, ns], swizzle_b=[sw, ns], output_dtype=torch.bfloat16)))
    print("_scaled_mm_v2 direct", eager_us(lambda: torch._scaled_mm_v2(
        xq.view(fp4), mod.weight_fp4.view(fp4).t(), [xs, outer], [int(bw), int(tw)], [int(sw), int(ns)],
        [mod.weight_scales_swizzled, mod.tensor_scale], [int(bw), int(tw)], [int(sw), int(ns)], None,
        torch.bfloat16, [], False)))
