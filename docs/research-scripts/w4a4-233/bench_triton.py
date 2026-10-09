"""W4A4 through ``F.scaled_mm`` NVFP4 against cuBLASLt FP8 and bf16 (#233; RTX 5070 Ti).

Per shape (M x K -> N), CUDA-graph time (20 calls per graph, best of 5 replays, warm L2) and
TFLOP/s of:

* ``nvfp4 2-level`` (``+ bias``: with its bias in the GEMM): ``F.scaled_mm``
  ``[BlockWise1x16, TensorWise]``, bf16 out, the operands
  quantised beforehand (the GEMM alone);
* ``nvfp4 fp32 + epi``: ``F.scaled_mm`` ``BlockWise1x16`` with fp32 out and the per-token
  epilogue in torch (``quant.fp4_w4a4_linear``'s path without the quantiser);
* ``triton example``: ``examples/triton_nvfp4_w4a4_gemm.py``'s forward (its two Triton
  quantiser launches + the two-level GEMM + bias);
* ``fp8 tensor-wise``: ``torch._scaled_mm`` e4m3 x e4m3 with scalar scales, bf16 out (the
  GEMM alone); ``bf16``: ``torch.mm``.

    python bench_triton.py   (results/bench_triton.out)
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from kernel_agent.kernels import quant

EX = Path(__file__).resolve().parents[3] / "src/kernel_agent/agent/examples/triton_nvfp4_w4a4_gemm.py"
SHAPES = [
    (4096, 4096, 4096),
    (352, 1024, 4096),  # LocDiT gate / up
    (352, 1024, 8192),  # gate | up merged
    (352, 4096, 1024),  # down
    (352, 1024, 2560),  # q | k | v merged
    (352, 2048, 1024),  # o
]


def load_example():
    spec = importlib.util.spec_from_file_location("triton_nvfp4_w4a4_gemm", EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def graph_ms(fn, calls=20, replays=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    best = float("inf")
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / calls)
    return best


def main():
    ex = load_example()
    fp4 = torch.float4_e2m1fn_x2
    bw, tw = F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise
    sw, ns = F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")
    print(
        "| M x K -> N | nvfp4 2-level | + bias | nvfp4 fp32 + epi | triton example | "
        "fp8 tensor-wise | bf16 | example's quantiser | GEMM vs fp8 | example vs fp8 GEMM |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m, k, n in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda").bfloat16()
        lin = torch.nn.Linear(k, n, device="cuda", dtype=torch.bfloat16)
        w = lin.weight.detach()
        wc, ws, wt = quant.quantize_fp4(w)
        wsw = quant.swizzle_fp4_scales(ws)
        xc, xs, xo = quant.quantize_fp4_activations(x, "nvfp4", "tensor")
        xsw = quant.swizzle_fp4_scales(xs)
        ga, gb = xo[:1].reshape(()).contiguous(), wt.reshape(()).float()
        wts = float(wt)

        def two_level(bias=None):
            return F.scaled_mm(
                xc.view(fp4), wc.view(fp4).t(), scale_a=[xsw, ga], scale_recipe_a=[bw, tw],
                scale_b=[wsw, gb], scale_recipe_b=[bw, tw], swizzle_a=[sw, ns],
                swizzle_b=[sw, ns], bias=bias, output_dtype=torch.bfloat16,
            )

        def two_level_bias():
            return two_level(lin.bias)

        tc, ts_, to = quant.quantize_fp4_activations(x)  # per token
        tsw = quant.swizzle_fp4_scales(ts_)

        def fp32_epi():
            acc = F.scaled_mm(
                tc.view(fp4), wc.view(fp4).t(), scale_a=tsw, scale_recipe_a=bw,
                scale_b=wsw, scale_recipe_b=bw, swizzle_a=sw, swizzle_b=sw,
                output_dtype=torch.float32,
            )
            return ((acc * (to[:, None] * wts)) + lin.bias.float()).to(torch.bfloat16)

        mod = ex.build(lin)

        def example():
            return mod(x)

        def quantiser():
            return ex._quantize(x, swizzle=True)

        q8, s8 = quant.quantize_fp8(w)
        x8 = (x.float() / 4).to(torch.float8_e4m3fn)
        one = torch.ones((), device="cuda")

        def fp8():
            return torch._scaled_mm(x8, q8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

        def bf16():
            return x @ w.t()

        with torch.no_grad():
            fns = (two_level, two_level_bias, fp32_epi, example, fp8, bf16)
            times = [graph_ms(f) for f in fns]
            q_us = graph_ms(quantiser) * 1e3
        flop = 2 * m * k * n
        cells = [f"{t * 1e3:.1f} us ({flop / t / 1e9:.0f})" for t in times]
        print(
            f"| {m} x {k} -> {n} | " + " | ".join(cells) + f" | {q_us:.1f} us | "
            f"{times[4] / times[0]:.2f}x | {times[4] / times[3]:.2f}x |"
        )
    print("(TFLOP/s in brackets; warm L2, CUDA graphs)")


if __name__ == "__main__":
    main()
