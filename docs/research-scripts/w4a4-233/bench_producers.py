"""NVFP4 quantisation fused into the producer against a separate pass (#233 follow-up 1):
``examples/triton_fp4_producers.py`` on any CUDA GPU (memory bound: no FP4 tensor cores
needed).

Per shape, CUDA-graph time per call (20 calls per graph, the median of 7 rounds that replay
every variant in turn, warm L2):

* ``bf16 out``: the producer alone writing bf16 (a Triton RMSNorm / SiLU-mul with the
  example's math): what the layer pays today before a W4A4 GEMM;
* ``fused NVFP4``: ``rmsnorm_fp4`` / ``silu_mul_fp4`` writing packed codes, swizzled e4m3
  scales and the per-token outer scale (what the GEMM reads); ``+ bias factor``: with the
  opt-in per-token epilogue scale (``unbiased=True``);
* ``bf16 + quantize_rows``: the bf16 producer, then the example's one-launch per-token
  quantiser (``quantize_rows``, swizzled) on its output: the separate pass;
* ``bf16 + torch reference``: the bf16 producer, then ``quant.quantize_fp4_activations`` and
  ``swizzle_fp4_scales`` in torch (a fallback's cost).

    PYTHONPATH=src python bench_producers.py   (results/bench_producers.out: NVIDIA A10)
"""

import importlib.util
import statistics

import torch
import triton
import triton.language as tl

from kernel_agent.agent import prompts
from kernel_agent.kernels import quant

spec = importlib.util.spec_from_file_location(
    "fp4_producers", prompts.EXAMPLES_DIR / "triton_fp4_producers.py"
)
prod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prod)


@triton.jit
def _rmsnorm_bf16(x, w, y, K, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    v = tl.load(x + row * K + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 0) / K + eps)
    hn = (v * r).to(tl.bfloat16).to(tl.float32)
    wv = tl.load(w + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y + row * K + cols, (hn * wv).to(tl.bfloat16), mask=mask)


@triton.jit
def _silu_mul_bf16(gu, y, K, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    g = tl.load(gu + row * 2 * K + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu + row * 2 * K + K + cols, mask=mask, other=0.0).to(tl.float32)
    a = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(y + row * K + cols, (a * u).to(tl.bfloat16), mask=mask)


CALLS = 20


def graph(fn, calls=CALLS):
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
    return g


def replay_us(g, calls=CALLS):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / calls


def interleaved_us(fns, rounds=7):
    """Median per function of ``rounds`` rounds that replay every function's graph in turn:
    the A10 is power capped (150 W), so its clocks drift under load and a function timed
    after another would carry that drift."""
    graphs = [graph(f) for f in fns]
    times = [[] for _ in fns]
    for _ in range(rounds):
        for i, g in enumerate(graphs):
            times[i].append(replay_us(g))
    return [statistics.median(t) for t in times]


def graph_us(fn, calls=CALLS, replays=7):
    g = graph(fn, calls)
    return statistics.median(replay_us(g, calls) for _ in range(replays))


def warps(k):
    return 4 if triton.next_power_of_2(k) <= 4096 else 8


def main():
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, triton {triton.__version__}")
    print("CUDA graphs, 20 calls per graph, median of 7 interleaved rounds, warm L2; us per call\n")
    print(
        "| producer, M x K | bf16 out | fused NVFP4 | + bias factor | bf16 + quantize_rows | "
        "bf16 + torch reference | fused vs bf16 + quantize_rows |"
    )
    print("|---|---|---|---|---|---|---|")
    for kind, m, k in (
        ("RMSNorm", 352, 1024),
        ("RMSNorm", 4096, 1024),
        ("RMSNorm", 352, 4096),
        ("SiLU-mul", 352, 3072),
        ("SiLU-mul", 352, 4096),
        ("SiLU-mul", 4096, 3072),
    ):
        torch.manual_seed(0)
        block = triton.next_power_of_2(k)
        y = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
        if kind == "RMSNorm":
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3
            w = (torch.randn(k, device="cuda") * 0.1 + 1).to(torch.bfloat16)

            def bf16(x=x, w=w, y=y, k=k, block=block, m=m):
                _rmsnorm_bf16[(m,)](x, w, y, k, 1e-6, BLOCK=block, num_warps=warps(k))

            def fused(x=x, w=w, unbiased=False):
                prod.rmsnorm_fp4(x, w, 1e-6, swizzle=True, unbiased=unbiased)

        else:
            x = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16) * 2

            def bf16(x=x, y=y, k=k, block=block, m=m):
                _silu_mul_bf16[(m,)](x, y, k, BLOCK=block, num_warps=warps(k))

            def fused(x=x, unbiased=False):
                prod.silu_mul_fp4(x, swizzle=True, unbiased=unbiased)

        def separate(bf16=bf16, y=y):
            bf16()
            prod.quantize_rows(y, swizzle=True)

        def torch_ref(bf16=bf16, y=y):
            bf16()
            q, s, o = quant.quantize_fp4_activations(y)
            quant.swizzle_fp4_scales(s)

        def unbiased(fused=fused):
            fused(unbiased=True)

        t = interleaved_us((bf16, fused, unbiased, separate, torch_ref))
        print(
            f"| {kind} {m} x {k} | {t[0]:.2f} | {t[1]:.2f} | {t[2]:.2f} | {t[3]:.2f} | "
            f"{t[4]:.1f} | {t[3] / t[1]:.2f}x |"
        )


if __name__ == "__main__":
    main()
