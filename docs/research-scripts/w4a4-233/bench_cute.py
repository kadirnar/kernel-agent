"""``examples/cute_nvfp4_w4a4_gemm.py`` (CuTe DSL NVFP4 W4A4 GEMM, sm_120) checked and timed
(#233): its quantiser bit for bit against ``quant.quantize_fp4_activations``, its output
against ``quant.fp4_w4a4_linear`` (the same codes, fp32 math), and CUDA-graph timings
against ``F.scaled_mm`` NVFP4 (cuBLASLt ``BlockWise1x16``, bf16 and fp32 out), cuBLASLt FP8
tensor-wise (``torch._scaled_mm``, scalar scales, bf16 out) and bf16 ``torch.mm``, at 4096^3
and at the VoxCPM2 LocDiT shapes at M = 352. "warm": one GEMM replayed back to back (its
operands stay in L2); "cold": enough copies of the operands to exceed L2, cycled (each
call streams its weight from DRAM, as in a model).

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 \
        python bench_cute.py [check|bench|all] > results/bench_cute.out
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from kernel_agent.kernels import quant

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "src/kernel_agent/agent/examples/cute_nvfp4_w4a4_gemm.py"
FP4 = torch.float4_e2m1fn_x2
L2_BYTES = 48 * 2**20


def load_example():
    spec = importlib.util.spec_from_file_location("cute_nvfp4_w4a4_gemm", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _graph_us(calls, reps):
    """Best us per call of ``calls`` (zero-argument callables) replayed in a CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for fn in calls:
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        for _ in range(reps):
            for fn in calls:
                fn()
    g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / (reps * len(calls)))
    return best


# ------------------------------------------------------------------ correctness


def check(ex):
    print("## quantiser: bit for bit against quant.quantize_fp4_activations")
    torch.manual_seed(0)
    for m, k in ((1, 1024), (17, 2048), (352, 1024), (352, 4096), (4096, 4096)):
        for kind in ("normal", "outliers"):
            x = torch.randn(m, k, device="cuda")
            if kind == "outliers":
                x = x * torch.logspace(-3, 3, m, device="cuda")[:, None]
                x[:, 5] *= 400.0
                x[:, 100:132] = 0.0
            x = x.to(torch.bfloat16)
            q, sf, outer = ex.quantize_activations(x)
            rq, rsf, router = quant.quantize_fp4_activations(x, "nvfp4")
            fq = ex.compile_quant(k)
            q2 = torch.empty_like(q)
            ssw = torch.zeros(ex.swizzled_scale_bytes(m, k), dtype=torch.uint8, device="cuda")
            o2 = torch.empty_like(outer)
            fq(x, q2, ssw, o2)
            sw_ref = quant.swizzle_fp4_scales(rsf).view(torch.uint8)
            same = (
                torch.equal(q, rq),
                torch.equal(sf.view(torch.uint8), rsf.view(torch.uint8)),
                torch.equal(outer, router),
                torch.equal(q2, rq) and torch.equal(ssw, sw_ref) and torch.equal(o2, router),
            )
            print(
                f"M {m:5d} K {k:5d} {kind:8s}: codes {same[0]}, scales {same[1]}, "
                f"outer {same[2]}, swizzled path {same[3]}"
            )
    print("\n## GEMM: module output against quant.fp4_w4a4_linear (same codes, fp32 math)")
    shapes = [
        (m, n, k)
        for m in (1, 17, 352, 4096)
        for n, k in ((2048, 1024), (256, 1024), (4096, 1024), (1024, 4096), (1024, 2048))
    ] + [(4096, 4096, 4096)]
    worst = 0.0
    for m, n, k in shapes:
        for bias in (True, False):
            lin = torch.nn.Linear(k, n, bias=bias, device="cuda", dtype=torch.bfloat16)
            mod = ex.build(lin)
            assert isinstance(mod, ex.CuteNvfp4Linear)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                y = mod(x).float()
                ref = quant.fp4_w4a4_linear(
                    x, mod.weight_codes, mod.weight_scales, mod.tensor_scale, lin.bias
                ).float()
                # the kernel's epilogue: one fma(acc, outer * tensor_scale, bias), one bf16
                # rounding (the reference rounds the product before adding the bias)
                xq, xs, outer = quant.quantize_fp4_activations(x, "nvfp4")
                acc = quant.fp4_values(xq, xs) @ quant.fp4_values(
                    mod.weight_codes, mod.weight_scales
                ).T
                s = (outer[:, None] * float(mod.tensor_scale)).double()
                fma = acc.double() * s
                if bias:
                    fma = fma + lin.bias.double()[None, :]
                fma = fma.float().to(torch.bfloat16).float()
            d = y - ref
            rel = float(d.norm() / ref.norm())
            differ = int((d != 0).sum())
            worst = max(worst, rel)
            print(
                f"M {m:5d} N {n:5d} K {k:5d} bias {bias!s:5s}: rel L2 {rel:.2e}, max |d| "
                f"{float(d.abs().max()):.3e} (max |ref| {float(ref.abs().max()):.3g}), "
                f"{differ} of {d.numel()} outputs differ; bit for bit to the fma epilogue: "
                f"{torch.equal(y, fma)}"
            )
    print(f"worst rel L2 {worst:.2e}")
    for persistent in (False,):
        lin = torch.nn.Linear(1024, 4096, device="cuda", dtype=torch.bfloat16)
        mod = ex.build(lin, persistent=persistent)
        x = torch.randn(352, 1024, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            ref = quant.fp4_w4a4_linear(
                x, mod.weight_codes, mod.weight_scales, mod.tensor_scale, lin.bias
            ).float()
            rel = float((mod(x).float() - ref).norm() / ref.norm())
        print(f"persistent={persistent}: rel L2 {rel:.2e}")
    print("selftest", ex.selftest())


# ------------------------------------------------------------------ timing


def _operands(m, n, k, copies):
    """``copies`` sets of operands of every path for an [m, k] x [n, k]^T GEMM."""
    sets = []
    for _ in range(copies):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
        xq, xs, outer = quant.quantize_fp4_activations(x, "nvfp4")
        wq, ws, wt = quant.quantize_fp4(w)
        x8 = (x.float() / (x.float().abs().amax() / 448)).to(torch.float8_e4m3fn)
        w8 = (w.float() / (w.float().abs().amax() / 448)).to(torch.float8_e4m3fn)
        sets.append(
            {
                "x": x,
                "w": w,
                "xq": xq,
                "xs": quant.swizzle_fp4_scales(xs),
                "outer": outer,
                "wq": wq,
                "ws": quant.swizzle_fp4_scales(ws),
                "wt": wt,
                "x8": x8,
                "w8": w8,
                "one": torch.ones((), device="cuda"),
            }
        )
    return sets


def _paths(ex, m, n, k, ops):
    """name -> callable(op set) of every timed path."""
    gemm = ex.compile_gemm(n, k, max_ctas=torch.cuda.get_device_properties(0).multi_processor_count)
    sw = torch.full((n,), 1.0, device="cuda")
    zero_bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty((1, m, n), device="cuda", dtype=torch.bfloat16)
    sfa = {id(o): o["xs"].view(torch.uint8) for o in ops}
    sfb = {id(o): o["ws"].view(torch.uint8) for o in ops}
    lin = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
    mods = []
    for o in ops:
        with torch.no_grad():
            lin.weight.copy_(o["w"])
        mods.append(ex.build(lin))
    mod_of = {id(o): mods[i] for i, o in enumerate(ops)}
    bw, swz = F.ScalingType.BlockWise1x16, F.SwizzleType.SWIZZLE_32_4_4

    def nvfp4(o, dtype):
        return F.scaled_mm(
            o["xq"].view(FP4),
            o["wq"].view(FP4).t(),
            scale_a=o["xs"],
            scale_recipe_a=bw,
            scale_b=o["ws"],
            scale_recipe_b=bw,
            swizzle_a=swz,
            swizzle_b=swz,
            output_dtype=dtype,
        )

    return {
        "bf16 torch.mm": lambda o: torch.mm(o["x"], o["w"].t()),
        "FP8 tensor-wise _scaled_mm": lambda o: torch._scaled_mm(
            o["x8"], o["w8"].t(), scale_a=o["one"], scale_b=o["one"], out_dtype=torch.bfloat16
        ),
        "NVFP4 F.scaled_mm, bf16 out": lambda o: nvfp4(o, torch.bfloat16),
        "NVFP4 F.scaled_mm, fp32 out": lambda o: nvfp4(o, torch.float32),
        "NVFP4 CuTe GEMM (A quantised)": lambda o: gemm(
            o["xq"].unsqueeze(0),
            o["wq"].unsqueeze(0),
            sfa[id(o)],
            sfb[id(o)],
            out,
            o["outer"],
            sw,
            zero_bias,
        ),
        "NVFP4 CuTe quantiser + GEMM (module)": lambda o: mod_of[id(o)](o["x"]),
    }


def _eager_us(fn, reps=20):
    """us per call of ``fn`` launched eagerly (paths a CUDA graph cannot capture)."""
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / reps


def bench(ex):
    print("\n## timing (CUDA graphs, best of 7 replays)")
    shapes = [
        (4096, 4096, 4096),
        (352, 4096, 1024),
        (352, 1024, 4096),
        (352, 2048, 1024),
        (352, 1024, 2048),
        (352, 8192, 1024),
    ]
    for m, n, k in shapes:
        flops = 2 * m * n * k
        weight_bytes = n * k * 2  # bf16: the largest variant
        copies = 1 + (2 * L2_BYTES) // max(weight_bytes, 1)
        cold_copies = min(copies, 24)
        for label, nset, reps in (("warm", 1, 20), ("cold", cold_copies, 2)):
            if m == 4096 and label == "cold":
                continue  # compute bound: warm only
            ops = _operands(m, n, k, nset)
            paths = _paths(ex, m, n, k, ops)
            print(f"\nM {m} N {n} K {k} ({label} L2, {nset} operand set(s))")
            base = None
            for name, fn in paths.items():
                us = _graph_us([lambda o=o, fn=fn: fn(o) for o in ops], reps)
                if name == "FP8 tensor-wise _scaled_mm":
                    base = us
                speed = f", {base / us:.2f}x FP8" if base else ""
                print(f"  {name:40s} {us:8.2f} us  {flops / us / 1e6:7.1f} TFLOP/s{speed}")
            if label == "warm":  # torch ops with host syncs: eager, informative only
                o = ops[0]
                ws, wt = quant.quantize_fp4(o["w"])[1], float(o["wt"])
                us = _eager_us(lambda o=o: quant.fp4_w4a4_linear(o["x"], o["wq"], ws, wt))
                print(f"  {'quant.fp4_w4a4_linear (eager reference)':40s} {us:8.2f} us")
            del ops, paths
            torch.cuda.empty_cache()
    # the quantiser alone
    print("\n## the CuTe quantiser alone (bf16 [M, K] -> codes, swizzled scales, outer)")
    for m, k in ((352, 1024), (352, 4096), (4096, 4096)):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        f = ex.compile_quant(k)
        q = torch.empty(m, k // 2, device="cuda", dtype=torch.uint8)
        s = torch.empty(ex.swizzled_scale_bytes(m, k), device="cuda", dtype=torch.uint8)
        o = torch.empty(m, device="cuda")
        us = _graph_us([lambda: f(x, q, s, o)], 20)
        gbs = (m * k * 2 + m * k // 2 + m * k // 16) / us / 1e3
        ref = _eager_us(lambda x=x: quant.quantize_fp4_activations(x))  # not graph-capturable
        print(
            f"  M {m:5d} K {k:5d}: {us:7.2f} us ({gbs:6.1f} GB/s); torch reference (eager) "
            f"{ref:8.2f} us"
        )


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ex = load_example()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}, sm_{props.major}{props.minor}, torch {torch.__version__}")
    if which in ("check", "all"):
        check(ex)
    if which in ("bench", "all"):
        bench(ex)


if __name__ == "__main__":
    main()
