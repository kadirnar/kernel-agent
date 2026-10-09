"""The FP4 producers (#233 follow-up 1, ``examples/triton_fp4_producers.py``): RMSNorm and
``silu(gate) * up`` writing NVFP4 / MXFP4 codes, block scales (row-major or cuBLASLt's
swizzled layout) and the per-token outer scale in their epilogue.

CPU, deterministic: the example's Triton kernels themselves, run by Triton's interpreter
(``TRITON_INTERPRET=1``, in a subprocess: the interpreter patches ``triton.language``), bit for
bit ``quant.quantize_fp4_activations`` of their own output on rows at scales from 1e-30 to
1e30 (subnormal and zero block scales, zero blocks and rows, bf16 subnormals), swizzled scales
= ``swizzle_fp4_scales``, MXFP4, and the opt-in per-token bias factor against
``quant.fp4_bias_correction``; the GPUs it declares (the GEMMs: sm_100+; the producers compile
from sm_75); the module's reference path and its scale-rule hook through the evaluator.
GPU (``-m gpu``, any CUDA GPU: no FP4 tensor cores needed for the producers): the same bit for
bit on the hardware, the W4A4 MLP against the reference chain and through the evaluator."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

from kernel_agent import gpu_arch, selftest
from kernel_agent.agent import prompts
from kernel_agent.kernels import compare, quant, scale_guard
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import capture_calls

EXAMPLE = prompts.EXAMPLES_DIR / "triton_fp4_producers.py"
W4A4, FP4A = "fp4_w4a4", "near-lossless-fp4a"


def _load():
    spec = importlib.util.spec_from_file_location("fp4_producers_test", EXAMPLE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The checks, run against the example's kernels on ``sys.argv[2]`` ("cpu" under the
#: interpreter, "cuda" on a GPU): a JSON list of what differs (empty: all bit for bit).
CHECKS = r"""
import importlib.util, json, sys
import torch
from kernel_agent.kernels import quant

spec = importlib.util.spec_from_file_location("fp4_producers_checks", sys.argv[1])
prod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prod)
dev = sys.argv[2]
# Triton's interpreter converts bf16 with numpy: subnormal bf16 inputs and the bf16 rounding of
# a producer's output are the hardware's only on a GPU (checked there)
hardware = dev != "cpu"
bad = []


def rows(m, k, seed):
    # rows at scales from 1e-30 to 1e30: subnormal and zero block scales, a zero block, an
    # outlier, a zero row (on a GPU: a row of bf16 subnormals)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=gen) * torch.logspace(-30, 30, m)[:, None]
    x[0, :16] = 0
    if m > 3:
        x[1, 5] = 1e3
        x[2] = 0
        if hardware:
            x[3] = torch.randn(k, generator=gen) * 1e-39
    return x.to(torch.bfloat16).to(dev)


def same(name, got, want):
    for i, (a, b) in enumerate(zip(got, want)):
        a, b = a.cpu(), b.cpu()
        if a.dtype != torch.float32:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        if not torch.equal(a, b):
            bad.append(f"{name}: output {i} differs in {int((a != b).sum())} places")


def factor(name, got, x, ref):
    c = quant.fp4_bias_correction(x.cpu(), *[t.cpu() for t in ref])
    want = ref[2].cpu() * c
    if not torch.allclose(got.cpu(), want, rtol=1e-5, atol=0):
        bad.append(f"{name}: bias factor off by {float(((got.cpu() - want) / want).abs().max())}")


for m, k in ((1, 16), (5, 48), (130, 64), (40, 256)):
    x = rows(m, k, m + k)
    ref = quant.quantize_fp4_activations(x.cpu(), "nvfp4")
    same(f"quantize_rows {m}x{k}", prod.quantize_rows(x), ref)
    q, s, o, e = prod.quantize_rows(x, swizzle=True, unbiased=True)
    same(f"swizzled {m}x{k}", (q, s, o), (ref[0], quant.swizzle_fp4_scales(ref[1]), ref[2]))
    factor(f"quantize_rows {m}x{k}", e, x, ref)
    if k % 32 == 0:
        mx = quant.quantize_fp4_activations(x.cpu(), "mxfp4")
        same(f"mxfp4 {m}x{k}", prod.quantize_rows(x, mx=True), mx)
        q, s, o = prod.quantize_rows(x, mx=True, swizzle=True)
        same(f"mxfp4 swizzled {m}x{k}", (q, s), (mx[0], quant.swizzle_fp4_scales(mx[1])))
    # the producers, on their own bf16 output (out=)
    gen = torch.Generator().manual_seed(k)
    xs = (torch.randn(m, k, generator=gen) * 3).to(torch.bfloat16).to(dev)
    w = (torch.randn(k, generator=gen) * 0.1 + 1).to(torch.bfloat16).to(dev)
    out = torch.empty_like(xs)
    q, s, o, e = prod.rmsnorm_fp4(xs, w, 1e-6, out=out, unbiased=True)
    ref = quant.quantize_fp4_activations(out.cpu())
    same(f"rmsnorm_fp4 {m}x{k}", (q, s, o), ref)
    factor(f"rmsnorm_fp4 {m}x{k}", e, out, ref)
    eager = (xs.float() * torch.rsqrt(xs.float().pow(2).mean(-1, keepdim=True) + 1e-6))
    eager = w * eager.to(torch.bfloat16)
    if hardware and float((out == eager).float().mean()) < 0.99:
        bad.append(f"rmsnorm_fp4 {m}x{k}: bf16 output differs from eager's in > 1 %")
    if k % 32 == 0:
        same(f"rmsnorm mxfp4 {m}x{k}", prod.rmsnorm_fp4(xs, w, 1e-6, mx=True),
             quant.quantize_fp4_activations(out.cpu(), "mxfp4"))
    gu = (torch.randn(m, 2 * k, generator=gen) * 2).to(torch.bfloat16).to(dev)
    out = torch.empty(m, k, dtype=torch.bfloat16, device=dev)
    g, u = gu[:, :k], gu[:, k:]
    merged = prod.silu_mul_fp4(gu, out=out)
    same(f"silu_mul_fp4 {m}x{k}", merged, quant.quantize_fp4_activations(out.cpu()))
    separate = prod.silu_mul_fp4(g.contiguous(), u.contiguous())
    same(f"silu_mul_fp4 separate {m}x{k}", separate, merged)
    eager = torch.nn.functional.silu(g) * u
    if hardware and float((out == eager).float().mean()) < 0.99:
        bad.append(f"silu_mul_fp4 {m}x{k}: bf16 output differs from eager's in > 1 %")
codes, scales, outer = prod.quantize_activations(rows(9, 64, 1))
assert codes.shape == (9, 32) and scales.shape == (9, 4) and outer.shape == (9,)
print(json.dumps(bad))
"""


def _checks(device: str, env: dict[str, str]) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-c", CHECKS, str(EXAMPLE), device],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_producers_are_the_reference_bit_for_bit_in_the_interpreter():
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    assert _checks("cpu", env) == []


def test_the_example_declares_where_it_runs_and_compiles():
    runs, why = gpu_arch.example_requirement(EXAMPLE)
    assert runs == "sm_100+" and "block-scaled FP4" in why and "sm_86" in why
    assert gpu_arch.example_compiles(EXAMPLE) == "sm_75+"
    for cap in ((8, 6), (8, 9), (9, 0)):
        assert "needs sm_100+" in gpu_arch.example_skip(EXAMPLE, cap)
    assert gpu_arch.example_skip(EXAMPLE, (12, 0)) is None
    assert gpu_arch.example_skip(EXAMPLE, (10, 0)) is None
    assert EXAMPLE.name in selftest.W4A4_PRODUCER_EXAMPLES  # doctor --smoke runs it there
    source = EXAMPLE.read_text()
    code = "\n".join(line.split("#")[0] for line in source.split('"""', 2)[2].splitlines())
    assert "float8e4nv" not in code  # no FP8 type in the kernels: e4m3 bits by integer math


def _mlp(hidden: int = 64, inter: int = 96, seed: int = 0) -> selftest.GatedMLP:
    torch.manual_seed(seed)
    module = selftest.GatedMLP(hidden, inter).to(torch.bfloat16)
    with torch.no_grad():
        for lin in (module.gate_proj, module.up_proj, module.down_proj):
            lin.weight.normal_(0.0, lin.in_features**-0.5)
    return module


@pytest.mark.parametrize("unbiased", [False, True])
def test_the_module_on_the_cpu_is_the_reference_math(unbiased):
    example = _load()
    reference = _mlp()
    module = example.build(reference, unbiased=unbiased)
    assert type(module).__name__ == "Fp4ProducerMLP" and module.unbiased is unbiased
    assert set(module.quant_error) == {"gate_up", "down"}
    x = torch.randn(3, 5, 64, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    w_gu = torch.cat([reference.gate_proj.weight, reference.up_proj.weight])
    gu = quant.fp4_w4a4_linear(x, *quant.quantize_fp4(w_gu, unbiased=unbiased), unbiased=unbiased)
    g, u = gu.chunk(2, dim=-1)
    d = quant.quantize_fp4(reference.down_proj.weight, unbiased=unbiased)
    want = quant.fp4_w4a4_linear(torch.nn.functional.silu(g) * u, *d, unbiased=unbiased)
    with torch.no_grad():
        got = module(x)
    assert got.shape == x.shape and torch.equal(got, want)
    # not a gated SiLU MLP, or a bias: the reference comes back
    reference.gate_proj.bias = torch.nn.Parameter(torch.zeros(96, dtype=torch.bfloat16))
    assert example.build(reference) is reference
    assert example.build(torch.nn.Linear(64, 64)).__class__ is torch.nn.Linear


def test_the_scale_rule_guard_and_the_evaluator_check_its_hook(tmp_path, monkeypatch):
    example = _load()
    x = (torch.randn(2, 11, 256) * torch.logspace(-3, 3, 256)).to(torch.bfloat16)
    report = scale_guard.check(example, [{"args": (x,), "kwargs": {}, "count": 4}], W4A4)
    assert report["ok"] and report["checked"] and report["stress"]["saturated"] == 0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    module = _mlp(256, 512)
    calls = [
        ((torch.randn(16, 11, 256, dtype=torch.bfloat16),), {}, 540),
        ((torch.randn(4, 3, 256, dtype=torch.bfloat16),), {}, 0),
    ]
    capture = tmp_path / "mlp.pt"
    capture_calls(module, calls, capture, tier=FP4A, precision=W4A4)
    try:
        result = evaluate(capture, EXAMPLE, device="cpu")
        assert result["correct"] and result["tolerance_tier"] == FP4A, result
        assert result["scale_rule"]["ok"] and result["scale_rule"]["checked"]
    finally:
        compare.TIER = compare.EXACT_TIER


# ------------------------------------------------------------------ GPU (any CUDA GPU)


def _gpu() -> tuple[int, int]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    return torch.cuda.get_device_capability()


@pytest.mark.gpu
def test_the_producers_on_the_gpu_are_the_reference_bit_for_bit():
    _gpu()
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    assert _checks("cuda", env) == []


@pytest.mark.gpu
@pytest.mark.parametrize("unbiased", [False, True])
def test_the_w4a4_producer_mlp_on_the_gpu(unbiased):
    capability = _gpu()
    example = _load()
    reference = _mlp(1024, 3072).cuda()
    module = example.build(reference, unbiased=unbiased)
    x = torch.randn(4, 88, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        got = module(x)
        gu_codes, gu_scales, gu_ts, d_codes, d_scales, d_ts = module.reference_args
        gu = quant.fp4_w4a4_linear(x, gu_codes, gu_scales, gu_ts, unbiased=unbiased)
        g, u = gu.chunk(2, dim=-1)
        silu = torch.nn.functional.silu(g) * u
        want = quant.fp4_w4a4_linear(silu, d_codes, d_scales, d_ts, unbiased=unbiased)
    if unbiased:  # the producers' factor is the reference's to fp32 rounding
        rel = float((got.float() - want.float()).norm() / want.float().norm())
        assert rel < 1e-3, rel
    else:  # the same codes, the same GEMM math (F.scaled_mm on sm_100+, fp32 elsewhere)
        assert torch.equal(got, want), capability


@pytest.mark.gpu
def test_the_producer_mlp_through_the_evaluator(tmp_path):
    # doctor --smoke's check (it runs it on the GPUs of the example's ARCHS): near-lossless-fp4a
    # passes with the scale-rule guard checking the hook, the 8-bit tier rejects it
    _gpu()
    examples = selftest.W4A4_PRODUCER_EXAMPLES
    capture = selftest.make_mlp_capture
    assert selftest.smoke_fp4(tmp_path, True, precision=W4A4, examples=examples, capture=capture)
