"""FP8 W8A8 precision class (#91): per-token activation quantisation helpers, the
``fp8_w8a8`` precision through planner, orchestrator, engineer prompt, capture and speed of
light, the near-lossless tier on a W8A8 candidate (passes) and broken scales (fail), and
(``gpu``) the bundled Triton W8A8 example on the evaluator."""

import asyncio
import copy
import math

import pytest
import torch
from torch import nn

from kernel_agent import charts, dryrun, orchestrator, selftest
from kernel_agent.agent import prompts
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels import compare, quant, roofline
from kernel_agent.kernels.evaluate import capture_precision, evaluate, run_evaluation
from kernel_agent.profiling.capture import capture_calls, load_capture
from kernel_agent.workspace import read_json

# ---------------------------------------------------------------- helpers


def test_quantize_fp8_activations_per_token():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 5, 256, generator=gen).to(torch.bfloat16)
    x[0, 1] = 0  # a token of zeros
    x[1, 2, 7] = 300.0  # an outlier token
    q, scale = quant.quantize_fp8_activations(x)
    assert q.shape == (10, 256) and q.dtype == torch.float8_e4m3fn and q.is_contiguous()
    assert scale.shape == (10,) and scale.dtype == torch.float32
    rows = x.float().reshape(10, 256)
    amax = rows.abs().amax(dim=1)
    assert scale[1] == 1.0 and not q[1].float().any()
    nonzero = amax > 0
    assert torch.allclose(scale[nonzero], amax[nonzero] / 448.0)
    assert torch.equal(q.float().abs().amax(dim=1)[nonzero], torch.full((9,), 448.0))
    deq = q.float() * scale[:, None]
    rel = (deq - rows).norm(dim=1) / rows.norm(dim=1).clamp_min(1e-30)
    assert float(rel[nonzero].max()) < 0.05  # e4m3 per token, other tokens unaffected
    with pytest.raises(ValueError, match="unsupported"):
        quant.quantize_fp8_activations(x, torch.bfloat16)


def test_fp8_w8a8_linear_and_error_report():
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(256, 512, generator=gen) * 512**-0.5
    bias = torch.randn(256, generator=gen)
    x = torch.randn(3, 7, 512, generator=gen)
    q, scale = quant.quantize_fp8(w)
    y = quant.fp8_w8a8_linear(x, q, scale, bias)
    xq, xs = quant.quantize_fp8_activations(x)
    expected = (xq.float() * xs[:, None]) @ (q.float() * scale[:, None]).T + bias
    assert y.shape == (3, 7, 256) and y.dtype == x.dtype
    assert torch.allclose(y.reshape(21, 256), expected)
    ref = x @ w.T + bias
    rel = float((y - ref).norm() / ref.norm())
    assert 0.01 < rel < 0.05  # weights and activations in e4m3: ~sqrt(2) x weight-only

    report = quant.fp8_w8a8_error(w, q, scale, x)
    assert report["activations"] == "e4m3 per token"
    assert report["rel_l2"] == quant.fp8_error(w, q, scale)["rel_l2"]
    assert 0.015 < report["activation_rel_l2"] < 0.035 and report["activation_underflow"] < 1e-3
    assert 3 < report["activation_crest"] < 6  # Gaussian tokens
    assert 0.02 < report["output_rel_l2"] < 0.05 and report["output_cosine"] > 0.998
    assert abs(report["output_norm_ratio"] - 1) < 0.01

    spiky = x.clone()
    spiky[0, 0, 0] = 1e5  # an outlier channel squeezes the rest of its token (crest √512)
    worse = quant.fp8_w8a8_error(w, q, scale, spiky)
    assert worse["activation_crest"] > 20 and worse["activation_underflow"] > 0.005


# ---------------------------------------------------------------- policy and prompts


def test_precision_class_and_tier():
    assert "fp8_w8a8" in compare.REDUCED_PRECISIONS and "fp8_w8a8" in compare.PRECISIONS
    enum = prompts.PLAN_SCHEMA["properties"]["targets"]["items"]["properties"]["precision"]
    assert "fp8_w8a8" in enum["enum"]
    assert compare.tier_for("near-lossless", "fp8_w8a8") == compare.NEAR_LOSSLESS_TIER
    assert compare.tier_for("exact", "fp8_w8a8") == compare.EXACT_TIER
    target = {"id": "dit", "precision": "fp8_w8a8", "precision_why": "M=352, compute bound"}
    assert "needs --quality near-lossless" in orchestrator._precision(dict(target), "exact")
    assert orchestrator._precision(target, "near-lossless") is None
    assert target == {"id": "dit", "precision": "fp8_w8a8", "precision_why": "M=352, compute bound"}


def test_planner_policy_steers_compute_bound_gemms_to_w8a8():
    near = prompts.precision_policy("near-lossless")
    assert '`precision: "fp8_w8a8"`' in near and '`precision: "fp8_weights"`' in near
    assert "compute-bound" in near and "FLOP-bound" in near
    assert "compute-bound" not in near.split("unset (exact)")[1]  # no longer left exact
    exact = prompts.precision_policy("exact")
    assert "`fp8_w8a8`" in exact and "is refused" in exact


def test_plan_keeps_a_w8a8_target_only_in_near_lossless(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", lambda: dryrun.SimToolchain())
    monkeypatch.setattr(charts, "available", lambda: False)
    plan = {
        "analysis": "the DiT is compute bound",
        "targets": [
            {
                "id": "dit",
                "module_class": "Qwen3MLP",
                "why": "w",
                "approach": "W8A8 e4m3 GEMMs",
                "backends": ["triton"],
                "precision": "fp8_w8a8",
                "precision_why": "M=352 GEMMs, 80 TFLOP per run, compute bound",
            }
        ],
        "transforms": [],
    }

    async def planner(name, *, system_append, result=None, **kwargs):
        return AgentResult(name=name, structured=copy.deepcopy(plan))

    for quality, kept in (("exact", []), ("near-lossless", ["dit"])):
        cfg = OptimizeConfig(
            model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path / quality, quality=quality
        )
        orch = orchestrator.Orchestrator(dryrun.create_run(cfg), cfg)
        orch.agent_runner = planner
        asyncio.run(orch.plan())
        targets = read_json(orch.run.plan_json)["targets"]
        assert [t["id"] for t in targets] == kept
    assert targets[0]["precision"] == "fp8_w8a8" and "compute bound" in targets[0]["precision_why"]


def test_engineer_prompt_states_the_w8a8_contract():
    target = {
        "id": "dit",
        "module_class": "Linear",
        "why": "w",
        "approach": "a",
        "backends": ["triton"],
        "precision": "fp8_w8a8",
        "precision_why": "M=352, compute bound",
    }
    cases = [{"signature": "a0[352, 1024]:bfloat16", "count": 540}]
    near = {"cases": cases, "tier": "near-lossless", "precision": "fp8_w8a8"}
    text = prompts.engineer_prompt(target, near, ["triton"], "python", "toolchain", 10, None)
    assert prompts.reduced_precision(target, near) == "fp8_w8a8"
    assert "# Precision: `fp8_w8a8`" in text and "(planner: M=352, compute bound)" in text
    for needle in (
        "FP8 W8A8",
        "per token on every call",
        "quantize_fp8, fp8_w8a8_linear, fp8_w8a8_error",
        "accumulate in fp32",
        "triton_fp8_w8a8_gemm.py",
        "# Low-precision weights",
        "## FP8 W8A8",
        "torch._scaled_mm",
    ):
        assert needle in text, needle
    assert "FP8 weight-only:" not in text


def test_knowledge_and_example_exist():
    text = prompts.knowledge("low_precision.md")
    for needle in ("fp8_w8a8", "per token", "_scaled_mm", "SmoothQuant", "20261006-004718"):
        assert needle in text, needle
    for name in selftest.W8A8_EXAMPLES:
        source = (prompts.EXAMPLES_DIR / name).read_text()
        for needle in ("def build(", "quantize_fp8", "custom_op", "register_fake", "float8e4nv"):
            assert needle in source, needle


# ---------------------------------------------------------------- the tier on a W8A8 candidate

CANDIDATE = """import torch
from kernel_agent.kernels.quant import fp8_w8a8_linear, quantize_fp8_activations, quantize_fp8


class W8A8(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.ref = ref
        self.q1, self.s1 = quantize_fp8(ref.up.weight)
        self.q2, self.s2 = quantize_fp8(ref.down.weight)
        self.s1 = self.s1 * {weight_scale}

    def mm(self, x, q, s):
        {mm}

    def forward(self, x):
        return self.mm(torch.nn.functional.silu(self.mm(x, self.q1, self.s1)), self.q2, self.s2)


def build(reference):
    return W8A8(reference)
"""
GOOD_MM = "return fp8_w8a8_linear(x, q, s)"
# the first token's scale for every token: a scale broadcast bug
ROW0_MM = """xq, xs = quantize_fp8_activations(x)
        y = (xq.float() * xs[:1, None]) @ (q.float() * s[:, None]).T
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)"""


class Mlp(nn.Module):
    def __init__(self, hidden: int = 256, inter: int = 1024) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(nn.functional.silu(self.up(x)))


@pytest.fixture
def w8a8_captures(tmp_path, monkeypatch):
    """A bf16 MLP captured twice: ``precision: fp8_w8a8`` (near-lossless) and exact."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    mlp = Mlp().to(torch.bfloat16)
    with torch.no_grad():
        mlp.up.weight.normal_(0, 256**-0.5)
        mlp.down.weight.normal_(0, 1024**-0.5)
    calls = [
        ((torch.randn(32, 11, 256, dtype=torch.bfloat16),), {}, 540),
        ((torch.randn(16, 11, 256, dtype=torch.bfloat16),), {}, 0),
    ]
    near, exact = tmp_path / "near.pt", tmp_path / "exact.pt"
    capture_calls(mlp, calls, near, tier="near-lossless", precision="fp8_w8a8")
    capture_calls(mlp, calls, exact)
    return near, exact


def _candidate(tmp_path, name, *, mm=GOOD_MM, weight_scale=1.0):
    path = tmp_path / f"{name}.py"
    path.write_text(CANDIDATE.format(mm=mm, weight_scale=weight_scale))
    return path


def test_w8a8_candidate_passes_the_near_lossless_tier(tmp_path, w8a8_captures):
    near, exact = w8a8_captures
    cap = load_capture(near)
    assert cap["tier"] == "near-lossless" and capture_precision(cap) == "fp8_w8a8"
    good = _candidate(tmp_path, "good")
    result = evaluate(near, good, device="cpu")
    assert result["correct"] and result["tolerance_tier"] == "near-lossless", result
    for case in result["cases"]:
        assert 0.02 < case["max_rel_l2"] < 0.06 and case["min_cosine"] > 0.998
    rejected = evaluate(exact, good, device="cpu")  # the exact tier still rejects W8A8
    assert rejected["status"] == "incorrect", rejected


@pytest.mark.parametrize(
    ("label", "kwargs", "why"),
    [
        ("scale_x1.05", {"weight_scale": 1.05}, "norm x1.05"),
        ("scale_row0", {"mm": ROW0_MM}, "relative L2"),
    ],
)
def test_w8a8_candidate_with_a_broken_scale_fails(tmp_path, w8a8_captures, label, kwargs, why):
    near, _ = w8a8_captures
    result = evaluate(near, _candidate(tmp_path, label, **kwargs), device="cpu")
    assert result["status"] == "incorrect", result
    errors = " ".join(f["error"] for c in result["cases"] for f in c.get("failures", []))
    assert why in errors, errors


# ---------------------------------------------------------------- speed of light


class _Gemm(nn.Module):
    """A weight GEMM (W8A8 in an fp8_w8a8 target) and an activation x activation one."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(256, 512, bias=False).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        return y @ y.transpose(-1, -2)


def test_speed_of_light_counts_w8a8_gemms_at_the_fp8_peak():
    torch.manual_seed(0)
    module, x = _Gemm(), torch.randn(352, 256, dtype=torch.bfloat16)
    exact = roofline.count_case(module, (x,), {})
    weights = roofline.count_case(module, (x,), {}, precision="fp8_weights")
    w8a8 = roofline.count_case(module, (x,), {}, precision="fp8_w8a8")
    gemm, attn = 2 * 352 * 256 * 512, 2 * 352 * 352 * 512
    assert exact.flops == weights.flops == {"bfloat16": gemm + attn}
    assert w8a8.flops == {roofline.FP8: gemm, "bfloat16": attn}
    assert w8a8.min_bytes == weights.min_bytes < exact.min_bytes  # weights at 1 byte

    peaks = {"dram_gbps": 1e9, "tflops": {"bfloat16": 100.0, roofline.FP8: 200.0}}
    full = roofline.sol_time(exact, peaks)
    fp8 = roofline.sol_time(w8a8, peaks)
    assert fp8["bound"] == full["bound"] == "compute" and "peak_missing" not in fp8
    assert fp8["sol_ms"] == pytest.approx((gemm / 200 + attn / 100) / 1e9)
    assert fp8["sol_ms"] < full["sol_ms"]

    # no FP8 peak (none on this GPU, an old cache): the ceiling is not one, no SOL advice
    old = {"dram_gbps": 1e9, "tflops": {"bfloat16": 100.0}}
    assert roofline.sol_time(w8a8, old)["peak_missing"] == roofline.FP8
    result = {"correct": True, "cases": [{"new_ms": 0.002, "ref_ms": 0.004, "calls_per_run": 9}]}
    roofline.apply_sol(result, [w8a8], old)
    assert result["cases"][0]["sol_unreliable"] and result["sol_unreliable"]
    assert "tflops_unavailable" in result["sol_note"] and roofline.sol_signal(result) is None
    fine = {"correct": True, "cases": [{"new_ms": 0.002, "ref_ms": 0.004, "calls_per_run": 9}]}
    roofline.apply_sol(fine, [w8a8], peaks)
    assert "sol_note" not in fine and roofline.sol_signal(fine) is not None


# ---------------------------------------------------------------- GPU: the example


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(selftest.W8A8_EXAMPLES))
def test_w8a8_example_passes_the_reduced_tier_and_fails_exact(name, tmp_path):
    from kernel_agent import toolchain

    if not selftest.w8a8_supported(toolchain.setup()):
        pytest.skip("needs the triton backend on sm_89+")
    k, n, calls = selftest.W8A8_EXAMPLES[name]
    near = selftest.make_linear_capture(
        tmp_path / "near.pt", k, n, calls, tier="near-lossless", precision="fp8_w8a8"
    )
    exact = selftest.make_linear_capture(tmp_path / "exact.pt", k, n, calls)
    example = prompts.EXAMPLES_DIR / name

    result = run_evaluation(near, example)  # subprocess + the checks outside it
    assert result["status"] == "ok" and result["correct"], result
    assert result["tolerance_tier"] == "near-lossless" and result["parent_check"] == "ok"
    assert result["speedup"] > 1.0  # FP8 tensor cores on a compute-bound GEMM
    for case, (shape, _) in zip(result["cases"], calls, strict=True):
        assert 0.02 < case["max_rel_l2"] < 0.05 and case["min_cosine"] > 0.998
        rows = math.prod(shape)
        if "min_bytes" in case:  # with measured peaks: the weights count at 1 byte
            assert case["min_bytes"] == n * k + 4 * n + rows * (k + n) * 2

    rejected = evaluate(exact, example)
    assert rejected["status"] == "incorrect", rejected
