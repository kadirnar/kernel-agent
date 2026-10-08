"""Low-precision weights (#72): FP8 quantisation helpers, the precision policy tied to
``--quality`` (planner, orchestrator, engineer prompt, capture, library), the
precision-aware speed of light, and (``gpu``) the bundled FP8 examples on the evaluator:
correct in the near-lossless tier, rejected by the exact tier."""

import asyncio
import math

import pytest
import torch
from torch import nn

from kernel_agent import charts, dryrun, library, orchestrator, selftest
from kernel_agent.agent import prompts
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels import compare, quant, roofline
from kernel_agent.kernels.evaluate import capture_precision, evaluate, run_evaluation
from kernel_agent.profiling.capture import capture_calls, load_capture
from kernel_agent.workspace import read_json, write_json

# ---------------------------------------------------------------- quantisation helpers


def test_quantize_fp8_per_output_channel():
    gen = torch.Generator().manual_seed(0)
    w = (torch.randn(64, 512, generator=gen) / 20).to(torch.bfloat16)
    w[3] = 0  # a row of zeros
    w[5, 7] = 2.0  # an outlier channel
    q, scale = quant.quantize_fp8(w)
    assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape and q.is_contiguous()
    assert scale.dtype == torch.float32 and scale.shape == (64,)
    amax = w.float().abs().amax(dim=1)
    nonzero = amax > 0
    assert torch.allclose(scale[nonzero], amax[nonzero] / 448.0)
    assert scale[3] == 1.0 and not q[3].float().any()
    # the largest element of every row maps to the largest finite e4m3 value
    assert torch.equal(q.float().abs().amax(dim=1)[nonzero], torch.full((63,), 448.0))
    deq = quant.dequantize_fp8(q, scale, torch.float32)
    assert torch.equal(deq, q.float() * scale[:, None])
    assert quant.dequantize_fp8(q, scale).dtype == torch.bfloat16
    rel = float((deq - w.float()).norm() / w.float().norm())
    assert 0.005 < rel < 0.04  # e4m3: 3 mantissa bits

    q5, s5 = quant.quantize_fp8(w, torch.float8_e5m2)
    assert q5.dtype == torch.float8_e5m2 and torch.allclose(s5[nonzero], amax[nonzero] / 57344.0)
    with pytest.raises(ValueError, match="2-D"):
        quant.quantize_fp8(w[0])
    bad = w.clone()
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        quant.quantize_fp8(bad)
    with pytest.raises(ValueError, match="unsupported"):
        quant.quantize_fp8(w, torch.bfloat16)


def test_fp8_error_report():
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(128, 1024, generator=gen) * 1024**-0.5
    q, scale = quant.quantize_fp8(w)
    report = quant.fp8_error(w, q, scale)
    deq = quant.dequantize_fp8(q, scale, torch.float32)
    assert report["format"] == "float8_e4m3fn"
    assert report["rel_l2"] == pytest.approx(float((deq - w).norm() / w.norm()), rel=1e-3)
    assert 0.02 < report["rel_l2"] < 0.035 and report["worst_channel_rel_l2"] >= report["rel_l2"]
    assert report["underflow"] < 1e-3 and 3 < report["crest"] < 6
    assert report["bytes"] == {"before": 128 * 1024 * 4, "after": 128 * 1024 + 128 * 4}
    assert "output_rel_l2" not in report

    x = torch.randn(4, 1024, generator=gen)
    with_x = quant.fp8_error(w, q, scale, x)
    assert with_x["output_rel_l2"] == pytest.approx(report["rel_l2"], rel=0.3)
    assert 0.999 < with_x["output_cosine"] < 1.0

    spiky = w.clone()
    spiky[0, 0] = 1e3  # one outlier sets its row's scale: small weights flush to zero
    report = quant.fp8_error(spiky, *quant.quantize_fp8(spiky))
    assert report["underflow"] > 1e-4 and report["crest"] > 30


# ---------------------------------------------------------------- policy


def test_precisions_and_tiers():
    assert set(
        prompts.PLAN_SCHEMA["properties"]["targets"]["items"]["properties"]["precision"]["enum"]
    ) == set(compare.PRECISIONS)
    assert compare.tier_for("near-lossless", "fp8_weights") == compare.NEAR_LOSSLESS_TIER
    assert compare.tier_for("near-lossless", "reduced") == compare.NEAR_LOSSLESS_TIER
    for quality, precision in (("exact", "fp8_weights"), ("near-lossless", "exact"), (None, None)):
        assert compare.tier_for(quality, precision) == compare.EXACT_TIER


def test_orchestrator_precision_policy():
    def check(precision, quality, why="decode GEMVs are memory bound"):
        target = {"id": "mlp", "precision": precision, "precision_why": why}
        return orchestrator._precision(target, quality), target

    for precision in ("fp8_weights", "reduced"):
        problem, target = check(precision, "exact")
        assert "needs --quality near-lossless" in problem and "exact" in problem
        problem, target = check(precision, "near-lossless")
        assert problem is None and target["precision"] == precision
        assert target["precision_why"] == "decode GEMVs are memory bound"
    for precision in ("exact", None, ""):
        for quality in ("exact", "near-lossless"):
            problem, target = check(precision, quality)
            assert problem is None and "precision" not in target and "precision_why" not in target
    assert "unknown precision" in check("int4", "near-lossless")[0]
    problem, target = check("fp8_weights", "near-lossless", why=None)
    assert problem is None and "precision_why" not in target


def _planned_targets(tmp_path, monkeypatch, quality):
    """``Orchestrator.plan`` on a simulated run with a fake planner that marks the MLP
    target ``fp8_weights``; returns (plan.json targets, the planner's system prompt)."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", lambda: dryrun.SimToolchain())
    monkeypatch.setattr(charts, "available", lambda: False)
    cfg = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path / quality, quality=quality)
    orch = orchestrator.Orchestrator(dryrun.create_run(cfg), cfg)
    seen = {}
    plan = {
        "analysis": "decode is memory bound",
        "targets": [
            {
                "id": "mlp",
                "module_class": "Qwen3MLP",
                "why": "w",
                "approach": "FP8 GEMVs",
                "backends": ["cuda"],
                "precision": "fp8_weights",
                "precision_why": "M=1 GEMVs stream the bf16 weights",
            },
            {
                "id": "norm",
                "module_class": "Qwen3RMSNorm",
                "why": "w",
                "approach": "a",
                "backends": ["cuda"],
            },
        ],
        "transforms": [],
    }

    async def planner(name, *, system_append, result=None, **kwargs):
        seen["system"] = system_append
        return AgentResult(name=name, structured=plan)

    orch.agent_runner = planner
    asyncio.run(orch.plan())
    return read_json(orch.run.plan_json)["targets"], seen["system"]


def test_plan_refuses_reduced_precision_in_exact_mode(tmp_path, monkeypatch):
    targets, system = _planned_targets(tmp_path, monkeypatch, "exact")
    assert [t["id"] for t in targets] == ["norm"]  # the fp8_weights target is refused
    assert "# Precision (`--quality exact`)" in system and "is refused" in system

    targets, system = _planned_targets(tmp_path, monkeypatch, "near-lossless")
    assert [t["id"] for t in targets] == ["mlp", "norm"]
    assert targets[0]["precision"] == "fp8_weights" and targets[0]["precision_why"]
    assert "precision" not in targets[1]
    assert '`precision: "fp8_weights"`' in system and "precision_why" in system


def test_engineer_prompt_explains_the_fp8_contract():
    target = {
        "id": "mlp",
        "module_class": "Linear",
        "why": "w",
        "approach": "a",
        "backends": ["cuda"],
        "precision": "fp8_weights",
        "precision_why": "decode GEMVs",
    }
    cases = [{"signature": "a0[1, 2048]:bfloat16", "count": 28}]
    near = {"cases": cases, "tier": "near-lossless", "precision": "fp8_weights"}
    args = (["cuda"], "python", "toolchain", 10, None)
    text = prompts.engineer_prompt(target, near, *args)
    assert prompts.reduced_precision(target, near) == "fp8_weights"
    assert "# Precision: `fp8_weights`" in text and "(planner: decode GEMVs)" in text
    for needle in (
        "quantize_fp8",
        "one fp32 scale per output channel",
        "activations stay bf16",
        "accumulate in fp32",
        "max_rel_l2",
        "cuda_fp8_gemv.py",
        "cuda_fp8_skinny_gemm.py",
        "`kernel-agent:fp8-weights`",  # the skill to load (#176)
        "`kernel-agent:precision-tiers`",
    ):
        assert needle in text, needle
    # an exact capture (an exact run, or no precision in the spec): no contract, no guide
    for spec, capture in ((target, {"cases": cases}), ({**target, "precision": None}, near)):
        text = prompts.engineer_prompt(spec, capture, *args)
        assert prompts.reduced_precision(spec, capture) is None
        assert "# Precision" not in text and "kernel-agent:precision-tiers" not in text
    generic = prompts.engineer_prompt({**target, "precision": "reduced"}, near, *args)
    assert "# Precision: `reduced`" in generic and "FP8 weight-only:" not in generic


def test_knowledge_and_examples_exist():
    text = prompts.knowledge("low_precision.md")
    for needle in ("e4m3", "per output channel", "sm_120a", "NVFP4", "fp8_error"):
        assert needle in text, needle
    for name in selftest.FP8_EXAMPLES:
        source = (prompts.EXAMPLES_DIR / name).read_text()
        assert "def build(" in source and "quantize_fp8" in source and "load_inline" in source


# ---------------------------------------------------------------- capture, roofline, library


def test_capture_and_speed_of_light_count_fp8_weight_bytes(tmp_path):
    torch.manual_seed(0)
    lin = nn.Linear(256, 512, bias=False).to(torch.bfloat16)
    x = torch.randn(1, 256, dtype=torch.bfloat16)
    path = tmp_path / "lin.pt"
    capture_calls(lin, [((x,), {}, 28)], path, tier="near-lossless", precision="fp8_weights")
    cap = load_capture(path)
    assert cap["tier"] == "near-lossless" and capture_precision(cap) == "fp8_weights"
    assert capture_precision({"precision": "fp8_weights"}) is None  # exact tier: no precision
    capture_calls(lin, [((x,), {}, 28)], tmp_path / "exact.pt")
    assert capture_precision(load_capture(tmp_path / "exact.pt")) is None

    io = 256 * 2 + 512 * 2  # x read, y written
    exact = roofline.count_case(lin, (x,), {})
    fp8 = roofline.count_case(lin, (x,), {}, precision="fp8_weights")
    assert exact.min_bytes == 512 * 256 * 2 + io
    assert fp8.min_bytes == 512 * 256 + 512 * 4 + io  # codes + one fp32 scale per channel
    assert fp8.flops == exact.flops
    assert roofline.count_case(lin, (x,), {}, precision="reduced").min_bytes == exact.min_bytes
    peaks = {"dram_gbps": 800.0, "tflops": {"bfloat16": 100.0}}  # memory bound: SOL ~ bytes
    ratio = roofline.sol_time(fp8, peaks)["sol_ms"] / roofline.sol_time(exact, peaks)["sol_ms"]
    assert ratio == pytest.approx(fp8.min_bytes / exact.min_bytes) and ratio < 0.52


def test_evaluator_reports_the_numerical_error_in_the_reduced_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    lin = nn.Linear(512, 256, bias=False)
    cand = tmp_path / "fp8.py"
    cand.write_text(
        "import torch\n"
        "from kernel_agent.kernels.quant import dequantize_fp8, quantize_fp8\n\n\n"
        "class Fp8(torch.nn.Module):\n"
        "    def __init__(self, ref):\n"
        "        super().__init__()\n"
        "        self.w = dequantize_fp8(*quantize_fp8(ref.weight), torch.float32)\n\n"
        "    def forward(self, x):\n"
        "        return x @ self.w.T\n\n\n"
        "def build(reference):\n"
        "    return Fp8(reference)\n"
    )
    calls = [((torch.randn(4, 512),), {}, 1), ((torch.randn(2, 3, 512),), {}, 1)]
    near, exact = tmp_path / "near.pt", tmp_path / "exact.pt"
    capture_calls(lin, calls, near, tier="near-lossless", precision="fp8_weights")
    capture_calls(lin, calls, exact)
    result = evaluate(near, cand, device="cpu")
    assert result["correct"] and result["tolerance_tier"] == "near-lossless", result
    assert all(0.01 < c["max_rel_l2"] < 0.05 for c in result["cases"])
    rejected = evaluate(exact, cand, device="cpu")
    assert rejected["status"] == "incorrect" and "max_rel_l2" not in rejected["cases"][0]


def _entry(root, eid, precision, *, cls="Qwen3MLP"):
    path = root / "sm_120" / cls / eid
    path.mkdir(parents=True)
    (path / "kernel.py").write_text(f"def build(reference):  # {eid}\n    return reference\n")
    sig = library.signature_summary({"cases": [{"signature": "a0[1, 1024]:bfloat16"}]})
    meta = {
        "id": eid,
        "sm_arch": "sm_120",
        "module_class": cls,
        "signature": sig,
        "speedup": 1.5,
        "files": {"kernel.py": eid},
    }
    write_json(path / "entry.json", meta | ({"precision": precision} if precision else {}))


def test_library_matches_precision(tmp_path, monkeypatch):
    monkeypatch.setenv(library.ENV, str(tmp_path))
    _entry(tmp_path, "old", None)  # stored before precisions existed: exact
    _entry(tmp_path, "bf16", "exact")
    _entry(tmp_path, "fp8", "fp8_weights")
    cases = [{"signature": "a0[1, 1024]:bfloat16", "count": 28}]
    exact = {"module_class": "Qwen3MLP", "capture": {"cases": cases}}
    fp8 = {
        "module_class": "Qwen3MLP",
        "capture": {"cases": cases, "tier": "near-lossless", "precision": "fp8_weights"},
    }
    assert library.precision_of(exact["capture"]) == "exact"
    assert library.precision_of(fp8["capture"]) == "fp8_weights"
    assert {e.id for e in library.matches("sm_120", exact)} == {"old", "bf16"}
    assert {e.id for e in library.matches("sm_120", fp8)} == {"old", "bf16", "fp8"}


def test_library_stores_the_precision(tmp_path, monkeypatch):
    from test_library import agent_eval, by_target, fake_worker, make

    monkeypatch.setenv(library.ENV, str(tmp_path / "library"))
    monkeypatch.setattr(charts, "available", lambda: False)
    orch = make(tmp_path, monkeypatch, "a")
    spec_path = orch.run.target("mlp") / "spec.json"
    spec = read_json(spec_path)
    spec["precision"] = "fp8_weights"
    spec["capture"].update(tier="near-lossless", precision="fp8_weights")
    write_json(spec_path, spec)
    agent_eval(orch, "mlp", 1.8, name="v1")
    agent_eval(orch, "rmsnorm", 1.5, name="v1")
    orch.worker = fake_worker({"rmsnorm": 1.05, "mlp": 1.3})
    asyncio.run(orch.integrate())
    entries = by_target()
    assert entries["mlp"].meta["precision"] == "fp8_weights"
    assert entries["rmsnorm"].meta["precision"] == "exact"


# ---------------------------------------------------------------- GPU: the examples


def _gpu_ready():
    from kernel_agent import toolchain

    tc = toolchain.setup()
    if not selftest.fp8_supported(tc):
        pytest.skip("needs the cuda backend on sm_89+")


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(selftest.FP8_EXAMPLES))
def test_fp8_example_passes_the_reduced_tier_and_fails_exact(name, tmp_path):
    _gpu_ready()
    k, n, calls = selftest.FP8_EXAMPLES[name]
    near = selftest.make_linear_capture(
        tmp_path / "near.pt", k, n, calls, tier="near-lossless", precision="fp8_weights"
    )
    exact = selftest.make_linear_capture(tmp_path / "exact.pt", k, n, calls)
    example = prompts.EXAMPLES_DIR / name

    result = run_evaluation(near, example)  # subprocess + the checks outside it
    assert result["status"] == "ok" and result["correct"], result
    assert result["tolerance_tier"] == "near-lossless" and result["parent_check"] == "ok"
    assert result["speedup"] > 1.0  # half the weight bytes, even with a warm L2
    for case, (shape, _) in zip(result["cases"], calls, strict=True):
        assert 0.015 < case["max_rel_l2"] < 0.04 and case["min_cosine"] > 0.999
        rows = math.prod(shape)
        if "min_bytes" in case:  # with measured peaks: the weights count at 1 byte
            assert case["min_bytes"] == n * k + 4 * n + rows * (k + n) * 2

    rejected = evaluate(exact, example)
    assert rejected["status"] == "incorrect", rejected
    assert any("relative L2" in f["error"] for f in rejected["cases"][0]["failures"])
