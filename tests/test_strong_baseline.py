"""Strong baselines (CPU): the reference_optimizations() hook, the compiled
baseline of ``analyze``, reports vs eager and vs compiled, and the
"reference optimisations + accepted kernels" integration step."""

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ab_fake import with_ab
from synthetic_run import BASELINE_MS, make_run
from torch import nn

from kernel_agent import dryrun, ledger, orchestrator, strong_baseline, toolchain, truth, worker
from kernel_agent.agent import prompts
from kernel_agent.config import OptimizeConfig
from kernel_agent.profiling import profiler
from kernel_agent.report import write_report
from kernel_agent.status import render
from kernel_agent.workloads import create_workload
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workloads.harness import load_harness
from kernel_agent.workloads.llm import LLMWorkload
from kernel_agent.workloads.stt import STTWorkload
from kernel_agent.workloads.tts import TTSWorkload
from kernel_agent.workspace import RunDir, read_json, write_json

TESTS = Path(__file__).parent
TOY = TESTS / "reference_toy.py"
PLAIN_TOY = TESTS / "chaotic_toy.py"


def _spec(harness=TOY, **options):
    return WorkloadSpec(
        repo_id="toy/ref", modality="tts", device="cpu", harness=str(harness), options=options
    )


@pytest.fixture
def cpu(monkeypatch):
    """CPU only, in this process and in the compiled-baseline child process."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # the child process imports the toy harness and kernel_agent like this one
    monkeypatch.setenv("PYTHONPATH", f"{Path(strong_baseline.__file__).parents[1]}:{TESTS}")


def _worker(capsys, *argv):
    worker.main([str(a) for a in argv])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)][-1]
    return json.loads(line[len(worker.MARKER) :])


def _no_profiler(monkeypatch):
    monkeypatch.setattr(profiler, "profile_workload", lambda w, i, **kw: {"classes": []})
    monkeypatch.setattr(profiler, "summarize", lambda p, ms, **kw: "# Profile summary\n")


# ------------------------------------------------------------------ the hook


def test_hook_is_optional():
    toy = load_harness(PLAIN_TOY, _spec(PLAIN_TOY))
    assert toy.reference_optimizations() is None  # the default: nothing to apply
    assert not strong_baseline.has_hook(toy)
    assert strong_baseline.apply(toy, generic=False) is None
    assert strong_baseline.has_hook(load_harness(TOY, _spec()))

    def spec(modality, family=None):
        return WorkloadSpec(repo_id="org/m", modality=modality, family=family)

    assert strong_baseline.has_hook(create_workload(spec("tts", family="voxcpm")))
    assert strong_baseline.has_hook(create_workload(spec("llm")))
    assert strong_baseline.has_hook(create_workload(spec("diffusion")))
    assert not strong_baseline.has_hook(TTSWorkload(spec("tts")))
    assert not strong_baseline.has_hook(STTWorkload(spec("stt")))


def test_apply_hook_and_generic_fallback():
    toy = load_harness(TOY, _spec())
    toy.load()
    before = toy.model.backbone.weight.clone()
    applied = strong_baseline.apply(toy)
    assert applied == {"source": "workload", "description": "toy reference optimisations (benign)"}
    assert not torch.equal(before, toy.model.backbone.weight)

    declined = load_harness(TOY, _spec(reference="none"))
    declined.load()
    assert strong_baseline.apply(declined, generic=False) is None
    generic = strong_baseline.apply(declined)  # --compile-baseline: torch.compile the roots
    assert generic is not None and generic["source"] == "generic"
    assert declined.model._compiled_call_impl is not None  # nn.Module.compile(), lazy


def test_builtin_hooks_on_fake_models():
    class FakeVoxCPM(nn.Module):
        def __init__(self, works: bool) -> None:
            super().__init__()
            self.works = works

        def optimize(self, disable: bool = False):
            if self.works:
                self._feat_encoder_raw = nn.Identity()
            return self  # like VoxCPM: only a warning on stderr when it cannot compile

    vox = create_workload(WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm"))
    vox.model = FakeVoxCPM(works=True)
    assert "model.optimize()" in vox.reference_optimizations()
    vox.model = FakeVoxCPM(works=False)
    with pytest.raises(RuntimeError, match="did not compile"):
        vox.reference_optimizations()
    vox.options["compile"] = True  # -o compile=true: the eager baseline is already compiled
    assert vox.reference_optimizations() is None

    llm = LLMWorkload(WorkloadSpec(repo_id="org/m", modality="llm"))
    llm.model = SimpleNamespace(_can_compile_fullgraph=True, generation_config=SimpleNamespace())
    assert "static KV cache" in llm.reference_optimizations()
    assert llm.model.generation_config.cache_implementation == "static"
    llm.model = SimpleNamespace(_can_compile_fullgraph=False, generation_config=SimpleNamespace())
    assert llm.reference_optimizations() is None


# ------------------------------------------------------------------ measurement


def _eager_run(tmp_path, monkeypatch, capsys, **options):
    root = tmp_path / "run"
    write_json(root / "run.json", {"workload": _spec(**options).to_dict()})
    eager = _worker(capsys, "analyze", "--run-dir", root, "--no-profile", "--iters", 1)
    return RunDir(root), eager


def test_measure_in_process(tmp_path, monkeypatch, capsys, cpu):
    run, eager = _eager_run(tmp_path, monkeypatch, capsys)
    assert "compiled_ms" not in eager  # --no-profile (harness checks): no compiled baseline
    detail = strong_baseline.measure(run, generic=False, warmup=2, iters=2)
    assert detail["status"] == "ok" and detail["source"] == "workload"
    assert detail["description"] == "toy reference optimisations (benign)"
    assert len(detail["warmup_s"]) == 2 and len(detail["times_ms"]) == 2
    assert detail["median_ms"] > 0 and detail["compile_s"] >= 0
    quality = detail["quality"]
    assert quality["passed"], quality  # teacher forced: the benign change passes
    assert quality["metrics"]["teacher_forced"]["passed"]
    assert quality["metrics"]["free_running"]["gating"] is False  # chaotic: informational

    for mode, check in (
        ("broken", lambda d: d["status"] == "ok" and not d["quality"]["passed"]),
        ("none", lambda d: d["status"] == "skipped"),
    ):
        cfg = run.load()
        cfg["workload"]["options"]["reference"] = mode
        write_json(run.run_json, cfg)
        assert check(strong_baseline.measure(run, generic=False, warmup=1, iters=1)), mode

    cfg = run.load()
    cfg["workload"]["options"]["reference"] = "raise"
    write_json(run.run_json, cfg)
    assert strong_baseline.main(["--run-dir", str(run.root), "--iters", "1"]) == 1
    out = capsys.readouterr().out.splitlines()[-1]
    result = json.loads(out[len(strong_baseline.MARKER) :])
    assert result["status"] == "error" and "cannot compile the toy" in result["error"]


def test_analyze_records_the_compiled_baseline(tmp_path, monkeypatch, capsys, cpu):
    _no_profiler(monkeypatch)
    root = tmp_path / "run"
    write_json(root / "run.json", {"workload": _spec().to_dict()})
    result = _worker(capsys, "analyze", "--run-dir", root, "--iters", 1)  # child process
    saved = read_json(root / "baseline.json")
    assert "profile" not in saved and result["compiled_ms"] == saved["compiled_ms"] > 0
    detail = saved["compiled_detail"]
    assert detail["status"] == "ok" and detail["source"] == "workload"
    assert detail["quality"]["passed"] and detail["warmup_s"] and "compile_s" in detail
    assert detail["speedup_vs_eager"] == pytest.approx(
        saved["median_ms"] / saved["compiled_ms"], rel=1e-3
    )
    assert (root / "logs" / "strong-baseline.log").exists()
    summary = (root / "profile" / "summary.md").read_text()
    assert "## Strong baseline (eager vs compiled)" in summary
    assert "toy reference optimisations (benign)" in summary and "vs eager" in summary

    # improve rounds re-profile without re-measuring, but keep the run's baselines
    rnd = tmp_path / "round"
    again = _worker(capsys, "analyze", "--run-dir", root, "--out-dir", rnd, "--no-profile")
    assert again["compiled_ms"] == saved["compiled_ms"]
    assert again["run_eager_ms"] == saved["median_ms"]
    assert "this profile" in strong_baseline.summary_section(again)


def test_compiled_baseline_failures_are_not_fatal(tmp_path, monkeypatch, capsys, cpu):
    _no_profiler(monkeypatch)
    root = tmp_path / "run"
    write_json(root / "run.json", {"workload": _spec(reference="raise").to_dict()})
    result = _worker(capsys, "analyze", "--run-dir", root, "--iters", 1)
    assert "error" not in result and result["median_ms"] > 0
    assert result["compiled_ms"] is None
    detail = result["compiled_detail"]
    assert detail["status"] == "error" and "cannot compile the toy" in detail["error"]
    line = strong_baseline.describe(result)
    assert line.startswith("compiled baseline not available (error: RuntimeError")
    assert "speedups are vs eager only" in strong_baseline.headroom(result)

    plain = tmp_path / "plain"  # no hook, no --compile-baseline: nothing attempted
    write_json(plain / "run.json", {"workload": _spec(PLAIN_TOY).to_dict()})
    result = _worker(capsys, "analyze", "--run-dir", plain, "--iters", 1)
    assert "compiled_ms" not in result and "compiled_detail" not in result
    assert strong_baseline.describe(result) is None and strong_baseline.headroom(result) == ""


# ------------------------------------------------------------------ rendering

BASE = {
    "median_ms": 5400.0,
    "workload": "VoxCPMWorkload(openbmb/VoxCPM2)",
    "compiled_ms": 3800.0,
    "compiled_detail": {
        "status": "ok",
        "source": "workload",
        "description": "VoxCPM `model.optimize()`",
        "median_ms": 3800.0,
        "compile_s": 61.2,
        "quality": {
            "passed": True,
            "metrics": {"teacher_forced": {"mean_step_cosine": 0.998, "min_step_cosine": 0.97}},
        },
    },
}


def test_speedups_and_text():
    assert strong_baseline.compiled_ms(BASE) == 3800.0
    assert strong_baseline.compiled_ms({"compiled_ms": None}) is None
    eager, compiled = strong_baseline.speedups(BASE, 2000.0)
    assert eager == pytest.approx(2.7) and compiled == pytest.approx(1.9)
    assert strong_baseline.vs_both(BASE, 2000.0) == "2.70x vs eager, 1.90x vs compiled"
    assert strong_baseline.vs_both({"median_ms": 100.0}, 50.0) == "2.00x vs eager"
    line = strong_baseline.describe(BASE)
    assert "**3,800.0 ms**, 1.42x vs eager" in line and "compile + warm-up 61.2 s" in line
    assert "teacher-forced quality ok vs eager (mean_step_cosine=0.998" in line
    failed = {"passed": False, "reason": "mean step cosine 0.9 < 0.99"}
    assert "FAILED vs eager: mean step cosine" in strong_baseline.quality_text(failed)

    text = strong_baseline.headroom(BASE)
    assert "**Real headroom**" in text and "**3,800.0 ms** (1.42x vs eager)" in text
    for prompt in (
        prompts.planner_prompt(
            {"repo_id": "o/m", "modality": "tts"}, BASE, "", ["triton"], 2, "py", "tc"
        ),
        prompts.systems_prompt({"repo_id": "o/m", "modality": "tts"}, BASE, "", [], "py", "tc", 3),
    ):
        assert "Real headroom" in prompt and "3,800.0 ms" in prompt
    plain = prompts.planner_prompt(
        {"repo_id": "o/m", "modality": "tts"}, {"median_ms": 1.0}, "", [], 2, "py", "tc"
    )
    assert "Real headroom" not in plain and "compiled_ms" not in plain


def test_combination_verdicts():
    ok = {"status": "ok", "passed": True, "median_ms": 3000.0, "speedup": 1.8}
    rec = strong_baseline.combination(ok, 3800.0)
    assert rec["verdict"] == "composes" and rec["speedup_vs_compiled"] == pytest.approx(1.2667)
    assert strong_baseline.combination_text(rec).startswith("composes: 3,000.0 ms (1.2667x")
    slow = strong_baseline.combination({**ok, "median_ms": 3790.0}, 3800.0)
    assert slow["verdict"] == "no_gain"
    bad = strong_baseline.combination({**ok, "passed": False, "reason": "tf"}, 3800.0)
    assert bad["verdict"] == "fails_quality"
    tb = (
        "Traceback (most recent call last):\n  File x\n"
        "torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped\n"
        "  Explanation: Dynamo developers have intentionally marked ...\n"
    )
    broken = strong_baseline.combination({"status": "runtime_error", "error": tb}, 3800.0)
    assert broken["verdict"] == "breaks"
    assert broken["error"].startswith("torch._dynamo.exc.Unsupported: Attempted")
    assert strong_baseline.combination_text(broken).startswith("breaks: torch._dynamo")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("runs"))


def test_report_status_dashboard_vs_compiled(synthetic, tmp_path, monkeypatch):
    from kernel_agent import charts

    monkeypatch.setattr(charts, "available", lambda: False)
    run = RunDir(tmp_path / "copy")
    shutil.copytree(synthetic.root, run.root)
    baseline = read_json(run.baseline_json)
    baseline["compiled_detail"] = {**BASE["compiled_detail"], "median_ms": 1104.6}
    write_json(run.baseline_json, baseline)
    integration = read_json(run.root / "integration.json")
    integration["reference"] = {
        "items": ["attn=/x/attn.py", strong_baseline.REFERENCE_LABEL],
        **strong_baseline.combination(
            {"status": "ok", "passed": True, "median_ms": 800.0, "speedup": 1.9}, 1104.6
        ),
    }
    write_json(run.root / "integration.json", integration)

    report = write_report(run).read_text()
    assert "| | latency (ms) | vs eager | vs compiled | quality |" in report
    assert f"| baseline (eager) | {BASELINE_MS:.1f} | 1.00x | 0.72x | reference |" in report
    assert "| compiled baseline | 1104.6 | 1.39x | 1.00x | teacher-forced quality ok" in report
    assert "| optimised | 889.0 | **1.7237x** | 1.24x |" in report
    assert "| compiled baseline + accepted kernels | 800.0 | 1.92x | 1.38x | composes" in report
    assert "* `attn`, `reference_optimizations`: composes: 800.0 ms" in report
    assert "# then the reference optimisations (they compose with the kernels)" in report

    text = render(run, width=200)
    assert "baseline 1,532.4 ms  |  compiled 1,104.6 ms (1.39x)" in text
    assert "measured 889.0 ms (1.72x vs eager, 1.24x vs compiled, integrated)" in text
    assert "compiled baseline + accepted kernels: composes: 800.0 ms" in text

    page = run.dashboard.read_text()
    assert "torch.compile 1,104.6 ms (1.39×)" in page and "1.24× vs. compiled" in page
    assert ledger.summary(run)["compiled_ms"] == 1104.6


# ------------------------------------------------------------------ integration


def _orchestrator(tmp_path, monkeypatch, compiled_ms):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    run = dryrun.create_run(config)
    baseline = read_json(run.baseline_json)
    if compiled_ms:  # as analyze writes it, before the orchestrator seals baseline.json
        baseline["compiled_ms"] = compiled_ms
        truth.writable(run.baseline_json)
        write_json(run.baseline_json, baseline)
        truth.of(run).seal_baseline(baseline["median_ms"])
    orch = orchestrator.Orchestrator(run, config)
    target = run.target_ids()[0]
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def build(reference):\n    return reference\n")
    item = f"{target}={kernel}"
    monkeypatch.setattr(orch, "_integration_items", lambda: ([("kernel", item)], {}))
    return orch, item


def test_integrate_tries_reference_plus_kernels(tmp_path, monkeypatch):
    orch, item = _orchestrator(tmp_path, monkeypatch, compiled_ms=1100.0)
    base = read_json(orch.run.baseline_json)["median_ms"]
    calls = []

    def fake_worker(run, command, *args):
        calls.append(list(args))
        if str(strong_baseline.REFERENCE_TRANSFORM) in args:
            return {
                "status": "runtime_error",
                "passed": False,
                "error": "Traceback\ntorch._dynamo.exc.Unsupported: Graph break due to "
                "unsupported builtin kernel.launch\n",
            }
        return {"status": "ok", "passed": True, "median_ms": 1200.0, "speedup": base / 1200}

    orch.worker = with_ab(fake_worker, base)
    asyncio.run(orch.integrate())
    data = read_json(orch.run.root / "integration.json")
    assert [a["item"] for a in data["accepted"]] == [item]  # greedy result unchanged
    assert data["final"]["median_ms"] == 1200.0
    ref = data["reference"]
    assert ref["items"] == [item, strong_baseline.REFERENCE_LABEL]
    assert ref["verdict"] == "breaks" and "Graph break" in ref["error"]
    assert calls[-1][:8] == [
        "--warmup",
        "2",
        "--iters",
        "5",
        "--kernel",
        item,
        "--transform",
        str(strong_baseline.REFERENCE_TRANSFORM),
    ]
    assert calls[-1][8:] == orch.truth.worker_args()  # verified baseline, like every e2e
    assert "--baseline-ms" in calls[-1]
    assert len(calls) == 2  # the kernel alone, then reference optimisations + kernel
    assert any(e["event"] == "reference_combination" for e in ledger.events(orch.run))
    assert all("reference" not in r["snapshot"] for r in ledger.rows(orch.run))

    # re-integration of the same kernels reuses the measurement
    calls.clear()
    asyncio.run(orch.integrate(reuse=True))
    assert calls == [] and read_json(orch.run.root / "integration.json")["reference"] == ref


def test_integrate_without_compiled_baseline_is_unchanged(tmp_path, monkeypatch):
    orch, _ = _orchestrator(tmp_path, monkeypatch, compiled_ms=None)
    calls = []

    def fake_worker(run, command, *args):
        calls.append(args)
        return {"status": "ok", "passed": True, "median_ms": 1200.0, "speedup": 1.27}

    orch.worker = with_ab(fake_worker, read_json(orch.run.baseline_json)["median_ms"])
    asyncio.run(orch.integrate())
    assert len(calls) == 1
    assert "reference" not in read_json(orch.run.root / "integration.json")


def test_reference_transform_applies_the_hook(cpu):
    from kernel_agent.integrate.patcher import PatchReport, apply_transforms

    toy = load_harness(TOY, _spec())
    toy.load()
    before = toy.model.backbone.weight.clone()
    report = apply_transforms(toy, [strong_baseline.REFERENCE_TRANSFORM], PatchReport())
    assert report.transforms == [strong_baseline.REFERENCE_LABEL]
    assert not torch.equal(before, toy.model.backbone.weight)
    assert ledger.item_label(str(strong_baseline.REFERENCE_TRANSFORM)) == "reference_optimizations"
