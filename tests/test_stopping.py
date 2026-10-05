"""Stop-condition check (#55): the comparison logic, the natural-length baseline and
the e2e check on a CPU toy with a stop head, end to end through the worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from stop_toy import STOP_AT
from test_truth import force_write, sealed_run, tamper_events

from kernel_agent import toolchain, truth, worker
from kernel_agent.report import _flat_metrics
from kernel_agent.workloads import create_workload, stopping
from kernel_agent.workloads.base import STOP_NEAR_TIE, WorkloadSpec, compare_stop
from kernel_agent.workloads.quality import probe_messages, summary_section
from kernel_agent.workspace import RunDir

TOY = Path(__file__).with_name("chaotic_toy.py")
STOP_TOY = Path(__file__).with_name("stop_toy.py")

#: Reference loop of the toy (stop_toy.StopModel.generate) and broken versions of it.
LOOP = """import types


def generate(self, x, steps, min_steps=None):
    min_steps = steps if min_steps is None else min_steps
    prev_flag = 0
    for i in range(steps):
        h = torch.tanh(self.backbone(x))
        x = self.sampler(h)
        stop_flag = int(self.stop_head(h, i).argmax(-1).flatten()[0])
{check}
    return x


def apply(workload):
    workload.model.generate = types.MethodType(generate, workload.model)
"""
SAME = "        if i > min_steps and stop_flag == 1:\n            break"
NEVER = "        pass  # the stop head is computed, never consulted"
LATE = (
    "        if i > min_steps + 1 and prev_flag == 1:\n            break\n"
    "        prev_flag = stop_flag"
)
EARLY = (  # a mis-calibrated threshold: stops while the stop head still says "continue"
    "        logits = self.stop_head(h, i).flatten()\n"
    "        if i > min_steps and float(logits[1] - logits[0]) > -2.0:\n            break"
)

BENIGN = """import torch


def apply(workload):
    with torch.no_grad():
        workload.model.backbone.weight.mul_(1 + 2**-10)
"""


def _ref(steps: int = 10, margins: list[float] | None = None, **extra: Any) -> dict[str, Any]:
    """A baseline result: ``steps`` steps (``len(margins)`` when given)."""
    if margins is None:
        margins = [-5.0] * (steps - 1) + [3.0]
    return {
        "steps": len(margins),
        "min_steps": 2,
        "max_steps": 40,
        "stop_margins": margins,
        **extra,
    }


# ------------------------------------------------------------------ comparison logic


def test_compare_stop_is_exact_by_default():
    ref = _ref()
    ok = compare_stop(ref, {"steps": 10})
    assert ok.passed and ok.reason == ""
    assert ok.metrics == {"steps": 10, "reference_steps": 10, "reference_stop_margin": 3.0}

    late = compare_stop(ref, {"steps": 11})
    assert not late.passed
    assert late.reason == (
        "the stop condition fires 1 step(s) late: 11 steps, the baseline 10 "
        "(baseline stop margin +3.000 at step 9)"
    )
    assert late.metrics["differing_step"] == 9 and late.metrics["differing_step_margin"] == 3.0

    early = compare_stop(ref, {"steps": 8})
    assert not early.passed and early.reason.startswith("the stop condition fires 2 step(s) early")
    assert early.metrics["differing_step"] == 7 and early.metrics["differing_step_margin"] == -5.0

    never = compare_stop(ref, {"steps": 40})
    assert not never.passed and "never fires (ran to max_steps=40): 40 steps" in never.reason


def test_compare_stop_tolerates_one_step_only_at_a_near_tie():
    tie_at_stop = _ref(margins=[-5.0] * 9 + [0.2])  # the baseline barely stopped
    tie_before = _ref(margins=[-5.0] * 8 + [-0.3, 3.0])  # ... and barely went on
    for ref, steps in ((tie_at_stop, 11), (tie_before, 9)):
        assert not compare_stop(ref, {"steps": steps}).passed  # exact by default
        cmp = compare_stop(ref, {"steps": steps}, tolerance=1)
        assert cmp.passed and cmp.reason == "", cmp
        assert "a near-tie (|margin| <= 0.5)" in str(cmp.metrics["tolerated"])
    # two steps away, a clear margin, a tighter near-tie or no margins: still rejected
    assert not compare_stop(tie_at_stop, {"steps": 12}, tolerance=1).passed
    assert not compare_stop(_ref(), {"steps": 11}, tolerance=1).passed
    assert not compare_stop(tie_at_stop, {"steps": 11}, tolerance=1, near_tie=0.1).passed
    no_margins = {**tie_at_stop, "stop_margins": None}
    cmp = compare_stop(no_margins, {"steps": 11}, tolerance=1)
    assert not cmp.passed and "margin" not in cmp.reason
    # a stop at or before min_steps is never the stop head's decision
    first = _ref(steps=4, margins=[-9.0, -9.0, -0.1, 2.0])
    assert not compare_stop(first, {"steps": 3}, tolerance=1).passed
    assert STOP_NEAR_TIE == 0.5


def test_compare_stop_checks_the_output_length_and_mismatched_margins():
    ref = _ref(output_length=1000)
    assert compare_stop(ref, {"steps": 10, "output_length": 1000}).passed
    assert compare_stop(ref, {"steps": 10}).passed  # the candidate reports none
    bad = compare_stop(ref, {"steps": 10, "output_length": 900})
    assert not bad.passed and bad.reason == "output length 900 != 1000 after the same 10 steps"
    odd = compare_stop({**ref, "stop_margins": [1.0, 2.0]}, {"steps": 10})  # not one per step
    assert odd.passed and "reference_stop_margin" not in odd.metrics


def test_stop_summary():
    info = stopping.stop_summary(_ref(margins=[9.0, 9.0, 9.0, -0.4, -6.0, 2.5]))
    assert info == {
        "steps": 6,
        "min_steps": 2,
        "max_steps": 40,
        "stopped": True,
        "stop_step": 5,
        "stop_margin": 2.5,
        "closest_margin": 0.4,  # steps <= min_steps (the 9.0s) are never consulted
    }
    never = stopping.stop_summary(_ref(steps=40, margins=[-3.0] * 40))
    assert not never["stopped"] and never["stop_step"] is None
    blind = stopping.stop_summary({"steps": 12, "max_steps": 40})  # no margins recorded
    assert blind["stopped"] and "stop_step" not in blind


def test_messages_and_summary():
    ok = {"status": "ok", "ms": 12.5, "near_tie": 0.5, **stopping.stop_summary(_ref())}
    baseline = {"natural_length": ok, "sensitivity": {"free_running_passed": True}}
    assert stopping.messages(baseline) == [
        "analyze: natural-length run: 10 steps, stop at step 9 (margin +3.000, closest "
        "before: 5.000) (12.5 ms)"
    ]
    assert "stop condition" in summary_section(baseline)
    assert any("natural-length run" in m for m in probe_messages(baseline))
    tie = {**ok, "closest_margin": 0.3}
    assert "stop_tolerance=1" in stopping.messages({"natural_length": tie})[-1]
    never = {**ok, **stopping.stop_summary(_ref(steps=40, margins=[-3.0] * 40))}
    assert "never stopped" in stopping.messages({"natural_length": never})[1]
    err = stopping.messages({"natural_length": {"status": "error", "error": "Boom"}})
    assert err[0].startswith("WARNING: the natural-length run failed") and "Boom" in err[0]
    assert stopping.messages({"natural_length": {"status": "none"}}) == []
    assert stopping.summary_lines({"natural_length": {"status": "none"}}) == []
    assert stopping.summary_text({"passed": True, "steps": 31}) == (
        "natural length passed (31 steps)"
    )
    assert stopping.summary_text({"passed": False, "reason": "late"}) == (
        "natural length FAILED: late"
    )
    assert "skipped (x)" in stopping.summary_text({"passed": True, "skipped": "x"})


# ------------------------------------------------------------------ the toy, in process


def _toy(path: Path = STOP_TOY) -> Any:
    wl = create_workload(
        WorkloadSpec(repo_id="toy/stop", modality="tts", device="cpu", harness=str(path))
    )
    wl.load()
    return wl


def _transform(tmp_path: Path, name: str, check: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text("import torch\n" + LOOP.format(check=check))
    return path


def test_record_and_check_on_the_stop_toy(tmp_path):
    from kernel_agent.integrate.patcher import PatchReport, apply_transforms

    wl = _toy()
    main = wl.run(wl.make_inputs())
    assert main["states"].shape[0] == 60  # the main run has a fixed length
    ref, info = stopping.record_baseline(wl)
    assert info["status"] == "ok" and info["stopped"] and info["stop_step"] == STOP_AT
    assert info["steps"] == STOP_AT + 1 and info["stop_margin"] > 0.5 and info["near_tie"] == 0.5
    baseline = {"natural_length": info}
    assert stopping.check(wl, ref, baseline) == {"passed": True, "reason": "", **_ok(info)}

    verdicts = {}
    for name, check in (("same", SAME), ("never", NEVER), ("late", LATE), ("early", EARLY)):
        fresh = _toy()
        apply_transforms(fresh, [_transform(tmp_path, name, check)], PatchReport())
        assert fresh.run(fresh.make_inputs())["states"].shape[0] == 60  # same main run
        verdicts[name] = stopping.check(fresh, ref, baseline)
    assert verdicts["same"]["passed"], verdicts["same"]
    assert not verdicts["never"]["passed"] and "never fires" in verdicts["never"]["reason"]
    assert not verdicts["late"]["passed"] and verdicts["late"]["steps"] == STOP_AT + 2
    assert not verdicts["early"]["passed"] and "early" in verdicts["early"]["reason"]

    broken = _toy()
    broken.natural_length_run = lambda reference=None: {}  # no "steps"
    failed = stopping.check(broken, ref, baseline)
    assert not failed["passed"] and "the natural-length run failed: KeyError" in failed["reason"]
    skipped = stopping.check(wl, None, baseline)
    assert skipped["passed"] and "re-run analyze" in skipped["skipped"]


def _ok(info: dict[str, Any]) -> dict[str, Any]:
    steps = info["steps"]
    return {"steps": steps, "reference_steps": steps, "reference_stop_margin": info["stop_margin"]}


def test_workloads_without_a_stop_condition_are_unaffected(tmp_path):
    wl = _toy(TOY)
    assert not stopping.declares(wl) and wl.natural_length_run() is None
    output, info = stopping.record_baseline(wl)
    assert output is None and info["status"] == "none"
    assert stopping.check(wl, None, {"natural_length": info}) is None
    assert stopping.check(wl, None, {}) is None
    path = tmp_path / "natural.pt"
    path.write_bytes(b"stale")
    assert stopping.save_baseline(wl, path)["status"] == "none" and not path.exists()


# ------------------------------------------------------------------ worker, end to end


def _call(capsys, run: RunDir, *argv: Any) -> dict[str, Any]:
    worker.main([str(a) for a in (argv[0], "--run-dir", run.root, *argv[1:])])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)]
    return json.loads(line[-1][len(worker.MARKER) :])


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)


def test_worker_records_verifies_and_checks_the_stop_condition(tmp_path, cpu, capsys):
    """analyze stores the natural-length baseline in .truth/ (sealed, verified by e2e);
    e2e passes a benign transform and rejects stop-condition bugs that pass every
    other check (the main and held-out runs have a fixed length)."""
    spec = WorkloadSpec(repo_id="toy/stop", modality="tts", device="cpu", harness=str(STOP_TOY))
    run = sealed_run(tmp_path, workload=spec.to_dict())
    keeper = truth.of(run)
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    natural = baseline["natural_length"]
    assert natural["status"] == "ok" and natural["stop_step"] == STOP_AT, natural
    path = run.baseline_output_natural()
    assert path == run.root / ".truth/baseline_output_natural.pt" and path.is_file()
    keeper.seal_baseline(baseline["median_ms"])
    assert f".truth/baseline_output_natural.pt={keeper.expect(path)}" in keeper.worker_args()
    assert any("natural-length run" in m for m in probe_messages(baseline))

    def e2e(name: str, source: str) -> dict[str, Any]:
        transform = tmp_path / f"{name}.py"
        transform.write_text(source)
        return _call(
            capsys, run, "e2e", "--transform", transform, "--iters", 1, *keeper.worker_args()
        )

    ok = e2e("benign", BENIGN)
    assert ok["passed"], ok
    assert ok["metrics"]["natural_length"]["passed"]
    assert ok["metrics"]["natural_length"]["steps"] == STOP_AT + 1

    for name, check, what in (
        ("never", NEVER, "never fires"),
        ("late", LATE, "fires 1 step(s) late"),
    ):
        bad = e2e(name, "import torch\n" + LOOP.format(check=check))
        assert bad["status"] == "ok" and not bad["passed"], bad
        assert bad["metrics"]["teacher_forced"]["passed"]  # every other check passes ...
        assert bad["metrics"]["holdout"]["passed"]
        assert bad["reason"].startswith(f"natural length: the stop condition {what}"), bad

    force_write(path, path.read_bytes() + b"\0")
    tampered = e2e("benign2", BENIGN)
    assert tampered["status"] == "tampered" and not tampered["passed"]
    assert ".truth/baseline_output_natural.pt" in {e["file"] for e in tamper_events(run)}


def test_worker_without_a_stop_condition_adds_nothing(tmp_path, cpu, capsys):
    spec = WorkloadSpec(repo_id="toy/chaotic", modality="tts", device="cpu", harness=str(TOY))
    run = sealed_run(tmp_path, workload=spec.to_dict())
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    assert baseline["natural_length"]["status"] == "none"
    assert not run.baseline_output_natural().exists()
    truth.of(run).seal_baseline(baseline["median_ms"])
    assert not any("natural" in a for a in truth.of(run).worker_args())
    result = _call(capsys, run, "e2e", "--iters", 1, *truth.of(run).worker_args())
    assert result["passed"] and "natural_length" not in result["metrics"]
    assert not any(k.startswith("natural") for k in _flat_metrics(result["metrics"]))
