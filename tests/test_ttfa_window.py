"""A time to first audio is timed up to the first chunk: every timed request of a measurement
but one stops there (``Workload.metric_window``), the one whole request gives the output the
quality checks judge and the full-run details.

On VoxCPM2 a request up to its first chunk takes 98 ms of a 5.8 s streamed run (eager), so
``e2e``, the paired A/B rounds of the integration and the diverse set time the same number
of requests for a fraction of the work. CPU: the streaming toy (``streaming_toy.py``) logs
the steps of every request; a simulated clock times them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fake_clock import Clock
from streaming_toy import FIRST_S, STEP_S, requests
from test_perceptual import _call, _cpu

from kernel_agent import truth
from kernel_agent.workloads import base, create_workload, diverse
from kernel_agent.workloads.base import WorkloadSpec, measure
from kernel_agent.workspace import RunDir, write_json

TOY = Path(__file__).with_name("streaming_toy.py")
STEPS = 60  # the toy's default
WHOLE_MS = 1000 * (FIRST_S + (STEPS - 1) * STEP_S)


def _spec(metric: str, log: Path) -> WorkloadSpec:
    options = {"metric": metric, "requests_log": str(log)}
    return WorkloadSpec("toy/streaming", "tts", device="cpu", harness=str(TOY), options=options)


def _sealed(tmp_path: Path, metric: str, log: Path) -> RunDir:
    run = RunDir.create(tmp_path, "toy/streaming")
    write_json(
        run.run_json,
        {
            "card": {"repo_id": "toy/streaming"},
            "workload": _spec(metric, log).to_dict(),
            "config": {"quality": "exact"},
            "truth": truth.new_section(),
        },
    )
    return run


@pytest.fixture
def clock(monkeypatch):
    _cpu(monkeypatch)
    monkeypatch.setattr(base, "time", Clock())


@pytest.mark.parametrize("metric", ["latency", "ttfa"])
def test_measurements_time_requests_up_to_the_first_chunk(tmp_path, clock, capsys, metric):
    log = tmp_path / "requests.log"
    run = _sealed(tmp_path, metric, log)
    keeper = truth.of(run)
    ttfa = metric == "ttfa"
    short = 1 if ttfa else STEPS  # a timed request: up to its first chunk, or whole
    value = 1000 * FIRST_S if ttfa else WHOLE_MS

    # analyze: one whole warm-up, two short timed requests, the last one whole
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 3)
    assert requests(log)[:4] == [STEPS, short, short, STEPS]
    assert baseline["median_ms"] == pytest.approx(value) and len(baseline["times_ms"]) == 3
    if ttfa:  # the steady state and the full run: from the whole request
        detail = baseline["metric_detail"]
        assert detail["chunks"] == STEPS and detail["steady_chunks"] == 8
        assert detail["run_ms"] == pytest.approx(WHOLE_MS)
        assert detail["chunk_ms"] == pytest.approx(1000 * STEP_S)
    keeper.seal_baseline(baseline["median_ms"])

    # e2e (evaluate_e2e's --warmup 2): the second warm-up is short too
    log.unlink()
    r = _call(capsys, run, "e2e", "--warmup", 2, "--iters", 3, *keeper.worker_args())
    assert r["status"] == "ok" and r["passed"], r
    assert requests(log)[:5] == [STEPS, short, short, short, STEPS]
    assert r["median_ms"] == pytest.approx(value) and r["speedup"] == pytest.approx(1.0)
    if ttfa:
        assert r["metric_detail"]["chunks"] == STEPS
    # judged on the whole streamed output, untimed checks included
    assert r["metrics"]["teacher_forced"]["passed"] and r["metrics"]["holdout"]["passed"]
    assert r["metrics"]["holdout"]["memoisation"]["fresh_over_repeat"] == pytest.approx(1.0)

    # the paired A/B: warm-ups and rounds of both states short, then one whole request of B
    noop = tmp_path / "noop.py"
    noop.write_text("def apply(workload):\n    workload.applied = True\n")
    log.unlink()
    r = _call(
        capsys,
        run,
        "e2e_ab",
        "--rounds",
        4,
        "--warmup",
        2,
        "--b-transform",
        noop,
        *keeper.worker_args(),
    )
    assert r["status"] == "ok" and r["passed"], r
    timed = 3 + 3 + 2 * 4  # warm-ups (--warmup + 1 per state) and rounds
    reqs = requests(log)
    assert reqs[:timed] == [short] * timed
    assert reqs[timed] == STEPS  # B's whole request (ttfa) or B's teacher-forced replay
    assert r["ab"]["rounds"] == 4 and r["ab"]["undo_check"]["B"] == "identical"
    assert r["ab"]["a_ms"] == r["ab"]["b_ms"] == [pytest.approx(value)] * 4
    if ttfa:
        assert r["metric_detail"]["chunks"] == STEPS
        assert r["metric_detail"]["run_ms"] == pytest.approx(WHOLE_MS)
    else:
        assert "metric_detail" not in r


def test_a_harness_that_ignores_the_window_is_timed_as_before(tmp_path, clock):
    """``measure`` on a streaming harness that streams to the end inside the window: the
    same value, the full-run details from the last request."""
    wl = create_workload(_spec("ttfa", tmp_path / "log"))
    wl.load()
    whole = wl.run

    def ignoring(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        saved, wl.in_window = wl.in_window, False
        try:
            return whole(inputs)
        finally:
            wl.in_window = saved

    wl.run = ignoring  # type: ignore[method-assign]
    timing = measure(wl, wl.make_inputs(), warmup=1, iters=3)
    assert requests(tmp_path / "log") == [STEPS] * 4
    assert timing["median_ms"] == pytest.approx(1000 * FIRST_S)
    assert timing["metric_detail"]["chunks"] == STEPS
    assert timing["metric_detail"]["run_ms"] == pytest.approx(WHOLE_MS)


@pytest.mark.parametrize("metric", ["latency", "ttfa"])
def test_the_diverse_set_times_short_requests_and_judges_a_whole_one(tmp_path, clock, metric):
    log = tmp_path / "log"
    wl = create_workload(_spec(metric, log))
    wl.load()
    _, output, timing = diverse.run_input(wl, {"seed": 3}, warmup=1, iters=3)
    short = 1 if metric == "ttfa" else STEPS
    assert requests(log) == [short, short, short, STEPS]
    assert output["states"].shape[0] == STEPS  # what the quality check judges
    value = 1000 * FIRST_S if metric == "ttfa" else WHOLE_MS
    assert timing["times_ms"] == [pytest.approx(value)] * 3
