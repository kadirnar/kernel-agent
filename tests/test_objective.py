"""The metric a run optimises (``-o metric=``, objective.py): validation, what ``measure``
and the A/B / held-out timings return for ``latency`` and ``ttfa`` on a CPU streaming toy,
and how reports, status and the live page name it."""

import shutil
import time
from typing import Any

import pytest
import torch
from synthetic_run import make_run
from torch import nn

from kernel_agent import objective, watch
from kernel_agent.integrate import ab
from kernel_agent.report import write_report
from kernel_agent.status import render
from kernel_agent.workloads import create_workload, holdout, validate_metric
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, measure, timed_run
from kernel_agent.workspace import RunDir, read_json, write_json

FIRST_S, CHUNK_S, CHUNKS = 0.04, 0.01, 6


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    """Timing on the CPU: no GPU warm-up, no synchronisation."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class Streamer(Workload):
    """Streams ``chunks`` audio chunks: the first after ``first_s``, then one every
    ``chunk_s`` (each 40 ms of audio)."""

    metrics = (objective.LATENCY, objective.TTFA)
    defaults = {"first_s": FIRST_S, "chunk_s": CHUNK_S, "chunks": CHUNKS, "seed": 0}

    def load(self) -> None:
        pass

    def roots(self) -> dict[str, nn.Module]:
        return {}

    def make_inputs(self) -> int:
        return int(self.options["seed"])

    def run(self, inputs: int) -> dict[str, Any]:
        time.sleep(float(self.options["first_s"]))
        chunks = []
        for i in range(int(self.options["chunks"])):
            if i:
                time.sleep(float(self.options["chunk_s"]))
            chunks.append(torch.full((4,), float(inputs + i)))
            self.mark_chunk(audio_ms=40.0)
        return {"audio": torch.cat(chunks)}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(torch.equal(reference["audio"], candidate["audio"]))


def _streamer(**options: Any) -> Streamer:
    return Streamer(WorkloadSpec(repo_id="toy/streamer", modality="tts", options=options))


def test_metric_names_and_validation():
    assert objective.get(None).name == "latency" == objective.of({}).name  # older runs
    assert objective.of({"metric": "ttfa"}).title == "Time to first audio"
    with pytest.raises(ValueError, match="unknown metric 'bogus'"):
        objective.get("bogus")
    assert _streamer().metric == "latency" and _streamer(metric="TTFA").metric == "ttfa"
    _streamer(metric="ttfa").check_metric()
    with pytest.raises(ValueError, match=r"metric=throughput is not implemented yet \(issue #74\)"):
        _streamer(metric="throughput").check_metric()

    class LatencyOnly(Streamer):
        metrics = (objective.LATENCY,)

    with pytest.raises(ValueError, match="LatencyOnly cannot time metric=ttfa"):
        LatencyOnly(WorkloadSpec("toy/l", "tts", options={"metric": "ttfa"})).check_metric()

    llm = WorkloadSpec(repo_id="org/llm", modality="llm", options={"metric": "ttfa"})
    with pytest.raises(ValueError, match="LLMWorkload cannot time metric=ttfa"):
        create_workload(llm)
    with pytest.raises(ValueError, match="LLMWorkload cannot time metric=ttfa"):
        validate_metric(llm)  # Orchestrator.create: before a run directory exists
    vox = WorkloadSpec("openbmb/VoxCPM2", "tts", family="voxcpm", options={"metric": "ttfa"})
    assert create_workload(vox).metric == "ttfa"
    validate_metric(vox)
    # no built-in workload: the harness agent writes one, any implemented metric may be asked
    validate_metric(WorkloadSpec("org/odd", "unknown", options={"metric": "ttfa"}))
    with pytest.raises(ValueError, match="issue #74"):
        validate_metric(WorkloadSpec("org/odd", "unknown", options={"metric": "throughput"}))


def test_latency_is_unchanged():
    wl = _streamer()
    timing = measure(wl, wl.make_inputs(), warmup=0, iters=2)
    assert timing["metric"] == "latency" and "metric_detail" not in timing
    total = 1000 * (FIRST_S + (CHUNKS - 1) * CHUNK_S)
    assert total * 0.95 <= timing["median_ms"] < total + 40


def test_ttfa_times_the_first_chunk_and_reports_the_steady_state():
    wl = _streamer(metric="ttfa")
    inputs = wl.make_inputs()
    timing = measure(wl, inputs, warmup=0, iters=3)
    assert timing["metric"] == "ttfa" and len(timing["times_ms"]) == 3
    assert 1000 * FIRST_S * 0.95 <= timing["median_ms"] < 1000 * FIRST_S + 25
    detail = timing["metric_detail"]
    assert detail["chunks"] == CHUNKS and detail["steady_chunks"] == CHUNKS - 1  # < 8 left
    assert 1000 * CHUNK_S * 0.95 <= detail["chunk_ms"] < 1000 * CHUNK_S + 15
    assert detail["rtf"] == pytest.approx(detail["chunk_ms"] / 40.0, rel=0.05)
    assert detail["run_ms"] >= 1000 * (FIRST_S + (CHUNKS - 1) * CHUNK_S) * 0.95
    assert "steady state" in objective.detail_text(detail) and "full streamed run" in (
        objective.detail_text(detail)
    )

    with wl.with_options({"steady_chunks": 2}):
        _, _, detail = timed_run(wl, inputs)
    assert detail["steady_chunks"] == 2

    # the A/B rounds and the held-out / memoisation runs time the same metric
    _, ms = ab.timed_run(wl, inputs)
    assert ms < 1000 * FIRST_S + 25
    _, _, ms = holdout.run_variant(wl, {"seed": 5})
    assert ms < 1000 * FIRST_S + 25


def test_ttfa_clock_starts_before_the_run():
    wl = _streamer(metric="ttfa")
    original = wl.run

    def wrapped(x):  # a transform that works before the run (precomputes, say)
        time.sleep(0.03)
        return original(x)

    wl.run = wrapped  # type: ignore[method-assign]
    _, ms, _ = timed_run(wl, wl.make_inputs())
    assert ms >= 1000 * (FIRST_S + 0.03) * 0.95


def test_ttfa_needs_marked_chunks():
    wl = _streamer(metric="ttfa")
    wl.run = lambda x: {"audio": torch.zeros(4)}  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match=r"no audio chunk.*the streaming path was bypassed"):
        timed_run(wl, wl.make_inputs())


def test_objective_text():
    baseline = {
        "metric": "ttfa",
        "median_ms": 99.3,
        "metric_detail": {"chunk_ms": 98.1, "rtf": 0.613, "steady_chunks": 8, "run_ms": 5897.0},
    }
    line = objective.describe(baseline)
    assert line is not None and line.startswith("metric: time to first audio (-o metric=ttfa)")
    assert "98.1 ms per chunk, RTF 0.613 (next 8 chunks)" in line and "5,897.0 ms" in line
    section = objective.summary_section(baseline)
    assert "## Objective: time to first audio" in section and "99.3 ms" in section
    assert "the profile above covers that window only" in section
    assert objective.describe({}) is None and objective.summary_section({}) == ""


@pytest.fixture(scope="module")
def ttfa_run(tmp_path_factory):
    """The synthetic run, its baseline measured as time to first audio."""
    run = make_run(tmp_path_factory.mktemp("runs"))
    copy = RunDir(tmp_path_factory.mktemp("ttfa") / "run")
    shutil.copytree(run.root, copy.root)
    baseline = read_json(copy.baseline_json, {})
    baseline.update(
        metric="ttfa",
        metric_detail={"chunk_ms": 14.9, "rtf": 0.093, "steady_chunks": 8, "run_ms": 897.0},
    )
    write_json(copy.baseline_json, baseline)
    return copy


def test_reports_name_the_metric(ttfa_run):
    text = render(ttfa_run, width=160)
    assert "metric: time to first audio (-o metric=ttfa); steady state 14.9 ms per chunk" in text
    s = watch.state(ttfa_run)["summary"]
    assert s["metric"]["title"] == "Time to first audio" and s["metric"]["per"] == "to first audio"
    pytest.importorskip("matplotlib")
    report = write_report(ttfa_run).read_text()
    assert "| | TTFA (ms) | vs eager | vs compiled | quality |" in report
    assert "* metric: time to first audio (-o metric=ttfa)" in report
