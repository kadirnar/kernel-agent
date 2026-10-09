"""Timing contexts (#226): which context a target is timed in (from fixture profiles), graph
timing's subtraction of the L2 flushes and its capture failures (a simulated GPU on a
simulated clock), the evaluator's context fields and the re-check's context; on the GPU,
graph timing against eager timing and the FP8 GEMM verdict that flips between them."""

from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from fake_clock import Clock

from kernel_agent.kernels import bench, evaluate, recheck
from kernel_agent.kernels import context as timing_context
from kernel_agent.kernels.context import GRAPH_SHARE, TimingContext
from kernel_agent.workspace import RunDir, write_json

FIXTURES = Path(__file__).parent / "fixtures" / "timing_context"
MB = 1024 * 1024


# ------------------------------------------------------------------ context selection


def qwen_profile() -> dict[str, Any]:
    """Qwen3-0.6B's eager decode profile (trimmed): no CUDA graph, 28 decoder layers whose
    other work between two decode calls of one layer is ~1.1 GB."""
    profile: dict[str, Any] = json.loads((FIXTURES / "qwen3_decode_profile.json").read_text())
    return profile


def graphed(profile: dict[str, Any], stage: str, share: float = 1.0) -> dict[str, Any]:
    """A re-profile of ``profile`` whose ``stage`` replays CUDA graphs (``share`` of its GPU
    events graph-launched); the hooks no longer see inside it (``module_gaps``)."""
    out = copy.deepcopy(profile)
    for row in out["kernel_view"]["timeline"]["stages"]:
        if row["stage"] == stage:
            row["graph_events"] = int(share * row["events"])
    out["module_gaps"] = {"graph_replays": {stage: 64}}
    out["classes"] = [c for c in out["classes"] if not c["example_qualname"].startswith(stage)]
    return out


def make_run(
    tmp_path: Path,
    profiles: list[dict[str, Any]],
    spec: dict[str, Any],
    l2_mb: float | None = 48.0,
) -> RunDir:
    """A run directory: its analyze profile (``profiles[0]``), one improve round per later
    profile, one target ``t`` of ``spec`` and the GPU's L2 in ``toolchain.json``."""
    run = RunDir(tmp_path / "run")
    write_json(run.profile_dir / "profile.json", profiles[0])
    for n, profile in enumerate(profiles[1:], start=2):
        write_json(run.root / "rounds" / str(n) / "profile" / "profile.json", profile)
    write_json(run.target("t") / "spec.json", spec)
    gpu = {"name": "fake", "l2_cache_mb": l2_mb} if l2_mb is not None else {}
    write_json(run.toolchain_json, {"gpu": gpu})
    return run


DECODER = {
    "id": "t",
    "module_class": "Qwen3DecoderLayer",
    "phase": "decode",
    "capture": {
        "qualname": "model.model.layers.0",
        "instance_groups": {"model.model.layers.*": {"instances": 28, "calls": 1764}},
    },
}


def test_an_eager_profile_times_eagerly_with_a_cold_l2(tmp_path):
    found = timing_context.for_target(make_run(tmp_path, [qwen_profile()], DECODER), "t")
    assert (found.context, found.l2) == ("eager", "cold")
    assert "0% of the 103268 GPU events of stage model.model" in found.reason
    assert "about 1,144 MB of other work" in found.reason and "48 MB L2" in found.reason
    assert found.kwargs() == {"context": "eager", "l2_flush": True, "context_reason": found.reason}


def test_a_graph_launched_stage_in_the_newest_profile_times_in_a_graph(tmp_path):
    base = qwen_profile()
    rounds = [base, graphed(base, "model.model", 0.2), graphed(base, "model.model")]
    run = make_run(tmp_path, rounds, DECODER)
    found = timing_context.for_target(run, "t")
    assert found.context == "graph"  # round 3's (newest) re-profile, not round 2's 20 %
    assert "100% of the 103268 GPU events" in found.reason
    assert "rounds/3/profile/profile.json" in found.reason
    # the L2 from the analyze profile: the re-profiles do not see inside the graphed stage
    assert found.l2 == "cold" and "MB of other work" in found.reason
    assert "(63 calls per run, profile/profile.json)" in found.reason

    below = make_run(
        tmp_path / "b", [base, graphed(base, "model.model", GRAPH_SHARE - 0.01)], DECODER
    )
    assert timing_context.for_target(below, "t").context == "eager"


def synthetic(between_mb: float, calls: int = 10) -> dict[str, Any]:
    """A profile in which one instance of ``Block`` (2 instances, ``calls`` calls each per run)
    has ``between_mb`` MB of other work between two of its calls."""
    own = 4 * MB  # one instance's bytes per run (weights + io over its calls)
    others = between_mb * MB * calls
    return {
        "classes": [
            {
                "cls": "Block",
                "is_leaf": False,
                "instances": 2,
                "work": [
                    {
                        "group": "net.blocks.*",
                        "phase": "decode",
                        "instances": 2,
                        "calls": 2 * calls,
                        "weight_bytes": 2 * own,
                        "io_bytes": 0,
                    }
                ],
            },
            {
                "cls": "Linear",
                "is_leaf": True,
                "instances": 3,
                "work": [{"group": "net.*", "weight_bytes": 2 * own + others, "io_bytes": 0}],
            },
        ],
        "kernel_view": {
            "timeline": {"stages": [{"stage": "net", "events": 100, "graph_events": 0}]}
        },
    }


BLOCK = {"id": "t", "module_class": "Block", "capture": {"qualname": "net.blocks.1"}}


@pytest.mark.parametrize("arch, l2_mb", [("sm_80", 40.0), ("sm_90", 50.0), ("sm_120", 48.0)])
def test_cold_when_the_work_between_two_calls_exceeds_the_gpus_l2(tmp_path, arch, l2_mb):
    for between, l2 in ((60.0, "cold"), (30.0, "warm"), (l2_mb + 1, "cold"), (l2_mb - 1, "warm")):
        found = timing_context.select([("p", synthetic(between))], BLOCK, l2_mb)
        assert found.l2 == l2, (arch, between, found.reason)
        assert f"about {between:,.0f} MB of other work" in found.reason
        assert f"the {l2_mb:g} MB L2" in found.reason
    # 45 MB between two calls: cold on an A100 (40 MB), warm on an H100 (50 MB) and an RTX
    # 5070 Ti (48 MB)
    found = timing_context.select([("p", synthetic(45.0))], BLOCK, l2_mb)
    assert found.l2 == ("cold" if l2_mb < 45 else "warm")


def test_missing_facts_choose_eager_and_warm_and_say_which(tmp_path):
    bare = {"id": "t", "module_class": "Nowhere"}
    found = timing_context.select([], bare, 48.0)
    assert (found.context, found.l2) == ("eager", "warm")
    assert (
        "no captured instance" in found.reason and "no profile of the whole model" in found.reason
    )
    found = timing_context.select([("p", synthetic(100.0))], BLOCK, None)
    assert found.l2 == "warm" and "L2 size is unknown" in found.reason
    other = {**BLOCK, "capture": {"qualname": "elsewhere.blocks.0"}}  # no stage, no group
    found = timing_context.select([("p", synthetic(100.0))], other, 48.0)
    assert found.context == "eager" and "no timeline stage of p holds elsewhere" in found.reason
    assert found.l2 == "warm"
    broken = make_run(tmp_path, [{"classes": "not a list"}], BLOCK)
    assert timing_context.for_target(broken, "t").context == "eager"  # never raises
    assert TimingContext().kwargs()["l2_flush"] is False


def test_stage_of_takes_the_innermost_stage_and_entrypoint_suffixes():
    timeline = {
        "stages": [
            {"stage": "model", "events": 10, "graph_events": 0},
            {"stage": "model.lm", "events": 30, "graph_events": 30},
            {"stage": "model.lm.forward_step", "events": 70, "graph_events": 70},
            {"stage": "model.dit.layers.*", "events": 50, "graph_events": 0},
            {"stage": "(no stage)", "events": 5, "graph_events": 5},
        ]
    }
    methods = {"forward_step"}
    assert timing_context.stage_of(timeline, "model.lm.layers.3", methods) == ("model.lm", 100, 100)
    assert timing_context.stage_of(timeline, "model.dit.layers.7.mlp", methods)[0] == (
        "model.dit.layers.*"
    )
    assert timing_context.stage_of(timeline, "model.vae", methods) == ("model", 10, 0)
    assert timing_context.stage_of(timeline, "other", methods) is None


# ------------------------------------------------------------------ graph timing, simulated


class FakeGPU:
    """A GPU on a simulated clock: work done eagerly moves the clock, work done while a graph
    is captured is recorded in the graph, and a replay moves the clock by the graph's work."""

    def __init__(self, clock: Clock, call_ms: float, flush_ms: float) -> None:
        self.clock, self.call_ms, self.flush_ms = clock, call_ms, flush_ms
        self.capturing: FakeGraph | None = None
        self.calls = 0

    def work(self, ms: float) -> None:
        if self.capturing is not None:
            self.capturing.ms += ms
        else:
            self.clock.sleep(ms / 1000)

    def call(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.work(self.call_ms)
        return x + 1


class FakeGraph:
    def __init__(self) -> None:
        self.ms = 0.0


class FakeEvent:
    def __init__(self, enable_timing: bool = True) -> None:
        self.t = 0.0


@pytest.fixture
def gpu(monkeypatch):
    clock = Clock()
    fake = FakeGPU(clock, call_ms=0.015, flush_ms=0.125)

    @contextlib.contextmanager
    def capture(graph: FakeGraph, stream: Any = None, pool: Any = None):
        fake.capturing = graph
        try:
            yield
        finally:
            fake.capturing = None

    class Stream:
        def wait_stream(self, other: Any) -> None:
            pass

    def record(event: FakeEvent, stream: Any = None) -> None:
        event.t = clock.now

    for name, value in (
        ("_CUDAGraph", FakeGraph),
        ("_graph_capture", capture),
        ("_replay", lambda graph: clock.sleep(graph.ms / 1000)),
        ("_Stream", Stream),
        ("_on_stream", lambda stream: contextlib.nullcontext()),
        ("_set_stream", lambda stream: None),
        ("_current_stream", Stream),
        ("_synchronize", lambda: None),
        ("_Event", FakeEvent),
        ("_record", record),
        ("_elapsed", lambda a, b: (b.t - a.t) * 1000),
        ("_perf_counter", clock.perf_counter),
        ("_gpu_sleep", lambda cycles: clock.sleep(cycles / 1e9)),
        ("_SLEEP_CYCLES_PER_US", 1000.0),
        ("flush_l2", lambda: fake.work(fake.flush_ms)),
        ("ensure_clocks", lambda: 1.0),
        ("clock_state", lambda: 1.0),
        ("_mem_get_info", lambda: (8 << 30, 16 << 30)),
    ):
        monkeypatch.setattr(bench, name, value)
    return fake


def test_graph_timing_subtracts_the_flush_only_graph(gpu):
    x = torch.zeros(4)
    warm = bench.time_call(gpu.call, (x,), {}, context=bench.GRAPH, target_ms=6.0)
    assert warm["context"] == bench.GRAPH and warm["calls_per_graph"] == bench.GRAPH_CALLS
    assert warm["median_ms"] == pytest.approx(gpu.call_ms)
    # 6 ms of timed calls per measurement: 6 / (12 x 0.015) replays, within the limits
    assert warm["replays"] == int(6.0 / (bench.GRAPH_CALLS * gpu.call_ms))
    assert warm["iters"] == warm["replays"] * bench.GRAPH_CALLS
    cold = bench.time_call(gpu.call, (x,), {}, context=bench.GRAPH, l2_flush=True)
    assert cold["median_ms"] == pytest.approx(gpu.call_ms)  # the flushes subtracted
    assert cold["flush_ms"] == pytest.approx(gpu.flush_ms)
    # the 150 ms default of timed graphs (the flushes in them included)
    assert cold["replays"] == int(150.0 / (bench.GRAPH_CALLS * (gpu.call_ms + gpu.flush_ms)))
    assert bench.per_call_ms([1.68, 1.70], [1.5, 1.5], 12) == pytest.approx([0.015, 0.2 / 12])
    assert bench.per_call_ms([1.2], None, 12) == pytest.approx([0.1])


def test_a_capture_that_fails_says_why_in_one_line(gpu):
    def syncs(x: torch.Tensor) -> torch.Tensor:
        if gpu.capturing is not None:
            raise RuntimeError("CUDA error: operation not permitted when stream is capturing\n...")
        return x

    with pytest.raises(bench.GraphUnavailable) as caught:
        bench.time_call(syncs, (torch.zeros(2),), {}, context=bench.GRAPH)
    assert str(caught.value) == (
        "RuntimeError: CUDA error: operation not permitted when stream is capturing"
    )
    assert bench.graph_probe(syncs, (torch.zeros(2),), {}) == str(caught.value)
    assert bench.graph_probe(gpu.call, (torch.zeros(2),), {}) is None
    with pytest.raises(ValueError, match="timing context"):
        bench.time_call(gpu.call, (torch.zeros(2),), {}, context="compiled")


class Cache:
    """Mutable state in a call's arguments (a KV cache object)."""

    def __init__(self) -> None:
        self.k = torch.zeros(4)


def test_inputs_whose_tensors_a_call_replaces_cannot_be_graph_timed(gpu):
    def in_place(x: torch.Tensor, cache: Cache) -> torch.Tensor:
        cache.k.add_(x)  # a static cache: the graph replays the write
        return gpu.call(x)

    def grows(x: torch.Tensor, cache: Cache) -> torch.Tensor:
        cache.k = torch.cat([cache.k, x])  # a grown cache: a replay reads the captured one
        return gpu.call(x)

    args = (torch.ones(4), Cache())
    result = bench.time_call(in_place, args, {}, context=bench.GRAPH, target_ms=3.0)
    assert result["median_ms"] == pytest.approx(gpu.call_ms)
    assert torch.equal(args[1].k, torch.zeros(4))  # timed on copies: the inputs untouched
    expected = "the call replaces `args[1].k` of its inputs instead of updating it in place"
    with pytest.raises(
        bench.GraphUnavailable, match=expected.replace("[", r"\[").replace("]", r"\]")
    ):
        bench.time_call(grows, args, {}, context=bench.GRAPH)
    assert expected in str(bench.graph_probe(grows, args, {}))


# ------------------------------------------------------------------ the evaluator's fields


def test_context_fields_report_both_contexts_and_why_one_is_unavailable():
    reports = [
        {
            "calls_per_run": 30,
            "timing": {
                "graph": {"ref_ms": 0.028, "new_ms": 0.014, "speedup": 2.0},
                "eager": {"ref_ms": 0.047, "new_ms": 0.074, "speedup": 0.635},
            },
        },
        {
            "calls_per_run": 10,
            "timing": {
                "graph": {"ref_ms": 0.1, "new_ms": 0.1, "speedup": 1.0},
                "eager": {"ref_ms": 0.12, "new_ms": 0.11, "speedup": 1.091},
            },
        },
        {"calls_per_run": 0},  # correctness only: not timed
    ]
    fields = evaluate.context_fields(reports, "graph", True, "graph: 100% of ...")
    assert fields == {
        "context": "graph",
        "l2": "cold",
        "context_reason": "graph: 100% of ...",
        "speedup_by_context": {
            "eager": round((30 * 0.047 + 10 * 0.12) / (30 * 0.074 + 10 * 0.11), 3),
            "graph": round((30 * 0.028 + 10 * 0.1) / (30 * 0.014 + 10 * 0.1), 3),
        },
    }
    for r in reports[:2]:  # a capture of the candidate failed: eager, and the warning
        r["timing"] = {"eager": r["timing"]["eager"], "graph": "unavailable (candidate, ...)"}
    fields = evaluate.context_fields(reports, "eager", False, "why")
    assert fields["speedup_by_context"]["graph"] == "unavailable (candidate, ...)"
    assert fields["l2"] == "warm" and isinstance(fields["speedup_by_context"]["eager"], float)
    reports[1]["timing"] = {"graph": reports[0]["timing"]["eager"]}  # not every case: none
    assert "eager" not in evaluate.context_fields(reports, "eager", False, "")["speedup_by_context"]


def test_the_probe_names_the_first_call_that_cannot_be_captured(monkeypatch):
    probes = {"ref0": None, "new0": None, "ref1": None, "new1": "RuntimeError: host sync"}
    monkeypatch.setattr(bench, "graph_probe", lambda fn, args, kwargs: probes[fn])
    timed = [(0, ("ref0", "new0"), {"args": (), "kwargs": {}})]
    assert evaluate._graph_unavailable(timed) is None
    timed.append((3, ("ref1", "new1"), {"args": (), "kwargs": {}}))
    assert evaluate._graph_unavailable(timed) == "candidate, case 3: RuntimeError: host sync"


def test_the_other_context_of_a_winner(monkeypatch):
    def rounds(ref, new, args, kwargs, *, rounds, l2_flush, verify, context):
        assert rounds == evaluate.OTHER_ROUNDS and verify == (context == bench.GRAPH)
        if new == "breaks":
            raise bench.GraphUnavailable("RuntimeError: host sync")
        checked = {"iteration": 5, "failures": [{"name": "output", "error": "x"}]}
        return [{"median_ms": 2.0}], [{"median_ms": 1.0}], checked if new == "wrong" else None

    monkeypatch.setattr(bench, "timing_rounds", rounds)
    monkeypatch.setattr(bench, "graph_probe", lambda fn, args, kwargs: None)

    def timed(new: str) -> list[Any]:
        report = {"timing": {"eager": {"ref_ms": 1.0, "new_ms": 0.5, "speedup": 2.0}}}
        return [evaluate._Rounds(0, report, {"args": (), "kwargs": {}}, ("ref", new), [], [])]

    ok = timed("fine")
    assert evaluate._other_context(ok, bench.GRAPH, True, None) is None
    assert ok[0].report["timing"]["graph"] == {"ref_ms": 2.0, "new_ms": 1.0, "speedup": 2.0}
    broken = timed("breaks")
    assert evaluate._other_context(broken, bench.GRAPH, True, None) == (
        "case 0: RuntimeError: host sync"
    )
    assert "graph" not in broken[0].report["timing"]
    wrong = evaluate._other_context(timed("wrong"), bench.GRAPH, False, None)
    assert wrong is not None and "the output of call #5 of a graph replay differs" in wrong
    known = timed("fine")  # the primary context's probe failed already: not timed again
    assert evaluate._other_context(known, bench.GRAPH, False, "reference, ...") == "reference, ..."


# ------------------------------------------------------------------ the re-check's context


def test_the_recheck_times_in_the_verdicts_context(monkeypatch, tmp_path):
    from kernel_agent import orchestrator

    rec = {"correct": True, "speedup": 2.0, "context": "graph", "l2": "cold", "cases": []}
    verdict = orchestrator._verdict(rec)
    assert verdict is not None and (verdict["context"], verdict["l2"]) == ("graph", "cold")
    commands: list[list[str]] = []

    def spawn(cmd: list[str], timeout: float, nonce: str) -> dict[str, Any]:
        commands.append(cmd)
        return {"status": "error", "error": "stop here"}

    monkeypatch.setattr(recheck, "_spawn", spawn)
    monkeypatch.setattr(recheck, "gpu_lock", lambda: contextlib.nullcontext(0))
    recheck.run_recheck(tmp_path / "c.pt", tmp_path / "k.py", verdict=verdict)
    assert commands[0][-3:] == ["--context", "graph", "--l2-flush"]
    result: dict[str, Any] = {}
    cases = [{"signature": "x", "count": 1}]
    ref = {"ms": [[1.0, 0.01]], "device": "cuda"}
    cand = {"timing_error": "graph timing unavailable: RuntimeError: host sync"}
    recheck._summarise(result, cases, [(0, 1)], [[]], ref, cand)
    assert "speedup" not in result and result["timing"].startswith("skipped: graph timing")


# ------------------------------------------------------------------ GPU


@pytest.mark.gpu
def test_graph_timing_leaves_out_the_host_time_of_launch_bound_calls():
    """40 tiny kernels per call: eagerly the GPU waits for each launch, in a graph it does
    not."""
    x = torch.zeros(256, device="cuda")

    def many(t: torch.Tensor) -> torch.Tensor:
        for _ in range(40):
            t = t + 1
        return t

    eager = bench.time_call(many, (x,), {}, target_ms=30.0)
    graph = bench.time_call(many, (x,), {}, target_ms=30.0, context=bench.GRAPH)
    assert graph["context"] == bench.GRAPH and graph["calls_per_graph"] == bench.GRAPH_CALLS
    assert graph["median_ms"] < 0.6 * eager["median_ms"], (graph, eager)


@pytest.mark.gpu
def test_graph_cold_l2_agrees_with_a_flush_outside_the_timing():
    """The per-call time of a weight-streaming GEMM in an L2-flushing graph (less the
    flush-only graph) against one call captured alone and timed after a flush outside its
    events (a 352 x 1024 x 2560 bf16 GEMM: 5 MB of weights, L2-resident when warm)."""
    torch.manual_seed(0)
    w = torch.randn(2560, 1024, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(352, 1024, device="cuda", dtype=torch.bfloat16)

    def gemm(a: torch.Tensor) -> torch.Tensor:
        return a @ w.t()

    cold = min(
        bench.time_call(gemm, (x,), {}, context=bench.GRAPH, l2_flush=True)["median_ms"]
        for _ in range(3)
    )
    gemm(x)  # the cuBLAS handle and workspace before the capture
    torch.cuda.synchronize()
    alone = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    with torch.cuda.graph(alone, stream=side):
        gemm(x)
    times = []
    for _ in range(50):
        bench.flush_l2()  # outside the events; it keeps the GPU busy while they are queued
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        alone.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    reference = sorted(times)[len(times) // 2]
    assert 0.8 * reference < cold < 1.15 * reference, (cold, reference)


class StaticCache:
    def __init__(self, n: int) -> None:
        self.k = torch.zeros(n, 64, device="cuda")


@pytest.mark.gpu
def test_graph_timing_of_caches_in_the_arguments():
    """A cache updated in place is restored before every replay and its timed output checks
    out; a cache the call replaces cannot be replayed from a graph."""
    from kernel_agent.kernels import compare

    w = torch.randn(64, 64, device="cuda")

    def static(x: torch.Tensor, cache: StaticCache) -> torch.Tensor:
        cache.k[3].copy_(x @ w)  # the new position, in place
        return torch.softmax(cache.k @ x, dim=0)

    def grown(x: torch.Tensor, cache: StaticCache) -> torch.Tensor:
        cache.k = torch.cat([cache.k, (x @ w)[None]])
        return cache.k.sum(0)

    x = torch.randn(64, device="cuda")
    args = (x, StaticCache(8))
    assert compare.TIER == compare.EXACT_TIER
    _, new = bench.compare_timing(static, static, args, {}, context=bench.GRAPH)
    assert new["context"] == bench.GRAPH and new["timed_output"]["failures"] == []
    assert not args[1].k.any()  # timed on copies
    assert "replaces `args[1].k`" in str(bench.graph_probe(grown, args, {}))
    current = torch.cuda.current_stream()  # a capture that fails leaves it as it was
    why = bench.graph_probe(lambda t: t * t.sum().item(), (x,), {})  # a host sync
    assert why is not None and "operation not permitted when stream is capturing" in why, why
    assert torch.cuda.current_stream() == current


@pytest.mark.gpu
def test_the_fp8_gemm_verdict_flips_from_eager_to_graph_with_a_cold_l2(tmp_path):
    """VoxCPM2's slice 8 (#226): an FP8 W8A8 GEMM of two Triton launches at M = 352 (the
    LocDiT QKV, 1024 -> 2560) loses to cuBLAS bf16 timed eagerly (launch bound) and wins in
    a CUDA graph with a cold L2, where the model runs it (RTX 5070 Ti: 0.58x eager warm,
    0.66x eager cold, 2.01x graph cold)."""
    from kernel_agent import selftest, toolchain
    from kernel_agent.agent import prompts
    from kernel_agent.kernels.evaluate import run_evaluation

    if not selftest.w8a8_supported(toolchain.setup()):
        pytest.skip("needs the triton backend on sm_89+")
    capture = selftest.make_linear_capture(
        tmp_path / "qkv.pt",
        1024,
        2560,
        [((32, 11), 540)],
        tier="near-lossless",
        precision="fp8_w8a8",
    )
    example = prompts.EXAMPLES_DIR / "triton_fp8_w8a8_gemm.py"
    eager = run_evaluation(capture, example, context_reason="test: eager")
    assert eager["status"] == "ok", eager
    assert (eager["context"], eager["l2"], eager["context_reason"]) == (
        "eager",
        "warm",
        "test: eager",
    )
    graph = run_evaluation(capture, example, context="graph", l2_flush=True)
    assert graph["status"] == "ok" and graph["reference_check"] == "ok", graph
    assert (graph["context"], graph["l2"]) == ("graph", "cold")
    assert eager["speedup"] < 1.0 < 1.3 < graph["speedup"], (eager["speedup"], graph["speedup"])
    both = graph["speedup_by_context"]  # a winner: timed eagerly as well
    assert both["graph"] == pytest.approx(graph["speedup"], rel=0.01), both  # rounding
    assert both["eager"] < 1.0, both
    assert set(graph["cases"][0]["timing"]) == {"graph", "eager"}
