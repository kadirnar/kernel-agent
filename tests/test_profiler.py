"""ModuleTimer bookkeeping on optimised models (#86): compiled modules, CUDA graphs captured
and replayed inside module calls, unmatched hooks and start/end events that do not pair."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.profiling import profiler
from kernel_agent.profiling.profiler import ModuleTimer, _Call, profile_workload, summarize
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec

KERNEL_VIEW = {
    "gpu_busy_ms": 1.0,
    "gpu_busy_fraction": 0.5,
    "kernel_launches": 1,
    "avg_kernel_us": 1.0,
    "kernels": [],
    "aten_ops": [],
}


class Block(nn.Module):
    """A user-defined class: Dynamo guards that its hooks are empty."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x)) + x


class Stack(nn.Module):
    """A user-defined container: its forward is the frame Dynamo compiles."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(d), Block(d)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Flaky(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise ValueError("boom")


class Tolerant(nn.Module):
    """Calls a submodule that raises (its post-hook never runs) and carries on."""

    def __init__(self) -> None:
        super().__init__()
        self.flaky = Flaky()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with contextlib.suppress(ValueError):
            self.flaky(x)
        return self.lin(x)


class _Event:
    """Stands in for ``torch.Event`` / ``torch.cuda.Event``: two kinds that cannot be paired."""

    def elapsed_time(self, other: Any) -> float:
        if type(other) is not type(self):
            raise RuntimeError("expected other to be a torch.Event object")
        return 1.0


class _CudaEvent(_Event):
    pass


class _Captured(_Event):
    """An event recorded into a CUDA graph: never timed."""

    def elapsed_time(self, other: Any) -> float:
        raise RuntimeError("CUDA error: invalid argument")


def test_a_call_without_post_hook_stays_untimed_and_its_parent_is_timed():
    model = Tolerant()
    with torch.inference_mode(), ModuleTimer({"model": model}, cuda=False) as timer:
        model(torch.randn(2, 4))
    calls = {c.qualname: c for c in timer.calls}
    assert set(calls) == {"model", "model.flaky", "model.lin"}
    # The parent's post-hook closes the parent, not the call left open above it.
    assert calls["model"].end is not None and calls["model.lin"].end is not None
    assert calls["model.flaky"].end is None
    stats = {s.cls: s for s in timer.class_stats()}
    assert stats["Tolerant"].inclusive_ms > 0 and stats["Flaky"].inclusive_ms == 0
    assert timer.gaps() == {"untimed": 1}


def test_unmatched_post_hooks_and_unpaired_events_are_counted_not_fatal():
    lin = nn.Linear(4, 4)
    timer = ModuleTimer({"m": lin}, methods={}, cuda=False)
    timer._post(lin, "forward", (), {}, None)  # no open call of this module
    assert timer.skipped["unmatched_post"] == 1 and not timer.calls
    timer.calls += [
        _Call("m", "Linear", "()", -1, 0.0, 0.002),  # 2 ms
        _Call("m", "Linear", "()", -1, _Event(), _CudaEvent()),  # the #86 crash
        _Call("m", "Linear", "()", -1, _Captured(), _Captured()),
        _Call("m", "Linear", "()", -1, 0.0),  # never ended
    ]
    (stat,) = timer.class_stats()
    assert stat.calls == 4 and stat.inclusive_ms == pytest.approx(2.0)
    assert timer.gaps() == {"unmatched_post": 1, "untimed": 3}


def test_calls_during_a_cuda_graph_capture_are_skipped(monkeypatch):
    capturing = [False]
    monkeypatch.setattr(profiler, "_capturing", lambda: capturing[0])

    class Capturing(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            capturing[0] = True
            try:
                return self.body(x)
            finally:
                capturing[0] = False

    model = nn.Sequential(Capturing(), nn.Linear(4, 4))
    with torch.inference_mode(), ModuleTimer({"model": model}, cuda=False) as timer:
        model(torch.randn(2, 4))
    assert [c.qualname for c in timer.calls] == ["model", "model.0", "model.1"]
    timer.class_stats()
    assert timer.gaps() == {"capture_calls": 3}  # body, body.0, body.1


def test_compiled_modules_are_timed_as_one_call_without_recompiling():
    graphs: list[Any] = []

    def backend(gm: torch.fx.GraphModule, example_inputs: Any) -> Any:
        graphs.append(gm)
        return gm.forward

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = Block()
            self.tail = torch.compile(Stack(), backend=backend)
            # registered as a plain module, called through a compiled wrapper the
            # timer cannot see (not a submodule): its hooks run inside Dynamo's context
            self.hidden = Stack()
            self.wrapped = [torch.compile(self.hidden, backend=backend)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.wrapped[0](self.tail(self.head(x)))

    model, x = Net(), torch.randn(2, 8)
    with torch.no_grad():
        model(x)
        compiled = len(graphs)
        with ModuleTimer({"model": model}, cuda=False) as timer:
            model(x)
        assert len(graphs) == compiled  # the hooks made nothing recompile
        names = [c.qualname for c in timer.calls]
        assert "model.tail" in names and not any("_orig_mod" in n for n in names)
        # Dynamo compiled no hook frame (a hook it compiles records nothing)
        assert "model.hidden" in names
        assert timer.compiled == ["model.tail"]
        model(x)  # hooks gone: the cached code runs again
        assert len(graphs) == compiled
    timer.class_stats()
    assert timer.gaps() == {"compiled": ["model.tail"]}  # every call timed


def test_summary_lists_what_the_module_view_does_not_time():
    profile = {
        "module_calls": 3,
        "module_gaps": {
            "compiled": ["model.audio_vae.decoder.inner"],
            "graph_replays": {"model.feat_decoder": 60},
            "capture_calls": 12,
            "untimed": 2,
        },
        "classes": [],
        "kernel_view": KERNEL_VIEW,
    }
    text = summarize(profile, 2.0)
    assert "`torch.compile`d modules" in text and "`model.audio_vae.decoder.inner`" in text
    assert "`model.feat_decoder` ×60" in text
    assert "12 during a CUDA-graph capture" in text and "2 calls without a usable" in text
    assert "kernel view below covers these regions" in text
    profile["module_gaps"] = {}
    assert "kernel view below covers" not in summarize(profile, 2.0)


# ------------------------------------------------------------------ kernel view guard (#95)

#: The round-3 VoxCPM2 re-profile (PR #89): 1056.3 ms of GPU kernels for a 522.9 ms run
#: (another process held ~13 GB of the GPU); another run gave 551 ms for the same launches.
E2E_MS = 522.9


def test_kernel_problems_on_synthetic_numbers():
    assert profiler.kernel_problems([520.4, 524.0], E2E_MS) == []
    assert profiler.kernel_problems([540.0, 545.0], E2E_MS) == []  # 103 %: within 105 %
    assert profiler.kernel_problems([500.0, 590.0], None) == []  # 18 % apart: within 20 %
    assert profiler.kernel_problems([], E2E_MS) == []
    (share,) = profiler.kernel_problems([1056.3, 1058.0], E2E_MS)
    assert "1056.3 ms is 202% of the 522.9 ms" in share
    (spread,) = profiler.kernel_problems([551.0, 1056.3], None)
    assert "551.0 vs 1056.3" in spread and "92% apart" in spread
    assert len(profiler.kernel_problems([560.0, 1056.3], E2E_MS)) == 2  # 107 % and 89 % apart


class Counted:
    """A workload stand-in for the guard: only ``run`` is called."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def run(self, inputs: Any) -> None:
        self.log.append("run")


@pytest.fixture
def guarded(monkeypatch, tmp_path):
    """CUDA on paper, without a GPU: the GPU lock in ``tmp_path``, fixed clock probes, GPU
    telemetry that sees another process, and a ``kernel_profile`` whose views have the GPU
    kernel times put in ``busy``, in turn. Returns (busy, log of the calls)."""
    from kernel_agent import gpulock

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiler, "synchronize", lambda: None)
    monkeypatch.setattr(gpulock, "CACHE_DIR", tmp_path)
    for name in (gpulock.ENV, gpulock.GPUS_ENV, gpulock.INDEX_ENV, "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gpulock, "_nvidia_smi", lambda: [gpulock.GPU(0, "fake", "GPU-0")])
    gpulock.pool.cache_clear()
    busy: list[float] = []
    log: list[str] = []
    probes = iter([0.97, 0.95, 0.99, 0.98])

    def full_clocks() -> float:
        log.append("ensure_clocks")
        return next(probes)

    class Monitor:
        def sample(self, label: str = "", *, loaded: bool = True) -> None:
            assert loaded == (label == "after")  # only right after a run is under load
            log.append(f"sample {label}")

        def processes(self) -> list[dict[str, Any]]:
            return [{"pid": 4242, "used_mib": 12928, "name": "python"}]

        def summary(self) -> dict[str, Any]:
            return {"sm_mhz": [2700, 2880, 2902], "throttled": False}

    def kernel_profile(workload: Any, inputs: Any, *, after: Any = None) -> dict[str, Any]:
        assert "gpu" in gpulock._held_gpus()  # profiled under the GPU lock
        log.append("profile")
        after()  # right after the run, before the profiler stops
        ms = busy.pop(0)
        return {**KERNEL_VIEW, "gpu_busy_ms": ms, "wall_ms": ms + 50.0}

    monkeypatch.setattr(profiler, "_full_clocks", full_clocks)
    monkeypatch.setattr(profiler.telemetry, "Monitor", Monitor)
    monkeypatch.setattr(profiler, "kernel_profile", kernel_profile)
    yield busy, log
    assert not gpulock._held_gpus()  # released
    gpulock.pool.cache_clear()


def test_the_kernel_view_is_locked_warmed_up_clock_guarded_and_checked(guarded):
    busy, log = guarded
    busy += [524.0, 520.4]
    view = profiler.guarded_kernel_profile(Counted(log), None, reference_ms=E2E_MS)
    run = ["ensure_clocks", "sample before", "profile", "sample after"]
    assert log == ["run", *run, *run]  # a warm-up, then the clock guard before every run
    assert view["reliable"] and "unreliable" not in view
    assert view["gpu_busy_ms"] == 520.4  # the run with the least kernel time
    assert view["reference_ms"] == E2E_MS and view["busy_share"] == 0.995
    (attempt,) = view["attempts"]
    assert attempt["gpu_busy_ms"] == [524.0, 520.4] and attempt["clock"] == [0.97, 0.95]
    assert attempt["other_gpu_processes"] == [{"pid": 4242, "used_mib": 12928, "name": "python"}]
    assert attempt["gpu"]["sm_mhz"] == [2700, 2880, 2902] and "problems" not in attempt


def test_a_suspicious_kernel_view_is_profiled_once_more(guarded):
    busy, log = guarded
    busy += [1056.3, 1058.0, 512.0, 509.5]
    view = profiler.guarded_kernel_profile(Counted(log), None, reference_ms=E2E_MS)
    assert log.count("run") == 2 and log.count("ensure_clocks") == 4
    assert view["reliable"] and view["gpu_busy_ms"] == 509.5
    first, second = view["attempts"]
    assert "202%" in first["problems"][0] and "problems" not in second


def test_a_kernel_view_that_fails_twice_is_marked_unreliable(guarded):
    busy, log = guarded
    busy += [1056.3, 1058.0, 510.0, 1056.3]
    view = profiler.guarded_kernel_profile(Counted(log), None, reference_ms=E2E_MS)
    assert log.count("profile") == 4  # two attempts, no third
    assert view["reliable"] is False and view["gpu_busy_ms"] == 510.0
    assert view["unreliable"] == view["attempts"][1]["problems"]
    assert view["unreliable"] == [
        "profiled runs disagree: 510.0 vs 1056.3 ms of GPU kernel time (107% apart)"
    ]

    profile = {"module_calls": 3, "classes": [], "kernel_view": view}
    text = profiler.summarize(profile, E2E_MS)
    assert "**UNRELIABLE kernel view** (profiled runs disagree: 510.0 vs 1056.3" in text
    assert "GPU bound: faster kernels" not in text and "LAUNCH/CPU BOUND" not in text
    assert "profiled 2 times (the first failed its sanity check)" in text
    assert "other processes on the GPU: python 4242 (12928 MiB)" in text
    assert "DRAM clock probe before each run 0.99 / 0.98" in text
    assert "SM clock 2700-2902 MHz at the end of the runs" in text


def test_summary_conclusions_follow_the_check():
    def text(kv: dict[str, Any], ms: float) -> str:
        return profiler.summarize({"module_calls": 1, "classes": [], "kernel_view": kv}, ms)

    reliable = {**KERNEL_VIEW, "gpu_busy_ms": 500.0, "reliable": True, "attempts": []}
    assert "GPU bound: faster kernels" in text(reliable, E2E_MS)
    assert "LAUNCH/CPU BOUND" in text({**reliable, "gpu_busy_ms": 100.0}, E2E_MS)
    # a profile from before the guard (no `reliable`): its share alone is checked
    old = {**KERNEL_VIEW, "gpu_busy_ms": 1056.3}
    assert "UNRELIABLE kernel view** (GPU kernel time 1056.3 ms is 202%" in text(old, E2E_MS)
    assert "GPU bound: faster kernels" in text({**old, "gpu_busy_ms": 500.0}, E2E_MS)
    assert "kernel view: under the GPU lock" not in text(old, E2E_MS)


def test_the_scheduler_ignores_an_unreliable_busy_fraction(tmp_path):
    from kernel_agent.scheduler import gpu_busy
    from kernel_agent.workspace import RunDir, write_json

    run = RunDir(tmp_path)
    write_json(run.profile_dir / "profile.json", {"kernel_view": {"gpu_busy_fraction": 0.8}})
    assert gpu_busy(run) == 0.8
    view = {"gpu_busy_fraction": 0.8, "reliable": False}
    write_json(run.profile_dir / "profile.json", {"kernel_view": view})
    assert gpu_busy(run) is None
    # the share of the run without the profiler, not of the slower profiled run
    view = {"gpu_busy_fraction": 0.131, "busy_share": 0.471, "reliable": True}
    write_json(run.profile_dir / "profile.json", {"kernel_view": view})
    assert gpu_busy(run) == 0.471
    write_json(run.profile_dir / "profile.json", {"kernel_view": {**view, "busy_share": 1.03}})
    assert gpu_busy(run) == 1.0


def test_without_cuda_one_unguarded_attempt(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    log: list[str] = []
    views = iter([{**KERNEL_VIEW, "gpu_busy_ms": 3.0}, {**KERNEL_VIEW, "gpu_busy_ms": 9.0}])
    monkeypatch.setattr(profiler, "kernel_profile", lambda w, i, after: next(views))
    view = profiler.guarded_kernel_profile(Counted(log), None, reference_ms=None)
    assert log == ["run"] and view["gpu_busy_ms"] == 3.0
    assert view["reliable"] is False  # 200 % apart, and no second attempt
    assert view["attempts"][0]["clock"] == [] and view["attempts"][0]["gpu"] == {}


# ------------------------------------------------------------------ GPU


class Graphed(nn.Module):
    """Captures its body in a CUDA graph on the first call per shape, then replays it
    (like VoxCPM's ``graph_prefill`` / ``hoist_cfm_invariants`` transforms)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.body = nn.Sequential(Block(d), Block(d))
        self.graphs: dict[Any, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        entry = self.graphs.get(x.shape)
        if entry is None:
            static = x.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self.body(static)  # warm-up outside the capture
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.body(static)
            entry = self.graphs[x.shape] = (graph, static, out)
        graph, static, out = entry
        static.copy_(x)
        graph.replay()
        return out.clone()


class GraphedModel(nn.Module):
    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.graphed = Graphed(d)
        self.tail = torch.compile(nn.Sequential(Block(d), Block(d)), backend="eager")
        self.head = nn.Linear(d, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tail(self.graphed(x)))


class GraphedWorkload(Workload):
    modality = Modality.LLM

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = GraphedModel().cuda().eval()

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> Any:
        return torch.randn(4, 64, device="cuda")

    def run(self, inputs: Any) -> Any:
        return self.model(inputs).cpu()

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.allclose(reference, candidate)))


@pytest.mark.gpu
def test_profile_with_a_cuda_graph_captured_inside_a_module_call():
    w = GraphedWorkload(WorkloadSpec(repo_id="toy/graphed", modality="llm"))
    w.load()
    inputs = w.make_inputs()
    with torch.inference_mode():
        w.model.tail(w.model.head.weight.new_zeros(4, 64))  # compiled before the profile
    # The graph of `graphed` is captured during the profiled run, then replayed.
    profile = profile_workload(w, inputs)
    gaps = profile["module_gaps"]
    assert gaps["compiled"] == ["model.tail"]
    assert gaps["capture_calls"] == 5  # body, its two blocks and their linears
    assert gaps["graph_replays"] == {"model.graphed": 1}
    assert "untimed" not in gaps and "unmatched_post" not in gaps
    classes = {c["cls"]: c for c in profile["classes"]}
    assert classes["Graphed"]["calls"] == 1 and classes["Graphed"]["inclusive_ms"] > 0
    assert classes["OptimizedModule"]["calls"] == 1
    # warm-up before the capture (side stream) is a normal call: body + 2 blocks + 2 linears
    assert classes["Block"]["calls"] == 2
    assert profile["kernel_view"]["kernel_launches"] > 0
    assert "CUDA-graph replays" in summarize(profile, 1.0)


@pytest.mark.gpu
def test_events_of_different_kinds_are_left_untimed():
    start = torch.Event(device="cuda", enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize()
    # what stopped the round-3 re-profile (torch 2.14 names torch.Event; torch 2.10 says
    # "Event device type CUDA does not match other's device type CPU.")
    with pytest.raises(RuntimeError, match=r"torch\.Event|does not match other's device type"):
        start.elapsed_time(end)
    assert ModuleTimer._elapsed_ms(_Call("m", "M", "()", -1, start, end)) is None
    same = torch.cuda.Event(enable_timing=True)
    same.record()
    torch.cuda.synchronize()
    assert ModuleTimer._elapsed_ms(_Call("m", "M", "()", -1, end, same)) is not None
