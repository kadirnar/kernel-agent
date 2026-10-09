"""Host-free generation loops (#232): kernel_agent.graphloop on a fake cuda.core and a fake
CUDA backend (CPU only): the WHILE graph's structure, the fallbacks and their reasons, the
unrolled schedule's masking on the toy decoder of examples/graph_while_decode.py (tokens
identical to the plain loop, nothing written past the stop), streaming chunks, teacher
forcing's watched callables, the unroll choice under a simulated clock, the doctor probe and
the skill links. GPU tests (marked) run the real graphs."""

from __future__ import annotations

import contextlib
import importlib.util
import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fake_clock import Clock

from kernel_agent import graphloop, probes, skills
from kernel_agent.agent.prompts import EXAMPLES_DIR

# ------------------------------------------------------------------ fake cuda.core


class FakeCondition:
    def __init__(self, graph: str, default: int | None) -> None:
        self.graph, self.default = graph, default

    def __repr__(self) -> str:
        return f"cond({self.graph})"


class FakeCoreStream:
    def __init__(self, handle: int) -> None:
        self.handle = handle


class FakeBuilder:
    def __init__(self, core: FakeCore, name: str) -> None:
        self.core, self.name = core, name
        self.stream = FakeCoreStream(next(core.handles))
        self.is_building = False

    def begin_building(self, mode: str = "relaxed") -> FakeBuilder:
        self.core.log.append(("begin", self.name))
        self.is_building = True
        return self

    def end_building(self) -> FakeBuilder:
        self.core.log.append(("end", self.name))
        self.is_building = False
        return self

    def create_condition(self, default_value: int | None = None) -> FakeCondition:
        self.core.log.append(("condition", self.name, default_value))
        return FakeCondition(self.name, default_value)

    def while_loop(self, condition: FakeCondition) -> FakeBuilder:
        self.core.log.append(("while", self.name, condition.graph))
        return FakeBuilder(self.core, "body")

    def if_then(self, condition: FakeCondition) -> FakeBuilder:
        self.core.log.append(("if", self.name, condition.graph))
        return FakeBuilder(self.core, "then")

    def complete(self) -> FakeGraph:
        self.core.log.append(("complete", self.name))
        if self.core.fail_complete:
            raise RuntimeError("cudaErrorNotSupported: a node type in a conditional body")
        return FakeGraph(self.core)

    def close(self) -> None:
        self.core.log.append(("close", self.name))


class FakeGraph:
    def __init__(self, core: FakeCore) -> None:
        self.core = core

    def upload(self, stream: Any) -> None:
        self.core.log.append(("upload",))

    def launch(self, stream: Any) -> None:
        self.core.log.append(("launch graph", stream.handle))


class FakeCore:
    """The parts of ``cuda.core`` graphloop uses; ``log`` records every call."""

    __version__ = "1.2.1"
    graph = SimpleNamespace(GraphBuilder=FakeBuilder)

    def __init__(self) -> None:
        self.log: list[tuple[Any, ...]] = []
        self.launches: list[tuple[str, str, tuple[Any, ...]]] = []
        self.handles = iter(range(500, 10_000))
        self.fail_complete = False
        core = self

        class Device:
            def __init__(self, index: int) -> None:
                self.index = index

            def set_current(self) -> None:
                pass

            def create_graph_builder(self) -> FakeBuilder:
                return FakeBuilder(core, "main")

            def create_stream(self, obj: Any) -> FakeCoreStream:
                return FakeCoreStream(obj.cuda_stream)

        self.Device = Device

    def LaunchConfig(self, grid: int, block: int) -> tuple[int, int]:
        return grid, block

    def ProgramOptions(self, **kw: Any) -> dict[str, Any]:
        return kw

    def Program(self, src: str, code_type: str, options: Any) -> Any:
        assert "cudaGraphSetConditional" in src and options["arch"] == "sm_120"
        return SimpleNamespace(compile=lambda target: SimpleNamespace(get_kernel=lambda name: name))

    def launch(self, builder: FakeBuilder, config: Any, kernel: str, *args: Any) -> None:
        self.log.append(("launch", builder.name, kernel))
        self.launches.append((builder.name, kernel, args))


class FakeEvent:
    def record(self, stream: Any) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True


class Eager:
    """A "captured" block on the CPU: replay runs it again."""

    def __init__(self, fn: Any) -> None:
        self.replay = fn


class FakeCuda(graphloop._Cuda):
    """graphloop's CUDA calls on the CPU: streams and pools are recorded (``current``,
    ``pool``: what the code under them runs on), captures run eagerly."""

    def __init__(self, core: FakeCore, driver: tuple[int, int] = (13, 0)) -> None:
        self._core, self.driver = core, driver
        self.current: Any = "caller"
        self.pool: Any = None
        self.captures = 0

    def core(self) -> Any:
        return self._core

    def driver_version(self) -> tuple[int, int] | None:
        return self.driver

    def capability(self, device: int) -> tuple[int, int]:
        return 12, 0

    def available(self) -> bool:
        return True

    def graphs(self, device: torch.device) -> bool:
        return True

    def current_stream(self, device: int) -> Any:
        return SimpleNamespace(cuda_stream=7)

    def external_stream(self, handle: int, device: int) -> Any:
        self._core.log.append(("external stream", handle))
        return f"ext{handle}"

    @contextlib.contextmanager
    def use_stream(self, stream: Any) -> Iterator[None]:
        saved, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = saved

    def mem_pool(self) -> Any:
        return "private pool"

    @contextlib.contextmanager
    def use_pool(self, pool: Any, device: int) -> Iterator[None]:
        self._core.log.append(("pool", pool))
        saved, self.pool = self.pool, pool
        try:
            yield
        finally:
            self.pool = saved

    def pool_handle(self) -> Any:
        return "graph pool"

    def capture_graph(self, fn: Any, pool: Any) -> Any:
        self.captures += 1
        return Eager(fn)

    def event(self) -> Any:
        return FakeEvent()

    def synchronize(self, device: int | None) -> None:
        pass

    def pinned(self, n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=torch.int64)

    runtime = False  # cudaGraphLaunch of torch's CUDA runtime found

    def end_capture(self, handle: int) -> None:
        self._core.log.append(("end capture", handle))

    def runtime_launch(self, graph: Any, stream: Any) -> bool:
        if self.runtime:
            self._core.log.append(("cudaGraphLaunch", stream.cuda_stream))
        return self.runtime


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
    cuda = FakeCuda(FakeCore())
    monkeypatch.setattr(graphloop, "_cuda", cuda)
    return cuda


def _counter_loop(fake: FakeCuda, stop_at: int = 3, **options: Any) -> tuple[Any, list[Any]]:
    """A loop whose step counts in ``x`` and records the stream and pool it ran under."""
    x = torch.zeros((), dtype=torch.int64)
    seen: list[Any] = []

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        seen.append(("step", fake.current, fake.pool))
        x.add_(active.to(x.dtype))

    loop = graphloop.device_loop(step, lambda: x >= stop_at, 8, device="cpu", **options)
    loop.x = x  # type: ignore[attr-defined]
    return loop, seen


# ------------------------------------------------------------------ the WHILE graph


def test_while_graph_structure_body_condition_and_pool(fake):
    chunks: list[Any] = []
    loop, seen = _counter_loop(
        fake,
        mode="while",
        warmup_runs=0,
        chunk_every=2,
        on_chunk=lambda: chunks.append(("chunk", fake.current, fake.pool)),
    )
    loop.run()
    assert loop.mode == "while" and loop.reason == "while built"
    body, then = 501, 502  # stream handles of the body and IF builders (main: 500)
    assert fake._core.log == [
        ("begin", "main"),
        ("condition", "main", 1),
        ("launch", "main", "ka_loop_begin"),
        ("while", "main", "main"),
        ("begin", "body"),
        ("external stream", body),
        ("pool", "private pool"),
        ("condition", "body", 0),
        ("launch", "body", "ka_loop_step"),
        ("if", "body", "body"),
        ("begin", "then"),
        ("external stream", then),
        ("pool", "private pool"),
        ("launch", "then", "ka_chunk_signal"),
        ("end", "then"),
        ("end", "body"),
        ("launch", "main", "ka_loop_end"),  # after the WHILE node, where the profiler sees it
        ("end", "main"),
        ("complete", "main"),
        ("upload",),
        ("launch graph", 7),  # on the caller's stream (cuda.core: no CUDA runtime found)
    ]
    # the step and on_chunk were captured on their builder's stream, from the private pool
    assert seen == [("step", f"ext{body}", "private pool")]
    assert chunks == [("chunk", f"ext{then}", "private pool")]
    launches = {kernel: (where, args) for where, kernel, args in fake._core.launches}
    loop_cond, chunk_cond, flags, n_flags, index, active, limits, every, host = launches[
        "ka_loop_step"
    ][1]
    assert (loop_cond.graph, chunk_cond.graph) == ("main", "body")
    assert (n_flags, every) == (1, 2) and flags != 0
    assert (index, active) == (loop.index.data_ptr(), loop.active.data_ptr())
    assert limits == loop._limits.data_ptr() and host == loop._host.data_ptr()
    assert launches["ka_loop_begin"][1] == (loop_cond, index, active)
    assert launches["ka_chunk_signal"][1] == (index, host)
    assert launches["ka_loop_end"][1] == (index, active, loop._status.data_ptr())
    assert loop.stats["launches"] == 1 and loop.stats["host_checks"] == 0
    # with torch's CUDA runtime the launch is a cudaGraphLaunch the profiler records
    fake.runtime = True
    loop.run()
    assert fake._core.log[-1] == ("cudaGraphLaunch", 7) and loop.stats["launches"] == 2


def test_while_graph_without_stop_flags_or_chunks(fake):
    loop = graphloop.device_loop(
        lambda i, a: None, None, 4, mode="while", warmup_runs=0, device="cpu"
    )
    loop.run()
    kinds = [entry[0] for entry in fake._core.log]
    assert "if" not in kinds and kinds.count("condition") == 1
    (args,) = [a for _, kernel, a in fake._core.launches if kernel == "ka_loop_step"]
    assert args[1:4] == (0, 0, 0) and args[-2:] == (0, 0)  # no chunk, no flags, no host


def test_the_warm_up_runs_are_host_runs_then_the_graph_is_built(fake):
    loop, seen = _counter_loop(fake, mode="while", warmup_runs=1)
    loop.run()
    assert loop.mode is None and loop.stats["host_runs"] == 1
    assert int(loop.x) == 3 and int(loop.index) == 3 and loop.stats["host_checks"] == 3
    assert all(s == ("step", "caller", None) for s in seen)  # eager, on the caller's stream
    loop.run()
    assert loop.mode == "while" and loop.stats["launches"] == 1


def test_a_failed_while_build_falls_back_with_the_reason(fake):
    fake._core.fail_complete = True
    loop, _ = _counter_loop(fake, warmup_runs=0, masked=True, unroll=2)
    loop.build()
    loop.x.fill_(0)  # the fake "captures" by running the step
    loop.run()
    assert loop.mode == "unrolled"
    assert loop.reason.startswith("while: RuntimeError: cudaErrorNotSupported")
    assert int(loop.index) == int(loop.x)

    unmasked, _ = _counter_loop(fake, warmup_runs=0)
    unmasked.run()
    assert unmasked.mode == "host"
    assert "while: RuntimeError" in unmasked.reason and "unrolled: Unsupported" in unmasked.reason
    assert "not declared masked" in unmasked.reason

    strict, _ = _counter_loop(fake, warmup_runs=0, strict=True)
    with pytest.raises(RuntimeError, match="cudaErrorNotSupported"):
        strict.run()


def test_a_capture_that_fails_in_the_body_ends_every_capture_and_keeps_the_builders(fake):
    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        if fake.pool is not None:  # under the capture: a host sync is refused
            raise RuntimeError("operation not permitted when stream is capturing")

    loop = graphloop.device_loop(step, None, 3, warmup_runs=0, device="cpu")
    loop.run()
    assert loop.mode == "host" and "while: RuntimeError: operation not permitted" in loop.reason
    builders = graphloop._ABANDONED[-1]
    assert [b.name for b in builders] == ["main", "body"]
    assert not any(b.is_building for b in builders)  # ended innermost first, never closed
    assert fake._core.log[-4:] == [
        ("end", "body"),
        ("end capture", builders[1].stream.handle),  # the driver's view, after cuda.core's
        ("end", "main"),
        ("end capture", builders[0].stream.handle),
    ]
    assert ("close", "main") not in fake._core.log


def test_an_old_driver_means_the_unrolled_fallback(fake):
    fake.driver = (12, 2)
    loop, _ = _counter_loop(fake, warmup_runs=0, masked=True, unroll=4)
    loop.run()
    assert loop.mode == "unrolled"
    assert "CUDA 12.2, conditional nodes need 12.4+" in loop.reason
    assert not fake._core.log  # no graph builder was touched


def test_support_reasons():
    core = FakeCore()
    ok = {"cuda": True, "driver": (12, 4), "core": core, "mem_pool": True}
    assert graphloop.support_reason(**ok) is None
    assert graphloop.support_reason(**{**ok, "cuda": False}) == "no CUDA device"
    assert "unknown" in graphloop.support_reason(**{**ok, "driver": None})
    assert "CUDA 12.3, conditional nodes need 12.4+" in graphloop.support_reason(
        **{**ok, "driver": (12, 3)}
    )
    assert "not installed" in graphloop.support_reason(**{**ok, "core": None})
    old = SimpleNamespace(__version__="0.2.0", graph=SimpleNamespace(GraphBuilder=object))
    assert "cuda.core 0.2.0 has no conditional" in graphloop.support_reason(**{**ok, "core": old})
    assert "MemPool" in graphloop.support_reason(**{**ok, "mem_pool": False})


# ------------------------------------------------------------------ the unrolled schedule


def _example() -> Any:
    spec = importlib.util.spec_from_file_location(
        "graph_while_decode", EXAMPLES_DIR / "graph_while_decode.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def toy() -> Any:
    ex = _example()
    model = ex.ToyDecoder(
        vocab=64, dim=32, heads=2, layers=2, max_len=64, device="cpu", dtype=torch.float32
    )
    prompt = [5, 17, 3, 42]
    req = ex.Request(model, 40)
    free, _ = ex.generate_host(req, prompt, eos=-1)
    late = range(12, 40)
    eos = next(free[i] for i in late if free[i] not in free[:i])
    return SimpleNamespace(ex=ex, model=model, prompt=prompt, req=req, free=free, eos=eos)


@pytest.mark.parametrize("unroll", [1, 3, 7, 40])
def test_unrolled_masking_gives_the_plain_loops_tokens_and_writes_nothing_past_the_stop(
    fake, toy, unroll
):
    ex, req, prompt = toy.ex, toy.req, toy.prompt
    reference, checks = ex.generate_host(req, prompt, toy.eos)
    cache = toy.model.k.clone(), toy.model.v.clone()
    n = len(reference)
    assert 12 < n < 40 and checks == n
    gen = ex.DeviceGenerator(req, toy.eos, mode="unrolled", unroll=unroll, device="cpu")
    gen.generate(prompt)  # warm-up: a host run
    tokens, host_checks = gen.generate(prompt)
    assert gen.loop.mode == "unrolled" and gen.loop.stats["unroll"] == unroll
    assert tokens == reference
    blocks = math.ceil(n / unroll)
    assert host_checks == min(blocks, math.ceil(40 / unroll) - 1)  # one read per block, behind
    # nothing written past the stop: the outputs and the KV cache are the plain loop's
    assert (req.out[n:] == -1).all()
    assert torch.equal(toy.model.k, cache[0]) and torch.equal(toy.model.v, cache[1])
    assert int(gen.loop.index) == n and not bool(gen.loop.active)


def test_unrolled_runs_to_the_limit_without_a_stop(fake, toy):
    ex, req = toy.ex, toy.req
    gen = ex.DeviceGenerator(req, eos=-1, mode="unrolled", unroll=6, device="cpu")
    gen.generate(toy.prompt)
    tokens, _ = gen.generate(toy.prompt)
    assert tokens == toy.free and len(tokens) == 40  # 40 = 6 * 6 + 4: a partial last block


def test_a_step_that_writes_when_masked_is_refused(fake):
    x = torch.zeros(4)

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        x[0] += 1  # ignores `active`

    loop = graphloop.device_loop(
        step, None, 5, masked=True, check_state=[x], warmup_runs=0, device="cpu", mode="unrolled"
    )
    loop.run()
    assert loop.mode == "host" and "a masked step changed check_state [0]" in loop.reason
    assert float(x[0]) == 6  # the check's masked step, then the host run's 5 steps


def test_host_mode_is_the_plain_loop(fake, toy):
    ex = toy.ex
    reference, _ = ex.generate_host(toy.req, toy.prompt, toy.eos)
    gen = ex.DeviceGenerator(toy.req, toy.eos, mode="host", device="cpu")
    for _ in range(2):
        tokens, checks = gen.generate(toy.prompt)
        assert tokens == reference and checks == len(reference)
    assert gen.loop.mode == "host"


def test_min_steps_and_a_lower_limit_per_run(fake):
    loop, _ = _counter_loop(fake, stop_at=1, min_steps=4, mode="host")
    assert int(loop.run()) == 4  # the flag is set from step 1, counted from step 4
    loop.x.fill_(0)
    assert int(loop.run(max_steps=2)) == 2
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        loop.run(max_steps=0)


# ------------------------------------------------------------------ streaming


@pytest.mark.parametrize(("mode", "want"), [("host", [3, 6, 7]), ("unrolled", [4, 6, 7])])
def test_chunks_yield_the_steps_done_at_chunk_boundaries(fake, mode, want):
    loop, _ = _counter_loop(
        fake, stop_at=7, mode=mode, masked=True, unroll=2, chunk_every=3, warmup_runs=0
    )
    assert list(loop.chunks()) == want  # unrolled: the boundaries its block reads saw
    loop.x.fill_(0)
    assert list(loop.chunks()) == want


def test_leaving_chunks_early_cancels_the_rest_of_the_loop(fake):
    loop, _ = _counter_loop(fake, stop_at=7, mode="host", chunk_every=2)
    for steps in loop.chunks():
        assert steps == 2
        break
    assert int(loop.x) == 2 and not loop._cancel  # stopped after the step it was at
    loop.x.fill_(0)
    assert list(loop.chunks()) == [2, 4, 6, 7]  # the next run is whole


def test_chunks_poll_a_while_graphs_host_mapped_counter(fake, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(graphloop, "time", clock)
    loop, _ = _counter_loop(fake, mode="while", warmup_runs=0, chunk_every=2)
    counts = iter([0, 2, 2, 4, 6, 7])
    done = iter([False] * 5 + [True])

    class Polled(FakeEvent):
        def query(self) -> bool:
            assert loop._host is not None
            loop._host[0] = next(counts)  # what the IF node's signal kernel wrote
            return next(done)

    monkeypatch.setattr(fake, "event", lambda: Polled())
    assert list(loop.chunks()) == [2, 4, 6, 7]
    assert clock.now == pytest.approx(1000.0 + 5 * graphloop.POLL_S)
    assert loop._host is not None and loop._host.tolist() == [0, 0]  # reset for the next run


def test_chunks_need_chunk_every_and_on_chunk_runs_on_the_host(fake):
    loop, _ = _counter_loop(fake)
    with pytest.raises(ValueError, match="chunk_every"):
        next(loop.chunks())
    marks: list[int] = []
    loop, _ = _counter_loop(fake, stop_at=5, mode="unrolled", masked=True, chunk_every=2)
    loop.on_chunk = lambda: marks.append(int(loop.index))
    assert list(loop.chunks()) == [2, 4, 5]  # the warm-up run: host steps
    assert marks == [2, 4, 5]
    loop.x.fill_(0)
    list(loop.chunks())
    assert loop.mode == "host" and "on_chunk runs at chunk boundaries" in loop.reason


# ------------------------------------------------------------------ teacher forcing


def test_a_replaced_watched_callable_runs_host_steps_that_call_it(fake):
    class Decoder:
        def __init__(self) -> None:
            self.calls = 0

        def forward(self) -> None:
            self.calls += 1

    decoder = Decoder()
    x = torch.zeros((), dtype=torch.int64)

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        decoder.forward()
        x.add_(active.to(x.dtype))

    loop = graphloop.device_loop(
        step,
        lambda: x >= 4,
        8,
        mode="unrolled",
        masked=True,
        unroll=2,
        watch=[(decoder, "forward")],
        device="cpu",
    )
    loop.run()
    x.zero_()
    loop.run()
    assert loop.mode == "unrolled" and loop.stats["host_runs"] == 1
    forced: list[int] = []
    decoder.forward = lambda: forced.append(1)  # type: ignore[method-assign]
    x.zero_()
    loop.run()
    assert len(forced) == 4  # teacher forcing's wrapper saw every step
    assert loop.stats["host_runs"] == 2 and "Decoder.forward is replaced" in loop.stats["host_why"]
    del decoder.forward
    x.zero_()
    launches = loop.stats["launches"]
    loop.run()
    assert loop.stats["host_runs"] == 2 and loop.stats["launches"] > launches


def test_no_graph_is_built_while_a_watched_callable_is_replaced(fake):
    module = torch.nn.Identity()
    loop = graphloop.device_loop(
        lambda i, a: module(i),
        None,
        3,
        mode="while",
        watch=[(module, "forward")],
        warmup_runs=0,
        device="cpu",
    )
    module.forward = lambda x: x  # type: ignore[method-assign]
    loop.run()
    assert loop.mode is None and not fake._core.log
    with pytest.raises(graphloop.Unsupported, match=r"Identity\.forward is replaced"):
        loop.build()


# ------------------------------------------------------------------ choosing K


class SimulatedGpu:
    """``replay`` costs the host ``host_s`` and queues ``step_s`` of GPU work; ``sync``
    waits for the GPU and wakes ``wake_s`` later (all on the simulated clock)."""

    def __init__(self, clock: Clock, step_s: float, host_s: float, wake_s: float) -> None:
        self.clock, self.step_s, self.host_s, self.wake_s = clock, step_s, host_s, wake_s
        self.free = clock.now

    def replay(self) -> None:
        self.clock.now += self.host_s
        self.free = max(self.free, self.clock.now) + self.step_s

    def sync(self) -> None:
        self.clock.now = max(self.clock.now, self.free) + self.wake_s


def test_measure_and_choose_unroll_under_a_simulated_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(graphloop, "time", clock)
    gpu = SimulatedGpu(clock, step_s=20e-6, host_s=4e-6, wake_s=6e-6)  # GPU bound
    step, launch = graphloop.measure_unroll(gpu.replay, gpu.sync)
    assert step == pytest.approx(20e-6) and launch == pytest.approx(10e-6)
    assert graphloop.choose_unroll(step, launch, 256) == 1  # launches hide behind the GPU

    gpu = SimulatedGpu(clock, step_s=2e-6, host_s=24e-6, wake_s=6e-6)  # host bound
    step, launch = graphloop.measure_unroll(gpu.replay, gpu.sync)
    assert step == pytest.approx(24e-6) and launch == pytest.approx(8e-6)
    # the "step" is the host's launch rate here; a 16-step block lets the GPU lead
    assert graphloop.choose_unroll(2e-6, 30e-6, 256) == 16
    assert graphloop.choose_unroll(2e-6, 30e-6, 3) == 3  # never more than the run's steps
    assert graphloop.choose_unroll(0.0, 0.0, 256) == 1
    assert graphloop.choose_unroll(0.0, 1e-6, 6400) == graphloop.MAX_UNROLL


def test_the_unrolled_build_measures_k_on_masked_replays(fake, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(graphloop, "time", clock)
    loop, _ = _counter_loop(fake, stop_at=100, mode="unrolled", masked=True, warmup_runs=0)
    gpu = SimulatedGpu(clock, step_s=2e-6, host_s=24e-6, wake_s=6e-6)

    def capture(fn: Any, pool: Any) -> Any:
        assert not bool(loop.active)  # measured masked: the counter stays
        return SimpleNamespace(replay=lambda: (fn(), gpu.replay()))

    monkeypatch.setattr(fake, "capture_graph", capture)
    monkeypatch.setattr(fake, "synchronize", lambda device: gpu.sync())
    loop.build()
    assert int(loop.x) == 0 and bool(loop.active)
    assert loop.stats["step_us"] == pytest.approx(24.0) and loop.stats["launch_us"] == 8.0
    assert loop.stats["unroll"] == graphloop.choose_unroll(24e-6, 8e-6, 8)


def test_masked_helpers_write_only_when_active():
    dst = torch.arange(6.0).view(2, 3)
    graphloop.masked_index_copy_(
        dst, 1, torch.tensor([1]), torch.full((2, 1), 9.0), torch.tensor(False)
    )
    assert torch.equal(dst, torch.arange(6.0).view(2, 3))
    graphloop.masked_index_copy_(
        dst, 1, torch.tensor([1]), torch.full((2, 1), 9.0), torch.tensor(True)
    )
    assert dst[:, 1].tolist() == [9.0, 9.0] and dst[:, 0].tolist() == [0.0, 3.0]
    last = torch.tensor([3])
    graphloop.masked_copy_(last, torch.tensor([5]), torch.tensor(False))
    assert last.tolist() == [3]
    graphloop.masked_copy_(last, torch.tensor([5]), torch.tensor(True))
    assert last.tolist() == [5]


def test_option_errors():
    step = lambda i, a: None  # noqa: E731
    with pytest.raises(ValueError, match="mode must be"):
        graphloop.device_loop(step, None, 3, mode="fast", device="cpu")
    with pytest.raises(ValueError, match="max_steps >= 1"):
        graphloop.device_loop(step, None, 0, device="cpu")
    with pytest.raises(ValueError, match="on_chunk needs chunk_every"):
        graphloop.device_loop(step, None, 3, on_chunk=lambda: None, device="cpu")
    with pytest.raises(ValueError, match="no stop flags"):
        graphloop.device_loop(step, torch.zeros(0), 3, mode="host", device="cpu").run()


# ------------------------------------------------------------------ doctor and skills


class FakeToolchain:
    gpu = None
    nvcc_version = "13.0"
    torch_version = "2.14"


def test_doctor_probe_reports_support_skip_and_failure(monkeypatch, tmp_path):
    from kernel_agent import toolchain

    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(toolchain, "setup", lambda apply_env=True: FakeToolchain())
    assert probes.NEEDS["graph_conditional"] == "cuda.core"
    for verdict, mark in ((True, "ok"), (None, "skipped"), (False, "FAILED")):
        monkeypatch.setattr(graphloop, "probe", lambda v=verdict: (v, f"detail {v}"))
        result = probes.run(True, {"graph_conditional": probes.PROBES["graph_conditional"]})
        assert result["probes"] == [
            {"name": "graph_conditional", "ok": verdict, "detail": f"detail {verdict}"}
        ]
        assert f"  graph_conditional: {mark}: detail {verdict}" in probes.describe(result)


def test_probe_without_conditional_support_is_a_skip(monkeypatch):
    monkeypatch.setattr(graphloop, "conditional_support", lambda: "no CUDA device")
    ok, detail = graphloop.probe()
    assert ok is None and detail.startswith("no CUDA device; device_loop uses K-step unrolled")


def test_the_skills_link_the_helper_and_the_example():
    text = skills.get("systems-patterns").text()
    assert "kernel_agent.graphloop" in text and "`examples/graph_while_decode.py`" in text
    assert "watch" in text and "masked" in text
    graphs = skills.get("cuda-graphs-streams-pdl").path.read_text()
    assert "graphloop" in graphs
    assert (EXAMPLES_DIR / "graph_while_decode.py").is_file()
    # problems() checks that every backticked example a skill names exists
    assert skills._EXAMPLE.findall("`examples/graph_while_decode.py`") == ["graph_while_decode.py"]
    assert not skills.problems()


# ------------------------------------------------------------------ GPU


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.gpu
@gpu
def test_toy_decoder_while_and_unrolled_vs_host_loop():
    ex = _example()
    with torch.inference_mode():
        rows = ex.compare(max_new=128, runs=3)
    for name, row in rows.items():
        print(f"{name:34s} {row}")
        assert row["same_tokens"], name
    n = rows["host loop"]["tokens"]
    assert rows["host loop"]["host_checks"] == n
    supported = graphloop.conditional_support() is None
    while_row = rows["device loop (while)"]
    if supported:
        assert while_row["mode"] == "while" and while_row["host_checks"] == 0
        assert while_row["host_us_per_token"] < 20  # the launch, not a host step per token
        assert while_row["ms_per_token"] < rows["host loop"]["ms_per_token"]
    unrolled = rows["device loop (unrolled)"]
    assert unrolled["mode"] == "unrolled" and unrolled["host_checks"] <= n


@pytest.mark.gpu
@gpu
def test_the_hidden_work_check_sees_the_while_loop():
    from kernel_agent.kernels import e2e_activity

    if graphloop.conditional_support() is not None:
        pytest.skip(graphloop.conditional_support())
    x = torch.zeros(1024, device="cuda")

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        torch.cuda._sleep(20_000)  # ~10 us: the loop's span is far above the join slack
        x.add_(1.0)

    loop = graphloop.device_loop(step, lambda: x[0] >= 50, 100, mode="while", warmup_runs=0)
    loop.run()

    def joined(_: Any) -> None:
        x.zero_()
        loop.run()

    evts, state = e2e_activity.profiled_run(joined, None, device=0)
    verdict = e2e_activity.analyse(evts, devices={0}, tag=state["tag"])
    assert verdict["passed"], verdict
    names = [e.name for e in evts if e.kind == "kernel"]
    print(f"{len(names)} kernels reported: {sorted(set(names))}")
    begin = next(e for e in evts if e.kind == "kernel" and "ka_loop_begin" in e.name)
    end = next(e for e in evts if e.kind == "kernel" and "ka_loop_end" in e.name)
    assert end.end - begin.start > 50 * 5_000  # the loop's span: 50 steps of >= 5 us
    assert float(x[0]) == 50 and loop._status.tolist() == [50, 0]

    side = torch.cuda.Stream()

    def unjoined(_: Any) -> None:  # what a transform must not do: the loop on a side stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            x.zero_()
            loop.run()

    evts, state = e2e_activity.profiled_run(unjoined, None, device=0)
    verdict = e2e_activity.analyse(evts, devices={0}, tag=state["tag"])
    assert not verdict["passed"] and any("ka_loop_end" in u for u in verdict["unjoined"])


@pytest.mark.gpu
@gpu
def test_while_chunks_reach_the_host_and_leaving_early_cancels():
    if graphloop.conditional_support() is not None:
        pytest.skip(graphloop.conditional_support())
    with torch.inference_mode(False):
        steps = torch.zeros((), dtype=torch.int64, device="cuda")

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        torch.cuda._sleep(200_000)  # ~0.1 ms a step: the host polls in time
        steps.add_(1)

    loop = graphloop.device_loop(step, None, 400, mode="while", warmup_runs=0, chunk_every=4)
    seen = []
    for count in loop.chunks():
        seen.append(count)
        break
    assert seen and seen[0] % 4 == 0 and seen[0] < 400
    assert int(steps) < 400  # cancelled after the step it was at, and joined
    assert int(steps) == int(loop.index)
    full = list(loop.chunks())
    assert full[-1] == 400 and all(c % 4 == 0 for c in full)


@pytest.mark.gpu
@gpu
def test_a_step_that_syncs_or_draws_falls_back_and_leaves_no_capture_open():
    if graphloop.conditional_support() is not None:
        pytest.skip(graphloop.conditional_support())
    x = torch.zeros(4, 8, device="cuda")

    def syncs(index: torch.Tensor, active: torch.Tensor) -> None:
        x[0] += float(x[0, 0].item())  # a host sync: refused under a capture

    loop = graphloop.device_loop(syncs, None, 4, masked=True, warmup_runs=0)
    loop.run()
    assert loop.mode == "host" and "while: " in loop.reason and "unrolled: " in loop.reason
    torch.cuda.synchronize()  # no capture left open by the failed builds

    def draws(index: torch.Tensor, active: torch.Tensor) -> None:
        graphloop.masked_index_copy_(x, 0, index.view(1), torch.randn(1, 8, device="cuda"), active)

    loop = graphloop.device_loop(draws, None, 4, masked=True, warmup_runs=0)
    loop.run()
    torch.cuda.synchronize()
    assert loop.mode == "unrolled" and "RNG op during graph capture" in loop.reason
    assert len({tuple(row.tolist()) for row in x.cpu()}) == 4  # every step drew its own


@pytest.mark.gpu
@gpu
def test_probe_on_this_gpu():  # last: the tests above start from a fresh cuda.core context
    ok, detail = graphloop.probe()
    assert ok in (True, None), detail
