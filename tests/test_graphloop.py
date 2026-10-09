"""Host-free generation loops (#232): kernel_agent.graphloop on a fake cuda.core and a fake
CUDA backend (CPU only): the WHILE graph's structure, the fallbacks and their reasons, the
unrolled schedule's masking on the toy decoder of examples/graph_while_decode.py (tokens
identical to the plain loop, nothing written past the stop), streaming chunks, teacher
forcing's watched callables, the unroll choice under a simulated clock, the doctor probe and
the skill links. GPU tests (marked) run the real graphs."""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import math
import subprocess
import sys
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
        self.core.ran()


class FakeCore:
    """The parts of ``cuda.core`` graphloop uses; ``log`` records every call."""

    __version__ = "1.2.1"
    graph = SimpleNamespace(GraphBuilder=FakeBuilder)

    def __init__(self) -> None:
        self.log: list[tuple[Any, ...]] = []
        self.launches: list[tuple[str, str, tuple[Any, ...]]] = []
        self.handles = iter(range(500, 10_000))
        self.fail_complete = False
        self.on_launch: Any = None  # what a launched graph "does" (the fake runs nothing)
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

    def ran(self) -> None:
        if self.on_launch is not None:
            self.on_launch()


def _round4(n: int) -> int:
    return (n + 3) // 4 * 4


class FakeGenerator:
    """torch's CUDA generator as graphloop uses it: a seed and a Philox offset (a multiple
    of 4) that each draw advances by its count rounded up to 4. Under a capture a draw is
    refused (torch: the generator state is not in capture mode) unless a FakeGraphRng has
    it in capture mode: then the draw is counted there (torch's offset within a capture)."""

    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda
        self.seed = self.offset = 0
        self.capture: FakeGraphRng | None = None

    def initial_seed(self) -> int:
        return self.seed

    def manual_seed(self, seed: int) -> None:
        self.seed, self.offset = seed, 0

    def get_offset(self) -> int:
        return self.offset

    def set_offset(self, offset: int) -> None:
        assert offset % 4 == 0 and offset >= 0, offset
        self.offset = offset

    def draw(self, n: int) -> torch.Tensor:
        """``n`` numbers that only the seed and the offset decide (a Philox draw)."""
        if self.capture is not None:
            self.capture.counted += _round4(n)
            return torch.zeros(n)
        if self.cuda.current != "caller":
            raise RuntimeError(
                "Attempt to increase offset for a CUDA generator not in capture mode."
            )
        out = torch.rand(n, generator=torch.Generator().manual_seed(self.seed * 7919 + self.offset))
        self.offset += _round4(n)
        return out


class FakeGraphRng:
    """graphloop._GraphRng on the CPU: the device scalars are CPU tensors."""

    def __init__(self, cuda: FakeCuda, generator: FakeGenerator) -> None:
        self.cuda, self.generator = cuda, generator
        self.seed = torch.zeros(1, dtype=torch.int64)
        self.offset = torch.zeros(1, dtype=torch.int64)
        self.counted = 0

    @contextlib.contextmanager
    def capture(self) -> Iterator[None]:
        self.cuda._core.log.append(("rng", "capture"))
        self.generator.capture = self
        try:
            yield
        finally:
            self.generator.capture = None
            self.cuda._core.log.append(("rng", "end"))

    def increment(self) -> int:
        return self.counted


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
        self.gen = FakeGenerator(self)
        self.graph_rngs: list[FakeGraphRng] = []
        self.rng_error: str | None = None  # graph_rng raises it

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
            self._core.ran()
        return self.runtime

    def generator(self, device: int) -> Any:
        return self.gen

    def graph_rng(self, generator: Any, device: int) -> Any:
        if self.rng_error is not None:
            raise RuntimeError(self.rng_error)
        self.graph_rngs.append(FakeGraphRng(self, generator))
        return self.graph_rngs[-1]

    def leave_rng_capture(self, device: int, *states: Any) -> None:
        self._core.log.append(("leave rng capture", len(states)))


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
        ("rng", "capture"),  # no warm-up run said whether the step draws
        ("external stream", body),
        ("pool", "private pool"),
        ("rng", "end"),
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
    loop_cond, chunk_cond, flags, n_flags, index, active, limits, every, host, *rng = launches[
        "ka_loop_step"
    ][1]
    assert rng == [0, 0]  # the step draws nothing: no offset to advance
    assert (loop_cond.graph, chunk_cond.graph) == ("main", "body")
    assert (n_flags, every) == (1, 2) and flags != 0
    assert (index, active) == (loop.index.data_ptr(), loop.active.data_ptr())
    assert limits == loop._limits.data_ptr() and host == loop._host.data_ptr()
    assert launches["ka_loop_begin"][1] == (loop_cond, index, active)
    assert launches["ka_chunk_signal"][1] == (index, host)
    status, total = loop._status.data_ptr(), loop._total.data_ptr()
    assert launches["ka_loop_end"][1] == (index, active, status, total)  # adds the run's steps
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
    assert args[1:4] == (0, 0, 0) and args[-4:] == (0, 0, 0, 0)  # no chunk, flags, host, draws


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


def test_chunks_need_chunk_every_and_on_chunk_runs_in_the_unrolled_blocks(fake):
    loop, _ = _counter_loop(fake)
    with pytest.raises(ValueError, match="chunk_every"):
        next(loop.chunks())
    marks: list[int] = []
    loop, _ = _counter_loop(fake, stop_at=5, mode="unrolled", masked=True, unroll=2, chunk_every=2)
    loop.on_chunk = lambda: marks.append(int(loop.index))
    assert list(loop.chunks()) == [2, 4, 5]  # the warm-up run: host steps
    assert marks == [2, 4, 5]
    loop.x.fill_(0)
    assert list(loop.chunks()) == [2, 4, 5]
    assert loop.mode == "unrolled" and loop.reason == "unrolled built"
    assert marks == [2, 4, 5] * 2


def _chunk_marks(
    fake: FakeCuda, mode: str, stop_at: int, every: int, unroll: int, limit: int | None
) -> tuple[Any, list[int], list[int], list[int]]:
    """A counter loop's ``on_chunk`` steps in ``mode`` (after its warm-up host run): the
    ones of ``run()`` and of ``chunks()``, and what ``chunks()`` yielded."""
    marks: list[int] = []
    loop, _ = _counter_loop(
        fake, stop_at=stop_at, mode=mode, masked=True, unroll=unroll, chunk_every=every
    )
    loop.on_chunk = lambda: marks.append(int(loop.index))
    loop.run(limit)  # the warm-up: host steps
    del marks[:]
    loop.x.fill_(0)
    loop.run(limit)
    ran, marks[:] = list(marks), []
    loop.x.fill_(0)
    yielded = list(loop.chunks(limit))
    return loop, ran, list(marks), yielded


@pytest.mark.parametrize(
    ("stop_at", "every", "unroll", "limit", "want"),
    [
        (7, 3, 3, None, [3, 6, 7]),  # every block ends at a boundary; the stop in one
        (7, 4, 2, None, [4, 7]),  # the stop in a block that ends with on_chunk
        (5, 4, 2, None, [4, 5]),  # the stop in a plain block, the next one ends a chunk
        (3, 6, 2, None, [3]),  # ... seen before the chunk block is launched
        (3, 8, 2, None, [3]),  # ... seen one plain block later (launched, all masked)
        (100, 3, 3, None, [3, 6, 8]),  # the limit (8) in a chunk block
        (100, 4, 2, 5, [4, 5]),  # this run's limit (5) in a plain last block
        (7, 3, 2, None, [3, 6, 7]),  # unroll 2 does not divide 3: blocks of 1
    ],
)
def test_unrolled_on_chunk_runs_where_the_while_graph_runs_it_once_each(
    fake, stop_at, every, unroll, limit, want
):
    _, ran, streamed, yielded = _chunk_marks(fake, "host", stop_at, every, unroll, limit)
    assert ran == streamed == yielded == want  # the host loop: the WHILE graph's rule
    loop, ran, streamed, yielded = _chunk_marks(fake, "unrolled", stop_at, every, unroll, limit)
    assert loop.mode == "unrolled" and every % loop.stats["unroll"] == 0
    assert ran == streamed == yielded == want


def test_unrolled_chunk_blocks_wait_for_the_block_before_and_nothing_else(fake):
    loop, ran, _, yielded = _chunk_marks(fake, "unrolled", 5, 4, 2, None)
    assert ran == yielded == [4, 5]
    # run() and chunks(): blocks [1, 2] plain, [3, 4] with on_chunk (after reading [1, 2]),
    # [5, 6] plain (stops at 5; read when the next would end a chunk), then on_chunk for the
    # stop (its two steps masked), whose status chunks() reads before it yields the count;
    # the warm-up host run read all of its 5 steps
    assert loop.stats["launches"] == 2 * 4 and loop.stats["host_checks"] == 5 + 3 + 4

    blocks: list[str] = []
    loop, _ = _counter_loop(fake, stop_at=100, mode="unrolled", masked=True, unroll=2)
    loop.chunk_every, loop.on_chunk = 4, lambda: blocks.append("chunk")
    loop.warmup_runs = 0
    loop.build()
    plain, chunk = loop._graph, loop._chunk_graph
    loop._graph = SimpleNamespace(replay=lambda: (blocks.append("plain"), plain.replay()))
    loop._chunk_graph = SimpleNamespace(replay=lambda: (blocks.append("with"), chunk.replay()))
    checks = loop.stats["host_checks"]
    loop.run()  # the limit (8): blocks to step 2, 4, 6 and 8
    assert blocks == ["plain", "with", "chunk", "plain", "with", "chunk"]
    # the reads of run(): before each chunk block and one block behind; none at the end
    assert loop.stats["host_checks"] - checks == 3 and int(loop.x) == 8


def test_unrolled_streaming_toy_decoder_matches_the_host_loop(fake, toy):
    """A non-idempotent on_chunk (a device log of the chunk's step count and its tokens so
    far): the same log as the host loop's, so on_chunk ran once per boundary and stop."""
    ex, req, prompt = toy.ex, toy.req, toy.prompt
    reference, _ = ex.generate_host(req, prompt, toy.eos)

    def streamed(mode: str, unroll: int) -> tuple[list[int], list[int], list[int]]:
        log = torch.full((16, 2), -1, dtype=torch.int64)
        n = torch.zeros(1, dtype=torch.int64)
        gen = ex.DeviceGenerator(
            req, toy.eos, mode=mode, unroll=unroll, chunk_every=4, device="cpu", warmup_runs=0
        )

        def on_chunk() -> None:
            index = gen.loop.index.view(1)
            row = torch.stack([index[0], req.out.ne(-1).sum()]).view(1, 2)
            log.index_copy_(0, n, row)
            n.add_(1)

        gen.loop.on_chunk = on_chunk
        tokens, _ = gen.generate(prompt)
        return tokens, log[: int(n)].flatten().tolist(), [gen.loop.mode]

    want = streamed("host", 1)
    assert want[0] == reference
    for unroll in (1, 2, 4, 8):
        assert streamed("unrolled", unroll) == (*want[:2], ["unrolled"]), unroll


def test_the_measured_k_is_aligned_to_the_chunks(fake, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(graphloop, "time", clock)
    x = torch.zeros((), dtype=torch.int64)
    loop = graphloop.device_loop(
        lambda i, a: x.add_(a.to(x.dtype)),
        None,
        256,
        mode="unrolled",
        masked=True,
        chunk_every=24,
        on_chunk=lambda: None,
        warmup_runs=0,
        device="cpu",
    )
    gpu = SimulatedGpu(clock, step_s=1e-6, host_s=2e-6, wake_s=30e-6)  # host bound
    pools = iter(range(10))
    captured: list[str] = []

    def capture(fn: Any, pool: Any) -> Any:
        captured.append(pool)
        return SimpleNamespace(replay=lambda: (fn(), gpu.replay()))

    monkeypatch.setattr(fake, "capture_graph", capture)
    monkeypatch.setattr(fake, "pool_handle", lambda: f"pool {next(pools)}")
    monkeypatch.setattr(fake, "synchronize", lambda device: gpu.sync())
    loop.build()
    assert (loop.stats["step_us"], loop.stats["launch_us"]) == (2.0, 31.0)
    assert graphloop.choose_unroll(2e-6, 31e-6, 256) == 16
    assert loop.stats["unroll"] == 12  # the largest divisor of chunk_every up to 16
    # K = 1 measured in the first pool, then both blocks; the one with on_chunk in its own
    assert captured == ["pool 0", "pool 0", "pool 1"]


def test_aligned_unroll():
    assert graphloop.aligned_unroll(16, 24) == 12
    assert graphloop.aligned_unroll(5, 4) == 4
    assert graphloop.aligned_unroll(3, 4) == 2
    assert graphloop.aligned_unroll(64, 7) == 7
    assert graphloop.aligned_unroll(6, 7) == 1


# ------------------------------------------------------------------ counters


def _toy_workload() -> Any:
    from diverse_toy import DiverseToy

    from kernel_agent.workloads.base import WorkloadSpec

    return DiverseToy(WorkloadSpec(repo_id="toy", modality="llm", device="cpu"))


class RecordingClock:
    """``time`` of ``workloads.base``: logs every read of the timed run's clock."""

    def __init__(self, log: list[str]) -> None:
        self.log, self.now = log, 1000.0

    def perf_counter(self) -> float:
        self.log.append("clock")
        self.now += 1.0
        return self.now


@pytest.mark.parametrize("mode", ["host", "unrolled", "while"])
def test_a_loops_steps_reach_decode_stats_read_outside_the_timed_run(fake, monkeypatch, mode):
    from kernel_agent.workloads import base

    workload = _toy_workload()
    x = torch.zeros((), dtype=torch.int64)
    stop = torch.zeros((), dtype=torch.int64)

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        x.add_(active.to(x.dtype))

    loop = graphloop.device_loop(
        step,
        lambda: x >= stop,
        16,
        mode=mode,
        masked=True,
        unroll=2,
        device="cpu",
        workload=workload,
        report=("steps", "tokens"),
    )
    if mode == "while":

        def launch(graph: Any, stream: Any) -> None:  # the fake graph runs until the stop
            loop._reset()
            while bool(loop.active):
                loop._body()

        monkeypatch.setattr(FakeGraph, "launch", launch)
    log: list[str] = []
    total = loop.steps_total
    monkeypatch.setattr(loop, "steps_total", lambda: (log.append("read"), total())[1])

    def run(inputs: Any) -> dict[str, Any]:  # two requests of 3 and 5 steps
        for n in (3, 5):
            x.zero_()
            stop.fill_(n)
            loop.run()
        log.append("run")
        return {}

    workload.run = run
    run(None)  # untimed: the warm-up host run and the build are not counted
    log.clear()
    monkeypatch.setattr(base, "time", RecordingClock(log))
    _, _, detail = base.timed_run(workload, None)
    assert loop.mode == mode and detail["decode_stats"] == {"steps": 8, "tokens": 8}
    assert log == ["read", "clock", "run", "clock", "read"]  # never inside the clock


def test_stats_sources_report_only_loops_that_ran(fake, monkeypatch):
    from kernel_agent.workloads import base

    workload = _toy_workload()
    loop, _ = _counter_loop(fake, stop_at=3, mode="host", workload=workload)
    workload.add_stats_source(loop)  # once only
    assert len(workload.stats_sources) == 1
    mark = loop.stats_mark()
    assert loop.stats_since(mark) == {}  # it did not run (an A/B's other state)
    loop.run()
    assert loop.stats_since(mark) == {"steps": 3}
    assert loop.stats_since(None) == {"steps": 3}  # added during the run: since it was made
    monkeypatch.setattr(base, "time", RecordingClock([]))
    workload.run = lambda inputs: {}
    assert "decode_stats" not in base.timed_run(workload, None)[2]  # the loop did not run

    made: list[Any] = []

    def run(inputs: Any) -> dict[str, Any]:  # a transform that makes its loop in the run
        made.append(_counter_loop(fake, stop_at=4, mode="host", workload=workload)[0])
        made[0].run()
        return {}

    workload.run = run
    assert base.timed_run(workload, None)[2]["decode_stats"] == {"steps": 4}
    del loop, made[:]
    gc.collect()
    assert base._stats_sources(workload) == []  # held weakly: gone with the loops
    quiet, _ = _counter_loop(fake, stop_at=2, mode="host", workload=workload, report=())
    quiet.run()
    assert quiet.stats_since(None) == {}


# ------------------------------------------------------------------ random numbers

LIMIT, STOP, PER_STEP = 8, 5, 8  # a step draws 6 numbers: 8 Philox offsets (rounded to 4)


def _drawing_loop(fake: FakeCuda, **options: Any) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """A loop whose step draws 6 numbers from the device's generator into ``out[index]``
    (masked) and stops after STOP steps."""
    out = torch.zeros(LIMIT, 6)
    count = torch.zeros((), dtype=torch.int64)

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        noise = fake.gen.draw(6).view(1, 6)
        graphloop.masked_index_copy_(out, 0, index.view(1), noise, active)
        graphloop.masked_copy_(count, count + 1, active)

    options = {"masked": True, "device": "cpu", **options}
    loop = graphloop.device_loop(step, lambda: count >= STOP, LIMIT, **options)
    return loop, out, count


def _plain_draws(fake: FakeCuda, seed: int) -> tuple[torch.Tensor, int]:
    """The plain loop's draws (seeded) and where it leaves the generator."""
    fake.gen.manual_seed(seed)
    out = torch.zeros(LIMIT, 6)
    for i in range(STOP):
        out[i] = fake.gen.draw(6)
    return out, fake.gen.offset


@pytest.mark.parametrize("rng", ["exact", "reserve"])
@pytest.mark.parametrize("mode", ["host", "unrolled"])
def test_draws_are_the_plain_loops_and_rng_says_where_the_generator_ends(fake, mode, rng):
    """The unrolled blocks replay torch's graphs (draws from the replay's offset on), so a
    block's steps draw the plain loop's numbers; its masked steps after the stop draw too
    and the run corrects the generator: ``exact`` leaves it where the plain loop does,
    ``reserve`` at max_steps' draws, in every mode. The build's own draws (the masked check
    and the measuring replay) are put back."""
    want, plain_end = _plain_draws(fake, seed=11)
    assert plain_end == STOP * PER_STEP
    loop, out, count = _drawing_loop(fake, mode=mode, unroll=3, rng=rng)
    for _ in range(3):  # the warm-up host run, then the built mode's runs
        fake.gen.manual_seed(11)
        out.zero_()
        count.zero_()
        loop.run()
        assert torch.equal(out, want)
        assert fake.gen.offset == (plain_end if rng == "exact" else LIMIT * PER_STEP)
    assert loop.mode == mode and loop.stats["rng_offsets_per_step"] == PER_STEP
    if mode == "unrolled":  # blocks of 3, read one behind: 9 steps drew
        assert loop.stats["launches"] == 2 * 3


def _launched(fake: FakeCuda, loop: Any, steps: int) -> list[tuple[int, int]]:
    """What each launch of the fake WHILE graph saw in the RNG scalars (seed, offset); the
    "graph" runs ``steps`` steps."""
    seen: list[tuple[int, int]] = []

    def run() -> None:
        rng = fake.graph_rngs[-1]
        seen.append((int(rng.seed), int(rng.offset)))
        loop.index.fill_(steps)

    fake._core.on_launch = run
    return seen


@pytest.mark.parametrize("rng", ["exact", "reserve"])
def test_a_while_body_that_draws_reads_offsets_its_step_kernel_advances(fake, rng):
    """The step is captured with the RNG state in capture mode (on_chunk is not: its draws
    are refused); ka_loop_step advances the offset scalar by the step's increment; each
    launch fills the scalars from the generator (the seed only when it changed) and takes
    max_steps' offsets; ``exact`` then reads the steps and gives back the rest."""
    loop, _, _ = _drawing_loop(fake, mode="while", rng=rng, chunk_every=4)
    fake.gen.manual_seed(11)
    loop.run()  # the warm-up host run measures the step's draws
    assert loop.mode is None and loop.stats["rng_offsets_per_step"] == PER_STEP
    seen = _launched(fake, loop, steps=STOP)
    loop.run()
    assert loop.mode == "while" and loop.reason == "while built"
    log = fake._core.log
    begin = log.index(("begin", "body"))
    assert log[begin + 1 : begin + 5] == [
        ("rng", "capture"),
        ("external stream", 501),
        ("pool", "private pool"),
        ("rng", "end"),
    ]
    assert log.index(("rng", "end")) < log.index(("begin", "then"))  # on_chunk's: refused
    (graph_rng,) = fake.graph_rngs
    (args,) = [a for _, kernel, a in fake._core.launches if kernel == "ka_loop_step"]
    assert args[-2:] == (graph_rng.offset.data_ptr(), PER_STEP)
    base = STOP * PER_STEP if rng == "exact" else LIMIT * PER_STEP  # after the warm-up
    assert seen == [(11, base)]
    after = base + (STOP if rng == "exact" else LIMIT) * PER_STEP
    assert fake.gen.offset == after
    graph_rng.seed.fill_(-1)  # the seed is not filled again while it stays the same
    assert list(loop.chunks()) == []  # (the fake's IF node signals nothing)
    assert seen[-1] == (-1, after) and fake.gen.offset == after + (after - base)
    fake.gen.manual_seed(12)
    loop.run()
    assert seen[-1] == (12, 0)


def test_a_while_build_sets_no_rng_state_up_for_a_step_that_drew_nothing(fake):
    loop, _ = _counter_loop(fake, mode="while")
    loop.run()  # the warm-up: no draws
    loop.run()
    assert loop.mode == "while" and not fake.graph_rngs and ("rng", "capture") not in fake._core.log
    assert "rng_offsets_per_step" not in loop.stats


def test_without_a_graph_safe_rng_state_a_drawing_step_falls_back(fake):
    fake.rng_error = "torch 9.9: a generator state's device seed and offset were not found"
    plain, _ = _counter_loop(fake, mode="while", warmup_runs=0)
    plain.run()  # draws nothing: built without it
    assert plain.mode == "while" and "were not found" in plain.stats["rng"]
    want, plain_end = _plain_draws(fake, seed=3)
    loop, out, _ = _drawing_loop(fake, warmup_runs=0, unroll=2)
    fake.gen.manual_seed(3)
    loop.run()
    assert loop.mode == "unrolled" and "not in capture mode" in loop.reason
    assert torch.equal(out, want) and fake.gen.offset == plain_end


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
    with pytest.raises(ValueError, match="rng must be one of"):
        graphloop.device_loop(step, None, 3, rng="fresh", device="cpu")
    assert graphloop._int64(2**64 - 1) == -1 and graphloop._int64(5) == 5  # a seed's bits


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


@pytest.mark.parametrize(
    ("compiled", "drawn", "want"),
    [
        (True, True, True),
        (None, None, True),  # None: no torch.compile step / no RNG capture here
        (False, True, False),
        (True, False, False),
    ],
)
def test_the_probe_adds_the_torch_compile_and_rng_checks(monkeypatch, compiled, drawn, want):
    monkeypatch.setattr(graphloop, "conditional_support", lambda: None)
    monkeypatch.setattr(graphloop, "_probe_while", lambda: (True, "while ok"))
    monkeypatch.setattr(graphloop, "_probe_compiled", lambda: (compiled, f"compile {compiled}"))
    monkeypatch.setattr(graphloop, "_probe_rng", lambda: (drawn, f"rng {drawn}"))
    assert graphloop.probe() == (want, f"while ok; compile {compiled}; rng {drawn}")
    monkeypatch.setattr(graphloop, "_probe_while", lambda: (False, "while wrong"))
    assert graphloop.probe() == (False, "while wrong")  # not tried after a wrong graph


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
def test_toy_decoder_sampling_gives_the_host_loops_tokens():
    """#232: temperature sampling (torch.multinomial) in the toy decoder's step: the WHILE
    graph and the unrolled blocks give the host loop's tokens, run after run."""
    ex = _example()
    with torch.inference_mode():
        model = ex.ToyDecoder(vocab=512, dim=128, heads=4, layers=1, max_len=160)
        req = ex.Request(model, 128, temperature=0.8, seed=5)
        prompt = list(range(3, 19))
        free, _ = ex.generate_host(req, prompt, eos=-1)
        eos = next(t for i, t in enumerate(free) if i >= 64 and t not in free[:i])
        reference, _ = ex.generate_host(req, prompt, eos)
        assert 64 < len(reference) <= 128
        modes = ["unrolled"] + (["while"] if graphloop.conditional_support() is None else [])
        for mode in modes:
            gen = ex.DeviceGenerator(req, eos, mode=mode)
            for _ in range(4):  # the warm-up host run, then the graph's
                tokens, _ = gen.generate(prompt)
                assert tokens == reference, mode
            assert gen.loop.mode == mode and gen.loop.stats["rng_offsets_per_step"] > 0


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

    # A profiled run before the graph is built (an earlier check in the process): CUPTI then
    # records every body iteration under correlation ids of other calls, now and then the
    # check's own marker's, which made the joined loop fail now and then (#232, A10)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
        x.add_(0.0)
    loop = graphloop.device_loop(step, lambda: x[0] >= 50, 100, mode="while", warmup_runs=0)
    loop.run()

    def joined(_: Any) -> None:
        x.zero_()
        loop.run()

    for _ in range(8):  # a body kernel takes the marker's id in about one run in three
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
def test_a_step_that_syncs_falls_back_and_leaves_no_capture_open():
    if graphloop.conditional_support() is not None:
        pytest.skip(graphloop.conditional_support())
    x = torch.zeros(4, 8, device="cuda")

    def syncs(index: torch.Tensor, active: torch.Tensor) -> None:
        x[0] += float(x[0, 0].item())  # a host sync: refused under a capture

    caller = torch.cuda.current_stream()
    loop = graphloop.device_loop(syncs, None, 4, masked=True, warmup_runs=0)
    loop.run()
    assert loop.mode == "host" and "while: " in loop.reason and "unrolled: " in loop.reason
    torch.cuda.synchronize()  # no capture left open by the failed builds
    # torch 2.10 (A10): its failed torch.cuda.graph left the capture stream current and
    # the allocator routing to the pool, and the next MemPool destructor aborted
    assert torch.cuda.current_stream() == caller
    pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(pool):
        torch.ones(1, device="cuda")
    del pool

    def draws(index: torch.Tensor, active: torch.Tensor) -> None:
        graphloop.masked_index_copy_(x, 0, index.view(1), torch.randn(1, 8, device="cuda"), active)

    loop = graphloop.device_loop(draws, None, 4, masked=True, warmup_runs=0)
    loop.run()
    torch.cuda.synchronize()
    assert loop.mode == "while" and loop.stats["rng_offsets_per_step"] > 0  # captured (#232)
    assert len({tuple(row.tolist()) for row in x.cpu()}) == 4  # every step drew its own

    def draws_and_syncs(index: torch.Tensor, active: torch.Tensor) -> None:
        x[0] += float(torch.randn(1, device="cuda").item())

    # torch 2.10 (A10): its failed capture of a step that drew left the default generator in
    # capture mode, and every later draw of the process raised
    loop = graphloop.device_loop(draws_and_syncs, None, 4, masked=True, warmup_runs=0)
    loop.run()
    assert loop.mode == "host" and "while: " in loop.reason and "unrolled: " in loop.reason
    torch.manual_seed(3)
    eager = torch.randn(4, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed = torch.randn(4, device="cuda")
    torch.manual_seed(3)
    graph.replay()
    assert torch.equal(graphed, eager)


def _drawing_step(out: torch.Tensor, tok: torch.Tensor, count: torch.Tensor) -> Any:
    """A sampling step: noise from torch.randn, a token from torch.multinomial, masked."""

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        noise = torch.randn(8, device="cuda")
        token = torch.multinomial(torch.softmax(noise * 3, 0), 1)
        graphloop.masked_index_copy_(out, 0, index.view(1), (noise + token).view(1, 8), active)
        graphloop.masked_copy_(tok, token, active)
        graphloop.masked_copy_(count, count + 1, active)

    return step


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("rng", ["exact", "reserve"])
@pytest.mark.parametrize("mode", ["while", "unrolled", "host"])
def test_a_step_that_draws_gives_the_plain_loops_numbers_in_every_mode(mode, rng):
    """#232: torch.randn / torch.multinomial in the step, the stop after 13 of 24 steps. Every
    run draws the plain loop's numbers, step for step; ``exact`` leaves the generator where
    the plain loop does (the next draw is the plain loop's next draw), ``reserve`` at 24
    steps' draws, in every mode."""
    if mode == "while" and graphloop.conditional_support() is not None:
        pytest.skip(graphloop.conditional_support())
    limit, stop = 24, 13
    with torch.inference_mode(False):
        out = torch.zeros(limit, 8, device="cuda")
        tok = torch.zeros(1, dtype=torch.int64, device="cuda")
        count = torch.zeros((), dtype=torch.int64, device="cuda")
    step = _drawing_step(out, tok, count)
    torch.manual_seed(7)
    for i in range(stop):  # the plain loop
        step(torch.tensor(i, device="cuda"), torch.ones((), dtype=torch.bool, device="cuda"))
    want, offset = out.clone(), torch.cuda.default_generators[0].get_offset()
    following = torch.randn(4, device="cuda")
    loop = graphloop.device_loop(
        step, lambda: count >= stop, limit, mode=mode, masked=True, unroll=4, rng=rng
    )
    per_step = None
    for _ in range(4):  # the warm-up host run, then the built mode's runs
        torch.manual_seed(7)
        out.zero_()
        count.zero_()
        loop.run()
        assert torch.equal(out, want) and int(loop.index) == stop
        per_step = loop.stats["rng_offsets_per_step"]
        at = torch.cuda.default_generators[0].get_offset()
        assert at == (offset if rng == "exact" else offset // stop * limit)
        assert torch.equal(torch.randn(4, device="cuda"), following) == (rng == "exact")
    assert loop.mode == mode and per_step == offset // stop


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("mode", ["while", "unrolled"])
def test_a_device_loops_steps_reach_decode_stats_without_a_sync(mode):
    from kernel_agent.workloads import base

    if mode == "while" and graphloop.conditional_support() is not None:
        pytest.skip(graphloop.conditional_support())
    workload = _toy_workload()
    with torch.inference_mode(False):
        x = torch.zeros((), dtype=torch.int64, device="cuda")
        stop = torch.zeros((), dtype=torch.int64, device="cuda")

    def step(index: torch.Tensor, active: torch.Tensor) -> None:
        graphloop.masked_copy_(x, x + 1, active)

    loop = graphloop.device_loop(
        step,
        lambda: x >= stop,
        64,
        mode=mode,
        masked=True,
        unroll=4,
        strict=True,
        workload=workload,
        report=("steps", "tokens"),
    )

    def run(inputs: Any) -> dict[str, Any]:  # two requests of 3 and 5 steps
        # a WHILE run makes no host sync (torch raises on one in this mode)
        torch.cuda.set_sync_debug_mode("error" if loop.mode == "while" else 0)
        try:
            for n in (3, 5):
                x.zero_()
                stop.fill_(n)
                loop.run()
        finally:
            torch.cuda.set_sync_debug_mode(0)
        return {}

    workload.run = run
    run(None)  # the warm-up host run
    run(None)  # the build
    _, _, detail = base.timed_run(workload, None)
    assert loop.mode == mode and detail["decode_stats"] == {"steps": 8, "tokens": 8}


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("stop_at", [5, 8, 11, 40])
def test_unrolled_on_chunk_runs_where_the_while_graph_runs_it(stop_at):
    """A device-side, non-idempotent on_chunk (it appends the step count to a log): the
    unrolled blocks give the host loop's log (and the WHILE graph's, where it builds)."""

    def logs(mode: str, unroll: int = 1) -> tuple[list[list[int]], list[int], str | None]:
        with torch.inference_mode(False):
            x = torch.zeros((), dtype=torch.int64, device="cuda")
            log = torch.full((32,), -1, dtype=torch.int64, device="cuda")
            n = torch.zeros(1, dtype=torch.int64, device="cuda")

        def step(index: torch.Tensor, active: torch.Tensor) -> None:
            graphloop.masked_copy_(x, x + 1, active)

        def on_chunk() -> None:
            log.index_copy_(0, n, loop.index.view(1))
            n.add_(1)

        loop = graphloop.device_loop(
            step,
            lambda: x >= stop_at,
            30,
            mode=mode,
            masked=True,
            unroll=unroll,
            chunk_every=4,
            on_chunk=on_chunk,
            warmup_runs=0,
            strict=mode != "host",
        )
        out, seen = [], []
        for streamed in (False, False, True):  # run() twice, then chunks()
            x.zero_()
            log.fill_(-1)
            n.zero_()
            if streamed:
                seen = list(loop.chunks())
            else:
                loop.run()
            torch.cuda.synchronize()
            out.append(log[: int(n)].tolist())
        return out, seen, loop.mode

    runs, seen, _ = logs("host")
    last = min(stop_at, 30)  # the stop flag or the limit
    want = [*range(4, last + 1, 4)] + ([last] if last % 4 else [])
    assert runs == [want] * 3 and seen == want
    for unroll in (1, 2, 4, 3):  # 3 does not divide 4: blocks of 2
        assert logs("unrolled", unroll) == (runs, seen, "unrolled"), unroll
    if graphloop.conditional_support() is None:
        got, streamed, mode = logs("while")
        assert got == runs and mode == "while" and streamed[-1:] == seen[-1:]


@pytest.mark.gpu
@gpu
def test_a_torch_compile_step_in_the_while_body():
    if graphloop.conditional_support() is not None:
        pytest.skip(graphloop.conditional_support())
    ok, detail = graphloop._probe_compiled()
    assert ok is True, detail
    ex = _example()
    with torch.inference_mode():
        rows = ex.compare(max_new=96, runs=3, compile=True, layers=1, dim=128, vocab=512)
    for name, row in rows.items():
        print(f"{name:42s} {row}")
        assert row["same_tokens"], name
    compiled = rows["device loop (while, torch.compile step)"]
    assert compiled["mode"] == "while" and compiled["host_checks"] == 0


FAILED_CAPTURE_EXIT = """
import torch
from kernel_agent import graphloop

x = torch.zeros(4, 8, device="cuda")

def syncs(index, active):
    x[0] += float(x[0, 0].item())

loop = graphloop.device_loop(syncs, None, 4, masked=True, warmup_runs=0)
loop.run()
torch.cuda.synchronize()
print("mode", loop.mode)

def draws(index, active):  # a WHILE graph that reads its RNG state's scalars, alive at exit
    graphloop.masked_index_copy_(x, 0, index.view(1), torch.randn(1, 8, device="cuda"), active)

drawing = graphloop.device_loop(draws, None, 4, masked=True, warmup_runs=0)
drawing.run()
torch.cuda.synchronize()
print("drawing", drawing.mode)
"""


@pytest.mark.gpu
@gpu
def test_a_process_with_a_failed_capture_exits_cleanly():
    """#246: the abandoned builders of a failed capture are never destroyed, not even at
    interpreter shutdown (where destroying them crashed the process)."""
    if graphloop.conditional_support() is not None:
        pytest.skip(graphloop.conditional_support())
    done = subprocess.run(
        [sys.executable, "-c", FAILED_CAPTURE_EXIT], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, (done.returncode, done.stderr[-2000:])
    assert "mode host" in done.stdout and "drawing while" in done.stdout


@pytest.mark.gpu
@gpu
def test_probe_on_this_gpu():  # last: the tests above start from a fresh cuda.core context
    ok, detail = graphloop.probe()
    assert ok in (True, None), detail
    if graphloop.conditional_support() is None:
        assert ok is True and "with the plain loop's numbers" in detail, detail
