"""Declared concurrency (#147): kernel_agent.concurrency on a fake CUDA backend (CPU only),
the ka_launch.cuh header and the PDL example's fallback; GPU tests (marked) run the real
fork/join, a multi-stream capture and the PDL example through the evaluator."""

from __future__ import annotations

import contextlib
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from kernel_agent import concurrency as cc
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.selftest import GemvChain

# ------------------------------------------------------------------ fake CUDA backend


class FakeDevice:
    def __init__(self, index: int) -> None:
        self.index = index


class FakeStream:
    def __init__(self, backend: FakeCuda, handle: int, device: int = 0) -> None:
        self.backend, self.cuda_stream, self.device = backend, handle, FakeDevice(device)

    def wait_stream(self, other: FakeStream) -> None:
        self.backend.log.append(("wait_stream", self.cuda_stream, other.cuda_stream))

    def wait_event(self, event: FakeEvent) -> None:
        self.backend.log.append(("wait_event", self.cuda_stream, event.recorded_on))


class FakeEvent:
    def __init__(self) -> None:
        self.recorded_on: int | None = None

    def record(self, stream: FakeStream) -> None:
        self.recorded_on = stream.cuda_stream


class FakeCuda:
    """Records every stream operation; ``current`` is the current stream."""

    def __init__(self) -> None:
        self.log: list[tuple[Any, ...]] = []
        self.handles = iter(range(100, 10_000))
        self.current = FakeStream(self, 7)
        self.is_capturing = False

    def current_stream(self, device: int | None = None) -> FakeStream:
        return self.current

    def new_stream(self, device: int) -> FakeStream:
        return FakeStream(self, next(self.handles), device)

    @contextlib.contextmanager
    def use(self, stream: FakeStream) -> Iterator[FakeStream]:
        previous, self.current = self.current, stream
        try:
            yield stream
        finally:
            self.current = previous

    def event(self) -> FakeEvent:
        return FakeEvent()

    def capturing(self) -> bool:
        return self.is_capturing

    def device_sms(self, device: int) -> int:
        return 70


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
    backend = FakeCuda()
    monkeypatch.setattr(cc, "_cuda", backend)
    for name in ("_named", "_labels", "_in_partition", "_used"):
        monkeypatch.setattr(cc, name, {})
    monkeypatch.setattr(cc, "_pending", [])
    recorded: list[tuple[Any, int]] = []
    monkeypatch.setattr(cc, "_record", lambda tree, s: recorded.append((tree, s.cuda_stream)))
    backend.recorded = recorded  # type: ignore[attr-defined]
    return backend


# ------------------------------------------------------------------ fork / join


def test_fork_waits_for_the_caller_runs_the_block_on_the_side_and_joins(fake: FakeCuda) -> None:
    caller = fake.current
    with cc.fork("aux") as side:
        assert fake.current is side and side is not caller
        fake.log.append(("work", fake.current.cuda_stream))
    assert fake.current is caller
    s = side.cuda_stream
    assert fake.log == [("wait_stream", s, 7), ("work", s), ("wait_stream", 7, s)]
    assert cc.used() == ["aux"]
    with cc.fork("aux") as again:  # one cached stream per name
        pass
    assert again is side and cc.streams() == {"aux": side}


def test_fork_joins_when_the_block_raises(fake: FakeCuda) -> None:
    with pytest.raises(RuntimeError), cc.fork("aux") as side:
        raise RuntimeError("boom")
    assert fake.log[-1] == ("wait_stream", 7, side.cuda_stream)
    assert fake.current.cuda_stream == 7


def test_stream_names_are_checked(fake: FakeCuda) -> None:
    for bad in ("", "ka::mark", None):
        with pytest.raises(ValueError):
            cc.stream(bad)  # type: ignore[arg-type]
    cc.stream("a")
    cc.stream("b", device=1)
    assert cc.used() == ["a", "b:cuda1"]
    cc.reset()
    assert cc.used() == [] and set(cc.streams()) == {"a", "b:cuda1"}  # streams stay


# ------------------------------------------------------------------ launch / join_all


def test_launch_enqueues_now_and_joins_exactly_its_work(fake: FakeCuda) -> None:
    x = torch.ones(3)
    first = cc.launch("aux", lambda t, scale=1: t * scale, x, scale=2)
    side = first.stream.cuda_stream
    assert fake.current.cuda_stream == 7  # back on the caller's stream
    assert cc.outstanding() == ["aux"] and not first.joined
    assert fake.log == [("wait_stream", side, 7)]
    assert fake.recorded == [(((x,), {"scale": 2}), side)]  # inputs live until side is done
    second = cc.launch("aux", lambda: "later")
    assert first.result().tolist() == [2.0, 2.0, 2.0]
    assert fake.log[-1] == ("wait_event", 7, side)  # the handle's event, not the stream
    assert fake.recorded[-1][1] == 7  # the result is recorded on the joining stream
    assert first.result() is first.value and cc.outstanding() == ["aux"]  # joined once
    assert cc.join_all() == 1 and second.joined and cc.outstanding() == []
    assert cc.join_all() == 0


def test_launch_inside_a_capture_records_no_streams(fake: FakeCuda) -> None:
    fake.is_capturing = True
    handle = cc.launch("aux", lambda: torch.zeros(1))
    handle.result()
    assert fake.recorded == []  # the graph's private pool owns the memory


def test_side_stream_cannot_be_the_callers(fake: FakeCuda) -> None:
    side = cc.stream("aux")
    with fake.use(side), pytest.raises(ValueError), cc.fork("aux"):
        pass
    with fake.use(side), pytest.raises(ValueError):
        cc.launch("aux", lambda: None)


# ------------------------------------------------------------------ partitions


class FakeContext:
    def __init__(self, backend: FakeCuda) -> None:
        self.backend = backend

    def Stream(self) -> FakeStream:  # torch's GreenContext API
        return FakeStream(self.backend, next(self.backend.handles))


def test_sm_count_is_the_partitions_inside_one(fake: FakeCuda) -> None:
    part = cc.Partition(16, 0, FakeContext(fake), "16sm")
    with cc.fork("aux", partition=part) as side:
        assert cc.sm_count() == 16
    assert cc.sm_count() == 70 and cc.sm_count(side) == 16
    assert cc.used() == ["aux@16sm"]
    plain = cc.stream("aux")  # the same name outside the partition: another stream
    assert plain is not side and cc.sm_count(plain) == 70


def test_partition_says_why_it_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cc.partition_unsupported() == "no CUDA device"
    assert cc.partition(8) is None
    assert cc.partition_error() == "no CUDA device"


# ------------------------------------------------------------------ the C++ header, PDL


def test_header_has_the_launch_helper_and_the_device_idiom() -> None:
    header = (cc.include_dir() / cc.HEADER).read_text()
    for needle in (
        "inline cudaError_t ka_launch(",
        "cudaLaunchAttributeProgrammaticStreamSerialization",
        "cudaLaunchAttributeCooperative",
        "griddepcontrol.launch_dependents;",
        "griddepcontrol.wait;",
        "__CUDA_ARCH__ >= 900",
        "inline int ka_sm_count(cudaStream_t stream)",
        "cuStreamGetGreenCtx",
        "inline int ka_coresident_blocks(",
    ):
        assert needle in header, needle


def _example(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"ka_example_{name}", EXAMPLES_DIR / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pdl_example_includes_the_header_and_keeps_unsupported_modules() -> None:
    example = _example("cuda_pdl_gemv_chain.py")
    assert '#include "ka_launch.cuh"' in example.CUDA_SRC
    assert "ka_pdl_wait();" in example.CUDA_SRC and "ka_launch(" in example.CUDA_SRC
    chain = GemvChain(256, 3).to(torch.bfloat16)  # on the CPU: nothing to build
    assert example.build(chain) is chain
    linear = torch.nn.Linear(256, 256)  # no `layers`: not a chain
    assert example.build(linear) is linear


# ------------------------------------------------------------------ GPU (not run on CPU)


@pytest.mark.gpu
def test_fork_and_launch_on_cuda_join_and_capture() -> None:
    x = torch.randn(512, 512, device="cuda")
    expected = (x @ x).relu() + x.sin()
    with cc.fork("test-aux"):
        a = (x @ x).relu()
    b = x.sin()
    torch.testing.assert_close(a + b, expected)
    handle = cc.launch("test-aux", lambda t: (t @ t).relu(), x)
    torch.testing.assert_close(handle.result() + x.sin(), expected)
    assert cc.join_all() == 0
    # multi-stream capture: the fork and the join become graph edges
    static = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with cc.fork("test-aux"):
            a = (static @ static).relu()
        out = a + static.sin()
    graph.replay()
    torch.testing.assert_close(out, expected)
    assert "test-aux" in cc.used()


@pytest.mark.gpu
def test_pdl_example_passes_the_evaluator(tmp_path: Path) -> None:
    from kernel_agent import selftest, toolchain

    if not selftest.pdl_supported(toolchain.setup()):
        pytest.skip("PDL needs the cuda backend on sm_90+")
    assert selftest.smoke_pdl(tmp_path, verbose=True)
