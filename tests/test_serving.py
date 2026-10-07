"""Generic serving helpers (issue #149): asynchronous flag reads, host copies, the post stage
on a side stream, and continuous batching over a fake batched loop (CPU); the CUDA paths
of the flags and the side stream (GPU)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from kernel_agent import objective
from kernel_agent.hub import Modality
from kernel_agent.workloads import create_workload, validate_metric
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, timed_run
from kernel_agent.workloads.serving import (
    AsyncFlags,
    HostCopy,
    ServingOptions,
    SideStage,
    per_request,
    serve,
    serving_options,
)


class Counter:
    """A fake batched generation loop (a :class:`SlotModel`): slot *s* holding request *m*
    outputs ``100 m + k`` at its step *k*; request *m*'s stop flag rises at its step
    ``stops[m] - 1``. ``finish`` returns a post-stage job that makes one row per request."""

    def __init__(self, slots: int, stops: list[int], *, send: bool = True) -> None:
        self.slots = slots
        self.stops = stops
        self.sends = send
        self.request = torch.full((slots,), -1)
        self.k = torch.zeros(slots, dtype=torch.long)
        self.records: list[torch.Tensor] = []
        self.log: list[tuple[Any, ...]] = []

    def admit(self, slots: list[int], requests: list[int]) -> None:
        self.log.append(("admit", slots, requests))
        self.request[slots] = torch.tensor(requests)
        self.k[slots] = 0

    def step(self, send: Callable[[torch.Tensor], int]) -> None:
        stop_at = torch.tensor([self.stops[m] if m >= 0 else 1 for m in self.request.tolist()])
        if self.sends:
            send(self.k + 1 >= stop_at)
        self.records.append(self.request * 100 + self.k)

    def advance(self) -> None:
        self.log.append(("advance",))
        self.k += 1

    def finish(
        self, slots: list[int], requests: list[int], steps: int
    ) -> Callable[[], torch.Tensor] | None:
        self.log.append(("finish", slots, requests, steps))
        return lambda: torch.stack([torch.arange(steps) + 100 * m for m in requests])


STOPS = [4, 1, 1, 1, 1]  # request 0 is long, the others stop after one step


def test_continuous_batching_refills_a_slot_as_soon_as_its_request_stops():
    model, ready = Counter(2, STOPS), []
    served = serve(
        model, 5, continuous=True, max_steps=10, on_ready=lambda m, n, out: ready.append(m)
    )
    assert served.steps == STOPS
    assert served.iterations == 4  # request 0's 4 steps; the short ones ride in slot 1
    assert served.slots == [0, 1, 1, 1, 1] and served.first == [0, 0, 1, 2, 3]
    assert served.tables[1] == [(0, 1), (2, 0)]
    admits = [e for e in model.log if e[0] == "admit"]
    assert admits == [("admit", [0, 1], [0, 1]), *(("admit", [1], [m]) for m in (2, 3, 4))]
    assert ready == [1, 2, 3, 0, 4]  # each request when its output arrived
    for m, out in enumerate(served.outputs):
        assert out is not None and out.tolist() == [100 * m + k for k in range(STOPS[m])]
    # the per-step batch records, regrouped per request
    latents, steps = per_request(model.records, served.tables, 5)
    assert steps == STOPS and latents.shape == (4, 5)
    assert latents[:, 0].tolist() == [0, 1, 2, 3]
    assert latents[:, 3].tolist() == [300, 0, 0, 0]  # zero past the request's last step


def test_static_batching_waits_for_the_whole_batch():
    model = Counter(2, STOPS)
    served = serve(model, 5, continuous=False, max_steps=10)
    assert served.steps == STOPS and served.iterations == 6
    assert served.first == [0, 0, 4, 4, 5]
    assert served.tables[1] == [(0, 1), None]  # slot 1 idles until request 0 stops
    assert ("admit", [0, 1], [2, 3]) in model.log and ("admit", [0], [4]) in model.log
    # no `advance` when nothing goes on: the next batch is prefilled instead
    assert model.log.count(("advance",)) == 3


def test_limits_min_steps_and_the_stop_check():
    model = Counter(3, [1, 1, 1, 1])
    served = serve(model, 4, continuous=True, max_steps=[3, 2, 5, 1], stop=False)
    assert served.steps == [3, 2, 5, 1]  # fixed lengths: no flags read
    served = serve(Counter(2, [1, 1]), 2, continuous=True, max_steps=6, min_steps=3)
    assert served.steps == [4, 4]  # a flag counts from step 3 on
    with pytest.raises(RuntimeError, match="sent no stop flags"):
        serve(Counter(2, [1, 1], send=False), 2, continuous=True, max_steps=4)
    with pytest.raises(ValueError, match="one limit"):
        serve(Counter(2, [1, 1]), 2, continuous=True, max_steps=[3])
    with pytest.raises(ValueError, match="requests must be >= 1"):
        serve(Counter(2, [1, 1]), 0, continuous=True, max_steps=3)


def test_post_stage_batches_requests_that_stop_together():
    model = Counter(4, [2, 2, 2, 3])
    served = serve(model, 4, continuous=False, max_steps=5, post_batch=2)
    finishes = [e for e in model.log if e[0] == "finish"]
    assert finishes == [
        ("finish", [0, 1], [0, 1], 2),
        ("finish", [2], [2], 2),
        ("finish", [3], [3], 3),
    ]
    assert served.outputs[3] is not None and served.outputs[3].tolist() == [300, 301, 302]
    stage = SideStage("cpu", pipelined=True)  # no CUDA: inline, the same results
    again = serve(Counter(4, [2, 2, 2, 3]), 4, continuous=False, max_steps=5, stage=stage)
    assert not stage.pipelined and stage.submitted == 2  # no post_batch: one call per length
    assert [o.tolist() for o in again.outputs if o is not None] == [
        o.tolist() for o in served.outputs if o is not None
    ]


def test_per_request_needs_one_record_per_step():
    with pytest.raises(ValueError, match="record exactly once per step"):
        per_request([torch.zeros(2)], [[(0, 0), None], [(0, 1), None]], 1)
    empty, steps = per_request([], [], 3)
    assert empty.numel() == 0 and steps == [0, 0, 0]


def test_async_flags_on_the_host():
    flags = AsyncFlags(depth=2)
    a = flags.send(torch.tensor([0, 1]))
    b = flags.send(torch.tensor([1, 1]))
    assert flags.read(a).tolist() == [0, 1] and flags.read().tolist() == [1, 1]
    flags.send(torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="overwritten"):
        flags.read(a)  # depth 2: a step later is fine, two steps later is not
    assert flags.read(b).tolist() == [1, 1]
    with pytest.raises(ValueError, match="never sent"):
        flags.read(7)
    with pytest.raises(ValueError, match="depth"):
        AsyncFlags(0)
    copy = HostCopy(torch.arange(3))
    assert copy.ready() and copy.wait().tolist() == [0, 1, 2]


def test_serving_options():
    assert serving_options({}) is None and serving_options({"serving": "off"}) is None
    opt = serving_options({"serving": "Continuous", "requests": "12", "pipeline": "true"})
    assert opt == ServingOptions("continuous", 12, True) and opt.continuous
    assert serving_options({"serving": "static"}) == ServingOptions("static", None, False)
    with pytest.raises(ValueError, match="choose one of static, continuous"):
        serving_options({"serving": "dynamic"})
    assert serving_options({"pipeline": True, "requests": 3}) is None  # not opted in
    with pytest.raises(ValueError, match="requests must be >= 1"):
        serving_options({"serving": "static", "requests": 0})


class Server(Workload):
    """A throughput workload served by :func:`serve` on the fake loop: one request per
    text, each marked when its output reached the host (40 ms of audio per step)."""

    modality = Modality.TTS
    metrics = (objective.THROUGHPUT,)
    supports_serving = True
    defaults = {"metric": "throughput", "batch_size": 2}

    def load(self) -> None:
        pass

    def roots(self) -> dict[str, nn.Module]:
        return {}

    def make_inputs(self) -> list[int]:
        opt = self.serving()
        return STOPS[: opt.requests] if opt and opt.requests else STOPS[:2]

    def run(self, inputs: list[int]) -> dict[str, Any]:
        opt = self.serving()
        model = Counter(int(self.options["batch_size"]), list(inputs))
        served = serve(
            model,
            len(inputs),
            continuous=bool(opt and opt.continuous),
            max_steps=10,
            flags=self.async_flags(),
            on_ready=lambda m, n, out: self.mark_ready(audio_ms=40.0 * n),
        )
        return {"steps": served.steps}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(reference == candidate)


def test_a_served_workload_reports_each_requests_latency():
    wl = Server(WorkloadSpec("toy/served", "tts", options={"serving": "continuous"}))
    with wl.with_options({"requests": 5}):
        out, _, detail = timed_run(wl, wl.make_inputs())
    assert out["steps"] == STOPS and detail["requests"] == 5
    assert detail["audio_s"] == pytest.approx(sum(STOPS) * 0.04)
    assert detail["request_ms"] <= detail["request_ms_max"] <= detail["run_ms"]
    assert 0 < detail["request_ms_mean"] <= detail["request_ms_max"]
    assert "(median; max " in objective.detail_text(detail)
    assert wl.serving() == ServingOptions("continuous")
    assert Server(WorkloadSpec("toy/served", "tts")).serving() is None  # the default
    wl.check_serving()


def test_a_workload_without_serving_rejects_the_option():
    spec = WorkloadSpec("openbmb/VoxCPM2", "tts", family="voxcpm", options={"batch_size": 4})
    create_workload(spec)  # the default benchmark
    spec.options["serving"] = "continuous"
    with pytest.raises(ValueError, match="VoxCPMBatchWorkload does not implement -o serving"):
        create_workload(spec)
    with pytest.raises(ValueError, match="does not implement"):
        validate_metric(spec)


# ---------------------------------------------------------------- GPU


@pytest.mark.gpu
def test_async_flags_and_the_side_stage_on_cuda():
    flags = AsyncFlags()
    x = torch.randn(64, 2, device="cuda")
    tickets = [flags.send((x * i).argmax(dim=-1)) for i in range(2)]
    torch.cuda._sleep(10_000_000)  # the GPU is busy: the reads wait for their copies only
    for i, ticket in enumerate(tickets):
        assert torch.equal(flags.read(ticket), (x * i).argmax(dim=-1).cpu())
    model = Counter(2, STOPS)
    stage = SideStage("cuda", pipelined=True)
    inline = serve(Counter(2, STOPS), 5, continuous=True, max_steps=10)
    served = serve(model, 5, continuous=True, max_steps=10, stage=stage)
    assert stage.pipelined and stage.submitted == 4
    assert [o.tolist() for o in served.outputs] == [o.tolist() for o in inline.outputs]
    copy = HostCopy(torch.arange(4, device="cuda"))
    assert copy.wait().tolist() == [0, 1, 2, 3] and copy.ready()
