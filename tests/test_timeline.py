"""The profile's timeline (#146): union busy over streams, stages, idle gaps by cause and the
critical path, on synthetic Kineto traces of two workload shapes (a TTS patch loop with a
second stream and a per-patch host sync, an LLM decode loop), and the stage ranges on CPU."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from kernel_agent.profiling import profiler, timeline
from kernel_agent.profiling.timeline import Event, StageRanges


class Trace:
    """A Kineto chrome trace written by hand (times in us, as Kineto writes them)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._corr = 0

    def range(self, name: str, ts: float, dur: float) -> None:
        self.events.append(
            {"ph": "X", "cat": "user_annotation", "name": f"ka::{name}", "ts": ts, "dur": dur}
        )

    def api(self, name: str, ts: float, dur: float = 2.0) -> int:
        self._corr += 1
        self.events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": name,
                "ts": ts,
                "dur": dur,
                "tid": 1,
                "args": {"correlation": self._corr},
            }
        )
        return self._corr

    def gpu(
        self,
        corr: int,
        ts: float,
        dur: float,
        name: str = "kernel",
        *,
        stream: int = 7,
        cat: str = "kernel",
        grid: list[int] | None = None,
    ) -> None:
        args: dict[str, Any] = {"correlation": corr, "stream": stream}
        if grid is not None:
            args |= {"grid": grid, "block": [128, 1, 1], "est. achieved occupancy %": 50}
        self.events.append(
            {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "args": args}
        )

    def kernel(
        self, launch_at: float, ts: float, dur: float, name: str = "kernel", **kw: Any
    ) -> int:
        corr = self.api("cudaLaunchKernel", launch_at)
        self.gpu(corr, ts, dur, name, **kw)
        return corr

    def events_(self) -> list[Event]:
        return timeline.from_chrome_trace({"traceEvents": self.events})


def test_union_and_flatten():
    assert timeline.union([(5, 9), (0, 3), (2, 4), (9, 10)]) == [[0, 4], [5, 10]]
    # a parent [0, 100) with children [10, 20) and [30, 40), the second with a grandchild
    segs = timeline.flatten([(0, 100, 0), (10, 20, 1), (30, 40, 2), (32, 35, 3)])
    assert segs == [
        (0, 10, 0),
        (10, 20, 1),
        (20, 30, 0),
        (30, 32, 2),
        (32, 35, 3),
        (35, 40, 2),
        (40, 100, 0),
    ]


def test_two_overlapping_streams_are_busy_once_and_the_view_stays_reliable():
    t = Trace()
    t.range("run", 0, 220)
    t.kernel(1, 5, 100, "a", stream=7)
    t.kernel(2, 105, 100, "b", stream=7)
    t.kernel(3, 55, 100, "side", stream=13)  # overlaps both
    tl = timeline.analyze(t.events_())
    assert tl["busy_ms"] == 0.2 and tl["kernel_time_ms"] == 0.3 and tl["overlap_ms"] == 0.1
    assert [(s["stream"], s["busy_ms"]) for s in tl["streams"]] == [(7, 0.2), (13, 0.1)]
    assert tl["idle_ms"] == pytest.approx(0.02) and tl["lead_in_ms"] == 0.005
    # the union is within the run measured without the profiler; the sum would not be
    assert profiler.kernel_problems([tl["busy_ms"]], 0.21) == []
    assert profiler.kernel_problems([tl["kernel_time_ms"]], 0.21)
    text = "\n".join(timeline.markdown(tl))
    assert "2 streams: stream 7 0.20 ms busy" in text and "they overlap 0.10 ms" in text


def patch_loop(patches: int = 3) -> tuple[Trace, list[dict[str, Any]]]:
    """A TTS-like patch loop: a graph-replayed solver, an encoder, an LM step, a stop head
    whose flag the host reads (device-to-host copy + stream sync), and in patch 1 a decoder
    of finished output on a second stream. Returns the trace and the stage calls with their
    tensors (storage pointers) as :class:`StageRanges` records them."""
    t = Trace()
    calls: list[dict[str, Any]] = []
    t.range("run", 0, 400 * patches + 50)
    lm_out = 1000

    def call(label: str, inputs: list[int], outputs: list[int]) -> None:
        calls.append(
            {
                "label": label,
                "parent": -1,
                "inputs": [(p, 4096, True) for p in inputs],
                "outputs": outputs,
            }
        )

    for p in range(patches):
        t0 = 10 + 400 * p
        dit_out, enc_out, new_lm, stop_out = 10 * p + 1, 10 * p + 2, 10 * p + 3, 10 * p + 4
        t.range("model.feat_decoder", t0, 100)
        corr = t.api("cudaGraphLaunch", t0 + 1)
        for i in range(3):  # a graph's kernels share the replay's correlation id
            t.gpu(corr, t0 + 5 + 30 * i, 30, "solver_step", grid=[140, 1, 1])
        call("model.feat_decoder", [lm_out], [dit_out])
        t.range("model.feat_encoder", t0 + 100, 20)
        t.kernel(t0 + 101, t0 + 103, 10, "encoder", grid=[35, 1, 1])  # host late: 8 us
        call("model.feat_encoder", [dit_out], [enc_out])
        t.range("model.base_lm.forward_step", t0 + 120, 20)
        t.kernel(t0 + 121, t0 + 125, 60, "lm_layer", grid=[70, 1, 1])
        call("model.base_lm.forward_step", [enc_out], [new_lm])
        if p == 1:  # finished output decoded on a second stream, overlapping the LM step
            t.range("model.audio_vae.decode", t0 + 130, 5)
            t.kernel(t0 + 131, t0 + 133, 40, "vae", stream=13)
            call("model.audio_vae.decode", [dit_out], [500])
        t.range("model.stop_head", t0 + 140, 10)
        t.kernel(t0 + 141, t0 + 185, 2, "stop")
        corr = t.api("cudaMemcpyAsync", t0 + 145)
        t.gpu(corr, t0 + 187, 1, "Memcpy DtoH (Device -> Pinned)", cat="gpu_memcpy")
        t.api("cudaStreamSynchronize", t0 + 146, 43)  # returns once the copy is done
        call("model.stop_head", [new_lm], [stop_out])
        lm_out = new_lm
    return t, calls


def test_a_patch_loop_attributes_stages_gaps_and_the_critical_path():
    t, calls = patch_loop()
    tl = timeline.analyze(t.events_(), calls=calls, sms=70)
    stages = {s["stage"]: s for s in tl["stages"]}
    solver = stages["model.feat_decoder"]
    assert solver["calls"] == 3 and solver["kernels"] == 9 and solver["graph_events"] == 9
    assert solver["gpu_ms"] == 0.27 and solver["sm_fill"] == 1.0 and solver["occupancy_pct"] == 50.0
    assert stages["model.feat_encoder"]["sm_fill"] == 0.5
    assert stages["model.stop_head"]["events"] == 6  # the kernel and the copy, per patch
    assert stages["model.audio_vae.decode"]["calls"] == 1
    assert {s["stream"] for s in tl["streams"]} == {7, 13}
    assert tl["overlap_ms"] > 0  # the decoder ran beside the LM step

    gaps = tl["gaps"]
    causes = gaps["causes"]
    # after each stop flag's device-to-host copy the GPU idles until the next solver replay
    assert causes["host sync"]["count"] == 2
    assert causes["host sync"]["ms"] == pytest.approx(2 * (405 - 188) / 1e3)
    # the encoder and the LM step were launched after the work before them had ended
    assert causes["host late"]["count"] == 6
    place = gaps["by_place"][0]
    assert place == {
        "cause": "host sync",
        "before": "model.stop_head",
        "after": "model.feat_decoder",
        "count": 2,
        "ms": pytest.approx(0.434),
        "median_us": 217.0,
        "after_event": "Memcpy DtoH (Device -> Pinned)",
    }
    assert gaps["largest"][0]["cause"] == "host sync"
    assert sum(b["count"] for b in gaps["bins"]) >= 7

    crit = tl["critical_path"]
    assert crit["calls"] == len(calls) and crit["assumed_edges"] == 0
    # solver -> encoder -> LM step through 3 patches, ending with the last stop head; the
    # other stop heads and the decoder on the second stream are off it
    assert crit["critical_path_ms"] == pytest.approx(3 * (0.09 + 0.01 + 0.06) + 0.003)
    off = {o["stage"]: o for o in crit["off_path"]}
    assert set(off) == {"model.audio_vae.decode", "model.stop_head"}
    assert off["model.stop_head"]["calls"] == 2
    assert crit["overlap_potential_ms"] == pytest.approx(0.04 + 2 * 0.003)

    text = "\n".join(timeline.markdown(tl))
    assert "| host sync | `model.stop_head` | `model.feat_decoder` | 2 |" in text
    assert "critical path" in text and "`model.audio_vae.decode` 0.04 ms" in text


def test_an_llm_decode_loop_with_item_syncs_and_pageable_token_copies():
    t = Trace()
    t.range("run", 0, 1000)
    t0 = 10
    for _ in range(4):  # tokens
        t.range("model.model", t0, 50)
        end = t0 + 5
        for i in range(5):  # one layer kernel per launch, launched early (host ahead)
            launch = t0 + 1 + i
            start = end + (3 if i == 2 else 0)  # one 3 us gap: launch latency
            t.kernel(launch, start, 20, f"layer{i}")
            end = start + 20
        t.range("model.lm_head", t0 + 50, 10)
        t.kernel(t0 + 51, end, 15, "lm_head_gemm")
        end += 15
        corr = t.api("cudaMemcpyAsync", t0 + 56)  # next_token.item()
        t.gpu(corr, end + 1, 1, "Memcpy DtoH (Device -> Pageable)", cat="gpu_memcpy")
        t.api("cudaStreamSynchronize", t0 + 57, end + 3 - (t0 + 57))
        host = end + 40  # Python decides the next token, then copies it from pageable memory
        corr = t.api("cudaMemcpyAsync", host)
        t.gpu(corr, host + 5, 2, "Memcpy HtoD (Pageable -> Device)", cat="gpu_memcpy")
        t0 = host + 20  # the next step's launches come after the copy returned
    tl = timeline.analyze(t.events_())
    causes = tl["gaps"]["causes"]
    assert causes["host sync"]["count"] == 4  # .item() before every pageable token copy
    assert causes["pageable copy"]["count"] == 3  # the next step waits for the copy
    assert causes["launch latency"]["count"] == 4
    stages = {s["stage"]: s for s in tl["stages"]}
    assert stages["model.model"]["kernels"] == 20 and stages["model.lm_head"]["calls"] == 4
    # the copies were launched outside any stage range: inside the run, no stage
    assert stages[timeline.OTHER]["events"] == 4
    assert tl["host_lead_ms"] is not None and "critical_path" not in tl
    text = "\n".join(timeline.markdown(tl))
    assert "pageable copy" in text and "| `model.model` | 4 | 20 |" in text


def test_events_without_ranges_or_gpu_work():
    t = Trace()
    t.kernel(0, 2, 10)
    t.kernel(1, 20, 10)
    tl = timeline.analyze(t.events_())
    assert tl["window_ms"] == 0.03 and tl["stages"][0]["stage"] == timeline.OTHER
    with pytest.raises(ValueError, match="no GPU events"):
        timeline.analyze([])


def test_the_critical_path_assumes_glue_edges_and_ignores_small_host_inputs():
    calls = [
        {"label": "a", "parent": -1, "inputs": [(1, 4096, True)], "outputs": [2]},
        # made from a's output by a torch.cat between stages: assumed to follow a
        {"label": "b", "parent": -1, "inputs": [(3, 4096, True)], "outputs": [4]},
        # a position tensor from the host and b's output
        {"label": "c", "parent": -1, "inputs": [(4, 4096, True), (5, 1, False)], "outputs": [6]},
        # independent of all (an input seen before, nobody wrote it): off the path
        {"label": "d", "parent": -1, "inputs": [(1, 4096, True)], "outputs": [7]},
    ]
    labels = ["a", "b", "c", "d"]
    per = {0: [(0, 1000)], 1: [(1000, 3000)], 2: [(3000, 4000)], 3: [(4000, 9000)]}
    crit = timeline.critical_path(calls, labels, per)
    assert crit["assumed_edges"] == 1 and crit["edges"] == 2
    assert crit["critical_path_ms"] == pytest.approx(0.005)  # d alone is the longest
    assert {o["stage"] for o in crit["off_path"]} == {"a", "b", "c"}
    mismatch = timeline.critical_path(calls[:3], labels, per)
    assert "disagree" in mismatch["error"]


class FakeKineto:
    """A ``_KinetoEvent`` (the methods :func:`timeline.from_profiler` calls)."""

    def __init__(self, kind: str, name: str, start: int, end: int, corr: int, where: int) -> None:
        self.kind, self._name, self.start, self.end = kind, name, start, end
        self.corr, self.where = corr, where

    def activity_type(self) -> str:
        return self.kind

    def name(self) -> str:
        return self._name

    def start_ns(self) -> int:
        return self.start

    def end_ns(self) -> int:
        return self.end

    def correlation_id(self) -> int:
        return self.corr

    def device_resource_id(self) -> int:
        return self.where

    def start_thread_id(self) -> int:
        return 1

    def metadata_json(self) -> str:
        return '"grid": [4, 1, 1], "stream": 7' if self.kind == "kernel" else ""


class FakeProf:
    def __init__(self, events: list[FakeKineto]) -> None:
        results = type("Results", (), {"events": lambda _self: events})()
        self.profiler = type("P", (), {"kineto_results": results})()


def fake_prof() -> FakeProf:
    return FakeProf(
        [
            FakeKineto("user_annotation", "ka::run", 0, 100_000, -1, 1),
            FakeKineto("user_annotation", "other annotation", 0, 1, -1, 1),
            FakeKineto("cpu_op", "aten::mm", 1_000, 2_000, -1, 1),
            FakeKineto("cuda_runtime", "cudaLaunchKernel", 1_000, 3_000, 9, 1),
            FakeKineto("kernel", "gemm", 5_000, 45_000, 9, 7),
        ]
    )


def test_from_profiler_reads_kineto_events():
    events = timeline.from_profiler(fake_prof())
    assert [e.kind for e in events] == ["user_annotation", "cuda_runtime", "kernel"]
    kernel = events[-1]
    assert kernel.stream == 7 and kernel.meta == {"grid": [4, 1, 1]}
    tl = timeline.analyze(events)
    assert tl["busy_ms"] == 0.04 and tl["host_lead_ms"] == 0.004


def test_the_kernel_view_uses_the_union_and_summarize_prints_the_timeline(monkeypatch):
    monkeypatch.setattr(profiler, "_roofline_note", list)  # no toolchain (CUDA) lookup
    view = profiler.timeline_view(fake_prof(), None)
    assert view["timeline"]["busy_ms"] == 0.04 and view["overlap_ms"] == 0.0
    assert "timeline_error" in profiler.timeline_view(FakeProf([]), None)
    t, calls = patch_loop()
    tl = timeline.analyze(t.events_(), calls=calls)
    kv = {
        "gpu_busy_ms": tl["busy_ms"],
        "gpu_busy_fraction": 0.9,
        "kernel_launches": 10,
        "avg_kernel_us": 1.0,
        "kernels": [],
        "aten_ops": [],
        "reliable": True,
        "attempts": [],
        "timeline": tl,
    }
    text = profiler.summarize({"module_calls": 1, "classes": [], "kernel_view": kv}, 1.3)
    assert "GPU busy (union over streams)" in text and "## Timeline" in text
    assert "host sync" in text


# ------------------------------------------------------------------ stage ranges (CPU)


class Lm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_step(x)  # the same instance: one stage call

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm = Lm()
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lm.forward_step(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def test_stage_modules_take_roots_and_children_with_lists_flattened():
    model = Model()
    found = {label for _, label in timeline.stage_modules({"model": model}).values()}
    assert found == {"model", "model.lm", "model.layers.*", "model.head"}
    deeper = {label for _, label in timeline.stage_modules({"model": model}, depth=2).values()}
    assert "model.lm.proj" in deeper


def test_stage_ranges_record_calls_and_restore_the_model():
    model = Model()
    x = torch.randn(2, 8)
    stages = StageRanges({"model": model}, {Lm: ["forward_step"]})
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof,
        stages,
        torch.inference_mode(),
    ):
        model(x)
        model.lm(x)  # forward -> forward_step of the same instance: one call
    labels = [c["label"] for c in stages.calls]
    assert labels == [
        "model",
        "model.lm.forward_step",
        "model.layers.*",
        "model.layers.*",
        "model.head",
        "model.lm",
    ]
    assert [c["parent"] for c in stages.calls] == [-1, 0, 0, 0, 0, -1]
    lm, first, second, head = stages.calls[1:5]
    assert first["inputs"][0][0] == lm["outputs"][0]  # producer -> consumer by storage
    assert head["inputs"][0][0] == second["outputs"][0]
    assert "forward_step" not in vars(model.lm) and not model._forward_pre_hooks
    names = [e.name for e in timeline.from_profiler(prof) if e.kind == timeline.ANNOTATION]
    assert "ka::model.lm.forward_step" in names and names.count("ka::model.layers.*") == 2
    stages.reset()
    assert stages.calls == []


def test_stage_ranges_of_a_workload_without_roots_is_none():
    assert StageRanges.of(object()) is None

    class W:
        options = {"stage_depth": 2}

        def roots(self) -> dict[str, nn.Module]:
            return {"model": Model()}

    stages = StageRanges.of(W())
    assert stages is not None and any(
        label == "model.lm.proj" for _, label in stages.modules.values()
    )
