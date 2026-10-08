"""Host-sync detector of the profile (issue #149): which torch calls make the host wait for
the GPU, their call sites, and the summary's transform opportunities. On the CPU a
predicate stands in for "this tensor is on the GPU"; the real classification on CUDA is a
GPU test."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.profiling import host_sync, profiler
from kernel_agent.profiling.host_sync import Recorder, classify, scan, summary_lines
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec

HOST, GPU = torch.zeros(4), torch.ones(4)


def gpu(t: Any) -> bool:
    return t is GPU or (isinstance(t, torch.Tensor) and t.dtype == torch.bool)


@pytest.mark.parametrize(
    ("func", "args", "kwargs", "want"),
    [
        (torch.Tensor.item, (GPU,), {}, ("item", ".item()")),
        (torch.Tensor.tolist, (GPU,), {}, ("item", ".tolist()")),
        (torch.Tensor.__bool__, (GPU,), {}, ("item", "bool(tensor)")),
        (torch.Tensor.item, (HOST,), {}, None),
        (torch.Tensor.cpu, (GPU,), {}, ("d2h", ".cpu()")),
        (torch.Tensor.to, (GPU, "cpu"), {}, ("d2h", '.to("cpu")')),
        (torch.Tensor.to, (GPU, "cpu"), {"non_blocking": True}, None),
        (torch.Tensor.to, (HOST, "cuda"), {}, ("h2d", ".to(device)")),
        (torch.Tensor.to, (HOST, "cuda"), {"non_blocking": True}, ("h2d", ".to(device)")),
        (torch.Tensor.to, (HOST, torch.float16), {}, None),
        (torch.Tensor.cuda, (HOST,), {}, ("h2d", ".cuda(device)")),
        (torch.Tensor.copy_, (GPU, HOST), {}, ("h2d", "device.copy_(host)")),
        (torch.Tensor.copy_, (HOST, GPU), {}, ("d2h", "host.copy_(device)")),
        (
            torch.tensor,
            ([3],),
            {"device": "cuda"},
            ("host_tensor", "torch.tensor(..., device=...)"),
        ),
        (torch.tensor, ([3],), {"device": "cpu"}, None),
        (torch.as_tensor, (GPU,), {"device": "cuda"}, None),
        (torch.Tensor.nonzero, (GPU,), {}, ("data_dependent", "nonzero()")),
        (torch.where, (GPU,), {}, ("data_dependent", "torch.where(condition)")),
        (torch.where, (GPU, HOST, HOST), {}, None),
        (torch.Tensor.__getitem__, (GPU, GPU > 0), {}, ("data_dependent", "tensor[mask]")),
        (torch.Tensor.__getitem__, (GPU, 0), {}, None),
        (torch.Tensor.add, (GPU, GPU), {}, None),
    ],
)
def test_classify(func, args, kwargs, want):
    assert classify(func, args, kwargs, gpu) == want


def loop(steps: int) -> torch.Tensor:
    x = torch.ones(4)
    for _ in range(steps):
        if (x > 0).all().item():  # the stop check of a decode loop
            x = x + 1
    return x.cpu()


class Looper(Workload):
    modality = Modality.LLM

    def load(self) -> None:
        self.model = nn.Linear(4, 4)

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> int:
        return 5

    def run(self, inputs: int) -> torch.Tensor:
        return self.model(loop(inputs))

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.equal(reference, candidate)))


def test_call_sites_are_counted_per_run():
    with Recorder(lambda t: isinstance(t, torch.Tensor)) as recorder:
        loop(5)
    rows = recorder.rows()
    first = inspect.getsourcelines(loop)[1]
    assert [(r["kind"], r["op"], r["calls"], r["line"] - first) for r in rows] == [
        ("item", ".item()", 5, 3),
        ("d2h", ".cpu()", 1, 5),
    ]
    item = rows[0]
    assert item["function"] == "loop" and item["file"].endswith("test_host_sync.py")
    assert item["code"] == "if (x > 0).all().item():  # the stop check of a decode loop"
    assert item["where"] == "other" and item["host_ms"] >= 0
    with Recorder() as recorder:  # the real predicate: nothing on the CPU syncs
        loop(5)
    assert recorder.rows() == []


def test_scan_and_the_summary(monkeypatch):
    wl = Looper(WorkloadSpec("toy/looper", "llm"))
    wl.load()
    found = scan(wl, wl.make_inputs(), is_device=lambda t: isinstance(t, torch.Tensor))
    assert found["calls"] == 6 and "error" not in found
    text = "\n".join(summary_lines(found))
    assert "## Host synchronisation (transform opportunities)" in text
    assert "6 calls in one run" in text and "1 of them more than once per run" in text
    assert "| 5 | device value read on the host | `.item()` | other |" in text
    assert "**async flag read**: " in text and "Workload.async_flags()" in text
    assert "build constant once" not in text  # only the fixes of the sites shown
    assert "none found" in "\n".join(summary_lines(scan(wl, 5)))  # real predicate, CPU
    assert summary_lines(None) == []

    def broken(inputs: int) -> None:
        torch.ones(1).item()
        raise RuntimeError("compiled region")

    monkeypatch.setattr(wl, "run", broken)
    failed = scan(wl, 5, is_device=lambda t: isinstance(t, torch.Tensor))
    assert failed["error"] == "RuntimeError: compiled region" and failed["calls"] == 1
    assert "the scan run failed (RuntimeError: compiled region)" in "\n".join(summary_lines(failed))


def test_the_profile_lists_host_syncs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(host_sync, "on_device", lambda t: isinstance(t, torch.Tensor))
    wl = Looper(WorkloadSpec("toy/looper", "llm"))
    wl.load()
    profile = profiler.profile_workload(wl, wl.make_inputs())
    sites = profile["host_syncs"]["sites"]
    assert sites[0]["op"] == ".item()" and sites[0]["calls"] == 5
    text = profiler.summarize(profile, 1.0)
    assert "## Host synchronisation (transform opportunities)" in text
    assert "(`loop`): `if (x > 0).all().item():" in text


@pytest.mark.gpu
def test_host_syncs_on_cuda():
    def step(x: torch.Tensor) -> torch.Tensor:
        pos = torch.tensor([3], device="cuda")  # a position built on the host per step
        if bool(x.sum() > 0):
            x = x + pos
        return x

    x = torch.ones(8, device="cuda")
    with Recorder() as recorder:
        for _ in range(3):
            x = step(x)
        x.cpu()
    kinds = {(r["kind"], r["calls"]) for r in recorder.rows()}
    assert kinds == {("host_tensor", 3), ("item", 3), ("d2h", 1)}


def test_the_systems_agent_gets_the_patterns():
    from kernel_agent.agent import prompts

    card = {"repo_id": "o/m", "modality": "tts"}
    text = prompts.systems_prompt(card, {"median_ms": 2.0}, "", [], "py", "tc", 4)
    assert "`kernel-agent:systems-patterns`" in text  # the skill to load (#176)
    patterns = prompts.knowledge("systems-patterns")
    assert "# Systems patterns: host syncs, side streams, serving" in patterns
    assert "workload.async_flags()" in patterns and "SideStage" in patterns
    assert "Host synchronisation" in prompts.knowledge("playbook.md")
