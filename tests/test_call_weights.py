"""The calls behind a kernel's estimated saving (issue #119, ``kernels/weights.py``).

VoxCPM2's ``dit_layer`` target (``MiniCPMDecoderLayer``) covers 12 LocDiT layers, called 540
times each per run, and 12 LocEnc layers, called 60 times each at a smaller shape. Its
capture keeps a LocDiT layer, and the evaluator used to count 540 calls × 24 instances =
12,960 calls (the even split) for the 7,212 of the run. A case now stands for the calls of
every instance with its primary input: the LocDiT layers' 6,480.

The toy model below is that class in two instance groups whose call counts differ by 9x.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from synthetic_run import make_run
from torch import nn

from kernel_agent import projection, report, scheduler
from kernel_agent.hub import Modality
from kernel_agent.kernels import weights
from kernel_agent.profiling.capture import capture_calls, capture_module, load_capture
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec
from kernel_agent.workspace import RunDir, append_jsonl, read_json, write_json

D = 8
DIT_CALLS, ENC_CALLS = 9, 1  # per run and layer: the LocDiT-like layers 9x the LocEnc-like
REGEX = r"dit\.|enc\."
DIT, ENC = "model.dit.layers.*", "model.enc.layers.*"


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin(x)


class Stack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(2))


class TwinModel(nn.Module):
    """One class in two instance groups: ``dit`` (first) and ``enc``."""

    def __init__(self) -> None:
        super().__init__()
        self.dit = Stack()
        self.enc = Stack()


class TwinWorkload(Workload):
    """Each ``dit`` layer runs 9 times per run at ``[2, 3, D]``, each ``enc`` layer once,
    at the same shape or (``enc_shape``) at another one, as VoxCPM2's LocEnc layers do."""

    modality = Modality.TTS
    enc_shape: tuple[int, ...] = (2, 3, D)

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = TwinModel().to(self.device).eval()

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(2, 3, D, device=self.device)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            x = inputs
            for _ in range(DIT_CALLS):
                for layer in self.model.dit.layers:
                    x = layer(x)
            y = torch.ones(self.enc_shape, device=self.device) * x.mean()
            for _ in range(ENC_CALLS):
                for layer in self.model.enc.layers:
                    y = layer(y)
            return torch.cat([x.flatten(), y.flatten()]).float().cpu()

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.allclose(reference, candidate, atol=1e-5)))


def _twin(device: str = "cpu", enc_shape: tuple[int, ...] | None = None) -> TwinWorkload:
    w = TwinWorkload(WorkloadSpec(repo_id="toy/twin", modality="tts", device=device))
    if enc_shape is not None:
        w.enc_shape = enc_shape
    w.load()
    return w


def _capture(w: TwinWorkload, path: Path, **kw: Any) -> dict[str, Any]:
    return capture_module(
        w, w.make_inputs(), "Block", path, qualname_regex=REGEX, phase="prefill", **kw
    )


def test_a_case_stands_for_the_calls_of_every_instance_with_its_input(tmp_path):
    w = _twin()
    info = _capture(w, tmp_path / "same" / "c.pt")
    assert info["qualname"] == "model.dit.layers.0"  # the busiest instance
    assert info["method_instances"] == {"forward": 4}
    (case,) = info["cases"]
    # the even split counted 9 calls × 4 instances = 36; the run makes 2 × 9 + 2 × 1 = 20
    assert case["count"] == DIT_CALLS and case["target_calls"] == 20
    assert info["instance_groups"] == {
        DIT: {"instances": 2, "calls": 18, "covered": 18},
        ENC: {"instances": 2, "calls": 2, "covered": 2},
    }
    cap = load_capture(tmp_path / "same" / "c.pt")
    assert weights.weights(cap) == weights.weights(info) == [20.0]
    assert weights.summary(cap, 20.0) == {"basis": "instance groups", "calls": 20.0, "uncovered": 0}
    assert weights.coverage(cap) == 1.0

    # whichever instance is captured, its case stands for the same 20 calls (the even split:
    # 1 call × 4 instances = 4 when a LocEnc-like layer is captured)
    info = _capture(w, tmp_path / "enc" / "c.pt", qualname="model.enc.layers.0")
    assert [(c["count"], c["target_calls"]) for c in info["cases"]] == [(ENC_CALLS, 20)]

    # VoxCPM2: the LocEnc-like layers run another shape, no case has it, so they are not
    # counted (the even split: 36 calls, 2x the 18 its case stands for)
    other = _twin(enc_shape=(1, 5, D))
    info = _capture(other, tmp_path / "other" / "c.pt")
    (case,) = info["cases"]
    assert (case["count"], case["target_calls"]) == (DIT_CALLS, 18)
    assert info["instance_groups"] == {
        DIT: {"instances": 2, "calls": 18, "covered": 18},
        ENC: {"instances": 2, "calls": 2, "covered": 0},
    }
    cap = load_capture(tmp_path / "other" / "c.pt")
    assert weights.summary(cap, 18.0) == {"basis": "instance groups", "calls": 18.0, "uncovered": 2}
    assert weights.coverage(cap) == pytest.approx(0.9)
    assert weights.group_calls(cap) == {DIT: 18.0, ENC: 0.0}


def test_captures_without_instance_counts_keep_the_even_split(tmp_path):
    """Older captures (and synthetic ones) have no ``target_calls``: the captured instance's
    calls × the instances calling the case's entrypoint, and the result says so."""
    path = tmp_path / "c.pt"
    capture_calls(Block(), [((torch.randn(2, 3, D),), {}, DIT_CALLS)], path, instances=4)
    cap = load_capture(path)
    assert weights.weights(cap) == [36.0]
    assert weights.basis(cap) == weights.EVEN_SPLIT
    assert weights.summary(cap, 36.0) == {"basis": "even split", "calls": 36.0}
    assert weights.coverage(cap) is None and weights.group_calls(cap) == {}
    # a spec.json written before #119, and an evaluation's case report
    old = {"cases": [{"method": "forward", "count": 9}], "method_instances": {"forward": 4}}
    assert weights.weights(old) == [36.0] and weights.basis(old) == weights.EVEN_SPLIT
    users = {"forward": 4, "forward_step": 2}
    assert weights.case_weight({"calls_per_run": 9}, users) == 36.0
    assert weights.case_weight({"method": "forward_step", "calls_per_run": 3}, users) == 6.0
    assert weights.case_weight({"calls_per_run": 9, "target_calls": 18.0}, users) == 18.0


def _specs(capture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``dit_layer`` (the blocks of both groups) and a kernel of the ``enc`` blocks' Linear."""
    return {
        "dit_layer": {"module_class": "Block", "qualname_regex": REGEX, "capture": capture},
        "enc_lin": {"module_class": "Linear", "qualname_regex": r"enc\."},
    }


PROFILE = {
    "classes": [
        {"cls": "Block", "instances": 4, "groups": {DIT: 2, ENC: 2}},
        {"cls": "Linear", "instances": 4, "groups": {f"{DIT}.lin": 2, f"{ENC}.lin": 2}},
    ]
}


def test_the_projection_puts_a_saving_where_its_cases_stand_for_calls(tmp_path):
    """The tree spreads ``dit_layer``'s saving over its groups by the calls its cases stand
    for: none in the ``enc`` blocks, whose Linear kernel then counts in full."""
    info = _capture(_twin(enc_shape=(1, 5, D)), tmp_path / "c.pt")
    tree = projection.build(_specs(info), [PROFILE])
    shares = {(g.target, g.pattern): g.share for g in tree.groups}
    assert shares[("dit_layer", DIT)] == 1.0 and shares[("dit_layer", ENC)] == 0.0
    proj = projection.project(tree, {"dit_layer": 90.0, "enc_lin": 10.0}, 1000.0)
    assert proj.counted == pytest.approx({"dit_layer": 90.0, "enc_lin": 10.0})

    # the even split put half of it in the enc blocks, where it hid the Linear kernel
    old = {"cases": [{"method": "forward", "count": 9}], "method_instances": {"forward": 4}}
    tree = projection.build(_specs(old), [PROFILE])
    assert {g.share for g in tree.groups if g.target == "dit_layer"} == {0.5}
    proj = projection.project(tree, {"dit_layer": 90.0, "enc_lin": 10.0}, 1000.0)
    assert proj.counted == pytest.approx({"dit_layer": 90.0}) and proj.left_out == ["enc_lin"]


def test_the_first_audio_window_counts_the_calls_a_case_stands_for(tmp_path):
    """metric=ttfa: the window's calls of the target's instances × the share of the run's
    calls a case stands for, ÷ the calls the estimate covers."""
    info = _capture(_twin(enc_shape=(1, 5, D)), tmp_path / "c.pt")
    spec = {"module_class": "Block", "phase": "prefill", "capture": info}
    window = {"classes": [{"cls": "Block", "calls": 10, "phases": {"prefill": {"calls": 10}}}]}
    assert projection.window(spec, [window]) == pytest.approx(10 * 0.9 / 18)
    whole = {"classes": [{"cls": "Block", "calls": 20, "phases": {"prefill": {"calls": 20}}}]}
    assert projection.window(spec, [whole]) == pytest.approx(1.0)  # the window: the whole run
    old = {"cases": [{"method": "forward", "count": 9}], "method_instances": {"forward": 4}}
    assert projection.window({**spec, "capture": old}, [window]) == pytest.approx(10 / 36)


def test_a_region_arm_weighs_its_timed_cases_as_the_evaluator(tmp_path):
    run = RunDir(tmp_path / "run")
    run.root.mkdir(parents=True)
    write_json(run.baseline_json, {"median_ms": 100.0})
    spec = {"kind": "region", "module_class": "Region_r", "parent_class": "Block"}
    spec["capture"] = {"method_instances": {"forward": 4}}
    case = {"method": "forward", "calls_per_run": 9, "ref_ms": 0.5}
    append_jsonl(run.results_file("r"), {"cases": [case]})
    assert scheduler._region_ref_ms(run, "r", spec, []) == pytest.approx(0.5 * 36)  # even split
    append_jsonl(run.results_file("r"), {"cases": [{**case, "target_calls": 18.0}]})
    assert scheduler._region_ref_ms(run, "r", spec, []) == pytest.approx(0.5 * 18)


def test_report_names_the_calls_behind_each_estimate(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    run = make_run(tmp_path)
    first, second = run.target_ids()[:2]
    rec = {
        "snapshot": "001_cuda_v1.py",
        "speedup": 2.0,
        "est_saved_ms_per_run": 9.0,
        "est_saved_calls": {"basis": "instance groups", "calls": 18.0, "uncovered": 2.0},
        "cases": [
            {"signature": "a0[2, 3, 8]", "calls_per_run": 9, "target_calls": 18.0, "ref_ms": 1.0}
        ],
    }
    old = {**rec, "est_saved_calls": None, "cases": [{**rec["cases"][0], "target_calls": None}]}
    best = {first: rec, second: old}
    monkeypatch.setattr(report, "best_for_target", lambda run, t, keeper=None: best.get(t))
    spec = read_json(run.target(second) / "spec.json", {})
    spec["capture"] = {"method_instances": {"forward": 4}}
    write_json(run.target(second) / "spec.json", spec)
    text = report.write_report(run).read_text()
    assert "`a0[2, 3, 8]` ×9 (×18 over the target's instances)" in text
    assert f"`{first}` 18 calls (not 2 with a primary input no case has)" in text
    assert f"`{second}` even split (the captured instance's calls ×" in text


# ------------------------------------------------------------------ GPU: the evaluator

FAST = """
import copy

import torch


def build(reference):
    class Fast(type(reference)):
        def forward(self, x):  # x + x W^T + b = x (I + W)^T + b: one GEMM, no add
            flat = x.reshape(-1, x.shape[-1])
            return torch.addmm(self.lin.bias, flat, self.fused_t).reshape(x.shape)

    new = copy.copy(reference)
    new.__class__ = Fast
    weight = reference.lin.weight
    eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
    new.fused_t = (weight + eye).t().contiguous()
    return new
"""


@pytest.mark.gpu
def test_evaluator_weighs_each_case_by_the_calls_it_stands_for_on_gpu(tmp_path):
    from kernel_agent.kernels.evaluate import evaluate

    _capture(_twin("cuda", enc_shape=(1, 5, D)), tmp_path / "c.pt")
    candidate = tmp_path / "fast.py"
    candidate.write_text(FAST)
    result = evaluate(tmp_path / "c.pt", candidate)
    assert result["status"] == "ok", result
    (case,) = result["cases"]
    assert (case["calls_per_run"], case["target_calls"]) == (DIT_CALLS, 18.0)
    assert result["est_saved_calls"] == {"basis": "instance groups", "calls": 18.0, "uncovered": 2}
    gain = case["ref_ms"] - case["new_ms"]
    assert result["est_saved_ms_per_run"] == pytest.approx(18 * gain, abs=2e-3)
