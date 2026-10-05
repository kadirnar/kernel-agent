"""Workload statistics at capture, phase-specific targets (capture, patching, export, prompts)."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from test_methods import LAYERS, MAX_LEN, PREFIX, STEPS, ToyWorkload, _candidate
from test_voxcpm import _voxcpm2_cached
from torch import nn

from kernel_agent import orchestrator, worker
from kernel_agent.agent import prompts
from kernel_agent.hub import Modality
from kernel_agent.integrate.export import export_optimized
from kernel_agent.integrate.patcher import KernelPatch, apply_kernels
from kernel_agent.phases import call_phase, route
from kernel_agent.profiling.capture import capture_module, load_capture
from kernel_agent.profiling.profiler import ModuleTimer, summarize
from kernel_agent.profiling.workload_stats import WorkloadStats, render
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec
from kernel_agent.workspace import RunDir, read_json, write_json

D = 8

# ------------------------------------------------------------------ phases


def test_call_phase():
    assert call_phase("forward_step", (torch.ones(1, D),), {}) == "decode"
    assert call_phase("generate_step", (), {}) == "decode"
    assert call_phase("forward", (torch.ones(1, 1, D),), {}) == "decode"
    assert call_phase("forward", (), {"hidden_states": torch.ones(2, 1, D)}) == "decode"
    assert call_phase("forward", (torch.ones(2, 11, D),), {}) == "prefill"
    assert call_phase("forward", (3, torch.ones(1, D)), {}) == "prefill"  # 2-D: not a step
    assert call_phase("decode", (torch.ones(1, 64, 100),), {}) == "prefill"  # a VAE decode
    assert call_phase("forward", (5,), {}) == "prefill"


# ------------------------------------------------------------------ statistics


class Probe(nn.Module):
    """Arguments named like attention layers': masks, positions, caches, biases, flags."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(D, D)
        self.out = nn.Linear(D, D, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        bias: torch.Tensor | None = None,
        rope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden_states

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_id: Any,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden_states


def _observe(stats: WorkloadStats, m: nn.Module, method: str, *args: Any, **kwargs: Any) -> None:
    stats.observe(m, method, args, kwargs, call_phase(method, args, kwargs))


def _arg(profile: dict[str, Any], method: str, name: str) -> dict[str, Any]:
    return next(a for a in profile["args"] if a["method"] == method and a["name"] == name)


def test_mask_kinds_and_flags():
    m, stats = Probe(), WorkloadStats()
    x = torch.randn(1, 4, D)
    tril = torch.ones(4, 4, dtype=torch.bool).tril()
    additive = torch.zeros(4, 4).masked_fill(~tril, float("-inf"))
    static_prefill = torch.arange(8) <= torch.arange(4)[:, None]  # [4, 8], rows 1..4
    padding = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])  # int64 [batch, slots]
    masks = [
        None,
        torch.ones(1, 1, 4, 8, dtype=torch.bool),  # all-ones
        tril[None, None],  # causal
        additive[None, None],  # causal (additive float)
        static_prefill[None, None],  # causal over the first rows of a static cache
        padding,  # prefix
        torch.tensor([[True, False, True, True]]),  # other
        torch.full((1, 1, 4, 4), -0.5),  # additive bias: other
    ]
    for i, mask in enumerate(masks):
        _observe(stats, m, "forward", x, attention_mask=mask, is_causal=i % 3 == 0)
    p = stats.finalize(m)
    mask = _arg(p, "forward", "attention_mask")
    assert mask["none"] == 1
    assert mask["masks"] == {"all-ones": 1, "causal": 3, "prefix": 1, "other": 2}
    assert _arg(p, "forward", "is_causal")["values"] == {"False": 5, "True": 3}
    assert any("`forward(attention_mask)` masks: causal 38 %" in f for f in p["facts"])
    assert any(f.startswith("`forward(is_causal)`: False ×5, True ×3") for f in p["facts"])
    assert p["groups"][0]["phase"] == "prefill" and p["groups"][0]["calls"] == len(masks)


def test_positions_valid_length_and_layouts():
    m, stats = Probe(), WorkloadStats()
    cache = torch.zeros(2, 1, 2, 64, 4)  # [K/V, batch, heads, slots, dim]
    for pos in range(5, 9):  # tensor positions: valid length pos + 1
        _observe(
            stats, m, "forward_step", torch.randn(1, D), torch.tensor([pos]), (cache[0], cache[1])
        )
    for pos in (20, 30):  # Python int positions
        _observe(stats, m, "forward_step", torch.randn(1, D), pos, (cache[0], cache[1]))
    decode_mask = (torch.arange(64) < 12)[None, None, None]  # no position: the mask's prefix
    _observe(stats, m, "forward_step", torch.randn(1, D), None, (cache[0], cache[1]), decode_mask)
    # forward: one non-contiguous input, a zero and a non-zero bias, an fp32 rope with bf16 x
    x = torch.randn(1, D, 4, dtype=torch.bfloat16)
    _observe(stats, m, "forward", x.transpose(1, 2), bias=torch.zeros(D), rope=torch.ones(4))
    _observe(stats, m, "forward", x.transpose(1, 2).contiguous(), bias=torch.ones(D))
    with torch.no_grad():
        m.proj.bias.zero_()
    p = stats.finalize(m, **{"class": "Probe", "qualname": "probe", "instances": 1})

    pos = _arg(p, "forward_step", "position_id")
    assert pos["ints"] == {"min": 5, "max": 30, "distinct": 6}
    assert pos["none"] == 1
    kv = _arg(p, "forward_step", "kv_cache")["cache"]
    assert kv["slots"] == [64]
    assert (kv["valid_min"], kv["valid_max"], kv["valid_from"]) == (6, 31, "position_id")
    cache_fact = next(f for f in p["facts"] if "static cache" in f)
    assert "static cache of 64 slots, valid length 6..31 (`position_id` 5..30)" in cache_fact
    assert "48 %" in cache_fact  # 31 / 64

    hidden = _arg(p, "forward", "hidden_states")
    assert hidden["contiguous"] == 0.5 and hidden["layout"] == "[1, 4, 8] strides (32, 1, 4)"
    assert _arg(p, "forward", "bias")["zero_frac"] == 0.5
    assert p["params"] == {"biases": 1, "zero_biases": ["proj.bias"]}
    facts = "\n".join(p["facts"])
    assert "`forward(hidden_states)`: non-contiguous in 50 %" in facts
    assert "`proj.bias` are all zeros" in facts
    assert "`forward` mixes float dtypes (bfloat16: hidden_states; float32: bias, rope)" in facts
    methods = p["methods"]
    assert methods["forward_step"]["phases"] == {"decode": 7} and methods["forward"]["calls"] == 2
    text = render(p)
    assert text.startswith("# Workload profile: `Probe`") and "| `forward_step` | decode |" in text


def test_mask_prefix_gives_valid_length():
    m, stats = Probe(), WorkloadStats()
    cache = torch.zeros(1, 2, 32, 4)
    for valid in (3, 9):
        mask = (torch.arange(32) < valid)[None, None, None]
        _observe(stats, m, "forward_step", torch.randn(1, D), None, (cache, cache), mask)
    kv = _arg(stats.finalize(), "forward_step", "kv_cache")["cache"]
    assert (kv["valid_min"], kv["valid_max"], kv["valid_from"]) == (3, 9, "mask")


# ------------------------------------------------------------------ capture


def test_capture_writes_workload_profile(tmp_path):
    w = ToyWorkload(WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cpu"))
    w.load()
    info = capture_module(w, w.make_inputs(), "Attention", tmp_path / "cap.pt")
    profile = json.loads((tmp_path / "workload_profile.json").read_text())
    # every call of both instances, not only the captured cases of layer 0
    assert profile["calls"] == LAYERS * (STEPS + 1) and profile["instances_called"] == LAYERS
    assert profile["methods"]["forward_step"] == {
        "calls": LAYERS * STEPS,
        "share": round(STEPS / (STEPS + 1), 4),
        "instances": LAYERS,
        "phases": {"decode": LAYERS * STEPS},
    }
    kv = _arg(profile, "forward_step", "kv_cache")["cache"]
    assert kv["slots"] == [MAX_LEN]
    assert (kv["valid_min"], kv["valid_max"]) == (PREFIX + 1, PREFIX + STEPS)
    assert info["workload"]["facts"] == profile["facts"]
    assert "valid length 6..11" in (tmp_path / "workload_profile.md").read_text()


class Mixed(nn.Module):
    """One class in three roles: encoder (1 prefill call), DiT (2 prefill calls per patch),
    LM layers (prefill once, then one decode step per patch)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        return self.lin(x)

    def forward_step(self, x: torch.Tensor, position_id: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * 2


class MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = Mixed()
        self.dit = Mixed()
        self.lm = nn.ModuleList(Mixed() for _ in range(2))


class MixedWorkload(Workload):
    modality = Modality.TTS

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = MixedModel().eval()

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> torch.Tensor:
        return torch.randn(1, 5, D)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        m = self.model
        with torch.inference_mode():
            h = m.enc(inputs)
            for layer in m.lm:
                h = layer(h)
            out = [h[:, -1]]
            for step in range(3):
                d = m.dit(torch.cat([h[:, :2], h[:, :2]]), is_causal=False)  # CFG batch 2
                d = m.dit(d, is_causal=False)
                x = d[:1, -1]
                for layer in m.lm:
                    x = layer.forward_step(x, torch.tensor([5 + step]))
                out.append(x)
            return torch.stack(out, 1)

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.allclose(reference, candidate, atol=1e-5)))


@pytest.fixture
def mixed() -> MixedWorkload:
    w = MixedWorkload(WorkloadSpec(repo_id="toy/mixed", modality="tts", device="cpu"))
    w.load()
    return w


def test_phase_capture_picks_the_busiest_instance(mixed, tmp_path):
    prefill = capture_module(
        mixed, mixed.make_inputs(), "Mixed", tmp_path / "p" / "c.pt", phase="prefill"
    )
    assert prefill["qualname"] == "model.dit" and prefill["phase"] == "prefill"
    assert prefill["methods"] == {"forward": 6}
    assert prefill["method_instances"] == {"forward": 4}  # forward_step calls are not counted
    assert {c["method"] for c in prefill["cases"]} == {"forward"}
    cap = load_capture(tmp_path / "p" / "c.pt")
    assert cap["phase"] == "prefill" and cap["instances"] == 4

    decode = capture_module(
        mixed, mixed.make_inputs(), "Mixed", tmp_path / "d" / "c.pt", phase="decode"
    )
    assert decode["qualname"] == "model.lm.0"  # first of the tied LM layers
    assert decode["methods"] == {"forward_step": 3}
    assert decode["method_instances"] == {"forward_step": 2}
    # the workload profile still covers both phases
    profile = json.loads((tmp_path / "d" / "workload_profile.json").read_text())
    assert profile["methods"]["forward"]["phases"] == {"prefill": 9}
    assert _arg(profile, "forward", "is_causal")["values"] == {"False": 6}  # passed by the DiT
    assert any("`forward_step` 6 calls (40 %, decode, 2 instances)" in f for f in profile["facts"])

    lm = capture_module(
        mixed, mixed.make_inputs(), "Mixed", tmp_path / "r" / "c.pt", qualname_regex=r"\.lm\."
    )
    assert lm["qualname"] == "model.lm.0"
    assert lm["method_instances"] == {"forward": 2, "forward_step": 2}
    profile = json.loads((tmp_path / "r" / "workload_profile.json").read_text())
    assert profile["instances"] == 2 and profile["calls"] == 2 + 6
    with pytest.raises(RuntimeError, match="no decode calls"):
        capture_module(
            mixed,
            mixed.make_inputs(),
            "Mixed",
            tmp_path / "x.pt",
            qualname="model.enc",
            phase="decode",
        )


def test_profile_summary_shows_phase_split(mixed):
    with torch.inference_mode(), ModuleTimer(mixed.roots(), cuda=False) as timer:
        mixed.run(mixed.make_inputs())
    stat = next(s for s in timer.class_stats() if s.cls == "Mixed")
    assert stat.phases["decode"]["calls"] == 6 and stat.phases["prefill"]["calls"] == 9
    assert stat.phases["decode"]["groups"] == {"model.lm.*": 2}
    assert stat.phases["decode"]["top_signature"].startswith("forward_step: a0[1, 8]")
    kernel_view = {
        "gpu_busy_ms": 1.0,
        "gpu_busy_fraction": 0.5,
        "kernel_launches": 1,
        "avg_kernel_us": 1.0,
        "kernels": [],
        "aten_ops": [],
    }
    mixed_stat = asdict(stat)
    mixed_stat["phases"]["decode"]["inclusive_ms"] = 3.0
    mixed_stat["phases"]["prefill"]["inclusive_ms"] = 7.0
    linear = copy.deepcopy(mixed_stat)  # a class whose decode time is negligible
    linear["cls"] = "Linear"
    linear["phases"]["decode"]["inclusive_ms"] = 0.01
    profile = {"module_calls": 1, "classes": [mixed_stat, linear], "kernel_view": kernel_view}
    text = summarize(profile, 2.0)
    assert "## Phase split" in text and "| `Mixed` | decode | 6 | 3.00 | 30% |" in text
    assert "`model.lm.*` (2)" in text and "| `Linear` | decode" not in text


# ------------------------------------------------------------------ patching


LOGGING = """
import copy

import torch


def build(reference):
    base = type(reference)

    class Fast(base):
        def forward(self, x, *args, **kwargs):
            self.log.append("{tag}.forward")
            return super().forward(x, *args, **kwargs)

        def forward_step(self, x, *args, **kwargs):
            self.log.append("{tag}.forward_step")
            return super().forward_step(x, *args, **kwargs)

    new = copy.copy(reference)
    new.__class__ = Fast
    new.log = reference.__dict__.setdefault("log", [])
    return new
"""

FALLBACK = """
import torch


class Wrapper(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.ref = ref

    def forward(self, x, *args, **kwargs):
        return self.ref(x, *args, **kwargs)  # falls back to the reference instance

    def forward_step(self, x, *args, **kwargs):
        return self.ref.forward_step(x, *args, **kwargs)


def build(reference):
    return Wrapper(reference)
"""


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    return path


def test_phase_patches_route_and_combine(mixed, tmp_path):
    inputs = mixed.make_inputs()
    reference = mixed.run(inputs)
    decode = _write(tmp_path, "dec", LOGGING.format(tag="dec"))
    prefill = _write(tmp_path, "pre", LOGGING.format(tag="pre"))
    patches = [
        KernelPatch("dec", "Mixed", decode, None, ["forward_step"], phase="decode"),
        KernelPatch("pre", "Mixed", prefill, r"model\.(dit|lm)", ["forward"], phase="prefill"),
    ]
    report = apply_kernels(mixed.roots(), patches)
    assert report.replaced == {"dec": 4, "pre": 3}
    m = mixed.model
    assert all(type(x) is Mixed for x in (m.enc, m.dit, *m.lm))  # routed, not replaced
    assert mixed.compare(reference, mixed.run(inputs)).passed
    assert m.dit.log == ["pre.forward"] * 6
    assert m.lm[0].log == ["pre.forward"] + ["dec.forward_step"] * 3
    assert m.enc.log == []  # only decode was patched there and it never decodes
    # A replacement that falls back to the reference instance does not loop back into itself.
    wrapper = _write(tmp_path, "wrap", FALLBACK)
    apply_kernels(
        mixed.roots(),
        [KernelPatch("w", "Mixed", wrapper, None, ["forward_step", "forward"], phase="decode")],
    )
    assert mixed.compare(reference, mixed.run(inputs)).passed


def test_route_strips_copied_routers():
    a = Mixed()
    first = Mixed()
    route(a, first, "decode", ["forward_step"])
    assert "forward_step" in vars(a)
    second = copy.copy(a)  # build() of the next target copies the router too
    route(a, second, "prefill", ["forward"])
    assert "forward_step" not in vars(second)
    x = torch.randn(1, 3, D)
    assert torch.equal(a(x), a.lin(x))
    with pytest.raises(ValueError):
        route(a, second, "all", ["forward"])


def test_export_routes_phase_kernels(tmp_path):
    w = ToyWorkload(WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cpu"))
    w.load()
    inputs = w.make_inputs()
    reference = w.run(inputs)
    run = RunDir.create(tmp_path, "toy/decoder")
    write_json(run.run_json, {"card": {"repo_id": "toy/decoder"}})
    write_json(
        run.profile_dir / "profile.json", {"classes": [{"root": "model", "cls": "Attention"}]}
    )
    write_json(
        run.target("attn") / "spec.json",
        {
            "module_class": "Attention",
            "phase": "decode",
            "qualname_regex": r"^model\.layers\.1\.",
            "capture": {"method_instances": {"forward_step": 2}},
        },
    )
    candidate = _candidate(tmp_path, "scaled", scale="1.5")
    out = export_optimized(run, [("kernel", f"attn={candidate}", 0.0)])
    manifest = read_json(out / "manifest.json")
    assert manifest["roots"] == ["model"] and (out / "phases.py").exists()
    entry = manifest["kernels"][0]
    assert entry["phase"] == "decode" and entry["routed"] == ["forward_step"]
    spec = importlib.util.spec_from_file_location("ka_test_apply", out / "apply.py")
    assert spec is not None and spec.loader is not None
    apply = importlib.util.module_from_spec(spec)
    sys.modules["ka_test_apply"] = apply
    spec.loader.exec_module(apply)
    assert apply.apply_kernels(w.model) == {"attn": 1}
    layers = w.model.layers
    assert "forward_step" in vars(layers[1].attn) and "forward_step" not in vars(layers[0].attn)
    assert type(layers[1].attn).__name__ == "Attention"
    assert not w.compare(reference, w.run(inputs)).passed  # the scaled kernel runs for decode


# ------------------------------------------------------------------ plan -> spec -> worker


def test_plan_schema_and_scope():
    props = prompts.PLAN_SCHEMA["properties"]["targets"]["items"]["properties"]
    assert props["phase"]["enum"] == ["all", "prefill", "decode"]
    assert props["qualname_regex"]["type"] == ["string", "null"]
    t = {"id": "a", "module_class": "C", "phase": "all", "qualname_regex": None}
    assert orchestrator._scope(t) is None and "phase" not in t and "qualname_regex" not in t
    t = {"id": "a", "module_class": "C", "phase": "decode", "qualname_regex": r"lm\."}
    assert orchestrator._scope(t) is None
    assert (t["phase"], t["qualname_regex"]) == ("decode", r"lm\.")
    assert "unknown phase" in (orchestrator._scope({"phase": "train"}) or "")
    assert "bad qualname_regex" in (orchestrator._scope({"qualname_regex": "("}) or "")


def test_worker_plumbs_phase_and_regex(mixed, tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "toy/mixed")
    spec = {
        "id": "dit_attn",
        "module_class": "Mixed",
        "phase": "prefill",
        "qualname_regex": r"\.dit$",
        "backends": ["triton"],
    }
    write_json(run.target("dit_attn") / "spec.json", spec)
    monkeypatch.setattr(worker, "_workload", lambda run: mixed)
    ns = argparse.Namespace(target="dit_attn", max_cases=4)
    info = worker.cmd_capture(run, ns)
    assert info["qualname"] == "model.dit" and info["phase"] == "prefill"
    saved = read_json(run.target("dit_attn") / "spec.json")
    assert saved["phase"] == "prefill" and saved["capture"]["method_instances"] == {"forward": 1}
    assert (run.target("dit_attn") / "workload_profile.md").exists()
    assert (run.target("dit_attn") / "reference_source.py").exists()
    patch = worker._kernel_patches(run, [f"dit_attn={tmp_path / 'k.py'}"])[0]
    assert (patch.phase, patch.qualname_regex, patch.methods) == ("prefill", r"\.dit$", ["forward"])


def test_engineer_prompt_shows_scope_and_workload_facts():
    target = {
        "id": "dec",
        "module_class": "MiniCPMAttention",
        "why": "w",
        "approach": "a",
        "phase": "decode",
        "qualname_regex": r"base_lm\.",
    }
    facts = [f"fact {i}" for i in range(9)]
    info = {
        "qualname": "model.base_lm.layers.0.self_attn",
        "methods": {"forward_step": 60},
        "method_instances": {"forward_step": 28},
        "cases": [
            {"method": "forward_step", "signature": "forward_step: a0[1, 2048]", "count": 60}
        ],
        "workload": {"calls": 9, "facts": facts},
    }
    text = prompts.engineer_prompt(target, info, ["triton"], "py", "tc", 4, None)
    assert "# Workload" in text and "* fact 0" in text and "* fact 5" in text
    assert "fact 6" not in text and "workload_profile.md" in text
    assert "* phase: `decode` only" in text and r"matches `base_lm\.`" in text
    plain = prompts.engineer_prompt(
        {"id": "x", "module_class": "C"}, {"cases": []}, ["triton"], "py", "tc", 4, None
    )
    assert "# Workload\n" not in plain and "* phase:" not in plain
    plan = prompts.planner_prompt(
        {"repo_id": "o/m", "modality": "tts"}, {}, "", ["triton"], 3, "p", "t"
    )
    assert "Phase-specific targets" in plan and r"`feat_decoder\.`" in plan


def test_workload_stats_skip_device_work_without_tensors():
    stats = WorkloadStats()
    m = Probe()
    _observe(stats, m, "forward", "text", attention_mask=None)
    p = stats.finalize(m)
    assert _arg(p, "forward", "hidden_states")["values"] == {"'text'": 1}
    assert math.isclose(p["groups"][0]["share"], 1.0)


# ------------------------------------------------------------------ GPU: VoxCPM2

SAME_ATTENTION = """
import copy


def build(reference):
    class Same(type(reference)):  # reference math; only the routing is under test
        pass

    new = copy.copy(reference)
    new.__class__ = Same
    return new
"""


@pytest.mark.gpu
@pytest.mark.skipif(not _voxcpm2_cached(), reason="needs voxcpm + openbmb/VoxCPM2 in the HF cache")
def test_voxcpm2_workload_profile_and_decode_routing(tmp_path):
    from kernel_agent.workloads import create_workload

    spec = WorkloadSpec("openbmb/VoxCPM2", "tts", family="voxcpm", options={"patches": 8})
    wl = create_workload(spec)
    wl.load()
    inputs = wl.make_inputs()
    with torch.inference_mode():
        reference = wl.run(inputs)
    info = capture_module(wl, inputs, "MiniCPMAttention", tmp_path / "cap.pt", phase="decode")
    assert set(info["methods"]) == {"forward_step"} and "base_lm." in info["qualname"]
    profile = json.loads((tmp_path / "workload_profile.json").read_text())
    methods = profile["methods"]
    assert methods["forward_step"]["phases"] == {"decode": methods["forward_step"]["calls"]}
    assert methods["forward"]["phases"] == {"prefill": methods["forward"]["calls"]}
    assert methods["forward"]["calls"] > methods["forward_step"]["calls"]  # DiT + LocEnc
    kv = _arg(profile, "forward_step", "kv_cache")["cache"]
    assert kv["slots"] == [8192] and kv["valid_from"] == "position_id"
    assert 1 < kv["valid_min"] < kv["valid_max"] < 8192 // 20
    assert any("static cache of 8192 slots" in f for f in profile["facts"])

    # A decode-phase patch keeps every instance and routes only forward_step calls.
    candidate = _write(tmp_path, "same", SAME_ATTENTION)
    patch = KernelPatch("dec", "MiniCPMAttention", candidate, None, ["forward_step"], "decode")
    report = apply_kernels(wl.roots(), [patch])
    assert report.replaced["dec"] == profile["instances"]
    attention = [m for m in wl.model.modules() if type(m).__name__ == "MiniCPMAttention"]
    assert len(attention) == profile["instances"]  # still the original instances
    assert all("forward_step" in vars(m) for m in attention)
    with torch.inference_mode():
        out = wl.run(inputs)
    assert torch.equal(out["latents"], reference["latents"])
