"""The library scout (issue #227): op families from what a reference calls, adapters probed
and skipped with reasons, candidate templates, the scout's ledger rows and its library bar,
the export's ``requirements.txt``.

CPU and deterministic: toy modules on CPU tensors (``libscout_toy.py``), fake library
modules and ``importlib.util.find_spec``, fake GPUs (``toolchain.GPUInfo``), a fake probe and
sweep for the run step. The ``gpu`` tests run the real probe and sweep on toy captures.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import operator
import os
import sys
import types
from pathlib import Path
from typing import Any

import libscout_toy
import pytest
import torch
from libscout_toy import (
    Block,
    Boxed,
    Branchy,
    Core,
    Counted,
    Feed,
    Fused,
    Head,
    Mixer,
    Plain,
    Promoted,
    Residual,
    Rotary,
    Scale,
)
from test_ceilings import PEAKS
from test_ceilings import _profile as ceilings_profile
from test_recheck import kernel, make, simulated  # noqa: F401 (fixture)
from test_truth import sealed_run

from kernel_agent import budget, critic, gpulock, ledger, scheduler, toolchain, truth
from kernel_agent.integrate.export import export_optimized
from kernel_agent.kernels import sweep
from kernel_agent.libscout import adapters, detect, fx_rewrites, probe, registry, template
from kernel_agent.libscout import scout as libscout
from kernel_agent.profiling import ceilings
from kernel_agent.report import write_report
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

TESTS = Path(__file__).parent


def found_in(module: torch.nn.Module, *args: Any) -> dict[str, Any]:
    return detect.summary(detect.families(detect.trace(module, args, module=module)))


def rope(seq: int = 5, dim: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randn(seq, dim), torch.randn(seq, dim)


# ------------------------------------------------------------------ detection


def test_detects_sdpa_rotary_norm_and_linears_of_an_attention_block():
    torch.manual_seed(0)
    found = found_in(Mixer(), torch.randn(2, 5, 32), *rope())
    assert list(found) == ["sdpa", "rms_norm", "linear", "rotary"]
    (site,) = found["sdpa"]["sites"]
    assert site == {
        "q": [2, 4, 5, 8],
        "k": [2, 1, 5, 8],
        "dtype": "float32",
        "batch": 2,
        "heads": 4,
        "kv_heads": 1,
        "seq_q": 5,
        "seq_kv": 5,
        "head_dim": 8,
        "mask": False,
        "causal": True,
        "gqa": True,
    }
    (norm,) = found["rms_norm"]["sites"]
    assert norm["form"] == "manual" and norm["hidden"] == 8 and norm["eps"] == 1e-6
    assert norm["weight"] == "n.weight" and norm["rows"] == 40
    assert [s["n"] for s in found["linear"]["sites"]] == [32, 8, 8, 32]
    assert found["rotary"]["sites"][0]["head_dim"] == 8


def test_detects_rms_norm_written_out_and_as_f_rms_norm():
    x = torch.randn(3, 4, 16, dtype=torch.bfloat16)
    manual = found_in(Scale(16), x)["rms_norm"]["sites"][0]
    functional = found_in(Plain(16).bfloat16(), x)["rms_norm"]["sites"][0]
    # Scale's weight is fp32: weight * x.to(bf16) is fp32, as the module returns it
    assert manual == {
        "form": "manual",
        "hidden": 16,
        "eps": 1e-6,
        "weight": "weight",
        "dtype": "float32",
        "rows": 12,
    }
    assert functional == {
        "form": "functional",
        "hidden": 16,
        "eps": 1e-5,
        "weight": "weight",
        "dtype": "bfloat16",
        "rows": 12,
    }
    assert detect.describe(found_in(Scale(16), x)) == "rms_norm x1 (manual)"


def test_detects_a_gated_mlp():
    found = found_in(Feed(), torch.randn(2, 3, 16))
    assert list(found) == ["linear", "gated_mlp"]
    assert found["gated_mlp"]["sites"] == [
        {
            "act": "silu",
            "merged": False,
            "hidden": 16,
            "intermediate": 32,
            "m": 6,
            "dtype": "float32",
        }
    ]


def test_sampling_is_a_draw_not_a_topk():
    torch.manual_seed(0)
    x = torch.randn(2, 16)
    drawn = found_in(Head(), x)
    assert {"softmax", "linear", "sampling"} <= set(drawn)
    assert drawn["sampling"]["sites"][0]["op"] == "multinomial"
    ranked = detect.summary(detect.families(detect.trace(lambda t: Head()(t, draw=False), (x,))))
    assert "sampling" not in ranked and "softmax" in ranked


# ------------------------------------------------------------------ probes and decisions


def _fake_libraries(monkeypatch, installed: dict[str, Any], broken: tuple[str, ...] = ()):
    """``installed``: import name -> fake module (its package gets version 9.9); ``broken``:
    found by ``find_spec`` but failing to import (an incompatible wheel). Every other
    library of the registry is missing."""
    real = importlib.util.find_spec
    ours = {a.module.split(".")[0] for a in adapters.ADAPTERS} - {"torch"}

    def find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        top = name.split(".")[0]
        if top in installed or top in broken:
            return importlib.machinery.ModuleSpec(name, None)
        if top in ours:
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    for name, module in installed.items():
        monkeypatch.setitem(sys.modules, name, module)
        for sub, value in getattr(module, "_submodules", {}).items():
            monkeypatch.setitem(sys.modules, f"{name}.{sub}", value)
    for name in broken:  # its import fails, installed or not (None in sys.modules)
        monkeypatch.setitem(sys.modules, name, None)
    packages = {a.module.split(".")[0]: a.package for a in adapters.ADAPTERS}
    versions = {packages[m]: "9.9" for m in [*installed, *broken]}
    real_version = registry.package_version
    monkeypatch.setattr(registry, "package_version", lambda p: versions.get(p) or real_version(p))


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_missing_and_broken_libraries_are_skipped_with_reasons(monkeypatch):
    _fake_libraries(monkeypatch, {}, broken=("flash_attn",))
    probes = registry.availability(adapters.ADAPTERS)
    assert probes["torch-sdpa"].ok and probes["torch-sdpa"].version
    assert probes["quack-rmsnorm"].reason == (
        "quack-kernels is not installed (pip install 'kernel-agent[libs]')"
    )
    assert probes["flash-attn"].version == "9.9"
    assert str(probes["flash-attn"].reason).startswith("import failed: ModuleNotFoundError")
    found = found_in(Block(), torch.randn(2, 5, 32), *rope())
    decisions = {
        d.adapter.name: d
        for d in registry.applicable(
            adapters.ADAPTERS,
            found,
            capability=(12, 0),
            precision="exact",
            available=probes,
            backends={"cuda": True},
        )
    }
    assert not decisions["flash-attn"].run
    assert decisions["flashinfer-norm"].reason == probes["flashinfer-norm"].reason
    assert decisions["torch-scaled-mm"].reason == "serves fp8_w8a8 targets; this one is exact"
    assert decisions["liger-rope"].reason == probes["liger-rope"].reason  # not installed
    assert decisions["torch-sdpa"].run and decisions["torch-rms-norm"].run
    assert decisions["cublaslt"].reason == (
        "no call site it takes: linear in float32 (takes bfloat16, float16)"
    )
    assert "flashinfer-sampling" not in decisions  # the block draws nothing
    # fp32 attention: no half-precision backend config, so flash and cuDNN are not swept
    assert decisions["torch-sdpa"].configs == [{"BACKEND": "efficient"}, {"BACKEND": "math"}]
    no_nvcc = registry.applicable(
        [adapters.BY_NAME["cublaslt"]],
        found,
        capability=(12, 0),
        precision="exact",
        available=probes,
        backends={"cuda": False},
    )
    assert no_nvcc[0].reason == "builds a CUDA extension and no CUDA toolkit (nvcc) was found"


def test_a_library_is_used_only_with_what_its_template_needs(monkeypatch):
    quack = _module("quack", rmsnorm=lambda *a, **k: None)  # no softmax in this version
    _fake_libraries(monkeypatch, {"quack": quack})
    probes = registry.availability(adapters.ADAPTERS)
    assert probes["quack-rmsnorm"].ok and probes["quack-rmsnorm"].version == "9.9"
    assert probes["quack-softmax"].reason == "quack has no softmax"


GPUS = {
    "sm_75": toolchain.GPUInfo("Fake T4", (7, 5), 16.0, 40, 4.0),
    "sm_86": toolchain.GPUInfo("Fake A10", (8, 6), 24.0, 72, 6.0),
    "sm_120": toolchain.GPUInfo("Fake RTX 5070 Ti", (12, 0), 16.0, 70, 48.0),
}


def test_adapters_follow_the_gpu_architecture(monkeypatch):
    """Each adapter declares its architectures: skipped elsewhere with the reason."""
    functional = _module("liger_kernel.transformers.functional", liger_rms_norm=print)
    functional.liger_swiglu = print
    transformers = _module("liger_kernel.transformers", functional=functional)
    liger = _module("liger_kernel", transformers=transformers)
    liger._submodules = {  # type: ignore[attr-defined]
        "transformers": transformers,
        "transformers.functional": functional,
    }
    installed = {
        "flash_attn": _module("flash_attn", flash_attn_func=print),
        "flash_attn_interface": _module("flash_attn_interface", flash_attn_func=print),
        "quack": _module("quack", rmsnorm=print, softmax=print),
        "liger_kernel": liger,
    }
    _fake_libraries(monkeypatch, installed)
    probes = registry.availability(adapters.ADAPTERS)
    cos, sin = (t.bfloat16() for t in rope())
    attention = found_in(Mixer().to(torch.bfloat16), torch.randn(2, 5, 32).bfloat16(), cos, sin)
    attention["gated_mlp"] = found_in(Feed().bfloat16(), torch.randn(2, 3, 16).bfloat16())[
        "gated_mlp"
    ]

    def decide(arch: str, precision: str = "exact") -> dict[str, registry.Decision]:
        gpu = GPUS[arch]
        return {
            d.adapter.name: d
            for d in registry.applicable(
                adapters.ADAPTERS,
                attention,
                capability=gpu.capability,
                precision=precision,
                available=probes,
                backends={"cuda": True},
            )
        }

    turing, ampere, blackwell = decide("sm_75"), decide("sm_86"), decide("sm_120")
    assert turing["torch-sdpa"].configs == [{"BACKEND": "efficient"}, {"BACKEND": "math"}]
    assert [c["BACKEND"] for c in ampere["torch-sdpa"].configs] == [
        "flash",
        "efficient",
        "cudnn",
        "math",
    ]
    assert turing["cublaslt"].reason == (  # fp16 sites run there (#254); these are bf16
        "no call site it takes: linear in bfloat16 needs sm_80+ (bf16 tensor cores); this GPU "
        "is sm_75"
    )
    assert turing["flash-attn"].reason == (
        "needs sm_80+ (FlashAttention 2: Ampere and newer); this GPU is sm_75"
    )
    assert ampere["flash-attn"].run and blackwell["flash-attn"].run
    assert ampere["flash-attn-3"].reason == (
        "needs sm_90 (FlashAttention 3: Hopper (wgmma, TMA)); this GPU is sm_86"
    )
    assert not blackwell["flash-attn-3"].run
    assert ampere["quack-rmsnorm"].reason == (
        "needs sm_90,sm_10x,sm_12x (QuACK lists H100, B200 / B300 and RTX 50); this GPU is sm_86"
    )
    assert blackwell["quack-rmsnorm"].run
    assert blackwell["liger-swiglu"].run and blackwell["liger-rmsnorm"].run
    assert turing["liger-swiglu"].reason.startswith("needs sm_80+")
    # FP8 W8A8 only on a target planned at it, and only where FP8 tensor cores exist
    fp8 = decide("sm_120", "fp8_w8a8")["torch-scaled-mm"]
    assert fp8.run and fp8.configs == [{"SCALING": "rowwise"}, {"SCALING": "tensorwise"}]
    assert decide("sm_86", "fp8_w8a8")["torch-scaled-mm"].reason.startswith("needs sm_89+")


def test_call_sites_a_template_cannot_take_are_named():
    flash = adapters.BY_NAME["flash-attn"]
    probes = {"flash-attn": registry.Availability("flash-attn", "flash-attn", "2.7")}
    masked = {"sdpa": {"sites": [{"dtype": "bfloat16", "head_dim": 64, "mask": True}]}}
    (decision,) = registry.applicable(
        [flash], masked, capability=(8, 0), precision=None, available=probes
    )
    assert decision.reason == "no call site it takes: attention with an explicit mask"
    fp32 = {"sdpa": {"sites": [{"dtype": "float32", "head_dim": 64}]}}
    (decision,) = registry.applicable(
        [flash], fp32, capability=(8, 0), precision=None, available=probes
    )
    assert decision.reason == "no call site it takes: sdpa in float32 (takes bfloat16, float16)"
    functional = {"rms_norm": {"sites": [{"form": "functional", "hidden": 64}]}}
    (decision,) = registry.applicable(
        [adapters.BY_NAME["torch-rms-norm"]],
        functional,
        capability=(8, 0),
        precision=None,
        available={"torch-rms-norm": registry.Availability("torch-rms-norm", "torch", "2")},
    )
    assert decision.reason == "no call site it takes: already F.rms_norm (torch's fused kernel)"


# ------------------------------------------------------------------ candidates


def _render(
    name: str, configs: list[dict[str, Any]] | None = None, methods: tuple[str, ...] = ()
) -> str:
    adapter = adapters.BY_NAME[name]
    return template.render(
        adapter,
        target="mix",
        version="2.14.1+cu130",
        configs=configs if configs is not None else [{}],
        families="sdpa x1",
        methods=["forward", *methods],
    )


TEMPLATED = [a.name for a in adapters.ADAPTERS if not a.no_template]


@pytest.mark.parametrize("name", TEMPLATED)
def test_generated_candidates_pass_the_critic_and_compile(name):
    configs = [{"BACKEND": "cudnn"}] if name == "torch-sdpa" else None
    source = _render(name, configs, ("forward_step",))
    compile(source, f"libscout_{name}.py", "exec")
    assert critic.static_checks(source) == []
    label = f"{adapters.BY_NAME[name].package}@2.14.1+cu130"
    assert template.library_of(source) == label
    assert ledger.detect_backend(source) == f"library:{label}"
    assert "def forward_step(self, *args, **kwargs):" in source


def _load(source: str, path: Path) -> Any:
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"ka_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidates_run_the_references_graph_on_the_cpu(tmp_path):
    """Dynamo traces the reference; the graph runs eagerly with the library's ops in it."""
    torch.manual_seed(0)
    block = Block().eval()
    args = (torch.randn(2, 5, 32), *rope())
    with torch.inference_mode():
        want = block(*args)
    sdpa = _load(_render("torch-sdpa", [{"BACKEND": "math"}]), tmp_path / "a.py")
    norm = _load(_render("torch-rms-norm", [{"FUSE": 0}, {"FUSE": 1}]), tmp_path / "b.py")
    for module, config, rewritten in (
        (sdpa, {}, 1),
        (norm, {"FUSE": 0}, 2),
        (norm, {"FUSE": 1}, 2),
    ):
        candidate = module.build(block, **config)
        with torch.inference_mode():
            got = candidate(*args)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        assert candidate.reference is block
        assert candidate.rewritten == rewritten  # SDPA once; both RMSNorms


def _graphs(module: torch.nn.Module, *args: Any) -> list[Any]:
    graphs: list[Any] = []

    def keep(gm: Any, example_inputs: Any) -> Any:
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    torch.compile(module, backend=keep)(*args)
    return graphs


def test_rewrites_fold_the_patterns_of_dynamo_graphs():
    def rms(x, weight, eps, dtype, mid):
        return (x, weight, eps, dtype, mid)

    (graph,) = _graphs(Scale(16), torch.randn(3, 16))
    assert fx_rewrites.fold_rms_norm(graph.graph, rms) == 1
    calls = [n for n in graph.graph.nodes if n.op in ("call_function", "call_method")]
    assert [n.target for n in calls] == [rms]  # six ops became one
    assert calls[0].args[2:] == (1e-6, torch.float32, torch.float32)  # rounds in fp32
    (graph,) = _graphs(Plain(16), torch.randn(3, 16))
    assert fx_rewrites.fold_rms_norm(graph.graph, rms) == 1
    # the input itself times the fp32 rsqrt (promotion), the weight last: the same norm
    (graph,) = _graphs(Promoted(16).bfloat16(), torch.randn(3, 16).bfloat16())
    assert fx_rewrites.fold_rms_norm(graph.graph, rms) == 1
    calls = [n for n in graph.graph.nodes if n.op in ("call_function", "call_method")]
    assert [n.target for n in calls] == [rms]
    assert calls[0].args[3:] == (torch.bfloat16, torch.bfloat16)  # rounded before the weight

    def linear(x, weight, bias, residual, act):
        return x

    (graph,) = _graphs(Residual(), torch.randn(3, 16))
    assert fx_rewrites.fold_linear(graph.graph, linear) == 2
    folded = [n for n in graph.graph.nodes if n.target is linear]
    assert [n.args[3] is not None for n in folded] == [True, False]  # the residual
    assert [n.args[4] for n in folded] == [None, "gelu"]
    (graph,) = _graphs(Residual(), torch.randn(3, 16))  # FUSE=0: the epilogue keeps the bias
    assert fx_rewrites.fold_linear(graph.graph, linear, residual=False, gelu=False) == 2
    folded = [n for n in graph.graph.nodes if n.target is linear]
    assert [n.args[3:] for n in folded] == [(None, None), (None, None)]

    def silu_mul(gate, up):
        return gate

    (graph,) = _graphs(Feed(), torch.randn(3, 16))
    assert fx_rewrites.fold_silu_mul(graph.graph, silu_mul) == 1
    assert fx_rewrites.fold_linear(graph.graph, linear) == 3


def test_op_bars_spare_the_sweep_its_losers():
    def bar(name: str, kernel: float, eager: float, ok: bool = True) -> dict[str, Any]:
        out = {"config": {"BACKEND": name}, "ok": ok, "speedup": kernel, "eager_speedup": eager}
        return out if ok else {**out, "error": "rel_l2 0.3"}

    configs = [{"BACKEND": b} for b in ("flash", "efficient", "cudnn", "math", "new")]
    bars = [
        bar("flash", 1.0, 1.0),
        bar("efficient", 0.71, 0.87),
        bar("cudnn", 2.34, 0.82),  # a kernel-time win slower per eager call: swept
        bar("math", 0.85, 0.2, ok=False),
    ]
    keep, pruned = probe.prune(configs, bars)
    assert keep == [{"BACKEND": "flash"}, {"BACKEND": "cudnn"}, {"BACKEND": "new"}]
    assert pruned == [
        {"config": {"BACKEND": "efficient"}, "why": "op bar at most 0.87x"},
        {"config": {"BACKEND": "math"}, "why": "op bar fails: rel_l2 0.3"},
    ]
    # nothing worth it: the fastest correct one still gets its module-level verdict
    keep, pruned = probe.prune(configs[1:2] + configs[3:4], bars)
    assert keep == [{"BACKEND": "efficient"}] and len(pruned) == 1


class InPlace(Scale):
    """The written-out RMSNorm with ``+=`` (``add_``): an RMSNorm to detect and fold, but no
    pattern to replay for an op bar (a replayed in-place call would change its recorded
    input)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        variance += self.eps
        return self.weight * (x * torch.rsqrt(variance))


def test_written_out_rms_norms_get_op_bars_against_the_fused_kernel(tmp_path):
    """The pattern's recorded calls replayed (the reference's kernels) against what the
    adapter's rewrite puts in its place, on the same x and weight; the GPU timer faked."""
    torch.manual_seed(0)
    block = Block().to(torch.bfloat16).eval()  # two RMSNorms: q's in Mixer, the block's
    with torch.no_grad():
        for p in block.parameters():
            p.add_(0.1 * torch.randn_like(p))
    cos, sin = (t.bfloat16() for t in rope())
    invocations: list[Any] = []
    x = torch.randn(2, 5, 32).bfloat16()
    calls = detect.trace(block, (x, cos, sin), module=block, invocations=invocations)
    spans = detect.rms_spans(calls)
    assert [[calls[i].name for i in s.calls] for s in spans] == [
        ["to", "pow", "mean", "add", "rsqrt", "mul", "to", "mul"]
    ] * 2
    assert [(s.weight, s.eps, s.dtype, s.mid) for s in spans] == [
        ("mix.n.weight", 1e-6, "bfloat16", "bfloat16"),
        ("norm.weight", 1e-6, "bfloat16", "bfloat16"),
    ]
    assert calls[spans[1].entry].name == "to" and calls[spans[1].calls[0] - 1].name == "add"
    in_place = InPlace(8)
    traced = detect.trace(in_place, (torch.randn(3, 8),), module=in_place)
    assert "rms_norm" in detect.families(traced) and detect.rms_spans(traced) == []
    assert detect._in_place("TensorBase.add_") and detect._in_place("Tensor.__imul__")
    assert not detect._in_place("Tensor.__add__") and not detect._in_place("TensorBase.pow")

    configs = [{"FUSE": 0}, {"FUSE": 1}]
    path = tmp_path / "libscout_torch_rms_norm.py"
    path.write_text(_render("torch-rms-norm", configs))
    module = probe._import(path)
    timed: list[int] = []

    def timer(fns: Any) -> dict[str, list[float] | None]:
        timed.append(len(fns))
        return {"eager": [12.0, 4.0], "graph": [6.0, 2.5]}

    wrong = types.SimpleNamespace(  # drops the weight: fails the tolerance
        patterns=lambda reference, **config: {"rms_norm": lambda x, w, eps, dtype, mid: x}
    )
    modules = {"torch-rms-norm": (module, configs), "wrong": (wrong, [{}])}
    bars = probe.pattern_bars(block, invocations, calls, modules, timer=timer)
    assert [(b["adapter"], b["config"]) for b in bars] == [
        ("torch-rms-norm", {"FUSE": 0}),
        ("torch-rms-norm", {"FUSE": 0}),
        ("torch-rms-norm", {"FUSE": 1}),
        ("torch-rms-norm", {"FUSE": 1}),
        ("wrong", {}),
        ("wrong", {}),
    ]
    assert [b["signature"] for b in bars[:2]] == [
        "rms_norm (written out)([2, 5, 4, 8] [8] bfloat16; eps=1e-06)",
        "rms_norm (written out)([2, 5, 32] [32] bfloat16; eps=1e-06)",
    ]
    assert all(b["ok"] for b in bars[:4]), [b.get("error") for b in bars[:4]]
    assert bars[0] == {
        "adapter": "torch-rms-norm",
        "config": {"FUSE": 0},
        "op": probe.RMS_PATTERN,
        "signature": "rms_norm (written out)([2, 5, 4, 8] [8] bfloat16; eps=1e-06)",
        "calls": 1,
        "pattern_ops": 8,
        "ok": True,
        "ref_eager_us": 12.0,
        "eager_us": 4.0,
        "eager_speedup": 3.0,
        "ref_us": 6.0,
        "us": 2.5,
        "speedup": 2.4,
    }
    assert not bars[4]["ok"] and bars[4]["error"] and timed == [2] * 6
    keep, pruned = probe.prune([*configs, {}], bars)
    assert keep == configs and pruned[0]["why"].startswith("op bar fails")
    assert libscout.bar_groups(bars)[1]["libraries"][0]["speedup"] == 2.4


def test_the_probe_sweeps_no_candidate_that_changes_nothing_or_fails(tmp_path):
    """The dry run of a candidate on the dominant case: a rewrite that finds nothing in
    Dynamo's graphs (here: no RMSNorm at all) is not swept, nor one that fails there."""
    feed = Feed().eval()
    case = {"args": (torch.randn(2, 3, 16),), "kwargs": {}}
    path = tmp_path / "libscout_torch_rms_norm.py"
    path.write_text(_render("torch-rms-norm"))
    module, why = probe._dry_run(path, feed, None, case, {})
    assert module is not None and why == probe.UNCHANGED
    path = tmp_path / "libscout_torch_sdpa.py"
    path.write_text(_render("torch-sdpa", [{"BACKEND": "math"}]))
    core = Core().eval()
    q, kv = torch.randn(1, 4, 3, 8), torch.randn(1, 2, 3, 8)
    case = {"args": (q, kv, kv), "kwargs": {}}
    assert probe._dry_run(path, core, None, case, {"BACKEND": "math"})[1] is None
    module, why = probe._dry_run(path, core, None, case, {"BACKEND": "fastest"})
    assert module is None
    assert why == "its candidate failed on the dominant case in the probe: KeyError: 'fastest'"


# ------------------------------------------------------------------ FlashInfer attention


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: Any):
    """One request's attention as FlashInfer computes it: q [Sq, Hq, D], k / v [Skv, Hkv,
    D], each KV head serving Hq / Hkv consecutive query heads, the causal mask aligned to
    the last key."""
    sq, hq, d = q.shape
    skv, hkv = k.shape[0], k.shape[1]
    keys, values = (t.repeat_interleave(hq // hkv, dim=1).double() for t in (k, v))
    scores = torch.einsum("qhd,khd->hqk", q.double(), keys) * (scale or d**-0.5)
    if causal:
        allowed = torch.ones(sq, skv, dtype=torch.bool).tril(skv - sq)
        scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).to(q.dtype)


def _nhd(t: torch.Tensor, layout: str) -> torch.Tensor:
    """Tokens first: [..., S, H, D] of a tensor in ``layout``."""
    return t if layout == "NHD" else t.transpose(-3, -2)


class _FakeFlashInfer:
    """A ``flashinfer`` module that computes what FlashInfer's kernels compute (on the CPU,
    in fp64) and records every call: which kernel, the KV layout, whether K / V were passed
    in place (their storage) and each wrapper's plan (the page table it was given)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.plans: list[dict[str, Any]] = []
        self.module = _module("flashinfer")
        self.module.single_decode_with_kv_cache = self.single_decode
        self.module.single_prefill_with_kv_cache = self.single_prefill
        fake = self

        class Wrapper:
            kind = ""

            def __init__(self, workspace: torch.Tensor, kv_layout: str = "NHD", **kw: Any):
                assert workspace.dtype == torch.uint8 and workspace.numel() == 128 << 20
                self.layout, self.options, self.plan_ = kv_layout, kw, None

            def plan(self, *args: Any, **kwargs: Any) -> None:
                names = ["indptr", "indices", "last", "hq", "hkv", "d", "page"]
                if self.kind == "prefill":
                    names = ["qo_indptr", *names]
                self.plan_ = {"kind": self.kind, **dict(zip(names, args, strict=True)), **kwargs}
                fake.plans.append(self.plan_)

            def run(self, q: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor]) -> Any:
                plan = self.plan_
                assert plan is not None, "run() before plan()"
                k_pages, v_pages = (_nhd(t, self.layout) for t in kv)
                assert k_pages.shape[1] == plan["page"] and q.shape[-2:] == (plan["hq"], plan["d"])
                fake.calls.append(
                    (f"batch {self.kind}", self.layout, kv[0].data_ptr(), self.options)
                )
                outs = []
                for i in range(len(plan["last"])):
                    pages = plan["indices"][plan["indptr"][i] : plan["indptr"][i + 1]].long()
                    k, v = (t[pages].flatten(0, 1) for t in (k_pages, v_pages))
                    n = (len(pages) - 1) * plan["page"] + int(plan["last"][i])
                    if self.kind == "decode":
                        rows = q[i : i + 1]
                    else:
                        rows = q[plan["qo_indptr"][i] : plan["qo_indptr"][i + 1]]
                    causal = bool(plan.get("causal"))
                    outs.append(_attend(rows, k[:n], v[:n], causal, plan.get("sm_scale")))
                return torch.cat(outs)

        class Decode(Wrapper):
            kind = "decode"

        class Prefill(Wrapper):
            kind = "prefill"

        self.module.BatchDecodeWithPagedKVCacheWrapper = Decode
        self.module.BatchPrefillWithPagedKVCacheWrapper = Prefill

    def single_decode(self, q, k, v, kv_layout="NHD", use_tensor_cores=False, sm_scale=None):
        self.calls.append(("single decode", kv_layout, k.data_ptr(), use_tensor_cores))
        return _attend(q[None], _nhd(k, kv_layout), _nhd(v, kv_layout), False, sm_scale)[0]

    def single_prefill(self, q, k, v, causal=False, kv_layout="NHD", sm_scale=None):
        self.calls.append(("single prefill", kv_layout, k.data_ptr(), causal))
        return _attend(q, _nhd(k, kv_layout), _nhd(v, kv_layout), causal, sm_scale)


#: (batch, heads, kv heads, seq_q, seq_kv, causal, [B, S, H, D] projections viewed as heads)
FLASHINFER_CALLS = [
    (1, 4, 2, 1, 7, False, False),  # single decode, a KV cache (HND)
    (1, 4, 2, 5, 5, True, True),  # single prefill, causal, projections (NHD)
    (3, 4, 2, 1, 6, False, False),  # batched decode
    (3, 4, 1, 1, 6, False, True),
    (2, 4, 2, 3, 3, False, False),  # batched prefill
    (2, 4, 2, 4, 4, True, True),
]


def test_flashinfer_attention_maps_single_and_batched_calls(tmp_path, monkeypatch):
    """The FlashInfer template on the CPU with a fake library: one request through the
    single-request kernels, a batch through the paged wrappers with one page per request
    (page size = its KV length), K / V passed in place in their layout, each wrapper
    planned once per call shape; outputs as SDPA's."""
    fake = _FakeFlashInfer()
    monkeypatch.setitem(sys.modules, "flashinfer", fake.module)
    configs = [{"TENSOR_CORES": 0}, {"TENSOR_CORES": 1}]
    module = _load(_render("flashinfer-attention", configs), tmp_path / "fi.py")
    # the template takes CUDA half tensors only: the fake computes on the CPU in fp64
    monkeypatch.setattr(module, "_takes", lambda q, k, v, mask, *rest: mask is None)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.manual_seed(0)
    for cores in (0, 1):
        attend = module.ops(None, TENSOR_CORES=cores)[sdpa]
        for b, h, hkv, sq, skv, causal, projected in FLASHINFER_CALLS:

            def heads(n: int, seq: int, b: int = b, projected: bool = projected) -> torch.Tensor:
                if projected:
                    return torch.randn(b, seq, n, 16, dtype=torch.float64).transpose(1, 2)
                return torch.randn(b, n, seq, 16, dtype=torch.float64)

            q, k, v = heads(h, sq), heads(hkv, skv), heads(hkv, skv)
            want = sdpa(q, k, v, is_causal=causal, enable_gqa=True)
            for _ in range(2):  # the second call reuses the first one's plan
                got = attend(q, k, v, is_causal=causal, enable_gqa=True)
                torch.testing.assert_close(got, want)
            kind, layout, storage = fake.calls[-1][:3]
            assert kind == adapters.flashinfer_path({"batch": b, "seq_q": sq})
            assert layout == ("NHD" if projected else "HND")
            assert storage == k.data_ptr()  # the pages are the K tensor itself: no copy
    # one plan per call shape and decode form (TENSOR_CORES is the wrapper's)
    assert [(p["kind"], len(p["last"]), p["page"]) for p in fake.plans] == [
        ("decode", 3, 6),
        ("decode", 3, 6),
        ("prefill", 2, 3),
        ("prefill", 2, 4),
    ] * 2
    first = fake.plans[0]
    assert first["indptr"].tolist() == [0, 1, 2, 3] and first["indices"].tolist() == [0, 1, 2]
    assert first["last"].tolist() == [6, 6, 6] and first["q_data_type"] == torch.float64
    assert fake.plans[3]["causal"] and fake.plans[3]["qo_indptr"].tolist() == [0, 4, 8]
    batched = [c for c in fake.calls if c[0] == "batch decode"]
    assert {c[3]["use_tensor_cores"] for c in batched} == {False, True}
    # a new shape first seen inside a CUDA graph capture: no plan there, torch's SDPA instead
    monkeypatch.setattr(module, "_capturing", lambda t: True)
    q, k = torch.randn(5, 4, 1, 16, dtype=torch.float64), torch.randn(5, 2, 9, 16).double()
    plans, calls = len(fake.plans), len(fake.calls)
    got = attend(q, k, k, enable_gqa=True)
    torch.testing.assert_close(got, sdpa(q, k, k, enable_gqa=True))
    assert (len(fake.plans), len(fake.calls)) == (plans, calls)


def test_flashinfer_attention_takes_batches_from_the_call_shapes():
    """Batched SDPA call sites go to FlashInfer's paged wrappers (they were refused while
    the adapter had only the single-request kernels); decode sites sweep CUDA-core and
    tensor-core decode."""
    adapter = adapters.BY_NAME["flashinfer-attention"]
    probes = {adapter.name: registry.Availability(adapter.name, adapter.package, "0.6.17")}

    def site(batch: int, seq_q: int, seq_kv: int, dim: int = 128) -> dict[str, Any]:
        return {
            "dtype": "bfloat16",
            "batch": batch,
            "heads": 16,
            "kv_heads": 2,
            "seq_q": seq_q,
            "seq_kv": seq_kv,
            "head_dim": dim,
            "mask": False,
            "causal": False,
        }

    def decide(*sites: dict[str, Any]) -> registry.Decision:
        (decision,) = registry.applicable(
            [adapter],
            {"sdpa": {"sites": list(sites)}},
            capability=(8, 6),
            precision=None,
            available=probes,
        )
        return decision

    prefill = decide(site(32, 11, 11))
    assert prefill.run and prefill.configs == [{}] and prefill.sites == 1
    decode = decide(site(16, 1, 1024), site(1, 1, 2048))
    assert decode.configs == [{"TENSOR_CORES": 0}, {"TENSOR_CORES": 1}] and decode.sites == 2
    assert decide(site(4, 1, 64, dim=96)).reason == (
        "no call site it takes: head size 96 (FlashInfer: 64, 128, 256)"
    )
    paths = [adapters.flashinfer_path(s) for s in (site(1, 1, 9), site(1, 9, 9))]
    paths += [adapters.flashinfer_path(s) for s in (site(8, 1, 9), site(8, 9, 9))]
    assert paths == ["single decode", "single prefill", "batch decode", "batch prefill"]


def test_a_library_check_imports_the_submodules_it_names(monkeypatch):
    """A package does not import its submodules by itself: the probe imports them on the
    way to the function (Liger's ``transformers.functional``, measured on Liger-Kernel
    0.8.4: an attribute walk alone skipped it as missing)."""
    package = _module("liger_kernel")
    package.__path__ = []  # a package, its submodules not imported into it
    functional = _module("liger_kernel.transformers.functional", liger_rms_norm=print)
    transformers = _module("liger_kernel.transformers", functional=functional)
    transformers.__path__ = []
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers.functional", functional)
    norm = adapters.BY_NAME["liger-rmsnorm"]
    assert norm.check(package) is None
    assert adapters.BY_NAME["flashinfer-norm"].check(_module("flashinfer")) == (
        "flashinfer has no norm.rmsnorm"
    )


# ------------------------------------------------------------------ the step in a run


ADAPTER = adapters.BY_NAME["torch-sdpa"]
CONFIGS = [{"BACKEND": "cudnn"}, {"BACKEND": "flash"}]
BARS = [
    {
        "adapter": "torch-sdpa",
        "config": {"BACKEND": name},
        "op": "scaled_dot_product_attention",
        "signature": "scaled_dot_product_attention([32, 16, 11, 128] [32, 2, 11, 128] bfloat16)",
        "calls": 1,
        "ok": True,
        "ref_us": 36.3,
        "us": us,
        "speedup": round(36.3 / us, 3),
        "ref_eager_us": 37.1,
        "eager_us": eager,
        "eager_speedup": round(37.1 / eager, 3),
    }
    for name, us, eager in (("flash", 36.3, 37.5), ("cudnn", 16.5, 65.6))
]


def _scouted_run(tmp_path: Path) -> tuple[RunDir, truth.Truth]:
    run = sealed_run(tmp_path)
    keeper = truth.of(run)
    write_json(run.target("mix") / "spec.json", {"id": "mix", "module_class": "Mixer"})
    capture = truth.replace(run.capture_file("mix"))  # a placeholder: probe and sweep are fake
    capture.write_bytes(b"capture")
    keeper.seal(capture)
    return run, keeper


def _prober(capture: Path, *, out_dir: Path, target: str, **kwargs: Any) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, configs in (("torch-sdpa", CONFIGS), ("torch-rms-norm", [{}])):
        path = out_dir / probe.candidate_name(name)
        path.write_text(
            template.render(
                adapters.BY_NAME[name], target=target, version="2.14.1", configs=configs
            )
        )
        written[name] = str(path)
    return {
        "case": "x=[32, 11, 1024]:bfloat16",
        "families": {"sdpa": {"count": 1}, "rms_norm": {"count": 2}},
        "described": "sdpa x1, rms_norm x2 (manual)",
        "capability": [12, 0],
        "available": {"torch-sdpa": {"version": "2.14.1"}, "torch-rms-norm": {"version": "2.14.1"}},
        "decisions": [
            {"adapter": "torch-sdpa", "run": True, "configs": CONFIGS, "sites": 1},
            {"adapter": "torch-rms-norm", "run": True, "configs": [{}], "sites": 2},
            {
                "adapter": "flash-attn",
                "package": "flash-attn",
                "run": False,
                "reason": "flash-attn is not installed (pip install 'kernel-agent[libs]')",
            },
        ],
        "candidates": written,
        "op_bars": BARS,
        "seconds": 1.5,
    }


def _sweeper(capture: Path, src: Path, configs: list[dict[str, Any]], **kwargs: Any) -> dict:
    """The SDPA candidate is a keep at 1.25x; the RMSNorm one a fallback."""
    best = configs[0]
    bound = src.parent.parent / "bound" / src.name
    bound.parent.mkdir(exist_ok=True)
    bound.write_text(sweep.bind_config(src.read_text(), best))
    kwargs["prepare"](bound)
    if "sdpa" in src.name:
        evaluation = {
            "status": "ok",
            "correct": True,
            "speedup": 1.25,
            "pct_of_sol": 40.0,
            "custom_kernel_share": 0.3,
            "cases": [],
        }
    else:
        evaluation = {
            "status": "fallback",
            "correct": False,
            "error": "case 0: launches only kernels the reference launches",
        }
    table = [{"config": c, "correct": True, "speedup": 1.25, "status": "ok"} for c in configs]
    info = {"configs": len(configs), "passed": len(configs), "failed": 0, "skipped": 0}
    return {"evaluation": evaluation, "config": best, "sweep": {**info, "table": table}}


def test_scout_records_library_rows_and_the_bar(tmp_path, monkeypatch):
    monkeypatch.setenv(gpulock.ENV, "1")  # the fake probe and sweep need no GPU
    run, keeper = _scouted_run(tmp_path)
    found = libscout.scout_target(
        run, "mix", keeper=keeper, prober=_prober, sweeper=_sweeper, say=lambda m: None
    )
    rows = [r for r in ledger.rows(run) if r["target"] == "mix"]
    assert [r["backend"] for r in rows] == ["library:torch@2.14.1"] * 2
    assert [r["status"] for r in rows] == ["keep", "fallback"]
    assert [r["session"] for r in rows] == ["libscout", "libscout"]
    assert [r["idea"] for r in rows] == ["library-torch-sdpa", "library-torch-rms-norm"]
    assert [ledger.title(r) for r in rows] == [
        "library torch-sdpa: BACKEND='cudnn'",
        "library torch-rms-norm: default",
    ]
    assert rows[0]["hypothesis"] == (
        "library scout: SDPA pinned to one backend (torch.nn.attention.sdpa_kernel) "
        "(torch 2.14.1) [sweep: BACKEND='cudnn'; best of 2/2 configs]"
    )
    records = read_jsonl(run.results_file("mix"))
    assert budget.non_improving_streak(records) == 0  # a scout row is no agent's attempt
    assert (
        records[0]["snapshot"].startswith("history/")
        and "libscout_torch_sdpa" in (records[0]["snapshot"])
    )
    assert read_json(run.target("mix") / libscout.FILE)["adapters"][0]["speedup"] == 1.25
    assert found["skipped"] == [
        {
            "adapter": "flash-attn",
            "package": "flash-attn",
            "reason": "flash-attn is not installed (pip install 'kernel-agent[libs]')",
        }
    ]
    snapshot = found["adapters"][0]["snapshot"]
    assert libscout.bar_lines(run, "mix") == [
        "",
        "## Library bar (the library scout: library kernels with no agent; a floor to beat, "
        "never a ceiling)",
        f"* Best: `{snapshot}` (SDPA pinned to one backend (torch.nn.attention.sdpa_kernel), "
        "`library:torch@2.14.1`, BACKEND='cudnn'): 1.25x, 40 % of SOL. Copy it or call its "
        "library inside your own kernel, then beat it.",
        "* Measured (module level, eager): torch-sdpa (BACKEND='cudnn'): 1.25x, 40 % of SOL "
        "(keep); torch-rms-norm: fallback: case 0: launches only kernels the reference "
        "launches",
        "* Op bars (the reference's op vs the library's on its recorded inputs):",
        "  - `scaled_dot_product_attention([32, 16, 11, 128] [32, 2, 11, 128] bfloat16)`: "
        "reference 36.3 us; BACKEND='cudnn' 16.5 us (2.20x); BACKEND='flash' 36.3 us (1.00x) "
        "[kernel time, CUDA graph]; eager per call: reference 37.1, BACKEND='cudnn' 65.6, "
        "BACKEND='flash' 37.5",
        "* Not run: flash-attn (flash-attn is not installed (pip install 'kernel-agent[libs]'))",
    ]
    assert libscout.planner_lines(run, ["mix", "other"]) == [
        "* `mix` (sdpa x1, rms_norm x2 (manual)): torch-sdpa 1.25x, 40 % of SOL; op bars: "
        "torch-sdpa BACKEND='cudnn' 2.20x on "
        "`scaled_dot_product_attention([32, 16, 11, 128] [32, 2, 11, 128] bfloat16)`"
    ]
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "ceilings.md").write_text("# Ceilings\n\n| a | b |\n")
    libscout.write_ceilings(run, profile)
    libscout.write_ceilings(run, profile)  # replaced, not appended twice
    text = (profile / "ceilings.md").read_text()
    assert text.startswith("# Ceilings\n\n| a | b |\n\n## Library bars (libscout)\n")
    assert text.count(libscout.CEILINGS_HEAD) == 1 and "* `mix` (sdpa x1" in text


def test_a_failed_probe_is_recorded_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv(gpulock.ENV, "1")
    run, keeper = _scouted_run(tmp_path)
    found = libscout.scout_target(
        run,
        "mix",
        keeper=keeper,
        prober=lambda *a, **k: {"error": "probe exited with 1: CUDA error"},
        say=lambda m: None,
    )
    assert found["error"] == "probe exited with 1: CUDA error"
    assert libscout.bar_lines(run, "mix")[-1] == (
        "* The scout failed on this target: probe exited with 1: CUDA error"
    )
    assert ledger.rows(run) == []


def test_a_scout_best_is_a_floor_never_a_stop():
    policy = scheduler.Policy(speedup_goal=2.0, sol_stop=0.9)
    rows = [
        {
            "exp": 1,
            "target": "mix",
            "status": "keep",
            "correct": True,
            "speedup": 2.5,
            "spread": None,
            "hypothesis": budget.LIBRARY_HYPOTHESIS + "SDPA pinned",
            "snapshot": "001_libscout_torch_sdpa_00000000.py",
        }
    ]
    arm = scheduler.Arm("mix", scheduler.KERNEL, 10.0)
    scheduler._kernel_history(arm, rows)
    assert arm.best == 2.5 and arm.scouted and arm.streak == 0
    arm.sol = 0.95
    assert scheduler.stop_reason(arm, policy) is None
    rows.append({**rows[0], "exp": 2, "speedup": 2.6, "hypothesis": "fused kernel"})
    agent = scheduler.Arm("mix", scheduler.KERNEL, 10.0)
    scheduler._kernel_history(agent, rows)
    agent.sol = 0.95
    assert not agent.scouted
    assert str(scheduler.stop_reason(agent, policy)).startswith("at 95% of its recipe's roofline")


def test_orchestrator_scouts_each_target_once(tmp_path, simulated):  # noqa: F811
    orch = make(tmp_path)
    run = orch.run
    write_json(run.target("mix") / "spec.json", {"id": "mix", "module_class": "Mixer"})
    capture = truth.replace(run.capture_file("mix"))
    capture.write_bytes(b"capture")
    calls: list[tuple[str, dict[str, Any]]] = []

    def scouter(run_: RunDir, target_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((target_id, kwargs))
        return {"seconds": 2.0, "adapters": [{"adapter": "torch-sdpa"}]}

    orch.scouter = scouter
    asyncio.run(orch.scout_libraries(["mix", "missing"]))
    asyncio.run(orch.scout_libraries(["mix"]))
    assert [c[0] for c in calls] == ["mix"]
    assert set(calls[0][1]) == {"keeper", "timeout", "backends", "race"}
    entry = libscout.scouted(run, "mix")
    assert entry is not None and entry.pop("key")["families"] is None  # it found no families
    assert entry == {"seconds": 2.0, "error": None, "adapters": ["torch-sdpa"]}
    assert [e["target"] for e in ledger.events(run) if e["event"] == "library_scout"] == ["mix"]
    off = make(tmp_path / "off", library_scout=False)
    write_json(off.run.target("mix") / "spec.json", {"id": "mix", "module_class": "Mixer"})
    truth.replace(off.run.capture_file("mix")).write_bytes(b"capture")
    off.scouter = scouter
    asyncio.run(off.scout_libraries(["mix"]))
    assert len(calls) == 1


# ------------------------------------------------------------------ the scout's key


def _versions(monkeypatch, installed: dict[str, str]) -> dict[str, str]:
    """Distribution metadata as ``installed`` says (a test upgrades or removes in place);
    no library module of the registry but torch importable."""
    _fake_libraries(monkeypatch, {})
    monkeypatch.setattr(registry, "package_version", lambda p: installed.get(p))
    return installed


H100 = toolchain.GPUInfo("Fake H100", (9, 0), 80.0, 132, 50.0)


def test_versions_are_read_from_the_metadata_without_an_import(monkeypatch):
    installed = _versions(
        monkeypatch,
        {
            "torch": "2.10.0+cu128",
            "nvidia-cublas-cu12": "12.8.4.1",
            "liger-kernel": "0.6.1",
            "triton": "3.6.0",
        },
    )

    def no_import(name: str, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"imported {name}")

    monkeypatch.setattr(registry.importlib, "import_module", no_import)
    assert registry.versions(adapters.ADAPTERS) == {
        "liger-kernel": "0.6.1",
        "nvidia-cublas-cu12": "12.8.4.1",
        "torch": "2.10.0+cu128",
        "triton": "3.6.0",
    }
    del installed["torch"]  # a source tree on PYTHONPATH: it imports, no metadata
    assert registry.versions([adapters.BY_NAME["torch-sdpa"]]) == {"torch": registry.NO_METADATA}


def test_a_scout_is_keyed_by_the_libraries_of_its_families_on_this_gpu(monkeypatch):
    installed = _versions(
        monkeypatch,
        {
            "torch": "2.10.0+cu128",
            "nvidia-cudnn-cu12": "9.10.2.21",
            "nvidia-cublas-cu12": "12.8.4.1",
            "flash-attn": "2.8.3",
            "flash-attn-3": "3.0.0b1",
            "liger-kernel": "0.6.1",
            "triton": "3.6.0",
        },
    )
    a10 = GPUS["sm_86"]
    sdpa = libscout.key(["sdpa"], a10)
    assert sdpa == {  # FlashAttention 3 needs sm_90: not this GPU's; FlashInfer: not here
        "gpu": "Fake A10 (sm_86)",
        "families": ["sdpa"],
        "libraries": {
            "flash-attn": "2.8.3",
            "nvidia-cudnn-cu12": "9.10.2.21",
            "torch": "2.10.0+cu128",
        },
    }
    assert libscout.key(["sdpa"], H100)["libraries"]["flash-attn-3"] == "3.0.0b1"
    assert libscout.key(["linear"], a10, nvcc="12.9")["libraries"] == {  # cuBLASLt compiles
        "nvidia-cublas-cu12": "12.8.4.1",
        "torch": "2.10.0+cu128",
        "nvcc (CUDA toolkit)": "12.9",
    }
    assert "nvcc (CUDA toolkit)" not in libscout.key(["sdpa"], a10, nvcc="12.9")["libraries"]
    assert libscout.key([], a10)["libraries"] == {}  # no family: no library matters
    assert set(libscout.key(None, a10)["libraries"]) == set(installed) - {"flash-attn-3"}
    norm = {"key": libscout.key(["rms_norm"], a10)}
    attention = {"key": sdpa}

    assert libscout.stale(attention, a10) is None and libscout.stale(norm, a10) is None
    installed.update({"liger-kernel": "0.6.2", "triton": "3.7.0", "flash-attn-3": "3.0.0"})
    assert libscout.stale(attention, a10) is None  # Liger and FA3 change no SDPA bar here
    assert libscout.stale(norm, a10) == "liger-kernel 0.6.1 → 0.6.2; triton 3.6.0 → 3.7.0"
    installed.update({"flash-attn": "2.8.4", "flashinfer-python": "0.2.6"})
    del installed["nvidia-cudnn-cu12"]
    assert libscout.stale(attention, a10) == (
        "flash-attn 2.8.3 → 2.8.4; flashinfer-python 0.2.6 installed; "
        "nvidia-cudnn-cu12 9.10.2.21 removed"
    )
    blackwell = libscout.stale({"key": libscout.key(["sdpa"], a10)}, GPUS["sm_120"])
    assert blackwell == "GPU Fake A10 (sm_86) → Fake RTX 5070 Ti (sm_120)"
    assert libscout.stale({"seconds": 2.0, "adapters": []}, a10) == libscout.NO_KEY


def test_orchestrator_scouts_again_when_a_library_changes(
    tmp_path,
    simulated,  # noqa: F811
    monkeypatch,
    capsys,
):
    installed = _versions(monkeypatch, {"torch": "2.10.0+cu128", "liger-kernel": "0.6.1"})
    installed["triton"] = "3.6.0"
    orch = make(tmp_path)
    monkeypatch.setattr(orch.tc, "gpu", GPUS["sm_86"], raising=False)
    run = orch.run
    families = {"core": {"sdpa": {"count": 1}}, "norm": {"rms_norm": {"count": 2}}}
    for target_id in families:
        write_json(run.target(target_id) / "spec.json", {"id": target_id})
        truth.replace(run.capture_file(target_id)).write_bytes(b"capture")
    calls: list[str] = []

    def scouter(run_: RunDir, target_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(target_id)
        found = {"seconds": 1.0, "families": families[target_id], "adapters": []}
        write_json(run_.target(target_id) / libscout.FILE, {"target": target_id, **found})
        return found

    orch.scouter = scouter
    asyncio.run(orch.scout_libraries(["core", "norm"]))
    asyncio.run(orch.scout_libraries(["core", "norm"]))  # nothing changed: no scout
    assert calls == ["core", "norm"]
    assert libscout.scouted(run, "norm")["key"] == {
        "gpu": "Fake A10 (sm_86)",
        "families": ["rms_norm"],
        "libraries": {"liger-kernel": "0.6.1", "torch": "2.10.0+cu128", "triton": "3.6.0"},
    }
    installed["liger-kernel"] = "0.6.2"  # Liger has RMSNorm, not attention: only `norm`
    asyncio.run(orch.scout_libraries(["core", "norm"]))
    assert calls == ["core", "norm", "norm"]
    assert "libscout: norm: scouting again: liger-kernel 0.6.1 → 0.6.2" in capsys.readouterr().out
    entry = libscout.scouted(run, "norm")
    assert entry["scouts"] == 2 and entry["rescouted"] == "liger-kernel 0.6.1 → 0.6.2"
    assert entry["key"]["libraries"]["liger-kernel"] == "0.6.2"
    # a scout remembered before the key: stale once, said why; then keyed like the rest
    libscout.remember(run, "core", {"seconds": 1.0, "error": None, "adapters": []})
    asyncio.run(orch.scout_libraries(["core", "norm"]))
    asyncio.run(orch.scout_libraries(["core", "norm"]))
    assert calls == ["core", "norm", "norm", "core"]
    assert f"libscout: core: scouting again: {libscout.NO_KEY}" in capsys.readouterr().out
    events = [e for e in ledger.events(run) if e["event"] == "library_scout"]
    assert [(e["target"], e.get("rescouted")) for e in events] == [
        ("core", None),
        ("norm", None),
        ("norm", "liger-kernel 0.6.1 → 0.6.2"),
        ("core", libscout.NO_KEY),
    ]
    text = write_report(run).read_text()  # report.md's section, from these files
    assert "## Library scout" in text
    assert "* `norm`: Fake A10 (sm_86); liger-kernel 0.6.2, torch 2.10.0+cu128, triton " in text


def test_the_report_shows_each_targets_scout_and_whether_an_agent_beat_it(tmp_path, monkeypatch):
    monkeypatch.setenv(gpulock.ENV, "1")
    run, keeper = _scouted_run(tmp_path)
    assert libscout.report_lines(run) == []  # nothing scouted: no section
    libscout.scout_target(
        run, "mix", keeper=keeper, prober=_prober, sweeper=_sweeper, say=lambda m: None
    )
    key = {
        "gpu": "Fake A10 (sm_86)",
        "families": ["rms_norm", "sdpa"],
        "libraries": {"nvidia-cudnn-cu12": "9.10.2.21", "torch": "2.14.1"},
    }
    libscout.remember(run, "mix", {"key": key, "scouts": 2, "rescouted": "torch 2.14.0 → 2.14.1"})
    write_json(run.target("lin") / "spec.json", {"id": "lin"})
    write_json(run.target("lin") / libscout.FILE, {"target": "lin", "error": "probe exited"})
    snapshot = libscout.summary(run, "mix")["adapters"][0]["snapshot"]
    head = [
        "",
        "## Library scout",
        "",
        "Library kernels on each target with no agent (`libscout/`, #227): every adapter of "
        "the op families the reference calls, swept and fully evaluated at module level. The "
        "best is the bar the engineers start from: a floor to beat, never a ceiling.",
        "",
        "| target | op families | adapters tried | best library candidate | best agent kernel "
        "| bar beaten |",
        "|---|---|---|---|---|---|",
        "| `lin` | scout failed: probe exited | — | — | — | — |",
    ]
    mix = (
        "| `mix` | sdpa x1, rms_norm x2 (manual) | torch-sdpa 1.25x; torch-rms-norm fallback "
        "| `library:torch@2.14.1` torch-sdpa (BACKEND='cudnn'): 1.25x, 40 % of SOL | {agent} "
        "| {beaten} |"
    )
    tail = [
        "",
        "Not run (and why):",
        "",
        "* `mix`: flash-attn: flash-attn is not installed (pip install 'kernel-agent[libs]')",
        "",
        "Scouted with (a target is scouted again when a library of its families or the GPU "
        "changes):",
        "",
        "* `lin`: no key (scouted before the key)",
        "* `mix`: Fake A10 (sm_86); nvidia-cudnn-cu12 9.10.2.21, torch 2.14.1; scouted 2 "
        "times, last after: torch 2.14.0 → 2.14.1",
    ]
    assert libscout.report_lines(run) == [
        *head,
        mix.format(agent="—", beaten="no agent kernel"),
        *tail,
    ]
    assert snapshot.startswith("history/")
    kernel(run, "mix", 1.1, name="slower")  # an agent's kernel below the bar
    assert libscout.report_lines(run)[8] == mix.format(
        agent="1.10x `" + Path(libscout.agent_best(run, "mix")["snapshot"]).name + "`",
        beaten="no (1.10x ≤ 1.25x)",
    )
    kernel(run, "mix", 1.6, name="fused")
    best = Path(libscout.agent_best(run, "mix")["snapshot"]).name
    assert libscout.report_lines(run)[8] == mix.format(
        agent=f"1.60x `{best}`", beaten="yes (1.28x the bar)"
    )


def test_the_ceilings_table_gets_the_library_bar_as_a_column(tmp_path):
    run = sealed_run(tmp_path)
    profile = ceilings_profile()
    run.profile_dir.mkdir(parents=True, exist_ok=True)
    write_json(run.profile_dir / "profile.json", profile)
    ceilings.write(run.profile_dir, profile, PEAKS, 5000.0)
    before = (run.profile_dir / "ceilings.md").read_text()
    specs = {
        "dit": {"module_class": "VoxCPMLocDiT", "qualname": "model.feat_decoder.estimator"},
        "mlp": {"module_class": "MiniCPMMLP", "qualname": "model.base_lm.layers.3.mlp"},
        "proj": {"module_class": "Linear"},  # no instance group known: its class's rows
        "norm": {"module_class": "RMSNorm"},  # not scouted (and under 1 % of the run: no row)
    }
    for target_id, spec in specs.items():
        write_json(run.target(target_id) / "spec.json", {"id": target_id, **spec})
    correct = {"adapter": "torch-sdpa", "correct": True, "speedup": 1.31, "status": "keep"}
    wrong = {"adapter": "liger-swiglu", "correct": False, "status": "fail"}
    found = {
        "dit": {"described": "sdpa x2", "adapters": [correct]},
        "mlp": {"described": "gated_mlp x1", "adapters": [wrong]},
        "proj": {"error": "probe exited with 1"},
    }
    for target_id, data in found.items():
        write_json(run.target(target_id) / libscout.FILE, {"target": target_id, **data})
    table = read_json(run.profile_dir / "ceilings.json")
    assert ceilings.markdown(table).lstrip("\n") == before  # the json renders it again
    libscout.write_ceilings(run)
    libscout.write_ceilings(run)  # rendered again, not a second column
    text = (run.profile_dir / "ceilings.md").read_text()
    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("| target |"))
    assert header.endswith("| saves ms | library bar |")

    def row(prefix: str) -> str:
        return next(line for line in lines if line.startswith(prefix))

    dit = row("| `VoxCPMLocDiT` `model.feat_decoder.estimator` |")
    assert dit.endswith(" | 1.31x torch-sdpa |")
    assert row("| `MiniCPMMLP` ").endswith(" | none correct |")
    assert row("| `Linear` ").endswith(" | scout failed |")
    assert row("| `UnifiedCFM` ").endswith(" | — |")  # holds the scouted LocDiT: not its bar
    assert "*library bar*: the module speedup of the best library scout candidate" in text
    assert text.count(libscout.CEILINGS_HEAD) == 1
    assert "* `dit` (sdpa x2): torch-sdpa 1.31x" in text
    # several targets on one row: each named
    write_json(run.target("dit2") / "spec.json", {"id": "dit2", **specs["dit"]})
    write_json(run.target("dit2") / libscout.FILE, {"target": "dit2", "adapters": []})
    bars = libscout.ceiling_bars(run, table)
    assert bars is not None
    assert bars[("VoxCPMLocDiT@model.feat_decoder.estimator", "prefill")] == (
        "`dit` 1.31x torch-sdpa; `dit2` no adapter"
    )
    assert libscout.ceiling_bars(sealed_run(tmp_path / "none"), table) is None
    # the newest round's profile too (what its re-planner reads), never an empty file
    newest = run.root / "rounds" / "2" / "profile"
    newest.mkdir(parents=True)
    ceilings.write(newest, profile, PEAKS, 5000.0)
    (run.root / "rounds" / "1" / "profile").mkdir(parents=True)
    libscout.write_ceilings(run)
    assert "| saves ms | library bar |" in (newest / "ceilings.md").read_text()
    assert not (run.root / "rounds" / "1" / "profile" / "ceilings.md").exists()
    bare = sealed_run(tmp_path / "bare")
    bare.profile_dir.mkdir(parents=True, exist_ok=True)
    libscout.write_ceilings(bare)  # nothing scouted, no table: nothing written
    assert not (bare.profile_dir / "ceilings.md").exists()


def test_the_export_lists_the_libraries_its_kernels_call(tmp_path):
    run = sealed_run(tmp_path)
    write_json(run.target("lin") / "spec.json", {"module_class": "Linear"})
    history = run.history_dir("lin")
    history.mkdir(parents=True)
    candidate = history / "001_libscout_cublaslt_0a0a0a0a.py"
    candidate.write_text(
        'KA_LIBRARY = "torch@2.14.1+cu130"\nKA_LICENCE = "BSD-3-Clause"\n\n\n'
        "def build(reference):\n    return reference\n"
    )
    out = export_optimized(run, [("kernel", f"lin={candidate}", 0.0)])
    assert (out / "requirements.txt").read_text() == (
        "# Libraries the exported kernels call (kernel-agent's library scout, #227).\n"
        "torch==2.14.1  # licence BSD-3-Clause; built as 2.14.1+cu130; used by kernels/lin.py\n"
    )
    assert read_json(out / "manifest.json")["libraries"] == [
        {
            "package": "torch",
            "version": "2.14.1+cu130",
            "licence": "BSD-3-Clause",
            "by": ["kernels/lin.py"],
        }
    ]
    plain = sealed_run(tmp_path / "plain")
    write_json(plain.target("lin") / "spec.json", {"module_class": "Linear"})
    plain.history_dir("lin").mkdir(parents=True)
    other = plain.history_dir("lin") / "001_lin_0b0b0b0b.py"
    other.write_text("def build(reference):\n    return reference\n")
    out = export_optimized(plain, [("kernel", f"lin={other}", 0.0)])
    assert not (out / "requirements.txt").exists()


def test_doctor_lists_libraries_and_where_their_adapters_run(monkeypatch):
    _fake_libraries(monkeypatch, {"flash_attn": _module("flash_attn", flash_attn_func=print)})
    lines = libscout.doctor_lines((8, 6), {"cuda": True})
    assert lines[0].startswith("library scout (#227)")
    text = "\n".join(lines)
    assert "  flash-attn 9.9 (BSD-3-Clause)" in text
    assert "    flash-attn [sdpa]: runs here" in text  # run on an A10 (#227)
    assert "    flash-attn-3 [sdpa, not verified on a GPU]: skipped: needs sm_90" in text
    assert "  quack-kernels not installed (Apache-2.0)" in text
    assert "    torch-sdpa [sdpa]: runs here" in text
    assert "    torch-scaled-mm [linear]: skipped: needs sm_89+" in text


# ------------------------------------------------------------------ follow-ups 3, 5, 6, 8


def test_a_written_out_rms_norm_with_an_in_place_add_is_detected_and_folded(tmp_path):
    """``variance += eps`` reaches the trace as ``add_`` and Dynamo's graph as
    ``operator.iadd``: the same RMSNorm, detected and folded."""
    torch.manual_seed(0)
    norm = InPlace(8).eval()
    with torch.no_grad():
        norm.weight.add_(torch.randn(8))
    x = torch.randn(3, 8)
    assert found_in(norm, x)["rms_norm"]["sites"] == [
        {
            "form": "manual",
            "hidden": 8,
            "eps": 1e-6,
            "weight": "weight",
            "dtype": "float32",
            "rows": 3,
        }
    ]
    (graph,) = _graphs(norm, x)
    assert any(n.target is operator.iadd for n in graph.graph.nodes)
    assert fx_rewrites.fold_rms_norm(graph.graph, lambda *a: a[0]) == 1
    module = _load(_render("torch-rms-norm", [{"FUSE": 0}]), tmp_path / "n.py")
    with torch.inference_mode():
        candidate = module.build(norm, FUSE=0)
        torch.testing.assert_close(candidate(x), norm(x), rtol=1e-5, atol=1e-5)
    assert candidate.rewritten == 1


def test_norms_in_torchscript_functions_are_run_as_python_and_inlined(tmp_path, monkeypatch):
    """A ``@torch.jit.script`` RMSNorm runs in C++, where the trace sees nothing: the trace
    runs its Python (its original, or its TorchScript code as Python when it has none, as a
    function of a TorchScript archive), and a candidate lets Dynamo trace into it."""
    torch.manual_seed(0)
    fused = Fused(16).eval()
    with torch.no_grad():
        fused.weight.add_(torch.randn(16))
    x = torch.randn(3, 16)
    with torch.inference_mode():
        want = fused(x)
    assert detect.trace(fused, (x,)) == []  # no module given: TorchScript, nothing seen
    inlined: list[str] = []
    calls = detect.trace(fused, (x,), module=fused, inlined=inlined)
    assert inlined == ["norm_scripted"]
    assert fx_rewrites.torchscript_functions(Block()) == []  # in its file, never called
    assert detect.describe(detect.families(calls)) == "rms_norm x1 (manual)"
    assert isinstance(libscout_toy.norm_scripted, torch.jit.ScriptFunction)  # put back
    assert fx_rewrites.torchscript_python(libscout_toy.norm_scripted) is libscout_toy._norm

    # a TorchScript function without its Python original: its code, run as Python
    monkeypatch.delattr(libscout_toy.norm_scripted, "_torchdynamo_inline")
    python = fx_rewrites.torchscript_python(libscout_toy.norm_scripted)
    assert python is not None and python is not libscout_toy._norm
    with torch.inference_mode():
        torch.testing.assert_close(python(x, fused.weight, 1e-6), want)
    assert "rms_norm" in detect.families(detect.trace(fused, (x,), module=fused))
    case = {"args": (x,), "kwargs": {}}
    torch._dynamo.reset()
    plain = tmp_path / "plain.py"  # Dynamo breaks its graph there: nothing to rewrite
    plain.write_text(_render("torch-rms-norm"))
    assert probe._dry_run(plain, fused, None, case, {})[1] == probe.UNCHANGED
    torch._dynamo.reset()
    inlining = tmp_path / "inlining.py"
    inlining.write_text(
        template.render(
            adapters.BY_NAME["torch-rms-norm"],
            target="t",
            version="2",
            configs=[{}],
            torchscript=True,
        )
    )
    output: list[Any] = []
    module, why = probe._dry_run(inlining, fused, None, case, {}, output)
    assert why is None and module is not None
    torch.testing.assert_close(output[0], want, rtol=1e-5, atol=1e-5)
    assert critic.static_checks(inlining.read_text()) == []


def cos_sin(batch: int = 2, seq: int = 5, dim: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half tables: their two halves repeat the frequencies, as models build them."""
    torch.manual_seed(2)
    freqs = torch.randn(batch, seq, dim // 2)
    table = torch.cat((freqs, freqs), dim=-1)
    return table.cos(), table.sin()


def _fake_liger(monkeypatch) -> None:
    functional = _module(
        "liger_kernel.transformers.functional",
        liger_rms_norm=print,
        liger_swiglu=print,
        liger_rope=lambda q, k, cos, sin: (q, k),
    )
    transformers = _module("liger_kernel.transformers", functional=functional)
    liger = _module("liger_kernel", transformers=transformers)
    liger._submodules = {  # type: ignore[attr-defined]
        "transformers": transformers,
        "transformers.functional": functional,
    }
    _fake_libraries(monkeypatch, {"liger_kernel": liger})


def test_rope_of_q_and_k_is_detected_as_a_pair_and_folded_into_one_call(tmp_path, monkeypatch):
    """Rotate-half RoPE of q and k with one cos and sin: two paired sites, one call of
    Liger's kernel; q alone is not Liger's; a q read again afterwards is copied first."""
    torch.manual_seed(0)
    rotary = Rotary().eval()
    x = torch.randn(2, 5, 32)
    cos, sin = cos_sin()
    found = found_in(rotary, x, cos, sin)
    assert found["rotary"]["sites"] == [
        {"shape": [2, 4, 5, 8], "head_dim": 8, "dtype": "float32", "paired": True},
        {"shape": [2, 2, 5, 8], "head_dim": 8, "dtype": "float32", "paired": True},
    ]
    alone = found_in(Mixer(), torch.randn(2, 5, 32), *rope())
    assert [s["paired"] for s in alone["rotary"]["sites"]] == [False]
    _fake_liger(monkeypatch)
    probes = registry.availability([adapters.BY_NAME["liger-rope"]])

    def decide(found: dict[str, Any]) -> registry.Decision:
        (decision,) = registry.applicable(
            [adapters.BY_NAME["liger-rope"]],
            found,
            capability=(8, 6),
            precision="exact",
            available=probes,
        )
        return decision

    assert decide(found).run and decide(found).sites == 2
    assert decide(alone).reason == (
        "no call site it takes: rotary embedding of one tensor (Liger's RoPE rotates q and k "
        "with one cos and sin together)"
    )

    def pair(q: Any, k: Any, cos: Any, sin: Any, q_free: bool, k_free: bool) -> Any:
        return q, k

    for module, free in ((rotary, (True, True)), (Rotary(keep=True), (False, True))):
        (graph,) = _graphs(module, x, cos, sin)
        assert fx_rewrites.fold_rope(graph.graph, pair) == 1
        (node,) = [n for n in graph.graph.nodes if n.target is pair]
        assert node.args[4:] == free  # q read again: the library must not write into it
        assert not any(n.target is torch.cat for n in graph.graph.nodes)
    (graph,) = _graphs(Mixer(), torch.randn(2, 5, 32), *rope())
    assert fx_rewrites.fold_rope(graph.graph, pair) == 0  # q alone

    module = _load(_render("liger-rope"), tmp_path / "rope.py")
    with torch.inference_mode():
        candidate = module.build(rotary)  # on the CPU: the reference's math
        torch.testing.assert_close(candidate(x, cos, sin), rotary(x, cos, sin))
    assert candidate.rewritten == 1
    q = torch.randn(2, 4, 5, 8)
    assert module._table(cos.unsqueeze(1), q).shape == (2, 5, 8)  # as Liger takes it
    assert module._table(cos[0], q).shape == (1, 5, 8)
    assert module._table(torch.randn(4, 5, 8), q) is None  # one table per head


def test_gemlite_runs_at_low_bit_weight_precisions_on_the_gpus_of_each_format():
    gemlite = adapters.BY_NAME["gemlite"]
    probes = {"gemlite": registry.Availability("gemlite", "gemlite", "0.6.0")}
    linear = {"linear": {"sites": [{"m": 1, "n": 4096, "k": 1024, "dtype": "bfloat16"}]}}

    def decide(arch: str, precision: str, found: dict[str, Any] = linear) -> registry.Decision:
        (decision,) = registry.applicable(
            [gemlite],
            found,
            capability=GPUS[arch].capability,
            precision=precision,
            available=probes,
        )
        return decision

    assert decide("sm_86", "exact").reason == (
        "serves fp8_weights, int8_weights, fp4_weights targets; this one is exact"
    )
    assert decide("sm_86", "fp8_w8a8").reason.startswith("serves fp8_weights")
    assert decide("sm_86", "int8_weights").configs == [{"FORMAT": "int8"}]
    assert decide("sm_86", "fp8_weights").reason == (
        "fp8 weights need sm_89+ (Triton converts e4m3 from sm_89; sm_86: a compile error); "
        "this GPU is sm_86"
    )
    assert decide("sm_120", "fp8_weights").configs == [{"FORMAT": "fp8"}]
    assert decide("sm_86", "fp4_weights").configs == [{"FORMAT": "mxfp4"}]
    assert decide("sm_120", "fp4_weights").configs == [{"FORMAT": "mxfp4"}, {"FORMAT": "nvfp4"}]
    assert decide("sm_75", "int8_weights").reason.startswith("needs sm_80+")
    odd = {"linear": {"sites": [{"m": 1, "n": 4096, "k": 1000, "dtype": "bfloat16"}]}}
    assert decide("sm_86", "int8_weights", odd).reason == (
        "no call site it takes: linear N=4096 K=1000 (needs multiples of 32)"
    )
    # its libraries key only the scouts of targets at its precisions
    a10 = GPUS["sm_86"]
    versions = {"torch": "2.10.0", "nvidia-cublas-cu12": "12.8", "gemlite": "0.6.0"}
    versions["triton"] = "3.6.0"
    exact = libscout.key(["linear"], a10, versions=versions)
    assert set(exact["libraries"]) == {"torch", "nvidia-cublas-cu12"} and "precision" not in exact
    low = libscout.key(["linear"], a10, versions=versions, precision="int8_weights")
    assert low["libraries"]["gemlite"] == "0.6.0" and low["precision"] == "int8_weights"
    assert libscout.stale({"key": low}, a10, versions={**versions, "triton": "3.7.0"}) == (
        "triton 3.6.0 → 3.7.0"
    )
    assert libscout.stale({"key": exact}, a10, versions={**versions, "triton": "3.7.0"}) is None


def test_low_bit_weights_are_quantised_once_per_reference(tmp_path, monkeypatch):
    """Dynamo calls ``rewrite`` again on every recompile, inside a CUDA graph capture too
    (an evaluator's copy of an input made in inference mode fails a guard there), where
    quantising cannot run: the weights of one reference are quantised once."""
    helper = _module(
        "gemlite.helper", **{n: object for n in ("A16W4_MXFP", "A16W4_NVFP", "A16W8_FP8")}
    )
    helper.A16W8_INT8 = object  # type: ignore[attr-defined]
    gemlite = _module("gemlite", helper=helper)
    monkeypatch.setitem(sys.modules, "gemlite", gemlite)
    monkeypatch.setitem(sys.modules, "gemlite.helper", helper)
    lin = torch.nn.Linear(64, 64)
    for name, configs in (("gemlite", ("int8", "mxfp4")), ("torch-scaled-mm", (True, False))):
        module = _load(_render(name), tmp_path / f"{name}.py")
        tables = getattr(module, "_layers" if name == "gemlite" else "_weights")
        first = [tables(lin, c) for c in configs]
        again = [tables(lin, c) for c in configs]  # the same tables, not new ones
        assert all(a is b for a, b in zip(first, again, strict=True))
        assert tables(torch.nn.Linear(64, 64), configs[0]) is not first[0]  # another reference


def test_the_guard_free_variant_calls_one_traced_graph_per_signature(tmp_path, monkeypatch):
    """``GUARDS=0``: each call signature's graph traced once and called directly; the
    guarded callable (Dynamo's guards) never runs, and the results are the reference's."""
    torch.manual_seed(0)
    block = Block().eval()
    path = tmp_path / "g.py"
    module = _load(_render("torch-rms-norm", template.guard_free([{"FUSE": 0}])), path)
    source = path.read_text()
    assert "def build(reference, FUSE=0, GUARDS=1):" in source
    assert "LibraryScout(reference, {'FUSE': FUSE}, guards=bool(GUARDS))" in source
    assert "FUSE=0, GUARDS=1; FUSE=0, GUARDS=0" in source  # the configs swept
    traced: list[Any] = []
    export = module.export_graph
    monkeypatch.setattr(
        module, "export_graph", lambda *a: traced.append(a[1][0].shape) or export(*a)
    )
    candidate = module.build(block, FUSE=0, GUARDS=0)
    guarded: list[int] = []
    compiled = candidate._compiled["forward"]
    candidate._compiled["forward"] = lambda *a, **k: guarded.append(1) or compiled(*a, **k)
    small, large = (torch.randn(2, 5, 32), *rope()), (torch.randn(3, 5, 32), *rope())
    with torch.inference_mode():
        for args in (small, large, small, large, small):
            torch.testing.assert_close(candidate(*args), block(*args), rtol=1e-5, atol=1e-5)
    assert traced == [torch.Size([2, 5, 32]), torch.Size([3, 5, 32])]  # once per signature
    assert guarded == [] and candidate.guarded == {}
    assert len(candidate.graphs) == 2 and candidate.rewritten == 4  # two norms, two graphs
    with torch.inference_mode():  # the key holds the grad and inference modes too
        key = fx_rewrites.call_key("forward", small, {})
    assert key is not None and key in candidate.graphs
    assert fx_rewrites.call_key("forward", small, {}) not in candidate.graphs
    assert fx_rewrites.call_key("forward", (small[0], libscout_toy.Box(small[1])), {}) is None
    with torch.inference_mode():  # GUARDS=1: Dynamo's compiled callable on every call
        guarded_candidate = module.build(block, FUSE=0)
        torch.testing.assert_close(guarded_candidate(*small), block(*small), rtol=1e-5, atol=1e-5)
    assert not guarded_candidate.graphs and guarded_candidate.rewritten == 2


def test_the_guard_free_variant_runs_guarded_where_it_must(tmp_path):
    """An object argument (a cache) or a graph break: the guarded graphs, and why; the
    probe then sweeps no guard-free variant, and says why."""
    torch.manual_seed(0)
    module = _load(_render("torch-rms-norm", template.guard_free([{"FUSE": 0}])), tmp_path / "g.py")
    x, box = torch.randn(3, 16), libscout_toy.Box(torch.full((16,), 2.0))
    boxed, branchy = Boxed(16).eval(), Branchy(16).eval()
    with torch.inference_mode():
        candidate = module.build(boxed, FUSE=0, GUARDS=0)
        torch.testing.assert_close(candidate(x, box), boxed(x, box), rtol=1e-5, atol=1e-5)
        assert candidate.guarded == {None: template.NOT_PLAIN} and not candidate.graphs
        candidate = module.build(branchy, FUSE=0, GUARDS=0)
        for sign in (1.0, -1.0, 1.0):
            y = sign * x.abs()
            torch.testing.assert_close(candidate(y), branchy(y), rtol=1e-5, atol=1e-5)
    (why,) = set(candidate.guarded.values())
    assert why.startswith("Dynamo does not trace the call as one graph (")
    assert candidate.rewritten > 0 and not candidate.graphs  # the guarded graphs fold it

    def verdict(reference: torch.nn.Module, *args: Any) -> str | None:
        with torch.inference_mode():
            expected = reference(*args)
        case = {"args": args, "kwargs": {}}
        return probe.guard_free(module, reference, case, {"FUSE": 0}, expected)

    assert verdict(Block().eval(), torch.randn(2, 5, 32), *rope()) is None
    assert verdict(boxed, x, box) == (f"its guard-free variant runs guarded: {template.NOT_PLAIN}")
    assert str(verdict(branchy, x)).startswith(
        "its guard-free variant runs guarded: Dynamo does not trace the call as one graph"
    )


def test_the_probe_sweeps_each_config_with_and_without_guards_where_it_may(tmp_path, monkeypatch):
    """On CPU captures (no op bars): every config twice, with and without Dynamo's guards;
    only the guarded ones for a graph-timed target or a module with state."""
    from kernel_agent.profiling.capture import capture_calls

    _fake_libraries(monkeypatch, {})
    torch.manual_seed(0)
    capture_calls(Block().eval(), [((torch.randn(2, 5, 32), *rope()), {}, 10)], tmp_path / "b.pt")
    counted = tmp_path / "c.pt"
    capture_calls(Counted(16).eval(), [((torch.randn(3, 16),), {}, 4)], counted)

    def norm(capture: Path, out: str, context: str | None = None) -> dict[str, Any]:
        info = probe.probe(capture, out_dir=tmp_path / out, target="t", context=context)
        (row,) = [d for d in info["decisions"] if d["adapter"] == "torch-rms-norm"]
        assert row["run"], row
        return {**row, "source": Path(info["candidates"]["torch-rms-norm"]).read_text()}

    both = norm(tmp_path / "b.pt", "eager")
    assert both["guard_free"] is True
    assert both["configs"] == [
        {"FUSE": 0, "GUARDS": 1},
        {"FUSE": 0, "GUARDS": 0},
        {"FUSE": 1, "GUARDS": 1},
        {"FUSE": 1, "GUARDS": 0},
    ]
    assert "def build(reference, FUSE=0, GUARDS=1):" in both["source"]
    graph = norm(tmp_path / "b.pt", "graph", context="graph")
    assert graph["guard_free"] == probe.GRAPH_TIMED
    assert graph["configs"] == [{"FUSE": 0}, {"FUSE": 1}]
    stateful = norm(counted, "state")
    assert stateful["guard_free"] == probe.STATEFUL and stateful["configs"] == [
        {"FUSE": 0},
        {"FUSE": 1},
    ]


def test_the_scout_times_its_candidates_in_the_targets_context(tmp_path, monkeypatch):
    """A graph-launched target's scout candidates are timed in a CUDA graph, as an agent's:
    the probe hears the context (no guard-free variant there), every sweep gets it."""
    from kernel_agent.kernels import context as timing_context

    monkeypatch.setenv(gpulock.ENV, "1")
    run, keeper = _scouted_run(tmp_path)
    seen: dict[str, Any] = {"sweeps": []}

    def prober(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen["probe"] = kwargs.get("context")
        info = _prober(*args, **kwargs)
        info["decisions"][0]["guard_free"] = probe.GRAPH_TIMED
        return info

    def sweeper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen["sweeps"].append({k: kwargs.get(k) for k in ("context", "l2_flush")})
        return _sweeper(*args, **kwargs)

    graph = timing_context.TimingContext("graph", "cold", "graph: 98% of the stage's events")
    monkeypatch.setattr(timing_context, "for_target", lambda run_, target_id: graph)
    found = libscout.scout_target(
        run, "mix", keeper=keeper, prober=prober, sweeper=sweeper, say=lambda m: None
    )
    assert seen == {"probe": "graph", "sweeps": [{"context": "graph", "l2_flush": True}] * 2}
    assert found["timing_context"] == {
        "context": "graph",
        "l2": "cold",
        "why": "graph: 98% of the stage's events",
    }
    assert found["precision"] == "exact"
    lines = libscout.bar_lines(run, "mix")
    assert lines[3].startswith("* Measured (module level, graph): torch-sdpa")
    assert lines[4] == f"* Swept with Dynamo's guards only: torch-sdpa ({probe.GRAPH_TIMED})"


class FakeLt:
    """The cuBLASLt extension's functions (``lt_linear``, ``lt_plans``) on the CPU: torch's
    linear, and a plan per shape (the timed pick: the heuristic's 4th of 8, 4.5 us)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.plans: dict[tuple[int, ...], list[float]] = {}

    def lt_linear(self, x: Any, w: Any, bias: Any, residual: Any, gelu: int, algo: int) -> Any:
        self.calls.append((tuple(x.shape), tuple(w.shape), algo))
        epilogue = 2 * (bias is not None) + int(gelu)
        key = (x.shape[0], w.shape[0], w.shape[1], epilogue, int(residual is not None), algo)
        self.plans.setdefault(key, [3.0, 8.0, 4.5] if algo < 0 else [float(algo), 8.0, -1.0])
        out = torch.nn.functional.linear(x, w, bias)
        if gelu:
            out = torch.nn.functional.gelu(out, approximate="tanh")
        return out if residual is None else out + residual

    def lt_plans(self) -> list[list[float]]:
        return [[*map(float, k), *v] for k, v in self.plans.items()]


def test_cublaslt_gets_op_bars_from_its_extension_built_in_the_probe(tmp_path, monkeypatch):
    """The extension is built by the candidate's prepare (the probe's dry run: once), its
    per-shape choice timed against F.linear on the recorded GEMMs, with the algorithm in the
    bar; a config that folds an epilogue the bar leaves out is not pruned for speed."""
    configs = [dict(c) for c in adapters.CublasLt.CONFIGS]
    path = tmp_path / "libscout_cublaslt.py"
    path.write_text(_render("cublaslt", configs))
    module = probe._import(path)
    fake, built = FakeLt(), []

    def builder(**kwargs: Any) -> FakeLt:
        built.append(kwargs)
        return fake

    monkeypatch.setattr(module, "load_inline", builder)
    with monkeypatch.context() as gpu:
        gpu.setattr(torch.cuda, "is_available", lambda: True)
        module.prepare(None)
        module.prepare(None)
    (build,) = built  # built once
    assert build["functions"] == ["lt_linear", "lt_plans"]
    assert build["extra_ldflags"] == ["-lcublasLt"]
    assert build["name"].startswith("ka_libscout_cublaslt_")
    monkeypatch.setattr(module, "_lt_takes", lambda *a: True)  # CPU tensors, the fake
    torch.manual_seed(0)
    lin = Residual().eval()
    invocations: list[Any] = []
    detect.trace(lin, (torch.randn(4, 16),), module=lin, invocations=invocations)

    def timer(fns: Any) -> dict[str, list[float] | None]:  # the GEMM 0.83x torch's
        return {"eager": [10.0, 12.0], "graph": [5.0, 6.0]}

    bars = probe.op_bars(lin, invocations, {"cublaslt": (module, configs)}, timer=timer)
    assert [(b["config"], b["calls"], b["ok"]) for b in bars] == [(c, 2, True) for c in configs]
    assert bars[0]["signature"] == "linear([4, 16] [16, 16] [16] float32)"
    assert [b.get("folds") for b in bars] == [
        "a residual add or a tanh GELU into the epilogue",
        None,
        None,
    ]
    assert {k: bars[0][k] for k in ("algorithm", "algorithms", "algorithm_us")} == {
        "algorithm": 3,
        "algorithms": 8,
        "algorithm_us": 4.5,
    }
    assert bars[2]["algorithm"] == 0 and "algorithm_us" not in bars[2]
    assert fake.calls and {c[2] for c in fake.calls} == {-1, 0}
    keep, pruned = probe.prune(configs, bars)
    assert keep == configs[:1]  # its epilogue folds a kernel the GEMM's bar does not count
    assert [p["config"] for p in pruned] == configs[1:]
    assert all(p["why"] == "op bar at most 0.83x" for p in pruned)


# ------------------------------------------------------------------ on the GPU


def _gpu_capture(path: Path, dim: int = 64) -> Path:
    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(0)
    core = Core().cuda().eval()

    def qkv(seq: int) -> tuple[torch.Tensor, ...]:
        def t(heads: int) -> torch.Tensor:
            return torch.randn(16, heads, seq, dim, device="cuda", dtype=torch.bfloat16)

        return (t(16), t(2), t(2))

    capture_calls(core, [(qkv(11), {}, 540), (qkv(7), {}, 60)], path)
    return path


@pytest.mark.gpu
def test_probe_measures_sdpa_backends_and_the_scout_evaluates_them(tmp_path, monkeypatch):
    """The SDPA core of a GQA attention at 11 tokens: op bars of every backend that runs
    here, and the scout's candidate fully evaluated (library kernels: not a fallback)."""
    monkeypatch.setenv(gpulock.ENV, "1")  # conftest holds this process's GPU lock
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])
    )
    capture = _gpu_capture(tmp_path / "core.pt")
    info = probe.run_probe(capture, out_dir=tmp_path / "out", target="core", timeout=600)
    assert "error" not in info, info.get("error")
    assert info["described"] == "sdpa x1"
    sdpa = [b for b in info["op_bars"] if b["adapter"] == "torch-sdpa"]  # flash-attn: its own
    every = {b["config"]["BACKEND"]: b for b in sdpa}
    assert set(every) >= {"efficient", "math"} and every["math"]["ok"]
    # a backend without a kernel for the call (mem-efficient on an A10, sm_86) fails its bar
    assert all(b["ok"] or "No available kernel" in str(b.get("error")) for b in every.values())
    bars = {name: b for name, b in every.items() if b["ok"]}
    assert all(b["ref_us"] > 0 and b["us"] > 0 for b in bars.values())
    (decision,) = [d for d in info["decisions"] if d["adapter"] == "torch-sdpa"]
    fastest = min(bars.values(), key=lambda b: b["us"])["config"]
    data = sweep.run_sweep(capture, Path(info["candidates"]["torch-sdpa"]), [fastest])
    evaluation = data["evaluation"]
    if fastest["BACKEND"] != "flash":  # flash is what torch picks here: a fallback, rightly
        assert evaluation["correct"], evaluation.get("error")
        assert evaluation["custom_kernel_share"] > 0
    assert decision["configs"]


@pytest.mark.gpu
def test_probe_measures_the_written_out_rms_norm_against_the_fused_kernel(tmp_path, monkeypatch):
    """A written-out RMSNorm (bf16, the cast to fp32 and back): its recorded calls replayed
    against ``F.rms_norm`` with the reference's roundings and with the weight inside."""
    from kernel_agent.profiling.capture import capture_calls

    monkeypatch.setenv(gpulock.ENV, "1")  # conftest holds this process's GPU lock
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])
    )
    torch.manual_seed(0)
    norm = Scale(1024).cuda().bfloat16().eval()
    rows = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    capture_calls(norm, [((rows,), {}, 100)], tmp_path / "norm.pt")
    info = probe.run_probe(tmp_path / "norm.pt", out_dir=tmp_path / "out", target="norm")
    assert "error" not in info, info.get("error")
    bars = [b for b in info["op_bars"] if b.get("op") == probe.RMS_PATTERN]
    mine = [b for b in bars if b["adapter"] == "torch-rms-norm"]  # libraries' too, installed
    assert sorted(b["config"]["FUSE"] for b in mine) == [0, 1]
    assert all(b["ok"] and b["ref_us"] > 0 and b["us"] > 0 for b in bars), bars
    assert all(b["pattern_ops"] == 8 for b in bars)
    # each installed library's RMSNorm: its op bar on the same pattern (FlashInfer, Liger)
    for adapter in ("flashinfer-norm", "liger-rmsnorm"):
        if info["available"][adapter]["reason"] is None:
            assert [b["adapter"] for b in bars].count(adapter) == 1, adapter


def _library(name: str) -> None:
    """Skip unless the adapter's library is installed and usable here."""
    found = adapters.BY_NAME[name].probe()
    if not found.ok:
        pytest.skip(str(found.reason))


@pytest.mark.gpu
def test_flashinfer_attention_on_the_gpu(tmp_path, monkeypatch):
    """FlashInfer installed: the template's op on one request and on batches (decode and
    prefill, K / V as a cache and as projections viewed as heads) against SDPA, then the
    probe on a batched prefill capture (GQA, 11 tokens): the adapter runs, its op bar is
    correct (measured on an NVIDIA A10, sm_86, FlashInfer 0.6.17: 21.6 vs 8.5 us at batch
    32; README)."""
    _library("flashinfer-attention")
    toolchain.setup()  # FlashInfer's JIT builds with the toolchain's nvcc flags
    configs = [{"TENSOR_CORES": 0}, {"TENSOR_CORES": 1}]
    module = _load(_render("flashinfer-attention", configs), tmp_path / "fi.py")
    sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.manual_seed(0)
    for cores in (0, 1):
        attend = module.ops(None, TENSOR_CORES=cores)[sdpa]
        for b, h, hkv, sq, skv, causal, projected in (
            (1, 16, 8, 1, 300, False, False),
            (4, 16, 2, 1, 300, False, True),
            (4, 16, 2, 11, 11, False, True),
            (2, 8, 8, 64, 64, True, False),
        ):

            def heads(n: int, seq: int, b: int = b, projected: bool = projected) -> torch.Tensor:
                shape = (b, seq, n, 128) if projected else (b, n, seq, 128)
                t = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
                return t.transpose(1, 2) if projected else t

            q, k, v = heads(h, sq), heads(hkv, skv), heads(hkv, skv)
            with torch.inference_mode():
                want = sdpa(q, k, v, is_causal=causal, enable_gqa=True)
                got = attend(q, k, v, is_causal=causal, enable_gqa=True)
            torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
    monkeypatch.setenv(gpulock.ENV, "1")  # conftest holds this process's GPU lock
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])
    )
    capture = _gpu_capture(tmp_path / "core.pt", dim=128)
    info = probe.run_probe(capture, out_dir=tmp_path / "out", target="core", timeout=1200)
    assert "error" not in info, info.get("error")
    (decision,) = [d for d in info["decisions"] if d["adapter"] == "flashinfer-attention"]
    assert decision["run"], decision
    bars = [b for b in info["op_bars"] if b["adapter"] == "flashinfer-attention"]
    assert bars and all(b["ok"] and b["us"] > 0 for b in bars), bars


@pytest.mark.gpu
def test_liger_swiglu_on_the_gpu(tmp_path):
    """Liger-Kernel installed: its SwiGLU folded into a gated MLP's Dynamo graph gives the
    reference's output."""
    _library("liger-swiglu")
    torch.manual_seed(0)
    feed = Feed(256, 512).cuda().bfloat16().eval()
    x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16)
    module = _load(_render("liger-swiglu"), tmp_path / "swiglu.py")
    candidate = module.build(feed)
    with torch.inference_mode():
        want, got = feed(x), candidate(x)
    assert candidate.rewritten == 1
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
def test_the_guard_free_variant_runs_the_library_on_the_gpu(tmp_path):
    """The SDPA core pinned to cuDNN: ``GUARDS=0`` computes what ``GUARDS=1`` does from one
    traced graph, and is timed in a CUDA graph with the evaluator's copy of an input (made in
    inference mode) without tracing again."""
    from kernel_agent.kernels import bench

    torch.manual_seed(0)
    core = Core().cuda().eval()
    q = torch.randn(16, 16, 11, 64, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(16, 2, 11, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    configs = template.guard_free([{"BACKEND": "cudnn"}])
    module = _load(_render("torch-sdpa", configs), tmp_path / "sdpa.py")
    with torch.inference_mode():
        free = module.build(core, BACKEND="cudnn", GUARDS=0)
        guarded = module.build(core, BACKEND="cudnn", GUARDS=1)
        torch.testing.assert_close(free(q, k, v), guarded(q, k, v))
    assert len(free.graphs) == 1 and free.guarded == {} and free.rewritten == 1
    timing = bench.time_call(free, (q, k, v), {}, context="graph", target_ms=5.0, keep=True)
    assert timing["median_ms"] > 0 and len(free.graphs) == 1 and free.guarded == {}


@pytest.mark.gpu
@pytest.mark.skipif(importlib.util.find_spec("liger_kernel") is None, reason="no liger-kernel")
def test_liger_rope_rotates_q_and_k_in_one_kernel_on_the_gpu(tmp_path, monkeypatch):
    torch.manual_seed(0)
    for keep in (False, True):  # q read again afterwards: Liger writes into a copy of it
        rotary = Rotary(hidden=256, heads=4, kv_heads=2, keep=keep).cuda().bfloat16().eval()
        x = torch.randn(2, 11, 256, device="cuda", dtype=torch.bfloat16)
        cos, sin = (t.cuda().bfloat16() for t in cos_sin(2, 11, 64))
        module = _load(_render("liger-rope"), tmp_path / f"rope_{keep}.py")
        calls: list[int] = []

        def counted(*a: Any, calls: list[int] = calls, liger: Any = module.liger_rope) -> Any:
            calls.append(1)
            return liger(*a)

        monkeypatch.setattr(module, "liger_rope", counted)
        with torch.inference_mode():
            want = rotary(x, cos, sin)
            candidate = module.build(rotary)
            got = candidate(x, cos, sin)
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
        assert candidate.rewritten == 1 and calls == [1]


@pytest.mark.gpu
@pytest.mark.skipif(importlib.util.find_spec("gemlite") is None, reason="no gemlite")
def test_gemlite_int8_weights_on_the_gpu(tmp_path):
    torch.manual_seed(0)
    lin = torch.nn.Sequential(torch.nn.Linear(1024, 2048, bias=False)).cuda().bfloat16().eval()
    with torch.no_grad():
        lin[0].weight.normal_(0, 0.02)
    x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    module = _load(_render("gemlite", [{"FORMAT": "int8"}]), tmp_path / "gemlite.py")
    with torch.inference_mode():
        want = lin(x).float()
        candidate = module.build(lin, FORMAT="int8")
        got = candidate(x).float()
    assert candidate.rewritten == 1
    assert ((got - want).norm() / want.norm()).item() < 0.02  # per-channel INT8 weights


@pytest.mark.gpu
def test_probe_times_cublaslts_choice_per_gemm_shape(tmp_path, monkeypatch):
    """cuBLASLt's extension built in the probe; its op bars on a skinny bf16 GEMM say which
    of the heuristic's algorithms it runs."""
    from kernel_agent.profiling.capture import capture_calls

    monkeypatch.setenv(gpulock.ENV, "1")  # conftest holds this process's GPU lock
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])
    )
    torch.manual_seed(0)
    lin = torch.nn.Linear(1024, 2048, bias=False).cuda().bfloat16().eval()
    rows = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    capture_calls(lin, [((rows,), {}, 10)], tmp_path / "lin.pt")
    info = probe.run_probe(
        tmp_path / "lin.pt", out_dir=tmp_path / "out", target="lin", backends={"cuda": True}
    )
    assert "error" not in info, info.get("error")
    bars = [b for b in info["op_bars"] if b["adapter"] == "cublaslt"]
    assert len(bars) == 3 and all(b["ok"] and b["us"] > 0 for b in bars), bars
    assert all(0 <= b["algorithm"] < b["algorithms"] for b in bars)
    timed = [b for b in bars if b["config"]["ALGO"] == -1]
    assert all(b["algorithm_us"] > 0 for b in timed)
