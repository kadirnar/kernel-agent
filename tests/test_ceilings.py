"""Ceilings table (#90): the profiler's per-call work and the floors per precision; on an
optimised model (#106), the work of the calls the hooks cannot see into."""

from __future__ import annotations

import functools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kernel_agent import toolchain, worker
from kernel_agent.agent import prompts
from kernel_agent.hub import Modality
from kernel_agent.kernels import roofline
from kernel_agent.profiling import ceilings, profiler
from kernel_agent.profiling.profiler import ModuleTimer
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec
from kernel_agent.workspace import read_json, write_json

# The RTX 5070 Ti's peaks (measured 2026-10-06; FP8 / NVFP4 via torch scaled_mm).
PEAKS = {
    "version": 2,
    "dram_gbps": 767.1,
    "launch_floor_us": 15.78,
    "tflops": {"bfloat16": 99.4, "float32": 34.4, "float8_e4m3fn": 332.6},
    "tflops_unavailable": {"float4_e2m1fn_x2": "RuntimeError: no NVFP4 kernel"},
}


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(16, 64, bias=False)
        self.down = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(16)
        self.mlp = Mlp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class Fused(nn.Module):
    """Holds a Linear but never calls it (a replaced kernel): estimated from its weights."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.proj.weight)


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])
        self.fused = Fused()
        self.conv = nn.Conv1d(4, 8, 3)
        self.deconv = nn.ConvTranspose1d(8, 4, 3)

    def forward(self, x: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        self.fused(x)
        self.deconv(self.conv(audio))
        return x


def _work(stats, cls):
    (stat,) = [s for s in stats if s.cls == cls]
    return {(w["group"], w["phase"]): w for w in stat.work}


def test_profiler_records_linear_and_conv_work_per_group_and_phase():
    torch.manual_seed(0)
    model = Toy()
    x, audio = torch.randn(2, 5, 16), torch.randn(2, 4, 9)
    with torch.inference_mode(), ModuleTimer({"m": model}, {}, cuda=False) as timer:
        model(x, audio)
        model(x[:, :1], audio)  # one position: the decode phase
    stats = timer.class_stats()

    mlp_elems = 16 * 64 + 64 * 16 + 16  # up (no bias) + down + its bias
    layer = _work(stats, "Layer")[("m.layers.*", "prefill")]
    assert layer["calls"] == 2 and layer["instances"] == 2
    assert layer["flops"] == {"float32": 2 * 2 * 10 * (16 * 64 + 64 * 16)}  # 2 calls x 10 rows
    assert layer["weight_elems"] == 2 * mlp_elems and layer["weight_bytes"] == 4 * 2 * mlp_elems
    assert layer["io_bytes"] == 2 * 2 * (2 * 5 * 16 * 4)  # input + output per call
    assert layer["signature"] == "a0[2, 5, 16]:float32" and "estimated_calls" not in layer
    decode = _work(stats, "Layer")[("m.layers.*", "decode")]
    assert decode["flops"] == {"float32": 2 * 2 * 2 * (16 * 64 + 64 * 16)}  # 2 rows per call

    linear = _work(stats, "Linear")
    up = linear[("m.layers.*.mlp.up", "prefill")]
    assert up["calls"] == 2 and up["weight_elems"] == 2 * 16 * 64
    assert linear[("m.layers.*.mlp.down", "prefill")]["weight_elems"] == 2 * (64 * 16 + 16)

    # convolutions: every output (conv) / input (transposed) element x C/groups x kernel
    conv = _work(stats, "Conv1d")[("m.conv", "prefill")]
    assert conv["flops"] == {"float32": 2 * 2 * (2 * 8 * 7) * (4 * 3)}  # two runs
    deconv = _work(stats, "ConvTranspose1d")[("m.deconv", "prefill")]
    assert deconv["flops"] == {"float32": 2 * 2 * (2 * 8 * 7) * (4 * 3)}
    assert deconv["weight_elems"] == 2 * (8 * 4 * 3 + 4)

    # a module that never calls its Linear: estimated from the weights at its input's rows
    fused = _work(stats, "Fused")[("m.fused", "prefill")]
    assert fused["flops"] == {"float32": 2 * 10 * 16 * 32} and fused["estimated_calls"] == 1
    toy = _work(stats, "Toy")[("m", "prefill")]  # the root sums everything inside it once
    assert toy["calls"] == 1 and "estimated_calls" in toy


def _profile() -> dict:
    """A VoxCPM2-like batch-16 profile: the LocDiT (inside the CFM solver) at 352 rows and
    the batched LM decode (Linear / MLP at 16 rows, 60 steps)."""
    dit_params, steps = 212_000_000, 540
    lm_attn, lm_mlp = 2048 * (2048 + 256 + 256) + 2048 * 2048, 3 * 2048 * 6144

    def work(group, calls, rows, params, ms, phase="prefill", io=0):
        return {
            "group": group,
            "phase": phase,
            "instances": 1,
            "calls": calls,
            "inclusive_ms": ms,
            "flops": {"bfloat16": 2 * rows * params * calls},
            "weight_elems": params * calls,
            "weight_bytes": 2 * params * calls,
            "io_bytes": io,
            "signature": f"a0[{rows}, 1024]:bfloat16",
        }

    dit = work("model.feat_decoder.estimator", steps, 352, dit_params, 7000.0, io=10**9)
    cfm = dict(work("model.feat_decoder", 60, 352, dit_params, 7100.0), calls=60)
    cfm["flops"], cfm["weight_elems"] = dit["flops"], dit["weight_elems"]  # the same work
    cfm["weight_bytes"] = dit["weight_bytes"]
    linears = [
        dict(work(f"model.base_lm.layers.*.self_attn.{name}", 28 * 60, 16, lm_attn // 4, 80.0))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    return {
        "hooked_wall_ms": 10_000.0,
        "classes": [
            {"cls": "UnifiedCFM", "is_leaf": False, "work": [cfm]},
            {"cls": "VoxCPMLocDiT", "is_leaf": False, "work": [dit]},
            {"cls": "Linear", "is_leaf": True, "work": linears},
            {
                "cls": "MiniCPMMLP",
                "is_leaf": False,
                "work": [work("model.base_lm.layers.*.mlp", 28 * 60, 16, lm_mlp, 600.0)],
            },
            {"cls": "RMSNorm", "is_leaf": True, "work": [work("model.norm", 20_000, 16, 0, 50.0)]},
        ],
    }


def test_ceilings_reproduce_the_hand_numbers():
    table = ceilings.build(_profile(), PEAKS, 5000.0, per="per batched run")
    rows = {r["target"]: r for r in table["rows"]}
    dit = rows["VoxCPMLocDiT@model.feat_decoder.estimator"]
    # 2 x 212 M x 352 rows x 540 calls = 80.6 TFLOP = 811 ms at 99.4 TFLOP/s: compute bound
    assert dit["flops_total"] == pytest.approx(80.6e12, rel=1e-3) and dit["m"] == 352
    assert dit["floors"]["exact"] == pytest.approx(811, rel=1e-3) and dit["bound"] == "compute"
    assert dit["floors"]["fp8_weights"] == dit["floors"]["exact"]  # bf16 math: no gain
    assert dit["floors"]["w8a8"] == pytest.approx(80.6e12 / 332.6e12 * 1e3, rel=1e-3)
    assert dit["floors"]["w4a4"] is None  # no NVFP4 peak: unknown, never a ratio
    assert dit["now_ms"] == pytest.approx(3500.0) and dit["share"] == pytest.approx(0.7)
    assert dit["saves_ms"]["exact"] == pytest.approx(3500 - 811, rel=1e-3)

    # The LM decode at 16 rows: the MLP streams its weights (memory bound); q/k/v/o share
    # one row, launch bound (four GEMV launches per layer cost more than their bytes).
    attn = rows["Linear@model.base_lm.layers.*.self_attn.{k_proj,o_proj,q_proj,v_proj}"]
    assert attn["instances"] == 4 and attn["calls"] == 4 * 28 * 60 and attn["bound"] == "launch"
    mlp = rows["MiniCPMMLP@model.base_lm.layers.*.mlp"]
    assert mlp["bound"] == "memory" and mlp["m"] == 16
    per_step = (attn["weight_bytes"] + mlp["weight_bytes"]) / 60
    assert per_step == pytest.approx(2 * 28 * (2048 * 2560 + 2048**2 + 3 * 2048 * 6144))
    assert mlp["floors"]["exact"] == pytest.approx(mlp["weight_bytes"] / 767.1e6, rel=1e-3)
    assert mlp["floors"]["fp8_weights"] == pytest.approx(mlp["floors"]["exact"] / 2, rel=1e-3)
    assert mlp["floors"]["fp4_weights"] == pytest.approx(mlp["floors"]["exact"] * 4.5 / 16, 1e-3)
    assert rows["RMSNorm@model.norm"]["bound"] == "launch"  # 20k tiny calls

    # End to end: the CFM holds the LocDiT, so the LocDiT's saving counts once.
    e2e = table["e2e"]
    assert table["e2e"]["w4a4"] is None
    saved = sum(r["saves_ms"]["exact"] for t, r in rows.items() if not t.startswith("VoxCPMLocDiT"))
    assert e2e["exact"]["floor_ms"] == pytest.approx(5000 - saved, rel=1e-3)
    assert e2e["exact"]["counted"][0] == "UnifiedCFM@model.feat_decoder"
    assert e2e["w8a8"]["floor_ms"] < e2e["fp8_weights"]["floor_ms"] < e2e["exact"]["floor_ms"]
    json.dumps(table)

    text = ceilings.markdown(table)
    assert "## Ceilings" in text and "| `VoxCPMLocDiT` `model.feat_decoder.estimator` |" in text
    assert "compute bound near M ≈ 130" in text and "W4A4 at NVFP4 unknown" in text
    assert "W4A4 unknown (RuntimeError: no NVFP4 kernel)" in text
    first = next(line for line in text.splitlines() if line.startswith("| `"))
    assert first.startswith("| `UnifiedCFM`")  # ranked by what reaching the floor saves


def test_a_row_below_its_exact_floor_ranks_by_its_best_lower_precision_saving():
    profile = _profile()
    for c in profile["classes"][:2]:  # the CFM and its LocDiT, already at W8A8: 600 ms now
        c["work"][0]["inclusive_ms"] = 1200.0
    table = ceilings.build(profile, PEAKS, 5000.0)
    first = table["rows"][0]
    assert first["target"] == "UnifiedCFM@model.feat_decoder" and first["now_ms"] == 600
    assert first["floors"]["exact"] > 600 and first["saves_ms"]["exact"] == 0
    w8a8 = 600 - 80.6e12 / 332.6e12 * 1e3
    assert first["saves_best"] == {"precision": "w8a8", "ms": pytest.approx(w8a8, rel=1e-3)}
    mlp = next(r for r in table["rows"] if r["cls"] == "MiniCPMMLP")
    assert 0 < mlp["saves_ms"]["exact"] < w8a8 and "saves_best" not in mlp
    assert "| 0 (W8A8 358) |" in ceilings.markdown(table)


def test_ceilings_without_peaks_or_work(tmp_path):
    table = ceilings.build(_profile(), None, 5000.0)
    assert all(v is None for r in table["rows"] for v in r["floors"].values())
    assert table["e2e"] == {} and "peaks not measured" in ceilings.markdown(table)
    old = {"hooked_wall_ms": 1.0, "classes": [{"cls": "A", "is_leaf": True, "params": 3}]}
    assert ceilings.markdown(ceilings.build(old, PEAKS, 1.0)) == ""  # a profile made before #90
    table = ceilings.write(tmp_path, {"classes": [{"work": [{}]}]}, PEAKS, 1.0)  # malformed
    assert "error" in table and (tmp_path / "ceilings.json").exists()
    assert (tmp_path / "ceilings.md").read_text() == ""
    ceilings.write(tmp_path, _profile(), PEAKS, 5000.0)
    assert (tmp_path / "ceilings.md").read_text().startswith("## Ceilings")
    assert json.loads((tmp_path / "ceilings.json").read_text())["rows"]


def test_planner_prompt_ranks_by_ceiling():
    card = {"repo_id": "org/m", "modality": "text"}
    plan = prompts.planner_prompt(card, {"median_ms": 1.0}, "# Profile", ["triton"], 3, "py", "tc")
    assert "*Ceilings* table" in plan and "ceiling × share" in plan


def test_peaks_cache_upgrade_and_low_precision_peaks(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("KERNEL_AGENT_LOCK_HELD", "1")
    monkeypatch.setattr(roofline, "_MEASURE_FAILED", False)
    gpu = toolchain.GPUInfo("Fake GPU", (12, 0), 16.0, 70, 48.0)
    tc = toolchain.Toolchain(
        gpu=gpu, torch_version="2.0", torch_cuda=None, cuda_home=None, nvcc_version=None
    )
    monkeypatch.setattr(toolchain, "setup", lambda: tc)
    path = toolchain.peaks_path("Fake GPU", "2.0")
    old = {"dram_gbps": 500.0, "tflops": {"bfloat16": 90.0}}  # before FP8 (no version)
    path.write_text(json.dumps(old))
    runs = []

    def failing_run(cmd, **kwargs):
        runs.append(cmd)
        raise OSError("GPU busy")

    monkeypatch.setattr(roofline.subprocess, "run", failing_run)
    assert roofline.ensure_peaks() == old and len(runs) == 1  # the old cache stays in use
    assert roofline.ensure_peaks() == old and len(runs) == 1  # not retried in this process

    monkeypatch.setattr(roofline, "_MEASURE_FAILED", False)
    new = {**old, "version": roofline.PEAKS_VERSION, "tflops": {**old["tflops"], "fp": 1.0}}

    def measuring_run(cmd, **kwargs):
        runs.append(cmd)
        path.write_text(json.dumps(new))
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(roofline.subprocess, "run", measuring_run)
    assert roofline.ensure_peaks() == new and len(runs) == 2  # measured once more
    assert roofline.ensure_peaks() == new and len(runs) == 2

    line = toolchain.format_peaks(
        {
            "dram_gbps": 767.1,
            "tflops": {"bfloat16": 99.4, roofline.FP8: 332.6, roofline.FP4: 640.7},
            "tflops_unavailable": {"x": "y"},
        }
    )
    assert line == "copy DRAM 767 GB/s, matmul bf16/fp8/nvfp4 99 / 333 / 641 TFLOP/s, no x matmul"


# ------------------------------------------------------------------ optimised models (#106)

D, STEPS, TOKENS = 64, 3, 22
ROWS = 2 * 4 * TOKENS  # the DiT's rows: CFG x 4 input rows x patch tokens = 176
# Peaks small enough that the toy's FLOPs and bytes, not launches, decide the bound.
TOY_PEAKS = {"dram_gbps": 1.0, "launch_floor_us": 0.0, "tflops": {"float32": 0.01}}


class Dit(nn.Module):
    """A LocDiT-like stack of Linears."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(D, D, bias=False) for _ in range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def dit_math(dit: Dit, x: torch.Tensor) -> torch.Tensor:
    """The DiT without module calls, as a captured CUDA graph or a fused kernel runs it."""
    for layer in dit.layers:
        x = torch.relu(F.linear(x, layer.weight))
    return x


class Solver(nn.Module):
    """A CFM-like solver: its input has 4 rows, its DiT runs STEPS times at ROWS rows.
    ``hide``: what an optimisation runs instead of the DiT's module calls."""

    def __init__(self) -> None:
        super().__init__()
        self.dit = Dit()
        self.hide: Any = None

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        x = mu[:, None].expand(-1, TOKENS, -1).repeat(2, 1, 1)
        for _ in range(STEPS):
            x = self.dit(x) if self.hide is None else self.hide(x)
        return x[: mu.shape[0], 0]


class Vae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, 8, 3)
        self.deconv = nn.ConvTranspose1d(8, 4, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.conv(z))


class Scope(nn.Module):
    """A transform's wrapper (VoxCPM's ``_TF32Scope``): the module moves to ``.inner``."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner(z)


class Tts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.solver = Solver()
        self.vae = Vae()

    def forward(self, mu: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.solver(mu), self.vae(z)


def _run(model: Tts, reference: Any = None, *, replay: bool = False) -> ModuleTimer:
    """One hooked run; ``replay``: the solver replays its DiT as a captured CUDA graph."""
    mu, z = torch.ones(4, D), torch.ones(2, 4, 9)
    with (
        torch.inference_mode(),
        ModuleTimer({"m": model}, {}, cuda=False, reference=reference) as timer,
    ):
        if replay:  # what torch's replay hook reports, then the graph's kernels

            def graph(x: torch.Tensor) -> torch.Tensor:
                timer._replay(None)
                return dit_math(model.solver.dit, x)

            model.solver.hide = graph
        model(mu, z)
    return timer


def _table(timer: ModuleTimer) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    profile = {"hooked_wall_ms": 1.0, "classes": [asdict(s) for s in timer.class_stats()]}
    table = ceilings.build(profile, TOY_PEAKS, 1.0)
    return {r["target"]: r for r in table["rows"]}, table


def test_a_graph_replayed_solver_takes_the_work_of_the_unmodified_model():
    torch.manual_seed(0)
    model = Tts()
    eager = _run(model)
    reference = eager.work_reference()  # before an optimisation hides the DiT
    rows, _ = _table(eager)
    flops = {"float32": 2 * ROWS * 2 * D * D * STEPS}
    solver = rows["Solver@m.solver"]
    assert solver["flops"] == flops and solver["m"] == ROWS and solver["bound"] == "compute"

    # The DiT is a CUDA graph replayed in the solver's call: the hooks see no DiT call. The
    # old estimate took the solver's Linear weights at its input's 4 rows: M = 4, memory.
    graphed = _run(model, reference, replay=True)
    assert "Dit" not in {c.cls for c in graphed.calls}
    rows, table = _table(graphed)
    row = rows["Solver@m.solver"]
    assert row["reference_calls"] == 1 and row["flops"] == flops
    assert row["weight_elems"] == solver["weight_elems"]
    assert row["m"] == ROWS and row["bound"] == "compute"
    assert rows["Tts@m"]["reference_calls"] == 1 and rows["Tts@m"]["work_known"]
    text = ceilings.markdown(table, min_share=0)
    assert "| `Solver` `m.solver` ‡ |" in text and "‡: the hooks did not see inside" in text

    # Without the unmodified model's work: unknown, never "M = 4, memory bound".
    rows, table = _table(_run(model, None, replay=True))
    row = rows["Solver@m.solver"]
    assert row["unknown_calls"] == 1 and not row["work_known"]
    assert row["flops_total"] is None and row["m"] is None and "bound" not in row
    assert all(v is None for v in [*row["floors"].values(), *row["saves_ms"].values()])
    assert sorted(table["unknown_work"]) == ["Solver@m.solver", "Tts@m"]  # the root holds it
    assert table["e2e"]["exact"]["counted"] == ["Vae@m.vae"]  # the solver keeps its time
    text = ceilings.markdown(table, min_share=0)
    cells = next(x for x in text.splitlines() if x.startswith("| `Solver`")).split(" | ")
    assert cells[4] == "?" and cells[7:10] == ["?", "?", "?"] and cells[10:15] == ["?"] * 5
    assert "the 2 rows with unknown work (`?`) at their time" in text
    assert "?: work unknown" in text


def test_a_replaced_kernel_takes_the_reference_or_is_estimated():
    torch.manual_seed(0)
    model = Tts()
    reference = _run(model).work_reference()
    model.solver.hide = functools.partial(dit_math, model.solver.dit)  # reads no weights
    rows, _ = _table(_run(model, reference))
    row = rows["Solver@m.solver"]
    assert row["reference_calls"] == 1 and row["m"] == ROWS and row["bound"] == "compute"
    # no reference (the first analyze): the module's Linear weights at its input's rows (†)
    rows, table = _table(_run(model))
    row = rows["Solver@m.solver"]
    assert row["estimated_calls"] == 1 and row["m"] == 4 and row["work_known"]
    assert "| `Solver` `m.solver` † |" in ceilings.markdown(table, min_share=0)


def test_a_compiled_module_moved_by_a_transform_keeps_its_conv_work():
    torch.manual_seed(0)
    model = Tts()
    eager = _run(model)
    reference = eager.work_reference()
    vae = _table(eager)[0]["Vae@m.vae"]
    assert vae["flops"] == {"float32": 2 * 2 * (2 * 8 * 7) * (4 * 3)}  # conv + transposed
    model.vae = Scope(torch.compile(model.vae, backend="eager"))
    with torch.inference_mode():
        model.vae(torch.ones(2, 4, 9))  # compiled before the profile

    timer = _run(model, reference)
    assert timer.compiled == ["m.vae.inner"]
    rows, table = _table(timer)
    # the wrapper at the old qualname, the compiled module by the identity of `_orig_mod`
    for target in ("Scope@m.vae", "OptimizedModule@m.vae.inner"):
        assert rows[target]["flops"] == vae["flops"] and rows[target]["reference_calls"] == 1
        assert rows[target]["floors"] == vae["floors"] and rows[target]["bound"] == vae["bound"]
    assert not table["unknown_work"]

    rows, table = _table(_run(model))  # without a reference: unknown, not "0 FLOP"
    assert not rows["OptimizedModule@m.vae.inner"]["work_known"]
    assert sorted(table["unknown_work"]) == ["OptimizedModule@m.vae.inner", "Scope@m.vae", "Tts@m"]


TRANSFORM = """import torch
from torch import nn


class Scope(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def apply(workload):
    for layer in workload.model.layers:
        layer.mlp = Scope(torch.compile(layer.mlp, backend="eager"))
"""


def test_reprofile_takes_the_work_of_the_unmodified_model(tmp_path, monkeypatch, capsys):
    """``worker analyze --out-dir --transform``: one hooked pass on the model before the
    transform is applied gives the MLPs that the transform compiles their work."""
    tests = Path(__file__).parent
    monkeypatch.syspath_prepend(str(tests))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    monkeypatch.setattr(roofline, "current_peaks", lambda: PEAKS)
    kernel_view = {"gpu_busy_ms": 1.0, "kernel_launches": 1, "kernels": [], "aten_ops": []}
    monkeypatch.setattr(profiler, "guarded_kernel_profile", lambda *a, **k: kernel_view)
    monkeypatch.setattr(profiler, "summarize", lambda p, ms, **kw: "# Profile summary\n")
    spec = WorkloadSpec(
        repo_id="toy/decoder",
        modality="llm",
        device="cpu",
        dtype="float32",
        harness=str(tests / "toy_decoder.py"),
    )
    root, out = tmp_path / "run", tmp_path / "round"
    write_json(root / "run.json", {"workload": spec.to_dict()})
    transform = tmp_path / "compile_mlp.py"
    transform.write_text(TRANSFORM)

    argv = ["analyze", "--run-dir", root, "--iters", 1, "--out-dir", out, "--transform", transform]
    assert worker.main([str(a) for a in argv]) == 0, capsys.readouterr()
    profile = read_json(out / "profile" / "profile.json")
    assert profile["work_reference"]["calls"] > 0 and profile["work_reference"]["seconds"] >= 0
    assert profile["module_gaps"]["compiled"] == [f"model.layers.{i}.mlp.inner" for i in (0, 1)]
    work = {(c["cls"], w["group"]): w for c in profile["classes"] for w in c["work"]}
    # 3 greedy steps over 8, 9 and 10 positions: 27 rows through gate_up (32 -> 128) and
    # down (64 -> 32) of each of the 2 layers
    flops = {"float32": 2 * 2 * 27 * (32 * 128 + 64 * 32)}
    for key in (("OptimizedModule", "model.layers.*.mlp.inner"), ("Scope", "model.layers.*.mlp")):
        assert work[key]["flops"] == flops and work[key]["reference_calls"] == 6
    table = read_json(out / "profile" / "ceilings.json")
    assert table["rows"] and not table["unknown_work"]
    assert "‡" in (out / "profile" / "ceilings.md").read_text()


class GraphedDit:
    """Captures the DiT in a CUDA graph on its first call, then replays it (as VoxCPM's
    ``graph_inductor_cfm_solver`` transform does with the LocDiT)."""

    def __init__(self, dit: Dit) -> None:
        self.dit = dit
        self.graph: torch.cuda.CUDAGraph | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            self.static = x.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self.dit(self.static)  # warm-up outside the capture
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.dit(self.static)
        self.static.copy_(x)
        self.graph.replay()
        return self.out.clone()


class TtsWorkload(Workload):
    modality = Modality.TTS

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = Tts().cuda().eval()

    def roots(self) -> dict[str, nn.Module]:
        return {"m": self.model}

    def make_inputs(self) -> Any:
        return torch.ones(4, D, device="cuda"), torch.ones(2, 4, 9, device="cuda")

    def run(self, inputs: Any) -> Any:
        return [t.cpu() for t in self.model(*inputs)]

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(
            all(torch.allclose(a, b) for a, b in zip(reference, candidate, strict=True))
        )


@pytest.mark.gpu
def test_reprofile_of_a_solver_that_replays_a_cuda_graph():
    w = TtsWorkload(WorkloadSpec(repo_id="toy/tts", modality="tts"))
    w.load()
    inputs = w.make_inputs()
    work = profiler.work_reference(w, inputs)  # the unmodified model, untimed
    assert work.info()["calls"] == 1 + 1 + 3 * (1 + 2) + 1 + 2  # Tts, solver, DiT, vae
    w.model.solver.hide = GraphedDit(w.model.solver.dit)  # the optimisation
    with torch.inference_mode():
        w.run(inputs)  # captured before the profile
    profile = profiler.profile_workload(w, inputs, work=work)
    assert profile["module_gaps"]["graph_replays"] == {"m.solver": STEPS}
    assert profile["work_reference"]["calls"] == work.info()["calls"]
    rows = {r["target"]: r for r in ceilings.build(profile, TOY_PEAKS, 1.0)["rows"]}
    solver = rows["Solver@m.solver"]
    assert solver["reference_calls"] == 1 and solver["calls"] == 1
    assert solver["flops"] == {"float32": 2 * ROWS * 2 * D * D * STEPS}
    assert solver["m"] == ROWS and solver["bound"] == "compute"
