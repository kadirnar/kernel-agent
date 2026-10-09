"""Fusion candidates (kernel_agent/profiling/fusion.py, issue #231): chains of memory-bound ops
found from tensor storages on toy models (CPU), their bytes, launches, parent class and
placements, the L2 rule, host syncs, the ranking, the summary section, the scheduler's
expected gain of a region arm and the native stage graph's group evidence."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.native import engine
from kernel_agent.profiling import fusion
from kernel_agent.profiling.profiler import ModuleTimer
from kernel_agent.scheduler import Policy, build_arms
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec
from kernel_agent.workspace import RunDir, write_json

B, T, D = 2, 8, 16
FULL = B * T * D * 4  # one [B, T, D] fp32 tensor: 1024 bytes
ROW = B * T * 4  # one [B, T, 1] fp32 tensor: 64 bytes
NORM_OPS = ["pow", "mean", "add", "rsqrt", "mul", "mul"]
PEAKS = {"dram_gbps": 1000.0, "launch_floor_us": 10.0}


class RMSNorm(nn.Module):
    def __init__(self, d: int = D) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class AddNorm(nn.Module):
    """A residual add in the parent's own code, then a norm and a projection (children)."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm()
        self.proj = nn.Linear(D, D, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x + h
        return self.proj(self.norm(y)), y


class Top(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = AddNorm()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        out, y = self.block(x, h)
        return out * y  # y is read again outside the block: its write stays


class Gate(nn.Module):
    """SiLU(gate(x)) · up(x) → down."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(D, 2 * D, bias=False)
        self.up = nn.Linear(D, 2 * D, bias=False)
        self.down = nn.Linear(2 * D, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = Gate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class Stack(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Synced(nn.Module):
    """Reads a device value on the host between two element-wise ops on the same tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        flag = bool((y.sum() > 0).item())
        z = y + 1 if flag else y - 1
        return z * 3


def mine(model: nn.Module, *args: torch.Tensor, l2: int | None = None) -> fusion.Miner:
    torch.manual_seed(0)
    miner = fusion.Miner(l2, "test")
    with (
        torch.inference_mode(),
        ModuleTimer({"m": model.eval()}, cuda=False) as timer,
        miner.recording(timer),
    ):
        model(*args)
    return miner


def inputs(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(1))


def chain_with(chains: list[dict[str, Any]], op: str) -> dict[str, Any]:
    found = [c for c in chains if op in [o["op"] for o in c["placements"]["chain"]["ops"]]]
    assert len(found) == 1, [c["placements"]["chain"]["ops"] for c in chains]
    return found[0]


def names(placement: dict[str, Any]) -> list[str]:
    return [o["op"] for o in placement["ops"]]


# ------------------------------------------------------------------ chains


def test_a_residual_add_and_a_norm_across_two_modules_with_the_projection_after_them():
    torch.manual_seed(0)
    miner = mine(Top(), inputs(B, T, D), inputs(B, T, D))
    row = chain_with(miner.chains(), "rsqrt")
    chain = row["placements"]["chain"]
    assert names(chain) == ["add", *NORM_OPS]
    assert [o["module"] for o in chain["ops"]] == ["m.block"] + ["m.block.norm"] * 6
    assert chain["parent_class"] == "AddNorm" and chain["group"] == "m.block"
    assert chain["launches"] == 7 and chain["launches_saved"] == 6
    assert chain["boundaries"] == 1  # the block's own code and its norm
    # written and read inside: y (read by pow and mul; Top reads it again: written anyway),
    # x², its mean, + eps, rsqrt, x · rsqrt; the norm's output goes to the projection
    assert chain["intermediate_bytes"] == FULL + FULL + ROW + ROW + ROW + FULL
    assert chain["round_trip_bytes"] == 2 * FULL + 2 * FULL + 2 * ROW * 3 + 2 * FULL
    assert chain["dram_bytes"] == chain["round_trip_bytes"]  # L2 unknown: all of it
    assert chain["tensors"] == 6 and chain["largest_bytes"] == FULL
    # the projection reads the norm's output: a prologue of its GEMM, one more boundary
    prologue = row["placements"]["prologue"]
    assert names(prologue) == ["add", *NORM_OPS, "linear"]
    assert prologue["ops"][-1]["anchor"] == "gemm"
    assert prologue["ops"][-1]["module"] == "m.block.proj"
    assert prologue["launches"] == 8 and prologue["launches_saved"] == 7
    assert prologue["boundaries"] == 2 and prologue["parent_class"] == "AddNorm"
    assert prologue["intermediate_bytes"] == chain["intermediate_bytes"] + FULL
    assert prologue["round_trip_bytes"] == chain["round_trip_bytes"] + 2 * FULL
    assert "epilogue" not in row["placements"]  # nothing anchors it from before
    assert row["calls"] == 1 and row["instances"] == 1 and row["phase"] == "prefill"
    assert row["method"] == "forward"
    assert row["modules"] == ["m.block", "m.block.norm", "m.block.proj"]


def test_silu_times_up_is_the_epilogue_of_the_merged_gate_and_up_projections():
    miner = mine(Gate(), inputs(B, T, D))
    row = chain_with(miner.chains(), "silu")
    chain, epilogue = row["placements"]["chain"], row["placements"]["epilogue"]
    assert names(chain) == ["silu", "mul"] and chain["boundaries"] == 0
    assert chain["parent_class"] == "Gate" and chain["launches_saved"] == 1
    wide = B * T * 2 * D * 4
    assert chain["intermediate_bytes"] == wide and chain["round_trip_bytes"] == 2 * wide
    # gate and up read the same input: one GEMM can take silu · mul as its epilogue
    assert names(epilogue) == ["linear", "silu", "linear", "mul"]
    assert [o["module"] for o in epilogue["ops"] if "anchor" in o] == ["m.gate", "m.up"]
    assert epilogue["launches"] == 4 and epilogue["launches_saved"] == 3
    assert epilogue["boundaries"] == 2 and epilogue["parent_class"] == "Gate"
    assert epilogue["intermediate_bytes"] == 3 * wide
    assert epilogue["round_trip_bytes"] == 6 * wide
    assert names(row["placements"]["prologue"]) == ["silu", "mul", "linear"]
    best = fusion.build({"fusions": miner.result()}, PEAKS, 1.0)["candidates"]
    top = next(c for c in best if c["id"] == row["id"])
    assert top["placement"] == "epilogue" and top["kind"] == "region"
    assert top["launches_saved"] == 3 and top["launch_ms"] == pytest.approx(3 * 10.0 / 1000)
    assert top["region"] == (
        "`linear` (in `gate`) → `silu` (the parent's own code) → `linear` (in `up`) → "
        "`mul` (the parent's own code)"
    )


def test_the_same_chain_in_every_layer_is_one_row():
    miner = mine(Stack(3), inputs(B, T, D))
    row = chain_with(miner.chains(), "silu")
    assert row["calls"] == 3 and row["instances"] == 3
    assert row["placements"]["epilogue"]["group"] == "m.layers.*.mlp"
    wide = B * T * 2 * D * 4
    assert row["placements"]["epilogue"]["round_trip_bytes"] == 3 * 6 * wide  # every layer
    ids = [c["id"] for c in miner.chains()]
    assert len(ids) == len(set(ids))


def test_an_intermediate_that_fits_in_l2_saves_no_bytes():
    wide = B * T * 2 * D * 4  # 2048 bytes, the largest intermediate of the gate
    small = chain_with(mine(Gate(), inputs(B, T, D), l2=wide).chains(), "silu")
    assert all(p["dram_bytes"] == 0 for p in small["placements"].values())
    assert small["placements"]["epilogue"]["round_trip_bytes"] == 6 * wide  # still counted
    big = chain_with(mine(Gate(), inputs(B, T, D), l2=wide - 1).chains(), "silu")
    assert big["placements"]["epilogue"]["dram_bytes"] == 6 * wide
    miner = mine(Gate(), inputs(B, T, D), l2=wide)
    table = fusion.build({"fusions": miner.result()}, PEAKS, 1.0)
    assert all(c["byte_ms"] == 0 for c in table["candidates"])
    assert any(c["launch_ms"] > 0 for c in table["candidates"])


def test_no_chain_crosses_a_host_sync():
    miner = mine(Synced(), inputs(B, T, D))
    ops = miner.ops
    sync = next(i for i, op in enumerate(ops) if op.name == "_local_scalar_dense")
    assert ops[sync].kind == fusion.BARRIER and miner.recorder.epoch == 1
    for group in miner.groups():
        assert len({ops[i].epoch for i in group}) == 1
        assert not (min(group) < sync < max(group))
    # without the sync, `y + 1` would join `x * 2` (it reads y)
    after = [names(c["placements"]["chain"]) for c in miner.chains()]
    assert ["add", "mul"] in after
    assert all(not ("sum" in n and "add" in n) for n in after)


def test_the_ranking_is_deterministic():
    first = mine(Stack(2), inputs(B, T, D)).result()
    second = mine(Stack(2), inputs(B, T, D)).result()
    first.pop("seconds"), second.pop("seconds")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    a = fusion.build({"fusions": first}, PEAKS, 1.0)
    b = fusion.build({"fusions": second}, PEAKS, 1.0)
    assert a == b
    savings = [(-c["saving_ms"], c["id"]) for c in a["candidates"]]
    assert savings == sorted(savings)  # by saving, ties by id
    assert [c["rank"] for c in a["candidates"]] == list(range(1, len(savings) + 1))


def test_views_and_allocations_are_not_ops():
    class Viewer(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = (x * 2).view(-1).reshape(B, T, D).transpose(0, 1)
            buf = torch.empty_like(x)
            buf.copy_(x)
            return y.contiguous() + buf.transpose(0, 1)

    miner = mine(Viewer(), inputs(B, T, D))
    # view / reshape of a contiguous tensor and transpose are views; contiguous() copies
    assert [op.name for op in miner.ops] == ["mul", "copy_", "clone", "add"]
    assert miner.recorder.views >= 4
    copy = miner.ops[1]
    assert not copy.reads  # it overwrites the fresh buffer: no earlier writer
    assert [p for p, _, _ in miner.ops[2].reads] == [0] and copy.consumers[0][0] == 3


# ------------------------------------------------------------------ the pass and the table


class GateWorkload(Workload):
    modality = Modality.LLM

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = Stack(2).eval()

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> Any:
        return inputs(B, T, D)

    def run(self, inputs: Any) -> Any:
        return self.model(inputs)

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.equal(reference, candidate)))


def test_scan_runs_the_hooked_work_pass_and_writes_the_table(tmp_path):
    w = GateWorkload(WorkloadSpec(repo_id="toy/gate", modality="llm"))
    w.load()
    found = fusion.scan(w, w.make_inputs(), l2_bytes=1 << 20, l2_source="test L2")
    assert "error" not in found and found["ops"] > 0 and found["l2_bytes"] == 1 << 20
    assert chain_with(found["chains"], "silu")["calls"] == 2
    profile = {"fusions": found, "kernel_view": {}}
    table = fusion.write(tmp_path, profile, PEAKS, 4.0, per="per run")
    assert (tmp_path / "fusions.json").exists() and (tmp_path / "fusions.md").exists()
    assert table["launch_mode"] == "eager" and table["launch_us"] == 10.0
    text = fusion.markdown(table)
    assert "## Fusion candidates (measured)" in text and "the 1 MB L2" in text
    assert f"`{table['candidates'][0]['id']}`" in text
    assert "`fusion` = its id" in text


def test_analyze_mines_the_unmodified_model_and_a_round_shows_the_runs_table(tmp_path, monkeypatch):
    from kernel_agent import worker
    from kernel_agent.kernels import roofline
    from kernel_agent.profiling import profiler

    view = {"gpu_busy_ms": 1.0, "kernel_launches": 1, "kernels": [], "aten_ops": []}
    monkeypatch.setattr(profiler, "guarded_kernel_profile", lambda *a, **k: dict(view))
    monkeypatch.setattr(roofline, "current_peaks", lambda: PEAKS)
    w = GateWorkload(WorkloadSpec(repo_id="toy/gate", modality="llm"))
    w.load()
    profile = profiler.profile_workload(w, w.make_inputs(), 4.0, fusions=True)
    assert chain_with(profile["fusions"]["chains"], "silu")["calls"] == 2
    assert "fusions" not in profiler.profile_workload(w, w.make_inputs(), 4.0)
    run = RunDir(tmp_path / "run")
    text = worker._fusion_section(run, run, profile, 4.0, "per run")
    assert "## Fusion candidates (measured)" in text
    assert (run.profile_dir / "fusions.json").exists()
    # a re-profile of an optimised model (its items hide the ops): the run's own table
    round_dir = RunDir(tmp_path / "run" / "rounds" / "2")
    again = worker._fusion_section(run, round_dir, {"classes": []}, 4.0, "per run")
    assert "## Fusion candidates (measured)" in again and "unmodified model" in again
    assert not (round_dir.profile_dir / "fusions.json").exists()


def test_a_failed_run_keeps_what_was_recorded():
    class Broken(GateWorkload):
        def run(self, inputs: Any) -> Any:
            self.model(inputs)
            raise RuntimeError("boom")

    w = Broken(WorkloadSpec(repo_id="toy/broken", modality="llm"))
    w.load()
    found = fusion.scan(w, w.make_inputs(), l2_bytes=None, l2_source="test")
    assert "RuntimeError: boom" in found["error"] and found["chains"]


def test_graph_launched_runs_price_a_launch_at_a_graph_boundary():
    timeline = {"stages": [{"events": 100, "graph_events": 90}]}
    profile = {"fusions": mine(Gate(), inputs(B, T, D)).result()}
    profile["kernel_view"] = {"timeline": timeline}
    table = fusion.build(profile, PEAKS, 1.0)
    assert table["launch_mode"] == "graph" and table["launch_us"] == fusion.GRAPH_BOUNDARY_US
    # peaks with a measured CUDA-graph launch floor (version 8, #226): that one
    measured = fusion.build(profile, {**PEAKS, "launch_floor_graph_us": 1.28}, 1.0)
    assert measured["launch_us"] == 1.28 and "measured CUDA-graph" in measured["launch_basis"]
    no_peaks = fusion.build({"fusions": profile["fusions"]}, {}, 1.0)
    assert any("not measured" in n for n in no_peaks["notes"])
    # eager: the launch floor, at most what an op of the run takes on average
    ops = profile["fusions"]["ops"]
    eager = fusion.build({"fusions": profile["fusions"]}, PEAKS, ops * 0.004)
    assert eager["launch_mode"] == "eager" and eager["launch_us"] == pytest.approx(4.0)
    assert "time per recorded op" in eager["launch_basis"]
    assert fusion.build({"fusions": profile["fusions"]}, PEAKS, ops * 0.02)["launch_us"] == 10.0


# ------------------------------------------------------------------ consumers


def _region_run(tmp_path, spec: dict[str, Any]) -> RunDir:
    root = tmp_path / "run"
    write_json(root / "run.json", {"config": {}})
    write_json(root / "baseline.json", {"median_ms": 100.0})
    classes = [{"cls": "Block", "root": "m", "inclusive_ms": 50.0, "instances": 2}]
    write_json(root / "profile" / "profile.json", {"classes": classes})
    candidates = [
        {"id": "fsmall", "saving_ms": 0.5, "parent_class": "Block", "kind": "region"},
        {"id": "fbig", "saving_ms": 2.0, "parent_class": "Block", "kind": "region"},
        {"id": "fmod", "saving_ms": 9.0, "parent_class": "Block", "kind": "module"},
    ]
    write_json(root / "profile" / "fusions.json", {"window_ms": 50.0, "candidates": candidates})
    write_json(root / "targets" / "r" / "spec.json", spec)
    return RunDir(root)


def test_a_region_arm_expects_the_saving_of_its_fusion(tmp_path):
    spec = {"kind": "region", "parent_class": "Block", "module_class": "Region_r"}
    run = _region_run(tmp_path, {**spec, "fusion": "fsmall"})
    arm = build_arms(run, Policy(systems=False), [], rows=[])[0]
    assert arm.fusion is not None and arm.fusion.id == "fsmall"
    assert arm.fusion.saving_ms == pytest.approx(0.5 * 100.0 / 50.0)  # metric ms
    assert arm.remaining_ms == pytest.approx(100.0)  # the parent class, an upper bound
    assert arm.expected_ms == pytest.approx(1.0)
    assert "fusion fsmall" in arm.why()
    # without an id: the largest region candidate of its parent class
    other = _region_run(tmp_path / "b", spec)
    arm = build_arms(other, Policy(systems=False), [], rows=[])[0]
    assert arm.fusion is not None and arm.fusion.id == "fbig"
    assert arm.expected_ms == pytest.approx(4.0)


def test_a_fusion_already_saved_leaves_a_share_of_its_prediction(tmp_path):
    spec = {"kind": "region", "parent_class": "Block", "module_class": "Region_r"}
    arm = build_arms(_region_run(tmp_path, spec), Policy(systems=False), [], rows=[])[0]
    arm.best = 2.0  # its kernel halved the region: 50 ms saved, far beyond the 4 predicted
    assert arm.remaining_ms == pytest.approx(50.0)
    assert arm.remaining_ms * arm.headroom == pytest.approx((1 - 1 / 1.1) * 4.0)


def _row(cls, group, now, calls=1):
    return {
        "cls": cls,
        "group": group,
        "now_ms": now,
        "share": now / 1000.0,
        "calls": calls,
        "instances": 1,
        "phase": "prefill",
        "floors": {"exact": now / 2},
    }


def test_chains_that_span_stages_are_group_evidence():
    table = {
        "columns": ["exact"],
        "rows": [
            _row("Encoder", "m.enc", 300),
            _row("Linear", "m.enc.fc", 290),
            _row("Decoder", "m.dec", 200),
            _row("Conv", "m.dec.conv", 190),
        ],
    }
    chains = [
        {"id": "f1", "saving_ms": 0.4, "modules": ["m.enc.fc", "m.dec"], "ops": [{"op": "add"}]},
        {"id": "f2", "saving_ms": 0.1, "modules": ["m.dec.conv"], "ops": [{"op": "mul"}]},
    ]
    stages = engine.stage_graph(table, fusions=chains)
    assert [s.id for s in stages] == ["enc", "dec", "fuse_dec_enc"]
    group = stages[-1]
    assert group.scope == "group" and group.members == ("dec", "enc")
    assert group.now_ms == 500 and group.floor_ms == 250
    assert "`f1` 0.4 ms (`add`)" in group.evidence and "f2" not in group.evidence
    assert group.evidence in group.describe()
    assert [s.id for s in engine.stage_graph(table)] == ["enc", "dec"]  # without the table
    # stages called equally often form the loop body: the chain is its evidence
    looped = {**table, "rows": [{**r, "calls": 4} for r in table["rows"]]}
    stages = engine.stage_graph(looped, fusions=chains)
    body = next(s for s in stages if s.id == "loop_body")
    assert "`f1`" in body.evidence and not any(s.id.startswith("fuse_") for s in stages)


# ------------------------------------------------------------------ GPU


class Attend(nn.Module):
    """q projection → RoPE-like element-wise ops → SDPA on CUDA (the anchor stays one op)."""

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(D, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = (self.q(x) * 2 + 1).view(B, T, 2, D // 2).transpose(1, 2)
        return F.scaled_dot_product_attention(q, q, q).transpose(1, 2).reshape(B, T, D)


@pytest.mark.gpu
def test_a_cuda_model_is_mined_with_the_gpus_l2():
    from kernel_agent import toolchain

    model = Attend().cuda().half()
    x = inputs(B, T, D).cuda().half()
    l2, source = fusion.gpu_l2()
    info = toolchain.gpu_info()
    assert info is not None and l2 == int(info.l2_cache_mb * 1024**2) and info.name in source
    miner = mine(model, x, l2=l2)
    anchors = [op.anchor for op in miner.ops if op.kind == fusion.ANCHOR]
    assert anchors == ["gemm", "attention"]
    row = chain_with(miner.chains(), "add")
    assert names(row["placements"]["chain"]) == ["mul", "add"]
    assert names(row["placements"]["prologue"])[-1] == "scaled_dot_product_attention"
    assert all(p["dram_bytes"] == 0 for p in row["placements"].values())  # KBs: in L2
    torch.cuda.synchronize()
