"""Fusion candidates (kernel_agent/profiling/fusion.py, issue #231): chains of memory-bound ops
found from tensor storages on toy models (CPU), their bytes, launches, parent class and
placements, the L2 rule (an intermediate's size and the traffic between its write and its
read), host syncs, the overlap-aware ranking, the summary section, the scheduler's expected
gain of a region arm and the native stage graph's group evidence; the optimised recorder
against the straightforward one, the kernel map (exact launches and kernel times per op,
from fake profiler events), the regions not mined (compiled modules, CUDA graphs, kernels
launched outside the dispatcher) and an improve round's re-mined table."""

from __future__ import annotations

import gc
import json
import random
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from kernel_agent.hub import Modality
from kernel_agent.native import engine
from kernel_agent.profiling import fusion, timeline
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


class Between(nn.Module):
    """``y = x · 2``, an op on a buffer of ``rows`` × D floats that does not touch y (its read
    and its write pass through the L2), then ``y + 1``."""

    def __init__(self, rows: int) -> None:
        super().__init__()
        self.register_buffer("buf", torch.ones(rows, D))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x * 2
        other = self.buf * 3
        return y + 1, other


class NormProj(nn.Module):
    """A norm, a projection that could take it as a prologue, then element-wise ops that it
    could take as an epilogue (``mul`` + ``add``; ``silu``: one op, nothing saved alone)."""

    def __init__(self, tail: str = "muladd") -> None:
        super().__init__()
        self.norm = RMSNorm()
        self.q = nn.Linear(D, D, bias=False)
        self.tail = tail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(self.norm(x))
        return F.silu(q) if self.tail == "silu" else q * 2 + 1


class Synced(nn.Module):
    """Reads a device value on the host between two element-wise ops on the same tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        flag = bool((y.sum() > 0).item())
        z = y + 1 if flag else y - 1
        return z * 3


def mine(
    model: nn.Module, *args: torch.Tensor, l2: int | None = None, kernels: bool = False
) -> fusion.Miner:
    torch.manual_seed(0)
    miner = fusion.Miner(l2, "test")
    with (
        torch.inference_mode(),
        ModuleTimer({"m": model.eval()}, cuda=False) as timer,
        miner.recording(timer, kernels=kernels),
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


def test_an_intermediate_that_fits_in_l2_with_its_traffic_saves_no_bytes():
    wide = B * T * 2 * D * 4  # 2048 bytes: the gate's and up's outputs, SiLU's
    # between SiLU's write and the mul that reads it, the up projection reads x (1024 bytes)
    # and its weight (2048) and writes its output (2048)
    up = FULL + wide + wide
    fits = chain_with(mine(Gate(), inputs(B, T, D), l2=wide + up).chains(), "silu")
    assert all(p["dram_bytes"] == 0 for p in fits["placements"].values())
    assert fits["placements"]["epilogue"]["round_trip_bytes"] == 6 * wide  # still counted
    found = fits["placements"]["epilogue"]["intermediates"]
    assert [(names(fits["placements"]["epilogue"])[i["op"]], i["l2"]) for i in found] == [
        ("linear", "in"),
        ("silu", "in"),
        ("linear", "in"),
    ]
    assert [i["traffic_bytes"] for i in found] == [0, up, 0]  # gate and up: read at once
    table = fusion.build(
        {"fusions": mine(Gate(), inputs(B, T, D), l2=wide + up).result()}, PEAKS, 1.0
    )
    assert all(c["byte_ms"] == 0 for c in table["candidates"])
    assert any(c["launch_ms"] > 0 for c in table["candidates"])
    # one byte less: the up projection evicts part of SiLU's output before the mul reads it
    tight = chain_with(mine(Gate(), inputs(B, T, D), l2=wide + up - 1).chains(), "silu")
    silu = tight["placements"]["epilogue"]["intermediates"][1]
    assert silu["l2"] == "traffic" and silu["dram_bytes"] == 2  # ⌈4096 × 1 / 2048⌉
    assert tight["placements"]["epilogue"]["dram_bytes"] == 2


def test_an_intermediate_smaller_than_l2_evicted_by_the_traffic_before_its_read_counts():
    rows = 128  # the buffer: 8 KB read and 8 KB written between y's write and its read
    traffic = 2 * rows * D * 4
    l2 = 4 * FULL  # y (1 KB) fits alone
    miner = mine(Between(rows), inputs(B, T, D), l2=l2)
    row = chain_with(miner.chains(), "mul")
    chain = row["placements"]["chain"]
    assert names(chain) == ["mul", "add"]
    (y,) = chain["intermediates"]
    assert y["bytes"] == FULL and y["round_trip_bytes"] == 2 * FULL
    assert y["traffic_bytes"] == traffic and y["l2"] == "traffic"
    assert y["dram_bytes"] == chain["dram_bytes"] == 2 * FULL  # all of it: evicted
    assert y["calls"] == 1 and y["dram_calls"] == 1
    # part of it: the newest L2 bytes hold half of y at its read
    half = chain_with(mine(Between(rows), inputs(B, T, D), l2=traffic + FULL // 2).chains(), "mul")
    assert half["placements"]["chain"]["dram_bytes"] == FULL
    assert half["placements"]["chain"]["intermediates"][0]["l2"] == "traffic"
    # the table says why
    profile = {"fusions": miner.result()}
    table = fusion.build(profile, {"dram_gbps": 1.0, "launch_floor_us": 10.0}, 1.0)
    (c,) = table["candidates"]
    assert c["placement"] == "chain" and c["dram_bytes"] == 2 * FULL
    assert c["byte_ms"] == pytest.approx(2 * FULL / 1e9 * 1e3)  # at 1 GB/s
    assert fusion.l2_text(c) == "0.00205 MB DRAM: 1 evicted by ≤ 0.0164 MB of traffic"
    text = fusion.markdown(table, details=True)
    assert "### Intermediates and the L2" in text
    assert "`mul` (`m`) 0.00102 MB: evicted by ≤ 0.0164 MB of traffic" in text
    assert "0.00205 MB DRAM in 1 of 1 calls" in text


def test_an_intermediate_that_fits_with_little_traffic_stays_in_l2():
    rows = 1  # 64 bytes read and 64 written in between
    row = chain_with(mine(Between(rows), inputs(B, T, D), l2=4 * FULL).chains(), "mul")
    (y,) = row["placements"]["chain"]["intermediates"]
    assert y["traffic_bytes"] == 2 * rows * D * 4 and y["l2"] == "in"
    assert y["dram_bytes"] == 0 and row["placements"]["chain"]["dram_bytes"] == 0
    assert y["dram_calls"] == 0


def test_with_the_l2_unknown_every_intermediate_counts():
    for rows in (1, 128):
        row = chain_with(mine(Between(rows), inputs(B, T, D), l2=None).chains(), "mul")
        (y,) = row["placements"]["chain"]["intermediates"]
        assert y["l2"] == "unknown" and y["dram_bytes"] == 2 * FULL  # the upper bound
    table = fusion.build({"fusions": mine(Between(1), inputs(B, T, D)).result()}, PEAKS, 1.0)
    assert fusion.l2_text(table["candidates"][0]) == "0.00205 MB DRAM: L2 unknown"


def test_a_large_intermediate_read_at_once_is_partly_in_l2():
    class Twice(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2 + 1

    # nothing between y's write and its read: the L2 still holds its newest quarter
    row = chain_with(mine(Twice(), inputs(B, T, D), l2=FULL // 4).chains(), "mul")
    (y,) = row["placements"]["chain"]["intermediates"]
    assert y["l2"] == "size" and y["traffic_bytes"] == 0
    assert y["dram_bytes"] == 3 * FULL // 2  # three quarters of its write and its read
    assert fusion.l2_round_trip(FULL, 2 * FULL, 0, FULL // 4) == (3 * FULL // 2, "size")
    assert fusion.l2_round_trip(FULL, 2 * FULL, 0, FULL) == (0, "in")
    assert fusion.l2_round_trip(FULL, 2 * FULL, 5 * FULL, 4 * FULL) == (2 * FULL, "traffic")
    assert fusion.l2_round_trip(FULL, 2 * FULL, 0, None) == (2 * FULL, "unknown")


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
    order = [(not c["counted"], -c["saving_ms"], c["id"]) for c in a["candidates"]]
    assert order == sorted(order)  # counted rows by saving, ties by id, then alternatives
    assert [c["rank"] for c in a["candidates"]] == list(range(1, len(order) + 1))


# ------------------------------------------------------------------ overlaps


def _placement(saved: int, overlaps: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "launches": saved + 1,
        "launches_saved": saved,
        "boundaries": 1,
        "parent_class": "P",
        "group": "m",
        "intermediate_bytes": 0,
        "round_trip_bytes": 0,
        "dram_bytes": 0,
        "largest_bytes": 0,
        "tensors": 0,
        "ops": [{"op": "add", "module": "m"}],
        "intermediates": [],
        "overlaps": overlaps,
    }


def _raw_chain(cid: str, placements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"id": cid, "phase": "", "method": "forward", "calls": 1, "placements": placements}


def test_placements_sharing_a_gemm_are_ranked_without_double_counting():
    miner = mine(NormProj(), inputs(B, T, D), l2=1 << 20)  # every intermediate in L2
    chains = miner.chains()
    norm = chain_with(chains, "rsqrt")
    (tail,) = [c for c in chains if c is not norm]
    assert names(tail["placements"]["chain"]) == ["mul", "add"]
    # the norm's prologue and the tail's epilogue both take in the projection
    assert norm["placements"]["prologue"]["overlaps"] == {tail["id"]: ["epilogue"]}
    assert tail["placements"]["epilogue"]["overlaps"] == {norm["id"]: ["prologue"]}
    assert norm["placements"]["chain"]["overlaps"] == tail["placements"]["chain"]["overlaps"] == {}
    table = fusion.build({"fusions": miner.result()}, PEAKS, 1.0)
    first, second = table["candidates"]
    assert first["id"] == norm["id"] and first["placement"] == "prologue"  # 7 → 1: 0.06 ms
    assert first["saving_ms"] == pytest.approx(0.06)
    # the tail's epilogue (0.02 ms) would take the projection in again: it is one kernel
    assert second["id"] == tail["id"] and second["placement"] == "chain"
    assert second["saving_ms"] == pytest.approx(0.01)
    assert second["blocked"] == {"epilogue": [norm["id"]]}
    assert second["placements"] == {"chain": 0.01, "epilogue": 0.02}
    assert first["counted"] and second["counted"] and first["overlaps"] == second["overlaps"] == []
    # 9 launches become 2: 0.07 ms, not the 0.08 of each row's best
    total = sum(c["saving_ms"] for c in fusion.additive(table["candidates"]))
    assert total == pytest.approx(7 * 10.0 / 1000)
    text = fusion.markdown(table)
    assert f"| chain (epilogue 0.02 ms: overlaps `{norm['id']}`) |" in text
    assert "The 2 counted rows share no op: together they save 0.07 ms" in text


def test_a_row_whose_saving_needs_a_gemm_taken_elsewhere_is_an_alternative():
    miner = mine(NormProj("silu"), inputs(B, T, D), l2=1 << 20)
    table = fusion.build({"fusions": miner.result()}, PEAKS, 1.0)
    first, alt = table["candidates"]
    assert first["placement"] == "prologue" and first["counted"]
    # SiLU alone saves nothing; its epilogue needs the projection the norm's prologue takes
    assert names(alt) == ["linear", "silu"] and alt["placement"] == "epilogue"
    assert not alt["counted"] and alt["placements"] == {"chain": 0.0, "epilogue": 0.01}
    assert first["overlaps"] == [alt["id"]] and alt["overlaps"] == [first["id"]]
    assert [c["rank"] for c in table["candidates"]] == [1, 2]
    assert fusion.additive(table["candidates"]) == [first]
    assert fusion.additive([alt]) == [alt]  # without the row it overlaps it adds up
    text = fusion.markdown(table)
    assert f"| ↳ `{alt['id']}` (alt. of `{first['id']}`) | 0.01 |" in text
    assert "The counted row saves 0.06 ms" in text and "1 more is an alternative" in text
    # beyond the top rows, "..." sums the counted rows only
    assert "| ... |" not in fusion.markdown(table, top=1)
    hit, how = fusion.match(table, {"fusion": alt["id"]}) or ({}, "")
    assert hit is alt and f"an alternative to `{first['id']}`" in how
    hit, how = fusion.match(table, {"fusion": first["id"]}) or ({}, "")
    assert hit is first and f"alternatives `{alt['id']}` overlap it" in how


def test_the_overlap_ranking_is_deterministic():
    # two chains read one GEMM's output: both are its epilogue, saving the same; the id
    # decides which, whatever order the chains come in
    chains = [
        _raw_chain(
            cid,
            {"chain": _placement(1, {}), "epilogue": _placement(2, {other: ["epilogue"]})},
        )
        for cid, other in (("fb", "fa"), ("fa", "fb"))
    ]
    tables = [
        fusion.build({"fusions": {"chains": order}}, PEAKS, 1.0) for order in (chains, chains[::-1])
    ]
    assert tables[0] == tables[1]
    fa, fb = tables[0]["candidates"]
    assert [(c["id"], c["placement"]) for c in (fa, fb)] == [("fa", "epilogue"), ("fb", "chain")]
    assert fb["blocked"] == {"epilogue": ["fa"]} and fa["counted"] and fb["counted"]
    # a mined model: the same table from either run and any order of its chains
    runs = [mine(Stack(3), inputs(B, T, D), l2=4 * FULL).result() for _ in range(2)]
    shuffled = list(runs[1]["chains"])
    random.Random(0).shuffle(shuffled)
    a = fusion.build({"fusions": runs[0]}, PEAKS, 1.0)
    b = fusion.build({"fusions": {**runs[1], "chains": shuffled}}, PEAKS, 1.0)
    assert a == b
    counted = {c["id"] for c in a["candidates"] if c["counted"]}
    assert all(not counted & set(c["overlaps"]) for c in a["candidates"] if c["counted"])


def test_a_table_from_before_the_traffic_and_the_overlaps_stays_readable():
    raw = mine(NormProj("silu"), inputs(B, T, D), l2=1 << 20).result()
    for chain in raw["chains"]:  # a version 1 profile: no intermediates, no overlaps
        for placement in chain["placements"].values():
            del placement["intermediates"], placement["overlaps"]
    table = fusion.build({"fusions": raw}, PEAKS, 1.0)
    assert all(c["counted"] and c["overlaps"] == [] for c in table["candidates"])
    assert [c["placement"] for c in table["candidates"]] == ["prologue", "epilogue"]
    old = [
        {k: v for k, v in c.items() if k not in ("counted", "overlaps", "intermediates")}
        for c in table["candidates"]
    ]
    text = fusion.markdown({**table, "candidates": old}, details=True)
    assert "The 2 counted rows" in text and "in L2" in text and "↳" not in text
    assert "### Intermediates" not in text
    assert fusion.additive(old) == old
    hit, how = fusion.match({"candidates": old}, {"fusion": old[1]["id"]}) or ({}, "")
    assert hit is old[1] and how == f"fusion {old[1]['id']}"  # no overlaps to name


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


# ------------------------------------------------------------------ the recorder's cost


def _ref_tensors(values: Any) -> list[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        return [values]
    out = []
    for v in values if isinstance(values, tuple | list) else ():
        if isinstance(v, torch.Tensor):
            out.append(v)
        elif isinstance(v, tuple | list):
            out += [x for x in v if isinstance(x, torch.Tensor)]
    return out


def _ref_storage(t: torch.Tensor) -> tuple[Any, int]:
    try:
        ptr = t.data_ptr()
    except Exception:
        return None, 0
    if not ptr:
        return None, 0
    size = t.element_size()
    n = int(t.numel())
    if not t.is_contiguous():
        span = 1 + sum((s - 1) * abs(st) for s, st in zip(t.shape, t.stride(), strict=True))
        n = min(n, int(span))
    return (t.get_device(), ptr - int(t.storage_offset()) * size), n * size


class ReferenceRecorder(TorchDispatchMode):
    """The recorder as it was before its cost was cut (#231), verbatim but for the module
    call it reads: the straightforward bookkeeping the optimised one must reproduce."""

    def __init__(self, stack: list[Any]) -> None:
        super().__init__()
        self.stack = stack
        self.ops: list[fusion._Op] = []
        self.writer: dict[Any, int] = {}
        self.epoch = self.views = self.failed = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs or {}
        info = fusion._func(func)
        if info.composite:
            with self:
                out = func.decompose(*args, **kwargs)
            if out is not NotImplemented:
                return out
        out = func(*args, **kwargs)
        try:
            self._record(info, args, kwargs, out)
        except Exception:
            self.failed += 1
        return out

    def _record(self, info: Any, args: Any, kwargs: dict[str, Any], out: Any) -> None:
        name = info.name
        written = _ref_tensors([args[i] for i in info.written if i < len(args)])
        written += _ref_tensors([kwargs[k] for k in info.written_kw if k in kwargs])
        ins = _ref_tensors(args) + (_ref_tensors(list(kwargs.values())) if kwargs else [])
        outs = _ref_tensors(out)
        in_st = [_ref_storage(t) for t in ins]
        out_st = [_ref_storage(t) for t in outs]
        if name in fusion._ALLOC:
            for key, _ in out_st:
                self.writer.pop(key or (0, 0), None)
            return
        in_keys = {key for key, _ in in_st}
        if not written and outs and all(key in in_keys for key, _ in out_st):
            self.views += 1
            return
        if not outs and not written and name not in fusion._SYNC:
            return
        cross = to_host = False
        if name in fusion._TRANSFER:
            devices = {t.get_device() for t in (*ins, *outs) if t.dim() or not t.is_cpu}
            cross = len(devices) > 1
            to_host = cross and any(t.is_cpu for t in outs)
        sync = name in fusion._SYNC or to_host
        barrier = sync or cross or not info.aten
        kind = fusion.BARRIER if barrier else fusion.ANCHOR if info.anchor else fusion.MEM
        first = outs[0] if outs else (written[0] if written else None)
        index = len(self.ops)
        op = fusion._Op(
            name,
            kind,
            int(self.stack[-1][0]) if self.stack else -1,
            self.epoch,
            tuple(first.shape) if first is not None else (),
            first.dtype if first is not None else None,
            info.anchor or "",
        )
        overwrites = name in fusion._OVERWRITE
        loads: dict[Any, int] = {}
        for t, (key, nbytes) in zip(ins, in_st, strict=True):
            if key is None or (overwrites and any(t is w for w in written)):
                continue
            loads[key] = max(loads.get(key, 0), nbytes)
            producer = self.writer.get(key)
            if producer is not None:
                op.reads.append((producer, key, nbytes))
        op.loads = tuple(loads.items())
        mutated = {id(t) for t in written}
        carried = None
        if name in fusion._PARTIAL:
            carried = sum(n for t, (_, n) in zip(ins, in_st, strict=True) if id(t) not in mutated)
        targets = [*zip(outs, out_st, strict=True), *((t, _ref_storage(t)) for t in written)]
        for t, (key, nbytes) in targets:
            if key is None:
                continue
            if carried is not None and id(t) in mutated:
                nbytes = min(nbytes, carried)
            op.writes[key] = max(op.writes.get(key, 0), nbytes)
        for producer, key, nbytes in op.reads:
            self.ops[producer].consumers.append((index, key, nbytes))
        for key in op.writes:
            self.writer[key] = index
        self.ops.append(op)
        if sync:
            self.epoch += 1


class Mixed(nn.Module):
    """In-place and ``out=`` ops, partial writes, lists of tensors, casts (``to`` decomposes),
    a broadcast, a host sync, an allocation and copies."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        y.add_(1)
        z = torch.empty_like(y)
        torch.mul(y, 3, out=z)
        z.index_put_((torch.arange(2),), torch.ones(2, x.shape[-1]))
        s = torch.zeros(4, x.shape[-1]).scatter_add_(0, torch.zeros(4, x.shape[-1]).long(), z[:4])
        c = torch.cat([y, z], 0).to(torch.float16).float()
        m = c.masked_fill(c > 3, 0.0) * torch.ones(1, x.shape[-1])
        if bool(m.sum() > 0):
            m = m - 1
        w = torch.where(m > 1, m, -m).t().contiguous()
        return w + s.sum()


def _recorded(ops: list[fusion._Op]) -> list[tuple[Any, ...]]:
    return [
        (
            o.name,
            o.kind,
            o.call,
            o.epoch,
            tuple(o.shape),
            o.dtype,
            o.anchor,
            o.reads,
            o.loads,
            list(o.writes.items()),
            o.consumers,
        )
        for o in ops
    ]


def test_the_optimised_recorder_records_what_the_straightforward_one_does():
    x, h = inputs(B, T, D), inputs(B, T, D)
    cases = [(Top(), (x, h)), (Stack(2), (x,)), (NormProj(), (x,)), (Mixed(), (inputs(4, D),))]
    for model, args in cases:
        torch.manual_seed(0)
        with torch.inference_mode(), ModuleTimer({"m": model.eval()}, cuda=False) as timer:
            ref = ReferenceRecorder(timer._stack)
            new = fusion.Recorder(timer._stack, ranges=True)  # ranges change nothing
            with ref, new:  # the same ops, the same tensors: new dispatches each one to ref
                model(*args)
        assert new.ops and _recorded(new.ops) == _recorded(ref.ops), type(model).__name__
        assert (new.views, new.epoch, new.failed) == (ref.views, ref.epoch, ref.failed)
        assert new.failed == 0 and sorted(new.seq.values()) == list(range(len(new.ops)))
    names = {o.name for o in new.ops}
    assert {"add_", "mul", "index_put_", "scatter_add_", "cat", "_to_copy"} <= names
    assert new.epoch == 1  # the host sync
    assert not fusion.Recorder._should_skip_dynamo()  # no Dynamo wrapper per op


def test_the_pass_collects_young_garbage_less_often_and_restores_the_gc():
    before = gc.get_threshold()
    seen = []

    class Probe(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen.append(gc.get_threshold())
            return x * 2

    mine(Probe(), inputs(B, T, D))
    assert seen[0][0] == max(before[0], fusion.GC_YOUNG) and gc.get_threshold() == before


# ------------------------------------------------------------------ the kernel map


def _ev(kind: str, name: str, start: int, end: int, corr: int = -1) -> timeline.Event:
    return timeline.Event(kind, name, start, end, corr)


def _launch(
    events: list[timeline.Event], corr: int, at: int, kernels: list[int], api: str = ""
) -> None:
    """A launch call at ``at`` (correlation ``corr``) and its GPU events of these ns."""
    events.append(_ev("cuda_runtime", api or "cudaLaunchKernel", at, at + 1, corr))
    for n, ns in enumerate(kernels):
        start = 100_000 + 10_000 * corr + 1_000 * n
        events.append(_ev("kernel", f"k{corr}.{n}", start, start + ns, corr))


def test_the_kernel_map_gives_each_gpu_event_to_the_op_that_launched_it():
    R = fusion.RANGE
    events = [
        _ev(R, "ka.call/0", 0, 1000),  # the model's call
        _ev(R, "ka.call/1", 100, 900),  # a child's
        _ev(R, "ka.op/0", 110, 150),  # launches one kernel
        _ev(R, "ka.op/1", 160, 200),  # a kernel and a memset
        _ev(R, "ka.op/2", 210, 250),  # nothing: an op on host tensors
        _ev(R, "ka.op/3", 260, 300),
    ]
    _launch(events, 11, 120, [10])
    _launch(events, 12, 170, [20])
    events += [
        _ev("cuda_runtime", "cudaMemsetAsync", 180, 182, 13),
        _ev("gpu_memset", "M", 1, 2, 13),
    ]
    _launch(events, 14, 270, [10])
    _launch(events, 15, 500, [100, 20], api="cudaGraphLaunch")  # in call 1, outside every op
    _launch(events, 16, 950, [30], api="cuLaunchKernel")  # in call 0 only: a Triton kernel
    events.append(_ev("kernel", "orphan", 5, 10, 99))  # no launch call in the trace
    events.append(_ev(R, "ka.op/4", 960, 990))
    km = fusion.kernel_map(events)
    assert km.ops == {0: (1, 10), 1: (2, 21), 3: (1, 10)}
    assert km.cuts == [4]  # the graph replay and the Triton kernel ran before op 4
    assert km.unmined == {
        ("graphed", 1): (2, 120),
        ("custom", 0): (1, 30),
        ("unattributed", -1): (1, 5),
    }
    assert km.events == 8 and km.ns == 10 + 21 + 10 + 120 + 30 + 5
    assert fusion.kernel_map([e for e in events if e.kind == R]).events == 0


def _fake_kernels(
    miner: fusion.Miner, launches: dict[str, list[list[int]]]
) -> list[timeline.Event]:
    """Profiler events for a pass recorded with ranges: the k-th recorded op named ``op``
    launches one GPU event per entry of ``launches[op][k]`` (its ns; default one of 1 us)."""
    rec = miner.recorder
    assert rec is not None
    by_index = {i: seq for seq, i in rec.seq.items()}
    events: list[timeline.Event] = []
    seen: dict[str, int] = {}
    corr = 1
    for i, op in enumerate(miner.ops):
        seq = by_index[i]
        events.append(_ev(fusion.RANGE, f"{fusion.OP_RANGE}{seq}", 100 * seq, 100 * seq + 50))
        k = seen.get(op.name, 0)
        seen[op.name] = k + 1
        plan = launches.get(op.name, [[1000]])
        for ns in plan[min(k, len(plan) - 1)]:
            _launch(events, corr, 100 * seq + 10, [ns])
            corr += 1
    return events


def test_exact_launches_and_kernel_times_from_the_kernel_map():
    # on the CPU the profiled pass sees no GPU work: no map, one launch per op
    miner = mine(Gate(), inputs(B, T, D), kernels=True)
    assert miner.kernels is None and "no GPU work" in miner.kernel_note
    assert [op.name for op in miner.ops] == ["linear", "silu", "linear", "mul", "linear"]
    # the gate projection launches a split-K GEMM and its reduction (2), the rest one each
    km = miner.map_kernels(
        _fake_kernels(miner, {"linear": [[2000, 500], [1000], [1000]], "silu": [[700]]})
    )
    assert km is not None and miner.kernels is km
    assert [op.launches for op in miner.ops] == [2, 1, 1, 1, 1]
    assert [op.kernel_ns for op in miner.ops] == [2500, 700, 1000, 1000, 1000]
    row = chain_with(miner.chains(), "silu")
    chain, epilogue = row["placements"]["chain"], row["placements"]["epilogue"]
    prologue = row["placements"]["prologue"]
    assert (chain["launches"], chain["launches_saved"], chain["kernel_ns"]) == (2, 1, 1700)
    # into the merged gate / up GEMM: it keeps the split-K reduction, 5 launches become 2
    assert (epilogue["launches"], epilogue["launches_saved"]) == (5, 3)
    assert epilogue["kernel_ns"] == 2500 + 700 + 1000 + 1000
    assert epilogue["ops"][0]["launches"] == 2 and "launches" not in epilogue["ops"][1]
    assert (prologue["launches"], prologue["launches_saved"]) == (3, 2)
    raw = miner.result()
    assert raw["kernel_map"]["launches"] == 6 and raw["kernel_map"]["gpu_events"] == 6
    assert raw["kernel_map"]["ops_with_several"] == 1 and "unmined" not in raw
    # priced per GPU launch of the run (6), not per recorded op (5); bytes capped at what
    # the kernels take: at 1 MB/s the L2-unknown round trips would take seconds
    table = fusion.build({"fusions": raw}, {"dram_gbps": 0.001, "launch_floor_us": 10.0}, 0.024)
    assert table["launch_us"] == pytest.approx(4.0) and "per GPU launch" in table["launch_basis"]
    c = next(c for c in table["candidates"] if c["id"] == row["id"])
    assert c["placement"] == "epilogue" and c["byte_capped"]
    assert c["kernel_ms"] == pytest.approx(0.0052) and c["byte_ms"] == pytest.approx(0.0052)
    assert c["launch_ms"] == pytest.approx(3 * 4.0 / 1000)
    text = fusion.markdown(table)
    assert "6 GPU launches (the kernel map" in text
    assert "| 5 → 2; 0.0052 ms GPU |" in text and "(≤ its kernels' time)" in text
    assert not any("no kernel map" in n for n in table.get("notes") or [])
    unmapped = fusion.build({"fusions": mine(Gate(), inputs(B, T, D)).result()}, PEAKS, 1.0)
    assert any("no kernel map" in n for n in unmapped["notes"])


def test_an_op_that_launched_nothing_saves_neither_launches_nor_bytes():
    miner = mine(Gate(), inputs(B, T, D), kernels=True)
    # silu and mul ran on the host (no GPU event in their ranges): nothing to fuse there
    miner.map_kernels(_fake_kernels(miner, {"silu": [[]], "mul": [[]]}))
    row = chain_with(miner.chains(), "silu")
    chain = row["placements"]["chain"]
    assert chain["launches"] == 0 and chain["launches_saved"] == 0
    assert chain["round_trip_bytes"] == 0 and chain["intermediates"] == []
    assert row["placements"]["epilogue"]["launches"] == 2  # the two GEMMs, merged: 1
    assert row["placements"]["epilogue"]["launches_saved"] == 1


def test_no_chain_crosses_a_kernel_launched_outside_the_dispatcher():
    class TwoChains(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x * 2 + 1, x * 3 - 1

    miner = mine(TwoChains(), inputs(B, T, D), kernels=True)
    assert sorted(names(c["placements"]["chain"]) for c in miner.chains()) == [
        ["mul", "add"],
        ["mul", "sub"],
    ]
    events = _fake_kernels(miner, {})
    rec = miner.recorder
    assert rec is not None
    add = next(seq for seq, i in rec.seq.items() if miner.ops[i].name == "add")
    # a Triton kernel between the first mul and its add: it may read y and write what the
    # add reads, the recorder cannot tell
    _launch(events, 999, 100 * add - 20, [100], api="cuLaunchKernel")
    for _ in range(2):  # mapped again: the same epochs
        km = miner.map_kernels(events)
        assert km is not None and km.cuts == [add]
        assert [op.epoch for op in miner.ops] == [0, 1, 1, 1]
    assert [names(c["placements"]["chain"]) for c in miner.chains()] == [["mul", "sub"]]
    raw = miner.result()
    assert raw["kernel_map"]["launch_cuts"] == 1 and raw["host_syncs"] == 0
    (custom,) = raw["unmined"]
    assert custom["kind"] == "custom" and custom["gpu_events"] == 1


def test_a_profiled_pass_puts_every_op_and_module_call_in_its_range():
    from torch.profiler import ProfilerActivity, profile

    model, x, h = Top().eval(), inputs(B, T, D), inputs(B, T, D)
    with torch.inference_mode(), ModuleTimer({"m": model}, cuda=False) as timer:
        timer.call_range = fusion._call_range
        rec = fusion.Recorder(timer._stack, ranges=True)
        with profile(activities=[ProfilerActivity.CPU]) as prof, rec:
            model(x, h)
    events = fusion.profiled_events(prof)
    assert all(e.kind == fusion.RANGE for e in events)  # the CPU: no GPU work, no launches
    ops = {int(e.name.removeprefix(fusion.OP_RANGE)): e for e in events if "op/" in e.name}
    calls = {int(e.name.removeprefix(fusion.CALL_RANGE)): e for e in events if "call/" in e.name}
    assert sorted(ops) == list(range(rec.dispatched)) and set(rec.seq) <= set(ops)
    assert sorted(calls) == list(range(len(timer.calls))) == [0, 1, 2, 3]
    for seq, i in rec.seq.items():  # each op's range is inside its module call's
        call = calls[rec.ops[i].call]
        assert call.start <= ops[seq].start <= ops[seq].end <= call.end
    assert not timer._ranges and fusion.kernel_map(events).events == 0
    # the pass's own profiler records the user-scope ranges (not every aten op)
    fresh = fusion.Recorder([], ranges=True)
    with torch.inference_mode(), fusion._kernel_profiler() as prof, fresh:
        x * 2 + 1
    seen = sorted(e.name for e in fusion.profiled_events(prof))
    assert fresh.dispatched == 2 and seen == [f"{fusion.OP_RANGE}{n}" for n in (0, 1)]
    # the miner's own pass: the profiler and the call ranges only while it records
    miner = mine(Top(), inputs(B, T, D), inputs(B, T, D), kernels=True)
    assert miner.recorder is not None and miner.recorder.dispatched >= len(miner.ops)
    assert miner.map_seconds >= 0 and miner.kernels is None


# ------------------------------------------------------------------ regions not mined


class CompiledMLP(nn.Module):
    """A residual add + norm around a ``torch.compile``'d gate MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.compile(Gate(), backend="eager")
        self.norm = RMSNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.mlp(x))


def _compiled() -> nn.Module:
    torch.manual_seed(0)
    model = CompiledMLP().eval()
    with torch.inference_mode():
        model(inputs(B, T, D))  # compiled here; under the miner Dynamo runs it eagerly
    return model


def test_the_ops_of_a_compiled_module_are_not_mined():
    miner = mine(_compiled(), inputs(B, T, D))
    hidden = [op for op in miner.ops if op.hidden]
    # its Python ran eagerly under the dispatch mode: recorded, but not what the run does
    assert [op.name for op in hidden] == ["linear", "silu", "linear", "mul", "linear"]
    assert all(op.kind == fusion.BARRIER for op in hidden)
    chains = miner.chains()
    assert all("silu" not in names(c["placements"]["chain"]) for c in chains)
    norm = chain_with(chains, "rsqrt")  # outside the compiled module: mined
    assert names(norm["placements"]["chain"]) == ["add", *NORM_OPS]
    assert "epilogue" not in norm["placements"]  # the compiled down projection is no anchor
    raw = miner.result()
    assert raw["unmined"] == [
        {"kind": "compiled", "where": "m.mlp", "instances": 1, "gpu_events": 0, "gpu_ms": 0.0}
        | {"ops": 5}
    ]
    # its share: the profile's module view (the compiled code's own time)
    classes = [
        {"cls": "CompiledMLP", "inclusive_ms": 10.0, "work": [{"group": "m", "inclusive_ms": 10}]},
        {
            "cls": "OptimizedModule",
            "inclusive_ms": 4.0,
            "work": [{"group": "m.mlp", "inclusive_ms": 4}],
        },
    ]
    table = fusion.build({"fusions": raw, "classes": classes}, PEAKS, 1.0)
    (row,) = table["unmined"]
    assert row["share"] == 0.4 and row["basis"] == "the module view's inclusive time"
    text = fusion.markdown(table)
    assert "* **not mined**: compiled code in `m.mlp` (40.0%, 5 ops recorded eagerly)" in text
    # without the module in the view and without a kernel map: its time is unknown
    alone = fusion.build({"fusions": raw}, PEAKS, 1.0)["unmined"][0]
    assert alone["share"] is None and "no kernel map" in alone["basis"]


def test_graphed_and_custom_regions_by_the_kernel_map_and_by_the_profile():
    # with a kernel map: the GPU work launched outside every op range, by its share
    unmined = [
        {"kind": "graphed", "where": "m.dec", "gpu_events": 64, "gpu_ms": 30.0},
        {"kind": "custom", "where": "m.enc.attn", "gpu_events": 8, "gpu_ms": 5.0},
    ]
    raw = {"chains": [], "kernel_map": {"gpu_ms": 50.0, "launches": 900}, "unmined": unmined}
    rows = fusion.not_mined({"fusions": raw})
    assert [(r["kind"], r["where"], r["share"]) for r in rows] == [
        ("graphed", "m.dec", 0.6),
        ("custom", "m.enc.attn", 0.1),
    ]
    assert all(r["basis"] == "GPU time in the miner's pass" for r in rows)
    table = fusion.build({"fusions": raw}, PEAKS, 1.0)
    text = fusion.markdown(table)  # no chain left, but what was not mined is said
    assert "CUDA-graph replays in `m.dec` (60.0%, 30 ms GPU)" in text
    assert "kernels launched outside the dispatcher in `m.enc.attn` (10.0%, 5 ms GPU)" in text
    # a run mostly graph launched: the ops the miner saw launch eagerly, priced as such
    timeline_ = {"stages": [{"events": 100, "graph_events": 90}]}
    graphed = fusion.build({"fusions": raw, "kernel_view": {"timeline": timeline_}}, PEAKS, 1.0)
    assert graphed["launch_mode"] == "eager" and graphed["launch_us"] == pytest.approx(1000 / 900)
    assert graphed["launch_basis"].startswith("the run's time per GPU launch")
    assert "the mined ops launch eagerly" in graphed["launch_basis"]
    # without one (a CPU pass, a table from before): the regions the profile itself saw
    classes = [
        {"cls": "Top", "inclusive_ms": 100.0, "work": [{"group": "m", "inclusive_ms": 100.0}]},
        {"cls": "Dec", "inclusive_ms": 60.0, "work": [{"group": "m.dec", "inclusive_ms": 60.0}]},
        {
            "cls": "OptimizedModule",
            "inclusive_ms": 20.0,
            "work": [{"group": "m.enc", "inclusive_ms": 20}],
        },
    ]
    stages = [
        {"stage": "m.post", "events": 100, "graph_events": 90, "gpu_ms": 10.0},
        {"stage": "m.enc", "events": 40, "graph_events": 0, "gpu_ms": 25.0},
    ]
    profile = {
        "fusions": {"chains": []},
        "classes": classes,
        "module_gaps": {"compiled": ["m.enc"], "graph_replays": {"m.dec": 64}},
        "kernel_view": {"timeline": {"kernel_time_ms": 50.0, "stages": stages}},
    }
    rows = fusion.not_mined(profile)
    assert [(r["kind"], r["where"], r["share"]) for r in rows] == [
        ("graphed", "m.dec", 0.6),
        ("compiled", "m.enc", 0.2),
        ("graphed", "m.post", 0.2),
    ]
    assert rows[2]["basis"].startswith("the timeline")


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


def test_an_improve_round_re_mines_the_optimised_model(tmp_path, monkeypatch):
    from kernel_agent import scheduler, worker
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
    assert "## Fusion candidates (measured)" in text and "this round" not in text
    first = json.loads((run.profile_dir / "fusions.json").read_text())
    # the round's item: layer 0's MLP compiled (what an accepted transform does); the
    # worker's re-profile (analyze --out-dir) mines the optimised model
    layer = w.model.layers[0]
    layer.mlp = torch.compile(layer.mlp, backend="eager")
    with torch.inference_mode():
        w.run(w.make_inputs())
    again = profiler.profile_workload(w, w.make_inputs(), 4.0, fusions=True)
    assert again["module_gaps"]["compiled"] == ["model.layers.0.mlp"]
    round_dir = RunDir(tmp_path / "run" / "rounds" / "2")
    text = worker._fusion_section(run, round_dir, again, 4.0, "per run")
    assert "Mined from the optimised model of this round" in text
    assert "compiled code in `model.layers.*.mlp` (1 instance) (" in text  # not mined
    table = json.loads((round_dir.profile_dir / "fusions.json").read_text())
    left = next(c for c in table["candidates"] if "silu" in [o["op"] for o in c["ops"]])
    assert left["calls"] == 1 and left["group"] == "model.layers.*.mlp"  # layer 1's only
    (compiled,) = table["unmined"]
    assert compiled["kind"] == "compiled" and compiled["where"] == "model.layers.*.mlp"
    assert compiled["instances"] == 1 and compiled["ops"] == 5
    assert 0 < compiled["share"] < 1 and compiled["basis"] == "the module view's inclusive time"
    # the scheduler expects a region arm's saving from the newest table: the round's
    gate = next(c for c in first["candidates"] if c["id"] == left["id"])
    assert left["saving_ms"] == pytest.approx(gate["saving_ms"] / 2, rel=1e-4)
    profiles = [{"fusions": table, "baseline_ms": 8.0}, {"fusions": first, "baseline_ms": 8.0}]
    gain = scheduler.fusion_gain({"kind": "region", "fusion": left["id"]}, profiles)
    assert gain is not None and gain.saving_ms == pytest.approx(left["saving_ms"] * 2)
    # a profile without chains (none mined) shows the run's own table
    fallback = worker._fusion_section(run, RunDir(tmp_path / "r3"), {"classes": []}, 4.0, "")
    assert "## Fusion candidates (measured)" in fallback and "unmodified model" in fallback


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


def _region_run(
    tmp_path, spec: dict[str, Any], candidates: list[dict[str, Any]] | None = None
) -> RunDir:
    root = tmp_path / "run"
    write_json(root / "run.json", {"config": {}})
    write_json(root / "baseline.json", {"median_ms": 100.0})
    classes = [{"cls": "Block", "root": "m", "inclusive_ms": 50.0, "instances": 2}]
    write_json(root / "profile" / "profile.json", {"classes": classes})
    candidates = candidates or [  # a version 1 table: no counted, no overlaps
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


def test_a_region_arm_expects_a_counted_row_before_an_alternative(tmp_path):
    spec = {"kind": "region", "parent_class": "Block", "module_class": "Region_r"}
    candidates = [
        {"id": "fcount", "saving_ms": 0.5, "parent_class": "Block", "kind": "region",
         "counted": True, "overlaps": ["falt"]},
        {"id": "falt", "saving_ms": 2.0, "parent_class": "Block", "kind": "region",
         "counted": False, "overlaps": ["fcount"]},
    ]  # fmt: skip
    arm = build_arms(_region_run(tmp_path, spec, candidates), Policy(systems=False), [], rows=[])[0]
    assert arm.fusion is not None and arm.fusion.id == "fcount"
    assert arm.expected_ms == pytest.approx(1.0)  # 0.5 ms of a 50 ms window, at 100 ms
    assert "alternatives `falt` overlap it" in arm.why()
    # the plan names the alternative: the arm expects its own saving, and says it overlaps
    named = _region_run(tmp_path / "b", {**spec, "fusion": "falt"}, candidates)
    arm = build_arms(named, Policy(systems=False), [], rows=[])[0]
    assert arm.fusion is not None and arm.fusion.id == "falt"
    assert arm.expected_ms == pytest.approx(4.0)
    assert "an alternative to `fcount`" in arm.why()


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


def test_the_native_stage_graph_takes_the_chains_of_the_newest_re_profile(tmp_path):
    run = RunDir(tmp_path / "run")
    write_json(run.profile_dir / "ceilings.json", {"rows": [{"group": "m"}]})
    write_json(run.profile_dir / "fusions.json", {"candidates": [{"id": "fall"}]})
    assert [c["id"] for c in engine.fusion_candidates(run)] == ["fall"]
    rounds = run.root / "rounds"
    write_json(rounds / "2" / "profile" / "ceilings.json", {"rows": [{"group": "m"}]})
    assert [c["id"] for c in engine.fusion_candidates(run)] == ["fall"]  # none mined there
    write_json(rounds / "2" / "profile" / "fusions.json", {"candidates": [{"id": "fleft"}]})
    assert [c["id"] for c in engine.fusion_candidates(run)] == ["fleft"]  # what is left


def test_group_evidence_adds_up_only_chains_that_share_no_op():
    table = {
        "columns": ["exact"],
        "rows": [
            _row("Encoder", "m.enc", 300),
            _row("Linear", "m.enc.fc", 290),
            _row("Decoder", "m.dec", 200),
            _row("Conv", "m.dec.conv", 190),
        ],
    }
    spans = ["m.enc.fc", "m.dec"]
    chains = [
        {"id": "f1", "saving_ms": 0.4, "counted": True, "overlaps": ["f3"], "modules": spans,
         "ops": [{"op": "add"}]},
        {"id": "f3", "saving_ms": 0.3, "counted": False, "overlaps": ["f1"], "modules": spans,
         "ops": [{"op": "mul"}]},
        {"id": "f4", "saving_ms": 0.2, "counted": True, "overlaps": [], "modules": spans,
         "ops": [{"op": "cat"}]},
    ]  # fmt: skip
    group = engine.stage_graph(table, fusions=chains)[-1]
    assert group.scope == "group" and group.members == ("dec", "enc")
    # f1 + f4; f3 shares an op with f1 (one GEMM in f1's prologue and f3's epilogue)
    assert "3 fusion chains across its stages' boundaries predict 0.6 ms" in group.evidence
    assert "1 of them share an op with one counted, not added" in group.evidence
    assert group.evidence.index("`f4`") < group.evidence.index("`f3` 0.3 ms (`mul`, an alt")
    # without the chain it overlaps, an alternative adds up
    alone = engine.stage_graph(table, fusions=[chains[1]])[-1]
    assert "predict 0.3 ms" in alone.evidence and "not added" not in alone.evidence


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


class Graphed(nn.Module):
    """``y = x · 2 + 1`` captured in a CUDA graph and replayed, then an eager ``+ 3``."""

    def __init__(self, x: torch.Tensor) -> None:
        super().__init__()
        self.static = x.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up off the capture stream
            self.static * 2 + 1
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.static * 2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.static.copy_(x)
        self.graph.replay()
        return self.out + 3


@pytest.mark.gpu
def test_a_cuda_model_maps_its_ops_to_their_kernels_and_reports_a_graph_replay():
    model = Attend().cuda().half()
    x = inputs(B, T, D).cuda().half()
    miner = mine(model, x, l2=1 << 20, kernels=True)
    assert miner.kernels is not None and miner.kernels.events > 0
    # every recorded op ran on the GPU: one launch or more, with its kernels' time
    assert all(op.launches >= 1 and op.kernel_ns > 0 for op in miner.ops)
    assert {op.name: op.launches for op in miner.ops}["mul"] == 1
    raw = miner.result()
    assert raw["kernel_map"]["launches"] == sum(op.launches for op in miner.ops)
    assert raw["kernel_map"]["gpu_events"] == raw["kernel_map"]["launches"]  # all mined
    assert "unmined" not in raw
    row = chain_with(raw["chains"], "add")
    assert row["placements"]["chain"]["launches"] == 2 and row["placements"]["chain"]["kernel_ns"]
    # a graph replay dispatches nothing: its kernels are not mined, by the call that replays
    graphed = Graphed(x)
    torch.cuda.synchronize()
    miner = mine(graphed, x, kernels=True)
    assert [op.name for op in miner.ops] == ["copy_", "add"]
    (region,) = miner.result()["unmined"]
    assert region["kind"] == "graphed" and region["where"] == "m" and region["gpu_events"] == 2
    torch.cuda.synchronize()
