"""Megakernel schedules built from a stage's captured ops (issue #225,
``kernel_agent.native.megakernel.captured``), on toy modules and CPU tensors: the DAG of an
RMSNorm → Linear → residual chain equals the one the example declares by hand (ops, tiles,
edges, counter targets, the program byte for byte), a gated MLP maps to fused GEMVs, a gated
activation and chunked counters (split over K on request), unsupported ops are reported by
name with why, the simulator finds no deadlock over random SM speeds, and every plan computes
its module when its instructions are executed with the opcodes' semantics on the CPU
(:func:`execute`). The GPU runs are in test_megakernel_gpu.py."""

from __future__ import annotations

import json
import struct

import pytest
import torch
from torch import nn

from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.native import project as proj
from kernel_agent.native.megakernel import captured, simulate
from kernel_agent.native.megakernel import opcodes as oc
from kernel_agent.native.megakernel import schedule as mks
from kernel_agent.native.megakernel.captured import Binding, TileMap
from kernel_agent.selftest import GatedMlpBlock, NormGemvChain, RMSNorm

EXAMPLE = EXAMPLES_DIR / "native_megakernel"
POOL = 11 * oc.PAGE_BYTES  # an A10's (or an RTX 5070 Ti's) page pool
L2 = 6 << 20  # an A10's L2
PEAKS = {"gpu": "Test GPU", "dram_gbps": 500.0, "tflops": {"float32": 20.0}}


def chain(n: int, layers: int, seed: int = 0) -> NormGemvChain:
    torch.manual_seed(seed)
    ref = NormGemvChain(n, layers).to(torch.bfloat16).eval()
    with torch.no_grad():
        for name, weight in ref.named_parameters():
            weight.normal_(1.0, 0.1) if name.startswith("norms.") else weight.normal_(0.0, n**-0.5)
    return ref


def mlp(hidden: int = 256, inter: int = 4096, seed: int = 0) -> GatedMlpBlock:
    torch.manual_seed(seed)
    block = GatedMlpBlock(hidden, inter).to(torch.bfloat16).eval()
    with torch.no_grad():
        for name, weight in block.named_parameters():
            fan_in = weight.shape[-1]
            weight.normal_(1.0, 0.1) if "norm" in name else weight.normal_(0.0, fan_in**-0.5)
    return block


def row(n: int, rows: int = 1, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(rows, n, dtype=torch.bfloat16)


# ------------------------------------------------------------------ a CPU interpreter


def _f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<i", word))[0]


def _norm(x: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    """The opcodes' RMSNorm: fp32, rounded to bf16, times the bf16 weight, rounded."""
    inv = torch.rsqrt(x.float().pow(2).mean() + eps)
    return (g.float() * (x.float() * inv).bfloat16().float()).bfloat16()


def execute(sched: mks.Schedule, tensors: list[torch.Tensor]) -> None:
    """Every instruction of ``sched``, in the simulator's start order, with the semantics of
    the example's opcodes (include/mk_ops.cuh) on the flat tensor table ``tensors``."""
    sim = simulate.simulate(sched)
    assert sim.ok, sim.problems()
    t = [x.view(-1) for x in tensors]
    for i in sorted(range(len(sched.instrs)), key=lambda i: (sim.start[i], i)):
        ins, a = sched.instrs[i], sched.instrs[i].args
        if ins.opcode == oc.RMSNORM:
            x, xo, g, go, eps, out, oo, n = a[:8]
            t[out][oo : oo + n] = _norm(t[x][xo : xo + n], t[g][go : go + n], _f32(eps))
        elif ins.opcode == oc.GEMV:
            x, xo, g, go, eps, out, oo, res, ro, row0, rows, k, k0, klen, part, po = a[:16]
            pf = ins.prefetch
            assert pf is not None
            w = t[pf.tensor][pf.offset // 2 : (pf.offset + pf.nbytes) // 2].view(rows, klen)
            h = t[x][xo : xo + k]
            if g >= 0:  # the norm over all K, the slice normalised
                h = _norm(h, t[g][go : go + k], _f32(eps))
            acc = w.float() @ h[k0 : k0 + klen].float()
            if part >= 0:
                t[part][po + row0 : po + row0 + rows] = acc
            else:
                y = acc.bfloat16()
                if res >= 0:
                    y = (t[res][ro + row0 : ro + row0 + rows].float() + y.float()).bfloat16()
                t[out][oo + row0 : oo + row0 + rows] = y
        elif ins.opcode in (oc.RESIDUAL, oc.GLU):
            x, xo, y, yo, out, oo, i0, n = a[:8]
            lhs, rhs = t[x][xo + i0 : xo + i0 + n].float(), t[y][yo + i0 : yo + i0 + n].float()
            if ins.opcode == oc.GLU:
                assert a[8] == oc.GLU_SILU
                lhs = torch.nn.functional.silu(lhs).bfloat16().float()
                t[out][oo + i0 : oo + i0 + n] = (lhs * rhs).bfloat16()
            else:
                t[out][oo + i0 : oo + i0 + n] = (lhs + rhs).bfloat16()
        elif ins.opcode == oc.SPLITK_REDUCE:
            part, po, splits, stride, row0, rows, out, oo, res, ro = a[:10]
            acc = sum(
                t[part][po + s * stride + row0 : po + s * stride + row0 + rows]
                for s in range(splits)
            )
            y = acc.bfloat16()
            if res >= 0:
                y = (t[res][ro + row0 : ro + row0 + rows].float() + y.float()).bfloat16()
            t[out][oo + row0 : oo + row0 + rows] = y
        elif ins.opcode == oc.ARGMAX:
            x, xo, n, out, oo = a[:5]
            t[out][oo] = int(torch.argmax(t[x][xo : xo + n].float()))
        else:
            raise AssertionError(f"opcode {ins.opcode}")


def run_plan(plan: captured.Plan, module: nn.Module, *inputs: torch.Tensor, queues: int = 13):
    """The plan's schedule executed on the CPU: its outputs."""
    tensors = plan.tensors(module)
    for binding, x in zip(plan.inputs, inputs, strict=True):
        assert binding is not None
        plan.bind(tensors, binding).copy_(x)
    execute(plan.schedule(queues), tensors)
    return [plan.bind(tensors, b) for b in plan.outputs if b is not None]


# ------------------------------------------------------------------ the example's chain


@pytest.mark.parametrize(("n", "layers"), [(1024, 4), (4096, 2), (1024, 28)])
def test_the_chains_dag_equals_the_one_the_example_declares_by_hand(
    n, layers, tmp_path, monkeypatch
):
    """Ops (opcode, tiles, every tile's arguments and weight prefetch), edges, counter targets
    and the serialised program: the captured chain is the hand-declared one, from nothing but
    one recorded call (16 rows per tile at 1024; 8 at 4096, where 16 rows exceed the pool)."""
    monkeypatch.setenv(proj.CACHE_ENV, str(tmp_path))
    mod = proj.import_project(EXAMPLE)
    ref = chain(n, layers)
    rows = mod.tile_rows(n, POOL)
    weights = sum(layer.weight.numel() * 2 for layer in ref.layers)
    hint = oc.l2_hint(weights, L2)
    hand_ops, hand_edges = mod.chain_dag(n, layers, rows, 1e-6, hint)
    plan = captured.from_module(ref, (row(n),), pool_bytes=POOL, l2_bytes=L2, peaks=PEAKS)
    assert plan.ok and plan.unsupported == []
    assert [(o.opcode, o.tiles) for o in plan.ops] == [(o.opcode, o.tiles) for o in hand_ops]
    for mine, theirs in zip(plan.ops, hand_ops, strict=True):
        for t in range(mine.tiles):
            assert mine.tile_args(t) == theirs.tile_args(t), (mine.name, t)
            assert mine.tile_prefetch(t) == theirs.tile_prefetch(t), (mine.name, t)
    index = {op.name: i for i, op in enumerate(plan.ops)}
    hand_index = {op.name: i for i, op in enumerate(hand_ops)}
    assert [(index[e.producer], index[e.consumer], e.needs) for e in plan.edges] == [
        (hand_index[e.producer], hand_index[e.consumer], e.needs) for e in hand_edges
    ]
    for queues in (72, 13):
        mine, theirs = plan.schedule(queues), mks.build(hand_ops, hand_edges, queues)
        assert [c.target for c in mine.counters] == [c.target for c in theirs.counters]
        assert [c.target for c in mine.counters] == [n // rows] * (layers - 1)
        assert mine.to_bytes() == theirs.to_bytes()
    # what each op stands for, the tensor table, the call's input and output in it
    first = plan.info[0]
    assert first.family == "gemv" and first.fused == ("RMSNorm prologue", "residual epilogue")
    assert [c.split()[1] for c in first.covers] == [
        "_to_copy", "pow", "mean", "add", "rsqrt", "mul", "_to_copy", "mul", "linear", "add"
    ]  # fmt: skip
    assert [(s.kind, s.name) for s in plan.slots] == [
        *(("weight", f"layers.{i}.weight") for i in range(layers)),
        *(("const", f"norms.{i}.weight") for i in range(layers)),
        ("arena", "bf16 arena"),
    ]
    assert plan.slots[-1].numel == (layers + 1) * n
    assert plan.inputs == [Binding(2 * layers, 0, (1, n), "bfloat16")]
    assert plan.outputs == [Binding(2 * layers, layers * n, (1, n), "bfloat16")]
    assert hint == (mks.EVICT_FIRST if weights > L2 // 2 else mks.EVICT_NORMAL)


def test_the_captured_chain_computes_the_module():
    ref = chain(1024, 4)
    x = row(1024)
    plan = captured.from_module(ref, (x,), pool_bytes=POOL)
    (out,) = run_plan(plan, ref, x)
    with torch.inference_mode():
        want = ref(x)
    torch.testing.assert_close(out.float(), want.float(), atol=0.05, rtol=0.02)


def test_the_same_input_gives_byte_identical_queues():
    """Fresh modules (other storage addresses) and fresh recordings: the same program."""
    for make, n, options in (
        (lambda: chain(1024, 6), 1024, {}),
        (mlp, 256, {}),
        (mlp, 256, {"split_k": "auto", "queues": 128}),
        (lambda: chain(512, 3), 512, {"fuse": False}),
    ):
        plans = [captured.from_module(make(), (row(n),), peaks=PEAKS, **options) for _ in range(3)]
        programs = {p.schedule(64).to_bytes() for p in plans}
        assert len(programs) == 1
        assert mks.check_words(plans[0].schedule(64).words()) is None
        assert plans[0].schedule(64).to_bytes() != plans[0].schedule(63).to_bytes()


# ------------------------------------------------------------------ a gated MLP


def test_a_gated_mlp_maps_to_fused_gemvs_a_gated_activation_and_chunked_counters():
    block = mlp()
    x = row(256)
    plan = captured.from_module(block, (x,), queues=64)
    assert plan.ok
    assert [(i.name, i.family, i.tiles) for i in plan.info] == [
        ("gemv0", "gemv", 256),  # gate: 16 rows per tile, the norm in its prologue
        ("gemv1", "gemv", 256),  # up: the same norm recomputed per GEMV
        ("glu0", "glu", 2),  # silu(gate) * up, 2048 elements per tile
        ("gemv2", "gemv", 32),  # down: K = 4096, 8 rows fit the pool, the residual added
    ]
    assert plan.info[0].fused == ("RMSNorm prologue (per GEMV)",)
    assert plan.info[3].fused == ("residual epilogue",)
    halves = TileMap((tuple(range(128)), tuple(range(128, 256))))
    assert [(e.producer, e.consumer, e.needs) for e in plan.edges] == [
        ("gemv0", "glu0", halves),
        ("gemv1", "glu0", halves),
        ("glu0", "gemv2", mks.ALL),
    ]
    sched = plan.schedule(64)
    # a GLU tile waits for the half of gate and of up it reads: two counters each (128)
    assert [(c.op, c.target) for c in sched.counters] == [
        ("gemv0", 128), ("gemv0", 128), ("gemv1", 128), ("gemv1", 128), ("glu0", 2)
    ]  # fmt: skip
    glu = next(i for i in sched.instrs if i.op == "glu0" and i.tile == 1)
    assert [sched.counters[c].tiles for c, _ in glu.waits] == [
        tuple(range(128, 256)),
        tuple(range(128, 256)),
    ]
    assert glu.args[8] == oc.GLU_SILU and glu.args[6:8] == (2048, 2048)
    assert simulate.check(sched, draws=1000, seed=225) == []
    (out,) = run_plan(plan, block, x)
    with torch.inference_mode():
        want = block(x)
    torch.testing.assert_close(out.float(), want.float(), atol=0.05, rtol=0.02)


def test_a_gemv_too_narrow_for_the_sms_is_split_over_k_with_a_reduce():
    """``split_k="auto"``: the down projection's 32 tiles fill less than half of 128 queues,
    so it splits K four ways (64 partial tiles, a packed copy of its weights, each tile's rows
    contiguous) and a reduce per row tile sums them and adds the residual."""
    block = mlp()
    x = row(256)
    plan = captured.from_module(block, (x,), queues=128, split_k="auto")
    assert [(i.name, i.family, i.tiles) for i in plan.info] == [
        ("gemv0", "gemv", 256),
        ("gemv1", "gemv", 256),
        ("glu0", "glu", 2),
        ("gemv2", "gemv", 64),
        ("reduce0", "splitk_reduce", 16),
    ]
    assert plan.info[3].shape == "M=1, N=256, K=4096, 16 rows per tile, K split 4 ways"
    assert plan.info[4].fused == ("residual epilogue",)
    packed = next(s for s in plan.slots if s.kind == "packed")
    assert (packed.name, packed.splits) == ("down_proj.weight", 4)
    edges = {(e.producer, e.consumer): e.needs for e in plan.edges}
    assert edges[("glu0", "gemv2")] == TileMap(tuple((t // 32,) for t in range(64)))
    assert edges[("gemv2", "reduce0")] == TileMap(
        tuple(tuple(s * 16 + r for s in range(4)) for r in range(16))
    )
    tensors = plan.tensors(block)
    want = block.down_proj.weight.view(256, 4, 1024).permute(1, 0, 2).contiguous()
    assert torch.equal(tensors[packed.index], want)
    sched = plan.schedule(128)
    assert simulate.check(sched, draws=1000, seed=7) == []
    (out,) = run_plan(plan, block, x, queues=128)
    with torch.inference_mode():
        torch.testing.assert_close(out.float(), block(x).float(), atol=0.05, rtol=0.02)


# ------------------------------------------------------------------ other forms


class TorchNorm(nn.Module):
    """``F.rms_norm`` (the weight applied before the bf16 rounding) → Linear → residual."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.proj = nn.Linear(n, n, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.rms_norm(x, (x.shape[-1],), self.weight, 1e-5)
        return x + self.proj(h)


class Bf16Norm(nn.Module):
    """An RMSNorm computed in bf16 throughout."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.proj = nn.Linear(n, n, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
        return self.proj(h) + x


class Head(nn.Module):
    """norm → LM head → greedy token (argmax)."""

    def __init__(self, n: int, vocab: int) -> None:
        super().__init__()
        self.norm = RMSNorm(n)
        self.head = nn.Linear(n, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.head(self.norm(x)), dim=-1)


class Mixed(nn.Module):
    """Ops no family takes: a LayerNorm, a biased Linear, a softmax, casts, a scaled mul."""

    def __init__(self, n: int = 256) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(n)
        self.proj = nn.Linear(n, n)
        self.q = nn.Linear(n, n, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.softmax(self.q(x).float(), -1)
        return self.proj(self.ln(x)) * 0.5 + h.to(x.dtype)


def test_other_norm_forms_map_with_a_note_on_their_rounding():
    for module, note in (
        (TorchNorm(512), "scales by the weight before rounding"),
        (Bf16Norm(512), "normalised in bf16 by the reference"),
    ):
        module = module.to(torch.bfloat16).eval()
        x = row(512)
        plan = captured.from_module(module, (x,))
        assert plan.ok and [i.family for i in plan.info] == ["gemv"], plan.describe()
        assert plan.info[0].fused == ("RMSNorm prologue", "residual epilogue")
        assert any(note in n for n in plan.info[0].notes), plan.info[0].notes
        (out,) = run_plan(plan, module, x)
        with torch.inference_mode():
            torch.testing.assert_close(out.float(), module(x).float(), atol=0.06, rtol=0.03)


def test_unfused_ops_keep_their_own_opcodes_and_edges():
    ref = chain(512, 3)
    x = row(512)
    plan = captured.from_module(ref, (x,), fuse=False)
    assert [i.name for i in plan.info] == [
        f"{family}{layer}" for layer in range(3) for family in ("rmsnorm", "gemv", "residual")
    ]
    assert [(e.producer, e.consumer, e.needs) for e in plan.edges][:5] == [
        ("rmsnorm0", "gemv0", mks.ALL),
        ("gemv0", "residual0", mks.ALL),
        ("residual0", "rmsnorm1", mks.ALL),
        ("rmsnorm1", "gemv1", mks.ALL),
        ("residual0", "residual1", mks.ALL),  # x1 + layer1(...): the residual reads x1 too
    ]
    assert simulate.check(plan.schedule(13), draws=1000, seed=3) == []
    (out,) = run_plan(plan, ref, x)
    with torch.inference_mode():
        torch.testing.assert_close(out.float(), ref(x).float(), atol=0.05, rtol=0.02)


def test_skinny_rows_an_argmax_and_a_gemm_too_wide_for_gemv_tiles():
    ref = chain(256, 2)
    x = row(256, rows=3)
    plan = captured.from_module(ref, (x,))
    assert [(i.family, i.tiles, i.shape) for i in plan.info] == [
        ("gemv", 48, "M=3, N=256, K=256, 16 rows per tile")
    ] * 2  # three activation rows: three GEMV tiles per weight tile
    (out,) = run_plan(plan, ref, x)
    with torch.inference_mode():
        torch.testing.assert_close(out.float(), ref(x).float(), atol=0.05, rtol=0.02)

    head = Head(256, 1024).to(torch.bfloat16).eval()
    plan = captured.from_module(head, (row(256),))
    assert [(i.family, i.tiles) for i in plan.info] == [("gemv", 64), ("argmax", 1)]
    assert [s.dtype for s in plan.slots if s.kind == "arena"] == ["bfloat16", "int64"]
    tensors = plan.tensors(head)
    plan.bind(tensors, plan.inputs[0]).copy_(row(256))
    execute(plan.schedule(8), tensors)
    logits = tensors[-2][256:].float()  # the head's output slice, after the input's
    with torch.inference_mode():
        torch.testing.assert_close(logits, head.head(head.norm(row(256))).float().view(-1))
    assert int(plan.bind(tensors, plan.outputs[0])) == int(torch.argmax(logits))

    wide = captured.from_module(chain(256, 1), (row(256, rows=16),))
    assert not wide.ok
    assert "a GEMM of 16 rows" in wide.unsupported[0].reason


def test_unsupported_ops_are_reported_with_their_names_and_why():
    module = Mixed().to(torch.bfloat16).eval()
    plan = captured.from_module(module, (row(256),))
    assert not plan.ok
    assert [(u.op, u.name, u.module) for u in plan.unsupported] == [
        (1, "_to_copy", ""),
        (2, "_softmax", ""),
        (3, "native_layer_norm", "ln"),
        (4, "linear", "proj"),
        (5, "mul", ""),
        (6, "_to_copy", ""),
    ]
    reasons = [u.reason for u in plan.unsupported]
    assert "softmax" in reasons[1] and "LayerNorm" in reasons[2] and "bias" in reasons[3]
    assert "#4 `aten.linear` in `proj` (1x256 bf16, 256x256 bf16, 256 bf16" in (
        plan.unsupported[3].describe()
    )
    assert [i.family for i in plan.info] == ["gemv", "residual"]  # what does map
    with pytest.raises(captured.CaptureError, match=r"6 unsupported op\(s\): #1 `aten._to_copy`"):
        plan.schedule(8)
    assert "Unsupported (6" in plan.describe()


def test_costs_come_from_the_roofline_of_the_peaks_and_say_where_from():
    ref = chain(1024, 2)
    plan = captured.from_module(ref, (row(1024),), queues=72, peaks=PEAKS, peaks_source="peaks")
    tile = 16 * 1024 * 2 + 2 * 1024 * 2 + 4 * 16  # weights, x and the norm's, residual + out
    assert plan.info[0].bytes == 64 * tile
    assert plan.ops[0].cost == pytest.approx(tile * 72 / 500e3)  # a 1/72 share of 500 GB/s
    assert plan.roofline.unit == "us" and "Test GPU" in plan.cost_source
    assert "1/72 share" in plan.cost_source and "20 TFLOP/s fp32 FMA" in plan.cost_source
    sim = simulate.simulate(plan.schedule(72))
    assert sim.makespan == pytest.approx(2 * tile * 72 / 500e3)  # one tile per queue a layer
    bare = captured.from_module(ref, (row(1024),))
    assert bare.ops[0].cost == float(tile) and bare.roofline.unit == "bytes"
    assert "relative costs" in bare.cost_source


# ------------------------------------------------------------------ captures, command line


def test_a_capture_file_gives_the_schedule_on_the_command_line(tmp_path, capsys):
    from kernel_agent.native.megakernel import schedule
    from kernel_agent.profiling.capture import capture_calls

    path = tmp_path / "capture.pt"
    ref = chain(512, 3)
    capture_calls(ref, [((row(512, rows=4),), {}, 0), ((row(512),), {}, 64)], path)
    plan = captured.from_capture(path)  # the case with the most calls: one row
    assert "case 1" in plan.label and [i.tiles for i in plan.info] == [32] * 3
    peaks = tmp_path / "peaks.json"
    peaks.write_text(json.dumps(PEAKS))
    argv = ["--from-capture", str(path), "--queues", "16", "--peaks", str(peaks), "--draws", "50"]
    assert schedule.main(argv) == 0
    text = capsys.readouterr().out
    assert text.startswith("Op DAG of `NormGemvChain.forward case 1")
    assert "Test GPU" in text and "no deadlock, no early start" in text
    assert "makespan" in text and "critical path" in text
    assert schedule.main([*argv, "--json", "--case", "0"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [o["tiles"] for o in out["ops"]] == [128] * 3 and out["schedule"]["problems"] == []
    edge = out["edges"][0]  # four rows: a tile of row m waits only for row m's producers
    assert (edge["producer"], edge["consumer"]) == ("gemv0", "gemv1")
    assert edge["needs"][0] == list(range(32)) and edge["needs"][-1] == list(range(96, 128))

    mixed = tmp_path / "mixed.pt"
    capture_calls(Mixed().to(torch.bfloat16).eval(), [((row(256),), {}, 1)], mixed)
    assert schedule.main(["--from-capture", str(mixed), "--queues", "8", "--no-peaks"]) == 1
    assert "Unsupported (6: no schedule until mapped)" in capsys.readouterr().out
    assert schedule.main(["--from-capture", str(tmp_path / "none.pt"), "--queues", "8"]) == 2
