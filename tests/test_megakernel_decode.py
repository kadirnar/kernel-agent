"""Split-KV attention and the on-device token advance of the megakernel kit (issue #225,
milestones 6 and 7): the chunking rule, the split-KV ops' counters, the decode step's schedule
(simulated over random SM speeds, the advance after every reader of the step state), the
opcode ABI mirrors and the decode interpreter compiled for every architecture (CPU only; the
GPU runs are in test_megakernel_decode_gpu.py)."""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.native import megakernel
from kernel_agent.native import project as proj
from kernel_agent.native.megakernel import decode as mkd
from kernel_agent.native.megakernel import opcodes as oc
from kernel_agent.native.megakernel import schedule as mks
from kernel_agent.native.megakernel import simulate

EXAMPLE = EXAMPLES_DIR / "native_megakernel"


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    """The example's entry module (imported on the CPU: nothing is compiled)."""
    cache = tmp_path_factory.mktemp("native-cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(proj.CACHE_ENV, str(cache))
        yield proj.import_project(EXAMPLE)


# ------------------------------------------------------------------ the chunking rule


@pytest.mark.parametrize(
    ("splits", "chunk", "capacity"),
    [(1, 0, 300), (3, 0, 700), (18, 0, 4096), (4, 128, 512), (5, 100, 500)],
)
def test_the_chunks_cover_every_key_of_every_length_once(splits, chunk, capacity):
    for length in [*range(0, 260), capacity - 1, capacity]:
        keys, nonempty = mkd.kv_split(length, splits, chunk)
        assert nonempty <= splits
        covered = [k for s in range(nonempty) for k in range(s * keys, min((s + 1) * keys, length))]
        assert covered == list(range(length)), (length, keys, nonempty)
        assert all(s * keys < length for s in range(nonempty))  # every listed chunk has keys
        if chunk == 0:
            assert keys % mkd.CHUNK_GRANULE == 0 and keys >= mkd.CHUNK_GRANULE
            assert keys < -(-length // splits) + mkd.CHUNK_GRANULE or length <= splits
        else:
            assert keys == chunk
    with pytest.raises(ValueError, match="splits"):
        mkd.kv_split(10, 0)


def test_q_blocks_split_a_group_evenly():
    assert mkd.q_blocks(1) == [(0, 1)]
    assert mkd.q_blocks(4) == [(0, 4)]
    assert mkd.q_blocks(5) == [(0, 3), (3, 2)]
    assert mkd.q_blocks(7) == [(0, 4), (4, 3)]
    assert mkd.q_blocks(8) == [(0, 4), (4, 4)]
    assert mkd.q_blocks(8, 2) == [(0, 2), (2, 2), (4, 2), (6, 2)]
    with pytest.raises(ValueError):
        mkd.q_blocks(4, mkd.MAX_Q_BLOCK + 1)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"head_dim": 96}, "head_dim"),
        ({"chunk": 100}, "do not cover"),
        ({"q_block": 5}, "q_block"),
        ({"splits": 0}, ">= 1"),
        ({"group": 8, "splits": 300}, "combine"),
    ],
)
def test_split_kv_layouts_the_opcodes_do_not_have_are_refused(kwargs, error):
    base = {"kv_heads": 2, "group": 4, "head_dim": 64, "capacity": 1000, "splits": 4}
    with pytest.raises(mks.ScheduleError, match=error):
        mkd.SplitKV(**{**base, **kwargs})


# ------------------------------------------------------------------ split-KV in a schedule


def attention_layer(split: mkd.SplitKV, layers: int = 2) -> tuple[list[mks.Op], list[mks.Edge]]:
    """Per layer: a projection, the RoPE / KV append (one tile per KV head), the split-KV
    attention and its combine, the output projection."""
    ops: list[mks.Op] = []
    edges: list[mks.Edge] = []
    for layer in range(layers):
        p = f"l{layer}."
        attn, comb, to_combine = split.ops(
            p + "attn", p + "combine", (oc.ATTN_DECODE, oc.ATTN_COMBINE),
            lambda a: [a.kv_head, a.q0, a.qn, a.split], lambda h, q0, qn: [h, q0, qn],
        )  # fmt: skip
        ops += [
            mks.Op(p + "qkv", 2, 24, cost=32768.0),
            mks.Op(p + "rope", 8, split.kv_heads),
            attn,
            comb,
            mks.Op(p + "o", 2, 16, cost=32768.0),
        ]
        if layer:
            edges.append(mks.Edge(f"l{layer - 1}.o", p + "qkv"))
        edges += [
            mks.Edge(p + "qkv", p + "rope"),
            mks.Edge(p + "rope", p + "attn", split.kv_head_of),
            to_combine,
            mks.Edge(p + "combine", p + "o"),
        ]
    return ops, edges


def test_each_combine_waits_on_one_counter_of_its_kv_heads_chunks():
    split = mkd.SplitKV(kv_heads=2, group=7, head_dim=64, capacity=1000, splits=5)
    assert split.attention_tiles == 2 * 2 * 5 and split.partial_rows == 14 * 5
    sched = mks.build(*attention_layer(split), queues=13)
    by_id = sched.instrs
    for instr in by_id:
        if instr.op.endswith("combine"):
            assert len(instr.waits) == 1
            counter = sched.counters[instr.waits[0][0]]
            assert counter.target == len(counter.tiles) == 2 * 5  # q blocks x splits
            heads = {split.attention_tile(t).kv_head for t in counter.tiles}
            assert heads == {instr.tile}
            assert instr.args == (instr.tile, instr.tile * 7, 7)
        if instr.op.endswith("attn"):
            tile = split.attention_tile(instr.tile)
            assert instr.args == (tile.kv_head, tile.q0, tile.qn, tile.split)
            assert len(instr.waits) == 1  # its KV head's RoPE tile only
            counter = sched.counters[instr.waits[0][0]]
            assert counter.op.endswith("rope") and counter.tiles == (tile.kv_head,)
            assert counter.target == 1
    tiles = [split.attention_tile(t) for t in range(split.attention_tiles)]
    assert [(t.kv_head, t.q0, t.qn, t.split) for t in tiles[:6]] == [
        (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 4, 4), (0, 4, 3, 0)
    ]  # fmt: skip
    assert tiles[-1] == mkd.AttentionTile(19, 1, 11, 3, 4)


def test_split_kv_schedules_are_byte_identical_and_never_deadlock():
    split = mkd.SplitKV(kv_heads=4, group=4, head_dim=128, capacity=4096, splits=18)
    first = mks.build(*attention_layer(split, 3), queues=72)
    assert first.to_bytes() == mks.build(*attention_layer(split, 3), queues=72).to_bytes()
    assert mks.check_words(first.words()) is None
    assert simulate.check(first, draws=1000, seed=225, latency=0.05) == []
    # at a short length most chunks are empty: durations from the length, still no deadlock
    durations = mkd.attention_durations(
        {f"l{k}.attn": split for k in range(3)}, 17, ns_per_byte=0.01, fixed_ns=500.0
    )
    assert simulate.check(first, draws=200, seed=7, durations=durations) == []


def test_attention_costs_and_durations_follow_the_length():
    split = mkd.SplitKV(kv_heads=2, group=2, head_dim=64, capacity=1024, splits=8)
    for length in (0, 1, 17, 500, 1024, 5000):
        keys = sum(split.keys(t, length) for t in range(split.attention_tiles))
        assert keys == 2 * min(length, 1024)  # every key of each KV head once
        assert sum(split.tile_bytes(t, length) for t in range(split.attention_tiles)) == (
            keys * 2 * 64 * 2
        )
    attn, _, _ = split.ops("attn", "combine", (10, 11), lambda a: [], lambda h, q0, qn: [])
    assert attn.tile_cost(0) == mkd.FIXED_COST + 2 * 128 * 64 * 2  # at the capacity
    short, _, _ = split.ops("attn", "c", (10, 11), lambda a: [], lambda *a: [], length=20)
    assert short.tile_cost(0) == mkd.FIXED_COST + 2 * 16 * 64 * 2
    assert short.tile_cost(1) == mkd.FIXED_COST + 2 * 4 * 64 * 2 and short.tile_cost(2) == (
        mkd.FIXED_COST
    )
    sched = mks.build(*attention_layer(split, 2), queues=8)
    spans = []
    for length in (1, 64, 512, 1024):
        durations = mkd.attention_durations(
            {"l0.attn": split, "l1.attn": split}, length, ns_per_byte=0.05, fixed_ns=300.0,
            other={"l0.qkv": 2000.0, "l1.qkv": 2000.0},
        )  # fmt: skip
        instr = next(i for i in sched.instrs if i.op == "l0.attn" and i.tile == 7)
        assert durations(instr) == 300.0 + split.tile_bytes(7, length) * 0.05
        spans.append(simulate.simulate(sched, durations=durations).makespan)
    assert spans == sorted(spans) and spans[-1] > spans[0]


# ------------------------------------------------------------------ the example's decode step

DIMS = {"vocab": 2048, "hidden": 1024, "heads": 16, "kv_heads": 4, "head_dim": 64,
        "inter": 2048, "capacity": 512, "layers": 2}  # fmt: skip


def decode_schedule(example, queues: int = 72, **kw):
    dims = {**DIMS, **kw}
    T, W = example.decode_table(dims["layers"])
    group = dims["heads"] // dims["kv_heads"]
    split = mkd.SplitKV(dims["kv_heads"], group, dims["head_dim"], dims["capacity"], 18)
    ops, edges, readers = example.decode_program(dims, T, W, split, pool_bytes=11 * 8192, eps=1e-6)
    return mks.build(ops, edges, queues), readers, split


def test_the_decode_step_schedule_is_deterministic_and_never_deadlocks(example):
    sched, _, split = decode_schedule(example)
    again, _, _ = decode_schedule(example)
    assert sched.to_bytes() == again.to_bytes()
    assert simulate.check(sched, draws=1000, seed=225, latency=0.05) == []
    odd, _, _ = decode_schedule(example, queues=13, layers=3, kv_heads=2)
    assert simulate.check(odd, draws=300, seed=1) == []
    ops = {i.op for i in sched.instrs}
    assert {"embed", "argmax", "lm", "l1.attn", "l1.combine", "l1.rope", "l1.glu"} <= ops
    # the counters: per KV head one RoPE counter over its q / k / v rows' QKV tiles, one
    # combine counter over its chunks, one GLU counter per tile over its gate and up rows
    by_op: dict[str, list[mks.Counter]] = {}
    for counter in sched.counters:
        by_op.setdefault(counter.op, []).append(counter)
    assert [c.target for c in by_op["l0.qkv"]] == [4 * 64 // 16 + 2 * 64 // 16] * 4
    assert [c.target for c in by_op["l0.rope"]] == [1] * 4
    assert [c.target for c in by_op["l0.attn"]] == [split.splits] * 4
    assert [c.target for c in by_op["l0.gate_up"]] == [2 * 256 // 16] * 8
    assert [c.target for c in by_op["lm"]] == [2048 // 16]
    assert "argmax" not in by_op  # nobody waits on the step's last instruction


def test_the_token_advance_follows_every_reader_of_the_step_state(example):
    sched, readers, _ = decode_schedule(example)
    assert set(readers) == {"embed"} | {f"l{k}.{op}" for k in range(2)
                                         for op in ("rope", "attn", "combine")}  # fmt: skip
    assert mkd.check_advance(sched, readers, "argmax") == []
    last = next(i for i in sched.instrs if i.op == "argmax")
    assert mkd.ancestors(sched, last.id) == {i.id for i in sched.instrs} - {last.id}
    # a step whose last combine does not wait for its attention tiles: they read the position
    # but nothing after them waits for them, so the advance could overwrite it under them;
    # the same holds for every reader before them (the step is one chain of edges: an output
    # projection reads the residual stream through it)
    T, W = example.decode_table(DIMS["layers"])
    split = mkd.SplitKV(4, 4, 64, 512, 18)
    ops, edges, readers = example.decode_program(DIMS, T, W, split, pool_bytes=11 * 8192, eps=1e-6)
    cut = [e for e in edges if e.producer != "l1.attn"]
    broken = mks.build(ops, cut, 72)
    problems = mkd.check_advance(broken, readers, "argmax")
    flagged = {p.split("[")[0] for p in problems}
    assert flagged == {"embed", "l0.rope", "l0.attn", "l0.combine", "l1.rope", "l1.attn"}
    assert sum(p.startswith("l1.attn[") for p in problems) == split.attention_tiles
    assert mkd.check_advance(broken, readers, "nothing") == ["no instruction of nothing"]


def test_a_mistargeted_combine_counter_is_a_deadlock_with_its_instruction(example):
    sched, _, _ = decode_schedule(example)
    victim = next(i for i in sched.instrs if i.op == "l1.combine" and i.tile == 2)
    counter, target = victim.waits[0]
    broken = dataclasses.replace(victim, waits=((counter, target + 1),))
    bad = dataclasses.replace(
        sched, instrs=tuple(broken if i.id == victim.id else i for i in sched.instrs)
    )
    problems = simulate.check(bad, draws=3)
    assert problems and f"instruction {victim.id} (l1.combine[2])" in problems[0]


def test_the_decode_tiles_fit_the_page_pool(example):
    pool = 11 * 8192  # a 99 KB GPU (A10, RTX 5070 Ti)
    assert example.gemv_rows(1536, 1024, pool) == 16
    assert example.gemv_rows(1024, 4096, pool) == 8  # 16 rows of 4096 are 128 KB
    assert example.gemv_rows(24, 1024, pool) == 12
    assert example.gemv_rows(1024, 1024, 1024) == 0
    assert example.tiles_covering([(0, 16), (1024, 1040)], 16) == [0, 64]
    assert example.tiles_covering([(8, 40)], 16) == [0, 1, 2]
    assert example.default_splits(72, 4) == 18 and example.default_splits(8, 16) == 1


# ------------------------------------------------------------------ ABI mirrors


def _enum(text: str, name: str) -> list[str]:
    body = re.search(rf"enum {name} : int \{{([^}}]*)\}}", text)
    assert body is not None, name
    return [
        w.split("=")[0].strip() for w in body.group(1).replace("\n", " ").split(",") if w.strip()
    ]


def test_the_decode_opcodes_and_their_argument_slots_mirror_the_header(example):
    header = (EXAMPLE / "include" / "mk_decode.cuh").read_text()
    ops = (EXAMPLE / "include" / "mk_ops.cuh").read_text()
    codes = dict(re.findall(r"(\w+) = (\d+)", re.search(r"enum DecodeOpcode[^}]*\}", header)[0]))
    assert {name: getattr(oc, name) for name in codes} == {k: int(v) for k, v in codes.items()}
    assert sorted(int(v) for v in codes.values()) == list(range(oc.GLU + 1, oc.GLU + 5))
    assert all(getattr(example, name) == getattr(oc, name) for name in codes)  # re-exported
    words = _enum(ops, "StepWord")
    assert [getattr(mkd, w) for w in words] == [0, 1] and len(words) == mkd.STEP_WORDS
    builders = {
        "EmbedArg": ("EM_", oc.embed_args, header),
        "RopeArg": ("RK_", oc.rope_kv_args, header),
        "AttnArg": ("AT_", oc.attn_args, header),
        "CombineArg": ("CB_", oc.combine_args, header),
        "ArgmaxArg": ("A_", oc.argmax_args, ops),
    }
    for enum, (prefix, builder, text) in builders.items():
        slots = _enum(text, enum)
        assert len(slots) <= mks.ARGS, enum
        kwargs = {slot.removeprefix(prefix).lower(): k for k, slot in enumerate(slots)}
        if enum == "AttnArg":  # the scale is a float word
            kwargs["scale"] = 0.0
            want = list(range(len(slots)))
            want[slots.index("AT_SCALE")] = 0
        else:
            want = list(range(len(slots)))
        assert builder(**kwargs) == want, enum
    assert {torch.bfloat16: oc.DTYPE_BF16, torch.float16: oc.DTYPE_FP16} == example.DTYPES


# ------------------------------------------------------------------ the decode interpreter


def _nvcc() -> str | None:
    root = toolchain._pip_cuda_root()
    if root is not None and (root / "bin" / "nvcc").exists():
        return str(root / "bin" / "nvcc")
    return shutil.which("nvcc")


ARCHS = ("sm_80", "sm_86", "sm_89", "sm_120a")


def _compile(nvcc: str, arch: str, src: Path, out: Path) -> tuple[int, str, str]:
    out.mkdir()
    cmd = [
        nvcc, f"-arch={arch}", "-cubin", "--keep", "--keep-dir", str(out), "-o",
        str(out / "tu.cubin"), "-O3", "-DKA_MK_TRACE=1", "-allow-unsupported-compiler",
        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK", f"-I{EXAMPLE / 'include'}",
        f"-I{megakernel.include_dir()}", f"-I{proj.toolkit_include()}", str(src),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    ptx = next(iter(out.glob("*.ptx")), None)
    return proc.returncode, proc.stderr[-3000:], ptx.read_text() if ptx else ""


def test_the_decode_interpreter_compiles_for_every_architecture(tmp_path):
    """kernel<DecodeOps> (the generic opcodes and the decode ones) on the CPU for sm_80, sm_86,
    sm_89 and sm_120a: counters by acquire / release, no grid sync, the attention's exp2
    softmax, the page path of the arch."""
    nvcc = _nvcc()
    if nvcc is None:
        pytest.skip("no nvcc")
    src = tmp_path / "tu.cu"
    src.write_text(
        '#include "mk_ops.cuh"\n'
        "template __global__ void ka_mk::kernel<mk::DecodeOps>(ka_mk::Params);\n"
    )
    with ThreadPoolExecutor(len(ARCHS)) as pool:
        results = dict(
            zip(ARCHS, pool.map(lambda a: _compile(nvcc, a, src, tmp_path / a), ARCHS), strict=True)
        )
    for arch, (code, err, ptx) in results.items():
        assert code == 0, f"{arch}: {err}"
        assert "ld.acquire.gpu" in ptx and "red.release.gpu" in ptx, arch
        assert not re.search(r"grid_group|cudaCGSynchronize|this_grid|cg::", ptx), arch
        assert "ex2.approx" in ptx and "shfl.sync.bfly" in ptx, arch  # the split softmax
        assert "DecodeOps" in ptx, arch
        bulk = int(re.match(r"sm_(\d+)", arch).group(1)) >= 90
        assert ("cp.async.bulk" in ptx) == bulk and ("cp.async.cg" in ptx) == (not bulk), arch
