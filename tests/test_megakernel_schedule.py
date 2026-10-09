"""The megakernel kit (kernel_agent/native/megakernel, issue #225): schedules, the simulator,
the serialised program, and the header compiled to PTX for every architecture (CPU only;
the GPU runs are in test_megakernel_gpu.py)."""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kernel_agent import toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.native import megakernel
from kernel_agent.native import project as proj
from kernel_agent.native.megakernel import schedule as mks
from kernel_agent.native.megakernel import simulate
from kernel_agent.native.megakernel.schedule import ALL, SAME, Edge, Op, Prefetch, shard, span

EXAMPLE = EXAMPLES_DIR / "native_megakernel"


def decode_step(layers: int = 2) -> tuple[list[Op], list[Edge]]:
    """An op DAG shaped like a decode step: per layer a norm, a split-K projection and its
    reduce, a one-to-one elementwise op, an MLP whose down projection waits per chunk of
    its input (4 counters), a residual; an argmax at the end. Costs and weights differ per
    op so the placement is not trivially round robin."""
    ops: list[Op] = []
    edges: list[Edge] = []
    prev = None
    for layer in range(layers):
        p = f"l{layer}."
        ops += [
            Op(p + "norm", 1, 1, cost=0.5, args=[[layer, 0, 7]]),
            Op(
                p + "qkv",
                2,
                24,
                cost=[1.0 + 0.1 * (t % 3) for t in range(24)],
                weights=lambda t, layer=layer: Prefetch(layer, t * 4096, 4096, mks.EVICT_FIRST),
                args=lambda t: [t, t * 2],
            ),
            Op(p + "reduce", 5, 6, cost=0.3),
            Op(p + "act", 4, 6, cost=0.2),
            Op(p + "up", 2, 16, cost=1.5, weights=[Prefetch(9, t * 8192, 8192) for t in range(16)]),
            Op(p + "down", 2, 8, cost=1.2),
            Op(p + "residual", 4, 2, cost=0.1),
        ]
        if prev is not None:
            edges.append(Edge(prev, p + "norm", ALL))
        edges += [
            Edge(p + "norm", p + "qkv", ALL),
            Edge(p + "qkv", p + "reduce", span(4)),  # 4 K-splits per output tile
            Edge(p + "reduce", p + "act", SAME),
            Edge(p + "act", p + "up", ALL),
            Edge(p + "up", p + "down", lambda t: range((t // 2) * 4, (t // 2) * 4 + 4)),
            Edge(p + "down", p + "residual", lambda t: range(t * 4, t * 4 + 4)),
            Edge(p + "act", p + "residual", shard(3)),  # a second producer of the residual
        ]
        prev = p + "residual"
    ops.append(Op("argmax", 6, 1, cost=0.4))
    edges.append(Edge(prev, "argmax", ALL))
    return ops, edges


# ------------------------------------------------------------------ the schedule


def test_the_same_op_dag_gives_byte_identical_queues():
    first = mks.build(*decode_step(), queues=13)
    again = mks.build(*decode_step(), queues=13)  # fresh Op / Edge objects
    assert first.to_bytes() == again.to_bytes()
    assert first.queues == again.queues
    assert mks.check_words(first.words()) is None
    assert mks.build(*decode_step(), queues=12).to_bytes() != first.to_bytes()
    words = first.words()
    assert words[mks.H_MAGIC] == mks.MAGIC and words[mks.H_QUEUES] == 13
    assert words[mks.H_INSTRS] == len(first.instrs) == sum(len(q) for q in first.queues)
    assert len(first.to_bytes()) == 4 * len(words)
    assert "instructions" in first.describe() and "13 queues" in first.describe()


def test_every_counter_target_equals_its_producers_tile_count():
    ops, edges = decode_step()
    sched = mks.build(ops, edges, queues=13)
    signals: dict[int, list[int]] = {}
    for instr in sched.instrs:
        if instr.signal != mks.NONE:
            signals.setdefault(instr.signal, []).append(instr.id)
    for counter in sched.counters:
        producers = signals.get(counter.id, [])
        assert counter.target == len(producers) == len(counter.tiles)
        assert {sched.instrs[i].tile for i in producers} == set(counter.tiles)
        assert {sched.instrs[i].op for i in producers} == {counter.op}
    # each consumer waits on exactly the chunks that cover the tiles it needs
    by_name = {op.name: op for op in ops}
    for instr in sched.instrs:
        needed = {(sched.instrs[d].op, sched.instrs[d].tile) for d in instr.deps}
        covered = {
            (sched.counters[c].op, t) for c, target in instr.waits for t in sched.counters[c].tiles
        }
        assert needed == covered, instr
        for c, target in instr.waits:
            assert target == sched.counters[c].target
    # chunk shapes: ALL -> one counter, SAME -> one per tile, the down projection's chunks of 4
    chunks = {(c.op, len(c.tiles)) for c in sched.counters}
    assert ("l0.norm", 1) in chunks and ("l0.reduce", 1) in chunks
    assert ("l0.up", 4) in chunks and ("l0.down", 4) in chunks
    assert sum(1 for c in sched.counters if c.op == "l0.act") == 2  # residual: shard(3) halves
    assert not [i for i in sched.instrs if i.op == "argmax" and i.signal != mks.NONE]
    assert by_name["l0.qkv"].tile_prefetch(3) == Prefetch(0, 3 * 4096, 4096, mks.EVICT_FIRST)
    # the first wait is inline in the row, the others in the wait list
    words, row0 = sched.words(), sched.words()[mks.H_INSTR_OFF]
    for instr in sched.instrs:
        row = words[row0 + instr.id * mks.WORDS : row0 + (instr.id + 1) * mks.WORDS]
        assert row[mks.ID] == instr.id and row[mks.OP] == instr.opcode
        assert (row[mks.W0_COUNTER], row[mks.W0_TARGET]) == (
            instr.waits[0] if instr.waits else (mks.NONE, 0)
        )
        assert row[mks.WAIT_COUNT] == max(0, len(instr.waits) - 1)
        assert row[mks.ARG0 : mks.ARG0 + len(instr.args)] == list(instr.args)
        if instr.prefetch:
            pf = instr.prefetch
            assert row[mks.PF_TENSOR : mks.PF_HINT + 1] == [
                pf.tensor, pf.offset // 16, pf.nbytes, pf.hint
            ]  # fmt: skip


def test_queues_run_in_wave_order_and_balance_the_estimate():
    sched = mks.build(*decode_step(), queues=13)
    order = {name: i for i, name in enumerate(sched.ops)}
    for queue in sched.queues:
        waves = [order[sched.instrs[i].op] for i in queue]
        assert waves == sorted(waves)  # a topological order on every queue
    loads = [sum(sched.instrs[i].cost for i in q) for q in sched.queues]
    assert max(loads) < 1.5 * sum(loads) / len(loads)  # serial ops do not pile up on one queue
    assert sched.est_makespan == max(i.est_finish for i in sched.instrs)
    one = mks.build([Op("a", 1, 5, cost=[5, 1, 1, 1, 1])], [], queues=2)
    assert [[one.instrs[i].tile for i in q] for q in one.queues] == [[0], [1, 2, 3, 4]]


@pytest.mark.parametrize(
    ("ops", "edges", "error"),
    [
        ([Op("a", 1, 2), Op("a", 1, 2)], [], "two ops named a"),
        ([Op("a", 1, 2), Op("b", 1, 3)], [Edge("a", "b", SAME)], "equal tile counts"),
        ([Op("a", 1, 2), Op("b", 1, 2)], [Edge("a", "b"), Edge("b", "a")], "cycle"),
        ([Op("a", 1, 2, wave=1), Op("b", 1, 2, wave=1)], [Edge("a", "b")], "must precede"),
        ([Op("a", 1, 2), Op("b", 1, 2)], [Edge("a", "b", lambda t: [t + 1])], "producer has 2"),
        ([Op("a", 1, 2), Op("b", 1, 2)], [Edge("a", "b", lambda t: [])], "needs no producer"),
        ([Op("a", 1, 1, args=[[0] * (mks.ARGS + 1)])], [], "arguments"),
        ([Op("a", 1, 1, weights=[Prefetch(0, 8, 16)])], [], "multiples of 16"),
        ([Op("a", 1, 1, weights=[Prefetch(0, 0, 16, hint=7)])], [], "hint"),
        ([Op("a", 1, 1)], [Edge("a", "x")], "unknown op"),
    ],
)
def test_ops_and_edges_that_make_no_schedule(ops, edges, error):
    with pytest.raises(mks.ScheduleError, match=error):
        mks.build(ops, edges, queues=4)


def test_check_words_catches_a_damaged_program():
    words = mks.build(*decode_step(1), queues=5).words()
    assert mks.check_words(words) is None
    bad = list(words)
    bad[mks.H_MAGIC] = 0
    assert "magic" in mks.check_words(bad)
    bad = list(words)
    bad[bad[mks.H_INSTR_OFF] + mks.WORDS + mks.ID] = 0
    assert "has id 0" in mks.check_words(bad)
    bad = list(words)
    bad[bad[mks.H_INSTR_OFF] + mks.SIGNAL] = 10**6
    assert "signal counter" in mks.check_words(bad)
    assert "total" in mks.check_words(words[:-1])


# ------------------------------------------------------------------ the simulator


def test_simulation_over_a_thousand_speed_draws_finds_no_deadlock_or_early_start():
    sched = mks.build(*decode_step(), queues=13)
    assert simulate.check(sched, draws=1000, seed=225, latency=0.05) == []
    sim = simulate.simulate(sched)
    assert sim.ok and sim.makespan > 0 and sim.critical
    assert sim.critical[-1] == max(range(len(sim.finish)), key=lambda i: sim.finish[i])
    text = simulate.report(sched, sim, " us")
    assert text.startswith("makespan") and "critical path" in text and "l0.up" in text


def test_a_mistargeted_counter_is_reported_as_a_deadlock_with_the_instruction_ids():
    sched = mks.build(*decode_step(), queues=13)
    victim = next(i for i in sched.instrs if i.op == "l1.down" and i.tile == 5)
    counter, target = victim.waits[0]
    broken = dataclasses.replace(victim, waits=((counter, target + 1), *victim.waits[1:]))
    bad = dataclasses.replace(
        sched, instrs=tuple(broken if i.id == victim.id else i for i in sched.instrs)
    )
    sim = simulate.simulate(bad)
    assert not sim.ok and sim.deadlock
    first = sim.deadlock[0]  # the cause first; the queues waiting on it after
    assert (first.instr, first.counter, first.value, first.target, first.stuck) == (
        victim.id, counter, target, target + 1, True
    )  # fmt: skip
    assert len(sim.deadlock) > 1 and not any(b.stuck for b in sim.deadlock[1:])
    problems = simulate.check(bad, draws=3)
    assert problems and f"instruction {victim.id} (l1.down[5])" in problems[0]
    assert simulate.report(bad, sim).startswith("deadlock")


def test_a_counter_target_set_too_low_starts_a_consumer_early():
    sched = mks.build(*decode_step(), queues=13)
    victim = next(i for i in sched.instrs if i.op == "l0.reduce" and i.tile == 2)
    counter, target = victim.waits[0]
    early = dataclasses.replace(victim, waits=((counter, target - 1), *victim.waits[1:]))
    bad = dataclasses.replace(
        sched, instrs=tuple(early if i.id == victim.id else i for i in sched.instrs)
    )
    problems = simulate.check(bad, draws=1000, seed=1)
    assert problems and f"instruction {victim.id} (l0.reduce[2]) started" in problems[0]


def test_trace_costs_predict_the_run_and_show_the_overlap():
    sched = mks.build(
        [Op("a", 1, 4, weights=[Prefetch(0, 0, 16)] * 4), Op("b", 2, 2)],
        [Edge("a", "b", ALL)],
        queues=2,
    )
    # rows: prefetch, start, ready, landed, end (ns), smid
    trace = []
    for instr in sched.instrs:
        base = 1000 * instr.id
        pf = base - 50 if instr.op == "a" and instr.tile != 3 else base + 300
        end = base + 700 if instr.op == "a" else base + 300
        trace.append([pf, base, base + 200, base + 250, end, 0, 0, 0])
    costs = simulate.costs_from_trace(sched, trace)
    assert costs == {"a": 500.0, "b": 100.0}
    summary = simulate.trace_summary(sched, trace)
    assert summary["ops"]["a"]["overlap"] == 0.75 and summary["ops"]["b"]["overlap"] is None
    assert summary["ops"]["a"]["wait_ns"] == 200.0 and summary["ops"]["a"]["land_ns"] == 50.0
    assert summary["ops"]["a"]["run_ns"] == 450.0
    predicted = simulate.simulate(sched, durations=costs)
    assert predicted.makespan == pytest.approx(2 * 500 + 100)  # two "a" per queue, then "b"


# ------------------------------------------------------------------ the header and the example


def _nvcc() -> str | None:
    root = toolchain._pip_cuda_root()
    if root is not None and (root / "bin" / "nvcc").exists():
        return str(root / "bin" / "nvcc")
    return shutil.which("nvcc")


ARCHS = ("sm_80", "sm_89", "sm_90a", "sm_100a", "sm_120a")


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


def test_the_header_and_the_example_compile_to_ptx_for_every_architecture(tmp_path):
    """The interpreter with the example's opcodes, compiled on the CPU (nvcc -cubin: PTX and
    ptxas) for sm_80, sm_89, sm_90a, sm_100a and sm_120a: acquire / release counters, no
    cooperative-groups grid sync; bulk copies from sm_90, cp.async below."""
    nvcc = _nvcc()
    if nvcc is None:
        pytest.skip("no nvcc")
    src = tmp_path / "tu.cu"
    src.write_text(
        '#include "mk_ops.cuh"\ntemplate __global__ void ka_mk::kernel<mk::Ops>(ka_mk::Params);\n'
    )
    with ThreadPoolExecutor(len(ARCHS)) as pool:
        results = dict(
            zip(ARCHS, pool.map(lambda a: _compile(nvcc, a, src, tmp_path / a), ARCHS), strict=True)
        )
    for arch, (code, err, ptx) in results.items():
        assert code == 0, f"{arch}: {err}"
        assert "ld.acquire.gpu" in ptx and "red.release.gpu" in ptx, arch
        assert not re.search(r"grid_group|cudaCGSynchronize|this_grid|cg::", ptx), arch
        bulk = int(re.match(r"sm_(\d+)", arch).group(1)) >= 90
        assert ("cp.async.bulk" in ptx) == bulk and ("cp.async.cg" in ptx) == (not bulk), arch


def test_the_example_is_a_valid_project_that_packs_into_a_bundle(tmp_path):
    project = proj.Project.from_dir(EXAMPLE)
    assert project.manifest.name == "native_megakernel" and project.manifest.kind == "kernel"
    assert "include/mk_ops.cuh" in project.files and "csrc/megakernel.cu" in project.files
    bundle = proj.write_bundle(EXAMPLE, tmp_path)
    payload = proj.read_bundle(bundle)
    assert payload is not None and payload["digest"] == project.digest
    assert proj.main(["check", str(EXAMPLE)]) == 0
    assert proj.main(["pack", str(EXAMPLE), "-o", str(tmp_path / "b.py")]) == 0
    assert (tmp_path / "b.py").read_text() == bundle.read_text()
    from kernel_agent import gpu_arch

    assert gpu_arch.example_requirement(EXAMPLE)[0] == "sm_80+"


def test_every_project_build_has_the_kit_on_its_include_path():
    dirs = proj.toolkit_includes()
    assert (dirs[0] / "ka_launch.cuh").is_file() and (dirs[1] / megakernel.HEADER).is_file()
    assert dirs[1] == megakernel.include_dir()


def test_the_example_mirrors_the_header_abi():
    """The opcode numbers and argument slots of include/mk_ops.cuh, and the instruction
    words of ka_mk.cuh, match their Python mirrors."""
    header = (megakernel.include_dir() / megakernel.HEADER).read_text()
    words = re.search(r"OP = 0, ([^}]*)\};", header).group(1)
    names = ["OP", *[w.strip() for w in words.replace("\n", " ").split(",") if w.strip()]]
    assert names[-1] == "ARG0" and len(names) - 1 == mks.ARG0
    for i, name in enumerate(names):
        assert getattr(mks, name) == i, name
    ops = (EXAMPLE / "include" / "mk_ops.cuh").read_text()
    codes = dict(re.findall(r"(\w+) = (\d+)", re.search(r"enum Opcode[^}]*\}", ops).group(0)))
    entry = (EXAMPLE / "candidate.py").read_text()
    mirror = re.search(r"^(NOP, .*) = range\((\d+)\)$", entry, re.M)
    assert mirror is not None
    assert [n.strip() for n in mirror.group(1).split(",")] == list(codes)
    assert [int(v) for v in codes.values()] == list(range(int(mirror.group(2))))
    gemv = re.search(r"enum GemvArg[^}]*\}", ops).group(0)
    slots = re.findall(r"G_\w+", gemv)
    assert len(slots) == 18 and len(slots) <= mks.ARGS
