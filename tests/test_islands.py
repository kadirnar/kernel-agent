"""Islands per target in ``improve`` (``--islands``, ``workers.py``, issue #189): island UCB,
lineages, migration (inspirations), culling and reseeding, the arm cap, and the dry run
with islands in sequence and at once.

No GPU and no Claude: synthetic ledger rows, and ``dryrun.World``'s simulated sessions (in
virtual time for ``--agents N``).
"""

import asyncio
import dataclasses
import itertools

import pytest

from kernel_agent import board, charts, cli, dryrun, improve, ledger, orchestrator, workers
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.scheduler import KERNEL, Arm, Policy, Running, assign, build_arms
from kernel_agent.workspace import read_json, write_json


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No real toolchain probing and no charts."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path, islands=None, *, virtual=False):
    config = OptimizeConfig(
        model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, dossier=False, seeds_per_target=islands
    )
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    return orch, dryrun.World(orch, virtual=virtual)


def loop(orch, world, **icfg):
    """The improve loop (``--agents N`` in virtual time when the world is virtual)."""
    improver = Improver(orch, ImproveConfig(**icfg), require_capture=False, live_charts=False)

    async def main():
        if world.virtual:
            with world.driving():
                return await improver.improve()
        return await improver.improve()

    with world.installed():
        reason = asyncio.run(main())
    return improver, reason


def row(exp, worker, speedup, *, session="s#1", backend="triton", parent="", idea="", t="t"):
    """A kernel ledger row of ``worker`` (``speedup`` None: a failed evaluation)."""
    return {
        "exp": exp,
        "target": t,
        "worker": str(worker or ""),
        "session": session,
        "status": "discard" if speedup else "incorrect",
        "correct": speedup is not None,
        "speedup": speedup,
        "spread": 0.0,
        "backend": backend,
        "snapshot": f"{exp:03d}_{backend}.py",
        "parent": parent,
        "idea": idea,
    }


# ------------------------------------------------------------------ an island's measures


def test_measure_lineage_gain_stale_and_migrations():
    rows = [
        row(1, 1, 1.5, session="w1#1"),
        row(2, 2, 1.2, session="w2#2", backend="cuda"),
        row(3, 1, 2.0, session="w1#3", parent="history/001_triton.py"),
        row(4, 2, None, session="w2#4", backend="cuda"),
        row(5, 2, 1.9, session="w2#5", backend="cuda", parent="history/002_cuda.py"),
        # island 2 builds on island 1's 2.0x (a migration): its lineage is at 2.0x then
        row(6, 2, 1.95, session="w2#6", parent="history/003_triton.py"),
        row(7, 2, 2.2, session="w2#6", parent="history/003_triton.py"),
        row(8, 1, 1.9, session="w1#7", parent="history/003_triton.py"),
    ]
    one = workers.measure(workers.Island("t", 1, backends=("triton",)), rows)
    assert (one.evals, one.best, one.head, one.streak) == (3, 2.0, "003_triton.py", 1)
    assert one.sessions == 3 and one.stale == 1  # its last session found nothing
    assert one.gain == pytest.approx(1 - 1 / 2.0)
    two = workers.measure(workers.Island("t", 2, backends=("cuda",)), rows, running=True)
    assert two.evals == 5 and two.best == 2.2 and two.head == "007_triton.py"
    assert two.backend == "triton" and two.stale == 0 and two.running
    # its own gains: 1.0 -> 1.2 -> 1.9, then 2.0 (the migration, not its gain) -> 2.2
    assert two.gain == pytest.approx((1 - 1 / 1.2) + (1 / 1.2 - 1 / 1.9) + (1 / 2.0 - 1 / 2.2))
    assert workers.adoptions(rows) == 1  # island 2 built on island 1's 003 (twice: one pair)
    # a reseeded island counts its rows after its generation started, from its parent
    later = workers.Island("t", 2, generation=2, since=6, parent="history/003_triton.py")
    later = workers.measure(later, rows)
    assert later.evals == 1 and later.best == 2.2 and later.head == "007_triton.py"


def test_island_ucb_explores_then_pays():
    fresh = [workers.Island("t", k) for k in (1, 2, 3)]
    assert [i.k for i in workers.rank(fresh, 0.7, 1.0)] == [1, 2, 3]  # ties: the lowest k
    paying = workers.Island("t", 1, evals=12, gain=0.5)
    idle = workers.Island("t", 2, evals=12, gain=0.05)
    untried = workers.Island("t", 3)
    ranked = workers.rank([paying, idle, untried], 0.7, 1.0)
    assert ranked[0] is untried  # the exploration bonus of an untried island
    assert ranked.index(paying) < ranked.index(idle)
    stale = workers.Island("t", 4, evals=12, gain=0.5, stale=3)
    assert workers.rank([paying, stale], 0.7, 1.0)[0] is paying  # decay ** stale sessions
    paying.running = True
    assert workers.choose(workers.rank([paying, idle], 0.7, 1.0)) is idle
    idle.running = True
    assert workers.choose([paying, idle]) is None


def test_inspirations_offer_the_elite_and_another_cell():
    rows = [
        row(1, 1, 1.5, idea="fuse"),
        row(2, 2, 2.1, backend="cuda", idea="splitk"),
        row(3, 3, 1.7, backend="cute", idea="fuse"),
        row(4, 3, 0.9, backend="tilelang", idea="naive"),  # slower than the reference
        row(5, 2, 2.0, backend="cuda", idea="persist"),
    ]
    island = workers.measure(workers.Island("t", 1, backends=("triton",)), rows)
    offered = workers.inspirations(island, rows)
    assert [(e["snapshot"], e["island"]) for e in offered] == [
        ("002_cuda.py", 2),
        ("003_cute.py", 3),
    ]
    assert "best result of the other islands" in offered[0]["why"]
    assert "`cute`" in offered[1]["why"]  # another backend: another cell
    # the island that has the best: no elite to offer it, only another cell's best
    top = workers.measure(workers.Island("t", 2, backends=("cuda",)), rows)
    assert [e["snapshot"] for e in workers.inspirations(top, rows)] == ["003_cute.py"]
    # one backend only: another idea is the other cell
    same = [row(1, 1, 1.5, idea="fuse"), row(2, 2, 1.8, idea="fuse"), row(3, 2, 1.6, idea="tile")]
    island = workers.measure(workers.Island("t", 1, backends=("triton",)), same)
    assert [e["snapshot"] for e in workers.inspirations(island, same)] == [
        "002_triton.py",
        "003_triton.py",
    ]
    lines = workers.island_lines(island, [island], 1.8, "002_triton.py", offered)
    text = "\n".join(lines)
    assert "## Your island (island 1 of 1, generation 1)" in text
    assert "## Inspirations" in text and "* `history/002_cuda.py`: 2.100x (`cuda`" in text


def test_migration_is_due_for_a_stalled_island_every_m_evaluations():
    island = workers.Island("t", 1, evals=4, stale=1, inspired=0)
    assert workers.migration_due(island, 4, 4)
    assert not workers.migration_due(island, 4, 0)  # --migrate-every 0: off
    assert not workers.migration_due(island, 3, 4)  # the target's evaluations since the last
    island.inspired = 2
    assert not workers.migration_due(island, 5, 4) and workers.migration_due(island, 6, 4)
    climbing = workers.Island("t", 1, evals=8, stale=0)
    assert not workers.migration_due(climbing, 20, 4)  # still improving: its own direction
    fresh = workers.Island("t", 1, evals=2, stale=1)
    assert not workers.migration_due(fresh, 20, 4)  # its own evaluations first


def test_cull_and_reseed(tmp_path):
    stuck = workers.Island("t", 2, evals=12, stale=2, best=1.3)
    assert "below the target's 2.000x" in workers.cull_reason(stuck, 2.0, 0.15)
    assert workers.cull_reason(stuck, 2.0, 0.0) is None  # --cull-gap 0: never
    assert workers.cull_reason(stuck, 1.5, 0.15) is None  # within the gap
    for change in ({"evals": 7}, {"stale": 1}, {"running": True}):
        assert workers.cull_reason(dataclasses.replace(stuck, **change), 2.0, 0.15) is None
    spec = {"approach": "fuse", "backends": ["triton", "cuda"]}
    available = ["cuda", "triton", "cute"]
    assert [workers.direction(spec, i, available)[1][0] for i in (1, 2, 3, 4)] == [
        "triton",
        "cuda",
        "cute",
        "triton",  # the backends in turn
    ]
    new = workers.reseed(stuck, spec, available, 3, ("009_cuda.py", 2.0), 40, 30)
    assert (new.k, new.generation, new.since, new.inspired) == (2, 2, 40, 30)
    assert new.parent == "history/009_cuda.py" and new.parent_speedup == 2.0
    assert new.origin == "reseed" and new.backends[0] == "cute"
    note = workers.island_note("t", new, [workers.Island("t", 1), new], 4)
    assert "# Island 2 of 2 on `t`" in note and "Reseeded (generation 2)" in note
    assert 'parent="history/009_cuda.py"' in note and "island 1:" in note
    assert workers.Island.of("t", 2, new.state()).state() == new.state()


# ------------------------------------------------------------------ the loop


def test_a_stagnant_island_is_reseeded_from_the_best(tmp_path):
    orch, _ = make(tmp_path, islands=2)
    run = orch.run
    improver = Improver(orch, ImproveConfig(), require_capture=False, live_charts=False)

    def evaluate(worker, session, speedups, parent=None):
        for s in speedups:
            result = {"status": "ok", "correct": True, "speedup": s, "cases": []}
            name = f"{len(ledger.rows(run)) + 1:03d}_v.py"
            ledger.record_kernel(
                run,
                "attn",
                result,
                snapshot=name,
                hypothesis="h",
                parent=parent,
                worker=worker,
                session=session,
                source="import triton\n",
            )

    evaluate(1, "kernel-attn-w1#1", [1.5, 2.0])
    evaluate(2, "kernel-attn-w2#2", [1.3, 1.1, 1.2, 1.2])  # a new island best, then nothing
    evaluate(2, "kernel-attn-w2#3", [1.2, 1.2, 1.25, 1.2])
    evaluate(2, "kernel-attn-w2#4", [1.2, 1.1, 1.2, 1.2])
    notes = workers.prepare(run, "attn", 2) / "NOTES.md"
    notes.write_text("- tried tiles\n")
    arms = improver.arms()
    attn = next(a for a in arms if a.id == "attn")
    assert attn.max_sessions == 2 and attn.best == 2.0  # the arm cap: its islands
    improver.state["islands"] = {}
    rec = improver._open_slice(attn, arms)
    (cull,) = improver.state["culls"]
    assert cull["island"] == 2 and cull["generation"] == 2 and cull["best"] == 1.3
    assert cull["parent"] == f"history/{attn.best_snapshot}" and cull["target_best"] == 2.0
    assert cull["archived"].endswith("workers/2/NOTES.gen1.md")
    assert notes.read_text() == "" and (notes.parent / "NOTES.gen1.md").read_text()
    saved = improver.state["islands"]["attn"]["2"]
    assert saved["origin"] == "reseed" and saved["since"] == len(ledger.rows(run))
    island = next(i for i in improver.islands(attn) if i.k == 2)
    assert island.evals == 0 and island.best == 2.0 and island.head == attn.best_snapshot
    assert rec["island"] in (1, 2) and rec["label"] == f"kernel-attn-w{rec['island']}#1"
    events = [e for e in ledger.events(run) if e["event"] == "island_reseeded"]
    assert events and events[0]["island"] == 2


def test_islands_one_is_the_classic_loop(tmp_path):
    def key(orch):
        rows = [
            (r["target"], r["status"], r["speedup"], r["worker"]) for r in ledger.rows(orch.run)
        ]
        state = read_json(orch.run.root / "improve.json")
        slices = [(s["label"], s["agent"], s["evals"]) for s in state["slices"]]
        return rows, slices, sorted(state)

    a, world = make(tmp_path / "a")
    loop(a, world, max_slices=5)
    b, world = make(tmp_path / "b", islands=1)
    loop(b, world, max_slices=5)
    assert key(a) == key(b)
    state = read_json(b.run.root / "improve.json")
    assert not {"islands", "migrations", "culls"} & set(state)
    assert all("island" not in s for s in state["slices"])
    assert not (b.run.target("attn") / workers.DIR).exists()


def concurrent_islands(path, **icfg):
    orch, world = make(path, islands=2, virtual=True)
    loop(orch, world, agents=3, **icfg)
    return orch, world


def test_islands_run_at_once_and_share_their_best(tmp_path):
    """``--agents 3 --islands 2`` in virtual time: a target runs one session per island at
    once, never two on one island; stalled islands get the others' best as inspirations and
    build on them."""
    orch, world = concurrent_islands(tmp_path, max_slices=30)
    state = read_json(orch.run.root / "improve.json")
    t0 = world.clock.t0
    spans = {}
    for s in state["slices"]:
        if "island" in s:
            spans.setdefault(s["arm"], []).append((s["island"], s["started"] - t0, s["ended"] - t0))
    together = 0
    for arm, items in spans.items():
        for (k1, a1, b1), (k2, a2, b2) in itertools.combinations(items, 2):
            if a1 < b2 and a2 < b1:  # overlapping sessions of one target
                assert k1 != k2, f"two sessions on island {k1} of {arm} at once"
                together += 1
    assert together  # the islands of a target ran at once
    assert state["migrations"] and all(m["inspirations"] for m in state["migrations"])
    inspired = [x for x in world.sessions if "\n## Inspirations" in x["prompt"]]
    assert inspired and all("# Island " in x["prompt"] for x in inspired)
    rows = ledger.rows(orch.run)
    assert sum(workers.adoptions([r for r in rows if r["target"] == t]) for t in spans)
    beside = [x for x in world.sessions if "other islands of your target" in x["prompt"]]
    assert beside  # the coordinator's note names the sibling islands
    assert "* islands of `attn` (`--islands`): 1: " in orch.run.report.read_text()


def test_concurrent_islands_are_reproducible(tmp_path):
    def key(orch):
        rows = [
            (r["target"], r["status"], r["speedup"], r["session"]) for r in ledger.rows(orch.run)
        ]
        state = read_json(orch.run.root / "improve.json")
        slices = [(s["label"], s.get("island"), s["evals"]) for s in state["slices"]]
        return rows, slices, state["islands"]

    a, _ = concurrent_islands(tmp_path / "a", max_slices=12)
    b, _ = concurrent_islands(tmp_path / "b", max_slices=12)
    assert key(a) == key(b)


# ------------------------------------------------------------------ scheduler, board, CLI


def test_the_arm_cap_is_its_islands(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    arms = build_arms(run, Policy(systems=False), [], islands={"attn": 3})
    caps = {a.id: a.max_sessions for a in arms}
    assert caps == {"attn": 3, "mlp": 1, "rmsnorm": 1}
    attn = next(a for a in arms if a.id == "attn")
    running = [
        Running("attn", KERNEL, "kernel-attn-w1", 4),
        Running("attn", KERNEL, "kernel-attn-w2", 4),
    ]
    only = [attn]
    assert [a.id for a in assign(only, 3, running, Policy())] == ["attn"]  # its third island
    assert assign(only, 3, [*running, Running("attn", KERNEL, "kernel-attn-w3", 4)], Policy()) == []
    single = Arm("x", KERNEL, 10.0)
    assert len(assign([single], 3, [], Policy())) == 1  # one session per arm without islands


def test_claims_close_with_their_own_sessions_release(tmp_path):
    from kernel_agent.workspace import RunDir

    run = RunDir.create(tmp_path, "org/m")
    write_json(run.target("attn") / "spec.json", {"id": "attn", "module_class": "Attn"})
    with board.opened(run):
        board.claim(run, "kernel-attn-w1#1", "kernel", "attn", [("Attn", None)])
        board.claim(run, "kernel-attn-w2#2", "kernel", "attn", [("Attn", None)])
        board.release(run, "kernel-attn-w1#1", "attn", [("Attn", None)])
        open_ = [e for e in board.open_claims(board.load(run)) if e["kind"] == board.CLAIM]
        assert [e["tags"]["session"] for e in open_] == ["kernel-attn-w2#2"]
        board.release(run, "kernel-attn-w2#2", "attn", [("Attn", None)])
        assert not [e for e in board.open_claims(board.load(run)) if e["kind"] == board.CLAIM]


def test_cli_islands(monkeypatch, tmp_path, capsys):
    seen = []
    run = dryrun.create_run(OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path))

    async def fake(ref, cfg, icfg, **kwargs):
        seen.append((cfg, icfg))
        return run

    monkeypatch.setattr(improve, "improve", fake)
    argv = ["improve", "org/m", "--islands", "3", "--migrate-every", "0", "--cull-gap", "0.2"]
    assert cli.main(argv) == 0
    cfg, icfg = seen[0]
    assert cfg.seeds_per_target == 3 and icfg.migrate_every == 0 and icfg.cull_gap == 0.2
    assert cli.main(["improve", "org/m"]) == 0
    cfg, icfg = seen[1]
    assert cfg.seeds_per_target is None  # --islands 1 by default
    assert (icfg.migrate_every, icfg.cull_gap) == (workers.MIGRATE_EVERY, workers.CULL_GAP)
    for bad in (["--cull-gap", "1.5"], ["--migrate-every", "-1"], ["--islands", "0"]):
        with pytest.raises(SystemExit):
            cli.main(["improve", "org/m", *bad])
    assert "--cull-gap" in capsys.readouterr().err
