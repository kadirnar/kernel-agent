"""``kernel-agent improve --agents N`` (``coordinator.py``, issue #183): concurrent agent
sessions in a virtual-time dry run (``dryrun.VirtualClock``), slot assignment with virtual
pulls, the round barrier, USD reservations, circuit breakers and the shared rate gate.

No GPU and no Claude: the sessions are ``dryrun.World``'s simulated ones; their GPU jobs go
through the real GPU job queue (``gpuqueue.holding``) in simulated time.
"""

import asyncio
import itertools
import time
from pathlib import Path

import pytest

from kernel_agent import (
    charts,
    cli,
    coordinator,
    dryrun,
    gpuqueue,
    improve,
    ledger,
    orchestrator,
    scheduler,
)
from kernel_agent.agent import runner
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.scheduler import (
    KERNEL,
    NATIVE,
    SYSTEMS,
    Arm,
    Policy,
    Running,
    assign,
    rank,
    virtual_pulls,
)
from kernel_agent.workspace import read_json, read_jsonl, write_json

GPU_KINDS = ("eval", "e2e", "integration", "reprofile", "capture")


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No real toolchain probing and no charts."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path, **cfg):
    cfg = {"dossier": False, **cfg}
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, **cfg)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    return orch, dryrun.World(orch, virtual=True)


def concurrent(orch, world, agents=3, **icfg):
    """The improve loop with ``agents`` sessions at once, in virtual time."""
    improver = Improver(
        orch, ImproveConfig(agents=agents, **icfg), require_capture=False, live_charts=False
    )

    async def main():
        with world.driving():
            return await improver.improve()

    with world.installed():
        reason = asyncio.run(main())
    return improver, reason


def spans(state, t0):
    """(label, agent, round, start, end) of every slice and research session, in seconds."""
    out = [
        (s["label"], s["agent"], s["round"], s["started"] - t0, s["ended"] - t0)
        for s in state["slices"]
    ]
    out += [
        (r["label"], f"research-{r['arm']}", r["round"], r["started"] - t0, r["ended"] - t0)
        for r in state["research"]
    ]
    return out


def most_at_once(intervals):
    """The most intervals (start, end) that overlap at one time."""
    points = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    now = peak = 0
    for _, step in points:
        now += step
        peak = max(peak, now)
    return peak


@pytest.fixture(scope="module")
def three(tmp_path_factory):
    """One dry run with ``--agents 3`` and two rounds, shared by the invariant tests."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
        mp.setattr(charts, "available", lambda: False)
        orch, world = make(tmp_path_factory.mktemp("three"))
        improver, reason = concurrent(orch, world, rounds=2)
    return orch, world, improver, reason


# ------------------------------------------------------------------ the dry run


def test_concurrent_dry_run_reaches_the_end(three):
    orch, world, _, reason = three
    state = read_json(orch.run.root / "improve.json")
    assert reason.startswith("every arm has stopped") and state["finished"]["reason"] == reason
    assert state["coordinator"] == {"agents": 3, "peak": 3, "rate_limit_waits": 0}
    assert [r["n"] for r in state["rounds"]] == [1, 2]
    final = read_json(orch.run.root / "integration.json")["final"]
    assert final["passed"] and final["speedup"] > 1.8
    assert any(i.get("background") for i in state["integrations"])  # beside the sessions
    sessions = spans(state, world.clock.t0)
    assert most_at_once([(a, b) for *_, a, b in sessions]) == 3
    assert "* --agents 3: at most 3 sessions at once" in orch.run.report.read_text()


def test_gpu_jobs_never_overlap(three):
    orch, *_ = three
    log = read_jsonl(orch.run.root / gpuqueue.FILE)
    start = {e["id"]: e for e in log if e["state"] == "start"}
    done = [e for e in log if e["state"] == "done"]
    jobs = sorted((start[e["id"]]["ts"], e["ts"]) for e in done)
    assert {start[e["id"]]["kind"] for e in done} >= {"eval", "e2e", "integration"}
    assert len(jobs) == len(start) > 50
    assert all(b[0] >= a[1] - 1e-3 for a, b in itertools.pairwise(jobs))
    assert any(e.get("wait_s") for e in start.values())  # they did queue


def test_every_row_and_record_is_a_sessions(three):
    orch, _, improver, _ = three
    state = read_json(orch.run.root / "improve.json")
    records = [*state["slices"], *state["research"], *state.get("dossiers", [])]
    assert all(r["status"] != "running" for r in records)
    mine = {label for s in state["slices"] for label in s["sessions"]}
    mine |= {r["label"] for r in state["research"]}
    rows = ledger.rows(orch.run)
    agents = [r for r in rows if r["backend"] != "integrate" and r["status"] != "re-evaluated"]
    assert agents and all(r["session"] in mine for r in agents)
    assert sum(s["evals"] for s in state["slices"]) == len(agents)
    assert all("queue_s" in s for s in state["slices"])
    # no two sessions of one agent name at once
    by_agent: dict[str, list[tuple[float, float]]] = {}
    for _, agent, _, a, b in spans(state, 0.0):
        by_agent.setdefault(agent, []).append((a, b))
    assert all(most_at_once(times) == 1 for times in by_agent.values())
    assert improver.doing is None


def test_round_barrier_waits_for_running_sessions(three):
    orch, world, *_ = three
    state = read_json(orch.run.root / "improve.json")
    t0 = world.clock.t0
    started = state["rounds"][1]["started"] - t0
    first = [s for s in spans(state, t0) if s[2] == 1]
    second = [s for s in spans(state, t0) if s[2] == 2]
    assert most_at_once([(a, b) for *_, a, b in first]) >= 2  # round 1 ran sessions at once
    assert max(b for *_, b in first) <= started  # ... and the round waited for every one
    assert second and min(a for *_, a, _ in second) >= started
    events = ledger.events(orch.run)
    reprofile = next(e["ts"] - t0 for e in events if e["event"] == "reprofile")
    assert max(b for *_, b in first) <= reprofile


def test_sessions_write_only_their_own_files(three):
    orch, world, *_ = three
    state = read_json(orch.run.root / "improve.json")
    assert world.writes and set(world.policies) >= {s["label"] for s in state["slices"]}
    for label, path in world.writes:
        if label not in world.policies:  # research sessions: exact files (writable)
            assert label.startswith("research-")
            continue
        roots, excluded = world.policies[label]
        assert any(path.is_relative_to(r) for r in roots), (label, path)
        assert not any(path.is_relative_to(x) for x in excluded), (label, path)
    systems = next(p for lab, p in world.policies.items() if lab.startswith("systems"))
    assert systems == ([orch.run.transforms_dir], [orch.run.transforms_dir / "native"])
    # sessions that ran at the same time wrote different files
    times = {label: (a, b) for label, _, _, a, b in spans(state, world.clock.t0)}
    files: dict[str, set[Path]] = {}
    for label, path in world.writes:
        files.setdefault(label, set()).add(path)
    for x in files:
        for y in files:
            if x < y and x in times and y in times:
                (a, b), (c, d) = times[x], times[y]
                if a < d and c < b:
                    assert not files[x] & files[y], (x, y)


def test_staggered_starts_and_digests_name_the_other_sessions(three):
    orch, world, *_ = three
    state = read_json(orch.run.root / "improve.json")
    t0 = world.clock.t0
    first = {}
    for s in state["slices"]:
        first.setdefault(s["agent"], s["started"] - t0)
    kernels = sorted(v for k, v in first.items() if k.startswith("kernel-"))
    assert first["systems"] == pytest.approx(kernels[0])  # another role: no wait
    assert kernels[1] - kernels[0] >= dryrun.FIRST_MESSAGE_S  # after the first one streamed
    kernel = [s for s in world.sessions if s["name"].startswith("kernel-")]
    assert "## Sessions running beside this one (--agents 3)" in kernel[1]["prompt"]
    assert kernel[0]["name"] != kernel[1]["name"]
    assert kernel[0]["system"] == kernel[1]["system"]  # one cached prefix for the role (#181)


def test_virtual_dry_run_is_reproducible(tmp_path):
    def key(orch):
        rows = [
            (r["target"], r["status"], r["speedup"], r["session"], r["eval_s"], r["queue_s"])
            for r in ledger.rows(orch.run)
        ]
        state = read_json(orch.run.root / "improve.json")
        t0 = min(s["started"] for s in state["slices"])
        slices = [(s["label"], s["evals"], round(s["started"] - t0, 3)) for s in state["slices"]]
        return rows, slices

    a, world = make(tmp_path / "a")
    concurrent(a, world, max_slices=10)
    b, world = make(tmp_path / "b")
    concurrent(b, world, max_slices=10)
    assert key(a) == key(b)
    assert len(read_json(a.run.root / "improve.json")["slices"]) == 10  # starts counted


def test_agents_one_is_the_sequential_loop(tmp_path, monkeypatch):
    async def refuse(self):
        raise AssertionError("--agents 1 must not start the coordinator")

    monkeypatch.setattr(coordinator.Coordinator, "run", refuse)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, dossier=False)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    world = dryrun.World(orch)
    improver = Improver(orch, ImproveConfig(max_slices=2), require_capture=False)
    with world.installed():
        assert asyncio.run(improver.improve()) == "--max-slices 2 reached"
    slices = read_json(orch.run.root / "improve.json")["slices"]
    assert not any("sessions" in s or "queue_s" in s for s in slices)  # records as before


# ------------------------------------------------------------------ budgets and breakers


def test_usd_reservations(tmp_path):
    orch, _ = make(tmp_path)
    budget = Budget(orch.run, max_usd=10.0)
    assert budget.free_usd() == 10.0 and budget.waiting() is None
    budget.reserve_usd("kernel-attn#1", 4.0)
    budget.reserve_usd("systems#2", 5.0)
    assert budget.free_usd() == pytest.approx(1.0)
    assert budget.free_usd("systems#2") == pytest.approx(6.0)  # its own is its own
    cfg = budget.agent_config(OptimizeConfig(model_ref="x"), "systems#2")
    assert cfg.budget_usd_per_agent == pytest.approx(6.0)  # left minus the others' reservations
    budget.reserve_usd("kernel-mlp#3", 0.9)
    assert "USD budget reserved" in (budget.waiting() or "")
    assert "USD budget reserved" in (budget.exhausted() or "")
    assert budget.exhausted("kernel-mlp#3") is None  # 0.25 > 0.1 left besides its own: ok
    budget.release_usd("systems#2")
    assert budget.waiting() is None and budget.free_usd() == pytest.approx(5.1)
    assert budget.expected_usd("kernel") == 1.77  # no session yet: the measured median
    orch.run.root.joinpath("costs.json").write_text(
        '{"kernel-attn#1": {"usd": 1.0}, "kernel-mlp-w2#3": {"usd": 3.0}, "systems#2": {"usd": 9}}'
    )
    assert budget.expected_usd("kernel") == 2.0


def test_usd_reservations_stop_new_starts(tmp_path):
    orch, world = make(tmp_path, max_usd=2.5)
    _, reason = concurrent(orch, world)
    state = read_json(orch.run.root / "improve.json")
    assert "USD budget spent" in reason
    assert state["coordinator"]["peak"] == 1  # one session's expected cost left no room
    assert orch.budget.spent_usd() <= 2.5


def test_time_budget_holds_with_concurrent_sessions(tmp_path):
    orch, world = make(tmp_path, max_hours=4.0)  # simulated hours
    _, reason = concurrent(orch, world)
    state = read_json(orch.run.root / "improve.json")
    t0, end = world.clock.t0, world.clock.t0 + 4 * 3600
    assert reason.startswith(("time budget spent", "time left"))
    assert state["coordinator"]["peak"] == 3 and len(state["rounds"]) >= 2  # rounds use it all
    assert max(s["ended"] for s in state["slices"]) <= end  # no session ran past the budget
    assert state["finished"]["budget"]["hours"] <= 4.0
    assert state["finished"]["at"] - t0 <= 4 * 3600 + 600  # the final integration about in it


def test_a_failing_arm_trips_its_breaker_not_the_run(tmp_path):
    orch, world = make(tmp_path)

    def hook(agent, evals):
        if agent == "kernel-mlp":  # its CLI dies at the first evaluation, every time
            raise RuntimeError("claude code exited with 1")

    world.hook = hook
    _, reason = concurrent(orch, world, rounds=1)
    state = read_json(orch.run.root / "improve.json")
    mlp = [s for s in state["slices"] if s["arm"] == "mlp"]
    assert [s["status"] for s in mlp] == ["failed"] * coordinator.ARM_FAILS
    assert "mlp: failing" in reason and "in a row failed" not in reason
    after = max(s["ended"] for s in mlp)
    assert any(s["status"] == "done" and s["started"] >= after for s in state["slices"])


def test_failures_across_arms_stop_the_run(tmp_path):
    orch, world = make(tmp_path)

    def hook(agent, evals):
        raise RuntimeError("claude code exited with 1")

    world.hook = hook
    _, reason = concurrent(orch, world)
    state = read_json(orch.run.root / "improve.json")
    # the third failure drains the loop; the sessions still running end (and fail) first
    assert reason.startswith(f"{len(state['slices'])} agent sessions in a row failed")
    assert len(state["slices"]) <= 3 + 2
    assert all(s["status"] == "failed" for s in state["slices"])
    assert orch.run.report.exists()


def test_rate_gate_pauses_and_resumes_every_session(tmp_path):
    orch, world = make(tmp_path)
    world.limit = (1800.0, 1800.0)  # 30 min in, the account's limit; it resets 30 min later
    _, reason = concurrent(orch, world, rounds=1)
    state = read_json(orch.run.root / "improve.json")
    assert reason.startswith("every arm has stopped")
    assert state["coordinator"]["rate_limit_waits"] == 1  # one wait for every session
    notes = orch.run.load()["phases"]["improve"]
    assert len(notes["usage_limit"]) == 1 and notes["usage_limit"][0]["shared"]
    costs = read_json(orch.run.root / "costs.json")
    waited = [label for label, c in costs.items() if c.get("usage_limit_waits")]
    assert len(waited) >= 2 and all(costs[label]["usage_limit_waits"] == 1 for label in waited)
    events = ledger.events(orch.run)
    gate = [(e["state"], e["ts"]) for e in events if e["event"] == "rate_gate"]
    assert [g[0] for g in gate] == ["closed", "open"] and gate[1][1] - gate[0][1] > 600
    starts = [e["ts"] for e in events if e["event"] == "agent_start"]
    assert not [s for s in starts if gate[0][1] < s < gate[1][1]]  # none while it was closed
    assert [s for s in starts if s > gate[1][1]]  # and new ones after
    resumed = {s["label"]: s["status"] for s in state["slices"] if s["label"] in waited}
    assert set(resumed.values()) == {"done"}


def test_rate_gate_alone():
    async def main():
        now = [1000.0]
        slept: list[float] = []

        async def sleep(seconds):
            slept.append(seconds)
            now[0] += seconds

        changes = []
        gate = coordinator.RateGate(
            clock=lambda: now[0], sleep=sleep, on_change=lambda s: changes.append((s, now[0]))
        )
        assert gate.closed() is None
        assert gate.close(60.0, "limit") is True
        assert gate.close(30.0, "again") is False  # closed already, an earlier reset
        assert "usage limit" in (gate.closed() or "")
        await gate.wait()
        assert gate.closed() is None and gate.waits == 1
        assert changes == [("closed", 1000.0), ("open", 1060.0)]
        gate.close(10.0, "limit")
        gate.close(100.0, "a later reset extends it")
        await gate.wait()
        assert now[0] == pytest.approx(1160.0) and gate.closings == 2 and gate.waits == 2
        gate.progress()
        assert gate.waits == 0

    asyncio.run(main())


def test_interrupt_closes_every_record(tmp_path):
    orch, world = make(tmp_path)
    calls = []

    def hook(agent, evals):
        calls.append(agent)
        if len(calls) == 9:
            raise KeyboardInterrupt

    world.hook = hook
    with pytest.raises(KeyboardInterrupt):
        concurrent(orch, world)
    state = read_json(orch.run.root / "improve.json")
    records = [*state["slices"], *state["research"]]
    assert all(r["status"] != "running" for r in records)
    assert sum(r["status"] == "interrupted" for r in state["slices"]) >= 2  # all that ran
    assert state["interrupted"]["during"].count("slice") >= 2
    assert not orch.budget.reservations


def test_restart_after_a_crash_with_several_running_sessions(tmp_path):
    orch, world = make(tmp_path)
    calls = []

    def hook(agent, evals):
        calls.append(agent)
        if len(calls) == 12:
            raise KeyboardInterrupt

    world.hook = hook
    with pytest.raises(KeyboardInterrupt):
        concurrent(orch, world)
    run = orch.run
    state = read_json(run.root / "improve.json")
    cut = [s for s in state["slices"] if s["status"] == "interrupted"]
    assert len(cut) >= 2
    for s in cut:  # as a crash leaves them: running, nothing closed
        s["status"] = "running"
        for key in ("ended", "seconds", "evals", "keeps", "failures", "improved", "queue_s"):
            s.pop(key, None)
    write_json(run.root / "improve.json", state)
    before = len(ledger.rows(run))

    config = OptimizeConfig(model_ref=str(run.root), runs_dir=run.root.parent)
    asyncio.run(improve.improve(str(run.root), config, ImproveConfig(agents=3), dry_run=True))
    state = read_json(run.root / "improve.json")
    assert all(s["status"] != "running" for s in state["slices"])
    rows = ledger.rows(run)
    for s in state["slices"]:  # each closed with its own sessions' rows
        own = [r for r in rows if r["session"] in s["sessions"]]
        assert s["evals"] == len([r for r in own if r["status"] not in ledger.UNMEASURED])
    assert len(rows) > before and state["finished"]["reason"].startswith("every arm has stopped")
    assert state["interruptions"]  # the crash's record moved there


# ------------------------------------------------------------------ slots


def _arm(name, ref_ms, kind=KERNEL, **kw):
    arm = Arm(name, kind, ref_ms, estimate=2.0, **kw)
    return arm


def test_virtual_pulls_spread_the_slots():
    policy = Policy()
    a, b = _arm("a", 100.0, max_sessions=2), _arm("b", 80.0, max_sessions=2)
    ranked = rank([a, b], policy)
    assert [x.id for x in ranked] == ["a", "b"] and a.max_sessions == 2  # a could take both
    assert [x.id for x in assign([a, b], 2, [], policy)] == ["a", "b"]
    pulled = {x.id: x for x in virtual_pulls([a, b], [Running("a", KERNEL, "x", 4)], policy)}
    assert pulled["a"].score < pulled["b"].score < b.score * 2
    assert (pulled["a"].evals, pulled["a"].stale) == (4, 1) and a.evals == 0  # copies


def test_assign_eligibility():
    policy = Policy()
    arms = [_arm("a", 100.0), _arm("b", 80.0), _arm(SYSTEMS, 60.0, SYSTEMS), _arm("c", 40.0)]
    native = _arm(NATIVE, 500.0, NATIVE, held="waiting: module arms still improving (a)")
    arms.append(native)
    # one session per arm; an agent name runs once; a role's cap
    got = assign(arms, 5, [], policy, role_max={KERNEL: 2})
    assert [x.id for x in got] == ["a", "b", SYSTEMS, NATIVE]  # the held native arm: last
    research = Running("a", "research", "research-a")
    assert "a" not in [x.id for x in assign(arms, 4, [research], policy)]  # paused
    # time: no slice that does not fit
    assert assign(arms, 3, [], policy, time_left=10.0) == []
    # the coordinator's own conditions
    assert [x.id for x in assign(arms, 1, [], policy, ok=lambda x: x.id == "c")] == ["c"]


def test_overlaps():
    attn = [("Qwen3Attention", None)]
    assert coordinator.overlaps(attn, [("Qwen3Attention", "model.layers.0.self_attn")])
    assert not coordinator.overlaps(attn, [("Qwen3MLP", None)])
    assert coordinator.overlaps([("*", None)], attn)
    assert not coordinator.overlaps(
        [("Linear", "model.a")], [("Linear", "model.b")]
    )  # other instances
    assert not coordinator.overlaps([], attn)  # the systems agent claims nothing


class _Group:
    """A stand-in for the coordinator's TaskGroup: records the sessions it would start."""

    def __init__(self):
        self.names: list[str] = []

    def create_task(self, coro, name=None):
        coro.close()
        self.names.append(name)
        return None


def _filled(tmp_path, monkeypatch, *, wait, slots=1, **icfg):
    orch, world = make(tmp_path)
    improver = Improver(
        orch, ImproveConfig(agents=slots, **icfg), require_capture=False, live_charts=False
    )
    monkeypatch.setattr(
        improver, "research_due", lambda arm: "plateau" if arm.id == "mlp" else None
    )
    monkeypatch.setattr(gpuqueue, "expected_wait", lambda *a, **k: wait)
    coord = coordinator.Coordinator(improver)

    async def main():
        coord.loop, coord.tg = asyncio.get_running_loop(), _Group()
        return coord._fill(), coord

    with world.installed():
        return asyncio.run(main())


def test_gpu_free_roles_first_on_a_saturated_gpu(tmp_path, monkeypatch):
    started, coord = _filled(tmp_path / "idle", monkeypatch, wait=0.0)
    assert started == 1 and [j.kind for j in coord.jobs.values()] == [coordinator.SLICE]
    started, coord = _filled(tmp_path / "busy", monkeypatch, wait=1000.0)
    job = next(iter(coord.jobs.values()))
    assert started == 1 and (job.kind, job.arm.id) == (coordinator.RESEARCH, "mlp")


def test_a_roles_other_sessions_wait_for_its_first(tmp_path, monkeypatch):
    started, coord = _filled(tmp_path, monkeypatch, wait=0.0, slots=4)
    roles = [j.role for j in coord.jobs.values()]
    assert started == 3 and roles.count(KERNEL) == 1 and SYSTEMS in roles  # mlp's research
    first = next(j for j in coord.jobs.values() if j.role == KERNEL)
    coord._warmed(first)
    assert coord._fill() == 1  # the next kernel arm, once the first kernel session streamed


# ------------------------------------------------------------------ pieces


def test_write_guard_roots(tmp_path):
    root, native = tmp_path / "transforms", tmp_path / "transforms" / "native"
    hooks = runner.write_guard([tmp_path / "plan.md"], root, roots=[root], excluded=[native])
    guard = hooks["PreToolUse"][0].hooks[0]

    def decide(path):
        out = asyncio.run(guard({"tool_input": {"file_path": str(path)}}, None, None))
        return "deny" if out else "allow"

    assert decide(root / "graph.py") == "allow"
    assert decide("graph.py") == "allow"  # relative to the session's directory
    assert decide(tmp_path / "plan.md") == "allow"
    assert decide(native / "stage" / "engine.cu") == "deny"
    assert decide(tmp_path / "targets" / "attn" / "x.py") == "deny"


def test_simulated_gpu_jobs_queue_in_virtual_time(tmp_path):
    clock = dryrun.VirtualClock(1000.0)
    order: list[str] = []

    async def job(kind, seconds):
        j = gpuqueue.Job(kind=kind, job_class=gpuqueue.KINDS[kind][0], estimate_s=seconds)
        async with gpuqueue.holding(j):
            order.append(kind)
            await asyncio.sleep(seconds)
        return j

    def worker():  # a thread's GPU job (the integration's): through the loop, without the baton
        return clock.run_in_loop(job("integration", 100.0))

    async def main():
        with clock.driving():
            first = asyncio.create_task(job("eval", 50.0))
            await asyncio.sleep(0)
            later = [asyncio.to_thread(worker), job("eval", 30.0)]
            done = await asyncio.gather(first, *later)
            return done, clock.elapsed

    real = time.monotonic()
    saved = gpuqueue.clock
    gpuqueue.clock = clock.time
    try:
        (_, integration, second), elapsed = asyncio.run(main())
    finally:
        gpuqueue.clock = saved
    assert time.monotonic() - real < 5.0  # 180 simulated seconds
    assert order == ["eval", "eval", "integration"]  # an evaluation before background work
    assert elapsed == pytest.approx(180.0)
    assert second.wait_s == pytest.approx(50.0) and integration.wait_s == pytest.approx(80.0)


def test_cli_agents(monkeypatch, tmp_path):
    seen = {}

    async def fake(model, cfg, icfg, *, dry_run=False, seed=0):
        seen.update(icfg=icfg)
        return orchestrator.RunDir(tmp_path)

    monkeypatch.setattr(improve, "improve", fake)
    args = ["improve", "x/y", "--dry-run", "--agents", "3", "--role-max", "kernel=2,research=1"]
    assert cli.main([*args, "--overlap", "avoid"]) == 0
    icfg = seen["icfg"]
    assert (icfg.agents, icfg.overlap) == (3, "avoid")
    assert icfg.role_max == {KERNEL: 2, "research": 1}  # over the registry's caps
    assert scheduler.role_caps(icfg.role_max) == {KERNEL: 2, SYSTEMS: 1, NATIVE: 1, "research": 1}
    assert cli.main(["improve", "x/y", "--dry-run"]) == 0
    assert seen["icfg"].agents == 1 and seen["icfg"].role_max == {}
    assert scheduler.role_caps() == {KERNEL: 4, SYSTEMS: 1, NATIVE: 1}  # max_concurrent
    for bad in (["--agents", "0"], ["--role-max", "critic=2"]):
        with pytest.raises(SystemExit):
            cli.main(["improve", "x/y", *bad])
