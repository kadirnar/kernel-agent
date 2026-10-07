"""``kernel-agent improve``: scheduler, stop rules, budgets, restarts and the dry run.

No GPU and no Claude: the loop runs against ``kernel_agent.dryrun`` (simulated
agents, worker and clock).
"""

import asyncio
import math
import struct

import pytest

from kernel_agent import charts, cli, dryrun, improve, ledger, orchestrator, scheduler, truth
from kernel_agent.agent.runner import AgentResult
from kernel_agent.agent.tools import record_candidate, record_e2e_result, snapshot
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver, kernel_digest, open_ideas
from kernel_agent.scheduler import KERNEL, SYSTEMS, Arm, Policy, build_arms, pick, rank, stop_reason
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

BASE = dryrun.BASELINE_MS
CHARTS = charts.available  # the real check; the fixture below turns charts off


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No real toolchain probing; charts only where a test turns them back on."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path, **cfg):
    cfg = {"dossier": False, **cfg}  # sessions counted: no dossier (test_web.py)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, **cfg)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    return orch, dryrun.World(orch)


def loop(orch, world, **icfg):
    improver = Improver(orch, ImproveConfig(**icfg), require_capture=False, live_charts=False)
    with world.installed():
        reason = asyncio.run(improver.improve())
    return improver, reason


def resume(run, **cfg):
    config = OptimizeConfig(model_ref=str(run.root), runs_dir=run.root.parent, **cfg)
    return improve.improve(str(run.root), config, ImproveConfig(), dry_run=True)


# ------------------------------------------------------------------ scheduler


def test_headroom_and_remaining():
    arm = Arm("a", KERNEL, ref_ms=100.0, best=1.25, estimate=2.0)
    assert arm.remaining_ms == pytest.approx(80.0)
    assert arm.headroom == pytest.approx(1 - 1.25 / 2.0)  # 1.6x still expected
    arm.sol = 0.8  # a speed-of-light estimate replaces the guess
    assert arm.headroom == pytest.approx(0.2)
    done = Arm("b", KERNEL, ref_ms=100.0, best=2.5, estimate=2.0)
    assert done.headroom == pytest.approx(1 - 1 / scheduler.MIN_FURTHER)


def test_ranking_follows_amdahl_and_decay():
    policy = Policy()
    big = Arm("big", KERNEL, ref_ms=600.0)
    small = Arm("small", KERNEL, ref_ms=200.0)
    near_sol = Arm("near_sol", KERNEL, ref_ms=600.0, sol=0.85)
    order = [a.id for a in rank([small, near_sol, big], policy)]
    assert order == ["big", "small", "near_sol"]  # untried: score ∝ remaining × headroom
    assert big.expected_ms == pytest.approx(600 * 0.5)
    assert near_sol.expected_ms == pytest.approx(600 * 0.15)

    stale = Arm("stale", KERNEL, ref_ms=600.0, stale=2)
    rank([big, stale], policy)
    assert stale.expected_ms == pytest.approx(big.expected_ms * 0.7**2)


def test_ucb_prefers_paying_arms_and_explores_untried_ones():
    policy = Policy()
    paying = Arm("paying", KERNEL, ref_ms=400.0, evals=10, gain_ms=100.0)
    barren = Arm("barren", KERNEL, ref_ms=400.0, evals=10)
    fresh = Arm("fresh", KERNEL, ref_ms=400.0)
    ranked = rank([barren, fresh, paying], policy)
    n = 20
    assert paying.index == pytest.approx(1.0 + math.sqrt(2 * math.log(n + 2) / 11))
    assert barren.index == pytest.approx(math.sqrt(2 * math.log(n + 2) / 11))
    assert fresh.index == pytest.approx(math.sqrt(2 * math.log(n + 2) / 1))
    assert fresh.index > paying.index > barren.index
    assert [a.id for a in ranked] == ["fresh", "paying", "barren"]
    assert pick(ranked) is fresh
    # an arm that keeps paying overtakes an untried one with less left to win
    small_fresh = Arm("small_fresh", KERNEL, ref_ms=100.0)
    assert pick(rank([small_fresh, paying], policy)) is paying


def test_stop_rules():
    policy = Policy(patience=5, sol_stop=0.9, target_hours=2.0, speedup_goal=2.0)
    assert stop_reason(Arm("a", KERNEL, 100.0, streak=4), policy) is None
    assert "plateau" in stop_reason(Arm("a", KERNEL, 100.0, streak=5), policy)
    assert "roofline" in stop_reason(Arm("a", KERNEL, 100.0, sol=0.91), policy)
    assert "time cap" in stop_reason(Arm("a", KERNEL, 100.0, hours=2.1), policy)
    assert "goal" in stop_reason(Arm("a", KERNEL, 100.0, best=2.05), policy)
    assert stop_reason(Arm(SYSTEMS, SYSTEMS, 100.0, best=2.05), policy) is None  # no module goal
    off = Policy(patience=0, sol_stop=None, target_hours=None, speedup_goal=None)
    assert stop_reason(Arm("a", KERNEL, 100.0, streak=9, sol=0.99, hours=9, best=9), off) is None
    stopped = Arm("stopped", KERNEL, ref_ms=900.0, stop="plateau")
    live = Arm("live", KERNEL, ref_ms=10.0)
    assert pick(rank([stopped, live], policy)) is live
    assert pick(rank([stopped], policy)) is None


def _kernel(speedup, *, pct=None):
    if isinstance(speedup, str):
        return {"status": speedup, "correct": False}
    case = {"timing_spread": 0.005, "calls_per_run": 1, "ref_ms": 1.0}
    result = {"status": "ok", "correct": True, "speedup": speedup, "cases": [case]}
    return result | ({"pct_of_sol": pct} if pct is not None else {})


def test_build_arms_from_ledger(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run

    def evaluate(target, outcome, pct=None):
        src = run.target(target) / "candidates" / "v.py"
        src.write_text("import torch\n")
        snap = snapshot(run, src, target)
        record_candidate(run, target, src, snap, _kernel(outcome, pct=pct), hypothesis="h")

    for outcome in (1.2, "incorrect", 1.5, 1.49, "build_error"):
        evaluate("attn", outcome)
    evaluate("mlp", 1.4, pct=72.0)
    evaluate("mlp", 1.7, pct=93.0)
    transform = run.transforms_dir / "static_cache.py"
    transform.write_text("def apply(workload): pass\n")
    snap = snapshot(run, transform)
    record_e2e_result(run, dryrun._e2e_result(BASE / 1.25), [snap], [], hypothesis="cache")
    integration = dryrun._e2e_result(BASE / 1.9)
    ledger.record_e2e(run, integration, backend="integrate", snapshot="x", hypothesis="all")
    slices = [
        {"arm": "attn", "improved": True, "seconds": 1800},
        {"arm": "attn", "improved": False, "seconds": 1800},
        {"arm": "attn", "improved": False, "seconds": 600},
    ]
    arms = {a.id: a for a in build_arms(run, Policy(), slices)}
    assert set(arms) == {"attn", "mlp", "rmsnorm", SYSTEMS}
    attn = arms["attn"]
    share = 856.0 / 2086.0
    assert attn.ref_ms == pytest.approx(share * BASE)
    assert attn.best == 1.5 and attn.streak == 2 and attn.evals == 5
    assert attn.gain_ms == pytest.approx(attn.ref_ms * (1 - 1 / 1.5))
    assert attn.stale == 2 and attn.hours == pytest.approx(4200 / 3600)
    assert attn.sol is None and attn.best_snapshot.startswith("003_")
    assert arms["mlp"].sol == pytest.approx(0.93) and "roofline" in arms["mlp"].stop
    assert arms["rmsnorm"].evals == 0 and arms["rmsnorm"].stop is None
    systems = arms[SYSTEMS]  # the integration row is not the systems agent's
    assert systems.evals == 1 and systems.best == pytest.approx(1.25, rel=1e-3)
    assert systems.estimate == pytest.approx(1 / dryrun.GPU_BUSY)  # launch bound: idle GPU
    no_systems = build_arms(run, Policy(systems=False), [])
    assert SYSTEMS not in {a.id for a in no_systems}


def test_move_on_rules_retire_an_arm_for_one_round(tmp_path):
    """Issue #166: patience, the SOL rule, the speedup goal and the time cap retire an arm for
    the round; in a new round (its record's ``exp``) it is back when the round's profile
    shows it still matters, else it stays retired."""
    orch, _ = make(tmp_path)
    run = orch.run

    def evaluate(target, outcome, pct=None):
        src = run.target(target) / "candidates" / "v.py"
        src.write_text(f"import torch  # {outcome}\n")
        snap = snapshot(run, src, target)
        record_candidate(run, target, src, snap, _kernel(outcome, pct=pct), hypothesis="h")

    for outcome in (1.5, 1.4, 1.45, 1.3, 1.2, 1.1):  # a best, then 5 without a new one
        evaluate("attn", outcome)
    evaluate("mlp", 2.1, pct=93.0)  # the goal and the SOL rule
    slices = [
        {"n": 1, "arm": "attn", "round": 1, "seconds": 7300, "evals": 6, "improved": True},
        {"n": 2, "arm": "mlp", "round": 1, "seconds": 600, "evals": 1, "improved": True},
    ]
    round1 = [{"n": 1, "speedup": 1.0}]
    arms = {a.id: a for a in build_arms(run, Policy(), slices, rounds=round1)}
    assert arms["attn"].stop == "plateau: 5 evaluations in a row without a new best"
    assert "93% of its recipe's roofline" in arms["mlp"].stop and arms["mlp"].fresh
    assert arms["attn"].hours == pytest.approx(7300 / 3600)

    round2 = [*round1, {"n": 2, "exp": len(ledger.rows(run)), "speedup": 1.4}]
    arms = {a.id: a for a in build_arms(run, Policy(), slices, rounds=round2)}
    attn, mlp = arms["attn"], arms["mlp"]
    assert attn.stop is None and attn.streak == 0 and attn.hours == 0.0  # back in round 2
    assert attn.best == 1.5 and attn.base == 1.5 and not attn.fresh
    # mlp's SOL stop was round 1's; at 93 % of its roofline it is not worth a slice now
    # (7 % of its 206 ms left: 14 ms < 2 % of the run), unless any gain revives an arm
    assert mlp.stop.startswith("retired: expected 14.4 ms in the round-2 profile, below 2%")
    revived = {a.id: a for a in build_arms(run, Policy(revive_share=0), slices, rounds=round2)}
    assert revived["mlp"].stop is None and revived["mlp"].base == pytest.approx(2.1)
    assert arms["rmsnorm"].stop is None  # no slice in round 1: never retired

    # in round 2 the rules count again from its start: the goal from the round's 1.5x
    slices.append({"n": 3, "arm": "attn", "round": 2, "seconds": 600, "evals": 1})

    def attn_arm():
        return next(a for a in build_arms(run, Policy(), slices, rounds=round2) if a.id == "attn")

    evaluate("attn", 2.4)
    assert attn_arm().stop is None and attn_arm().fresh  # past round 1's goal: 1.6x in round 2
    evaluate("attn", 3.1)
    attn = attn_arm()
    assert attn.fresh and attn.stop == "speedup goal reached: 3.10x (2.07x this round) (goal 2x)"
    for _ in range(5):
        evaluate("attn", 1.0)
    assert attn_arm().stop == "plateau: 5 evaluations in a row without a new best"


def test_systems_arm_credits_transforms_on_top_of_kernels(tmp_path):
    """Issue #40, the VoxCPM2 run: transforms on top of the integrated kernels beat the
    integration (ledger rows 14, 17-19); that is the systems agent's gain, the kernels' is not."""
    orch, _ = make(tmp_path)
    run = orch.run

    def transforms(*stems):
        snaps = []
        for stem in stems:
            src = run.transforms_dir / f"{stem}.py"
            src.write_text(f"def apply(workload): pass  # {stem}\n")
            snaps.append(snapshot(run, src))
        return snaps

    def kernel(target, speedup):
        src = run.target(target) / "candidates" / "v.py"
        src.write_text(f"import torch  # {speedup}\n")
        snap = snapshot(run, src, target)
        record_candidate(run, target, src, snap, _kernel(speedup), hypothesis="h")
        return f"{target}={run.target(target) / 'history' / snap.name}"  # as _kernel_winners

    def e2e(speedup, snaps, kernels=()):
        result = dryrun._e2e_result(BASE / speedup)
        record_e2e_result(run, result, snaps, list(kernels), hypothesis="h")

    def integrate(speedup, *items):
        result = dryrun._e2e_result(BASE / speedup)
        ledger.record_e2e(run, result, backend="integrate", snapshot="+".join(items), hypothesis="")

    def systems(policy=None):
        return next(a for a in build_arms(run, policy or Policy(), []) if a.id == SYSTEMS)

    stack = ("graph_cfm_solver", "compile_dit", "graph_lm_step")
    e2e(2.0843, transforms("graph_cfm_solver"))
    e2e(1.2366, transforms("graph_lm_step"))
    e2e(5.0473, transforms(*stack))  # the best transform-only run
    kernel("attn", 9.84)
    attn = kernel("attn", 46.89)
    integrate(1.6786, "attn")
    integrate(4.6775, "graph_cfm_solver", "compile_dit", "attn")
    integrate(5.4441, "graph_cfm_solver", "compile_dit", "attn", "graph_lm_step")  # row 14
    assert systems().best == pytest.approx(5.0473, rel=1e-3)
    for speedup in (5.9027, 6.1677, 6.3778):  # rows 17-19: each beats the previous
        e2e(speedup, transforms(*stack), [attn])
    arm = systems(Policy(patience=3))
    assert arm.best == pytest.approx(6.3778, rel=1e-3) and arm.best_snapshot.endswith("+attn")
    assert arm.streak == 0 and arm.stop is None and arm.evals == 6
    alone = BASE * (1 - 1 / 5.0473)
    on_kernels = BASE * (1 / 5.4441 - 1 / 6.3778)  # over the integration with the same kernels
    assert arm.gain_ms == pytest.approx(alone + on_kernels, rel=1e-3)
    assert arm.remaining_ms == pytest.approx(BASE / 6.3778, rel=1e-3)

    # a faster kernel (or one more kernel) is not the systems agent's gain: no run with
    # those kernels to beat yet, the first run sets the bar
    faster = kernel("attn", 60.0)
    e2e(6.9, transforms(*stack), [faster])
    e2e(7.5, transforms(*stack), [attn, kernel("mlp", 1.4)])
    arm = systems()
    assert arm.best == pytest.approx(6.3778, rel=1e-3) and arm.streak == 2
    e2e(7.1, transforms(*stack), [faster])  # beats 6.9 with the same kernels
    e2e(5.2, transforms(*stack))  # transform-only: beats 5.0473, as before
    arm = systems()
    assert arm.best == pytest.approx(7.1, rel=1e-3) and arm.streak == 0
    more = BASE * (1 / 6.9 - 1 / 7.1) + BASE * (1 / 5.0473 - 1 / 5.2)
    assert arm.gain_ms == pytest.approx(alone + on_kernels + more, rel=1e-3)


# ------------------------------------------------------------------ digests


def test_open_ideas_and_digest_are_bounded(tmp_path):
    notes = "# log\n- tried a\n\n## Open ideas\n- split-K\n- persistent CTAs\n\n## Other\nx\n"
    assert open_ideas(notes) == "- split-K\n- persistent CTAs"
    assert open_ideas("nothing here") == ""

    orch, _ = make(tmp_path)
    run = orch.run
    for i in range(40):
        src = run.target("attn") / "candidates" / f"v{i}.py"
        src.write_text("import triton\n")
        snap = snapshot(run, src, "attn")
        record_candidate(run, "attn", src, snap, _kernel(1.0 + 0.03 * i), hypothesis=f"idea {i}")
    (run.target("attn") / "NOTES.md").write_text("x" * 50_000 + "\n## Open ideas\n- tile 64\n")
    arm = next(a for a in build_arms(run, Policy(), []) if a.id == "attn")
    digest = kernel_digest(run, arm, 7, 4, Policy())
    assert "# Improve slice 7" in digest and "about 4 evaluations" in digest
    assert f"history/{arm.best_snapshot}" in digest and "2.170x module speedup" in digest
    assert "idea 39" in digest and "idea 24" not in digest  # the last 15 rows only
    assert "- tile 64" in digest
    assert len(digest) < improve.NOTES_CHARS + 6000  # NOTES.md is cut to its tail
    assert f"costs about {arm.remaining_ms:.1f} ms per model run now" in digest
    # the scheduler's ms are the metric's (#114): per second of audio for metric=throughput
    baseline = read_json(run.baseline_json)
    truth.writable(run.baseline_json)
    write_json(run.baseline_json, {**baseline, "metric": "throughput"})
    digest = kernel_digest(run, arm, 7, 4, Policy())
    assert f"{arm.remaining_ms:.1f} ms per second of generated audio now" in digest


# ------------------------------------------------------------------ the loop (dry run)


def test_dry_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "available", CHARTS)
    orch, world = make(tmp_path)
    improver, reason = loop(orch, world, rounds=2)
    run = orch.run
    state = read_json(run.root / "improve.json")

    assert reason.startswith("every arm has stopped")
    assert state["finished"]["reason"] == reason
    slices = state["slices"]
    assert [s["n"] for s in slices] == list(range(1, len(slices) + 1))
    assert all(s["status"] == "done" for s in slices)
    assert {s["arm"] for s in slices} == {"attn", "mlp", "rmsnorm", "rope", SYSTEMS}
    assert all(1 <= s["evals"] <= 4 for s in slices)  # --slice 4: fresh session per slice
    # budget moves to where it pays: the hot targets get more slices than rope
    count = {a: sum(s["arm"] == a for s in slices) for a in ("attn", "rope")}
    assert count["attn"] > count["rope"]

    rows = ledger.rows(run)
    statuses = {r["status"] for r in rows}
    assert {"keep", "discard", "incorrect"} <= statuses
    assert sum(s["evals"] for s in slices) == sum(r["backend"] != "integrate" for r in rows)
    arms = {a.id: a for a in improver.arms()}
    assert all(a.stop for a in arms.values())
    assert any("plateau" in a.stop for a in arms.values())
    # the move-on rules retire an arm for a round (#166): an arm retired in round 1 (the
    # SOL rule, the goal) comes back in round 2 when its profile shows it still matters
    # (attn: its goal counts from its round-1 best), else it stays retired
    assert any(a.stop.startswith("retired: expected") for a in arms.values())
    assert {s["arm"] for s in slices if s["round"] == 2} >= {"attn", "rope"}
    assert "attn" in {s["arm"] for s in slices if s["round"] == 1}

    # re-integration every 4 kept results, measured end to end
    integrations = state["integrations"]
    assert len(integrations) >= 2
    final = read_json(run.root / "integration.json")["final"]
    assert final["passed"] and final["speedup"] > 1.8
    assert integrations[-1]["speedup"] == final["speedup"]
    assert "mlp" not in integrations[-1]["accepted"]  # incompatible with the CUDA graph

    # round 2: re-profile with the integration applied, re-plan, new target captured
    assert [r["n"] for r in state["rounds"]] == [1, 2]
    round2 = state["rounds"][1]
    assert round2["targets"] == ["rope"] and round2["applied"]
    assert (run.root / "rounds/2/profile/profile.json").exists()
    assert read_json(run.root / "rounds/2/plan.json")["targets"][0]["id"] == "rope"
    assert read_json(run.target("rope") / "spec.json")["capture"]["cases"]

    # fresh sessions seeded with a digest, through Orchestrator._agent (program.md, costs)
    kernel_sessions = [s for s in world.sessions if s["name"].startswith("kernel-")]
    assert all("# Improve slice" in s["system"] for s in kernel_sessions)
    assert all("# Program" in s["system"] for s in world.sessions)
    later = [s for s in world.sessions if s["name"] == "kernel-attn"][1]
    assert "## Best so far" in later["system"] and "history/" in later["system"]
    assert "(none recorded yet)" not in later["system"] and "- exp " in later["system"]
    planner = next(s for s in world.sessions if s["name"] == "planner")
    assert "# Round 2" in planner["system"] and "`attn` (`Qwen3Attention`)" in planner["system"]
    costs = read_json(run.root / "costs.json")
    assert {s["label"] for s in slices} <= set(costs)
    assert "planner#round2" in costs

    events = [e["event"] for e in ledger.events(run)]
    assert events.count("slice_start") == events.count("slice_done") == len(slices)
    assert "phase_done" in events and "round_start" in events
    assert run.load()["phases"]["improve"]["done"]
    report = run.report.read_text()
    assert "## Improve loop" in report and "every arm has stopped" in report
    if charts.available():
        for png in ("progress.png", "improve.png", "amdahl.png", "integration.png"):
            data = (run.root / png).read_bytes()[:24]
            assert data[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", data[16:24])[0] > 800
        assert "improve.png" in report


def test_dry_run_uses_the_whole_budget_with_new_rounds(tmp_path):
    """Issue #166: every arm retires early in round 1; with budget left the loop starts new
    rounds (re-profile, re-plan, the arms that still matter) instead of ending, and the
    report shows the budget left unused (about none) and why; ``--rounds 1`` keeps the old
    behaviour."""
    orch, world = make(tmp_path / "budget", max_hours=30.0)
    _, reason = loop(orch, world)
    state = read_json(orch.run.root / "improve.json")
    assert len(state["rounds"]) > 2 and reason.startswith("time left")  # the budget ended it
    assert all(r["exp"] > 0 for r in state["rounds"][1:])
    assert {s["round"] for s in state["slices"]} == {r["n"] for r in state["rounds"]}
    used = state["finished"]["budget"]
    assert used["max_hours"] == 30.0 and used["unused_hours"] < 0.25
    report = orch.run.report.read_text()
    assert "* budget: " in report and "h of 30 h used" in report

    orch, world = make(tmp_path / "one", max_hours=30.0)
    _, reason = loop(orch, world, rounds=1)
    state = read_json(orch.run.root / "improve.json")
    assert [r["n"] for r in state["rounds"]] == [1]
    assert reason.startswith("every arm has stopped") and "no new round" not in reason
    assert state["finished"]["budget"]["unused_hours"] > 10
    assert "unused because: every arm has stopped" in orch.run.report.read_text()

    # no budget: new rounds while each one brings a real end-to-end gain
    orch, world = make(tmp_path / "unbounded")
    _, reason = loop(orch, world)
    assert reason.endswith("; no new round: round 2 brought no real end-to-end gain")


def test_budget_lines():
    assert improve._budget_lines({"reason": "x"}) == []
    done = {"reason": "every arm has stopped (a: plateau)"}
    used = {"hours": 2.5, "max_hours": 4.0, "unused_hours": 1.5, "usd": 12.3, "max_usd": 60.0}
    assert improve._budget_lines({**done, "budget": used}) == [
        "* budget: 2.50 h of 4 h used, 90 min (38%) unused, $12.30 of $60.00; unused "
        "because: every arm has stopped (a: plateau)"
    ]
    used = {"hours": 4.0, "max_hours": 4.0, "unused_hours": 0.0, "sessions": 9, "max_sessions": 9}
    assert improve._budget_lines({**done, "budget": used}) == [
        "* budget: 4.00 h of 4 h used, 0 min (0%) unused, 9 of 9 agent sessions"
    ]
    assert improve._budget_lines({**done, "budget": {"hours": 1.0}}) == [
        "* budget: 1.00 h (no --max-hours)"
    ]


def test_dry_run_is_reproducible(tmp_path):
    a, _ = loop(*make(tmp_path / "a"), max_slices=6)
    b, _ = loop(*make(tmp_path / "b"), max_slices=6)

    def key(run):
        return [(r["target"], r["status"], r["speedup"]) for r in ledger.rows(run)]

    assert key(a.run) == key(b.run)


def test_usd_budget_stops_the_loop(tmp_path):
    orch, world = make(tmp_path, max_usd=5.0)
    _, reason = loop(orch, world)
    assert "USD budget spent" in reason
    state = read_json(orch.run.root / "improve.json")
    assert 1 <= len(state["slices"]) <= 4
    assert orch.budget.spent_usd() < 5.0 + 3.0  # at most one slice over
    assert (orch.run.root / "integration.json").exists() and orch.run.report.exists()


def test_time_budget_stops_the_loop(tmp_path):
    orch, world = make(tmp_path, max_hours=1.5)  # simulated hours
    _, reason = loop(orch, world)
    assert "time budget spent" in reason
    state = read_json(orch.run.root / "improve.json")
    hours = (state["slices"][-1]["ended"] - state["slices"][0]["started"]) / 3600
    assert hours < 1.5 and state["finished"]
    assert orch.run.report.exists()


def test_max_slices_and_integrate_every(tmp_path):
    orch, world = make(tmp_path)
    _, reason = loop(orch, world, max_slices=3, integrate_every=1000)
    assert reason == "--max-slices 3 reached"
    state = read_json(orch.run.root / "improve.json")
    assert len(state["slices"]) == 3
    assert [i["why"] for i in state["integrations"]] == ["final integration"]


def test_restart_after_ctrl_c_continues_consistently(tmp_path):
    orch, world = make(tmp_path)
    calls = []

    def hook(agent, evals):
        calls.append(agent)
        if len(calls) == 9:  # 3rd evaluation of a later slice
            raise KeyboardInterrupt

    world.hook = hook
    with pytest.raises(KeyboardInterrupt):
        loop(orch, world)
    run = orch.run
    state = read_json(run.root / "improve.json")
    cut = state["slices"][-1]
    assert cut["status"] == "interrupted" and cut["evals"] >= 1
    assert all(s["status"] == "done" for s in state["slices"][:-1])
    assert "finished" not in state
    events = ledger.events(run)
    assert events[-1]["event"] == "phase_failed" and events[-1]["phase"] == "improve"
    rows_before = len(ledger.rows(run))

    asyncio.run(resume(run))  # `kernel-agent improve <run_dir> --dry-run`
    state = read_json(run.root / "improve.json")
    slices = state["slices"]
    assert [s["n"] for s in slices] == list(range(1, len(slices) + 1))
    assert slices[-1]["status"] == "done" and "running" not in {s["status"] for s in slices}
    assert state["finished"]["reason"].startswith("every arm has stopped")
    rows = ledger.rows(run)
    assert [r["exp"] for r in rows] == list(range(1, len(rows) + 1))
    assert len(rows) > rows_before
    assert sum(s["evals"] for s in slices) == sum(r["backend"] != "integrate" for r in rows)

    # a finished run stays finished: no new slices and no new evaluations
    asyncio.run(resume(run))
    again = read_json(run.root / "improve.json")
    assert len(again["slices"]) == len(slices)
    assert len(ledger.rows(run)) == len(rows)


def test_failing_sessions_do_not_end_the_loop_until_three_in_a_row(tmp_path):
    orch, world = make(tmp_path)

    def hook(agent, evals):
        if evals == 1 and len(world.sessions) in (2, 5, 6, 7):  # the CLI dies mid-session
            raise RuntimeError("claude code exited with 1")

    world.hook = hook
    _, reason = loop(orch, world)
    assert reason.startswith("3 agent sessions in a row failed")
    statuses = [s["status"] for s in read_json(orch.run.root / "improve.json")["slices"]]
    assert statuses.count("failed") == 4 and statuses[-3:] == ["failed"] * 3
    assert "done" in statuses[statuses.index("failed") + 1 :]  # it went on after one failure
    assert orch.run.report.exists()


def test_idle_arms_stop(tmp_path):
    orch, _ = make(tmp_path)
    idle = [{"arm": "attn", "status": "done", "evals": 0, "seconds": 60}] * 2
    arms = {a.id: a for a in build_arms(orch.run, Policy(), idle)}
    assert arms["attn"].idle == 2 and "no evaluation" in arms["attn"].stop
    interrupted = [*idle, {"arm": "attn", "status": "interrupted", "evals": 0}]
    assert next(a for a in build_arms(orch.run, Policy(), interrupted) if a.id == "attn").idle == 0
    # slices the time budget cut short count neither as idle nor as stale (issue #100)
    short = [{**s, "budget_short": True} for s in idle]
    attn = next(a for a in build_arms(orch.run, Policy(), short) if a.id == "attn")
    assert (attn.idle, attn.stale, attn.stop, attn.short) == (0, 0, None, True)
    attn = next(a for a in build_arms(orch.run, Policy(), [*short, idle[0]]) if a.id == "attn")
    assert (attn.idle, attn.stale, attn.stop, attn.short) == (1, 1, None, False)


# ------------------------------------------------------------------ time (issue #100)


def test_slice_seconds():
    kernel = Arm("a", KERNEL, ref_ms=1.0, rows=[{"eval_s": s} for s in (10.0, 30.0, None, 20.0)])
    extra = scheduler.WARMUP_SECONDS + scheduler.WRAP_UP_SECONDS
    assert scheduler.slice_seconds(kernel) == pytest.approx(extra + 20.0)  # the median
    assert scheduler.slice_seconds(Arm("s", SYSTEMS, ref_ms=1.0)) == pytest.approx(
        extra + scheduler.EVAL_SECONDS[SYSTEMS]  # none timed yet
    )


def test_integration_estimate_covers_the_measured_integration(tmp_path):
    orch, world = make(tmp_path)
    improver = Improver(orch, ImproveConfig(), require_capture=False, live_charts=False)
    run = orch.run
    with world.installed():
        assert improver._integration_estimate() == (0, scheduler.AB_SECONDS)  # nothing yet
        for k in range(4):
            arms = improver._pickable()
            asyncio.run(improver._slice(pick(arms), arms))
            if k % 2 == 0:
                continue
            steps, each = improver._integration_estimate()
            before = len(ledger.rows(run))
            asyncio.run(orch.integrate(reuse=True))
            measured = len(ledger.rows(run)) - before
            assert steps >= measured > 0  # the first measures everything, the second reuses
        steps, each = improver._integration_estimate()
    assert steps == 0  # nothing changed since: a re-integration reuses every measurement
    integrated = [r["eval_s"] for r in ledger.rows(run) if r["backend"] == "integrate"]
    assert min(integrated) <= each <= max(integrated) and each > 100  # simulated A/B time


def test_little_time_left_goes_to_the_final_integration(tmp_path):
    orch, world = make(tmp_path, max_hours=5.0)  # simulated hours
    _, reason = loop(orch, world, integrate_every=0)
    state = read_json(orch.run.root / "improve.json")
    assert reason.startswith("time left") and "< one slice of" in reason
    assert "kept for the final integration (" in reason and "A/B measurements" in reason
    slices = state["slices"]
    assert all(s["limit_s"] >= s["need_s"] for s in slices if "limit_s" in s)
    assert any("limit_s" in s for s in slices)  # the last ones ran on what the budget left
    assert [i["why"] for i in state["integrations"]] == ["final integration"]
    # the final integration fits in the time the loop kept for it
    assert state["finished"]["at"] - world.clock.t0 <= 5.0 * 3600


def test_the_reserve_is_capped_and_the_final_integration_runs_past_the_budget(tmp_path, capsys):
    """Round 2 of runs/openbmb--VoxCPM2/20261006-004718 kept 113 of its 180 min for the
    final integration (issue #108). The reserve now takes at most a third of --max-hours;
    the final integration runs to its end after it, on the GPU alone."""
    orch, world = make(tmp_path, max_hours=2.0)  # simulated hours
    _, reason = loop(orch, world, integrate_every=0)
    out = capsys.readouterr().out
    assert "40 min kept for the final integration (33% of --max-hours; estimated" in out
    assert "40 min kept for integrate + report" in reason
    assert "runs past --max-hours" in out and "it uses the GPU only, no agent sessions" in out
    state = read_json(orch.run.root / "improve.json")
    assert [i["why"] for i in state["integrations"]] == ["final integration"]
    assert state["slices"][-1]["ended"] - world.clock.t0 > 2.0 * 3600 * 2 / 3 - 600
    assert state["finished"]["at"] - world.clock.t0 > 2.0 * 3600  # past the budget


@pytest.mark.parametrize(
    "estimate, reserve, kept_min",
    [
        ((30, 228.0), None, 60.0),  # 114 min: capped at a third of 3 h
        ((10, 228.0), None, 38.0),  # below the cap: the estimate
        ((30, 228.0), 30.0, 30.0),  # --integration-reserve 30
        ((30, 228.0), 0.0, 27.0),  # --integration-reserve 0: --budget-reserve (15 %) only
    ],
)
def test_integration_reserve(tmp_path, monkeypatch, capsys, estimate, reserve, kept_min):
    orch, _ = make(tmp_path, max_hours=3.0)
    icfg = ImproveConfig(integration_reserve=reserve)
    improver = Improver(orch, icfg, require_capture=False, live_charts=False)
    monkeypatch.setattr(improver, "_integration_estimate", lambda: estimate)
    improver._keep_time()
    assert orch.budget.reserve_s() == pytest.approx(kept_min * 60)
    assert orch.budget.estimate_reserve() == orch.budget.final_reserve_s  # after evaluations
    assert capsys.readouterr().out.count(f"{kept_min:.0f} min kept for") == 1


def test_cli_integration_reserve(monkeypatch, tmp_path):
    seen = []

    async def fake_improve(ref, cfg, icfg, *, dry_run=False, seed=0):
        seen.append(icfg.integration_reserve)
        return RunDir(tmp_path)

    monkeypatch.setattr(improve, "improve", fake_improve)
    for flag, value in (([], None), (["auto"], None), (["45"], 45.0), (["0"], 0.0)):
        argv = ["improve", "runs/x", *(["--integration-reserve", *flag] if flag else [])]
        assert cli.main(argv) == 0 and seen[-1] == value
    with pytest.raises(SystemExit):
        cli.main(["improve", "runs/x", "--integration-reserve", "-5"])


def test_a_slice_out_of_time_does_not_stop_its_arm(tmp_path):
    orch, world = make(tmp_path)
    # 10 min for agents: enough for one slice of any arm (7-8 min), not for twice that
    orch.budget.max_hours = 600 / 3600 / (1 - orch.budget.reserve)

    async def decline(name, *, result=None, **_):  # too little time to write a candidate
        world.clock.advance(30)
        return result or AgentResult(name=name)

    world.run_agent = decline
    improver, reason = loop(orch, world)
    slices = read_json(orch.run.root / "improve.json")["slices"]
    assert slices and all(s["budget_short"] and s["evals"] == 0 for s in slices)
    assert len({s["arm"] for s in slices}) == len(slices)  # no arm twice: it needs 2x now
    arms = {a.id: a for a in improver.arms()}
    assert all(arms[s["arm"]].stop is None and arms[s["arm"]].short for s in slices)
    assert reason.startswith("time left") and "kept for integrate + report" in reason


def test_recover_a_slice_left_running(tmp_path):
    orch, world = make(tmp_path)
    loop(orch, world, max_slices=2)
    path = orch.run.root / "improve.json"
    state = read_json(path)
    last = state["slices"][-1]
    evals, ended = last["evals"], last["ended"]
    last.update(status="running", evals=None, ended=None)  # the process was killed
    write_json(path, state)
    Improver(orch, ImproveConfig(), require_capture=False)._recover()
    recovered = read_json(path)["slices"][-1]
    assert recovered["status"] == "interrupted" and recovered["evals"] == evals
    assert recovered["started"] <= recovered["ended"] <= ended


def test_dry_run_guards(tmp_path):
    orch, _ = make(tmp_path)
    cfg = OptimizeConfig(model_ref="x", runs_dir=tmp_path)
    with pytest.raises(SystemExit, match="pass --dry-run"):
        asyncio.run(improve.improve(str(orch.run.root), cfg, ImproveConfig()))
    data = orch.run.load()
    data.pop("dry_run")
    write_json(orch.run.run_json, data)
    with pytest.raises(SystemExit, match="only continues dry-run runs"):
        asyncio.run(improve.improve(str(orch.run.root), cfg, ImproveConfig(), dry_run=True))


def test_cli_improve(monkeypatch, tmp_path):
    seen = []

    async def fake_improve(ref, cfg, icfg, *, dry_run=False, seed=0):
        seen.append((ref, cfg, icfg, dry_run, seed))
        return RunDir(tmp_path)

    monkeypatch.setattr(improve, "improve", fake_improve)
    argv = ["improve", "runs/x", "--max-hours", "3", "--max-usd", "20", "--slice", "6"]
    argv += ["--rounds", "2", "--speedup-goal", "0", "--target-hours", "1.5", "--dry-run"]
    assert cli.main(argv) == 0
    ref, cfg, icfg, dry_run, seed = seen[0]
    assert ref == "runs/x" and dry_run and seed == 0
    assert cfg.max_hours == 3.0 and cfg.max_usd == 20.0
    assert icfg.slice == 6 and icfg.rounds == 2 and icfg.integrate_every == 4
    assert icfg.policy.speedup_goal is None and icfg.policy.target_hours == 1.5
    assert icfg.policy.patience == 5 and icfg.policy.sol_stop == 0.9
    assert cli.main(["improve", "runs/x", "--dry-run"]) == 0
    assert seen[1][2].rounds is None  # as many rounds as the budget allows (#166)


def test_cli_dry_run(tmp_path, capsys):
    argv = ["improve", "org/model", "--dry-run", "--runs-dir", str(tmp_path), "--max-slices", "3"]
    assert cli.main(argv) == 0
    run = RunDir(next(tmp_path.glob("org--model/*")))
    assert len(ledger.rows(run)) >= 3 and run.report.exists()
    assert "run directory:" in capsys.readouterr().out


# ------------------------------------------------------------------ orchestrator hooks


def test_integrate_reuses_measured_combinations(tmp_path):
    orch, world = make(tmp_path)
    loop(orch, world, max_slices=4, integrate_every=1000)  # ends with one integration
    calls = []
    real = world.worker

    def counting(run, command, *args, **kwargs):
        calls.append(args)
        return real(run, command, *args, **kwargs)

    with world.installed():
        orch.worker = counting
        asyncio.run(orch.integrate(reuse=True))
        assert calls == []  # nothing changed: every combination was measured before
        asyncio.run(orch.integrate())
        assert calls  # without reuse every combination is measured again
    history = read_json(orch.run.root / "integration.json")["history"]
    assert len(calls) == len(history)


def test_worker_analyze_needs_out_dir_for_patches(tmp_path, capsys):
    from kernel_agent import worker

    run = RunDir.create(tmp_path, "org/m")
    assert worker.main(["analyze", "--run-dir", str(run.root), "--kernel", "t=x.py"]) == 1
    assert "needs --out-dir" in capsys.readouterr().out
    assert not run.baseline_json.exists()


def test_slice_events_and_costs_are_labelled(tmp_path):
    orch, world = make(tmp_path)
    loop(orch, world, max_slices=2)
    starts = [e for e in ledger.events(orch.run) if e["event"] == "agent_start"]
    slices = read_json(orch.run.root / "improve.json")["slices"]
    assert [e.get("label") for e in starts] == [s["label"] for s in slices]
    results = read_jsonl(orch.run.target("attn") / "results.jsonl")
    assert all(r["hypothesis"] for r in results)
