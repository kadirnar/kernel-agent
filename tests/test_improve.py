"""``kernel-agent improve``: scheduler, stop rules, budgets, restarts and the dry run.

No GPU and no Claude: the loop runs against ``kernel_agent.dryrun`` (simulated
agents, worker and clock).
"""

import asyncio
import math
import struct

import pytest

from kernel_agent import charts, cli, dryrun, improve, ledger, orchestrator, scheduler
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
    assert "speed of light" in stop_reason(Arm("a", KERNEL, 100.0, sol=0.91), policy)
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
    assert arms["mlp"].sol == pytest.approx(0.93) and "speed of light" in arms["mlp"].stop
    assert arms["rmsnorm"].evals == 0 and arms["rmsnorm"].stop is None
    systems = arms[SYSTEMS]  # the integration row is not the systems agent's
    assert systems.evals == 1 and systems.best == pytest.approx(1.25, rel=1e-3)
    assert systems.estimate == pytest.approx(1 / dryrun.GPU_BUSY)  # launch bound: idle GPU
    no_systems = build_arms(run, Policy(systems=False), [])
    assert SYSTEMS not in {a.id for a in no_systems}


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
    assert any("speed of light" in a.stop for a in arms.values())
    assert any("plateau" in a.stop for a in arms.values())

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
