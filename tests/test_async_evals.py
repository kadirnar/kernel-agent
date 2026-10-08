"""``kernel-agent improve --async-evals`` (issue #191): ``submit_evaluation`` /
``evaluation_result``, their advice semantics, the session clock of a submitted evaluation
(``gpuqueue.Job.attach``), the session's states, and a virtual-time dry run whose simulated
engineers write their next candidate while the last one is evaluated.

No GPU and no Claude.
"""

import asyncio
import json
import threading

import pytest

from kernel_agent import charts, dryrun, gpuqueue, ledger, orchestrator, sessions
from kernel_agent.agent import tools as tools_mod
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.workspace import RunDir, read_json, read_jsonl

LABEL = "kernel-t#1"


def text(out):
    return json.loads(out["content"][0]["text"])


@pytest.fixture
def server(tmp_path, monkeypatch):
    """One session's tools with --async-evals over a fake evaluator: a full evaluation waits
    for ``gate``; it records what it evaluated (the snapshot's text)."""
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    (tdir / "capture.pt").write_bytes(b"")
    gate, evaluated = threading.Event(), []

    def fake_eval(capture, snap, *, quick=False, **_):
        if not quick:
            evaluated.append(snap.read_text())
            assert gate.wait(10)
        return {"status": "ok", "correct": True, "speedup": 1.0 + 0.1 * len(evaluated)}

    monkeypatch.setattr(tools_mod, "run_evaluation", fake_eval)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    binding = tools_mod.SessionBinding(label=LABEL, evaluations=3)
    built = tools_mod.build_server(run, Budget(run), binding=binding, async_evals=True)
    tools = {t.name: t for t in built}

    async def call(name, **args):
        return text(await tools[name].handler(args))

    def write(name, version):
        (tdir / "candidates" / name).write_text(f"V = {version}\ndef build(r): ...\n")
        return {"target_id": "t", "candidate": f"candidates/{name}", "hypothesis": "h"}

    return run, call, write, gate, evaluated


def test_submit_then_collect(server):
    run, call, write, gate, evaluated = server

    async def main():
        out = await call("submit_evaluation", **write("v1.py", 1))
        assert out["ticket"] == "e1" and out["state"] == "submitted"
        assert out["snapshot"].startswith("history/") and out["evaluations_left"] == 2
        assert "advice" not in out and "evaluation_result" in out["next"]
        write("v1.py", 2)  # the agent edits its file: the evaluation measures the snapshot
        busy = await call("submit_evaluation", **write("v2.py", 3))
        assert busy["status"] == "error" and "e1" in busy["error"] and "in flight" in busy["error"]
        busy = await call("evaluate_candidate", **write("v2.py", 3))
        assert busy["status"] == "error" and "e1" in busy["error"]
        quick = await call("evaluate_candidate", mode="quick", **write("v2.py", 3))
        assert quick["mode"] == "quick" and quick["correct"]  # a quick check runs meanwhile
        gate.set()
        result = await call("evaluation_result", ticket="e1")
        assert result["advice"] == "continue" and result["budget"]["evals_used"] == 1
        assert result["ledger"]["status"] and result["speedup"] == pytest.approx(1.1)
        none = await call("evaluation_result")
        assert none["status"] == "error" and "none in flight" in none["error"]
        again = await call("submit_evaluation", **write("v3.py", 1))  # v1's source: a duplicate
        assert "duplicate" in again and "ticket" not in again  # its result at once
        quick_submit = await call("submit_evaluation", mode="quick", **write("v4.py", 4))
        assert quick_submit["status"] == "error" and "quick" in quick_submit["error"]

    asyncio.run(main())
    assert evaluated == ["V = 1\ndef build(r): ...\n"]
    rows = ledger.rows(run)
    assert [r["session"] for r in rows if r["status"] not in ledger.UNMEASURED] == [LABEL]


def test_an_evaluation_in_flight_when_the_session_ends_is_still_recorded(server):
    run, call, write, gate, _ = server

    async def main():
        out = await call("submit_evaluation", **write("v1.py", 1))
        assert out["ticket"] == "e1"
        threading.Timer(0.2, gate.set).start()
        return await tools_mod.settle(run, LABEL)  # the session ended (Orchestrator._agent)

    assert asyncio.run(main()) == 1
    assert [r["session"] for r in ledger.rows(run) if r["speedup"]] == [LABEL]
    assert asyncio.run(tools_mod.settle(run, LABEL)) == 0  # nothing left


def test_a_detached_job_stops_the_session_clock_only_once_collected(monkeypatch):
    monkeypatch.setattr(gpuqueue, "_gates", {})
    events = []
    stop = gpuqueue.listen(events.append)

    async def main():
        clock = gpuqueue.SessionClock(LABEL)
        with clock.running(None):
            mine = gpuqueue.Job.of(None, "eval", "t", detached=True)
            with gpuqueue.using(mine):
                nested = gpuqueue.Job.of(None, "eval", "t")
        assert mine.session == LABEL and mine.clock is None and mine.detached
        assert nested.detached and nested.clock is None
        other, release = gpuqueue.Job.of(None, "eval", "u"), asyncio.Event()

        async def hold_other():
            async with gpuqueue.holding(other):
                await release.wait()

        async def run_mine():
            async with gpuqueue.holding(mine):
                pass

        first = asyncio.create_task(hold_other())
        await asyncio.sleep(0.05)
        second = asyncio.create_task(run_mine())
        await asyncio.sleep(0.3)  # it waits for the GPU while its session works
        assert mine.waiting and clock.waited == 0.0
        mine.attach(clock)  # evaluation_result: the session waits for it now
        await asyncio.sleep(0.3)
        release.set()
        await asyncio.gather(first, second)
        await asyncio.sleep(0)
        return clock.waited, mine.wait_s

    try:
        waited, wait_s = asyncio.run(main())
    finally:
        stop()
    assert wait_s >= 0.55 and 0.25 <= waited <= wait_s - 0.2
    assert any(e.get("detached") and e["state"] == "queued" for e in events)


def test_a_detached_job_is_the_sessions_state_only_while_it_collects(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    observer = sessions.Observer(run)
    tracker = observer.open(LABEL, agent="kernel-t", role="kernel")
    tracker.turn(sessions.THINKING)
    tracker.tool_started("s1", "mcp__ka__submit_evaluation")
    tracker.tool_ended("s1")
    assert tracker.evaluations == 1  # a submitted evaluation counts
    tracker.gpu({"state": "queued", "id": 7, "class": "eval", "detached": True})
    assert tracker.state == sessions.THINKING  # it writes its next candidate meanwhile
    tracker.tool_started("c1", "mcp__ka__evaluation_result")
    assert tracker.state == sessions.QUEUED
    tracker.gpu({"state": "start", "id": 7, "kind": "eval", "detached": True})
    assert (tracker.state, tracker.detail) == (sessions.ON_GPU, "eval")
    tracker.gpu({"state": "done", "id": 7, "detached": True})
    tracker.tool_ended("c1")
    assert tracker.state == sessions.THINKING and tracker.evaluations == 1
    tracker.close(status="done")
    observer.detach()


# ------------------------------------------------------------------ the dry run


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def test_dry_run_with_async_evals(tmp_path, fast):
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, dossier=False)
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    world = dryrun.World(orch, virtual=True)
    icfg = ImproveConfig(agents=3, rounds=1, async_evals=True)
    improver = Improver(orch, icfg, require_capture=False, live_charts=False)

    async def main():
        with world.driving():
            return await improver.improve()

    with world.installed():
        reason = asyncio.run(main())
    assert reason.startswith("every arm has stopped") and not orch.async_evals
    state = read_json(orch.run.root / "improve.json")
    assert state["config"]["async_evals"] and all(s["status"] == "done" for s in state["slices"])
    log = read_jsonl(orch.run.root / gpuqueue.FILE)
    assert any(e.get("detached") for e in log if e["kind"] == "eval")
    # every submitted evaluation became a row of its session (none lost when a session ended)
    ended = [r for r in read_jsonl(orch.run.root / sessions.FILE) if r["state"] == "ended"]
    rows = [r for r in ledger.rows(orch.run) if r["backend"] != "integrate" and r["session"]]
    for record in ended:
        if record["label"].startswith("kernel-"):
            mine = [r for r in rows if r["session"] == record["label"]]
            assert len(mine) == record["evaluations"]
    kernel = [s for s in world.sessions if s["name"].startswith("kernel-")]
    other = [s for s in world.sessions if s["name"] == "systems"]
    assert kernel and all("# Evaluations in flight" in s["system"] for s in kernel)
    assert other and not any("# Evaluations in flight" in s["system"] for s in other)
