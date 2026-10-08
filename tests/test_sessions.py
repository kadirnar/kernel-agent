"""Per-agent observability (``sessions.py``, issue #184): session states from the runner's
hooks, messages and the GPU queue; the time split; sessions.jsonl and the rate-limited
events; improve.json's live sessions / gpu; ``kernel-agent status``'s Agents table and GPU
block; ``watch``'s swimlanes; improve.png's sub-lanes and GPU strip; report.md's Concurrency.

No GPU and no Claude: fake sessions on a fake clock, and a virtual-time dry run with
``--agents 3`` (``dryrun.World``)."""

import asyncio
import itertools
import json
import math
import re
import shutil
import subprocess
import time

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    UserMessage,
)

from kernel_agent import (
    charts,
    dryrun,
    gpuqueue,
    improve,
    ledger,
    orchestrator,
    sessions,
    status,
    watch,
)
from kernel_agent.agent import runner
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

LABEL = "kernel-attn#1"


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(ledger, "clock", c)
    return c


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    return RunDir(root)


def gpu_event(state, job, t, **extra):
    base = {"ts": t, "event": "gpu_job", "state": state, "id": job, "kind": "eval"}
    return {**base, "class": "eval", "session": LABEL, "target": "attn", **extra}


def result(cost=0.5):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=3,
        session_id="s",
        total_cost_usd=cost,
    )


# ------------------------------------------------------------------ one session


def test_states_and_the_time_split(run, clock):
    """Every source of a state: the init message, tool hooks, an evaluation tool, its GPU
    job's queue events, a denied call closed by its tool result, the turn's result."""
    clock.t = 100.0
    obs = sessions.Observer(run, reserved=lambda label: 1.5)
    tracker = obs.open(LABEL, agent="kernel-attn", role="kernel")
    assert tracker.arm == "attn" and tracker.reserved == 1.5
    steps = [
        (102, lambda: tracker.see(SystemMessage(subtype="init", data={})), "thinking"),
        (110, lambda: tracker.tool_started("t1", "Bash", {"command": "nvcc"}), "tool"),
        (130, lambda: tracker.tool_ended("t1"), "thinking"),
        (
            140,
            lambda: tracker.tool_started("t2", "mcp__ka__evaluate_candidate", {"candidate": "c"}),
            "evaluating",
        ),
        (141, lambda: obs._gpu(gpu_event("queued", 7, 141)), "queued"),
        (150, lambda: obs._gpu(gpu_event("start", 7, 150, wait_s=9.0)), "on_gpu"),
        (160, lambda: obs._gpu(gpu_event("done", 7, 160, hold_s=10.0)), "evaluating"),
        (162, lambda: tracker.tool_ended("t2"), "thinking"),
        (163, lambda: tracker.tool_started("t3", "Write"), "tool"),
        (  # a denied call: no PostToolUse, its tool result closes the span
            165,
            lambda: tracker.see(UserMessage(content=[ToolResultBlock("t3", "denied", True)])),
            "thinking",
        ),
        (
            166,
            lambda: tracker.tool_started("t4", "Agent", {"subagent_type": "doc-lookup"}),
            "helper",
        ),
        (168, lambda: tracker.tool_ended("t4"), "thinking"),
        (170, lambda: tracker.see(result()), "idle"),
    ]
    for t, step, want in steps:
        clock.t = t
        step()
        assert tracker.state == want, (t, want)
    clock.t = 175.0
    split = tracker.close(AgentResult(name="kernel-attn", cost_usd=0.5))
    assert split == {
        "model": 24.0,
        "gpu": 10.0,
        "queued": 9.0,
        "eval": 3.0,
        "tools": 22.0,
        "idle": 7.0,
    }
    assert tracker.evaluations == 1 and tracker.status == "done"
    assert not obs.trackers and obs._unlisten is None  # nothing open: not listening

    # sessions.jsonl tells the same session: its states, split, evaluations, USD
    [session] = sessions.replay(sessions.read(run))
    assert session.split() == pytest.approx(split)
    assert (session.role, session.arm, session.status, session.evaluations) == (
        "kernel",
        "attn",
        "done",
        1,
    )
    assert session.usd == 0.5 and session.reserved == 1.5
    states = [s for _, _, s, _ in session.segments]
    assert states[:4] == ["starting", "thinking", "tool", "thinking"]
    assert ("on_gpu", "eval") in [(s, d) for _, _, s, d in session.segments]
    # events.jsonl: the start and the end only (one state change per EVENT_S at most)
    events = [e for e in ledger.events(run) if e["event"] == "session_state"]
    assert [e["state"] for e in events] == ["starting", "ended"]
    assert events[-1]["split"] == split and events[-1]["evaluations"] == 1


def test_quick_checks_are_not_evaluations(run, clock):
    obs = sessions.Observer(run)
    tracker = obs.open("systems#2", agent="systems", role="systems")
    tracker.tool_started("q", "mcp__ka__evaluate_candidate", {"mode": "quick"})
    tracker.tool_ended("q")
    tracker.tool_started("e", "mcp__ka__evaluate_e2e", {})
    tracker.tool_ended("e")
    tracker.tool_ended("e")  # a second end (the hook, then the tool result): once
    assert tracker.evaluations == 1 and tracker.arm == "systems"
    assert sessions.arm_of("kernel-attn-w2") == "attn" and sessions.arm_of("planner") is None
    tracker.close(status="failed")
    tracker.close()  # once
    assert [r["state"] for r in sessions.read(run)].count("ended") == 1


def test_session_state_events_are_rate_limited(run, clock):
    """A state change every 10 s for an hour: every one in sessions.jsonl, at most one
    session_state event per EVENT_S in events.jsonl."""
    obs = sessions.Observer(run)
    tracker = obs.open(LABEL, agent="kernel-attn", role="kernel")
    for i in range(360):
        clock.t += 10.0
        if i % 2:
            tracker.tool_ended(f"t{i - 1}")
        else:
            tracker.tool_started(f"t{i}", "Bash")
    clock.t += 1.0
    tracker.close()
    records = sessions.read(run)
    assert len(records) == 362
    events = [e for e in ledger.events(run) if e["event"] == "session_state"]
    assert len(events) <= 2 + math.ceil(3600 / sessions.EVENT_S)
    assert len(events) >= 3 and all(e["label"] == LABEL for e in events)
    gaps = [b["ts"] - a["ts"] for a, b in itertools.pairwise(events[1:-1])]
    assert all(g >= sessions.EVENT_S for g in gaps)


def test_recover_closes_what_a_crash_left_open(run, clock):
    path = run.root / sessions.FILE
    path.write_text(
        json.dumps(
            {"ts": 10.0, "label": "a#1", "state": "starting", "agent": "a", "role": "kernel"}
        )
        + "\n"
        + json.dumps({"ts": 40.0, "label": "a#1", "state": "on_gpu", "detail": "eval"})
        + "\n"
        + '{"ts": 41, "label": "a#1", "sta'  # a line being written: left out
    )
    sessions.Observer(run).open("b#1", agent="b", role="kernel")  # the next process's first session
    a = next(s for s in sessions.replay(sessions.read(run)) if s.label == "a#1")
    assert a.status == "interrupted" and a.end == 40.0
    assert a.split() == pytest.approx({**dict.fromkeys(sessions.PARTS, 0.0), "idle": 30.0})


# ------------------------------------------------------------------ the runner's hooks


def test_run_agent_reports_tool_spans_and_turns(tmp_path, monkeypatch, clock):
    """runner.run_agent installs the session's PreToolUse / PostToolUse hooks (every tool,
    beside the write guards) and hands it every message."""
    run = RunDir(tmp_path)
    seen = []
    obs = sessions.Observer(run)
    box = {}

    async def fake_query(*, prompt, options):
        tracker = box["tracker"]
        hooks = options.hooks
        assert any(m.matcher is None for m in hooks["PostToolUseFailure"])
        pre = [h for m in hooks["PreToolUse"] if m.matcher is None for h in m.hooks]
        post = [h for m in hooks["PostToolUse"] if m.matcher is None for h in m.hooks]
        yield SystemMessage(subtype="init", data={"session_id": "s"})
        seen.append(tracker.state)
        clock.t += 5
        yield AssistantMessage(content=[TextBlock("go")], model="m")
        clock.t += 5
        for hook in pre:
            assert await hook({"tool_name": "Bash", "tool_input": {}}, "b1", None) == {}
        seen.append(tracker.state)
        clock.t += 20
        for hook in post:
            await hook({"tool_name": "Bash"}, "b1", None)
        seen.append(tracker.state)
        clock.t += 5
        yield result()
        seen.append(tracker.state)

    monkeypatch.setattr(runner, "query", fake_query)
    common = {"prompt": "go", "system_append": "P", "cwd": tmp_path, "env": {}, "mcp_tools": []}
    common |= {"cfg": OptimizeConfig(model_ref="m"), "mcp_server": None, "log_dir": tmp_path}

    async def main():
        tracker = box["tracker"] = obs.open(LABEL, agent="kernel-attn", role="kernel")
        out = AgentResult(name="kernel-attn")
        async with tracker.running(out):
            await runner.run_agent("kernel-attn", result=out, **common)
        assert tracker.state == "idle"
        return tracker.close(out)

    split = asyncio.run(main())
    assert seen == ["thinking", "tool", "thinking", "idle"]
    assert split["tools"] == 20.0 and split["model"] == 15.0


def test_a_session_that_raises_ends_failed(run, clock):
    obs = sessions.Observer(run)

    async def main():
        tracker = obs.open(LABEL, agent="kernel-attn", role="kernel")
        with pytest.raises(RuntimeError):
            async with tracker.running():
                raise RuntimeError("the CLI died")
        return tracker

    tracker = asyncio.run(main())
    assert tracker.status == "failed" and not obs.trackers


# ------------------------------------------------------------------ the live snapshot


def test_improve_json_snapshot_of_sessions_and_gpu(run, clock, monkeypatch):
    monkeypatch.setattr(sessions, "SAVE_S", 0.01)
    saved = []

    async def main():
        obs = sessions.Observer(run)
        obs.attach(saved.append)  # the improve loop's save: a snapshot at once
        tracker = obs.open(LABEL, agent="kernel-attn", role="kernel")
        tracker.tool_started("e", "evaluate_candidate", {})
        clock.t += 4
        obs._gpu(gpu_event("start", 1, clock.t, wait_s=0.0))
        background = {"ts": clock.t, "event": "gpu_job", "state": "queued", "id": 2}
        obs._gpu({**background, "kind": "integration", "class": "background"})
        for _ in range(3):  # many changes: a few saves (at most one per SAVE_S)
            tracker.turn("thinking")
            tracker.turn("idle")
        await asyncio.sleep(0.05)
        clock.t += 6
        snap = obs.snapshot()
        obs._gpu(gpu_event("done", 1, clock.t, hold_s=6.0))
        tracker.tool_ended("e")
        tracker.close()
        obs.detach()
        return snap

    snap = asyncio.run(main())
    assert 2 <= len(saved) < 10
    [live] = snap["sessions"]
    assert (live["label"], live["state"], live["detail"]) == (LABEL, "on_gpu", "eval")
    assert live["split"]["gpu"] == 6.0 and live["split"]["eval"] == 4.0
    gpu = snap["gpu"]
    assert [h["session"] for h in gpu["holders"]] == [LABEL]
    assert gpu["queue"] == {"background": {"jobs": 1, "max_wait_s": 6.0}}
    assert gpu["waiting"][0]["kind"] == "integration" and gpu["busy_1h"] == 0.6
    assert saved[-1]["sessions"] == []  # the last save: nothing runs


# ------------------------------------------------------------------ views


def test_lanes_pack_concurrent_sessions_and_coarsen():
    records = []
    for label, a, b in (("a#1", 0, 100), ("b#2", 10, 50), ("c#3", 60, 90), ("d#4", 100, 130)):
        records.append(
            {"ts": a, "label": label, "state": "starting", "agent": label, "role": "kernel"}
        )
        records += [
            {"ts": a + 1 + i, "label": label, "state": ("thinking", "tool")[i % 2]}
            for i in range(10)
        ]
        records.append({"ts": b, "label": label, "state": "ended", "status": "done"})
    holds = []
    for i, (a, b, who) in enumerate(((5, 10, "a#1"), (10.5, 20, None), (20.2, 30, None))):
        holds += [
            {
                "ts": a,
                "state": "start",
                "id": i,
                "kind": "eval" if who else "integration",
                "session": who,
            },
            {"ts": b, "state": "done", "id": i},
        ]
    lanes = sessions.lanes(records, holds, columns=13)  # 10 s steps
    assert [(s["label"], s["lane"]) for s in lanes["sessions"]] == [
        ("a#1", 0),
        ("b#2", 1),
        ("c#3", 1),
        ("d#4", 0),
    ]
    assert lanes["lanes"] == 2 and lanes["start"] == 0 and lanes["end"] == 130
    a = lanes["sessions"][0]
    assert len(a["segments"]) < 11  # 1 s states at 10 s steps: coarsened
    assert a["segments"][0][0] == 0 and a["segments"][-1][1] == 100
    assert lanes["gpu"] == [
        [5, 10, "agent", "eval", "a#1"],
        [10.5, 30, "background", "integration", None],
    ]
    assert lanes["busy"] == pytest.approx(24.3 / 130, abs=1e-3)
    json.dumps(lanes, allow_nan=False)
    assert sessions.lanes([], []) == {}


def test_improve_png_sub_lanes():
    slices = [
        {"arm": "a", "started": 0, "ended": 50, "sessions": ["a-w1#1", "a-w2#1"]},
        {"arm": "a", "started": 60, "ended": 80, "sessions": ["a#2"]},
        {"arm": "b", "started": 0, "ended": 30},
    ]
    spans = {"a-w1#1": (0, 50), "a-w2#1": (5, 40), "a#2": (60, 80)}
    bars = [(s["arm"], a, b, k, n) for s, a, b, k, n in improve._bars(slices, spans)]
    assert bars == [("a", 0, 50, 0, 2), ("b", 0, 30, 0, 1), ("a", 5, 40, 1, 2), ("a", 60, 80, 0, 2)]


# ------------------------------------------------------------------ the dry run, --agents 3


@pytest.fixture(scope="module")
def three(tmp_path_factory):
    """A virtual-time dry run with --agents 3, and what status and the obs showed in it."""
    seen: dict[str, object] = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
        mp.setattr(charts, "available", lambda: False)
        config = OptimizeConfig(
            model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path_factory.mktemp("three"), dossier=False
        )
        orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
        world = dryrun.World(orch, virtual=True)
        clock = world.clock

        def hook(agent, used):  # mid-run: the status and the live snapshot
            if "status" not in seen and len(orch.observer.trackers) >= 3:
                with pytest.MonkeyPatch.context() as now:  # the simulated wall clock
                    now.setattr(status.time, "time", clock.now)
                    seen["status"] = status.render(orch.run, width=140)
                seen["snapshot"] = orch.observer.snapshot()

        world.hook = hook
        improver = Improver(
            orch, ImproveConfig(agents=3, max_slices=8), require_capture=False, live_charts=False
        )

        async def main():
            with world.driving():
                return await improver.improve()

        with world.installed():
            asyncio.run(main())
    return orch.run, seen


def test_dry_run_records_every_session(three):
    run, _ = three
    done = sessions.replay(sessions.read(run))
    costs = read_json(run.root / "costs.json")
    assert {s.label for s in done} == set(costs) and all(s.end is not None for s in done)
    last = {s.label: s for s in done}  # costs.json: a label's last session
    for s in done:  # the split covers the session
        assert sum(s.split().values()) == pytest.approx(s.seconds(), abs=0.01)
    for label, s in last.items():
        assert costs[label]["time"] == pytest.approx(s.split(), abs=0.11)
    kernel = sessions.total_split(s for s in done if s.role == "kernel")
    assert all(kernel[p] > 0 for p in ("model", "gpu", "queued", "tools"))
    assert 0.3 < kernel["tools"] / (kernel["tools"] + kernel["model"]) <= dryrun.OWN_RUNS
    evaluations = [
        r for r in ledger.rows(run) if r.get("session") and r["status"] not in ledger.UNMEASURED
    ]
    assert sum(s.evaluations for s in done) == len(evaluations)
    # GPU states come from the queue: a session's on_gpu time is its jobs' hold time
    holds = sessions.gpu_holds(sessions.read_gpu(run))
    for s in done:
        mine = sum(h.end - h.start for h in holds if h.session == s.label)
        assert s.split()["gpu"] == pytest.approx(mine, abs=0.05)
    # events.jsonl stays small: session_state at the start, the end, at most every EVENT_S
    events = ledger.events(run)
    states = [e for e in events if e["event"] == "session_state"]
    hours = sum(s.seconds() for s in done) / 3600
    assert len(states) <= 2 * len(done) + hours * 3600 / sessions.EVENT_S + len(done)
    assert len(sessions.read(run)) > 3 * len(states) // 2
    waited = [h for h in holds if h.wait >= sessions.GPU_EVENT_S]
    assert len([e for e in events if e["event"] == "gpu_job"]) == len(waited) > 0


def test_dry_run_live_views(three):
    run, seen = three
    text = str(seen["status"])
    assert "agents: 3 running (--agents 3)" in text
    head = next(line for line in text.splitlines() if line.startswith("session "))
    assert head.split() == ["session", "role", "arm", "for", "evals", "$", "state"]
    assert "agent time (" in text and "GPU queue:" in text
    snap = seen["snapshot"]
    assert len(snap["sessions"]) == 3 and "busy_1h" in snap["gpu"]
    states = {s["state"] for s in snap["sessions"]}
    assert states <= {"thinking", "tool", "evaluating", "queued", "on_gpu", "idle", "starting"}
    final = read_json(run.root / "improve.json")
    assert final["sessions"] == [] and "busy_1h" in final["gpu"]


def test_dry_run_watch_status_report(three):
    run, _ = three
    state = watch.state(run)
    lanes = state["lanes"]
    assert 1 < lanes["lanes"] <= 3 and lanes["gpu"]
    assert len(lanes["sessions"]) == len(sessions.read(run)) - len(
        [r for r in sessions.read(run) if "role" not in r]
    )
    assert not any(e["event"] == "session_state" for e in state["events"])  # not in the feed
    json.dumps(state, allow_nan=False)
    text = status.render(run, width=160)
    assert "agent time (" in text and "running (--agents" not in text
    assert "last hour of the queue: GPU busy" in text
    report = run.report.read_text()
    section = report.split("## Concurrency", 1)[1].split("\n## ", 1)[0]
    assert re.search(r"sessions at once on average, at most \d \(--agents 3\)", section)
    assert "| **all** |" in section and "| own runs | idle |" in section
    assert "GPU busy" in section and "p95" in section


def test_dry_run_improve_png_has_the_gpu_strip(three, tmp_path):
    if not charts.available():
        pytest.skip("matplotlib not installed")
    run, _ = three
    copy = RunDir(tmp_path / "run")
    shutil.copytree(run.root, copy.root)
    assert improve.slices_chart(copy) == copy.root / "improve.png"


def test_watch_page_script_parses(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = watch.PAGE.read_text()
    script = page.split('<script nonce="__NONCE__">', 1)[1].split("</script>", 1)[0]
    (tmp_path / "watch.js").write_text(script)
    subprocess.run([node, "--check", str(tmp_path / "watch.js")], check=True)
    assert "drawLanes" in script and 'id="chart-lanes"' in page


def test_watch_sends_lanes_in_deltas(tmp_path, clock, monkeypatch):
    run = RunDir(tmp_path)
    write_json(run.run_json, {"created": "2026-10-08 10:00:00"})
    watcher = watch.Watcher(run)
    assert watcher.snapshot()["lanes"] == {}
    obs = sessions.Observer(run)
    tracker = obs.open(LABEL, agent="kernel-attn", role="kernel")
    tracker.tool_started("b", "Bash")
    monkeypatch.setattr(watch, "LANES_S", 0.0)
    kind, delta = watcher.poll()
    assert kind == "delta" and [s["label"] for s in delta["lanes"]["sessions"]] == [LABEL]
    assert "events" not in delta  # the session_state event is not in the feed
    tracker.tool_ended("b")
    monkeypatch.setattr(watch, "LANES_S", 60.0)
    assert watcher.poll() is None  # within LANES_S of the last lanes: later
    monkeypatch.setattr(watch, "LANES_S", 0.0)
    assert "lanes" in watcher.poll()[1]
    tracker.close()


def test_gpu_lines_and_queue_by_class(tmp_path):
    run = RunDir(tmp_path)
    now = time.time()
    for record in (
        {"ts": now - 100, "state": "start", "id": 1, "kind": "eval", "class": "eval", "wait_s": 0},
        {"ts": now - 40, "state": "done", "id": 1, "hold_s": 60},
        {"ts": now - 30, "state": "queued", "id": 2, "kind": "e2e", "class": "e2e"},
        {"ts": now - 20, "state": "start", "id": 3, "kind": "eval", "class": "eval", "wait_s": 30},
    ):
        gpuqueue_file = run.root / gpuqueue.FILE
        with gpuqueue_file.open("a") as fh:
            fh.write(json.dumps({"event": "gpu_job", **record}) + "\n")
    lines = sessions.gpu_lines(run)
    assert lines[0].startswith("  queue by class: e2e 1 (longest 30 s)")
    assert "GPU busy 80%" in lines[1] and "p95" in lines[1]
    assert read_jsonl(run.root / gpuqueue.FILE)[0]["id"] == 1
