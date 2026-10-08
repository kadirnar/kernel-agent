"""Run budgets: no GPU and no Claude (fake ``run_agent`` / fake Claude Code CLI)."""

import asyncio
import functools
import json
import os
import sys
import time
from pathlib import Path

import pytest
from ab_fake import with_ab
from claude_agent_sdk import ClaudeAgentOptions

from kernel_agent import cli, orchestrator
from kernel_agent.agent import runner
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.runner import AgentResult
from kernel_agent.budget import (
    PLATEAU,
    Budget,
    Standing,
    improves,
    non_improving_streak,
    results_streak,
)
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels.evaluate import COMPILE_MARKER, _timeout_result
from kernel_agent.workspace import RunDir, append_jsonl, read_json, read_jsonl, write_json

CARD = {"repo_id": "org/m", "modality": "llm", "architectures": ["X"], "params": 1}


# ------------------------------------------------------------ advice signals


def test_non_improving_streak():
    recs = [
        {"status": "build_error", "correct": False},
        {"correct": True, "speedup": 1.2},  # new best (beats the 1.0 reference)
        {"correct": True, "speedup": 1.205},  # within 1 % noise
        {"correct": False, "speedup": 3.0},  # incorrect never counts
        {"correct": True, "speedup": 1.3},  # new best
        {"correct": True, "speedup": 1.4, "cases": [{"timing_spread": 0.1}]},  # < 1.3 * 1.2
        {"status": "timeout", "correct": False},
    ]
    assert non_improving_streak(recs[:2]) == 0
    assert non_improving_streak(recs) == 2
    assert non_improving_streak([{"correct": True, "speedup": 0.9}]) == 1  # slower than reference
    assert improves({"passed": True, "speedup": 1.5}, 1.0, ok_key="passed")
    assert not improves({"passed": True, "speedup": 1.5}, 1.0)  # wrong ok key
    assert non_improving_streak([{"passed": True, "speedup": 1.1}], ok_key="passed") == 0


def test_streak_uses_the_standing_records():
    """attn_fused of the VoxCPM2 run (#64): 46.89x recorded by the evaluator before #6/#7,
    re-evaluated at 18.1x; a later 20x is a new best, as in the ledger (``keep``)."""

    def rec(speedup, name, **extra):
        out = {"correct": speedup is not None, "speedup": speedup, "snapshot": f"history/{name}"}
        return out | {"cases": [{"timing_spread": 0.005}], **extra}

    stale = rec(46.89, "001_v1.py")
    records = [stale, rec(15.0, "002_v2.py"), rec(None, "003_v3.py")]
    again = rec(18.1, "001_v1.py", reevaluates={"exp": 1, "speedup": 46.89})
    assert non_improving_streak([*records, again]) == 2  # the re-evaluation is not the agent's
    assert non_improving_streak([*records, again, rec(20.0, "004_v4.py")]) == 0
    assert non_improving_streak([*records, rec(20.0, "004_v4.py")]) == 3  # not re-evaluated
    assert non_improving_streak([*records, again, rec(18.2, "004_v4.py")]) == 3  # within noise

    stand = Standing()
    stand.keep(stale)
    assert stand.best == 46.89 and stand.top is stale
    stand.replace(again)  # stands instead of the stale record
    assert stand.rows == [again] and stand.best == 18.1 and stand.top is again
    stand.replace(rec(None, "001_v1.py", reevaluates={"exp": 4}))  # wrong by the current one
    assert stand.rows == [] and stand.best == 1.0 and stand.top is None
    slower = Standing(rows=[rec(0.8, "001_v1.py")])  # never below the reference
    assert slower.best == 1.0 and slower.top is None


def test_feedback_restart_counts_evaluations_only(tmp_path):
    """A research plan restarts the streak after 3 evaluations; a re-evaluation appended
    since is not an evaluation of the agent."""
    run = RunDir.create(tmp_path, "org/m")
    results = run.target("t") / "results.jsonl"
    for name, speedup in (("a", 1.5), ("b", 1.2), ("c", 1.1)):
        append_jsonl(results, {"correct": True, "speedup": speedup, "snapshot": f"history/{name}"})
    again = {"correct": True, "speedup": 1.45, "snapshot": "history/a", "reevaluates": {"exp": 1}}
    append_jsonl(results, again)
    append_jsonl(results, {"correct": True, "speedup": 1.3, "snapshot": "history/d"})
    budget = Budget(run, restarted={"kernel-t": 3})
    assert results_streak(results) == 3
    assert budget.feedback("kernel-t", results, None)["budget"]["non_improving"] == 1


def test_feedback_advice(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    results = run.target("t") / "results.jsonl"
    budget, evals = Budget(run), PLATEAU + 2  # the session's evaluation budget
    budget.start_agent("kernel-t")
    append_jsonl(results, {"correct": True, "speedup": 1.5})
    fb = budget.feedback("kernel-t", results, evals)
    assert fb["advice"] == "continue"
    assert fb["budget"] == {
        "evals_used": 1,
        "evals_budget": PLATEAU + 2,
        "minutes_left": None,
        "non_improving": 0,
    }
    for _ in range(PLATEAU):
        append_jsonl(results, {"correct": True, "speedup": 1.4})
        fb = budget.feedback("kernel-t", results, evals)
    assert fb["advice"] == "consider_stopping" and fb["budget"]["non_improving"] == PLATEAU
    assert results_streak(results) == PLATEAU
    append_jsonl(results, {"correct": True, "speedup": 2.0})
    fb = budget.feedback("kernel-t", results, evals)
    assert fb["advice"] == "stop" and fb["budget"]["evals_used"] == PLATEAU + 2
    assert "evaluation budget" in fb["advice_reason"]

    # A new session restarts the count; a near deadline means "stop".
    budget = Budget(run, agent_minutes=1.0)
    budget.start_agent("kernel-t")
    fb = budget.feedback("kernel-t", results, None)
    assert fb["advice"] == "stop" and fb["budget"]["evals_used"] == 1
    assert 0.9 < fb["budget"]["minutes_left"] <= 1.0


def test_budget_limits(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    cfg = OptimizeConfig(model_ref="org/m", budget_usd_per_agent=5.0)
    assert Budget(run).exhausted() is None
    assert Budget(run).start_agent("a") is None
    assert Budget(run).agent_config(cfg) is cfg
    assert Budget(run).prompt_note("a", OptimizeConfig(model_ref="x"), []) == ""

    write_json(run.root / "costs.json", {"planner": {"usd": 1.5}, "x": {"usd": 0.5}})
    budget = Budget(run, max_usd=4.0, max_hours=1.0, agent_minutes=30)
    assert budget.spent_usd() == 2.0
    assert budget.agent_config(cfg).budget_usd_per_agent == 2.0  # min(5, 4 - 2)
    assert budget.exhausted() is None
    timeout = budget.start_agent("kernel-t")
    assert timeout == 30 * 60  # the run still has 51 min for agents
    note = budget.prompt_note("kernel-t", budget.agent_config(cfg), ["mcp__ka__evaluate_candidate"])
    assert "about 30 min" in note and "$2.00" in note and "consider_stopping" in note

    budget.started -= 0.85 * 3600 - 60  # 1 min before the 15 % reserve starts
    assert "time budget spent" in (budget.exhausted() or "")
    assert budget.start_agent("planner") == 120.0  # never less than MIN_AGENT_SECONDS
    write_json(run.root / "costs.json", {"planner": {"usd": 3.9}})
    assert "USD budget spent" in (Budget(run, max_usd=4.0).exhausted() or "")


def test_final_integration_reserve(tmp_path):
    """The improve loop's estimate of its final integration is kept from the agents when it
    is longer than ``reserve`` of the time budget, and estimated again after every
    evaluation (issue #100)."""
    run = RunDir.create(tmp_path, "org/m")
    budget = Budget(run, max_hours=2.0, final_reserve_s=600.0)  # less than 15 % (18 min)
    assert budget.reserve_s() == pytest.approx(0.15 * 2 * 3600)
    budget.final_reserve_s = 3600.0
    assert budget.reserve_s() == 3600.0
    assert budget.agent_seconds_left() == pytest.approx(3600.0, abs=5)
    assert budget.start_agent("kernel-t") == pytest.approx(3600.0, abs=5)  # sessions too
    budget.started -= 3600 - 60
    assert "60 min kept for integrate + report" in (budget.exhausted() or "")

    results = run.root / "results.jsonl"
    append_jsonl(results, {"correct": True, "speedup": 1.5})
    budget = Budget(run, max_hours=2.0, estimate_reserve=lambda: 2 * 3600 - 60)
    budget.start_agent("kernel-t")
    fb = budget.feedback("kernel-t", results, None)  # it added an item: no time left
    assert budget.final_reserve_s == 2 * 3600 - 60
    assert fb["advice"] == "stop" and "time is up" in fb["advice_reason"]


def test_timeout_result_reports_compile_time():
    late = _timeout_result(300, f"warn\n{COMPILE_MARKER}42.5\nmore\n".encode())
    assert late["status"] == "timeout" and late["compile_s"] == 42.5
    assert "compile + first call took 42s" in late["error"]
    early = _timeout_result(300, None)
    assert early["compile_s"] is None and "compiling?" in early["error"]


def test_cli_budget_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_optimize", lambda ns: seen.append(cli._config(ns)) or 0)
    flags = ["--max-hours", "2", "--max-usd", "5", "--agent-minutes", "20", "--eval-timeout", "90"]
    assert cli.main(["optimize", "org/m", *flags]) == 0
    cfg = seen[0]
    assert (cfg.max_hours, cfg.max_usd, cfg.agent_minutes) == (2.0, 5.0, 20.0)
    assert (cfg.eval_timeout_s, cfg.budget_reserve) == (90.0, 0.15)
    assert OptimizeConfig.from_dict(cfg.to_dict()).max_usd == 5.0


# ------------------------------------------------------------ evaluation tool


def test_evaluate_candidate_carries_budget(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    (tdir / "capture.pt").write_bytes(b"")
    (tdir / "candidates" / "v1.py").write_text("def build(r): ...\n")
    timeouts = []

    def fake_eval(capture, snap, *, profile, timeout, **_):
        timeouts.append(timeout)
        return {"status": "ok", "correct": True, "speedup": 1.0, "compile_s": 3.0}

    monkeypatch.setattr(tools_mod, "run_evaluation", fake_eval)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    budget = Budget(run, eval_timeout_s=123.0)
    session = tools_mod.SessionBinding(evaluations=PLATEAU + 1)
    server = {t.name: t for t in tools_mod.build_server(run, budget, binding=session)}

    async def evaluate(version: int) -> dict:
        # a new kernel each time: the same source again would be a duplicate (dedup.py)
        (tdir / "candidates" / "v1.py").write_text(f"V = {version}\ndef build(r): ...\n")
        args = {"target_id": "t", "candidate": "candidates/v1.py", "hypothesis": "same kernel"}
        out = await server["evaluate_candidate"].handler(args)
        return json.loads(out["content"][0]["text"])

    outs = [asyncio.run(evaluate(i)) for i in range(PLATEAU + 1)]
    assert timeouts == [123.0] * (PLATEAU + 1)
    assert outs[0]["compile_s"] == 3.0 and outs[0]["advice"] == "continue"
    assert outs[PLATEAU - 1]["advice"] == "consider_stopping"  # 1.0x never beats the reference
    assert outs[PLATEAU]["advice"] == "stop"
    assert outs[PLATEAU]["budget"]["evals_used"] == PLATEAU + 1


# ------------------------------------------------------------ orchestrator


class FakeToolchain:
    env: dict[str, str] = {}
    backends = {"triton": True}
    gpu = None

    def summary(self) -> str:
        return "GPU: none (test)"


def make_orchestrator(tmp_path, monkeypatch, calls, *, sleep=0.0, costs=None, **budget):
    """Synthetic run (analyze/plan/capture done, targets t1 + t2) with a fake run_agent."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", FakeToolchain)
    run = RunDir.create(tmp_path, "org/m")
    budget = {"dossier": False, **budget}  # sessions counted: no dossier (test_web.py)
    cfg = OptimizeConfig(model_ref="org/m", runs_dir=tmp_path, backends=["triton"], **budget)
    phases = {p: {"done": True} for p in ("analyze", "plan", "capture")}
    write_json(
        run.run_json, {"card": CARD, "workload": {}, "config": cfg.to_dict(), "phases": phases}
    )
    write_json(run.baseline_json, {"median_ms": 10.0, "workload": "test"})
    run.profile_dir.mkdir()
    (run.profile_dir / "summary.md").write_text("# Profile\n")
    write_json(run.plan_json, {"analysis": "a", "targets": [], "transforms": []})
    for tid in ("t1", "t2"):
        spec = {"id": tid, "module_class": "M", "backends": ["triton"], "why": "w", "approach": "a"}
        write_json(run.target(tid) / "spec.json", spec)
    (run.root / "events.jsonl").touch()

    async def fake_run_agent(name, *, cfg, system_append, cwd, result=None, **kwargs):
        result = result or AgentResult(name=name)
        calls.append({"name": name, "usd_cap": cfg.budget_usd_per_agent, "system": system_append})
        result.session_id, result.turns = f"sess-{name}", 1
        try:
            await asyncio.sleep(sleep)
        except asyncio.CancelledError:
            calls[-1]["cancelled"] = True
            raise
        if name == "kernel-t1":  # leaves one correct kernel behind
            snap = cwd / "history" / "001_v1.py"
            snap.parent.mkdir()
            snap.write_text("def build(r): ...\n")
            append_jsonl(
                cwd / "results.jsonl",
                {"correct": True, "speedup": 1.5, "snapshot": "history/001_v1.py"},
            )
        result.cost_usd = (costs or {}).get(name, 0.0)
        return result

    monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
    e2e = []

    def fake_worker(run_dir, command, *args, **kwargs):
        e2e.append(args)
        return {"status": "ok", "passed": True, "median_ms": 8.0, "speedup": 1.25}

    monkeypatch.setattr(orchestrator, "call_worker", with_ab(fake_worker, base_ms=10.0))
    return orchestrator.Orchestrator(run, cfg), e2e


def test_agent_timeout_is_enforced(tmp_path, monkeypatch):
    calls: list[dict] = []
    orch, _ = make_orchestrator(
        tmp_path, monkeypatch, calls, sleep=60.0, agent_minutes=0.01, do_transforms=False
    )
    t0 = time.monotonic()
    asyncio.run(orch.run_all(until="kernels"))
    assert time.monotonic() - t0 < 10  # 2 agents x 0.6 s, not 2 x 60 s
    assert [c["name"] for c in calls] == ["kernel-t1", "kernel-t2"]
    assert all(c.get("cancelled") for c in calls)
    assert "about 1 min for this session" in calls[0]["system"]
    costs = read_json(orch.run.root / "costs.json")
    assert costs["kernel-t1"]["timed_out"] and costs["kernel-t1"]["session_id"] == "sess-kernel-t1"
    assert costs["kernel-t1"]["turns"] == 1
    kernels = orch.run.load()["phases"]["kernels"]
    assert kernels["done"] and kernels["finished"] == ["t1", "t2"]
    assert [t["agent"] for t in kernels["timed_out"]] == ["kernel-t1", "kernel-t2"]
    events = read_jsonl(orch.run.root / "events.jsonl")
    timed_out = [e for e in events if e["event"] == "timed_out"]
    assert [e["agent"] for e in timed_out] == ["kernel-t1", "kernel-t2"]
    assert orch.agent_results[0].timed_out


def test_usd_budget_stops_new_agents_but_integrate_and_report_run(tmp_path, monkeypatch):
    calls: list[dict] = []
    orch, e2e = make_orchestrator(
        tmp_path, monkeypatch, calls, costs={"kernel-t1": 2.5}, max_usd=3.0
    )
    write_json(orch.run.root / "costs.json", {"planner": {"usd": 1.0}})
    asyncio.run(orch.run_all())

    assert [c["name"] for c in calls] == ["kernel-t1"]  # t2 + systems skipped: $3.5 spent
    assert calls[0]["usd_cap"] == 2.0  # the agent itself is capped at what was left
    assert "at most $2.00" in calls[0]["system"]
    phases = orch.run.load()["phases"]
    assert [s["agent"] for s in phases["kernels"]["budget_skipped"]] == ["kernel-t2"]
    assert "USD budget spent" in phases["kernels"]["budget_skipped"][0]["reason"]
    assert phases["transforms"]["skipped"]
    assert phases["transforms"]["budget_skipped"][0]["agent"] == "systems"
    # integrate + report still ran on what exists (t1's kernel)
    assert phases["integrate"]["done"] and phases["integrate"]["speedup"] == 1.25
    assert any("t1=" in " ".join(a) for a in e2e)
    assert read_json(orch.run.root / "integration.json")["accepted"][0]["kind"] == "kernel"
    assert phases["report"]["done"]
    report = orch.run.report.read_text()
    assert "## Budget stops" in report and "kernel-t2" in report
    assert read_json(orch.run.root / "costs.json")["kernel-t1"]["usd"] == 2.5


def test_time_budget_skips_agents(tmp_path, monkeypatch):
    calls: list[dict] = []
    orch, _ = make_orchestrator(tmp_path, monkeypatch, calls, max_hours=0.01)  # 36 s
    asyncio.run(orch.run_all())
    assert calls == []
    phases = orch.run.load()["phases"]
    assert "time budget spent" in phases["kernels"]["budget_skipped"][0]["reason"]
    assert phases["integrate"]["done"] and phases["report"]["done"]
    assert read_json(orch.run.root / "integration.json")["final"] is None


# ------------------------------------------------------------ real SDK, fake CLI

FAKE_CLI = """#!{python}
import json, os, sys, time
if sys.argv[1:] == ["-v"]:
    print("2.1.286 (Claude Code)")
    sys.exit(0)
open(os.environ["FAKE_CLI_PIDFILE"], "w").write(str(os.getpid()))
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        emit({{"type": "control_response", "response": {{"subtype": "success",
              "request_id": msg["request_id"], "response": {{}}}}}})
    elif msg.get("type") == "user":
        emit({{"type": "system", "subtype": "init", "session_id": "sess-fake"}})
        tool = {{"type": "tool_use", "id": "tu1", "name": "mcp__ka__evaluate_candidate",
                "input": {{"candidate": "candidates/v1.py"}}}}
        emit({{"type": "assistant", "session_id": "sess-fake", "message": {{"id": "msg_1",
              "model": "fake", "content": [tool]}}}})
        time.sleep(3600)  # a long tool call that ignores stdin EOF: needs SIGTERM
"""


def _alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text().split()[2] != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake CLI")
def test_timeout_terminates_claude_subprocess(tmp_path, monkeypatch):
    """Cancelling run_agent (as asyncio.timeout does) kills the CLI and keeps partial stats."""
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLI.format(python=sys.executable))
    fake.chmod(0o755)
    pidfile = tmp_path / "cli.pid"
    monkeypatch.setenv("FAKE_CLI_PIDFILE", str(pidfile))
    options = functools.partial(ClaudeAgentOptions, cli_path=str(fake))
    monkeypatch.setattr(runner, "ClaudeAgentOptions", options)
    run = RunDir.create(tmp_path, "org/m")
    result = AgentResult(name="k")

    async def main() -> None:
        async with asyncio.timeout(2.0):
            await runner.run_agent(
                "k",
                prompt="go",
                system_append="",
                cwd=tmp_path,
                cfg=OptimizeConfig(model_ref="org/m"),
                mcp_server=tools_mod.build_server(run),
                mcp_tools=[],
                env={},
                log_dir=tmp_path / "logs",
                result=result,
            )

    with pytest.raises(TimeoutError):
        asyncio.run(main())
    assert not _alive(int(pidfile.read_text()))
    assert result.session_id == "sess-fake" and result.turns == 1
    assert result.tool_calls == {"evaluate_candidate": 1} and result.seconds >= 2.0
