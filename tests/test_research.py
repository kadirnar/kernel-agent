"""Idea ledger and clean-context research sessions on plateaued targets (issue #15).

CPU only, no Claude: the improve loop runs against ``kernel_agent.dryrun``.
"""

import asyncio
import itertools
import json
import re
from pathlib import Path

import pytest

from kernel_agent import charts, dryrun, ledger, orchestrator, research, status
from kernel_agent.agent import prompts, runner
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.runner import AgentResult
from kernel_agent.agent.tools import record_candidate, snapshot
from kernel_agent.budget import PLATEAU, Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver, kernel_digest, systems_digest
from kernel_agent.scheduler import KERNEL, SYSTEMS, Arm, Policy, build_arms, plateau
from kernel_agent.workspace import RunDir, append_jsonl, read_json, read_jsonl, write_json


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No real toolchain probing and no charts."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path, seed=0, **cfg):
    cfg.setdefault("quality", "exact")  # an exact run (new runs default to relaxed, #175)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, **cfg)
    orch = orchestrator.Orchestrator(dryrun.create_run(config, seed), config)
    return orch, dryrun.World(orch)


def loop(orch, world, **icfg):
    improver = Improver(orch, ImproveConfig(**icfg), require_capture=False, live_charts=False)
    with world.installed():
        reason = asyncio.run(improver.improve())
    return improver, reason


def evaluate(run, target, outcome, *, idea="", expected=None):
    """Record one kernel evaluation (``outcome``: a speedup or a failure status)."""
    src = run.target(target) / "candidates" / "v.py"
    src.write_text("import triton\n")
    snap = snapshot(run, src, target)
    result = dryrun.kernel_result(outcome, 300.0, 28, dryrun.TARGETS[0])
    _, row = record_candidate(
        run,
        target,
        src,
        snap,
        result,
        hypothesis=f"h {outcome}",
        idea=idea,
        expected_speedup=expected,
    )
    return row


def row(idea, status, speedup=None, exp=1, **extra):
    correct = status in (ledger.KEEP, ledger.DISCARD)
    out = {"idea": idea, "status": status, "correct": correct, "speedup": speedup, "exp": exp}
    return out | {"hypothesis": f"h{exp}", **extra}


# ------------------------------------------------------------------ idea ledger


def test_idea_slug_and_aggregation():
    assert ledger.idea_slug("  Split-K GEMV (v2)!") == "split-k_gemv_v2"
    assert ledger.idea_slug("__launch_bounds__(256)") == "launch_bounds_256"
    assert ledger.idea_slug(None) == "" and ledger.idea_slug("___") == ""
    assert len(ledger.idea_slug("x" * 99)) == 40

    rows = [
        row("splitk", "build_error", exp=1),
        row("splitk", "incorrect", exp=2),
        row("splitk", "keep", 1.4, exp=3, expected_speedup=1.6),
        row("", "discard", 1.1, exp=4),  # untagged rows are not an idea
        row("tile64", "discard", 1.2, exp=5),
        row("tile64", "discard", 1.3, exp=6),
        row("persist", "timeout", exp=7),
        row("persist", "runtime_error", exp=8),
    ]
    stats = {s["idea"]: s for s in ledger.ideas(rows)}
    assert list(stats) == ["splitk", "tile64", "persist"]  # order of the first try
    splitk = stats["splitk"]
    assert (splitk["tries"], splitk["bugs"], splitk["kept"], splitk["slow"]) == (3, 2, 1, 0)
    assert splitk["statuses"] == {"build_error": 1, "incorrect": 1, "keep": 1}
    assert splitk["best"] == 1.4 and splitk["expected"] == 1.6 and splitk["verdict"] == "kept"
    assert splitk["exps"] == [1, 2, 3] and splitk["last_hypothesis"] == "h3"
    tile = stats["tile64"]  # measured correct and never a new best
    assert tile["verdict"] == "slow" and tile["best"] == 1.3 and tile["bugs"] == 0
    persist = stats["persist"]  # never correct: untested, not refuted
    assert persist["verdict"] == "buggy" and persist["best"] is None and persist["slow"] == 0

    assert ledger.labelled({"idea": "splitk", "hypothesis": "h"}) == "[splitk] h"
    assert ledger.labelled({"hypothesis": "h"}) == "h"
    table = research.ideas_table(ledger.ideas(rows))
    assert table[-1].startswith("| `persist` | 2 |") and "buggy: retry" in table[-1]
    assert "2 (timeout 1, runtime_error 1)" in table[-1]
    assert "| 1.400x | 1.60x | 1 | 0 | 2 (build_error 1, incorrect 1) | kept |" in table[2]


def test_ledger_idea_column_and_old_ledgers(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    result = {"status": "ok", "correct": True, "speedup": 1.4, "cases": []}
    new = ledger.record_kernel(run, "t", result, snapshot="001_a.py", hypothesis="h", idea="splitk")
    assert new["idea"] == "splitk" and ledger.rows(run)[0]["idea"] == "splitk"
    header = run.ledger.read_text().splitlines()[0].split("\t")
    assert header[-3:] == ["idea", "title", "hypothesis"] and tuple(header) == ledger.COLUMNS
    assert "[splitk] h" in status.render(run, width=200)

    # a ledger written before the column existed keeps its own layout
    old = RunDir.create(tmp_path / "old", "org/m")
    old.ledger.write_text("\t".join(c for c in ledger.COLUMNS if c != "idea") + "\n")
    ledger.record_kernel(old, "t", result, snapshot="001_a.py", hypothesis="aligned", idea="x")
    (r,) = ledger.rows(old)
    assert r["hypothesis"] == "aligned" and r["speedup"] == 1.4 and "idea" not in r
    assert ledger.ideas(ledger.rows(old)) == [] and ledger.labelled(r) == "aligned"
    assert "aligned" in status.render(old, width=200)

    # runs without a ledger: rebuilt from results.jsonl, with the idea
    legacy = RunDir.create(tmp_path / "legacy", "org/m")
    write_json(legacy.run_json, {"created": "2026-10-05 10:00:00"})
    write_json(legacy.target("t") / "spec.json", {"id": "t"})
    rec = {"time": "10:01:00", "snapshot": "history/001_a.py", "status": "incorrect"}
    append_jsonl(legacy.results_file("t"), rec | {"correct": False, "idea": "splitk"})
    (r,) = ledger.backfill(legacy)
    assert r["idea"] == "splitk" and r["status"] == "incorrect"
    assert ledger.ideas([r])[0]["verdict"] == "buggy"


def test_evaluate_candidate_echoes_ideas_and_best_result_aggregates(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    (tdir / "capture.pt").write_bytes(b"")
    (tdir / "candidates" / "v1.py").write_text("def build(r): ...\n")
    outcomes = iter(
        [
            {"status": "build_error", "correct": False, "error": "nvcc: error"},
            {"status": "ok", "correct": True, "speedup": 1.3, "cases": []},
            {"status": "ok", "correct": True, "speedup": 1.31, "cases": []},
        ]
    )
    monkeypatch.setattr(tools_mod, "run_evaluation", lambda *a, **k: next(outcomes))
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    server = {t.name: t for t in tools_mod.build_server(run, Budget(run))}

    def call(tool, **args):
        out = asyncio.run(server[tool].handler(args))
        return json.loads(out["content"][0]["text"])

    versions = iter(range(10))

    def evaluate_candidate(hypothesis, **args):  # a changed kernel each time (no duplicates)
        (tdir / "candidates" / "v1.py").write_text(f"V = {next(versions)}\ndef build(r): ...\n")
        return call(
            "evaluate_candidate",
            target_id="t",
            candidate="candidates/v1.py",
            hypothesis=hypothesis,
            **args,
        )

    bug = evaluate_candidate("split-K GEMV", idea_id="Split-K GEMV", expected_speedup=1.5)
    idea = bug["idea"]
    assert idea["id"] == "split-k_gemv" and idea["expected_speedup"] == 1.5
    assert idea["speedup"] is None and idea["verdict"] == "buggy" and idea["bugs"] == 1
    assert "not evidence against the idea" in idea["note"] and "'split-k_gemv'" in idea["note"]

    fixed = evaluate_candidate(
        "split-K, index fixed", idea_id="split-k_gemv", expected_speedup="1.5x"
    )
    idea = fixed["idea"]
    assert idea["vs_expected"] == "1.300x measured vs 1.500x expected"
    assert idea["tries"] == 2 and idea["verdict"] == "kept" and idea["best"] == 1.3
    assert "note" not in idea

    untagged = evaluate_candidate("same kernel again", expected_speedup="fast")
    assert "idea" not in untagged  # neither an idea nor a usable expectation

    records = read_jsonl(run.results_file("t"))
    assert [r["idea"] for r in records] == ["split-k_gemv", "split-k_gemv", None]
    assert [r["expected_speedup"] for r in records] == [1.5, 1.5, None]
    assert [r["idea"] for r in ledger.rows(run)] == ["split-k_gemv", "split-k_gemv", ""]

    best = call("best_result", target_id="t")
    (agg,) = best["ideas"]
    assert agg["idea"] == "split-k_gemv" and agg["tries"] == 2 and agg["bugs"] == 1
    assert agg["statuses"] == {"build_error": 1, "keep": 1} and agg["expected"] == 1.5
    assert best["untagged"] == 1 and best["history"][0]["idea"] == "split-k_gemv"


# ------------------------------------------------------------------ plateau trigger


def test_plateau_triggers():
    policy = Policy()  # patience 5

    def arm(**kw):
        return Arm("a", KERNEL, 100.0, **kw)

    assert plateau(arm(streak=PLATEAU - 1), policy) is None
    assert "4 evaluations in a row" in plateau(arm(streak=PLATEAU), policy)  # consider_stopping
    assert "5 evaluations in a row" in plateau(arm(streak=5), policy)  # patience reached
    assert "3 failed evaluations" in plateau(arm(streak=3, fails=3), policy)
    assert plateau(arm(streak=2, fails=2), policy) is None
    # other stop rules are not plateaus: no research session for them
    assert plateau(arm(streak=9, sol=0.95), policy) is None
    assert plateau(arm(streak=9, best=2.1), policy) is None
    assert plateau(arm(streak=9, hours=3.0), policy) is None
    assert plateau(Arm(SYSTEMS, SYSTEMS, 100.0, streak=9), policy) is None
    assert plateau(arm(streak=2), Policy(patience=2)) is not None  # a smaller patience
    assert plateau(arm(streak=PLATEAU), Policy(patience=0)) is not None


def test_research_plan_restarts_the_streak(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    for outcome in (1.3, "incorrect", 1.2, "build_error", "incorrect", "timeout"):
        evaluate(run, "attn", outcome)

    def attn(research=None):
        return next(a for a in build_arms(run, Policy(), [], research=research) if a.id == "attn")

    before = attn()
    assert (before.streak, before.fails, before.best) == (5, 3, 1.3)
    assert "plateau" in before.stop
    planned = [{"arm": "attn", "exp": 6, "plan": True}, {"arm": "mlp", "exp": 6, "plan": True}]
    after = attn(planned)
    assert (after.streak, after.fails, after.best, after.stop) == (0, 0, 1.3, None)
    assert attn([{"arm": "attn", "exp": 6, "plan": False}]).streak == 5  # no plan: no restart
    evaluate(run, "attn", "incorrect")
    assert (attn(planned).streak, attn(planned).fails) == (1, 1)

    # the evaluation advice of the next slice counts from the plan too
    improver = Improver(orch, ImproveConfig(), require_capture=False, live_charts=False)
    improver.state["research"] = planned
    improver._restart_advice(attn(planned))
    assert orch.budget.restarted == {"kernel-attn": 6}
    out = orch.budget.feedback("kernel-attn", run.results_file("attn"), None)
    assert out["advice"] == "continue" and out["budget"]["non_improving"] == 1
    orch.budget.restarted.clear()
    out = orch.budget.feedback("kernel-attn", run.results_file("attn"), None)
    assert out["advice"] == "consider_stopping" and out["budget"]["non_improving"] == 6
    improver.state["research"] = []
    orch.budget.restarted["kernel-attn"] = 3
    improver._restart_advice(attn())
    assert orch.budget.restarted == {}


def test_research_due_cap_and_stopped_arms(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    for outcome in (1.3, 1.2, 1.25, 1.1, 1.0):
        evaluate(run, "attn", outcome)
    improver = Improver(
        orch, ImproveConfig(policy=Policy(patience=4)), require_capture=False, live_charts=False
    )

    def attn():
        return next(a for a in improver.arms() if a.id == "attn")

    assert "plateau" in attn().stop  # patience 4 reached ...
    assert "4 evaluations" in improver.research_due(attn())  # ... but research comes first
    live = {a.id: a for a in improver._pickable()}
    assert live["attn"].stop is None and live["mlp"].stop is None

    # a plan that brought no new best: the plateau stands, the arm stays stopped
    plan = {"arm": "attn", "after_slice": 0, "exp": 5, "plan": True, "best": 1.3}
    improver.state["research"] = [plan]
    assert attn().stop is None and attn().streak == 0  # the plan restarted the count
    for outcome in (1.1, 1.2, 1.0, 1.25):
        evaluate(run, "attn", outcome)
    assert "plateau" in attn().stop and improver.research_due(attn()) is None
    assert "plateau" in next(a for a in improver._pickable() if a.id == "attn").stop

    # a plan that paid: again after research_every (3) slices of the arm, not before
    plan["best"] = 1.2
    slices = [{"n": n, "arm": a, "evals": 4} for n, a in enumerate(["attn", "mlp", "attn"], 1)]
    improver.state["slices"] = slices
    assert improver.research_due(attn()) is None
    improver.state["slices"].append({"n": 4, "arm": "attn", "evals": 4})
    assert "4 evaluations" in improver.research_due(attn())
    improver.icfg.research_every = 0  # off
    assert improver.research_due(attn()) is None


# ------------------------------------------------------------------ the loop (dry run)


def test_dry_run_research_trigger_and_cap(tmp_path):
    orch, world = make(tmp_path, seed=6)  # this seed plateaus in every way
    _, reason = loop(orch, world, research_every=2)
    run = orch.run
    assert reason.startswith("every arm has stopped")
    state = read_json(run.root / "improve.json")
    slices, sessions = state["slices"], state["research"]
    assert sessions and all(s["status"] == "done" and s["plan"] for s in sessions)
    whys = [s["why"] for s in sessions]
    assert any(re.match(r"[3-9] failed evaluations in a row", w) for w in whys)
    assert any(re.match(rf"{PLATEAU} evaluations in a row without a new best", w) for w in whys)
    # past --patience (5): the arm had stopped and was picked again for its research session
    assert any(re.match(r"([5-9]|\d\d) evaluations in a row without a new best", w) for w in whys)
    for s in sessions:  # every research session is followed by a slice of its arm
        assert any(x["arm"] == s["arm"] and x["n"] > s["after_slice"] for x in slices)
        assert (run.target(s["arm"]) / research.PLAN_FILE).is_file()
    arms = {s["arm"] for s in sessions}
    assert any(sum(s["arm"] == arm for s in sessions) > 1 for arm in arms)
    for arm in arms:  # the cap: research_every slices of the arm and a gain in between
        mine = [s for s in sessions if s["arm"] == arm]
        for a, b in itertools.pairwise(mine):
            between = [x for x in slices if x["arm"] == arm]
            between = [x for x in between if a["after_slice"] < x["n"] <= b["after_slice"]]
            assert len(between) >= 2 and b["best"] > a["best"]

    # through Orchestrator._agent: program.md, events, costs
    reviews = [x for x in world.sessions if x["name"].startswith("research-")]
    assert len(reviews) == len(sessions)
    system, brief = reviews[0]["system"], reviews[0]["prompt"]
    assert "pathology checklist" in system and "# Evidence" in brief and "## Ideas" in brief
    assert "# Program" in system and "## research" in system and "## kernel" not in system
    assert {r["system"] for r in reviews} == {system}  # one prefix for every review (#181)
    assert {s["label"] for s in sessions} <= set(read_json(run.root / "costs.json"))
    events = ledger.events(run)
    starts = [e for e in events if e["event"] == "agent_start" and e["agent"].startswith("res")]
    assert len(starts) == len(sessions) == sum(e["event"] == "research_done" for e in events)

    # the next slice of the arm is a fresh session that reads the plan and starts from it
    first = sessions[0]
    at = world.sessions.index(reviews[0])
    after = next(x for x in world.sessions[at:] if x["name"] == f"kernel-{first['arm']}")
    assert "## Research plan" in after["prompt"] and "## Ranked directions" in after["prompt"]
    top = re.search(r"^1\. `([a-z0-9_-]+)`", after["prompt"].split("## Research plan")[1], re.M)
    rows = [r for r in ledger.rows(run) if r["target"] == first["arm"]]
    assert next(r for r in rows if r["exp"] > first["exp"])["idea"] == top[1]

    # every kernel evaluation carries an idea; buggy ideas are retried under the same id
    kernel_rows = [r for r in ledger.rows(run) if r["target"] != ledger.E2E]
    assert all(r["idea"] for r in kernel_rows)
    assert any(r["hypothesis"].startswith("fix of exp ") for r in kernel_rows)
    assert "Research sessions on plateaued targets" in run.report.read_text()


def test_dry_run_without_research(tmp_path):
    orch, world = make(tmp_path)
    loop(orch, world, research_every=0)
    assert read_json(orch.run.root / "improve.json")["research"] == []
    assert not any(x["name"].startswith("research-") for x in world.sessions)
    assert not list(orch.run.targets_dir.glob(f"*/{research.PLAN_FILE}"))


def test_interrupted_research_is_recovered_without_a_plan(tmp_path):
    orch, world = make(tmp_path)
    loop(orch, world, max_slices=2)
    path = orch.run.root / "improve.json"
    state = read_json(path)
    state["research"] = [
        {"n": 1, "arm": "attn", "after_slice": 2, "exp": 3, "status": "running", "started": 1.0}
    ]
    write_json(path, state)
    Improver(orch, ImproveConfig(), require_capture=False)._recover()
    (rec,) = read_json(path)["research"]
    assert rec["status"] == "interrupted" and rec["plan"] is False and rec["ended"] == 1.0


# ------------------------------------------------------------------ the session


def test_orchestrator_research_session(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    evaluate(run, "attn", "build_error", idea="splitk")
    evaluate(run, "attn", 1.5, idea="splitk", expected=1.8)
    evaluate(run, "attn", 1.4, idea="tile64")
    calls = []

    async def fake(name, **kwargs):
        calls.append((name, kwargs))
        return AgentResult(name=name, cost_usd=0.5)

    orch.agent_runner = fake
    why = "4 evaluations in a row without a new best"
    asyncio.run(orch.research("attn", reason=why, label="research-attn#3"))
    ((name, kw),) = calls
    plan = run.target("attn") / "plan.md"
    assert name == "research-attn" and kw["cwd"] == run.target("attn")
    dossier = run.target("attn") / "research.md"  # the web tools on: it may update it
    assert kw["tools"] == ["Read", "Glob", "Grep", "Write"] and kw["writable"] == [plan, dossier]
    assert kw["mcp_tools"] == ["mcp__ka__best_result"] and why in kw["prompt"]
    # the role's part (system prompt) and the target's (first message, #181)
    system, brief = kw["system_append"], kw["prompt"]
    for text in ("# Diagnose: pathology checklist", "Repetition loop", "Correctness wall"):
        assert text in system
    assert "the ceiling, not the current number" in system and str(plan) in brief
    assert "## Do not try" in system and "## Retry (failed, not refuted)" in system
    assert "attn" not in system and str(run.root) not in system
    # the evidence: why, the best result with its speed of light per case, ideas, rows
    assert f"* {why}. 3 evaluations so far: 1 kept, 1 failed." in brief
    assert "1.500x module speedup" in brief
    assert "% of its recipe's roofline (memory bound)" in brief
    assert "| `a0[1, 1, 1024]:bfloat16` | 127 |" in brief
    assert "| `splitk` | 2 | 1.500x | 1.80x |" in brief and "| `tile64` | 1 | 1.400x |" in brief
    assert "| exp | status | speedup | % SOL | idea |" in brief
    assert brief.index("# Evidence") < brief.index("# Task\nTarget `attn` has plateaued")
    assert "# Program" in system and "untested, not refuted" in system  # `## research`
    assert read_json(run.root / "costs.json")["research-attn#3"]["usd"] == 0.5


def test_write_guard_and_restricted_tools(tmp_path, monkeypatch):
    plan = tmp_path / "plan.md"
    (matcher,) = runner.write_guard([plan], tmp_path)["PreToolUse"]
    assert matcher.matcher == runner.WRITE_TOOLS

    def decision(tool, **tool_input):
        out = asyncio.run(matcher.hooks[0]({"tool_name": tool, "tool_input": tool_input}, "t", {}))
        return (out.get("hookSpecificOutput") or {}).get("permissionDecision", "allow")

    assert decision("Write", file_path=str(plan), content="# Plan") == "allow"
    assert decision("Write", file_path="plan.md") == "allow"  # relative to the cwd
    assert decision("Edit", file_path=str(tmp_path / "x" / ".." / "plan.md")) == "allow"
    assert decision("Write", file_path=str(tmp_path / "NOTES.md")) == "deny"
    assert decision("Edit", file_path=str(tmp_path / "x" / "plan.md")) == "deny"
    assert decision("NotebookEdit", notebook_path=str(tmp_path / "n.ipynb")) == "deny"
    assert decision("Write") == "deny"

    seen = []

    async def fake_query(*, prompt, options):
        seen.append(options)
        if False:
            yield

    monkeypatch.setattr(runner, "query", fake_query)
    cfg = OptimizeConfig(model_ref="org/m", allow_web=False)

    def run_agent(**kwargs):
        common = {"prompt": "go", "system_append": "", "cwd": tmp_path, "cfg": cfg}
        common |= {"mcp_server": None, "env": {}, "log_dir": tmp_path / "logs"}
        return asyncio.run(runner.run_agent("a", **common, **kwargs))

    run_agent(mcp_tools=tools_mod.tool_names("best_result"), tools=[*runner.READ_TOOLS, "Write"])
    restricted = seen[-1]
    assert restricted.tools == ["Read", "Glob", "Grep", "Write", "Skill"]  # + skills (#176)
    assert len(restricted.hooks["PreToolUse"]) == 1  # only the Claude files guard (#126)
    assert set(restricted.allowed_tools) == {
        "Read",
        "Glob",
        "Grep",
        "Write",
        "mcp__ka__best_result",
        "mcp__ka__doc_search",  # the doc library: every session (#177)
        "mcp__ka__doc_read",
    }
    run_agent(mcp_tools=[], tools=["Read", "Write"], writable=[plan])
    assert len(seen[-1].hooks["PreToolUse"]) == 2  # write guard + Claude files guard
    run_agent(mcp_tools=[])  # the other agents: unchanged
    assert seen[-1].tools is None and "Bash" in seen[-1].allowed_tools
    assert len(seen[-1].hooks["PreToolUse"]) == 1  # the Claude files guard (#126)


# ------------------------------------------------------------------ prompts and digests


def test_engineer_prompt_ideas_and_research_prompt():
    target = {"id": "rms", "module_class": "LlamaRMSNorm", "why": "w", "approach": "a"}
    info = {"qualname": "model.norm", "cases": [{"signature": "a0[1, 1, 8]", "count": 3}]}
    text = prompts.engineer_prompt(target, info, ["cuda", "cute"], "py", "GPU: x", 8, None)
    assert 'idea_id="<slug>", expected_speedup=1.4' in text and "# Ideas" in text
    assert "3-5 distinct ideas" in text and "abandoned after N attempts" in text
    assert "`plan.md` (when present)" in text and "bugs (failed attempts) vs slow" in text
    # the backend skills the prompt names (#176) hold the process traps of long debug loops
    assert "`kernel-agent:cuda-kernels`" in text and "`kernel-agent:cute-dsl`" in text
    for guide in ("cuda.md", "cute_dsl.md", "cuda-kernels", "cute-dsl"):
        assert "abandoned after N" in prompts.knowledge(guide)
        assert "Plan amnesia" in prompts.knowledge(guide)
        assert "False infeasibility" in prompts.knowledge(guide)

    review = prompts.research_prompt(target, info, "EVIDENCE", Path("/r/plan.md"), "GPU: x")
    assert "EVIDENCE" in review and "`/r/plan.md`" in review and "`a0[1, 1, 8]`" in review
    assert all(f"\n{i}. **" in review for i in range(1, 10))  # the nine pathologies
    assert "You do not write kernel" in review and "best_result" in review


def test_digest_has_ideas_and_the_plan(tmp_path):
    orch, _ = make(tmp_path)
    run = orch.run
    evaluate(run, "attn", "incorrect", idea="splitk")
    evaluate(run, "attn", 1.4, idea="splitk")
    plan = "# Plan\n\n## Ranked directions\n1. `persist`: persistent CTAs\n" + "x\n" * 4000
    (run.target("attn") / "plan.md").write_text(plan)
    arm = next(a for a in build_arms(run, Policy(), []) if a.id == "attn")
    digest = kernel_digest(run, arm, 3, 4, Policy())
    assert "## Ideas so far" in digest and "| `splitk` | 2 | 1.400x |" in digest
    assert "| exp | status | speedup | idea |" in digest
    assert "## Research plan" in digest and "1. `persist`: persistent CTAs" in digest
    assert "(cut: read plan.md for the rest)" in digest
    assert len(digest) < research.PLAN_CHARS + 8000
    (run.target("attn") / "plan.md").unlink()
    assert "## Research plan" not in kernel_digest(run, arm, 3, 4, Policy())

    # the systems arm's best may be transforms on top of kernels (#40)
    systems = next(a for a in build_arms(run, Policy(), []) if a.id == SYSTEMS)
    systems.best, systems.best_snapshot = 1.6, "002_cuda_graph_0a1b2c3d.py+attn"
    text = systems_digest(run, systems, 4, 4, Policy())
    assert "## Best end-to-end configuration so far (transforms, plus kernels if listed)" in text
    assert "`002_cuda_graph_0a1b2c3d.py+attn`" in text and "Best transform so far" not in text
