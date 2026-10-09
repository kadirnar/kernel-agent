"""Records timed in another timing context than their target's current one (#226): a
target's context comes from its run's newest profile, and an improve round's re-profile can
flip it (eager -> graph once the systems agent graphs a stage). What a record's speedup is in
a context (``kernels/context.py``: ``comparable``, ``in_context``), and every place that
compares an evaluation with older records of its target, from fixture records and profiles:
the ledger's keep bar, the early-discard bar, the budget's streak, an idea's ``refuted``
verdict, the duplicate cache, the best-record ranking, the scheduler's arms and the
integration's stale check, re-check reuse and choice of a target's kernel."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from kernel_agent import budget, critic, dryrun, ledger, orchestrator, scheduler
from kernel_agent.agent import tools as tools_mod
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels import evaluate
from kernel_agent.kernels.context import comparable, current, in_context, timed_in
from kernel_agent.kernels.roofline import sol_signal
from kernel_agent.workspace import RunDir, append_jsonl, write_json

EAGER, GRAPH = ("eager", "warm"), ("graph", "warm")
SCHEMA = {"evaluator_version": {"schema": evaluate.EVALUATOR_SCHEMA}}
UNCAPTURABLE = "unavailable (case 0: operation not permitted when stream is capturing)"


def result(speedup: float, context: str = "eager", other: Any = None, **extra: Any) -> dict:
    """A correct evaluation timed in ``context`` (``other``: its speedup in the other context,
    a winner's; a string: why it could not be timed there)."""
    by_context: dict[str, Any] = {context: speedup}
    if other is not None:
        by_context["graph" if context == "eager" else "eager"] = other
    return {
        "status": "ok",
        "correct": True,
        "speedup": speedup,
        "context": context,
        "l2": "warm",
        "speedup_by_context": by_context,
        **SCHEMA,
        **extra,
    }


#: An eager winner, timed in a CUDA graph too: 2.0x eagerly (host time saved), 1.2x there.
EAGER_WIN = result(
    2.0,
    other=1.2,
    ref_ms_weighted=4.0,
    new_ms_weighted=2.0,
    est_saved_ms_per_run=2.0,
    pct_of_sol=40.0,
    bound="launch",
    launch_floor_ms=0.0194,
    sol_ms_weighted=0.8,
    cases=[
        {
            "calls_per_run": 2,
            "target_calls": 2,
            "ref_ms": 2.0,
            "new_ms": 1.0,
            "speedup": 2.0,
            "sol_ms": 0.4,
            "pct_of_sol": 40.0,
            "bound": "launch",
            "timing": {
                "eager": {"ref_ms": 2.0, "new_ms": 1.0, "speedup": 2.0},
                "graph": {"ref_ms": 0.6, "new_ms": 0.5, "speedup": 1.2},
            },
        }
    ],
)


def test_a_records_speedup_in_a_timing_context():
    assert timed_in({"speedup": 1.5}) == EAGER  # a record from before #226: eager, warm
    assert comparable({"correct": True, "speedup": 1.5}, EAGER) == (1.5, None)
    assert comparable(EAGER_WIN, EAGER) == (2.0, None)
    assert comparable(EAGER_WIN, GRAPH) == (1.2, None)  # its graph timing (a winner's)
    loss_speed, why = comparable(result(0.6), GRAPH)  # a loser is timed in its context only
    assert loss_speed is None and why is not None
    assert why.startswith(
        "the timing context changed: timed eagerly with a warm L2, the target is now timed "
        "in a CUDA graph with a warm L2"
    )
    assert "only a winner is timed in the other context too" in why
    unsafe = comparable(result(3.0, other=UNCAPTURABLE), GRAPH)[1]
    assert unsafe is not None and f"graph: {UNCAPTURABLE}" in unsafe
    cold = comparable(EAGER_WIN, ("eager", "cold"))[1]  # another L2: never measured
    assert cold is not None and "cold L2 was never measured" in cold
    assert comparable({"status": "build_error", "correct": False}, GRAPH) == (None, None)


def test_a_record_seen_from_the_other_context():
    original = copy.deepcopy(EAGER_WIN)
    assert in_context(EAGER_WIN, EAGER) is EAGER_WIN
    view = in_context(EAGER_WIN, GRAPH)
    assert view is not None and original == EAGER_WIN  # a copy: the record is unchanged
    assert (view["speedup"], view["context"], view["l2"]) == (1.2, "graph", "warm")
    assert view["measured_in"] == {"context": "eager", "l2": "warm", "speedup": 2.0}
    assert "timed eagerly with a warm L2 at 2.0x" in view["context_note"]
    (case,) = view["cases"]
    assert (case["ref_ms"], case["new_ms"], case["speedup"]) == (0.6, 0.5, 1.2)
    assert (view["ref_ms_weighted"], view["new_ms_weighted"]) == (1.2, 1.0)  # 2 calls per run
    assert view["est_saved_ms_per_run"] == 0.2  # 2 target calls x 0.1 ms
    # the roofline of the eager times and launch floor is not this context's; the work is
    for key in ("pct_of_sol", "bound", "launch_floor_ms"):
        assert key not in view and key not in case
    assert view["sol_ms_weighted"] == 0.8 and case["sol_ms"] == 0.4
    assert sol_signal(view) is None and sol_signal(EAGER_WIN) == 40.0
    assert in_context(result(0.6), GRAPH) is None  # no graph speedup: not comparable
    failure = {"status": "incorrect", "correct": False}
    assert in_context(failure, GRAPH) is failure  # nothing timed: the same in any context


def test_the_keep_bar_is_the_best_in_the_new_evaluations_context(tmp_path):
    run = RunDir.create(tmp_path, "org/m")

    def record(res: dict[str, Any], name: str, *, write: bool = True) -> str:
        row = ledger.record_kernel(run, "t", res, snapshot=name, hypothesis=name)
        if write:  # the tool appends the record right after its ledger row
            append_jsonl(run.results_file("t"), {**res, "exp": row["exp"], "snapshot": name})
        return str(row["status"])

    assert record(EAGER_WIN, "001_a.py") == ledger.KEEP
    assert record(result(3.0, other=UNCAPTURABLE), "002_b.py") == ledger.KEEP
    # after a re-profile flipped the target to graph: 1.2x (the eager winner's graph timing)
    # is the bar; the 3.0x of a kernel that cannot be captured sets none (an eager bar of
    # 3.0x would discard both)
    assert record(result(1.3, "graph"), "003_c.py") == ledger.KEEP
    assert record(result(1.25, "graph"), "004_d.py") == ledger.DISCARD
    assert record(result(2.5), "005_e.py") == ledger.DISCARD  # eager: 3.0x is its bar
    assert ledger.bar(run, "t", GRAPH) == 1.3 and ledger.bar(run, "t", EAGER) == 3.0
    # a keep row whose record is not written yet (by another session, this moment) keeps its
    # own speedup
    assert record(result(1.6, "graph"), "006_f.py", write=False) == ledger.KEEP
    assert ledger.bar(run, "t", GRAPH) == 1.6
    rows = ledger.in_context(run, "t", ledger.rows(run), GRAPH)
    assert [r["speedup"] for r in rows] == [1.2, None, 1.3, 1.25, 2.5, 1.6]  # keeps mapped


def test_the_streak_compares_each_evaluation_in_its_own_context():
    # without contexts, 1.3x and 1.25x in a CUDA graph would be two evaluations without a new
    # best after 2.0x eagerly
    records = [EAGER_WIN, result(1.3, "graph"), result(1.25, "graph")]
    assert budget.non_improving_streak(records) == 1  # 1.3 beat 1.2 in a graph; 1.25 did not
    assert budget.non_improving_streak(records[:2]) == 0
    assert budget.non_improving_streak([EAGER_WIN, result(1.9)]) == 1  # eager: 2.0x stands
    stand = budget.Standing()
    stand.keep(EAGER_WIN)
    stand.keep(result(3.0, other=UNCAPTURABLE))
    assert (stand.best_in(EAGER), stand.best_in(GRAPH), stand.best) == (3.0, 1.2, 3.0)


def test_ideas_tried_in_another_context_are_not_refuted_there():
    records = [{**EAGER_WIN, "idea": "base", "ledger_status": ledger.KEEP}]
    records += [
        {**result(s), "idea": "fp8", "ledger_status": ledger.DISCARD} for s in (0.6, 0.62, 0.58)
    ]
    eager = tools_mod.idea_stats(records, "fp8", EAGER)
    assert eager is not None and eager["verdict"] == "refuted"  # 3 tries far below 2.0x
    assert eager["refuted"] == {"tries": 3, "best": 0.62, "target_best": 2.0}
    graph = tools_mod.idea_stats(records, "fp8", GRAPH)
    assert graph is not None and graph["verdict"] == "slow" and graph["tries"] == 3
    assert graph["best"] is None  # no try has a speedup in a CUDA graph: untested there
    graph_rows = tools_mod.idea_rows(records, GRAPH)
    assert graph_rows[0]["speedup"] == 1.2 and "context_stale" not in graph_rows[0]
    assert all("the timing context changed" in r["context_stale"] for r in graph_rows[1:])
    assert tools_mod.idea_rows(records) == tools_mod.idea_rows(records, EAGER)  # as before


def test_stale_says_when_the_timing_context_changed():
    loss = result(0.6)
    assert evaluate.stale(loss) is None and evaluate.stale(loss, EAGER) is None
    why = evaluate.stale(loss, GRAPH)
    assert why is not None and why.startswith("the timing context changed")
    assert evaluate.stale(EAGER_WIN, GRAPH) is None  # its graph speedup compares
    old = {k: v for k, v in EAGER_WIN.items() if k != "evaluator_version"}
    older = evaluate.stale(old, GRAPH)  # an older evaluator first
    assert older is not None and "evaluator_version was recorded" in older


# ------------------------------------------------------------------ the tools


def profile(graph: bool) -> dict[str, Any]:
    """A profile whose timeline stage ``model`` holds the target's instance, launched by
    CUDA-graph replays or not."""
    stage = {"stage": "model", "events": 1000, "graph_events": 1000 if graph else 0}
    return {"kernel_view": {"timeline": {"stages": [stage]}}}


def kernel_run(tmp_path: Path) -> tuple[RunDir, Path]:
    """A run whose target ``t`` (instance ``model.layers.0``) runs eagerly in its profile."""
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    run.capture_file("t").write_bytes(b"")
    spec = {"id": "t", "module_class": "Layer", "capture": {"qualname": "model.layers.0"}}
    write_json(tdir / "spec.json", spec)
    write_json(run.profile_dir / "profile.json", profile(graph=False))
    return run, tdir


def server(run: RunDir, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The tools of ``run`` with the static critic; returns call(tool, **args) -> result."""
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    critic.open_critic(run, critic.STATIC)
    tools = {t.name: t for t in tools_mod.build_server(run, Budget(run))}

    def call(tool: str, **args: Any) -> dict[str, Any]:
        if tool.startswith("evaluate_candidate"):
            args = {"target_id": "t", "hypothesis": "h", **args}
        out = asyncio.run(tools[tool].handler(args))
        return dict(json.loads(out["content"][0]["text"]))

    return call


def test_a_target_whose_stage_was_graphed_compares_in_a_cuda_graph(tmp_path, monkeypatch):
    """The VoxCPM2 slice-8 story (#226): an FP8 GEMM idea loses three times eagerly (launch
    bound) and is refuted; once a re-profile shows its stage CUDA-graphed, the idea is
    untested there, a variant wins at its graph speedup and every bar is the graph one."""
    run, tdir = kernel_run(tmp_path)
    graph_results = {1: result(1.4, "graph", other=0.9), 4: result(1.9, "graph", other=1.1)}
    seen: list[dict[str, Any]] = []

    def fake(capture: Path, snap: Path, **kw: Any) -> dict[str, Any]:
        seen.append(kw)
        i = int(snap.read_text().split("return ")[1])
        if kw["context"] == "graph":
            return graph_results[i]
        return EAGER_WIN if i == 0 else result(0.6 + i / 100)

    monkeypatch.setattr(tools_mod, "run_evaluation", fake)
    call = server(run, monkeypatch)
    out = []
    for i, idea in enumerate(["base", "fp8", "fp8", "fp8"]):
        (tdir / "candidates" / f"v{i}.py").write_text(f"def build(r):\n    return {i}\n")
        out.append(call("evaluate_candidate", candidate=f"candidates/v{i}.py", idea_id=idea))
    assert [kw["context"] for kw in seen] == ["eager"] * 4
    assert out[3]["idea"]["verdict"] == "refuted"

    # round 2's re-profile: the systems agent graphed the stage that holds the target
    write_json(run.root / "rounds" / "2" / "profile" / "profile.json", profile(graph=True))
    assert current(run, "t").key == GRAPH
    (tdir / "candidates" / "v4.py").write_text("def build(r):\n    return 4\n")
    new = call("evaluate_candidate", candidate="candidates/v4.py", idea_id="fp8")
    assert new["status"] == "ok", new  # not withdrawn as a variant of a refuted idea
    assert seen[-1]["context"] == "graph" and seen[-1]["early_best"] == 1.2  # not 2.0
    assert new["ledger"]["status"] == ledger.KEEP  # 1.9x > 1.2x (eager bar 2.0x: discard)
    assert new["idea"]["verdict"] == "kept" and new["budget"]["non_improving"] == 0
    assert new["best_so_far"]["speedup"] == 1.9

    ranked = list(tools_mod.ranked_for_target(run, "t"))
    # the graph timings first (the eager winner's: its view from a graph), then the eager
    # losers by their own speedups, each saying why it does not compare
    assert [r["speedup"] for r in ranked] == [1.9, 1.2, 0.63, 0.62, 0.61]
    assert ranked[1]["measured_in"]["speedup"] == 2.0 and "context_stale" not in ranked[1]
    assert all("the timing context changed" in r["context_stale"] for r in ranked[2:])
    eager_ranked = [r["speedup"] for r in tools_mod.ranked_for_target(run, "t", context=EAGER)]
    assert eager_ranked == [2.0, 1.1, 0.63, 0.62, 0.61]
    best = call("best_result", target_id="t")
    assert best["best"]["speedup"] == 1.9
    assert {s["idea"]: s["verdict"] for s in best["ideas"]} == {"base": "kept", "fp8": "kept"}

    # the duplicate cache: the eager winner's source comes back as its graph view; a loser's
    # (no graph timing) is evaluated again, in a CUDA graph
    n = len(seen)
    dup = call("evaluate_candidate", candidate="candidates/v0.py", hypothesis="again")
    assert "duplicate" in dup and len(seen) == n
    assert dup["speedup"] == 1.2 and dup["measured_in"]["speedup"] == 2.0
    assert dup["context"] == "graph" and "context_note" in dup
    again = call("evaluate_candidate", candidate="candidates/v1.py", hypothesis="again")
    assert "duplicate" not in again and len(seen) == n + 1 and again["speedup"] == 1.4


def test_the_scheduler_takes_an_arms_best_in_its_context(tmp_path):
    run, _ = kernel_run(tmp_path)
    for i, res in enumerate([EAGER_WIN, result(1.3, "graph"), result(1.25, "graph")]):
        name = f"{i:03d}_v{i}.py"
        row = ledger.record_kernel(run, "t", res, snapshot=name, hypothesis="h")
        append_jsonl(run.results_file("t"), {**res, "exp": row["exp"], "snapshot": name})

    def arm() -> scheduler.Arm:
        arms = scheduler.build_arms(run, scheduler.Policy(), [], targets=["t"])
        return next(a for a in arms if a.id == "t")

    assert (arm().best, arm().best_snapshot) == (2.0, "000_v0.py")  # eager: the eager winner
    write_json(run.root / "rounds" / "2" / "profile" / "profile.json", profile(graph=True))
    assert (arm().best, arm().best_snapshot) == (1.3, "001_v1.py")


# ------------------------------------------------------------------ the integration


def sim_orchestrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    return orchestrator.Orchestrator(dryrun.create_run(config), config)


def records_of(orch: Any, target: str, results: list[dict[str, Any]]) -> list[Path]:
    """``results`` recorded as evaluations of snapshots of ``target`` (as the tool does)."""
    snaps = []
    for i, res in enumerate(results):
        src = orch.run.target(target) / "candidates" / f"k{i}.py"
        src.write_text(f"def build(r):\n    return {i}\n")
        snap = orch.run.history_dir(target) / f"{i:03d}_k{i}.py"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(src.read_text())
        tools_mod.record_candidate(
            orch.run, target, src, snap, res, hypothesis=f"k{i}", keeper=orch.truth
        )
        snaps.append(snap)
    return snaps


def graphed(run: RunDir) -> None:
    """Round 2's re-profile: every decoder layer inside a CUDA-graph replay."""
    stage = {"stage": "model.layers.*", "events": 1000, "graph_events": 1000}
    write_json(
        run.root / "rounds" / "2" / "profile" / "profile.json",
        {"kernel_view": {"timeline": {"stages": [stage]}}},
    )


def test_the_integration_takes_a_targets_kernel_in_its_context(tmp_path, monkeypatch):
    orch = sim_orchestrator(tmp_path, monkeypatch)
    target = orch.run.target_ids()[0]
    eager_only = result(2.5)  # the fastest eagerly; never timed in a CUDA graph
    records_of(orch, target, [EAGER_WIN, eager_only, result(1.3, "graph")])
    assert orch._kernel_best(target)["speedup"] == 2.5
    graphed(orch.run)
    best = orch._kernel_best(target)
    assert best["speedup"] == 1.3 and best["context"] == "graph"  # not the stale 2.5x
    # with no record that has a speedup in a CUDA graph, the stale best is taken (and
    # re-evaluated before it is integrated: the next test)
    other = orch.run.target_ids()[1]
    records_of(orch, other, [result(1.5), result(1.8)])
    stale = orch._kernel_best(other)
    assert stale["speedup"] == 1.8 and "the timing context changed" in stale["context_stale"]


def test_a_stale_context_is_re_evaluated_before_the_recheck(tmp_path, monkeypatch):
    orch = sim_orchestrator(tmp_path, monkeypatch)
    target = orch.run.target_ids()[0]
    capture = orch.run.capture_file(target)
    capture.parent.mkdir(parents=True, exist_ok=True)
    capture.write_bytes(b"capture")
    orch.truth.seal(capture)
    (snap,) = records_of(orch, target, [result(1.8)])
    calls: dict[str, list[dict[str, Any]]] = {"reevaluate": [], "recheck": []}

    def reevaluator(capture_path: Path, path: Path, **kw: Any) -> dict[str, Any]:
        calls["reevaluate"].append(kw)
        return result(1.4, "graph")

    def rechecker(capture_path: Path, path: Path, **kw: Any) -> dict[str, Any]:
        calls["recheck"].append(kw["verdict"])
        return {"status": "ok", "passed": True, "speedup": kw["verdict"]["speedup"]}

    orch.reevaluator, orch.rechecker = reevaluator, rechecker
    rec = orch._kernel_record(target, snap.name)
    assert orch._recheck_one(target, snap, rec)["status"] == "ok"
    assert calls["reevaluate"] == []  # eager, as its record: nothing stale
    graphed(orch.run)
    checked = orch._recheck_one(target, snap, rec)
    (kw,) = calls["reevaluate"]
    assert kw["context"] == "graph" and kw["l2_flush"] is False
    assert checked["reevaluated"]["why"].startswith("the timing context changed")
    verdict = {"correct": True, "speedup": 1.4, "context": "graph", "l2": "warm"}
    assert calls["recheck"][-1] == verdict  # the re-evaluation's, timed in a CUDA graph
    # the re-evaluation replaced the record: the snapshot ranks by its graph speedup now
    best = orch._kernel_best(target)
    assert best["speedup"] == 1.4 and "context_stale" not in best


def test_a_recheck_is_reused_only_in_the_same_timing_context(tmp_path, monkeypatch):
    orch = sim_orchestrator(tmp_path, monkeypatch)
    target = orch.run.target_ids()[0]
    (snap,) = records_of(orch, target, [EAGER_WIN])
    rec = orch._kernel_record(target, snap.name)
    item = f"{target}={snap}"
    done = {
        "status": "ok",
        "passed": True,
        "item": item,
        "sha256": rec["snapshot_sha256"],
        "evaluator_schema": evaluate.EVALUATOR_SCHEMA,
        "memcheck": {"status": "skipped", "reason": "test"},
    }
    ran: list[str] = []

    def recheck_one(target_id: str, path: Path, record: Any, context: Any = None) -> dict:
        ran.append(str(context))
        return {"status": "ok", "passed": True, "memcheck": done["memcheck"]}

    monkeypatch.setattr(orch, "_recheck_one", recheck_one)
    previous = {"recheck": [done]}  # an earlier integration's, from before the context key
    _, _, records = orch._recheck_kernels([("kernel", item)], None, previous)
    assert ran == [] and records[0]["timing_context"] == list(EAGER)  # reused: eager, eager
    graphed(orch.run)
    _, _, records = orch._recheck_kernels([("kernel", item)], None, {"recheck": records})
    assert ran == [str(GRAPH)] and records[0]["timing_context"] == list(GRAPH)
