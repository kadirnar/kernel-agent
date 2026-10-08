"""Parallel workers per target, duplicate candidates and the quick evaluation tier.

CPU only: fake agents (``run_agent``) that evaluate through their own bound tools,
a fake evaluator, the dry-run world for ``improve``, and the real evaluator on a
CPU toy for the quick tier.
"""

import asyncio
import copy
import json
import re
import threading
import time

import pytest
import torch

from kernel_agent import (
    charts,
    cli,
    dedup,
    dryrun,
    ledger,
    orchestrator,
    scheduler,
    status,
    workers,
)
from kernel_agent.agent import prompts
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.runner import AgentResult
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.kernels.evaluate import evaluate, quick_cases
from kernel_agent.profiling.capture import capture_calls
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

CARD = {"repo_id": "org/m", "modality": "llm", "architectures": ["M"], "params": 1}


# ------------------------------------------------------------------ worker counts, seeds


def test_counts_budget_split_and_rounds():
    assert workers.count(None, 0.9) == 1  # not set: one session, as before
    assert workers.count(3, 0.0) == 3
    assert workers.count("auto", 0.25) == workers.AUTO_WORKERS
    assert workers.count("auto", 0.19) == 1
    assert workers.parse("AUTO") == "auto" and workers.parse("2") == 2
    with pytest.raises(ValueError):
        workers.parse("0")
    # the target's total stays fixed: split, not multiplied (depth beats breadth)
    assert workers.split(12, 2) == [6, 6] and workers.split(10, 3) == [4, 3, 3]
    assert sum(workers.split(7, 4)) == 7 and workers.split(1, 2) == [1, 1]  # at least one each
    assert workers.rounds(12, False) == (12, 0) and workers.rounds(12, True) == (6, 6)
    assert workers.rounds(7, True) == (4, 3)


def test_seeds_use_alternatives_then_other_backends():
    spec = {
        "approach": "fuse the block",
        "backends": ["triton", "cuda"],
        "alternatives": [{"approach": "split-K GEMV", "backends": ["cute", "nope"]}],
    }
    team = workers.seeds(spec, 3, 12, ["cuda", "triton", "cute", "tilelang"])
    assert [s.worker for s in team] == [1, 2, 3]
    assert [s.evaluations for s in team] == [4, 4, 4]
    assert team[0].approach == "fuse the block" and team[0].backends == ("triton", "cuda")
    assert team[1].origin == "alternative" and team[1].approach == "split-K GEMV"
    assert team[1].backends == ("cute",)  # unavailable backends dropped
    # no third alternative: the planned approach from another primary backend
    assert team[2].origin == "backend" and team[2].backends[0] == "cute"
    assert "instead of `triton`" in team[2].approach

    note = workers.prompt_note("t", team[1], team)
    assert "# Worker 2 of 3 on `t`" in note and "split-K GEMV" in note
    assert "worker 1: fuse the block" in note and "worker 3:" in note
    assert "budget in this session: 4 evaluations" in note
    assert workers.parse_agent(workers.agent_name("t", 2)) == ("t", 2)
    assert workers.parse_agent("kernel-t") == ("t", None)


def test_prepare_links_the_shared_files(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.target("t") / "spec.json", {"id": "t"})
    (run.target("t") / "history").mkdir()
    (run.target("t") / "history" / "001_a.py").write_text("x = 1\n")
    home = workers.prepare(run, "t", 2)
    assert home == run.target("t") / "workers" / "2"
    assert (home / "candidates").is_dir() and (home / "NOTES.md").is_file()
    assert read_json(home / "spec.json") == {"id": "t"}
    assert (home / "history" / "001_a.py").read_text() == "x = 1\n"
    assert not (home / "plan.md").exists()  # a link to a file that comes later
    (run.target("t") / "plan.md").write_text("plan")
    assert (home / "plan.md").read_text() == "plan"
    assert workers.prepare(run, "t", 2) == home  # idempotent
    assert workers.notes_file(run, "t", "2") == home / "NOTES.md"
    assert workers.notes_file(run, "t", None) == run.target("t") / "NOTES.md"


# ------------------------------------------------------------------ dedup keys


def test_source_key_ignores_comments_and_formatting():
    base = "import torch\n\ndef build(r):\n    return r  # identity\n"
    same = "# a header comment\nimport torch\ndef build( r ):\n\n    return r\n"
    quoted = base.replace("import torch", "import torch\nX = 'a'")
    assert dedup.source_key(base) == dedup.source_key(same)
    assert dedup.source_key(quoted) == dedup.source_key(quoted.replace("'a'", '"a"'))
    assert dedup.source_key(base) != dedup.source_key(base.replace("return r", "return None"))
    # docstrings are code (a different string constant), not comments
    assert dedup.source_key(base) != dedup.source_key('"""v2"""\n' + base)
    # not Python: compared line by line without trailing whitespace
    assert dedup.source_key("def (:\n  x  \n") == dedup.source_key("def (:\n  x\n")


# ------------------------------------------------------------------ the tools


def _server(run, monkeypatch, outcomes=None, *, binding=None, budget=None, delay=0.0):
    """Tools of ``run`` with a fake evaluator; returns (call, calls of the evaluator)."""
    seen = []
    lock = threading.Lock()

    def fake_eval(capture, snap, **kwargs):
        with lock:
            seen.append({"snap": snap, **kwargs})
        time.sleep(delay)
        if outcomes is not None:
            return outcomes(snap, kwargs)
        return {"status": "ok", "correct": True, "speedup": 1.2, "cases": []}

    monkeypatch.setattr(tools_mod, "run_evaluation", fake_eval)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    server = {t.name: t for t in tools_mod.build_server(run, budget or Budget(run), None, binding)}

    async def acall(tool, **args):
        out = await server[tool].handler(args)
        return json.loads(out["content"][0]["text"])

    def call(tool, **args):
        return asyncio.run(acall(tool, **args))

    call.acall = acall
    return call, seen


def _target(run, target_id="t", spec=None):
    tdir = run.target(target_id)
    (tdir / "candidates").mkdir(parents=True, exist_ok=True)
    run.capture_file(target_id).write_bytes(b"")  # a run without .truth/: no digest to check
    write_json(tdir / "spec.json", spec or {"id": target_id, "module_class": "M"})
    return tdir


def test_duplicates_return_the_cached_result_and_cost_nothing(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = _target(run)
    call, seen = _server(run, monkeypatch, binding=tools_mod.SessionBinding(evaluations=3))
    src = tdir / "candidates" / "v1.py"
    src.write_text("import triton\n\ndef build(r):\n    return r\n")

    def evaluate_candidate(**args):
        return call("evaluate_candidate", target_id="t", candidate="candidates/v1.py", **args)

    first = evaluate_candidate(hypothesis="fused", idea_id="fuse")
    assert first["ledger"]["status"] == "keep" and first["budget"]["evals_used"] == 1
    snapshot = first["best_so_far"]["snapshot"]

    # only whitespace and a comment changed: the same kernel
    src.write_text("import triton\n# tuned?\n\ndef build( r ):\n    return r   \n")
    again = evaluate_candidate(hypothesis="same, reformatted", idea_id="fuse")
    assert len(seen) == 1  # not evaluated again: no GPU lock, no subprocess
    assert again["ledger"]["status"] == ledger.DUPLICATE
    assert (
        again["duplicate_of"] == snapshot and "duplicate of history/001_v1_" in again["duplicate"]
    )
    assert again["speedup"] == 1.2 and again["status"] == "ok"  # the cached result
    assert again["budget"] == {"evals_used": 1, "evals_budget": 3, "counted": False}

    rows = ledger.rows(run)
    assert [r["status"] for r in rows] == ["keep", "duplicate"]
    assert rows[1]["snapshot"] == rows[0]["snapshot"] and rows[1]["speedup"] is None
    assert len(read_jsonl(run.results_file("t"))) == 1  # no record, no snapshot
    assert len(list(run.history_dir("t").glob("*.py"))) == 1
    assert ledger.measured(rows) == rows[:1]
    (idea,) = ledger.ideas(rows)
    assert idea["tries"] == 1  # a duplicate is not a try
    assert call("best_result", target_id="t")["evaluations"] == 1

    # a real change is evaluated; the streak and the budget count it
    src.write_text("import triton\n\ndef build(r):\n    return r.eval()\n")
    changed = evaluate_candidate(hypothesis="really new")
    assert len(seen) == 2 and changed["ledger"]["status"] == "discard"
    assert changed["budget"]["evals_used"] == 2 and changed["budget"]["non_improving"] == 1
    # profile tables are never stored: a profile=true call runs it again
    profiled = evaluate_candidate(hypothesis="profile it", profile=True)
    assert len(seen) == 3 and seen[-1]["profile"] and "duplicate" not in profiled


def test_repeatable_results_only(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = _target(run)
    statuses = iter(["timeout", "build_error"])

    def outcome(snap, kwargs):
        return {"status": next(statuses), "correct": False, "error": "x"}

    call, seen = _server(run, monkeypatch, outcome)
    (tdir / "candidates" / "v1.py").write_text("def build(r): ...\n")

    def evaluate_candidate():
        args = {"target_id": "t", "candidate": "candidates/v1.py", "hypothesis": "h"}
        return call("evaluate_candidate", **args)

    assert evaluate_candidate()["status"] == "timeout"
    assert evaluate_candidate()["status"] == "build_error"  # a timeout may not repeat: rerun
    dup = evaluate_candidate()  # a build error does
    assert len(seen) == 2 and dup["ledger"]["status"] == "duplicate"
    assert dup["status"] == "build_error"


def test_identical_candidates_in_flight_are_evaluated_once(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = _target(run)
    call, seen = _server(run, monkeypatch, delay=0.3)
    for name in ("a", "b"):  # two workers write the same kernel at the same time
        (tdir / "candidates" / f"{name}.py").write_text("def build(r):\n    return r\n")

    async def both():
        return await asyncio.gather(
            *(
                call.acall("evaluate_candidate", target_id="t", candidate=c, hypothesis="h")
                for c in ("candidates/a.py", "candidates/b.py")
            )
        )

    first, second = asyncio.run(both())
    assert len(seen) == 1
    assert "duplicate" not in first and second["ledger"]["status"] == "duplicate"


def test_quick_tier(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = _target(run)

    def outcome(snap, kwargs):
        if kwargs.get("quick"):
            ok = "bug" not in snap.read_text()
            result = {"status": "ok" if ok else "incorrect", "correct": ok, "cases": []}
            return result | {"quick": {"cases": [0, 2], "of": 3}, "timing": "skipped: quick"}
        return {"status": "ok", "correct": True, "speedup": 1.5, "cases": []}

    binding = tools_mod.SessionBinding(evaluations=2)
    call, seen = _server(run, monkeypatch, outcome, binding=binding)
    src = tdir / "candidates" / "v1.py"

    def evaluate_candidate(code, **args):
        src.write_text(code)
        return call(
            "evaluate_candidate",
            target_id="t",
            candidate="candidates/v1.py",
            hypothesis="h",
            **args,
        )

    bad = evaluate_candidate("def build(r):\n    return 'bug'\n", mode="quick")
    assert seen[-1]["quick"] and not seen[-1]["profile"]
    assert bad["mode"] == "quick" and "NOT a benchmark" in bad["not_a_benchmark"]
    assert bad["ledger"]["status"] == ledger.QUICK_FAIL and "speedup" not in bad
    assert bad["budget"]["counted"] is False and bad["budget"]["evals_used"] == 0
    assert "advice" not in bad

    good = evaluate_candidate("def build(r):\n    return r\n", mode="quick")
    assert good["ledger"]["status"] == ledger.QUICK_OK and good["best_so_far"] is None
    # quick checks are their own records: never a winner, never a full evaluation
    assert read_jsonl(run.results_file("t")) == []
    quick = read_jsonl(dedup.quick_file(run, "t"))
    assert [r["mode"] for r in quick] == ["quick", "quick"]
    assert tools_mod.best_for_target(run, "t") is None

    # a full evaluation of a quick-checked source is not a duplicate: it gets timed
    full = evaluate_candidate("def build(r):\n    return r\n")
    assert "quick" not in seen[-1] and full["ledger"]["status"] == "keep"
    assert full["budget"]["evals_used"] == 1  # the two quick checks were free
    # ... and a quick check of a fully evaluated source returns the full result
    again = evaluate_candidate("def build(r):\n    return r  # same\n", mode="quick")
    assert again["ledger"]["status"] == "duplicate" and again["speedup"] == 1.5
    assert len(seen) == 3

    rows = ledger.rows(run)
    assert [r["status"] for r in rows] == ["quick_fail", "quick_ok", "keep", "duplicate"]
    s = ledger.summary(run)
    assert s["evaluations"] == 1 and s["targets"][0]["evals"] == 1
    assert s["targets"][0]["failures"] == 0  # a quick_fail is not a failed evaluation
    arms = scheduler.build_arms(run, scheduler.Policy(systems=False), [], targets=["t"])
    assert arms[0].evals == 1 and arms[0].streak == 0 and arms[0].fails == 0
    assert call("evaluate_candidate", target_id="t", candidate="candidates/v1.py",
                hypothesis="h", mode="fast")["status"] == "error"  # fmt: skip


def test_quick_mode_of_the_evaluator_checks_the_smallest_and_largest_case(tmp_path):
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 4)
    calls = [((torch.randn(2, 4),), {}, 3), ((torch.randn(8, 4),), {}, 1)]
    calls += [((torch.randn(1, 4),), {}, 5), ((torch.randn(3, 4),), {}, 0)]
    capture = tmp_path / "c.pt"
    capture_calls(module, calls, capture)
    candidate = tmp_path / "same.py"
    candidate.write_text(
        "import copy\n\n\ndef build(reference):\n    return copy.deepcopy(reference)\n"
    )

    quick = evaluate(capture, candidate, device="cpu", quick=True)
    assert quick["correct"] and quick["status"] == "ok"
    assert quick["quick"] == {"cases": [1, 2], "of": 4}  # 8x4 is the largest, 1x4 the smallest
    assert [c["signature"] for c in quick["cases"]] == ["a0[8, 4]:float32", "a0[1, 4]:float32"]
    assert quick["timing"] == "skipped: quick check" and "speedup" not in quick
    full = evaluate(capture, candidate, device="cpu")
    assert len(full["cases"]) == 4 and "quick" not in full

    cases = [{"args": (torch.zeros(5),), "kwargs": {"m": [torch.zeros(2, 2)]}}]
    assert quick_cases(cases) == [0]
    cases.append({"args": (torch.zeros(1),), "kwargs": {}})
    assert quick_cases(cases) == [0, 1]
    assert quick_cases([]) == []


def test_worker_tools_resolve_paths_label_rows_and_rank_across_workers(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    _target(run)
    speed = {"w1": 1.2, "w2": 1.6}

    def outcome(snap, kwargs):
        tag = re.search(r"TAG = '(\w+)'", snap.read_text())[1]
        return {"status": "ok", "correct": True, "speedup": speed[tag], "cases": []}

    calls = {}
    for k in (1, 2):
        home = workers.prepare(run, "t", k)
        (home / "candidates" / "v1.py").write_text(f"TAG = 'w{k}'\n\ndef build(r):\n    return r\n")
        binding = tools_mod.SessionBinding(
            target_id="t", worker=k, agent=workers.agent_name("t", k), evaluations=2
        )
        calls[k], _ = _server(run, monkeypatch, outcome, binding=binding)
    one = calls[1](
        "evaluate_candidate", target_id="t", candidate="candidates/v1.py", hypothesis="a"
    )
    two = calls[2](
        "evaluate_candidate", target_id="t", candidate="candidates/v1.py", hypothesis="b"
    )
    assert one["budget"]["evals_budget"] == 2 and two["budget"]["evals_used"] == 1
    assert two["ledger"]["status"] == "keep" and two["best_so_far"]["speedup"] == 1.6

    rows = ledger.rows(run)
    assert [r["worker"] for r in rows] == ["1", "2"]
    records = read_jsonl(run.results_file("t"))
    assert [r["worker"] for r in records] == [1, 2]
    assert records[0]["candidate"] == "workers/1/candidates/v1.py"
    best = tools_mod.best_for_target(run, "t")
    assert best["worker"] == 2 and best["speedup"] == 1.6
    assert [r["speedup"] for r in tools_mod.ranked_for_target(run, "t")] == [1.6, 1.2]
    shared = call_best = calls[1]("best_result", target_id="t")
    assert shared["best"]["speedup"] == 1.6 and call_best["history"][1]["worker"] == 2

    # worker 1 resubmits worker 2's kernel: a duplicate of its snapshot
    (workers.directory(run, "t", 1) / "candidates" / "v2.py").write_text(
        "TAG = 'w2'\ndef build(r):\n    return r\n"
    )
    dup = calls[1](
        "evaluate_candidate", target_id="t", candidate="candidates/v2.py", hypothesis="c"
    )
    assert dup["duplicate_of"] == records[1]["snapshot"] and ledger.rows(run)[-1]["worker"] == "1"

    # round 2 of --reseed-workers: the two best snapshots, each with the worker that made it
    team = workers.seeds({"approach": "a", "backends": ["triton"]}, 2, 4, ["triton", "cuda"])
    again = workers.reseeds(run, "t", team, 4)
    assert [(s.worker, s.parent, s.parent_speedup) for s in again] == [
        (2, records[1]["snapshot"], 1.6),
        (1, records[0]["snapshot"], 1.2),
    ]
    assert [s.evaluations for s in again] == [2, 2] and again[0].backends == team[1].backends
    assert "Round 2: start from" in workers.prompt_note("t", again[0], again)
    assert workers.reseeds(RunDir.create(tmp_path / "x", "org/m"), "t", team, 4) == []

    # status and the charts show the workers
    text = status.render(run, width=200)
    assert "t/w2" in text and "(2 workers)" in text
    if charts.available():
        assert charts.target_progress(run, "t").is_file()


# ------------------------------------------------------------------ orchestrator.kernels


class FakeToolchain:
    env: dict[str, str] = {}
    backends = {"triton": True, "cuda": True}
    gpu = None

    def summary(self) -> str:
        return "GPU: none (test)"


def make_orchestrator(tmp_path, monkeypatch, **cfg):
    """A run (analyze/plan/capture done) with a hot target (60 % of the profile) and a cold
    one; fake agents that spend their evaluation budget through their own tools."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", FakeToolchain)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    run = RunDir.create(tmp_path, "org/m")
    config = OptimizeConfig(
        model_ref="org/m",
        runs_dir=tmp_path,
        backends=["triton", "cuda"],
        use_library=False,
        librarian=False,
        dossier=False,
        **cfg,
    )
    phases = {p: {"done": True} for p in ("analyze", "plan", "capture")}
    write_json(
        run.run_json, {"card": CARD, "workload": {}, "config": config.to_dict(), "phases": phases}
    )
    write_json(run.baseline_json, {"median_ms": 10.0})
    classes = [
        {"cls": "Model", "root": "Model", "inclusive_ms": 100.0, "instances": 1},
        {"cls": "Hot", "root": "Model", "inclusive_ms": 60.0, "instances": 1},
        {"cls": "Cold", "root": "Model", "inclusive_ms": 10.0, "instances": 1},
    ]
    write_json(run.profile_dir / "profile.json", {"classes": classes})
    alternative = {"approach": "split-K alternative", "backends": ["cuda"]}
    for tid, cls in (("hot", "Hot"), ("cold", "Cold")):
        spec = {"id": tid, "module_class": cls, "backends": ["triton"], "why": "w"}
        spec |= {"approach": f"{tid} plan", "alternatives": [alternative]}
        _target(run, tid, spec)
    sessions = []
    active = [0, 0]  # now, most at once
    rng = iter(range(10_000))

    async def fake_run_agent(name, *, cfg, system_append, cwd, mcp_server, result=None, **_):
        target_id, worker = workers.parse_agent(name)
        found = re.search(r"budget in this session: (\d+) evaluations", system_append)
        n = int(found[1]) if found else config.evaluations_per_target
        sessions.append({"name": name, "cwd": cwd, "system": system_append, "evals": n})
        active[0] += 1
        active[1] = max(active)
        tools = {t.name: t for t in mcp_server}
        for i in range(n):
            (cwd / "candidates" / f"v{i}.py").write_text(
                f"V = {next(rng)}\nW = {worker or 0}\n\ndef build(r):\n    return r\n"
            )
            args = {"target_id": target_id, "candidate": f"candidates/v{i}.py", "hypothesis": "h"}
            await tools["evaluate_candidate"].handler(args)
            await asyncio.sleep(0.01)
        active[0] -= 1
        return result or AgentResult(name=name)

    def fake_eval(capture, snap, **kwargs):
        source = snap.read_text()
        worker, v = int(re.search(r"W = (\d+)", source)[1]), int(re.search(r"V = (\d+)", source)[1])
        speedup = 1.0 + 0.1 * worker + 0.001 * v
        return {"status": "ok", "correct": True, "speedup": speedup, "cases": []}

    monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
    monkeypatch.setattr(tools_mod, "run_evaluation", fake_eval)
    return orchestrator.Orchestrator(run, config), sessions, active


def test_kernels_phase_runs_workers_with_a_split_budget(tmp_path, monkeypatch):
    orch, sessions, active = make_orchestrator(
        tmp_path, monkeypatch, seeds_per_target="auto", parallel=2, evaluations_per_target=6
    )
    run = orch.run
    asyncio.run(orch.kernels())

    names = sorted(s["name"] for s in sessions)
    assert names == ["kernel-cold", "kernel-hot-w1", "kernel-hot-w2"]  # auto: only hot (60 %)
    by = {s["name"]: s for s in sessions}
    assert [by[f"kernel-hot-w{k}"]["evals"] for k in (1, 2)] == [3, 3]  # 6 split, not 2 x 6
    assert by["kernel-cold"]["evals"] == 6
    assert by["kernel-hot-w1"]["cwd"] == run.target("hot") / "workers" / "1"
    assert "suggested approach: hot plan" in by["kernel-hot-w1"]["system"]
    assert "suggested approach: split-K alternative" in by["kernel-hot-w2"]["system"]
    assert (
        "`cuda`" in by["kernel-hot-w2"]["system"]
        and "# Worker 2 of 2" in by["kernel-hot-w2"]["system"]
    )
    assert by["kernel-cold"]["cwd"] == run.target("cold")
    assert 1 < active[1] <= 2  # concurrent, within --parallel

    hot = [r for r in ledger.rows(run) if r["target"] == "hot"]
    assert len(hot) == 6 and sorted({r["worker"] for r in hot}) == ["1", "2"]
    assert all(not r["worker"] for r in ledger.rows(run) if r["target"] == "cold")
    best = tools_mod.best_for_target(run, "hot")
    assert best["worker"] == 2  # the best across the workers
    kernels = run.load()["phases"]["kernels"]
    assert sorted(kernels["workers"]) == ["hot/w1", "hot/w2"]
    assert sorted(kernels["finished"]) == ["cold", "hot"]
    costs = read_json(run.root / "costs.json")
    assert {"kernel-hot-w1", "kernel-hot-w2", "kernel-cold"} <= set(costs)

    # a resumed phase does not repeat finished worker sessions
    data = run.load()
    data["phases"]["kernels"]["finished"] = []
    write_json(run.run_json, data)
    sessions.clear()
    asyncio.run(orch.kernels())
    assert [s["name"] for s in sessions] == ["kernel-cold"]


def test_kernels_phase_reseeds_round_two_from_the_best_snapshots(tmp_path, monkeypatch):
    orch, sessions, active = make_orchestrator(
        tmp_path,
        monkeypatch,
        seeds_per_target=2,
        reseed_workers=True,
        parallel=1,
        evaluations_per_target=8,
    )
    run = orch.run
    asyncio.run(orch.kernels())
    hot = [s for s in sessions if s["name"].startswith("kernel-hot")]
    assert [s["evals"] for s in hot] == [2, 2, 2, 2]  # 8 = 4 (round 1) + 4 (round 2)
    assert active[1] == 1  # --parallel 1: one session at a time
    round2 = [s for s in hot if "Round 2: start from" in s["system"]]
    assert [s["name"] for s in round2] == ["kernel-hot-w2", "kernel-hot-w1"]
    # they start from the two fastest snapshots of round 1 (both by worker 2 here)
    hot_rows = [r for r in ledger.rows(run) if r["target"] == "hot"]
    first = sorted(hot_rows[:4], key=lambda r: -r["speedup"])
    starts = [re.search(r"start from `history/([^`]+)`", s["system"])[1] for s in round2]
    assert starts == [r["snapshot"] for r in first[:2]] and first[0]["worker"] == "2"
    assert len(hot_rows) == 8
    workers_done = run.load()["phases"]["kernels"]["workers"]
    assert sorted(w for w in workers_done if w.startswith("hot/")) == [
        "hot/r2w1",
        "hot/r2w2",
        "hot/w1",
        "hot/w2",
    ]


# ------------------------------------------------------------------ improve


@pytest.fixture
def sim(monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def test_improve_slices_run_the_workers_of_a_target(tmp_path, sim):
    config = OptimizeConfig(
        model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, seeds_per_target=2, parallel=2
    )
    orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
    world = dryrun.World(orch)
    improver = Improver(
        orch, ImproveConfig(slice=4, max_slices=4), require_capture=False, live_charts=False
    )
    with world.installed():
        asyncio.run(improver.improve())
    run = orch.run
    kernel_slices = [s for s in improver.state["slices"] if s["arm"] != "systems"]
    assert kernel_slices and all(s["workers"] == [1, 2] for s in kernel_slices)
    for s in kernel_slices:
        assert s["evals"] <= 4  # the slice's 4 evaluations, split over two workers
        assert s["status"] in ("done", "timed_out")
    names = [x["name"] for x in world.sessions if x["name"].startswith("kernel-")]
    arm = kernel_slices[0]["arm"]
    assert {f"kernel-{arm}-w1", f"kernel-{arm}-w2"} <= set(names)
    rows = [r for r in ledger.rows(run) if r["target"] == arm]
    assert {r["worker"] for r in rows} == {"1", "2"}
    assert (workers.directory(run, arm, 2) / "NOTES.md").read_text().strip()
    costs = read_json(run.root / "costs.json")
    assert f"kernel-{arm}-w1#{kernel_slices[0]['n']}" in costs
    # the second worker's digest is its own notes, the target's shared ledger
    second = [x for x in world.sessions if x["name"] == f"kernel-{arm}-w2"][-1]["system"]
    assert "# Worker 2 of 2" in second and "| worker |" in second


def test_unmeasured_rows_do_not_feed_the_scheduler(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.target("t") / "spec.json", {"id": "t", "module_class": "M"})
    ok = {"status": "ok", "correct": True, "speedup": 1.5, "cases": []}
    ledger.record_kernel(run, "t", ok, snapshot="001_a.py", hypothesis="a", worker=1)
    for kind in (ledger.DUPLICATE, ledger.QUICK_OK, ledger.QUICK_FAIL):
        ledger.record_kernel(run, "t", {}, snapshot="001_a.py", hypothesis="x", status=kind)
    (arm,) = scheduler.build_arms(run, scheduler.Policy(systems=False), [], targets=["t"])
    assert arm.evals == 1 and arm.streak == 0 and arm.fails == 0 and arm.best == 1.5
    events = [e for e in ledger.events(run) if e["event"] == "evaluation"]
    assert events[0]["worker"] == 1 and "worker" not in events[1]


def test_planner_schema_and_engineer_prompt():
    item = prompts.PLAN_SCHEMA["properties"]["targets"]["items"]
    assert "alternatives" in item["properties"] and "alternatives" not in item["required"]
    plan = prompts.planner_prompt(CARD, {"median_ms": 1.0}, "# P", ["triton"], 2, "py", "tc")
    assert "`alternatives`" in plan
    target = {"id": "t", "module_class": "M", "why": "w", "approach": "a"}
    text = prompts.engineer_prompt(target, {"cases": []}, ["triton"], "py", "tc", 4, None)
    assert 'mode="quick"' in text and "duplicate" in text
    copy.deepcopy(prompts.PLAN_SCHEMA)


def test_cli_flags_and_a_resumed_improve(monkeypatch, tmp_path, sim, capsys):
    seen = []
    monkeypatch.setattr(cli, "cmd_optimize", lambda ns: seen.append(cli._config(ns)) or 0)
    assert cli.main(["optimize", "org/m", "--seeds-per-target", "auto", "--reseed-workers"]) == 0
    assert seen[0].seeds_per_target == "auto" and seen[0].reseed_workers
    assert OptimizeConfig.from_dict(seen[0].to_dict()).seeds_per_target == "auto"
    assert cli.main(["optimize", "org/m"]) == 0
    assert seen[1].seeds_per_target is None and not seen[1].reseed_workers
    with pytest.raises(SystemExit):
        cli.main(["optimize", "org/m", "--seeds-per-target", "0"])
    assert "--seeds-per-target" in capsys.readouterr().err

    # `improve <run_dir> --seeds-per-target 2 --parallel 2` applies to a run made without
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    run = dryrun.create_run(config)
    argv = ["improve", str(run.root), "--dry-run", "--max-slices", "2"]
    assert cli.main([*argv, "--seeds-per-target", "2", "--parallel", "2"]) == 0
    slices = read_json(run.root / "improve.json")["slices"]
    kernel = [s for s in slices if s["arm"] != "systems"]
    assert kernel and kernel[0]["workers"] == [1, 2]
