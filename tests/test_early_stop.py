"""Early termination and batching in measurement (issue #190, docs/MULTIAGENT.md
§3.12.5-3.12.6): the evaluator's early discard, sweep racing, the sequential A/B, refuted
ideas, lease batching, ``evaluate_candidates`` and ``evaluate_e2e_batch``. CPU, but for the
two ``gpu`` checks at the end (the real evaluator and a batch of real evaluations)."""

from __future__ import annotations

import asyncio
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ab_fake
import pytest
import torch
import toy_decoder
from fake_clock import Clock

from kernel_agent import abtest, budget, critic, gpuqueue, ledger, toolchain, worker
from kernel_agent.agent import tools as tools_mod
from kernel_agent.budget import Budget
from kernel_agent.integrate import reuse
from kernel_agent.kernels import early, sweep
from kernel_agent.kernels import evaluate as evaluate_mod
from kernel_agent.profiling.capture import capture_calls
from kernel_agent.selftest import RMSNorm
from kernel_agent.workloads import base
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workspace import RunDir, read_jsonl, write_json

TESTS = Path(__file__).parent


# ------------------------------------------------------------------ the sequential A/B


def test_sequential_ab_matches_the_fixed_round_verdict_on_recorded_data():
    """Every paired A/B of our runs' integrations (tests/fixtures/ab_rounds.json): the
    rounds a sequential A/B runs reach the verdict of all 8, with a third fewer rounds."""
    records = ab_fake.recorded()
    assert len(records) >= 200
    ran = total = stops = 0
    for rec in records:
        a, b = rec["a_ms"], rec["b_ms"]
        fixed = abtest.judge({"mode": "paired", "a_ms": a, "b_ms": b})["accepted"]
        a_run, b_run, stopped = ab_fake.sequential_rounds(a, b, len(a))
        judged = abtest.judge({"mode": "paired", "a_ms": a_run, "b_ms": b_run})
        assert judged["accepted"] == fixed, (rec["run"], a, b, stopped)
        if stopped is not None:
            stops += 1
            assert (stopped["verdict"] == "accept") == fixed and stopped["of"] == len(a)
        ran += len(a_run)
        total += len(a)
    assert stops > 0.8 * len(records)
    assert ran < 0.7 * total, f"{ran} of {total} rounds"


def test_sequential_stop_rules():
    a = [100.0] * 8
    assert abtest.sequential(a[:2], [101.0, 102.0], 8) is None  # before SEQ_MIN_ROUNDS
    lost = abtest.sequential(a[:3], [101.0, 102.0, 99.0], 8)  # can no longer win 7 of 8
    assert lost is not None and lost["verdict"] == "reject" and "no longer win 7" in lost["why"]
    tiny = abtest.sequential(a[:3], [99.8, 99.79, 99.81], 8)  # wins, but 0.2 %: below 1 %
    assert tiny is not None and tiny["verdict"] == "reject" and "99 % CI" in tiny["why"]
    fast = [95.0, 95.2, 94.9, 95.1, 95.0, 94.8, 95.3]
    # a clear 5 % gain: accepted once the win rate holds whatever the last round brings
    assert all(abtest.sequential(a[:n], fast[:n], 8) is None for n in range(3, 7))
    win = abtest.sequential(a[:7], fast, 8)
    assert win is not None and win["verdict"] == "accept" and (win["rounds"], win["of"]) == (7, 8)
    assert abtest.sequential(a, [95.0] * 8, 8) is None  # all rounds ran: nothing to stop
    assert abtest.sequential(a[:3], [101.0] * 3, 3) is None
    judged = abtest.judge({"mode": "paired", "a_ms": a[:7], "b_ms": fast, "stopped": win})
    assert judged["accepted"] and "stopped after 7 of 8" in abtest.describe(judged)


def test_sequential_measurements_are_reused_only_under_their_rule(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    run.transforms_dir.mkdir(parents=True, exist_ok=True)
    item = run.transforms_dir / "t.py"
    item.write_text("def apply(workload):\n    pass\n")
    context = {"evaluator_schema": 2, "ab_rounds": 8}
    plain = reuse.Keys(run, context)
    ruled = reuse.Keys(run, context, abtest.stop_rule())
    other = reuse.Keys(run, context, abtest.stop_rule(min_gain=0.02))
    step = ([], [("transform", str(item))])
    # all rounds ran: the same key whatever the rule (it serves either way)
    assert plain.step(*step) == ruled.step(*step) is not None
    # stopped early: a key of its rule, none without a rule
    assert plain.step(*step, ruled=True) is None
    assert ruled.step(*step, ruled=True) not in (
        None,
        ruled.step(*step),
        other.step(*step, ruled=True),
    )


def test_worker_e2e_ab_stops_once_the_verdict_is_decided(tmp_path, capsys, cpu):
    root = _toy_run(tmp_path)
    _worker(capsys, "analyze", "--run-dir", root, "--no-profile", "--iters", 1)
    norm = _file(tmp_path, "norm.py", NORM)
    args = ["--run-dir", root, "--rounds", 8, "--warmup", 1, "--b-kernel", f"norm={norm}"]
    r = _worker(capsys, "e2e_ab", *args, "--sequential", "--ab-min-gain", "0.5")
    assert r["status"] == "ok" and r["passed"], r
    rec = r["ab"]  # a 50 % gain is out of reach: rejected after 3 rounds, B still judged
    assert rec["rounds"] == 3 and rec["stopped"]["verdict"] == "reject", rec
    assert len(rec["a_ms"]) == len(rec["b_ms"]) == 3 and r["metrics"]["holdout"]["passed"]
    full = _worker(capsys, "e2e_ab", *args)
    assert full["ab"]["rounds"] == 8 and "stopped" not in full["ab"]


# ------------------------------------------------------------------ early discard


def test_min_rounds_bound_the_median_of_all_rounds():
    """Once :func:`early.min_rounds` rounds ran, the median of all (bench.median_round's upper
    median) lies between their smallest and largest value, whatever the rest measure."""
    rng = random.Random(0)
    assert [early.min_rounds(r) for r in (2, 3, 4, 5, 6)] == [2, 2, 3, 3, 4]
    for _ in range(2000):
        rounds = rng.randint(2, 6)
        values = [rng.uniform(0.5, 2.0) for _ in range(rounds)]
        k = early.min_rounds(rounds)
        median = sorted(values)[rounds // 2]
        assert min(values[:k]) <= median <= max(values[:k])


def _rounds(rng: random.Random, ms: float, n: int) -> list[float]:
    """``n`` round medians of a function of ``ms``: 1 % noise, now and then a slow round."""
    out = []
    for _ in range(n):
        value = ms * math.exp(rng.gauss(0.0, 0.01))
        if rng.random() < 0.05:  # a disturbed round (up to the worst spread we recorded)
            value *= rng.uniform(1.05, 1.5)
        out.append(value)
    return out


def test_early_discard_never_flips_a_winner():
    """Synthetic timings around the bar (the noise edge): whenever the evaluator would stop
    after 2 of 3 rounds, all 3 rounds would not have made a keep (budget.improves)."""
    rng = random.Random(190)
    stopped = winners = 0
    for _ in range(5000):
        bar = rng.uniform(1.0, 2.5)
        speedup = bar * rng.uniform(0.85, 1.15)
        cases = []
        for _ in range(rng.randint(1, 4)):
            ref_ms = rng.uniform(0.01, 2.0)
            cases.append(
                (rng.randint(1, 40), _rounds(rng, ref_ms, 3), _rounds(rng, ref_ms / speedup, 3))
            )
        ref = sum(n * sorted(r)[1] for n, r, _ in cases)
        new = sum(n * sorted(x)[1] for n, _, x in cases)
        spreads = [max(early.spread(r), early.spread(x)) for _, r, x in cases]
        rec = {
            "correct": True,
            "speedup": ref / new,
            "cases": [{"timing_spread": s} for s in spreads],
        }
        keep = budget.improves(rec, bar)
        winners += keep
        stop = early.discard([early.Timed(n, r[:2], x[:2]) for n, r, x in cases], bar, 3)
        if stop is not None:
            stopped += 1
            assert not keep, (bar, cases, stop)
            assert stop["bound"] * (1 + stop["noise"]) < bar and stop["rounds"] == 2
    assert stopped > 800 and winners > 500  # it stops clear losers; winners exist


def test_early_discard_needs_its_rounds_and_a_clear_loser():
    one = [early.Timed(10, [1.0], [2.0])]
    assert early.discard(one, 1.0, 3) is None  # one round of three: not bounded yet
    two = [early.Timed(10, [1.0, 1.01], [2.0, 2.02])]
    stop = early.discard(two, 1.0, 3)
    assert stop is not None and stop["bound"] == pytest.approx(1.01 / 2.0, rel=1e-3)
    assert "cannot be a new best" in stop["why"]
    three = [early.Timed(10, [1.0, 1.01, 1.0], [2.0, 2.02, 2.0])]
    assert early.discard(three, 1.0, 3) is None  # all rounds ran
    close = [early.Timed(10, [1.0, 1.0], [0.99, 0.995])]
    assert early.discard(close, 1.0, 3) is None  # faster than the bar: timed in full


# ------------------------------------------------------------------ racing


class _Replay:
    """A sweep's case replay that names what is called: (holder, case index)."""

    def call(self, case: dict[str, Any], *holders: Any) -> tuple[Any, int]:
        return (holders[0], case["ci"])


def _race(seed: int, race: bool) -> tuple[int, int]:
    """One synthetic sweep (configs within a few % of the best and far slower ones, 1 %
    round noise, now and then a disturbed round), timed by :func:`sweep._time_configs`:
    (the index of its best config, timed config-rounds)."""
    rng = random.Random(seed)
    n = rng.randint(4, 32)
    cases = [{"ci": ci, "count": rng.randint(1, 30), "args": (), "kwargs": {}} for ci in range(2)]
    ref_ms = [rng.uniform(0.02, 1.0) for _ in cases]
    factor = [
        1.0 if i == 0 else rng.choice([rng.uniform(1.0, 1.05), rng.uniform(1.1, 6.0)])
        for i in range(n)
    ]
    calls: dict[tuple[Any, int], int] = {}
    timed = 0

    def timer(fn: tuple[Any, int], args: Any, kwargs: Any, **_: Any) -> dict[str, float]:
        nonlocal timed
        who, ci = fn
        rounds = calls[fn] = calls.get(fn, 0) + 1
        noise = random.Random(f"{seed}:{who}:{ci}:{rounds}")  # the same in both modes
        ms = ref_ms[ci] * (1.0 if who == "ref" else factor[who] * (1 + 0.003 * ci))
        timed += who != "ref"
        return {"median_ms": _rounds(noise, ms, 1)[0]}

    rows = [{"index": i, "config": {"i": i}, "correct": True} for i in range(n)]
    passing = [(row, (row["index"],)) for row in rows]
    sweep._time_configs(
        "ref",
        cases,
        [0, 1],
        passing,
        timer=timer,
        check_output=None,
        l2_flush=False,
        deadline=None,
        emit=lambda event: None,
        replay=_Replay(),
        race=race,
    )
    return sweep.rank(rows)[0]["index"], timed


def test_racing_keeps_the_best_config_on_fixtures():
    timed = {False: 0, True: 0}
    for seed in range(300):
        best, spent = _race(seed, False)
        raced, spent_raced = _race(seed, True)
        timed[False] += spent
        timed[True] += spent_raced
        assert raced == best, seed
    assert timed[True] < 0.7 * timed[False], timed


def test_race_never_drops_the_leader_nor_more_than_half():
    def cfg(*new: float) -> list[early.Timed]:
        return [early.Timed(1, [1.0] * len(new), list(new))]

    first = {0: cfg(1.0), 1: cfg(1.4), 2: cfg(1.6), 3: cfg(9.0), 4: cfg(8.0)}
    out = early.race(first, 1, 3)  # one round: only far beyond the noise (50 %), half at most
    assert sorted(out) == [3, 4] and out[3]["leader"] == 0 and out[3]["after"] == 1
    second = {0: cfg(1.0, 1.01), 1: cfg(1.03, 1.04), 2: cfg(1.5, 1.6)}
    # certainly slower beyond the noise: config 2 (config 1 is within 2 % of the leader)
    assert sorted(early.race(second, 2, 3)) == [2]
    assert early.race(second, 3, 3) == {}  # the last round: nothing left to save
    assert early.race({0: cfg(1.0)}, 1, 3) == {}


# ------------------------------------------------------------------ refuted ideas


def _row(idea: str, status: str, speedup: float | None, exp: int, spread: float = 0.005) -> dict:
    return {
        "idea": idea,
        "status": status,
        "correct": status in (ledger.KEEP, ledger.DISCARD),
        "speedup": speedup,
        "spread": spread,
        "exp": exp,
        "snapshot": f"{exp:03d}_x.py",
    }


def test_ideas_are_refuted_after_three_correct_tries_none_near_the_best():
    rows = [_row("base", ledger.KEEP, 1.5, 1)]
    rows += [_row("tile", ledger.DISCARD, s, 2 + i) for i, s in enumerate((1.2, 1.25, 1.3))]
    rows += [_row("near", ledger.DISCARD, s, 5 + i) for i, s in enumerate((1.2, 1.3, 1.49))]
    rows += [_row("bugs", "build_error", None, 8 + i) for i in range(4)]
    rows += [_row("two", ledger.DISCARD, s, 12 + i) for i, s in enumerate((1.1, 1.2))]
    stats = {s["idea"]: s for s in ledger.ideas(rows)}
    assert stats["tile"]["verdict"] == "refuted"
    assert stats["tile"]["refuted"] == {"tries": 3, "best": 1.3, "target_best": 1.5}
    assert stats["near"]["verdict"] == "slow"  # 1.48 is within the noise of 1.5
    assert stats["bugs"]["verdict"] == "buggy"  # bugs never refute an idea
    assert stats["two"]["verdict"] == "slow"  # two tries are not enough
    assert stats["base"]["verdict"] == "kept"


def test_the_critic_rejects_variants_of_a_refuted_idea(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    run.capture_file("t").write_bytes(b"")
    speedups = iter([1.5, 1.2, 1.25, 1.3, 1.1])
    monkeypatch.setattr(
        tools_mod,
        "run_evaluation",
        lambda capture, snap, **kw: {"status": "ok", "correct": True, "speedup": next(speedups)},
    )
    call = _server(run, monkeypatch, critic_mode=critic.STATIC)
    out = []
    for i, idea in enumerate(["base", "tile", "tile", "tile", "tile"]):
        (tdir / "candidates" / f"v{i}.py").write_text(f"def build(r):\n    return {i}\n")
        out.append(call("evaluate_candidate", candidate=f"candidates/v{i}.py", idea_id=idea))
    assert out[3]["idea"]["verdict"] == "refuted" and "refuted: stop" in out[3]["idea"]["note"]
    rejected = out[4]  # the fourth variant: withdrawn, no evaluation used
    assert rejected["status"] == "reviewed" and not rejected["evaluated"], rejected
    assert (
        rejected["review"]["check"] == critic.REFUTED and rejected["review"]["by"] == critic.LEDGER
    )
    assert rejected["budget"]["counted"] is False
    forced = call("evaluate_candidate", candidate="candidates/v4.py", idea_id="tile", force=True)
    assert forced["ledger"]["status"] == ledger.DISCARD
    assert "refuted idea" in forced["review"]["note"]
    reviews = critic.stats(critic.load(run).values())
    assert reviews["labels"] == {critic.STATIC: [0, 0], critic.MODEL: [0, 0]}  # no label
    critic.close(run)


# ------------------------------------------------------------------ lease batching


def _job(target: str | None, kind: str = "eval", session: str = "s", seq: int = 0) -> gpuqueue.Job:
    job = gpuqueue.Job(
        kind=kind, job_class=gpuqueue.KINDS[kind][0], estimate_s=9.0, session=session, target=target
    )
    job.seq, job.submitted = seq, gpuqueue.clock()
    return job


def test_lease_batching_runs_one_targets_jobs_back_to_back():
    gate = gpuqueue.Gate()
    gate._admitted(_job("attn", session="a"), (0,))  # an attn evaluation took the GPU
    other = _job("mlp", session="m", seq=1)  # never served: first in a plain round robin
    same = _job("attn", session="b", seq=2)
    gate.waiting += [other, same]
    assert gate.head() is same  # the same target and class goes next
    quick = _job("mlp", "quick", session="m", seq=3)
    gate.waiting.append(quick)
    assert gate.head() is quick  # never before a better class
    gate.waiting.remove(quick)
    for i in range(gpuqueue.LEASE_JOBS - 1):
        gate._admitted(_job("attn", session=f"x{i}"), (0,))
    assert gate.lease is not None and gate.lease.jobs == gpuqueue.LEASE_JOBS
    assert gate.head() is other  # the lease is full: the others' turn
    gate._admitted(_job(None), (0,))  # a job without a target ends it
    assert gate.lease is None


# ------------------------------------------------------------------ the tools


def _server(run, monkeypatch, *, budget_=None, critic_mode=None):
    """The tools of ``run`` (target ``t``); returns call(tool, **args) -> result."""
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    if critic_mode is not None:
        critic.open_critic(run, critic_mode)
    server = {t.name: t for t in tools_mod.build_server(run, budget_ or Budget(run))}

    def call(tool: str, **args: Any) -> dict[str, Any]:
        if tool.startswith("evaluate_candidate"):
            args = {"target_id": "t", "hypothesis": "h", **args}
        out = asyncio.run(server[tool].handler(args))
        return json.loads(out["content"][0]["text"])

    return call


def _kernel_run(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    run.capture_file("t").write_bytes(b"")
    write_json(tdir / "spec.json", {"id": "t", "module_class": "M"})
    return run, tdir


def test_evaluate_candidate_passes_the_keep_bar_for_an_early_discard(tmp_path, monkeypatch):
    run, tdir = _kernel_run(tmp_path)
    seen = []

    def fake(capture, snap, **kw):
        seen.append(kw)
        return {"status": "ok", "correct": True, "speedup": 1.5}

    monkeypatch.setattr(tools_mod, "run_evaluation", fake)
    call = _server(run, monkeypatch)
    for i in range(2):
        (tdir / "candidates" / f"v{i}.py").write_text(f"def build(r):\n    return {i}\n")
        call("evaluate_candidate", candidate=f"candidates/v{i}.py")
    assert [kw.get("early_best") for kw in seen] == [1.0, 1.5]  # the reference, then the best
    (tdir / "candidates" / "q.py").write_text("def build(r):\n    return 'q'\n")
    call("evaluate_candidate", candidate="candidates/q.py", mode="quick")
    assert "early_best" not in seen[-1]  # a quick check is never timed
    off = _server(run, monkeypatch, budget_=Budget(run, early_stop=False))
    (tdir / "candidates" / "v9.py").write_text("def build(r):\n    return 9\n")
    off("evaluate_candidate", candidate="candidates/v9.py")
    assert "early_best" not in seen[-1]  # --early-stop off


def test_early_discards_are_marked_in_the_record_and_the_ledger(tmp_path, monkeypatch):
    run, tdir = _kernel_run(tmp_path)
    stop = {"rounds": 2, "of": 3, "bound": 0.6, "bar": 1.0, "noise": 0.01, "why": "..."}
    result = {"status": "ok", "correct": True, "speedup": 0.55, "early": stop}
    monkeypatch.setattr(tools_mod, "run_evaluation", lambda c, s, **kw: dict(result))
    call = _server(run, monkeypatch)
    (tdir / "candidates" / "v.py").write_text("def build(r):\n    return 1\n")
    out = call("evaluate_candidate", candidate="candidates/v.py")
    assert out["early"] == stop and out["ledger"]["status"] == ledger.DISCARD
    row = ledger.rows(run)[-1]
    assert row["early"] is True and row["status"] == ledger.DISCARD
    assert read_jsonl(run.results_file("t"))[-1]["early"] == stop


def test_evaluate_candidates_records_each_and_counts_each(tmp_path, monkeypatch):
    run, tdir = _kernel_run(tmp_path)
    batches = []

    def fake(capture, snaps, **kw):
        batches.append((list(snaps), kw))
        outcomes = {"v0": 1.3, "v1": None, "v2": 1.1}
        return [
            {"status": "ok", "correct": True, "speedup": outcomes[s.stem.split("_")[1]]}
            if outcomes[s.stem.split("_")[1]]
            else {"status": "incorrect", "correct": False}
            for s in snaps
        ]

    monkeypatch.setattr(tools_mod, "run_evaluations", fake)
    monkeypatch.setattr(
        tools_mod,
        "run_evaluation",
        lambda c, s, **kw: {"status": "ok", "correct": True, "speedup": 1.2},
    )
    call = _server(run, monkeypatch)
    for i in range(3):
        (tdir / "candidates" / f"v{i}.py").write_text(f"def build(r):\n    return {i}\n")
    (tdir / "candidates" / "old.py").write_text("def build(r):\n    return 'old'\n")
    call("evaluate_candidate", candidate="candidates/old.py")  # evaluated before: a duplicate
    items = [{"candidate": f"candidates/v{i}.py", "hypothesis": f"variant {i}"} for i in range(3)]
    items += [{"candidate": "candidates/old.py", "hypothesis": "again"}]
    items += [{"candidate": "candidates/missing.py", "hypothesis": "gone"}]
    out = call("evaluate_candidates", candidates=items, idea_id="tile")
    ((snaps, kw),) = batches
    assert [s.stem.split("_")[1] for s in snaps] == ["v0", "v1", "v2"] and kw["early_best"] == 1.2
    results = out["results"]
    assert [r.get("ledger", {}).get("status") for r in results[:3]] == [
        "keep",
        "incorrect",
        "discard",
    ]
    assert results[3]["duplicate_of"] and results[4]["status"] == "error"
    assert out["budget"]["evals_used"] == 1 + 3  # old.py before, one per candidate that ran
    rows = [r for r in ledger.rows(run) if r["status"] != ledger.DUPLICATE]
    assert [r["idea"] for r in rows] == ["", "tile", "tile", "tile"]
    assert results[2]["idea"]["tries"] == 3


def test_evaluate_e2e_batch_records_each_set(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    run.transforms_dir.mkdir(parents=True, exist_ok=True)
    for name in ("a", "b"):
        (run.transforms_dir / f"{name}.py").write_text(
            f"# {name}\ndef apply(workload):\n    pass\n"
        )
    calls = []

    def fake(run_, sets, common, **kw):
        calls.append((sets, common))
        return [
            {"status": "ok", "passed": True, "median_ms": 8.0, "speedup": 1.25, "batch": True},
            {"status": "ok", "passed": False, "reason": "outputs differ", "median_ms": 9.0},
        ]

    monkeypatch.setattr(tools_mod, "e2e_batch", fake)
    call = _server(run, monkeypatch)
    sets = [{"transforms": ["a.py"], "hypothesis": "a"}, {"transforms": ["b.py"]}, {}]
    out = call("evaluate_e2e_batch", sets=sets)
    ((given, common),) = calls
    assert [len(s["transform"]) for s in given] == [1, 1] and common[:2] == ["--warmup", "2"]
    results = out["results"]
    assert (
        results[0]["ledger"]["status"] == "keep" and results[1]["ledger"]["status"] == "incorrect"
    )
    assert results[2]["status"] == "error" and out["budget"]["evals_used"] == 2
    assert [r["target"] for r in ledger.rows(run)] == ["e2e", "e2e"]


# ------------------------------------------------------------------ batch evaluation (CPU)


def _cpu_capture(path: Path) -> Path:
    torch.manual_seed(0)
    module = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        module.weight.copy_(torch.randn(64) * 0.1 + 1)
    capture_calls(
        module.eval(), [((torch.randn(1, 16, 64),), {}, 1), ((torch.randn(1, 1, 64),), {}, 7)], path
    )
    return path


GOOD = """from torch import nn


class Norm(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.weight, self.eps = ref.weight, ref.variance_epsilon

    def forward(self, x):
        h = x.float()
        return self.weight * (h * (h.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()).to(x.dtype)


def build(reference):
    return Norm(reference)
"""


def test_a_batch_evaluates_each_candidate_in_one_process(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    capture = _cpu_capture(tmp_path / "c.pt")
    good, wrong, broken = (tmp_path / f"{n}.py" for n in ("good", "wrong", "broken"))
    good.write_text(GOOD)
    wrong.write_text(GOOD.replace("self.weight * (", "2 * self.weight * ("))
    broken.write_text("def build(reference):\n    raise RuntimeError('nope')\n")
    outputs = tmp_path / "out"
    outputs.mkdir()
    argv = [
        str(capture),
        str(good),
        "--also",
        str(wrong),
        "--also",
        str(broken),
        "--also",
        str(good),
    ]
    assert evaluate_mod.main([*argv, "--outputs-dir", str(outputs)]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert [(x["index"], x["event"]) for x in lines[:2]] == [(0, "running"), (0, "result")]
    results = [x for x in lines if x["event"] == "result"]
    assert [x["index"] for x in results] == [0, 1, 2, 3]
    assert [x["status"] for x in results] == ["ok", "incorrect", "build_error", "ok"]
    assert sorted(p.name for p in outputs.iterdir()) == ["0.pt", "3.pt"]  # the correct ones


def test_run_evaluations_goes_on_after_a_candidate_kills_its_process(
    tmp_path, monkeypatch, request
):
    from kernel_agent import gpulock

    # a private lock and one fake GPU (no nvidia-smi through the fake subprocess.run)
    request.addfinalizer(gpulock.pool.cache_clear)
    monkeypatch.setattr(gpulock, "CACHE_DIR", tmp_path / "locks")
    for name in (gpulock.ENV, gpulock.GPUS_ENV, gpulock.INDEX_ENV, "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gpulock, "_nvidia_smi", lambda: [gpulock.GPU(0, "Fake", "GPU-0-fake")])
    gpulock.pool.cache_clear()
    monkeypatch.setattr(gpuqueue, "_gates", {})
    monkeypatch.setattr(evaluate_mod, "ensure_peaks", lambda: None)
    spawned = []

    def run(cmd, *, input, **kwargs):
        tag = f"{evaluate_mod.RESULT_MARKER}{input.strip()}@@"
        paths = [cmd[4], *[cmd[i + 1] for i, x in enumerate(cmd) if x == "--also"]]
        spawned.append([Path(p).name for p in paths])
        lines = []
        for i, path in enumerate(paths):
            if Path(path).name == "crash.py":  # it dies in this one
                return subprocess.CompletedProcess(cmd, -11, "\n".join(lines), "Segmentation fault")
            lines.append(tag + json.dumps({"index": i, "status": "incorrect", "correct": False}))
        return subprocess.CompletedProcess(cmd, 0, "\n".join(lines), "")

    monkeypatch.setattr(subprocess, "run", run)
    names = ["a.py", "crash.py", "b.py"]
    results = evaluate_mod.run_evaluations(tmp_path / "c.pt", [tmp_path / n for n in names])
    assert spawned == [names, ["b.py"]]  # after the crash: the rest in a new process
    assert [r["status"] for r in results] == ["incorrect", "crash", "incorrect"]
    assert "Segmentation fault" in results[1]["error"]
    assert all("evaluator_version" in r for r in results)


# ------------------------------------------------------------------ e2e batch (toy decoder)

NORM = """from torch import nn


class Norm(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.weight = ref.weight
        self.variance_epsilon = ref.variance_epsilon

    def forward(self, x):
        y = x.float()
        y = y * (y.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).rsqrt()
        return self.weight * y.to(x.dtype)


def build(ref):
    return Norm(ref)
"""

WRAP = """from torch import nn


class Wrapped(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def apply(workload):
    workload.model.norm = Wrapped(workload.model.norm)
"""

#: Changes the output: a set that fails its quality check (and must not leak into the next)
SCALE = """import torch


def apply(workload):
    head = workload.model.lm_head
    head.weight = torch.nn.Parameter(head.weight.detach() * -1.0)
"""

#: Modifies a weight in place: cannot be undone in-process
INPLACE = """import torch


def apply(workload):
    with torch.no_grad():
        workload.model.lm_head.weight.mul_(1.0)
"""


@pytest.fixture
def cpu(monkeypatch):
    """The toy decoder on the CPU, every timed run 1 ms of simulated time (#211: real CPU
    timings of the toy on a busy machine are not a verdict)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    monkeypatch.setattr(base, "time", Clock(tick=1e-3))


def _file(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def _toy_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    spec = WorkloadSpec(
        repo_id="toy/decoder",
        modality="llm",
        device="cpu",
        dtype="float32",
        harness=str(TESTS / "toy_decoder.py"),
    )
    write_json(root / "run.json", {"workload": spec.to_dict()})
    write_json(
        root / "targets" / "norm" / "spec.json",
        {"module_class": "ToyRMSNorm", "capture": {"method_instances": {"forward": 5}}},
    )
    return root


def _worker(capsys, *argv: Any) -> dict[str, Any]:
    worker.main([str(a) for a in argv])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)][-1]
    return json.loads(line[len(worker.MARKER) :])


def _verdict(result: dict[str, Any]) -> tuple[Any, ...]:
    """What must not depend on the process a set was measured in: the verdict, the quality
    metrics and each check's verdict (their timings are measurements)."""
    checks = {
        key: (value.get("passed"), value.get("reason")) if isinstance(value, dict) else value
        for key, value in (result.get("metrics") or {}).items()
    }
    return result["status"], result.get("passed"), result.get("reason"), checks


def test_e2e_batch_equals_separate_runs_on_the_toy_decoder(tmp_path, capsys, cpu):
    """One model load for every set: each set's verdict as in its own `e2e` process, its
    time within the noise, the model undone between sets (a set that changes the output
    fails alone), an irreversible set ends the batch for the sets after it."""
    assert toy_decoder.ToyDecoderWorkload  # the harness of the run
    root = _toy_run(tmp_path)
    _worker(capsys, "analyze", "--run-dir", root, "--no-profile", "--iters", 1)
    norm = _file(tmp_path, "norm.py", NORM)
    sets = [
        {"kernel": [f"norm={norm}"]},
        {"transform": [str(_file(tmp_path, "scale.py", SCALE))]},
        {"kernel": [f"norm={norm}"], "transform": [str(_file(tmp_path, "wrap.py", WRAP))]},
        {"transform": [str(_file(tmp_path, "inplace.py", INPLACE))]},
        {"transform": [str(_file(tmp_path, "wrap2.py", WRAP))]},
    ]
    spec = _file(tmp_path, "sets.json", json.dumps(sets))
    batch = _worker(capsys, "e2e_batch", "--run-dir", root, "--sets", spec, "--iters", 5)
    assert batch["status"] == "ok" and batch["undo_check"] == "identical"
    got = batch["sets"]
    assert [r["status"] for r in got] == ["ok", "ok", "ok", "ok", worker.NOT_RUN]
    assert [r["passed"] for r in got[:3]] == [True, False, True]
    assert "cannot be undone in-process" in got[4]["reason"]
    for items, result in list(zip(sets, got, strict=True))[:4]:
        alone = _worker(capsys, "e2e", "--run-dir", root, *worker._set_flags(items), "--iters", 5)
        assert _verdict(result) == _verdict(alone), (items, result, alone)
        assert result["patches"]["replaced"] == alone["patches"]["replaced"]
        if result["passed"]:  # the same timed runs (simulated time: no noise to allow for)
            assert result["median_ms"] == alone["median_ms"] == 1.0
            assert result["times_ms"] == alone["times_ms"]


def test_e2e_batch_falls_back_to_separate_processes(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    calls = []

    def fake(run_, command, *args, **kwargs):
        calls.append((command, list(args)))
        if command == "e2e_batch":
            sets = json.loads(Path(args[1]).read_text())
            assert len(sets) == 3
            return {
                "status": "ok",
                "sets": [
                    {"status": "ok", "passed": True, "median_ms": 1.0},
                    {"status": "irreversible", "passed": False, "reason": "declares undo = False"},
                    {"status": worker.NOT_RUN, "passed": False, "reason": "set 2 ..."},
                ],
                "gpu_index": 0,
            }
        return {"status": "ok", "passed": True, "median_ms": 2.0, "gpu_index": 0}

    monkeypatch.setattr(worker, "call_worker", fake)
    sets = [{"transform": ["a.py"]}, {"transform": ["b.py"]}, {"kernel": ["t=c.py"]}]
    results = worker.e2e_batch(run, sets, ["--warmup", "2"])
    assert [c for c, _ in calls] == ["e2e_batch", "e2e", "e2e"]
    assert calls[1][1] == ["--transform", "b.py", "--warmup", "2"]
    assert calls[2][1] == ["--kernel", "t=c.py", "--warmup", "2"]
    assert results[0]["batch"] and "declares undo" in results[1]["batch_fallback"]
    one = worker.e2e_batch(run, sets[:1], [])
    assert one[0]["median_ms"] == 2.0 and calls[-1][0] == "e2e"  # one set: an e2e
    calls.clear()
    monkeypatch.setattr(worker, "call_worker", lambda *a, **k: {"status": "crash", "error": "x"})
    crashed = worker.e2e_batch(run, sets, [])
    assert all("the batch failed" in r["batch_fallback"] for r in crashed)


def test_ledger_marks_an_ab_that_stopped_early(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    result = {"status": "ok", "passed": True, "median_ms": 9.0, "baseline_ms": 10.0}
    result |= {"speedup": 1.11, "ab": {"stopped": {"verdict": "accept", "rounds": 7, "of": 8}}}
    row = ledger.record_e2e(run, result, backend="integrate", snapshot="x", hypothesis="h")
    plain = ledger.record_e2e(
        run, {**result, "ab": {}}, backend="integrate", snapshot="y", hypothesis="h"
    )
    assert row["early"] is True and plain["early"] is None
    assert [r["early"] for r in ledger.rows(run)] == [True, False]


# ------------------------------------------------------------------ on the GPU

SLOW = """import torch
from torch import nn


class Slow(nn.Module):
    \"\"\"RMSNorm in float64, eight times over: correct, and slower than the reference.\"\"\"

    def __init__(self, ref):
        super().__init__()
        self.weight, self.eps = ref.weight, ref.variance_epsilon

    def forward(self, x):
        h = x.double()
        for _ in range(8):
            y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.double() * y).to(x.dtype)


def build(reference):
    return Slow(reference)
"""


@pytest.mark.gpu
def test_the_evaluator_discards_a_clear_loser_early(tmp_path):
    from kernel_agent.agent.prompts import EXAMPLES_DIR
    from kernel_agent.selftest import make_rmsnorm_capture

    capture = make_rmsnorm_capture(tmp_path / "rms.pt")
    slow = _file(tmp_path, "slow.py", SLOW)
    t0 = time.perf_counter()
    stopped = evaluate_mod.run_evaluation(capture, slow, early_best=1.0)
    early_s = time.perf_counter() - t0
    assert stopped["status"] == "ok" and stopped["correct"], stopped
    assert stopped["early"]["rounds"] == 2 and stopped["speedup"] < 1.0
    assert "perturbed" in stopped["checks"] and stopped["parent_check"] == "ok"  # in full
    t0 = time.perf_counter()
    full = evaluate_mod.run_evaluation(capture, slow)
    full_s = time.perf_counter() - t0
    assert "early" not in full and full["correct"]
    print(f"early discard: {early_s:.1f} s, full: {full_s:.1f} s", file=sys.stderr)
    fast = evaluate_mod.run_evaluation(capture, EXAMPLES_DIR / "triton_rmsnorm.py", early_best=1.0)
    assert fast["correct"] and "early" not in fast  # it can win: timed in full


@pytest.mark.gpu
def test_a_batch_of_real_evaluations_matches_separate_ones(tmp_path):
    from kernel_agent.agent.prompts import EXAMPLES_DIR
    from kernel_agent.selftest import make_rmsnorm_capture

    capture = make_rmsnorm_capture(tmp_path / "rms.pt")
    paths = [EXAMPLES_DIR / "triton_rmsnorm.py", _file(tmp_path, "slow.py", SLOW)]
    t0 = time.perf_counter()
    batch = evaluate_mod.run_evaluations(capture, paths)
    batch_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    alone = [evaluate_mod.run_evaluation(capture, p) for p in paths]
    alone_s = time.perf_counter() - t0
    for b, a in zip(batch, alone, strict=True):
        assert b["status"] == a["status"] == "ok" and b["correct"], (b, a)
        assert b["parent_check"] == "ok" and b["reference_check"] == "ok"
        assert 0.8 < b["speedup"] / a["speedup"] < 1.25  # the same candidate, within noise
    print(f"batch of 2: {batch_s:.1f} s, one by one: {alone_s:.1f} s", file=sys.stderr)
