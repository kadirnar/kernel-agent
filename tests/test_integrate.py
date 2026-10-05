"""Integration candidates from every passing e2e record, and the measured combination as the
seed of the greedy search (issue #43); every comparison of two sets a paired A/B (issue #11).

CPU only: a simulated run (``dryrun.create_run``), records written through the evaluation
tools' own code, and a fake e2e worker that knows the latency of the combinations of the
VoxCPM2 run in the issue (runs/openbmb--VoxCPM2/20261005-042829).
"""

import argparse
import asyncio
import os
import random
import statistics
from pathlib import Path

import pytest

from kernel_agent import abtest, charts, dryrun, ledger, orchestrator, truth
from kernel_agent.agent.tools import record_candidate, record_e2e_result, snapshot
from kernel_agent.config import OptimizeConfig
from kernel_agent.report import write_report
from kernel_agent.workspace import read_json

BASE = dryrun.BASELINE_MS

# End-to-end speedups by what is applied: transform idea (+ version), kernel target.
ALONE = {
    "graph_cfm_solver": 2.055,
    "graph_lm_step v1": 1.223,
    "graph_lm_step v2": 1.24,
    "compile_dit_estimator": 1.688,
    "autotune_dit_estimator": 1.72,
    "compile_lm_layers": 1.03,
    "attn": 1.679,
    "rmsnorm": 1.18,
    "mlp": 0.99,  # slower than the baseline end to end
}
EXP = ("graph_cfm_solver", "autotune_dit_estimator", "graph_lm_step v2", "compile_lm_layers")
BEST = frozenset({*EXP, "attn"})  # the issue's 847 ms (6.38x) combination
COMBINATIONS = {
    BEST: 6.378,
    BEST | {"compile_dit_estimator"}: 6.31,  # the DiT compiled twice
    BEST | {"rmsnorm"}: 6.36,  # compile_lm_layers fuses the LM's norms already
}


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)


def make(tmp_path):
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    return orchestrator.Orchestrator(dryrun.create_run(config), config)


def label(item: str) -> str:
    """What an integration item applies: a kernel's target, a transform's first line."""
    if not item.startswith("/"):
        return item.partition("=")[0]
    return Path(item).read_text().splitlines()[0].removeprefix("# ")


#: Run-to-run wiggle of a paired A/B round (median 1.0, ±0.04 %).
WIGGLE = (1.0004, 0.9996, 1.0002, 0.9998, 1.0001, 0.9999, 1.0003, 0.9997)


def parse(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    for flag in ("--kernel", "--transform", "--b-kernel", "--b-transform"):
        parser.add_argument(flag, action="append", default=[])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--iters", type=int, default=5)
    ns, _ = parser.parse_known_args(list(args))
    return ns


def paired(a_ms: float, b_ms: float, rounds: int = 8, passed: bool = True) -> dict:
    """An ``e2e_ab`` result: B's e2e result plus the timings of both states."""
    a = [round(a_ms * WIGGLE[i % 8], 3) for i in range(rounds)]
    b = [round(b_ms * WIGGLE[-1 - i % 8], 3) for i in range(rounds)]
    result = dryrun._e2e_result(statistics.median(b))
    result.update(times_ms=b, ab={"mode": "paired", "a_ms": a, "b_ms": b})
    if not passed:
        result.update(passed=False, reason="teacher-forced: step 8 cosine 0.68385 < 0.7")
    return result


def fake_worker(table: dict[frozenset[str], float], calls: list[list[str]]):
    def speedup(items: list[str]) -> float:
        key = frozenset(label(i) for i in items)
        if not key:
            return 1.0
        if len(key) == 1:
            return ALONE[next(iter(key))]
        assert key in table, f"unexpected combination {sorted(key)}"
        return table[key]

    def worker(run, command, *args):
        assert command in ("e2e", "e2e_ab")
        ns = parse(args)
        calls.append(list(args))
        a = speedup([*ns.kernel, *ns.transform])
        if command == "e2e":
            return dryrun._e2e_result(BASE / a)
        return paired(BASE / a, BASE / speedup([*ns.b_kernel, *ns.b_transform]), ns.rounds)

    return worker


def kernel(run, target: str, speedup: float, saved_ms: float | None = None) -> str:
    """A correct kernel evaluation; returns ``target=<snapshot>`` as the systems agent's
    prompt lists it (the agent's copy in ``targets/<id>/history/``)."""
    src = run.target(target) / "candidates" / "v4.py"
    src.write_text(f"# {target}\n")
    snap = snapshot(run, src, target)
    result = {"status": "ok", "correct": True, "speedup": speedup, "cases": []}
    if saved_ms is not None:
        result["est_saved_ms_per_run"] = saved_ms
    record_candidate(run, target, src, snap, result, hypothesis="v4")
    return f"{target}={run.target(target) / 'history' / snap.name}"


def e2e(run, speedup: float | None, files: list[Path], kernels: list[str] | None = None) -> dict:
    """An ``evaluate_e2e`` record of ``files`` (+ ``kernels``); ``None``: failed quality."""
    snaps = [snapshot(run, f) for f in files]
    result = dryrun._e2e_result(BASE / (speedup or 6.5))
    if speedup is None:
        result.update(passed=False, reason="teacher-forced: step 8 cosine 0.68385 < 0.7")
    record, _ = record_e2e_result(run, result, snaps, kernels or [], hypothesis="")
    return record


def write(run, name: str, version: str = "") -> Path:
    path = run.transforms_dir / f"{name}.py"
    path.write_text(f"# {name} {version}".strip() + "\n\ndef apply(workload):\n    pass\n")
    return path


def again(run, record: dict) -> Path:
    """The agent's copy of a record's first snapshot (``history/001_..._<sha>.py``)."""
    return run.transforms_dir / record["transforms"][0]


def voxcpm2(run) -> dict[str, dict]:
    """The issue's transforms/results.jsonl: the improved transforms were only ever
    evaluated on top of the attention kernel; snapshot names as in the real run."""
    attn = kernel(run, "attn", 2.1)
    kernel(run, "rmsnorm", 1.5)
    kernel(run, "mlp", 1.3)
    cfm = e2e(run, 2.0843, [write(run, "graph_cfm_solver")])  # 001
    e2e(run, 1.2366, [write(run, "graph_lm_step", "v1")])  # 002
    dit = e2e(run, 1.375, [write(run, "compile_dit_estimator")])  # 003
    names = ("graph_cfm_solver", "graph_lm_step", "compile_dit_estimator")
    e2e(run, 5.0473, [run.transforms_dir / f"{n}.py" for n in names])  # 004-006, lm v1
    lm = write(run, "graph_lm_step", "v2")
    e2e(run, 5.9027, [again(run, cfm), again(run, dit), lm], [attn])  # 007-009
    layers = write(run, "compile_lm_layers")
    exp6 = e2e(run, 6.1677, [again(run, cfm), again(run, dit), lm, layers], [attn])  # 010-013
    files = [again(run, cfm), write(run, "autotune_dit_estimator"), lm, layers]
    exp7 = e2e(run, 6.3778, files, [attn])  # 014_001_graph_cfm_solver_<sha>_<sha>.py, 015-017
    e2e(run, None, [again(run, cfm), *files[1:], write(run, "compile_feat_encoder")], [attn])
    return {"exp6": exp6, "exp7": exp7}


def test_snapshot_stem_strips_every_history_prefix_and_suffix():
    stem = ledger.snapshot_stem
    assert stem("history/014_001_graph_cfm_solver_300928ff_300928ff.py") == "graph_cfm_solver"
    assert stem("/r/.truth/transforms/history/016_graph_lm_step_4710b7af.py") == "graph_lm_step"
    assert stem("graph_lm_step.py") == "graph_lm_step"
    assert stem("v2_graph.py") == "v2_graph" and stem("2_stage.py") == "2_stage"
    item = "/x/history/010_001_graph_cfm_solver_300928ff_300928ff.py"
    assert ledger.item_label(item) == "graph_cfm_solver"


def test_candidates_come_from_every_passing_record(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    records = voxcpm2(run)
    items, digests = orch._integration_items()
    transforms = {label(a): Path(a) for k, a in items if k == "transform"}
    # one version per idea, from the fastest passing record that has it; the failed
    # record's compile_feat_encoder is none
    assert set(transforms) == {*EXP, "compile_dit_estimator"}
    newest = {Path(t).name for t in records["exp7"]["transforms"]}
    assert {transforms[t].name for t in EXP} == newest
    assert transforms["graph_cfm_solver"].name.startswith("014_001_graph_cfm_solver_")
    assert transforms["compile_dit_estimator"].name.startswith("011_003_compile_dit_estimator_")
    assert all(p.parent == run.history_dir() for p in transforms.values())  # .truth/
    assert all(digests[str(p)] == truth.sha256_file(p) for p in transforms.values())
    assert sorted(label(a) for k, a in items if k == "kernel") == ["attn", "mlp", "rmsnorm"]

    combo, record = orch._integration_composite(digests)
    assert record["exp"] == records["exp7"]["exp"]
    assert [label(a) for _, a in combo] == [*EXP, "attn"]
    attn = Path(combo[-1][1].partition("=")[2])
    assert attn.parent == run.history_dir("attn") and ".truth" in attn.parts
    assert digests[str(attn)] == truth.sha256_file(attn)


def test_integration_seeds_with_the_measured_combination(tmp_path):
    """The issue's run ended at 928.6 ms (5.82x) although 847 ms (6.38x) was measured."""
    orch = make(tmp_path)
    run = orch.run
    exp = voxcpm2(run)["exp7"]["exp"]
    calls: list[list[str]] = []
    orch.worker = fake_worker(COMBINATIONS, calls)
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    accepted = [a["item"] for a in data["accepted"]]
    assert [label(a) for a in accepted] == [*EXP, "attn"]  # graph_lm_step v2 included
    assert data["final"]["passed"] and data["final"]["speedup"] == pytest.approx(6.378, 1e-3)
    assert data["composite"] == {"items": accepted, "exp": exp, "seeded": True}
    # everything measured again: 8 items alone, the combination, then 2 additions
    measured = [frozenset(label(i) for i in h["items"]) for h in data["history"]]
    assert measured[8:] == [BEST, BEST | {"compile_dit_estimator"}, BEST | {"rmsnorm"}]
    assert len(calls) == len(measured) == 11
    assert all("--baseline-ms" in c and "--verify" in c for c in calls)
    assert "graph_lm_step v1" not in {label(i) for h in data["history"] for i in h["items"]}
    rows = [r for r in ledger.rows(run) if r["backend"] == "integrate"]
    assert rows[8]["hypothesis"].endswith(f"(the combination of exp {exp})")

    manifest = read_json(run.optimized_dir / "manifest.json")
    assert [k["target"] for k in manifest["kernels"]] == ["attn"]
    assert len(manifest["transforms"]) == 4

    steps = charts.integration_steps(data)
    assert [s["kind"] for s in steps] == ["total", "keep", "nogain", "nogain", "total"]
    assert steps[1]["label"] == f"exp {exp} combination" and steps[1]["source"] == "5 items"
    assert [steps[2]["label"], steps[3]["label"]] == ["compile_dit_estimator", "rmsnorm"]
    assert steps[-1]["ms"] == pytest.approx(BASE / 6.378, rel=1e-3)
    report = write_report(run).read_text()
    assert f"measured combination of exp {exp} (5 items) seeded the search" in report

    calls.clear()  # a re-integration of the same files measures nothing again
    asyncio.run(orch.integrate(reuse=True))
    assert calls == []
    assert read_json(run.root / "integration.json")["accepted"] == data["accepted"]


def test_a_slower_combination_does_not_seed(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    exp = voxcpm2(run)["exp7"]["exp"]
    seed = frozenset({"graph_cfm_solver", "autotune_dit_estimator"})
    table = {
        BEST: 1.9,  # slower now than graph_cfm_solver alone (2.055x)
        seed: 3.4,
        seed | {"compile_dit_estimator"}: 3.3,
        seed | {"attn"}: 5.0,
        **{seed | {"attn", extra}: 4.9 for extra in (*EXP[2:], "rmsnorm")},
    }
    orch.worker = fake_worker(table, [])
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    assert data["composite"]["seeded"] is False and data["composite"]["exp"] == exp
    accepted = [label(a["item"]) for a in data["accepted"]]
    assert accepted == ["graph_cfm_solver", "autotune_dit_estimator", "attn"]  # greedy as before
    assert data["final"]["speedup"] == pytest.approx(5.0, 1e-3)
    kinds = [s["kind"] for s in charts.integration_steps(data)]
    assert kinds == ["total", "keep", "nogain", "keep", "nogain", "keep", *["nogain"] * 3, "total"]


def test_combination_with_an_unverified_file_is_no_seed(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    attn = kernel(run, "attn", 2.1)
    wip = run.target("attn") / "candidates" / "wip.py"
    wip.write_text("# attn\n")
    graph, lm = write(run, "graph_cfm_solver"), write(run, "graph_lm_step")
    first = e2e(run, 2.5, [graph], [f"attn={wip}"])  # a kernel candidate, not a snapshot
    best = e2e(run, 2.2, [graph, lm], [attn])
    assert orch._integration_composite({})[1]["exp"] == best["exp"]

    snap = run.history_dir() / Path(best["transforms"][1]).name
    os.chmod(snap, 0o644)  # what an agent's Bash can do
    snap.write_text("# graph_lm_step\nCACHED = True\n")
    assert orch._integration_composite({}) is None
    items, _ = orch._integration_items()  # the verified transforms still count
    assert [Path(a).name for k, a in items if k == "transform"] == [
        Path(first["transforms"][0]).name
    ]


# ------------------------------------------------------------------ paired A/B (issue #11)

#: True latency factor of each item (product over a set).
FACTOR = {"attn": 0.6, "gain3": 0.97, "noise": 0.996, "inplace": 0.96, "mlp": 0.95}


def noisy_worker(calls: list[tuple[str, list[str]]], drift: dict[frozenset[str], float]):
    """A consumer GPU: every ``e2e`` process runs at its own clock (``drift`` per set of
    items, default 1.0) with ±0.3 % run-to-run jitter; the rounds of an ``e2e_ab`` share a
    drift of up to ±3 % per round, plus ±0.6 % jitter per run. ``inplace`` cannot be undone
    in-process."""

    def ms(items: list[str]) -> float:
        out = BASE
        for item in items:
            out *= FACTOR[label(item)]
        return out

    def worker(run, command, *args):
        ns = parse(args)
        calls.append((command, list(args)))
        a, b = [*ns.kernel, *ns.transform], [*ns.b_kernel, *ns.b_transform]
        rng = random.Random(f"{command} {sorted(map(label, a))} {sorted(map(label, b))}")
        if command == "e2e":
            factor = drift.get(frozenset(map(label, a)), 1.0)
            times = [ms(a) * factor * rng.uniform(0.997, 1.003) for _ in range(ns.iters)]
            result = dryrun._e2e_result(statistics.median(times))
            return {**result, "times_ms": [round(t, 3) for t in times]}
        if any(label(i) == "inplace" for i in [*a, *b]):
            path = next(i for i in [*a, *b] if label(i) == "inplace")
            return {"status": "irreversible", "passed": False, "irreversible": [path]}
        a_ms, b_ms = [], []
        for _ in range(ns.rounds):
            shared = rng.uniform(0.97, 1.03)
            a_ms.append(round(ms(a) * shared * rng.uniform(0.994, 1.006), 3))
            b_ms.append(round(ms(b) * shared * rng.uniform(0.994, 1.006), 3))
        result = dryrun._e2e_result(statistics.median(b_ms))
        return {**result, "times_ms": b_ms, "ab": {"mode": "paired", "a_ms": a_ms, "b_ms": b_ms}}

    return worker


def test_decision_rule_on_synthetic_timings():
    base = [100.0, 103.0, 98.0, 101.0, 99.0, 102.0, 100.5, 97.5]  # ±3 % drift per round
    win = abtest.judge({"mode": "paired", "a_ms": base, "b_ms": [t * 0.97 for t in base]})
    assert win["accepted"] and win["wins"] == 8 and win["gain"] == pytest.approx(0.03)
    assert win["ci95"][0] > 0.01 and win["why"] == ""
    assert "B won 8/8 rounds, gain +3.0 %" in abtest.describe(win)

    # B equal to A up to ±0.6 % per run: about half of the rounds, no gain
    rng = random.Random(1)
    same = [t * rng.uniform(0.994, 1.006) for t in base]
    tie = abtest.judge({"mode": "paired", "a_ms": base, "b_ms": same})
    assert not tie["accepted"] and tie["wins"] < 8 * 0.8 and "needs 80 %" in tie["why"]

    # 1.5 % faster in 7 of 8 rounds, one round 4 % slower: the interval reaches below 1 %
    gains = [0.985] * 7 + [1.04]
    b_ms = [t * g for t, g in zip(base, gains, strict=True)]
    mixed = abtest.judge({"mode": "paired", "a_ms": base, "b_ms": b_ms})
    assert mixed["wins"] == 7 and mixed["win_rate"] == 0.875 and mixed["gain"] > 0.0
    assert not mixed["accepted"] and "95 % CI of the gain starts at" in mixed["why"]

    # wins every round by 0.8 %: significant, but not the 1 % the rule asks for
    small = {"mode": "paired", "a_ms": base, "b_ms": [t * 0.992 for t in base]}
    assert abtest.judge(small)["wins"] == 8 and not abtest.judge(small)["accepted"]
    assert abtest.judge(small, min_gain=0.005)["accepted"]

    # separate processes: unpaired runs, the win rate is over all (A, B) pairs
    a = [100.0 + 0.1 * i for i in range(10)]
    sep = abtest.judge({"mode": "separate", "a_ms": a, "b_ms": [t * 0.95 for t in a]})
    assert sep["accepted"] and sep["win_rate"] == 1.0 and sep["wins"] is None
    assert "B won 100 % of run pairs" in abtest.describe(sep)
    assert not abtest.judge({"mode": "separate", "a_ms": a, "b_ms": a})["accepted"]


def test_paired_ab_rejects_what_the_old_rule_accepted_as_noise(tmp_path):
    """``gain3`` saves 3 %, ``noise`` 0.4 % (less than the 1 % the rule asks for). A
    separate process that happens to run 2.5 % faster made the old rule (B's median <
    0.99 x A's median from another process) keep ``noise``."""
    orch = make(tmp_path)
    run = orch.run
    kernel(run, "attn", 2.1, saved_ms=0.45 * BASE)
    e2e(run, 1.02, [write(run, "noise")])
    e2e(run, 1.03, [write(run, "gain3")])
    calls: list[tuple[str, list[str]]] = []
    drift = {frozenset({"attn", "gain3", "noise"}): 0.975}
    orch.worker = noisy_worker(calls, drift)
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    assert [label(a["item"]) for a in data["accepted"]] == ["attn", "gain3"]
    assert [c for c, _ in calls] == ["e2e_ab"] * 5  # 3 items alone, 2 additions
    singles, (plus_gain, plus_noise) = data["history"][:3], data["history"][3:]
    assert all(h["ab"]["a_items"] == [] and h["passed"] for h in singles)  # vs the baseline
    assert plus_gain["ab"]["accepted"] and plus_gain["ab"]["wins"] == 8
    assert [label(i) for i in plus_gain["ab"]["a_items"]] == ["attn"]
    assert not plus_noise["ab"]["accepted"] and plus_noise["passed"]
    assert plus_noise["ab"]["ci95"][0] < 0.01 < plus_gain["ab"]["ci95"][0]
    assert plus_noise["ab"]["a_items"] == [a["item"] for a in data["accepted"]]

    # what the old rule saw: B from a separate process vs A's median from another one
    final = data["final"]
    items = plus_noise["items"]
    combo = [("kernel", items[0]), *(("transform", t) for t in items[1:])]
    old = orch.worker(run, "e2e", *orchestrator._cli(combo, iters=5))
    assert old["passed"] and old["median_ms"] < final["median_ms"] * 0.99

    # projected (baseline − Σ est. saved) vs measured of every accepted set
    first, second = data["projection"]
    assert first["items"] == plus_gain["ab"]["a_items"]
    assert first["projected_ms"] == pytest.approx(0.55 * BASE)
    assert first["measured_ms"] == pytest.approx(0.6 * BASE, rel=0.03)  # its session
    alone = next(h for h in singles if h["items"] == [plus_gain["items"][1]])
    saved = BASE * alone["ab"]["gain"]  # the paired gain alone
    assert saved == pytest.approx(0.03 * BASE, rel=0.1)
    assert second["projected_ms"] == pytest.approx(0.55 * BASE - saved, abs=0.01)
    assert second["measured_ms"] == final["median_ms"]
    report = write_report(run).read_text()
    assert "| accepted set | projected ms | measured ms | measured / projected |" in report
    assert "paired A/B: B won 8/8 rounds" in report

    steps = charts.integration_steps(data)
    assert [s["kind"] for s in steps] == ["total", "keep", "keep", "nogain", "total"]
    assert steps[2]["from"] == plus_gain["ab"]["a_median_ms"]  # A of that session
    rows = [r for r in ledger.rows(run) if r["backend"] == "integrate"]
    assert len(rows) == 5 and rows[-1]["hypothesis"].endswith("noise")

    calls.clear()  # the same A and B: nothing measured again
    asyncio.run(orch.integrate(reuse=True))
    assert calls == []
    assert read_json(run.root / "integration.json")["accepted"] == data["accepted"]


def test_paired_singles_keep_a_kernel_a_busy_process_hid(tmp_path):
    """Issue #49's case: a kernel 5 % faster end to end, measured alone in a process that
    ran 8 % slow (507 ms vs the 469 ms baseline), was dropped as slower than the baseline.
    Paired against the unmodified model of the same session, it wins every round."""
    orch = make(tmp_path)
    run = orch.run
    kernel(run, "mlp", 1.4)
    calls: list[tuple[str, list[str]]] = []
    orch.worker = noisy_worker(calls, {frozenset({"mlp"}): 1.08})
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    assert [label(a["item"]) for a in data["accepted"]] == ["mlp"]
    (alone,) = data["history"]
    assert alone["ab"]["wins"] == 8 and alone["ab"]["gain"] == pytest.approx(0.05, abs=0.005)
    old = orch.worker(run, "e2e", "--kernel", alone["items"][0], "--iters", "5")
    assert old["median_ms"] > BASE  # the old rule: "drop (not faster than the baseline)"


def test_a_single_within_the_noise_seeds_nothing(tmp_path):
    """0.4 % faster alone: a candidate (positive paired gain), but no seed (the rule)."""
    orch = make(tmp_path)
    run = orch.run
    e2e(run, 1.01, [write(run, "noise")])
    orch.worker = noisy_worker([], {})
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    (alone,) = data["history"]
    assert alone["passed"] and alone["ab"]["gain"] > 0 and not alone["ab"]["accepted"]
    assert data["accepted"] == [] and data["final"] is None and data["projection"] == []
    assert orch.run.load()["phases"]["integrate"]["speedup"] == 1.0


def test_irreversible_items_are_measured_in_separate_processes(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    kernel(run, "attn", 2.1)
    e2e(run, 1.04, [write(run, "inplace")])
    e2e(run, 1.03, [write(run, "gain3")])
    calls: list[tuple[str, list[str]]] = []
    orch.worker = noisy_worker(calls, {})
    asyncio.run(orch.integrate())

    data = read_json(run.root / "integration.json")
    assert [label(a["item"]) for a in data["accepted"]] == ["attn", "inplace", "gain3"]
    # `inplace` alone falls back to two processes (the unmodified model, then it); the
    # additions involving it make no in-process attempt
    commands = [c for c, _ in calls]
    assert commands == ["e2e_ab", "e2e_ab", "e2e", "e2e", "e2e_ab"] + ["e2e"] * 4
    separate = [args for c, args in calls if c == "e2e"]
    assert {a[a.index("--iters") + 1] for a in separate} == {str(abtest.SEPARATE_ITERS)}
    assert [a.count("--kernel") + a.count("--transform") for a in separate] == [0, 1, 1, 2, 2, 3]
    modes = [h["ab"]["mode"] for h in data["history"]]
    assert modes == ["paired", "separate", "paired", "separate", "separate"]
    assert all(h["ab"]["accepted"] for h in data["history"][3:])
    assert all(len(h["ab"]["a_ms"]) == 10 for h in data["history"] if h["ab"]["mode"] == "separate")
    rows = [r for r in ledger.rows(run) if r["backend"] == "integrate"]
    assert len(rows) == 8 and "(A of an A/B in separate processes)" in rows[1]["hypothesis"]
    assert rows[1]["snapshot"] == "baseline"
    assert [label(i) for i in data["irreversible"]] == ["inplace"]  # remembered
