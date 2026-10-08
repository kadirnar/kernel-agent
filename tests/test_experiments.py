"""Experiments as first-class records (issue #222): the ledger's ``kind`` and ``title``
columns, the measured ``files`` of every ``evaluation`` event, ``experiments.py`` and
``kernel-agent exp``. CPU only and deterministic: synthetic runs, fake evaluators, threads
and a child process released by a barrier or a file, never by a sleep."""

from __future__ import annotations

import asyncio
import html
import itertools
import json
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from synthetic_run import SINGLES, make_run

from kernel_agent import dashboard, experiments, ledger, status, watch
from kernel_agent.agent import tools as tools_mod
from kernel_agent.budget import Budget, Standing
from kernel_agent.cli import main
from kernel_agent.workspace import RunDir, read_json, write_json

OK = {"status": "ok", "correct": True, "cases": []}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("runs"))


# ------------------------------------------------------------------ kind and title


#: Rows of a VoxCPM2 ledger written before ``kind`` and ``title`` existed (exp 1, 3, 6, 12,
#: 31 (...), 48, 97 of runs/openbmb--VoxCPM2/20261005-192504), plus a baseline measured
#: again for a probe's A/B in separate processes.
OLD_ROWS = [
    ("dit_layer_fp8", "cuda", "001_prior_8eea4be3-8cc6a848_a7b4d179.py", "prior winner"),
    ("e2e", "transform+kernels", "001_hoist_cfm_invariants_285cb8f0.py", "Porting the stack"),
    ("e2e", "integrate", "dit_layer_fp8", "integration: dit_layer_fp8 alone"),
    (
        "e2e",
        "integrate",
        "hoist_cfm_invariants+skip_dead_work+dit_layer_fp8",
        "integration: hoist_cfm_invariants + skip_dead_work + dit_layer_fp8 (the combination "
        "of exp 4)",
    ),
    ("e2e", "integrate", "a+b+c", "integration: a + b + c"),
    (
        "e2e",
        "integrate",
        "a+dit_layer_fp8",
        "integration: a + dit_layer_fp8 (swap dit_layer_fp8 012_cuda_v17_a20565c5.py -> "
        "001_prior_8eea4be3-8cc6a848_a7b4d179.py)",
    ),
    ("e2e", "integrate", "a+b", "integration: a + b (A of an A/B in separate processes)"),
    ("e2e", "integrate", "baseline", "integration: baseline (A of an A/B in separate processes)"),
]
OLD_HEADER = [c for c in ledger.COLUMNS if c not in ("kind", "title")]


def _old_ledger(run: RunDir) -> None:
    lines = ["\t".join(OLD_HEADER)]
    for i, (target, backend, snapshot, hypothesis) in enumerate(OLD_ROWS, 1):
        row = {"exp": i, "target": target, "backend": backend, "snapshot": snapshot}
        row |= {"status": "discard", "correct": "true", "speedup": "1.5", "hypothesis": hypothesis}
        lines.append("\t".join(str(row.get(c, "")) for c in OLD_HEADER))
    run.ledger.write_text("\n".join(lines) + "\n")


def test_kind_of_old_rows_and_the_old_header_stays(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    _old_ledger(run)
    rows = ledger.rows(run)
    assert "kind" not in rows[0] and "title" not in rows[0]
    kinds = [ledger.kind(r) for r in rows]
    integration = ["integration"] * 4
    assert kinds == ["kernel", "e2e", "probe", *integration, "probe"]
    # a row appended to the old ledger keeps its layout: no title, kind derived again
    new = ledger.record_e2e(
        run,
        {"status": "ok", "passed": True, "speedup": 2.0},
        backend="integrate",
        snapshot="x",
        hypothesis="integration: x alone",
        title="probe x #003 alone",
    )
    assert new["kind"] == "probe" and new["exp"] == len(OLD_ROWS) + 1
    assert run.ledger.read_text().splitlines()[0].split("\t") == OLD_HEADER
    last = ledger.rows(run)[-1]
    assert "title" not in last and ledger.kind(last) == "probe"
    assert ledger.title(last) == "probe x alone"  # made up from its hypothesis


def test_new_columns_round_trip(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    title = "split-K=4, RED epilogue"
    ledger.record_kernel(run, "t", OK | {"speedup": 1.3}, snapshot="001_a.py", hypothesis="h")
    ledger.record_kernel(
        run, "t", OK | {"speedup": 1.5}, snapshot="002_b.py", hypothesis="h", title=title
    )
    ledger.record_e2e(
        run,
        {"status": "ok", "passed": True, "speedup": 1.2},
        backend="transform",
        snapshot="001_g.py",
        hypothesis="graph it",
        title="  CUDA graph\tof the\ndecode step ",
    )
    header = run.ledger.read_text().splitlines()[0].split("\t")
    assert tuple(header) == ledger.COLUMNS
    assert header.index("kind") == header.index("status") + 1
    assert header.index("title") == header.index("hypothesis") - 1
    rows = ledger.rows(run)
    assert [r["kind"] for r in rows] == ["kernel", "kernel", "e2e"]
    assert [r["title"] for r in rows] == ["", title, "CUDA graph of the decode step"]
    assert [ledger.title(r) for r in rows][:2] == ["h", title]


def test_status_dashboard_and_watch_show_titles(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}, "phases": {}})
    write_json(run.target("t") / "spec.json", {"id": "t", "module_class": "M"})
    hypothesis = "four k-slices halve the idle CTAs at M=16, so the GEMV is no longer latency bound"
    ledger.record_kernel(
        run,
        "t",
        OK | {"speedup": 1.3},
        snapshot="001_a.py",
        hypothesis=hypothesis,
        idea="splitk",
        title="split-K=4 & RED epilogue",
    )
    latest = status.render(run, width=200).split("last 1 evaluations")[1]  # the rows table
    assert "split-K=4 & RED epilogue" in latest and hypothesis not in latest
    page = dashboard.write_dashboard(run).read_text()
    hover = html.escape(f"[splitk] {hypothesis}")
    assert f'<span title="{hover}">split-K=4 &amp; RED epilogue</span>' in page
    (row,) = watch.state(run)["rows"]
    assert row["title"] == "split-K=4 & RED epilogue" and row["kind"] == "kernel"


def test_title_falls_back_at_a_word_boundary():
    hypothesis = (
        "Prefetching one layer ahead keeps DRAM streaming across layer boundaries instead "
        "of bursting twice per layer"
    )
    made = ledger.title({"target": "t", "idea": "stack_coop", "hypothesis": hypothesis})
    clause = made.removeprefix("[stack_coop] ")
    assert made.startswith("[stack_coop] ") and clause.endswith("…") and len(clause) <= 60
    assert hypothesis.startswith(clause[:-1]) and hypothesis[len(clause) - 1] == " "
    # the first clause, never broken at the point of a number or inside brackets
    text = "split-K=4 with a 17.5 KB (RED, fp32) epilogue: fewer idle CTAs. Then more"
    assert ledger.title({"target": "t", "hypothesis": text}) == (
        "split-K=4 with a 17.5 KB (RED, fp32) epilogue"
    )
    assert ledger.title({"target": "t", "hypothesis": "re-run of exp 1 (5.1x), again"}) == (
        "re-run of exp 1 (5.1x)"
    )
    # no hypothesis: the snapshot's stem
    snap = {"target": "t", "snapshot": "014_001_graph_cfm_solver_300928ff_300928ff.py"}
    assert ledger.title(snap) == "graph_cfm_solver"
    # an agent's title is one line of at most 72 characters; a sweep's suffix stays whole
    long = "fused " + " ".join(f"step{i}" for i in range(30))
    for given in (long, "x" * 100):
        cut = ledger.clean_title(given, " [cfg 3]")
        assert len(cut) <= ledger.TITLE_MAX and cut.endswith("… [cfg 3]")
    assert ledger.clean_title("", " [cfg 3]") == ""


def test_integration_titles_are_generated():
    truth = "/r/.truth"
    dit12 = f"dit_layer_fp8={truth}/targets/dit_layer_fp8/history/012_cuda_v17_a20565c5.py"
    dit1 = f"dit_layer_fp8={truth}/targets/dit_layer_fp8/history/001_prior_a7b4d179.py"
    enc = f"enc_dit_stack_fp8={truth}/targets/enc_dit_stack_fp8/history/013_v22_bc80857c.py"
    hoist = f"{truth}/transforms/history/008_004_hoist_cfm_invariants_285cb8f0_285cb8f0.py"
    title = ledger.integration_title
    assert title([dit12], []) == "probe dit_layer_fp8 #012 alone"
    assert title([hoist, dit12, enc], [hoist, dit12]) == "integrate +enc_dit_stack_fp8 (#013)"
    assert title([hoist, dit1], [hoist, dit12]) == "integrate dit_layer_fp8 #012 -> #001"
    assert title([hoist, enc], [hoist, dit12]) == (
        "integrate +enc_dit_stack_fp8 (#013) -dit_layer_fp8 (#012)"
    )
    assert title([hoist, dit12], [hoist, dit12]) == (
        "measure A again: hoist_cfm_invariants + dit_layer_fp8"
    )
    assert title([], []) == "measure the baseline again"
    many = title(
        [hoist, dit12, enc, *(f"/r/transforms/history/00{i}_t{i}.py" for i in range(9))], []
    )
    assert len(many) <= ledger.TITLE_MAX and many.endswith("…")
    # rows recorded before titles: made up from the hypothesis, versions as #NNN
    rows = [
        {"target": "e2e", "backend": "integrate", "snapshot": s, "hypothesis": h}
        for _, _, s, h in OLD_ROWS[2:]
    ]
    assert [ledger.title(r) for r in rows] == [
        "probe dit_layer_fp8 alone",
        "integrate the combination of exp 4",
        "integrate +c",
        "integrate dit_layer_fp8 #012 -> #001",
        "measure A again: a + b",
        "measure the baseline again",
    ]


def test_item_files_are_run_relative_snapshots(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    history = run.history_dir("attn")
    history.mkdir(parents=True)
    (history / "003_cuda_v2_0a1b2c3d.py").write_text("x")
    transform = run.history_dir() / "002_static_cache_0a1b2c3d.py"
    items = [
        str(transform),
        f"attn={run.target('attn')}/history/003_cuda_v2_0a1b2c3d.py",  # the agent's copy
        "attn=candidates/v9.py",  # not a snapshot: the file given, in the target's directory
    ]
    assert ledger.item_files(run, items) == [
        "transforms/history/002_static_cache_0a1b2c3d.py",
        "attn=targets/attn/history/003_cuda_v2_0a1b2c3d.py",
        "attn=targets/attn/candidates/v9.py",
    ]
    (history / "rewrite.py").write_text("x")  # a region target's sealed rewrite
    assert ledger.kernel_files(run, "attn", "003_cuda_v2_0a1b2c3d.py") == [
        "targets/attn/history/003_cuda_v2_0a1b2c3d.py",
        "targets/attn/history/rewrite.py",
    ]


# ------------------------------------------------------------------ the synthetic run


def test_experiments_of_the_synthetic_run(synthetic):
    run = synthetic
    s = experiments.summary(run)
    totals = ledger.summary(run)
    assert s["experiments"] + s["probes"] == totals["evaluations"]
    assert s["kept"] == totals["keeps"] and s["failed"] == totals["failures"]
    assert s["probes"] == len(SINGLES) and s["by_kind"]["probe"] == len(SINGLES)
    assert sum(s["by_kind"].values()) == totals["evaluations"]
    assert s["repo_id"] == "Qwen/Qwen3-0.6B" and s["baseline_ms"] == pytest.approx(1532.4)
    assert 0 < s["keep_rate"] < 1 and 0 < s["failure_rate"] < 1

    lineages = {ln["lineage"]: ln for ln in s["lineages"]}
    assert next(iter(lineages)) == experiments.MODEL
    assert lineages["model"]["start"] == pytest.approx(1532.4)
    assert lineages["model"]["best"] == pytest.approx(889.0)  # the last kept integration step
    for target in totals["targets"]:
        mine = lineages[target["id"]]
        assert mine["experiments"] == target["evals"] and mine["kept"] == target["keeps"]
        assert mine["best"] == pytest.approx(target["best_speedup"])
        assert mine["failed"] == target["failures"]

    items = experiments.experiments(run)
    assert all(e.measured for e in items) and len(items) == totals["evaluations"]
    top = experiments.kept(run)
    assert len(top) == s["kept"] and all(e.status == ledger.KEEP for e in top)
    gains = [e.gain for e in top]
    assert all(g is not None and g > 0 for g in gains)
    assert gains == sorted(gains, reverse=True)  # ranked by gain
    for lineage in lineages:  # kept values improve along exp
        values = [e.speedup for e in items if e.lineage == lineage and e.status == ledger.KEEP]
        assert values == sorted(values) and len(set(values)) == len(values)
    # every kernel experiment's files are its snapshot, in the run directory
    for e in items:
        if e.kind == ledger.KERNEL:
            assert e.files == (f"targets/{e.lineage}/history/{e.row['snapshot']}",)
            assert (run.root / e.files[0]).is_file()
    e2e = [e for e in items if e.kind == ledger.E2E]
    assert e2e and all(e.files for e in e2e)
    with_kernels = next(e for e in e2e if "attn" in e.row["snapshot"])
    assert any(f.startswith("attn=") for f in with_kernels.files)


def test_parents_and_values(synthetic):
    items = {e.exp: e for e in experiments.all_rows(synthetic)}
    attn = [e for e in items.values() if e.lineage == "attn"]
    assert attn[0].parent_exp is None and attn[0].best_before == 1.0  # the reference
    for e in attn[1:]:  # no `parent` given: the standing best when it was recorded
        best = max((p for p in attn if p.exp < e.exp and p.status == "keep"), key=lambda p: p.exp)
        assert e.parent_exp == best.exp and e.best_before == pytest.approx(best.speedup)
    model = [e for e in items.values() if e.lineage == experiments.MODEL]
    assert model[0].best_before == pytest.approx(1532.4) and model[0].unit == experiments.MS
    failed = [e for e in items.values() if e.failed]
    assert failed and all(e.value is None and e.gain is None for e in failed)


def test_a_parent_snapshot_names_the_parent_experiment(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    for i, (speedup, parent) in enumerate(
        [(1.5, None), (2.0, None), (1.8, "history/001_a_0a1b2c3d.py"), (1.9, "candidates/a.py")]
    ):
        name = "001_a_0a1b2c3d.py" if i == 0 else f"00{i + 1}_{'b' if i == 1 else 'c'}_x.py"
        ledger.record_kernel(
            run, "t", OK | {"speedup": speedup}, snapshot=name, hypothesis="h", parent=parent
        )
    parents = [e.parent_exp for e in experiments.experiments(run)]
    assert parents == [None, 1, 1, 1]  # the best (exp 2) for exp 2; the named file's exp 1


# ------------------------------------------------------------------ concurrency


THREADS, CALLS = 8, 25


def test_numbers_and_keeps_from_threads(tmp_path):
    """8 threads × 25 recordings on 2 targets, released at once: ``exp`` is 1..200, dense
    and unique, and per target the kept speedups strictly increase in ``exp`` order."""
    run = RunDir.create(tmp_path, "org/m")
    start = threading.Barrier(THREADS)
    errors: list[BaseException] = []

    def job(k: int) -> None:
        rng = random.Random(k)
        start.wait(timeout=60)
        try:
            for i in range(CALLS):
                result = OK | {"speedup": round(rng.uniform(0.8, 4.0), 4)}
                ledger.record_kernel(
                    run, f"t{k % 2}", result, snapshot=f"{k}_{i}.py", hypothesis=f"{k}/{i}"
                )
        except BaseException as exc:  # reported below
            errors.append(exc)

    threads = [threading.Thread(target=job, args=(k,)) for k in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    rows = ledger.rows(run)
    assert [r["exp"] for r in rows] == list(range(1, THREADS * CALLS + 1))
    events = [e for e in ledger.events(run) if e["event"] == "evaluation"]
    assert sorted(e["exp"] for e in events) == list(range(1, THREADS * CALLS + 1))
    for target in ("t0", "t1"):
        mine = [r for r in rows if r["target"] == target]
        kept = [r["speedup"] for r in mine if r["status"] == ledger.KEEP]
        assert kept and all(b > a for a, b in itertools.pairwise(kept))
        stand = Standing()  # every row's status relative to every row before it
        for r in mine:
            beats = r["speedup"] > stand.best * 1.01
            assert r["status"] == (ledger.KEEP if beats else ledger.DISCARD)
            if beats:
                stand.keep(r)
    gains = [e.gain for e in experiments.experiments(run) if e.status == ledger.KEEP]
    assert all(g is not None and g > 0.01 for g in gains)


_CHILD = """
import sys
import time
from pathlib import Path
from kernel_agent import ledger
from kernel_agent.workspace import RunDir
run, ready, go = RunDir(Path(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3])
ready.write_text("ready")
while not go.exists():  # the parent's file: both write from now on
    time.sleep(0.001)
for i in range(int(sys.argv[4])):
    ledger.append(run, {"target": "child", "status": "keep", "hypothesis": str(i)})
"""


def test_a_second_writer_process_never_reuses_a_number(tmp_path):
    """The ``flock`` of ``results.tsv``: this process's lock does not reach another's."""
    run = RunDir.create(tmp_path, "org/m")
    ready, go, n = tmp_path / "ready", tmp_path / "go", 40
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(run.root), str(ready), str(go), str(n)]
    )
    try:
        while not ready.exists():  # the child's file: it imported everything
            if child.poll() is not None:
                pytest.fail(f"the child exited with {child.returncode}")
            time.sleep(0.001)
        go.write_text("go")
        for i in range(n):
            ledger.append(run, {"target": "parent", "status": "keep", "hypothesis": str(i)})
        assert child.wait(timeout=120) == 0
    finally:
        child.kill()
    rows = ledger.rows(run)
    assert sorted(r["exp"] for r in rows) == list(range(1, 2 * n + 1))
    for who in ("parent", "child"):
        assert [r["hypothesis"] for r in rows if r["target"] == who] == [str(i) for i in range(n)]


# ------------------------------------------------------------------ the tools


def _kernel_server(tmp_path, monkeypatch, **kw: Any) -> tuple[RunDir, Path, Any]:
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    run.capture_file("t").write_bytes(b"")
    write_json(tdir / "spec.json", {"id": "t", "module_class": "M"})
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    server = {t.name: t for t in tools_mod.build_server(run, Budget(run), **kw)}

    def call(tool: str, **args: Any) -> dict[str, Any]:
        out = asyncio.run(server[tool].handler(args))
        return json.loads(out["content"][0]["text"])

    return run, tdir, (server, call)


def test_tool_schemas_carry_a_title(tmp_path, monkeypatch):
    _, _, (server, _) = _kernel_server(tmp_path, monkeypatch, async_evals=True)
    for name in ("evaluate_candidate", "submit_evaluation", "sweep_candidate", "evaluate_e2e"):
        props = server[name].input_schema["properties"]
        assert props["title"]["type"] == "string" and "72" in props["title"]["description"]
        assert "title" not in server[name].input_schema.get("required", [])
    batch = server["evaluate_candidates"].input_schema["properties"]["candidates"]["items"]
    assert "title" in batch["properties"]
    sets = server["evaluate_e2e_batch"].input_schema["properties"]["sets"]["items"]
    assert "title" in sets["properties"]
    assert "[cfg k]" in server["sweep_candidate"].input_schema["properties"]["title"]["description"]


def test_a_title_reaches_the_row_and_the_record(tmp_path, monkeypatch):
    run, tdir, (_, call) = _kernel_server(tmp_path, monkeypatch)
    monkeypatch.setattr(tools_mod, "run_evaluation", lambda c, s, **kw: OK | {"speedup": 1.4})
    monkeypatch.setattr(  # 1.5x, 1.7x: each a new best
        tools_mod,
        "run_evaluations",
        lambda c, snaps, **kw: [OK | {"speedup": 1.5 + 0.2 * i} for i in range(len(snaps))],
    )
    for name in ("a", "b", "c"):
        (tdir / "candidates" / f"{name}.py").write_text(f"def build(r):\n    return {name!r}\n")
    call("evaluate_candidate", target_id="t", candidate="candidates/a.py", hypothesis="h",
         title="split-K=4, RED epilogue")  # fmt: skip
    call("evaluate_candidate", target_id="t", candidate="candidates/a.py", hypothesis="again",
         title="the same code")  # fmt: skip
    items = [
        {"candidate": "candidates/b.py", "hypothesis": "h", "title": "BLOCK_N 128"},
        {"candidate": "candidates/c.py", "hypothesis": "h"},
    ]
    call("evaluate_candidates", target_id="t", candidates=items)
    rows = ledger.rows(run)
    assert [r["status"] for r in rows] == ["keep", "duplicate", "keep", "keep"]
    assert [r["title"] for r in rows] == [
        "split-K=4, RED epilogue",
        "the same code",
        "BLOCK_N 128",
        "",
    ]
    records = [json.loads(line) for line in run.results_file("t").read_text().splitlines()]
    assert [r.get("title") for r in records] == ["split-K=4, RED epilogue", "BLOCK_N 128", None]
    event = next(e for e in ledger.events(run) if e["event"] == "evaluation")
    assert event["files"] == [f"targets/t/history/{rows[0]['snapshot']}"]


def test_a_sweeps_title_names_its_config(tmp_path, monkeypatch):
    run, tdir, (_, call) = _kernel_server(tmp_path, monkeypatch)
    configs = [{"BLOCK": 64}, {"BLOCK": 128}, {"BLOCK": 256}]

    def run_sweep(capture, src, swept, *, prepare, **kw):
        bound = tdir / "candidates" / "bound.py"
        bound.write_text("def build(r, BLOCK=128):\n    return r\n")
        prepare(bound)
        table = [
            {"index": i, "config": c, "correct": True, "status": "ok", "speedup": 1.0 + i / 10}
            for i, c in enumerate(swept)
        ]
        sweep = {"configs": 3, "passed": 3, "failed": 0, "skipped": 0, "seconds": 1.0}
        return {
            "evaluation": OK | {"speedup": 1.2},
            "config": swept[1],
            "sweep": sweep | {"table": table},
        }

    monkeypatch.setattr(tools_mod.sweep_mod, "run_sweep", run_sweep)
    (tdir / "candidates" / "s.py").write_text("def build(r, BLOCK=64):\n    return r\n")
    out = call("sweep_candidate", target_id="t", candidate="candidates/s.py", configs=configs,
               hypothesis="tile sizes", title="sweep BLOCK")  # fmt: skip
    assert out["config"] == {"BLOCK": 128}
    (row,) = ledger.rows(run)
    assert row["title"] == "sweep BLOCK [cfg 1]"


def test_an_e2e_title_and_its_files(tmp_path, monkeypatch):
    run, _, (_, call) = _kernel_server(tmp_path, monkeypatch)
    run.transforms_dir.mkdir(parents=True, exist_ok=True)
    (run.transforms_dir / "graph.py").write_text("def apply(workload):\n    pass\n")
    result = {"status": "ok", "passed": True, "speedup": 1.3, "times_ms": [1.0, 1.0]}
    monkeypatch.setattr(tools_mod, "call_worker", lambda *a, **kw: dict(result))
    monkeypatch.setattr(
        tools_mod, "e2e_batch", lambda r, sets, common, **kw: [dict(result)] * len(sets)
    )
    run.history_dir("t").mkdir(parents=True)  # a snapshot (the agent's copy in a sealed run)
    (run.history_dir("t") / "001_v_0a1b2c3d.py").write_text("x")
    call("evaluate_e2e", transforms=["graph.py"], kernels=["t=history/001_v_0a1b2c3d.py"],
         hypothesis="graph the step", title="CUDA graph of the decode step")  # fmt: skip
    sets = [{"transforms": ["graph.py"], "title": "graph again"}, {"transforms": ["graph.py"]}]
    call("evaluate_e2e_batch", sets=sets)
    rows = ledger.rows(run)
    assert [r["title"] for r in rows] == ["CUDA graph of the decode step", "graph again", ""]
    assert [r["kind"] for r in rows] == ["e2e"] * 3
    event = next(e for e in ledger.events(run) if e["event"] == "evaluation")
    transform = f"transforms/history/{rows[0]['snapshot'].split('+')[0]}"
    assert event["files"] == [transform, "t=targets/t/history/001_v_0a1b2c3d.py"]
    assert (run.root / transform).is_file()


# ------------------------------------------------------------------ the integration


def test_the_integration_records_titles_kinds_and_files(tmp_path, monkeypatch):
    from test_integrate import COMBINATIONS, dryrun, fake_worker, make, orchestrator, voxcpm2

    from kernel_agent import charts

    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)
    orch = make(tmp_path)
    run = orch.run
    voxcpm2(run)
    orch.worker = fake_worker(COMBINATIONS, [])
    asyncio.run(orch.integrate())

    history = read_json(run.root / "integration.json")["history"]
    rows = [r for r in ledger.rows(run) if r["backend"] == "integrate"]
    assert len(rows) == len(history) == 11  # 8 items alone, the combination, 2 additions
    assert [r["kind"] for r in rows] == ["probe"] * 8 + ["integration"] * 3
    for row, step in zip(rows, history, strict=True):  # B against its A
        assert row["title"] == ledger.integration_title(step["items"], step["ab"]["a_items"])
    singles = [h["items"][0] for h in history[:8]]
    assert [r["title"] for r in rows[:8]] == [
        f"probe {ledger.item_label(x)} {ledger.item_version(x)} alone" for x in singles
    ]
    assert all(re.fullmatch(r"integrate \+\w+ \(#\d{3}\)", r["title"]) for r in rows[9:])
    files = {e["exp"]: e["files"] for e in ledger.events(run) if e["event"] == "evaluation"}
    for row, step in zip(rows, history, strict=True):
        assert files[row["exp"]] == ledger.item_files(run, step["items"])
        for f in files[row["exp"]]:
            assert not Path(f.partition("=")[2] or f).is_absolute()
            assert (run.root / (f.partition("=")[2] or f)).is_file()
    s = experiments.summary(run)
    assert s["probes"] == 8 and s["by_kind"]["integration"] == 3


# ------------------------------------------------------------------ the CLI


def test_exp_cli_on_the_synthetic_run(synthetic, capsys):
    root = str(synthetic.root)
    assert main(["exp", root]) == 0
    out = capsys.readouterr().out
    s = experiments.summary(synthetic)
    assert out.startswith(f"Qwen/Qwen3-0.6B: {s['experiments']} experiments, {s['kept']} kept")
    assert f"+{s['probes']} integration probes" in out
    assert "kept improvements, ranked by gain" in out and "lineage" in out
    top = experiments.kept(synthetic)[0]
    assert f"{experiments.gain_text(top.gain)}  {top.title}" in out

    assert main(["exp", "list", root, "--status", "keep"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:5] == ["exp", "time", "kind", "lineage", "status"]
    assert len(lines) == 1 + s["kept"] and all(" keep " in line for line in lines[1:])
    assert main(["exp", "list", root, "--lineage", "attn", "--status", "failed", "--tsv"]) == 0
    tsv = capsys.readouterr().out.splitlines()
    assert tsv[0].split("\t")[:5] == ["exp", "time", "kind", "lineage", "status"]
    assert len(tsv) == 1 + 5 and all(line.split("\t")[3] == "attn" for line in tsv[1:])
    assert main(["exp", "list", root, "--kind", "probe", "--last", "2"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 3

    attn = [e for e in experiments.experiments(synthetic) if e.lineage == "attn"]
    kept = [e for e in attn if e.status == "keep"]
    assert main(["exp", "show", root, str(kept[2].exp)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"exp {kept[2].exp} · kernel · attn · keep")
    assert kept[2].hypothesis in out and f"parent exp  exp {kept[1].exp}" in out
    assert f"diff against exp {kept[1].exp}" in out and f'+"""{kept[2].hypothesis}' in out
    failed = next(e for e in attn if e.status == "build_error")
    assert main(["exp", "show", root, str(failed.exp)]) == 0
    out = capsys.readouterr().out
    assert "evaluator   build_error" in out and "error tail\n    build_error: synthetic" in out

    assert main(["exp", "diff", root, str(kept[-1].exp), str(kept[0].exp)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"--- exp {kept[0].exp}: targets/attn/history/")
    assert f"+++ exp {kept[-1].exp}: targets/attn/history/" in out
    assert main(["exp", "diff", root, str(attn[0].exp)]) == 1  # no parent: the reference
    assert "give B" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no exp 999"):
        main(["exp", "show", root, "999"])


def test_exp_list_leaves_out_a_row_still_being_written(tmp_path, capsys):
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"card": {"repo_id": "org/m"}})
    ledger.record_kernel(run, "t", OK | {"speedup": 1.3}, snapshot="001_a.py", hypothesis="h")
    with run.ledger.open("a") as fh:
        fh.write("2\t2026-10-08 12:00:00\tt\ttorch\t002_b.py")  # another writer, mid-row
    with run.events.open("a") as fh:
        fh.write('{"ts": 1, "event": "evalu')  # and mid-event
    assert main(["exp", "list", str(run.root)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and lines[1].split()[0] == "1"
    assert main(["exp", str(run.root)]) == 0
    assert "org/m: 1 experiments, 1 kept improvements" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not a run directory"):
        main(["exp", str(tmp_path / "nothing")])
