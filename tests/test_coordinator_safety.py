"""Concurrency safety of one coordinator process (issue #180, docs/MULTIAGENT.md §2 and §5
PR 0): what several agent sessions and worker threads share stays consistent, and long GPU
work does not hold the event loop.

* ``run.json``: every read-modify-write goes through ``workspace.update_json`` (a per-path
  lock), so the truth digests, budget notes and phase lists never overwrite each other;
  ``write_json`` uses a temporary file of its own per process and thread.
* the integration and the captures run in threads (``asyncio.to_thread``);
* every session has its own tools (``SessionBinding``): its evaluation budget and the
  ``session`` of its ledger rows and records;
* ledger rows are read under the ledger's lock, without a row still being written;
* snapshot numbers are taken under a per-directory lock and created with ``O_EXCL``;
* the dashboard is rewritten by one debounced thread per run (``dashboard.Refresher``);
* one process per run (``workspace.coordinator_lock``).

CPU only.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fake_clock import Clock

import kernel_agent
from kernel_agent import budget as budget_mod
from kernel_agent import charts, dashboard, dryrun, ledger, library, orchestrator, program, truth
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.runner import AgentResult
from kernel_agent.agent.tools import SessionBinding, record_candidate, snapshot
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver, improve
from kernel_agent.workspace import (
    COORDINATOR_LOCK,
    RunDir,
    coordinator_lock,
    read_json,
    update_json,
    write_json,
)

THREADS = 6
ROUNDS = 40


def _sealed_run(tmp_path: Path) -> RunDir:
    """A run that keeps its ground truth in ``.truth/`` (``run.json`` has ``truth``)."""
    run = RunDir.create(tmp_path, "org/m")
    write_json(run.run_json, {"config": {}, "phases": {}, "truth": truth.new_section()})
    return run


def _hammer(*jobs) -> None:
    """Run every job in its own thread, all released at once; re-raise the first error."""
    start = threading.Barrier(len(jobs))
    errors: list[BaseException] = []

    def run(job) -> None:
        start.wait()
        try:
            job()
        except BaseException as exc:  # reported by the test
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(job,)) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


# ------------------------------------------------------------------ run.json


def test_threads_writing_run_json_lose_no_truth_digest(tmp_path):
    """Truth.append (the tools and the library seeding thread), budget.note, phase marks,
    program versions and library records at once: every update survives, and a fresh
    process (a resumed run) trusts every record."""
    run = _sealed_run(tmp_path)
    keeper = truth.Truth(run)  # this process's authority
    results = [run.truth_dir / "targets" / f"t{i}" / "results.jsonl" for i in range(3)]

    def append(path: Path):
        return lambda: [keeper.append(path, {"i": i, "pad": "x" * 200}) for i in range(ROUNDS)]

    def note(k: int):
        return lambda: [run_note(k, i) for i in range(ROUNDS)]

    def run_note(k: int, i: int) -> None:
        budget_mod.note(run, "improve", f"note{k}", {"i": i})

    def count() -> None:
        for _ in range(ROUNDS):
            run.update(lambda d: d.__setitem__("counter", d.get("counter", 0) + 1))

    def seeds() -> None:
        for i in range(ROUNDS):
            library.remember_seed(run, f"t{i}", [{"entry": f"e{i}"}])

    def versions() -> None:
        (run.root / program.FILENAME).write_text("# program\n")
        for i in range(ROUNDS):
            program.for_agent(run, f"agent{i}", log=lambda _: None)

    _hammer(*(append(p) for p in results), note(0), note(1), count, seeds, versions)

    data = run.load()
    assert data["counter"] == ROUNDS
    assert [len(data["phases"]["improve"][f"note{k}"]) for k in (0, 1)] == [ROUNDS, ROUNDS]
    assert len(data["library"]["seeded"]) == ROUNDS
    assert len(data["program"]["versions"]) == 1  # one version, recorded once
    fresh = truth.Truth(run)  # digests from run.json only, as a resumed run reads them
    assert fresh.files == keeper.files
    for path in results:
        assert [r["i"] for r in fresh.records(path)] == list(range(ROUNDS))


def test_write_json_writers_of_one_file_never_tear_it(tmp_path):
    path = tmp_path / "shared.json"

    def writer(k: int):
        return lambda: [write_json(path, {"k": k, "data": [k] * 2000}) for _ in range(ROUNDS)]

    _hammer(*(writer(k) for k in range(THREADS)))
    data = json.loads(path.read_text())
    assert data["data"] == [data["k"]] * 2000
    assert [p.name for p in tmp_path.iterdir()] == ["shared.json"]  # no temporary file left


def test_update_json_does_not_rewrite_an_unchanged_file(tmp_path):
    path = tmp_path / "x.json"
    assert update_json(path, lambda d: d.update(a=1)) == {"a": 1}
    before = path.stat().st_mtime_ns
    time.sleep(0.01)
    assert update_json(path, lambda d: None) == {"a": 1}
    assert path.stat().st_mtime_ns == before
    assert update_json(path, lambda d: {"b": 2}) == {"b": 2}  # or returns new data
    assert read_json(path) == {"b": 2}


# ------------------------------------------------------------------ the event loop


@pytest.fixture
def sim(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, dossier=False)
    return orchestrator.Orchestrator(dryrun.create_run(config), config)


class Ticker:
    """A task that ticks while the event loop is free; ``seen`` records, for every fake
    GPU worker call, whether the loop ticked while that call waited for it (a call that
    blocks the loop waits in vain, until ``timeout``)."""

    def __init__(self) -> None:
        self.ticked = threading.Event()
        self.seen: list[bool] = []

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await asyncio.sleep(0.005)
            self.ticked.set()

    def sleep(self, timeout: float = 30.0) -> None:  # in the worker's thread
        self.ticked.clear()
        self.seen.append(self.ticked.wait(timeout))

    def around(self, coro):
        async def main():
            stop = asyncio.Event()
            task = asyncio.create_task(self.run(stop))
            try:
                return await coro
            finally:
                stop.set()
                await task

        return asyncio.run(main())


def test_integration_does_not_block_the_event_loop(sim):
    run, ticker = sim.run, Ticker()
    src = run.target("attn") / "candidates" / "v1.py"
    src.write_text("# attn\n")
    snap = snapshot(run, src, "attn")
    ok = {"status": "ok", "correct": True, "speedup": 2.0, "est_saved_ms_per_run": 10.0}
    record_candidate(run, "attn", src, snap, {**ok, "cases": []}, hypothesis="v1")

    def worker(run, command, *args):  # a GPU worker process: an A/B of the kernel
        ticker.sleep()
        a = [100.0 * (1 + 0.0001 * i) for i in range(8)]
        result = dryrun._e2e_result(60.0)
        return {
            **result,
            "times_ms": [60.0] * 8,
            "ab": {"mode": "paired", "a_ms": a, "b_ms": [60.0] * 8},
        }

    sim.worker = worker
    ticker.around(sim.integrate())
    assert ticker.seen and all(ticker.seen)  # the loop ran while every A/B step slept
    integration = read_json(run.root / "integration.json")
    assert [ledger.item_label(a["item"]) for a in integration["accepted"]] == ["attn"]


def test_captures_do_not_block_the_event_loop(sim):
    ticker = Ticker()

    def worker(run, command, *args):
        assert command == "capture"
        ticker.sleep()
        return {"cases": [{"signature": "x[1]:float32", "count": 1}]}

    sim.worker = worker
    specs = [{"id": f"c{i}", "module_class": "Qwen3MLP", "backends": ["triton"]} for i in (1, 2)]
    assert ticker.around(sim.capture_targets(specs)) == ["c1", "c2"]
    assert ticker.seen == [True, True]


# ------------------------------------------------------------------ sessions


def test_concurrent_sessions_get_their_own_budget_and_session_rows(sim, monkeypatch):
    """Two kernel slices at once: each session's tools have its own evaluation budget
    (no run-wide counter that the second slice would overwrite) and stamp its label."""
    run = sim.run
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(tools_mod, "refresh", lambda *a: None)
    monkeypatch.setattr(
        tools_mod,
        "run_evaluation",
        lambda capture, snap, **kw: {"status": "ok", "correct": True, "speedup": 1.3, "cases": []},
    )
    for target_id in ("attn", "mlp"):  # a simulated run has no captures: stand-ins
        capture = run.capture_file(target_id)
        capture.parent.mkdir(parents=True, exist_ok=True)
        capture.write_bytes(target_id.encode())
        sim.truth.seal(capture)
    started = asyncio.Event()
    running: list[str] = []
    seen: dict[str, dict] = {}

    async def fake_agent(name, *, cwd, mcp_server, result=None, **_):
        target_id = name.removeprefix("kernel-")
        running.append(name)
        if len(running) == 2:
            started.set()
        await started.wait()  # both sessions have started before either evaluates
        assert set(sim.bindings) == {"kernel-attn", "kernel-mlp"}
        tools = {t.name: t for t in mcp_server}
        (cwd / "candidates" / "v1.py").write_text(f"# {target_id}\ndef build(r):\n    return r\n")
        args = {"target_id": target_id, "candidate": "candidates/v1.py", "hypothesis": "h"}
        out = await tools["evaluate_candidate"].handler(args)
        seen[name] = json.loads(out["content"][0]["text"])
        return result or AgentResult(name=name)

    sim.agent_runner = fake_agent

    async def both():
        await asyncio.gather(
            sim.kernel_slice("attn", evaluations=2, digest="", label="kernel-attn#1"),
            sim.kernel_slice("mlp", evaluations=5, digest="", label="kernel-mlp#2"),
        )

    asyncio.run(both())
    assert seen["kernel-attn"]["budget"]["evals_budget"] == 2
    assert seen["kernel-mlp"]["budget"]["evals_budget"] == 5
    assert not hasattr(sim.budget, "kernel_evals")  # no run-wide evaluation budget
    assert sim.bindings == {}  # none running
    rows = {r["target"]: r for r in ledger.rows(run)}
    assert rows["attn"]["session"] == "kernel-attn#1" and rows["mlp"]["session"] == "kernel-mlp#2"
    for target_id, label in (("attn", "kernel-attn#1"), ("mlp", "kernel-mlp#2")):
        (record,) = sim.truth.records(run.results_file(target_id))
        assert record["session"] == label
    events = [e for e in ledger.events(run) if e["event"] == "evaluation"]
    assert {e["session"] for e in events} == {"kernel-attn#1", "kernel-mlp#2"}


def test_dry_run_rows_carry_their_slice(tmp_path, monkeypatch):
    """N = 1: the dry run stays reproducible with the session column, and every agent row
    belongs to the slice (its label) that ran it, which the ledger now says directly."""
    monkeypatch.setattr(charts, "available", lambda: False)

    def loop(base: Path) -> RunDir:
        config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=base, dossier=False)
        orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
        icfg = ImproveConfig(max_slices=5)
        improver = Improver(orch, icfg, require_capture=False, live_charts=False)
        with dryrun.World(orch).installed():
            asyncio.run(improver.improve())
        return orch.run

    a, b = loop(tmp_path / "a"), loop(tmp_path / "b")

    def key(run: RunDir) -> list[tuple]:
        return [(r["target"], r["status"], r["speedup"], r["session"]) for r in ledger.rows(run)]

    assert key(a) == key(b)
    slices = read_json(a.root / "improve.json")["slices"]
    rows = ledger.rows(a)
    for rec in slices:
        mine = [r for r in rows if r["session"] == rec["label"]]
        assert len(mine) == rec["evals"] > 0  # the exp-range attribution agrees at N = 1
    agents = [r for r in rows if not r["hypothesis"].startswith("integration")]
    assert all(r["session"] for r in agents)
    assert all(not r["session"] for r in rows if r not in agents)  # integration: none


# ------------------------------------------------------------------ ledger, snapshots


def test_ledger_rows_leave_out_a_row_still_being_written(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    result = {"status": "ok", "correct": True, "speedup": 1.2, "cases": []}
    ledger.record_kernel(run, "t", result, snapshot="001_a.py", hypothesis="h", session="s#1")
    with run.ledger.open("a") as fh:
        fh.write("2\t2026-10-08 12:00:00\tt\ttorch\t002_b.py")  # another writer, mid-row
    assert [r["exp"] for r in ledger.rows(run)] == [1]
    assert ledger.rows(run)[0]["session"] == "s#1"
    with run.ledger.open("a") as fh:
        fh.write("\n")
    assert [r["exp"] for r in ledger.rows(run)] == [1, 2]
    header = run.ledger.read_text().splitlines()[0].split("\t")
    assert header[-4:] == ["worker", "session", "idea", "hypothesis"] and "queue_s" in header


def test_ledger_rows_and_appends_from_threads(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    result = {"status": "ok", "correct": True, "speedup": 1.2, "cases": []}

    def writer(k: int):
        def job() -> None:
            for i in range(ROUNDS // 2):
                ledger.record_kernel(
                    run, "t", result, snapshot=f"{k}_{i}.py", hypothesis="h", session=f"s{k}"
                )

        return job

    def reader() -> None:
        for _ in range(ROUNDS):
            assert all(r["exp"] for r in ledger.rows(run))

    _hammer(*(writer(k) for k in range(THREADS)), reader)
    rows = ledger.rows(run)
    assert [r["exp"] for r in rows] == list(range(1, THREADS * ROUNDS // 2 + 1))
    for k in range(THREADS):
        assert sum(r["session"] == f"s{k}" for r in rows) == ROUNDS // 2


def test_snapshots_taken_at_once_get_distinct_numbers(tmp_path):
    history = tmp_path / "history"
    sources = []
    for k in range(THREADS):
        for i in range(8):
            src = tmp_path / f"w{k}_v{i}.py"
            src.write_text(f"K = {k}\nI = {i}\n")
            sources.append(src)
    snaps: list[Path] = []

    def worker(k: int):
        return lambda: snaps.extend(
            tools_mod._snapshot(s, history) for s in sources if s.stem.startswith(f"w{k}_")
        )

    _hammer(*(worker(k) for k in range(THREADS)))
    numbers = sorted(int(p.name[:3]) for p in history.glob("*.py"))
    assert numbers == list(range(1, len(sources) + 1))
    for snap in snaps:
        stem = snap.name[4:].rsplit("_", 1)[0]
        assert snap.read_text() == (tmp_path / f"{stem}.py").read_text()


# ------------------------------------------------------------------ dashboard


def test_refresher_coalesces_and_debounces_requests(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    # The interval passes in simulated seconds, when the test says so: on a busy machine a
    # real one passed before the test had made all the requests it coalesces.
    clock = Clock()
    monkeypatch.setattr(dashboard, "time", clock)
    calls: list[tuple[float, list[str] | None]] = []
    rendered = threading.Semaphore(0)

    def render(r: RunDir, targets: list[str] | None) -> None:
        assert r is run
        calls.append((clock.monotonic(), targets))
        rendered.release()

    refresher = dashboard.Refresher(run, interval=0.3, render=render)
    refresher.request("a")  # the first one is served at once (no simulated time passes)
    assert rendered.acquire(timeout=60)
    for target in ("b", "c", "b"):  # within the interval: one rewrite later, coalesced
        refresher.request(target)
    assert refresher.flush(timeout=60) and rendered.acquire(timeout=0)
    assert [t for _, t in calls] == [["a"], ["b", "c"]]
    refresher.request("d")
    refresher.request(None)  # every target
    assert not rendered.acquire(timeout=0.4)  # not before the interval has passed
    clock.sleep(0.3)
    assert rendered.acquire(timeout=60)
    assert [t for _, t in calls] == [["a"], ["b", "c"], None]
    assert calls[2][0] - calls[1][0] == pytest.approx(0.3)  # at most once per interval
    refresher.close(timeout=60)
    refresher.request("e")  # closed: dropped
    time.sleep(0.05)
    assert len(calls) == 3


def test_tools_ask_the_refresher_and_do_not_wait(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    asked: list[str | None] = []
    monkeypatch.setattr(dashboard.Refresher, "request", lambda self, t=None: asked.append(t))
    tools_mod.refresh(run, "t")
    tools_mod.refresh(run)
    assert asked == ["t", None]
    assert dashboard.refresher(run) is dashboard.refresher(RunDir(run.root))  # one per run


# ------------------------------------------------------------------ one process per run


def test_coordinator_lock_admits_one_process_per_run(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    with coordinator_lock(run), coordinator_lock(run):  # re-entrant in a process
        assert (run.root / COORDINATOR_LOCK).read_text().strip() == str(os.getpid())
        src = Path(kernel_agent.__file__).parents[1]
        code = (
            "import sys; from pathlib import Path\n"
            "from kernel_agent.workspace import RunDir, coordinator_lock\n"
            "with coordinator_lock(RunDir(Path(sys.argv[1]))): print('got it')\n"
        )
        other = subprocess.run(
            [sys.executable, "-c", code, str(run.root)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(src)},
            timeout=60,
        )
        assert other.returncode != 0 and "got it" not in other.stdout
        assert f"in use by another kernel-agent process (pid {os.getpid()})" in other.stderr
    with coordinator_lock(run):  # released
        pass


def test_improve_refuses_a_run_another_process_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path)
    run = dryrun.create_run(config)
    fd = os.open(run.root / COORDINATOR_LOCK, os.O_RDWR | os.O_CREAT)  # "another process"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, b"4242\n")
        before = run.run_json.read_text()
        with pytest.raises(SystemExit, match="pid 4242"):
            asyncio.run(improve(str(run.root), config, ImproveConfig(), dry_run=True))
        assert run.run_json.read_text() == before  # refused before anything was written
    finally:
        os.close(fd)


def test_budget_has_no_run_wide_evaluation_counter(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    budget = Budget.from_config(run, OptimizeConfig(model_ref="org/m"))
    assert not hasattr(budget, "kernel_evals") and not hasattr(budget, "transform_evals")
    native = SessionBinding(role="native", agent="native", evaluations=3)
    worker = SessionBinding(role="kernel", target_id="t", worker=2, agent="kernel-t-w2")
    assert native.kernel_agent("t") == "native" and native.e2e_agent() == "native"
    assert worker.kernel_agent("t") == "kernel-t-w2" and worker.kernel_agent("u") == "kernel-u"
    assert worker.e2e_agent() == "systems" and SessionBinding().kernel_agent("t") == "kernel-t"
