"""The GPU job queue (gpuqueue.py, admission in gpulock.gpu_lock): fake jobs in threads,
no GPU."""

import asyncio
import fcntl
import functools
import json
import multiprocessing
import os
import queue
import threading
import time
from pathlib import Path

import pytest
from fake_clock import Clock

from kernel_agent import gpulock, gpuqueue, ledger, scheduler
from kernel_agent.agent import tools as tools_mod
from kernel_agent.budget import Budget
from kernel_agent.workspace import RunDir, read_jsonl

POOL_ENV = (gpulock.ENV, gpulock.GPUS_ENV, gpulock.INDEX_ENV, "CUDA_VISIBLE_DEVICES")
TIMEOUT = 60  # seconds a thread waits for the test's next step (never reached when it works)


@pytest.fixture(autouse=True)
def lock_dir(tmp_path, monkeypatch):
    """Lock files in a temporary directory, one fake GPU, a fresh queue."""
    locks = tmp_path / "locks"
    monkeypatch.setattr(gpulock, "CACHE_DIR", locks)
    for name in POOL_ENV:
        monkeypatch.delenv(name, raising=False)
    gpus = [gpulock.GPU(0, "NVIDIA Fake GPU", "GPU-000-fake")]
    monkeypatch.setattr(gpulock, "_nvidia_smi", lambda: list(gpus))
    gpulock.pool.cache_clear()
    monkeypatch.setattr(gpuqueue, "_gates", {})
    yield locks
    gpulock.pool.cache_clear()


def job(kind="eval", job_class=None, estimate=10.0, session=None, **kw):
    return gpuqueue.Job(
        kind=kind,
        job_class=job_class or gpuqueue.KINDS[kind][0],
        estimate_s=estimate,
        session=session,
        **kw,
    )


def until(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def queued(n):
    until(lambda: len(gpuqueue.gate().waiting) >= n)


class Holder:
    """The GPU held by another thread (an untagged job) until :meth:`release`."""

    def __init__(self, job_=None):
        self.go, self.took = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._hold, args=(job_,), daemon=True)
        self.thread.start()
        assert self.took.wait(5)

    def _hold(self, job_):
        with gpuqueue.using(job_ or gpuqueue.Job()), gpulock.gpu_lock():
            self.took.set()
            self.go.wait(30)

    def release(self):
        self.go.set()
        self.thread.join(5)

    def release_when_queued(self, seconds, clock=None):
        """Let the GPU go ``seconds`` after a job queued behind it (not after a timer that
        ran while a busy machine was still getting the job there); on ``clock`` (a
        simulated one) at once, ``seconds`` later in its time."""

        def wait_and_release():
            try:
                queued(1)
            finally:  # (a job that never queued fails the test, it does not hang it)
                if clock is None:
                    time.sleep(seconds)
                else:
                    clock.sleep(seconds)
                self.release()

        threading.Thread(target=wait_and_release, daemon=True).start()


def submit(job_, order, label=None, hold=0.0):
    """A thread that takes the GPU as ``job_``, appends ``label`` to ``order`` and holds the
    GPU ``hold`` seconds (an Event: until it is set)."""

    def work():
        with gpuqueue.using(job_), gpulock.gpu_lock():
            order.append(label or job_.kind)
            if isinstance(hold, threading.Event):
                hold.wait(TIMEOUT)
            else:
                time.sleep(hold)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread


def run_all(jobs, labels=None):
    """Queue ``jobs`` (in this order) behind a holder; returns the order they took the GPU."""
    holder, order, threads = Holder(), [], []
    for i, j in enumerate(jobs):
        threads.append(submit(j, order, labels[i] if labels else None))
        queued(i + 1)
    holder.release()
    for t in threads:
        t.join(10)
    return order


# ------------------------------------------------------------------ order


def test_class_order():
    kinds = ["capture", "dev", "sweep", "e2e", "eval", "quick"]
    jobs = [job(k) for k in kinds] + [job("integration", gpuqueue.DEADLINE)]
    order = run_all(jobs, labels=[*kinds, "final integration"])
    assert order == ["final integration", "quick", "eval", "e2e", "sweep", "dev", "capture"]
    assert not gpuqueue.gate().waiting and not any(gpuqueue.gate().holders.values())


def test_untagged_calls_rank_as_evaluations():
    order = run_all([job("sweep"), gpuqueue.Job(), job("e2e")], labels=["sweep", "cli", "e2e"])
    assert order == ["cli", "e2e", "sweep"]


def test_shortest_job_first_within_a_class():
    jobs = [job("eval", estimate=s, session=f"s{s}") for s in (30.0, 5.0, 10.0)]
    assert run_all(jobs, labels=["30", "5", "10"]) == ["5", "10", "30"]


def test_round_robin_across_sessions():
    """Within a class the session served least recently goes first, before the shortest:
    one session's stream of short jobs does not starve another session."""
    a = [job("sweep", estimate=1.0, session="kernel-a#1") for _ in range(3)]
    b = [job("sweep", estimate=60.0, session="kernel-b#2") for _ in range(3)]
    order = run_all([*a, *b], labels=["a1", "a2", "a3", "b1", "b2", "b3"])
    assert order == ["a1", "b1", "a2", "b2", "a3", "b3"]


def test_aging_moves_a_waiting_job_up(monkeypatch):
    # The queue's clock is simulated: the later jobs do not age while the test (on a busy
    # machine, slowly) queues them, only the first one does, by exactly 5 steps.
    clock = Clock()
    monkeypatch.setattr(gpuqueue, "clock", clock.monotonic)
    monkeypatch.setattr(gpuqueue, "AGE_S", 0.1)
    holder, order = Holder(), []
    old = submit(job("integration"), order, "background")
    queued(1)
    clock.sleep(0.55)  # 5 steps: background -> interactive at best
    new = [submit(job(k), order, k) for k in ("eval", "sweep")]
    queued(3)
    holder.release()
    for t in (old, *new):
        t.join(10)
    assert order == ["background", "eval", "sweep"]
    assert gpuqueue.rank(job("quick"), time.monotonic()) == gpuqueue.RANK[gpuqueue.INTERACTIVE]
    stale = job("integration", gpuqueue.BACKGROUND, submitted=time.monotonic() - 100)
    assert gpuqueue.rank(stale, time.monotonic()) == gpuqueue.RANK[gpuqueue.INTERACTIVE]
    deadline = job("integration", gpuqueue.DEADLINE)
    assert gpuqueue.rank(deadline, time.monotonic()) == 0  # only a deadline job is better


def test_integration_steps_yield_to_waiting_evaluations():
    """A long sequence of acquisitions (an integration: one A/B step each) lets a job of a
    better class go between its steps; a step that runs is not interrupted."""
    order: list[str] = []
    step, waiting = threading.Event(), threading.Event()

    def integration():
        with gpuqueue.using(job("integration")):
            for i in range(3):
                with gpulock.gpu_lock():
                    order.append(f"step{i}")
                    step.set()
                    if i == 0:  # the first step runs until the evaluation waits for it
                        waiting.wait(TIMEOUT)

    background = threading.Thread(target=integration, daemon=True)
    background.start()
    assert step.wait(5)
    evaluation = submit(job("eval"), order, "eval")
    queued(1)
    waiting.set()
    background.join(10)
    evaluation.join(10)
    assert order == ["step0", "eval", "step1", "step2"]


# ------------------------------------------------------------------ lock paths


def test_reentrant_hold_skips_the_queue():
    order: list[str] = []
    inside = threading.Event()
    nested = threading.Event()
    release = threading.Event()

    def holder():
        with gpuqueue.using(job("integration")), gpulock.gpu_lock():
            inside.set()
            release.wait(10)
            with gpulock.gpu_lock() as gpu:  # nested: no turn in the queue, no deadlock
                assert gpu == 0
                nested.set()

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert inside.wait(5)
    waiter = submit(job("integration", gpuqueue.DEADLINE), order, "deadline")
    queued(1)  # a better job waits for the GPU this thread holds
    release.set()
    assert nested.wait(5)
    thread.join(5)
    waiter.join(5)
    assert order == ["deadline"]


def test_child_process_skips_the_queue(monkeypatch):
    holder, order = Holder(), []
    waiter = submit(job("quick"), order, "quick")
    queued(1)
    monkeypatch.setenv(gpulock.ENV, "1")  # a child of the holder
    done = threading.Event()

    def child():
        with gpulock.gpu_lock() as gpu:
            assert gpu == 0
            done.set()

    threading.Thread(target=child, daemon=True).start()
    assert done.wait(5)
    assert len(gpuqueue.gate().waiting) == 1 and not order  # it skipped the queue
    monkeypatch.delenv(gpulock.ENV)
    holder.release()
    waiter.join(5)
    assert order == ["quick"]


def _hold_in_process(lock_dir, events, release):
    gpulock.CACHE_DIR = Path(lock_dir)
    os.environ.pop(gpulock.ENV, None)
    os.environ[gpulock.GPUS_ENV] = "0"
    with gpulock.gpu_lock():
        events.put("took")
        release.wait(60)


def test_another_process_still_excludes(lock_dir):
    """The flock stays the outer layer: while another process holds the GPU, the queue's
    head waits on the flock and the rest in the queue; this process's order is the queue's."""
    ctx = multiprocessing.get_context("spawn")
    events, release = ctx.Queue(), ctx.Event()
    other = ctx.Process(target=_hold_in_process, args=(str(lock_dir), events, release))
    other.daemon = True
    other.start()
    order: list[str] = []
    try:
        assert events.get(timeout=60) == "took"
        first = submit(job("capture"), order, "first")
        until(lambda: bool(gpuqueue.gate().unplaced))  # admitted, waiting on the flock
        rest = [submit(job(k), order, k) for k in ("integration", "quick")]
        queued(2)
        with pytest.raises(queue.Empty):
            events.get(timeout=0.3)
        assert not order
        release.set()
        for t in (first, *rest):
            t.join(30)
    finally:
        release.set()
        other.join(30)
        if other.is_alive():
            other.kill()
    assert order == ["first", "quick", "integration"]
    assert other.exitcode == 0


# ------------------------------------------------------------------ shared jobs


def test_non_exclusive_jobs_share_within_memory(monkeypatch, lock_dir):
    """Correctness-only jobs (exclusive=False, the dev class hook) share a GPU while their
    memory estimates fit in it, never with an exclusive job; two big ones never share."""
    monkeypatch.setattr(gpulock, "_memory_gb", lambda index: 16.0)
    inside: list[str] = []
    peak: list[int] = []
    guard = threading.Lock()

    def work(j, label, hold):
        with gpuqueue.using(j), gpulock.gpu_lock():
            with guard:
                inside.append(label)
                peak.append(len(inside))
            if isinstance(hold, threading.Barrier):
                hold.wait(TIMEOUT)  # all of them on the GPU at once (broken: they were not)
            else:
                time.sleep(hold)  # a chance for another to join it (it must not)
            with guard:
                inside.remove(label)

    def together(jobs, hold=0.3):
        """Queue ``jobs`` behind an exclusive holder; the most of them on the GPU at once."""
        peak.clear()
        holder = Holder()
        threads = []
        for i, j in enumerate(jobs):
            threads.append(threading.Thread(target=work, args=(j, f"j{i}", hold), daemon=True))
            threads[-1].start()
            queued(i + 1)  # none shares the exclusive holder's GPU
        assert not inside
        holder.release()
        for t in threads:
            t.join(10)
        return max(peak)

    small = [job("dev", exclusive=False, mem_gb=1.0) for _ in range(3)]
    assert together(small, threading.Barrier(3)) == 3  # 3 GB of 16: all at once
    big = [job("dev", exclusive=False, mem_gb=8.0) for _ in range(2)]
    assert together(big) == 1  # 16 GB > 90 % of 16
    unknown = [job("dev", exclusive=False) for _ in range(2)]
    assert together(unknown) == 1  # no memory estimate: alone
    # an exclusive job waits for the shared ones, and the queue does not pass it
    order: list[str] = []
    release = threading.Event()
    first = submit(small[0], order, "shared", hold=release)
    until(lambda: bool(order))
    later = [submit(job("eval"), order, "eval")]
    queued(1)
    later.append(submit(small[1], order, "shared again"))  # it would fit, but waits its turn
    queued(2)
    release.set()
    for t in (first, *later):
        t.join(10)
    assert order == ["shared", "eval", "shared again"]
    # a shared hold is a shared flock: another process cannot take the GPU meanwhile
    held, release = threading.Event(), threading.Event()

    def shared():
        with gpuqueue.using(small[0]), gpulock.gpu_lock():
            held.set()
            release.wait(10)

    thread = threading.Thread(target=shared, daemon=True)
    thread.start()
    assert held.wait(5)
    with open(lock_dir / "gpu.lock", "w") as fh, pytest.raises(BlockingIOError):
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    release.set()
    thread.join(5)


def test_fits():
    group = [job("dev", exclusive=False, mem_gb=4.0)]
    assert gpuqueue.fits(group, job("dev", exclusive=False, mem_gb=10.0), 16.0)
    assert not gpuqueue.fits(group, job("dev", exclusive=False, mem_gb=11.0), 16.0)
    assert not gpuqueue.fits(group, job("dev", exclusive=False, mem_gb=1.0), None)
    assert not gpuqueue.fits([job("dev", exclusive=False)], job("dev", mem_gb=1.0), 16.0)


# ------------------------------------------------------------------ withdrawal, events


def test_a_cancelled_caller_withdraws_its_queued_job(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    ran = []
    holder = Holder()

    def evaluate():
        with gpulock.gpu_lock():
            ran.append(True)

    async def main():
        j = gpuqueue.Job.of(run, "eval", "t")
        task = asyncio.create_task(gpuqueue.run(j, evaluate))
        await asyncio.to_thread(queued, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return j

    j = asyncio.run(main())
    until(lambda: not gpuqueue.gate().waiting)
    holder.release()
    assert not ran and j.withdrawn
    states = [e["state"] for e in read_jsonl(run.root / gpuqueue.FILE)]
    assert states == ["queued", "withdrawn"]


def test_events_and_listeners(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    seen: list[dict] = []
    stop = gpuqueue.listen(seen.append)
    try:
        holder = Holder()
        order: list[str] = []
        j = gpuqueue.Job.of(run, "sweep", "t1")
        thread = submit(j, order, hold=0.05)
        queued(1)
        time.sleep(0.2)
        holder.release()
        thread.join(5)
        with gpulock.gpu_lock():  # untagged: class default, no record (no run)
            pass
    finally:
        stop()
    log = read_jsonl(run.root / gpuqueue.FILE)
    assert [e["state"] for e in log] == ["queued", "start", "done"]
    assert {e["event"] for e in log} == {"gpu_job"}
    assert all(e["class"] == "sweep" and e["target"] == "t1" and e["id"] == j.id for e in log)
    assert log[1]["wait_s"] >= 0.2 and log[2]["hold_s"] >= 0.05
    assert j.wait_s >= 0.2 and j.queue_s == round(j.wait_s, 1) and j.holds == 1
    tagged = [e for e in seen if e.get("id") == j.id]
    assert [e["state"] for e in tagged] == ["queued", "start", "done"]
    untagged = [e for e in seen if e["kind"] == gpuqueue.DEFAULT][-2:]  # (the holder's first)
    assert [e["state"] for e in untagged] == ["start", "done"]  # never waited: no "queued"
    assert untagged[0]["class"] == gpuqueue.DEFAULT


def test_nested_tags_account_to_the_outer_job(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    with gpuqueue.tagged("integration", run, job_class=gpuqueue.DEADLINE) as outer:
        with gpuqueue.tagged("recheck") as inner:
            assert inner.job_class == gpuqueue.DEADLINE and inner.run == run
            with gpulock.gpu_lock():
                time.sleep(0.05)
        assert gpuqueue.current() is outer
    assert gpuqueue.current() is None
    assert inner.holds == outer.holds == 1 and outer.hold_s >= 0.05
    assert outer.hold_s == inner.hold_s and outer.queue_s is not None
    assert gpuqueue.Job.of(run, "eval").queue_s is None  # never took the GPU


# ------------------------------------------------------------------ estimates


def test_estimates_from_ledger_medians(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    assert gpuqueue.estimate(run, "eval", "t") == gpuqueue.KINDS["eval"][1]  # no ledger yet
    ok = {"correct": True, "speedup": 1.0}
    for s in (4.0, 6.0, 20.0):
        ledger.record_kernel(run, "t", ok, snapshot="a.py", hypothesis="h", eval_s=s)
    ledger.record_kernel(
        run, "t", ok, snapshot="q.py", hypothesis="h", eval_s=2.0, status=ledger.QUICK_OK
    )
    ledger.record_kernel(run, "t", ok, snapshot="s.py", hypothesis="h [sweep: x]", eval_s=99.0)
    ledger.record_e2e(run, {"passed": True}, backend="transform", snapshot="x", hypothesis="h")
    ledger.record_e2e(
        run, {"passed": True}, backend="transform", snapshot="x", hypothesis="h", eval_s=50.0
    )
    ledger.record_e2e(
        run, {"passed": True}, backend="integrate", snapshot="x", hypothesis="h", eval_s=200.0
    )
    assert gpuqueue.estimate(run, "eval", "t") == 6.0
    assert gpuqueue.estimate(run, "quick", "t") == 2.0
    assert gpuqueue.estimate(run, "sweep", "t") == 99.0
    assert gpuqueue.estimate(run, "e2e") == 50.0
    assert gpuqueue.estimate(run, "integration") == 200.0
    assert gpuqueue.estimate(run, "eval", "other") == gpuqueue.KINDS["eval"][1]
    assert gpuqueue.estimate(run, "capture") == gpuqueue.KINDS["capture"][1]
    assert gpuqueue.memory_gb(run, "e2e") is None
    (run.root / "baseline.json").write_text(json.dumps({"peak_mem_gb": 8.7}))
    assert gpuqueue.memory_gb(run, "e2e") == 8.7 and gpuqueue.memory_gb(run, "eval") is None


def test_expected_wait_and_slice_seconds(monkeypatch):
    gate = gpuqueue.gate()
    assert gpuqueue.expected_wait(gpuqueue.EVAL) == 0.0
    now = time.monotonic()
    monkeypatch.setattr(gpuqueue, "clock", lambda: now)  # the test's own time does not count
    on_gpu = job("e2e", estimate=100.0, started=now - 40.0)
    gate.holders[0] = [on_gpu]
    gate.waiting = [
        job("eval", estimate=10.0, submitted=now),
        job("integration", estimate=50.0, submitted=now),
    ]
    assert gpuqueue.expected_wait(gpuqueue.EVAL) == pytest.approx(70.0, abs=1.0)
    assert gpuqueue.expected_wait(gpuqueue.BACKGROUND) == pytest.approx(120.0, abs=1.0)
    gate.gpus = 2
    assert gpuqueue.expected_wait(gpuqueue.EVAL) == pytest.approx(35.0, abs=1.0)
    arm = scheduler.Arm("t", scheduler.KERNEL, ref_ms=1.0)
    with_queue = scheduler.slice_seconds(arm)
    gate.holders.clear()
    gate.waiting = []
    assert with_queue - scheduler.slice_seconds(arm) == pytest.approx(35.0, abs=1.0)


# ------------------------------------------------------------------ session clocks


def _budget_session(run, agent_minutes):
    budget = Budget(run, agent_minutes=agent_minutes)
    timeout = budget.start_agent("kernel-t")
    clock = gpuqueue.SessionClock(
        "kernel-t#1",
        cap=budget.agent_seconds_left,
        extend=functools.partial(budget.extend_deadline, "kernel-t"),
    )
    return budget, timeout, clock


def test_queue_wait_extends_the_session_timeout_and_deadline(tmp_path, monkeypatch):
    """A session whose evaluation waits 1.5 s behind another job keeps its 0.6 s of work:
    its timeout and Budget.deadlines are pushed back by the wait (1.5 s of the queue's
    clock, simulated: exactly that much)."""
    queue_clock = Clock()
    monkeypatch.setattr(gpuqueue, "clock", queue_clock.monotonic)
    run = RunDir.create(tmp_path, "org/m")
    budget, timeout, clock = _budget_session(run, agent_minutes=0.01)
    deadline = budget.deadlines["kernel-t"]
    holder = Holder()
    holder.release_when_queued(1.5, queue_clock)

    async def session():
        async with asyncio.timeout(timeout) as timer:
            with clock.running(timer):
                when = timer.when()
                j = gpuqueue.Job.of(run, "eval", "t")
                assert j.session == "kernel-t#1" and j.clock is clock
                await gpuqueue.run(j, _hold_gpu)
                await asyncio.sleep(0)  # the clock's callbacks run on the loop
                return j, when, timer.when()

    j, before, after = asyncio.run(session())
    assert j.wait_s == clock.waited == 1.5
    assert after - before == pytest.approx(1.5)
    assert budget.deadlines["kernel-t"] - deadline == pytest.approx(1.5)
    holder.thread.join(5)


def _hold_gpu():
    with gpulock.gpu_lock():
        pass


def test_session_clock_never_passes_the_run_budget(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    budget = Budget(run, agent_minutes=10.0, max_hours=1.0, reserve=0.0)
    budget.start_agent("kernel-t")
    deadline = budget.deadlines["kernel-t"]
    budget.started -= 3600 - 30  # 30 s of the run left
    budget.extend_deadline("kernel-t", 120.0)
    assert budget.deadlines["kernel-t"] == deadline  # it was already past the run's end
    budget.started += 1200  # 20 min left
    budget.extend_deadline("kernel-t", 120.0)
    assert budget.deadlines["kernel-t"] == pytest.approx(deadline + 120.0, abs=1.0)
    budget.extend_deadline("nobody", 120.0)  # not running: nothing to extend
    assert "nobody" not in budget.deadlines

    left = {"s": 5.0}
    clock = gpuqueue.SessionClock("kernel-t#1", cap=lambda: left["s"])

    async def session():
        async with asyncio.timeout(1.0) as timer:
            with clock.running(timer):
                when = timer.when()
                clock.pause()
                await asyncio.sleep(0.05)
                held = timer.when()  # held at the run's deadline while it waits
                await asyncio.sleep(1.2)  # longer than the session's own second
                left["s"] = 0.5
                clock.resume()
                await asyncio.sleep(0)
                return when, held, timer.when(), asyncio.get_running_loop().time()

    when, held, after, now = asyncio.run(session())
    assert held == pytest.approx(when - 1.0 + 5.0, abs=0.1)
    assert after == pytest.approx(now + 0.5, abs=0.1)  # capped by the run's time left


def test_orchestrator_session_does_not_time_out_while_queued(tmp_path, monkeypatch):
    from test_budget import make_orchestrator

    from kernel_agent import orchestrator
    from kernel_agent.agent.runner import AgentResult

    calls: list[dict] = []
    orch, _ = make_orchestrator(tmp_path, monkeypatch, calls, agent_minutes=0.02)
    labels = []

    async def fake_run_agent(name, *, result=None, **kwargs):
        result = result or AgentResult(name=name)
        j = gpuqueue.Job.of(orch.run, "eval", "t1")
        labels.append(j.session)
        await gpuqueue.run(j, _hold_gpu)  # waits 2 s: longer than the 1.2 s session
        await asyncio.sleep(0.3)
        return result

    monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
    holder = Holder()
    holder.release_when_queued(2.0)
    result = asyncio.run(
        orch._agent("kernel-t1", label="kernel-t1#1", prompt="p", system_append="", mcp_tools=[])
    )
    holder.thread.join(5)
    assert not result.timed_out and labels == ["kernel-t1#1"]
    costs = json.loads((orch.run.root / "costs.json").read_text())
    assert costs["kernel-t1#1"]["gpu_wait_s"] >= 2.0 and "timed_out" not in costs["kernel-t1#1"]


def test_integration_jobs_class(tmp_path, monkeypatch):
    from test_budget import make_orchestrator

    orch, _ = make_orchestrator(tmp_path, monkeypatch, [], max_hours=1.0)
    budget = orch.budget
    with orch._gpu_job("integration") as j:
        assert j.job_class == gpuqueue.BACKGROUND and j.run == orch.run
    with orch._gpu_job("capture") as j:
        assert j.job_class == gpuqueue.BACKGROUND
    budget.final_reserve_s = scheduler.INTEGRATION_SHARE * 3600  # its estimate fills the share
    with orch._gpu_job("integration") as j:
        assert j.job_class == gpuqueue.EVAL
    with orch._gpu_job("seed") as j:
        assert j.job_class == gpuqueue.BACKGROUND
    budget.started -= 3600  # the run is in the time kept for the final integration
    with orch._gpu_job("recheck") as j:
        assert j.job_class == gpuqueue.DEADLINE


# ------------------------------------------------------------------ tools and records


def test_evaluate_candidate_records_queue_s(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    (tdir / "capture.pt").write_bytes(b"")
    (tdir / "candidates" / "v1.py").write_text("def build(r): ...\n")

    def fake_eval(capture, snap, **_):
        with gpulock.gpu_lock():
            time.sleep(0.2)
        return {"status": "ok", "correct": True, "speedup": 1.2}

    monkeypatch.setattr(tools_mod, "run_evaluation", fake_eval)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    server = {t.name: t for t in tools_mod.build_server(run, Budget(run))}
    holder = Holder()
    holder.release_when_queued(2.0)
    args = {"target_id": "t", "candidate": "candidates/v1.py", "hypothesis": "h"}
    asyncio.run(server["evaluate_candidate"].handler(args))
    holder.thread.join(5)
    row = ledger.rows(run)[-1]
    assert row["queue_s"] >= 2.0 and row["eval_s"] < 1.5  # the hold (0.2 s), without the wait
    record = read_jsonl(run.results_file("t"))[-1]
    assert record["queue_s"] == row["queue_s"]
    log = read_jsonl(run.root / gpuqueue.FILE)
    assert [e["state"] for e in log] == ["queued", "start", "done"]
    assert log[0]["class"] == gpuqueue.EVAL and log[0]["target"] == "t"


def test_status_gpu_block(tmp_path):
    run = RunDir.create(tmp_path, "org/m")
    assert gpuqueue.status_lines(run) == []
    now = time.time()
    evaluation = {"id": 1, "class": "eval", "kind": "eval", "target": "t"}
    integration = {"id": 2, "class": "background", "kind": "integration"}
    events = [
        {**evaluation, "state": "start", "wait_s": 0.0},
        {**evaluation, "state": "done", "hold_s": 30.0, "wait_s": 0.0},
        {**integration, "state": "queued"},
        {**integration, "state": "start", "wait_s": 120.0},
        {**integration, "state": "done", "hold_s": 90.0, "wait_s": 120.0},
        {"id": 3, "state": "start", "class": "e2e", "kind": "e2e", "session": "systems#4"},
        {"id": 4, "state": "queued", "class": "sweep", "kind": "sweep", "target": "t2"},
    ]
    with (run.root / gpuqueue.FILE).open("w") as fh:
        for e in events:
            fh.write(json.dumps({"ts": now - 60, "event": "gpu_job", **e}) + "\n")
    lines = gpuqueue.status_lines(run)
    assert lines[0] == "GPU queue: 3 jobs, 2.0 min on the GPU, 2.0 min waiting (max 2.0 min)"
    assert lines[1] == "  eval 1, e2e 1, background 1 (waited 2.0 min)"
    assert lines[2] == "  on the GPU (1): e2e (systems#4) 60 s"
    assert lines[3] == "  waiting (1): sweep (t2) 60 s"
