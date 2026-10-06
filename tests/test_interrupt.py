"""Ctrl-C / SIGTERM (``kernel_agent.interrupt``, issue #94): the first signal stops new
work, ends the run's processes and records what was interrupted; nothing outlives the
run; a restart continues. CPU only: the improve loop is the dry run, with real child
processes that hang until they are killed (``interrupt_driver.py``).

Every signal goes to a process this test started (the driver or its descendants).
"""

import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kernel_agent import gpulock, interrupt, ledger
from kernel_agent.workspace import RunDir, read_json

TESTS = Path(__file__).parent
SRC = TESTS.parent / "src"
LINUX = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")


@pytest.fixture(autouse=True)
def no_stop():
    """A stop requested by a test does not leak into the next one."""
    yield
    interrupt._stop.clear()
    interrupt._signal.clear()


def env(tmp_path, **extra):
    out = {k: v for k, v in os.environ.items() if not k.startswith("KERNEL_AGENT_")}
    out.update(
        PYTHONPATH=os.pathsep.join([str(SRC), str(TESTS)]),
        KERNEL_AGENT_LIBRARY=str(tmp_path / "library"),
        **extra,
    )
    return out


def driver(tmp_path, mode, *args, **extra):
    """``kernel-agent improve --dry-run`` with a hanging child at ``mode`` (a process of its
    own: the test signals it)."""
    log = (tmp_path / f"driver-{mode}.log").open("a")
    cmd = [sys.executable, str(TESTS / "interrupt_driver.py"), mode, str(tmp_path / "pids")]
    return subprocess.Popen(
        [*cmd, *args, "--dry-run"],
        env=env(tmp_path, **extra),
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=tmp_path,
    )


def started(proc, pids, timeout=60.0):
    """(child, grandchild) once the hanging child has started."""
    deadline = time.monotonic() + timeout
    while not (pids.exists() and pids.read_text().strip()):
        assert proc.poll() is None, f"the driver ended first ({proc.returncode})"
        assert time.monotonic() < deadline, "no child started"
        time.sleep(0.05)
    child, grandchild = map(int, pids.read_text().split())
    return child, grandchild


def gone(pid, timeout=5.0):
    """``pid`` has ended (a zombie counts as ended) within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return True
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def cleanup(*pids):
    """Kill what a failed test left behind (only processes this test started)."""
    for pid in pids:
        if not gone(pid, timeout=0):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def run_of(tmp_path):
    return RunDir(next((tmp_path / "runs").glob("org--model/*")))


NEW = ("org/model", "--runs-dir", "runs", "--max-slices", "3")


@LINUX
@pytest.mark.parametrize("mode", ["integration", "evaluation", "agent"])
def test_first_ctrl_c_stops_the_run_and_its_processes(tmp_path, mode):
    proc = driver(tmp_path, mode, *NEW)
    child, grandchild = started(proc, tmp_path / "pids")
    try:
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == interrupt.EXIT_CODE
        assert gone(child) and gone(grandchild)  # the whole tree, the grandchild too
    finally:
        cleanup(child, grandchild)
    run = run_of(tmp_path)
    state = read_json(run.root / "improve.json")
    stop = state["interrupted"]
    assert stop["signal"] == "SIGINT" and "finished" not in state
    if mode == "integration":  # the integration did not go on to its next A/B
        assert stop["during"].startswith("integration (")
        assert all(s["status"] == "done" for s in state["slices"])
        events = [e["event"] for e in ledger.events(run)]
        tail = events[events.index("reintegrate") :]
        assert tail == ["reintegrate", "interrupted", "phase_failed"]
    else:  # the kernel session's slice
        assert stop["during"] == f"slice {len(state['slices'])} ({state['slices'][-1]['arm']})"
        assert state["slices"][-1]["status"] == "interrupted"
        assert state["slices"][-1]["evals"] == 0  # the killed evaluation is no result
    assert [e["event"] for e in ledger.events(run)][-2:] == ["interrupted", "phase_failed"]
    rows = ledger.rows(run)
    assert not any(r["status"] in ("crash", "timeout") for r in rows)
    text = (tmp_path / f"driver-{mode}.log").read_text()
    assert "SIGINT: stopping" in text and f"`kernel-agent improve {run.root}` continues" in text

    # the same command continues the run
    again = driver(tmp_path, "none", str(run.root), "--max-slices", "2")
    assert again.wait(timeout=120) == 0
    state = read_json(run.root / "improve.json")
    assert "interrupted" not in state and state["interruptions"] == [stop]
    assert state["finished"]["reason"] == "--max-slices 2 reached"
    slices = state["slices"]
    assert [s["n"] for s in slices] == list(range(1, len(slices) + 1))
    assert "running" not in {s["status"] for s in slices}
    assert len(ledger.rows(run)) > len(rows)


@LINUX
def test_second_signal_exits_at_once(tmp_path):
    """Children that ignore SIGTERM keep the first stop waiting for its grace period; a
    second Ctrl-C kills them and exits at once."""
    proc = driver(tmp_path, "integration", *NEW, KA_TEST_TRAP_SIGTERM="1")
    child, grandchild = started(proc, tmp_path / "pids")
    try:
        proc.send_signal(signal.SIGINT)
        time.sleep(1.0)
        assert proc.poll() is None and not gone(child, timeout=0)  # SIGTERM is ignored
        start = time.monotonic()
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == interrupt.EXIT_CODE
        assert time.monotonic() - start < 5.0
        assert gone(child) and gone(grandchild)
    finally:
        cleanup(child, grandchild)


@LINUX
def test_sigterm_and_sigkill_after_the_grace_period(tmp_path):
    proc = driver(tmp_path, "integration", *NEW, KA_TEST_TRAP_SIGTERM="1", KA_TEST_GRACE="1")
    child, grandchild = started(proc, tmp_path / "pids")
    try:
        start = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == interrupt.EXIT_CODE
        assert 0.9 < time.monotonic() - start < 15.0  # SIGKILL after the 1 s grace
        assert gone(child) and gone(grandchild)
    finally:
        cleanup(child, grandchild)
    state = read_json(run_of(tmp_path).root / "improve.json")
    assert state["interrupted"]["signal"] == "SIGTERM"


# ------------------------------------------------------------------ the pieces


@pytest.fixture
def lock_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gpulock, "CACHE_DIR", tmp_path)
    for name in (gpulock.ENV, gpulock.GPUS_ENV, gpulock.INDEX_ENV, "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gpulock, "_nvidia_smi", lambda: [gpulock.GPU(0, "fake", "GPU-0")])
    gpulock.pool.cache_clear()
    yield tmp_path
    gpulock.pool.cache_clear()


def test_a_stopping_run_takes_no_gpu_and_starts_no_child(lock_dir):
    with gpulock.gpu_lock():
        env = gpulock.child_env()
    assert env[interrupt.PARENT_ENV] == str(os.getpid())
    interrupt._stop.set()
    with pytest.raises(interrupt.Interrupted), gpulock.gpu_lock():
        pass
    with pytest.raises(interrupt.Interrupted):
        gpulock.child_env()


def test_work_under_the_lock_when_the_stop_comes_is_no_result(lock_dir):
    """A measurement whose process the stop killed returns a crash: the lock turns it
    into ``Interrupted`` as it is let go, so nobody records it."""
    result = None
    with pytest.raises(interrupt.Interrupted), gpulock.gpu_lock(), gpulock.gpu_lock():
        interrupt._stop.set()  # (the inner, re-entrant one raises already)
        result = {"status": "crash"}
    assert result is not None
    interrupt._stop.clear()
    with gpulock.gpu_lock():  # the lock was released
        pass


def test_a_waiter_stops_waiting_and_leaves_the_lock_free(lock_dir):
    foreign = open(lock_dir / "gpu.lock", "w")  # another process holds the GPU  # noqa: SIM115
    fcntl.flock(foreign, fcntl.LOCK_EX)
    errors = []

    def wait():
        try:
            with gpulock.gpu_lock():
                errors.append("got the lock")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    time.sleep(0.5)
    assert thread.is_alive()  # waiting
    interrupt._stop.set()
    thread.join(timeout=3 * gpulock.WAIT_S + 1)
    assert not thread.is_alive() and isinstance(errors[0], interrupt.Interrupted)
    foreign.close()  # the abandoned wait gets the lock now and lets it go at once
    probe = open(lock_dir / "gpu.lock", "w")  # noqa: SIM115
    deadline = time.monotonic() + 5
    while True:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            assert time.monotonic() < deadline, "the abandoned wait kept the lock"
            time.sleep(0.05)
    probe.close()


def test_a_waiter_gets_the_lock_when_it_is_freed(lock_dir):
    foreign = open(lock_dir / "gpu.lock", "w")  # noqa: SIM115
    fcntl.flock(foreign, fcntl.LOCK_EX)
    got = []

    def hold():
        with gpulock.gpu_lock() as gpu:
            got.append(gpu)
            probe = open(lock_dir / "gpu.lock", "w")  # noqa: SIM115
            try:  # held through the caller's descriptor once the helper thread is done
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got.append("not exclusive")
            except BlockingIOError:
                pass
            probe.close()

    waiter = threading.Thread(target=hold, daemon=True)
    waiter.start()
    time.sleep(0.3)
    assert not got
    foreign.close()
    waiter.join(timeout=5)
    assert got == [0]


@LINUX
def test_terminate_ends_a_process_tree_and_escalates(tmp_path):
    code = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(600)\n"
    )
    before = interrupt.descendants()
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    grandchild = int(child.stdout.readline())
    try:
        mine = interrupt.descendants() - before
        assert {pid for pid, _ in mine} == {child.pid, grandchild}
        start = time.monotonic()
        interrupt.terminate(mine, grace=0.5)  # SIGTERM is ignored: SIGKILL after 0.5 s
        assert child.wait(timeout=5) == -signal.SIGKILL
        assert 0.4 < time.monotonic() - start < 5
        assert gone(grandchild)
        assert not any(interrupt.alive(p) for p in mine)
        interrupt.terminate(mine, grace=0.5)  # ended processes: nothing is sent
    finally:
        cleanup(child.pid, grandchild)


@LINUX
def test_a_child_dies_with_the_process_that_started_it(tmp_path):
    """``child_env`` children set PR_SET_PDEATHSIG on ``import kernel_agent``: killing their
    parent outright (SIGKILL, no cleanup at all) ends them too. A process that only
    inherited the variable is left alone."""
    parent_code = (
        "import os, subprocess, sys, time\n"
        "from kernel_agent import gpulock\n"
        "code = 'import kernel_agent, time; time.sleep(600)'\n"
        "child = subprocess.Popen([sys.executable, '-c', code], env=gpulock.child_env())\n"
        "plain = subprocess.Popen([sys.executable, '-c', code])\n"
        "print(child.pid, plain.pid, flush=True)\n"
        "time.sleep(600)\n"
    )
    env_ = env(tmp_path, **{interrupt.PARENT_ENV: "1"})  # inherited, naming another parent
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True, env=env_
    )
    child, plain = map(int, parent.stdout.readline().split())
    try:
        time.sleep(1.0)  # both have imported kernel_agent
        assert not gone(child, timeout=0) and not gone(plain, timeout=0)
        parent.kill()
        parent.wait()
        assert gone(child)
        assert not gone(plain, timeout=0.5)  # it was not started with child_env
    finally:
        cleanup(child, plain)


def test_run_returns_the_result_and_restores_the_handlers():
    async def work():
        return 42

    before = signal.getsignal(signal.SIGINT)
    assert interrupt.run(work()) == 42
    assert signal.getsignal(signal.SIGINT) is before
    assert not interrupt.requested()
