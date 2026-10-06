import fcntl
import json
import multiprocessing
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import pytest

from kernel_agent import gpulock, interrupt

NVIDIA_SMI = gpulock._nvidia_smi  # before the fixture fakes it
POOL_ENV = (
    gpulock.ENV,
    gpulock.GPUS_ENV,
    gpulock.INDEX_ENV,
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
)


def parent():
    """What every child environment gains: its parent, to die with it (interrupt.py)."""
    return {interrupt.PARENT_ENV: str(os.getpid())}


def fake_gpus(monkeypatch, n, name="NVIDIA Fake GPU"):
    """nvidia-smi lists ``n`` GPUs (the pool is found again)."""
    gpus = [gpulock.GPU(i, name, f"GPU-{i}{i}{i}-fake") for i in range(n)]
    monkeypatch.setattr(gpulock, "_nvidia_smi", lambda: list(gpus))
    gpulock.pool.cache_clear()
    return gpus


@pytest.fixture(autouse=True)
def lock_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gpulock, "CACHE_DIR", tmp_path)
    for name in POOL_ENV:
        monkeypatch.delenv(name, raising=False)
    fake_gpus(monkeypatch, 1)
    yield tmp_path
    gpulock.pool.cache_clear()


@pytest.fixture
def foreign(lock_dir):
    """Lock files held by "another process" (a separate open file, as an old
    kernel-agent or the `flock` wrapper would hold them)."""
    held = {}

    def hold(name):
        fh = held[name] = open(lock_dir / name, "w")  # noqa: SIM115
        fcntl.flock(fh, fcntl.LOCK_EX)

    def release(name):
        held.pop(name).close()

    yield hold, release
    for fh in held.values():
        fh.close()


def _in_thread(fn):
    """Run ``fn`` in a thread; returns (thread, list that gets its result)."""
    out = []
    thread = threading.Thread(target=lambda: out.append(fn()), daemon=True)
    thread.start()
    return thread, out


def _hold_gpu():
    with gpulock.gpu_lock() as gpu:
        return gpu


def test_threads_of_one_process_exclude_each_other():
    inside = 0
    overlap = False
    guard = threading.Lock()

    def work():
        nonlocal inside, overlap
        with gpulock.gpu_lock():
            with guard:
                inside += 1
                overlap |= inside > 1
            time.sleep(0.05)
            with guard:
                inside -= 1

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not overlap


def test_reentrant_in_one_thread_and_child_env(lock_dir):
    assert gpulock.ENV not in gpulock.child_env()
    with gpulock.gpu_lock() as gpu:
        assert gpu == 0
        with gpulock.gpu_lock() as again:  # nested: must not deadlock
            assert again == 0
            assert gpulock.child_env()[gpulock.ENV] == "1"
        # another process cannot take the file lock while we hold it
        with open(lock_dir / "gpu.lock", "w") as fh, pytest.raises(BlockingIOError):
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert gpulock.ENV not in gpulock.child_env()

    assert gpulock.ENV not in os.environ  # never leaks into this process


def test_child_process_flag_skips_locking(monkeypatch, lock_dir):
    monkeypatch.setenv(gpulock.ENV, "1")
    monkeypatch.setattr(gpulock, "_nvidia_smi", lambda: pytest.fail("a child never looks"))
    gpulock.pool.cache_clear()
    with open(lock_dir / "gpu.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # held by the "parent"
        with gpulock.gpu_lock() as gpu:  # the child must not wait for it
            assert gpu == 0
            assert gpulock.child_env() == dict(os.environ) | parent()  # its children: the same
    monkeypatch.setenv(gpulock.INDEX_ENV, "2")  # a pinned child: the GPU its parent locked
    with gpulock.gpu_lock() as gpu:
        assert gpu == 2
        assert gpulock.child_env()[gpulock.ENV] == "1"


# ------------------------------------------------------------------ discovery


def _smi(monkeypatch, stdout, returncode=0):
    """The real discovery, parsing a fake nvidia-smi."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    monkeypatch.setattr(gpulock.subprocess, "run", run)
    monkeypatch.setattr(gpulock, "_nvidia_smi", NVIDIA_SMI)
    return calls


SMI = (
    "0, NVIDIA A100-SXM4-80GB, GPU-aaaa-0000\n"
    "1, NVIDIA A100-SXM4-80GB, GPU-bbbb-1111\n"
    "2, NVIDIA H100 80GB HBM3, GPU-cccc-2222\n"
)


def test_pool_from_nvidia_smi(monkeypatch):
    def pool(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        gpulock.pool.cache_clear()
        try:
            found = gpulock.pool()
        finally:
            for k in env:
                monkeypatch.delenv(k)
        return [g.index for g in found.gpus], found.pin

    calls = _smi(monkeypatch, SMI)
    assert pool() == ([0, 1, 2], True)
    assert calls[0][:2] == ["nvidia-smi", "--query-gpu=index,name,uuid"]
    assert gpulock.pool().gpus[2] == gpulock.GPU(2, "NVIDIA H100 80GB HBM3", "GPU-cccc-2222")
    # an inherited CUDA_VISIBLE_DEVICES: indices or UUID prefixes, in its order; like
    # CUDA, the list ends at the first entry that names no GPU
    assert pool(CUDA_VISIBLE_DEVICES="2,GPU-aaaa") == ([2, 0], True)
    assert pool(CUDA_VISIBLE_DEVICES="1") == ([1], False)  # one GPU: children unpinned
    assert pool(CUDA_VISIBLE_DEVICES="1,-1,0") == ([1], False)
    assert pool(CUDA_VISIBLE_DEVICES="") == ([0], False)  # none visible: GPU 0's gpu.lock
    assert pool(CUDA_VISIBLE_DEVICES="MIG-1234") == ([0], False)
    # KERNEL_AGENT_GPUS replaces both, without nvidia-smi
    calls.clear()
    assert pool(KERNEL_AGENT_GPUS="2, 0", CUDA_VISIBLE_DEVICES="1") == ([2, 0], True)
    assert pool(KERNEL_AGENT_GPUS="3") == ([3], True)
    assert not calls
    with pytest.raises(ValueError, match="KERNEL_AGENT_GPUS"):
        pool(KERNEL_AGENT_GPUS="0,gpu1")
    # no nvidia-smi, or a failing one: GPU 0 alone, as before the pool
    _smi(monkeypatch, "", returncode=9)
    assert pool() == ([0], False)

    def missing(cmd, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpulock.subprocess, "run", missing)
    assert pool() == ([0], False)
    gpulock.pool.cache_clear()


def test_describe_names_the_lock_files(monkeypatch):
    fake_gpus(monkeypatch, 2)
    line = gpulock.describe()
    assert line == "GPU locks: 0 NVIDIA Fake GPU (gpu.lock), 1 NVIDIA Fake GPU (gpu1.lock)"
    gpus = [gpulock.GPU(0, "A100"), gpulock.GPU(1, "H100")]
    monkeypatch.setattr(gpulock, "_nvidia_smi", lambda: gpus)
    gpulock.pool.cache_clear()
    assert "mixed GPU models" in gpulock.describe()


# ------------------------------------------------------------------ the pool


def test_single_gpu_keeps_the_plain_lock_file_and_environment(lock_dir, foreign, monkeypatch):
    hold, release = foreign
    hold("gpu.lock")  # a kernel-agent from before the pool, on the same GPU
    thread, got = _in_thread(_hold_gpu)
    thread.join(0.3)
    assert thread.is_alive() and not got  # waits for it
    release("gpu.lock")
    thread.join(5)
    assert got == [0]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    expected = dict(os.environ) | {gpulock.ENV: "1"} | parent()
    with gpulock.gpu_lock():
        assert gpulock.child_env() == expected  # only the flag, as before the pool
    assert sorted(p.name for p in lock_dir.iterdir()) == ["gpu.lock"]


def test_one_visible_gpu_of_several_uses_its_own_lock_file(lock_dir, monkeypatch):
    fake_gpus(monkeypatch, 4)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-222")
    gpulock.pool.cache_clear()
    with gpulock.gpu_lock() as gpu:
        assert gpu == 2
        assert gpulock.child_env()["CUDA_VISIBLE_DEVICES"] == "GPU-222"  # inherited
        assert gpulock.INDEX_ENV not in gpulock.child_env()
    assert sorted(p.name for p in lock_dir.iterdir()) == ["gpu2.lock"]


def test_free_gpu_first_reentrant_keeps_it_and_child_env_pins_it(lock_dir, foreign, monkeypatch):
    fake_gpus(monkeypatch, 3)
    hold, release = foreign
    hold("gpu.lock")  # GPU 0 is busy in another process
    with gpulock.gpu_lock() as gpu:
        assert gpu == 1  # the first free GPU, without waiting
        release("gpu.lock")
        with gpulock.gpu_lock() as again:
            assert again == 1  # re-entrant: the same GPU, although GPU 0 is free now
            env = gpulock.child_env()
        assert env[gpulock.ENV] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"  # nvidia-smi's numbering
        assert env[gpulock.INDEX_ENV] == "1"
        with open(lock_dir / "gpu1.lock", "w") as fh, pytest.raises(BlockingIOError):
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # another thread of this process gets GPU 0 meanwhile
        thread, got = _in_thread(_hold_gpu)
        thread.join(5)
        assert got == [0]
    env = gpulock.child_env()
    assert gpulock.ENV not in env and "CUDA_VISIBLE_DEVICES" not in env
    assert "CUDA_VISIBLE_DEVICES" not in os.environ  # never leaks into this process


def test_threads_spread_over_the_pool_never_two_on_one_gpu(monkeypatch):
    fake_gpus(monkeypatch, 2)
    inside = {0: 0, 1: 0}
    peak = {0: 0, 1: 0}
    guard = threading.Lock()

    def work():
        with gpulock.gpu_lock() as gpu:
            with guard:
                inside[gpu] += 1
                peak[gpu] = max(peak[gpu], inside[gpu])
            time.sleep(0.05)
            with guard:
                inside[gpu] -= 1

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == {0: 1, 1: 1}  # both GPUs used, never shared


def test_waiters_wait_on_different_gpus(lock_dir, foreign, monkeypatch):
    """Every GPU busy: each waiter blocks on the GPU with the fewest waiters."""
    fake_gpus(monkeypatch, 3)
    hold, release = foreign
    files = ["gpu.lock", "gpu1.lock", "gpu2.lock"]
    for name in files:
        hold(name)
    waiters = [_in_thread(_hold_gpu) for _ in range(3)]
    deadline = time.monotonic() + 5
    while sum(gpulock._waiting[f] for f in files) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [gpulock._waiting[f] for f in files] == [1, 1, 1]
    assert not any(got for _, got in waiters)
    for name in files:
        release(name)
    for thread, _ in waiters:
        thread.join(5)
    assert sorted(got[0] for _, got in waiters) == [0, 1, 2]
    assert [gpulock._waiting[f] for f in files] == [0, 0, 0]


def _hold_in_process(lock_dir, gpus, events, release):
    """Child process: take a GPU of the pool ``gpus`` and hold it until ``release``."""
    gpulock.CACHE_DIR = Path(lock_dir)
    os.environ.pop(gpulock.ENV, None)
    os.environ[gpulock.GPUS_ENV] = gpus
    events.put(("waiting", None))
    with gpulock.gpu_lock() as gpu:
        events.put(("took", gpu))
        release.wait(60)


def test_processes_exclude_each_other(lock_dir, monkeypatch):
    monkeypatch.setenv(gpulock.GPUS_ENV, "0,1")
    gpulock.pool.cache_clear()
    ctx = multiprocessing.get_context("spawn")
    events, release = ctx.Queue(), ctx.Event()
    args = (str(lock_dir), "0,1", events, release)
    first = ctx.Process(target=_hold_in_process, args=args, daemon=True)
    second = ctx.Process(target=_hold_in_process, args=args, daemon=True)
    first.start()
    try:
        assert events.get(timeout=60) == ("waiting", None)
        assert events.get(timeout=60) == ("took", 0)
        with gpulock.gpu_lock() as gpu:
            assert gpu == 1  # GPU 0 is the other process's
            with open(lock_dir / "gpu.lock", "w") as fh, pytest.raises(BlockingIOError):
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            second.start()
            assert events.get(timeout=60) == ("waiting", None)
            with pytest.raises(queue.Empty):
                events.get(timeout=0.5)  # both GPUs taken: it waits
            release.set()  # the first process lets GPU 0 go, this one keeps GPU 1
            assert events.get(timeout=30) == ("took", 0)
    finally:
        release.set()
        for proc in (first, second):
            proc.join(30)
            if proc.is_alive():
                proc.kill()
    assert first.exitcode == 0 and second.exitcode == 0


# ------------------------------------------------------------------ call sites


def test_evaluation_and_worker_run_on_the_locked_gpu(lock_dir, foreign, monkeypatch, tmp_path):
    from kernel_agent import worker
    from kernel_agent.kernels import evaluate
    from kernel_agent.workspace import RunDir

    fake_gpus(monkeypatch, 2)
    hold, _ = foreign
    hold("gpu.lock")
    envs = []

    def run(cmd, *, env, **kwargs):
        envs.append(env)
        marker = "@@KA_RESULT@@" if "evaluate" in cmd[2] else worker.MARKER
        nonce = (kwargs.get("input") or "").strip()  # run_evaluation's result-line nonce
        marker += f"{nonce}@@" if nonce else ""
        return subprocess.CompletedProcess(cmd, 0, marker + json.dumps({"status": "ok"}), "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(evaluate, "ensure_peaks", lambda: None)
    result = evaluate.run_evaluation(tmp_path / "c.pt", tmp_path / "k.py")
    version = evaluate.evaluator_version()  # stamped outside the candidate's process
    assert result == {"status": "ok", "gpu_index": 1, "evaluator_version": version}
    assert worker.call_worker(RunDir(tmp_path), "e2e") == {"status": "ok", "gpu_index": 1}
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["1", "1"]
    assert all(e[gpulock.ENV] == "1" for e in envs)


def test_parallel_evaluations_run_one_per_gpu(monkeypatch, tmp_path):
    """--parallel N with N GPUs: N evaluations at once, never two on one GPU."""
    from concurrent.futures import ThreadPoolExecutor

    from kernel_agent.kernels import evaluate

    fake_gpus(monkeypatch, 2)
    busy = {"0": 0, "1": 0}
    peak = {"0": 0, "1": 0, "all": 0}
    guard = threading.Lock()

    def run(cmd, *, env, **kwargs):
        gpu = env["CUDA_VISIBLE_DEVICES"]
        with guard:
            busy[gpu] += 1
            peak[gpu] = max(peak[gpu], busy[gpu])
            peak["all"] = max(peak["all"], sum(busy.values()))
        time.sleep(0.2)
        with guard:
            busy[gpu] -= 1
        tag = kwargs["input"].strip() + "@@"  # run_evaluation's result-line nonce
        return subprocess.CompletedProcess(cmd, 0, "@@KA_RESULT@@" + tag + json.dumps({}), "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(evaluate, "ensure_peaks", lambda: None)
    with ThreadPoolExecutor(4) as agents:  # the agents' tool calls run in worker threads
        results = list(
            agents.map(
                lambda _: evaluate.run_evaluation(tmp_path / "c.pt", tmp_path / "k.py"), range(4)
            )
        )
    assert peak == {"0": 1, "1": 1, "all": 2}
    assert sorted(r["gpu_index"] for r in results) == [0, 0, 1, 1]


def test_one_gpu_results_say_gpu_0_and_children_see_the_usual_environment(monkeypatch, tmp_path):
    from kernel_agent.kernels import evaluate

    envs = []

    def run(cmd, *, env, **kwargs):
        envs.append(env)
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(evaluate, "ensure_peaks", lambda: None)
    result = evaluate.run_evaluation(tmp_path / "c.pt", tmp_path / "k.py")
    assert result["status"] == "crash" and result["gpu_index"] == 0
    assert envs == [dict(os.environ) | {gpulock.ENV: "1"} | parent()]


@pytest.mark.gpu
def test_gpu_tests_hold_the_lock():
    # conftest wraps every `gpu` test in gpu_lock() on the real lock file (taken before
    # this module's lock_dir fixture redirects CACHE_DIR): nobody else can take it now.
    from kernel_agent.toolchain import CACHE_DIR

    with open(CACHE_DIR / "gpu.lock", "w") as fh, pytest.raises(BlockingIOError):
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
