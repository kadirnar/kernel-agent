import fcntl
import threading
import time

import pytest

from kernel_agent import gpulock


@pytest.fixture(autouse=True)
def lock_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gpulock, "CACHE_DIR", tmp_path)
    monkeypatch.delenv(gpulock.ENV, raising=False)
    return tmp_path


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
    with gpulock.gpu_lock():
        with gpulock.gpu_lock():  # nested: must not deadlock
            assert gpulock.child_env()[gpulock.ENV] == "1"
        # another process cannot take the file lock while we hold it
        with open(lock_dir / "gpu.lock", "w") as fh, pytest.raises(BlockingIOError):
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert gpulock.ENV not in gpulock.child_env()
    import os

    assert gpulock.ENV not in os.environ  # never leaks into this process


def test_child_process_flag_skips_locking(monkeypatch, lock_dir):
    monkeypatch.setenv(gpulock.ENV, "1")
    with open(lock_dir / "gpu.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # held by the "parent"
        with gpulock.gpu_lock():  # the child must not wait for it
            pass


@pytest.mark.gpu
def test_gpu_tests_hold_the_lock():
    # conftest wraps every `gpu` test in gpu_lock() on the real lock file (taken before
    # this module's lock_dir fixture redirects CACHE_DIR): nobody else can take it now.
    from kernel_agent.toolchain import CACHE_DIR

    with open(CACHE_DIR / "gpu.lock", "w") as fh, pytest.raises(BlockingIOError):
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
