import gc
import os
import subprocess
import sys

import pytest
import torch

#: A busy loop for ``--load``: it ends when its parent (this pytest process) does.
BUSY = "import os\nparent = os.getppid()\nwhile os.getppid() == parent:\n    sum(range(100_000))\n"
_BUSY = pytest.StashKey[list[subprocess.Popen]]()


def pytest_addoption(parser):
    """The flakiness check of AGENTS.md: ``--repeat 50 --load 8`` runs every selected test
    50 times while 8 busy processes keep the CPUs loaded."""
    group = parser.getgroup("kernel-agent")
    group.addoption("--repeat", type=int, default=1, metavar="N", help="run each test N times")
    group.addoption(
        "--load", type=int, default=0, metavar="N", help="N busy processes during the session"
    )


def pytest_generate_tests(metafunc):
    if (n := metafunc.config.getoption("repeat")) > 1:
        metafunc.fixturenames.append("_repeat")
        metafunc.parametrize("_repeat", range(n), ids=lambda i: f"x{i + 1}")


@pytest.fixture
def _repeat():
    """The repetition of ``--repeat`` (parametrized)."""


def pytest_configure(config):
    """With several GPUs in kernel-agent's lock pool, this process and its children use
    the first one, the GPU `gpu` tests lock (set before CUDA starts). Under
    ``KERNEL_AGENT_EMULATE_ARCH`` (``emulate.py``) the GPU tests run as on that GPU.
    ``--load N``: the busy processes start."""
    from kernel_agent import gpulock

    pool = gpulock.pool()
    if pool.pin:
        first = pool.gpus[0].index
        os.environ.update(gpulock.pinned(first), **{gpulock.GPUS_ENV: str(first)})
        gpulock.pool.cache_clear()
    if os.environ.get("KERNEL_AGENT_EMULATE_ARCH"):  # `-m gpu` as on that older GPU (#252)
        from kernel_agent import toolchain

        toolchain.setup()  # this process emulates too, not only the evaluator's children
    n = config.getoption("load")
    config.stash[_BUSY] = [subprocess.Popen([sys.executable, "-c", BUSY]) for _ in range(n)]


def pytest_unconfigure(config):
    for proc in config.stash.get(_BUSY, []):
        proc.kill()
        proc.wait()


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def _private_library(tmp_path_factory):
    """Tests never read or write the user's cross-run kernel library (library.py)."""
    old = os.environ.get("KERNEL_AGENT_LIBRARY")
    os.environ["KERNEL_AGENT_LIBRARY"] = str(tmp_path_factory.mktemp("library"))
    yield
    if old is None:
        os.environ.pop("KERNEL_AGENT_LIBRARY", None)
    else:
        os.environ["KERNEL_AGENT_LIBRARY"] = old


@pytest.fixture(autouse=True, scope="session")
def _private_tuned_configs(tmp_path_factory):
    """Tests never read or write the user's tuned-config cache (kernels/tuned.py)."""
    old = os.environ.get("KERNEL_AGENT_TUNED_DB")
    os.environ["KERNEL_AGENT_TUNED_DB"] = str(tmp_path_factory.mktemp("tuned") / "tuned.sqlite")
    yield
    if old is None:
        os.environ.pop("KERNEL_AGENT_TUNED_DB", None)
    else:
        os.environ["KERNEL_AGENT_TUNED_DB"] = old


@pytest.fixture(autouse=True, scope="session")
def _private_docs(tmp_path_factory):
    """Tests never read or write the user's doc library (doclib, #177), and a run under
    test never builds or fetches it in the background."""
    names = ("KERNEL_AGENT_DOCS", "KERNEL_AGENT_DOCS_PREPARE")
    old = {name: os.environ.get(name) for name in names}
    os.environ["KERNEL_AGENT_DOCS"] = str(tmp_path_factory.mktemp("docs"))
    os.environ["KERNEL_AGENT_DOCS_PREPARE"] = "0"
    yield
    for name, value in old.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def _gpu_lock_for_gpu_tests(request):
    """`gpu` tests hold kernel-agent's GPU lock (of the GPU this process uses, see
    `pytest_configure`), so a plain `pytest` never benchmarks on top of a running
    optimisation (or another test session)."""
    if request.node.get_closest_marker("gpu") is None:
        yield
        return
    from kernel_agent.gpulock import ENV, gpu_lock

    with gpu_lock():
        # The test process holds the GPU for all its threads: evaluation tools run in
        # worker threads (asyncio.to_thread), which would otherwise wait for this one.
        old = os.environ.get(ENV)
        os.environ[ENV] = "1"
        try:
            yield
        finally:
            if old is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = old
            # Hand cached GPU memory back before the lock is released: other
            # processes (agents, optimisation runs) allocate as soon as they get it.
            # A loaded model (VoxCPM2: hooks and closures) is freed only by the cycle
            # collector; without it the models of several tests add up to an OOM.
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
