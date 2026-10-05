import os

import pytest
import torch


def pytest_configure(config):
    """With several GPUs in kernel-agent's lock pool, this process and its children use
    the first one, the GPU `gpu` tests lock (set before CUDA starts)."""
    from kernel_agent import gpulock

    pool = gpulock.pool()
    if pool.pin:
        first = pool.gpus[0].index
        os.environ.update(gpulock.pinned(first), **{gpulock.GPUS_ENV: str(first)})
        gpulock.pool.cache_clear()


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
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
