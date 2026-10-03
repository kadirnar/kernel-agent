"""Process-wide exclusive GPU lock so concurrent agents never benchmark at once."""

from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator

from kernel_agent.toolchain import CACHE_DIR


@contextlib.contextmanager
def gpu_lock(name: str = "gpu") -> Iterator[None]:
    if os.environ.get("KERNEL_AGENT_LOCK_HELD") == "1":
        yield
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{name}.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        os.environ["KERNEL_AGENT_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            os.environ.pop("KERNEL_AGENT_LOCK_HELD", None)
            fcntl.flock(fh, fcntl.LOCK_UN)
