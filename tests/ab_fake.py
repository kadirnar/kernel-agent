"""Test helper: ``e2e_ab`` answers for fake workers that only know ``e2e`` (issue #11)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

B_FLAGS = {"--b-kernel": "--kernel", "--b-transform": "--transform"}


def with_ab(e2e: Callable[..., dict[str, Any]], base_ms: float, rounds: int = 8) -> Callable:
    """Wrap a fake worker: ``e2e_ab`` returns B's ``e2e`` result plus ``rounds`` paired
    timings, A's from its own ``e2e`` (the unmodified model: ``base_ms``) and B's, every
    round alike. Other commands go to the fake unchanged."""

    def worker(run: Any, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        if command != "e2e_ab":
            return e2e(run, command, *args, **kwargs)
        a_args: list[str] = []
        b_args: list[str] = []
        common: list[str] = []
        it = iter(args)
        for flag in it:
            if flag in ("--kernel", "--transform"):
                a_args += [flag, next(it)]
            elif flag in B_FLAGS:
                b_args += [B_FLAGS[flag], next(it)]
            elif flag in ("--rounds", "--warmup"):
                next(it)
            else:  # --baseline-ms / --verify and their values
                common.append(flag)
        a_ms = e2e(run, "e2e", *a_args, *common)["median_ms"] if a_args else base_ms
        result = dict(e2e(run, "e2e", *b_args, *common))
        if result.get("median_ms") is not None:
            b_ms = float(result["median_ms"])
            result["times_ms"] = [b_ms] * rounds
            result["ab"] = {"mode": "paired", "a_ms": [a_ms] * rounds, "b_ms": [b_ms] * rounds}
        return result

    return worker
