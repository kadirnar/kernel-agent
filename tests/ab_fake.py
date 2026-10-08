"""Test helper: ``e2e_ab`` answers for fake workers that only know ``e2e`` (issue #11), and
the paired A/B rounds recorded in our runs' integrations (issue #190)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kernel_agent import abtest

B_FLAGS = {"--b-kernel": "--kernel", "--b-transform": "--transform"}
#: ``e2e_ab`` flags with a value that are not the e2e worker's
AB_FLAGS = ("--rounds", "--warmup", "--ab-min-win-rate", "--ab-min-gain")
#: Paired A/B rounds of the integrations of our runs (8 rounds each, see its ``about``)
RECORDED = Path(__file__).parent / "fixtures" / "ab_rounds.json"


def recorded() -> list[dict[str, Any]]:
    """The recorded paired A/Bs: ``run``, ``metric``, ``a_ms`` and ``b_ms`` per round."""
    records: list[dict[str, Any]] = json.loads(RECORDED.read_text())["records"]
    return records


def sequential_rounds(
    a_ms: list[float], b_ms: list[float], rounds: int, **rule: float
) -> tuple[list[float], list[float], dict[str, Any] | None]:
    """The rounds of ``a_ms`` / ``b_ms`` that a ``worker e2e_ab --sequential`` would have
    run (``abtest.sequential`` asked before every round after the first, as
    ``integrate/ab.alternate`` does) and its stop (None: all ``rounds`` ran)."""
    for n in range(1, rounds):
        stopped = abtest.sequential(a_ms[:n], b_ms[:n], rounds, **rule)
        if stopped is not None:
            return a_ms[:n], b_ms[:n], stopped
    return a_ms[:rounds], b_ms[:rounds], None


def with_ab(e2e: Callable[..., dict[str, Any]], base_ms: float, rounds: int = 8) -> Callable:
    """Wrap a fake worker: ``e2e_ab`` returns B's ``e2e`` result plus ``rounds`` paired
    timings, A's from its own ``e2e`` (the unmodified model: ``base_ms``) and B's, every
    round alike; with ``--sequential`` only the rounds before its stop. Other commands go to
    the fake unchanged."""

    def worker(run: Any, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        if command != "e2e_ab":
            return e2e(run, command, *args, **kwargs)
        a_args: list[str] = []
        b_args: list[str] = []
        common: list[str] = []
        values: dict[str, str] = {}
        sequential = False
        it = iter(args)
        for flag in it:
            if flag in ("--kernel", "--transform"):
                a_args += [flag, next(it)]
            elif flag in B_FLAGS:
                b_args += [B_FLAGS[flag], next(it)]
            elif flag in AB_FLAGS:
                values[flag] = next(it)
            elif flag == "--sequential":
                sequential = True
            else:  # --baseline-ms / --verify and their values
                common.append(flag)
        a_ms = e2e(run, "e2e", *a_args, *common)["median_ms"] if a_args else base_ms
        result = dict(e2e(run, "e2e", *b_args, *common))
        if result.get("median_ms") is not None:
            b_ms = float(result["median_ms"])
            a_runs, b_runs = [a_ms] * rounds, [b_ms] * rounds
            stopped = None
            if sequential:
                rule = {
                    "min_win_rate": float(values.get("--ab-min-win-rate", abtest.MIN_WIN_RATE)),
                    "min_gain": float(values.get("--ab-min-gain", abtest.MIN_GAIN)),
                }
                a_runs, b_runs, stopped = sequential_rounds(a_runs, b_runs, rounds, **rule)
            result["times_ms"] = b_runs
            result["ab"] = {"mode": "paired", "a_ms": a_runs, "b_ms": b_runs}
            if stopped is not None:
                result["ab"]["stopped"] = stopped
        return result

    return worker
