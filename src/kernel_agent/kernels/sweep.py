"""Parameter sweeps: many configs of one candidate under one GPU-lock acquisition.

InferenceBench measured a plain hyperparameter sweep (11.5x) beating agents (8.1x)
that spent their budget re-measuring one config at a time; KernelFoundry tunes
template parameters apart from the LLM (docs/RESEARCH.md).  A candidate exposes
its parameters as keyword arguments of ``build`` with defaults::

    def build(reference, BLOCK=1024, num_warps=4): ...

and :func:`run_sweep` (the ``sweep_candidate`` tool, ``kernel-agent eval --sweep``)
tries a list of configs while it holds one GPU of the pool:

1. **Check** (one subprocess, :func:`sweep`): every config goes through the
   evaluator's quick tier (:func:`kernels.evaluate.evaluate` with ``quick``: build,
   the smallest and the largest captured case with aliasing, fallback detection,
   perturbed re-verification and the integrity snapshot).  A failing config is
   rejected with its error.  The capture and the candidate module are loaded once,
   and the configs share the reference's weights: a config whose ``build()`` or
   calls modify them is an ``integrity_violation`` (and the weights are restored).
2. **Time** (same subprocess, CUDA only): the passing configs against the reference
   with the evaluator's timing (:func:`kernels.bench.time_call`: 60 ms rounds,
   input sets rotated, median of :data:`ROUNDS` rounds), interleaved: each round
   times the reference and then every config, in an order rotated per round, on
   every timed case.  One timed call of each config's first round runs on redrawn
   inputs and is checked against the reference (``incorrect_timed_output``).  The
   table is sorted by weighted speedup (calls per run x time) and has the
   speedup and ``pct_of_sol`` of every case.  **Racing** (``--early-stop on``, issue #190,
   :func:`kernels.early.race`): after each round but the last, the configs beyond the
   noise of the leader (more than 50 % slower after the first round, certainly slower
   after the second) drop out, at most half of those still timed per round (successive
   halving); their rows keep the speedups of the rounds they had and say why
   (``raced``).
3. **Evaluate** the best config with the full evaluator (:func:`run_evaluation`
   in a fresh process: every stage and anti-gaming guard, the checks outside the
   candidate's process included), with its config bound into the file
   (:func:`bind_config`); that file is what gets recorded, ranked and integrated.
   When no config passed, the first failure stands for the sweep.

**Search** (issue #229, :mod:`kernels.search`): instead of a list, a *space* (the values
of every parameter), constraints and a strategy. The subprocess then repeats steps 1 and
2 on batches of :data:`kernels.search.BATCH` configs the search asks for (a batch timed
with :data:`SEARCH_TARGET_MS` rounds, racing on) and tells it each config's weighted
speedup, until the time bound (no limit on the number of configs). The
:data:`FINALISTS` best are then timed against each other with the evaluator's rounds
(``final`` in their rows), and the best of them goes to step 3. Each config's row has the
registers it spilled and the shared memory it uses (the compiler's stats); a spill or an
out-of-resources failure prunes the configs that need at least as much. Every measured
point goes to the tuned-config cache (:meth:`kernels.tuned.TunedConfigs.add_points`),
and a later search of the same GPU, versions, op (the captured module's class and the
parameters) and shape bucket starts from its best points.

A crash or a broken CUDA context (an illegal memory access) in one config ends
the subprocess; it starts again without that config (at most :data:`MAX_RUNS`
processes; a search continues from the points measured so far).  The configs share
:data:`TIMEOUT_FACTOR` x the evaluation timeout (a search: until then, however many
configs that is): configs not checked by then are ``skipped``, and timing stops after the last
whole round that fits.  The full evaluation has the usual timeout, so a sweep
takes at most ``(TIMEOUT_FACTOR + 1) x`` the evaluation timeout.  The tool counts
a sweep as one evaluation of the budget.

Run as a subprocess (results arrive on lines tagged with a nonce from stdin)::

    python -m kernel_agent.kernels.sweep CAPTURE CANDIDATE --configs FILE [--deadline-s S]
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import keyword
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from kernel_agent.gpulock import child_env, gpu_lock
from kernel_agent.kernels.evaluate import _first_error, _short_tb, run_evaluation
from kernel_agent.kernels.roofline import ensure_peaks
from kernel_agent.truth import TamperError

DEFAULT_MAX_CONFIGS = 32
MAX_CONFIGS = 64  # hard cap of one sweep, whatever max_configs asks for
TIMEOUT_FACTOR = 3.0  # the configs' share: this x the evaluation timeout
MAX_RUNS = 3  # sweep subprocesses (a crash or a broken CUDA context starts another one)
MIN_RUN_S = 10.0  # do not start a sweep subprocess with less time left
SOFT_SHARE = 0.8  # a subprocess starts no new work after this share of its time limit
CHECK_SHARE = 0.6  # ... and no new config check after this share (the rest is for timing)
ROUNDS = 3  # timing rounds, as kernels.bench.compare_timing
TARGET_MS = 60.0  # timed per round and function, as kernels.bench.compare_timing
SEARCH_TARGET_MS = 20.0  # a search's exploration rounds; its finalists get TARGET_MS
FINALISTS = 4  # a search's best configs, timed against each other at the end
TABLE_ROWS = 24  # a search's table as text and in the tool's result: the best rows
ERROR_CHARS = 600  # the end of a rejected config's error (where the exception is)
MARKER = "@@KA_SWEEP@@"
BROKEN_CONTEXT = 3  # exit code of a subprocess whose CUDA context a config broke
SKIPPED = "not run: the sweep's time limit was reached (sweep fewer configs at a time)"


class BrokenContext(Exception):
    """A config left the CUDA context unusable; nothing else can run in this process."""

    def __init__(self, index: int, error: str) -> None:
        super().__init__(error)
        self.index = index
        self.error = error


# ------------------------------------------------------------------ configs


def _literal(value: Any) -> bool:
    if value is None or isinstance(value, bool | int | str):
        return True
    return isinstance(value, float) and math.isfinite(value)


def configs_from(value: Any, max_configs: Any = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Validated configs and notes.  ``value``: a list of dicts of ``build`` keyword
    arguments, or a dict of lists (every combination), possibly as JSON text.
    Duplicates are dropped; beyond ``max_configs`` (default :data:`DEFAULT_MAX_CONFIGS`,
    at most :data:`MAX_CONFIGS`) the rest is.  Raises ``ValueError``."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"configs is not valid JSON: {exc}") from None
    if isinstance(value, dict):
        if not value or not all(isinstance(v, list) and v for v in value.values()):
            raise ValueError(
                'a grid is a dict of non-empty lists, e.g. {"BLOCK": [512, 1024], '
                '"num_warps": [4, 8]}'
            )
        names = list(value)
        value = [
            dict(zip(names, combo, strict=True)) for combo in itertools.product(*value.values())
        ]
    if not isinstance(value, list) or not value:
        raise ValueError(
            "configs is a non-empty list of dicts of build() keyword arguments, e.g. "
            '[{"BLOCK": 512, "num_warps": 4}, {"BLOCK": 1024, "num_warps": 8}] (or a dict '
            "of lists: every combination)"
        )
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, config in enumerate(value):
        if not isinstance(config, dict):
            raise ValueError(f"config {i} is not a dict of keyword arguments: {config!r}"[:300])
        for name, arg in config.items():
            if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"config {i}: {name!r} is not a keyword argument name")
            if not _literal(arg):
                raise ValueError(
                    f"config {i}: {name}={arg!r} is not a number, string, bool or null"[:300]
                )
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(dict(config))
    notes = []
    if len(configs) < len(value):
        notes.append(f"{len(value) - len(configs)} duplicate config(s) dropped")
    try:
        limit = DEFAULT_MAX_CONFIGS if max_configs is None else int(max_configs)
    except (TypeError, ValueError):
        raise ValueError(f"max_configs is a number, not {max_configs!r}") from None
    if limit > MAX_CONFIGS:
        notes.append(f"max_configs is at most {MAX_CONFIGS}")
    limit = min(max(limit, 1), MAX_CONFIGS)
    if len(configs) > limit:
        notes.append(f"swept the first {limit} of {len(configs)} configs (max_configs)")
        configs = configs[:limit]
    return configs, notes


def label(config: dict[str, Any]) -> str:
    """``BLOCK=1024, num_warps=4`` (``default`` for the empty config; a long value, such as
    tuned Helion configs, shortened: the record has it whole)."""

    def short(value: Any) -> str:
        text = repr(value)
        return text if len(text) <= 80 else text[:77] + "..."

    return ", ".join(f"{k}={short(v)}" for k, v in config.items()) or "default"


def bind_config(source: str, config: dict[str, Any]) -> str:
    """``source`` with ``config`` bound as the defaults of ``build`` (keyword arguments
    still override them): the file a sweep evaluates and records for its best config,
    so integration and export build what was measured."""
    if not config:
        return source
    return (
        source.rstrip()
        + "\n\n\n# Bound by kernel-agent's sweep_candidate: the best config of the sweep.\n"
        + f"_KA_SWEEP_CONFIG = {config!r}\n"
        + "_ka_unbound_build = build\n\n\n"
        + "def build(reference, **config):\n"
        + "    return _ka_unbound_build(reference, **{**_KA_SWEEP_CONFIG, **config})\n"
    )


# ------------------------------------------------------------------ the table


def rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Passing configs by weighted speedup (a search's finalists first: their speedups come
    from the same rounds; untimed ones after them, in the given order), then the rejected
    ones, then the skipped ones."""

    def key(row: dict[str, Any]) -> tuple[int, int, float, int]:
        if row.get("correct"):
            speedup = row.get("speedup")
            if speedup is None:
                return (1, 0, 0, row["index"])
            return (0, 0 if row.get("final") else 1, -float(speedup), row["index"])
        return (3 if row.get("status") == "skipped" else 2, 0, 0, row["index"])

    return sorted(rows, key=key)


def _tail(text: Any) -> str:
    text = str(text).strip()
    return text if len(text) <= ERROR_CHARS else "..." + text[-ERROR_CHARS:]


def _row(result: dict[str, Any]) -> dict[str, Any]:
    """Table row of a config's quick check (:func:`kernels.evaluate.evaluate`)."""
    row: dict[str, Any] = {
        "status": str(result.get("status") or "error"),
        "correct": bool(result.get("correct")),
    }
    if result.get("compile_s") is not None:
        row["compile_s"] = result["compile_s"]
    if row["correct"]:
        return row
    error = result.get("error")
    if not error:  # incorrect: the failures are in the case reports
        bad = next((c for c in result.get("cases") or [] if not c.get("ok")), None)
        error = (
            f"case {bad['signature']}: {_first_error(bad['failures'])}" if bad else row["status"]
        )
    row["error"] = _tail(error)
    for key in ("stage", "failed_case"):
        if key in result:
            row[key] = result[key]
    return row


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """A table row as the agent sees it (per-case values in the order of ``cases``)."""
    out: dict[str, Any] = {"config": row["config"], "correct": row["correct"]}
    if not row["correct"]:
        out["status"] = row["status"]
        out["error"] = row.get("error")
        return out
    for key in ("speedup", "pct_of_sol", "timing_spread", "compile_s"):
        if row.get(key) is not None:
            out[key] = row[key]
    if row.get("cases"):
        out["speedup_per_case"] = [c.get("speedup") for c in row["cases"]]
        if any(c.get("pct_of_sol") is not None for c in row["cases"]):
            out["pct_of_sol_per_case"] = [c.get("pct_of_sol") for c in row["cases"]]
    for flag in ("suspicious_faster_than_sol", "sol_unreliable", "final"):
        if row.get(flag):
            out[flag] = True
    if raced := row.get("raced"):  # racing (kernels/early.py): the rounds it had
        out["raced"] = raced["why"]
    for key in ("spills", "shared_bytes"):  # a search's compiler stats of the config
        if row.get(key):
            out[key] = row[key]
    return out


def slim_table(table: list[dict[str, Any]], keep: int = MAX_CONFIGS) -> list[dict[str, Any]]:
    """``table`` as a record keeps it: the first ``keep`` rows whole, the rest (a search's
    hundreds) as their config, status and speedup."""
    slim = ("index", "config", "status", "correct", "speedup", "final", "spills")
    return table[:keep] + [{k: r[k] for k in slim if k in r} for r in table[keep:]]


def format_table(data: dict[str, Any]) -> str:
    """The sweep of :func:`run_sweep` as text (``kernel-agent eval --sweep``)."""
    info = data["sweep"]
    head = (
        f"sweep: {info['configs']} configs, {info['passed']} passed, {info['failed']} "
        f"rejected, {info['skipped']} skipped, {info.get('rounds') or 0} timing rounds, "
        f"{info['seconds']} s" + (f", {info['raced']} raced out" if info.get("raced") else "")
    )
    lines = [head]
    if found := info.get("search"):
        pruned = ", ".join(f"{n} by {kind}" for kind, n in (found.get("pruned") or {}).items())
        lines.append(
            f"  search: {found['strategy']} (seed {found['seed']}), {found['measured']} timed "
            f"of {found['valid'] if found.get('valid') is not None else '?'} valid configs "
            f"({found['space']} in the space" + (f"; pruned {pruned}" if pruned else "") + ")"
        )
    if tuned := info.get("helion"):
        what = tuned.get("error") or (
            f"tuned {', '.join(tuned.get('kernels') or [])} in {tuned.get('seconds')} s"
        )
        lines.append(f"  helion: {what} (on case {tuned.get('case')})")
    for case in info.get("cases") or []:
        ref = f", reference {case['ref_ms']} ms" if case.get("ref_ms") is not None else ""
        lines.append(f"  case {case['signature']} x{case['calls_per_run']}{ref}")
    lines.append(f"  {'speedup':>8}  {'%SOL':>6}  {'per case':<24}  config")
    shown = info["table"][:TABLE_ROWS] if info.get("search") else info["table"]
    for row in shown:
        if row["correct"]:
            speedup = f"{row['speedup']:.3f}x" if row.get("speedup") is not None else "untimed"
            sol = f"{row['pct_of_sol']:.1f}" if row.get("pct_of_sol") is not None else "-"
            cases = " ".join(f"{c.get('speedup')}x" for c in row.get("cases") or [])
            raced = f" (raced out after round {row['raced']['after']})" if row.get("raced") else ""
            raced += " (finalist)" if row.get("final") else ""
            raced += f" ({row['spills']} spilled)" if row.get("spills") else ""
            lines.append(f"  {speedup:>8}  {sol:>6}  {cases:<24}  {label(row['config'])}{raced}")
        else:
            error = (row.get("error") or "").strip().splitlines() or [""]
            lines.append(f"  {row['status']:>16}  {label(row['config'])}: {error[-1][:120]}")
    if len(shown) < len(info["table"]):
        lines.append(f"  ... {len(info['table']) - len(shown)} more rows (the JSON has them all)")
    evaluation = data["evaluation"]
    lines.append(
        f"best: {label(data['config'])} -> full evaluation {evaluation.get('status')}"
        + (f", {evaluation['speedup']}x" if evaluation.get("speedup") is not None else "")
    )
    return "\n".join(lines)


# ------------------------------------------------------------------ in the subprocess


def _ignore(event: dict[str, Any]) -> None:
    pass


def _restore_weights(named: list[tuple[str, Any]], backup: list[Any]) -> list[str]:
    """Names of the shared weights a config changed (restored from ``backup``)."""
    import torch

    changed = []
    with torch.no_grad():
        for (name, weight), saved in zip(named, backup, strict=True):
            same = (weight.shape, weight.dtype, weight.device) == (
                saved.shape,
                saved.dtype,
                saved.device,
            )
            if same and weight.is_floating_point():
                same = bool(torch.allclose(weight, saved, rtol=0, atol=0, equal_nan=True))
            elif same:
                same = torch.equal(weight, saved)
            if not same:
                weight.data = saved.clone()
                changed.append(name)
    return changed


def _check_context(index: int) -> None:
    """Raise :class:`BrokenContext` when the CUDA context no longer works."""
    import torch

    try:
        torch.cuda.synchronize()
        torch.ones(1, device="cuda").add_(1).item()
    except Exception as exc:
        raise BrokenContext(index, _tail(f"{type(exc).__name__}: {exc}")) from None


def _time_configs(
    reference: Any,
    cases: list[dict[str, Any]],
    timed: list[int],
    passing: list[tuple[dict[str, Any], tuple[Any, ...]]],
    *,
    timer: Callable[..., dict[str, Any]],
    check_output: Callable[[Any, dict[str, Any]], dict[str, Any]] | None,
    l2_flush: bool,
    deadline: float | None,
    emit: Callable[[dict[str, Any]], None],
    replay: Any,
    race: bool = False,
    target_ms: float = TARGET_MS,
) -> tuple[int, dict[int, float]]:
    """Interleaved timing rounds of the passing configs (``(row, (candidate, the reference
    copy it was built from))``); fills their rows (speedup per case and weighted). Every
    call runs from its case's module state (``replay``, profiling/state.py). With ``race``
    the configs beyond the noise of the leader drop out after each round (``raced`` in their
    row: :func:`kernels.early.race`; their speedups are of the rounds they had). Returns the
    number of whole rounds and the reference's time per timed case (``target_ms`` per round
    and function)."""
    from kernel_agent.kernels import early
    from kernel_agent.kernels.bench import median_round

    ref_rounds: dict[int, list[dict[str, Any]]] = {ci: [] for ci in timed}
    new_rounds: dict[int, dict[int, list[dict[str, Any]]]] = {
        row["index"]: {ci: [] for ci in timed} for row, _ in passing
    }
    done, round_s = 0, 0.0
    for r in range(ROUNDS):
        if r and deadline is not None and time.monotonic() + round_s > deadline:
            break
        began = time.monotonic()
        shift = r % len(passing)
        order = passing[shift:] + passing[:shift]
        for ci in timed:
            case = cases[ci]
            args, kwargs = case["args"], case["kwargs"]
            ref_fn = replay.call(case, reference)
            ref_rounds[ci].append(
                timer(ref_fn, args, kwargs, l2_flush=l2_flush, target_ms=target_ms)
            )
            for row, holders in order:
                if not row["correct"] or row.get("raced"):
                    continue
                emit({"event": "running", "index": row["index"]})
                try:
                    t = timer(
                        replay.call(case, *holders),
                        args,
                        kwargs,
                        l2_flush=l2_flush,
                        target_ms=target_ms,
                        keep=r == 0,
                    )
                    kept = t.pop("kept", None)
                    checked = check_output(ref_fn, kept) if kept and check_output else {}
                except Exception:
                    row.update(
                        status="runtime_error",
                        correct=False,
                        error=_tail(_short_tb()),
                        failed_case=ci,
                    )
                    if check_output is not None:  # CUDA
                        _check_context(row["index"])
                    continue
                if checked.get("failures"):
                    row.update(
                        status="incorrect_timed_output",
                        correct=False,
                        stage="timed_output",
                        failed_case=ci,
                        error=f"case {ci} ({case['signature']}): the output of timed call "
                        f"#{checked['iteration']} differs from the reference on the same inputs "
                        f"({_first_error(checked['failures'])})",
                    )
                    continue
                new_rounds[row["index"]][ci].append(t)
        done += 1
        round_s = time.monotonic() - began
        if race:  # successive halving: the configs clearly slower than the leader drop out
            refs = {ci: [x["median_ms"] for x in ref_rounds[ci]] for ci in timed}
            still = {
                row["index"]: [
                    early.Timed(
                        cases[ci]["count"],
                        refs[ci],
                        [x["median_ms"] for x in new_rounds[row["index"]][ci]],
                    )
                    for ci in timed
                ]
                for row, _ in passing
                if row["correct"] and not row.get("raced")
            }
            rows = {row["index"]: row for row, _ in passing}
            for index, why in early.race(still, done, ROUNDS).items():
                rows[index]["raced"] = why
    emit({"event": "running", "index": None})

    ref = {ci: median_round(ref_rounds[ci]) for ci in timed}
    for row, _ in passing:
        if not row["correct"]:
            continue
        reports, ref_total, new_total = [], 0.0, 0.0
        for ci in timed:
            ref_t, new_t = ref[ci], median_round(new_rounds[row["index"]][ci])
            n = cases[ci]["count"]
            ref_total += n * ref_t["median_ms"]
            new_total += n * new_t["median_ms"]
            reports.append(
                {
                    "case": ci,
                    "calls_per_run": n,
                    "ref_ms": round(ref_t["median_ms"], 5),
                    "new_ms": round(new_t["median_ms"], 5),
                    "speedup": round(ref_t["median_ms"] / max(new_t["median_ms"], 1e-9), 3),
                    "timing_spread": round(max(ref_t["spread"], new_t["spread"]), 3),
                }
            )
        row.update(
            speedup=round(ref_total / max(new_total, 1e-9), 3),
            ref_ms_weighted=round(ref_total, 4),
            new_ms_weighted=round(new_total, 4),
            timing_spread=max(c["timing_spread"] for c in reports),
            cases=reports,
        )
    return done, {ci: round(ref[ci]["median_ms"], 5) for ci in timed}


def _speed_of_light(
    rows: list[dict[str, Any]],
    reference: Any,
    cases: list[dict[str, Any]],
    timed: list[int],
    l2_flush: bool,
    precision: str | None = None,
    replay: Any = None,
) -> str | None:
    """``pct_of_sol`` of every timed row (:func:`kernels.roofline.apply_sol`, the work
    counted once per case from its module state (``replay``), weights at the capture's
    ``precision``); a note when it cannot be computed."""
    from kernel_agent.kernels.roofline import apply_sol, count_case, current_peaks

    peaks = current_peaks()
    if not peaks:
        return "GPU peaks not measured yet (`kernel-agent doctor` measures them)"
    try:
        costs = []
        for ci in timed:
            if replay is not None:
                replay.restore(cases[ci], reference)
            costs.append(
                count_case(
                    reference,
                    cases[ci]["args"],
                    cases[ci]["kwargs"],
                    method=cases[ci]["method"],
                    precision=precision,
                )
            )
        for row in rows:
            if row.get("correct") and row.get("cases"):
                apply_sol(row, costs, peaks, hot_l2=not l2_flush)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


@contextlib.contextmanager
def _tier(capture: dict[str, Any]) -> Iterator[None]:
    """The capture's tolerance tier while configs are timed: the timed-output check compares
    with it, and :func:`kernels.evaluate.evaluate` leaves the exact tier behind when it
    returns (a near-lossless target's sweep failed every config with the exact bounds)."""
    from kernel_agent.kernels import compare

    compare.TIER = compare.tier_of(capture)
    try:
        yield
    finally:
        compare.TIER = compare.EXACT_TIER


def _compiled(module: Any) -> Counter[tuple[tuple[str, Any], ...]]:
    """The Triton kernels compiled so far from ``module`` (:func:`kernels.ncu.triton_stats`)."""
    from kernel_agent.kernels.ncu import triton_stats

    try:
        rows = triton_stats(dict(vars(module))) if module is not None else []
    except Exception:  # compiler stats never fail a sweep
        return Counter()
    return Counter(tuple(sorted(r.items())) for r in rows)


def _stats(before: Counter[tuple[tuple[str, Any], ...]], module: Any) -> dict[str, Any]:
    """Spilled registers and shared memory of the kernels a config compiled (``before``: the
    module's kernels before its check)."""
    new = [dict(k) for k in (_compiled(module) - before).elements()]
    out: dict[str, Any] = {}
    if spills := sum(int(k.get("spills") or 0) for k in new):
        out["spills"] = spills
    if shared := max((int(k.get("shared_bytes") or 0) for k in new), default=0):
        out["shared_bytes"] = shared
    return out


def _search(
    engine: Any,
    check: Callable[..., tuple[dict[str, Any], tuple[Any, ...] | None]],
    time_batch: Callable[[list[tuple[dict[str, Any], tuple[Any, ...]]], float | None], float],
    *,
    timed_cases: int,
    start: int,
    history: list[dict[str, Any]],
    deadline: float | None,
    emit: Callable[[dict[str, Any]], None],
    remember: Callable[[list[dict[str, Any]]], None],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], tuple[Any, ...]]]]:
    """A search's batches (module docstring, **Search**): ask ``engine``
    (:class:`kernels.search.Search`) for configs, ``check`` them, ``time_batch`` the passing
    ones (it returns the seconds of one timed call, 0 when nothing is timed) and tell their
    speedups, until the ``deadline`` leaves just the time of the final rounds. ``history``:
    the rows of an earlier process of this sweep (told first, run again only as finalists).
    Returns this process's rows and the finalists with their built candidates (``final``)."""
    for row in history:
        score = row.get("speedup") if row.get("correct") else None
        engine.tell(row["config"], score, error=row.get("error"), spills=row.get("spills"))
    rows: list[dict[str, Any]] = []
    best: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
    index, batch_s, measure_s, check_s, checked = start, 0.0, 0.0, 0.0, 0

    def reserve() -> float:  # the final rounds (TARGET_MS) and rebuilding earlier finalists
        rounds = ROUNDS * (FINALISTS + 1) * timed_cases * measure_s * TARGET_MS / SEARCH_TARGET_MS
        return rounds + (FINALISTS * check_s if history else 0.0)

    def by_speed(found: list[tuple[dict[str, Any], tuple[Any, ...]]]) -> list[Any]:
        return sorted(found, key=lambda p: -float(p[0]["speedup"]))[:FINALISTS]

    while True:
        began = time.monotonic()
        if deadline is not None and began + batch_s + reserve() > deadline:
            break
        configs = engine.ask()
        if not configs:  # the space is exhausted
            break
        passing, batch = [], []
        for config in configs:
            if deadline is not None and time.monotonic() + reserve() > deadline:
                break
            t0 = time.monotonic()
            row, holders = check(index, config, stats=True)
            check_s = (check_s * checked + time.monotonic() - t0) / (checked + 1)
            checked += 1
            index += 1
            batch.append(row)
            if holders is not None:
                passing.append((row, holders))
            else:
                engine.tell(config, None, error=row.get("error"), spills=row.get("spills"))
        rows += batch
        if passing:
            until = None if deadline is None else deadline - reserve()
            measure_s = time_batch(passing, until) or measure_s
        for row, _ in passing:
            score = row.get("speedup") if row["correct"] else None
            engine.tell(row["config"], score, error=row.get("error"), spills=row.get("spills"))
            emit({"event": "row", **row})  # with its speedup: a restarted process keeps it
        remember(batch)
        best = by_speed(
            best + [p for p in passing if p[0]["correct"] and p[0].get("speedup") is not None]
        )
        batch_s = time.monotonic() - began

    # an earlier process's best configs compete again: built here for the final rounds
    have = {row["index"] for row, _ in best}
    earlier = [
        r
        for r in history
        if r.get("correct") and r.get("speedup") is not None and r["index"] not in have
    ]
    for old in sorted(earlier, key=lambda r: -float(r["speedup"])):
        if len(best) >= FINALISTS and float(old["speedup"]) <= float(best[-1][0]["speedup"]):
            break
        row, holders = check(old["index"], old["config"], stats=True)
        rows.append(row)
        if holders is not None:
            row["speedup"] = old["speedup"]
            best = by_speed([*best, (row, holders)])
    for row, _ in best:
        row["search_speedup"] = row.get("speedup")  # of its batch, before the final rounds
        row.pop("raced", None)
        row["final"] = True
    return rows, best


#: A Helion sweep's subprocess: checks and timings use each kernel's declared or default
#: config (no tuning inside an evaluation), and no progress bar on the result stream
_HELION_ENV = {"HELION_AUTOTUNE_EFFORT": "none", "HELION_AUTOTUNE_PROGRESS_BAR": "0"}


@contextlib.contextmanager
def _defaults(env: dict[str, str]) -> Iterator[None]:
    """``env``'s variables that are not set, while the block runs."""
    added = [k for k in env if k not in os.environ]
    os.environ.update({k: env[k] for k in added})
    try:
        yield
    finally:
        for k in added:
            os.environ.pop(k, None)


def _elements(case: dict[str, Any]) -> int:
    import torch

    values = [*case["args"], *case["kwargs"].values()]
    return sum(v.numel() for v in values if isinstance(v, torch.Tensor))


def _helion(
    check: Callable[..., tuple[dict[str, Any], tuple[Any, ...] | None]],
    session: dict[str, Any],
    cases: list[dict[str, Any]],
    timed: list[int],
    *,
    start: int,
    seed: int,
    deadline: float | None,
    warp_specialize: bool,
    seeds: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], tuple[Any, ...]]], dict[str, Any]]:
    """A Helion candidate's sweep (``strategy="helion"``): its default config checked, Helion's
    autotuner on its kernels (:func:`kernels.helion_tune.autotune`) with the arguments of
    the case with the most work (calls per run x elements) and most of the time left, then
    the tuned configs (``helion_configs``) checked. Returns the rows, the passing configs
    (timed next) and what the tuning did."""
    import torch

    from kernel_agent.kernels import helion_tune

    rows: list[dict[str, Any]] = []
    passing: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
    row, built = check(start, {})
    rows.append(row)
    if built is None:
        return rows, passing, {"error": "the default config failed its check: fix that first"}
    passing.append((row, built))
    pool = timed or list(range(len(cases)))
    ci = max(pool, key=lambda i: (max(cases[i]["count"], 1) * _elements(cases[i]), -i))
    case, replay, holders = cases[ci], session["replay"], built
    seconds = None if deadline is None else max(10.0, 0.7 * (deadline - time.monotonic()))

    def call() -> None:
        replay.call(case, *holders)(*case["args"], **case["kwargs"])

    on = [dict(vars(session["module"])), dict(getattr(built[0], "__dict__", {}))]
    try:
        with torch.no_grad():
            found = helion_tune.autotune(
                helion_tune.kernels(*on),
                call,
                seconds=seconds,
                seed=seed,
                warp_specialize=warp_specialize,
                seeds=seeds,
            )
    except Exception:
        found = {"error": _tail(_short_tb())}
    info: dict[str, Any] = {k: found[k] for k in ("kernels", "seconds", "error") if k in found}
    info["case"] = case["signature"]
    if "configs" in found:
        row, tuned = check(start + 1, {helion_tune.KEYWORD: found["configs"]})
        rows.append(row)
        if tuned is not None:
            passing.append((row, tuned))
    return rows, passing, info


class _TunedPoints:
    """A search's points in the tuned-config cache (:mod:`kernels.tuned`): op ``sweep:<the
    captured module's class>:<the parameters>``, the bucket of the timed cases' signatures,
    the candidate's backend (whose library versions an entry is valid for). The cache never
    fails a sweep: its errors leave the search without warm starts or unrecorded."""

    def __init__(
        self,
        reference: Any,
        cases: list[dict[str, Any]],
        timed: list[int],
        candidate_path: Path,
        names: list[str],
    ) -> None:
        from kernel_agent.backends import classify
        from kernel_agent.kernels.tuned import signature_bucket

        self.op = f"sweep:{type(reference).__name__}:{','.join(names)}"
        self.bucket = signature_bucket(cases[i]["signature"] for i in (timed or range(len(cases))))
        try:
            self.backend = classify(candidate_path.read_text())
        except (OSError, UnicodeDecodeError):
            self.backend = "torch"

    def warm(self) -> list[dict[str, Any]]:
        from kernel_agent.kernels import tuned

        try:
            found = tuned.default().points(self.op, self.bucket, backend=self.backend)
        except Exception:
            return []
        return [p["config"] for p in found if p.get("score") is not None]

    def helion_seeds(self) -> dict[str, list[dict[str, Any]]]:
        """Earlier tuned Helion configs by kernel name, the best first (Helion's seeds)."""
        from kernel_agent.kernels.helion_tune import KEYWORD

        out: dict[str, list[dict[str, Any]]] = {}
        for config in self.warm():
            for name, found in (config.get(KEYWORD) or {}).items():
                out.setdefault(name, []).append(found)
        return out

    def remember(self, rows: list[dict[str, Any]]) -> None:
        from kernel_agent.kernels import tuned

        points = [
            {
                "config": r["config"],
                "score": r.get("speedup") if r.get("correct") else None,
                "status": r.get("status"),
            }
            for r in rows
            if r.get("status") != "skipped" and (r.get("speedup") is not None or not r["correct"])
        ]
        with contextlib.suppress(Exception):
            tuned.default().add_points(self.op, self.bucket, points, backend=self.backend)


def sweep(
    capture_path: Path,
    candidate_path: Path,
    configs: list[dict[str, Any]],
    *,
    indices: list[int] | None = None,
    device: str | None = None,
    l2_flush: bool = False,
    capture_sha256: str | None = None,
    deadline: float | None = None,
    emit: Callable[[dict[str, Any]], None] = _ignore,
    timer: Callable[..., dict[str, Any]] | None = None,
    race: bool = True,
    search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Steps 1 and 2 of the module docstring, in this process: the sorted ``table``.

    ``indices``: the configs' numbers in the whole sweep (rows carry them as
    ``index``); ``deadline`` (``time.monotonic()``): no new work after it; ``emit``
    gets progress events (``running`` before a config's check or timing, ``row``
    after its check).  Timing needs CUDA, or ``timer``: a stand-in for
    :func:`kernels.bench.time_call` (tests). ``race``: racing (``--early-stop``).
    ``search``: a search instead of ``configs`` (:func:`kernels.search.spec_from`, and
    ``history``: the rows of an earlier process of this sweep, ``start``: its first index,
    ``arch``: GPU facts instead of this GPU's (tests), ``warm``: False for no warm starts)."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.startswith("cuda")
    if cuda:
        from kernel_agent import toolchain

        toolchain.setup()
    # Imported before the candidate: kernels.bench binds the timer when it is imported.
    from kernel_agent.kernels import bench
    from kernel_agent.kernels.evaluate import capture_precision, evaluate, quick_cases
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.state import Replay

    start = time.monotonic()
    indices = list(range(len(configs))) if indices is None else indices
    try:
        capture = load_capture(capture_path, device=device, sha256=capture_sha256)
    except TamperError as exc:
        return {"status": "tampered", "error": str(exc), "table": []}
    if capture.get("inputs_only"):
        error = f"{capture_path} is an inputs-only capture (no reference outputs)"
        return {"status": "error", "error": error, "table": []}
    reference = capture["module"].eval()
    cases = capture["cases"]
    for case in cases:
        case.setdefault("method", "forward")
    named = list(reference.named_parameters())
    backup = [w.detach().clone() for _, w in named]
    replay = Replay(capture, reference)  # each case's module state, before any call
    session: dict[str, Any] = {
        "capture": capture,
        "memo": {id(w): w for _, w in named},
        "replay": replay,
    }
    checks_until = None if deadline is None else start + CHECK_SHARE * (deadline - start)

    def check(
        index: int, config: dict[str, Any], stats: bool = False
    ) -> tuple[dict[str, Any], tuple[Any, ...] | None]:
        """One config through the quick tier: its row, and what it built when it passed
        (``stats``: the spills and shared memory of the kernels it compiled)."""
        row: dict[str, Any] = {"index": index, "config": config}
        emit({"event": "running", "index": index, "config": config})
        before = _compiled(session.get("module")) if stats else Counter()
        session.pop("candidate", None)
        try:
            result = evaluate(
                capture_path,
                candidate_path,
                device=device,
                quick=True,
                config=config,
                session=session,
            )
        except Exception:
            result = {"status": "harness_error", "correct": False, "error": _short_tb()}
        if changed := _restore_weights(named, backup):
            result = {
                "status": "integrity_violation",
                "correct": False,
                "stage": "integrity",
                "error": f"modified the reference's weights in place ({', '.join(changed[:4])}); "
                "the configs of a sweep share them, so build() and the calls must leave them "
                "alone (pack into new tensors instead)",
            }
        row.update(_row(result))
        if stats:
            row.update(_stats(before, session.get("module")))
        candidate = session.pop("candidate", None)
        holders = session.pop("holders", None) or (candidate,)
        passed = row["correct"] and candidate is not None
        if not passed and cuda:
            _check_context(index)
        emit({"event": "row", **row})
        return row, (holders if passed else None)

    timed = [i for i, c in enumerate(cases) if c["count"]]
    time_fn = timer or (bench.time_call if cuda else None)
    check_output = bench.check_timed_output if cuda else None
    out: dict[str, Any] = {
        "status": "ok",
        "device": device,
        "quick_cases": [cases[i]["signature"] for i in quick_cases(cases)],
        "cases": [
            {"case": i, "signature": cases[i]["signature"], "calls_per_run": cases[i]["count"]}
            for i in timed
        ],
    }

    # 1. every config through the quick tier (a search: batches, checked and timed)
    rows: list[dict[str, Any]] = []
    passing: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
    points: _TunedPoints | None = None
    from kernel_agent.kernels import search as search_mod

    arch = search_mod.Arch()  # this GPU's facts (a search's pruning), a spec's in tests
    if search is not None:
        arch = search_mod.Arch.from_dict(search["arch"]) if search.get("arch") else arch
        if not search.get("arch") and cuda:
            arch = search_mod.Arch.detect()
    if search is not None and search.get("strategy") == search_mod.HELION:
        from kernel_agent.kernels.helion_tune import KEYWORD

        points = _TunedPoints(reference, cases, timed, candidate_path, [KEYWORD])
        with _defaults(_HELION_ENV):  # read when the candidate's kernels are declared
            rows, passing, out["helion"] = _helion(
                check,
                session,
                cases,
                timed,
                start=int(search.get("start") or 0),
                seed=int(search.get("seed") or 0),
                deadline=deadline,
                warp_specialize=search_mod.arch_reason({"warp_specialize": 1}, arch) is None,
                seeds=points.helion_seeds() if search.get("warm", True) else {},
            )
    elif search is not None:
        names = search_mod.parse_space(search["space"]).names
        points = _TunedPoints(reference, cases, timed, candidate_path, names)
        warm = points.warm() if search.get("warm", True) else []
        engine = search_mod.Search.from_spec(search, arch=arch, warm=warm)

        def time_batch(
            batch: list[tuple[dict[str, Any], tuple[Any, ...]]], until: float | None
        ) -> float:
            if not timed or time_fn is None:
                return 0.0
            began = time.monotonic()
            with _tier(capture):
                done, _ = _time_configs(
                    reference,
                    cases,
                    timed,
                    batch,
                    timer=time_fn,
                    check_output=check_output,
                    l2_flush=l2_flush,
                    deadline=until,
                    emit=emit,
                    replay=replay,
                    race=race,
                    target_ms=SEARCH_TARGET_MS,
                )
            return (time.monotonic() - began) / max(done * (len(batch) + 1) * len(timed), 1)

        if cuda and timed:
            bench.warm_gpu()
        rows, passing = _search(
            engine,
            check,
            time_batch,
            timed_cases=len(timed),
            start=int(search.get("start") or 0),
            history=list(search.get("history") or []),
            deadline=deadline,
            emit=emit,
            remember=points.remember,
        )
        out["search"] = engine.summary()
    else:
        for index, config in zip(indices, configs, strict=True):
            if checks_until is not None and time.monotonic() > checks_until:
                skipped = {"status": "skipped", "correct": False, "error": SKIPPED}
                rows.append({"index": index, "config": config, **skipped})
                continue
            row, holders = check(index, config)
            rows.append(row)
            if holders is not None:
                passing.append((row, holders))

    # 2. the passing configs (a search's finalists), timed against the reference
    if passing and timed and time_fn is not None:
        if cuda:
            bench.warm_gpu()
        with _tier(capture):
            out["rounds"], ref_ms = _time_configs(
                reference,
                cases,
                timed,
                passing,
                timer=time_fn,
                check_output=check_output,
                l2_flush=l2_flush,
                deadline=deadline,
                emit=emit,
                replay=replay,
                race=race,
            )
        if raced := sum(bool(row.get("raced")) for row in rows):
            out["raced"] = raced
        for case in out["cases"]:
            case["ref_ms"] = ref_ms[case["case"]]
        precision = capture_precision(capture)  # reduced-precision weights: their own bytes
        if cuda and (
            note := _speed_of_light(rows, reference, cases, timed, l2_flush, precision, replay)
        ):
            out["sol_note"] = note
        if points is not None:  # the finalists' speedups of the evaluator's rounds
            points.remember([row for row, _ in passing])
    elif passing or any(r.get("correct") for r in rows):
        out["timing"] = "skipped: no CUDA device" if timed else "skipped: no timed case"
    out["table"] = rank(rows)
    out["seconds"] = round(time.monotonic() - start, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    # Bound before the candidate is imported: it cannot redirect the result lines.
    write, flush, dumps = sys.stdout.write, sys.stdout.flush, json.dumps
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--configs", type=Path, required=True, help="JSON: configs, indices")
    parser.add_argument("--deadline-s", type=float, help="start no new work after this")
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--capture-sha256", help="refuse a capture without this digest")
    parser.add_argument("--nonce-stdin", action="store_true", help="tag result lines")
    parser.add_argument("--no-race", action="store_true", help="time every config every round")
    ns = parser.parse_args(argv)
    tag = (sys.stdin.readline().strip() + "@@") if ns.nonce_stdin else ""

    def emit(event: dict[str, Any]) -> None:
        write("\n" + MARKER + tag + dumps(event, default=str) + "\n")  # after any partial line
        flush()

    spec = json.loads(ns.configs.read_text())
    deadline = time.monotonic() + ns.deadline_s if ns.deadline_s else None
    try:
        result = sweep(
            ns.capture,
            ns.candidate,
            spec["configs"],
            indices=spec.get("indices"),
            l2_flush=ns.l2_flush,
            capture_sha256=ns.capture_sha256,
            deadline=deadline,
            emit=emit,
            race=not ns.no_race,
            search=spec.get("search"),
        )
    except BrokenContext as exc:
        emit({"event": "broken_context", "index": exc.index, "error": exc.error})
        os._exit(BROKEN_CONTEXT)  # no teardown in a broken CUDA context
    except Exception:
        result = {"status": "harness_error", "error": _short_tb(), "table": []}
    emit({"event": "result", **result})
    return 0


# ------------------------------------------------------------------ in the orchestrator


def _text(value: str | bytes | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def _spawn(
    capture_path: Path,
    swept: Path,
    configs: list[dict[str, Any]],
    indices: list[int],
    *,
    soft_s: float,
    hard_s: float,
    l2_flush: bool,
    capture_sha256: str | None,
    race: bool = True,
    search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One sweep subprocess: its checked ``rows``, final ``result`` (None if it did not
    finish) and the ``culprit``: the config that ran when it died (``culprit_config``)."""
    spec = swept.parent / "configs.json"
    payload: dict[str, Any] = {"configs": configs, "indices": indices}
    if search is not None:
        payload["search"] = search
    spec.write_text(json.dumps(payload))
    nonce = secrets.token_hex(16)
    cmd = [
        sys.executable,
        "-m",
        "kernel_agent.kernels.sweep",
        str(capture_path),
        str(swept),
        "--configs",
        str(spec),
        "--deadline-s",
        f"{soft_s:.1f}",
        "--nonce-stdin",
    ]
    if l2_flush:
        cmd.append("--l2-flush")
    if capture_sha256:
        cmd += ["--capture-sha256", capture_sha256]
    if not race:
        cmd.append("--no-race")
    timed_out, code = False, None
    try:
        proc = subprocess.run(
            cmd, input=nonce + "\n", capture_output=True, text=True, timeout=hard_s, env=child_env()
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, timed_out = _text(exc.stdout), _text(exc.stderr), True
    marker = f"{MARKER}{nonce}@@"
    rows: dict[int, dict[str, Any]] = {}
    result = broken = running = None
    configs_of: dict[int, Any] = {}  # a search's configs by index (a list's are known)
    for line in stdout.splitlines():
        if not line.startswith(marker):
            continue
        with contextlib.suppress(json.JSONDecodeError):
            event = json.loads(line[len(marker) :])
            kind = event.pop("event", None)
            if kind == "running":
                running = event.get("index")
                if "config" in event:
                    configs_of[running] = event["config"]
            elif kind == "row":
                rows[event["index"]] = event
            elif kind == "result":
                result = event
            elif kind == "broken_context":
                broken, running = event, event.get("index")
    if broken is not None:
        error = f"broke the CUDA context: {broken.get('error')}"
    elif timed_out:
        error = f"the sweep process exceeded its time limit ({hard_s:.0f} s) in this config"
    else:
        error = (
            f"the sweep process died (exit code {code}) in this config: {_tail(stderr or stdout)}"
        )
    culprit_config = None
    if running is not None:
        culprit_config = configs_of.get(running, rows.get(running, {}).get("config"))
    return {
        "rows": rows,
        "result": result,
        "culprit": running,
        "culprit_config": culprit_config,
        "timed_out": timed_out,
        "error": error,
    }


def _check_and_time(
    capture_path: Path,
    swept: Path,
    configs: list[dict[str, Any]],
    *,
    budget_s: float,
    l2_flush: bool,
    capture_sha256: str | None,
    race: bool = True,
    search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Steps 1 and 2 in subprocesses (another one without a config that crashed); every
    config gets a row. A ``search`` (:func:`kernels.search.spec_from`) instead of
    ``configs``: a restarted process continues from every row so far (``history``)."""
    start = time.monotonic()
    failed: dict[int, dict[str, Any]] = {}
    checked: dict[int, dict[str, Any]] = {}
    seen: dict[int, dict[str, Any]] = {}  # a search: the rows of every process so far
    data: dict[str, Any] | None = None
    runs = 0
    while runs < MAX_RUNS:
        left = budget_s - (time.monotonic() - start)
        todo = [i for i in range(len(configs)) if i not in failed]
        if (search is None and not todo) or left < MIN_RUN_S:
            break
        runs += 1
        resumed: dict[str, Any] | None = None
        if search is not None:  # everything measured so far, and the next free index
            history = [*seen.values(), *failed.values()]
            start_at = 1 + max(seen | failed, default=-1)
            todo, resumed = [], {**search, "history": history, "start": start_at}
        run = _spawn(
            capture_path,
            swept,
            [configs[i] for i in todo],
            todo,
            soft_s=SOFT_SHARE * left,
            hard_s=left,
            l2_flush=l2_flush,
            capture_sha256=capture_sha256,
            race=race,
            search=resumed,
        )
        checked = run["rows"]
        seen.update(checked)
        if run["result"] is not None:
            data = run["result"]
            break
        status = "timeout" if run["timed_out"] else "crash"
        culprit = run["culprit"]
        if culprit is None:  # outside any config (loading the capture, after timing)
            data = {"status": status, "error": run["error"], "table": []}
            break
        seen.pop(culprit, None)
        failed[culprit] = {
            "index": culprit,
            "config": configs[culprit] if search is None else run["culprit_config"],
            "status": status,
            "correct": False,
            "error": run["error"],
        }
    if data is None:
        data = {
            "status": "incomplete",
            "note": f"the sweep did not finish ({runs} processes): configs that passed their "
            "check are untimed",
        }
    # checked, maybe untimed: a later process did not finish
    rows = dict(checked if search is None else seen)
    rows.update({row["index"]: row for row in data.get("table") or []})
    rows.update(failed)
    for i, config in enumerate(configs):
        if i not in rows:
            status = data["status"] if data["status"] not in ("ok", "incomplete") else "skipped"
            error = data.get("error") if status != "skipped" else SKIPPED
            rows[i] = {"index": i, "config": config, "status": status, "correct": False}
            rows[i]["error"] = _tail(error)
    data["table"] = rank(list(rows.values()))
    data["runs"] = runs
    return data


def _stand_in(top: dict[str, Any]) -> dict[str, Any]:
    """The recorded result of a sweep in which no config passed: its first failure."""
    status = "timeout" if top["status"] == "skipped" else top["status"]
    result = {
        "status": status,
        "correct": False,
        "stage": top.get("stage") or "sweep",
        "error": f"no config of the sweep passed its quick check; the first, "
        f"{label(top['config'])}: {top.get('error')}",
    }
    if "failed_case" in top:
        result["failed_case"] = top["failed_case"]
    return result


def _nothing_tried(data: dict[str, Any]) -> dict[str, Any]:
    """The row that stands for a sweep that tried no config (a search whose whole space was
    pruned on this GPU, or a subprocess that failed before its first config)."""
    error = data.get("error")
    if not error:
        found = data.get("search") or {}
        why = "; ".join(f"{k}: {v}" for k, v in (found.get("pruned_examples") or {}).items())
        error = "the search tried no config: every config of the space was pruned on this GPU"
        error += f" ({why})" if why else ""
    status = data.get("status") if data.get("status") not in (None, "ok") else "error"
    return {"index": 0, "config": {}, "status": status, "correct": False, "error": error}


def _prebuild(
    swept: Path,
    capture_path: Path,
    configs: list[dict[str, Any]],
    timeout: float,
    search: dict[str, Any] | None = None,
) -> None:
    """With clean timing on (``hygiene.py``): every config's ``load_inline`` extensions built
    before the GPU lock, without a GPU (``kernels/prebuild.py``), so the sweep under the lock
    only loads them. A search's configs depend on what is measured: its first batch (the
    seeded initial design, :meth:`kernels.search.Search.ask`) is built here."""
    from kernel_agent import hygiene
    from kernel_agent.gpulock import holding

    if hygiene.current() is None or holding():
        return
    from kernel_agent.kernels import prebuild

    if search is not None:
        from kernel_agent.kernels.search import Search

        configs = Search.from_spec(search).ask() if search.get("space") is not None else [{}]

    inputs = prebuild.inputs_capture(capture_path)
    prebuild.prebuild(swept, inputs, configs, timeout=timeout)


def run_sweep(
    capture_path: Path,
    candidate_path: Path,
    configs: list[dict[str, Any]],
    *,
    timeout: float = 300.0,
    capture_sha256: str | None = None,
    l2_flush: bool = False,
    compile_check: bool = False,
    prepare: Callable[[Path], Path] | None = None,
    race: bool = True,
    search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sweep ``configs`` of a candidate and fully evaluate the best one, all under one
    GPU-lock acquisition (see the module docstring); ``timeout``: the evaluation timeout.

    ``prepare`` turns the file of the config that stands for the sweep (the best one
    bound into the source, :func:`bind_config`) into the file that is evaluated and
    recorded (the tool: its snapshot).  ``race``: racing of the configs (``--early-stop``,
    :func:`kernels.early.race`).  ``search``: a search (:func:`kernels.search.spec_from`)
    instead of ``configs`` (module docstring, **Search**).  Returns ``evaluation`` (the full
    evaluator's result, or the first failure when no config passed), ``config``,
    ``evaluated`` (that file), ``sweep`` (counts, ``cases``, the sorted ``table``; a
    search's summary in ``search``) and ``gpu_index``."""
    from kernel_agent.native import project

    # read once: what is swept is what is bound (a project directory: its bundle)
    source = project.source_of(candidate_path)
    name = project.bundle_name(candidate_path) if candidate_path.is_dir() else candidate_path.name
    workdir = Path(tempfile.mkdtemp(prefix="ka-sweep-"))
    try:
        swept = workdir / "swept" / name
        swept.parent.mkdir()
        swept.write_text(source)
        _prebuild(swept, capture_path, configs, TIMEOUT_FACTOR * timeout, search)
        start = time.monotonic()
        with gpu_lock() as gpu:  # one acquisition: the configs and the full evaluation
            ensure_peaks()
            data = _check_and_time(
                capture_path,
                swept,
                configs,
                budget_s=TIMEOUT_FACTOR * timeout,
                l2_flush=l2_flush,
                capture_sha256=capture_sha256,
                race=race,
                search=search,
            )
            seconds = round(time.monotonic() - start, 1)
            top = data["table"][0] if data["table"] else _nothing_tried(data)
            bound = workdir / "best" / name
            bound.parent.mkdir()
            bound.write_text(bind_config(source, top["config"]))
            evaluated = prepare(bound) if prepare is not None else bound
            if top["correct"]:  # same thread: the nested lock keeps this GPU
                evaluation = run_evaluation(
                    capture_path,
                    evaluated,
                    timeout=timeout,
                    capture_sha256=capture_sha256,
                    l2_flush=l2_flush,
                    compile_check=compile_check,
                )
            else:
                evaluation = _stand_in(top)
        table = data.pop("table")
        info = {
            "status": data.pop("status"),
            "configs": len(table),
            "passed": sum(bool(r["correct"]) for r in table),
            "failed": sum(not r["correct"] and r["status"] != "skipped" for r in table),
            "skipped": sum(r["status"] == "skipped" for r in table),
            "seconds": seconds,
            **data,
            "table": table,
        }
        return {
            "evaluation": evaluation,
            "config": top["config"],
            "evaluated": str(evaluated),
            "sweep": info,
            "gpu_index": gpu,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
