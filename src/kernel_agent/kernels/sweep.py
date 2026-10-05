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
   speedup and ``pct_of_sol`` of every case.
3. **Evaluate** the best config with the full evaluator (:func:`run_evaluation`
   in a fresh process: every stage and anti-gaming guard, the checks outside the
   candidate's process included), with its config bound into the file
   (:func:`bind_config`); that file is what gets recorded, ranked and integrated.
   When no config passed, the first failure stands for the sweep.

A crash or a broken CUDA context (an illegal memory access) in one config ends
the subprocess; it starts again without that config (at most :data:`MAX_RUNS`
processes).  The configs share :data:`TIMEOUT_FACTOR` x the evaluation timeout:
configs not checked by then are ``skipped``, and timing stops after the last
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
from collections.abc import Callable
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
    """``BLOCK=1024, num_warps=4`` (``default`` for the empty config)."""
    return ", ".join(f"{k}={v!r}" for k, v in config.items()) or "default"


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
    """Passing configs by weighted speedup (untimed ones after them, in the given
    order), then the rejected ones, then the skipped ones."""

    def key(row: dict[str, Any]) -> tuple[int, float, int]:
        if row.get("correct"):
            speedup = row.get("speedup")
            return (
                (0, -float(speedup), row["index"]) if speedup is not None else (1, 0, row["index"])
            )
        return (3 if row.get("status") == "skipped" else 2, 0, row["index"])

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
    for flag in ("suspicious_faster_than_sol", "sol_unreliable"):
        if row.get(flag):
            out[flag] = True
    return out


def format_table(data: dict[str, Any]) -> str:
    """The sweep of :func:`run_sweep` as text (``kernel-agent eval --sweep``)."""
    info = data["sweep"]
    head = (
        f"sweep: {info['configs']} configs, {info['passed']} passed, {info['failed']} "
        f"rejected, {info['skipped']} skipped, {info.get('rounds') or 0} timing rounds, "
        f"{info['seconds']} s"
    )
    lines = [head]
    for case in info.get("cases") or []:
        ref = f", reference {case['ref_ms']} ms" if case.get("ref_ms") is not None else ""
        lines.append(f"  case {case['signature']} x{case['calls_per_run']}{ref}")
    lines.append(f"  {'speedup':>8}  {'%SOL':>6}  {'per case':<24}  config")
    for row in info["table"]:
        if row["correct"]:
            speedup = f"{row['speedup']:.3f}x" if row.get("speedup") is not None else "untimed"
            sol = f"{row['pct_of_sol']:.1f}" if row.get("pct_of_sol") is not None else "-"
            cases = " ".join(f"{c.get('speedup')}x" for c in row.get("cases") or [])
            lines.append(f"  {speedup:>8}  {sol:>6}  {cases:<24}  {label(row['config'])}")
        else:
            error = (row.get("error") or "").strip().splitlines() or [""]
            lines.append(f"  {row['status']:>16}  {label(row['config'])}: {error[-1][:120]}")
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
    passing: list[tuple[dict[str, Any], Any]],
    *,
    timer: Callable[..., dict[str, Any]],
    check_output: Callable[[Any, dict[str, Any]], dict[str, Any]] | None,
    l2_flush: bool,
    deadline: float | None,
    emit: Callable[[dict[str, Any]], None],
) -> tuple[int, dict[int, float]]:
    """Interleaved timing rounds of the passing configs; fills their rows (speedup per
    case and weighted).  Returns the number of whole rounds and the reference's time
    per timed case."""
    from kernel_agent.kernels.bench import median_round
    from kernel_agent.profiling.methods import entrypoint

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
            args, kwargs, method = case["args"], case["kwargs"], case["method"]
            ref_fn = entrypoint(reference, method)
            ref_rounds[ci].append(
                timer(ref_fn, args, kwargs, l2_flush=l2_flush, target_ms=TARGET_MS)
            )
            for row, candidate in order:
                if not row["correct"]:
                    continue
                emit({"event": "running", "index": row["index"]})
                try:
                    t = timer(
                        entrypoint(candidate, method),
                        args,
                        kwargs,
                        l2_flush=l2_flush,
                        target_ms=TARGET_MS,
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
) -> str | None:
    """``pct_of_sol`` of every timed row (:func:`kernels.roofline.apply_sol`, the work
    counted once per case, weights at the capture's ``precision``); a note when it cannot
    be computed."""
    from kernel_agent.kernels.roofline import apply_sol, count_case, current_peaks

    peaks = current_peaks()
    if not peaks:
        return "GPU peaks not measured yet (`kernel-agent doctor` measures them)"
    try:
        costs = [
            count_case(
                reference,
                cases[ci]["args"],
                cases[ci]["kwargs"],
                method=cases[ci]["method"],
                precision=precision,
            )
            for ci in timed
        ]
        for row in rows:
            if row.get("correct") and row.get("cases"):
                apply_sol(row, costs, peaks, hot_l2=not l2_flush)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


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
) -> dict[str, Any]:
    """Steps 1 and 2 of the module docstring, in this process: the sorted ``table``.

    ``indices``: the configs' numbers in the whole sweep (rows carry them as
    ``index``); ``deadline`` (``time.monotonic()``): no new work after it; ``emit``
    gets progress events (``running`` before a config's check or timing, ``row``
    after its check).  Timing needs CUDA, or ``timer``: a stand-in for
    :func:`kernels.bench.time_call` (tests)."""
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
    session: dict[str, Any] = {"capture": capture, "memo": {id(w): w for _, w in named}}
    checks_until = None if deadline is None else start + CHECK_SHARE * (deadline - start)

    # 1. every config through the quick tier
    rows: list[dict[str, Any]] = []
    passing: list[tuple[dict[str, Any], Any]] = []
    for index, config in zip(indices, configs, strict=True):
        row: dict[str, Any] = {"index": index, "config": config}
        rows.append(row)
        if checks_until is not None and time.monotonic() > checks_until:
            row.update(status="skipped", correct=False, error=SKIPPED)
            continue
        emit({"event": "running", "index": index})
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
        candidate = session.pop("candidate", None)
        if row["correct"] and candidate is not None:
            passing.append((row, candidate))
        elif cuda:
            _check_context(index)
        emit({"event": "row", **row})

    # 2. the passing configs, timed against the reference
    timed = [i for i, c in enumerate(cases) if c["count"]]
    out: dict[str, Any] = {
        "status": "ok",
        "device": device,
        "quick_cases": [cases[i]["signature"] for i in quick_cases(cases)],
        "cases": [
            {"case": i, "signature": cases[i]["signature"], "calls_per_run": cases[i]["count"]}
            for i in timed
        ],
    }
    time_fn = timer or (bench.time_call if cuda else None)
    if passing and timed and time_fn is not None:
        if cuda:
            bench.warm_gpu()
        out["rounds"], ref_ms = _time_configs(
            reference,
            cases,
            timed,
            passing,
            timer=time_fn,
            check_output=bench.check_timed_output if cuda else None,
            l2_flush=l2_flush,
            deadline=deadline,
            emit=emit,
        )
        for case in out["cases"]:
            case["ref_ms"] = ref_ms[case["case"]]
        precision = capture_precision(capture)  # reduced-precision weights: their own bytes
        if cuda and (note := _speed_of_light(rows, reference, cases, timed, l2_flush, precision)):
            out["sol_note"] = note
    elif passing:
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
) -> dict[str, Any]:
    """One sweep subprocess: its checked ``rows``, final ``result`` (None if it did not
    finish) and the ``culprit``: the config that ran when it died."""
    spec = swept.parent / "configs.json"
    spec.write_text(json.dumps({"configs": configs, "indices": indices}))
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
    for line in stdout.splitlines():
        if not line.startswith(marker):
            continue
        with contextlib.suppress(json.JSONDecodeError):
            event = json.loads(line[len(marker) :])
            kind = event.pop("event", None)
            if kind == "running":
                running = event.get("index")
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
    return {
        "rows": rows,
        "result": result,
        "culprit": running,
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
) -> dict[str, Any]:
    """Steps 1 and 2 in subprocesses (another one without a config that crashed); every
    config gets a row."""
    start = time.monotonic()
    failed: dict[int, dict[str, Any]] = {}
    checked: dict[int, dict[str, Any]] = {}
    data: dict[str, Any] | None = None
    runs = 0
    while runs < MAX_RUNS:
        left = budget_s - (time.monotonic() - start)
        todo = [i for i in range(len(configs)) if i not in failed]
        if not todo or left < MIN_RUN_S:
            break
        runs += 1
        run = _spawn(
            capture_path,
            swept,
            [configs[i] for i in todo],
            todo,
            soft_s=SOFT_SHARE * left,
            hard_s=left,
            l2_flush=l2_flush,
            capture_sha256=capture_sha256,
        )
        checked = run["rows"]
        if run["result"] is not None:
            data = run["result"]
            break
        status = "timeout" if run["timed_out"] else "crash"
        culprit = run["culprit"]
        if culprit is None:  # outside any config (loading the capture, after timing)
            data = {"status": status, "error": run["error"], "table": []}
            break
        failed[culprit] = {
            "index": culprit,
            "config": configs[culprit],
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
    rows = dict(checked)  # checked, maybe untimed: a later process did not finish
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
) -> dict[str, Any]:
    """Sweep ``configs`` of a candidate and fully evaluate the best one, all under one
    GPU-lock acquisition (see the module docstring); ``timeout``: the evaluation timeout.

    ``prepare`` turns the file of the config that stands for the sweep (the best one
    bound into the source, :func:`bind_config`) into the file that is evaluated and
    recorded (the tool: its snapshot).  Returns ``evaluation`` (the full evaluator's
    result, or the first failure when no config passed), ``config``, ``evaluated``
    (that file), ``sweep`` (counts, ``cases``, the sorted ``table``) and ``gpu_index``."""
    source = candidate_path.read_text()  # read once: what is swept is what is bound
    workdir = Path(tempfile.mkdtemp(prefix="ka-sweep-"))
    try:
        swept = workdir / "swept" / candidate_path.name
        swept.parent.mkdir()
        swept.write_text(source)
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
            )
            seconds = round(time.monotonic() - start, 1)
            top = data["table"][0]
            bound = workdir / "best" / candidate_path.name
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
