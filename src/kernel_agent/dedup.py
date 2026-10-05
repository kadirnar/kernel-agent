"""Duplicate candidates: a kernel evaluated again returns its earlier result.

Before ``evaluate_candidate`` takes the GPU lock it hashes the candidate's
normalised source (:func:`source_key`: ``ast.unparse(ast.parse(source))``, so
comments, blank lines, quoting and formatting do not matter; docstrings do) and
looks the key up in the target's verified records: ``results.jsonl`` and the
quick checks' ``quick.jsonl`` (:func:`quick_file`), which all workers of a
target share (:mod:`kernel_agent.workers`). A match is not evaluated again: the
tool returns the earlier result labelled ``duplicate of history/NNN``, the
ledger gets a ``duplicate`` row (:data:`kernel_agent.ledger.DUPLICATE`) and no
budget or streak counts it (KernelAgent dedups by PTX fingerprint; a source key
needs no compiler and catches the common case, a candidate resubmitted as is).

A full evaluation reuses only full results; a quick check reuses either. Results
that may not repeat (``timeout``, ``crash``, ``tampered``) are not reused, and
neither is a result that lacks what the call asks for (``profile`` tables are
never stored; ``compile_check`` only when the earlier call asked for it).
Every new record stores its ``source_key``; older records get theirs from their
snapshot, if it is still the evaluated file.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from kernel_agent.truth import TamperError, Truth
from kernel_agent.workspace import RunDir

#: Evaluator statuses that the same source reproduces (failures included: a build error
#: does not go away by evaluating the same file again).
REPEATABLE = ("ok", "incorrect", "build_error", "runtime_error")
QUICK = "quick"
FULL = "full"


def source_key(source: str) -> str:
    """Hash of a candidate's source, insensitive to comments and formatting."""
    try:
        text = ast.unparse(ast.parse(source))
    except (SyntaxError, ValueError, RecursionError):  # not Python: compare the stripped lines
        text = "\n".join(line.rstrip() for line in source.strip().splitlines())
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def quick_file(run: RunDir, target_id: str) -> Path:
    """Records of a target's quick checks (next to its ``results.jsonl``)."""
    return run.results_file(target_id).with_name("quick.jsonl")


def _key(rec: dict[str, Any], history: Path, keeper: Truth) -> str | None:
    if rec.get("source_key"):
        return str(rec["source_key"])
    snap = history / Path(str(rec.get("snapshot") or "")).name
    if not snap.is_file() or not keeper.snapshot_ok(snap, rec.get("snapshot_sha256")):
        return None
    return source_key(snap.read_text(errors="replace"))


def _records(keeper: Truth, path: Path) -> list[dict[str, Any]]:
    try:
        return keeper.records(path)
    except TamperError:
        return []


def find(
    run: RunDir,
    target_id: str,
    key: str,
    keeper: Truth,
    *,
    mode: str = FULL,
    compile_check: bool = False,
) -> dict[str, Any] | None:
    """The latest verified record of ``target_id`` with source ``key`` that a ``mode``
    evaluation may return instead of evaluating again (None: evaluate it)."""
    history = run.history_dir(target_id)
    pools = [_records(keeper, run.results_file(target_id))]
    if mode == QUICK:
        pools.append(_records(keeper, quick_file(run, target_id)))
    for records in pools:  # a full result first: it says more
        for rec in reversed(records):
            if rec.get("status") not in REPEATABLE:
                continue
            if compile_check and "compile_check" not in rec:
                continue
            if _key(rec, history, keeper) == key:
                return rec
    return None


def label(rec: dict[str, Any]) -> str:
    """``duplicate of history/NNN_...py (exp N)`` of the record a duplicate returns."""
    exp = f" (exp {rec['exp']})" if rec.get("exp") is not None else ""
    return f"duplicate of {rec.get('snapshot')}{exp}"
