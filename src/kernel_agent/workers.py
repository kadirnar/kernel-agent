"""Parallel isolated workers per hot target (``--seeds-per-target``).

By default one engineer session works on a target at a time. With
``--seeds-per-target k`` (``auto``: :data:`AUTO_WORKERS` for targets with at least
:data:`AUTO_SHARE` of the profiled time, else 1) a target gets ``k`` workers: separate
agent sessions in ``targets/<id>/workers/<k>/``, each with its own ``candidates/``
and ``NOTES.md`` and its own starting point (:func:`seeds`): the planner's
``alternatives`` for the target (approach + backends), else the target's approach
on another primary backend (KernelFalcon's seeded workers, METR's diversity, GEAK's
workspace per agent). Everything else of the target is shared: the verified
evaluation store (``results.jsonl``, ``history/``), the GPU lock and the duplicate
check (:mod:`kernel_agent.dedup`), so ``best_for_target`` is the best of all
workers and a candidate another worker already evaluated returns its result.

The evaluation budget of a target stays what it is and is split across its
workers (:func:`split`): serial refinement beats parallel sampling at a fixed
budget (Kevin), so breadth costs depth and ``k = 1`` stays the default.
``--reseed-workers`` adds a second round: after a target's first worker sessions,
the next sessions start from its two best snapshots (:func:`reseeds`), with half
the budget (:func:`rounds`). Worker sessions run concurrently up to ``--parallel``
(``orchestrator.kernels`` and the slices of ``improve``).

A worker directory::

    targets/<id>/workers/<k>/
      candidates/  NOTES.md        the worker's own
      spec.json  reference_source.py  workload_profile.md  capture_inputs.pt
      history/  results.jsonl  quick.jsonl  plan.md     links to the target's shared files
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent.truth import Truth
from kernel_agent.workspace import RunDir

AUTO = "auto"
AUTO_SHARE = 0.20  # profile share from which `auto` gives a target several workers
AUTO_WORKERS = 2
MAX_WORKERS = 8
DIR = "workers"
#: Files of the target directory a worker sees through links in its own directory.
SHARED = (
    "spec.json",
    "reference_source.py",
    "workload_profile.md",
    "capture_inputs.pt",
    "history",
    "results.jsonl",
    "quick.jsonl",
    "plan.md",
)
RESEED_TOP = 2  # snapshots a second round starts from


@dataclass(frozen=True)
class Seed:
    """One worker session of a target: where it starts and what it may spend."""

    worker: int  # 1-based
    approach: str
    backends: tuple[str, ...]
    evaluations: int
    origin: str = "plan"  # plan | alternative | backend | reseed
    parent: str | None = None  # reseed: the snapshot (history/...) the session starts from
    parent_speedup: float | None = None


@dataclass(frozen=True)
class Binding:
    """What the evaluation tools of one worker session are bound to (``tools.build_server``)."""

    target_id: str
    worker: int
    agent: str
    evaluations: int | None = None


# ------------------------------------------------------------------ how many, which budget


def parse(value: str | int | None) -> int | str | None:
    """A ``--seeds-per-target`` value: a count ≥ 1, ``auto``, or None (not set: 1)."""
    if value is None:
        return None
    if str(value).strip().lower() == AUTO:
        return AUTO
    count = int(value)
    if not 1 <= count <= MAX_WORKERS:
        raise ValueError(f"--seeds-per-target must be {AUTO} or 1..{MAX_WORKERS}, got {value}")
    return count


def count(setting: str | int | None, share: float | None) -> int:
    """Workers of a target: ``setting`` (None: 1), or for ``auto`` its profile ``share``."""
    value = parse(setting)
    if value == AUTO:
        return AUTO_WORKERS if (share or 0.0) >= AUTO_SHARE else 1
    return int(value or 1)


def split(total: int, k: int) -> list[int]:
    """``total`` evaluations split over ``k`` workers (at least one each)."""
    k = max(k, 1)
    base, extra = divmod(max(total, 0), k)
    return [max(base + (i < extra), 1) for i in range(k)]


def rounds(total: int, reseed: bool) -> tuple[int, int]:
    """Evaluations of the first and the second (``--reseed-workers``) round of a target."""
    if not reseed or total < 2:
        return total, 0
    second = total // 2
    return total - second, second


# ------------------------------------------------------------------ seeds


def _rotated(pool: list[str], worker: int) -> list[str]:
    if not pool:
        return []
    primary = pool[(worker - 1) % len(pool)]
    return [primary, *(b for b in pool if b != primary)][:2]


def seeds(spec: dict[str, Any], k: int, total: int, available: list[str]) -> list[Seed]:
    """The first-round sessions of a target with ``k`` workers and ``total`` evaluations.

    Worker 1 follows the plan (``approach``, ``backends``); worker ``i`` takes the
    planner's ``alternatives[i - 2]`` when there is one, else the target's approach
    with another primary backend (the target's backends, then the other available ones).
    """
    planned = [b for b in spec.get("backends") or [] if b in available] or available[:2]
    pool = planned + [b for b in available if b not in planned]
    alternatives = [a for a in spec.get("alternatives") or [] if isinstance(a, dict)]
    budgets = split(total, k)
    out = [Seed(1, str(spec.get("approach") or ""), tuple(planned), budgets[0])]
    for i in range(2, k + 1):
        alt = alternatives[i - 2] if i - 2 < len(alternatives) else None
        if alt and str(alt.get("approach") or "").strip():
            backends = [b for b in alt.get("backends") or [] if b in available]
            backends = backends or _rotated(pool, i)
            out.append(
                Seed(i, str(alt["approach"]), tuple(backends), budgets[i - 1], "alternative")
            )
            continue
        backends = _rotated(pool, i)
        approach = str(spec.get("approach") or "")
        if backends and backends[0] != planned[0]:
            approach += f" (this worker starts with `{backends[0]}` instead of `{planned[0]}`)"
        else:
            approach += " (this worker: a different algorithm than the other workers)"
        out.append(Seed(i, approach, tuple(backends), budgets[i - 1], "backend"))
    return out


def reseeds(
    run: RunDir,
    target_id: str,
    previous: list[Seed],
    total: int,
    keeper: Truth | None = None,
    top: int = RESEED_TOP,
) -> list[Seed]:
    """Second-round sessions: one per best distinct snapshot (at most ``top``) of the target.

    A session goes to the worker that produced its snapshot when that worker is free,
    else to the lowest free worker; [] when the target has no correct result yet."""
    from kernel_agent.agent.tools import ranked_for_target

    best: list[dict[str, Any]] = []
    for rec in ranked_for_target(run, target_id, keeper):
        if rec.get("snapshot") not in {b.get("snapshot") for b in best}:
            best.append(rec)
        if len(best) == top:
            break
    if not best:
        return []
    by_worker = {s.worker: s for s in previous}
    free = sorted(by_worker) or list(range(1, len(best) + 1))
    budgets = split(total, len(best))
    out = []
    for i, rec in enumerate(best):
        try:
            mine = int(rec.get("worker") or 0)
        except (TypeError, ValueError):
            mine = 0
        worker = mine if mine in free else free[0]
        free.remove(worker)
        base = by_worker.get(worker)
        out.append(
            Seed(
                worker,
                base.approach if base else "",
                base.backends if base else (),
                budgets[i],
                "reseed",
                parent=str(rec["snapshot"]),
                parent_speedup=float(rec["speedup"]),
            )
        )
    return out


# ------------------------------------------------------------------ names and directories


def agent_name(target_id: str, worker: int) -> str:
    """``kernel-<target>-w<k>`` (target ids have no ``-``; ``program.role_of`` → kernel)."""
    return f"kernel-{target_id}-w{worker}"


def parse_agent(name: str) -> tuple[str, int | None]:
    """``(target, worker)`` of a kernel agent name (``worker`` None: the classic session)."""
    target, _, worker = name.removeprefix("kernel-").partition("-w")
    return target, int(worker) if worker.isdigit() else None


def directory(run: RunDir, target_id: str, worker: int) -> Path:
    return run.target(target_id) / DIR / str(worker)


def prepare(run: RunDir, target_id: str, worker: int) -> Path:
    """Create a worker's directory: own ``candidates/`` + ``NOTES.md``, links to the rest."""
    target_dir = run.target(target_id)
    path = directory(run, target_id, worker)
    (path / "candidates").mkdir(parents=True, exist_ok=True)
    (path / "NOTES.md").touch()
    for name in SHARED:
        link = path / name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(os.path.relpath(target_dir / name, path))
    return path


def notes_file(run: RunDir, target_id: str, worker: Any = None) -> Path:
    """``NOTES.md`` of a target's worker (``worker`` empty: of the classic session)."""
    if worker in (None, "", 0, "0"):
        return run.target(target_id) / "NOTES.md"
    return directory(run, target_id, int(worker)) / "NOTES.md"


def all_notes(run: RunDir, target_id: str) -> list[tuple[str, Path]]:
    """``(label, NOTES.md)`` of the target and of each of its workers that exist."""
    out = [("", notes_file(run, target_id))]
    base = run.target(target_id) / DIR
    numbers = (
        sorted(int(p.name) for p in base.iterdir() if p.name.isdigit()) if base.is_dir() else []
    )
    out += [(f"worker {k}", notes_file(run, target_id, k)) for k in numbers]
    return [(label, path) for label, path in out if path.is_file()]


# ------------------------------------------------------------------ prompt


def prompt_note(target_id: str, seed: Seed, team: list[Seed]) -> str:
    """``# Worker`` section of a worker's engineer prompt (``team``: every session of the round)."""
    others = [s for s in team if s.worker != seed.worker]
    lines = [
        "",
        "",
        f"# Worker {seed.worker} of {len(team)} on `{target_id}`",
        f"This target has {len(team)} isolated workers, each an agent session like you with its "
        "own starting point. They share one evaluation store: `best_result`, `best_so_far`, "
        "`history/` and `results.jsonl` include every worker's results, and a candidate "
        "identical to an evaluated one (comments and formatting aside) returns that result as "
        "a duplicate without a new evaluation.",
        f"* Your direction: {seed.approach or 'the planned approach'} "
        f"(backends: {', '.join(seed.backends) or 'any'}). Stay on it: the others cover:",
    ]
    lines += [
        f"  * worker {s.worker}: {s.approach[:200] or 'the planned approach'} "
        f"({', '.join(s.backends) or 'any backend'})"
        for s in others
    ]
    lines += [
        f"* Your working directory is `workers/{seed.worker}/` of the target: your own "
        "`candidates/` and `NOTES.md`. `spec.json`, `reference_source.py`, "
        "`workload_profile.md`, `capture_inputs.pt`, `history/`, `results.jsonl` and "
        "`plan.md` are links to the target's shared files (read-only for you).",
        f"* Your evaluation budget in this session: {seed.evaluations} evaluations (the "
        'target\'s budget is split across its workers). `mode="quick"` checks are free.',
    ]
    if seed.parent:
        speed = f" ({seed.parent_speedup:.3f}x)" if seed.parent_speedup else ""
        lines.append(
            f"* Round 2: start from `{seed.parent}`{speed}, one of the two best snapshots "
            f'so far. Refine it (`parent="{seed.parent}"`) rather than starting over.'
        )
    return "\n".join(lines)
