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
(``orchestrator.kernels``).

In ``improve`` the workers are **islands** (docs/MULTIAGENT.md §3.5, issue #189;
``--islands``, an alias of ``--seeds-per-target``): persistent workers (:class:`Island`)
with their own direction, lineage and scores, and no fixed split of the budget. Every
session of the target is one island's, with a whole ``--slice`` of evaluations:

* **Which island**: the arm's score decides whether the target gets a slot; the island
  with the best island UCB (:func:`rank`: its own gain per evaluation, plus exploration
  on few evaluations, decayed by its stale sessions) and no running session gets it.
  With ``--agents N`` the target runs up to one session per island at once.
* **Migration** (OpenEvolve / ShinkaEvolve islands, MAP-Elites cells of backend × idea):
  every ``--migrate-every`` evaluations of the target an island's next digest gets
  ``## Inspirations`` (:func:`inspirations`): the target's best result of another island
  when it beats the island's own, and the best one of another cell. New bests also reach
  running sessions as board winners (``board.py``).
* **Culling** (:func:`cull_reason`): an island ``--cull-gap`` below the target's best after
  :data:`CULL_AFTER` evaluations and :data:`CULL_STALE` sessions without a new island best
  is reseeded (:func:`reseed`): the next unused direction, the target's best as parent,
  its ``NOTES.md`` archived.

Breadth only pays when it shares conclusions and uses otherwise idle time (MKEvolve,
KernelArc, Kevin in docs/MULTIAGENT-LITERATURE.md), so ``--islands 1`` stays the default.

A worker directory::

    targets/<id>/workers/<k>/
      candidates/  NOTES.md        the worker's own
      spec.json  reference_source.py  workload_profile.md  capture_inputs.pt
      history/  results.jsonl  quick.jsonl  plan.md     links to the target's shared files
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent import ledger
from kernel_agent.budget import improves
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
    "research.md",
)
RESEED_TOP = 2  # snapshots a second round starts from
# islands (improve, #189): the target's evaluations between two offers of inspirations to an
# island, and its own evaluations before the first (--migrate-every; 0: no migration)
MIGRATE_EVERY = 4
CULL_GAP = 0.15  # --cull-gap: an island this far below the target's best is reseeded ...
CULL_AFTER = 8  # ... after this many evaluations of its generation
CULL_STALE = 2  # ... and this many of its sessions in a row without a new island best
INSPIRATIONS = 2  # elites in an island's digest: the best of the others and another cell's
ISLAND_MARK = "# Island "  # the heading of an island session's section (island_note)


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


def direction(
    spec: dict[str, Any], i: int, available: list[str]
) -> tuple[str, tuple[str, ...], str]:
    """``(approach, backends, origin)`` of the ``i``-th starting point of a target (1-based):
    1 follows the plan (``approach``, ``backends``); ``i`` the planner's
    ``alternatives[i - 2]`` when there is one, else the target's approach with another
    primary backend (the target's backends, then the other available ones, in turn)."""
    planned = [b for b in spec.get("backends") or [] if b in available] or available[:2]
    if i <= 1:
        return str(spec.get("approach") or ""), tuple(planned), "plan"
    pool = planned + [b for b in available if b not in planned]
    alternatives = [a for a in spec.get("alternatives") or [] if isinstance(a, dict)]
    alt = alternatives[i - 2] if i - 2 < len(alternatives) else None
    if alt and str(alt.get("approach") or "").strip():
        backends = [b for b in alt.get("backends") or [] if b in available]
        return str(alt["approach"]), tuple(backends or _rotated(pool, i)), "alternative"
    backends = _rotated(pool, i)
    approach = str(spec.get("approach") or "")
    if backends and backends[0] != planned[0]:
        approach += f" (this worker starts with `{backends[0]}` instead of `{planned[0]}`)"
    else:
        approach += " (this worker: a different algorithm than the other workers)"
    return approach, tuple(backends), "backend"


def seeds(spec: dict[str, Any], k: int, total: int, available: list[str]) -> list[Seed]:
    """The first-round sessions of a target with ``k`` workers and ``total`` evaluations,
    each from its :func:`direction`."""
    budgets = split(total, k)
    out = []
    for i in range(1, k + 1):
        approach, backends, origin = direction(spec, i, available)
        out.append(Seed(i, approach, backends, budgets[i - 1], origin))
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
        "`workload_profile.md`, `capture_inputs.pt`, `history/`, `results.jsonl`, "
        "`plan.md` and `research.md` are links to the target's shared files (read-only for "
        "you).",
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


# ------------------------------------------------------------------ islands (improve, #189)

#: What ``improve.json`` → ``islands`` keeps of an island (the ledger cannot say it).
STATE = (
    "approach",
    "backends",
    "origin",
    "generation",
    "since",
    "parent",
    "parent_speedup",
    "inspired",
)


@dataclass
class Island:
    """A persistent worker ``k`` of a target in ``improve`` (docs/MULTIAGENT.md §3.5): its
    direction (:func:`direction`), its lineage (the ``parent`` chain of its candidates) and
    its own ``NOTES.md`` in ``workers/<k>/``. ``improve.json`` → ``islands`` keeps
    :data:`STATE`; the rest is measured from the ledger rows of its ``worker`` since its
    generation started (:func:`measure`)."""

    target: str
    k: int
    approach: str = ""
    backends: tuple[str, ...] = ()
    origin: str = "plan"  # plan | alternative | backend | reseed
    generation: int = 1  # 1 + the times it was culled and reseeded
    since: int = 0  # the ledger's length when its generation started: its rows come after
    parent: str | None = None  # ``history/...`` its generation starts from (None: reference)
    parent_speedup: float | None = None
    inspired: int = 0  # the target's evaluations when it was last offered inspirations
    # measured (:func:`measure`)
    evals: int = 0
    best: float = 1.0  # its lineage's speedup: its own best, or a better parent it built on
    head: str | None = None  # the snapshot of that speedup (what it builds on next)
    backend: str = ""  # that snapshot's backend (its direction's first one before any)
    gain: float = 0.0  # Σ 1/old − 1/new over its new island bests (its rate's numerator)
    streak: int = 0  # its evaluations since its last new island best
    sessions: int = 0  # its sessions with an evaluation in this generation
    stale: int = 0  # its last sessions in a row without a new island best
    running: bool = False  # a session of it runs
    index: float = 0.0  # its UCB index (:func:`rank`)
    score: float = 0.0  # index × decay ** stale

    @property
    def agent(self) -> str:
        return agent_name(self.target, self.k)

    @property
    def rate(self) -> float:
        """Its gain per evaluation (in 1 / speedup: the share of the target's time it saved)."""
        return self.gain / self.evals if self.evals else 0.0

    def state(self) -> dict[str, Any]:
        out = {key: getattr(self, key) for key in STATE}
        out["backends"] = list(self.backends)
        return out

    @classmethod
    def of(cls, target: str, k: int, saved: dict[str, Any]) -> Island:
        known = {key: saved[key] for key in STATE if key in saved}
        known["backends"] = tuple(known.get("backends") or ())
        return cls(target, k, **known)

    def seed(self, evaluations: int) -> Seed:
        """The island as a worker session's :class:`Seed` (``Orchestrator.worker_session``)."""
        return Seed(
            self.k,
            self.approach,
            self.backends,
            evaluations,
            self.origin,
            self.parent,
            self.parent_speedup,
        )


def new_islands(
    target: str,
    k: int,
    spec: dict[str, Any],
    available: list[str],
    saved: dict[str, dict[str, Any]],
    elite: tuple[str, float] | None = None,
) -> list[Island]:
    """The ``k`` islands of a target: those ``saved`` (``improve.json``), the others new, each
    from its :func:`direction`; a new island 1 continues the target's best so far (``elite``:
    its snapshot and speedup: a prior from the library, a classic session's), the others
    start from the reference in their own direction."""
    out = []
    for i in range(1, k + 1):
        if str(i) in saved:
            out.append(Island.of(target, i, saved[str(i)]))
            continue
        approach, backends, origin = direction(spec, i, available)
        parent, speedup = (f"history/{elite[0]}", elite[1]) if elite and i == 1 else (None, None)
        out.append(
            Island(target, i, approach, backends, origin, parent=parent, parent_speedup=speedup)
        )
    return out


def _levels(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Snapshot name → its correct ledger row of a target (its island's evaluation; an
    integration's re-evaluation replaces its speedup, or drops it when it was not correct
    again)."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("snapshot") or "")
        if row["status"] == ledger.REEVALUATED:
            if row["correct"] and row["speedup"] and name in out:
                out[name] = {**out[name], "speedup": row["speedup"]}
            else:
                out.pop(name, None)
        elif row["status"] not in ledger.UNMEASURED and row["correct"] and row["speedup"]:
            out[name] = row
    return out


def _beats(row: dict[str, Any], level: float) -> bool:
    """Whether a correct ledger row beats ``level`` by more than the noise (the keep rule)."""
    rec = {"correct": True, "speedup": row["speedup"], "timing_spread": row.get("spread") or 0.0}
    return improves(rec, level)


def _mine(island: Island, row: dict[str, Any]) -> bool:
    return str(row.get("worker") or "") == str(island.k) and (row["exp"] or 0) > island.since


def measure(island: Island, rows: list[dict[str, Any]], running: bool = False) -> Island:
    """Fill in what the ledger says of ``island`` from its target's ``rows``: its evaluations
    in this generation, its lineage's best (a candidate it built on a better snapshot of
    another island, a migration, lifts it), its gain, streak, sessions and stale sessions
    (by the ``session`` of its rows)."""
    levels = _levels(rows)
    start = levels.get(Path(island.parent or "").name) if island.parent else None
    level = float(start["speedup"]) if start else float(island.parent_speedup or 1.0)
    island.head = Path(island.parent).name if island.parent else None
    island.backend = str(start["backend"]) if start else (island.backends or ("",))[0]
    island.evals = island.streak = 0
    island.gain = 0.0
    order: list[str] = []
    better: set[str] = set()
    for row in ledger.measured(rows):
        if not _mine(island, row):
            continue
        island.evals += 1
        label = str(row.get("session") or "")
        if label not in order:
            order.append(label)
        parent = levels.get(Path(str(row.get("parent") or "")).name)
        if parent is not None and float(parent["speedup"]) > level:  # built on a better one
            level, island.head = float(parent["speedup"]), str(parent["snapshot"])
            island.backend = str(parent["backend"])
        if row["correct"] and row["speedup"] and _beats(row, level):
            island.gain += 1.0 / level - 1.0 / float(row["speedup"])
            level, island.head = float(row["speedup"]), str(row["snapshot"])
            island.backend = str(row["backend"])
            island.streak = 0
            better.add(label)
        else:
            island.streak += 1
    island.best = level
    island.sessions = len(order)
    island.stale = 0
    for label in reversed(order):
        if label in better:
            break
        island.stale += 1
    island.running = running
    return island


def rank(islands: list[Island], decay: float, explore: float) -> list[Island]:
    """Island UCB (the arm's rule, ``scheduler.rank``, one level down): ``index = rate / best
    rate + explore × sqrt(2 ln(N + 2) / (n + 1))`` (``n`` its evaluations, ``N`` all of the
    target's islands'), ``score = index × decay ** stale``; best first (ties: the lowest k)."""
    total = sum(i.evals for i in islands)
    top = max((i.rate for i in islands), default=0.0)
    for island in islands:
        exploit = island.rate / top if top > 0 else 0.0
        bonus = explore * math.sqrt(2 * math.log(total + 2) / (island.evals + 1))
        island.index = exploit + bonus
        island.score = island.index * decay**island.stale
    return sorted(islands, key=lambda i: (-i.score, i.k))


def choose(islands: list[Island]) -> Island | None:
    """The island that gets the target's next session: the best ranked (:func:`rank`) without
    a running session (None: every island runs one)."""
    return next((i for i in islands if not i.running), None)


def cull_reason(
    island: Island,
    best: float,
    gap: float,
    after: int = CULL_AFTER,
    stale: int = CULL_STALE,
) -> str | None:
    """Why ``island`` is culled and reseeded from the target's ``best``, or None: it is not
    running, made at least ``after`` evaluations in this generation, its last ``stale``
    sessions found no new island best, and its best is more than ``gap`` below the target's."""
    if gap <= 0 or island.running or best <= 1.0 or island.evals < after:
        return None
    if island.stale < stale or island.best >= (1.0 - gap) * best:
        return None
    return (
        f"island {island.k}: {island.best:.3f}x after {island.evals} evaluations, "
        f"{island.stale} sessions without a new island best, {1 - island.best / best:.0%} "
        f"below the target's {best:.3f}x"
    )


def reseed(
    island: Island,
    spec: dict[str, Any],
    available: list[str],
    i: int,
    elite: tuple[str, float],
    since: int,
    inspired: int,
) -> Island:
    """``island``'s next generation: the ``i``-th :func:`direction` of the target (the next
    unused one) with the target's best (``elite``: snapshot and speedup) as its parent; its
    rows count from ledger length ``since``."""
    approach, backends, _ = direction(spec, i, available)
    return Island(
        island.target,
        island.k,
        approach,
        backends,
        "reseed",
        island.generation + 1,
        since,
        f"history/{elite[0]}",
        elite[1],
        inspired,
    )


def archive_notes(run: RunDir, target_id: str, island: Island) -> Path | None:
    """Move a culled island's ``NOTES.md`` aside (``NOTES.gen<g>.md``) for a fresh one; the
    archived file (None: there was nothing in it)."""
    notes = notes_file(run, target_id, island.k)
    if not notes.is_file() or not notes.read_text().strip():
        return None
    archived = notes.with_name(f"NOTES.gen{island.generation}.md")
    notes.replace(archived)
    notes.touch()
    return archived


def migration_due(island: Island, evaluations: int, every: int) -> bool:
    """Whether ``island``'s next digest gets inspirations: migration is on (``every`` > 0),
    the island made ``every`` evaluations of its own in this generation, the target
    (``evaluations``: all its islands') ``every`` since its last offer, and its last session
    found no new island best (one that still climbs keeps its own direction: the islands are
    worth more apart)."""
    if every <= 0 or island.evals < every or island.stale < 1:
        return False
    return evaluations - island.inspired >= every


def inspirations(
    island: Island, rows: list[dict[str, Any]], limit: int = INSPIRATIONS
) -> list[dict[str, Any]]:
    """The elites of the other islands for ``island``'s digest (MAP-Elites cells: a result's
    backend × idea), each faster than the reference: the best result of the others when it
    beats the island's lineage, then the best of another cell (a backend neither the island
    nor that elite uses, else another idea), at most ``limit``."""
    others = [
        r
        for r in _levels(rows).values()
        if str(r.get("worker") or "") != str(island.k) and _beats(r, 1.0)
    ]
    ranked = sorted(others, key=lambda r: (-float(r["speedup"]), r["exp"] or 0))
    ranked = [r for r in ranked if r["snapshot"] != island.head]
    out: list[dict[str, Any]] = []
    if ranked and _beats(ranked[0], island.best):
        out.append(_inspiration(ranked[0], "the best result of the other islands"))
    backends = {island.backend, *(e["backend"] for e in out)}
    ideas = {e["idea"] for e in out if e["idea"]}
    rest = [r for r in ranked if r["snapshot"] not in {e["snapshot"] for e in out}]
    cell = next((r for r in rest if str(r.get("backend") or "") not in backends), None)
    if cell is not None:
        out.append(_inspiration(cell, f"the best `{cell['backend']}` result: another approach"))
    elif idea := next((r for r in rest if r.get("idea") and r["idea"] not in ideas), None):
        out.append(_inspiration(idea, f"the best result of idea `{idea['idea']}`: another idea"))
    return out[:limit]


def _inspiration(row: dict[str, Any], why: str) -> dict[str, Any]:
    worker = str(row.get("worker") or "")
    return {
        "snapshot": str(row["snapshot"]),
        "speedup": float(row["speedup"]),
        "backend": str(row.get("backend") or ""),
        "idea": str(row.get("idea") or ""),
        "island": int(worker) if worker.isdigit() else None,
        "exp": row["exp"],
        "why": why,
    }


def adoptions(rows: list[dict[str, Any]]) -> int:
    """How often an island built on another island's snapshot (a migration used, a reseed
    from the best): distinct (island, parent) pairs of a target's ``rows``."""
    made: dict[str, str] = {}  # snapshot -> the island that evaluated it first
    for row in ledger.measured(rows):
        made.setdefault(str(row["snapshot"]), str(row.get("worker") or ""))
    pairs = set()
    for row in ledger.measured(rows):
        worker = str(row.get("worker") or "")
        parent = Path(str(row.get("parent") or "")).name
        if worker and parent in made and made[parent] != worker:
            pairs.add((worker, parent))
    return len(pairs)


def island_note(target_id: str, island: Island, islands: list[Island], evaluations: int) -> str:
    """``# Island`` section of an island session's engineer prompt (``islands``: the target's)."""
    others = [i for i in islands if i.k != island.k]
    lines = [
        "",
        "",
        f"{ISLAND_MARK}{island.k} of {len(islands)} on `{target_id}`",
        f"This target is searched by {len(islands)} islands: agent sessions like you, each with "
        "its own direction, its own lineage (the `parent` chain of its candidates) and its own "
        "`NOTES.md`. They share one evaluation store: `best_result`, `best_so_far`, `history/` "
        "and `results.jsonl` include every island's results (a row's `worker` is its island), "
        "and a candidate identical to an evaluated one returns that result as a duplicate "
        "without a new evaluation.",
        f"* Your direction: {island.approach or 'the planned approach'} "
        f"(backends: {', '.join(island.backends) or 'any'}). Stay on it: the other islands "
        "cover:",
    ]
    lines += [
        f"  * island {o.k}: {o.approach[:200] or 'the planned approach'} "
        f"({', '.join(o.backends) or 'any backend'})"
        for o in others
    ]
    lines += [
        f"* Your working directory is `workers/{island.k}/` of the target: your own "
        "`candidates/` and `NOTES.md`. `spec.json`, `reference_source.py`, "
        "`workload_profile.md`, `capture_inputs.pt`, `history/`, `results.jsonl`, "
        "`plan.md` and `research.md` are links to the target's shared files (read-only for "
        "you).",
        f"* Your evaluation budget in this session: {evaluations} evaluations. The target's "
        "next session goes to the island whose results pay most per evaluation, and an island "
        "that stays well behind the target's best is reseeded from it, so make each one "
        'count. `mode="quick"` checks are free.',
        "* Migration: every few evaluations your digest's `## Inspirations` lists the best "
        "results of the other islands. Build on one (`parent=...`) when it beats your island's "
        "best, or carry its idea into your direction. Their new bests also reach you on the "
        "board when the run has one: take ideas from them, but keep your own lineage until "
        "`## Inspirations` offers theirs (the islands are worth more apart).",
    ]
    if island.origin == "reseed" and island.parent:
        speed = f" ({island.parent_speedup:.3f}x)" if island.parent_speedup else ""
        lines.append(
            f"* Reseeded (generation {island.generation}): this island's earlier lineage fell "
            f"behind. Start from `{island.parent}`{speed}, the target's best, and take it "
            f'further in your new direction (`parent="{island.parent}"`).'
        )
    return "\n".join(lines)


def island_lines(
    island: Island,
    islands: list[Island],
    best: float,
    best_snapshot: str | None,
    offered: list[dict[str, Any]],
) -> list[str]:
    """``## Your island`` (and ``## Inspirations`` when ``offered``) of an island session's
    digest, in place of the target's ``## Best so far``."""
    lines = [
        "",
        f"## Your island (island {island.k} of {len(islands)}, generation {island.generation})",
    ]
    if island.head:
        lines.append(
            f"* Your island's best: `history/{island.head}` at {island.best:.3f}x "
            f'(`{island.backend}`). Build on it (`parent="history/{island.head}"`) unless you '
            "test a different idea of your direction."
        )
    else:
        lines.append(
            "* No correct candidate of this island yet: start from the reference in your "
            "direction (the `# Island` section)."
        )
    lines.append(
        f"* {island.evals} evaluations of this island in this generation, {island.streak} since "
        "its last new best."
    )
    if best_snapshot:
        lines.append(f"* The target's best: `history/{best_snapshot}` at {best:.3f}x.")
    if not offered:
        return lines
    lines += [
        "",
        "## Inspirations (migration: the best results of the other islands)",
        'Build on one (`parent="history/..."`) when it beats your island\'s best, or carry its '
        "idea into your direction:",
    ]
    for e in offered:
        what = [f"`{e['backend']}`"] if e["backend"] else []
        what += [f"idea `{e['idea']}`"] if e["idea"] else []
        what += [f"island {e['island']}"] if e["island"] else []
        lines.append(
            f"* `history/{e['snapshot']}`: {e['speedup']:.3f}x ({', '.join(what)}): {e['why']}"
        )
    return lines
