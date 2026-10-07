"""Precision pivots: a plateaued target moves to another precision tier mid-run.

A target's precision is fixed when it is planned and captured (``spec.json`` →
``precision``, recorded with its tolerance tier in the sealed capture). When the
evidence later shows that its remaining gain lies in another precision (the VoxCPM2
throughput run: the exact-tier ``dit_layer`` arm stayed at 3.01x while W8A8 FP8 on the
same GEMMs, done as transforms, took the run from 3.3x to 4.7x), a pivot moves it there:

* **Proposals.** The research session of a plateaued arm may write ``pivot.json`` next
  to its ``plan.md`` (:data:`PIVOT_FILE`), and a round re-plan may list ``pivots`` in its
  plan; each is ``{"target", "precision", "precision_why"[, "approach"]}``, the
  ``precision_why`` backed by numbers (the bound, the ceilings table, a passing transform
  that already uses that precision).
* **Check** (:func:`check`): ``--quality near-lossless`` only, a reduced precision
  (:data:`kernel_agent.kernels.compare.REDUCED_PRECISIONS`) that the run allows
  (``--precisions``, :mod:`kernel_agent.precisions`: no 4-bit by default) other than the
  target's own, a ``precision_why`` with a number in it, a module target (not a region
  target), and no earlier pivot of the target to that precision.
* **New arm.** The orchestrator writes the spec of a new target ``<id>__<precision>``
  (:func:`pivot_spec`, ``pivot_of``: the original id) and captures it in the new tier
  (``Orchestrator.pivot``), so the scheduler sees a new arm with a fresh capture while
  the old arm keeps its history and its exact results stay comparable.
* **Integration** treats a target and its pivots as versions of one item
  (:func:`family`, ``orchestrator._item_key``): one of them is applied, the version swap
  step measures the other one in its place.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kernel_agent.workspace import RunDir

PIVOT_FILE = "pivot.json"
SEP = "__"  # <target id>__<precision>: the id of a pivot's new target
_NUMBER = re.compile(r"\d")


def proposal_path(run: RunDir, target_id: str) -> Path:
    """Where the research session of ``target_id`` writes its pivot proposal."""
    return run.target(target_id) / PIVOT_FILE


def family(target_id: str) -> str:
    """The original target of ``target_id``: its id without a ``__<precision>`` suffix."""
    from kernel_agent.kernels.compare import PRECISIONS

    base, sep, suffix = target_id.rpartition(SEP)
    return base if sep and base and suffix in PRECISIONS else target_id


def pivot_id(target_id: str, precision: str) -> str:
    """Id of the target that moves ``target_id``'s family to ``precision``."""
    return f"{family(target_id)}{SEP}{precision}"


def taken(run: RunDir) -> set[str]:
    """Ids of every target directory of the run, failed captures (``spec.failed.json``)
    included: a pivot is tried once."""
    root = run.targets_dir
    return {p.name for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()


def precision_of(spec: dict[str, Any]) -> str:
    """A spec's precision (``exact`` when it has none)."""
    return str(spec.get("precision") or "exact")


def read_proposal(path: Path) -> dict[str, Any] | None:
    """The proposal in ``path`` (``pivot.json``), or None when there is none or it is not a
    JSON object."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check(
    spec: dict[str, Any],
    proposal: dict[str, Any],
    *,
    quality: str,
    taken: set[str],
    allowed: Iterable[str] | None = None,
) -> str | None:
    """Why ``proposal`` cannot move the target of ``spec`` (None: it can). ``taken``: ids
    of the run's targets (also failed captures: a pivot is tried once); ``allowed``: the
    precisions the run allows (``precisions.py``; None: the default of ``quality``, no
    4-bit)."""
    from kernel_agent import precisions
    from kernel_agent.kernels.compare import NEAR_LOSSLESS_TIER, REDUCED_PRECISIONS

    if not spec.get("id"):
        return "no such target"
    if quality != NEAR_LOSSLESS_TIER:
        return f"a precision pivot needs --quality near-lossless (this run: {quality})"
    precision = str(proposal.get("precision") or "")
    if precision not in REDUCED_PRECISIONS:
        return f"precision {precision!r} is not one of {', '.join(REDUCED_PRECISIONS)}"
    allowed = precisions.default(quality) if allowed is None else tuple(allowed)
    if problem := precisions.refusal(precision, allowed):
        return problem
    if precision == precision_of(spec):
        return f"the target is {precision} already"
    why = str(proposal.get("precision_why") or "").strip()
    if not _NUMBER.search(why):
        return "precision_why must give the numbers behind the pivot"
    if spec.get("kind") == "region":
        return "a region target cannot pivot (its rewrite is tied to its id)"
    if (new := pivot_id(str(spec["id"]), precision)) in taken:
        return f"{new} exists already"
    return None


def pivot_spec(spec: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    """The spec of the pivot's new target: ``spec``'s target, scope and approach (or the
    proposal's ``approach``), the new ``precision`` / ``precision_why`` and ``pivot_of``."""
    keep = (
        "module_class",
        "qualname",
        "qualname_regex",
        "phase",
        "why",
        "approach",
        "backends",
        "alternatives",
        "expected_speedup",
    )
    precision = str(proposal["precision"])
    new = {k: spec[k] for k in keep if spec.get(k) is not None}
    new.update(
        id=pivot_id(str(spec["id"]), precision),
        precision=precision,
        precision_why=str(proposal["precision_why"]).strip(),
        pivot_of=str(spec["id"]),
    )
    if approach := str(proposal.get("approach") or "").strip():
        new["approach"] = approach
    return new


def label(spec: dict[str, Any]) -> str:
    """``fp8_w8a8 (pivot of dit_layer)``, ``exact``: an arm's precision for tables."""
    text = precision_of(spec)
    return f"{text} (pivot of {spec['pivot_of']})" if spec.get("pivot_of") else text
