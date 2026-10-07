"""The calls per run behind a kernel's estimated saving (``est_saved_ms_per_run``).

The evaluator times the cases of one captured instance. Each case stands for some calls
per run of the target's instances (its *weight*, ``target_calls`` in the evaluation's
case reports), and the estimate is Σ (reference − candidate ms per call) × weight:

* ``instance groups`` (captures since #119): during the capture run every instance of the
  target counts its calls (in the target's ``phase``) per entrypoint and primary input
  (:func:`kernel_agent.profiling.capture.capture_module`). A case stands for the calls of
  every instance with its primary input (``target_calls``, split over decode buckets like
  its ``count``). Calls with a primary input that no case has are not counted: no case
  times them. On VoxCPM2 the ``MiniCPMDecoderLayer`` target of the LocDiT + LocEnc layers
  keeps a LocDiT layer (540 calls at ``[32, 11, 1024]``): its case stands for the 12
  LocDiT layers' 6,480 calls, and the 12 LocEnc layers' 732 calls at ``[16, 5, 1024]``
  are ``uncovered``. ``instance_groups`` holds the same counts per instance group
  (qualname with the layer indices folded): its instances, calls and the calls a case
  covers.
* ``even split`` (older captures, :func:`kernel_agent.profiling.capture.capture_calls`):
  the captured instance's calls × the instances that call the case's entrypoint, as if
  every instance made the same calls. That counted the LocEnc layers as 540 LocDiT calls
  each: 12,960 calls for the 7,212 of the run.

The evaluator (``est_saved_calls`` names the basis), the ``ttfa`` window share and the
instance groups of the projection tree (:mod:`kernel_agent.projection`) and the
scheduler's region arms use these weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

INSTANCE_GROUPS = "instance groups"
EVEN_SPLIT = "even split"


def case_weight(case: Mapping[str, Any], users: Mapping[str, Any], instances: int = 1) -> float:
    """Calls per run of the target's instances that one case stands for: its
    ``target_calls``, else (the even split) its calls per run (a capture's ``count``, an
    evaluation's ``calls_per_run``) × the instances calling its entrypoint (``users``:
    ``method_instances``; ``instances`` when it is not there)."""
    if case.get("target_calls") is not None:
        return float(case["target_calls"])
    count = case.get("count", case.get("calls_per_run"))
    method = str(case.get("method") or "forward")
    return float(count or 0) * float(users.get(method) or instances or 1)


def weights(capture: Mapping[str, Any]) -> list[float]:
    """:func:`case_weight` of every case of a capture (the file, or ``spec.json``
    ``capture``)."""
    users = capture.get("method_instances") or {}
    instances = int(capture.get("instances") or 1)
    return [case_weight(c, users, instances) for c in capture.get("cases") or []]


def basis(capture: Mapping[str, Any]) -> str:
    """``instance groups`` when every timed case of the capture knows its calls over the
    target's instances, else ``even split``."""
    timed = [c for c in capture.get("cases") or [] if c.get("count")]
    known = timed and all(c.get("target_calls") is not None for c in timed)
    return INSTANCE_GROUPS if known else EVEN_SPLIT


def _groups(capture: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """Instance group → (calls, covered calls); {} for an even split."""
    groups = capture.get("instance_groups")
    if basis(capture) != INSTANCE_GROUPS or not isinstance(groups, Mapping):
        return {}
    return {
        str(p): (float(g.get("calls") or 0), float(g.get("covered") or 0))
        for p, g in groups.items()
    }


def uncovered(capture: Mapping[str, Any]) -> float | None:
    """Calls of the target's instances (in its phase) with a primary input no case has
    (None: unknown, an even split)."""
    groups = _groups(capture)
    return sum(calls - covered for calls, covered in groups.values()) if groups else None


def coverage(capture: Mapping[str, Any]) -> float | None:
    """Share of the calls of the target's instances (in its phase) that a case stands for
    (None: unknown, an even split)."""
    groups = _groups(capture)
    calls = sum(n for n, _ in groups.values())
    return sum(c for _, c in groups.values()) / calls if calls else None


def group_calls(capture: Mapping[str, Any]) -> dict[str, float]:
    """Instance group (folded qualname) → the calls its cases stand for ({}: unknown)."""
    return {p: covered for p, (_, covered) in _groups(capture).items()}


def summary(capture: Mapping[str, Any], calls: float) -> dict[str, Any]:
    """``est_saved_calls`` of an evaluation: the ``basis``, the ``calls`` its timed cases
    stand for and (instance groups) the ``uncovered`` calls."""
    out: dict[str, Any] = {"basis": basis(capture), "calls": round(calls, 3)}
    if (rest := uncovered(capture)) is not None:
        out["uncovered"] = round(rest, 3)
    return out
