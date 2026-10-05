"""Two optimised states of one loaded model, for the paired A/B of ``worker e2e_ab``.

Each state is built the way a fresh ``e2e`` process applies it (kernels, then
transforms, in order) with undo handles (:mod:`kernel_agent.integrate.undo`):

* kernels are shared when B's kernels extend A's and the extra ones bring no
  region rewrite (rewrites must come before every kernel);
* transforms are applied afresh for each state, so what one state captured or
  compiled lazily (CUDA graphs on the first call, say) never serves the other.

:meth:`Session.to` switches the model between states: it undoes the applied
handles the target state does not share (newest first) and redoes its own,
with the lazily-built state of each intact (no warm-up needed), or applies a
transform again when it defines its own ``undo()``.
"""

from __future__ import annotations

import statistics
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from kernel_agent.integrate.patcher import KernelPatch, PatchReport, apply_kernels, apply_transforms
from kernel_agent.integrate.undo import Undo
from kernel_agent.workloads.base import synchronize
from kernel_agent.workloads.holdout import outputs_equal


@dataclass
class State:
    name: str
    kernels: list[str]  # TARGET_ID=PATH
    transforms: list[str]
    handles: list[Undo] = field(default_factory=list)
    #: Leading handles shared with the state it was built on (never undone by a switch).
    shared: int = 0
    report: PatchReport = field(default_factory=PatchReport)
    #: handle id -> what it applied (kernel items, or the transform path)
    items: dict[int, list[str]] = field(default_factory=dict)

    def irreversible(self, keep: int | None = None) -> list[tuple[list[str], str]]:
        """(items, why) of the handles after the first ``keep`` (default: the shared ones)
        that a switch would have to undo but cannot."""
        start = self.shared if keep is None else keep
        return [
            (self.items.get(id(h), [h.label]), "; ".join(h.problems))
            for h in self.handles[start:]
            if not h.reversible
        ]


class Session:
    def __init__(self, workload: Any, patches: Callable[[list[str]], list[KernelPatch]]) -> None:
        self.workload = workload
        self.patches = patches  # TARGET_ID=PATH items -> KernelPatch list
        self.applied: list[Undo] = []

    def shareable(self, on: State, kernels: list[str]) -> int:
        """Handles of ``on`` a state with ``kernels`` shares: its kernel handle (1) when
        ``kernels`` extend ``on``'s and the extra ones bring no region rewrite, else 0."""
        if not on.kernels or kernels[: len(on.kernels)] != on.kernels:
            return 0
        return 0 if any(p.rewrite for p in self.patches(kernels[len(on.kernels) :])) else 1

    def build(
        self, name: str, kernels: list[str], transforms: list[str], on: State | None = None
    ) -> State:
        """Apply a state (kernels, then transforms) on the pristine model, or on ``on``'s
        kernels when it can share them (:meth:`shareable`). A failure propagates with the
        partially applied handles in ``self.applied``."""
        state = State(name, list(kernels), list(transforms))
        own = list(kernels)
        if on is not None and (shared := self.shareable(on, kernels)):
            state.handles, state.shared = on.handles[:shared], shared
            own = own[len(on.kernels) :]
            for key in ("replaced", "skipped", "rewritten"):
                getattr(state.report, key).update(getattr(on.report, key))
            state.report.errors += on.report.errors
        self.to(state)
        try:
            if own:
                roots = self.workload.roots()
                apply_kernels(roots, self.patches(own), state.report, handles=state.handles)
                state.items[id(state.handles[-1])] = own
            for path in transforms:
                apply_transforms(self.workload, [Path(path)], state.report, handles=state.handles)
                state.items[id(state.handles[-1])] = [path]
        finally:
            self.applied = list(state.handles)
        return state

    def to(self, state: State) -> bool:
        """Make ``state`` (its handles so far) the model's state. True when a transform was
        applied again: the next run is a warm-up."""
        target = list(state.handles)
        n = 0
        while n < min(len(self.applied), len(target)) and self.applied[n] is target[n]:
            n += 1
        for handle in reversed(self.applied[n:]):
            handle.undo()
        self.applied = self.applied[:n]
        again = False
        for i in range(n, len(target)):
            handle = target[i]
            if handle.reapply:
                handle = self._reapply(state, handle)
                again = True
            else:
                handle.redo()
            self.applied.append(handle)
        return again

    def _reapply(self, state: State, old: Undo) -> Undo:
        """Apply a transform with its own ``undo()`` again; its new handle replaces ``old``."""
        (path,) = state.items.pop(id(old))
        handles: list[Undo] = []
        apply_transforms(self.workload, [Path(path)], PatchReport(), handles=handles)
        state.handles[state.handles.index(old)] = handles[0]
        state.items[id(handles[0])] = [path]
        return handles[0]


def timed_run(workload: Any, inputs: Any) -> tuple[Any, float]:
    """One synchronised end-to-end run: (output, ms)."""
    with torch.inference_mode():
        synchronize()
        start = time.perf_counter()
        output = workload.run(inputs)
        synchronize()
    return output, (time.perf_counter() - start) * 1000


@dataclass
class Rounds:
    """Timed rounds of two states; outputs checked against each state's warm-up output."""

    a_ms: list[float] = field(default_factory=list)
    b_ms: list[float] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    output: Any = None  # B's last output
    #: per state: its warm-up output was bit-identical twice, so every round must match it
    reproducible: dict[str, bool] = field(default_factory=dict)
    mismatch: str | None = None
    failed: str | None = None  # the state whose run raised
    error: str | None = None

    def ab(self) -> dict[str, Any]:
        return {
            "a_ms": [round(t, 3) for t in self.a_ms],
            "b_ms": [round(t, 3) for t in self.b_ms],
            "order": " ".join(self.order),
            "undo_check": {
                name: "identical" if ok else "skipped: the output is not bit-reproducible"
                for name, ok in self.reproducible.items()
            },
        }


def warm(session: Session, state: State, inputs: Any, runs: int) -> tuple[Any, bool]:
    """``runs`` (>= 2) untimed runs of ``state``: (last output, whether the last two
    outputs were bit-identical)."""
    session.to(state)
    outputs = [timed_run(session.workload, inputs)[0] for _ in range(max(runs, 2))]
    return outputs[-1], outputs_equal(outputs[-2], outputs[-1])


def alternate(
    session: Session,
    states: tuple[State, State],
    inputs: Any,
    reference: dict[str, Any],
    reproducible: dict[str, bool],
    *,
    rounds: int,
    sample: Callable[[str], Any] | None = None,
) -> Rounds:
    """``rounds`` rounds of one run per state, A first in even rounds and B first in odd
    ones. Stops at the first output of a reproducible state that differs from its warm-up
    output (``mismatch``: a switch did not restore the state) or at an error (``failed``)."""
    a, b = states
    out = Rounds(reproducible=dict(reproducible))
    for i in range(rounds):
        for state in (a, b) if i % 2 == 0 else (b, a):
            try:
                if session.to(state):
                    timed_run(session.workload, inputs)  # warm-up of a transform applied again
                output, ms = timed_run(session.workload, inputs)
            except Exception:
                out.failed, out.error = state.name, traceback.format_exc()[-4000:]
                return out
            if sample is not None:
                sample(state.name)
            (out.a_ms if state is a else out.b_ms).append(ms)
            out.order.append(state.name)
            if state is b:
                out.output = output
            if reproducible.get(state.name) and not outputs_equal(reference[state.name], output):
                other = b.name if state is a else a.name
                out.mismatch = (
                    f"the output of {state.name} changed after switching from {other} (round "
                    f"{i + 1}): the in-process undo is incomplete"
                )
                return out
    return out


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")
