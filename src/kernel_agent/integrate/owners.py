"""Which modules each integration item changes, and which items are alternatives (#112).

The patcher records while it applies an item (``PatchReport.touched`` / ``owns``, per
kernel target and per transform file stem):

* ``touched``: the modules whose bindings it changed (:meth:`Snapshot.where
  <kernel_agent.integrate.undo.Snapshot.where>`: an attribute, a hook, a parameter or
  buffer, the class, a class attribute, a child), ``workload.<name>`` for an attribute of
  the workload object (its ``lm_step``, say);
* ``owns``: the modules it replaced: of a kernel every module it replaces or routes, of a
  transform the children it swapped for others and the instances whose class it changed.
  Whatever lies inside a module an item replaced is that item's.

Two items *overlap* when they touch the same module, or one of them touches something
inside a module the other owns: a kernel of the LocEnc and a transform that CUDA-graphs the
LocEnc's ``forward``; a VAE decoder kernel and a transform that converts the decoder's
weights. Such items are alternatives: one cannot be stacked on the other (the transform
finds the kernel's module where it expects the original, or the kernel replaces what the
transform changed), so the integration also tries the new item in their place (a
``replace`` step of ``orchestrator.integrate``). A module that merely contains another
item's (a CUDA graph of the solver around the DiT layers a kernel replaces) is no overlap:
the container still calls what is inside.

Names are compact: list indices become ``*`` (``layers.*.mlp`` for the MLPs of every
layer), so a record stays small and a kernel of ``layers.*`` overlaps a transform of
``layers.*.mlp``."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kernel_agent import ledger

_INDEX = re.compile(r"(?<=\.)\d+(?=\.|$)")


def compact(names: Iterable[str]) -> list[str]:
    """Sorted unique ``names`` with list indices as ``*``."""
    return sorted({_INDEX.sub("*", name) for name in names})


def inside(name: str, module: str) -> bool:
    """Whether ``name`` is ``module`` or lies inside it."""
    return name == module or name.startswith(module + ".")


def report_key(kind: str, arg: str) -> str:
    """An integration item's key in a ``PatchReport``: a kernel's target, a transform's
    file stem."""
    return arg.partition("=")[0] if kind == "kernel" else Path(arg).stem


def topmost(names: Iterable[str]) -> list[str]:
    """``names`` without those inside another of them."""
    names = sorted(set(names))
    return [n for n in names if not any(m != n and inside(n, m) for m in names)]


class Owners:
    """What the integration's measurements recorded of each item (the ``patches`` of a
    worker result: B's in ``patches``, A's in ``ab.a_patches``), by item."""

    def __init__(self) -> None:
        self.touched: dict[str, set[str]] = {}
        self.owns: dict[str, set[str]] = {}

    def note(self, items: list[tuple[str, str]], patches: dict[str, Any] | None) -> None:
        """Take what one state's ``patches`` recorded of its ``items`` (merged with what
        other measurements recorded of them: an item can touch more on top of others)."""
        touched = (patches or {}).get("touched") or {}
        owns = (patches or {}).get("owns") or {}
        for kind, arg in items:
            key = report_key(kind, arg)
            if key in touched:
                self.touched.setdefault(arg, set()).update(touched[key])
                self.owns.setdefault(arg, set()).update(owns.get(key) or [])

    def shared(self, x: str, y: str) -> list[str]:
        """Where items ``x`` and ``y`` overlap: the modules both touch, and those one touches
        inside a module the other owns (empty: no overlap, or one of them is unknown)."""
        tx, ty = self.touched.get(x, set()), self.touched.get(y, set())
        ox, oy = self.owns.get(x, set()), self.owns.get(y, set())
        out = tx & ty
        out |= {t for t in ty if any(inside(t, o) for o in ox)}
        out |= {t for t in tx if any(inside(t, o) for o in oy)}
        return sorted(out)

    def overlapping(
        self, item: tuple[str, str], others: list[tuple[str, str]]
    ) -> list[tuple[tuple[str, str], list[str]]]:
        """The ``others`` that ``item`` overlaps, with where."""
        return [(o, s) for o in others if o != item and (s := self.shared(item[1], o[1]))]

    def record(self, items: list[tuple[str, str]]) -> dict[str, dict[str, list[str]]]:
        """``integration.json`` ``owners``: what each of ``items`` touches and owns."""
        return {
            arg: {"touched": sorted(self.touched[arg]), "owns": sorted(self.owns.get(arg, ()))}
            for _, arg in items
            if arg in self.touched
        }


def context_line(owners: dict[str, dict[str, list[str]]] | None, *, limit: int = 4) -> str:
    """The re-plan context's line on the modules the accepted transforms change ("" when
    there are none): a kernel of such a module is tried in the transform's place."""
    parts = []
    for item, rec in (owners or {}).items():
        target, sep, _ = item.partition("=")
        if (sep and "/" not in target) or not rec.get("touched"):
            continue  # a kernel: its target is listed already
        names = topmost(rec["touched"])
        more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
        listed = ", ".join(f"`{n}`" for n in names[:limit])
        parts.append(f"`{ledger.item_label(item)}`: {listed}{more}")
    if not parts:
        return ""
    return (
        "Modules the accepted transforms change: " + "; ".join(parts) + ". A kernel whose "
        "module is one of these, or contains one, is no addition on top of that transform: "
        "the integration tries it in the transform's place, where it must beat the transform."
    )
