"""Call phases of phase-specific targets: ``prefill`` vs ``decode``.

One module class can run in two regimes that want different kernels. VoxCPM's
``MiniCPMAttention.forward`` runs multi-position blocks (the LM prompt prefill,
the LocDiT and LocEnc layers), while ``forward_step`` decodes one position
against a static KV cache. A target with a ``phase`` captures only the calls of
that phase, and integration sends only those calls to its kernel
(:func:`route`). Every other call keeps the implementation it had, so two
targets on the same class, one per phase, combine.

* ``decode``: a call through an entrypoint whose name contains ``step``
  (``forward_step``, ``generate_step``, ``step``), or whose first tensor
  argument has at least 3 dims and one position (``[batch, 1, ...]``).
* ``prefill``: every other call (multi-position).

This file imports nothing from kernel_agent: ``optimized/`` ships a copy of it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

PHASES = ("prefill", "decode")


def call_phase(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """``"decode"`` or ``"prefill"`` for one entrypoint call (host-side shape check only)."""
    if "step" in method:
        return "decode"
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return "decode" if value.dim() >= 3 and value.shape[1] == 1 else "prefill"
    return "prefill"


def _router(
    name: str, phase: str, fast: Callable[..., Any], slow: Callable[..., Any]
) -> Callable[..., Any]:
    active = [False]  # a fallback of `fast` to the reference instance must not loop back

    def router(*args: Any, **kwargs: Any) -> Any:
        if active[0] or call_phase(name, args, kwargs) != phase:
            return slow(*args, **kwargs)
        active[0] = True
        try:
            return fast(*args, **kwargs)
        finally:
            active[0] = False

    router.ka_phase = phase  # type: ignore[attr-defined]
    return router


def route(instance: nn.Module, new: nn.Module, phase: str, methods: Iterable[str]) -> list[str]:
    """Send the ``phase`` calls of ``methods`` on ``instance`` to the same methods of ``new``.

    ``instance`` stays in the model (same class, same qualname, so a target for
    the other phase still finds it); each method becomes an instance attribute
    that dispatches per call. Calls of the other phase go to what the method
    was before (the class method, or the router of a target for the other
    phase). Returns the routed method names."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    if new is not instance:
        # `copy.copy(reference)` in build() copies routers of an earlier phase target.
        for key, value in list(vars(new).items()):
            if getattr(value, "ka_phase", None) is not None:
                del vars(new)[key]
    routed = []
    for name in dict.fromkeys(methods):
        fast, slow = getattr(new, name, None), getattr(instance, name, None)
        if callable(fast) and callable(slow):
            object.__setattr__(instance, name, _router(name, phase, fast, slow))
            routed.append(name)
    return routed
