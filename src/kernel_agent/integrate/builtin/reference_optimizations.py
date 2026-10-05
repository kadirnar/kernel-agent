"""Built-in transform: the workload's reference optimisations, i.e. the compiled
baseline of ``analyze`` (:mod:`kernel_agent.strong_baseline`).  The integration
applies it after the accepted kernels to see whether they compose with
``torch.compile`` / CUDA graphs."""

from __future__ import annotations

from typing import Any


def apply(workload: Any) -> None:
    from kernel_agent.strong_baseline import apply as apply_reference

    if apply_reference(workload) is None:
        raise RuntimeError("the workload has no reference optimisations")
