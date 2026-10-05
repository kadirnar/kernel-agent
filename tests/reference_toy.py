"""The chaotic toy (CPU) with a ``reference_optimizations()`` hook, for the
strong-baseline tests.  Usable as ``harness.py``; ``-o reference=<mode>``:

* ``benign``: a one-rounding-step weight change (stands in for torch.compile's
  numerics), passes teacher forcing;
* ``broken``: a real change of the math, fails teacher forcing;
* ``none``: the hook declines; ``raise``: the hook fails.
"""

from __future__ import annotations

import torch
from chaotic_toy import ChaoticToy

from kernel_agent.workloads.base import Workload, WorkloadSpec


class ReferenceToy(ChaoticToy):
    defaults = {**ChaoticToy.defaults, "reference": "benign"}

    def reference_optimizations(self) -> str | None:
        mode = self.options["reference"]
        if mode == "none":
            return None
        if mode == "raise":
            raise RuntimeError("cannot compile the toy")
        with torch.no_grad():
            self.model.backbone.weight.mul_(1 + 2**-10 if mode == "benign" else 1.3)
        return f"toy reference optimisations ({mode})"


def create(spec: WorkloadSpec) -> Workload:
    return ReferenceToy(spec)
