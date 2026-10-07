"""Template of a multi-file project candidate (``kernel_agent.native.project``, issue #134).

Layout::

    kernel_project.toml   name, entry, kind; build backend, sources (globs), flags
    candidate.py          this entry module: build(reference) for kind = "kernel"
    include/ka_native.cuh device helpers (conversions, warp sums, PDL wait / launch helpers)
    include/ops.h         host entry points of the project
    csrc/rmsnorm.cu       kernels and their launchers (one .cu per stage or op family)
    csrc/binding.cpp      PYBIND11_MODULE: the functions the entry module calls

``project.load(__file__)`` compiles the project once per content digest and toolchain
(``~/.cache/kernel-agent/native``) and returns it; its attributes are the bound functions.
``python -m kernel_agent.native.project check|build <dir>`` validates / compiles it without
a GPU evaluation. Evaluate the directory itself (``evaluate_candidate(candidate="<dir>")``):
the tool snapshots its bundle, one ``.py`` file holding every project file.

A project that replaces a stage, the stages of a loop iteration or the whole loop end to
end is a transform: ``kind = "transform"`` and an entry with ``apply(workload)``::

    def apply(workload) -> None:
        engine = project.load(__file__)
        model = workload.model
        model.stage = NativeStage(model.stage, engine)  # shares the stage's weights

and ``evaluate_e2e(transforms=["<dir>"])``.
"""

import torch
from torch import nn

from kernel_agent.native import project


class NativeRMSNorm(nn.Module):
    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.weight = reference.weight  # shared, not copied
        self.eps = float(getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-6)))
        self.ext = project.load(__file__)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ext.rmsnorm(hidden_states, self.weight, self.eps)


def build(reference: nn.Module) -> nn.Module:
    return NativeRMSNorm(reference)
