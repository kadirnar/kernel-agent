"""The chaotic toy with a stop head (CPU), shaped like VoxCPM's decode loop: every
step computes the stop logits from the backbone state, and the run ends after the
first step ``i > min_steps`` whose argmax is "stop". The main run has a fixed length
(``min_steps`` = ``steps``), so only ``natural_length_run`` lets the stop head decide.
Also usable as ``harness.py``.
"""

from __future__ import annotations

from typing import Any

import torch
from chaotic_toy import ChaoticToy, ToyModel
from torch import nn

from kernel_agent.workloads.base import Workload, WorkloadSpec

#: The natural-length run: the stop fires at step ``STOP_AT`` (margin about +1).
STOP_AT = 12
NATURAL = {"steps": 40, "min_steps": 2}


class StopHead(nn.Module):
    """``[continue, stop]`` logits: a clock (the step) plus a little of the state."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, step: int) -> torch.Tensor:
        stop = 2.0 * (step - STOP_AT + 0.5) + 0.2 * torch.tanh(self.proj(h))[..., 0]
        return torch.stack([torch.zeros_like(stop), stop], -1)


class StopModel(ToyModel):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.stop_head = StopHead(dim)

    def generate(self, x: torch.Tensor, steps: int, min_steps: int | None = None) -> torch.Tensor:
        min_steps = steps if min_steps is None else min_steps
        for i in range(steps):
            h = torch.tanh(self.backbone(x))
            x = self.sampler(h)
            stop_flag = int(self.stop_head(h, i).argmax(-1).flatten()[0])
            if i > min_steps and stop_flag == 1:
                break
        return x


class StopToy(ChaoticToy):
    def load(self) -> None:
        super().load()
        dim = int(self.options["dim"])
        model = StopModel(dim)
        model.load_state_dict(self.model.state_dict(), strict=False)
        gen = torch.Generator().manual_seed(4321)
        with torch.no_grad():
            model.stop_head.proj.weight.copy_(torch.randn(1, dim, generator=gen) / dim**0.5)
            model.stop_head.proj.bias.zero_()
        self.model = model

    def run(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        torch.manual_seed(int(self.options["seed"]))
        states: list[torch.Tensor] = []

        def record(y: torch.Tensor) -> torch.Tensor:
            states.append(y.detach().clone())
            return y

        with self._hook(record):
            self.model.generate(inputs, int(self.options["steps"]), self.options.get("min_steps"))
        return {"states": torch.stack(states)}

    def run_teacher_forced(self, inputs: torch.Tensor, reference: Any) -> dict[str, torch.Tensor]:
        ref = reference["states"]
        preds: list[torch.Tensor] = []

        def force(y: torch.Tensor) -> torch.Tensor:
            preds.append(y.detach().clone())
            return ref[len(preds) - 1].to(y.dtype) if len(preds) <= len(ref) else y

        with self._hook(force):
            self.run(inputs)
        return {"states": torch.stack(preds)}

    def natural_length_run(self, reference: Any = None) -> dict[str, Any]:
        logits: list[torch.Tensor] = []
        hook = self.model.stop_head.register_forward_hook(
            lambda module, args, output: logits.append(output.detach().reshape(-1, 2)[0])
        )
        try:
            with self.with_options(NATURAL):
                inputs = self.make_inputs()
                if reference is None:
                    out = self.run(inputs)
                else:
                    out = self.run_teacher_forced(inputs, reference)
        finally:
            hook.remove()
        steps = len(out["states"])
        result: dict[str, Any] = {
            "steps": steps,
            "min_steps": NATURAL["min_steps"],
            "max_steps": NATURAL["steps"],
            "states": out["states"],
        }
        if reference is None and len(logits) == steps:
            result["stop_margins"] = [float(x[1] - x[0]) for x in logits]
        return result


def create(spec: WorkloadSpec) -> Workload:
    return StopToy(spec)
