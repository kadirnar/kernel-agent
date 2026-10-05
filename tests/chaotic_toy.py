"""A tiny chaotic autoregressive workload (CPU) for the teacher-forcing tests.

Shaped like VoxCPM: a backbone feeds a sampler that adds fresh Gaussian noise
every step, and the sample is fed back.  The recurrence has a gain well above
one, so a one-rounding-step change diverges the free-running trajectory while
teacher-forced per-step predictions stay close.  Also usable as ``harness.py``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, compare_steps, cosine


class Sampler(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(h)) + 0.05 * torch.randn(h.shape, dtype=h.dtype)


class ToyModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.backbone = nn.Linear(dim, dim)
        self.sampler = Sampler(dim)

    def generate(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            x = self.sampler(torch.tanh(self.backbone(x)))
        return x


class ChaoticToy(Workload):
    modality = Modality.TTS
    supports_teacher_forcing = True
    defaults = {"steps": 60, "seed": 0, "dim": 32, "gain": 2.5}

    def load(self) -> None:
        dim = int(self.options["dim"])
        gen = torch.Generator().manual_seed(1234)
        self.model = ToyModel(dim)
        with torch.no_grad():
            for layer in (self.model.backbone, self.model.sampler.proj):
                weight = torch.randn(dim, dim, generator=gen) * float(self.options["gain"])
                layer.weight.copy_(weight / dim**0.5)
                layer.bias.zero_()

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> torch.Tensor:
        return torch.full((1, int(self.options["dim"])), 0.1)

    @contextlib.contextmanager
    def _hook(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
        sampler = self.model.sampler
        had_own = "forward" in vars(sampler)
        forward = sampler.forward
        vars(sampler)["forward"] = lambda *a, **k: fn(forward(*a, **k))
        try:
            yield
        finally:
            if had_own:
                vars(sampler)["forward"] = forward
            else:
                vars(sampler).pop("forward", None)

    def run(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        torch.manual_seed(int(self.options["seed"]))
        states: list[torch.Tensor] = []

        def record(y: torch.Tensor) -> torch.Tensor:
            states.append(y.detach().clone())
            return y

        with self._hook(record):
            self.model.generate(inputs, int(self.options["steps"]))
        return {"states": torch.stack(states)}

    def run_teacher_forced(self, inputs: torch.Tensor, reference: Any) -> dict[str, torch.Tensor]:
        ref = reference["states"]
        preds: list[torch.Tensor] = []

        def force(y: torch.Tensor) -> torch.Tensor:
            preds.append(y.detach().clone())
            return ref[len(preds) - 1].to(y.dtype)

        with self._hook(force):
            self.run(inputs)
        return {"states": torch.stack(preds)}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        cos = cosine(reference["states"][-10:], candidate["states"][-10:])
        passed = cos >= 0.99
        return Comparison(passed, {"tail_cosine": round(cos, 5)}, "" if passed else "diverged")

    def compare_teacher_forced(self, reference: Any, candidate: Any) -> Comparison:
        return compare_steps(
            reference["states"],
            candidate["states"],
            min_step_cosine=0.999,
            min_mean_step_cosine=0.9999,
        )


def create(spec: WorkloadSpec) -> Workload:
    return ChaoticToy(spec)
