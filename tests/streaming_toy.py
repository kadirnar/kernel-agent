"""The perceptual toy (``perceptual_toy.py``) as a streaming TTS: one audio chunk per step,
``-o metric=ttfa`` timed to the first one. Also usable as ``harness.py``.

Every step takes simulated time (``base.time``: put a ``fake_clock.Clock`` in its place) and
every request appends the steps it generated to the file ``-o requests_log=PATH``, so a test
sees what each measurement costs: a request inside the metric's window stops at its first
chunk (:attr:`~kernel_agent.workloads.base.Workload.in_window`), a whole one streams every
step.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import torch

from kernel_agent import objective
from kernel_agent.workloads import base
from kernel_agent.workloads.base import Workload, WorkloadSpec

_spec = importlib.util.spec_from_file_location(
    "perceptual_toy", Path(__file__).with_name("perceptual_toy.py")
)
assert _spec is not None and _spec.loader is not None
_toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toy)

#: Simulated seconds of the first step (the prompt prefill) and of every later one.
FIRST_S, STEP_S = 0.04, 0.01


def requests(log: Path) -> list[int]:
    """The steps of every request logged to ``log``, in order."""
    return [int(x) for x in log.read_text().split()] if log.exists() else []


class StreamingToy(_toy.PerceptualToy):
    metrics = (objective.LATENCY, objective.TTFA)

    def _steps(self, x: torch.Tensor, steps: int) -> Iterator[torch.Tensor]:
        model = self.model
        for i in range(steps):
            base.time.sleep(STEP_S if i else FIRST_S)
            x = model.sampler(torch.tanh(model.backbone(x)))
            yield x

    def run(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        torch.manual_seed(int(self.options["seed"]))
        states: list[torch.Tensor] = []

        def record(y: torch.Tensor) -> torch.Tensor:
            states.append(y.detach().clone())
            return y

        with self._hook(record):
            for _ in self._steps(inputs, int(self.options["steps"])):
                self.mark_chunk(audio_ms=40.0)
                if self.in_window:  # the metric's value is known: the client stops listening
                    break
        if log := self.options.get("requests_log"):
            with open(log, "a") as fh:
                fh.write(f"{len(states)}\n")
        return {"states": torch.stack(states)}


def create(spec: WorkloadSpec) -> Workload:
    return StreamingToy(spec)
