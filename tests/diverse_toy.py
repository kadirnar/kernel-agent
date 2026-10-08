"""A toy greedy "decoder" (CPU) with a diverse input set, for the #170 tests; also usable as
``harness.py``. Its runs cost simulated time (:class:`Clock`, which the tests put in place
of the ``time`` module of ``workloads.base``): ``step_ms`` per decode step, ``new_tokens``
steps per run. The tokens depend on the text only; a transform may take fewer steps
(speculative decoding) and must emit the same tokens.

The transforms of the tests (``apply(workload)``) are the module-level strings below."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads import base
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec


class Clock:
    """Simulated seconds: ``perf_counter`` for ``workloads.base``, advanced by the runs."""

    def __init__(self) -> None:
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now

    def spend(self, ms: float) -> None:
        self.now += ms / 1000


def spend(ms: float) -> None:
    """Advance the simulated clock (a no-op under the real ``time`` module)."""
    clock = base.time
    if hasattr(clock, "spend"):  # a Clock (of whichever copy of this module)
        clock.spend(ms)


TEXTS = {
    "prose": "a lighthouse keeper kept a log of every storm and every ship",
    "code": "a=a+1;b=b+2;" * 6,
    "table": "1,2,3\n1,2,3\n1,2,3\n1,2,3\n1,2,3\n1,2,3\n1,2,3\n1,2,3\n1,2,3",
}


def tokens_of(text: str, n: int) -> list[int]:
    """The toy model's greedy continuation of ``text``: deterministic in the content."""
    return [(ord(text[i % len(text)]) * 7 + i) % 101 for i in range(n)]


def repeats(text: str) -> float:
    """How much of ``text`` repeats (0..1): the acceptance of a prompt-lookup draft."""
    chunks = [text[i : i + 4] for i in range(0, len(text) - 3, 4)]
    return 1 - len(set(chunks)) / max(len(chunks), 1)


class DiverseToy(Workload):
    modality = Modality.LLM
    defaults = {
        "text": "the history of computing is a story of abstraction layers",
        "new_tokens": 16,
        "step_ms": 10.0,
    }

    def load(self) -> None:
        self.model = nn.Linear(4, 4)

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def diverse_inputs(self) -> dict[str, dict[str, Any]]:
        return {label: {"text": text} for label, text in TEXTS.items()}

    def make_inputs(self) -> str:
        return str(self.options["text"])

    def run(self, inputs: str) -> dict[str, Any]:
        n = int(self.options["new_tokens"])
        spend(float(self.options["step_ms"]) * n)
        return {"tokens": torch.tensor(tokens_of(inputs, n))}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        same = torch.equal(reference["tokens"], candidate["tokens"])
        return Comparison(same, {}, "" if same else "tokens differ")


def create(spec: WorkloadSpec) -> DiverseToy:
    return DiverseToy(spec)


#: A kernel-like transform: every step twice as fast, whatever the content.
FASTER = """
def apply(workload):
    workload.options["step_ms"] = float(workload.options["step_ms"]) / 2
"""

#: Exact speculative decoding: its steps depend on how much the text repeats; it reports
#: its counters.
SPECULATIVE = """
import math

import torch

from diverse_toy import repeats, spend, tokens_of


def apply(workload):
    def run(text):
        n = int(workload.options["new_tokens"])
        steps = max(1, math.ceil(n * (1 - repeats(text))))
        spend(float(workload.options["step_ms"]) * steps)
        workload.report_stats(
            steps=steps, verifies=steps, drafted=4 * steps, accepted=n - steps, tokens=n
        )
        return {"tokens": torch.tensor(tokens_of(text, n))}

    workload.run = run
"""

#: Twice as fast and wrong on code: a content-dependent bug the benchmark input misses.
WRONG_ON_CODE = """
import torch

from diverse_toy import spend, tokens_of


def apply(workload):
    def run(text):
        n = int(workload.options["new_tokens"])
        spend(float(workload.options["step_ms"]) * n / 2)
        tokens = tokens_of(text, n)
        if "a=a+1" in text:
            tokens[3] += 1
        return {"tokens": torch.tensor(tokens)}

    workload.run = run
"""
