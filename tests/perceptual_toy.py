"""The chaotic toy (``chaotic_toy.py``) with a perceptual gate, for the near-lossless and
relaxed tests.

Teacher-forcing thresholds are options (so ``near_lossless_options`` can loosen them), the
perceptual samples are two short runs with other seeds, and the "scorer" is cheap: the RMS
energy of every sample (a stand-in for WER / speaker similarity / MOS), which a correct
numerical change keeps and a broken one does not. Also usable as ``harness.py``.
"""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path
from typing import Any

from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, compare_steps

_spec = importlib.util.spec_from_file_location(
    "chaotic_toy", Path(__file__).with_name("chaotic_toy.py")
)
assert _spec is not None and _spec.loader is not None
_toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toy)


class PerceptualToy(_toy.ChaoticToy):
    defaults = {
        **_toy.ChaoticToy.defaults,
        "min_step_cosine": 0.999,
        "min_mean_step_cosine": 0.9999,
        "max_rms_change": 0.1,
    }
    near_lossless_options = {"min_step_cosine": 0.9, "min_mean_step_cosine": 0.95}
    # --quality relaxed (#175): twice the floor's budgets and twice the gate's
    relaxed_options = {"min_step_cosine": 0.8, "min_mean_step_cosine": 0.9, "max_rms_change": 0.2}

    def compare_teacher_forced(self, reference: Any, candidate: Any) -> Comparison:
        return compare_steps(
            reference["states"],
            candidate["states"],
            min_step_cosine=float(self.options["min_step_cosine"]),
            min_mean_step_cosine=float(self.options["min_mean_step_cosine"]),
        )

    def perceptual_samples(self) -> list[dict[str, Any]]:
        return [{"seed": seed, "steps": 40} for seed in (10, 11)]

    def perceptual_quality(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"rms": round(float(s["output"]["states"].pow(2).mean().sqrt()), 6), "steps": 40}
            for s in samples
        ]

    def compare_perceptual(
        self, reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
    ) -> Comparison:
        ref = statistics.fmean(r["rms"] for r in reference)
        new = statistics.fmean(c["rms"] for c in candidate)
        ratio = new / ref
        passed = abs(ratio - 1) <= float(self.options["max_rms_change"])
        reason = "" if passed else f"sample energy x{ratio:.3f}"
        return Comparison(passed, {"rms_ratio": round(ratio, 4)}, reason)


def create(spec: WorkloadSpec) -> Workload:
    return PerceptualToy(spec)
