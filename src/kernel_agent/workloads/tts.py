"""Text-to-speech through the ``transformers`` text-to-audio pipeline.

Covers VITS/MMS, Bark, MusicGen and similar.  Most modern TTS models ship
custom inference code; for those the orchestrator asks Claude to write a
``harness.py`` workload instead (see :mod:`kernel_agent.workloads.harness`).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, compare_audio

TEXT = (
    "Kernel level optimisation makes speech synthesis fast enough for real time "
    "conversation, even on a single consumer graphics card."
)
#: Held-out e2e input (``holdout_options``) and a short extra capture setting (``variants``).
HOLDOUT_TEXT = (
    "Every sentence a listener hears must be synthesised from scratch, never replayed "
    "from an earlier run, whatever words it happens to contain."
)
SHORT_TEXT = "Short sentences matter too."


class TTSWorkload(Workload):
    modality = Modality.TTS
    defaults = {"text": TEXT, "seed": 0, "min_spec_cosine": 0.97}

    def load(self) -> None:
        from transformers import pipeline

        self.pipe = pipeline(
            "text-to-audio",
            model=self.spec.repo_id,
            revision=self.spec.revision,
            trust_remote_code=self.spec.trust_remote_code,
            dtype=self.dtype,
            device=self.device,
        )

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.pipe.model}

    def holdout_options(self, variant: int = 1) -> dict[str, Any] | None:
        """Another sentence; variants ``>= 2`` change only the seed."""
        return {"text": HOLDOUT_TEXT, "seed": int(self.options["seed"]) + variant}

    def variants(self) -> list[dict[str, Any]]:
        return [{"text": SHORT_TEXT}]

    def make_inputs(self) -> str:
        return str(self.options["text"])

    def run(self, inputs: str) -> dict[str, Any]:
        torch.manual_seed(int(self.options["seed"]))
        out = self.pipe(inputs)
        audio = torch.as_tensor(out["audio"]).float().flatten()
        return {"audio": audio, "sampling_rate": int(out["sampling_rate"])}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return compare_audio(
            reference["audio"],
            candidate["audio"],
            min_spec_cosine=float(self.options["min_spec_cosine"]),
        )
