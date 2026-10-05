"""Workloads: deterministic end-to-end runs of a model, one per modality, plus
built-in workloads for model families whose inference code is not
``transformers``/``diffusers`` (``spec.family``)."""

from __future__ import annotations

from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, measure

__all__ = ["Comparison", "Workload", "WorkloadSpec", "create_workload", "measure"]


def create_workload(spec: WorkloadSpec) -> Workload:
    """Instantiate (but do not load) the workload for ``spec``."""
    if spec.harness:
        from kernel_agent.workloads.harness import load_harness

        return load_harness(spec.harness, spec)
    if spec.family == "voxcpm":
        from kernel_agent.workloads.voxcpm import VoxCPMWorkload

        return VoxCPMWorkload(spec)
    modality = Modality(spec.modality)
    if modality is Modality.LLM:
        from kernel_agent.workloads.llm import LLMWorkload

        return LLMWorkload(spec)
    if modality is Modality.STT:
        from kernel_agent.workloads.stt import STTWorkload

        return STTWorkload(spec)
    if modality is Modality.TTS:
        from kernel_agent.workloads.tts import TTSWorkload

        return TTSWorkload(spec)
    if modality is Modality.DIFFUSION:
        from kernel_agent.workloads.diffusion import DiffusionWorkload

        return DiffusionWorkload(spec)
    raise ValueError(f"no built-in workload for modality {spec.modality!r}; a harness is required")
