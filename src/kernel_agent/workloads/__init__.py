"""Workloads: deterministic end-to-end runs of a model, one per modality, plus
built-in workloads for model families whose inference code is not
``transformers``/``diffusers`` (``spec.family``)."""

from __future__ import annotations

from kernel_agent import objective
from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, measure

__all__ = [
    "Comparison",
    "Workload",
    "WorkloadSpec",
    "create_workload",
    "measure",
    "validate_metric",
]


def create_workload(spec: WorkloadSpec) -> Workload:
    """Instantiate (but do not load) the workload for ``spec``; ``ValueError`` when it
    cannot time the requested metric (``-o metric=``)."""
    workload = _create(spec)
    workload.check_metric()
    workload.check_serving()
    return workload


def validate_metric(spec: WorkloadSpec) -> None:
    """Fail fast, before a run starts, on a metric (``-o metric=``) that the built-in
    workload for ``spec`` cannot time (``ValueError``). A harness is checked when
    ``create_workload`` loads it."""
    metric = objective.get(spec.options.get("metric"))
    if spec.harness:
        objective.check(metric.name, tuple(objective.METRICS), "a harness")
        return
    try:
        workload = _create(spec)
    except ValueError:  # no built-in workload: the harness agent writes one
        objective.check(metric.name, tuple(objective.METRICS), "a harness")
        return
    workload.check_metric()
    workload.check_serving()


def _create(spec: WorkloadSpec) -> Workload:
    if spec.harness:
        from kernel_agent.workloads.harness import load_harness

        return load_harness(spec.harness, spec)
    if spec.family == "voxcpm":
        options = spec.options
        if "batch_size" in options or str(options.get("metric")).lower() == objective.THROUGHPUT:
            from kernel_agent.workloads.voxcpm_batch import VoxCPMBatchWorkload

            return VoxCPMBatchWorkload(spec)  # a batch of requests (metric=throughput)
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
