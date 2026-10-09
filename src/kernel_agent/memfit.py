"""Memory preflight (#255): whether a run's model fits the GPU, said before analyze loads it.

``ModelCard.size_gb`` (the checkpoint's weight files) scaled to the run's dtype is the
weights' size; the workload's footprint is that times a factor learnt from the runs recorded
in the runs directory (``baseline.json`` ``peak_mem_gb`` over the weights of a run of the
same model, metric and batch size: VoxCPM2 1.29 for latency, 2.46 for throughput at batch
16), else :data:`DEFAULT_FACTORS`. The GPU's memory (``GPUInfo.memory_gb``, or
``KERNEL_AGENT_EMULATE_MEM_GB`` when smaller) minus :data:`OVERHEAD_GB` is what it may use.

* :func:`preflight`: ``run.json`` → ``memory_fit``: the estimate, whether it fits, whether
  float32 would (the reference of the float16 check and its fallback, ``workloads/dtypes.py``)
  and whether the integration's in-process A/B fits (two states of the model in one
  process; when it does not, every A/B starts in two processes, ``Orchestrator._paired``).
* :func:`refusal`: what ``analyze`` says, before loading, when the model does not fit, with
  the options (8-bit weight storage, a smaller batch, half precision, ``--no-memory-check``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kernel_agent.workloads.dtypes import BFLOAT16, BYTES, FLOAT32, canonical, checkpoint_dtype

#: The footprint (baseline peak / weights) of a model and metric without a recorded run:
#: measured on VoxCPM2 (RTX 5070 Ti): latency 5.48 GB / 4.27 GB of bf16 weights = 1.29
#: (ttfa likewise), throughput at batch 16 10.51 / 4.27 = 2.46 (Qwen3-0.6B, latency: 0.9).
DEFAULT_FACTORS = {"latency": 1.29, "ttfa": 1.29, "throughput": 2.46}
#: GPU memory a worker holds outside torch's allocations: CUDA context and library handles
#: (0.25 GB measured on an RTX 5070 Ti after a cuBLAS GEMM and SDPA) plus the caching
#: allocator's slack.
OVERHEAD_GB = 0.5
#: The in-process A/B of the integration holds two states of the model: each state's own
#: copies (an FP8 weight copy is half the bf16 weights; CUDA graph pools), about one more
#: weight-sized copy for both (#112, #137: a 17-transform composite on VoxCPM2 did not fit
#: in 16 GB).
AB_EXTRA_WEIGHTS = 1.0
#: The GPU memory to assume instead of the real one when smaller (``--emulate-arch``'s
#: memory, issue #252).
EMULATE_ENV = "KERNEL_AGENT_EMULATE_MEM_GB"


def gpu_memory_gb(gpu: Any) -> float | None:
    """The memory of ``gpu`` (``toolchain.GPUInfo``; None: unknown), capped by
    :data:`EMULATE_ENV`."""
    total = getattr(gpu, "memory_gb", None)
    emulated = os.environ.get(EMULATE_ENV)
    if emulated:
        cap = float(emulated)
        total = cap if not isinstance(total, int | float) else min(float(total), cap)
    return float(total) if isinstance(total, int | float) else None


def weights_gb(size_gb: float | None, checkpoint: str | None, runtime: str) -> float | None:
    """The weights of a checkpoint of ``size_gb`` in ``runtime`` (an unstated checkpoint
    dtype is taken as bfloat16, the usual one)."""
    if not size_gb:
        return None
    stored = BYTES.get(canonical(checkpoint) or BFLOAT16, 2)
    return float(size_gb) * BYTES.get(canonical(runtime) or BFLOAT16, 2) / stored


def _metric(options: Mapping[str, Any] | None) -> str:
    return str((options or {}).get("metric") or "latency").lower()


def footprint_factor(
    runs_dir: Path | None, repo_id: str, options: Mapping[str, Any] | None
) -> tuple[float, str]:
    """``(peak / weights, where it comes from)`` of ``repo_id`` under workload ``options``:
    the largest of the runs recorded in ``runs_dir`` with the same model, metric and
    ``batch_size`` (each run's ``peak_mem_gb`` over its weights in the dtype it ran in),
    else :data:`DEFAULT_FACTORS` of the metric."""
    metric, batch = _metric(options), (options or {}).get("batch_size")
    found: list[tuple[float, str]] = []
    root = Path(runs_dir) if runs_dir else None
    for run_json in sorted(root.glob("*/*/run.json")) if root and root.is_dir() else []:
        ratio = _recorded(run_json, repo_id, metric, batch)
        if ratio is not None:
            found.append((ratio, run_json.parent.name))
    if found:
        best, where = max(found)
        return best, f"measured: {len(found)} recorded run(s) of this model and metric ({where})"
    default = DEFAULT_FACTORS.get(metric, DEFAULT_FACTORS["latency"])
    return default, "default: no recorded run of this model and metric (measured on VoxCPM2)"


def _recorded(run_json: Path, repo_id: str, metric: str, batch: Any) -> float | None:
    """Peak over weights of one recorded run when it matches (else None)."""
    import json

    try:
        data = json.loads(run_json.read_text())
        baseline = json.loads((run_json.parent / "baseline.json").read_text())
    except (OSError, ValueError):
        return None
    card, workload = data.get("card") or {}, data.get("workload") or {}
    options = workload.get("options") or {}
    if "dry_run" in data or card.get("repo_id") != repo_id or _metric(options) != metric:
        return None
    if options.get("batch_size") != batch:
        return None
    choice = data.get("dtype") or {}
    ckpt = choice.get("checkpoint") or checkpoint_dtype(card.get("config"))
    runtime = choice.get("runtime") or workload.get("dtype") or BFLOAT16
    weights = weights_gb(card.get("size_gb"), ckpt, runtime)
    peak = baseline.get("peak_mem_gb")
    if not weights or not isinstance(peak, int | float) or peak <= 0:
        return None
    return float(peak) / weights


def preflight(
    card: Mapping[str, Any],
    workload: Mapping[str, Any],
    runtime: str,
    gpu_gb: float | None,
    runs_dir: Path | None = None,
    *,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """``run.json`` → ``memory_fit`` of a run of ``card`` (``ModelCard.to_dict()``) with
    ``workload`` (its recorded spec) in ``runtime`` on a GPU of ``gpu_gb``: ``status`` ``ok``
    with ``estimate_gb``, ``fits``, ``fp32_fits``, ``ab_estimate_gb`` and ``ab_in_process``,
    or ``unknown`` (no checkpoint size or GPU memory: everything is assumed to fit)."""
    ckpt = checkpoint or checkpoint_dtype(card.get("config"))
    weights = weights_gb(card.get("size_gb"), ckpt, runtime)
    if weights is None or not gpu_gb:
        what = "the checkpoint size" if weights is None else "the GPU memory"
        return {
            "status": "unknown",
            "why": f"{what} is unknown",
            "runtime": runtime,
            "fits": True,
            "fp32_fits": True,
            "ab_in_process": True,
        }
    options = workload.get("options") or {}
    factor, source = footprint_factor(runs_dir, str(card.get("repo_id")), options)
    usable = max(float(gpu_gb) - OVERHEAD_GB, 0.0)
    estimate = weights * factor
    fp32 = (weights_gb(card.get("size_gb"), ckpt, FLOAT32) or weights) * factor
    ab = estimate + weights * AB_EXTRA_WEIGHTS
    fit: dict[str, Any] = {
        "status": "ok",
        "runtime": runtime,
        "gpu_gb": round(float(gpu_gb), 2),
        "usable_gb": round(usable, 2),
        "weights_gb": round(weights, 2),
        "factor": round(factor, 3),
        "factor_source": source,
        "estimate_gb": round(estimate, 2),
        "fits": estimate <= usable,
        "fp32_estimate_gb": round(fp32, 2),
        "fp32_fits": fp32 <= usable,
        "ab_estimate_gb": round(ab, 2),
        "ab_in_process": ab <= usable,
    }
    if not fit["ab_in_process"]:
        fit["ab_why"] = (
            f"two states of the model need about {ab:.1f} GB of the {usable:.1f} GB usable"
        )
    if not fit["fits"]:
        fit["options"] = options_for(fit, options)
    return fit


def options_for(fit: Mapping[str, Any], options: Mapping[str, Any]) -> list[str]:
    """What the user can do when the model does not fit."""
    weights = float(fit.get("weights_gb") or 0)
    found = [
        f"8-bit weight storage (weight-only INT8 or FP8, about {weights / 2:.1f} GB of "
        "weights): an INT8 / FP8 checkpoint of the model",
    ]
    if fit.get("runtime") == FLOAT32:
        found.append("--dtype float16 or bfloat16 (float32 doubles the weights)")
    batch = options.get("batch_size")
    if batch is not None or _metric(options) == "throughput":
        now = f" (now {batch})" if batch is not None else ""
        found.append(f"a smaller batch: -o batch_size=N{now}")
    else:
        found.append("a smaller workload variant (fewer tokens, patches or steps: -o ...)")
    found += [
        "a GPU with more memory",
        "--no-memory-check: try anyway (the estimate is approximate)",
    ]
    return found


def refusal(fit: Mapping[str, Any]) -> str:
    """The message of a run whose model does not fit (``fit["fits"]`` False)."""
    lines = [
        f"memory preflight: the model in {fit.get('runtime')} needs about "
        f"{fit.get('estimate_gb')} GB ({fit.get('weights_gb')} GB of weights x footprint "
        f"{fit.get('factor')}, {fit.get('factor_source')}); this GPU has {fit.get('gpu_gb')} "
        f"GB ({fit.get('usable_gb')} GB usable). Options:",
        *(f"  * {o}" for o in fit.get("options") or ()),
    ]
    return "\n".join(lines)


def describe(fit: Mapping[str, Any]) -> str:
    """One line for logs and ``report.md``."""
    if fit.get("status") != "ok":
        return f"not estimated ({fit.get('why')})"
    verdict = "fits" if fit.get("fits") else "does NOT fit"
    ab = "in one process" if fit.get("ab_in_process") else "in two processes"
    return (
        f"{fit.get('estimate_gb')} GB estimated in {fit.get('runtime')} ({fit.get('weights_gb')} "
        f"GB of weights x {fit.get('factor')}) of {fit.get('usable_gb')} GB usable: {verdict}; "
        f"integration A/B {ab}"
    )
