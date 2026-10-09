"""The dtype a run's model runs in (#255): the checkpoint's where the GPU has tensor cores
for it, float16 where it has none for bfloat16, after a measured check.

``--dtype auto`` (the default of new runs) decides it from the checkpoint and the GPU:

* :func:`checkpoint_dtype`: what the checkpoint stores (``config.json`` ``torch_dtype`` or
  ``dtype``; None when it does not say).
* :func:`runtime_dtype`: the dtype a run starts from on a compute capability. A float16
  checkpoint runs in float16 everywhere. A bfloat16 one runs in bfloat16 on a GPU with bf16
  tensor cores (``gpu_arch`` feature ``bf16_tc``: sm_80+). Below sm_80 (Turing) bf16 has no
  tensor cores: Triton's bf16 ``tl.dot`` becomes FMA, SDPA's flash and mem-efficient
  backends refuse bf16 there and the math backend runs, so every speedup would be measured
  against a crippled baseline (T4 datasheet: 65 TFLOP/s fp16 tensor cores, 8.1 fp32). The
  run's candidate is then float16, which overflows where bfloat16 does not, so it is only a
  candidate (``check`` :data:`PENDING`). A float32 checkpoint (or one that does not say)
  runs in half precision as kernel-agent always ran it: bfloat16, or the float16 candidate.
* :func:`resolve`: ``--dtype``: ``auto`` as above, an explicit dtype as given (an explicit
  bfloat16 on a GPU without bf16 tensor cores is kept, with a warning recorded).
* :func:`check`: ``analyze``'s check of the candidate (``worker dtype_check``): the
  workload runs once in the reference dtype (float32 where it fits, ``memfit.py``) and once
  in float16 on its inputs. float16 passes when its outputs are finite where the
  reference's are and within the workload's relaxed bounds of the reference (its own
  comparison, teacher forcing for chaotic workloads, and in near-lossless / relaxed runs
  the perceptual gate); :func:`settle` then keeps float16, or takes float32 when it fits
  (else the checkpoint's dtype), and says why.

``run.json`` → ``dtype`` records ``requested``, ``checkpoint``, ``runtime``, ``why`` (and
``check``, ``reference``, ``warning``); ``workload.dtype`` is the runtime dtype every worker
loads (:meth:`Workload.set_dtype <kernel_agent.workloads.base.Workload.set_dtype>`). A run
from before #255 has no ``dtype`` section and keeps the dtype it recorded.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kernel_agent import gpu_arch

if TYPE_CHECKING:
    from kernel_agent.workloads.base import Workload

AUTO = "auto"
FLOAT16, BFLOAT16, FLOAT32 = "float16", "bfloat16", "float32"
#: The canonical name of every spelling of a dtype a run can use (configs, the CLI).
NAMES = {
    "float16": FLOAT16,
    "fp16": FLOAT16,
    "half": FLOAT16,
    "bfloat16": BFLOAT16,
    "bf16": BFLOAT16,
    "float32": FLOAT32,
    "fp32": FLOAT32,
    "float": FLOAT32,
}
#: Bytes per element (``memfit.py``).
BYTES = {FLOAT16: 2, BFLOAT16: 2, FLOAT32: 4}
#: ``run.json`` → ``dtype.check`` of a float16 candidate until analyze has checked it.
PENDING = "pending"


def canonical(name: Any) -> str | None:
    """``float16`` / ``bfloat16`` / ``float32`` for any spelling (``bf16``,
    ``torch.float16``, ...); None for anything else (``auto``, None, ``int8``)."""
    text = str(name or "").strip().lower().removeprefix("torch.")
    return NAMES.get(text)


def checkpoint_dtype(config: Mapping[str, Any] | None) -> str | None:
    """The dtype a checkpoint stores: its ``config.json`` ``torch_dtype`` (``dtype`` in newer
    transformers and in VoxCPM's config); None when it does not say."""
    for key in ("torch_dtype", "dtype"):
        if (found := canonical((config or {}).get(key))) is not None:
            return found
    return None


@dataclass
class DtypeChoice:
    """The dtype of a run (``run.json`` → ``dtype``)."""

    runtime: str
    why: str
    checkpoint: str | None = None
    requested: str = AUTO
    #: :data:`PENDING`: ``candidate`` (float16) is checked by ``analyze`` against a reference
    #: run; afterwards the check's record (:func:`settle`).
    check: str | None = None
    candidate: str | None = None
    warning: str | None = None
    arch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "requested": self.requested,
            "checkpoint": self.checkpoint,
            "runtime": self.runtime,
            "why": self.why,
        }
        extra = {
            "check": self.check,
            "candidate": self.candidate,
            "warning": self.warning,
            "arch": self.arch,
        }
        return {**data, **{k: v for k, v in extra.items() if v is not None}}


def _no_bf16_cores(capability: tuple[int, ...] | None) -> bool:
    return capability is not None and not gpu_arch.has(capability, "bf16_tc")


def runtime_dtype(checkpoint: str | None, capability: tuple[int, ...] | None) -> DtypeChoice:
    """``--dtype auto``: the dtype a checkpoint storing ``checkpoint`` runs in on a GPU of
    ``capability`` (None: no GPU known). See the module docstring."""
    ckpt = canonical(checkpoint)
    arch = gpu_arch.arch_of(tuple(capability[:2])) if capability else None
    if ckpt == FLOAT16:
        return DtypeChoice(FLOAT16, "the checkpoint's dtype", ckpt, arch=arch)
    if not _no_bf16_cores(capability):
        why = {
            BFLOAT16: "the checkpoint's dtype",
            FLOAT32: "a float32 checkpoint runs in bfloat16, as kernel-agent always ran it "
            "(--dtype float32 keeps float32)",
        }.get(ckpt or "", "the checkpoint does not state its dtype: bfloat16, the default")
        return DtypeChoice(BFLOAT16, why, ckpt, arch=arch)
    why = (
        f"{arch} has no bf16 tensor cores (bf16 matmuls run on FMA units and attention on "
        "SDPA's math backend there); float16 runs on its tensor cores, once analyze has "
        "checked it against a float32 run of the workload"
    )
    return DtypeChoice(FLOAT16, why, ckpt, check=PENDING, candidate=FLOAT16, arch=arch)


def needs_check(choice: Mapping[str, Any] | None, *, retry: bool = False) -> bool:
    """Whether ``analyze`` has to check the float16 candidate of ``choice`` (``run.json`` →
    ``dtype``): it is :data:`PENDING`, or (``retry``: a harness was written since) the last
    check could not run."""
    check = (choice or {}).get("check")
    if check == PENDING:
        return True
    return retry and isinstance(check, Mapping) and "passed" not in check


def resolve(
    requested: str | None, checkpoint: str | None, capability: tuple[int, ...] | None
) -> DtypeChoice:
    """The dtype of a new run for ``--dtype requested``: ``auto`` (or None) is
    :func:`runtime_dtype`; an explicit dtype is kept as given, with a ``warning`` when it is
    bfloat16 on a GPU without bf16 tensor cores. ``ValueError`` for an unknown dtype."""
    if requested in (None, "", AUTO):
        return runtime_dtype(checkpoint, capability)
    wanted = canonical(requested)
    if wanted is None:
        raise ValueError(f"unknown dtype {requested!r}: auto, {', '.join(BYTES)}")
    arch = gpu_arch.arch_of(tuple(capability[:2])) if capability else None
    choice = DtypeChoice(wanted, f"--dtype {wanted}", canonical(checkpoint), wanted, arch=arch)
    if wanted == BFLOAT16 and _no_bf16_cores(capability):
        choice.warning = (
            f"--dtype bfloat16 on {arch}: this GPU has no bf16 tensor cores, so the model's "
            "matmuls run on FMA units and its attention on SDPA's math backend (a slow "
            "baseline); --dtype auto runs float16 after checking it against float32"
        )
    return choice


# ------------------------------------------------------------------ analyze: the check


def nonfinite(reference: Any, candidate: Any) -> list[str]:
    """Outputs of ``candidate`` with non-finite values where ``reference`` has finite ones
    (an overflow): ``["out.first_logits: 3 non-finite values"]``. Masks the reference has
    as well (-inf logits) are not counted."""
    import torch

    from kernel_agent.kernels.compare import flatten

    ref = flatten(reference)
    found = []
    for name, new in flatten(candidate).items():
        if not new.is_floating_point():
            continue
        bad = ~torch.isfinite(new.detach())
        old = ref.get(name)
        if old is not None and old.shape == new.shape and old.is_floating_point():
            bad &= torch.isfinite(old.detach().to(new.device))
        if count := int(bad.sum()):
            found.append(f"{name}: {count} non-finite values")
    return found


def _free() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _numbers(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """The numeric metrics of a (nested) comparison, flattened (``teacher_forced.x``)."""
    found: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            found.update(_numbers(value, f"{prefix}{key}."))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            found[f"{prefix}{key}"] = float(value)
    return found


def brief(metrics: Mapping[str, Any], limit: int = 3) -> str:
    """A few similarity metrics of a check for its ``why``: ``first_logits_cosine 0.9998``."""
    found = _numbers(metrics)
    keys = [k for k in found if any(w in k for w in ("cosine", "psnr", "kl", "top1"))]
    return ", ".join(f"{k.rsplit('.', 1)[-1]} {found[k]:.6g}" for k in keys[:limit])


def check(
    load: Callable[[str], Workload], candidate: str, reference: str, *, gate: bool = True
) -> dict[str, Any]:
    """``analyze``'s check of the ``candidate`` dtype (float16) against a ``reference`` run
    (float32 where it fits): ``load(dtype)`` returns the workload loaded in that dtype. The
    reference runs on its inputs (and, with ``gate``, the perceptual samples are generated
    and scored), is freed, then the candidate runs on its own inputs and is judged:

    * outputs non-finite where the reference's are finite fail (an overflow, :func:`nonfinite`);
    * the workload's end-to-end checks (:func:`~kernel_agent.workloads.quality.assess`: its
      comparison, teacher forcing for teacher-forced workloads) with its relaxed bounds
      (:func:`~kernel_agent.workloads.perceptual.floor_options`);
    * with ``gate`` and declared samples, the perceptual gate at the relaxed thresholds.
      A gate that cannot run (its scoring models) is recorded and does not decide.

    ``{"passed", "reason", "candidate", "reference", "metrics", "perceptual", "seconds"}``.
    A failure of the candidate's run is a failed check (with its ``traceback``); one of the
    reference's raises, and so does running out of memory (no verdict)."""
    import torch

    from kernel_agent.workloads import perceptual, quality

    start = time.perf_counter()
    result: dict[str, Any] = {"candidate": candidate, "reference": reference}
    workload = load(reference)
    inputs = workload.make_inputs()
    with torch.inference_mode():
        ref = workload.run(inputs)
    samples = None
    if gate:
        try:
            samples, info = perceptual.record_baseline(workload)
            if samples is None:
                result["perceptual"] = info
        except torch.OutOfMemoryError:
            raise
        except Exception:
            result["perceptual"] = {"status": "error", "error": traceback.format_exc()[-1500:]}
    del workload, inputs
    _free()
    try:
        workload = load(candidate)
        inputs = workload.make_inputs()
        with torch.inference_mode():
            out = workload.run(inputs)
        problems = nonfinite(ref, out)
        with workload.with_options(perceptual.floor_options(workload, perceptual.RELAXED)):
            verdict = quality.assess(workload, inputs, ref, out, chaotic=workload.chaotic)
        reasons = [*problems, *([verdict["reason"]] if not verdict["passed"] else [])]
        if samples is not None:
            gated = perceptual.check(workload, samples, quality=perceptual.RELAXED)
            gated.pop("per_sample", None)
            if "error" in gated:  # the gate could not run: no verdict of the dtype
                gated = {"status": "error", "reason": gated.get("reason")}
            elif not gated["passed"]:
                reasons.append(f"perceptual: {gated['reason']}")
            result["perceptual"] = gated
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:
        what = " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
        reasons = [f"the {candidate} run failed ({what})"]
        verdict = {"metrics": {}, "traceback": traceback.format_exc()[-1500:]}
    result.update(
        passed=not reasons,
        reason="; ".join(r for r in reasons if r),
        metrics=verdict["metrics"],
        seconds=round(time.perf_counter() - start, 1),
    )
    if "traceback" in verdict:  # the candidate's run failed: where
        result["traceback"] = verdict["traceback"]
    return result


def _native(choice: Mapping[str, Any], fp32_fits: bool) -> str:
    """The dtype that cannot overflow where the checkpoint does not: the checkpoint's
    (float32 only when it fits), else bfloat16."""
    ckpt = choice.get("checkpoint")
    return FLOAT32 if ckpt == FLOAT32 and fp32_fits else BFLOAT16


def settle(
    choice: Mapping[str, Any], result: Mapping[str, Any], *, fp32_fits: bool
) -> dict[str, Any]:
    """``run.json`` → ``dtype`` after ``analyze``'s :func:`check` of a pending float16
    candidate (``result``: the worker's, with ``passed``; anything else is no verdict):

    * passed: float16, with the measured reason;
    * failed: float32 when it fits the GPU (``fp32_fits``, ``memfit.py``), else the
      checkpoint's dtype (bfloat16: no tensor cores here, but no overflow either);
    * no verdict (the check could not run): the checkpoint's dtype, as before #255; a
      harness written afterwards checks it again (:func:`needs_check` with ``retry``)."""
    out = {**choice, "check": _record(result)}
    arch = choice.get("arch") or "this GPU"
    candidate = choice.get("candidate") or FLOAT16
    reference = result.get("reference") or choice.get("reference") or FLOAT32
    if result.get("passed") is True:
        found = brief(result.get("metrics") or {})
        out["runtime"] = candidate
        out["why"] = (
            f"{arch} has no bf16 tensor cores; {candidate} runs on them and matched a "
            f"{reference} run of the workload within its relaxed bounds"
            + (f" ({found})" if found else "")
        )
        return out
    if result.get("passed") is False:
        fallback = FLOAT32 if fp32_fits else _native(choice, fp32_fits)
        then = (
            "float32, which fits this GPU"
            if fallback == FLOAT32
            else f"float32 does not fit this GPU, so {fallback} (no tensor cores on {arch})"
        )
        out["runtime"] = fallback
        out["why"] = f"{candidate} failed against {reference} ({result.get('reason')}): {then}"
        return out
    native = _native(choice, fp32_fits)
    error = " ".join(str(result.get("error") or result.get("status") or "no result").split())
    out["runtime"] = native
    out["why"] = f"the {candidate} check could not run ({error[-300:]}): {native}"
    return out


def _record(result: Mapping[str, Any]) -> dict[str, Any]:
    """What ``run.json`` keeps of a check: the verdict, its reason and metrics."""
    keep = ("passed", "reason", "reference", "candidate", "metrics", "perceptual", "seconds")
    found = {k: result[k] for k in keep if k in result}
    if "passed" not in result:
        found["status"] = str(result.get("status") or "error")
        found["error"] = str(result.get("error") or "")[-1500:]
    return found


# ------------------------------------------------------------------ reporting


def summary_section(choice: Mapping[str, Any] | None) -> str:
    """``profile/summary.md`` (the planner's, systems' and native engineers' prompts): the
    run's dtype when it is not the checkpoint's or carries a warning; "" otherwise (the
    workload line already names it)."""
    if not choice:
        return ""
    runtime, ckpt = choice.get("runtime"), choice.get("checkpoint")
    if runtime == ckpt and not choice.get("warning"):
        return ""
    lines = [
        "",
        "## Run dtype",
        "",
        f"* **{runtime}** (checkpoint: {ckpt or 'not stated'}): {choice.get('why')}.",
        f"* The captured tensors, references and tolerances are {runtime}'s: kernels take "
        f"and return {runtime}.",
    ]
    if choice.get("warning"):
        lines.append(f"* WARNING: {choice['warning']}.")
    return "\n".join(lines) + "\n"


def report_lines(data: Mapping[str, Any]) -> list[str]:
    """``report.md`` bullets: the run's dtype and why, and the memory preflight."""
    from kernel_agent import memfit

    choice = data.get("dtype") or {}
    lines = []
    if choice:
        ckpt = choice.get("checkpoint") or "not stated"
        lines.append(
            f"* dtype: **{choice.get('runtime')}** (checkpoint: {ckpt}; {choice.get('why')})"
        )
        if choice.get("warning"):
            lines.append(f"* **dtype warning**: {choice['warning']}")
    if fit := data.get("memory_fit"):
        lines.append(f"* memory preflight: {memfit.describe(fit)}")
    return lines
