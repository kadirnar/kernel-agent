"""End-to-end quality decision and the sensitivity probe.

* :func:`assess` turns one candidate run into a verdict.  Workloads without
  teacher forcing are judged by their own free-running :meth:`compare`, as
  before.  Teacher-forced workloads are judged by
  :meth:`~kernel_agent.workloads.base.Workload.compare_teacher_forced` plus a
  cheap free-running *sanity* check; the full free-running comparison only gates
  workloads that are not chaotic.
* :func:`probe` runs in ``analyze``: one extra run with a benign perturbation
  tells whether the free-running output is chaotic, and teacher-forced
  workloads replay their own trajectory once (it must reproduce it exactly).
"""

from __future__ import annotations

import contextlib
import math
import traceback
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from kernel_agent.kernels.compare import flatten
from kernel_agent.workloads import holdout, perceptual, stopping
from kernel_agent.workloads.base import Comparison, Workload

#: Free-running sanity check: RMS energy of every float output within ±25 %.
MAX_RMS_CHANGE = 0.25
#: Sensitivity probe: every ``nn.Linear`` output is multiplied by ``1 ± 2**-8``
#: (random sign per element), about one bf16 rounding step — the kind of change
#: a numerically-correct fused kernel makes.
PROBE_REL = 2.0**-8
PROBE_DESCRIPTION = "every nn.Linear output x (1 ± 2^-8), random sign per element"


def free_running_sanity(
    reference: Any,
    candidate: Any,
    *,
    max_rms_change: float = MAX_RMS_CHANGE,
    tolerances: dict[str, float] | None = None,
) -> Comparison:
    """Cheap check of a free-running output whose trajectory may legitimately
    diverge: every floating-point tensor of the reference exists in the
    candidate with the same shape (length), is finite and has an RMS energy
    within ``±max_rms_change`` of the reference (``tolerances[key]`` overrides
    it per output, e.g. ``{"audio": 0.6}``)."""
    ref = {k: v for k, v in flatten(reference).items() if v.is_floating_point()}
    new = flatten(candidate)
    metrics: dict[str, float | int | str] = {}
    problems: list[str] = []
    for name, r in ref.items():
        key = name.removeprefix("out.")
        allowed = (tolerances or {}).get(key, max_rms_change)
        n = new.get(name)
        if n is None:
            problems.append(f"{key} missing")
            continue
        if tuple(n.shape) != tuple(r.shape):
            problems.append(f"{key} shape {list(n.shape)} != {list(r.shape)}")
            continue
        if not bool(torch.isfinite(n).all()):
            problems.append(f"{key} has non-finite values")
            continue
        if r.numel() == 0:
            continue
        r_rms = float(r.detach().float().pow(2).mean().sqrt())
        n_rms = float(n.detach().float().pow(2).mean().sqrt())
        ratio = n_rms / r_rms if r_rms > 0 else (1.0 if n_rms == 0 else math.inf)
        metrics[f"{key}_rms_ratio"] = round(ratio, 4)
        if abs(ratio - 1.0) > allowed:
            problems.append(f"{key} RMS energy x{ratio:.3f} (allowed ±{allowed:.0%})")
    return Comparison(not problems, metrics, "; ".join(problems))


def judge(
    free_running: Comparison,
    teacher_forced: Comparison | None = None,
    sanity: Comparison | None = None,
    *,
    chaotic: bool = False,
) -> tuple[bool, str]:
    """Combine the end-to-end checks into ``(passed, reason)``.

    Without teacher forcing the free-running comparison decides alone.  With it,
    teacher forcing and the free-running sanity check must pass; the full
    free-running comparison is informational for chaotic workloads."""
    if teacher_forced is None:
        return free_running.passed, free_running.reason
    reasons = []
    if not teacher_forced.passed:
        reasons.append(f"teacher-forced: {teacher_forced.reason}")
    if sanity is not None and not sanity.passed:
        reasons.append(f"free-running sanity: {sanity.reason}")
    if not chaotic and not free_running.passed:
        reasons.append(f"free-running: {free_running.reason}")
    return not reasons, "; ".join(reasons)


def is_chaotic(workload: Workload, baseline: dict[str, Any]) -> bool:
    """Flagged by the workload itself or by the sensitivity probe of ``analyze``."""
    sensitivity = baseline.get("sensitivity") or {}
    return workload.chaotic or sensitivity.get("free_running_passed") is False


def _section(cmp: Comparison) -> dict[str, Any]:
    return {"passed": cmp.passed, "reason": cmp.reason, **cmp.metrics}


def assess(
    workload: Workload, inputs: Any, reference: Any, output: Any, *, chaotic: bool
) -> dict[str, Any]:
    """Verdict for a candidate whose free-running output is ``output``.

    Teacher-forced workloads run once more (teacher forced, not timed).
    Returns ``{"passed", "reason", "metrics"}``; for teacher-forced workloads
    ``metrics`` holds ``teacher_forced`` and ``free_running`` sections."""
    free = workload.compare(reference, output)
    if not workload.supports_teacher_forcing:
        return {"passed": free.passed, "reason": free.reason, "metrics": free.metrics}
    sanity = free_running_sanity(reference, output, tolerances=workload.sanity_rms_tolerance)
    with torch.inference_mode():
        forced = workload.run_teacher_forced(inputs, reference)
    tf = workload.compare_teacher_forced(reference, forced)
    passed, reason = judge(free, tf, sanity, chaotic=chaotic)
    return {
        "passed": passed,
        "reason": reason,
        "metrics": {
            "teacher_forced": _section(tf),
            "free_running": {
                **_section(free),
                "gating": not chaotic,
                "sanity_passed": sanity.passed,
                "sanity_reason": sanity.reason,
                **sanity.metrics,
            },
        },
    }


@contextlib.contextmanager
def perturb_linears(
    roots: dict[str, nn.Module], *, rel: float = PROBE_REL, seed: int = 0
) -> Iterator[int]:
    """Multiply the output of every ``nn.Linear`` under ``roots`` by
    ``1 + rel * (±1)``.  The signs come from a private generator, so the
    workload's own sampling noise (global RNG) is untouched.  Yields the number
    of hooked modules."""
    generators: dict[torch.device, torch.Generator] = {}

    def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        if not isinstance(output, torch.Tensor) or not output.is_floating_point():
            return None
        gen = generators.get(output.device)
        if gen is None:
            gen = torch.Generator(device=output.device).manual_seed(seed)
            generators[output.device] = gen
        sign = torch.randint(
            0, 2, output.shape, generator=gen, device=output.device, dtype=torch.float32
        )
        return (output.float() * (1.0 + rel * (2.0 * sign - 1.0))).to(output.dtype)

    handles = []
    seen: set[int] = set()
    for root in roots.values():
        for module in root.modules():
            if isinstance(module, nn.Linear) and id(module) not in seen:
                seen.add(id(module))
                handles.append(module.register_forward_hook(hook))
    try:
        yield len(handles)
    finally:
        for handle in handles:
            handle.remove()


def sensitivity_probe(workload: Workload, inputs: Any, reference: Any) -> dict[str, Any]:
    """One free-running run under :func:`perturb_linears`, compared with
    ``reference`` by the workload's own free-running comparison."""
    with torch.inference_mode(), perturb_linears(workload.roots()) as hooked:
        if hooked == 0:
            return {"free_running_passed": None, "reason": "no nn.Linear modules to perturb"}
        output = workload.run(inputs)
    cmp = workload.compare(reference, output)
    return {
        "free_running_passed": cmp.passed,
        "reason": cmp.reason,
        "metrics": cmp.metrics,
        "perturbation": f"{PROBE_DESCRIPTION} ({hooked} modules)",
    }


def teacher_forcing_selfcheck(workload: Workload, inputs: Any, reference: Any) -> dict[str, Any]:
    """Teacher-force the unmodified model on its own trajectory.  It must pass
    (and is normally exact); otherwise the replay does not draw the same noise
    as the free run and every candidate would be rejected."""
    with torch.inference_mode():
        forced = workload.run_teacher_forced(inputs, reference)
    cmp = workload.compare_teacher_forced(reference, forced)
    return {"passed": cmp.passed, "reason": cmp.reason, "metrics": cmp.metrics}


def probe(workload: Workload, inputs: Any, reference: Any) -> dict[str, Any]:
    """Quality probes for ``baseline.json`` (``sensitivity`` and, for
    teacher-forced workloads, ``teacher_forcing``).  Failures are recorded, not
    raised: they inform the user and the agents but must not abort ``analyze``."""
    result: dict[str, Any] = {}
    try:
        result["sensitivity"] = sensitivity_probe(workload, inputs, reference)
    except Exception:
        result["sensitivity"] = {
            "free_running_passed": None,
            "error": traceback.format_exc()[-2000:],
        }
    if workload.supports_teacher_forcing:
        try:
            result["teacher_forcing"] = teacher_forcing_selfcheck(workload, inputs, reference)
        except Exception:
            result["teacher_forcing"] = {"passed": False, "error": traceback.format_exc()[-2000:]}
        if workload.teacher_forcing_note:
            result["teacher_forcing"]["note"] = workload.teacher_forcing_note
    result["chaotic"] = is_chaotic(workload, result)
    return result


def probe_messages(baseline: dict[str, Any]) -> list[str]:
    """Log lines for the orchestrator; loud when no kernel can pass end to end."""
    sens = baseline.get("sensitivity") or {}
    tf = baseline.get("teacher_forcing")
    messages = []
    if sens.get("free_running_passed") is False:
        if tf is None:
            messages.append(
                "WARNING " + "!" * 60 + "\n  the output is CHAOTIC: a benign perturbation "
                f"({PROBE_DESCRIPTION}) already fails the comparison ({sens.get('reason')}),\n"
                "  and this workload has no teacher forcing, so end-to-end validation will "
                "reject (almost) every kernel.\n  " + "!" * 68
            )
        else:
            messages.append(
                f"analyze: free-running output is chaotic ({sens.get('reason')}); "
                "quality is judged by teacher forcing"
            )
    if tf is not None:
        if tf.get("passed"):
            messages.append(f"analyze: teacher-forcing self-check ok ({_fmt(tf.get('metrics'))})")
        else:
            detail = tf.get("reason") or str(tf.get("error", ""))[-500:]
            messages.append(
                f"WARNING: teacher-forcing self-check FAILED ({detail}); every candidate will "
                "fail end-to-end validation"
            )
    return (
        messages
        + holdout.messages(baseline)
        + stopping.messages(baseline)
        + perceptual.messages(baseline)
    )


def _fmt(metrics: dict[str, Any] | None) -> str:
    return ", ".join(f"{k}={v}" for k, v in (metrics or {}).items())


def summary_section(baseline: dict[str, Any]) -> str:
    """Markdown for ``profile/summary.md`` so the planner and systems agents
    know how end-to-end quality is judged for this run."""
    sens = baseline.get("sensitivity")
    if not sens:
        return ""
    tf = baseline.get("teacher_forcing")
    lines = ["", "## End-to-end quality check", ""]
    status = sens.get("free_running_passed")
    if status is None:
        lines.append(
            f"* sensitivity probe skipped: {sens.get('reason') or sens.get('error', '')[-300:]}"
        )
    elif status:
        lines.append(
            f"* sensitivity probe ({PROBE_DESCRIPTION}): the free-running output still "
            f"passes the workload comparison ({_fmt(sens.get('metrics'))}). Standard "
            "end-to-end validation applies."
        )
    else:
        lines.append(
            f"* **CHAOTIC OUTPUT.** A benign perturbation ({PROBE_DESCRIPTION}) already "
            f"fails the free-running comparison: {sens.get('reason')} "
            f"({_fmt(sens.get('metrics'))}). Every numerically-correct kernel makes the "
            "free-running trajectory diverge."
        )
        if not tf:
            lines.append(
                "* **WARNING: this workload has no teacher forcing**, so end-to-end "
                "validation will reject (almost) every kernel. Only bit-exact changes "
                "can pass; a harness implementing `run_teacher_forced` fixes this."
            )
    if tf:
        check = "passes" if tf.get("passed") else "**FAILS**"
        lines += [
            "* quality is judged by **teacher forcing**: the candidate replays the baseline "
            "trajectory and its per-step predictions are compared with the baseline's "
            f"({_fmt(tf.get('metrics'))}; self-check on the unmodified model {check}). "
            "The free-running output only has to pass a sanity check (finite, same "
            f"shape, RMS energy within ±{MAX_RMS_CHANGE:.0%} unless the workload sets a "
            "per-output tolerance)"
            + (" and is otherwise informational." if baseline.get("chaotic") else "."),
            "* teacher forcing hooks into the step loop from Python, so a transform that "
            "captures a whole step (or the whole loop) in one CUDA graph, or changes how "
            "sampling noise is drawn, cannot be validated and is rejected.",
        ]
        if tf.get("note"):
            lines.append(f"* {tf['note']}")
        if tf.get("error"):
            lines.append(f"* teacher-forcing self-check error: `{tf['error'][-300:]}`")
    lines += holdout.summary_lines(baseline) + stopping.summary_lines(baseline)
    lines += perceptual.summary_lines(baseline)
    return "\n".join(lines) + "\n"
