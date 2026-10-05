"""Strong baseline: what a competent user gets without custom kernels.

:meth:`~kernel_agent.workloads.base.Workload.reference_optimizations` applies the
model's own fast path (VoxCPM's ``model.optimize()``, a static KV cache +
``torch.compile`` for LLMs, a compiled denoiser for diffusion).  ``analyze``
measures it after the eager baseline in a fresh process (``python -m
kernel_agent.strong_baseline``; the eager model is released first) when the
workload implements the hook or the run was started with ``--compile-baseline``
(workloads without the hook then get a generic ``torch.compile`` of their root
modules).  ``baseline.json`` gets

* ``compiled_ms``: median latency with the reference optimisations (warm-up and
  compilation excluded), ``None`` when the measurement failed;
* ``compiled_detail``: what was applied, the timings, the compile + warm-up
  seconds, the quality verdict against the eager output (the workload's own
  check: teacher forced for chaotic workloads) or the error.

Failures are recorded, never fatal.  The helpers below give the reports and the
agents' prompts the speedups vs eager AND vs compiled; :data:`REFERENCE_TRANSFORM`
lets the integration measure "reference optimisations + accepted kernels".
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from kernel_agent.gpulock import child_env
from kernel_agent.workspace import RunDir, read_json

MARKER = "@@KA_STRONG_BASELINE@@"
#: Transform file (``apply(workload)``) that applies the reference optimisations.
REFERENCE_TRANSFORM = Path(__file__).parent / "integrate" / "builtin" / "reference_optimizations.py"
#: Label of :data:`REFERENCE_TRANSFORM` in integration records.
REFERENCE_LABEL = "reference_optimizations"
#: Untimed runs of the optimised model: compilation, then CUDA-graph recording.
WARMUP = 2
TIMEOUT_S = 1800.0
#: The combination must beat the compiled baseline by more than this to "compose".
MIN_GAIN = 0.01


def has_hook(workload: Any) -> bool:
    """The workload overrides :meth:`Workload.reference_optimizations`."""
    from kernel_agent.workloads.base import Workload

    return type(workload).reference_optimizations is not Workload.reference_optimizations


def generic_compile(workload: Any) -> str:
    """``nn.Module.compile()`` (in place, default mode) of every root module."""
    roots = workload.roots()
    for module in roots.values():
        module.compile()
    return f"generic torch.compile (default mode) of the root modules: {', '.join(roots)}"


def apply(workload: Any, *, generic: bool = True) -> dict[str, str] | None:
    """Apply the reference optimisations in place: ``{"source", "description"}``,
    or ``None`` when there is nothing to apply (no hook, or the hook declined,
    and no ``generic`` fallback)."""
    if has_hook(workload):
        description = workload.reference_optimizations()
        if description:
            return {"source": "workload", "description": str(description)}
    if generic:
        return {"source": "generic", "description": generic_compile(workload)}
    return None


# ------------------------------------------------------------------ measurement


def measure(run: RunDir, *, generic: bool, warmup: int = WARMUP, iters: int = 3) -> dict[str, Any]:
    """Load the workload in this process, apply the reference optimisations and
    time them like the eager baseline; quality is judged against the eager
    ``baseline_output.pt`` with the workload's own end-to-end check."""
    import torch

    from kernel_agent.worker import _workload
    from kernel_agent.workloads.base import measure as measure_runs
    from kernel_agent.workloads.base import synchronize
    from kernel_agent.workloads.quality import assess, is_chaotic

    t0 = time.perf_counter()
    workload = _workload(run)
    detail: dict[str, Any] = {"load_s": round(time.perf_counter() - t0, 1)}
    t0 = time.perf_counter()
    applied = apply(workload, generic=generic)
    if applied is None:
        return {
            "status": "skipped",
            "reason": "the workload's hook has no reference optimisations for this model",
        }
    detail.update(applied, apply_s=round(time.perf_counter() - t0, 1))
    inputs = workload.make_inputs()
    warm = []
    with torch.inference_mode():
        for _ in range(warmup):
            t0 = time.perf_counter()
            workload.run(inputs)
            synchronize()
            warm.append(round(time.perf_counter() - t0, 2))
    timing = measure_runs(workload, inputs, warmup=0, iters=iters)
    output = timing.pop("output")
    median = float(timing["median_ms"])
    # a whole run (median_ms is the time to first audio for metric=ttfa, objective.py)
    run_ms = float((timing.get("metric_detail") or {}).get("run_ms") or median)
    detail.update(
        status="ok",
        median_ms=round(median, 3),
        min_ms=round(timing["min_ms"], 3),
        times_ms=[round(t, 3) for t in timing["times_ms"]],
        metric=timing["metric"],
        **({"metric_detail": d} if (d := timing.get("metric_detail")) else {}),
        peak_mem_gb=round(timing["peak_mem_gb"], 3),
        warmup_s=warm,
        # warm-up runs minus steady-state runs: compilation, autotuning, graph recording
        compile_s=round(max(sum(warm) - len(warm) * run_ms / 1000, 0.0), 1),
    )
    reference = torch.load(run.baseline_output(), weights_only=False)
    chaotic = is_chaotic(workload, read_json(run.baseline_json, {}) or {})
    try:
        detail["quality"] = assess(workload, inputs, reference, output, chaotic=chaotic)
    except Exception as exc:
        detail["quality"] = {"passed": False, "reason": f"quality check failed: {exc}"[:500]}
    return detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernel_agent.strong_baseline")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--generic", action="store_true", help="torch.compile without a hook")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=3)
    ns = parser.parse_args(argv)
    try:
        result = measure(
            RunDir(ns.run_dir.resolve()), generic=ns.generic, warmup=ns.warmup, iters=ns.iters
        )
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()[-4000:]}
    print(MARKER + json.dumps(result, default=str), flush=True)
    return 0 if result.get("status") == "ok" else 1


def run_measurement(
    run: RunDir,
    *,
    eager_ms: float,
    generic: bool,
    warmup: int = WARMUP,
    iters: int = 3,
    timeout: float = TIMEOUT_S,
) -> dict[str, Any]:
    """``{"compiled_ms", "compiled_detail"}`` for ``baseline.json``, measured by
    :func:`main` in a fresh process.  The caller holds the GPU lock (the child
    inherits it) and has released its own model.  Never raises."""
    cmd = [
        sys.executable,
        "-m",
        "kernel_agent.strong_baseline",
        "--run-dir",
        str(run.root),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    if generic:
        cmd.append("--generic")
    start = time.perf_counter()
    stdout = stderr = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=child_env())
        stdout, stderr = proc.stdout, proc.stderr
        detail = next(
            (
                json.loads(line[len(MARKER) :])
                for line in stdout.splitlines()[::-1]
                if line.startswith(MARKER)
            ),
            {"status": "crash", "returncode": proc.returncode, "error": (stderr or stdout)[-4000:]},
        )
    except subprocess.TimeoutExpired:
        detail = {"status": "timeout", "error": f"the compiled baseline exceeded {timeout:.0f}s"}
    except Exception:
        detail = {"status": "error", "error": traceback.format_exc()[-4000:]}
    detail["wall_s"] = round(time.perf_counter() - start, 1)
    log = run.root / "logs" / "strong-baseline.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n{stdout[-20000:]}\n{stderr[-20000:]}\n")
    ms = _num(detail.get("median_ms")) if detail.get("status") == "ok" else None
    if ms:
        detail["speedup_vs_eager"] = round(eager_ms / ms, 4)
    return {"compiled_ms": ms, "compiled_detail": detail}


# ------------------------------------------------------------------ reporting


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return None
    return float(value)


def compiled_ms(baseline: dict[str, Any] | None) -> float | None:
    """The compiled baseline latency, ``None`` when it was not (successfully) measured."""
    return _num((baseline or {}).get("compiled_ms"))


def eager_ms(baseline: dict[str, Any] | None) -> float | None:
    """The run's eager baseline (a re-profile of an improve round keeps it in ``run_eager_ms``)."""
    baseline = baseline or {}
    return _num(baseline.get("run_eager_ms")) or _num(baseline.get("median_ms"))


def speedups(
    baseline: dict[str, Any] | None, ms: float | None
) -> tuple[float | None, float | None]:
    """``(vs eager, vs compiled)`` of a latency ``ms``; ``None`` where unknown."""
    ms = _num(ms)
    if ms is None:
        return None, None
    eager, compiled = eager_ms(baseline), compiled_ms(baseline)
    return (eager / ms if eager else None), (compiled / ms if compiled else None)


def vs_both(baseline: dict[str, Any] | None, ms: float | None) -> str:
    """``"1.72x vs eager, 1.21x vs compiled"`` (vs eager only without a compiled baseline)."""
    eager, compiled = speedups(baseline, ms)
    parts = [f"{eager:.2f}x vs eager"] if eager else []
    if compiled:
        parts.append(f"{compiled:.2f}x vs compiled")
    return ", ".join(parts)


def quality_text(quality: dict[str, Any] | None) -> str:
    """One line on the compiled (or combined) output vs the eager output."""
    if not quality:
        return "quality not checked"
    metrics = quality.get("metrics") or {}
    tf = metrics.get("teacher_forced") if isinstance(metrics, dict) else None
    shown = tf if isinstance(tf, dict) else metrics
    keys = ("mean_step_cosine", "min_step_cosine", "rms_ratio")
    picked = {k: shown[k] for k in keys if k in shown} or {
        k: v for k, v in list(shown.items())[:3] if not isinstance(v, dict)
    }
    what = "teacher-forced " if isinstance(tf, dict) else ""
    detail = ", ".join(f"{k}={v}" for k, v in picked.items())
    if quality.get("passed"):
        return f"{what}quality ok vs eager" + (f" ({detail})" if detail else "")
    return f"{what}quality check FAILED vs eager: {quality.get('reason') or 'see metrics'}"


def describe(baseline: dict[str, Any] | None) -> str | None:
    """One line on the compiled baseline (``None`` when none was attempted)."""
    detail = (baseline or {}).get("compiled_detail")
    if not isinstance(detail, dict):
        return None
    ms = compiled_ms(baseline)
    if ms is None:
        why = detail.get("reason") or _last_error(str(detail.get("error") or ""))
        return f"compiled baseline not available ({detail.get('status')}: {why})"
    eager = eager_ms(baseline)
    gain = f", {eager / ms:.2f}x vs eager" if eager else ""
    warm = detail.get("warmup_s") or []
    warmup = (
        f"{len(warm)} warm-up runs, {sum(warm):.1f} s, of which {detail.get('compile_s')} s "
        "compilation / graph recording; excluded"
        if warm
        else f"compile + warm-up {detail.get('compile_s')} s, excluded"
    )
    return (
        f"compiled baseline **{ms:,.1f} ms**{gain}: {detail.get('description')} "
        f"({warmup}; {quality_text(detail.get('quality'))})"
    )


def summary_section(baseline: dict[str, Any]) -> str:
    """Markdown for ``profile/summary.md``: eager vs compiled and the real headroom."""
    line = describe(baseline)
    if line is None:
        return ""
    lines = ["", "## Strong baseline (eager vs compiled)", ""]
    eager, ms = eager_ms(baseline), compiled_ms(baseline)
    current = _num(baseline.get("median_ms"))
    if baseline.get("run_eager_ms") and current:
        lines.append(
            f"* this profile (accepted optimisations applied): **{current:,.1f} ms** = "
            f"{vs_both(baseline, current)}"
        )
    if eager:
        lines.append(f"* eager baseline: **{eager:,.1f} ms** (the workload as written)")
    lines.append(f"* {line}")
    if ms is not None:
        lines.append(
            "* results are reported vs eager AND vs compiled: an optimisation only helps "
            f"users when the model beats {ms:,.1f} ms"
        )
    return "\n".join(lines) + "\n"


def headroom(baseline: dict[str, Any]) -> str:
    """The real headroom vs the compiled baseline, for the agents' prompts."""
    eager, ms = eager_ms(baseline), compiled_ms(baseline)
    if ms is None:
        line = describe(baseline)
        return f"{line}; speedups are vs eager only." if line else ""
    gain = f" ({eager / ms:.2f}x vs eager)" if eager else ""
    current = _num(baseline.get("median_ms"))
    now = ""
    if baseline.get("run_eager_ms") and current:
        now = (
            f" The model of this profile already runs in {current:,.1f} ms "
            f"({vs_both(baseline, current)})."
        )
    return (
        f"**Real headroom**: the workload's reference optimisations "
        f"({(baseline.get('compiled_detail') or {}).get('description')}) already run it in "
        f"**{ms:,.1f} ms**{gain}: users get that for free. Speedups are reported vs eager "
        f"AND vs compiled; an optimisation only helps users when the model beats "
        f"{ms:,.1f} ms.{now} Re-implementing the same torch.compile / CUDA-graph capture "
        "is no progress; aim at what it cannot do (fused kernels, algorithmic changes), and "
        "keep kernels compatible with it (the integration also measures reference "
        "optimisations + accepted kernels; `torch.library.custom_op` + `register_fake` make "
        "a kernel opaque to torch.compile instead of a graph break)."
    )


def headroom_note(baseline: dict[str, Any]) -> str:
    """``headroom`` as a prompt paragraph (empty without a compiled baseline attempt)."""
    text = headroom(baseline)
    return f"\n{text}\n" if text else ""


# ------------------------------------------------------------------ integration


def _last_error(text: str) -> str:
    """The exception line of a traceback tail (or its last line)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    errors = [
        i
        for i, line in enumerate(lines)
        if re.match(r"^[\w.]*(Error|Exception|Unsupported|Exit)\w*:", line)
    ]
    if not errors:
        return (lines or [""])[-1][:300]
    i = errors[-1]
    line = lines[i]
    if i + 1 < len(lines) and lines[i + 1].startswith("Explanation:"):  # Dynamo says which call
        line += " — " + lines[i + 1].removeprefix("Explanation:").strip()
    return line[:300]


def combination(result: dict[str, Any], compiled: float) -> dict[str, Any]:
    """Verdict of "reference optimisations + accepted kernels" (one e2e result).

    ``composes``: passes and beats the compiled baseline by more than 1 %;
    ``no_gain``: passes but is not faster than the compiled baseline;
    ``fails_quality``: runs, but the output fails the quality check;
    ``breaks``: does not run (e.g. a kernel is a graph break under ``fullgraph=True``)."""
    ms = _num(result.get("median_ms"))
    out: dict[str, Any] = {
        "compiled_ms": compiled,
        **{k: result.get(k) for k in ("status", "passed", "reason", "median_ms", "speedup")},
        "metrics": result.get("metrics"),
        "patches": result.get("patches"),
    }
    if ms is not None:
        out["speedup_vs_compiled"] = round(compiled / ms, 4)
    if result.get("status") != "ok":
        out["verdict"] = "breaks"
        out["error"] = _last_error(str(result.get("error") or result.get("reason") or ""))
        out["error_tail"] = str(result.get("error") or "")[-2000:]
    elif not result.get("passed"):
        out["verdict"] = "fails_quality"
    elif ms is not None and ms < compiled * (1 - MIN_GAIN):
        out["verdict"] = "composes"
    else:
        out["verdict"] = "no_gain"
    return out


def combination_text(record: dict[str, Any]) -> str:
    """One line for the report / status / log."""
    verdict = record.get("verdict")
    ms = _num(record.get("median_ms"))
    timing = f"{ms:,.1f} ms ({record.get('speedup_vs_compiled')}x vs compiled)" if ms else ""
    if verdict == "composes":
        return f"composes: {timing}"
    if verdict == "no_gain":
        return f"runs and passes, but no gain over the compiled baseline: {timing}"
    if verdict == "fails_quality":
        return f"runs ({timing}) but fails the quality check: {record.get('reason')}"
    return f"breaks: {record.get('error') or record.get('status')}"


if __name__ == "__main__":
    raise SystemExit(main())
