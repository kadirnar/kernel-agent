"""Evaluate one kernel candidate against a captured module.

Candidate contract (``candidates/<name>.py``)::

    def build(reference: torch.nn.Module) -> torch.nn.Module:
        '''Return a drop-in replacement for ``reference``: same forward
        signature, same outputs (within tolerance), same in-place side effects
        (e.g. KV-cache updates).  Reuse ``reference``'s parameters/buffers.
        Return ``reference`` itself for instances the kernel does not support.'''

Each captured case is replayed through the entrypoint it was recorded from:
``candidate(*args, **kwargs)`` for ``forward`` cases and
``candidate.<method>(*args, **kwargs)`` otherwise (e.g. ``forward_step`` of a
custom decode loop), for correctness and timing alike.  A candidate that lacks
a captured method is a ``build_error``.

Run as a subprocess (so compiler crashes and illegal memory accesses cannot
take down the orchestrator)::

    python -m kernel_agent.kernels.evaluate CAPTURE CANDIDATE [--profile] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from kernel_agent.gpulock import gpu_lock

COMPILE_MARKER = "@@KA_COMPILE_S@@"  # on stderr, so a timed-out run still reports it


def load_candidate_module(path: Path) -> Any:
    path = path.resolve()
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    name = f"ka_candidate_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise AttributeError(f"{path.name} must define build(reference) -> nn.Module")
    return module


def _short_tb(limit: int = 4000) -> str:
    text = traceback.format_exc()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def _kernel_table(fn: Any, args: Any, kwargs: Any, top: int = 15) -> list[dict[str, Any]]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from kernel_agent.kernels.bench import has_mutable_state

    a, k = (
        (copy.deepcopy(args), copy.deepcopy(kwargs))
        if has_mutable_state(args, kwargs)
        else (
            args,
            kwargs,
        )
    )
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*a, **k)
        torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        us = getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)
        if us:
            rows.append({"kernel": evt.key[:140], "calls": int(evt.count), "us": round(us, 2)})
    rows.sort(key=lambda r: -r["us"])
    return rows[:top]


def evaluate(
    capture_path: Path,
    candidate_path: Path,
    *,
    profile: bool = False,
    l2_flush: bool = False,
    compile_baseline: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Build, check and time one candidate.  ``device`` defaults to CUDA when
    available; on CPU only correctness is checked (timing needs CUDA events)."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        from kernel_agent import toolchain

        toolchain.setup()

    from kernel_agent.kernels.bench import compare_timing, time_call
    from kernel_agent.kernels.compare import compare_side_effects, compare_structures
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.methods import entrypoint
    from kernel_agent.workloads.base import synchronize

    result: dict[str, Any] = {
        "candidate": str(candidate_path),
        "status": "error",
        "correct": False,
    }
    t0 = time.perf_counter()
    capture = load_capture(capture_path, device=device)
    reference = capture["module"].eval()
    cases = capture["cases"]
    for case in cases:
        case.setdefault("method", "forward")  # captures written before entrypoints existed

    # 1. import + build
    t_build = time.perf_counter()
    try:
        module = load_candidate_module(candidate_path)
        given = copy.deepcopy(reference)
        candidate = module.build(given)
        if candidate is None:
            raise TypeError("build() returned None")
        candidate = candidate.eval() if hasattr(candidate, "eval") else candidate
    except Exception:
        result.update(status="build_error", error=_short_tb())
        return result
    if candidate is given or candidate is reference:
        result.update(
            status="build_error",
            error="build() returned the reference module unchanged; nothing to evaluate "
            "(that fallback is only for instances the kernel does not support)",
        )
        return result
    calls: collections.Counter[str] = collections.Counter()
    for case in cases:
        calls[case["method"]] += case["count"]
    missing = [m for m in calls if m != "forward" and not callable(getattr(candidate, m, None))]
    if missing:
        result.update(
            status="build_error",
            error=f"build() returned a {type(candidate).__name__} without "
            + ", ".join(f"`{m}()`" for m in missing)
            + "; the model calls this module through "
            + ", ".join(f"`{m}` ({n} calls per run)" for m, n in calls.items())
            + ". Implement every captured entrypoint with the reference signature and side "
            "effects (subclassing the reference class keeps the ones you do not optimise).",
        )
        return result

    # 2. correctness on every captured case (outputs + in-place side effects)
    case_reports: list[dict[str, Any]] = []
    all_ok = True
    for i, case in enumerate(cases):
        args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
        try:
            with torch.inference_mode():
                out = entrypoint(candidate, case["method"])(*args, **kwargs)
            synchronize()
        except Exception:
            result.update(status="runtime_error", error=_short_tb(), failed_case=i)
            return result
        if i == 0:
            # import + build + first call: where load_inline / JIT backends compile
            result["compile_s"] = round(time.perf_counter() - t_build, 1)
            print(f"{COMPILE_MARKER}{result['compile_s']}", file=sys.stderr, flush=True)
        checks = compare_structures(case["output"], out, "output")
        checks += compare_side_effects(case["args"], case["post_args"], args, "args")
        checks += compare_side_effects(case["kwargs"], case["post_kwargs"], kwargs, "kwargs")
        bad = [c for c in checks if not c.get("ok")]
        ok = not bad
        all_ok &= ok
        worst = max(
            (c for c in checks if "max_abs_err" in c), key=lambda c: c["max_abs_err"], default={}
        )
        case_reports.append(
            {
                "case": i,
                "method": case["method"],
                "signature": case["signature"],
                "calls_per_run": case["count"],
                "ok": ok,
                "tensors_checked": len(checks),
                "max_abs_err": worst.get("max_abs_err"),
                "min_cosine": min((c.get("cosine", 1.0) for c in checks), default=1.0),
                "failures": bad[:5],
            }
        )
    result["cases"] = case_reports
    if not all_ok:
        result.update(status="incorrect")
        return result
    result["correct"] = True
    if not device.startswith("cuda"):
        result.update(status="ok", timing="skipped: no CUDA device")
        result["eval_seconds"] = round(time.perf_counter() - t0, 1)
        return result

    # 3. performance (reference vs candidate, same inputs, same entrypoint)
    compiled_ref = copy.deepcopy(reference) if compile_baseline else None
    compiled: dict[str, Any] = {}
    saved = 0.0
    ref_total = 0.0
    new_total = 0.0
    for report, case in zip(case_reports, cases, strict=True):
        method = case["method"]
        try:
            ref_t, new_t = compare_timing(
                entrypoint(reference, method),
                entrypoint(candidate, method),
                case["args"],
                case["kwargs"],
                l2_flush=l2_flush,
            )
        except Exception:
            result.update(status="runtime_error", error=_short_tb())
            return result
        report["ref_ms"] = round(ref_t["median_ms"], 5)
        report["new_ms"] = round(new_t["median_ms"], 5)
        report["speedup"] = round(ref_t["median_ms"] / max(new_t["median_ms"], 1e-9), 3)
        report["timing_spread"] = round(max(ref_t["spread"], new_t["spread"]), 3)
        if compiled_ref is not None:
            try:
                if method not in compiled:
                    compiled[method] = torch.compile(
                        entrypoint(compiled_ref, method), mode="max-autotune-no-cudagraphs"
                    )
                comp_t = time_call(
                    compiled[method], case["args"], case["kwargs"], l2_flush=l2_flush
                )
                report["torch_compile_ms"] = round(comp_t["median_ms"], 5)
            except Exception as exc:
                report["torch_compile_ms"] = f"failed: {exc}"[:200]
        n = case["count"]
        ref_total += n * ref_t["median_ms"]
        new_total += n * new_t["median_ms"]
        # times the instances that call this entrypoint (all instances for old captures)
        users = capture.get("method_instances", {}).get(method, capture.get("instances", 1))
        saved += n * (ref_t["median_ms"] - new_t["median_ms"]) * users

    result.update(
        status="ok",
        speedup=round(ref_total / max(new_total, 1e-9), 3),
        est_saved_ms_per_run=round(saved, 3),
        ref_ms_weighted=round(ref_total, 4),
        new_ms_weighted=round(new_total, 4),
    )
    if profile:
        try:
            first = cases[0]
            a, k, m = first["args"], first["kwargs"], first["method"]
            result["kernels_candidate"] = _kernel_table(entrypoint(candidate, m), a, k)
            result["kernels_reference"] = _kernel_table(entrypoint(reference, m), a, k)
        except Exception as exc:
            result["profile_error"] = str(exc)[:500]
    result["eval_seconds"] = round(time.perf_counter() - t0, 1)
    return result


def _timeout_result(timeout: float, stderr: str | bytes | None) -> dict[str, Any]:
    """Timeout result; says whether compiling or checking/benchmarking ran out of time."""
    text = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr or ""
    compile_s = None
    for line in text.splitlines():
        if line.startswith(COMPILE_MARKER):
            compile_s = float(line[len(COMPILE_MARKER) :])
    result: dict[str, Any] = {"status": "timeout", "correct": False, "compile_s": compile_s}
    if compile_s is None:
        result["error"] = f"exceeded {timeout:.0f}s before the first call finished (compiling?)"
    else:
        result["error"] = (
            f"exceeded {timeout:.0f}s; compile + first call took {compile_s:.0f}s, "
            "the rest went to correctness checks and benchmarking"
        )
    return result


def run_evaluation(
    capture_path: Path,
    candidate_path: Path,
    *,
    profile: bool = False,
    l2_flush: bool = False,
    compile_baseline: bool = False,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Evaluate in a fresh subprocess under the GPU lock."""
    cmd = [
        sys.executable,
        "-m",
        "kernel_agent.kernels.evaluate",
        str(capture_path),
        str(candidate_path),
        "--json",
        "-",
    ]
    if profile:
        cmd.append("--profile")
    if l2_flush:
        cmd.append("--l2-flush")
    if compile_baseline:
        cmd.append("--compile-baseline")
    with gpu_lock():
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return _timeout_result(timeout, exc.stderr)
    marker = "@@KA_RESULT@@"
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith(marker):
            data: dict[str, Any] = json.loads(line[len(marker) :])
            return data
    tail = (proc.stderr or proc.stdout)[-4000:]
    return {
        "status": "crash",
        "correct": False,
        "returncode": proc.returncode,
        "error": tail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--profile", action="store_true", help="add per-kernel tables")
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--json", default=None, help="write result JSON ('-' = stdout marker)")
    ns = parser.parse_args(argv)
    try:
        result = evaluate(
            ns.capture,
            ns.candidate,
            profile=ns.profile,
            l2_flush=ns.l2_flush,
            compile_baseline=ns.compile_baseline,
        )
    except Exception:
        result = {"status": "harness_error", "correct": False, "error": _short_tb()}
    payload = json.dumps(result, default=str)
    if ns.json == "-":
        print("@@KA_RESULT@@" + payload, flush=True)
    elif ns.json:
        Path(ns.json).write_text(payload)
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("correct") else 1


if __name__ == "__main__":
    raise SystemExit(main())
