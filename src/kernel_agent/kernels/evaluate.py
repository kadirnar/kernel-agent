"""Evaluate one kernel candidate against a captured module.

Candidate contract (``candidates/<name>.py``)::

    def build(reference: torch.nn.Module) -> torch.nn.Module:
        '''Return a drop-in replacement for ``reference``: same forward
        signature, same outputs (within tolerance), same in-place side effects
        (e.g. KV-cache updates).  Reuse ``reference``'s parameters/buffers.
        Return ``reference`` itself for instances the kernel does not support.'''

Run as a subprocess (so compiler crashes and illegal memory accesses cannot
take down the orchestrator)::

    python -m kernel_agent.kernels.evaluate CAPTURE CANDIDATE [--profile] [--json OUT]
"""

from __future__ import annotations

import argparse
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
) -> dict[str, Any]:
    from kernel_agent import toolchain

    toolchain.setup()

    import torch

    from kernel_agent.kernels.bench import compare_timing, time_call
    from kernel_agent.kernels.compare import compare_structures
    from kernel_agent.profiling.capture import load_capture

    result: dict[str, Any] = {
        "candidate": str(candidate_path),
        "status": "error",
        "correct": False,
    }
    t0 = time.perf_counter()
    capture = load_capture(capture_path, device="cuda")
    reference = capture["module"].eval()
    cases = capture["cases"]

    # 1. import + build
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

    # 2. correctness on every captured case (outputs + in-place side effects)
    case_reports: list[dict[str, Any]] = []
    all_ok = True
    for i, case in enumerate(cases):
        args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
        try:
            with torch.inference_mode():
                out = candidate(*args, **kwargs)
            torch.cuda.synchronize()
        except Exception:
            result.update(status="runtime_error", error=_short_tb(), failed_case=i)
            return result
        checks = compare_structures(case["output"], out, "output")
        checks += compare_structures(case["post_args"], args, "args")
        checks += compare_structures(case["post_kwargs"], kwargs, "kwargs")
        bad = [c for c in checks if not c.get("ok")]
        ok = not bad
        all_ok &= ok
        worst = max(
            (c for c in checks if "max_abs_err" in c), key=lambda c: c["max_abs_err"], default={}
        )
        case_reports.append(
            {
                "case": i,
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

    # 3. performance (reference vs candidate, same inputs)
    compiled = None
    if compile_baseline:
        try:
            compiled = torch.compile(copy.deepcopy(reference), mode="max-autotune-no-cudagraphs")
        except Exception:
            compiled = None
    saved = 0.0
    ref_total = 0.0
    new_total = 0.0
    for report, case in zip(case_reports, cases, strict=True):
        try:
            ref_t, new_t = compare_timing(
                reference, candidate, case["args"], case["kwargs"], l2_flush=l2_flush
            )
        except Exception:
            result.update(status="runtime_error", error=_short_tb())
            return result
        report["ref_ms"] = round(ref_t["median_ms"], 5)
        report["new_ms"] = round(new_t["median_ms"], 5)
        report["speedup"] = round(ref_t["median_ms"] / max(new_t["median_ms"], 1e-9), 3)
        report["timing_spread"] = round(max(ref_t["spread"], new_t["spread"]), 3)
        if compiled is not None:
            try:
                comp_t = time_call(compiled, case["args"], case["kwargs"], l2_flush=l2_flush)
                report["torch_compile_ms"] = round(comp_t["median_ms"], 5)
            except Exception as exc:
                report["torch_compile_ms"] = f"failed: {exc}"[:200]
        n = case["count"]
        ref_total += n * ref_t["median_ms"]
        new_total += n * new_t["median_ms"]
        saved += n * (ref_t["median_ms"] - new_t["median_ms"])

    result.update(
        status="ok",
        speedup=round(ref_total / max(new_total, 1e-9), 3),
        est_saved_ms_per_run=round(saved * capture.get("instances", 1), 3),
        ref_ms_weighted=round(ref_total, 4),
        new_ms_weighted=round(new_total, 4),
    )
    if profile:
        try:
            first = cases[0]
            result["kernels_candidate"] = _kernel_table(candidate, first["args"], first["kwargs"])
            result["kernels_reference"] = _kernel_table(reference, first["args"], first["kwargs"])
        except Exception as exc:
            result["profile_error"] = str(exc)[:500]
    result["eval_seconds"] = round(time.perf_counter() - t0, 1)
    return result


def run_evaluation(
    capture_path: Path,
    candidate_path: Path,
    *,
    profile: bool = False,
    l2_flush: bool = False,
    compile_baseline: bool = False,
    timeout: float = 900.0,
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
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "correct": False, "error": f"exceeded {timeout:.0f}s"}
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
