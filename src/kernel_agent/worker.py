"""GPU-side worker.  Every model-level step runs in a fresh subprocess so the
orchestrator never holds GPU memory and a crashing kernel cannot kill a run.

    python -m kernel_agent.worker analyze --run-dir R [--out-dir D --kernel ID=PATH ...]
    python -m kernel_agent.worker capture --run-dir R --target ID
    python -m kernel_agent.worker e2e     --run-dir R [--kernel ID=PATH ...] [--transform PATH ...]
                                          [--baseline-ms MS] [--verify REL=SHA256 ...]
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from kernel_agent import truth
from kernel_agent.gpulock import child_env, gpu_lock
from kernel_agent.workspace import RunDir, read_json, write_json

MARKER = "@@KA_WORKER@@"


def _workload(run: RunDir) -> Any:
    from kernel_agent import toolchain
    from kernel_agent.workloads import WorkloadSpec, create_workload

    toolchain.setup()
    cfg = run.load()
    spec = WorkloadSpec.from_dict(cfg["workload"])
    if spec.harness is None and run.harness.exists():
        spec.harness = str(run.harness)
    workload = create_workload(spec)
    workload.load()
    return workload


def _output_summary(output: Any) -> dict[str, Any]:
    import torch

    from kernel_agent.kernels.compare import flatten

    summary: dict[str, Any] = {}
    for name, t in list(flatten(output).items())[:8]:
        if isinstance(t, torch.Tensor):
            summary[name] = {"shape": list(t.shape), "dtype": str(t.dtype)}
    return summary


def cmd_analyze(run: RunDir, ns: argparse.Namespace) -> dict[str, Any]:
    import gc

    import torch

    from kernel_agent import strong_baseline
    from kernel_agent.profiling.profiler import profile_workload, summarize
    from kernel_agent.workloads import holdout, quality
    from kernel_agent.workloads.base import measure

    if (ns.kernel or ns.transform) and not ns.out_dir:
        raise ValueError("analyze --kernel/--transform needs --out-dir (keeps the run's baseline)")
    t0 = time.perf_counter()
    workload = _workload(run)
    load_s = time.perf_counter() - t0
    out = run  # where baseline.json + profile/ go
    if ns.out_dir:  # improve rounds: profile the optimised model next to the run's baseline
        out = RunDir(ns.out_dir.resolve())
        out.root.mkdir(parents=True, exist_ok=True)
        _apply_patches(run, workload, ns.kernel or [], ns.transform or [])
    inputs = workload.make_inputs()
    timing = measure(workload, inputs, warmup=ns.warmup, iters=ns.iters)
    output = timing.pop("output")
    torch.save(output, truth.replace(out.baseline_output()))  # .truth/ of a sealed run

    # Determinism check: a second run must pass the workload's own comparison.
    with torch.inference_mode():
        again = workload.run(inputs)
    det = workload.compare(output, again)

    baseline = {
        "workload": workload.describe(),
        "load_seconds": round(load_s, 1),
        **timing,
        "deterministic": det.passed,
        "determinism_metrics": det.metrics,
        "determinism_reason": det.reason,
        # Sensitivity probe (+ teacher-forcing self-check): is the output chaotic?
        **quality.probe(workload, inputs, output),
        "output_summary": _output_summary(output),
        "roots": {
            name: {
                "class": type(m).__name__,
                "params": sum(p.numel() for p in m.parameters()),
            }
            for name, m in workload.roots().items()
        },
    }
    if ns.out_dir:  # improve round: keep the run's eager + compiled baselines for reference
        first = read_json(run.baseline_json, {}) or {}
        baseline["run_eager_ms"] = first.get("median_ms")
        baseline.update({k: first[k] for k in ("compiled_ms", "compiled_detail") if k in first})
    else:  # the held-out input: e2e judges candidates on it too (workloads/holdout.py)
        target = truth.replace(out.baseline_output_holdout())
        baseline["holdout"] = holdout.save_baseline(workload, output, target)
    write_json(out.baseline_json, baseline)
    # Strong baseline (strong_baseline.py): the full analyze of a run measures the
    # workload's reference optimisations (or, with --compile-baseline, a generic
    # torch.compile) in a fresh process; harness checks and re-profiles do not.
    generic = bool((run.load().get("config") or {}).get("compile_baseline"))
    compiled = not (ns.out_dir or ns.no_profile) and (generic or strong_baseline.has_hook(workload))

    if not ns.no_profile:
        profile = profile_workload(workload, inputs)
        write_json(out.profile_dir / "profile.json", profile)
        summary = summarize(profile, timing["median_ms"]) + quality.summary_section(baseline)
        (out.profile_dir / "summary.md").write_text(
            summary + strong_baseline.summary_section(baseline)
        )
        if compiled:
            del workload, inputs, output, again  # release the eager model for the compiled copy
            gc.collect()
            torch.cuda.empty_cache()
            baseline.update(
                strong_baseline.run_measurement(
                    out, eager_ms=timing["median_ms"], generic=generic, iters=ns.iters
                )
            )
            write_json(out.baseline_json, baseline)
            (out.profile_dir / "summary.md").write_text(
                summary + strong_baseline.summary_section(baseline)
            )
        baseline["profile"] = str(out.profile_dir / "summary.md")
    return baseline


def _kernel_patches(run: RunDir, kernels: list[str]) -> list[Any]:
    """``KernelPatch`` per ``TARGET_ID=PATH`` item (``e2e`` and the re-profile of ``analyze``)."""
    from kernel_agent.integrate.patcher import KernelPatch

    patches = []
    for item in kernels:
        target_id, _, path = item.partition("=")
        spec = read_json(run.target(target_id) / "spec.json")
        patches.append(
            KernelPatch(
                target_id=target_id,
                module_class=spec["module_class"],
                candidate=Path(path),
                qualname_regex=spec.get("qualname_regex"),
                methods=list(spec.get("capture", {}).get("method_instances", [])),
                phase=spec.get("phase"),
            )
        )
    return patches


def _apply_patches(run: RunDir, workload: Any, kernels: list[str], transforms: list[str]) -> None:
    """Apply kernel replacements and transform files as ``e2e`` does (errors propagate)."""
    from kernel_agent.integrate.patcher import PatchReport, apply_kernels, apply_transforms

    report = PatchReport()
    apply_kernels(workload.roots(), _kernel_patches(run, kernels), report)
    apply_transforms(workload, [Path(p) for p in transforms], report)


def cmd_capture(run: RunDir, ns: argparse.Namespace) -> dict[str, Any]:
    from kernel_agent.profiling.capture import capture_module

    target_dir = run.target(ns.target)
    spec = read_json(target_dir / "spec.json")
    workload = _workload(run)
    inputs = workload.make_inputs()
    with __import__("torch").inference_mode():
        workload.run(inputs)  # warm-up so lazily-initialised state exists
    capture = truth.replace(run.capture_file(ns.target))  # .truth/captures/ of a sealed run
    info = capture_module(
        workload,
        inputs,
        spec["module_class"],
        capture,
        qualname=spec.get("qualname"),
        max_cases=int(ns.max_cases),
        qualname_regex=spec.get("qualname_regex"),
        phase=spec.get("phase"),
        profile_dir=target_dir,  # workload_profile.md is for the agent
        variants=workload.variants(),  # extra settings: correctness-only cases
    )
    if run.sealed():  # the agent's copy: module + inputs, no reference outputs
        truth.write_inputs_capture(capture, target_dir / "capture_inputs.pt")
    spec["capture"] = info
    write_json(target_dir / "spec.json", spec)
    _write_reference_source(workload, spec, target_dir / "reference_source.py")
    return info


def _write_reference_source(workload: Any, spec: dict[str, Any], path: Path) -> None:
    """Dump the target class source (plus its file path) for the kernel engineer."""
    import inspect

    from kernel_agent.profiling.capture import find_instance

    qualname = spec.get("capture", {}).get("qualname") or spec.get("qualname")  # captured one
    _, module = find_instance(workload.roots(), spec["module_class"], qualname)
    cls = type(module)
    try:
        file = inspect.getsourcefile(cls)
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        file, source = None, "# source unavailable"
    children = "\n".join(
        f"#   {name}: {type(child).__name__}" for name, child in module.named_children()
    )
    path.write_text(
        f"# Reference implementation of {cls.__module__}.{cls.__qualname__}\n"
        f"# defined in: {file}\n# instance repr:\n"
        + "\n".join(f"#   {line}" for line in repr(module).splitlines()[:40])
        + f"\n# direct children:\n{children}\n\n{source}\n"
    )


def cmd_e2e(run: RunDir, ns: argparse.Namespace) -> dict[str, Any]:
    import torch

    from kernel_agent.integrate.patcher import PatchReport, apply_kernels, apply_transforms
    from kernel_agent.workloads import holdout
    from kernel_agent.workloads.base import measure
    from kernel_agent.workloads.quality import assess, is_chaotic

    # The baseline outputs and baseline.json, checked against the digests the
    # orchestrator holds (--verify) before anything runs; read once, used later.
    expected = dict(item.partition("=")[::2] for item in ns.verify or [])
    try:
        reference_bytes = _truth_bytes(run, run.baseline_output(), expected)
        baseline_bytes = _truth_bytes(run, run.baseline_json, expected, required=False)
        holdout_bytes = _truth_bytes(run, run.baseline_output_holdout(), expected, required=False)
    except truth.TamperError as exc:
        return {"status": "tampered", "passed": False, "error": str(exc)}
    workload = _workload(run)
    patches = _kernel_patches(run, ns.kernel or [])
    report = PatchReport()
    try:
        apply_kernels(workload.roots(), patches, report)
        apply_transforms(workload, [Path(p) for p in ns.transform or []], report)
    except Exception:
        return {"status": "patch_error", "passed": False, "error": traceback.format_exc()[-4000:]}

    inputs = workload.make_inputs()
    try:
        timing = measure(workload, inputs, warmup=ns.warmup, iters=ns.iters)
    except Exception:
        return {
            "status": "runtime_error",
            "passed": False,
            "patches": report.__dict__,
            "error": traceback.format_exc()[-4000:],
        }
    output = timing.pop("output")
    reference = torch.load(io.BytesIO(reference_bytes), weights_only=False)
    baseline = json.loads(baseline_bytes or b"{}")
    # the orchestrator's baseline latency (--baseline-ms), not what baseline.json says now
    base_ms = ns.baseline_ms or float(baseline.get("median_ms", 0.0)) or float("nan")
    # Timing is free-running; quality is teacher-forced when the workload supports it.
    chaotic = is_chaotic(workload, baseline)
    try:
        verdict = assess(workload, inputs, reference, output, chaotic=chaotic)
    except Exception as exc:
        return {
            "status": "runtime_error",
            "passed": False,
            "reason": f"quality check failed: {exc}"[:500],
            "median_ms": round(timing["median_ms"], 3),
            "patches": report.__dict__,
            "error": traceback.format_exc()[-4000:],
        }
    # Held-out input (untimed) + memoisation probe; failures are verdicts, not errors.
    held = holdout.check(
        workload,
        torch.load(io.BytesIO(holdout_bytes), weights_only=False) if holdout_bytes else None,
        reference,
        main_output=output,
        main_ms=timing["median_ms"],
        baseline=baseline,
        chaotic=chaotic,
    )
    reasons = [verdict["reason"], held["reason"] and f"held-out input: {held['reason']}"]
    return {
        "status": "ok",
        "passed": verdict["passed"] and held["passed"],
        "reason": "; ".join(r for r in reasons if r),
        "metrics": {**verdict["metrics"], "holdout": held},
        "median_ms": round(timing["median_ms"], 3),
        "times_ms": [round(t, 3) for t in timing["times_ms"]],
        "baseline_ms": round(base_ms, 3),
        "speedup": round(base_ms / timing["median_ms"], 4),
        "peak_mem_gb": round(timing["peak_mem_gb"], 3),
        "patches": report.__dict__,
    }


def _truth_bytes(
    run: RunDir, path: Path, expected: dict[str, str], *, required: bool = True
) -> bytes:
    """A run file checked against its ``--verify`` digest (unchecked and missing: b""
    unless ``required``)."""
    rel = path.relative_to(run.root).as_posix()
    sha256 = expected.get(rel)
    if sha256 is None:
        return path.read_bytes() if required or path.exists() else b""
    try:
        return truth.read_verified(path, sha256)
    except (truth.TamperError, OSError) as exc:
        truth.alarm(run, rel, str(exc) if isinstance(exc, truth.TamperError) else "missing")
        raise truth.TamperError(f"{rel}: {exc}") from None


COMMANDS = {"analyze": cmd_analyze, "capture": cmd_capture, "e2e": cmd_e2e}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernel_agent.worker")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--target")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--kernel", action="append", help="TARGET_ID=CANDIDATE_PATH")
    parser.add_argument("--transform", action="append", help="transform .py path")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--out-dir", type=Path, help="analyze: write baseline + profile here")
    parser.add_argument("--baseline-ms", type=float, help="e2e: the speedup denominator")
    parser.add_argument(
        "--verify", action="append", help="e2e: REL=SHA256, refuse a run file without it"
    )
    ns = parser.parse_args(argv)
    run = RunDir(ns.run_dir.resolve())
    try:
        result = COMMANDS[ns.command](run, ns)
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()[-6000:]}
    print(MARKER + json.dumps(result, default=str), flush=True)
    return 1 if "error" in result else 0


def call_worker(run: RunDir, command: str, *args: str, timeout: float = 3600.0) -> dict[str, Any]:
    """Run a worker command under the GPU lock; returns its JSON result."""
    cmd = [sys.executable, "-m", "kernel_agent.worker", command, "--run-dir", str(run.root), *args]
    log = run.root / "logs" / f"worker-{command}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock():
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=child_env()
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"worker {command} exceeded {timeout:.0f}s"}
    with log.open("a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n{proc.stdout[-20000:]}\n{proc.stderr[-20000:]}\n")
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith(MARKER):
            data: dict[str, Any] = json.loads(line[len(MARKER) :])
            return data
    return {
        "status": "crash",
        "returncode": proc.returncode,
        "error": (proc.stderr or proc.stdout)[-6000:],
    }


if __name__ == "__main__":
    raise SystemExit(main())
