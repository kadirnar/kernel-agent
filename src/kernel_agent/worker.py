"""GPU-side worker.  Every model-level step runs in a fresh subprocess so the
orchestrator never holds GPU memory and a crashing kernel cannot kill a run.

    python -m kernel_agent.worker analyze --run-dir R [--out-dir D --kernel ID=PATH ...]
    python -m kernel_agent.worker capture --run-dir R --target ID [--parent]
    python -m kernel_agent.worker e2e     --run-dir R [--kernel ID=PATH ...] [--transform PATH ...]
                                          [--baseline-ms MS] [--verify REL=SHA256 ...]
    python -m kernel_agent.worker e2e_ab  --run-dir R [A: --kernel ... --transform ...]
                                          [B: --b-kernel ... --b-transform ...] [--rounds K]

A step that runs out of GPU memory has status ``oom`` (``abtest.OOM``), also when a check
that catches its own errors ran out (the perceptual gate, say: no quality verdict); the
integration measures it again in separate processes with ``--expandable-segments``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kernel_agent import precisions, truth
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

    from kernel_agent import objective, strong_baseline
    from kernel_agent.kernels.roofline import current_peaks
    from kernel_agent.profiling import ceilings
    from kernel_agent.profiling.profiler import profile_workload, summarize
    from kernel_agent.workloads import holdout, perceptual, quality, stopping
    from kernel_agent.workloads.base import measure

    if (ns.kernel or ns.transform) and not ns.out_dir:
        raise ValueError("analyze --kernel/--transform needs --out-dir (keeps the run's baseline)")
    t0 = time.perf_counter()
    workload = _workload(run)
    load_s = time.perf_counter() - t0
    out = run  # where baseline.json + profile/ go
    work = None
    if ns.out_dir:  # improve rounds: profile the optimised model next to the run's baseline
        out = RunDir(ns.out_dir.resolve())
        out.root.mkdir(parents=True, exist_ok=True)
        if (ns.kernel or ns.transform) and not ns.no_profile:
            work = _work_reference(workload)  # before the items hide it from the hooks
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
        # ... and the stop condition on a natural-length run (workloads/stopping.py)
        target = truth.replace(out.baseline_output_natural())
        baseline["natural_length"] = stopping.save_baseline(workload, target)
        # ... and, with --quality near-lossless, the perceptual samples (workloads/perceptual.py)
        target = truth.replace(out.baseline_output_perceptual())
        mode = _quality(ns, run)
        baseline["perceptual"] = perceptual.save_baseline(workload, target, quality=mode)
    write_json(out.baseline_json, baseline)
    # Strong baseline (strong_baseline.py): the full analyze of a run measures the
    # workload's reference optimisations (or, with --compile-baseline, a generic
    # torch.compile) in a fresh process; harness checks and re-profiles do not.
    generic = bool((run.load().get("config") or {}).get("compile_baseline"))
    compiled = not (ns.out_dir or ns.no_profile) and (generic or strong_baseline.has_hook(workload))

    if not ns.no_profile:
        window_ms, what, per = objective.profile_window(baseline)  # metric=throughput: a run
        with workload.metric_window():  # metric=ttfa: the run up to the first audio chunk
            profile = profile_workload(workload, inputs, reference_ms=window_ms, work=work)
        write_json(out.profile_dir / "profile.json", profile)
        # Floors per class at bf16 / FP8 / FP4 (profile/ceilings.json + .md; issue #90); the
        # markdown shows the precisions the run allows (precisions.py, issue #131).
        table = ceilings.write(
            out.profile_dir, profile, current_peaks(), window_ms, per=per, allowed=_allowed(ns, run)
        )
        summary = (
            summarize(profile, window_ms, metric=what, per=per)
            + ceilings.markdown(table)
            + objective.summary_section(baseline)
            + quality.summary_section(baseline)
        )
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


def _work_reference(workload: Any) -> Any:
    """The work of the unmodified model's calls in the profiled window
    (``profiler.work_reference``): the ceilings of the optimised model take it for the
    calls whose insides the hooks do not see (#106). None when the pass fails."""
    import gc

    import torch

    from kernel_agent.profiling.profiler import work_reference

    try:
        inputs = workload.make_inputs()
        with workload.metric_window():
            work = work_reference(workload, inputs)
        print(f"analyze: work of the unmodified model: {work.info()}", file=sys.stderr)
        return work
    except Exception:
        print(f"analyze: no work of the unmodified model:\n{_tb()}", file=sys.stderr)
        return None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()  # peak_mem_gb is the optimised model's


def _kernel_patches(run: RunDir, kernels: list[str]) -> list[Any]:
    """``KernelPatch`` per ``TARGET_ID=PATH`` item (``e2e`` and the re-profile of ``analyze``)."""
    from kernel_agent.integrate.patcher import KernelPatch
    from kernel_agent.region import rewrite_of

    patches = []
    for item in kernels:
        target_id, _, path = item.partition("=")
        spec = read_json(run.target(target_id) / "spec.json")
        rewrite = rewrite_of(run, target_id, spec)  # a region target: its regex is the parent's
        patches.append(
            KernelPatch(
                target_id=target_id,
                module_class=spec["module_class"],
                candidate=Path(path),
                qualname_regex=None if rewrite else spec.get("qualname_regex"),
                methods=list(spec.get("capture", {}).get("method_instances", [])),
                phase=spec.get("phase"),
                rewrite=rewrite,
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
    from kernel_agent import region
    from kernel_agent.kernels.compare import EXACT_TIER, tier_for
    from kernel_agent.profiling.capture import capture_module

    target_dir = run.target(ns.target)
    spec = read_json(target_dir / "spec.json")
    reduced = tier_for(_quality(ns, run), spec.get("precision")) != EXACT_TIER
    if reduced and (why := precisions.refusal(spec.get("precision"), _allowed(ns, run))):
        raise ValueError(f"capture of {ns.target} refused: {why}")  # e.g. 4-bit, not asked for
    workload = _workload(run)
    cls, out, capture = spec["module_class"], target_dir, run.capture_file(ns.target)
    scope = {k: spec.get(k) for k in ("qualname", "qualname_regex", "phase")}
    parent = getattr(ns, "parent", False)
    if parent:  # region target (region.py), first: its parent class, every phase
        cls, out = spec["parent_class"], target_dir / region.PARENT_DIR
        capture, scope["phase"] = region.parent_capture(run, ns.target), None
    elif (rewrite := region.rewrite_of(run, ns.target, spec)) is not None:
        region.apply_rewrites(workload.roots(), [rewrite])  # then Region_<id> in the rewrite
        scope["qualname"] = scope["qualname_regex"] = None  # they select parent instances
    inputs = workload.make_inputs()
    with __import__("torch").inference_mode():
        workload.run(inputs)  # warm-up so lazily-initialised state exists
    capture = truth.replace(capture)  # .truth/captures/ of a sealed run
    # near-lossless tolerances for a target whose spec allows reduced precision, recorded
    # in the sealed capture (kernels/compare.py): the agent's spec.json cannot change it
    tier = tier_for(_quality(ns, run), spec.get("precision")) if not parent else EXACT_TIER
    info = capture_module(
        workload,
        inputs,
        cls,
        capture,
        qualname=scope["qualname"],
        max_cases=int(ns.max_cases),
        qualname_regex=scope["qualname_regex"],
        phase=scope["phase"],
        profile_dir=out,  # workload_profile.md is for the agent
        variants=workload.variants(),  # extra settings: correctness-only cases
        tier=None if tier == EXACT_TIER else tier,
        precision=None if tier == EXACT_TIER else spec.get("precision"),  # roofline bytes
    )
    if run.sealed():  # the agent's copy: module + inputs, no reference outputs
        truth.write_inputs_capture(capture, out / "capture_inputs.pt")
    spec["parent_capture" if parent else "capture"] = info
    write_json(target_dir / "spec.json", spec)
    _write_reference_source(
        workload, {"module_class": cls, "capture": info}, out / "reference_source.py"
    )
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
    from kernel_agent.integrate.patcher import PatchReport, apply_kernels, apply_transforms
    from kernel_agent.telemetry import Monitor
    from kernel_agent.workloads.base import measure

    try:
        truth_files = _truth_files(run, ns)
    except truth.TamperError as exc:
        return {"status": "tampered", "passed": False, "error": str(exc)}
    workload = _workload(run)
    patches = _kernel_patches(run, ns.kernel or [])
    report = PatchReport()
    try:
        apply_kernels(workload.roots(), patches, report)
        apply_transforms(workload, [Path(p) for p in ns.transform or []], report)
    except Exception:
        return _failed("patch_error", patches=report.__dict__)

    inputs = workload.make_inputs()
    monitor = Monitor()  # GPU clocks / temperature / power before and after the timing
    monitor.sample("before", loaded=False)  # the clocks may still be idling
    try:
        timing = measure(workload, inputs, warmup=ns.warmup, iters=ns.iters)
    except Exception:
        return _failed("runtime_error", patches=report.__dict__)
    monitor.sample("after")
    output = timing.pop("output")
    base_ms, verdict = _judge(ns, workload, inputs, output, timing["median_ms"], truth_files)
    if verdict["status"] != "ok":
        return {**verdict, "median_ms": round(timing["median_ms"], 3), "patches": report.__dict__}
    return {
        **verdict,
        "median_ms": round(timing["median_ms"], 3),
        "times_ms": [round(t, 3) for t in timing["times_ms"]],
        "baseline_ms": round(base_ms, 3),
        "speedup": round(base_ms / timing["median_ms"], 4),
        "metric": timing["metric"],  # what median_ms is (objective.py)
        **({"metric_detail": d} if (d := timing.get("metric_detail")) else {}),
        "peak_mem_gb": round(timing["peak_mem_gb"], 3),
        "patches": report.__dict__,
        **({"gpu": gpu} if (gpu := monitor.summary()) else {}),
    }


def _truth_files(run: RunDir, ns: argparse.Namespace) -> tuple[bytes, ...]:
    """The baseline output, baseline.json, the held-out baseline output, the
    natural-length baseline and the perceptual baseline, checked against the digests the
    orchestrator holds (``--verify``) before anything runs; read once."""
    expected = dict(item.partition("=")[::2] for item in ns.verify or [])
    return (
        _truth_bytes(run, run.baseline_output(), expected),
        _truth_bytes(run, run.baseline_json, expected, required=False),
        _truth_bytes(run, run.baseline_output_holdout(), expected, required=False),
        _truth_bytes(run, run.baseline_output_natural(), expected, required=False),
        _truth_bytes(run, run.baseline_output_perceptual(), expected, required=False),
    )


def _quality(ns: argparse.Namespace, run: RunDir | None = None) -> str:
    """The quality mode: ``--quality`` (the orchestrator's, from its memory), else
    ``run.json``'s (``exact`` when unset)."""
    from kernel_agent.workloads.perceptual import mode_of

    if getattr(ns, "quality", None):
        return mode_of(ns.quality)
    if run is None and getattr(ns, "run_dir", None) is not None:
        run = RunDir(Path(ns.run_dir).resolve())
    return mode_of(((run.load() if run else {}).get("config") or {}).get("quality"))


def _allowed(ns: argparse.Namespace, run: RunDir) -> tuple[str, ...]:
    """The target precisions the run allows (``precisions.py``): ``--precisions`` (the
    orchestrator's, from its memory), else ``run.json``'s, in the run's quality mode."""
    raw = getattr(ns, "precisions", None)
    listed = raw.split(",") if raw else (run.load().get("config") or {}).get("precisions")
    return precisions.allowed(_quality(ns, run), listed)


def _judge(
    ns: argparse.Namespace,
    workload: Any,
    inputs: Any,
    output: Any,
    median_ms: float,
    truth_files: tuple[bytes, ...],
    before_scoring: Callable[[], None] | None = None,
) -> tuple[float, dict[str, Any]]:
    """(baseline ms, quality verdict) of a timed candidate output: ``status`` ok with
    ``passed`` / ``reason`` / ``metrics``, ``runtime_error`` when a check crashed, or
    ``oom`` when one ran out of GPU memory (no verdict). ``--quality near-lossless`` with a
    perceptual baseline: the checks of :func:`_checks` with the workload's sanity floor,
    then the perceptual gate (workloads/perceptual.py; ``before_scoring`` runs after its
    samples, the model's last run)."""
    import torch

    from kernel_agent.workloads import perceptual

    *files, perceptual_bytes = truth_files
    mode = _quality(ns)
    reference: dict[str, Any] | None = None
    if perceptual_bytes and mode == perceptual.NEAR_LOSSLESS:
        reference = torch.load(io.BytesIO(perceptual_bytes), weights_only=False)
    with perceptual.judging(workload, mode, reference) as gate:
        base_ms, verdict = _checks(ns, workload, inputs, output, median_ms, tuple(files))
    verdict = _out_of_memory(verdict)
    if verdict["status"] != "ok":
        return base_ms, verdict
    result: dict[str, Any] | None
    if reference is None or not gate:
        result = perceptual.skipped(mode, json.loads(files[1] or b"{}"))
    elif not verdict["passed"]:  # rejected already: spare the gate's runs
        result = {"passed": False, "reason": "", "skipped": "the other checks failed"}
    else:
        result = perceptual.check(workload, reference, before_scoring=before_scoring)
    if result is not None:
        verdict["metrics"]["perceptual"] = result
        verdict["passed"] = verdict["passed"] and result["passed"]
        reasons = [verdict["reason"], result["reason"] and f"perceptual: {result['reason']}"]
        verdict["reason"] = "; ".join(r for r in reasons if r)
    return base_ms, _out_of_memory(verdict)


def _out_of_memory(verdict: dict[str, Any]) -> dict[str, Any]:
    """``verdict`` as status ``oom`` when one of the checks that catch their own errors (the
    held-out input, the natural-length run, the perceptual gate: ``abtest.CHECKS``) ran out
    of GPU memory: what else the process held (two states of an A/B, the gate's scoring
    models next to them), not a quality verdict. Such a check's record says so (``oom``,
    ``passed`` None); the integration measures the step again in separate processes (#137)."""
    from kernel_agent import abtest

    if verdict.get("status") != "ok":
        return verdict
    found = abtest.checks_out_of_memory(verdict.get("metrics"))
    if not found:
        return verdict
    metrics = dict(verdict["metrics"])
    for key, line in found.items():
        reason = f"out of GPU memory, not judged: {line}"
        metrics[key] = {**metrics[key], "passed": None, "oom": True, "reason": reason}
    reason = abtest.checks_reason(found)
    return {**verdict, "status": abtest.OOM, "passed": False, "reason": reason, "metrics": metrics}


def _checks(
    ns: argparse.Namespace,
    workload: Any,
    inputs: Any,
    output: Any,
    median_ms: float,
    truth_files: tuple[bytes, ...],
) -> tuple[float, dict[str, Any]]:
    """:func:`_judge` without the perceptual gate: teacher forcing (or the workload's own
    comparison), the held-out input and the stop condition."""
    import torch

    from kernel_agent.workloads import holdout, stopping
    from kernel_agent.workloads.quality import assess, is_chaotic

    reference_bytes, baseline_bytes, holdout_bytes, natural_bytes = truth_files
    reference = torch.load(io.BytesIO(reference_bytes), weights_only=False)
    baseline = json.loads(baseline_bytes or b"{}")
    # the orchestrator's baseline latency (--baseline-ms), not what baseline.json says now
    base_ms = ns.baseline_ms or float(baseline.get("median_ms", 0.0)) or float("nan")
    # Timing is free-running; quality is teacher-forced when the workload supports it.
    chaotic = is_chaotic(workload, baseline)
    try:
        verdict = assess(workload, inputs, reference, output, chaotic=chaotic)
    except Exception as exc:
        return base_ms, _failed("runtime_error", reason=f"quality check failed: {exc}"[:500])
    # Held-out input (untimed) + memoisation probe; failures are verdicts, not errors.
    held = holdout.check(
        workload,
        torch.load(io.BytesIO(holdout_bytes), weights_only=False) if holdout_bytes else None,
        reference,
        main_output=output,
        main_ms=median_ms,
        baseline=baseline,
        chaotic=chaotic,
    )
    reasons = [verdict["reason"], held["reason"] and f"held-out input: {held['reason']}"]
    metrics = {**verdict["metrics"], "holdout": held}
    # Stop condition (untimed natural-length run); None: the workload has none.
    natural = stopping.check(
        workload,
        torch.load(io.BytesIO(natural_bytes), weights_only=False) if natural_bytes else None,
        baseline,
    )
    if natural is not None:
        metrics["natural_length"] = natural
        reasons.append(natural["reason"] and f"natural length: {natural['reason']}")
    return base_ms, {
        "status": "ok",
        "passed": verdict["passed"] and held["passed"] and (natural or {}).get("passed", True),
        "reason": "; ".join(r for r in reasons if r),
        "metrics": metrics,
    }


def cmd_e2e_ab(run: RunDir, ns: argparse.Namespace) -> dict[str, Any]:
    """Paired A/B (``abtest.py``): the model loaded once, A (``--kernel`` / ``--transform``)
    and B (``--b-kernel`` / ``--b-transform``) built on it with undo handles
    (``integrate/ab.py``), warmed up, then alternated for ``--rounds`` timed rounds; B's
    quality is checked once, as in ``e2e``; A's state is freed before the perceptual gate's
    scoring models load (``ab.released_gb``).
    Status ``irreversible`` / ``undo_failed``: the states cannot be switched in-process,
    measure them in separate processes; so does ``oom`` (out of GPU memory anywhere in the
    step, the perceptual gate included). A failure reports what each state applied so far
    (``patches``, ``ab.a_patches``: the modules each item touched)."""
    import torch

    from kernel_agent import abtest, telemetry
    from kernel_agent.integrate import ab, undo

    try:
        truth_files = _truth_files(run, ns)
    except truth.TamperError as exc:
        return {"status": "tampered", "passed": False, "error": str(exc)}
    a_kernels, a_transforms = ns.kernel or [], ns.transform or []
    b_kernels, b_transforms = ns.b_kernel or [], ns.b_transform or []
    declared = [t for t in [*a_transforms, *b_transforms] if undo.declaration(Path(t)) is False]
    if declared:
        return _irreversible([([t], "it declares undo = False") for t in dict.fromkeys(declared)])
    workload = _workload(run)
    session = ab.Session(workload, lambda items: _kernel_patches(run, items))
    inputs = workload.make_inputs()
    monitor = telemetry.Monitor()
    if torch.cuda.is_available():
        from kernel_agent.kernels.bench import warm_gpu

        torch.cuda.reset_peak_memory_stats()
        warm_gpu(500.0)  # the same clock warm-up as `e2e`
    reference: dict[str, Any] = {}
    reproducible: dict[str, bool] = {}
    states: dict[str, ab.State] = {}

    def failed(status: str, why: str | None = None, error: str | None = None) -> dict[str, Any]:
        if session.built is not None:  # the state that failed to build: what it applied
            states.setdefault(session.built.name, session.built)
        a_patches = states["A"].report.__dict__ if "A" in states else None
        return _failed(
            status,
            error,
            reason=why,
            patches=states["B"].report.__dict__ if "B" in states else None,
            ab={"a_patches": a_patches} if a_patches is not None else None,
        )

    try:
        a = states["A"] = session.build("A", a_kernels, a_transforms)
        if bad := a.irreversible(keep=session.shareable(a, b_kernels)):
            return _irreversible(bad)
        reference["A"], reproducible["A"] = ab.warm(session, a, inputs, ns.warmup + 1)
    except Exception:
        return failed("undo_failed", "the accepted set A failed in-process")
    # No torch.cuda.empty_cache() between A and B: releasing the cached segments turned an
    # out-of-bounds access of a VAE kernel that cached memory absorbs into an illegal
    # address (#112); a step that does not fit is measured in two processes instead.
    try:
        b = states["B"] = session.build("B", b_kernels, b_transforms, on=a)
    except Exception:  # its own items, or A's applied again: the separate processes tell
        return failed("undo_failed", "B failed to apply in-process")
    if bad := b.irreversible():
        return _irreversible(bad)
    try:
        reference["B"], reproducible["B"] = ab.warm(session, b, inputs, ns.warmup + 1)
    except Exception:
        return failed("runtime_error")
    rounds = ab.alternate(
        session, (a, b), inputs, reference, reproducible, rounds=ns.rounds, sample=monitor.sample
    )
    if rounds.failed == "A" or rounds.mismatch:
        why = rounds.mismatch or "A failed after a switch from B"
        return failed("undo_failed", why, rounds.error or "")
    if rounds.failed == "B":
        return failed("runtime_error", None, rounds.error or "")
    session.to(b)
    released: list[int] = []

    def free_a() -> None:
        """After the gate's samples, the model's last run: keep B for good and free A's state
        (and the original modules the states replaced), so the gate's scoring models load next
        to one state, as in a fresh `e2e` process of B (#137). Not before B's checks: a model
        whose memory moved under it can turn an out-of-bounds read that cached memory
        absorbs into an illegal address (#112; an FP8 swap of VoxCPM2 did, #137)."""
        held = _allocated()
        session.keep(b, a)
        released.append(held - _allocated())

    b_ms = ab.median(rounds.b_ms)
    base_ms, verdict = _judge(ns, workload, inputs, rounds.output, b_ms, truth_files, free_a)
    gpu = monitor.summary()
    if message := telemetry.warning(gpu):
        print(f"e2e_ab: WARNING {message}", file=sys.stderr, flush=True)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    result = {
        **verdict,
        "median_ms": round(b_ms, 3),
        "times_ms": [round(t, 3) for t in rounds.b_ms],
        "baseline_ms": round(base_ms, 3),
        "speedup": round(base_ms / b_ms, 4),
        "metric": workload.metric,  # what the rounds timed (objective.py)
        "peak_mem_gb": round(peak, 3),
        "patches": b.report.__dict__,
    }
    result["ab"] = {
        **abtest.paired(rounds.a_ms, rounds.b_ms),
        **rounds.ab(),
        "warmup": max(ns.warmup + 1, 2),
        "shared_kernels": bool(b.shared),
        **({"released_gb": round(released[0] / 1024**3, 3)} if released else {}),  # free_a
        "a_patches": a.report.__dict__,
        **({"gpu": gpu} if gpu else {}),
    }
    return result


def _allocated() -> int:
    """Bytes of GPU memory the tensors of this process hold (0 without CUDA)."""
    import torch

    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


def _irreversible(bad: list[tuple[list[str], str]]) -> dict[str, Any]:
    """``e2e_ab`` result: these items cannot be undone in-process ((items, why) pairs)."""
    why = "; ".join(f"{', '.join(Path(i).name for i in items)}: {w}" for items, w in bad)
    return {
        "status": "irreversible",
        "passed": False,
        "reason": f"cannot be undone in-process: {why}",
        "irreversible": [i for items, _ in bad for i in items],
    }


def _tb() -> str:
    return traceback.format_exc()[-4000:]


def _failed(status: str, error: str | None = None, **extra: Any) -> dict[str, Any]:
    """A failed step: ``status``, or ``oom`` when ``error`` (default: the exception being
    handled) says the GPU ran out of memory; ``extra`` keys that are None are left out."""
    from kernel_agent import abtest

    error = _tb() if error is None else error
    result: dict[str, Any] = {"status": status, "passed": False}
    result.update({k: v for k, v in extra.items() if v is not None})
    if oom := abtest.out_of_memory(error):
        reasons = [result.get("reason"), f"out of GPU memory: {oom}"]
        result.update(status=abtest.OOM, reason="; ".join(r for r in reasons if r))
    return {**result, "error": error}


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


COMMANDS = {
    "analyze": cmd_analyze,
    "capture": cmd_capture,
    "e2e": cmd_e2e,
    "e2e_ab": cmd_e2e_ab,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernel_agent.worker")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--target")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--parent", action="store_true", help="capture: a region's parent class")
    parser.add_argument("--kernel", action="append", help="TARGET_ID=CANDIDATE_PATH")
    parser.add_argument("--transform", action="append", help="transform .py path")
    parser.add_argument("--b-kernel", action="append", help="e2e_ab: a kernel of state B")
    parser.add_argument("--b-transform", action="append", help="e2e_ab: a transform of B")
    parser.add_argument("--rounds", type=int, default=8, help="e2e_ab: timed A/B rounds")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--out-dir", type=Path, help="analyze: write baseline + profile here")
    parser.add_argument("--baseline-ms", type=float, help="e2e: the speedup denominator")
    parser.add_argument(
        "--verify", action="append", help="e2e: REL=SHA256, refuse a run file without it"
    )
    parser.add_argument("--quality", help="exact | near-lossless (default: run.json's)")
    parser.add_argument(
        "--precisions", help="P,P,...: the precisions the run allows (default: run.json's)"
    )
    parser.add_argument(
        "--expandable-segments",
        action="store_true",
        help="PYTORCH_CUDA_ALLOC_CONF expandable_segments:True (the retry of an oom step)",
    )
    ns = parser.parse_args(argv)
    if ns.expandable_segments:  # before the CUDA caching allocator starts
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
            filter(None, [conf, "expandable_segments:True"])
        )
    run = RunDir(ns.run_dir.resolve())
    try:
        result = COMMANDS[ns.command](run, ns)
    except Exception:
        result = _failed("error", traceback.format_exc()[-6000:])
        result.pop("passed")  # an error of the command, not a verdict (as before)
    print(MARKER + json.dumps(result, default=str), flush=True)
    return 1 if "error" in result else 0


def call_worker(run: RunDir, command: str, *args: str, timeout: float = 3600.0) -> dict[str, Any]:
    """Run a worker command under the GPU lock; returns its JSON result, with the GPU of
    the pool it ran on (``gpu_index``)."""
    cmd = [sys.executable, "-m", "kernel_agent.worker", command, "--run-dir", str(run.root), *args]
    log = run.root / "logs" / f"worker-{command}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock() as gpu:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=child_env()
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"worker {command} exceeded {timeout:.0f}s",
                "gpu_index": gpu,
            }
    with log.open("a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n{proc.stdout[-20000:]}\n{proc.stderr[-20000:]}\n")
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith(MARKER):
            data: dict[str, Any] = json.loads(line[len(MARKER) :])
            data["gpu_index"] = gpu
            return data
    return {
        "status": "crash",
        "returncode": proc.returncode,
        "error": (proc.stderr or proc.stdout)[-6000:],
        "gpu_index": gpu,
    }


if __name__ == "__main__":
    raise SystemExit(main())
