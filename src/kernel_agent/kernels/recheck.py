"""Independent re-check of a kernel winner (issue #21).

:mod:`kernels.evaluate` checks a candidate on the captured inputs and times it next
to the reference, in the candidate's own process. The re-check repeats that verdict
from scratch, with nothing shared with the evaluation:

1. a *reference* subprocess (it never imports a candidate) draws ``seeds`` fresh
   inputs per captured case (:func:`fresh_inputs`): the captured shapes, dtypes,
   strides and aliasing; floating-point tensors redrawn from each channel's own
   statistics (:func:`kernels.verify.perturb_`: a normal draw for the first seed, a
   uniform / Laplace / log-normal one for the others; masks, rotary tables and non-finite
   tensors stay); integer and boolean tensors (ids, positions, masks) as captured; mutable
   state objects (KV caches) copied as captured, and a stateful module's state set to
   the case's before every call (:mod:`kernel_agent.profiling.state`, in both
   subprocesses). It saves the inputs, computes the
   reference outputs and post-call state on them and times the reference on the
   timed cases (``count`` > 0, the first seed's inputs);
2. the parent loads those expected results into its memory and deletes them from
   disk; then a *candidate* subprocess builds the candidate, runs it on the same
   fresh inputs, saves its outputs and post-call state, and times it the same way
   (:func:`kernels.bench.time_call`, median of ``rounds`` rounds, after
   :func:`kernels.bench.warm_gpu`, in the timing context of the evaluator's verdict: eager or
   CUDA graph, warm or cold L2, #226; a graph capture that fails leaves the re-check untimed)
   and runs the fresh inputs once more after timing
   (a kernel that changes behaviour after its first calls), under the evaluator's
   integrity snapshot;
3. the parent compares them with the strict comparator (:mod:`kernels.compare`:
   outputs and in-place side effects, a reduced-precision tier with its bounds for
   redrawn inputs; files loaded with ``weights_only``) and the
   weighted speedup (calls per run x ms, as the evaluator weighs it) with the
   evaluator's verdict.

The re-check fails (``passed`` False) when the candidate is wrong on any fresh input
(``incorrect``), fails to build or run, changes watched state, or when its speedup and
the evaluator's disagree (``disagrees``, :func:`speedups_agree`: faster or slower).
``agrees`` says whether correctness and speedup match the evaluator's verdict. The
integration keeps a correct kernel whose speedups disagree as a warning
(:func:`speed_warning`) and lets its end-to-end A/B decide. A kernel that passes then runs
once under ``compute-sanitizer`` memcheck (:mod:`kernels.memcheck`, ``memcheck`` in the
record); memory errors refuse it with status ``memcheck``.

Subprocesses::

    python -m kernel_agent.kernels.recheck reference CAPTURE --workdir DIR --seed S --seeds N
    python -m kernel_agent.kernels.recheck candidate CAPTURE CANDIDATE --workdir DIR
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from kernel_agent.gpulock import child_env, gpu_lock

RESULT_MARKER = "@@KA_RECHECK@@"
#: Two speedups of a kernel agree when each is within a factor ``1 + tolerance`` of the
#: other (:func:`speedups_agree`); the tolerance is twice the larger timing spread of the
#: two measurements, at least SPEEDUP_TOLERANCE and at most SPEEDUP_TOLERANCE_MAX (noisy
#: rounds cannot make 3x and 5x agree).
SPEEDUP_TOLERANCE = 0.25
SPEEDUP_TOLERANCE_MAX = 0.5
#: ``status`` of a re-check whose speedup disagrees with the evaluator's (set by :func:`judge`).
DISAGREES = "disagrees"
#: ... the same, kept as a warning (:func:`speed_warning`): the kernel is correct, so the
#: integration's end-to-end A/B decides on its speed.
SPEED_DISAGREES = "speed_disagrees"
ROUNDS = 3
INPUTS, EXPECTED, CANDIDATE = "inputs.pt", "expected.pt", "candidate.pt"
_dumps = json.dumps  # bound before any candidate is imported
_stdout = sys.stdout


# ------------------------------------------------------------------ fresh inputs


def _plain_tensors(value: Any, depth: int = 0) -> list[Any]:
    """Tensors directly in (nested) tuples, lists and dicts; the contents of other objects
    (mutable state such as KV caches) are not visited."""
    import torch

    if depth > 6:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    items: list[Any] = []
    if isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, tuple | list):
        items = list(value)
    return [t for v in items for t in _plain_tensors(v, depth + 1)]


def fresh_inputs(args: Any, kwargs: Any, gen: Any, kind: str) -> tuple[Any, Any, int]:
    """A copy of one case's inputs with redrawn floating-point tensors (see the module
    docstring); returns ``(args, kwargs, tensors redrawn)``."""
    from kernel_agent.kernels.verify import perturb_

    new_args, new_kwargs = copy.deepcopy((args, kwargs))  # one copy keeps their aliasing
    seen: set[int] = set()
    tensors = []
    for t in _plain_tensors((new_args, new_kwargs)):
        if id(t) not in seen:
            seen.add(id(t))
            tensors.append(t)
    redrawn = perturb_(tensors, gen, kind)
    return new_args, new_kwargs, redrawn


def _flat(**parts: Any) -> dict[str, dict[str, Any]]:
    """``{part: {name: plain CPU tensor}}`` (:func:`kernels.integrity.flat_outputs`)."""
    from kernel_agent.kernels.integrity import flat_outputs

    return flat_outputs([parts])[0]


def _rerounded(
    fn: Any, entry: dict[str, Any], expected: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """The reference ``fn`` once more on an entry's fresh inputs with its rounding redrawn
    (:func:`kernels.verify.rerounded_call`): its outputs and the arguments it computed
    (flattened as ``expected``'s; an argument it left as the plain call did is not kept),
    for :func:`compare_entries` to judge a failed check with the reference's own rounding
    spread (#250). Empty when it cannot run so."""
    import torch

    from kernel_agent.kernels.verify import rerounded_call

    alt = rerounded_call(fn, entry["args"], entry["kwargs"])
    if alt is None:
        return {}
    flat = _flat(output=alt[0], args=alt[1], kwargs=alt[2])
    for key in ("args", "kwargs"):
        plain = expected.get(key) or {}
        flat[key] = {
            name: t
            for name, t in (flat.get(key) or {}).items()
            if not (
                isinstance(t, torch.Tensor)
                and isinstance(plain.get(name), torch.Tensor)
                and t.shape == plain[name].shape
                and t.dtype == plain[name].dtype
                and torch.equal(t, plain[name])
            )
        }
    return flat


# ------------------------------------------------------------------ subprocesses


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        from kernel_agent import toolchain

        toolchain.setup()
        return "cuda"
    return "cpu"


def _load(capture: Path, device: str, sha256: str | None) -> dict[str, Any]:
    from kernel_agent.profiling.capture import load_capture

    data = load_capture(capture, device=device, sha256=sha256)  # inputs-only captures work too
    for case in data["cases"]:
        case.setdefault("method", "forward")
    return data


def _time_cases(
    holders: tuple[Any, ...],
    cases: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    rounds: int,
    replay: Any,
    timing: dict[str, Any] | None = None,
) -> list[Any]:
    """``[median ms, spread]`` per case (None: not timed) on the first seed's inputs, of the
    module ``holders[0]`` (every call from the case's module state: ``replay``), in the
    timing context ``timing`` (``context``, ``l2_flush``: :func:`kernels.bench.time_call`)."""
    from kernel_agent.kernels.bench import median_round, time_call, warm_gpu

    timing = timing or {}

    warm_gpu()
    first = {e["case"]: e for e in reversed(entries)}  # the first seed of every case
    out: list[Any] = []
    for i, case in enumerate(cases):
        if not case.get("count") or i not in first:
            out.append(None)
            continue
        args, kwargs = copy.deepcopy((first[i]["args"], first[i]["kwargs"]))
        fn = replay.call(case, *holders)
        best = median_round(
            [time_call(fn, args, kwargs, target_ms=60.0, **timing) for _ in range(rounds)]
        )
        out.append([best["median_ms"], best["spread"]])
    return out


def reference_main(
    capture: Path,
    workdir: Path,
    *,
    seed: int,
    seeds: int,
    capture_sha256: str | None,
    timing: bool,
    rounds: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fresh inputs, the reference's outputs on them and its timing (no candidate here; in
    the timing ``context``: ``context``, ``l2_flush``)."""
    import torch

    from kernel_agent.profiling.state import Replay
    from kernel_agent.workloads.base import synchronize

    device = _device()
    data = _load(capture, device, capture_sha256)
    reference, cases = data["module"].eval(), data["cases"]
    replay = Replay(data, reference)  # each case's module state (profiling/state.py)
    entries: list[dict[str, Any]] = []
    redrawn = [0] * len(cases)
    for j in range(seeds):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + j)
        for i, case in enumerate(cases):
            kind = "normal" if j == 0 else "mix"
            args, kwargs, n = fresh_inputs(case["args"], case["kwargs"], gen, kind)
            redrawn[i] = n
            entries.append({"case": i, "seed": seed + j, "args": args, "kwargs": kwargs})
    torch.save(entries, workdir / INPUTS)  # before any call can change them
    expected = []
    for entry in entries:
        case = cases[entry["case"]]
        args, kwargs = copy.deepcopy((entry["args"], entry["kwargs"]))
        pre = _flat(args=args, kwargs=kwargs)
        with torch.inference_mode():
            out = replay.call(case, reference)(*args, **kwargs)
        synchronize()
        done = {"pre": pre, **_flat(output=out, args=args, kwargs=kwargs)}
        done["rerounded"] = _rerounded(replay.call(case, reference), entry, done)
        synchronize()
        expected.append(done)
    torch.save({"entries": expected}, workdir / EXPECTED)
    keys = ("method", "signature", "count")
    result: dict[str, Any] = {
        "status": "ok",
        "device": device,
        "redrawn": redrawn,
        "cases": [{k: c.get(k) for k in keys} for c in cases],
        "tier": data.get("tier"),  # the capture's tolerance tier (kernels/compare.py)
    }
    if timing and device == "cuda":
        _timed(result, (reference,), cases, entries, rounds, replay, context)
    return result


def _timed(
    result: dict[str, Any],
    holders: tuple[Any, ...],
    cases: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    rounds: int,
    replay: Any,
    context: dict[str, Any] | None,
) -> None:
    """``ms`` of a subprocess's result (:func:`_time_cases`), or ``timing_error`` when the
    calls cannot be captured in the CUDA graph of a graph-timed verdict."""
    from kernel_agent.kernels.bench import GraphUnavailable

    try:
        result["ms"] = _time_cases(holders, cases, entries, rounds, replay, context)
    except GraphUnavailable as exc:
        result["timing_error"] = f"graph timing unavailable: {exc}"


def candidate_main(
    capture: Path,
    candidate: Path,
    workdir: Path,
    *,
    capture_sha256: str | None,
    timing: bool,
    rounds: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The candidate's outputs on the fresh inputs and its timing (in the timing
    ``context``), under the evaluator's integrity snapshot (taken before the candidate is
    imported)."""
    import torch

    import kernel_agent.kernels.bench  # noqa: F401  (binds the timer before the candidate)
    from kernel_agent.kernels import integrity
    from kernel_agent.kernels.evaluate import load_candidate_module
    from kernel_agent.profiling.state import Replay
    from kernel_agent.workloads.base import synchronize

    flat, save = _flat, torch.save  # bound before the candidate is imported
    device = _device()
    data = _load(capture, device, capture_sha256)
    reference, cases = data["module"].eval(), data["cases"]
    replay = Replay(data, reference)  # each case's module state (profiling/state.py)
    entries = torch.load(workdir / INPUTS, map_location=device, weights_only=False)
    guard = integrity.Snapshot(reference, state=replay.keys)

    def violation(where: str) -> dict[str, Any] | None:
        problems = guard.changes()
        problems += [
            f"thread {t} runs candidate code" for t in integrity.candidate_threads(candidate.parent)
        ]
        if not problems:
            return None
        return {
            "status": "integrity_violation",
            "error": f"{where}: " + "; ".join(problems[:6]),
        }

    try:
        try:
            module = load_candidate_module(candidate)
            given = copy.deepcopy(reference)
            new = module.build(given)
            if new is None:
                raise TypeError("build() returned None")
            new = new.eval() if hasattr(new, "eval") else new
        except Exception:
            return {"status": "build_error", "error": traceback.format_exc()[-3000:]}
        if new is given or new is reference:
            return {"status": "build_error", "error": "build() returned the reference module"}
        methods = {c["method"] for c in cases} - {"forward"}
        if missing := [m for m in methods if not callable(getattr(new, m, None))]:
            return {"status": "build_error", "error": f"no {', '.join(missing)} entrypoint"}
        if (bad := violation("after build()")) is not None:
            return bad

        def calls() -> list[dict[str, Any]] | dict[str, Any]:
            """The candidate on every fresh input (or the runtime error)."""
            outputs = []
            for entry in entries:
                args, kwargs = copy.deepcopy((entry["args"], entry["kwargs"]))
                try:
                    with torch.inference_mode():
                        out = replay.call(cases[entry["case"]], new, given)(*args, **kwargs)
                    synchronize()
                except Exception:
                    return {
                        "status": "runtime_error",
                        "case": entry["case"],
                        "seed": entry["seed"],
                        "error": traceback.format_exc()[-3000:],
                    }
                outputs.append(flat(output=out, args=args, kwargs=kwargs))
            return outputs

        saved: dict[str, Any] = {"entries": calls()}
        if isinstance(saved["entries"], dict):
            return saved["entries"]
        if (bad := violation("after the fresh-input calls")) is not None:
            return bad
        result: dict[str, Any] = {"status": "ok", "device": device}
        if timing and device == "cuda":
            _timed(result, (new, given), cases, entries, rounds, replay, context)
            # the same inputs again: a kernel that changes behaviour after its first calls
            saved["after_timing"] = calls()
            if isinstance(saved["after_timing"], dict):
                return saved["after_timing"]
            if (bad := violation("after timing")) is not None:
                return bad
        save(saved, workdir / CANDIDATE)
        return result
    finally:
        guard.restore()


# ------------------------------------------------------------------ the parent


def _spawn(cmd: list[str], timeout: float, nonce: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            input=nonce + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env(),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {timeout:.0f}s"}
    marker = f"{RESULT_MARKER}{nonce}@@"
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith(marker):
            data: dict[str, Any] = json.loads(line[len(marker) :])
            return data
    tail = (proc.stderr or proc.stdout)[-3000:]
    return {"status": "crash", "returncode": proc.returncode, "error": tail}


def _first_error(failures: list[dict[str, Any]]) -> str:
    first = failures[0] if failures else {}
    detail = first.get("error") or (
        f"{first.get('mismatch_frac')} of elements outside tolerance, max abs err "
        f"{first.get('max_abs_err')}"
    )
    return f"{first.get('name', '?')}: {detail}"


def compare_entries(
    expected: list[dict[str, Any]],
    saved: Any,
    entries: list[tuple[int, int]],
    *,
    tier: str | None = None,
) -> list[list[dict[str, Any]]]:
    """Failed checks per entry (``(case, seed)`` of ``entries``): the candidate's saved
    outputs and post-call state against the reference's (:mod:`kernels.compare`, in the
    capture's tolerance ``tier``, with its bounds for redrawn inputs and the reference's own
    rounding spread on them where the reference subprocess measured it: ``rerounded``)."""
    from kernel_agent.kernels.compare import compare_output, compare_side_effects_flat

    if not isinstance(saved, list) or len(saved) != len(expected):
        bad = {"name": "outputs", "ok": False, "error": "saved outputs do not match the inputs"}
        return [[bad] for _ in entries]
    failures = []
    for exp, new in zip(expected, saved, strict=True):
        new = new if isinstance(new, dict) else {}
        checks = []
        new_out = new.get("output") or {}
        pres = exp.get("pre") or {}
        alt = exp.get("rerounded") or {}
        inputs = [*(pres.get("args") or {}).values(), *(pres.get("kwargs") or {}).values()]
        for name, ref in (exp.get("output") or {}).items():
            if name not in new_out:
                checks.append({"name": name, "ok": False, "error": "missing in candidate output"})
            else:
                checks.append(
                    compare_output(
                        name,
                        ref,
                        new_out[name],
                        inputs,
                        tier=tier,
                        perturbed=True,
                        rerounded=(alt.get("output") or {}).get(name),
                    )
                )
        for key in ("args", "kwargs"):
            pre = (exp.get("pre") or {}).get(key) or {}
            checks += compare_side_effects_flat(
                pre,
                exp.get(key) or {},
                new.get(key) or {},
                tier=tier,
                perturbed=True,
                rerounded_flat=alt.get(key) or {},
            )
        failures.append([c for c in checks if not c.get("ok")])
    return failures


def spread(result: dict[str, Any] | None) -> float | None:
    """The timing spread of a measurement: the largest of its ``timing_spread`` and its
    cases' (an evaluator result or record); None when it reports none."""
    result = result or {}
    found = [result.get("timing_spread")]
    found += [c.get("timing_spread") for c in result.get("cases") or [] if isinstance(c, dict)]
    return max((float(v) for v in found if isinstance(v, int | float)), default=None)


def speedup_tolerance(*spreads: float | None, floor: float = SPEEDUP_TOLERANCE) -> float:
    """The agreement tolerance of speedups measured with these timing spreads: twice the
    larger spread, within [``floor``, :data:`SPEEDUP_TOLERANCE_MAX`]."""
    noise = 2 * max((float(s) for s in spreads if s is not None), default=0.0)
    return min(max(floor, noise), max(floor, SPEEDUP_TOLERANCE_MAX))


def speedups_agree(a: float, b: float, tolerance: float) -> bool:
    """Whether speedups ``a`` and ``b`` are within a factor ``1 + tolerance`` of each other:
    ``|log a - log b| <= log(1 + tolerance)``, the same band whichever is larger."""
    return abs(math.log(float(a)) - math.log(float(b))) <= math.log1p(tolerance)


def judge(
    result: dict[str, Any], verdict: dict[str, Any] | None, tolerance: float = SPEEDUP_TOLERANCE
) -> dict[str, Any]:
    """Agreement with the evaluator's ``verdict`` (``correct``, ``speedup``, its timing
    spread: :func:`spread`), the re-check's own ``passed`` and, when it failed, ``reason``.

    The speedups must agree (:func:`speedups_agree` within :func:`speedup_tolerance`,
    ``tolerance`` its floor), whichever is larger. A result judged before is judged again
    from its measurement (against another verdict: a re-evaluation)."""
    verdict = verdict or {}
    if result.get("status") in (DISAGREES, SPEED_DISAGREES):  # judged before: measured fine
        result["status"] = "ok"
        result.pop("reason", None)
    for key in ("speedup_ratio", "tolerance", "warning", "conservative_speedup"):
        result.pop(key, None)
    claimed, measured = verdict.get("speedup"), result.get("speedup")
    result["evaluator"] = {"correct": verdict.get("correct"), "speedup": claimed}
    if "status" in verdict:
        result["evaluator"]["status"] = verdict["status"]
    agrees: dict[str, Any] = {}
    if verdict.get("correct") is not None and result.get("correct") is not None:
        agrees["correct"] = bool(verdict["correct"]) == bool(result["correct"])
    if isinstance(claimed, int | float) and claimed > 0 and measured:
        tol = speedup_tolerance(result.get("timing_spread"), spread(verdict), floor=tolerance)
        result["speedup_ratio"] = round(float(measured) / float(claimed), 3)
        result["tolerance"] = round(tol, 3)
        agrees["speedup"] = speedups_agree(measured, claimed, tol)
    result["agrees"] = agrees
    if result.get("status") == "ok" and agrees.get("speedup") is False:
        tol = result["tolerance"]
        result["status"] = DISAGREES
        result["reason"] = (
            f"{measured}x in separate processes vs {claimed}x claimed by the evaluator "
            f"(ratio {result['speedup_ratio']}, outside {1 / (1 + tol):.2f}-{1 + tol:.2f})"
        )
    result["passed"] = result.get("status") == "ok"
    return result


def disagreement(result: dict[str, Any]) -> float:
    """How far a judged result's speedup is from the evaluator's: ``|log ratio|`` (0: no
    timed comparison)."""
    ratio = result.get("speedup_ratio")
    return abs(math.log(ratio)) if isinstance(ratio, int | float) and ratio > 0 else 0.0


def speed_warning(result: dict[str, Any]) -> dict[str, Any]:
    """A correct result whose speedup disagrees with the evaluator's (:data:`DISAGREES`)
    as a warning, not a failure: :data:`SPEED_DISAGREES`, ``passed``, the disagreement in
    ``warning`` and the smaller of the two speedups as ``conservative_speedup``."""
    claimed = float((result.get("evaluator") or {}).get("speedup") or 0.0)
    measured = float(result.get("speedup") or 0.0)
    result["warning"] = result.pop("reason", None)
    result.update(status=SPEED_DISAGREES, passed=True)
    result["conservative_speedup"] = round(min(claimed, measured), 3)
    return result


def run_recheck(
    capture: Path,
    candidate: Path,
    *,
    seeds: int = 3,
    seed: int | None = None,
    verdict: dict[str, Any] | None = None,
    capture_sha256: str | None = None,
    timeout: float = 600.0,
    timing: bool = True,
    rounds: int = ROUNDS,
    tolerance: float = SPEEDUP_TOLERANCE,
) -> dict[str, Any]:
    """Re-check ``candidate`` on ``capture`` (see the module docstring) under the GPU lock.

    ``verdict``: the evaluator's result (or record) to compare with (timed in its
    ``context`` and ``l2``), ``seed``: the first input seed (default: random),
    ``capture_sha256``: refuse a capture without this digest (both subprocesses check the
    bytes they load)."""
    import torch

    capture, candidate = Path(capture).resolve(), Path(candidate).resolve()
    base = secrets.randbits(31) if seed is None else int(seed)
    result: dict[str, Any] = {
        "status": "error",
        "passed": False,
        "correct": None,
        "speedup": None,
        "seed": base,
        "seeds": seeds,
    }
    start = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix="ka-recheck-"))
    common = ["--workdir", str(workdir), "--rounds", str(rounds)]
    common += ["--capture-sha256", capture_sha256] if capture_sha256 else []
    common += [] if timing else ["--no-timing"]
    context = str((verdict or {}).get("context") or "eager")  # the verdict's (#226)
    common += ["--context", context] if context != "eager" else []
    common += ["--l2-flush"] if (verdict or {}).get("l2") == "cold" else []
    module = [sys.executable, "-m", "kernel_agent.kernels.recheck"]
    try:
        with gpu_lock() as gpu:  # both processes on this GPU, one after the other
            result["gpu_index"] = gpu
            seeding = ["--seed", str(base), "--seeds", str(seeds)]
            ref = _spawn(
                [*module, "reference", str(capture), *seeding, *common],
                timeout,
                secrets.token_hex(16),
            )
            if ref.get("status") != "ok":
                why = str(ref.get("error") or ref.get("status"))
                result["reason"] = f"the reference process failed ({ref.get('status')}): {why}"
                return judge(result, verdict, tolerance)
            expected = torch.load(workdir / EXPECTED, map_location="cpu", weights_only=True)
            (workdir / EXPECTED).unlink()  # the candidate's process never sees the answer
            cand = _spawn(
                [*module, "candidate", str(capture), str(candidate), *common],
                timeout,
                secrets.token_hex(16),
            )
        cases = ref.get("cases") or []  # from the candidate-free process
        entries = [(i, base + j) for j in range(seeds) for i in range(len(cases))]
        if cand.get("status") != "ok":
            result["status"] = str(cand.get("status"))
            result["reason"] = f"{cand.get('status')}: {str(cand.get('error') or '')[-1500:]}"
            if "case" in cand:
                result["failed_case"] = {"case": cand["case"], "seed": cand.get("seed")}
            return judge(result, verdict, tolerance)
        try:
            got = torch.load(workdir / CANDIDATE, map_location="cpu", weights_only=True)
        except Exception as exc:
            got = None
            result["reason"] = f"the candidate's outputs are unreadable: {exc}"[:500]
        got = got if isinstance(got, dict) else {}
        tier = ref.get("tier")
        failures = compare_entries(expected["entries"], got.get("entries"), entries, tier=tier)
        if cand.get("ms"):  # timed: the same inputs once more after timing
            after = compare_entries(
                expected["entries"], got.get("after_timing"), entries, tier=tier
            )
            failures = [
                first + [{**f, "after_timing": True} for f in again]
                for first, again in zip(failures, after, strict=True)
            ]
        _summarise(result, cases, entries, failures, ref, cand)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        result["seconds"] = round(time.perf_counter() - start, 1)
    return judge(result, verdict, tolerance)


def _summarise(
    result: dict[str, Any],
    cases: list[dict[str, Any]],
    entries: list[tuple[int, int]],
    failures: list[list[dict[str, Any]]],
    ref: dict[str, Any],
    cand: dict[str, Any],
) -> None:
    """Per-case correctness and timing, the weighted speedup and the first failure."""
    reports = []
    ref_ms, new_ms = ref.get("ms") or [], cand.get("ms") or []
    total_ref = total_new = spread = 0.0
    for i, case in enumerate(cases):
        bad = [
            {"seed": seed, **f}
            for (c, seed), found in zip(entries, failures, strict=True)
            if c == i
            for f in found
        ]
        report: dict[str, Any] = {
            "case": i,
            "method": case.get("method") or "forward",
            "signature": case.get("signature"),
            "calls_per_run": case.get("count"),
            "redrawn": (ref.get("redrawn") or [None] * len(cases))[i],
            "ok": not bad,
            "failures": bad[:5],
        }
        r = ref_ms[i] if i < len(ref_ms) else None
        n = new_ms[i] if i < len(new_ms) else None
        if r and n:
            report.update(
                ref_ms=round(r[0], 5),
                new_ms=round(n[0], 5),
                speedup=round(r[0] / max(n[0], 1e-9), 3),
            )
            count = float(case.get("count") or 0)
            total_ref += count * r[0]
            total_new += count * n[0]
            spread = max(spread, r[1], n[1])
        reports.append(report)
    result["cases"] = reports
    result["correct"] = all(r["ok"] for r in reports)
    result["status"] = "ok" if result["correct"] else "incorrect"
    if why := ref.get("timing_error") or cand.get("timing_error"):
        result["timing"] = f"skipped: {why}"  # no speedup to compare with the verdict's
    elif total_ref and total_new:
        result["speedup"] = round(total_ref / total_new, 3)
        result["timing_spread"] = round(spread, 3)
    elif ref.get("device") != "cuda":
        result["timing"] = "skipped: no CUDA device"
    else:
        result["timing"] = "skipped" if not ref.get("ms") else "no timed case"
    if not result["correct"]:
        first = next(r for r in reports if not r["ok"])
        fail = first["failures"][0]
        when = " (the same input again, after timing)" if fail.get("after_timing") else ""
        result["reason"] = (
            f"case {first['case']} ({first['signature']}), seed {fail['seed']}{when}: "
            f"{_first_error([fail])}: wrong on fresh inputs of the captured shapes (it "
            "passed the evaluator on the captured ones)"
        )


def describe(result: dict[str, Any]) -> str:
    """One line on a re-check: its verdict against the evaluator's, after the
    re-evaluation of a stale or disagreeing record when there was one (``reevaluated``)."""
    status = result.get("status")
    if status == "skipped":
        return f"skipped: {result.get('reason')}"
    prefix = ""
    if again := result.get("reevaluated"):
        new_speed, new_status = again.get("new_speedup"), again.get("new_status")
        now = f"{new_speed}x" if again.get("new_correct") else new_status
        prefix = (
            f"re-evaluated ({again.get('why')}): {again.get('old_speedup')}x recorded, "
            f"{now} by the current evaluator; "
        )
    seeds = result.get("seeds")
    cases = len(result.get("cases") or [])
    parts = []
    if result.get("correct"):
        parts.append(f"correct on {seeds} fresh draws of {cases} case(s)")
    speed = result.get("speedup")
    claimed = (result.get("evaluator") or {}).get("speedup")
    if speed:
        parts.append(
            f"{speed}x in separate processes"
            + (f" vs {claimed}x in the evaluator" if claimed else "")
        )
    text = "; ".join(parts) or str(status)
    if status == SPEED_DISAGREES:
        runs = [f"{r.get('speedup')}x" for r in result.get("rechecks") or []]
        text += (
            f"; WARNING: the speedups disagree (ratio {result.get('speedup_ratio')}"
            + (f"; re-checks: {', '.join(runs)}" if len(runs) > 1 else "")
            + f"): not refused, ranked by the conservative {result.get('conservative_speedup')}x,"
            " the end-to-end A/B decides"
        )
    if (check := result.get("memcheck")) and result.get("passed"):  # kernels/memcheck.py
        from kernel_agent.kernels.memcheck import describe as memcheck_line

        text += f"; {memcheck_line(check)}"
    if not result.get("passed"):
        text = f"FAILED ({status}): {result.get('reason') or ''}".strip()
    return prefix + text


# ------------------------------------------------------------------ entry point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("role", choices=["reference", "candidate"])
    parser.add_argument("capture", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--capture-sha256")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--context", choices=("eager", "graph"), default="eager")
    parser.add_argument("--l2-flush", action="store_true")
    ns = parser.parse_args(argv)
    tag = sys.stdin.readline().strip()  # read before a candidate is imported
    context = {"context": ns.context, "l2_flush": ns.l2_flush}
    try:
        if ns.role == "reference":
            result = reference_main(
                ns.capture,
                ns.workdir,
                seed=ns.seed,
                seeds=ns.seeds,
                capture_sha256=ns.capture_sha256,
                timing=not ns.no_timing,
                rounds=ns.rounds,
                context=context,
            )
        else:
            if ns.candidate is None:
                parser.error("the candidate role needs a candidate file")
            result = candidate_main(
                ns.capture,
                ns.candidate,
                ns.workdir,
                capture_sha256=ns.capture_sha256,
                timing=not ns.no_timing,
                rounds=ns.rounds,
                context=context,
            )
    except Exception:
        result = {"status": "harness_error", "error": traceback.format_exc()[-3000:]}
    _stdout.write(f"{RESULT_MARKER}{tag}@@" + _dumps(result, default=str) + "\n")
    _stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # no interpreter teardown: a candidate's atexit hooks cannot print a result


if __name__ == "__main__":
    raise SystemExit(main())
