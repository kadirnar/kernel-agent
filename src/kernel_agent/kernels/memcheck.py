"""Out-of-bounds accesses of a kernel under ``compute-sanitizer --tool memcheck`` (issue #115).

A kernel can read past a buffer and still pass every check: the extra elements are masked
out of its result, or change it too little. PyTorch's caching allocator keeps the memory
after a tensor mapped, so nothing faults until that memory is released (``empty_cache()``,
another state of an A/B) and the read lands on an unmapped page: an illegal memory
access, far from the kernel that caused it.

:func:`run_memcheck` runs a kernel candidate once on every captured case under memcheck, in
a process of its own with ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` (every tensor is its own
``cudaMalloc``, so the sanitizer knows its exact bounds), and on one *odd-size variant* of
each case (:func:`odd_variants`): the sizes that differ between the captured cases (batch,
sequence length, ...) one smaller, which leaves a partial tile at the end of every tiled
loop. A variant the reference itself cannot run is dropped; an exception of the candidate
on a variant is not an error (it may refuse sizes it was not written for), a memory error
is. The captured shapes alone can be exact tile multiples: the VAE decoder kernel of
issue #115 overruns only on a partial tile, which the captured shapes never have.

``status`` of a result: ``ok``, :data:`STATUS` (memory errors: their count in ``errors``,
the sanitizer's first report in ``report``), ``skipped`` (no usable ``compute-sanitizer``:
:func:`kernel_agent.toolchain.find_sanitizer`; no CUDA device), ``error`` (the sanitizer or
the process failed without a memory error) or ``timeout``; ``seconds`` is its wall time.

:func:`selftest` proves that the sanitizer works here (``kernel-agent doctor``): a deliberate
one-block overrun of a Triton kernel (:mod:`kernel_agent.kernels.memcheck_probe`) must be
reported, an in-bounds kernel must not.

Subprocesses (under the sanitizer)::

    python -m kernel_agent.kernels.memcheck CAPTURE CANDIDATE [--capture-sha256 S]
    python -m kernel_agent.kernels.memcheck_probe
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from kernel_agent import interrupt
from kernel_agent.gpulock import child_env, gpu_lock

#: ``status`` of a kernel with memory errors under memcheck; the integration refuses it.
STATUS = "memcheck"
RESULT_MARKER = "@@KA_MEMCHECK@@"
#: The sanitizer's options: memory errors only (no CUDA API errors, which libraries make on
#: purpose, nor the exit code); a faulting kernel is stopped, not the CUDA context (with no
#: caching allocator, the first ``cudaFree`` in a dead context aborts the process: no
#: result line); device backtraces (the host's are interpreter frames); the Python process
#: alone (not the ptxas a Triton compile starts).
SANITIZER_ARGS = [
    "--tool",
    "memcheck",
    "--report-api-errors",
    "no",
    "--check-exit-code",
    "no",
    "--destroy-on-device-error",
    "kernel",
    "--show-backtrace",
    "device",
    "--print-limit",
    "10",
    "--target-processes",
    "application-only",
]
#: Every tensor its own ``cudaMalloc``: the caching allocator would hide overruns.
ENV = {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
PREFIX = "========="
#: Lines of the sanitizer's log that are not an error report.
_META = (
    "COMPUTE-SANITIZER",
    "ERROR SUMMARY",
    "LEAK SUMMARY",
    "Error: process didn't terminate successfully",
    "Target application returned an error",
    "Error: couldn't find exit code",
    "Error: Target application terminated before first instrumented API call",
)
REPORT_LINES = 25
_dumps = json.dumps  # bound before any candidate is imported
_stdout = sys.stdout


# ------------------------------------------------------------------ odd-size variants


def _rebuild(value: Any, new: dict[int, Any], depth: int = 0) -> Any:
    """``value`` with the tensors of ``new`` (by id) swapped in, in (nested) tuples, lists
    and dicts; other objects are kept as they are."""
    if id(value) in new:
        return new[id(value)]
    if depth > 6:
        return value
    if isinstance(value, dict):
        return type(value)((k, _rebuild(v, new, depth + 1)) for k, v in value.items())
    if isinstance(value, list):
        return [_rebuild(v, new, depth + 1) for v in value]
    if isinstance(value, tuple):
        items = [_rebuild(v, new, depth + 1) for v in value]
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    return value


def odd_variants(cases: list[dict[str, Any]]) -> list[tuple[int, list[list[int]], Any, Any]]:
    """``(case, shapes, args, kwargs)``: one variant of each captured case whose dynamic sizes
    are one smaller. A size is dynamic when it differs between the cases of one entrypoint
    with the same tensor ranks (the batch or sequence length of a workload); cases alone of
    their kind have none. The variant's changed tensors are fresh copies of the leading part
    of the captured ones (same dtype, device and dimension order); ``shapes``: of its plain
    tensors."""
    import torch

    from kernel_agent.kernels.recheck import _plain_tensors

    groups: dict[tuple[Any, ...], list[int]] = {}
    tensors = [_plain_tensors((c["args"], c["kwargs"])) for c in cases]
    for i, case in enumerate(cases):
        key = (case.get("method") or "forward", tuple(t.dim() for t in tensors[i]))
        groups.setdefault(key, []).append(i)
    out: list[tuple[int, list[list[int]], Any, Any]] = []
    for (_, ranks), members in groups.items():
        dynamic = {
            (p, d)
            for p, rank in enumerate(ranks)
            for d in range(rank)
            if len({tensors[i][p].shape[d] for i in members}) > 1
        }
        for i in members:
            new: dict[int, Any] = {}
            for p, t in enumerate(tensors[i]):
                dims = [d for d in range(t.dim()) if (p, d) in dynamic and t.shape[d] > 1]
                if not dims or id(t) in new:
                    continue
                part = t
                for d in dims:
                    part = part.narrow(d, 0, t.shape[d] - 1)
                new[id(t)] = torch.empty_like(part, memory_format=torch.preserve_format)
                new[id(t)].copy_(part)
            if new:
                args, kwargs = _rebuild((cases[i]["args"], cases[i]["kwargs"]), new)
                shapes = [list(t.shape) for t in _plain_tensors((args, kwargs))]
                out.append((i, shapes, args, kwargs))
    return out


# ------------------------------------------------------------------ the checked process


def cases_main(
    capture: Path, candidate: Path, *, capture_sha256: str | None, variants: bool = True
) -> dict[str, Any]:
    """The candidate once on every captured case and on the odd-size variants the reference
    runs (run by the reference before the candidate is imported), each call from its
    case's module state (:mod:`kernel_agent.profiling.state`)."""
    import torch

    from kernel_agent.kernels.evaluate import load_candidate_module
    from kernel_agent.kernels.recheck import _device, _load
    from kernel_agent.profiling.state import Replay
    from kernel_agent.workloads.base import synchronize

    if _device() != "cuda":
        return {"status": "skipped", "reason": "no CUDA device"}
    data = _load(capture, "cuda", capture_sha256)
    reference, cases = data["module"].eval(), data["cases"]
    replay = Replay(data, reference)  # each case's module state (profiling/state.py)
    given = copy.deepcopy(reference)  # what build() gets: untouched by the variants' calls

    def call(holders: tuple[Any, ...], i: int, args: Any, kwargs: Any) -> None:
        args, kwargs = copy.deepcopy((args, kwargs))
        with torch.inference_mode():
            replay.call(cases[i], *holders)(*args, **kwargs)
        synchronize()

    runs: list[tuple[int, list[list[int]] | None, Any, Any]] = [
        (i, None, c["args"], c["kwargs"]) for i, c in enumerate(cases)
    ]
    dropped = 0
    for i, shapes, args, kwargs in odd_variants(cases) if variants else []:
        try:
            call((reference,), i, args, kwargs)
        except Exception:  # not an input the module takes
            dropped += 1
            continue
        runs.append((i, shapes, args, kwargs))
    del reference
    try:
        new = load_candidate_module(candidate).build(given)
        if new is None:
            raise TypeError("build() returned None")
        new = new.eval() if hasattr(new, "eval") else new
    except Exception:
        return {"status": "build_error", "error": traceback.format_exc()[-3000:]}
    refused: list[dict[str, Any]] = []
    for i, sizes, args, kwargs in runs:
        try:
            call((new, given), i, args, kwargs)
        except Exception:
            if sizes is None:
                tb = traceback.format_exc()[-3000:]
                return {"status": "runtime_error", "case": i, "error": tb}
            refused.append({"case": i, "shapes": sizes, "error": traceback.format_exc()[-300:]})
    return {
        "status": "ok",
        "cases": len(cases),
        "variants": [{"case": i, "shapes": s} for i, s, _, _ in runs if s is not None],
        "variants_dropped": dropped,  # the reference cannot run them
        "variants_refused": refused,  # the candidate raised (not a memory error)
    }


# ------------------------------------------------------------------ the parent


def parse_log(text: str) -> tuple[int, str]:
    """``(errors, the first error report)`` of a sanitizer log (its default prefix)."""
    counts = re.findall(r"ERROR SUMMARY: (\d+) error", text)
    errors = int(counts[0]) if counts else 0
    report: list[str] = []
    for line in text.splitlines():
        if not line.startswith(PREFIX):
            if report:
                break
            continue
        rest = line[len(PREFIX) :].rstrip()
        if report:
            if not rest.startswith("  "):  # a blank line or the next report
                break
            report.append(rest.strip())
        elif rest.strip() and not rest.startswith("  ") and not rest.strip().startswith(_META):
            report.append(rest.strip())
    if report and not errors:
        errors = 1
    lines = report[:REPORT_LINES] + (["..."] if len(report) > REPORT_LINES else [])
    return errors, "\n".join(lines)


def _first_lines(report: str, n: int = 2) -> str:
    return " ".join(line for line in report.splitlines()[:n])


def _sanitize(
    cmd: list[str], timeout: float, workdir: Path, tool: Any
) -> tuple[dict[str, Any] | None, str, str]:
    """Run ``cmd`` (a Python command line) under the sanitizer: ``(the result line of the
    process or None, the sanitizer's log, the process's output tail)``; raises
    ``subprocess.TimeoutExpired`` after killing the sanitizer and what it launched (the
    process runs under its launcher, so it does not die with this one)."""
    nonce = secrets.token_hex(16)
    log_path = workdir / "memcheck.log"
    full = [tool.path, *SANITIZER_ARGS, "--log-file", str(log_path), *cmd]
    with subprocess.Popen(
        full,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**child_env(), **ENV},
    ) as proc:
        try:
            stdout, stderr = proc.communicate(nonce + "\n", timeout=timeout)
        except subprocess.TimeoutExpired:
            tree = interrupt.descendants(proc.pid)
            proc.kill()
            interrupt.terminate(tree, grace=2.0)
            proc.communicate()
            raise
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    marker = f"{RESULT_MARKER}{nonce}@@"
    child = None
    for line in stdout.splitlines()[::-1]:
        if line.startswith(marker):
            child = json.loads(line[len(marker) :])
            break
    tail = (stderr + stdout)[-2000:]
    return child, log, tail


def run_memcheck(
    capture: Path,
    candidate: Path,
    *,
    capture_sha256: str | None = None,
    timeout: float = 900.0,
    variants: bool = True,
    tool: Any = None,
) -> dict[str, Any]:
    """``candidate`` on the cases of ``capture`` (and their odd-size variants) under
    ``compute-sanitizer --tool memcheck`` (``tool``: a :class:`toolchain.Sanitizer`, default
    the one :func:`toolchain.sanitizer` finds), under the GPU lock. See the module
    docstring for the result."""
    from kernel_agent import toolchain

    tool = tool or toolchain.sanitizer()
    if not tool.path:
        return {"status": "skipped", "reason": tool.reason, "seconds": 0.0}
    result: dict[str, Any] = {"status": "error", "sanitizer": tool.version}
    start = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix="ka-memcheck-"))
    cmd = [sys.executable, "-m", "kernel_agent.kernels.memcheck"]
    cmd += [str(Path(capture).resolve()), str(Path(candidate).resolve())]
    cmd += ["--capture-sha256", capture_sha256] if capture_sha256 else []
    cmd += [] if variants else ["--no-variants"]
    try:
        with gpu_lock():
            child, log, tail = _sanitize(cmd, timeout, workdir, tool)
    except subprocess.TimeoutExpired:
        result.update(status="timeout", reason=f"exceeded {timeout:.0f}s under the sanitizer")
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        result["seconds"] = round(time.perf_counter() - start, 1)
    errors, report = parse_log(log)
    for key in ("cases", "variants", "variants_dropped", "variants_refused"):
        if child and key in child:
            result[key] = child[key]
    if errors:
        result.update(status=STATUS, errors=errors, report=report)
        result["reason"] = (
            f"{errors} memory error(s) under compute-sanitizer memcheck, the first: "
            f"{_first_lines(report)}"
        )
    elif child is None:
        why = (log.strip().splitlines() or [""])[-1] if log.strip() else tail.strip()
        result["reason"] = f"the checked process gave no result: {why[-1500:]}"
    elif child.get("status") in ("ok", "skipped"):
        result["status"] = child["status"]
        if child.get("reason"):
            result["reason"] = child["reason"]
    else:
        result["reason"] = f"{child.get('status')}: {str(child.get('error') or '')[-1500:]}"
    return result


def describe(result: dict[str, Any]) -> str:
    """One line on a memcheck result."""
    status, seconds = result.get("status"), result.get("seconds")
    took = f" ({seconds} s)" if seconds else ""
    if status == "ok":
        n = len(result.get("variants") or [])
        on = f"{result.get('cases')} case(s)" + (f" + {n} odd-size variant(s)" if n else "")
        return f"memcheck clean on {on}{took}"
    if status == STATUS:
        return f"memcheck FAILED{took}: {result.get('reason')}"
    return f"memcheck {status}{took}: {result.get('reason')}"


def refuse(result: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    """A passed re-check ``result`` refused for the memory errors of ``check``."""
    result.update(status=STATUS, passed=False, memcheck=check)
    result.pop("warning", None)
    result["reason"] = (
        f"{check.get('reason')} (it passed the evaluator and the re-check: the overrun is "
        "masked out of its outputs, but faults once the memory after the buffer is released)"
    )
    return result


def pending(result: dict[str, Any]) -> bool:
    """Whether a passed re-check record still needs its memcheck: none yet, or one that
    did not decide (skipped, error, timeout)."""
    done = (result.get("memcheck") or {}).get("status") in ("ok", STATUS)
    return bool(result.get("passed")) and not done


# ------------------------------------------------------------------ the self-test


def selftest(tool: Any = None, timeout: float = 300.0) -> dict[str, Any]:
    """``ok`` when the sanitizer reports the deliberate overrun of
    :mod:`kernel_agent.kernels.memcheck_probe` and nothing else (``kernel-agent doctor``)."""
    from kernel_agent import toolchain

    tool = tool or toolchain.sanitizer()
    if not tool.path:
        return {"ok": False, "reason": tool.reason}
    start = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix="ka-memcheck-"))
    cmd = [sys.executable, "-m", "kernel_agent.kernels.memcheck_probe"]
    try:
        with gpu_lock():
            child, log, tail = _sanitize(cmd, timeout, workdir, tool)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"exceeded {timeout:.0f}s"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    errors, report = parse_log(log)
    out: dict[str, Any] = {"seconds": round(time.perf_counter() - start, 1), "errors": errors}
    out["report"] = report
    if child is None or child.get("status") != "ok":
        detail = (child or {}).get("error") or log.strip()[-500:] or tail.strip()[-500:]
        out.update(ok=False, reason=f"the probe did not run: {detail}")
    elif "_ka_memcheck_inbounds" in log:
        out.update(ok=False, reason=f"an in-bounds kernel was reported: {_first_lines(report)}")
    elif not errors or "_ka_memcheck_overrun" not in report:
        out.update(ok=False, reason="the deliberate out-of-bounds read was not reported")
    else:
        out.update(ok=True, reason=_first_lines(report))
    return out


# ------------------------------------------------------------------ entry point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--capture-sha256")
    parser.add_argument("--no-variants", action="store_true")
    ns = parser.parse_args(argv)
    tag = sys.stdin.readline().strip()  # read before the candidate is imported
    try:
        result = cases_main(
            ns.capture,
            ns.candidate,
            capture_sha256=ns.capture_sha256,
            variants=not ns.no_variants,
        )
    except Exception:
        result = {"status": "harness_error", "error": traceback.format_exc()[-3000:]}
    _stdout.write(f"{RESULT_MARKER}{tag}@@" + _dumps(result, default=str) + "\n")
    _stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # no interpreter teardown: a candidate's atexit hooks cannot print a result


if __name__ == "__main__":
    raise SystemExit(main())
