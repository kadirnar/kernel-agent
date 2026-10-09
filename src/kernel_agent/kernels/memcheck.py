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
``unchecked`` (:data:`UNCHECKED`, with ``ok``): the checked process made no CUDA call the
sanitizer saw (a CPU-only run; CUDA work in a child process, which ``--target-processes
application-only`` does not track), so nothing was checked: the log's notice about child
processes is no error (:func:`unchecked`).

**racecheck and synccheck** (issue #225). A candidate that synchronises inside a kernel
(:func:`extra_tools`: every native project; a source with shared memory, ``cp.async`` or
mbarrier pipelines, acquire / release counters or atomics) also runs once on its captured
cases under ``--tool racecheck`` (hazards between a block's threads on shared memory: a read
before the barrier that orders it, a page reused while still read) and ``--tool synccheck``
(a barrier some threads of a block or warp never reach). Either one's errors refuse it like a
memory error (status ``racecheck`` / ``synccheck``, :data:`FAILED`); ``tools`` in the
result has every tool's own record (status, errors, first report, seconds). Racecheck sees
shared memory only: ordering bugs of global memory are what the evaluator's determinism and
perturbed checks are for.

:func:`selftest` proves that each tool works here (``kernel-agent doctor``, :data:`SELFTESTS`):
memcheck must report a deliberate one-block overrun of a Triton kernel
(:mod:`kernel_agent.kernels.memcheck_probe`) and not an in-bounds kernel; racecheck a
shared-memory read-after-write between two warps without the barrier, synccheck a
``__syncthreads()`` half of a warp reaches (:mod:`kernel_agent.kernels.sanitizer_probe`,
NVRTC), and neither the same kernels with their barrier in place.

Subprocesses (under the sanitizer)::

    python -m kernel_agent.kernels.memcheck CAPTURE CANDIDATE [--capture-sha256 S]
    python -m kernel_agent.kernels.memcheck_probe
    python -m kernel_agent.kernels.sanitizer_probe race|sync
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
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
#: The sanitizer tools for kernels that synchronise inside (:func:`extra_tools`), and the
#: statuses the integration refuses (each tool's name: its errors).
EXTRA_TOOLS = ("racecheck", "synccheck")
FAILED = (STATUS, *EXTRA_TOOLS)
#: What makes a single-file candidate synchronise inside its kernels (native projects always
#: run every tool): shared-memory pipelines, mbarriers, counters and atomics, the megakernel kit.
_SYNC_SOURCE = re.compile(
    r"__shared__|mbarrier|cp\.async|ld\.acquire|red\.release|atomicAdd|atomicCAS|atomicExch"
    r"|tl\.atomic_|ka_mk"
)
RESULT_MARKER = "@@KA_MEMCHECK@@"
#: The sanitizer's options: memory errors only (no CUDA API errors, which libraries make on
#: purpose, nor the exit code); a faulting kernel is stopped, not the CUDA context (with no
#: caching allocator, the first ``cudaFree`` in a dead context aborts the process: no
#: result line); device backtraces (the host's are interpreter frames); the Python process
#: alone (not the ptxas a Triton compile starts).
_COMMON_ARGS = [
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
SANITIZER_ARGS = ["--tool", "memcheck", *_COMMON_ARGS]
#: Each tool's options: racecheck reports the hazards it can prove (errors) and the
#: likely ones (warnings) per access pair, synccheck every divergent or misused barrier.
TOOL_ARGS = {
    "memcheck": SANITIZER_ARGS,
    "racecheck": ["--tool", "racecheck", "--racecheck-report", "analysis", *_COMMON_ARGS],
    "synccheck": ["--tool", "synccheck", *_COMMON_ARGS],
}
#: Every tensor its own ``cudaMalloc``: the caching allocator would hide overruns. A
#: megakernel's counter waits last ~100x longer under a sanitizer: its watchdog
#: (``kernel_agent.native.megakernel.runtime``) gets 10 minutes, not 2 s.
ENV = {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1", "KA_MK_WATCHDOG_MS": "600000"}
PREFIX = "========="
#: Lines of the sanitizer's log that are not an error report.
_META = (
    "COMPUTE-SANITIZER",
    "ERROR SUMMARY",
    "RACECHECK SUMMARY",
    "LEAK SUMMARY",
    "Error: process didn't terminate successfully",
    "Target application returned an error",
    "Error: couldn't find exit code",
    "Error: Target application terminated before first instrumented API call",
    # compute-sanitizer 2025.2 adds this notice after the line above: a hint, no error
    "Tracking kernels launched by child processes requires",
)
#: What the log of a process that made no CUDA call under the sanitizer says (measured with
#: compute-sanitizer 2025.2.1 on an NVIDIA A10: a CPU-only run, or CUDA work in a child
#: process only; with CUDA in the process itself the log is a plain ERROR SUMMARY).
_NO_CUDA_CALL = (
    "terminated before first instrumented API call",
    "Tracking kernels launched by child processes",
)
#: ``unchecked`` of such a run: it passes, and says so instead of claiming a clean kernel.
UNCHECKED = (
    "the checked process made no CUDA call the sanitizer saw, so nothing was checked: "
    "kernels launched from a child process are not tracked (--target-processes "
    "application-only), and a CPU-only run launches none"
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


def unchecked(text: str) -> bool:
    """Whether a sanitizer log says its process made no CUDA call it saw (:data:`UNCHECKED`):
    no error, and nothing checked either."""
    return any(sign in text for sign in _NO_CUDA_CALL)


def parse_race_log(text: str) -> tuple[int, int, str]:
    """``(errors, warnings, the first hazard report)`` of a racecheck log: its ``RACECHECK
    SUMMARY: N hazards displayed (E errors, W warnings)``, else one error per report."""
    _, report = parse_log(text)
    summary = re.findall(
        r"RACECHECK SUMMARY: \d+ hazards? displayed \((\d+) errors?, (\d+) warn", text
    )
    if summary:
        return int(summary[-1][0]), int(summary[-1][1]), report
    return (1 if report else 0), 0, report


def extra_tools(candidate: Path) -> tuple[tuple[str, ...], str]:
    """The tools beyond memcheck that ``candidate`` gets (:data:`EXTRA_TOOLS` or none) and
    why: a native project (a directory or its bundle) always, a single file when its source
    synchronises inside its kernels (:data:`_SYNC_SOURCE`)."""
    from kernel_agent.native import project as native_project

    candidate = Path(candidate)
    if candidate.is_dir() or native_project.read_bundle(candidate) is not None:
        return EXTRA_TOOLS, "a native project"
    try:
        found = _SYNC_SOURCE.search(candidate.read_text(errors="replace"))
    except OSError:
        return (), ""
    return (EXTRA_TOOLS, f"its source uses {found.group(0)}") if found else ((), "")


def _first_lines(report: str, n: int = 2) -> str:
    return " ".join(line for line in report.splitlines()[:n])


def _sanitize(
    cmd: list[str], timeout: float, workdir: Path, tool: Any, name: str = "memcheck"
) -> tuple[dict[str, Any] | None, str, str]:
    """Run ``cmd`` (a Python command line) under the sanitizer: ``(the result line of the
    process or None, the sanitizer's log, the process's output tail)``; raises
    ``subprocess.TimeoutExpired`` after killing the sanitizer and what it launched (the
    process runs under its launcher, so it does not die with this one)."""
    nonce = secrets.token_hex(16)
    log_path = workdir / f"{name}.log"
    full = [tool.path, *TOOL_ARGS[name], "--log-file", str(log_path), *cmd]
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
    from kernel_agent import emulate

    if (emulated := emulate.record()) is not None:
        result["emulated"] = emulated  # the child checks an older GPU's code paths (#252)
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
    if not errors and unchecked(log):
        result["unchecked"] = UNCHECKED
    tools, why = extra_tools(Path(candidate))
    if tools:
        result["tools"] = {"memcheck": _tool_record(result)}
        first = result["status"]  # memcheck's verdict: the others run only after a clean one
        for name in tools:
            if first != "ok":
                result["tools"][name] = {"status": "skipped", "reason": f"memcheck {first}"}
                continue
            check = _run_tool(name, [*cmd, "--no-variants"], timeout, tool)
            check["why"] = why
            result["tools"][name] = check
            if check["status"] == name and result["status"] == "ok":  # refused: the first
                result.update(status=name, errors=check["errors"], report=check["report"])
                result["reason"] = check["reason"]
        result["seconds"] = round(time.perf_counter() - start, 1)
    return result


def _tool_record(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("status", "errors", "report", "reason", "unchecked", "seconds")
    return {k: result[k] for k in keys if k in result}


def _run_tool(name: str, cmd: list[str], timeout: float, tool: Any) -> dict[str, Any]:
    """One run of ``cmd`` (the checked process, captured cases only) under ``--tool name``:
    status ``ok``, ``name`` (its errors), ``error`` or ``timeout``."""
    start = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix=f"ka-{name}-"))
    out: dict[str, Any] = {"status": "error"}
    try:
        with gpu_lock():
            child, log, tail = _sanitize(cmd, timeout, workdir, tool, name)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": f"exceeded {timeout:.0f}s under {name}",
                "seconds": round(time.perf_counter() - start, 1)}  # fmt: skip
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    out["seconds"] = round(time.perf_counter() - start, 1)
    if name == "racecheck":
        errors, warnings, report = parse_race_log(log)
        out["warnings"] = warnings
    else:
        errors, report = parse_log(log)
    if errors:
        what = "shared-memory race(s)" if name == "racecheck" else "barrier error(s)"
        out.update(status=name, errors=errors, report=report)
        out["reason"] = (
            f"{errors} {what} under compute-sanitizer {name}, the first: {_first_lines(report)}"
        )
    elif child is not None and child.get("status") in ("ok", "skipped"):
        out["status"] = child["status"]
    else:
        why = (log.strip().splitlines() or [""])[-1] if child is None else child.get("error")
        out["reason"] = f"the checked process gave no result: {str(why or tail)[-1500:]}"
    if not errors and unchecked(log):
        out["unchecked"] = UNCHECKED
    return out


def describe(result: dict[str, Any]) -> str:
    """One line on a memcheck result (and its racecheck and synccheck, when they ran)."""
    status, seconds = result.get("status"), result.get("seconds")
    took = f" ({seconds} s)" if seconds else ""

    def verdict(rec: dict[str, Any]) -> str:
        if rec.get("status") != "ok":
            return str(rec.get("status"))
        return "checked nothing" if rec.get("unchecked") else "clean"

    others = [
        f"{name} {verdict(rec)}"
        for name, rec in (result.get("tools") or {}).items()
        if name != "memcheck"
    ]
    also = f"; {', '.join(others)}" if others else ""
    if status == "ok":
        n = len(result.get("variants") or [])
        on = f"{result.get('cases')} case(s)" + (f" + {n} odd-size variant(s)" if n else "")
        if result.get("unchecked"):
            return f"memcheck checked nothing on {on}{also}{took}: {result['unchecked']}"
        return f"memcheck clean on {on}{also}{took}"
    if status in FAILED:
        return f"{status} FAILED{took}: {result.get('reason')}"
    return f"memcheck {status}{took}: {result.get('reason')}{also}"


#: Why a kernel that passed the evaluator and the re-check is refused anyway, per tool.
_REFUSED = {
    STATUS: "the overrun is masked out of its outputs, but faults once the memory after the "
    "buffer is released",
    "racecheck": "the race is lost the same way on this GPU today, another clock, schedule or "
    "GPU changes its outputs",
    "synccheck": "a barrier not every thread reaches is undefined behaviour: it hangs or "
    "corrupts on another GPU or driver",
}


def failed(check: dict[str, Any]) -> bool:
    """Whether a memcheck record refuses its kernel (memory, race or barrier errors)."""
    return check.get("status") in FAILED


def refuse(result: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    """A passed re-check ``result`` refused for the errors of ``check`` (memcheck,
    racecheck or synccheck)."""
    status = str(check.get("status") or STATUS)
    result.update(status=status, passed=False, memcheck=check)
    result.pop("warning", None)
    result["reason"] = (
        f"{check.get('reason')} (it passed the evaluator and the re-check: "
        f"{_REFUSED.get(status, _REFUSED[STATUS])})"
    )
    return result


def pending(result: dict[str, Any]) -> bool:
    """Whether a passed re-check record still needs its memcheck: none yet, or one that
    did not decide (skipped, error, timeout)."""
    done = (result.get("memcheck") or {}).get("status") in ("ok", *FAILED)
    return bool(result.get("passed")) and not done


# ------------------------------------------------------------------ the self-test


@dataclasses.dataclass(frozen=True)
class SelfTest:
    """One tool's self-test: ``program`` (a module and its arguments) runs ``clean`` (a
    kernel the tool must not report) and then ``hazard`` (one it must report as an error)."""

    program: tuple[str, ...]
    hazard: str
    clean: str
    hazard_is: str  # what the hazard is, for the verdict's reason
    clean_is: str


#: The self-test of every tool (``kernel-agent doctor``): memcheck's deliberate overrun of a
#: Triton kernel (:mod:`kernel_agent.kernels.memcheck_probe`); racecheck's shared-memory race
#: and synccheck's divergent ``__syncthreads()`` (:mod:`kernel_agent.kernels.sanitizer_probe`,
#: issue #225).
SELFTESTS = {
    "memcheck": SelfTest(
        ("kernel_agent.kernels.memcheck_probe",),
        "_ka_memcheck_overrun",
        "_ka_memcheck_inbounds",
        "the deliberate out-of-bounds read",
        "an in-bounds kernel",
    ),
    "racecheck": SelfTest(
        ("kernel_agent.kernels.sanitizer_probe", "race"),
        "ka_race",
        "ka_ordered",
        "the deliberate shared-memory race",
        "a kernel ordered by its barrier",
    ),
    "synccheck": SelfTest(
        ("kernel_agent.kernels.sanitizer_probe", "sync"),
        "ka_divergent",
        "ka_uniform",
        "the deliberate divergent __syncthreads()",
        "a barrier every thread reaches",
    ),
}


def selftest(tool: Any = None, timeout: float = 300.0, name: str = "memcheck") -> dict[str, Any]:
    """``ok`` when ``--tool name`` reports the deliberate hazard of its probe as an error and
    nothing else (:data:`SELFTESTS`; ``kernel-agent doctor``). ``reason``: the first report
    (or why not); ``errors`` / ``warnings`` / ``report`` as the tool's log has them."""
    from kernel_agent import toolchain

    tool = tool or toolchain.sanitizer()
    if not tool.path:
        return {"ok": False, "reason": tool.reason}
    test = SELFTESTS[name]
    start = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix=f"ka-{name}-"))
    cmd = [sys.executable, "-m", *test.program]
    try:
        with gpu_lock():
            child, log, tail = _sanitize(cmd, timeout, workdir, tool, name)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"exceeded {timeout:.0f}s"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    out: dict[str, Any] = {"seconds": round(time.perf_counter() - start, 1)}
    if name == "racecheck":
        errors, out["warnings"], report = parse_race_log(log)
    else:
        errors, report = parse_log(log)
    out.update(errors=errors, report=report)
    if child is None or child.get("status") != "ok":
        detail = (child or {}).get("error") or log.strip()[-500:] or tail.strip()[-500:]
        out.update(ok=False, reason=f"the probe did not run: {detail}")
    elif test.clean in log:
        out.update(ok=False, reason=f"{test.clean_is} was reported: {_first_lines(report)}")
    elif child.get("clean") is False:  # the sanitizer changed a correct kernel's result
        out.update(ok=False, reason=f"{test.clean_is} ({test.clean}) computed a wrong result")
    elif not errors or test.hazard not in report:
        warned = " (as a warning only)" if out.get("warnings") and test.hazard in report else ""
        out.update(
            ok=False,
            reason=f"{test.hazard_is} was not reported{warned}: a clean {name} run is no "
            "evidence on this GPU",
        )
    else:
        out.update(ok=True, reason=_first_lines(report))
    return out


def selftests(tool: Any = None, timeout: float = 300.0) -> dict[str, dict[str, Any]]:
    """:func:`selftest` of every tool the integration runs (memcheck, racecheck, synccheck)."""
    return {name: selftest(tool, timeout, name) for name in SELFTESTS}


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
