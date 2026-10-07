"""Nsight Compute metrics and compiler stats as feedback for kernel candidates (issue #10).

``evaluate_candidate(profile="ncu")`` adds, after a correct evaluation, a curated set of
Nsight Compute metrics per candidate kernel (CudaForge: a few chosen metrics beat both no
metrics and the full report; KernelAgent: a SOL roofline classifier):

* :func:`profile_candidate` runs ``ncu --csv --page raw --replay-mode kernel --metrics
  <METRICS>`` over ``python -m kernel_agent.kernels.evaluate CAPTURE CANDIDATE
  --ncu-mode`` (:func:`ncu_entry`: build the candidate, call it on its most-called case
  once to compile, then :data:`NCU_CALLS` times inside the NVTX range
  :data:`NVTX_RANGE`, the only kernels ncu profiles) under the GPU lock. A metric this
  ncu or GPU does not know is dropped and the run repeated.
* :func:`parse_csv` reads the raw page (one row per kernel launch, a units row);
  :func:`kernels` aggregates launches per kernel and :func:`classify` labels each
  ``memory`` / ``compute`` bound by the larger of its SM and memory throughput (% of peak),
  or ``under-utilised`` when both are below :data:`UNDER_UTILISED_PCT` (latency, occupancy,
  too few blocks: the top warp stalls and the occupancy say which).
* :func:`availability` finds ``ncu`` (``KERNEL_AGENT_NCU``, ``PATH``, ``CUDA_HOME``, the
  usual install prefixes) and reads whether the driver restricts the GPU performance
  counters to admin users (``RmProfilingAdminOnly`` in ``/proc/driver/nvidia/params``;
  ncu then fails with ``ERR_NVGPUCTRPERM``): ``profile="ncu"`` then returns
  ``{"status": "unavailable", "reason": ...}`` with the fix instead of failing.
* :func:`compiler_stats` (no GPU work, every ``profile`` evaluation): registers, spills
  and shared memory of the candidate's Triton kernels (``n_regs`` / ``n_spills`` of the
  compiled kernels), its NVRTC kernels (``cuda.core`` kernel attributes) and its
  ``load_inline`` extensions (``cuobjdump --dump-resource-usage``, the ``ptxas -v``
  numbers of the cubins inside); spills are the first thing to fix.

Times under ncu are per launch with caches flushed and clocks locked to base (ncu's
defaults): compare kernels with each other, not with the evaluator's timings.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

#: The curated metrics: ncu name -> short name in the result.
METRICS: dict[str, str] = {
    "gpu__time_duration.sum": "time_ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor_pipe_pct",
    "launch__registers_per_thread": "registers",
    "launch__shared_mem_per_block_static": "smem_static_bytes",
    "launch__shared_mem_per_block_dynamic": "smem_dynamic_bytes",
    "launch__grid_size": "grid_size",
    "launch__block_size": "block_size",
    "launch__waves_per_multiprocessor": "waves",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": (
        "stall_long_scoreboard"
    ),
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": (
        "stall_short_scoreboard"
    ),
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "stall_mio_throttle",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": (
        "stall_math_pipe_throttle"
    ),
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "stall_barrier",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "stall_wait",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio": "stall_lg_throttle",
}
#: What each warp stall means, for the classification's hint.
STALLS = {
    "stall_long_scoreboard": "waiting on global / local memory loads",
    "stall_short_scoreboard": "waiting on shared memory or MUFU results",
    "stall_mio_throttle": "shared-memory / special-instruction queue full",
    "stall_math_pipe_throttle": "a math pipe is saturated",
    "stall_barrier": "waiting at __syncthreads / barriers",
    "stall_wait": "fixed-latency dependencies (too little ILP)",
    "stall_lg_throttle": "global-memory instruction queue full (too many small loads)",
}
#: Both throughputs below this share of peak: the kernel is not limited by either.
UNDER_UTILISED_PCT = 60.0
NVTX_RANGE = "ka_candidate"
NCU_CALLS = 3
LAUNCH_COUNT = 40  # kernel launches ncu profiles at most
TIMEOUT_S = 600.0
NCU_ENV = "KERNEL_AGENT_NCU"
PERMISSION_HINT = (
    "the driver restricts GPU performance counters to admin users (ERR_NVGPUCTRPERM): set "
    "`options nvidia NVreg_RestrictProfilingToAdminUsers=0` in a file in /etc/modprobe.d/, "
    "rebuild the initramfs and reboot (or run as root); "
    "https://developer.nvidia.com/ERR_NVGPUCTRPERM"
)
_TIME_UNITS = {"nsecond": 1.0, "usecond": 1e3, "msecond": 1e6, "second": 1e9}
_BYTE_UNITS = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}


# ------------------------------------------------------------------ finding ncu


def _places() -> list[Path]:
    places: list[Path] = []
    if override := os.environ.get(NCU_ENV):
        places.append(Path(override).expanduser())
    if found := shutil.which("ncu"):
        places.append(Path(found))
    for home in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if home:
            places.append(Path(home) / "bin" / "ncu")
    places.append(Path("/opt/cuda/bin/ncu"))
    for pattern in ("/opt/nvidia/nsight-compute/*/ncu", "/usr/local/NVIDIA-Nsight-Compute*/ncu"):
        places += [Path(p) for p in sorted(glob.glob(pattern), reverse=True)]
    return list(dict.fromkeys(places))


def find_ncu(places: Iterable[Path] | None = None) -> str | None:
    """The first executable ``ncu`` of ``places`` (default: :func:`_places`)."""
    for path in _places() if places is None else places:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def ncu_version(ncu: str) -> str | None:
    try:
        out = subprocess.run([ncu, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"Version (\S+)", out.stdout)
    return match.group(1) if match else None


def counters_restricted(params: str | None = None, *, root: bool | None = None) -> bool | None:
    """Whether the NVIDIA driver restricts the performance counters to admin users and this
    process is not one (``params``: the text of ``/proc/driver/nvidia/params``; None: it
    cannot be read, so unknown)."""
    if params is None:
        try:
            params = Path("/proc/driver/nvidia/params").read_text()
        except OSError:
            return None
    match = re.search(r"RmProfilingAdminOnly:\s*(\d+)", params)
    if match is None:
        return None
    if root is None:
        root = hasattr(os, "geteuid") and os.geteuid() == 0
    return match.group(1) != "0" and not root


def availability() -> dict[str, Any]:
    """``{"ncu", "version", "usable", "reason"}``: whether ``profile="ncu"`` can work here
    (CPU only: no profiling run)."""
    ncu = find_ncu()
    if ncu is None:
        return {
            "ncu": None,
            "usable": False,
            "reason": "Nsight Compute (`ncu`) not found: install it (CUDA toolkit or the "
            f"standalone Nsight Compute) or set {NCU_ENV}",
        }
    out: dict[str, Any] = {"ncu": ncu, "version": ncu_version(ncu), "usable": True}
    if counters_restricted():
        out.update(usable=False, reason=PERMISSION_HINT)
    return out


# ------------------------------------------------------------------ CSV


def _number(raw: str) -> float | str | None:
    text = raw.strip()
    if not text or text.lower() in {"n/a", "nan", "-"}:
        return None
    try:
        return float(text.replace(",", ""))  # thousands separators
    except ValueError:
        return text


def _scale(value: Any, unit: str) -> Any:
    """``value`` in ns or bytes when ``unit`` is a time or size unit."""
    factor = _TIME_UNITS.get(unit) or _BYTE_UNITS.get(unit)
    return value * factor if isinstance(value, float) and factor else value


def parse_csv(text: str) -> list[dict[str, Any]]:
    """The launches of an ``ncu --csv --page raw`` report: ``[{"id", "kernel", "grid",
    "block", "stream", "metrics": {short name or ncu name: value}}]``, times in ns and
    sizes in bytes whatever ``--print-units`` was (lines before the header, ncu's
    ``==PROF==`` log, are skipped)."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        return []
    reader = csv.reader(io.StringIO("\n".join(lines[start:])))
    header = next(reader)
    units: dict[str, str] = {}
    out = []
    for row in reader:
        if len(row) != len(header):
            continue
        record = dict(zip(header, row, strict=True))
        if not record.get("ID"):  # the units row
            units = {k: v for k, v in record.items() if v}
            continue
        metrics = {
            METRICS.get(name, name): _scale(_number(record[name]), units.get(name, ""))
            for name in header
            if "__" in name
        }
        out.append(
            {
                "id": int(_number(record["ID"]) or 0),
                "kernel": record.get("Kernel Name", ""),
                "grid": record.get("Grid Size", ""),
                "block": record.get("Block Size", ""),
                "stream": record.get("Stream", ""),
                "metrics": metrics,
            }
        )
    return out


#: Metrics that add up over launches (the others are averaged, weighted by time).
_SUMMED = {"time_ns", "dram_read_bytes", "dram_write_bytes"}


def kernels(launches: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Launches aggregated per kernel name, by total time: ``launches``, summed time and
    bytes, other metrics time-weighted, the classification (:func:`classify`)."""
    grouped: dict[str, dict[str, Any]] = {}
    for launch in launches:
        m = launch["metrics"]
        weight = m.get("time_ns") if isinstance(m.get("time_ns"), float) else 1.0
        g = grouped.setdefault(
            launch["kernel"],
            {
                "kernel": launch["kernel"],
                "launches": 0,
                "grid": launch.get("grid"),
                "block": launch.get("block"),
                "_sum": {},
                "_w": {},
            },
        )
        g["launches"] += 1
        for name, value in m.items():
            if not isinstance(value, float):
                continue
            if name in _SUMMED:
                g["_sum"][name] = g["_sum"].get(name, 0.0) + value
            else:
                acc = g["_w"].setdefault(name, [0.0, 0.0])
                acc[0] += value * weight
                acc[1] += weight
    out = []
    for g in grouped.values():
        metrics = {k: round(v, 3) for k, v in g.pop("_sum").items()}
        metrics |= {k: round(a / w, 3) for k, (a, w) in g.pop("_w").items() if w}
        g["metrics"] = metrics
        g.update(classify(metrics))
        out.append(g)
    out.sort(key=lambda g: -float(g["metrics"].get("time_ns") or 0.0))
    return out


def top_stalls(metrics: Mapping[str, Any], n: int = 3) -> list[tuple[str, float]]:
    """The largest warp-stall ratios (cycles a warp waits per issued instruction)."""
    stalls = [(k, float(v)) for k, v in metrics.items() if k in STALLS and isinstance(v, float)]
    return sorted(stalls, key=lambda kv: -kv[1])[:n]


def classify(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """``{"bound": memory | compute | under-utilised | unknown, "why": ...}`` from the SM and
    memory throughputs (% of peak), like KernelAgent's ``ncu_roofline.py``."""
    sm, mem = metrics.get("sm_pct"), metrics.get("memory_pct")
    if not isinstance(sm, float) or not isinstance(mem, float):
        return {"bound": "unknown", "why": "no SM / memory throughput in the report"}
    stalls = ", ".join(
        f"{k.removeprefix('stall_')} {v:.1f} ({STALLS[k]})" for k, v in top_stalls(metrics)
    )
    occ, theo = metrics.get("achieved_occupancy_pct"), metrics.get("theoretical_occupancy_pct")
    occupancy = (
        f"achieved occupancy {occ:.0f} % of {theo:.0f} % possible"
        if isinstance(occ, float) and isinstance(theo, float)
        else ""
    )
    if max(sm, mem) < UNDER_UTILISED_PCT:
        why = [f"SM {sm:.0f} % and memory {mem:.0f} % of peak: latency bound"]
        waves = metrics.get("waves")
        if isinstance(waves, float) and waves < 1.0:
            why.append(f"{waves:.2f} waves: too few blocks to fill the GPU")
        why += [x for x in (occupancy, f"top stalls: {stalls}" if stalls else "") if x]
        return {"bound": "under-utilised", "why": "; ".join(why)}
    if mem >= sm:
        why = [f"memory {mem:.0f} % of peak (SM {sm:.0f} %)"]
        for key, label in (("dram_pct", "DRAM"), ("l2_hit_pct", "L2 hit rate")):
            if isinstance(v := metrics.get(key), float):
                why.append(f"{label} {v:.0f} %")
        return {"bound": "memory", "why": "; ".join(why)}
    why = [f"SM {sm:.0f} % of peak (memory {mem:.0f} %)"]
    if isinstance(tensor := metrics.get("tensor_pipe_pct"), float):
        why.append(f"tensor pipe {tensor:.0f} %")
    if stalls:
        why.append(f"top stalls: {stalls}")
    return {"bound": "compute", "why": "; ".join(why)}


# ------------------------------------------------------------------ running ncu


def command(
    ncu: str,
    capture: Path,
    candidate: Path,
    metrics: Iterable[str],
    log: Path,
    *,
    capture_sha256: str | None = None,
    calls: int = NCU_CALLS,
) -> list[str]:
    """The ncu command line over the evaluator's ``--ncu-mode`` entry."""
    cmd = [
        ncu,
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--replay-mode",
        "kernel",
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        f"{NVTX_RANGE}/",
        "--launch-count",
        str(LAUNCH_COUNT),
        "--metrics",
        ",".join(metrics),
        "--log-file",
        str(log),
        sys.executable,
        "-m",
        "kernel_agent.kernels.evaluate",
        str(capture),
        str(candidate),
        "--ncu-mode",
        "--ncu-calls",
        str(calls),
    ]
    if capture_sha256:
        cmd += ["--capture-sha256", capture_sha256]
    return cmd


_UNKNOWN_METRIC = re.compile(
    r"(?:Failed to find metric|unknown metric|Invalid metric)s?\W*(?:regex:\^?)?([\w.]+)", re.I
)


def unknown_metrics(output: str, metrics: Iterable[str]) -> list[str]:
    """The requested metrics ncu's ``output`` says it does not know."""
    named = {m.group(1) for m in _UNKNOWN_METRIC.finditer(output)}
    return [m for m in metrics if m in named or any(m.startswith(n) for n in named if n)]


def profile_candidate(
    capture: Path,
    candidate: Path,
    *,
    capture_sha256: str | None = None,
    timeout: float = TIMEOUT_S,
    run: Any = subprocess.run,
) -> dict[str, Any]:
    """Nsight Compute metrics of ``candidate``'s kernels on its capture (module docstring):
    ``{"status": "ok", "kernels": [...], ...}``, or ``"unavailable"`` / ``"error"`` with
    the reason. Never raises."""
    found = availability()
    if not found["usable"]:
        return {"status": "unavailable", "reason": found["reason"]}
    from kernel_agent.gpulock import child_env, gpu_lock

    metrics = list(METRICS)
    dropped: list[str] = []
    workdir = Path(tempfile.mkdtemp(prefix="ka-ncu-"))
    log = workdir / "ncu.csv"
    try:
        with gpu_lock():
            for _ in range(3):  # drop unknown metrics and try again
                cmd = command(
                    found["ncu"], capture, candidate, metrics, log, capture_sha256=capture_sha256
                )
                log.unlink(missing_ok=True)
                try:
                    proc = run(
                        cmd, capture_output=True, text=True, timeout=timeout, env=child_env()
                    )
                except subprocess.TimeoutExpired:
                    return {"status": "error", "reason": f"ncu exceeded {timeout:.0f} s"}
                text = log.read_text(errors="replace") if log.exists() else ""
                output = "\n".join((text, proc.stdout or "", proc.stderr or ""))
                if "ERR_NVGPUCTRPERM" in output:
                    return {"status": "unavailable", "reason": PERMISSION_HINT}
                launches = parse_csv(text)
                if launches:
                    break
                bad = unknown_metrics(output, metrics)
                if not bad:
                    return {
                        "status": "error",
                        "reason": f"ncu exited {proc.returncode} without a report: "
                        + output.strip()[-1500:],
                    }
                dropped += bad
                metrics = [m for m in metrics if m not in bad]
            else:
                return {"status": "error", "reason": "ncu rejected the metrics", "dropped": dropped}
    except Exception as exc:  # OSError, a broken lock, ...
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:500]}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    found_kernels = kernels(launches)
    return {
        "status": "ok",
        "ncu": found.get("version"),
        "launches": len(launches),
        "kernels": found_kernels[:8],
        **({"dropped_metrics": dropped} if dropped else {}),
        "note": "per launch under ncu (caches flushed, base clocks): compare kernels with "
        "each other; bound = the larger of SM and memory throughput, under-utilised when "
        f"both are below {UNDER_UTILISED_PCT:.0f} % of peak",
    }


def ncu_entry(
    capture_path: Path,
    candidate_path: Path,
    *,
    capture_sha256: str | None = None,
    calls: int = NCU_CALLS,
) -> int:
    """The evaluator's ``--ncu-mode`` (run under ncu): build the candidate, call it once on
    its most-called captured case (compiles, autotunes), then ``calls`` times inside the
    NVTX range :data:`NVTX_RANGE` with fresh copies of the inputs (copied outside it)."""
    import torch

    from kernel_agent import toolchain
    from kernel_agent.kernels.evaluate import load_candidate_module
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.state import Replay, split

    toolchain.setup()
    capture = load_capture(capture_path, device="cuda", sha256=capture_sha256)
    reference = capture["module"].eval()
    replay = Replay(capture, reference)  # the case's module state, outside the NVTX range
    case = max(capture["cases"], key=lambda c: int(c.get("count") or 0))
    module = load_candidate_module(candidate_path)
    given = copy.deepcopy(reference)
    candidate = module.build(given)
    fn, restore = split(replay.call(case, candidate, given))
    with torch.inference_mode():
        for i in range(calls + 1):
            args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
            if restore is not None:
                restore()
            torch.cuda.synchronize()
            ranged = torch.cuda.nvtx.range(NVTX_RANGE) if i else contextlib.nullcontext()
            with ranged:
                fn(*args, **kwargs)
                torch.cuda.synchronize()
    return 0


# ------------------------------------------------------------------ compiler stats


def _triton_kernels(obj: Any) -> Iterator[Any]:
    """Compiled kernels of a Triton ``JITFunction`` (or an ``Autotuner`` / ``Heuristics``
    around one); empty for anything else or a Triton whose caches look different."""
    for _ in range(4):  # Autotuner(Heuristics(JITFunction))
        if hasattr(obj, "device_caches"):
            break
        obj = getattr(obj, "fn", None)
        if obj is None:
            return
    caches = getattr(obj, "device_caches", None)
    if not isinstance(caches, Mapping):
        return
    for entry in caches.values():
        cache = entry[0] if isinstance(entry, tuple) and entry else entry
        if isinstance(cache, Mapping):
            yield from cache.values()


def triton_stats(namespace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Registers, spills and shared memory of the Triton kernels compiled from the
    functions in ``namespace`` (a candidate module's globals)."""
    out = []
    for name, obj in namespace.items():
        for kernel in _triton_kernels(obj):
            meta = getattr(kernel, "metadata", None)
            out.append(
                {
                    "kernel": str(getattr(meta, "name", None) or name),
                    "registers": getattr(kernel, "n_regs", None),
                    "spills": getattr(kernel, "n_spills", None),
                    "shared_bytes": getattr(meta, "shared", None),
                    "num_warps": getattr(meta, "num_warps", None),
                    "num_stages": getattr(meta, "num_stages", None),
                }
            )
    return out


def _cuda_core_kernels(values: Iterable[Any]) -> Iterator[tuple[str, Any]]:
    for name, obj in values:
        if type(obj).__module__.startswith("cuda.core") and hasattr(obj, "attributes"):
            yield name, obj


def nvrtc_stats(namespace: Mapping[str, Any], candidate: Any = None) -> list[dict[str, Any]]:
    """Registers, local memory (spills) and static shared memory of the ``cuda.core``
    kernels in ``namespace`` and in ``candidate``'s attributes."""
    values = list(namespace.items())
    if candidate is not None:
        values += list(getattr(candidate, "__dict__", {}).items())
    out, seen = [], set()
    for name, kernel in _cuda_core_kernels(values):
        if id(kernel) in seen:
            continue
        seen.add(id(kernel))
        attrs = kernel.attributes
        row: dict[str, Any] = {"kernel": name}
        for key, attr in (
            ("registers", "num_regs"),
            ("local_bytes", "local_size_bytes"),
            ("shared_bytes", "shared_size_bytes"),
            ("max_threads", "max_threads_per_block"),
        ):
            with contextlib.suppress(Exception):
                value = getattr(attrs, attr)
                row[key] = value() if callable(value) else value
        out.append(row)
    return out


def parse_resource_usage(text: str) -> list[dict[str, Any]]:
    """Kernels of ``cuobjdump --dump-resource-usage`` output: ``[{"kernel", "registers",
    "stack_bytes", "shared_bytes", "local_bytes", "arch"}]``."""
    out = []
    arch = None
    name = None
    for line in text.splitlines():
        if m := re.search(r"arch\s*=\s*(sm_\w+)", line):
            arch = m.group(1)
        if m := re.match(r"\s*Function\s+(\S+?):?\s*$", line):
            name = m.group(1)
            continue
        if name and (m := re.search(r"\bREG:(\d+)", line)):
            values = dict(re.findall(r"\b(STACK|SHARED|LOCAL):(\d+)", line))
            out.append(
                {
                    "kernel": name,
                    "registers": int(m.group(1)),
                    "stack_bytes": int(values.get("STACK", 0)),
                    "shared_bytes": int(values.get("SHARED", 0)),
                    "local_bytes": int(values.get("LOCAL", 0)),
                    **({"arch": arch} if arch else {}),
                }
            )
            name = None
    return out


def parse_ptxas(text: str) -> list[dict[str, Any]]:
    """Kernels of ``ptxas -v`` output: ``[{"kernel", "registers", "spill_stores",
    "spill_loads", "stack_bytes", "shared_bytes"}]``."""
    out: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        m = re.search(r"Function properties for (\S+)", line) or re.search(
            r"Compiling entry function '([^']+)'", line
        )
        if m:
            current = next((k for k in out if k["kernel"] == m.group(1)), None)
            if current is None:
                current = {"kernel": m.group(1)}
                out.append(current)
        elif current is not None and (
            m := re.search(
                r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line
            )
        ):
            current.update(
                stack_bytes=int(m.group(1)),
                spill_stores=int(m.group(2)),
                spill_loads=int(m.group(3)),
            )
        elif current is not None and (m := re.search(r"Used (\d+) registers", line)):
            current["registers"] = int(m.group(1))
            if s := re.search(r"(\d+) bytes smem", line):
                current["shared_bytes"] = int(s.group(1))
    return out


def cuobjdump() -> str | None:
    """A ``cuobjdump``: the CUDA toolkit's, else the one Triton bundles."""
    from kernel_agent import toolchain

    home = toolchain.setup().cuda_home
    places = [Path(home) / "bin" / "cuobjdump"] if home else []
    if found := shutil.which("cuobjdump"):
        places.append(Path(found))
    with contextlib.suppress(ImportError):
        import triton

        places.append(Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump")
    return next((str(p) for p in places if p.is_file() and os.access(p, os.X_OK)), None)


def extension_files() -> list[Path]:
    """Shared libraries of the loaded ``torch.utils.cpp_extension`` builds (``load_inline``:
    under ``TORCH_EXTENSIONS_DIR`` or a ``torch_extensions`` cache directory)."""
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    out = []
    for module in list(sys.modules.values()):
        path = str(getattr(module, "__file__", "") or "")
        if path.endswith(".so") and (
            "torch_extensions" in path or (root and path.startswith(root))
        ):
            out.append(Path(path))
    return sorted(set(out))


def cuda_stats(files: Iterable[Path], tool: str | None = None) -> list[dict[str, Any]]:
    """Per-kernel resource usage of the cubins in ``files`` (``cuobjdump``)."""
    tool = tool or cuobjdump()
    if tool is None:
        return []
    out = []
    for path in files:
        try:
            proc = subprocess.run(
                [tool, "--dump-resource-usage", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        out += [{**k, "file": path.name} for k in parse_resource_usage(proc.stdout)]
    return out


def compiler_stats(module: Any, candidate: Any = None) -> dict[str, Any]:
    """Compiler stats of a candidate (module docstring): ``{"triton": [...], "nvrtc": [...],
    "cuda": [...], "warnings": [...]}`` with the empty lists left out. No GPU work."""
    namespace = dict(vars(module)) if module is not None else {}
    out: dict[str, Any] = {}
    for key, collect in (
        ("triton", lambda: triton_stats(namespace)),
        ("nvrtc", lambda: nvrtc_stats(namespace, candidate)),
        ("cuda", lambda: cuda_stats(extension_files())),
    ):
        try:
            rows = collect()
        except Exception as exc:  # never breaks an evaluation
            out[f"{key}_error"] = f"{type(exc).__name__}: {exc}"[:200]
            continue
        if rows:
            out[key] = rows[:12]
    warnings = []
    for rows in (out.get("triton") or [], out.get("nvrtc") or [], out.get("cuda") or []):
        for k in rows:
            spills = k.get("spills") or k.get("local_bytes") or k.get("spill_stores")
            if spills:
                warnings.append(
                    f"{k['kernel'][:80]}: {spills} spilled (registers {k.get('registers')}): "
                    "spills go to local memory; fewer live values, smaller tiles or fewer "
                    "warps' worth of state fix it"
                )
    if warnings:
        out["warnings"] = warnings[:6]
    return out


def describe(found: Mapping[str, Any]) -> str:
    """One ``doctor`` line on :func:`availability`."""
    if found.get("ncu") is None:
        return f"ncu: unavailable ({found.get('reason')})"
    text = f"ncu {found.get('version') or '?'} ({found['ncu']})"
    if not found.get("usable"):
        text += f': profile="ncu" unavailable: {found.get("reason")}'
    return text
