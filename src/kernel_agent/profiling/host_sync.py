"""Host synchronisation in a workload's run, with call sites (issue #149).

A generation loop that reads a device value on the host every step (``.item()`` of a stop
token, ``.cpu()`` of a finished mask), or copies a tensor from pageable host memory (a
position built with ``torch.tensor([...], device=...)`` each step), makes the host wait
until the GPU has drained the work queued so far; the GPU then idles while the host
launches the next step. The kernel view shows only the idle time, not the line of Python
that caused it. :func:`scan` runs the workload once under a ``TorchFunctionMode`` that
records every such call (no kernels change; the run is only slower in Python), grouped by
call site (the first frame outside torch: the workload's loop, the model's package, a
transform), with the host time spent in it. :func:`summary_lines` turns the sites into
transform opportunities for the planner and the systems agent (``profile/summary.md``):

* ``item``: a device value read on the host (``.item()``, ``.tolist()``, ``.numpy()``,
  ``bool()`` / ``int()`` / ``float()`` of a device tensor, ``if tensor:``). Fix: *async
  flag read* (``Workload.async_flags()``) or keep the value on the device.
* ``d2h``: a blocking device-to-host copy (``.cpu()``, ``.to("cpu")``, ``copy_`` into a
  host tensor). Fix: *async flag read* / a pinned, non-blocking copy read later
  (``serving.HostCopy``).
* ``h2d``: a host-to-device copy that is blocking or from pageable memory (``.to(device)``,
  ``.cuda()``, ``copy_`` of an unpinned host tensor). Fix: *pinned non-blocking copy*, or
  keep the tensor on the device.
* ``host_tensor``: ``torch.tensor(...)`` / ``torch.as_tensor(...)`` of host data for a
  device: built on the host and copied from pageable memory every call. Fix: *build the
  constant once* (cache it on the device) or compute it on the device.
* ``data_dependent``: an output size only the device knows (``nonzero``, ``unique``,
  ``masked_select``, boolean-mask indexing, one-argument ``torch.where``). Fix: *static
  shapes* (masks and ``torch.where`` instead of gathering the selected rows).

A CUDA-graph replay makes no Python calls: what it captured is not seen. A
``torch.compile``'d region may trace or graph-break around the mode; when the scanned run
fails under it, :func:`scan` reports the error with what it recorded until then.
"""

from __future__ import annotations

import linecache
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from types import FrameType
from typing import Any

import torch
from torch.overrides import TorchFunctionMode

#: kind -> (what it is, the fix's name in :data:`FIXES`)
KINDS: dict[str, tuple[str, str]] = {
    "item": ("device value read on the host", "async flag read"),
    "d2h": ("blocking device-to-host copy", "async flag read"),
    "h2d": ("blocking / pageable host-to-device copy", "pinned non-blocking copy"),
    "host_tensor": ("tensor built on the host, copied to the device", "build constant once"),
    "data_dependent": ("output size read on the host", "static shapes"),
}
FIXES: dict[str, str] = {
    "async flag read": (
        "compute the flag (stop token, finished mask, a count) as early in the step as it "
        "exists, `ticket = flags.send(x)` (pinned memory, `non_blocking=True`, an event), queue "
        "the rest of the step and `flags.read(ticket)` only then, or a step later "
        "(`Workload.async_flags()`, `kernel_agent.workloads.serving.AsyncFlags`; the same "
        "values, bit-identical when read in the same step); or keep the value on the device "
        "and branch there (`torch.where`). A transform can do it where the loop is the "
        "model's code (wrap or replace the method that holds the call site)"
    ),
    "pinned non-blocking copy": (
        "pin the host tensor once (`pin_memory()`) and copy it with `non_blocking=True`, or "
        "keep it on the device across calls"
    ),
    "build constant once": (
        "make the tensor once (at load / first call) and keep it on the device, or build "
        "it on the device (`torch.arange(..., device=...)`, an index kept on the device and "
        "advanced there) instead of `torch.tensor([...], device=...)` per call"
    ),
    "static shapes": (
        "keep shapes static: masks and `torch.where` instead of selecting rows, a "
        "`nonzero_static` / `output_size=` with a known bound"
    ),
}
#: Sites listed in the summary (the rest are counted).
SHOWN = 12

_SCALAR = {"item", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__"}
_DATA_DEPENDENT = {"nonzero", "argwhere", "unique", "unique_consecutive", "masked_select"}
_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep
_SKIP_FILES = (_TORCH_DIR, os.path.abspath(__file__))


def on_device(x: Any) -> bool:
    """A tensor on an accelerator (the default ``is_device`` of :class:`Recorder`)."""
    return isinstance(x, torch.Tensor) and x.device.type not in ("cpu", "meta")


def _target(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.device | None:
    """The device a ``Tensor.to(...)`` / ``.cuda(...)`` call copies to (None: no device)."""
    for value in (kwargs.get("device"), *args):
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, torch.device):
            return value
        if isinstance(value, str):
            try:
                return torch.device(value)
            except RuntimeError:
                continue
    return None


def _non_blocking(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if "non_blocking" in kwargs:
        return bool(kwargs["non_blocking"])
    return any(a is True for a in args)  # .to(device, dtype, True) / copy_(src, True)


def _pinned(t: Any) -> bool:
    if not isinstance(t, torch.Tensor):
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False  # no pointer queries inside a graph capture (it cannot copy anyway)
    try:
        return bool(t.is_pinned())
    except RuntimeError:
        return False


def _bool_index(index: Any, is_device: Callable[[Any], bool]) -> bool:
    items = index if isinstance(index, tuple) else (index,)
    return any(
        isinstance(i, torch.Tensor) and i.dtype == torch.bool and is_device(i) for i in items
    )


def classify(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    is_device: Callable[[Any], bool] = on_device,
) -> tuple[str, str] | None:
    """``(kind, op)`` of a torch function call that synchronises the host (or copies from
    pageable memory), None for every other call. ``is_device(tensor)``: is it on the GPU
    (injectable for CPU tests)."""
    name = getattr(func, "__name__", "")
    first = args[0] if args else None
    if name in _SCALAR:
        if is_device(first):
            return "item", f"{name.strip('_')}(tensor)" if name.startswith("__") else f".{name}()"
        return None
    if name == "cpu":
        return ("d2h", ".cpu()") if is_device(first) else None
    if name in ("to", "cuda"):
        if not isinstance(first, torch.Tensor):
            return None
        target = _target(args[1:], kwargs)
        if name == "cuda":
            target = torch.device("cuda")
        if target is None:
            return None
        if is_device(first) and target.type == "cpu" and not _non_blocking(args[1:], kwargs):
            return "d2h", '.to("cpu")'
        host = not is_device(first) and target.type not in ("cpu", "meta")
        if host and not (_non_blocking(args[1:], kwargs) and _pinned(first)):
            return "h2d", f".{name}(device)"
        return None
    if name == "copy_" and len(args) >= 2:
        dst, src = args[0], args[1]
        fast = _non_blocking(args[2:], kwargs)
        if is_device(src) and not is_device(dst) and not (fast and _pinned(dst)):
            return "d2h", "host.copy_(device)"
        host = is_device(dst) and not is_device(src) and isinstance(src, torch.Tensor)
        if host and not (fast and _pinned(src)):
            return "h2d", "device.copy_(host)"
        return None
    if func is torch.tensor or func is torch.as_tensor:
        device = kwargs.get("device")
        if device is None or torch.device(device).type in ("cpu", "meta"):
            return None
        if is_device(first):  # already there: as_tensor is a no-op, tensor a device copy
            return None
        return "host_tensor", f"torch.{name}(..., device=...)"
    if name in _DATA_DEPENDENT:
        return ("data_dependent", f"{name}()") if is_device(first) else None
    if name == "where" and len(args) == 1 and is_device(first):
        return "data_dependent", "torch.where(condition)"
    if name == "repeat_interleave" and is_device(first) and "output_size" not in kwargs:
        repeats = args[1] if len(args) > 1 else kwargs.get("repeats")
        return ("data_dependent", "repeat_interleave()") if is_device(repeats) else None
    indexing = name in ("__getitem__", "__setitem__") and len(args) >= 2
    if indexing and is_device(first) and _bool_index(args[1], is_device):
        return "data_dependent", f"tensor[mask]{' = ...' if name == '__setitem__' else ''}"
    return None


@dataclass
class Site:
    """One call site of host synchronisation: what, where, how often per run."""

    kind: str
    op: str
    file: str
    line: int
    function: str
    code: str
    where: str  # "workload" | "model" | "other" (a transform, a kernel, a harness)
    calls: int = 0
    host_ms: float = 0.0


def _short(path: str) -> tuple[str, str]:
    """``(shortened path, where)`` of a call site's file."""
    norm = path.replace(os.sep, "/")
    if "/kernel_agent/" in norm:
        return "kernel_agent/" + norm.rsplit("/kernel_agent/", 1)[1], "workload"
    if "-packages/" in norm:
        return norm.rsplit("-packages/", 1)[1], "model"
    try:
        rel = os.path.relpath(path)
    except ValueError:
        rel = path
    return (rel if len(rel) < len(path) else path), "other"


def _call_site() -> tuple[str, int, str]:
    """The first frame outside torch and this module: the code that made the call."""
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_SKIP_FILES):
        frame = frame.f_back
    if frame is None:
        return "?", 0, "?"
    return frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name


class Recorder(TorchFunctionMode):
    """Records every synchronising torch call (:func:`classify`) by call site while active.
    Never changes a call: it runs it and times it."""

    def __init__(self, is_device: Callable[[Any], bool] | None = None) -> None:
        super().__init__()
        self.is_device = is_device or on_device  # (resolved now: tests patch it)
        self.sites: dict[tuple[str, str, str, int], Site] = {}

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        try:
            hit = classify(func, args, kwargs, self.is_device)
        except Exception:  # never let the bookkeeping break the run
            hit = None
        if hit is None:
            return func(*args, **kwargs)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self._record(hit, (time.perf_counter() - start) * 1000)

    def _record(self, hit: tuple[str, str], ms: float) -> None:
        path, line, function = _call_site()
        key = (hit[0], hit[1], path, line)
        site = self.sites.get(key)
        if site is None:
            short, where = _short(path)
            code = linecache.getline(path, line).strip()[:120]
            site = self.sites[key] = Site(hit[0], hit[1], short, line, function, code, where)
        site.calls += 1
        site.host_ms += ms

    def rows(self) -> list[dict[str, Any]]:
        """The sites, most calls first."""
        sites = sorted(self.sites.values(), key=lambda s: (-s.calls, -s.host_ms))
        return [{**asdict(s), "host_ms": round(s.host_ms, 3)} for s in sites]


def scan(
    workload: Any, inputs: Any, *, is_device: Callable[[Any], bool] | None = None
) -> dict[str, Any]:
    """One run of ``workload`` (warmed up) under a :class:`Recorder`: ``{"sites": [...],
    "calls": n, "host_ms": ms}``, plus ``error`` when the run failed under it (what was
    recorded until then is kept; a compiled region that cannot trace the mode, say)."""
    from kernel_agent.workloads.base import synchronize

    recorder = Recorder(is_device)
    error = None
    synchronize()
    try:
        with torch.inference_mode(), recorder:
            workload.run(inputs)
        synchronize()
    except Exception as exc:
        error = " ".join(f"{type(exc).__name__}: {exc}".split())[:400]
    rows = recorder.rows()
    out: dict[str, Any] = {
        "sites": rows,
        "calls": sum(r["calls"] for r in rows),
        "host_ms": round(sum(r["host_ms"] for r in rows), 3),
    }
    if error:
        out["error"] = error
    return out


def summary_lines(found: dict[str, Any] | None, shown: int = SHOWN) -> list[str]:
    """The profile summary's section on :func:`scan` (``[]`` for a profile without one)."""
    if not found:
        return []
    sites = found.get("sites") or []
    lines = ["", "## Host synchronisation (transform opportunities)", ""]
    if found.get("error"):
        lines.append(
            f"* the scan run failed ({found['error']}); the sites below were recorded before "
            "the failure"
        )
    if not sites:
        lines.append(
            "* none found: no `.item()` / `.cpu()` / pageable host-to-device copy / "
            "`torch.tensor(..., device=...)` / data-dependent shape in the run's Python code"
        )
        return lines
    loops = [s for s in sites if s["calls"] > 1]
    lines += [
        f"{found['calls']} calls in one run make the host wait for the GPU (or copy from "
        f"pageable host memory) at {len(sites)} call sites, {len(loops)} of them more than "
        "once per run (a loop). Each one in a generation loop idles the GPU while the host "
        "catches up: it cannot queue the next step until the GPU has drained. *host ms*: the "
        "host's time inside these calls during the scanned run (indicative: the scan runs "
        "Python slower). *where*: the workload's own code, the model's package, or other "
        "code (a transform, a kernel, a harness).",
        "",
        "| calls | kind | op | where | call site | host ms | fix |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sites[:shown]:
        what, fix = KINDS.get(s["kind"], (s["kind"], ""))
        code = f": `{s['code']}`" if s.get("code") else ""
        lines.append(
            f"| {s['calls']} | {what} | `{s['op']}` | {s['where']} | "
            f"`{s['file']}:{s['line']}` (`{s['function']}`){code} | {s['host_ms']:.2f} | "
            f"{fix} |"
        )
    if len(sites) > shown:
        rest = sites[shown:]
        lines.append(
            f"| {sum(s['calls'] for s in rest)} | ... | | | {len(rest)} more sites | "
            f"{sum(s['host_ms'] for s in rest):.2f} | |"
        )
    used = dict.fromkeys(KINDS.get(s["kind"], ("", ""))[1] for s in sites[:shown])
    lines += ["", "Fixes:"]
    lines += [f"* **{fix}**: {FIXES[fix]}." for fix in used if fix in FIXES]
    return lines
