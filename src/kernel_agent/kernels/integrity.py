"""Guards against a candidate that games the evaluator from inside its process (#7).

The evaluator imports the candidate into its own process, so the candidate can
monkey-patch the timer, the comparator or the reference, flip backend flags to
slow the reference, hide work on side streams or threads, or fall back to the
reference.  The guards (all run by :mod:`kernels.evaluate`):

* :class:`Snapshot`: identities of the timing / synchronisation primitives
  (``torch.cuda.Event``, ``torch.cuda.synchronize``, ``time.perf_counter``), the
  evaluator's own functions and constants (``compare.TOLERANCES``, ...),
  ``torch``, ``torch.Tensor``, ``torch.nn.functional`` and the ``forward`` of
  every ``torch.nn`` module class, every method of the reference's module classes
  and instances (``type(reference).forward``), the reference's weights, and the
  backend flags (TF32, cuDNN, SDPA backends, matmul precision, deterministic
  algorithms, default dtype, current stream, torch function / dispatch modes,
  an active profiler).  Taken before the candidate is imported and verified
  after ``build()``, after the correctness and timing stages and at the end;
  any change is an ``integrity_violation`` (and is undone).  The timer itself
  is bound at import time (:mod:`kernels.bench`), so a patch cannot change a
  measurement even before it is detected.
* :func:`candidate_threads`: threads running code from the candidate's directory.
* :func:`activity_check`: one profiled pass over the dominant ("main") case: GPU
  work launched from another thread, or still running on another stream when
  the timed stream continues (not joined back), is a violation; it also
  measures ``custom_kernel_share`` (GPU time in kernels the reference does not
  launch), the kernel launches per call of both, and counts calls into the
  reference's entrypoint code (:func:`count_calls`, ``sys.monitoring``), which
  decide ``fallback`` (:func:`fallback_reason`).
* Outside the candidate's process (:func:`compare_saved_outputs`,
  :func:`reference_slowdown`): ``run_evaluation`` compares the candidate's saved
  outputs with the capture itself and compares the reference timing with one
  measured in a candidate-free process.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import dataclasses
import inspect
import json
import math
import sys
import sysconfig
import threading
import time
import types
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

#: A reference entrypoint that runs on the main case with less than this share of
#: the candidate's GPU time in kernels of its own is a fallback to the reference.
MIN_CUSTOM_SHARE = 0.5
#: GPU work of a call may end this long after the timed stream moved on (ns).
JOIN_SLACK_NS = 2_000
#: Patched by ``torch._dynamo`` when it is first imported (e.g. by torch.compile).
_ALLOWED_CHANGES = {"torch.manual_seed"}
_WATCHED_DUNDERS = {"__call__", "__torch_function__", "__torch_dispatch__"}
_MISSING = object()


# ------------------------------------------------------------------ snapshot


def _frozen(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _simple(value: Any) -> bool:
    """A constant whose value (not just identity) is watched (tolerances, limits)."""
    if isinstance(value, bool | int | float | str | torch.dtype):
        return True
    if isinstance(value, tuple | frozenset):
        return all(_simple(v) for v in value)
    if isinstance(value, dict):
        return all(_simple(k) and _simple(v) for k, v in value.items())
    return False


def _members(obj: Any) -> dict[str, Any]:
    """Raw attributes (``staticmethod`` objects stay wrapped, so identities are stable)."""
    try:
        return dict(vars(obj))
    except TypeError:  # e.g. torch._C._VariableFunctions
        return {name: getattr(obj, name, None) for name in dir(obj)}


def _member(obj: Any, name: str) -> Any:
    try:
        return vars(obj).get(name, _MISSING)
    except TypeError:
        return getattr(obj, name, _MISSING)


@dataclasses.dataclass
class _Watch:
    """Watched attributes of one object: callables by identity, constants by value."""

    label: str
    obj: Any
    attrs: dict[str, tuple[Any, Any]]  # name -> (identity, frozen value or _MISSING)


def _watch(label: str, obj: Any, *, constants: bool = False) -> _Watch:
    attrs: dict[str, tuple[Any, Any]] = {}
    for name, value in _members(obj).items():
        if name.startswith("__") and name not in _WATCHED_DUNDERS:
            continue
        if callable(value) and not isinstance(value, type | types.ModuleType):
            attrs[name] = (value, _MISSING)
        elif constants and name.isupper() and not name.startswith("_") and _simple(value):
            attrs[name] = (value, _frozen(value))  # public constants, not module state
    return _Watch(label, obj, attrs)


def _flags() -> dict[str, tuple[Callable[[], Any], Callable[[Any], Any] | None]]:
    """Global state the reference's speed or numerics depend on: getter, setter."""
    b = torch.backends
    m = b.cuda.matmul

    def attr(owner: Any, name: str) -> tuple[Callable[[], Any], Callable[[Any], Any]]:
        return (lambda: getattr(owner, name)), (lambda v: setattr(owner, name, v))

    flags: dict[str, tuple[Callable[[], Any], Callable[[Any], Any] | None]] = {
        "torch.backends.cuda.matmul.allow_tf32": attr(m, "allow_tf32"),
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction": attr(
            m, "allow_fp16_reduced_precision_reduction"
        ),
        "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction": attr(
            m, "allow_bf16_reduced_precision_reduction"
        ),
        "torch.backends.cudnn.allow_tf32": attr(b.cudnn, "allow_tf32"),
        "torch.backends.cudnn.enabled": attr(b.cudnn, "enabled"),
        "torch.backends.cudnn.benchmark": attr(b.cudnn, "benchmark"),
        "torch.backends.cudnn.deterministic": attr(b.cudnn, "deterministic"),
        "torch.backends.cuda.flash_sdp_enabled()": (
            b.cuda.flash_sdp_enabled,
            b.cuda.enable_flash_sdp,
        ),
        "torch.backends.cuda.mem_efficient_sdp_enabled()": (
            b.cuda.mem_efficient_sdp_enabled,
            b.cuda.enable_mem_efficient_sdp,
        ),
        "torch.backends.cuda.math_sdp_enabled()": (b.cuda.math_sdp_enabled, b.cuda.enable_math_sdp),
        "torch.backends.cuda.cudnn_sdp_enabled()": (
            b.cuda.cudnn_sdp_enabled,
            b.cuda.enable_cudnn_sdp,
        ),
        "torch.get_float32_matmul_precision()": (
            torch.get_float32_matmul_precision,
            torch.set_float32_matmul_precision,
        ),
        "torch.are_deterministic_algorithms_enabled()": (
            torch.are_deterministic_algorithms_enabled,
            torch.use_deterministic_algorithms,
        ),
        "torch.get_default_dtype()": (torch.get_default_dtype, torch.set_default_dtype),
        "torch function modes": (torch._C._len_torch_function_stack, None),
        "torch dispatch modes": (torch._C._len_torch_dispatch_stack, None),
        "an active torch profiler": (torch._C._autograd._profiler_enabled, None),
    }
    if hasattr(m, "allow_fp16_accumulation"):
        flags["torch.backends.cuda.matmul.allow_fp16_accumulation"] = attr(
            m, "allow_fp16_accumulation"
        )
    if torch.cuda.is_available():
        flags["torch.cuda.current_stream()"] = (torch.cuda.current_stream, torch.cuda.set_stream)
        flags["torch.cuda.current_device()"] = (torch.cuda.current_device, torch.cuda.set_device)
    return flags


def _module_classes(reference: nn.Module) -> list[type]:
    """The reference's own module classes (and their non-torch bases)."""
    found: list[type] = []
    for module in reference.modules():
        for klass in type(module).__mro__:
            if klass in (nn.Module, object) or klass.__module__.startswith("torch."):
                continue
            if klass not in found:
                found.append(klass)
    return found


class Snapshot:
    """What a candidate must not change; see the module docstring."""

    def __init__(self, reference: nn.Module | None = None) -> None:
        import kernel_agent.kernels.bench as bench
        import kernel_agent.kernels.compare as compare
        import kernel_agent.kernels.evaluate as evaluate
        import kernel_agent.kernels.verify as verify
        import kernel_agent.profiling.methods as methods
        import kernel_agent.workloads.base as base

        self.watches = [
            _watch("time", time),
            _watch("json", json),
            _watch("torch", torch),
            _watch("torch.Tensor", torch.Tensor),
            _watch("torch.nn.functional", torch.nn.functional),
            _watch("torch._C._nn", torch._C._nn),
            _watch("torch._C._VariableFunctions", torch._C._VariableFunctions),
            _watch("torch.nn.Module", nn.Module),
            _watch("torch.cuda", torch.cuda),
            _watch("torch.cuda.Event", torch.cuda.Event),
            _watch("torch.cuda.Stream", torch.cuda.Stream),
            _watch("kernel_agent.kernels.compare", compare, constants=True),
            _watch("kernel_agent.kernels.bench", bench, constants=True),
            _watch("kernel_agent.kernels.verify", verify, constants=True),
            _watch("kernel_agent.kernels.evaluate", evaluate, constants=True),
            _watch("kernel_agent.kernels.integrity", sys.modules[__name__], constants=True),
            _watch("kernel_agent.profiling.methods", methods),
            _watch("kernel_agent.workloads.base", base),
        ]
        for name in dir(nn):
            klass = getattr(nn, name)
            if isinstance(klass, type) and issubclass(klass, nn.Module):
                self.watches.append(_watch(f"torch.nn.{name}", klass))
        self.instances: list[tuple[str, nn.Module, dict[str, Any]]] = []
        self.weights: list[tuple[str, torch.Tensor, int, int]] = []
        if reference is not None:
            for klass in _module_classes(reference):
                self.watches.append(_watch(f"{klass.__module__}.{klass.__qualname__}", klass))
            for name, module in reference.named_modules():
                label = f"reference.{name}" if name else "reference"
                self.instances.append((label, module, dict(vars(module))))
            for name, t in [*reference.named_parameters(), *reference.named_buffers()]:
                self.weights.append((name, t, t._version, t.data_ptr()))
        self.flags = _flags()
        self.flag_values = {name: get() for name, (get, _) in self.flags.items()}

    def changes(self) -> list[str]:
        """Human-readable list of what changed since the snapshot."""
        found = []
        for watch in self.watches:
            for name, (value, frozen) in watch.attrs.items():
                label = f"{watch.label}.{name}"
                if label in _ALLOWED_CHANGES:
                    continue
                now = _member(watch.obj, name)
                if frozen is _MISSING:
                    if now is not value:
                        owner = getattr(now, "__module__", None) or type(now).__name__
                        found.append(f"{label} was replaced (by {owner})")
                elif now is not value or now != frozen:
                    found.append(f"{label} changed: {frozen!r} -> {now!r}"[:300])
        for label, module, attrs in self.instances:
            current = vars(module)
            for name, value in current.items():
                if name.startswith("__"):
                    continue
                if name not in attrs:
                    if callable(value):
                        found.append(f"{label}.{name} was added (patches the reference)")
                elif callable(attrs[name]) and value is not attrs[name]:
                    found.append(f"{label}.{name} was replaced (patches the reference)")
        for name, t, version, ptr in self.weights:
            if t._version != version or t.data_ptr() != ptr:
                found.append(f"reference weight {name} was modified")
        for name, value in self.flag_values.items():
            try:
                now = self.flags[name][0]()
            except Exception as exc:
                now = f"unreadable ({exc})"
            if now != value:
                found.append(f"{name} changed: {value} -> {now}")
        return found

    def restore(self) -> None:
        """Undo the changes (best effort), so later code and other evaluations are clean."""
        for watch in self.watches:
            for name, (value, frozen) in watch.attrs.items():
                with contextlib.suppress(Exception):
                    if _member(watch.obj, name) is not value:
                        setattr(watch.obj, name, value)
                    elif frozen is not _MISSING and value != frozen and isinstance(value, dict):
                        value.clear()
                        value.update(frozen)
        for _, module, attrs in self.instances:
            current = vars(module)
            for name in [n for n in current if n not in attrs and callable(current[n])]:
                del current[name]
            for name, value in attrs.items():
                if callable(value):
                    current[name] = value
        for name, value in self.flag_values.items():
            get, put = self.flags[name]
            with contextlib.suppress(Exception):
                if put is not None and get() != value:
                    put(value)
        with contextlib.suppress(Exception):
            while torch._C._len_torch_function_stack() > self.flag_values["torch function modes"]:
                torch._C._pop_torch_function_stack()


# ------------------------------------------------------------------ threads and code


def candidate_threads(directory: Path) -> list[str]:
    """Live threads whose code (target or ``run``) lives under ``directory``."""
    root = str(directory.resolve())
    found = []
    for thread in threading.enumerate():
        if thread is threading.main_thread():
            continue
        fn: Any = getattr(thread, "_target", None)
        if fn is None and type(thread).run is not threading.Thread.run:
            fn = type(thread).run
        fn = getattr(fn, "__func__", fn)
        with contextlib.suppress(Exception):
            fn = inspect.unwrap(fn)
        code = getattr(fn, "__code__", None)
        if code is not None and str(Path(code.co_filename).resolve()).startswith(root):
            found.append(f"{thread.name} ({code.co_name} in {Path(code.co_filename).name})")
    return found


def _library_file(filename: str) -> bool:
    torch_dir = str(Path(torch.__file__).parent)
    return filename.startswith((torch_dir, sysconfig.get_paths()["stdlib"]))


def reference_codes(reference: nn.Module, methods: Iterable[str]) -> dict[Any, str]:
    """Code objects of the reference class's entrypoints (``forward`` and ``methods``),
    unwrapped from decorators, with a label each."""
    codes: dict[Any, str] = {}
    cls = type(reference)
    for name in dict.fromkeys(["forward", *methods]):
        fn = getattr(cls, name, None)
        if fn is None:
            continue
        with contextlib.suppress(Exception):
            fn = inspect.unwrap(fn)
        code = getattr(fn, "__code__", None)
        if code is not None and not _library_file(code.co_filename):
            codes[code] = f"{cls.__name__}.{name}"
    return codes


@contextlib.contextmanager
def count_calls(codes: Iterable[Any]) -> Iterator[collections.Counter[Any]]:
    """Count starts of the given code objects in any thread (``sys.monitoring``)."""
    counts: collections.Counter[Any] = collections.Counter()
    codes = list(codes)
    monitoring = getattr(sys, "monitoring", None)
    tool = None
    if monitoring is not None and codes:
        tool = next((i for i in (4, 3) if monitoring.get_tool(i) is None), None)
    if tool is None:
        yield counts
        return
    assert monitoring is not None
    start = monitoring.events.PY_START

    def on_start(code: Any, offset: int) -> None:
        counts[code] += 1

    monitoring.use_tool_id(tool, "kernel-agent")
    try:
        monitoring.register_callback(tool, start, on_start)
        for code in codes:
            monitoring.set_local_events(tool, code, start)
        yield counts
    finally:
        for code in codes:
            with contextlib.suppress(Exception):
                monitoring.set_local_events(tool, code, 0)
        monitoring.register_callback(tool, start, None)
        monitoring.free_tool_id(tool)


# ------------------------------------------------------------------ activity pass


def main_case(cases: list[dict[str, Any]], reports: list[dict[str, Any]]) -> int:
    """Index of the dominant case: largest calls per run x reference time."""

    def weight(i: int) -> float:
        ref_ms = reports[i].get("ref_ms") if i < len(reports) else None
        return float(cases[i].get("count", 1)) * float(ref_ms or 1.0)

    return max(range(len(cases)), key=weight)


@dataclasses.dataclass
class _Event:
    kind: str  # Kineto activity type: kernel, gpu_memcpy, cuda_runtime, user_annotation, ...
    name: str
    start: int
    end: int
    corr: int
    resource: int  # GPU work: stream; CPU events: thread


_GPU_WORK = {"kernel", "gpu_memcpy", "gpu_memset"}
_API_CALLS = {"cuda_runtime", "cuda_driver"}


def _events(prof: Any) -> list[_Event]:
    out = []
    for e in prof.profiler.kineto_results.events():
        start = int(e.start_ns())
        out.append(
            _Event(
                str(e.activity_type()),
                e.name(),
                start,
                start + int(e.duration_ns()),
                int(e.correlation_id()),
                int(e.device_resource_id()),
            )
        )
    return out


def activity_check(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    inputs: list[tuple[Any, Any]],
    codes: dict[Any, str],
    *,
    settle_s: float = 0.02,
) -> dict[str, Any]:
    """Profile one reference call and one candidate call per entry of ``inputs`` (fresh
    copies of the main case) and report:

    * ``foreign_threads``: GPU work the candidate's calls caused that was launched
      from a thread other than the evaluator's;
    * ``unjoined``: GPU work launched during a candidate call that was still
      running when the timed (current) stream moved on, i.e. on another stream that
      the timed stream never waited for;
    * ``custom_kernel_share``: share of the candidate's kernel time in kernels whose
      names the reference's call does not launch (None without kernels);
    * ``reference_calls``: starts of the reference's entrypoint code during the
      candidate's calls (``{label: count}``);
    * ``reference_kernels`` / ``candidate_kernels``: kernel launches per call by name
      (:func:`fallback_reason`).
    """
    from torch.autograd.profiler import record_function
    from torch.profiler import ProfilerActivity, profile

    out: dict[str, Any] = {"foreign_threads": [], "unjoined": [], "reference_calls": {}}
    ref_args, ref_kwargs = inputs[0]
    ref_args, ref_kwargs = copy.deepcopy(ref_args), copy.deepcopy(ref_kwargs)
    torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities) as prof, torch.inference_mode():
        with record_function("ka::reference"):
            reference(*ref_args, **ref_kwargs)
        torch.cuda.synchronize()
        with count_calls(codes) as counts:
            for i, (args, kwargs) in enumerate(inputs):
                with record_function(f"ka::call{i}"):
                    candidate(*args, **kwargs)
                with record_function(f"ka::mark{i}"):
                    torch.cuda._sleep(1)  # the timed stream moves on
                torch.cuda.synchronize()
            time.sleep(settle_s)  # late launches from other threads
            torch.cuda.synchronize()
    out["reference_calls"] = {codes[c]: n for c, n in counts.items() if n}

    events = _events(prof)
    ranges = {e.name: e for e in events if e.kind == "user_annotation" and e.name[:4] == "ka::"}
    gpu: dict[int, list[_Event]] = collections.defaultdict(list)  # a graph launch: several
    for e in events:
        if e.kind in _GPU_WORK:
            gpu[e.corr].append(e)
    launches = [e for e in events if e.kind in _API_CALLS and e.corr in gpu]

    def within(name: str) -> list[_Event]:
        span = ranges.get(name)
        return [] if span is None else [e for e in launches if span.start <= e.start <= span.end]

    marks = [within(f"ka::mark{i}") for i in range(len(inputs))]
    if not all(marks) or not gpu:
        out["note"] = "the profiler recorded no GPU activity; activity checks skipped"
        return out
    main_thread = marks[0][0].resource
    ref_kernels: collections.Counter[str] = collections.Counter(
        w.name for e in within("ka::reference") for w in gpu[e.corr] if w.kind == "kernel"
    )
    new_kernels: collections.Counter[str] = collections.Counter()
    own = total = 0
    for i in range(len(inputs)):
        mark = min((w.start for w in gpu[marks[i][0].corr]), default=None)
        for launch in within(f"ka::call{i}"):
            if launch.resource != main_thread:
                continue
            for work in gpu[launch.corr]:
                if mark is not None and work.end > mark + JOIN_SLACK_NS:
                    out["unjoined"].append(
                        f"{work.name[:80]} (stream {work.resource}) ran "
                        f"{(work.end - mark) / 1e3:.0f} us past the end of call {i}"
                    )
                if work.kind == "kernel":
                    new_kernels[work.name] += 1
                    total += work.end - work.start
                    own += (work.end - work.start) * (work.name not in ref_kernels)
    first = ranges.get("ka::call0")
    for launch in launches:
        if launch.resource == main_thread or first is None or launch.start < first.start:
            continue
        label = gpu[launch.corr][0].name[:80]
        out["foreign_threads"].append(f"{label} launched from thread {launch.resource}")
    out["custom_kernel_share"] = round(own / total, 3) if total else None
    # kernel launches per call, by name (the candidate's averaged over its calls)
    out["reference_kernels"] = dict(ref_kernels)
    out["candidate_kernels"] = {k: n / len(inputs) for k, n in new_kernels.items()}
    for key in ("unjoined", "foreign_threads"):
        out[key] = sorted(set(out[key]))[:5]
    return out


def fallback_reason(
    ran: dict[str, int],
    share: float | None,
    ref_kernels: dict[str, float] | None = None,
    new_kernels: dict[str, float] | None = None,
) -> str | None:
    """Why the candidate's call of the dominant case falls back to the reference (None:
    it does not).

    * The reference's entrypoint code ran (``ran``) and less than
      :data:`MIN_CUSTOM_SHARE` of the GPU time is in kernels of its own (``share``;
      None: not profiled, the code alone decides).
    * Or none of it is (``share`` 0) and the candidate launches every kernel of the
      reference at least as often (the same multiset of kernel names or a superset):
      it re-runs the reference's ops.  Fewer launches of some kernel is a genuine
      restructuring (merged QKV or gate/up projections: one GEMM instead of three or
      two; dropped casts and copies) and is allowed.
    """
    if ran and (share is None or share < MIN_CUSTOM_SHARE):
        why = "runs the reference's " + ", ".join(f"`{k}`" for k in sorted(ran))
        if share is not None:
            why += f" and spends {share:.0%} of its GPU time in kernels of its own"
        return why
    ref, new = ref_kernels or {}, new_kernels or {}
    if share == 0 and ref and all(new.get(k, 0) >= n for k, n in ref.items()):
        return (
            f"launches only kernels the reference launches, each at least as often "
            f"({sum(new.values()):g} launches per call vs {sum(ref.values()):g}): it re-runs "
            "the reference's ops"
        )
    return None


# ------------------------------------------------------------------ outside the candidate


def flat_outputs(case_outputs: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    """What the evaluator saves of the candidate's correctness calls: per case the
    flattened output and post-call arguments, plain CPU tensors only."""
    from kernel_agent.kernels.compare import flatten, type_error

    saved = []
    for item in case_outputs:
        entry: dict[str, dict[str, Any]] = {}
        for key, value in item.items():
            entry[key] = {
                name: t.detach().cpu().clone()
                for name, t in flatten(value, key).items()
                if type_error(t) is None
            }
        saved.append(entry)
    return saved


def compare_saved_outputs(capture: dict[str, Any], saved: Any) -> list[dict[str, Any]]:
    """Failed checks of the saved candidate outputs (``{"cases": capture indices,
    "outputs": flat_outputs(...)}``) against the capture: the same checks as the
    evaluator's correctness stage, run outside the candidate's process."""
    from kernel_agent.kernels.compare import (
        compare_side_effects_flat,
        compare_tensors,
        flatten,
        tier_of,
    )

    failures: list[dict[str, Any]] = []
    cases = capture["cases"]
    tier = tier_of(capture)
    indices = saved.get("cases") if isinstance(saved, dict) else None
    outputs = saved.get("outputs") if isinstance(saved, dict) else None
    if (
        not isinstance(indices, list)
        or not isinstance(outputs, list)
        or len(indices) != len(outputs)
        or not indices
        or any(not isinstance(i, int) or not 0 <= i < len(cases) for i in indices)
    ):
        return [{"case": None, "error": "saved outputs do not match the captured cases"}]
    for i, new in zip(indices, outputs, strict=True):
        case = cases[i]
        checks = []
        new_out = new.get("output", {})
        for name, ref in flatten(case["output"], "output").items():
            if name not in new_out:
                checks.append({"name": name, "ok": False, "error": "missing in saved outputs"})
            else:
                checks.append(compare_tensors(name, ref, new_out[name], tier=tier))
        for key in ("args", "kwargs"):
            checks += compare_side_effects_flat(
                flatten(case[key], key),
                flatten(case[f"post_{key}"], key),
                new.get(key, {}),
                tier=tier,
            )
        failures += [{"case": i, **c} for c in checks if not c.get("ok")]
    return failures


def reference_slowdown(
    result: dict[str, Any], clean: list[list[float]], *, tolerance: float = 0.10
) -> list[str]:
    """Cases whose reference time next to the candidate (``ref_ms``) exceeds the
    candidate-free one (``[ms, instability]`` per case) by more than ``tolerance``, twice
    the case's timing spread, twice the candidate-free instability, and 2 us."""
    slow = []
    for i, (case, (ms, unstable)) in enumerate(zip(result.get("cases") or [], clean, strict=False)):
        ref = case.get("ref_ms")
        if not ref or not ms:
            continue
        spread = float(case.get("timing_spread") or 0)
        allowed = ms * (1 + max(tolerance, 2 * spread, 2 * unstable))
        if ref > allowed + 0.002:
            slow.append(
                f"case {i} ({case.get('signature')}): reference {ref:.4f} ms in the "
                f"candidate's process vs {ms:.4f} ms without it ({ref / ms:.2f}x)"
            )
    return slow


def describe_share(share: float | None) -> str:
    return "n/a" if share is None or math.isnan(share) else f"{share:.0%}"
