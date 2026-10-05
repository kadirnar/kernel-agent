"""Capture a hot module together with real inputs/outputs from the model run.

The capture file is the self-contained unit an agent optimises: it holds the
original module (weights included), up to ``max_cases`` calls with distinct
shape signatures, the reference outputs and the post-call state of the
arguments (so in-place updates such as KV-cache appends are verified too).

Calls are recorded from every entrypoint of the module: ``forward`` and the
methods found by :mod:`kernel_agent.profiling.methods` (e.g. ``forward_step``
of a custom decode loop).  Each case records the ``method`` it came from.

Arguments are deep-copied, with one exception for memory: a tensor that is a
view into a much larger storage (one layer's ``K``/``V`` slice of a static
``[2, layers, B, H, T, D]`` KV-cache buffer, say) is copied compactly — only
the memory its views span, with the same shape, strides and aliasing between
overlapping views.  Deep-copying such a view would copy the whole buffer for
every case and for both the pre- and post-call state.  As a consequence,
side effects are checked on the memory the arguments cover, not on the rest
of a larger buffer they were sliced from.

While recording, every call of the target's instances (not only the saved
cases) feeds :class:`~kernel_agent.profiling.workload_stats.WorkloadStats`,
written to ``workload_profile.md`` / ``.json`` (in the target directory). A
target with a ``phase`` (:mod:`kernel_agent.phases`) records only that phase's
calls as cases; ``qualname_regex`` restricts the instances it covers.
"""

from __future__ import annotations

import collections
import copy
import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent.kernels.compare import flatten
from kernel_agent.phases import PHASES, call_phase
from kernel_agent.profiling.methods import (
    entrypoint,
    entrypoints_of,
    instrument,
    workload_entrypoints,
)
from kernel_agent.profiling.profiler import call_signature, signature_of
from kernel_agent.profiling.workload_stats import WorkloadStats, write_profile
from kernel_agent.workloads.base import Workload, synchronize

#: A storage is copied compactly only when that saves at least this many bytes.
COMPACT_MIN_BYTES = 1 << 20


def _span(t: torch.Tensor) -> int:
    """Number of storage elements ``t`` spans (from its first to its last element)."""
    if t.numel() == 0:
        return 0
    return 1 + sum((int(n) - 1) * int(s) for n, s in zip(t.shape, t.stride(), strict=True))


def _compact_views(value: Any) -> dict[int, torch.Tensor]:
    """``copy.deepcopy`` memo entries that copy views of large storages compactly.

    Tensors are grouped by storage; overlapping views are merged into one
    interval, so aliasing between them survives the copy.  Groups that cover
    most of their storage, mix dtypes or are not plain strided tensors are left
    to the regular deep copy."""
    groups: dict[tuple[str, int], list[torch.Tensor]] = collections.defaultdict(list)
    seen: set[int] = set()
    for t in flatten(value).values():
        if id(t) in seen or not isinstance(t, torch.Tensor):
            continue
        seen.add(id(t))
        if t.layout != torch.strided or t.is_quantized or t.is_conj() or t.is_neg():
            continue
        if t.device.type == "meta" or t.numel() == 0:
            continue
        try:
            groups[(str(t.device), t.untyped_storage().data_ptr())].append(t)
        except (RuntimeError, NotImplementedError):
            continue

    memo: dict[int, torch.Tensor] = {}
    for tensors in groups.values():
        if len({t.dtype for t in tensors}) != 1:
            continue
        intervals = sorted(
            ((int(t.storage_offset()), int(t.storage_offset()) + _span(t), t) for t in tensors),
            key=lambda item: (item[0], item[1]),
        )
        merged: list[tuple[int, int, list[torch.Tensor]]] = []
        for start, end, t in intervals:
            if merged and start < merged[-1][1]:
                first, last, members = merged[-1]
                merged[-1] = (first, max(last, end), [*members, t])
            else:
                merged.append((start, end, [t]))
        size = tensors[0].element_size()
        covered = sum(end - start for start, end, _ in merged) * size
        storage = tensors[0].untyped_storage().nbytes()
        if storage - covered < COMPACT_MIN_BYTES or storage < 2 * covered:
            continue
        for start, end, members in merged:
            chunk = members[0].detach().as_strided((end - start,), (1,), start).clone()
            for t in members:
                memo[id(t)] = chunk.as_strided(t.shape, t.stride(), t.storage_offset() - start)
    return memo


def _detach(value: Any) -> Any:
    """Deep copy of call arguments/outputs (views of large storages copied compactly)."""
    try:
        memo: dict[int, Any] = dict(_compact_views(value))
    except Exception:
        memo = {}
    try:
        return copy.deepcopy(value, memo)
    except Exception:
        return value


class _Recorder:
    """Records the calls of one module instance through all of its entrypoints.

    ``peers`` (other instances of the class) are only watched to count which
    instances call which entrypoint (``callers``). With ``phase`` only calls of
    that phase are recorded and counted. ``stats`` sees every call of the
    module and its peers, whatever the phase."""

    def __init__(
        self,
        module: nn.Module,
        max_cases: int,
        methods: Sequence[str] = (),
        peers: Sequence[nn.Module] = (),
        *,
        phase: str | None = None,
        stats: WorkloadStats | None = None,
    ) -> None:
        self.module = module
        self.max_cases = max_cases
        self.phase = phase if phase in PHASES else None
        self.stats = stats
        self.cases: dict[tuple[str, str], dict[str, Any]] = {}
        #: Calls per entrypoint during the run (all signatures, captured or not).
        self.calls: collections.Counter[str] = collections.Counter()
        #: Entrypoint -> ids of the instances (``module`` and peers) that called it.
        self.callers: dict[str, set[int]] = collections.defaultdict(set)
        self._pending: list[tuple[tuple[str, str], Any, Any] | None] = []
        self._ctx: Any = instrument(
            [module, *peers], {type(module): list(methods)}, self._pre, self._post
        )
        self._ctx.__enter__()

    def _pre(
        self, module: nn.Module, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        phase = call_phase(method, args, kwargs)
        if self.stats is not None:
            self.stats.observe(module, method, args, kwargs, phase)
        wanted = self.phase is None or phase == self.phase
        if wanted:
            self.callers[method].add(id(module))
        if module is not self.module:
            return
        if not wanted:
            self._pending.append(None)
            return
        self.calls[method] += 1
        # Group calls by entrypoint and primary input only: decode steps share
        # it even though masks / cache positions grow every step.
        key = (method, call_signature(method, args, kwargs, limit=1))
        case = self.cases.get(key)
        if case is not None:
            case["count"] += 1
            self._pending.append(None)
        elif self._make_room(method):
            self._pending.append((key, _detach(args), _detach(kwargs)))
        else:
            self._pending.append(None)

    def _make_room(self, method: str) -> bool:
        """Whether a new case of ``method`` may be recorded.

        Every entrypoint keeps at least one case: when the budget is full, an
        entrypoint without a case evicts the least-called case of an entrypoint
        that has several."""
        pending = [p[0] for p in self._pending if p is not None]
        if len(self.cases) + len(pending) < self.max_cases:
            return True
        if method in {m for m, _ in [*self.cases, *pending]}:
            return False
        per_method = collections.Counter(m for m, _ in self.cases)
        victims = [key for key in self.cases if per_method[key[0]] > 1]
        if not victims:
            return False
        del self.cases[min(victims, key=lambda key: self.cases[key]["count"])]
        return True

    def _post(
        self,
        module: nn.Module,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if module is not self.module:
            return
        pending = self._pending.pop() if self._pending else None
        if pending is None:
            return
        key, pre_args, pre_kwargs = pending
        self.cases[key] = {
            "method": method,
            "signature": key[1],
            "full_signature": signature_of(pre_args, pre_kwargs, limit=8),
            "count": 1,
            "args": pre_args,
            "kwargs": pre_kwargs,
            "output": _detach(output),
            "post_args": _detach(args),
            "post_kwargs": _detach(kwargs),
        }

    def remove(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None


def instances_of(
    roots: dict[str, nn.Module], cls: str, qualname_regex: str | None = None
) -> list[tuple[str, nn.Module]]:
    """``(qualname, module)`` of every instance of ``cls`` whose qualname the regex
    matches (``re.search``, as in the patcher), in module order."""
    pattern = re.compile(qualname_regex) if qualname_regex else None
    found: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for root_name, root in roots.items():
        for name, module in root.named_modules():
            full = f"{root_name}.{name}" if name else root_name
            if type(module).__name__ != cls or id(module) in seen:
                continue
            if pattern is None or pattern.search(full):
                seen.add(id(module))
                found.append((full, module))
    return found


def find_instance(
    roots: dict[str, nn.Module],
    cls: str,
    qualname: str | None = None,
    qualname_regex: str | None = None,
) -> tuple[str, nn.Module]:
    for full, module in instances_of(roots, cls, None if qualname else qualname_regex):
        if qualname is None or full == qualname:
            return full, module
    raise LookupError(
        f"no module of class {cls!r} (qualname={qualname!r}, qualname_regex={qualname_regex!r})"
    )


def count_calls(roots: dict[str, nn.Module], cls: str) -> int:
    return sum(1 for root in roots.values() for m in root.modules() if type(m).__name__ == cls)


def _busiest(
    workload: Workload, inputs: Any, candidates: list[tuple[str, nn.Module]], phase: str
) -> tuple[str, nn.Module]:
    """The candidate instance with the most ``phase`` calls in one run (the first on ties)."""
    counts: collections.Counter[int] = collections.Counter()
    extra = workload_entrypoints(workload)
    methods = {t: entrypoints_of(t, extra) for t in {type(m) for _, m in candidates}}

    def pre(module: nn.Module, method: str, args: Any, kwargs: Any) -> None:
        if call_phase(method, args, kwargs) == phase:
            counts[id(module)] += 1

    with (
        instrument([m for _, m in candidates], methods, pre, lambda *_: None),
        torch.inference_mode(),
    ):
        workload.run(inputs)
        synchronize()
    return max(candidates, key=lambda c: counts[id(c[1])])


def capture_module(
    workload: Workload,
    inputs: Any,
    cls: str,
    path: Path,
    *,
    qualname: str | None = None,
    max_cases: int = 4,
    qualname_regex: str | None = None,
    phase: str | None = None,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the workload once and save one instance of ``cls`` plus its calls.

    The target's instances are those a patch replaces (``cls``, matching
    ``qualname_regex``); every call of theirs goes into the workload profile
    (``workload_profile.md`` in ``profile_dir``, default: next to ``path``).
    The captured instance is ``qualname``, else the one with the most
    ``phase`` calls when a phase is given (one extra run), else the first.
    With ``phase`` only that phase's calls become cases."""
    phase = phase if phase in PHASES else None
    roots = workload.roots()
    candidates = instances_of(roots, cls, qualname_regex)
    if qualname is not None:
        full, module = find_instance(roots, cls, qualname)
    elif phase is not None and len(candidates) > 1:
        full, module = _busiest(workload, inputs, candidates, phase)
    elif candidates:
        full, module = candidates[0]
    else:
        raise LookupError(f"no module of class {cls!r} (qualname_regex={qualname_regex!r})")
    methods = entrypoints_of(type(module), workload_entrypoints(workload))
    peers = [m for _, m in candidates if m is not module]
    stats = WorkloadStats()
    recorder = _Recorder(module, max_cases, methods, peers, phase=phase, stats=stats)
    try:
        with torch.inference_mode():
            workload.run(inputs)
            synchronize()
    finally:
        recorder.remove()
    with torch.inference_mode():
        meta = {"class": cls, "qualname": full, "instances": len(candidates) or 1}
        profile = stats.finalize(module, **meta, qualname_regex=qualname_regex, phase=phase)
    write_profile(profile, profile_dir or path.parent)
    if not recorder.cases:
        what = f"{phase} calls" if phase else "calls"
        raise RuntimeError(f"{full} ({cls}) had no {what} during the workload run")
    cases = sorted(recorder.cases.values(), key=lambda c: -c["count"])
    calls = dict(recorder.calls.most_common())
    # Instances that call each entrypoint (VoxCPM: forward_step only on the LM
    # layers, forward on every MiniCPMAttention) – weights the estimated saving.
    method_instances = {m: len(ids) for m, ids in recorder.callers.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module": module,
            "qualname": full,
            "class": cls,
            "module_path": f"{type(module).__module__}.{type(module).__qualname__}",
            "instances": len(candidates) or count_calls(roots, cls),
            "method_instances": method_instances,
            "methods": calls,
            "cases": cases,
            **({"phase": phase} if phase else {}),
        },
        path,
    )
    return {
        "qualname": full,
        # calls per run of this instance, per entrypoint (captured or not)
        "methods": calls,
        "method_instances": method_instances,
        "cases": [
            {"method": c["method"], "signature": c["signature"], "count": c["count"]} for c in cases
        ],
        "bytes": path.stat().st_size,
        **({"phase": phase} if phase else {}),
        # every call of the target's instances; the facts go into the engineer prompt
        "workload": {"calls": profile["calls"], "facts": profile["facts"]},
    }


def load_capture(
    path: Path, device: str | None = None, *, sha256: str | None = None
) -> dict[str, Any]:
    """Load a capture file; with ``sha256`` it is read once and refused (``TamperError``)
    unless it has that digest (``kernel_agent.truth``)."""
    from kernel_agent.truth import read_verified

    data = read_verified(path, sha256) if sha256 is not None else None

    def load() -> dict[str, Any]:
        source = path if data is None else io.BytesIO(data)
        return torch.load(source, map_location=device, weights_only=False)

    try:
        return load()
    except ModuleNotFoundError:
        # Classes from ``trust_remote_code`` repos live in the HF modules cache.
        from transformers.dynamic_module_utils import init_hf_modules

        init_hf_modules()
        return load()


def capture_calls(
    module: nn.Module,
    calls: Sequence[tuple[Any, ...]],
    path: Path,
    *,
    instances: int = 1,
) -> None:
    """Build a capture file from explicit ``(args, kwargs, count[, method])`` calls.

    Used by tests and for synthetic shapes (e.g. other batch sizes) that the
    workload run did not exercise.  ``method`` defaults to ``"forward"``."""
    cases = []
    totals: collections.Counter[str] = collections.Counter()
    with torch.inference_mode():
        for call in calls:
            args, kwargs, count = call[0], call[1], int(call[2])
            method = str(call[3]) if len(call) > 3 else "forward"
            pre_args, pre_kwargs = _detach(args), _detach(kwargs)
            output = entrypoint(module, method)(*args, **kwargs)
            synchronize()
            totals[method] += count
            cases.append(
                {
                    "method": method,
                    "signature": call_signature(method, pre_args, pre_kwargs, limit=1),
                    "full_signature": signature_of(pre_args, pre_kwargs, limit=8),
                    "count": count,
                    "args": pre_args,
                    "kwargs": pre_kwargs,
                    "output": _detach(output),
                    "post_args": _detach(args),
                    "post_kwargs": _detach(kwargs),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module": module,
            "qualname": type(module).__name__,
            "class": type(module).__name__,
            "module_path": f"{type(module).__module__}.{type(module).__qualname__}",
            "instances": instances,
            "methods": dict(totals.most_common()),
            "cases": cases,
        },
        path,
    )
