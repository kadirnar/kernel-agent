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
calls as cases; ``qualname_regex`` restricts the instances it covers. Their calls
are also counted per instance and primary input: each case's ``target_calls``
(the calls of every instance with its primary input it stands for) and the
``instance_groups`` weight the estimated saving (:mod:`kernel_agent.kernels.weights`).

Correctness coverage beyond one call per signature:

* KV-length buckets: decode steps share their primary input while the KV
  cache grows, so a survey run counts them first and the recording keeps the
  first, middle and last step of every decode signature (``bucket``,
  ``decode_step``); each stands for a third of the calls in the timing weights.
* Extra settings: ``Workload.variants()`` (other prompt lengths, batch sizes,
  texts) are run once more each; calls with primary inputs the main run lacks
  become *correctness-only* cases (``count`` 0, ``correctness_only``) that the
  evaluator checks but does not time.

Module state outside the arguments (a KV-cache attribute, a step counter: issue #162,
:mod:`kernel_agent.profiling.state`): the module is saved with its state at the end of the
run, so every case also keeps the state its call saw (``state``, a diff against the saved
module) and what its call changed (``post_state``); the evaluator restores a case's state
before each call. Self-check (:func:`self_check`): the saved capture is replayed through the
evaluator's correctness flow with the reference; a capture the reference fails is refused
(:class:`UnverifiableCapture`, the file removed) with the reason, so no agent ever gets a
target nothing can pass.
"""

from __future__ import annotations

import collections
import copy
import io
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent.kernels.compare import flatten
from kernel_agent.phases import PHASES, call_phase
from kernel_agent.profiling import state
from kernel_agent.profiling.methods import (
    entrypoint,
    entrypoints_of,
    instrument,
    workload_entrypoints,
)
from kernel_agent.profiling.profiler import call_signature, signature_of
from kernel_agent.profiling.workload_stats import WorkloadStats, write_profile
from kernel_agent.projection import fold
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


def bucket_plan(n: int) -> list[tuple[str, int, int]]:
    """``(bucket, call index, calls it stands for)`` of ``n`` decode calls that share
    a primary input: the first, middle and last step (thirds of the calls each), so
    the KV cache is checked short, half full and full, and timings weighted by
    third.  One call: no buckets."""
    if n < 2:
        return []
    if n == 2:
        return [("first", 0, 1), ("last", 1, 1)]
    third = n // 3
    return [("first", 0, third), ("middle", n // 2, n - 2 * third), ("last", n - 1, third)]


class _Recorder:
    """Records the calls of one module instance through all of its entrypoints.

    ``peers`` (other instances of the class) are only watched to count which
    instances call which entrypoint (``callers``). With ``phase`` only calls of
    that phase are recorded and counted. ``stats`` sees every call of the
    module and its peers, whatever the phase.

    ``steps`` (decode calls per ``(method, primary signature)`` in one run, from a
    survey run) splits decode calls into KV-length buckets (:func:`bucket_plan`):
    besides its first call, a decode signature keeps its middle and last call as
    cases. Those extra cases do not count against ``max_cases``. Keys in ``skip``
    are never recorded."""

    def __init__(
        self,
        module: nn.Module,
        max_cases: int,
        methods: Sequence[str] = (),
        peers: Sequence[nn.Module] = (),
        *,
        phase: str | None = None,
        stats: WorkloadStats | None = None,
        steps: dict[tuple[str, str], int] | None = None,
        skip: set[tuple[str, str]] | None = None,
    ) -> None:
        self.module = module
        self.max_cases = max_cases
        self.phase = phase if phase in PHASES else None
        self.stats = stats
        self.steps = steps or {}
        self.skip = skip or set()
        self.cases: dict[tuple[str, str], dict[str, Any]] = {}
        #: Middle / last decode-step cases of a bucketed key (see ``steps``).
        self.extra: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._index: collections.Counter[tuple[str, str]] = collections.Counter()
        #: Calls per entrypoint during the run (all signatures, captured or not).
        self.calls: collections.Counter[str] = collections.Counter()
        #: Entrypoint -> ids of the instances (``module`` and peers) that called it.
        self.callers: dict[str, set[int]] = collections.defaultdict(set)
        #: (entrypoint, primary input) -> calls per instance id (``module`` and peers): the
        #: calls of the target's instances each case stands for (``kernels/weights.py``).
        self.instance_calls: dict[tuple[str, str], collections.Counter[int]] = (
            collections.defaultdict(collections.Counter)
        )
        #: Per open call: (key, (bucket, decode step) or None, args, kwargs, index of the
        #: module-state snapshot before it), or None.
        self._pending: list[
            tuple[tuple[str, str], tuple[str, int] | None, Any, Any, int | None] | None
        ] = []
        #: The module's state around every recorded call (``profiling/state.py``).
        self.log = state.Log(module)
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
        # Group calls by entrypoint and primary input only: decode steps share
        # it even though masks / cache positions grow every step.
        key = (method, call_signature(method, args, kwargs, limit=1)) if wanted else None
        if key is not None:
            self.callers[method].add(id(module))
            self.instance_calls[key][id(module)] += 1
        if module is not self.module:
            return
        if key is None:
            self._pending.append(None)
            return
        self.calls[method] += 1
        index = self._index[key]
        self._index[key] += 1
        plan = bucket_plan(self.steps.get(key, 0)) if phase == "decode" else []
        bucket = next(((name, index) for name, i, _ in plan if i == index), None)
        case = self.cases.get(key)
        if case is not None:
            case["count"] += 1
        if key in self.skip or (case is not None and bucket is None):
            self._pending.append(None)
        elif case is not None or self._make_room(method):
            # a new case, or the middle / last step of a bucketed decode signature
            pre_args, pre_kwargs = _detach(args), _detach(kwargs)
            self._pending.append((key, bucket, pre_args, pre_kwargs, self.log.take()))
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
        victim = min(victims, key=lambda key: self.cases[key]["count"])
        del self.cases[victim]
        self.extra.pop(victim, None)
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
        key, bucket, pre_args, pre_kwargs, before = pending
        case: dict[str, Any] = {
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
        state.record(case, self.log, before, self.log.take())
        if bucket is not None:
            case["bucket"], case["decode_step"] = bucket
        if key in self.cases:  # a middle / last decode step
            case["count"] = 0
            self.extra.setdefault(key, []).append(case)
        else:
            self.cases[key] = case

    def remove(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None

    def recorded(self) -> list[dict[str, Any]]:
        """The cases, most-called signature first, each followed by its middle / last
        decode-step cases; bucketed cases share their signature's calls by bucket.

        ``target_calls``: the calls of every instance with the case's entrypoint and
        primary input that it stands for (its ``count`` × theirs ÷ the captured
        instance's), the weight of its gain in the estimated saving (``kernels/weights.py``)."""
        out: list[dict[str, Any]] = []
        for key, case in sorted(self.cases.items(), key=lambda kv: -kv[1]["count"]):
            members = [case, *self.extra.get(key, [])]
            if any("bucket" in c for c in members):
                _split_calls(members, total=case["count"])
            per = self.instance_calls.get(key) or collections.Counter()
            scale = sum(per.values()) / max(per[id(self.module)], 1)
            for member in members:
                member["target_calls"] = round(member["count"] * scale, 3)
            out += members
        return out

    def instance_groups(self, qualnames: dict[int, str]) -> dict[str, dict[str, int]]:
        """Instance group (qualname with layer indices folded) → its ``instances``, their
        ``calls`` and the calls with a primary input a case has (``covered``)."""
        groups: dict[str, dict[str, int]] = {}
        for full in qualnames.values():
            group = groups.setdefault(fold(full), {"instances": 0, "calls": 0, "covered": 0})
            group["instances"] += 1
        for key, per in self.instance_calls.items():
            for mid, n in per.items():
                group = groups.setdefault(
                    fold(qualnames.get(mid, "")), {"instances": 0, "calls": 0, "covered": 0}
                )
                group["calls"] += n
                group["covered"] += n if key in self.cases else 0
        return groups


def _split_calls(members: list[dict[str, Any]], total: int) -> None:
    """Counts and signatures of the bucket cases of one decode signature.

    The ``total`` calls are split as :func:`bucket_plan` says; a bucket without a
    case (the run made other calls than the survey run) adds its calls to the
    closest earlier case, so the counts still add up to ``total``."""
    members.sort(key=lambda c: c.get("decode_step", 0))
    for case in members:
        case["count"] = 0
    for _, index, count in bucket_plan(total) or [("first", 0, total)]:
        owner = [c for c in members if c.get("decode_step", 0) <= index] or members
        owner[-1]["count"] += count
    for case in members:
        step = case.get("decode_step", 0) + 1
        case["decode_steps"] = total
        case["signature"] = f"{case['signature']} @ decode step {step}/{total}"


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


#: A survey count: ``(id(instance), method, primary-input signature, phase)``.
SurveyKey = tuple[int, str, str, str]


def _survey(
    workload: Workload, inputs: Any, modules: Sequence[nn.Module]
) -> collections.Counter[SurveyKey]:
    """Calls per instance, entrypoint, primary input and phase in one run."""
    counts: collections.Counter[SurveyKey] = collections.Counter()
    extra = workload_entrypoints(workload)
    methods = {t: entrypoints_of(t, extra) for t in {type(m) for m in modules}}

    def pre(module: nn.Module, method: str, args: Any, kwargs: Any) -> None:
        signature = call_signature(method, args, kwargs, limit=1)
        counts[(id(module), method, signature, call_phase(method, args, kwargs))] += 1

    with instrument(modules, methods, pre, lambda *_: None), torch.inference_mode():
        workload.run(inputs)
        synchronize()
    return counts


def _record_variant(
    workload: Workload,
    module: nn.Module,
    methods: Sequence[str],
    overrides: dict[str, Any],
    known: set[tuple[str, str]],
    *,
    max_cases: int,
    phase: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cases of ``module`` with primary inputs not in ``known`` from one run under the
    option ``overrides`` (``Workload.variants``): correctness-only, ``count`` 0."""
    label = ", ".join(f"{k}={v}" for k, v in overrides.items())
    info: dict[str, Any] = {"options": dict(overrides)}
    recorder: _Recorder | None = None
    try:
        with workload.with_options(overrides):
            inputs = workload.make_inputs()
            recorder = _Recorder(module, max_cases, methods, phase=phase, skip=known)
            with torch.inference_mode():
                workload.run(inputs)
                synchronize()
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return [], info
    finally:
        if recorder is not None:
            recorder.remove()
    known |= set(recorder.cases)
    cases = recorder.recorded()
    for case in cases:
        case.update(
            count=0,
            target_calls=0,
            correctness_only=True,
            variant=label,
            signature=f"{case['signature']} [correctness only: {label}]",
        )
    info["cases"] = len(cases)
    return cases, info


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
    decode_buckets: bool = True,
    variants: Sequence[dict[str, Any]] = (),
    variant_cases: int = 2,
    tier: str | None = None,
    precision: str | None = None,
    self_check: bool = True,
) -> dict[str, Any]:
    """Run the workload and save one instance of ``cls`` plus its calls (``tier``: the
    tolerance tier the evaluator applies, :mod:`kernel_agent.kernels.compare`;
    ``precision``: the reduced precision the target may use, e.g. ``fp8_weights``, for the
    speed of light of :mod:`kernel_agent.kernels.roofline`). With ``self_check`` the saved
    capture is refused (:class:`UnverifiableCapture`) when the reference fails it
    (:func:`self_check`).

    The target's instances are those a patch replaces (``cls``, matching
    ``qualname_regex``); every call of theirs goes into the workload profile
    (``workload_profile.md`` in ``profile_dir``, default: next to ``path``).
    The captured instance is ``qualname``, else the one with the most
    ``phase`` calls when a phase is given, else the first.
    With ``phase`` only that phase's calls become cases.

    A survey run counts the calls first (unless ``phase`` is ``prefill`` and no
    instance has to be picked): with ``decode_buckets`` every decode signature
    keeps its first, middle and last step (:func:`bucket_plan`), so caching bugs
    at longer KV lengths are caught and timings are weighted per bucket.  Each
    of ``variants`` (option overrides, ``Workload.variants``) is one more run
    that adds up to ``variant_cases`` cases with primary inputs the main run
    lacks, as correctness-only cases (``count`` 0: checked, not timed)."""
    phase = phase if phase in PHASES else None
    roots = workload.roots()
    candidates = instances_of(roots, cls, qualname_regex)
    chosen = find_instance(roots, cls, qualname) if qualname is not None else None
    if chosen is None and not candidates:
        raise LookupError(f"no module of class {cls!r} (qualname_regex={qualname_regex!r})")
    busiest = chosen is None and phase is not None and len(candidates) > 1
    survey: collections.Counter[SurveyKey] = collections.Counter()
    if busiest or (decode_buckets and phase != "prefill"):
        modules = [m for _, m in candidates]
        if chosen is not None and all(m is not chosen[1] for m in modules):
            modules.append(chosen[1])
        survey = _survey(workload, inputs, modules)
    if chosen is None and busiest:
        per_instance: collections.Counter[int] = collections.Counter()
        for (mid, _, _, call), n in survey.items():
            per_instance[mid] += n if call == phase else 0
        chosen = max(candidates, key=lambda c: per_instance[id(c[1])])  # first on ties
    full, module = chosen or candidates[0]
    steps = {
        (method, sig): n
        for (mid, method, sig, call), n in survey.items()
        if mid == id(module) and call == "decode" and decode_buckets
    }
    methods = entrypoints_of(type(module), workload_entrypoints(workload))
    peers = [m for _, m in candidates if m is not module]
    stats = WorkloadStats()
    recorder = _Recorder(module, max_cases, methods, peers, phase=phase, stats=stats, steps=steps)
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
    cases = recorder.recorded()
    known = set(recorder.cases)
    variant_info = []
    for overrides in variants:
        found, info = _record_variant(
            workload, module, methods, overrides, known, max_cases=variant_cases, phase=phase
        )
        cases += found
        variant_info.append(info)
    calls = dict(recorder.calls.most_common())
    # Instances that call each entrypoint (VoxCPM: forward_step only on the LM
    # layers, forward on every MiniCPMAttention) – weights the estimated saving.
    method_instances = {m: len(ids) for m, ids in recorder.callers.items()}
    # The calls of every instance group, and those a case stands for (target_calls): the
    # weights of the estimated saving (kernels/weights.py).
    qualnames = {id(m): q for q, m in [*candidates, (full, module)]}
    instance_groups = recorder.instance_groups(qualnames)
    # the state each case saw, as diffs against the module as saved (profiling/state.py)
    tracked = state.finalize(cases, module)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module": module,
            "qualname": full,
            "class": cls,
            "module_path": f"{type(module).__module__}.{type(module).__qualname__}",
            "instances": len(candidates) or count_calls(roots, cls),
            "method_instances": method_instances,
            "instance_groups": instance_groups,
            "methods": calls,
            "cases": cases,
            **({"state": tracked} if tracked else {}),
            **({"phase": phase} if phase else {}),
            **({"tier": tier} if tier else {}),
            **({"precision": precision} if precision else {}),
        },
        path,
    )
    checked = refuse_unverifiable(path, tracked) if self_check else None
    return {
        "qualname": full,
        # calls per run of this instance, per entrypoint (captured or not)
        "methods": calls,
        "method_instances": method_instances,
        "instance_groups": instance_groups,
        "cases": [
            {
                "method": c["method"],
                "signature": c["signature"],
                "count": c["count"],
                "target_calls": c["target_calls"],
                **{k: c[k] for k in ("bucket", "correctness_only") if k in c},
            }
            for c in cases
        ],
        "bytes": path.stat().st_size,
        **({"state": tracked} if tracked else {}),
        **({"self_check": checked} if checked else {}),
        **({"phase": phase} if phase else {}),
        **({"tier": tier} if tier else {}),
        **({"precision": precision} if precision else {}),
        **({"variants": variant_info} if variant_info else {}),
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
    tier: str | None = None,
    precision: str | None = None,
    self_check: bool = False,
) -> None:
    """Build a capture file from explicit ``(args, kwargs, count[, method])`` calls.

    Used by tests and for synthetic shapes (e.g. other batch sizes) that the
    workload run did not exercise.  ``method`` defaults to ``"forward"``; ``tier``,
    ``precision`` and ``self_check`` as in :func:`capture_module` (the module's state
    is recorded per case the same way)."""
    cases = []
    totals: collections.Counter[str] = collections.Counter()
    log = state.Log(module)
    with torch.inference_mode():
        for call in calls:
            args, kwargs, count = call[0], call[1], int(call[2])
            method = str(call[3]) if len(call) > 3 else "forward"
            pre_args, pre_kwargs = _detach(args), _detach(kwargs)
            before = log.take()
            output = entrypoint(module, method)(*args, **kwargs)
            synchronize()
            totals[method] += count
            case = {
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
            state.record(case, log, before, log.take())
            cases.append(case)
    tracked = state.finalize(cases, module)
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
            **({"state": tracked} if tracked else {}),
            **({"tier": tier} if tier else {}),
            **({"precision": precision} if precision else {}),
        },
        path,
    )
    if self_check:
        refuse_unverifiable(path, tracked)


class UnverifiableCapture(RuntimeError):
    """The unmodified reference fails its own capture (:func:`self_check`): refused."""


def self_check(path: Path, device: str | None = None) -> dict[str, Any]:
    """The reference through the evaluator's correctness flow on the saved capture: every
    case in order, its module state restored first (:class:`state.Replay`), its outputs,
    in-place argument updates and state changes compared with the recorded ones in the
    capture's tolerance tier (:mod:`kernel_agent.kernels.compare`). Returns ``ok``,
    ``cases``, ``failures`` (per failing case: ``case``, ``signature`` and the failed
    checks or the ``error`` it raised) and ``seconds``."""
    from kernel_agent.kernels.compare import compare_side_effects, compare_structures, tier_of

    t0 = time.perf_counter()
    capture = load_capture(path, device=device)
    reference = capture["module"].eval()
    replay = state.Replay(capture, reference)  # its state as saved: before any call
    tier = tier_of(capture)
    failures: list[dict[str, Any]] = []
    for i, case in enumerate(capture["cases"]):
        args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
        where = {"case": i, "signature": case.get("signature")}
        try:
            with torch.inference_mode():
                out = replay.call(case, reference)(*args, **kwargs)
            synchronize()
        except Exception as exc:
            failures.append({**where, "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        inputs = (case["args"], case["kwargs"])
        checks = compare_structures(case["output"], out, "output", inputs=inputs, tier=tier)
        checks += compare_side_effects(case["args"], case["post_args"], args, "args", tier=tier)
        checks += compare_side_effects(
            case["kwargs"], case["post_kwargs"], kwargs, "kwargs", tier=tier
        )
        checks += replay.check(case, reference, tier=tier)
        if bad := [c for c in checks if not c.get("ok")]:
            failures.append({**where, "failures": bad[:3]})
    seconds = round(time.perf_counter() - t0, 2)
    cases = len(capture["cases"])
    return {"ok": not failures, "cases": cases, "failures": failures, "seconds": seconds}


def refuse_unverifiable(path: Path, tracked: dict[str, Any]) -> dict[str, Any]:
    """:func:`self_check` of a capture just saved; removes the file and raises
    :class:`UnverifiableCapture` with the reason when the reference fails it."""
    result = self_check(path)
    if result["ok"]:
        return {k: result[k] for k in ("cases", "seconds")}
    path.unlink(missing_ok=True)
    first = result["failures"][0]
    detail = first.get("error")
    if detail is None:
        check = first["failures"][0]
        detail = f"{check.get('name')}: " + str(
            check.get("error") or f"max abs err {check.get('max_abs_err')}"
        )
    raise UnverifiableCapture(
        f"the unmodified reference fails its own capture on {len(result['failures'])} of "
        f"{result['cases']} cases (first: case {first['case']}, {first['signature']}: "
        f"{detail[:240]}); {state.describe(tracked)}. Its calls depend on something the "
        "capture does not hold (state outside the module and its arguments, randomness), "
        "so no candidate could be checked against it: capture refused."
    )
