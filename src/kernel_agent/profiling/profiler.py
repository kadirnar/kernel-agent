"""Find where an end-to-end run spends its GPU time.

Two complementary views are collected:

* **Module view** – CUDA events recorded by forward pre/post hooks on every
  ``nn.Module`` give inclusive and self time per module *class* and per input
  shape signature (e.g. prefill ``[1, 512, 896]`` vs decode ``[1, 1, 896]``).
  These classes are the units the agent rewrites; ``work`` adds, per instance
  group and phase, the ``nn.Linear`` / convolution FLOPs and weights inside the
  calls (the ceilings table, :mod:`.ceilings`).  Entrypoints other than
  ``forward`` (``forward_step`` in a custom decode loop, ``decode`` of a VAE)
  bypass hooks; they are wrapped per instance (:mod:`.methods`) and reported
  per method.
* **Kernel view** – ``torch.profiler`` gives the CUDA kernels and aten ops that
  actually ran, the number of launches and the GPU busy fraction.  A low busy
  fraction means the run is launch/CPU bound, which calls for fusion, CUDA
  graphs or static caches rather than faster individual kernels.  Its timeline
  (:mod:`.timeline`) is the GPU busy time as the union over every stream (two
  overlapping streams are not counted twice), per stream, per stage (the calls of
  the workload's top-level modules) and the idle gaps by cause.  Kernel times
  inflate when the GPU's clocks are low or another process computes on it, so
  the kernel view is guarded like a timing (:func:`guarded_kernel_profile`):
  under the GPU lock, after a warm-up and the clock guard of ``time_call``, two
  profiled runs that must agree with each other and with the end-to-end latency
  (:func:`kernel_problems`), or the view is profiled again and, failing again,
  marked unreliable: :func:`summarize` then draws no conclusion from it.
* **Host syncs** – one more run under :mod:`.host_sync` lists every call that makes the
  host wait for the GPU (``.item()``, ``.cpu()``, pageable copies, ``torch.tensor(...,
  device=...)``) by call site, as transform opportunities.

Regions the module view cannot see into (an optimised model in a later improve
round has them): a ``torch.compile``'d module is timed as one call (hooks inside
it would make Dynamo recompile it and compile the bookkeeping), a module that
replays a CUDA graph has no child calls, and calls made while a graph is being
captured are not timed. :meth:`ModuleTimer.gaps` lists them; their kernels are
in the kernel view. Their work comes from the same calls of the unmodified model
(:func:`work_reference`, a hooked pass before an improve round's items are applied),
or is unknown.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import inspect
import re
import sys
import time
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from kernel_agent import telemetry
from kernel_agent.gpulock import gpu_lock
from kernel_agent.phases import call_phase
from kernel_agent.profiling import host_sync, timeline
from kernel_agent.profiling.methods import (
    describe,
    discover_entrypoints,
    instrument,
    workload_entrypoints,
)
from kernel_agent.projection import fold
from kernel_agent.workloads.base import Workload, synchronize


def signature_of(args: tuple[Any, ...], kwargs: dict[str, Any], limit: int = 4) -> str:
    """Compact shape/dtype signature of the tensor arguments of a call."""
    parts: list[str] = []

    def add(name: str, value: Any) -> None:
        if len(parts) >= limit:
            return
        if isinstance(value, torch.Tensor):
            dtype = str(value.dtype).removeprefix("torch.")
            parts.append(f"{name}{list(value.shape)}:{dtype}")
        elif isinstance(value, tuple | list) and value and isinstance(value[0], torch.Tensor):
            add(f"{name}[0]", value[0])

    for i, arg in enumerate(args):
        add(f"a{i}", arg)
    for key, value in kwargs.items():
        add(f"{key}=", value)
    return ", ".join(parts) or "()"


def call_signature(
    method: str, args: tuple[Any, ...], kwargs: dict[str, Any], limit: int = 1
) -> str:
    """:func:`signature_of`, prefixed with the method unless it is ``forward``."""
    sig = signature_of(args, kwargs, limit=limit)
    return sig if method == "forward" else f"{method}: {sig}"


@dataclass
class _Call:
    qualname: str
    cls: str
    signature: str
    parent: int
    start: Any  # torch.cuda.Event, or perf_counter() seconds without CUDA
    end: Any = None
    children: list[int] = field(default_factory=list)
    method: str = "forward"
    phase: str = "prefill"
    # Work of this call alone (_input_work / _output_work; class_stats adds its children's):
    rows: int = 0  # of the first tensor argument: numel / last dim
    shape: tuple[int, ...] = ()  # of the first tensor argument (the WorkReference key)
    io_bytes: int = 0  # the first tensor argument + the first tensor of the output
    flops: int = 0  # nn.Linear / convolution FLOPs, at ``dtype``
    dtype: str = ""
    weight_elems: int = 0  # weights (and bias) it reads
    weight_bytes: int = 0
    #: The hooks did not see inside: a compiled module, or a CUDA graph replayed in the call.
    opaque: bool = False
    #: A decode call's KV cache (:func:`_kv_cache`): bytes per cached position of the cache
    #: tensors passed to it, their positions, the position argument (an int or a tensor,
    #: read after the run) and, once read, the bytes the call reads from the cache.
    kv_slot_bytes: int = 0
    kv_slots: int = 0
    kv_position: Any = None
    kv_bytes: int = 0


@dataclass
class ClassStat:
    root: str
    cls: str
    module_path: str
    source_file: str | None
    instances: int
    calls: int
    inclusive_ms: float
    self_ms: float
    params: int
    is_leaf: bool
    example_qualname: str
    signatures: list[dict[str, Any]]
    #: Per entrypoint: ``{"forward_step": {"calls": 2160, "inclusive_ms": 1151.2}, ...}``.
    methods: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Per phase (:mod:`kernel_agent.phases`): calls, inclusive_ms, instances,
    #: top_signature and ``groups`` (qualname pattern -> instances).
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Where the instances are: qualname with layer indices folded -> instances
    #: (``model.base_lm.layers.*.self_attn``: 28); the module tree of
    #: :mod:`kernel_agent.projection`.
    groups: dict[str, int] = field(default_factory=dict)
    #: Per (folded qualname, phase): calls, inclusive_ms and the work of those calls, the
    #: input of :mod:`kernel_agent.profiling.ceilings` (:meth:`ModuleTimer.class_stats`).
    work: list[dict[str, Any]] = field(default_factory=list)


def _capturing() -> bool:
    """A CUDA graph is being captured on the current stream: its work is recorded, not run,
    so a module call there cannot be timed (an event recorded into a graph has no time)."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _compiled(module: nn.Module) -> nn.Module | None:
    """The original module of a ``torch.compile``'d one (``OptimizedModule._orig_mod``)."""
    orig = getattr(module, "_orig_mod", None)
    return orig if isinstance(orig, nn.Module) else None


_CONV = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


def _first_tensor(values: Any) -> torch.Tensor | None:
    """The first tensor of a call's arguments or output (a tensor, or one level of a tuple,
    list or dict-like output)."""
    if isinstance(values, torch.Tensor):
        return values
    if isinstance(values, dict):
        values = list(values.values())
    if isinstance(values, tuple | list):
        return next((v for v in values if isinstance(v, torch.Tensor)), None)
    return None


def _weights(module: nn.Module) -> list[torch.Tensor]:
    return [
        t
        for t in (getattr(module, "weight", None), getattr(module, "bias", None))
        if isinstance(t, torch.Tensor)
    ]


def _input_work(call: _Call, module: nn.Module, x: torch.Tensor | None) -> None:
    """Rows and input bytes of a call; FLOPs and weights of an ``nn.Linear`` call
    (``2 × rows × in × out``, its weight and bias read once)."""
    if x is None:
        return
    call.io_bytes = x.numel() * x.element_size()
    call.dtype = str(x.dtype).removeprefix("torch.")
    call.shape = tuple(x.shape)
    floating = x.is_floating_point() and x.dim() > 0
    call.rows = x.numel() // x.shape[-1] if floating and x.shape[-1] else x.numel()
    weights = _weights(module) if isinstance(module, nn.Linear) else []
    if weights and weights[0].dim() == 2:
        call.flops = 2 * call.rows * weights[0].numel()
        call.weight_elems = sum(t.numel() for t in weights)
        call.weight_bytes = sum(t.numel() * t.element_size() for t in weights)


def _output_work(
    call: _Call, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
) -> None:
    """Output bytes of a call; FLOPs and weights of a convolution call."""
    out = _first_tensor(output)
    if out is None:
        return
    call.io_bytes += out.numel() * out.element_size()
    weights = _weights(module) if isinstance(module, _CONV) else []
    x = _first_tensor((*args, *kwargs.values()))
    if weights and x is not None and weights[0].dim() > 2:
        weight = weights[0]
        # conv: every output element sums C_in/groups × kernel products; transposed: every
        # input element feeds C_out/groups × kernel outputs (weight dim 0 is C_out / C_in).
        per = weight.numel() // weight.shape[0]
        call.flops = 2 * (x.numel() if getattr(module, "transposed", False) else out.numel()) * per
        call.weight_elems = sum(t.numel() for t in weights)
        call.weight_bytes = sum(t.numel() * t.element_size() for t in weights)


#: Arguments of a decode call that hold its KV cache (a tensor, or a tuple / list of them),
#: and that give the position of the step in it; matched by parameter name.
_KV_ARGS = re.compile(
    r"kv_caches?|past_key_values?|layer_past|(?:key|value|k|v)_caches?|caches?|kv",
)
_POSITION_ARGS = (
    "position_id",
    "position_ids",
    "cache_position",
    "input_pos",
    "start_pos",
    "pos",
    "position",
    "positions",
    "offset",
)


_KV_PARAMS: dict[tuple[type, str], tuple[inspect.Signature, tuple[str, ...]] | None] = {}


def _kv_params(cls: type, method: str) -> tuple[inspect.Signature, tuple[str, ...]] | None:
    """The signature of ``cls.method`` and its KV-cache parameters (None: it has none)."""
    key = (cls, method)
    if key not in _KV_PARAMS:
        try:
            sig = inspect.signature(getattr(cls, method))
        except (TypeError, ValueError, AttributeError):
            _KV_PARAMS[key] = None
        else:
            names = tuple(n for n in sig.parameters if _KV_ARGS.fullmatch(n))
            _KV_PARAMS[key] = (sig, names) if names else None
    return _KV_PARAMS[key]


def _cache_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, tuple | list):
        return [t for v in value for t in _cache_tensors(v)]
    return []


def _kv_cache(
    call: _Call, module: nn.Module, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """The KV cache a decode call reads (issue #146): the tensors of its KV-cache arguments
    (by name: ``kv_cache``, ``past_key_value``, ``layer_past``, ...) of at least 3 dims, each
    holding ``slots`` positions along its longest non-last dim, read up to the call's
    position argument (``position_id``, ``cache_position``, ...: the newest position), or
    whole without one (a cache that grows holds exactly the context). A cache held by the
    module instead of passed in is not seen; a position tensor updated in place after the
    call counts its last value."""
    found = _kv_params(type(module), method)
    if found is None:
        return
    sig, names = found
    try:
        bound = sig.bind_partial(module, *args, **kwargs).arguments
    except TypeError:
        return
    per_slot = slots = 0
    for name in names:
        for t in _cache_tensors(bound.get(name)):
            if t.dim() < 3 or not t.numel():
                continue
            n = max(t.shape[:-1])
            per_slot += t.numel() // n * t.element_size()
            slots = max(slots, n)
    if not per_slot:
        return
    call.kv_slot_bytes, call.kv_slots = per_slot, slots
    call.kv_position = next((bound[n] for n in _POSITION_ARGS if bound.get(n) is not None), None)


def _read_kv(call: _Call) -> None:
    """``call.kv_bytes`` from its cache and position (after the run: may synchronise)."""
    if not call.kv_slot_bytes or call.kv_bytes:
        return
    pos = call.kv_position
    newest: int | None = None
    with contextlib.suppress(Exception):
        if isinstance(pos, torch.Tensor):
            newest = int(pos.max().item()) if pos.numel() else None
        elif isinstance(pos, int | float):
            newest = int(pos)
    context = call.kv_slots if newest is None or newest < 0 else min(newest + 1, call.kv_slots)
    call.kv_bytes = call.kv_slot_bytes * context
    call.kv_position = None  # do not keep the tensor alive


class ModuleTimer:
    """Context manager that times every module call with CUDA events.

    ``methods`` maps module classes to their non-``forward`` entrypoints (by
    default :func:`~kernel_agent.profiling.methods.discover_entrypoints`); those
    calls are timed too and attributed to the class with their method name.
    With ``cuda=False`` (default when no GPU) host wall-clock time is used.

    The bookkeeping never crashes the profile: calls during a CUDA-graph capture
    or inside code Dynamo traces are skipped, a post-hook without its pre-hook and
    a call whose start/end cannot be paired are counted (:meth:`gaps`) and left
    untimed. While the hooks are installed Dynamo compiles nothing new: compiled
    code whose guards they break runs eagerly (``eager_on_recompile``).

    ``reference``: the work of the unmodified model's calls (:func:`work_reference`),
    for the calls whose insides the hooks do not see (:meth:`_subtree_work`)."""

    def __init__(
        self,
        roots: dict[str, nn.Module],
        methods: dict[type, list[str]] | None = None,
        *,
        cuda: bool | None = None,
        reference: WorkReference | None = None,
    ) -> None:
        self.roots = roots
        self.methods = discover_entrypoints(roots) if methods is None else methods
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        self.reference = reference
        self.calls: list[_Call] = []
        #: Open calls: (index in ``calls``, id of the module, method).
        self._stack: list[tuple[int, int, str]] = []
        self._names: dict[int, tuple[str, str]] = {}
        self._ctx: contextlib.ExitStack | None = None
        #: ``torch.compile``'d modules (qualnames): timed as one call, inside not hooked.
        self.compiled: list[str] = []
        self._compiled_ids: set[int] = set()
        #: CUDA-graph replays per qualname of the module call that replayed them.
        self.replays: collections.Counter[str] = collections.Counter()
        #: ``capture``: calls during a CUDA-graph capture; ``unmatched_post``: post-hooks
        #: without an open call of their module.
        self.skipped: collections.Counter[str] = collections.Counter()
        #: Calls :meth:`class_stats` could not time (no end, or start/end that do not pair).
        self.untimed = 0

    def __enter__(self) -> ModuleTimer:
        # Modules that run inside a compiled module are not hooked (#86): their hooks would
        # break its guards (Dynamo recompiles it in the profiled run) or run in Dynamo's
        # context, which compiles the hook frames too; an event made in compiled code is a
        # ``torch.Event`` that cannot be paired with the eager ``torch.cuda.Event``s.
        inside = {
            id(m)
            for root in self.roots.values()
            for wrapper in root.modules()
            if (orig := _compiled(wrapper)) is not None
            for m in orig.modules()
        }
        modules: list[nn.Module] = []
        for root_name, root in self.roots.items():
            for qualname, module in root.named_modules():
                full = f"{root_name}.{qualname}" if qualname else root_name
                if id(module) in inside or id(module) in self._names:
                    continue
                if _compiled(module) is not None:
                    self.compiled.append(full)
                    self._compiled_ids.add(id(module))
                self._names[id(module)] = (root_name, full)
                modules.append(module)
        with contextlib.ExitStack() as stack:
            if "torch._dynamo" in sys.modules:  # something may be compiled
                stack.enter_context(torch.compiler.set_stance("eager_on_recompile"))
            register = getattr(torch.cuda.graphs, "register_graph_replay_start_hook", None)
            if torch.cuda.is_available() and register is not None:  # untimed passes too
                stack.callback(register(self._replay).remove)
            stack.enter_context(instrument(modules, self.methods, self._pre, self._post))
            self._ctx = stack.pop_all()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def _now(self) -> Any:
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def _elapsed_ms(call: _Call) -> float | None:
        """Inclusive ms of ``call``; None when it cannot be timed: its post-hook never ran,
        or its start and end do not pair (#86: a ``torch.Event`` made in compiled code next
        to a ``torch.cuda.Event``; an event recorded into a CUDA graph)."""
        if call.end is None or type(call.start) is not type(call.end):
            return None
        if isinstance(call.start, float):
            return (call.end - call.start) * 1000.0
        try:
            return float(call.start.elapsed_time(call.end))
        except (RuntimeError, TypeError):
            return None

    def _replay(self, graph: Any) -> None:
        if self._stack:
            call = self.calls[self._stack[-1][0]]
            call.opaque = True  # the graph's work ran without module calls
            self.replays[call.qualname] += 1

    def _pre(
        self, module: nn.Module, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if torch.compiler.is_compiling():  # traced by Dynamo: keep the bookkeeping out
            return
        if _capturing():
            self.skipped["capture"] += 1
            return
        _, full = self._names.get(id(module), ("?", type(module).__name__))
        start = self._now()
        parent = self._stack[-1][0] if self._stack else -1
        call = _Call(
            full,
            type(module).__name__,
            call_signature(method, args, kwargs, limit=1),
            parent,
            start,
            method=method,
            phase=call_phase(method, args, kwargs),
            opaque=id(module) in self._compiled_ids,  # its insides are not hooked
        )
        with contextlib.suppress(Exception):  # the work estimate never breaks a profile
            _input_work(call, module, _first_tensor((*args, *kwargs.values())))
            if call.phase == "decode":
                _kv_cache(call, module, method, args, kwargs)
        self.calls.append(call)
        index = len(self.calls) - 1
        if parent >= 0:
            self.calls[parent].children.append(index)
        self._stack.append((index, id(module), method))

    def _post(
        self,
        module: nn.Module,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if torch.compiler.is_compiling() or _capturing():
            return
        key = (id(module), method)
        depth = len(self._stack) - 1
        while depth >= 0 and self._stack[depth][1:] != key:
            depth -= 1
        if depth < 0:  # its pre-hook did not record (e.g. it ran during a capture)
            self.skipped["unmatched_post"] += 1
            return
        # Calls opened above it never saw their post-hook: they stay untimed.
        index = self._stack[depth][0]
        del self._stack[depth:]
        self.calls[index].end = self._now()
        with contextlib.suppress(Exception):
            _output_work(self.calls[index], module, args, kwargs, output)

    def class_stats(self) -> list[ClassStat]:
        if self.cuda:
            synchronize()
        times = [self._elapsed_ms(call) for call in self.calls]
        for call in self.calls:
            _read_kv(call)
        self.untimed = sum(t is None for t in times)
        inclusive = [t or 0.0 for t in times]

        modules = self._modules()
        work = self._subtree_work(modules)

        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for idx, call in enumerate(self.calls):
            root = call.qualname.split(".", 1)[0]
            own = inclusive[idx] - sum(inclusive[c] for c in call.children)
            g = groups.setdefault(
                (root, call.cls),
                {
                    "qualnames": set(),
                    "calls": 0,
                    "inclusive": 0.0,
                    "self": 0.0,
                    "sigs": collections.defaultdict(lambda: [0, 0.0]),
                    "methods": collections.defaultdict(lambda: [0, 0.0]),
                    "phases": collections.defaultdict(
                        lambda: {
                            "calls": 0,
                            "inclusive": 0.0,
                            "qualnames": set(),
                            "sigs": collections.Counter(),
                        }
                    ),
                    "example": call.qualname,
                    "work": collections.defaultdict(_new_work),
                },
            )
            # Do not double count recursive containers of the same class.
            ancestor = call.parent
            nested = False
            while ancestor >= 0:
                if self.calls[ancestor].cls == call.cls:
                    nested = True
                    break
                ancestor = self.calls[ancestor].parent
            g["qualnames"].add(call.qualname)
            g["calls"] += 1
            g["self"] += max(own, 0.0)
            meth = g["methods"][call.method]
            meth[0] += 1
            phase = g["phases"][call.phase]
            phase["calls"] += 1
            phase["qualnames"].add(call.qualname)
            phase["sigs"][call.signature] += inclusive[idx]
            if not nested:
                g["inclusive"] += inclusive[idx]
                meth[1] += inclusive[idx]
                phase["inclusive"] += inclusive[idx]
                _add_work(
                    g["work"][(fold(call.qualname), call.phase)], call, work[idx], inclusive[idx]
                )
            sig = g["sigs"][call.signature]
            sig[0] += 1
            sig[1] += inclusive[idx]

        stats: list[ClassStat] = []
        for (root, cls), g in groups.items():
            example = modules.get(g["example"])
            params = sum(p.numel() for p in example.parameters()) if example is not None else 0
            is_leaf = example is not None and not any(True for _ in example.children())
            source = None
            module_path = ""
            if example is not None:
                module_path = f"{type(example).__module__}.{type(example).__qualname__}"
                try:
                    source = inspect.getsourcefile(type(example))
                except TypeError:
                    source = None
            sigs = sorted(g["sigs"].items(), key=lambda kv: -kv[1][1])[:6]
            stats.append(
                ClassStat(
                    root=root,
                    cls=cls,
                    module_path=module_path,
                    source_file=source,
                    instances=len(g["qualnames"]),
                    calls=g["calls"],
                    inclusive_ms=round(g["inclusive"], 4),
                    self_ms=round(g["self"], 4),
                    params=params,
                    is_leaf=is_leaf,
                    example_qualname=g["example"],
                    signatures=[
                        {"signature": s, "calls": c, "inclusive_ms": round(t, 4)}
                        for s, (c, t) in sigs
                    ],
                    methods={
                        m: {"calls": c, "inclusive_ms": round(t, 4)}
                        for m, (c, t) in sorted(g["methods"].items(), key=lambda kv: -kv[1][1])
                    },
                    phases={
                        name: _phase_stat(p)
                        for name, p in sorted(
                            g["phases"].items(), key=lambda kv: -kv[1]["inclusive"]
                        )
                    },
                    groups=dict(_folded(g["qualnames"]).most_common()),
                    work=[
                        _work_entry(pattern, phase_name, w)
                        for (pattern, phase_name), w in sorted(
                            g["work"].items(), key=lambda kv: -kv[1]["inclusive"]
                        )
                    ],
                )
            )
        stats.sort(key=lambda s: -s.inclusive_ms)
        return stats

    def _modules(self) -> dict[str, nn.Module]:
        """Every module of the roots by qualname (the first one of a shared module)."""
        modules: dict[str, nn.Module] = {}
        for root_name, root_module in self.roots.items():
            for qualname, module in root_module.named_modules():
                full = f"{root_name}.{qualname}" if qualname else root_name
                modules.setdefault(full, module)
        return modules

    def _subtree_work(self, modules: dict[str, nn.Module]) -> list[_Work]:
        """Per call: its own work (``nn.Linear`` / convolution) plus that of every call inside
        it.

        Some calls hide their insides from the hooks: a compiled module or a call that
        replays a CUDA graph (``opaque``), and a call that reads no weights although its
        module holds ``nn.Linear`` / convolution layers (a replaced kernel, ``F.linear`` on a
        child's weight). Such a call, and one with such a call inside, takes the work of the
        same call in the unmodified model (``self.reference``, :class:`WorkReference`). Without
        one, the work of an opaque call is ``unknown`` (a CUDA graph or compiled module may run
        its layers at other row counts, or more often, than its input suggests: the LocDiT of
        a CFM solver, #106), and that of the other kind is estimated from its module's
        ``nn.Linear`` weights at its input's rows (``estimated``)."""
        static: dict[str, tuple[int, int, int, str, bool]] = {}
        out: list[_Work] = [_Work()] * len(self.calls)
        for i in range(len(self.calls) - 1, -1, -1):  # a call's children come after it
            call = self.calls[i]
            w = _Work({call.dtype: call.flops} if call.flops else {})
            w.weight_elems, w.weight_bytes = call.weight_elems, call.weight_bytes
            for c in call.children:
                w.add(out[c])
            # a layer that passes its cache on to its attention reads it once
            w.kv_bytes = max(w.kv_bytes, call.kv_bytes)
            module = modules.get(call.qualname)
            bypassed = False
            if not w.weight_elems and not call.opaque and module is not None:
                if call.qualname not in static:
                    static[call.qualname] = _layer_weights(module)
                bypassed = static[call.qualname][1] > 0 or static[call.qualname][4]
            if call.opaque or bypassed or w.unknown or w.estimated:
                ref = self.reference.get(call, module) if self.reference else None
                if ref is not None:
                    out[i] = ref
                    continue
            if call.opaque:
                w.unknown = True
            elif bypassed and call.rows:
                matmul, elems, nbytes, dtype, _ = static[call.qualname]
                if matmul:
                    dtype = call.dtype if "float" in call.dtype else dtype  # ids: the weights'
                    w.flops = {dtype: 2 * call.rows * matmul}
                    w.weight_elems, w.weight_bytes, w.estimated = elems, nbytes, True
            out[i] = w
        return out

    def work_reference(self) -> WorkReference:
        """The work of the calls of this run (:meth:`_subtree_work`) per (qualname, phase,
        method, input shape), and where each module was: the reference for a later profile
        of the same workload whose optimisations hide their insides from the hooks."""
        modules = self._modules()
        for call in self.calls:
            _read_kv(call)
        calls: dict[tuple[str, str, str, tuple[int, ...]], tuple[int, _Work]] = {}
        for call, w in zip(self.calls, self._subtree_work(modules), strict=True):
            if w.unknown or w.estimated:  # only what the hooks saw
                continue
            key = (call.qualname, call.phase, call.method, call.shape)
            n, total = calls.get(key) or (0, _Work())
            total.add(w)
            calls[key] = (n + 1, total)
        names = {id(m): (weakref.ref(m), q) for q, m in reversed(list(modules.items()))}
        return WorkReference(calls, names)

    def gaps(self) -> dict[str, Any]:
        """What the module view does not time (empty when it timed everything); after
        :meth:`class_stats`. ``graph_replays``: qualnames with layer indices folded."""
        replays: collections.Counter[str] = collections.Counter()
        for qualname, n in self.replays.items():
            replays[fold(qualname)] += n
        out: dict[str, Any] = {
            "compiled": self.compiled,
            "graph_replays": dict(replays.most_common()),
            "capture_calls": self.skipped["capture"],
            "unmatched_post": self.skipped["unmatched_post"],
            "untimed": self.untimed,
        }
        return {k: v for k, v in out.items() if v}


@dataclass
class _Work:
    """Work of one call and the calls inside it."""

    flops: dict[str, int] = field(default_factory=dict)  # per dtype
    weight_elems: int = 0
    weight_bytes: int = 0
    kv_bytes: int = 0  # KV cache read by decode calls (_kv_cache)
    # Where some of it comes from (ModuleTimer._subtree_work):
    estimated: bool = False  # module weights at the input's rows
    reference: bool = False  # the same call of the unmodified model (WorkReference)
    unknown: bool = False  # hidden from the hooks, no reference: the total is incomplete

    def add(self, other: _Work) -> None:
        for dtype, n in other.flops.items():
            self.flops[dtype] = self.flops.get(dtype, 0) + n
        self.weight_elems += other.weight_elems
        self.weight_bytes += other.weight_bytes
        self.kv_bytes += other.kv_bytes
        self.estimated |= other.estimated
        self.reference |= other.reference
        self.unknown |= other.unknown


@dataclass
class WorkReference:
    """The work of an unmodified model's calls (:func:`work_reference`; issue #106).

    An optimised model hides work from the hooks (a compiled module, a CUDA-graph replay, a
    replaced kernel), but that work is a property of the math, not of its implementation:
    the same call of the unmodified model did it. ``calls``: (qualname, phase, method, shape
    of the first tensor argument) -> (calls, their summed work); ``names``: id of each
    module -> (weak reference, its qualname then), which finds a module that a transform
    moved or wrapped (``torch.compile`` keeps it as ``_orig_mod``)."""

    calls: dict[tuple[str, str, str, tuple[int, ...]], tuple[int, _Work]]
    names: dict[int, tuple[weakref.ref[nn.Module], str]] = field(default_factory=dict)
    seconds: float = 0.0  # what the hooked pass took

    def get(self, call: _Call, module: nn.Module | None) -> _Work | None:
        """The work of ``call`` per call in the unmodified model (``reference`` set), at its
        qualname then or now; None when the unmodified model made no such call."""
        qualnames = []
        for m in (_compiled(module), module) if module is not None else ():
            entry = self.names.get(id(m)) if m is not None else None
            if entry is not None and entry[0]() is m:  # not a new module at a reused id
                qualnames.append(entry[1])
        for qualname in [*qualnames, call.qualname]:
            hit = self.calls.get((qualname, call.phase, call.method, call.shape))
            if hit is not None:
                n, total = hit
                return _Work(
                    {dtype: round(f / n) for dtype, f in total.flops.items()},
                    round(total.weight_elems / n),
                    round(total.weight_bytes / n),
                    round(total.kv_bytes / n),
                    reference=True,
                )
        return None

    def info(self) -> dict[str, Any]:
        """``profile.json`` → ``work_reference``."""
        return {
            "source": "the unmodified model (one hooked pass before the round's items)",
            "calls": sum(n for n, _ in self.calls.values()),
            "keys": len(self.calls),
            "seconds": self.seconds,
        }


def work_reference(workload: Workload, inputs: Any, *, miner: Any = None) -> WorkReference:
    """The work of every module call of ``workload`` as loaded (the unmodified model,
    before an improve round's items are applied): one hooked run, untimed (#106).
    ``miner``: a :class:`~kernel_agent.profiling.fusion.Miner` that records every op of the
    run with its module call (the fusion candidates, #231)."""
    start = time.perf_counter()
    roots = workload.roots()
    methods = discover_entrypoints(roots, workload_entrypoints(workload))
    with (
        torch.inference_mode(),
        ModuleTimer(roots, methods, cuda=False) as timer,
        miner.recording(timer) if miner is not None else contextlib.nullcontext(),
    ):
        workload.run(inputs)
        synchronize()
    reference = timer.work_reference()
    reference.seconds = round(time.perf_counter() - start, 2)
    return reference


def _layer_weights(module: nn.Module) -> tuple[int, int, int, str, bool]:
    """(matmul weight elements, elements with bias, bytes, dtype) of the ``nn.Linear``
    layers in ``module``, and whether it holds a convolution."""
    matmul = elems = nbytes = 0
    dtype = ""
    conv = False
    for sub in module.modules():
        conv |= isinstance(sub, _CONV)
        weights = _weights(sub) if isinstance(sub, nn.Linear) else []
        if weights and weights[0].dim() == 2:
            matmul += weights[0].numel()
            elems += sum(t.numel() for t in weights)
            nbytes += sum(t.numel() * t.element_size() for t in weights)
            dtype = dtype or str(weights[0].dtype).removeprefix("torch.")
    return matmul, elems, nbytes, dtype, conv


def _new_work() -> dict[str, Any]:
    return {
        "qualnames": set(),
        "calls": 0,
        "inclusive": 0.0,
        "flops": collections.Counter(),
        "weight_elems": 0,
        "weight_bytes": 0,
        "io_bytes": 0,
        "kv_bytes": 0,
        "estimated": 0,
        "reference": 0,
        "unknown": 0,
        "sigs": collections.Counter(),
    }


def _add_work(acc: dict[str, Any], call: _Call, work: _Work, ms: float) -> None:
    acc["qualnames"].add(call.qualname)
    acc["calls"] += 1
    acc["inclusive"] += ms
    acc["flops"].update(work.flops)
    acc["weight_elems"] += work.weight_elems
    acc["weight_bytes"] += work.weight_bytes
    acc["io_bytes"] += call.io_bytes
    acc["kv_bytes"] += work.kv_bytes
    acc["estimated"] += work.estimated
    acc["reference"] += work.reference
    acc["unknown"] += work.unknown
    acc["sigs"][call.signature] += ms


def _work_entry(group: str, phase: str, acc: dict[str, Any]) -> dict[str, Any]:
    """One ``ClassStat.work`` entry: the calls of one instance group in one phase, totals
    per run. ``flops`` per dtype and ``weight_*`` (read once per call) count the
    ``nn.Linear`` and convolution calls inside; ``io_bytes`` the first input and output of
    each call; ``kv_bytes`` (decode calls only, when non-zero) the KV cache read by the calls
    inside (:func:`_kv_cache`). Calls whose insides the hooks did not see
    (ModuleTimer._subtree_work): ``reference_calls`` took the work of the unmodified model's
    call, ``unknown_calls`` have none (their totals are incomplete), ``estimated_calls`` are
    estimated from module weights."""
    top = acc["sigs"].most_common(1)
    entry = {
        "group": group,
        "phase": phase,
        "instances": len(acc["qualnames"]),
        "calls": acc["calls"],
        "inclusive_ms": round(acc["inclusive"], 4),
        "flops": dict(acc["flops"]),
        "weight_elems": acc["weight_elems"],
        "weight_bytes": acc["weight_bytes"],
        "io_bytes": acc["io_bytes"],
        "signature": top[0][0] if top else "",
    }
    if acc["kv_bytes"]:
        entry["kv_bytes"] = acc["kv_bytes"]
    for name in ("estimated", "reference", "unknown"):
        if acc[name]:
            entry[f"{name}_calls"] = acc[name]
    return entry


def _phase_stat(p: dict[str, Any]) -> dict[str, Any]:
    """One phase of a class: totals, the top signature and where its instances live
    (qualnames with layer indices folded: ``model.base_lm.layers.*.self_attn``)."""
    top = p["sigs"].most_common(1)
    return {
        "calls": p["calls"],
        "inclusive_ms": round(p["inclusive"], 4),
        "instances": len(p["qualnames"]),
        "top_signature": top[0][0] if top else "",
        "groups": dict(_folded(p["qualnames"]).most_common(6)),
    }


def _folded(qualnames: set[str]) -> collections.Counter[str]:
    """Instances per qualname with layer indices folded (``model.layers.*.mlp``)."""
    return collections.Counter(fold(q) for q in qualnames)


def _device_time(evt: Any, self_only: bool) -> float:
    names = (
        ("self_device_time_total", "self_cuda_time_total")
        if self_only
        else ("device_time_total", "cuda_time_total")
    )
    for name in names:
        value = getattr(evt, name, None)
        if value is not None:
            return float(value) / 1000.0  # us -> ms
    return 0.0


#: Kernel-view sanity check (#95): GPU kernel time above this share of the end-to-end
#: latency measured without the profiler, or two profiled runs further apart than
#: MAX_RUN_SPREAD (of the smaller), cannot be right: the GPU ran slow (low clocks, another
#: process computing on it), as in a re-profile that reported 1056 ms of kernels (202 %)
#: for a 523 ms run and 551 ms for the same 29,122 launches in another run.
MAX_BUSY_SHARE = 1.05
MAX_RUN_SPREAD = 0.2
KERNEL_RUNS = 2  # profiled runs per attempt
KERNEL_ATTEMPTS = 2  # a kernel view that fails the check is profiled once more


def kernel_problems(busy_ms: list[float], reference_ms: float | None) -> list[str]:
    """Why profiled runs with ``busy_ms`` of GPU kernel time each cannot be right for a run
    that takes ``reference_ms`` without the profiler ([]: plausible)."""
    if not busy_ms:
        return []
    low, high = min(busy_ms), max(busy_ms)
    problems = []
    if reference_ms and low > MAX_BUSY_SHARE * reference_ms:
        problems.append(
            f"GPU kernel time {low:.1f} ms is {low / reference_ms:.0%} of the "
            f"{reference_ms:.1f} ms measured without the profiler"
        )
    if low > 0 and (high - low) / low > MAX_RUN_SPREAD:
        problems.append(
            "profiled runs disagree: "
            + " vs ".join(f"{b:.1f}" for b in busy_ms)
            + f" ms of GPU kernel time ({(high - low) / low:.0%} apart)"
        )
    return problems


def guarded_kernel_profile(
    workload: Workload, inputs: Any, reference_ms: float | None = None
) -> dict[str, Any]:
    """:func:`kernel_profile` measured like a timing and sanity-checked.

    Under the GPU lock (``gpulock.gpu_lock``: a no-op in a worker whose parent holds it),
    each attempt runs the workload once unprofiled (warm-up), then profiles KERNEL_RUNS
    runs, each after spinning the GPU to full clocks (``bench.ensure_clocks``, the clock
    guard of ``time_call``: the GPU idles while the profiler parses the previous run, a
    minute for eager VoxCPM2), with GPU telemetry right before and right after each run
    (only the latter counts as a load sample: ``nvidia-smi``'s clocks lag by a second or
    so) and the other processes on the GPU. Runs that fail :func:`kernel_problems` against
    ``reference_ms`` (the end-to-end time of the profiled window) are profiled once more;
    if they fail again the view says ``reliable: false`` and why (``unreliable``). The view
    returned is the run with the least GPU kernel time of the last attempt, with its share
    of ``reference_ms`` (``busy_share``); ``attempts`` keeps every attempt (GPU kernel time
    and clock probe per run, ``gpu`` telemetry, ``other_gpu_processes``). Without CUDA: one
    attempt, no lock and no clock guard."""
    cuda = torch.cuda.is_available()
    attempts: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    problems: list[str] = []
    # Stage ranges for the timeline, installed before the warm-up: hooks can make compiled
    # code recompile once, which must not happen in a profiled run.
    stages = timeline.StageRanges.of(workload)
    staged: dict[str, Any] = {"stages": stages} if stages is not None else {}
    with (
        gpu_lock() if cuda else contextlib.nullcontext(),
        stages if stages is not None else contextlib.nullcontext(),
    ):
        for _ in range(KERNEL_ATTEMPTS if cuda else 1):
            with torch.inference_mode():
                workload.run(inputs)  # warm-up: caches, allocator, lazy init
            synchronize()
            monitor = telemetry.Monitor() if cuda else None
            others = monitor.processes() if monitor is not None else []
            clocks: list[float | None] = []
            views = []
            for _ in range(KERNEL_RUNS):
                if cuda:
                    clocks.append(_full_clocks())
                if monitor is not None:  # before the work: not a load sample
                    monitor.sample("before", loaded=False)
                after = functools.partial(monitor.sample, "after") if monitor else None
                views.append(kernel_profile(workload, inputs, after=after, **staged))
            busy = [v["gpu_busy_ms"] for v in views]
            problems = kernel_problems(busy, reference_ms)
            attempts.append(
                {
                    "gpu_busy_ms": busy,
                    "clock": clocks,
                    "gpu": monitor.summary() if monitor is not None else {},
                    "other_gpu_processes": others,
                    **({"problems": problems} if problems else {}),
                }
            )
            if not problems:
                break
    view = min(views, key=lambda v: v["gpu_busy_ms"])
    view.update(
        reference_ms=round(reference_ms, 2) if reference_ms else None,
        busy_share=round(view["gpu_busy_ms"] / reference_ms, 3) if reference_ms else None,
        reliable=not problems,
        **({"unreliable": problems} if problems else {}),
        attempts=attempts,
    )
    return view


def _full_clocks() -> float | None:
    """Spin the GPU to full clocks (``bench.ensure_clocks``); its last DRAM clock probe
    (about 1: full clocks; None: no probe)."""
    from kernel_agent.kernels import bench

    try:
        state = bench.ensure_clocks()
    except Exception:  # out of memory for the probe: the profile goes on unguarded
        return None
    return round(state, 3) if state is not None else None


def kernel_view_problems(kv: dict[str, Any], window_ms: float | None) -> list[str]:
    """Why ``summarize`` must not draw conclusions from the kernel view ``kv`` ([]: none):
    its own check failed, or (a profile from before the check) its GPU kernel time is above
    MAX_BUSY_SHARE of ``window_ms``."""
    if kv.get("reliable") is False:
        return list(kv.get("unreliable") or ["it failed its sanity check"])
    if "reliable" not in kv:
        return kernel_problems([float(kv.get("gpu_busy_ms") or 0.0)], window_ms)
    return []


def kernel_profile(
    workload: Workload,
    inputs: Any,
    top: int = 40,
    *,
    after: Callable[[], Any] | None = None,
    stages: timeline.StageRanges | None = None,
) -> dict[str, Any]:
    """One profiled run. ``wall_ms`` ends when its GPU work does, before the profiler stops
    (which parses every event: longer than the run itself for eager VoxCPM2); ``after`` is
    called then (GPU telemetry while the clocks are still up). ``stages``: installed stage
    ranges (:class:`.timeline.StageRanges`) for the ``timeline``.

    ``gpu_busy_ms`` is the union of the GPU work over every stream (:func:`.timeline.analyze`)
    and ``kernel_time_ms`` the summed time of every kernel, copy and memset (they differ when
    streams overlap); without a timeline (it failed: ``timeline_error``) both are the sum."""
    from torch.profiler import ProfilerActivity, profile, record_function

    synchronize()
    if stages is not None:
        stages.reset()
    start = time.perf_counter()
    with (
        torch.inference_mode(),
        profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof,
    ):
        with record_function(timeline.RUN):
            workload.run(inputs)
            synchronize()
        wall_ms = (time.perf_counter() - start) * 1000
        if after is not None:
            after()

    kernels: list[dict[str, Any]] = []
    ops: list[dict[str, Any]] = []
    launches = 0
    gpu_busy = 0.0
    for evt in prof.key_averages():
        is_device = str(getattr(evt, "device_type", "")).endswith("CUDA")
        if is_device:
            ms = _device_time(evt, self_only=True)
            launches += int(evt.count)
            gpu_busy += ms
            kernels.append(
                {"name": evt.key[:160], "calls": int(evt.count), "total_ms": round(ms, 4)}
            )
        elif evt.key.startswith("aten::"):
            ms = _device_time(evt, self_only=False)
            if ms > 0:
                ops.append({"name": evt.key, "calls": int(evt.count), "device_ms": round(ms, 4)})
    kernels.sort(key=lambda k: -k["total_ms"])
    ops.sort(key=lambda o: -o["device_ms"])
    sms = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        if torch.cuda.is_available()
        else None
    )
    busy, extra = gpu_busy, timeline_view(prof, stages, sms)
    if "timeline" in extra:
        busy = float(extra["timeline"]["busy_ms"])
    return {
        "wall_ms": round(wall_ms, 2),
        "gpu_busy_ms": round(busy, 2),
        "kernel_time_ms": round(gpu_busy, 2),
        "gpu_busy_fraction": round(busy / wall_ms, 3) if wall_ms else 0.0,
        "kernel_launches": launches,
        "avg_kernel_us": round(1000 * gpu_busy / launches, 2) if launches else 0.0,
        "kernels": kernels[:top],
        "aten_ops": ops[:top],
        **extra,
    }


def timeline_view(
    prof: Any, stages: timeline.StageRanges | None, sms: int | None = None
) -> dict[str, Any]:
    """``{"timeline", "overlap_ms"}`` of a finished profile (:func:`.timeline.analyze`; ``sms``:
    the GPU's SM count, for the SM fill), or ``{"timeline_error"}``: the timeline never
    breaks the kernel view."""
    try:
        tl = timeline.analyze(
            timeline.from_profiler(prof),
            calls=stages.calls if stages is not None else None,
            sms=sms,
        )
    except Exception as exc:
        return {"timeline_error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"overlap_ms": tl["overlap_ms"], "timeline": tl}


def profile_workload(
    workload: Workload,
    inputs: Any,
    reference_ms: float | None = None,
    *,
    work: WorkReference | None = None,
    fusions: bool = False,
) -> dict[str, Any]:
    """Module view + kernel view (:func:`guarded_kernel_profile`; ``reference_ms``: the
    end-to-end time of the profiled window, measured without the profiler). Assumes the
    workload is warmed up. ``work``: the unmodified model's (:func:`work_reference`), for
    the calls of an optimised model whose insides the hooks do not see. ``fusions``: the
    model is the unmodified one; one more hooked run records its fusion chains
    (:func:`.fusion.scan`, ``fusions``)."""
    roots = workload.roots()
    methods = discover_entrypoints(roots, workload_entrypoints(workload))
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), ModuleTimer(roots, methods, reference=work) as timer:
        workload.run(inputs)
        synchronize()
    hooked_ms = (time.perf_counter() - start) * 1000
    classes = timer.class_stats()
    syncs = host_sync.scan(workload, inputs)  # one more run: host syncs by call site
    kernel_view = guarded_kernel_profile(workload, inputs, reference_ms)
    mined: dict[str, Any] = {}
    if fusions:  # after the kernel view: a run that fails under the miner cannot spoil it
        from kernel_agent.profiling import fusion

        mined["fusions"] = fusion.scan(workload, inputs)
    return {
        "hooked_wall_ms": round(hooked_ms, 2),
        "module_calls": len(timer.calls),
        "entrypoints": describe(methods),
        "module_gaps": timer.gaps(),
        **({"work_reference": work.info()} if work is not None else {}),
        "classes": [asdict(c) for c in classes],
        "kernel_view": kernel_view,
        "host_syncs": syncs,
        **mined,
    }


def _methods_cell(methods: dict[str, dict[str, Any]]) -> str:
    """``forward_step×2160 (1151.2 ms) · forward×36 (12.3 ms)``; empty for forward-only classes."""
    if set(methods) <= {"forward"}:
        return ""
    return " · ".join(
        f"{name}×{m['calls']} ({m['inclusive_ms']:.1f} ms)" for name, m in methods.items()
    )


def summarize(
    profile: dict[str, Any],
    baseline_ms: float,
    top: int = 30,
    *,
    metric: str = "end-to-end latency",
    per: str = "per run",
) -> str:
    """Markdown summary handed to the planner. ``metric`` / ``per``: what ``baseline_ms``
    measures (``objective.py``; the profile covers the same window)."""
    kv = profile["kernel_view"]
    busy = kv["gpu_busy_ms"] / baseline_ms if baseline_ms else kv["gpu_busy_fraction"]
    tl = kv.get("timeline") or {}
    total = max((c["inclusive_ms"] for c in profile["classes"]), default=0.0) or 1.0
    unreliable = kernel_view_problems(kv, baseline_ms)
    if unreliable:
        verdict = (
            "**UNRELIABLE kernel view** ("
            + "; ".join(unreliable)
            + "): no conclusion (launch/CPU or GPU bound) is drawn from it, and the kernel "
            "and aten-op times below may be inflated: compare them with each other, not "
            "with the latency."
        )
    elif busy < 0.6:
        verdict = (
            "LAUNCH/CPU BOUND: the GPU idles between tiny kernels. Fusing many small ops into "
            "few kernels, CUDA graphs and static caches matter more than faster math."
        )
    else:
        verdict = "GPU bound: faster kernels pay off directly"
    lines = [
        "# Profile summary",
        "",
        f"* {metric} (no hooks): **{baseline_ms:.1f} ms**",
        f"* GPU {'busy (union over streams)' if tl else 'kernel time'}: "
        f"{kv['gpu_busy_ms']:.1f} ms {per} = **{busy:.0%}** of the {metric} — {verdict}",
        *_guard_notes(kv),
        f"* kernel launches: {kv['kernel_launches']} (avg {kv['avg_kernel_us']:.1f} us/kernel)",
        f"* module calls: {profile['module_calls']}",
    ]
    if profile.get("entrypoints"):
        lines.append(
            "* non-`forward` entrypoints (they bypass hooks and are instrumented separately): "
            + ", ".join(f"`{e}`" for e in profile["entrypoints"])
        )
    lines += _gap_notes(profile.get("module_gaps") or {})
    lines += [
        "",
        "## Module classes by inclusive time",
        "",
        "Times are measured with per-module CUDA events (hooks add overhead, so use the "
        "*share* column; it includes CPU launch gaps, which is what a fused kernel removes). "
        "*methods* splits calls and time by entrypoint when a class is also called through "
        "a method other than `forward` (e.g. `forward_step` in a decode loop); a replacement "
        "for such a class must implement those methods too.",
        "",
        "| root | class | leaf | inst | calls | methods | share | inclusive ms | self ms "
        "| params | top signature |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in profile["classes"][:top]:
        sig = c["signatures"][0]["signature"] if c["signatures"] else ""
        lines.append(
            f"| {c['root']} | `{c['cls']}` | {'y' if c['is_leaf'] else ''} | {c['instances']} | "
            f"{c['calls']} | {_methods_cell(c.get('methods', {}))} | "
            f"{c['inclusive_ms'] / total:.1%} | {c['inclusive_ms']:.2f} | "
            f"{c['self_ms']:.2f} | {c['params']:,} | "
            f"`{sig[:70]}` |"
        )
    lines += _phase_split(profile["classes"][:top])
    lines += [
        "",
        "## Top CUDA kernels",
        "",
        "| kernel | calls | total ms |",
        "|---|---|---|",
    ]
    for k in kv["kernels"][:top]:
        lines.append(f"| `{k['name'][:110]}` | {k['calls']} | {k['total_ms']:.3f} |")
    lines += [
        "",
        "## Top aten ops (inclusive device time; nested ops overlap)",
        "",
        "| op | calls | ms |",
        "|---|---|---|",
    ]
    for o in kv["aten_ops"][:20]:
        lines.append(f"| `{o['name']}` | {o['calls']} | {o['device_ms']:.3f} |")
    if tl:
        lines += timeline.markdown(tl)
    lines += host_sync.summary_lines(profile.get("host_syncs"))
    lines += _roofline_note()
    return "\n".join(lines) + "\n"


def _guard_notes(kv: dict[str, Any]) -> list[str]:
    """How the kernel view was measured (:func:`guarded_kernel_profile`; [] for a profile
    from before the guard)."""
    attempts = kv.get("attempts") or []
    if not attempts:
        return []
    last = attempts[-1]
    busy = last.get("gpu_busy_ms") or []
    clocks = last.get("clock")  # [] without CUDA: no lock, no clock guard
    note = (
        "* kernel view: "
        + ("under the GPU lock after a warm-up and the clock guard; " if clocks else "")
        + f"{len(busy)} profiled runs ({' and '.join(f'{b:.1f}' for b in busy)} ms of GPU "
        "kernel time)"
    )
    if clocks:
        probes = " / ".join("n/a" if c is None else f"{c:.2f}" for c in clocks)
        note += f", DRAM clock probe before each run {probes} (1 = full clocks)"
    if (sm := (last.get("gpu") or {}).get("sm_mhz")) and len(sm) == 3:
        note += f", SM clock {sm[0]}-{sm[2]} MHz at the end of the runs"
    if len(attempts) > 1:
        note += f"; profiled {len(attempts)} times (the first failed its sanity check)"
    if others := last.get("other_gpu_processes") or []:
        listed = ", ".join(
            f"{p.get('name') or 'pid'} {p['pid']} ({p.get('used_mib') or '?'} MiB)"
            for p in others[:3]
        )
        more = f" and {len(others) - 3} more" if len(others) > 3 else ""
        note += f"; other processes on the GPU: {listed}{more}"
    return [note]


def _gap_notes(gaps: dict[str, Any]) -> list[str]:
    """Bullets on the regions the module view does not see into (:meth:`ModuleTimer.gaps`)."""
    notes = []
    if gaps.get("compiled"):
        notes.append(
            "* `torch.compile`d modules, timed as one call (no rows for their submodules): "
            + ", ".join(f"`{q}`" for q in gaps["compiled"])
        )
    if gaps.get("graph_replays"):
        notes.append(
            "* CUDA-graph replays, by the module call that replays them (the modules captured "
            "in the graph have no rows): "
            + ", ".join(f"`{q}` ×{n}" for q, n in gaps["graph_replays"].items())
        )
    untimed = {
        "during a CUDA-graph capture": gaps.get("capture_calls", 0),
        "post-hooks without an open call": gaps.get("unmatched_post", 0),
        "calls without a usable start/end": gaps.get("untimed", 0),
    }
    if any(untimed.values()):
        notes.append(
            "* module calls not timed: "
            + ", ".join(f"{n} {what}" for what, n in untimed.items() if n)
        )
    if notes:
        notes.append("* the kernel view below covers these regions (it sees every kernel)")
    return notes


def _phase_split(classes: list[dict[str, Any]], min_share: float = 0.05) -> list[str]:
    """Classes that spend at least ``min_share`` of their time in each of both phases:
    candidates for one target per phase."""

    def both(c: dict[str, Any]) -> bool:
        times = [p["inclusive_ms"] for p in (c.get("phases") or {}).values()]
        return len(times) > 1 and min(times) >= min_share * (sum(times) or 1.0)

    split = [c for c in classes if both(c)]
    if not split:
        return []
    lines = [
        "",
        "## Phase split",
        "",
        "Classes called both on several positions at once (`prefill`: prompt prefill, "
        "encoder and DiT/denoiser blocks) and one position at a time (`decode`: `*step*` "
        "entrypoints, `[batch, 1, ...]` inputs). Each phase can be its own target "
        "(`phase` in the plan), restricted to some instances with `qualname_regex`.",
        "",
        "| class | phase | calls | inclusive ms | share of class | instances | top signature "
        "| instances by qualname |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in split:
        total = sum(p["inclusive_ms"] for p in c["phases"].values()) or 1.0
        for name, p in c["phases"].items():
            groups = ", ".join(f"`{q}` ({n})" for q, n in p["groups"].items())
            lines.append(
                f"| `{c['cls']}` | {name} | {p['calls']} | {p['inclusive_ms']:.2f} | "
                f"{p['inclusive_ms'] / total:.0%} | {p['instances']} | "
                f"`{p['top_signature'][:60]}` | {groups} |"
            )
    return lines


def _roofline_note() -> list[str]:
    """Measured peaks + how to read headroom from them (empty when not measured)."""
    from kernel_agent import toolchain

    peaks = toolchain.setup().peaks
    if not peaks:
        return []
    gbps = float(peaks["dram_gbps"])
    return [
        "",
        "## Roofline (peaks measured on this GPU)",
        "",
        f"{toolchain.format_peaks(peaks)}. A module call takes at least max(FLOPs / peak, "
        "bytes it must read and write / bandwidth), and at least the launch floor if it "
        "launches anything (inside a CUDA-graph replay the graph one: no host time per "
        "launch). Weights are read once per call, so a class with P bf16 "
        f"parameters needs ≥ 2·P bytes / {gbps:.0f} GB/s per call (≈ {2e3 / gbps:.1f} us per "
        "million parameters): when *inclusive ms / calls* is far above that, there is "
        "headroom. Kernel "
        "engineers get `sol_ms` / `pct_of_sol` per captured case.",
    ]
