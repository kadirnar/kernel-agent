"""Fusion candidates: measured chains of ops across module boundaries (issue #231).

A module target fuses what one ``nn.Module`` does; the steps that paid most in our runs fused
glue *between* modules (a residual add and the next norm, a gate · up product, the update of
a solver step), and some fusions bought nothing because their intermediates stayed in L2.
This module measures those chains instead of leaving them to the planner's reading of the
source. ``analyze`` runs the model once more in the hooked work pass
(:func:`~kernel_agent.profiling.profiler.work_reference`: module calls without timing; the
unmodified model, and in an improve round's re-profile the optimised one) under a
``TorchDispatchMode`` (:class:`Recorder`) that records every aten op:

* its name, the storages it reads and writes (views resolved to their storage) and their
  bytes, and the innermost module call around it (qualname, class, entrypoint, phase); a
  composite op is recorded as the kernels it decomposes into (as the flop counter does),
  except the GEMMs, convolutions and attention (*anchors*), which stay one op each;
* producer → consumer edges from the storages: an op that reads a storage depends on the
  last op that wrote it (an in-place op reads and writes it; a fresh allocation forgets the
  storage's earlier writer). Views and allocations launch nothing and are not ops here.

Chains (:meth:`Miner.chains`): the memory-bound ops (element-wise, norms, reductions, casts,
copies) grouped into maximal sets that can be one kernel: an op joins the group of an op it
reads from when no path from that group to it runs through an op outside the group (the
fused kernel would need that op's result first), within :data:`MAX_GAP` ops of the producer
(a loop's state carried to the next step is not a fusion) and never across a host
synchronisation (``.item()``, ``nonzero``, a device-to-host copy: the host needs the value
before it launches what follows). Each chain is measured in three *placements*:

* ``chain``: its ops as one kernel;
* ``epilogue``: inside the anchor(s) it reads from (several only when they read a common
  input: one merged GEMM, e.g. gate and up projections);
* ``prologue``: inside the anchor(s) that read its output (recomputed per anchor, e.g. a norm
  in front of the q / k / v projections).

Per placement: the ops in order with their module (layer indices folded), the launches now
and saved (the placement becomes one kernel; the kernel map below), the intermediates written
and read back (a tensor one op of the placement writes and another reads: its write, unless
an op outside also reads it, and those reads), the module boundaries it crosses (distinct
module calls − 1) and the lowest common ancestor call: its class is a region target's
``parent_class``, its ops the ``region`` description. Chains with the same ops in the same
instance group and phase are one row (layers deduplicated): ``calls`` occurrences per run,
``instances`` distinct parents.

Estimate (:func:`build`, with the peaks measured on the GPU at hand)::

    saves = Σ DRAM bytes of the intermediates' round trips / DRAM bandwidth
            + calls × launches saved × per-launch cost

The L2 rule (:func:`l2_round_trip`), LRU-ish: the L2 (the GPU's,
:class:`kernel_agent.toolchain.GPUInfo`) is taken to hold the newest bytes touched. The ops
between an intermediate's write and its last read inside the placement touch ``traffic``
distinct bytes (what they read, weights included, and write, outside its storage; per
storage the most bytes one op touches in it: :meth:`Miner._traffic`), so ``min(its bytes,
L2 − traffic)`` of it is still in the L2 at that read and the rest counts, as that share of
its round trip. An intermediate that fits with its traffic saves no DRAM bytes (``in``); a
small one that the traffic evicts counts (``traffic``); a large one read at once counts the
part beyond the L2 (``size``); with the L2 unknown every one counts (``unknown``, an upper
bound). Per placement, ``intermediates`` records each one's bytes, traffic, DRAM bytes and
why (``fusions.md`` says it). Not the hardware's replacement policy: the reads in between
are not taken to refresh it, the L2 is one pool of its full size, and work the recorder
does not see (another process, a graph replay) is not traffic.

The per-launch cost is the measured launch floor when the run launches eagerly, at most the
run's own time per GPU launch (window ÷ launches; per recorded op without a kernel map: the
floor times a module call, whose Python a bare op does not pay), and the measured CUDA-graph
launch floor (``launch_floor_graph_us``; :data:`GRAPH_BOUNDARY_US` for peaks without one) per
kernel boundary when the kernel view shows most GPU work launched by CUDA graphs and no
kernel map says that the mined ops launch eagerly.

Overlap-aware ranking (:func:`build`): rows can share an op, an anchor in one's prologue
and another's epilogue (a norm before q / k / v and the q norm after q), and one GEMM
cannot take in both, so their savings do not add up. :meth:`Miner.chains` records per
placement the other rows whose placements take in one of its ops in some occurrence
(``overlaps``); :func:`_disjoint` takes the placements of every row greedily by saving,
each when its row has none yet and it shares no op with one taken (a row whose best
placement is taken in elsewhere takes its next one: ``blocked``). The rows taken are
``counted``: they share no op, their savings add up, ranked first by saving. A row with
none taken is an alternative of the rows it overlaps (``overlaps``), ranked after them.
Greedy and deterministic, not the best combination in general.

Kernel map (:func:`kernel_map`; on a GPU): the pass runs under the profiler (Kineto: the
GPU's work, the launch calls and, on the host, only user-scope ranges, not every aten op:
:func:`_kernel_profiler`) and the recorder runs each op it dispatches in a range
(:data:`OP_RANGE` + its dispatch number), each module call in another (:data:`CALL_RANGE` +
its index). A GPU event (kernel, copy, memset) belongs to the innermost op range around the
host call that launched it (its correlation id), so every recorded op carries its exact
``launches`` (0 for an op on host tensors; several for an op that launches several kernels)
and the GPU time of its kernels (``kernel_ns``, measured in this pass: each kernel runs alone
while the recorder works).
A placement's launches are then the sum over its ops, and it becomes one kernel, or as
many as its largest anchor launches (a split-K reduction stays); its ``kernel_ns`` is what
its kernels take now. :func:`build` prices an eager launch at most at the run's time per
GPU launch (window ÷ launches, not ÷ recorded ops) and caps a placement's byte saving at
the measured time of its kernels (fusing cannot save more GPU time than they take). On the
CPU, or when the profiler fails, each recorded op counts one launch (``kernel_map`` absent,
the table says so). GPU work launched between two recorded ops outside every op range (a
graph replay, a Triton kernel: the recorder saw neither what it read nor what it wrote)
separates them like a host sync: no chain crosses it (``launch_cuts``).

Not mined (:meth:`Miner.unmined`): a CUDA-graph replay launches its kernels without
dispatching an op, and so do Triton kernels, extensions and other kernels launched outside
the dispatcher; compiled code does not run under a dispatch mode at all (Dynamo runs the
module's Python eagerly instead), so its ops are recorded but are not what the run does.
The ops inside a ``torch.compile``'d module call are kept as barriers (never in a chain),
and the kernel map attributes the GPU work launched outside every op range to the innermost
module call around its launch: ``unmined`` lists each such region (``graphed``,
``compiled``, ``custom`` kernels) with its GPU events and time; :func:`build` gives each its
share of the run (of the pass's GPU time; a compiled region by the profile's module view,
its eager kernels being no measure of the compiled code), and without a kernel map the
regions of the profile's ``module_gaps`` and its graph-launched timeline stages.

``analyze`` writes ``profile/fusions.json`` + ``fusions.md`` and the top :data:`TOP` counted
rows with their alternatives into ``summary.md`` (*Fusion candidates (measured)*), and so
does every improve round's re-profile of the optimised model (``rounds/<n>/profile/``): its
table lists what is left. The improve scheduler takes a region arm's expected gain from its
candidate in the newest table (``fusion`` id in the plan, else the largest of its parent
class, counted rows first: :func:`match`), and the native stage graph takes chains that
span stages as evidence for a group, adding up only the savings of chains that share no op
(:func:`additive`, ``native/engine.py``). Candidates are evidence: the planner decides.
Tables from before the traffic and the overlaps were recorded (``fusions.json`` version 1)
stay readable: every row counts; tables from before the kernel map (version 2) count one
launch per op.

Recorder cost (:class:`Recorder`): the dispatch mode itself costs ~25 us per op in Python
(a pass-through mode, measured on the CPU of the A10 host); the bookkeeping, 55 → 43 us per
op on Qwen3-0.6B (measured on an NVIDIA A10, torch 2.10), is kept small:
per-overload facts looked up once (:class:`_Func`), each tensor's storage looked up once per
op, no Dynamo wrapper around ``__torch_dispatch__`` (torch ≥ 2.11 adds one; Dynamo skips
every frame while this mode is active anyway), and the cyclic GC collecting young objects
less often during the pass (the recorded ops are long-lived; cyclic garbage is still
collected). The recorded ops are the same as before (``tests/test_fusion.py`` compares
them with the straightforward recorder).

Approximate: bytes are the logical bytes of each tensor (at most its storage's); state a
module keeps outside its arguments is a storage like any other; a kernel launched on
another thread at the time of an op range counts for that op; the profiler may lose a
GPU event now and then (1 in 40,000 on a toy, measured on an NVIDIA A10): that op counts
one launch less.

    python -m kernel_agent.profiling.fusion profile.json --window-ms MS [--peaks peaks.json]
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import gc
import hashlib
import itertools
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kernel_agent.kernels.roofline import GRAPH_LAUNCH_FLOOR_US
from kernel_agent.profiling import timeline
from kernel_agent.projection import fold

# 2: the L2 traffic of each intermediate, overlaps and counted rows; 3: the kernel map (exact
# launches and kernel times per op), regions not mined
VERSION = 3
#: Ops between a producer and its consumer beyond which they are not linked: a value a loop
#: carries to its next step (a position counter) is not a fusion.
MAX_GAP = 64
#: A kernel boundary inside a CUDA graph (drain, launch, ramp-up of the next grid): ~0.9 us,
#: measured on an RTX 5070 Ti (docs/PARALLEL.md §4.6), used where the peaks have no measured
#: CUDA-graph launch floor (the roofline's stand-in for it).
GRAPH_BOUNDARY_US = GRAPH_LAUNCH_FLOOR_US
#: Candidates shown in ``summary.md`` (``fusions.md`` shows :data:`SHOWN`).
TOP = 10
SHOWN = 40
#: Placements, in the order a tie prefers them.
PLACEMENTS = ("chain", "epilogue", "prologue")
#: Profiler ranges of the kernel map (:func:`kernel_map`): around each op the recorder
#: dispatches (+ its dispatch number) and each module call of the pass (+ its index). Not
#: ``ka::``: the timeline (:mod:`.timeline`) takes those for stages.
OP_RANGE = "ka.op/"
CALL_RANGE = "ka.call/"
#: Regions not mined (:meth:`Miner.unmined`), by how their GPU work escaped the recorder.
UNMINED = ("graphed", "compiled", "custom", "unattributed")
#: The cyclic GC's young-generation threshold during the pass: the recorded ops are long
#: lived, and collecting every 700 allocations cost ~10 % of the pass (measured on a toy
#: decoder on the CPU of the A10 host); cyclic garbage still goes every 50,000.
GC_YOUNG = 50_000

MEM, ANCHOR, BARRIER = "mem", "anchor", "barrier"
_GEMM = frozenset(
    {
        "mm",
        "addmm",
        "bmm",
        "baddbmm",
        "addbmm",
        "matmul",
        "linear",
        "mv",
        "addmv",
        "_addmm_activation",
        "_scaled_mm",
        "_scaled_grouped_mm",
        "_grouped_mm",
        "_int_mm",
        "_weight_int8pack_mm",
        "_weight_int4pack_mm",
    }
)
_CONV_PREFIXES = (
    "conv",
    "_conv",
    "cudnn_conv",
    "miopen_conv",
    "mkldnn_conv",
    "slow_conv",
    "thnn_conv",
    "_slow_conv",
    "_mps_conv",
)
#: Allocations: they launch nothing, and what their storage held before is forgotten.
_ALLOC = frozenset(
    {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "empty_permuted"}
)
#: Ops that make the host wait for the device (their result decides what runs next).
_SYNC = frozenset(
    {
        "_local_scalar_dense",
        "item",
        "is_nonzero",
        "equal",
        "nonzero",
        "argwhere",
        "masked_select",
        "unique",
        "_unique",
        "_unique2",
        "unique_dim",
        "unique_consecutive",
    }
)
_TRANSFER = frozenset({"_to_copy", "copy_", "_copy_from", "_copy_from_and_resize", "to"})
#: In-place ops that overwrite their first argument without reading it.
_OVERWRITE = frozenset(
    {
        "copy_",
        "fill_",
        "zero_",
        "normal_",
        "uniform_",
        "random_",
        "bernoulli_",
        "exponential_",
        "index_put_",
        "_index_put_impl_",
        "index_copy_",
        "index_fill_",
        "scatter_",
        "masked_scatter_",
        "put_",
    }
)
#: In-place ops that write only the elements their other arguments name or carry.
_PARTIAL = frozenset(
    {
        "index_put_",
        "_index_put_impl_",
        "index_copy_",
        "index_add_",
        "index_fill_",
        "scatter_",
        "scatter_add_",
        "scatter_reduce_",
        "masked_scatter_",
        "put_",
    }
)
_DTYPES = {
    "bfloat16": "bf16",
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "float8_e4m3fn": "fp8",
    "int64": "i64",
    "int32": "i32",
    "bool": "bool",
}

Key = tuple[int, int]  # device index (-1: the host), storage data pointer


def anchor_kind(name: str) -> str | None:
    """``gemm`` / ``conv`` / ``attention`` for an aten op that a fusion keeps as one kernel
    and can take memory-bound ops into (None: any other op)."""
    if name in _GEMM:
        return "gemm"
    if name.startswith(_CONV_PREFIXES) and "convert" not in name:
        return "conv"
    if "attention" in name:
        return "attention"
    return None


@dataclass(frozen=True, slots=True)
class _Func:
    """What the recorder needs of an op overload, looked up once per overload."""

    name: str
    aten: bool
    anchor: str | None
    composite: bool  # it decomposes into other aten ops
    written: tuple[int, ...]  # positional arguments it writes (in place)
    written_kw: tuple[str, ...]  # keyword-only ones (``out=``)
    alloc: bool = False  # _ALLOC
    sync: bool = False  # _SYNC
    transfer: bool = False  # _TRANSFER
    overwrite: bool = False  # _OVERWRITE
    partial: bool = False  # _PARTIAL


_FUNCS: dict[Any, _Func] = {}


def _func(func: Any) -> _Func:
    found = _FUNCS.get(func)
    if found is None:
        name = func.overloadpacket.__name__
        aten = func.namespace == "aten"
        anchor = anchor_kind(name) if aten else None
        key = torch._C.DispatchKey.CompositeImplicitAutograd
        try:
            composite = key in func.py_kernels or torch._C._dispatch_has_kernel_for_dispatch_key(
                func.name(), key
            )
        except Exception:
            composite = True  # ``decompose`` decides
        args = func._schema.arguments
        written = [i for i, a in enumerate(args) if a.alias_info and a.alias_info.is_write]
        found = _FUNCS[func] = _Func(
            name,
            aten,
            anchor,
            aten and anchor is None and composite,
            tuple(i for i in written if not args[i].kwarg_only),
            tuple(args[i].name for i in written if args[i].kwarg_only),
            name in _ALLOC,
            name in _SYNC,
            name in _TRANSFER,
            name in _OVERWRITE,
            name in _PARTIAL,
        )
    return found


# a tuple, not ``tuple | list``: that union is built anew at each isinstance call
_SEQ = (tuple, list)
_TENSOR = torch.Tensor


def _tensors(values: Any) -> list[torch.Tensor]:
    """The tensors in ``values`` (a tensor, or a sequence of tensors and of lists of them, as
    aten arguments and outputs are)."""
    if isinstance(values, _TENSOR):
        return [values]
    out = []
    if isinstance(values, _SEQ):
        for v in values:
            if isinstance(v, _TENSOR):
                out.append(v)
            elif isinstance(v, _SEQ):
                out += [x for x in v if isinstance(x, _TENSOR)]
    return out


def _storage(t: torch.Tensor) -> tuple[Key | None, int]:
    """The storage of ``t`` (None: none to track, e.g. a meta or empty tensor) and the bytes
    ``t`` covers in it: its logical bytes, at most the span its strides reach (a broadcast
    reads one copy)."""
    try:
        ptr = t.data_ptr()
    except Exception:  # no storage (a sparse or wrapper tensor)
        return None, 0
    if not ptr:
        return None, 0
    size = t.element_size()
    n = t.numel()
    if not t.is_contiguous():
        span = 1 + sum((s - 1) * abs(st) for s, st in zip(t.shape, t.stride(), strict=True))
        n = min(n, span)
    return (t.get_device(), ptr - int(t.storage_offset()) * size), n * size


def _record_function_enter(name: str) -> Any:
    rf = torch.autograd.profiler.record_function(name)
    rf.__enter__()
    return rf


def _record_function_exit(handle: Any) -> None:
    handle.__exit__(None, None, None)


#: Open and close a profiler range in the user scope (``record_function``'s), through the
#: direct bindings where torch has them (no op to dispatch, no object per range): the pass's
#: profiler records only that scope (:func:`_kernel_profiler`).
_range_enter: Callable[[str], Any] = getattr(
    torch.autograd, "_record_function_with_args_enter", _record_function_enter
)
_range_exit: Callable[[Any], None] = getattr(
    torch.autograd, "_record_function_with_args_exit", _record_function_exit
)


class _Range:
    """A profiler range named ``name`` (:func:`_range_enter`), a context manager."""

    __slots__ = ("handle", "name")

    def __init__(self, name: str) -> None:
        self.name = name
        self.handle: Any = None

    def __enter__(self) -> None:
        self.handle = _range_enter(self.name)

    def __exit__(self, *exc: object) -> None:
        _range_exit(self.handle)


def _call_range(index: int) -> Any:
    return _Range(f"{CALL_RANGE}{index}")


def _shape(shape: Sequence[int], dtype: Any) -> str:
    if dtype is None:
        return ""
    name = str(dtype).removeprefix("torch.")
    return "x".join(str(s) for s in shape) + " " + _DTYPES.get(name, name)


@dataclass(slots=True)
class _Op:
    """One recorded op (a kernel launch)."""

    name: str
    kind: str  # MEM | ANCHOR | BARRIER
    call: int  # innermost module call (index into the timer's calls), -1: none
    epoch: int  # host synchronisations before it
    shape: tuple[int, ...]  # of its first output
    dtype: Any
    anchor: str = ""  # gemm | conv | attention
    reads: list[tuple[int, Key, int]] = field(default_factory=list)  # producer, storage, bytes
    # every storage it reads (weights and inputs too, a written-only argument not) and the
    # bytes: an anchor's inputs (which anchors share one), the L2 traffic of a window
    loads: tuple[tuple[Key, int], ...] = ()
    writes: dict[Key, int] = field(default_factory=dict)  # storage -> bytes written
    consumers: list[tuple[int, Key, int]] = field(default_factory=list)  # op, storage, bytes
    # the kernel map (Miner._apply): GPU events it launched (-1: no map) and their GPU time
    launches: int = -1
    kernel_ns: int = 0
    hidden: bool = False  # inside a compiled module call: not what the run does (Miner._hide)


class Recorder(TorchDispatchMode):
    """Records every aten op of a run (:class:`_Op`) while active; never changes one.
    ``stack``: the open module calls, innermost last, each with its index into the timer's
    calls first (``ModuleTimer._stack``). ``ranges``: each op it dispatches runs inside a
    profiler range named :data:`OP_RANGE` + its dispatch number, for :func:`kernel_map`;
    ``seq`` maps dispatch numbers to ops.

    Kept cheap (the module docstring): it records the same ops as a straightforward
    recorder, which ``tests/test_fusion.py`` checks."""

    @classmethod
    def _should_skip_dynamo(cls) -> bool:
        # torch >= 2.11 wraps __torch_dispatch__ in torch._disable_dynamo (a few us per op);
        # Dynamo already skips every frame while a mode like this one is active
        return False

    def __init__(
        self, stack: list[tuple[int, int, str]] | None = None, *, ranges: bool = False
    ) -> None:
        super().__init__()
        self.stack = stack if stack is not None else []
        self.ranges = ranges
        self.ops: list[_Op] = []
        self.writer: dict[Key, int] = {}
        self.epoch = 0
        self.views = 0
        self.failed = 0  # ops the bookkeeping could not record (the run went on)
        self.dispatched = 0  # ops run in a range (their dispatch numbers)
        self.seq: dict[int, int] = {}  # dispatch number -> index of the op recorded

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        info = _FUNCS.get(func) or _func(func)
        if info.composite:
            # record the kernels it runs (re-entering the mode for them). An error
            # propagates: running the op again could repeat an in-place update.
            with self:
                out = func.decompose(*args, **kwargs)
            if out is not NotImplemented:
                return out
        seq = -1
        if not self.ranges:
            out = func(*args, **kwargs)
        else:  # the kernels it launches are launched in its range (kernel_map)
            seq = self.dispatched
            self.dispatched = seq + 1
            handle = _range_enter(f"{OP_RANGE}{seq}")
            try:
                out = func(*args, **kwargs)
            finally:
                _range_exit(handle)
        try:
            self._record(info, args, kwargs, out, seq)
        except Exception:  # the bookkeeping never breaks the run
            self.failed += 1
        return out

    def _record(
        self, info: _Func, args: tuple[Any, ...], kwargs: dict[str, Any], out: Any, seq: int = -1
    ) -> None:
        if info.written or info.written_kw:
            written = _tensors([args[i] for i in info.written if i < len(args)])
            written += _tensors([kwargs[k] for k in info.written_kw if k in kwargs])
        else:
            written = []
        ins = _tensors(args)
        if kwargs:
            ins += _tensors(list(kwargs.values()))
        outs = _tensors(out)
        out_st = [_storage(t) for t in outs]
        writer = self.writer
        if info.alloc:
            for key, _ in out_st:
                writer.pop(key or (0, 0), None)
            return
        in_st = [_storage(t) for t in ins]
        if not written and outs:
            in_keys = {key for key, _ in in_st}
            if all(key in in_keys for key, _ in out_st):
                self.views += 1  # a view (or an op that returned its input): no kernel
                return
        if not outs and not written and not info.sync:
            return  # sizes, metadata
        cross = to_host = False
        if info.transfer:
            devices = {t.get_device() for t in (*ins, *outs) if t.dim() or not t.is_cpu}
            cross = len(devices) > 1
            to_host = cross and any(t.is_cpu for t in outs)
        sync = info.sync or to_host
        # a host synchronisation, a transfer, a custom op: never fused
        barrier = sync or cross or not info.aten
        kind = BARRIER if barrier else ANCHOR if info.anchor else MEM
        overwrites = info.overwrite and written  # its written argument is not read
        loads: dict[Key, int] = {}
        reads: list[tuple[int, Key, int]] = []
        for t, (key, nbytes) in zip(ins, in_st, strict=True):
            if key is None or (overwrites and any(t is w for w in written)):
                continue
            known = loads.get(key)
            if known is None or nbytes > known:
                loads[key] = nbytes
            producer = writer.get(key)
            if producer is not None:
                reads.append((producer, key, nbytes))
        writes: dict[Key, int] = {}
        targets = list(zip(outs, out_st, strict=True))
        if written:  # an in-place op: its storages were looked up as inputs
            looked = {id(t): st for t, st in zip(ins, in_st, strict=True)}
            targets += [(t, looked.get(id(t)) or _storage(t)) for t in written]
        carried = None
        mutated: set[int] = set()
        if info.partial:  # it writes the rows its other arguments carry, not the buffer
            mutated = {id(t) for t in written}
            carried = sum(n for t, (_, n) in zip(ins, in_st, strict=True) if id(t) not in mutated)
        for t, (key, nbytes) in targets:
            if key is None:
                continue
            if carried is not None and id(t) in mutated:
                nbytes = min(nbytes, carried)
            known = writes.get(key)
            if known is None or nbytes > known:
                writes[key] = nbytes
        first = outs[0] if outs else (written[0] if written else None)
        stack = self.stack
        ops = self.ops
        index = len(ops)
        op = _Op(
            info.name,
            kind,
            stack[-1][0] if stack else -1,
            self.epoch,
            first.shape if first is not None else (),  # a torch.Size: a tuple
            first.dtype if first is not None else None,
            info.anchor or "",
            reads,
            tuple(loads.items()),
            writes,
            [],
        )
        # committed only once the op is complete: a failure above leaves no half an op
        for producer, key, nbytes in reads:
            ops[producer].consumers.append((index, key, nbytes))
        for key in writes:
            writer[key] = index
        ops.append(op)
        if seq >= 0:
            self.seq[seq] = index
        if sync:
            self.epoch += 1


# ------------------------------------------------------------------ chains


class _Groups:
    """Union-find over the memory-bound ops, with each group's last member and the groups
    it depends on (through any path)."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.last: dict[int, int] = {}
        self.deps: dict[int, set[int]] = {}

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def resolve(self, groups: set[int]) -> set[int]:
        return {self.find(g) for g in groups}


#: A placement's numbers that add up over a row's occurrences (:meth:`Miner.chains`).
_SUMMED = (
    "intermediate_bytes",
    "round_trip_bytes",
    "dram_bytes",
    "tensors",
    "launches_run",
    "launches_saved_run",
    "kernel_ns",
)

#: Why an intermediate's round trip counts (:func:`l2_round_trip`), from none to all of it.
L2_WHY = ("in", "traffic", "size", "unknown")


def l2_round_trip(written: int, trip: int, traffic: int, l2: int | None) -> tuple[int, str]:
    """The DRAM bytes of an intermediate's round trip (``trip``: its write and its reads back,
    ``written`` its bytes) and why (:data:`L2_WHY`), with ``traffic`` the distinct bytes
    other ops touch between its write and its last read (:meth:`Miner._traffic`).

    LRU-ish: the L2 is taken to hold the newest ``l2`` bytes touched, so at its last read
    ``min(written, l2 − traffic)`` bytes of it are still there; the rest of it, as a share of
    the round trip, goes through DRAM. ``in``: all of it is there (0 bytes); ``traffic``: it
    fits the L2 alone, the traffic evicted all or part of it; ``size``: it is larger than
    the L2 (the part the L2 holds still counts as there); ``unknown``: no L2 known, all of
    it counts (an upper bound). Not the hardware's replacement policy: an approximation
    from the bytes, with reads in between not counted as refreshing it."""
    if l2 is None:
        return trip, "unknown"
    resident = min(written, max(l2 - traffic, 0))
    if written <= 0 or resident >= written:
        return 0, "in"
    return -(-trip * (written - resident) // written), "size" if written > l2 else "traffic"


#: Kind of this module's ranges in :func:`profiled_events`.
RANGE = "range"


@dataclass
class KernelMap:
    """The GPU events of a profiled pass by the op range that launched them
    (:func:`kernel_map`)."""

    #: dispatch number -> (GPU events launched inside its op range, their GPU ns)
    ops: dict[int, tuple[int, int]]
    #: (kind (:data:`UNMINED`), innermost module call around the launch, -1: none) -> the same
    unmined: dict[tuple[str, int], tuple[int, int]]
    events: int = 0  # every GPU event of the pass
    ns: int = 0
    #: GPU events in the range of a dispatch that recorded no op (Miner.map_kernels)
    unrecorded: tuple[int, int] = (0, 0)
    #: Dispatch numbers before which GPU work was launched outside every op range (a graph
    #: replay, a Triton kernel), sorted: the recorder saw neither what it read nor what it
    #: wrote, so no chain crosses one (Miner.map_kernels)
    cuts: list[int] = field(default_factory=list)


def _innermost(ranges: list[tuple[int, int, int]]) -> Callable[[int], int | None]:
    """The key of the innermost range around a time (None: none), for properly nested
    ``(start, end, key)`` ranges."""
    segs = timeline.flatten(ranges)
    starts = [s for s, _, _ in segs]

    def at(t: int) -> int | None:
        j = bisect.bisect_right(starts, t) - 1
        return segs[j][2] if j >= 0 and segs[j][0] <= t < segs[j][1] else None

    return at


def kernel_map(events: Iterable[timeline.Event]) -> KernelMap:
    """Each GPU event (kernel, copy, memset; ``timeline.GPU_KINDS``) of a profiled pass to
    the op whose range (:data:`OP_RANGE` + its dispatch number) is the innermost around the
    host call that launched it (the launch call with its correlation id), and one launched
    outside every op range to how (``graphed``: by a CUDA-graph launch, ``custom``: by any
    other launch outside the dispatcher: Triton, an extension; ``unattributed``: no launch
    call in the trace) and the innermost module call (:data:`CALL_RANGE` + its index) around
    the launch. Ranges are matched by time on any thread: the profiler numbers the threads
    of its ranges and of its launch calls differently, and the recorder runs on one."""
    op_ranges: list[tuple[int, int, int]] = []
    call_ranges: list[tuple[int, int, int]] = []
    gpu: list[timeline.Event] = []
    api: dict[int, timeline.Event] = {}
    for e in events:
        if e.kind == RANGE:
            for prefix, found in ((OP_RANGE, op_ranges), (CALL_RANGE, call_ranges)):
                if e.name.startswith(prefix):
                    with contextlib.suppress(ValueError):
                        found.append((e.start, e.end, int(e.name[len(prefix) :])))
        elif e.kind in timeline.GPU_KINDS:
            gpu.append(e)
        elif e.kind in timeline.API_KINDS and e.correlation >= 0:
            api[e.correlation] = e
    op_at, call_at = _innermost(op_ranges), _innermost(call_ranges)
    op_ranges.sort()
    starts = [s for s, _, _ in op_ranges]
    per_op: dict[int, list[int]] = {}
    outside: dict[tuple[str, int], list[int]] = {}
    cuts: set[int] = set()
    total = 0
    for g in gpu:
        ns = max(g.end - g.start, 0)
        total += ns
        launch = api.get(g.correlation)
        seq = op_at(launch.start) if launch is not None else None
        if seq is not None:
            entry = per_op.setdefault(seq, [0, 0])
        elif launch is None:
            entry = outside.setdefault(("unattributed", -1), [0, 0])
        else:
            kind = "graphed" if "Graph" in launch.name else "custom"
            call = call_at(launch.start)
            entry = outside.setdefault((kind, call if call is not None else -1), [0, 0])
            k = bisect.bisect_left(starts, launch.start)  # the first op range after it
            if k < len(op_ranges):  # (after the last one it separates nothing)
                cuts.add(op_ranges[k][2])
        entry[0] += 1
        entry[1] += ns
    return KernelMap(
        {k: (v[0], v[1]) for k, v in per_op.items()},
        {k: (v[0], v[1]) for k, v in outside.items()},
        len(gpu),
        total,
        cuts=sorted(cuts),
    )


def profiled_events(prof: Any) -> list[timeline.Event]:
    """The events :func:`kernel_map` reads from a finished profiler (``torch.profiler``'s or
    the autograd one of :func:`_kernel_profiler`): GPU work and launch calls
    (``timeline.activity_type``) and this module's ranges on the host (kind :data:`RANGE`;
    a user range's GPU-side span is not one), from its Kineto results (no chrome trace)."""
    results = getattr(prof, "kineto_results", None)
    if results is None:  # torch.profiler.profile: the autograd profiler inside
        results = prof.profiler.kineto_results
    from torch._C._autograd import DeviceType

    keep = timeline.GPU_KINDS | timeline.API_KINDS
    ranges = (OP_RANGE, CALL_RANGE)
    host = DeviceType.CPU
    event = timeline.Event
    out: list[timeline.Event] = []
    append = out.append
    # half a million events for Qwen3-0.6B: the cheap tests first (activity_type reads
    # three or four fields on a torch without _KinetoEvent.activity_type)
    for e in results.events():
        name = e.name()
        on_host = e.device_type() == host
        if name.startswith(ranges):
            if on_host:  # not Kineto's GPU span of the range
                append(event(RANGE, name, e.start_ns(), e.end_ns()))
            continue
        if on_host and not name.startswith(_LAUNCH_CALLS):
            continue  # an aten op, cudaGetDevice, cudaStreamIsCapturing, ...
        kind = timeline.activity_type(e)
        if kind in keep:
            append(event(kind, name, e.start_ns(), e.end_ns(), e.correlation_id()))
    return out


#: Host calls that put work on the GPU (CUDA runtime and driver): the launches of
#: :func:`kernel_map` (a GPU event whose launch is none of these is ``unattributed``).
_LAUNCH_CALLS = (
    "cudaLaunch",
    "cudaGraphLaunch",
    "cudaMemcpy",
    "cudaMemset",
    "cuLaunch",
    "cuGraphLaunch",
    "cuMemcpy",
    "cuMemset",
)


def _kernel_profiler() -> Any:
    """The profiler of a pass with a kernel map: the host's user-scope ranges (this
    module's; every aten op too made the Qwen3-0.6B pass 25.7 s instead of 19.1 s, measured
    on an NVIDIA A10), the launch calls and the GPU's work; no shapes, stacks or memory, and
    no external correlation (the map pairs launches and kernels by CUPTI's own correlation
    id; Kineto's GPU spans of the ranges were a quarter of the events). Falls back to
    torch's defaults where its profiler differs."""
    from torch._C._profiler import RecordScope, _ExperimentalConfig
    from torch.autograd import _enable_profiler
    from torch.autograd.profiler import profile

    class UserScope(profile):
        def _start_trace(self) -> None:  # torch's, with the scopes
            self.entered = True
            config = self.config(create_trace_id=False)
            try:
                scopes = {RecordScope.USER_SCOPE}  # the binding takes them; its stub does not
                _enable_profiler(config, self.kineto_activities, scopes)  # type: ignore[call-arg]
            except TypeError:  # a torch without scopes: every op too
                _enable_profiler(config, self.kineto_activities)
            self.profiling_start_time_ns = time.perf_counter_ns()

    try:
        experimental = _ExperimentalConfig(disable_external_correlation=True)
    except TypeError:
        experimental = None
    return UserScope(
        use_device="cuda" if torch.cuda.is_available() else None,
        use_kineto=True,
        experimental_config=experimental,
    )


@contextlib.contextmanager
def _young_gc_less_often() -> Iterator[None]:
    """The cyclic GC's young generation collected every :data:`GC_YOUNG` allocations (not
    700) while active, unless it is disabled; restored after."""
    old = gc.get_threshold()
    if old[0]:
        gc.set_threshold(max(old[0], GC_YOUNG), max(old[1], 50), old[2])
    try:
        yield
    finally:
        gc.set_threshold(*old)


def _error(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]


class Miner:
    """Records a run's ops in the hooked work pass (:meth:`recording`) and finds its fusion
    chains (:meth:`chains`, :meth:`result`). ``l2_bytes``: the GPU's L2 (None: unknown,
    every intermediate counts as DRAM traffic) and where that number is from. ``kernels``:
    map the ops to their kernels (:meth:`recording`; default: with CUDA)."""

    def __init__(
        self,
        l2_bytes: int | None = None,
        l2_source: str = "",
        *,
        max_gap: int = MAX_GAP,
        kernels: bool | None = None,
    ) -> None:
        self.l2_bytes = l2_bytes
        self.l2_source = l2_source
        self.max_gap = max_gap
        self.want_kernels = kernels
        self.recorder: Recorder | None = None
        self.calls: list[Any] = []  # the timer's calls (qualname, cls, method, phase, parent)
        self.compiled: set[str] = set()  # qualnames of the torch.compile'd modules (the timer's)
        self.kernels: KernelMap | None = None  # the kernel map (map_kernels)
        self.kernel_note = ""  # why there is none, when one was asked for
        self.map_seconds = 0.0
        self._ancestry: dict[int, tuple[int, ...]] = {}
        self._folded: dict[int, str] = {}
        self._windows: dict[tuple[int, int], tuple[int, dict[Key, int]]] = {}
        self._inside: dict[int, str] = {}  # call -> the compiled call around it
        self._hid = False
        self._epochs: list[int] | None = None  # the ops' epochs before the launch cuts

    @contextlib.contextmanager
    def recording(self, timer: Any, *, kernels: bool | None = None) -> Iterator[Recorder]:
        """Record every op while active, each with the innermost open call of ``timer`` (a
        ``ModuleTimer`` that is active around this). ``kernels`` (default: the miner's, else
        with CUDA): run the pass under the profiler (:func:`_kernel_profiler`) with the op and
        module-call ranges and map the ops to their kernels at the end (:meth:`map_kernels`),
        also when the run fails."""
        if kernels is None:
            kernels = self.want_kernels
        mapped = torch.cuda.is_available() if kernels is None else kernels
        self.recorder = Recorder(timer._stack, ranges=mapped)
        self.calls = timer.calls
        self.compiled = set(getattr(timer, "compiled", ()))
        self.kernel_note = "" if mapped else "no CUDA: a pass on the CPU"
        prof = None
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(_young_gc_less_often())
                if mapped:
                    try:
                        prof = stack.enter_context(_kernel_profiler())
                    except Exception as exc:  # another profiler is active, say: no map
                        self.kernel_note = f"the profiler did not start: {_error(exc)}"
                        self.recorder.ranges = False
                    else:
                        timer.call_range = _call_range
                        stack.callback(setattr, timer, "call_range", None)
                with self.recorder:
                    yield self.recorder
        finally:
            if prof is not None:
                start = time.perf_counter()
                try:
                    with _young_gc_less_often():  # hundreds of thousands of events
                        self.map_kernels(profiled_events(prof))
                except Exception as exc:  # the map never breaks the pass
                    self.kernels, self.kernel_note = None, f"the kernel map failed: {_error(exc)}"
                self.map_seconds = time.perf_counter() - start
            self._hide()

    @property
    def ops(self) -> list[_Op]:
        return self.recorder.ops if self.recorder is not None else []

    def map_kernels(self, events: Iterable[timeline.Event]) -> KernelMap | None:
        """Each recorded op's ``launches`` and ``kernel_ns`` from the events of its pass
        (:func:`kernel_map`; the recorder ran with ranges): 0 for an op whose range launched
        nothing. A pass without GPU events (a workload on the CPU) maps nothing: every op
        then counts one launch. GPU work launched between two recorded ops outside the
        dispatcher (a graph replay, a Triton kernel) separates them as a host sync does
        (``epoch``): the recorder did not see what it read and wrote, so a chain across it
        could skip a dependency (a residual add, a Triton norm, the next add)."""
        km = kernel_map(events)
        rec = self.recorder
        if not km.events or rec is None:
            self.kernels = None
            self.kernel_note = "no GPU work in the pass (a workload on the CPU)"
            return None
        ops = rec.ops
        if self._epochs is None:  # the host syncs' epochs, before any cut
            self._epochs = [op.epoch for op in ops]
        for seq, i in rec.seq.items():
            ops[i].epoch = self._epochs[i] + bisect.bisect_right(km.cuts, seq)
        for op in ops:
            op.launches, op.kernel_ns = 0, 0
        lost = [0, 0]
        for seq, (events_n, ns) in km.ops.items():
            index = rec.seq.get(seq)
            if index is None:  # a view, an allocation, an op the bookkeeping failed on
                lost[0] += events_n
                lost[1] += ns
                continue
            ops[index].launches, ops[index].kernel_ns = events_n, ns
        km.unrecorded = (lost[0], lost[1])
        self.kernels = km
        self.kernel_note = ""
        return km

    # -------------------------------------------------------------- compiled regions

    def _compiled_around(self, call: int) -> str:
        """The qualname of the compiled module call around ``call`` ("" for none)."""
        found = self._inside.get(call)
        if found is None:
            found = next(
                (
                    self.calls[a].qualname
                    for a in self._ancestors(call)
                    if self.calls[a].qualname in self.compiled
                ),
                "",
            )
            self._inside[call] = found
        return found

    def _hide(self) -> None:
        """The ops inside a compiled module call become barriers (``hidden``): under the
        dispatch mode its Python ran eagerly, not the compiled code the run executes."""
        if self._hid or not self.compiled:
            return
        self._hid = True
        for op in self.ops:
            if op.call >= 0 and self._compiled_around(op.call):
                op.hidden = True
                op.kind = BARRIER

    def unmined(self) -> list[dict[str, Any]]:
        """The regions of the pass whose ops were not mined (the module docstring), per kind
        (:data:`UNMINED`) and instance group: their GPU events and ms (with a kernel map)
        and, for a compiled region, the ops recorded inside it. Largest first."""
        self._hide()
        rows: dict[tuple[str, str], list[int]] = {}
        instances: dict[tuple[str, str], set[str]] = {}

        def add(kind: str, qualname: str, events: int, ns: int, ops: int) -> None:
            key = (kind, fold(qualname))
            row = rows.setdefault(key, [0, 0, 0])
            row[0] += events
            row[1] += ns
            row[2] += ops
            instances.setdefault(key, set()).add(qualname)

        for op in self.ops:
            if op.hidden:
                where = self._compiled_around(op.call)
                add("compiled", where, max(op.launches, 0), op.kernel_ns, 1)
        if self.kernels is not None:
            for (kind, call), (events, ns) in self.kernels.unmined.items():
                add(kind, self.calls[call].qualname if call >= 0 else "", events, ns, 0)
        out = [
            {
                "kind": kind,
                "where": where,
                "instances": len(instances[(kind, where)] - {""}),
                "gpu_events": n,
                "gpu_ms": round(ns / 1e6, 4),
                **({"ops": ops} if ops else {}),
            }
            for (kind, where), (n, ns, ops) in rows.items()
        ]
        return sorted(out, key=lambda r: (-r["gpu_ms"], -r.get("ops", 0), r["kind"], r["where"]))

    # -------------------------------------------------------------- grouping

    def groups(self) -> list[list[int]]:
        """The memory-bound ops in maximal fusible groups (the module docstring), each in
        execution order, groups by their first op."""
        self._hide()
        ops, gap = self.ops, self.max_gap
        g = _Groups()
        dep: list[set[int]] = []

        def active(root: int, i: int, epoch: int) -> bool:
            return g.last[root] >= i - gap and ops[root].epoch == epoch

        for i, op in enumerate(ops):
            preds = sorted({p for p, _, _ in op.reads}, reverse=True)  # the newest first
            mine: set[int] = set()
            for p in preds:
                mine |= dep[p]
            mine = {r for r in g.resolve(mine) if active(r, i, op.epoch)}
            if op.kind != MEM:
                dep.append(mine)
                continue
            chosen = []
            for p in preds:
                near = ops[p].kind == MEM and ops[p].epoch == op.epoch and i - p <= gap
                if near and (root := g.find(p)) not in chosen:
                    chosen.append(root)
            while True:  # drop groups that reach this op through an op outside them
                outside: set[int] = set()
                for q in preds:
                    if not (ops[q].kind == MEM and g.find(q) in chosen):
                        outside |= g.resolve(dep[q])
                kept: list[int] = []
                for root in chosen:
                    if root in outside:
                        continue
                    if any(
                        root in g.resolve(g.deps[h]) or h in g.resolve(g.deps[root]) for h in kept
                    ):
                        continue  # one depends on the other: they were not fused for a reason
                    kept.append(root)
                if kept == chosen:
                    break
                chosen = kept
            g.parent[i] = i
            g.deps[i] = set()
            for root in chosen:
                g.parent[root] = i
                g.deps[i] |= g.deps.pop(root)
                g.last.pop(root, None)
            g.deps[i] = {r for r in g.resolve(g.deps[i] | mine) if r != i}
            g.last[i] = i  # the newest member is the root: its epoch is the group's
            dep.append(mine | {i})
        members: dict[int, list[int]] = {}
        for i, op in enumerate(ops):
            if op.kind == MEM:
                members.setdefault(g.find(i), []).append(i)
        return sorted(members.values(), key=lambda m: m[0])

    # -------------------------------------------------------------- one chain

    def _ancestors(self, call: int) -> tuple[int, ...]:
        """The calls from the outermost to ``call`` (empty for -1)."""
        if call < 0:
            return ()
        found = self._ancestry.get(call)
        if found is None:
            found = (*self._ancestors(int(self.calls[call].parent)), call)
            self._ancestry[call] = found
        return found

    def _lca(self, members: Sequence[int]) -> int:
        """The innermost module call around every op of ``members`` (-1: none)."""
        chains = [self._ancestors(self.ops[i].call) for i in members]
        if not chains or any(not c for c in chains):
            return -1
        common = -1
        for level in zip(*chains, strict=False):
            if len(set(level)) != 1:
                break
            common = level[0]
        return common

    def _module(self, call: int) -> str:
        """The instance group of a call (its qualname, layer indices folded)."""
        found = self._folded.get(call)
        if found is None:
            found = self._folded[call] = fold(self.calls[call].qualname) if call >= 0 else ""
        return found

    def _describe(self, i: int) -> dict[str, Any]:
        op = self.ops[i]
        out: dict[str, Any] = {
            "op": op.name,
            "module": self._module(op.call),
            "shape": _shape(op.shape, op.dtype),
        }
        if op.kind == ANCHOR:
            out["anchor"] = op.anchor
        if self.kernels is not None and op.launches != 1:  # none (host tensors) or several
            out["launches"] = op.launches
        return out

    def _traffic(self, first: int, last: int, key: Key) -> int:
        """The distinct bytes the ops strictly between ``first`` and ``last`` touch (read or
        write; per storage the most bytes one of them touches in it) outside storage
        ``key``: what passes through the L2 between an intermediate's write and its last
        read. Windows are cached per chain (:meth:`chains` clears the cache)."""
        window = self._windows.get((first, last))
        if window is None:
            touched: dict[Key, int] = {}
            for op in self.ops[first + 1 : last]:
                for k, n in itertools.chain(op.loads, op.writes.items()):
                    if n > touched.get(k, 0):
                        touched[k] = n
            window = self._windows[(first, last)] = (sum(touched.values()), touched)
        total, touched = window
        return total - touched.get(key, 0)

    def _measure(self, members: Sequence[int]) -> dict[str, Any]:
        """Launches, intermediates and module boundaries of fusing ``members`` (the module
        docstring), for one occurrence; ``intermediates``: each one's bytes, round trip, the
        traffic between its write and its last read inside, and its DRAM bytes and why
        (:func:`l2_round_trip`), ``op`` its producer's position in ``members``. With a kernel
        map the launches are its ops' GPU launches (after the fusion: one kernel, or as many
        as its largest anchor launches), ``kernel_ns`` their GPU time, and the outputs of an
        op that launched no GPU work (host tensors) move no GPU bytes; without, one launch per
        op."""
        inside = set(members)
        l2 = self.l2_bytes
        ops = self.ops
        mapped = self.kernels is not None
        if mapped:  # exact: its ops' GPU launches, after: one kernel or its largest anchor's
            launches = sum(ops[i].launches for i in members)
            anchors = [ops[i].launches for i in members if ops[i].kind == ANCHOR]
            after = max([1, *anchors]) if launches else 0
        else:  # one launch per recorded op
            launches, after = len(members), 1
        totals = {"intermediate": 0, "round_trip": 0, "dram": 0, "largest": 0, "tensors": 0}
        found: list[dict[str, Any]] = []
        for pos, p in enumerate(members):
            op = ops[p]
            for out, (key, written) in enumerate(op.writes.items()):
                if mapped and not op.launches:
                    continue  # it launched no GPU work (host tensors): no GPU bytes to save
                readers = [(c, b) for c, k, b in op.consumers if k == key]
                mine = [(c, b) for c, b in readers if c in inside]
                if not mine:
                    continue
                escapes = any(c not in inside for c, _ in readers)
                trip = (0 if escapes else written) + sum(min(b, written) for _, b in mine)
                traffic = self._traffic(p, max(c for c, _ in mine), key)
                dram, why = l2_round_trip(written, trip, traffic, l2)
                totals["intermediate"] += written
                totals["round_trip"] += trip
                totals["largest"] = max(totals["largest"], written)
                totals["tensors"] += 1
                totals["dram"] += dram
                found.append(
                    {
                        "op": pos,
                        "out": out,
                        "bytes": written,
                        "round_trip_bytes": trip,
                        "traffic_bytes": traffic,
                        "dram_bytes": dram,
                        "l2": why,
                    }
                )
        lca = self._lca(members)
        call = self.calls[lca] if lca >= 0 else None
        saved = max(launches - after, 0)
        return {
            "launches": launches,
            "launches_saved": saved,
            # per run (chains() sums the occurrences; an op's launches may differ by call)
            "launches_run": launches,
            "launches_saved_run": saved,
            **({"kernel_ns": sum(ops[i].kernel_ns for i in members)} if mapped else {}),
            "boundaries": len({ops[i].call for i in members}) - 1,
            "parent_class": call.cls if call is not None else None,
            "group": fold(call.qualname) if call is not None else "",
            "intermediate_bytes": totals["intermediate"],
            "round_trip_bytes": totals["round_trip"],
            "dram_bytes": totals["dram"],
            "largest_bytes": totals["largest"],
            "tensors": totals["tensors"],
            "intermediates": found,
        }

    def _placements(self, members: list[int]) -> dict[str, list[int]]:
        """The op sets of a chain's placements (the module docstring)."""
        ops, gap = self.ops, self.max_gap
        epoch = ops[members[0]].epoch
        producers = sorted(
            {
                p
                for m in members
                for p, _, _ in ops[m].reads
                if ops[p].kind == ANCHOR and ops[p].epoch == epoch and m - p <= gap
            }
        )
        consumers = sorted(
            {
                c
                for m in members
                for c, _, _ in ops[m].consumers
                if ops[c].kind == ANCHOR and ops[c].epoch == epoch and c - m <= gap
            }
        )
        out = {"chain": list(members)}
        if producers:  # the newest, and those that read an input of it: one merged GEMM
            last = producers[-1]
            inputs = {key for key, _ in ops[last].loads}
            shared = [p for p in producers if any(key in inputs for key, _ in ops[p].loads)]
            out["epilogue"] = sorted({*members, *shared, last})
        if consumers:
            out["prologue"] = sorted({*members, *consumers})
        return out

    # -------------------------------------------------------------- all chains

    def chains(self) -> list[dict[str, Any]]:
        """Every chain that a fusion could save something on (a launch or an intermediate),
        occurrences with the same ops, modules and phase merged (layers deduplicated):
        ``calls`` per run and the totals of every placement, with its ``intermediates``
        (summed over the occurrences: :func:`_merge_intermediates`) and its ``overlaps``
        (``{id: [placements]}``: the other rows whose placements take in one of its ops in
        some occurrence)."""
        found: dict[str, dict[str, Any]] = {}
        # the rows (and placements) that would take each anchor op in: overlaps
        owners: dict[int, set[tuple[str, str]]] = {}
        for members in self.groups():
            self._windows.clear()
            sets = self._placements(members)
            measured = {name: self._measure(ms) for name, ms in sets.items()}
            if not any(
                m["launches_saved"] > 0 or m["round_trip_bytes"] > 0 for m in measured.values()
            ):
                continue
            lca = self._lca(members)
            call = self.calls[lca] if lca >= 0 else None
            first = self.ops[members[0]].call
            inner = self.calls[first] if first >= 0 else call
            # the phase of the call that runs it (a model's call of a decode step has 2-dim
            # token ids, its layers [batch, 1, ...]: prefill and decode apart), the parent's
            # entrypoint (where a region's ops are)
            phase = inner.phase if inner is not None else ""
            parent = call if call is not None else inner
            method = parent.method if parent is not None else ""
            ops = self.ops
            signature = [
                phase,
                {
                    name: [(ops[i].name, self._module(ops[i].call)) for i in ms]
                    for name, ms in sets.items()
                },
            ]
            key = json.dumps(signature, sort_keys=True)
            row = found.get(key)
            if row is None:  # its first occurrence describes the row's ops
                for name, ms in sets.items():
                    m = measured[name]
                    m["intermediates"] = _merge_intermediates({}, m["intermediates"])
                    measured[name] = {"ops": [self._describe(i) for i in ms], **m}
                row = found[key] = {
                    "id": "f" + hashlib.sha1(key.encode()).hexdigest()[:7],
                    "phase": phase,
                    "method": method,
                    "calls": 0,
                    "instances": set(),
                    "placements": measured,
                }
            else:
                for name, m in measured.items():
                    total = row["placements"][name]
                    for k in _SUMMED:
                        if k in m:
                            total[k] += m[k]
                    total["largest_bytes"] = max(total["largest_bytes"], m["largest_bytes"])
                    _merge_intermediates(total["intermediates"], m["intermediates"])
            row["calls"] += 1
            row["instances"].add(call.qualname if call is not None else "")
            for name, ms in sets.items():
                for i in ms:
                    if ops[i].kind == ANCHOR:
                        owners.setdefault(i, set()).add((row["id"], name))
        # two rows overlap when, in some occurrence, their placements take in the same op (an
        # anchor: chains are disjoint): one GEMM cannot absorb both
        shared: dict[tuple[str, str], dict[str, set[str]]] = {}
        for owned in owners.values():
            for (a, pa), (b, pb) in itertools.permutations(owned, 2):
                if a != b:
                    shared.setdefault((a, pa), {}).setdefault(b, set()).add(pb)
        rows = []
        for row in sorted(found.values(), key=lambda r: r["id"]):
            for name, m in row["placements"].items():
                m["intermediates"] = [
                    {k: v for k, v in entry.items() if k != "out"}
                    for _, entry in sorted(m["intermediates"].items())
                ]
                others = shared.get((row["id"], name), {})
                m["overlaps"] = {
                    b: sorted(ps, key=PLACEMENTS.index) for b, ps in sorted(others.items())
                }
            modules = {
                o["module"] for m in row["placements"].values() for o in m["ops"] if o["module"]
            }
            rows.append({**row, "instances": len(row["instances"]), "modules": sorted(modules)})
        return rows

    def result(self, *, seconds: float = 0.0, error: str | None = None) -> dict[str, Any]:
        """``profile.json`` → ``fusions``: the chains, how they were found, the kernel map
        (``kernel_map``: GPU events and launches of the pass; ``kernel_map_note`` when there
        is none) and the regions not mined (``unmined``)."""
        rec = self.recorder
        out: dict[str, Any] = {
            "version": VERSION,
            "ops": len(self.ops),
            "views": rec.views if rec is not None else 0,
            "host_syncs": rec.epoch if rec is not None else 0,
            "l2_bytes": self.l2_bytes,
            "l2_source": self.l2_source,
            "max_gap": self.max_gap,
            "seconds": round(seconds, 2),
            "chains": self.chains() if self.ops else [],
        }
        km = self.kernels
        if km is not None:
            ops = self.ops
            mined = [op for op in ops if not op.hidden]
            custom = sum(n for (kind, _), (n, _) in km.unmined.items() if kind == "custom")
            out["kernel_map"] = {
                "gpu_events": km.events,
                "gpu_ms": round(km.ns / 1e6, 4),
                # the run's eager launches (build's price per launch): the mined ops' and
                # those outside the dispatcher; not a graph's, not compiled code's (its
                # eager ops' launches here are not the run's)
                "launches": sum(op.launches for op in mined) + custom,
                "mined_ms": round(sum(op.kernel_ns for op in mined) / 1e6, 4),
                "ops_without_launch": sum(op.launches == 0 for op in ops),
                "ops_with_several": sum(op.launches > 1 for op in ops),
                "launch_cuts": len(km.cuts),  # no chain crosses one (map_kernels)
                "unrecorded_events": km.unrecorded[0],
                "seconds": round(self.map_seconds, 2),
            }
        elif self.kernel_note:
            out["kernel_map_note"] = self.kernel_note
        if unmined := self.unmined():
            out["unmined"] = unmined
        if rec is not None and rec.failed:
            out["unrecorded_ops"] = rec.failed
        if error:
            out["error"] = error
        return out


def _merge_intermediates(
    into: dict[tuple[int, int], dict[str, Any]], found: list[dict[str, Any]]
) -> dict[tuple[int, int], dict[str, Any]]:
    """Add one occurrence's intermediates (:meth:`Miner._measure`) to a row's, by producer
    position and output: bytes summed, the largest traffic, the strongest reason
    (:data:`L2_WHY`), ``calls`` seen and ``dram_calls`` whose round trip counts DRAM bytes."""
    for i in found:
        dram = int(i["dram_bytes"] > 0)
        entry = into.get((i["op"], i["out"]))
        if entry is None:
            into[(i["op"], i["out"])] = {**i, "calls": 1, "dram_calls": dram}
            continue
        for k in ("bytes", "round_trip_bytes", "dram_bytes"):
            entry[k] += i[k]
        entry["traffic_bytes"] = max(entry["traffic_bytes"], i["traffic_bytes"])
        entry["l2"] = max(entry["l2"], i["l2"], key=L2_WHY.index)
        entry["calls"] += 1
        entry["dram_calls"] += dram
    return into


def gpu_l2() -> tuple[int | None, str]:
    """The L2 of the GPU at hand in bytes and where that is from (None: no GPU)."""
    from kernel_agent import toolchain

    info = toolchain.gpu_info()
    if info is None or info.l2_cache_mb <= 0:
        return None, "unknown (no CUDA GPU): every intermediate counts as DRAM traffic"
    return int(info.l2_cache_mb * 1024**2), f"{info.name} ({info.l2_cache_mb:.0f} MB)"


def scan(
    workload: Any, inputs: Any, *, l2_bytes: int | None = None, l2_source: str = ""
) -> dict[str, Any]:
    """One hooked work pass of ``workload`` (warmed up; ``profiler.work_reference``) under a
    :class:`Recorder`: :meth:`Miner.result`, with ``error`` when the run failed under it
    (the chains of what was recorded until then are kept). ``l2_bytes``: default the GPU's."""
    from kernel_agent.profiling.profiler import work_reference

    if l2_bytes is None and not l2_source:
        l2_bytes, l2_source = gpu_l2()
    miner = Miner(l2_bytes, l2_source)
    start = time.perf_counter()
    error = None
    try:
        work_reference(workload, inputs, miner=miner)
    except Exception as exc:
        error = " ".join(f"{type(exc).__name__}: {exc}".split())[:400]
    try:
        return miner.result(seconds=time.perf_counter() - start, error=error)
    except Exception as exc:  # the miner never breaks an analyze
        return {"version": VERSION, "chains": [], "error": f"{type(exc).__name__}: {exc}"[:400]}


# ------------------------------------------------------------------ the estimate


def launch_mode(profile: Mapping[str, Any]) -> tuple[str, str]:
    """``eager`` or ``graph`` (most GPU events of the profiled run launched by CUDA-graph
    replays, the kernel view's timeline) and why."""
    tl = (profile.get("kernel_view") or {}).get("timeline") or {}
    stages = tl.get("stages") or []
    events = sum(int(s.get("events") or 0) for s in stages)
    graph = sum(int(s.get("graph_events") or 0) for s in stages)
    if events and graph / events >= 0.5:
        return "graph", f"{graph / events:.0%} of the run's GPU events are CUDA-graph launched"
    if not events:
        return "eager", "no stage timeline: assumed eager"
    return "eager", f"{graph / events:.0%} of the run's GPU events are CUDA-graph launched"


def build(
    profile: Mapping[str, Any],
    peaks: Mapping[str, Any] | None,
    window_ms: float,
    *,
    per: str = "per run",
) -> dict[str, Any]:
    """The ranked candidates of a profile's chains (``profile["fusions"]``, :func:`scan`) at
    the measured ``peaks`` (DRAM bandwidth, launch floor), ms per profiled window
    (``window_ms``, ``per``): each chain at its best placement, by ``saving_ms``, ties by id
    (deterministic)."""
    raw = profile.get("fusions") or {}
    peaks = peaks or {}
    dram = float(peaks.get("dram_gbps") or 0.0)
    floor_us = float(peaks.get("launch_floor_us") or 0.0)
    graph_us = float(peaks.get("launch_floor_graph_us") or 0.0)  # measured since peaks v8
    mode, mode_basis = launch_mode(profile)
    km = raw.get("kernel_map") or {}
    if km and mode == "graph":  # graph-launched work is not mined: the mined ops are eager
        mode_basis = f"the mined ops launch eagerly (the kernel map); {mode_basis}, not mined"
        mode = "eager"
    # an eager launch saves at most what a launch of this run takes on average: the launch
    # floor (a module call with one kernel) includes Python a bare op does not pay. With
    # the kernel map, per GPU launch of the recorded ops; without, per recorded op
    ops = int(raw.get("ops") or 0)
    count, unit = (int(km.get("launches") or 0), "GPU launch") if km else (ops, "recorded op")
    per_op_us = 1000.0 * window_ms / count if count and window_ms > 0 else None
    notes = []
    if mode == "graph" and graph_us:
        boundary, basis = graph_us, f"the measured CUDA-graph launch floor; {mode_basis}"
    elif mode == "graph":
        boundary, basis = GRAPH_BOUNDARY_US, f"a CUDA-graph kernel boundary; {mode_basis}"
    elif floor_us and (per_op_us is None or floor_us <= per_op_us):
        boundary, basis = floor_us, f"the measured launch floor of an eager run; {mode_basis}"
    elif per_op_us is not None:
        boundary = per_op_us
        basis = f"the run's time per {unit}, {window_ms:,.4g} ms / {count:,}" + (
            f", below the measured launch floor {floor_us:.3g} us" if floor_us else ""
        )
        basis += f"; {mode_basis}"
    else:
        boundary = GRAPH_BOUNDARY_US
        basis = "a CUDA-graph kernel boundary: the launch floor was not measured"
        notes.append("launch floor not measured (`kernel-agent doctor`): launches priced low")
    if dram <= 0:
        notes.append("DRAM bandwidth not measured (`kernel-agent doctor`): no byte saving")
    if not km and raw.get("chains"):
        why = raw.get("kernel_map_note") or "a table from before it"
        notes.append(f"no kernel map ({why}): every recorded op counts one launch")
    priced: dict[str, dict[str, Any]] = {}  # id -> the chain and its options
    for chain in raw.get("chains") or []:
        options = []
        for order, name in enumerate(PLACEMENTS):
            p = (chain.get("placements") or {}).get(name)
            if not p:
                continue
            byte_ms = float(p["dram_bytes"]) / (dram * 1e6) if dram > 0 else 0.0
            # fusing cannot save more GPU time than its kernels take now (the kernel map)
            kernel_ms = float(p["kernel_ns"]) / 1e6 if "kernel_ns" in p else None
            capped = kernel_ms is not None and byte_ms > kernel_ms
            if capped:
                byte_ms = float(kernel_ms or 0.0)
            saved = p.get("launches_saved_run", int(chain["calls"]) * int(p["launches_saved"]))
            launch_ms = int(saved) * boundary / 1000
            # a tie goes to the innermost parent (the smallest region), then the order
            depth = len(str(p["group"]).split(".")) if p["group"] else 0
            saving = round(byte_ms + launch_ms, 9)
            options.append((saving, depth, -order, name, byte_ms, launch_ms, capped))
        if options:
            options.sort(reverse=True)  # the row's best first
            priced[str(chain["id"])] = {"chain": chain, "options": options}
    shown, blocked = _disjoint(priced)
    candidates = []
    for cid, entry in priced.items():
        chain, options = entry["chain"], entry["options"]
        name = shown[cid]
        saving, _, _, _, byte_ms, launch_ms, capped = next(o for o in options if o[3] == name)
        p = chain["placements"][name]
        kind = "glue" if not p["parent_class"] else "region" if p["boundaries"] > 0 else "module"
        # the other rows whose shown placement takes in an op of this one's
        overlaps = sorted(o for o, ps in (p.get("overlaps") or {}).items() if shown.get(o) in ps)
        candidate = {
            "id": chain["id"],
            "saving_ms": round(saving, 6),
            "byte_ms": round(byte_ms, 6),
            "launch_ms": round(launch_ms, 6),
            "share": round(saving / window_ms, 6) if window_ms > 0 else None,
            "placement": name,
            "counted": cid in blocked,
            "overlaps": overlaps,
            "kind": kind,
            "parent_class": p["parent_class"],
            "group": p["group"],
            "phase": chain.get("phase", ""),
            "method": chain.get("method", ""),
            "calls": chain["calls"],
            "instances": chain.get("instances", 1),
            "boundaries": p["boundaries"],
            "launches": p["launches"],
            "launches_saved": p["launches_saved"],
            "intermediate_bytes": p["intermediate_bytes"],
            "round_trip_bytes": p["round_trip_bytes"],
            "dram_bytes": p["dram_bytes"],
            "largest_bytes": p["largest_bytes"],
            "intermediates": p.get("intermediates") or [],
            "ops": p["ops"],
            "region": region_text(p["ops"], p["group"]),
            "modules": chain.get("modules", []),
            "placements": {o[3]: o[0] for o in sorted(options, key=lambda o: -o[2])},
        }
        # a counted row's placements that would save more but take in a counted row's op
        better = {
            n: ids
            for n, ids in (blocked.get(cid) or {}).items()
            if candidate["placements"][n] > saving
        }
        if better:
            candidate["blocked"] = better
        if "kernel_ns" in p:  # what its kernels take now (the kernel map), per run
            candidate["kernel_ms"] = round(float(p["kernel_ns"]) / 1e6, 6)
        if capped:
            candidate["byte_capped"] = True
        candidates.append(candidate)
    # the disjoint rows (their savings add up) by saving, then the rows that overlap them
    candidates.sort(key=lambda c: (not c["counted"], -c["saving_ms"], c["id"]))
    for rank, c in enumerate(candidates, 1):
        c["rank"] = rank
    out: dict[str, Any] = {
        "version": VERSION,
        "window_ms": round(window_ms, 4),
        "per": per,
        "dram_gbps": dram or None,
        "launch_us": boundary,
        "launch_mode": mode,
        "launch_basis": basis,
        "l2_bytes": raw.get("l2_bytes"),
        "l2_source": raw.get("l2_source", ""),
        "ops": raw.get("ops", 0),
        "host_syncs": raw.get("host_syncs", 0),
        "seconds": raw.get("seconds"),
        "candidates": candidates,
    }
    if km:
        out["kernel_map"] = dict(km)
    if unmined := not_mined(profile):
        out["unmined"] = unmined
    if raw.get("error"):
        out["error"] = raw["error"]
    if notes:
        out["notes"] = notes
    return out


#: The class of a ``torch.compile``'d module's calls in the module view (its wrapper's).
COMPILED_CLASS = "OptimizedModule"


def _module_view(profile: Mapping[str, Any]) -> tuple[dict[tuple[str, str], float], float]:
    """Inclusive ms per class and instance group (folded qualname) of the profile's module
    view, and its total (the largest class: the roots), as ``summarize`` shares it."""
    groups: dict[tuple[str, str], float] = {}
    total = 0.0
    for c in profile.get("classes") or []:
        total = max(total, float(c.get("inclusive_ms") or 0.0))
        for w in c.get("work") or []:
            key = (str(c.get("cls") or ""), str(w.get("group") or ""))
            groups[key] = groups.get(key, 0.0) + float(w.get("inclusive_ms") or 0.0)
    return groups, total


def not_mined(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The regions of a profile whose ops the miner did not mine (the module docstring),
    each with its ``share`` of the run and its ``basis``: the miner's ``unmined`` rows (a
    graphed or custom region by its GPU time in the pass; a compiled one by the profile's
    module view when it has the module, its eager kernels being no measure of the compiled
    code) and, without a kernel map, the profile's own: the compiled modules and CUDA-graph
    replays of its ``module_gaps`` (module view) and its graph-launched timeline stages
    (their share of the timeline's GPU time). Largest share first."""
    raw = profile.get("fusions") or {}
    km = raw.get("kernel_map") or {}
    gpu_ms = float(km.get("gpu_ms") or 0.0)
    groups, total = _module_view(profile)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def inclusive(kind: str, where: str) -> float | None:
        """The module view's ms of a region: a compiled one's wrapper calls (the other
        instances of its group may run eagerly), else every class of its group."""
        found = [
            ms
            for (cls, group), ms in groups.items()
            if group == where and (kind != "compiled" or cls == COMPILED_CLASS)
        ]
        return sum(found) if found and total > 0 else None

    def view(kind: str, where: str, extra: Mapping[str, Any]) -> None:
        ms = inclusive(kind, where)
        if (kind, where) in seen or ms is None:
            return
        seen.add((kind, where))
        share = round(ms / total, 4)
        basis = "the module view's inclusive time"
        rows.append({**extra, "kind": kind, "where": where, "share": share, "basis": basis})

    for r in raw.get("unmined") or []:
        kind, where = str(r.get("kind")), str(r.get("where") or "")
        if kind == "compiled" and inclusive(kind, where) is not None:
            view(kind, where, r)
        elif gpu_ms > 0:
            share = float(r.get("gpu_ms") or 0.0) / gpu_ms
            basis = "GPU time in the miner's pass"
            if kind == "compiled":
                basis += " (its eager kernels: the compiled code did not run under the miner)"
            rows.append({**r, "share": round(share, 4), "basis": basis})
            seen.add((kind, where))
        else:
            rows.append({**r, "share": None, "basis": "no kernel map: its time is unknown"})
            seen.add((kind, where))
    if not km:  # the profile's own view of what the miner could not see
        gaps = profile.get("module_gaps") or {}
        for qualname in gaps.get("compiled") or []:
            view("compiled", fold(str(qualname)), {})
        for where in gaps.get("graph_replays") or {}:
            view("graphed", str(where), {})
        tl = (profile.get("kernel_view") or {}).get("timeline") or {}
        kernel_ms = float(tl.get("kernel_time_ms") or 0.0)
        for s in tl.get("stages") or []:
            events, graph = int(s.get("events") or 0), int(s.get("graph_events") or 0)
            where = str(s.get("stage") or "")
            graphed = kernel_ms > 0 and events and graph / events >= 0.5
            if graphed and ("graphed", where) not in seen:
                seen.add(("graphed", where))
                share = float(s.get("gpu_ms") or 0.0) / kernel_ms
                rows.append(
                    {
                        "kind": "graphed",
                        "where": where,
                        "gpu_events": events,
                        "gpu_ms": s.get("gpu_ms"),
                        "share": round(share, 4),
                        "basis": "the timeline: a stage whose GPU work is mostly graph launched",
                    }
                )
    return sorted(rows, key=lambda r: (-(r.get("share") or 0.0), r["kind"], r["where"]))


def _disjoint(
    priced: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, list[str]]]]:
    """Overlap-aware ranking of the rows of :func:`build`: every placement of every row, by
    saving (ties: the innermost parent, the placement order, the id), is taken when its row
    has none yet and it shares no op with a placement taken (``overlaps``: an anchor two
    rows' placements take in); a placement that saves nothing waits while a better one of
    its row is blocked. Greedy, deterministic, not the best combination in general.

    Returns the placement shown per row (the one taken; a row with none taken is an
    alternative of the rows it overlaps, shown at its best) and, per taken row, its
    placements that taken rows blocked (``{placement: [ids]}``): its keys are the rows
    whose savings add up."""
    order = sorted(
        ((o[0], o[1], o[2], cid, o[3]) for cid, e in priced.items() for o in e["options"]),
        key=lambda o: (-o[0], -o[1], -o[2], o[3]),
    )
    taken: dict[str, str] = {}
    blocked: dict[str, dict[str, list[str]]] = {}
    for saving, _, _, cid, name in order:
        if cid in taken:
            continue
        if saving <= 0 and any(o[0] > 0 for o in priced[cid]["options"]):
            continue  # nothing saved while a better placement of it overlaps: an alternative
        place = priced[cid]["chain"]["placements"][name]
        hits = sorted(o for o, ps in (place.get("overlaps") or {}).items() if taken.get(o) in ps)
        if hits:
            blocked.setdefault(cid, {})[name] = hits
            continue
        taken[cid] = name
    shown = {cid: taken.get(cid, e["options"][0][3]) for cid, e in priced.items()}
    return shown, {cid: blocked.get(cid, {}) for cid in taken}


def additive(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The candidates whose savings add up, from any set of them (a stage group's chains,
    the rows of a table): the counted ones, then by saving each alternative that overlaps
    none kept so far (its ``overlaps``). A table from before overlaps were recorded counts
    every row."""
    kept: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    ranked = sorted(
        candidates,
        key=lambda c: (
            not c.get("counted", True),
            -float(c.get("saving_ms") or 0.0),
            str(c.get("id")),
        ),
    )
    for c in ranked:
        if not ids.intersection(c.get("overlaps") or ()):
            kept.append(c)
            ids.add(str(c.get("id")))
    return kept


def region_text(ops: Sequence[Mapping[str, Any]], group: str) -> str:
    """The ops of a placement for a region target's ``region``: runs of ops in one module,
    named relative to the parent (``group``): ```add` (the parent's own code) → `pow`, ...
    (in `norm`) → `linear` (in `proj`)``."""
    parts = []
    for module, run in itertools.groupby(ops, key=lambda o: str(o.get("module") or "")):
        names = ", ".join(f"`{o['op']}`" for o in run)
        if not module:
            where = "outside every module"
        elif module == group:
            where = "the parent's own code"
        elif group and module.startswith(group + "."):
            where = f"in `{module[len(group) + 1 :]}`"
        else:
            where = f"in `{module}`"
        parts.append(f"{names} ({where})")
    return " → ".join(parts)


def match(table: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> tuple[dict, str] | None:
    """The candidate a target stands for, and how it was found: its ``fusion`` id, else the
    largest region candidate of its ``parent_class`` (in its ``phase``, if it has one) among
    the counted rows (:func:`build`: their savings add up), then among the alternatives."""
    candidates = (table or {}).get("candidates") or []
    fid = spec.get("fusion")
    if fid:
        hit = next((c for c in candidates if c.get("id") == fid), None)
        if hit is not None:
            return hit, f"fusion {fid}{overlap_note(hit)}"
    parent = spec.get("parent_class")
    phase = spec.get("phase") if spec.get("phase") not in (None, "", "all") else None
    same = [
        c
        for c in candidates
        if parent
        and c.get("parent_class") == parent
        and c.get("kind") == "region"
        and (phase is None or c.get("phase") == phase)
    ]
    if not same:
        return None
    best = max(
        same,
        key=lambda c: (bool(c.get("counted", True)), float(c.get("saving_ms") or 0.0), c["id"]),
    )
    how = f"fusion {best['id']}, the largest of parent class {parent}{overlap_note(best)}"
    return best, how


def overlap_note(candidate: Mapping[str, Any]) -> str:
    """`` (an alternative to f1, f2: …)`` / `` (alternatives f3 overlap it)``: the rows that
    share an op with it ("" when none, or in a table from before overlaps)."""
    ids = [f"`{i}`" for i in candidate.get("overlaps") or []]
    if not ids:
        return ""
    names = ", ".join(ids[:3]) + (f" and {len(ids) - 3} more" if len(ids) > 3 else "")
    if candidate.get("counted", True):
        return f" (alternatives {names} overlap it)"
    return f" (an alternative to {names}: they share an op, its saving does not add to theirs)"


# ------------------------------------------------------------------ markdown


def _mb(n: float) -> str:
    return f"{n / 1e6:,.3g} MB"


def l2_text(candidate: Mapping[str, Any]) -> str:
    """Why its intermediates' bytes count, for the table: ``in L2``, ``1.2 MB DRAM: 2
    evicted by ≤ 7.3 MB of traffic, 1 larger than the L2`` (a table from before the
    traffic was recorded: its DRAM bytes only)."""
    found = candidate.get("intermediates")
    dram = int(candidate.get("dram_bytes") or 0)
    if not found:
        return f"{_mb(dram)} DRAM" if dram else "in L2"
    if any(i.get("l2") == "unknown" for i in found):
        return f"{_mb(dram)} DRAM: L2 unknown"
    if not dram:
        return "in L2"
    parts = []
    evicted = [i for i in found if i.get("l2") == "traffic"]
    if evicted:
        most = max(int(i.get("traffic_bytes") or 0) for i in evicted)
        parts.append(f"{len(evicted)} evicted by ≤ {_mb(most)} of traffic")
    larger = sum(1 for i in found if i.get("l2") == "size")
    if larger:
        parts.append(f"{larger} larger than the L2")
    return f"{_mb(dram)} DRAM: " + ", ".join(parts)


def _intermediate_text(i: Mapping[str, Any], ops: Sequence[Mapping[str, Any]]) -> str:
    """```mul` (`m.norm`) 1.05 MB: evicted by ≤ 7.34 MB of traffic, 2.1 MB DRAM in 28 of 28
    calls`` (bytes per call)."""
    pos = int(i.get("op", -1))
    op = ops[pos] if 0 <= pos < len(ops) else {}
    calls = max(int(i.get("calls") or 1), 1)
    where = f" (`{op.get('module')}`)" if op.get("module") else ""
    traffic = _mb(int(i.get("traffic_bytes") or 0))
    why = str(i.get("l2"))
    text = {
        "in": "in L2",
        "traffic": f"evicted by ≤ {traffic} of traffic",
        "size": f"larger than the L2 (≤ {traffic} of traffic too)",
        "unknown": "L2 unknown",
    }.get(why, why)
    if why in ("traffic", "size"):
        dram = _mb(int(i.get("dram_bytes") or 0))
        text += f", {dram} DRAM in {i.get('dram_calls')} of {calls} calls"
    return f"`{op.get('op', '?')}`{where} {_mb(int(i.get('bytes') or 0) / calls)}: {text}"


def markdown(
    table: Mapping[str, Any], *, top: int = TOP, title: bool = True, details: bool = False
) -> str:
    """``## Fusion candidates (measured)`` of ``summary.md`` ("" without chains and regions
    not mined): the top counted rows (their savings add up), each followed by its
    alternatives (rows that share an op with it), and the regions not mined; ``details``: why
    each intermediate of the rows shown counts or not."""
    candidates = table.get("candidates") or []
    if not candidates and not table.get("error") and not table.get("unmined"):
        return ""
    lines = ["", "## Fusion candidates (measured)", ""] if title else [""]
    l2 = table.get("l2_bytes")
    l2_size = f"the {l2 / 2**20:.3g} MB L2" if l2 else "the L2 (unknown here: all of it counts)"
    dram = table.get("dram_gbps")
    km = table.get("kernel_map") or {}
    counts = (
        f"{table.get('ops', 0):,} recorded ops, {int(km.get('launches') or 0):,} GPU launches "
        "(the kernel map: each op's own launches and kernel time)"
        if km
        else f"{table.get('ops', 0):,} kernel ops"
    )
    lines.append(
        f"Chains of memory-bound ops (element-wise, norms, reductions, casts, copies) with the "
        "GEMM / convolution / attention next to them that could take them as an epilogue or "
        f"prologue, from the tensor storages of every op of one run "
        f"({counts}, {table.get('host_syncs', 0)} host syncs; no "
        "chain crosses one). *saves* (ms "
        f"{table.get('per', 'per run')}) = the DRAM round trips of the intermediates written "
        f"and read back / "
        + (f"DRAM {dram:.0f} GB/s" if dram else "DRAM bandwidth (not measured)")
        + f" (LRU-ish: {l2_size} keeps the newest bytes touched, so an intermediate stays "
        "there while it and the bytes other ops touch between its write and its last read "
        "fit; the share of it evicted counts) + launches saved × "
        f"{table.get('launch_us', GRAPH_BOUNDARY_US):.3g} us ({table.get('launch_basis', '')}), "
        "× calls per run. One row per chain and instance group (layers deduplicated); *fuse* "
        "says where its ops go: one kernel of their own (*chain*), into the GEMM / attention "
        "they read from (*epilogue*) or that reads them (*prologue*). Every row: "
        "`profile/fusions.md`, `fusions.json`."
    )
    if table.get("error"):
        lines += ["", f"* the recorded run failed ({table['error']}); chains recorded until then"]
    for note in table.get("notes") or []:
        lines.append(f"* {note}")
    lines += unmined_lines(table.get("unmined") or [])
    if not candidates:
        return "\n".join(lines) + "\n"
    counted = [c for c in candidates if c.get("counted", True)]
    rank = {c["id"]: n for n, c in enumerate(counted)}
    alternatives: dict[str, list[Mapping[str, Any]]] = {}
    for c in candidates:  # under the best-ranked counted row it overlaps
        hosts = sorted((rank[i], i) for i in c.get("overlaps") or [] if i in rank)
        if not c.get("counted", True) and hosts:
            alternatives.setdefault(hosts[0][1], []).append(c)
    lines += [
        "",
        "| id | saves ms | bytes + launches ms | calls | fuse | parent class @ group | crosses "
        "| launches | intermediates (largest; L2) | ops |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    shown: list[Mapping[str, Any]] = []
    for c in counted[:top]:
        for row in (c, *alternatives.get(c["id"], [])):
            lines.append(_table_row(row, c))
            shown.append(row)
    rest = counted[top:]
    if rest:
        lines.append(
            f"| ... | {sum(c['saving_ms'] for c in rest):,.4g} | | | | {len(rest)} more | | | | |"
        )
    total = sum(float(c["saving_ms"]) for c in counted)
    others = len(candidates) - len(counted)
    window = float(table.get("window_ms") or 0.0)
    share = f" ({total / window:.1%} of the window)" if window > 0 else ""
    lines += [
        "",
        (
            f"The {len(counted)} counted rows share no op: together they save {total:,.4g} ms"
            if len(counted) != 1
            else f"The counted row saves {total:,.4g} ms"
        )
        + f"{share}. "
        + (
            f"{others} more {'is an alternative' if others == 1 else 'are alternatives'} (↳): "
            "each shares an op (a GEMM in one's prologue and the other's epilogue) with the "
            "row above it, so their savings do not add up; plan one or the other. "
            if others
            else ""
        )
        + "Ranked greedily by saving over rows that share no op (a row whose best placement "
        "overlaps a row ranked before it takes its next one, *fuse* names what that would "
        "cost). To plan one: a `region` target with `parent_class` = its parent class, "
        "`region` = its ops and `fusion` = its id (the improve scheduler then expects its "
        "*saves*). *crosses* 0: inside one module call, a module target covers it. Estimates "
        "from bytes and launches, not a fused kernel's measurement: the planner still decides.",
    ]
    if details:
        lines += _details(shown)
    return "\n".join(lines) + "\n"


def _table_row(c: Mapping[str, Any], host: Mapping[str, Any]) -> str:
    where = f"`{c['parent_class']}` `{c['group']}`" if c["parent_class"] else "(no module)"
    phase = f" ({c['phase']})" if c.get("phase") else ""
    region = c["region"] if len(c["region"]) <= 400 else c["region"][:400] + " …"
    fuse = str(c["placement"])
    for name, ids in (c.get("blocked") or {}).items():  # a better placement it gave up
        fuse += f" ({name} {c['placements'][name]:,.3g} ms: overlaps " + ", ".join(
            f"`{i}`" for i in ids
        )
        fuse += ")"
    ident = f"`{c['id']}`" if c is host else f"↳ `{c['id']}` (alt. of `{host['id']}`)"
    launches = int(c["launches"])
    after = str(launches - int(c["launches_saved"]))
    if "kernel_ms" in c:  # the kernel map: its kernels' GPU time now, per run
        after += f"; {c['kernel_ms']:,.3g} ms GPU"
    capped = " (≤ its kernels' time)" if c.get("byte_capped") else ""
    return (
        f"| {ident} | {c['saving_ms']:,.4g} | {c['byte_ms']:,.3g}{capped} + "
        f"{c['launch_ms']:,.3g} | {c['calls']:,} | {fuse} | {where}{phase} | "
        f"{c['boundaries']} | {launches} → {after} | {_mb(c['intermediate_bytes'])} "
        f"({_mb(c['largest_bytes'])}; {l2_text(c)}) | {region} |"
    )


#: How each kind of region escaped the recorder, for the table.
_UNMINED_TEXT = {
    "graphed": "CUDA-graph replays",
    "compiled": "compiled code",
    "custom": "kernels launched outside the dispatcher",
    "unattributed": "GPU work without a launch call in the trace",
}


def unmined_lines(rows: Sequence[Mapping[str, Any]], top: int = 6) -> list[str]:
    """The bullet on the regions not mined (:func:`not_mined`; [] without)."""
    if not rows:
        return []
    items = []
    for r in rows[:top]:
        where = f" in `{r['where']}`" if r.get("where") else " outside every module"
        n = int(r.get("instances") or 0)
        if n and "*" in str(r.get("where")):  # which of the group's instances
            where += f" ({n} instance{'s' if n > 1 else ''})"
        share = r.get("share")
        amount = f"{share:.1%}" if share is not None else "time unknown"
        if r.get("gpu_ms") and str(r.get("basis", "")).startswith("GPU time"):
            amount += f", {float(r['gpu_ms']):,.3g} ms GPU"
        if r.get("ops"):
            amount += f", {int(r['ops']):,} ops recorded eagerly"
        items.append(f"{_UNMINED_TEXT.get(str(r['kind']), r['kind'])}{where} ({amount})")
    more = f" and {len(rows) - top} more" if len(rows) > top else ""
    bases = sorted({str(r["basis"]) for r in rows[:top] if r.get("basis")})
    return [
        "* **not mined**: "
        + "; ".join(items)
        + more
        + f" (shares: {'; '.join(bases)}). A CUDA-graph replay and a kernel launched outside "
        "the dispatcher (Triton, an extension) dispatch no op to the recorder, and compiled "
        "code runs eagerly under it: no chain inside these regions is listed, their fusions "
        "are the graph's, the kernel's or the compiler's (`fusions.json` → `unmined`)"
    ]


def _details(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """``### Intermediates and the L2``: per row shown, the intermediates whose round trip
    counts and why (:func:`l2_round_trip`), and how many stay in the L2."""
    out = []
    for c in rows:
        found = c.get("intermediates") or []
        if not found:
            continue
        counted = [i for i in found if i.get("l2") in ("traffic", "size")]
        unknown = sum(1 for i in found if i.get("l2") == "unknown")
        stay = [i for i in found if i.get("l2") == "in"]
        items = [_intermediate_text(i, c.get("ops") or []) for i in counted]
        if unknown:
            items.append(f"{unknown} counted: L2 unknown")
        if stay:
            most = max(int(i.get("traffic_bytes") or 0) for i in stay)
            traffic = f"≤ {_mb(most)} of traffic" if most else "no traffic in between"
            items.append(f"{len(stay)} in L2 ({traffic})")
        out.append(f"* `{c['id']}` ({c['placement']}): " + "; ".join(items))
    if not out:
        return []
    return [
        "",
        "### Intermediates and the L2",
        "",
        "Per row shown, the intermediates whose round trip counts (by the op that writes "
        "it, bytes per call) and why: *traffic* = the distinct bytes the ops between its "
        "write and its last read touch, the most of any call; the L2 keeps the newest bytes "
        "(LRU-ish), so the part of it beyond L2 − traffic counts.",
        "",
        *out,
    ]


def write(
    profile_dir: Path,
    profile: Mapping[str, Any],
    peaks: Mapping[str, Any] | None,
    window_ms: float,
    *,
    per: str = "per run",
) -> dict[str, Any]:
    """Rank the profile's chains (:func:`build`) and write ``fusions.json`` + ``fusions.md``
    into ``profile_dir``. Never raises: the table then holds the error."""
    from kernel_agent.workspace import write_json

    try:
        table = build(profile, peaks, window_ms, per=per)
    except Exception as exc:  # the fusion table never breaks an analyze
        table = {"version": VERSION, "error": f"{type(exc).__name__}: {exc}"[:300]}
        table["candidates"] = []
    write_json(profile_dir / "fusions.json", table)
    text = markdown(table, top=SHOWN, details=True)
    (profile_dir / "fusions.md").write_text(text.lstrip("\n"))
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fusion candidates of a profile.json.")
    parser.add_argument("profile", type=Path, help="profile/profile.json of a run")
    parser.add_argument("--window-ms", type=float, required=True, help="unhooked window ms")
    parser.add_argument("--per", default="per run")
    parser.add_argument("--peaks", type=Path, default=None, help="peaks JSON (default: cached)")
    parser.add_argument("--top", type=int, default=SHOWN)
    parser.add_argument("--json", action="store_true", help="print the JSON, not the markdown")
    ns = parser.parse_args(argv)
    if ns.peaks is not None:
        peaks = json.loads(ns.peaks.read_text())
    else:
        from kernel_agent.kernels.roofline import current_peaks

        peaks = current_peaks()
    table = build(json.loads(ns.profile.read_text()), peaks, ns.window_ms, per=ns.per)
    print(json.dumps(table, indent=1) if ns.json else markdown(table, top=ns.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
