"""Megakernel schedules built from a stage's captured ops (issue #225).

The input of :func:`.schedule.build`, a stage's ops with their tiles, weights, costs and
tile-level edges, comes from one recorded call of the stage instead of a hand-written list:
``python -m kernel_agent.native.megakernel.schedule --from-capture capture.pt`` (a target's
``capture.pt`` or the inputs-only ``capture_inputs.pt``), :func:`from_capture`,
:func:`from_module` (in a project's ``build()``) or :func:`build` (a :class:`Recording`).
Model-agnostic: every decision comes from the recorded ops, their operands' shapes, dtypes,
strides and storages, and the peaks of the GPU at hand; module names only label the report.

1. **Record** (:func:`record`): one call runs under the fusion miner's ``TorchDispatchMode``
   (:class:`kernel_agent.profiling.fusion.Recorder`: every aten op, composite ones as the
   kernels they decompose into except GEMMs, views and allocations not ops), and each op's
   tensor operands are kept as :class:`View` (storage, offset, shape, strides, dtype, and
   its *version*: the recorded op that last wrote the storage before, -1 for none).
2. **Map** each op to a family of the kit's opcodes (:mod:`.opcodes`, :data:`OPCODES`):

   * ``rmsnorm``: rsqrt(mean(v²) + eps) times v (v: x, or x cast to fp32), then the cast
     back and a multiplication by a constant 1-D weight in either order, or
     ``aten.rms_norm`` / ``_fused_rms_norm`` as one op;
   * ``gemv``: ``linear`` / ``mm`` / ``matmul`` / ``mv`` of at most :data:`SKINNY_ROWS`
     activation rows (decode; a skinny GEMM is M GEMV tiles per weight tile) by a bf16
     weight the stage only reads (a parameter, buffer or constant), without a bias;
   * ``residual``: ``add`` of two bf16 activations of one shape (element-wise);
   * ``glu``: ``silu`` / ``gelu`` of one bf16 activation times another (a gated MLP);
   * ``argmax`` over the last dimension of bf16 rows;
   * ``splitk_reduce``: no recorded op, the second half of a GEMV split over K (``split_k``;
     a K wider than a tile's slice, :data:`.opcodes.MAX_SLICE`, always is).

   Anything else is **unsupported** (:class:`Unsupported`: the op, its module, its shapes,
   why), never dropped: :meth:`Plan.schedule` refuses a plan with any.
3. **Fuse** (``fuse=True``, the example's layer): an RMSNorm read only by GEMVs into their
   prologue (each tile normalises its x), an add of a GEMV's output (read by nothing else)
   and another activation into the GEMV's epilogue, or into the reduce of a split GEMV.
4. **Lay out** the tensor table (:class:`Slot`): the GEMV weights in op order (prefetched
   into the page pool; a split GEMV's weights packed once as [splits][N][K / splits], each
   tile's rows contiguous), the other constants (norm weights), then one *arena* per dtype:
   the call's inputs first, then each value an op writes, every version its own 16-byte
   aligned slice. Nothing is reused within a call, so no write-after-read edges are needed.
5. **Tile**: a GEMV per :func:`.opcodes.tile_rows` output rows (and activation row, and K
   split), an RMSNorm and an argmax per row, element-wise ops per
   :data:`.opcodes.ELEMENTWISE_TILE` elements, a reduce per GEMV row tile.
6. **Edges at tile granularity from the storages**: each tile's byte ranges read and
   written; a consumer tile needs the producer tiles whose writes its reads overlap
   (:data:`.schedule.ALL`, :data:`.schedule.SAME` or a :class:`TileMap`), so the schedule's
   counters are chunked wherever consumers read part of a producer.
7. **Costs** per tile from the roofline of the GPU at hand (:class:`Roofline`: max(bytes /
   DRAM bandwidth, FLOPs / the fp32 FMA peak the opcodes compute at), µs) from measured
   peaks (``kernel-agent doctor``'s, or a ceilings table's ``peaks``); without peaks the
   bytes moved (relative costs). The source is labelled (:attr:`Plan.cost_source`).

Then :meth:`Plan.schedule` builds the queues and counters (:func:`.schedule.build`) and
:func:`.simulate.check` validates them; :meth:`Plan.tensors` gives the runtime's tensor table
(:class:`.runtime.Runtime`) and :meth:`Plan.bind` the call's inputs and outputs in it.
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import itertools
import json
import math
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent.native.megakernel import opcodes as oc
from kernel_agent.native.megakernel import schedule as mks

Key = tuple[int, int]  # device index (-1: the host), storage base pointer
Value = tuple[Any, int]  # (storage key or a synthetic name, version)

#: Activation rows a GEMV op takes: M GEMV tiles per weight tile, the later ones reading it
#: from L2. More rows: a tensor-core GEMM fits better.
SKINNY_ROWS = 8
#: The page pool assumed without a launch's (``pages × page bytes``, the example's
#: ``mk_info()``): 11 pages of 8 KB, what fits next to the example's scratch on a GPU with
#: 99 KB of shared memory per block (an A10, an RTX 5070 Ti).
POOL_BYTES = 11 * oc.PAGE_BYTES
#: Arena slices start on this many bytes (16-byte vector loads).
ALIGN = 16
#: The smallest K slice ``split_k="auto"`` makes (narrower slices are mostly overhead).
MIN_SLICE = 256
FAMILIES = ("rmsnorm", "gemv", "splitk_reduce", "residual", "glu", "argmax")
#: The kit's opcode of each family (``opcodes=`` replaces them for a project's own numbers).
OPCODES = {
    "rmsnorm": oc.RMSNORM,
    "gemv": oc.GEMV,
    "splitk_reduce": oc.SPLITK_REDUCE,
    "residual": oc.RESIDUAL,
    "glu": oc.GLU,
    "argmax": oc.ARGMAX,
}
_SHORT = {
    "bfloat16": "bf16",
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "float8_e4m3fn": "fp8",
    "int64": "i64",
    "int32": "i32",
    "bool": "bool",
}
#: Why common ops have no family (the rest: "no opcode family").
_WHY = {
    "_to_copy": "a dtype cast outside an RMSNorm: the kit's opcodes read and write bf16",
    "clone": "a copy: no opcode (the kit's ops write fresh slices)",
    "copy_": "an in-place copy (a cache or state update?): no opcode",
    "index_put_": "writes part of a buffer (a cache append?): no opcode",
    "index_copy_": "writes part of a buffer (a cache append?): no opcode",
    "scatter_": "writes part of a buffer: no opcode",
    "softmax": "softmax: no opcode family (attention is milestone 6 of megakernel.md)",
    "_softmax": "softmax: no opcode family (attention is milestone 6 of megakernel.md)",
    "native_layer_norm": "LayerNorm: no opcode (the kit's norm is RMSNorm)",
    "layer_norm": "LayerNorm: no opcode (the kit's norm is RMSNorm)",
    "embedding": "a gather (embedding): no opcode",
    "index_select": "a gather: no opcode",
    "gather": "a gather: no opcode",
    "cat": "a concatenation: no opcode (write the parts into one buffer instead)",
    "bmm": "a batched matmul: each batch has its own weights, not a GEMV of a constant",
    "baddbmm": "a batched matmul: each batch has its own weights, not a GEMV of a constant",
    "addmm": "a matmul with a bias (addmm): the kit's GEMV has no bias epilogue",
    "addmv": "a matrix-vector product with a bias (addmv): the GEMV has no bias epilogue",
    "mul": "a multiplication outside an RMSNorm or a gated activation (act(a) * b): no opcode",
    "silu": "an activation that gates no multiplication (act(a) * b): no opcode",
    "gelu": "an activation that gates no multiplication (act(a) * b): no opcode",
    "_scaled_mm": "a scaled FP8 matmul of FP8 activations: the GEMV_FP8 opcode takes bf16 "
    "activations and per-row weight scales",
}


class CaptureError(ValueError):
    """A recording or a plan that makes no schedule (unsupported ops, a bad capture)."""


# ------------------------------------------------------------------ recording


@dataclass(frozen=True)
class View:
    """A tensor as one op sees it: its storage (``key``), offset (elements), shape, strides,
    dtype (a torch name: ``bfloat16``), element size, and ``version``: the recorded op that
    last wrote the storage before this op read it (an output: the op itself; -1: none)."""

    key: Key | None
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    esize: int
    version: int = -1

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def value(self) -> Value:
        return self.key, self.version

    def contiguous(self) -> bool:
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.stride), strict=True):
            if size != 1 and stride != expected:
                return False
            expected *= size
        return True

    def describe(self) -> str:
        return "x".join(map(str, self.shape)) + " " + _SHORT.get(self.dtype, self.dtype)


@dataclass(frozen=True)
class Rec:
    """One recorded op (a kernel launch): its aten name and overload, its arguments
    (tensors as :class:`View`, lists as tuples, dtypes by name, other objects as text),
    outputs, and the innermost module call around it (for the report)."""

    index: int
    name: str
    overload: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    outs: tuple[View, ...]
    module: str = ""
    broken: bool = False  # its arguments could not be recorded

    def tensors(self) -> list[View]:
        """Every tensor argument (positional and keyword, inside lists too)."""
        found: list[View] = []
        for value in (*self.args, *self.kwargs.values()):
            if isinstance(value, View):
                found.append(value)
            elif isinstance(value, tuple):
                found += [v for v in value if isinstance(v, View)]
        return found

    def arg(self, position: int, name: str, default: Any = None) -> Any:
        if position < len(self.args):
            return self.args[position]
        return self.kwargs.get(name, default)

    def label(self) -> str:
        return f"#{self.index} {self.name}"


@dataclass
class Recording:
    """One call of a stage: its ops in launch order, its tensor inputs (version -1) and
    outputs (the version they hold at the end), the qualnames of the module's parameters and
    buffers by storage, every storage some op wrote, a tensor of each storage the ops
    touched (the runtime's weights) and the ops the recorder could not record."""

    ops: list[Rec]
    inputs: list[View]
    outputs: list[View]
    names: dict[Key, str] = field(default_factory=dict)
    written: set[Key] = field(default_factory=set)
    tensors: dict[Key, Any] = field(default_factory=dict, repr=False)
    unrecorded: int = 0
    label: str = ""


def _view(t: Any, version: int) -> View:
    from kernel_agent.profiling.fusion import _storage

    key, _ = _storage(t)
    return View(
        key,
        int(t.storage_offset()),
        tuple(int(s) for s in t.shape),
        tuple(int(s) for s in t.stride()),
        str(t.dtype).removeprefix("torch."),
        int(t.element_size()),
        version,
    )


def _arg(value: Any, versions: Mapping[Key, int]) -> Any:
    import torch

    from kernel_agent.profiling.fusion import _storage

    if isinstance(value, torch.Tensor):
        key, _ = _storage(value)
        return _view(value, versions.get(key, -1) if key is not None else -1)
    if isinstance(value, tuple | list):
        return tuple(_arg(v, versions) for v in value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return repr(value)  # devices, layouts, memory formats: as text


def _make_recorder(module_of: Callable[[], str]) -> Any:
    """A :class:`kernel_agent.profiling.fusion.Recorder` that also keeps each op as a
    :class:`Rec` (``recs``, aligned with its ``ops``)."""
    from kernel_agent.profiling import fusion

    class _Recorder(fusion.Recorder):
        def __init__(self) -> None:
            super().__init__(lambda: -1)
            self.recs: list[Rec] = []
            self.seen: dict[Key, Any] = {}
            self._funcs: list[Any] = []

        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            self._funcs.append(func)
            try:
                return super().__torch_dispatch__(func, types, args, kwargs)
            finally:
                self._funcs.pop()

        def _record(self, info: Any, args: Any, kwargs: Any, out: Any) -> None:
            ins = fusion._tensors(args) + fusion._tensors(list(kwargs.values()))
            versions: dict[Key, int] = {}
            for t in ins:
                key, _ = fusion._storage(t)
                if key is not None:
                    versions[key] = self.writer.get(key, -1)
            index = len(self.ops)
            super()._record(info, args, kwargs, out)
            if len(self.ops) == index:
                return  # a view, an allocation or metadata: no kernel
            func = self._funcs[-1] if self._funcs else None
            # a custom op by its namespace (``myops::fused``), an aten op by its name
            name = info.name if info.aten else f"{getattr(func, 'namespace', '?')}::{info.name}"
            try:
                outs = fusion._tensors(out)
                rec = Rec(
                    index,
                    name,
                    str(getattr(func, "_overloadname", "")),
                    tuple(_arg(a, versions) for a in args),
                    {k: _arg(v, versions) for k, v in kwargs.items()},
                    tuple(_view(t, index) for t in outs),
                    module_of(),
                )
                for t in (*ins, *outs):
                    key, _ = fusion._storage(t)
                    if key is not None:
                        self.seen.setdefault(key, t)
            except Exception:  # never breaks the run; reported as unsupported
                rec = Rec(index, name, "", (), {}, (), module_of(), broken=True)
            self.recs.append(rec)

    return _Recorder()


def _flat(value: Any) -> Iterator[Any]:
    """The tensors in nested tuples, lists and dicts, in order."""
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, tuple | list):
        for v in value:
            yield from _flat(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _flat(v)


def record(
    module: Any,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    method: str = "forward",
    label: str = "",
) -> Recording:
    """One call of ``module`` (``method``) on ``args`` / ``kwargs``, recorded (inference
    mode). The call runs for real: its side effects (a cache append) happen once more."""
    import torch

    from kernel_agent.profiling.fusion import _storage
    from kernel_agent.profiling.methods import entrypoint

    kwargs = dict(kwargs or {})
    names: dict[int, str] = {}
    for name, sub in module.named_modules(remove_duplicate=False):
        names.setdefault(id(sub), name)
    stack: list[str] = []

    def enter(sub: Any, args: Any) -> None:
        stack.append(names.get(id(sub), ""))

    def leave(sub: Any, args: Any, out: Any) -> None:  # returns None: the output stays
        if stack:
            stack.pop()

    handles = []
    for sub in module.modules():
        handles += [sub.register_forward_pre_hook(enter), sub.register_forward_hook(leave)]
    rec = _make_recorder(lambda: stack[-1] if stack else "")
    try:
        with torch.inference_mode(), rec:
            out = entrypoint(module, method)(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    inputs = [_view(t, -1) for t in _flat((tuple(args), kwargs))]
    outputs = []
    for t in _flat(out):
        key, _ = _storage(t)
        outputs.append(_view(t, rec.writer.get(key, -1) if key is not None else -1))
    weights: dict[Key, str] = {}
    for name, p in [
        *module.named_parameters(remove_duplicate=False),
        *module.named_buffers(remove_duplicate=False),
    ]:
        key, _ = _storage(p)
        if key is not None:
            weights.setdefault(key, name)
            rec.seen[key] = p  # the parameter itself, not a view an op took of it
    return Recording(
        ops=list(rec.recs),
        inputs=inputs,
        outputs=outputs,
        names=weights,
        written=set(rec.writer),
        tensors=dict(rec.seen),
        unrecorded=int(rec.failed),
        label=label or f"{type(module).__name__}.{method}",
    )


# ------------------------------------------------------------------ the op DAG


@dataclass(frozen=True)
class Unsupported:
    """A recorded op no family takes: why, where (its module, for the report), its shapes."""

    op: int  # its index in the recording (-1: the recording as a whole)
    name: str
    module: str
    shapes: str
    reason: str

    def describe(self) -> str:
        where = f" in `{self.module}`" if self.module else ""
        shapes = f" ({self.shapes})" if self.shapes else ""
        name = self.name if "::" in self.name else f"aten.{self.name}"
        head = f"#{self.op} `{name}`" if self.op >= 0 else self.name
        return f"{head}{where}{shapes}: {self.reason}"


@dataclass(frozen=True)
class TileMap:
    """``needs`` of an :class:`.schedule.Edge` from the storages: consumer tile t needs the
    producer tiles ``needs[t]`` (hashable, comparable, deterministic)."""

    needs: tuple[tuple[int, ...], ...]

    def __call__(self, t: int) -> tuple[int, ...]:
        return self.needs[t]


@dataclass(frozen=True)
class Slot:
    """One entry of the tensor table. ``kind``: ``weight`` (a GEMV's weights, prefetched;
    the storage of ``name``), ``packed`` (a split GEMV's weights as [splits][N][K / splits],
    built once from ``name``'s ``view``), ``const`` (another constant: a norm's weight), or
    ``arena`` (the activations of one dtype, ``numel`` elements, zeroed)."""

    index: int
    kind: str
    name: str
    dtype: str
    numel: int
    key: Key | None = None
    view: View | None = None
    splits: int = 1


@dataclass(frozen=True)
class Binding:
    """A tensor of the call in the table: slot, element offset, shape, dtype."""

    slot: int
    offset: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class OpInfo:
    """What one op of the plan stands for: its family, the recorded ops it covers, its
    shape, fusions, the bytes and FLOPs of all its tiles and its cost per tile."""

    name: str
    family: str
    opcode: int
    tiles: int
    covers: tuple[str, ...]
    shape: str
    fused: tuple[str, ...]
    bytes: int
    flops: int
    cost: float
    notes: tuple[str, ...] = ()

    def describe(self, unit: str) -> str:
        fused = f"; {', '.join(self.fused)}" if self.fused else ""
        covers = ", ".join(self.covers)
        notes = "".join(f" [{n}]" for n in self.notes)
        return (
            f"`{self.name}`: {self.family} x {self.tiles} tiles ({self.shape}{fused}) <- "
            f"{covers}; {self.bytes / 1e6:.4g} MB, {self.cost:.4g} {unit} per tile{notes}"
        )


@dataclass(frozen=True)
class Roofline:
    """Cost of a tile on one of ``share`` queues that stream at once: max(bytes / (DRAM
    bandwidth / share), FLOPs / (peak / share)) in µs (the GPU's rates split evenly between
    its busy SMs, so a wave of tiles on every queue takes its bytes over the whole
    bandwidth), or its bytes without a measured bandwidth (``unit`` says which); ``source``
    labels the peaks."""

    dram_gbps: float | None
    tflops: float | None
    source: str
    share: int = 1

    @property
    def unit(self) -> str:
        return "us" if self.dram_gbps else "bytes"

    def cost(self, nbytes: int, flops: int) -> float:
        if not self.dram_gbps:
            return float(nbytes)
        compute = flops * self.share / (self.tflops * 1e6) if self.tflops else 0.0
        return max(nbytes * self.share / (self.dram_gbps * 1e3), compute)

    def describe(self) -> str:
        if not self.dram_gbps:
            return f"bytes moved per tile ({self.source}: relative costs, not time)"
        peak = f", FLOPs / {self.tflops:.3g} TFLOP/s fp32 FMA" if self.tflops else ""
        share = (
            f"a 1/{self.share} share of" if self.share > 1 else "all of (pass queues for a share)"
        )
        return (
            f"roofline per tile on {share} max(bytes / {self.dram_gbps:.4g} GB/s{peak}), µs: "
            f"{self.source}"
        )


def roofline(peaks: Mapping[str, Any] | None, source: str = "", share: int = 1) -> Roofline:
    """The cost model of measured ``peaks`` (``kernel-agent doctor``'s, a ceilings table's
    ``peaks``): DRAM bandwidth and the fp32 matmul peak (the kit's opcodes compute with
    CUDA-core FMAs; a power-capped board's sustained rate where measured), each split
    between ``share`` queues."""
    from kernel_agent.kernels.roofline import floor_tflops

    if not peaks or not peaks.get("dram_gbps"):
        return Roofline(None, None, source or "no measured peaks (`kernel-agent doctor`)")
    tflops = floor_tflops(peaks).get("float32")
    gpu = peaks.get("gpu") or "this GPU"
    when = f", measured {peaks['measured_at']}" if peaks.get("measured_at") else ""
    label = f"{source or 'peaks'} of {gpu}{when}"
    return Roofline(float(peaks["dram_gbps"]), tflops, label, max(int(share), 1))


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _alpha(op: Rec) -> Any:
    return op.arg(2, "alpha", 1)


@dataclass
class _Unit:
    """One op of the DAG before tiling: a family and its operands (views of values)."""

    family: str
    covers: list[int]
    out: View
    x: View | None = None
    w: View | None = None
    gamma: View | None = None
    eps: float = 0.0
    a: View | None = None
    b: View | None = None
    res: View | None = None
    act: int = 0
    m: int = 1  # activation rows
    n: int = 0  # gemv: N; rmsnorm / argmax: the row; element-wise: the elements
    k: int = 0
    rows: int = 0
    splits: int = 1
    fused: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def operands(self) -> list[View]:
        return [v for v in (self.x, self.a, self.b, self.res) if v is not None]


class _Graph:
    """The recording's values: who wrote and who reads each, what the call returns."""

    def __init__(self, recording: Recording) -> None:
        self.rec = recording
        self.ops = recording.ops
        self.readers: dict[Value, list[int]] = {}
        for op in recording.ops:
            for v in op.tensors():
                readers = self.readers.setdefault(v.value, [])
                if op.index not in readers:
                    readers.append(op.index)
        self.outputs = {v.value for v in recording.outputs}
        self.input_keys = {v.key for v in recording.inputs}

    def producer(self, v: Any) -> Rec | None:
        if not isinstance(v, View) or v.version < 0:
            return None
        return self.ops[v.version]

    def inside(self, value: Value, ops: set[int]) -> bool:
        """Whether only ``ops`` read ``value`` and the call does not return it."""
        return value not in self.outputs and set(self.readers.get(value, [])) <= ops

    def const(self, v: Any) -> bool:
        """A storage the stage only reads that is not an input: a weight."""
        return (
            isinstance(v, View)
            and v.version < 0
            and v.key is not None
            and v.key not in self.rec.written
            and v.key not in self.input_keys
        )


def _rmsnorm(g: _Graph, r: Rec) -> _Unit | str | None:
    """The RMSNorm whose rsqrt is ``r`` (None: not a norm's; text: why it cannot map)."""
    add = g.producer(r.arg(0, "self"))
    if add is None or add.name not in ("add", "add_") or not _number(add.arg(1, "other")):
        return None
    if _alpha(add) != 1:
        return None
    mean = g.producer(add.arg(0, "self"))
    if mean is None or mean.name != "mean" or not isinstance(mean.arg(0, "self"), View):
        return None
    src: View = mean.arg(0, "self")
    dims = mean.arg(1, "dim")
    if not mean.arg(2, "keepdim", False) or not isinstance(dims, tuple) or len(dims) != 1:
        return None
    if dims[0] not in (-1, len(src.shape) - 1):
        return None
    sq = g.producer(src)
    if sq is None:
        return None
    exponent: Any = sq.arg(1, "exponent") if sq.name == "pow" else None
    squared = _number(exponent) and float(exponent) == 2
    if not squared and not (sq.name == "mul" and sq.arg(0, "self") == sq.arg(1, "other")):
        return None
    v = sq.arg(0, "self")
    if not isinstance(v, View):
        return None
    covers = [sq, mean, add, r]
    x, cast = v, g.producer(v)
    if cast is not None and cast.name == "_to_copy" and isinstance(cast.arg(0, "self"), View):
        x = cast.arg(0, "self")
        covers.insert(0, cast)
    found = g.readers.get(r.outs[0].value, [])
    if len(found) != 1:
        return "its rsqrt is read by more than the normalisation"
    norm = g.ops[found[0]]
    operands = {norm.arg(0, "self"), norm.arg(1, "other")}
    if norm.name != "mul" or operands != {v, r.outs[0]}:
        return "its rsqrt does not scale the input it squared"
    covers.append(norm)
    out = norm.outs[0]
    gamma: View | None = None
    rounded_first = True  # the cast back to the input's dtype before the weight (the opcode's)
    back = False
    for _ in range(2):
        nxt = g.readers.get(out.value, [])
        if len(nxt) != 1 or out.value in g.outputs:
            break
        op = g.ops[nxt[0]]
        if op.name == "_to_copy" and not back and op.outs and op.outs[0].dtype == x.dtype:
            back = True
        elif op.name == "mul" and gamma is None:
            other = op.arg(1, "other") if op.arg(0, "self") == out else op.arg(0, "self")
            if not g.const(other) or other.numel != x.shape[-1]:
                break
            gamma, rounded_first = other, back
        else:
            break
        covers.append(op)
        out = op.outs[0]
    inside = {op.index for op in covers}
    for op in covers[:-1]:
        for o in op.outs:
            if not g.inside(o.value, inside):
                return f"its intermediate `{op.name}` (#{op.index}) is read outside it"
    unit = _Unit(
        "rmsnorm",
        sorted(inside),
        out,
        x=x,
        gamma=gamma,
        eps=float(add.arg(1, "other")),
    )
    if v.dtype == x.dtype and x.dtype != "float32":
        unit.notes.append(f"normalised in {_SHORT.get(x.dtype)} by the reference, in fp32 here")
    elif gamma is not None and not rounded_first:
        unit.notes.append(
            "the reference scales by the weight before rounding to bf16, the opcode after"
        )
    return _norm_checks(unit)


def _norm_checks(unit: _Unit) -> _Unit | str:
    x, gamma, out = unit.x, unit.gamma, unit.out
    assert x is not None
    if x.dtype != "bfloat16" or out.dtype != "bfloat16":
        return f"normalises {x.describe()} into {out.describe()}: the opcode reads and writes bf16"
    if gamma is None:
        return "an RMSNorm without a weight: the opcode multiplies by one (pass a ones vector)"
    if gamma.dtype != "bfloat16" or not gamma.contiguous() or gamma.numel != x.shape[-1]:
        return f"its weight is {gamma.describe()}: the opcode reads a bf16 vector of the row"
    if not x.contiguous() or not out.contiguous() or x.shape[-1] % 8:
        return "a strided input or output, or a row not a multiple of 8 elements"
    unit.n = x.shape[-1]
    unit.m = x.numel // unit.n
    return unit


def _rmsnorm_op(g: _Graph, op: Rec) -> _Unit | str:
    """``aten.rms_norm`` / ``_fused_rms_norm`` recorded as one op."""
    x, shape, gamma = op.arg(0, "input"), op.arg(1, "normalized_shape"), op.arg(2, "weight")
    if not isinstance(x, View) or not isinstance(shape, tuple) or len(shape) != 1:
        return "an RMSNorm over more than the last dimension"
    eps = op.arg(3, "eps")
    if eps is None:
        import torch

        eps = float(torch.finfo(getattr(torch, x.dtype)).eps)  # torch's default
    for extra in op.outs[1:]:  # _fused_rms_norm's rstd
        if g.readers.get(extra.value) or extra.value in g.outputs:
            return "its rstd output is read: the opcode does not write it"
    unit = _Unit("rmsnorm", [op.index], op.outs[0], x=x, gamma=gamma, eps=float(eps))
    if gamma is not None and not g.const(gamma):
        return "its weight is written in the stage: the opcode reads a constant"
    unit.notes.append("the fused kernel's rounding of the weight may differ by a bf16 ulp")
    return _norm_checks(unit)


def _untranspose(wt: View) -> View | None:
    """The row-major [N, K] weight of which ``wt`` ([K, N]) is the transpose."""
    if len(wt.shape) != 2:
        return None
    k, n = wt.shape
    if wt.stride != (1, k) and not (n == 1 and wt.stride[0] == 1):
        return None
    return View(wt.key, wt.offset, (n, k), (k, 1), wt.dtype, wt.esize, wt.version)


def _gemv(g: _Graph, op: Rec) -> _Unit | str:
    if op.name == "linear":
        x, w = op.arg(0, "input"), op.arg(1, "weight")
        if op.arg(2, "bias") is not None:
            return "a linear with a bias: the kit's GEMV has no bias epilogue"
    elif op.name in ("mm", "matmul"):
        x, wt = op.arg(0, "self"), op.arg(1, "other")
        w = _untranspose(wt) if isinstance(wt, View) else None
        if w is None:
            return "its second operand is not a transposed row-major [N, K] weight"
    elif op.name == "mv":
        w, x = op.arg(0, "self"), op.arg(1, "vec")
    else:
        return _WHY.get(op.name, f"a {op.name}: no GEMV form")
    if not isinstance(x, View) or not isinstance(w, View) or len(w.shape) != 2:
        return "operands that are not an activation and a 2-D weight"
    if not g.const(w):
        why = "an input of the call" if w.key in g.input_keys else "written in the stage"
        return (
            f"its weight is {why}: a GEMV tile prefetches constant weights before its "
            "counters are met"
        )
    out = op.outs[0]
    if x.dtype != "bfloat16" or w.dtype != "bfloat16" or out.dtype != "bfloat16":
        return (
            f"{x.describe()} by {w.describe()}: the GEMV opcode takes bf16 (GEMV_FP8: e4m3 "
            "weights with per-row scales, which the recording does not show)"
        )
    n, k = w.shape
    if x.shape[-1] != k or not x.contiguous() or not out.contiguous() or w.stride != (k, 1):
        return "strided operands: the GEMV reads contiguous rows"
    if k % 8 or (w.offset * w.esize) % mks.PREFETCH_ALIGN:
        return "K not a multiple of 8 or a weight not 16-byte aligned (prefetches move 16 bytes)"
    m = x.numel // k
    if m > SKINNY_ROWS:
        return (
            f"a GEMM of {m} rows: GEMV tiles stream each weight tile once per row (at most "
            f"{SKINNY_ROWS}); a tensor-core GEMM kernel fits better"
        )
    return _Unit("gemv", [op.index], out, x=x, w=w, m=m, n=n, k=k)


def _residual(g: _Graph, op: Rec) -> _Unit | str:
    a, b = op.arg(0, "self"), op.arg(1, "other")
    if not isinstance(a, View) or not isinstance(b, View):
        return "an add of a scalar: no opcode"
    if _alpha(op) != 1:
        return "an add with alpha: no opcode"
    if a.shape != b.shape:
        return f"a broadcast add ({a.describe()} + {b.describe()}: a bias?): no opcode"
    out = op.outs[0]
    if {a.dtype, b.dtype, out.dtype} != {"bfloat16"}:
        return f"an add of {a.describe()} and {b.describe()}: the opcode adds bf16"
    if not (a.contiguous() and b.contiguous() and out.contiguous()):
        return "strided operands: the opcode reads contiguous ranges"
    return _Unit("residual", [op.index], out, a=a, b=b, n=a.numel)


_ACTS = {"silu": oc.GLU_SILU, "gelu": oc.GLU_GELU}


def _glu(g: _Graph, op: Rec, claimed: set[int]) -> _Unit | str | None:
    """``act(a) * b`` ending at the mul ``op`` (None: not a gated activation)."""
    p, q = op.arg(0, "self"), op.arg(1, "other")
    if not isinstance(p, View) or not isinstance(q, View):
        return None
    for gated, other in ((p, q), (q, p)):
        act = g.producer(gated)
        if act is None or act.name not in _ACTS or act.index in claimed:
            continue
        if not g.inside(gated.value, {op.index}):
            continue
        a = act.arg(0, "self")
        if not isinstance(a, View):
            continue
        code = _ACTS[act.name]
        if act.name == "gelu":
            approximate = act.arg(1, "approximate", "none")
            if approximate not in ("none", "tanh"):
                return f"gelu(approximate={approximate!r}): no opcode form"
            code = oc.GLU_GELU_TANH if approximate == "tanh" else oc.GLU_GELU
        out = op.outs[0]
        if {a.dtype, other.dtype, gated.dtype, out.dtype} != {"bfloat16"}:
            return f"a gated activation of {a.describe()}: the opcode reads and writes bf16"
        if a.shape != other.shape or a.shape != out.shape:
            return f"a broadcast gate ({a.describe()} x {other.describe()}): no opcode"
        if not (a.contiguous() and other.contiguous() and out.contiguous()):
            return "strided operands: the opcode reads contiguous ranges"
        return _Unit("glu", [act.index, op.index], out, a=a, b=other, act=code, n=a.numel)
    return None


def _argmax(g: _Graph, op: Rec) -> _Unit | str:
    x, dim = op.arg(0, "self"), op.arg(1, "dim")
    if not isinstance(x, View):
        return "an argmax of no tensor"
    if x.dtype != "bfloat16" or not x.contiguous():
        return f"an argmax over {x.describe()}: the opcode reads contiguous bf16 rows"
    if dim is None:
        n = x.numel
    elif dim in (-1, len(x.shape) - 1):
        n = x.shape[-1]
    else:
        return "an argmax over another dimension than the last"
    out = op.outs[0]
    if out.dtype != "int64" or not out.contiguous():
        return "an argmax into a strided or non-int64 output"
    return _Unit("argmax", [op.index], out, x=x, m=x.numel // max(n, 1), n=n)


def _describe(op: Rec) -> str:
    ins = ", ".join(v.describe() for v in op.tensors())
    outs = ", ".join(v.describe() for v in op.outs)
    return f"{ins} -> {outs}" if outs else ins


def _map(g: _Graph) -> tuple[list[_Unit], list[Unsupported]]:
    """Every recorded op into a family's unit, or unsupported with why."""
    units: list[_Unit] = []
    claimed: set[int] = set()
    hints: dict[int, str] = {}

    def take(found: _Unit | str | None, anchor: Rec) -> None:
        if isinstance(found, _Unit):
            if claimed.isdisjoint(found.covers):
                units.append(found)
                claimed.update(found.covers)
        elif isinstance(found, str):
            hints.setdefault(anchor.index, found)

    for op in g.ops:  # the patterns of several ops first
        if op.broken:
            continue
        if op.name == "rsqrt":
            take(_rmsnorm(g, op), op)
        elif op.name in ("rms_norm", "_fused_rms_norm"):
            take(_rmsnorm_op(g, op), op)
    for op in g.ops:
        if op.index in claimed or op.broken:
            continue
        if op.name == "mul":
            take(_glu(g, op, claimed), op)
        elif op.name in ("linear", "mm", "matmul", "mv"):
            take(_gemv(g, op), op)
        elif op.name in ("add", "add_"):
            take(_residual(g, op), op)
        elif op.name == "argmax":
            take(_argmax(g, op), op)
    unsupported = []
    for op in g.ops:
        if op.index in claimed:
            continue
        if op.broken:
            reason = "its arguments could not be recorded"
        else:
            reason = hints.get(op.index) or _why(op.name)
        unsupported.append(Unsupported(op.index, op.name, op.module, _describe(op), reason))
    units.sort(key=lambda u: max(u.covers))
    return units, unsupported


def _why(name: str) -> str:
    """Why an op no pattern took has no family."""
    from kernel_agent.profiling import fusion

    if name in _WHY:
        return _WHY[name]
    if "::" in name:
        return "a custom op: no opcode (write its opcode in the project's Ops)"
    if name in fusion._SYNC:
        return "a host synchronisation (the host reads the result): no instruction can"
    kind = fusion.anchor_kind(name)
    if kind == "attention":
        return "attention: split-KV attention and its combine are milestone 6 of megakernel.md"
    if kind == "conv":
        return "a convolution: no opcode"
    return f"no opcode family for `{name}`"


def _same(a: View, b: View) -> bool:
    return a.value == b.value and a.offset == b.offset and a.numel == b.numel


def _fuse(units: list[_Unit], g: _Graph) -> list[_Unit]:
    """The example's layer: RMSNorm prologues and residual epilogues of GEMVs."""
    by_value: dict[Value, list[_Unit]] = {}
    for u in units:
        for v in u.operands():
            by_value.setdefault(v.value, []).append(u)
    dropped: set[int] = set()
    for u in units:
        if u.family != "rmsnorm" or u.out.value in g.outputs:
            continue
        users = by_value.get(u.out.value, [])
        covered = {i for c in users for i in c.covers}
        if (
            users
            and set(g.readers.get(u.out.value, [])) <= covered
            and all(
                c.family == "gemv"
                and c.x is not None
                and _same(c.x, u.out)
                and c.k == u.n
                and c.k <= oc.MAX_SLICE
                and c.gamma is None
                for c in users
            )
        ):
            for c in users:
                c.x, c.gamma, c.eps = u.x, u.gamma, u.eps
                c.covers = sorted({*u.covers, *c.covers})
                c.fused.append("RMSNorm prologue" + (" (per GEMV)" if len(users) > 1 else ""))
                c.notes += u.notes
            dropped.add(id(u))
    producers = {u.out.value: u for u in units if id(u) not in dropped}
    for r in units:
        if r.family != "residual" or id(r) in dropped:
            continue
        assert r.a is not None and r.b is not None
        for mine, other in ((r.b, r.a), (r.a, r.b)):
            p = producers.get(mine.value)
            if (
                p is None
                or p.family != "gemv"
                or p.res is not None
                or not _same(mine, p.out)
                or not g.inside(mine.value, set(r.covers))
            ):
                continue
            p.res, p.out = other, r.out
            p.covers = sorted({*p.covers, *r.covers})
            p.fused.append("residual epilogue")
            dropped.add(id(r))
            break
    kept = [u for u in units if id(u) not in dropped]
    return sorted(kept, key=lambda u: max(u.covers))  # a unit after the producers it reads


def _splits(
    u: _Unit, pool_bytes: int, rows: int, split_k: int | str, queues: int | None
) -> str | None:
    """Rows per tile and K splits of a GEMV (None), or why no tile fits."""
    valid: list[tuple[int, int]] = []  # (splits, rows)
    for s in (1, 2, 4, 8, 16, 32, 64):
        ks = u.k // s
        if u.k % (8 * s) or ks > oc.MAX_SLICE:
            continue
        r = rows or oc.tile_rows(u.n, ks * 2, pool_bytes)
        if r > 0 and u.n % r == 0 and r <= oc.MAX_ROWS and r * ks * 2 <= pool_bytes:
            valid.append((s, r))
    if not valid:
        return (
            f"no GEMV tile of N = {u.n}, K = {u.k} fits a {pool_bytes}-byte page pool "
            f"({rows or 'any'} rows, K slices of at most {oc.MAX_SLICE})"
        )
    pick = valid[0]
    if split_k == "auto":
        want = (queues or 0) // 2
        wider = [
            (s, r) for s, r in valid if s >= pick[0] and (u.k // s >= MIN_SLICE or s == pick[0])
        ]
        pick = next((sr for sr in wider if u.m * u.n // sr[1] * sr[0] >= want), wider[-1])
    elif int(split_k) > pick[0]:
        pick = next((sr for sr in valid if sr[0] >= int(split_k)), valid[-1])
    u.splits, u.rows = pick  # a fused norm has K <= the slice (_fuse): any split takes it
    return None


# ------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class _Tile:
    args: list[int]
    prefetch: mks.Prefetch | None
    reads: list[tuple[int, int, int]]  # (slot, first byte, end byte)
    writes: list[tuple[int, int, int]]
    nbytes: int
    flops: int


@dataclass
class Plan:
    """A stage's op DAG for :func:`.schedule.build` (``ops``, ``edges``), what each op stands
    for (``info``), the tensor table (``slots``), the call's inputs and outputs in it, the
    unsupported ops, the cost model and notes."""

    ops: list[mks.Op]
    edges: list[mks.Edge]
    info: list[OpInfo]
    slots: list[Slot]
    inputs: list[Binding | None]
    outputs: list[Binding | None]
    unsupported: list[Unsupported]
    roofline: Roofline
    notes: list[str] = field(default_factory=list)
    label: str = ""
    recording: Recording | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        """Every recorded op is in an op of the plan."""
        return not self.unsupported and bool(self.ops)

    @property
    def cost_source(self) -> str:
        return self.roofline.describe()

    def schedule(self, queues: int, *, meta: Mapping[str, Any] | None = None) -> mks.Schedule:
        """The queues and counters (:func:`.schedule.build`) on ``queues`` resident blocks;
        :class:`CaptureError` when an op is unsupported (a schedule with a hole computes
        garbage) or nothing was recorded."""
        if self.unsupported:
            raise CaptureError(
                f"{len(self.unsupported)} unsupported op(s): "
                + "; ".join(u.describe() for u in self.unsupported[:5])
            )
        if not self.ops:
            raise CaptureError("no op recorded")
        info = {"source": "captured", "cost": self.cost_source, "label": self.label}
        return mks.build(self.ops, self.edges, queues, meta={**info, **(meta or {})})

    def tensors(self, module: Any = None, device: Any = None) -> list[Any]:
        """The runtime's tensor table (:class:`.runtime.Runtime`): every weight slot the
        storage of its parameter (``module``'s, by name, else the recorded tensor), a packed
        copy for a split GEMV, zeroed arenas; on ``device`` (default the weights')."""
        import torch

        sources = self.recording.tensors if self.recording is not None else {}
        weights: list[Any] = []
        for slot in self.slots:
            if slot.kind == "arena":
                weights.append(None)
                continue
            t = _named(module, slot.name) if module is not None else None
            if t is None:
                t = sources.get(slot.key) if slot.key is not None else None
            if t is None:
                raise CaptureError(f"no tensor for slot {slot.index} ({slot.name})")
            weights.append(t)
        if device is None:
            device = next((t.device for t in weights if t is not None), torch.device("cpu"))
        out = []
        for slot, t in zip(self.slots, weights, strict=True):
            if slot.kind == "arena":
                out.append(torch.zeros(slot.numel, dtype=getattr(torch, slot.dtype), device=device))
                continue
            base = torch.empty(0, dtype=t.dtype, device=t.device).set_(t.untyped_storage())
            if slot.kind == "packed":
                v = slot.view
                assert v is not None
                n, k = v.shape
                ks = k // slot.splits
                w = base.as_strided((n, k), (k, 1), v.offset)
                base = w.view(n, slot.splits, ks).permute(1, 0, 2).contiguous()
            out.append(base.to(device))
        return out

    def bind(self, tensors: Sequence[Any], binding: Binding) -> Any:
        """The tensor ``binding`` names in a table from :meth:`tensors` (a view)."""
        size = math.prod(binding.shape)
        flat = tensors[binding.slot][binding.offset : binding.offset + size]
        return flat.view(binding.shape)

    def describe(self) -> str:
        """The DAG, its edges, the cost model and the unsupported ops (markdown lines)."""
        lines = [
            f"Op DAG of `{self.label}`: {len(self.ops)} ops, {len(self.edges)} edges, "
            f"{len(self.slots)} tensor slots; costs: {self.cost_source}."
        ]
        unit = self.roofline.unit
        lines += [f"* {i.describe(unit)}" for i in self.info]
        for e in self.edges:
            needs = e.needs if isinstance(e.needs, str) else "per tile (chunked counters)"
            lines.append(f"* edge `{e.producer}` -> `{e.consumer}`: {needs}")
        lines += [f"* note: {n}" for n in self.notes]
        if self.unsupported:
            lines.append(f"Unsupported ({len(self.unsupported)}: no schedule until mapped):")
            lines += [f"* {u.describe()}" for u in self.unsupported]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        def needs(e: mks.Edge) -> Any:
            if isinstance(e.needs, TileMap):
                return [list(n) for n in e.needs.needs]
            return e.needs if isinstance(e.needs, str) else "function"

        return {
            "label": self.label,
            "cost_source": self.cost_source,
            "cost_unit": self.roofline.unit,
            "ops": [
                {
                    "name": i.name,
                    "family": i.family,
                    "opcode": i.opcode,
                    "tiles": i.tiles,
                    "covers": list(i.covers),
                    "shape": i.shape,
                    "fused": list(i.fused),
                    "bytes": i.bytes,
                    "flops": i.flops,
                    "cost_per_tile": i.cost,
                    "notes": list(i.notes),
                }
                for i in self.info
            ],
            "edges": [
                {"producer": e.producer, "consumer": e.consumer, "needs": needs(e)}
                for e in self.edges
            ],
            "slots": [
                {
                    "index": s.index,
                    "kind": s.kind,
                    "name": s.name,
                    "dtype": s.dtype,
                    "numel": s.numel,
                    "splits": s.splits,
                }
                for s in self.slots
            ],
            "inputs": [dataclasses.asdict(b) if b else None for b in self.inputs],
            "outputs": [dataclasses.asdict(b) if b else None for b in self.outputs],
            "unsupported": [u.describe() for u in self.unsupported],
            "notes": list(self.notes),
        }


def _named(module: Any, name: str) -> Any:
    for get in ("get_parameter", "get_buffer"):
        try:
            return getattr(module, get)(name)
        except (AttributeError, TypeError):
            continue
    return None


class _Layout:
    """The tensor table: weight and constant slots by storage, then one arena per dtype with
    a slice per value (placed in the order :func:`build` meets them)."""

    def __init__(self, names: Mapping[Key, str]) -> None:
        self.names = names
        self.slots: list[Slot] = []  # weights and constants; the arenas after finish()
        self.tags: dict[tuple[Any, int], int] = {}
        self.sizes: dict[str, list[int]] = {}  # dtype -> [elements so far, element size]
        self.slices: dict[Any, tuple[str, int, int, int]] = {}  # value -> dtype, start, lo, hi
        self.arena: dict[str, int] = {}  # dtype -> its slot (finish())

    def weight(self, v: View, kind: str, splits: int = 1) -> int:
        """The slot of a constant's storage (a packed copy per split count)."""
        tag = (v.key, splits if kind == "packed" else 0)
        if tag not in self.tags:
            name = (self.names.get(v.key, "") if v.key is not None else "") or (
                f"constant{len(self.slots)}"
            )
            numel = v.numel if kind == "packed" else 0
            self.slots.append(Slot(len(self.slots), kind, name, v.dtype, numel, v.key, v, splits))
            self.tags[tag] = len(self.slots) - 1
        return self.tags[tag]

    def place(self, value: Any, dtype: str, esize: int, lo: int, hi: int) -> None:
        """Give ``value`` (its storage's elements [lo, hi)) a slice of its dtype's arena,
        starting on :data:`ALIGN` bytes; a placed value keeps its slice."""
        if value in self.slices:
            return
        size = self.sizes.setdefault(dtype, [0, esize])
        per = max(ALIGN // esize, 1)
        start = -(-size[0] // per) * per
        size[0] = start + hi - lo
        self.slices[value] = (dtype, start, lo, hi)

    def finish(self) -> None:
        """The arena slots, after every weight and constant, in the order of first use."""
        for dtype, (numel, _) in self.sizes.items():
            self.arena[dtype] = len(self.slots)
            short = _SHORT.get(dtype, dtype)
            self.slots.append(Slot(len(self.slots), "arena", f"{short} arena", dtype, numel))

    def at(self, value: Any, offset: int = 0) -> tuple[int, int]:
        """(slot, element offset) of element ``offset`` of a placed value's region."""
        dtype, start, lo, _ = self.slices[value]
        return self.arena[dtype], start + offset - lo

    def loc(self, v: View) -> tuple[int, int]:
        """(slot, element offset) of a view of a placed value."""
        _, _, lo, hi = self.slices[v.value]
        if v.offset < lo or v.offset + v.numel > hi:
            raise CaptureError(f"a view {v.describe()} outside its value's slice")
        return self.at(v.value, v.offset)


def _region(v: View) -> tuple[int, int, int]:
    return v.esize, v.offset, v.offset + v.numel


def build(
    recording: Recording,
    *,
    pool_bytes: int = POOL_BYTES,
    rows: int = 0,
    split_k: int | str = 1,
    queues: int | None = None,
    fuse: bool = True,
    peaks: Mapping[str, Any] | None = None,
    peaks_source: str = "",
    l2_bytes: int | None = None,
    opcodes: Mapping[str, int] | None = None,
) -> Plan:
    """The op DAG of ``recording`` (the module docstring). ``pool_bytes``: the launch's page
    pool (pages × page bytes); ``rows``: output rows per GEMV tile (0: the most up to 16
    that fit it, :func:`.opcodes.tile_rows`); ``split_k``: K splits per GEMV beyond those
    its width needs (an int: at least that many; ``"auto"``: enough for the tiles of each
    GEMV to fill half the ``queues``); ``fuse``: norm prologues and residual epilogues;
    ``peaks``: measured peaks for the costs (None: bytes; ``peaks_source`` labels them);
    ``l2_bytes``: the L2 the prefetches' eviction hint is chosen for (None: unknown, normal);
    ``opcodes``: a project's numbers per family (default :data:`OPCODES`)."""
    codes = {**OPCODES, **(opcodes or {})}
    model = roofline(peaks, peaks_source, queues or 1)
    g = _Graph(recording)
    units, unsupported = _map(g)
    if recording.unrecorded:
        lost = f"{recording.unrecorded} op(s) the recorder could not record"
        unsupported.insert(0, Unsupported(-1, "the recording", "", "", lost))
    if fuse:
        units = _fuse(units, g)
    for u in units:
        if u.family == "gemv" and (why := _splits(u, pool_bytes, rows, split_k, queues)):
            for i in u.covers:
                op = g.ops[i]
                unsupported.append(Unsupported(i, op.name, op.module, _describe(op), why))
    unsupported.sort(key=lambda x: x.op)
    units = [u for u in units if u.family != "gemv" or u.rows]

    layout = _Layout(recording.names)
    for u in units:  # the streamed weights first, in op order, then the other constants
        if u.w is not None:
            layout.weight(u.w, "packed" if u.splits > 1 else "weight", u.splits)
    for u in units:
        if u.gamma is not None:
            layout.weight(u.gamma, "const")
    read = {v.value for u in units for v in u.operands()}
    for v in recording.inputs:
        if v.value in read or v.value in g.outputs:
            layout.place(v.value, v.dtype, *_region(v))
    for i, u in enumerate(units):
        for v in u.operands():  # an unsupported op's value: a slice of its own as well
            layout.place(v.value, v.dtype, *_region(_written(g, v)))
        if u.splits > 1:
            layout.place(("part", i), "float32", 4, 0, u.splits * u.m * u.n)
        layout.place(u.out.value, u.out.dtype, *_region(u.out))
    layout.finish()

    streamed = [s.view for s in layout.slots if s.view is not None and s.kind != "const"]
    weight_bytes = sum(v.numel * v.esize for v in streamed)
    hint = oc.l2_hint(weight_bytes, l2_bytes)
    counts: dict[str, int] = {}
    pieces: list[tuple[str, _Piece]] = []
    for i, u in enumerate(units):
        for piece in _tiles(u, i, layout, hint):
            k = counts.get(piece.family, 0)
            counts[piece.family] = k + 1
            pieces.append((f"{_PREFIX.get(piece.family, piece.family)}{k}", piece))
    ops, info = [], []
    for k, (name, piece) in enumerate(pieces):
        costs = [model.cost(t.nbytes, t.flops) for t in piece.tiles]
        prefetch = [t.prefetch for t in piece.tiles]
        ops.append(
            mks.Op(
                name,
                codes[piece.family],
                len(piece.tiles),
                cost=costs[0] if len(set(costs)) == 1 else costs,
                args=[t.args for t in piece.tiles],
                weights=prefetch if any(p is not None for p in prefetch) else None,
            )
        )
        info.append(
            OpInfo(
                name,
                piece.family,
                codes[piece.family],
                len(piece.tiles),
                tuple(g.ops[c].label() for c in piece.covers)
                or (f"the partials of `{pieces[k - 1][0]}`",),  # a split GEMV's reduce
                piece.shape,
                tuple(piece.fused),
                sum(t.nbytes for t in piece.tiles),
                sum(t.flops for t in piece.tiles),
                max(costs),
                tuple(piece.notes),
            )
        )
    edges = _edges([(name, piece.tiles) for name, piece in pieces])
    notes = []
    if hint == mks.EVICT_FIRST and l2_bytes:
        notes.append(
            f"weights {weight_bytes / 1e6:.4g} MB > half the {l2_bytes / 2**20:.3g} MB L2: "
            "prefetched evict-first"
        )
    for i, v in enumerate(recording.outputs):
        if v.version < 0:
            notes.append(f"output {i} is not written in the stage (an input or a weight)")
    for v in recording.inputs:
        if v.key in recording.written:
            notes.append(
                f"the stage writes its input {v.describe()} in place: the plan writes a new "
                "slice (copy it back)"
            )
    return Plan(
        ops,
        edges,
        info,
        layout.slots,
        [_binding(layout, v) for v in recording.inputs],
        [_binding(layout, v) for v in recording.outputs],
        unsupported,
        model,
        notes,
        recording.label,
        recording,
    )


def _written(g: _Graph, v: View) -> View:
    """The region of ``v``'s value: what its producer wrote there (``v`` for an input)."""
    op = g.producer(v)
    for out in op.outs if op is not None else ():
        if out.key == v.key:
            return out
    return v


def _binding(layout: _Layout, v: View) -> Binding | None:
    if v.value not in layout.slices:
        return None
    slot, offset = layout.loc(v)
    return Binding(slot, offset, v.shape, v.dtype)


#: Op names: the family's prefix and a count.
_PREFIX = {"splitk_reduce": "reduce"}


@dataclass
class _Piece:
    """One op of the plan: a unit's tiles (a split GEMV makes two pieces)."""

    family: str
    tiles: list[_Tile]
    covers: list[int]
    shape: str
    fused: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _rng(slot: int, offset: int, count: int, esize: int) -> tuple[int, int, int]:
    return slot, offset * esize, (offset + count) * esize


def _tiles(u: _Unit, index: int, layout: _Layout, hint: int) -> list[_Piece]:
    """The tiles of unit ``index``: one op, or a split GEMV's partials and its reduce."""
    os_, oo = layout.loc(u.out)
    if u.family == "gemv":
        return _gemv_tiles(u, index, layout, hint, os_, oo)
    tiles: list[_Tile] = []
    if u.family == "rmsnorm":
        assert u.x is not None and u.gamma is not None
        xs, xo = layout.loc(u.x)
        gs = layout.weight(u.gamma, "const")
        for m in range(u.m):
            args = oc.rmsnorm_args(
                x=xs, x_off=xo + m * u.n, gamma=gs, gamma_off=u.gamma.offset, eps=u.eps,
                out=os_, out_off=oo + m * u.n, n=u.n,
            )  # fmt: skip
            reads, writes = [_rng(xs, xo + m * u.n, u.n, 2)], [_rng(os_, oo + m * u.n, u.n, 2)]
            tiles.append(_Tile(args, None, reads, writes, 6 * u.n, 4 * u.n))
        shape = f"{u.m} x {u.n}"
    elif u.family in ("residual", "glu"):
        assert u.a is not None and u.b is not None
        (as_, ao), (bs, bo) = layout.loc(u.a), layout.loc(u.b)
        for i0 in range(0, u.n, oc.ELEMENTWISE_TILE):
            cnt = min(oc.ELEMENTWISE_TILE, u.n - i0)
            if u.family == "residual":
                args = oc.residual_args(
                    a=as_, a_off=ao, b=bs, b_off=bo, out=os_, out_off=oo, i0=i0, n=cnt
                )
            else:
                args = oc.glu_args(
                    a=as_, a_off=ao, b=bs, b_off=bo, out=os_, out_off=oo, i0=i0, n=cnt, act=u.act
                )
            reads = [_rng(as_, ao + i0, cnt, 2), _rng(bs, bo + i0, cnt, 2)]
            flops = cnt if u.family == "residual" else 8 * cnt
            tiles.append(_Tile(args, None, reads, [_rng(os_, oo + i0, cnt, 2)], 6 * cnt, flops))
        shape = f"{u.n} elements"
    else:  # argmax
        assert u.x is not None
        xs, xo = layout.loc(u.x)
        for m in range(u.m):
            args = oc.argmax_args(x=xs, x_off=xo + m * u.n, n=u.n, out=os_, out_off=oo + m)
            reads, writes = [_rng(xs, xo + m * u.n, u.n, 2)], [_rng(os_, oo + m, 1, 8)]
            tiles.append(_Tile(args, None, reads, writes, 2 * u.n + 8, u.n))
        shape = f"{u.m} x {u.n}"
    return [_Piece(u.family, tiles, u.covers, shape, u.fused, u.notes)]


def _gemv_tiles(
    u: _Unit, index: int, layout: _Layout, hint: int, os_: int, oo: int
) -> list[_Piece]:
    """A GEMV's tiles: per K split, activation row and row tile (in that order), weights
    prefetched; a split one writes fp32 partials ([splits][M][N]) and a reduce per row tile
    sums them (and adds the residual)."""
    assert u.x is not None and u.w is not None
    xs, xo = layout.loc(u.x)
    gs, go = (layout.weight(u.gamma, "const"), u.gamma.offset) if u.gamma is not None else (-1, 0)
    rs, ro = layout.loc(u.res) if u.res is not None else (-1, 0)
    n, k, m_rows, rows, splits = u.n, u.k, u.m, u.rows, u.splits
    ks, per, norm = k // splits, n // rows, u.gamma is not None
    w_slot = layout.weight(u.w, "packed" if splits > 1 else "weight", splits)
    ps, po = layout.at(("part", index)) if splits > 1 else (-1, 0)
    partials: list[_Tile] = []
    for s in range(splits):
        for m in range(m_rows):
            for r in range(per):
                row0 = r * rows
                # with the norm fused a tile reads all of x (its sum of squares)
                x_lo, x_len = (xo + m * k, k) if norm or splits == 1 else (xo + m * k + s * ks, ks)
                reads = [_rng(xs, x_lo, x_len, 2)]
                if splits == 1:
                    if rs >= 0:
                        reads.append(_rng(rs, ro + m * n + row0, rows, 2))
                    args = oc.gemv_args(
                        x=xs, x_off=xo + m * k, gamma=gs, gamma_off=go, eps=u.eps, out=os_,
                        out_off=oo + m * n, res=rs, res_off=ro + m * n, row0=row0, rows=rows,
                        k=k,
                    )  # fmt: skip
                    pf = mks.Prefetch(w_slot, (u.w.offset + row0 * k) * 2, rows * k * 2, hint)
                    writes = [_rng(os_, oo + m * n + row0, rows, 2)]
                    out_bytes = (4 if rs >= 0 else 2) * rows
                else:
                    args = oc.gemv_args(
                        x=xs, x_off=xo + m * k, gamma=gs, gamma_off=go, eps=u.eps, out=-1,
                        out_off=0, row0=row0, rows=rows, k=k, k0=s * ks, klen=ks, part=ps,
                        part_off=po + (s * m_rows + m) * n,
                    )  # fmt: skip
                    pf = mks.Prefetch(w_slot, (s * n + row0) * ks * 2, rows * ks * 2, hint)
                    writes = [_rng(ps, po + (s * m_rows + m) * n + row0, rows, 4)]
                    out_bytes = 4 * rows
                nbytes = pf.nbytes + 2 * x_len + (2 * k if norm else 0) + out_bytes
                flops = 2 * rows * ks + (3 * k if norm else 0)
                partials.append(_Tile(args, pf, reads, writes, nbytes, flops))
    split = f", K split {splits} ways" if splits > 1 else ""
    shape = f"M={m_rows}, N={n}, K={k}, {rows} rows per tile{split}"
    if splits == 1:
        return [_Piece("gemv", partials, u.covers, shape, u.fused, u.notes)]
    reduce: list[_Tile] = []
    for m in range(m_rows):
        for r in range(per):
            row0 = r * rows
            reads = [_rng(ps, po + (s * m_rows + m) * n + row0, rows, 4) for s in range(splits)]
            if rs >= 0:
                reads.append(_rng(rs, ro + m * n + row0, rows, 2))
            args = oc.reduce_args(
                part=ps, part_off=po + m * n, splits=splits, stride=m_rows * n, row0=row0,
                rows=rows, out=os_, out_off=oo + m * n, res=rs, res_off=ro + m * n,
            )  # fmt: skip
            writes = [_rng(os_, oo + m * n + row0, rows, 2)]
            nbytes = 4 * rows * splits + (4 if rs >= 0 else 2) * rows
            reduce.append(_Tile(args, None, reads, writes, nbytes, rows * splits))
    epilogue = [f for f in u.fused if f == "residual epilogue"]
    return [
        _Piece(
            "gemv", partials, u.covers, shape, [f for f in u.fused if f not in epilogue], u.notes
        ),
        _Piece("splitk_reduce", reduce, [], f"{splits} partials of M={m_rows}, N={n}", epilogue),
    ]


def _edges(pieces: Sequence[tuple[str, list[_Tile]]]) -> list[mks.Edge]:
    """Producer → consumer at tile granularity: a consumer tile needs the producer tiles
    whose written byte ranges its read ranges overlap (every value has one writer)."""
    writes: dict[int, list[tuple[int, int, int, int]]] = {}  # slot -> (lo, hi, op, tile)
    for o, (_, tiles) in enumerate(pieces):
        for t, tile in enumerate(tiles):
            for slot, lo, hi in tile.writes:
                writes.setdefault(slot, []).append((lo, hi, o, t))
    starts: dict[int, list[int]] = {}
    for slot, found in writes.items():
        found.sort()
        for (_, hi, a, _), (lo, _, b, _) in itertools.pairwise(found):
            if lo < hi:
                raise CaptureError(f"ops {a} and {b} write the same bytes of slot {slot}")
        starts[slot] = [w[0] for w in found]
    edges = []
    for c, (consumer, tiles) in enumerate(pieces):
        needs: dict[int, list[set[int]]] = {}
        for t, tile in enumerate(tiles):
            for slot, lo, hi in tile.reads:
                found = writes.get(slot, [])
                i = max(bisect.bisect_right(starts.get(slot, []), lo) - 1, 0)
                while i < len(found) and found[i][0] < hi:
                    _, w_hi, p, pt = found[i]
                    if w_hi > lo and p != c:
                        needs.setdefault(p, [set() for _ in tiles])[t].add(pt)
                    i += 1
        for p in sorted(needs):
            sets = [tuple(sorted(s)) for s in needs[p]]
            if any(not s for s in sets):  # every family reads all of what it reads from p
                raise CaptureError(f"a tile of {consumer} reads nothing of {pieces[p][0]}")
            edges.append(mks.Edge(pieces[p][0], consumer, _canonical(sets, len(pieces[p][1]))))
    return edges


def _canonical(sets: list[tuple[int, ...]], producer_tiles: int) -> str | TileMap:
    """:data:`.schedule.ALL`, :data:`.schedule.SAME` or the per-tile map."""
    every = tuple(range(producer_tiles))
    if all(s == every for s in sets):
        return mks.ALL
    if len(sets) == producer_tiles and all(s == (t,) for t, s in enumerate(sets)):
        return mks.SAME
    return TileMap(tuple(sets))


# ------------------------------------------------------------------ captures and modules


def peaks_here() -> tuple[dict[str, Any] | None, str]:
    """The measured peaks of the GPU at hand (``kernel-agent doctor``'s cache) and their
    label, or (None, why)."""
    try:
        from kernel_agent.kernels.roofline import current_peaks

        peaks = current_peaks()
    except Exception as exc:  # no GPU, no cache: costs in bytes
        return None, f"no peaks ({type(exc).__name__})"
    if not peaks:
        return None, "no measured peaks of this GPU (`kernel-agent doctor` measures them)"
    return peaks, "`kernel-agent doctor`'s peaks"


def from_module(
    module: Any,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    method: str = "forward",
    **options: Any,
) -> Plan:
    """:func:`build` of one recorded call (:func:`record`) of ``module``: in a project's
    ``build()``, the module the engine replaces on an input of the call it serves. Its
    weights are the tensor table's (:meth:`Plan.tensors`)."""
    return build(record(module, args, kwargs, method=method), **options)


def main_case(cases: Sequence[Mapping[str, Any]]) -> int:
    """The case a stage's schedule serves: the most calls per run (``count``), the first on
    a tie (a decode step's, for a decode stage)."""
    if not cases:
        raise CaptureError("the capture has no case")
    return max(range(len(cases)), key=lambda i: (int(cases[i].get("count") or 0), -i))


def from_capture(
    path: str | Path, *, case: int | None = None, device: str = "cpu", **options: Any
) -> Plan:
    """:func:`build` of one captured call: case ``case`` (default :func:`main_case`) of the
    capture file ``path`` (a target's ``capture.pt`` or ``capture_inputs.pt``), its module
    loaded on ``device`` with the state that call saw."""
    from kernel_agent.profiling import state
    from kernel_agent.profiling.capture import load_capture

    capture = load_capture(Path(path), device)
    module = capture["module"]
    cases = capture.get("cases") or []
    index = main_case(cases) if case is None else int(case)
    if not 0 <= index < len(cases):
        raise CaptureError(f"case {index} of {len(cases)}")
    chosen = cases[index]
    state.Replay(capture, module).restore(chosen, module)
    method = str(chosen.get("method") or "forward")
    label = f"{capture.get('qualname') or type(module).__name__}.{method} case {index}"
    if chosen.get("signature"):
        label += f" ({chosen['signature']})"
    recording = record(
        module, chosen.get("args") or (), chosen.get("kwargs") or {}, method=method, label=label
    )
    return build(recording, **options)


# ------------------------------------------------------------------ command line


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", type=int, default=None, help="case index (default: most calls)")
    parser.add_argument("--device", default="cpu", help="where to record (default cpu)")
    parser.add_argument("--queues", type=int, default=0, help="resident blocks (default: SMs)")
    parser.add_argument("--pool-bytes", type=int, default=POOL_BYTES, help="the page pool")
    parser.add_argument("--rows", type=int, default=0, help="rows per GEMV tile (0: fit)")
    parser.add_argument("--split-k", default="1", help="K splits per GEMV: N or auto")
    parser.add_argument("--no-fuse", action="store_true", help="no norm / residual fusion")
    parser.add_argument("--peaks", type=Path, default=None, help="a peaks JSON")
    parser.add_argument("--ceilings", type=Path, default=None, help="a ceilings.json")
    parser.add_argument("--no-peaks", action="store_true", help="costs in bytes")
    parser.add_argument("--l2-mb", type=float, default=None, help="L2 for the prefetch hint")
    parser.add_argument("--draws", type=int, default=1000, help="simulated SM-speed draws")
    parser.add_argument("--json", action="store_true", help="print JSON")


def run(path: Path, ns: argparse.Namespace) -> int:
    """``--from-capture``: the plan, its schedule and the simulator's verdict."""
    from kernel_agent import toolchain
    from kernel_agent.native.megakernel import simulate

    peaks: Mapping[str, Any] | None = None
    source = ""
    if ns.ceilings is not None:
        peaks = (json.loads(ns.ceilings.read_text()) or {}).get("peaks")
        source = f"the ceilings table {ns.ceilings}"
    elif ns.peaks is not None:
        peaks, source = json.loads(ns.peaks.read_text()), f"peaks {ns.peaks}"
    elif not ns.no_peaks:
        peaks, source = peaks_here()
    gpu = toolchain.gpu_info() if not ns.queues or ns.l2_mb is None else None
    l2 = ns.l2_mb if ns.l2_mb is not None else (peaks or {}).get("l2_mb")
    if l2 is None and gpu is not None:
        l2 = gpu.l2_cache_mb
    queues = ns.queues or (gpu.sm_count if gpu is not None else 0)
    split: int | str = ns.split_k if ns.split_k == "auto" else int(ns.split_k)
    try:
        plan = from_capture(
            path,
            case=ns.case,
            device=ns.device,
            pool_bytes=ns.pool_bytes,
            rows=ns.rows,
            split_k=split,
            queues=queues or None,
            fuse=not ns.no_fuse,
            peaks=peaks,
            peaks_source=source,
            l2_bytes=int(l2 * 2**20) if l2 else None,
        )
    except Exception as exc:
        print(f"cannot build the op DAG: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    out: dict[str, Any] = plan.to_dict()
    text = [plan.describe()]
    code = 0
    if not plan.ok:
        code = 1
    elif not queues:
        text.append("No schedule: pass --queues (no GPU here: one queue per resident block).")
        code = 1
    else:
        sched = plan.schedule(queues)
        problems = simulate.check(sched, draws=ns.draws)
        sim = simulate.simulate(sched)
        out["schedule"] = {
            "queues": queues,
            "describe": sched.describe(),
            "words": len(sched.words()),
            "problems": problems,
            "predicted": sim.makespan,
        }
        text += [
            "",
            f"Schedule on {queues} queues: {sched.describe()}",
            f"Simulator over {ns.draws} random SM-speed draws: "
            + ("no deadlock, no early start" if not problems else problems[0]),
            simulate.report(sched, sim, " " + plan.roofline.unit),
        ]
        code = 1 if problems else 0
    print(json.dumps(out, indent=1) if ns.json else "\n".join(text))
    return code
