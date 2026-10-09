"""The megakernel kit's generic opcodes (issue #225): their numbers, argument slots and
limits, as the instruction words of a :class:`~.schedule.Op` hold them.

The device code is the example project's ``include/mk_ops.cuh``
(``kernel_agent/agent/examples/native_megakernel``; its ``Opcode`` enum and argument enums
mirror this file, ``tests/test_megakernel_schedule.py`` checks it): a project that copies
the example keeps these numbers and slots, and :mod:`.captured` maps a stage's recorded ops
onto them (``opcodes=``: a project's own numbers for the same families).

==============  ==========================================================================
RMSNORM         y = gamma * bf16(x * rsqrt(mean(x^2) + eps)) (one vector per instruction)
GEMV            rows [row0, row0 + rows) of y = W h, bf16 weights from the page pool; h = x
                or the RMSNorm of x fused in; a residual epilogue or fp32 split-K partials
GEMV_FP8        the same with e4m3 weights and an fp32 scale per output row
RESIDUAL        y = a + b over a range of elements
SPLITK_REDUCE   y[r] = bf16(sum_s part[s][r]) (+ a residual)
ARGMAX          out = argmax(x[0:n]) (int64; the first maximum, NaN the largest, as torch)
GLU             y = bf16(act(a)) * b over a range (act: silu, gelu tanh, gelu erf)
==============  ==========================================================================
"""

from __future__ import annotations

import struct

from kernel_agent.native.megakernel.schedule import EVICT_FIRST, EVICT_NORMAL

NOP, RMSNORM, GEMV, GEMV_FP8, RESIDUAL, SPLITK_REDUCE, ARGMAX, GLU = range(8)
#: ``GLU``'s activation argument.
GLU_SILU, GLU_GELU_TANH, GLU_GELU = range(3)

#: Bytes of one page of the pool (``mk::kPageBytes``).
PAGE_BYTES = 8192
#: Columns a GEMV tile stages (``mk::kMaxSlice``): K of a slice, and all of K with the norm
#: fused (x sits in registers).
MAX_SLICE = 4096
#: Output rows of one GEMV tile at most (``32 * kWarps``).
MAX_ROWS = 256
#: Output rows per GEMV tile where they fit the page pool (the whole-row path takes up to 16).
ROWS = 16
#: Elements per instruction of the element-wise opcodes (RESIDUAL, GLU): 8 per thread.
ELEMENTWISE_TILE = 2048


def f32_bits(value: float) -> int:
    """A float argument as the int32 word the device reads with ``ctx.arg_f``."""
    return int(struct.unpack("<i", struct.pack("<f", value))[0])


def gemv_args(
    *,
    x: int,
    x_off: int,
    out: int,
    out_off: int,
    row0: int,
    rows: int,
    k: int,
    gamma: int = -1,
    gamma_off: int = 0,
    eps: float = 0.0,
    res: int = -1,
    res_off: int = 0,
    k0: int = 0,
    klen: int | None = None,
    part: int = -1,
    part_off: int = 0,
    scale: int = -1,
    scale_off: int = 0,
) -> list[int]:
    """Arguments of a GEMV / GEMV_FP8 tile (tensors by table index, offsets in elements):
    rows [row0, row0 + rows) of ``W h`` over columns [k0, k0 + klen); ``gamma``: fuse the
    RMSNorm of x (over all k columns); ``res``: add a residual; ``part``: write fp32
    partials there instead (split-K); ``scale``: the e4m3 weights' per-row scales."""
    return [
        x, x_off, gamma, gamma_off, f32_bits(eps), out, out_off, res, res_off, row0, rows, k,
        k0, k if klen is None else klen, part, part_off, scale, scale_off,
    ]  # fmt: skip


def rmsnorm_args(
    *, x: int, x_off: int, gamma: int, gamma_off: int, eps: float, out: int, out_off: int, n: int
) -> list[int]:
    return [x, x_off, gamma, gamma_off, f32_bits(eps), out, out_off, n]


def residual_args(
    *, a: int, a_off: int, b: int, b_off: int, out: int, out_off: int, i0: int, n: int
) -> list[int]:
    return [a, a_off, b, b_off, out, out_off, i0, n]


def glu_args(
    *, a: int, a_off: int, b: int, b_off: int, out: int, out_off: int, i0: int, n: int, act: int
) -> list[int]:
    """``out[i] = bf16(act(a[i])) * b[i]`` for i in [i0, i0 + n) (``act``: :data:`GLU_SILU`,
    :data:`GLU_GELU_TANH` or :data:`GLU_GELU`)."""
    return [a, a_off, b, b_off, out, out_off, i0, n, act]


def reduce_args(
    *,
    part: int,
    part_off: int,
    splits: int,
    stride: int,
    row0: int,
    rows: int,
    out: int,
    out_off: int,
    res: int = -1,
    res_off: int = 0,
) -> list[int]:
    return [part, part_off, splits, stride, row0, rows, out, out_off, res, res_off]


def argmax_args(*, x: int, x_off: int, n: int, out: int, out_off: int) -> list[int]:
    return [x, x_off, n, out, out_off]


def tile_rows(n_out: int, row_bytes: int, pool_bytes: int, want: int = ROWS) -> int:
    """The most output rows per GEMV tile, at most ``want``, that divide ``n_out`` and whose
    weights (rows x ``row_bytes``) fit a page pool of ``pool_bytes`` (0: not one row fits).
    16 bf16 rows of a 4096-wide layer are 128 KB: more than the pool of a GPU with 99 KB of
    shared memory per block (an A10 or an RTX 5070 Ti: 11 pages of 8 KB), so 8 there; an
    A100's 163 KB holds 16."""
    rows = min(want, n_out)
    while rows > 0 and (n_out % rows or rows * row_bytes > pool_bytes):
        rows -= 1
    return rows


def l2_hint(weight_bytes: int, l2_bytes: int | None) -> int:
    """The prefetch L2 policy of a stage's weights: evict-first when they exceed half the
    L2 (streamed once per call, they leave the program and the activations in L2), else
    normal (``schedule.EVICT_*``; an unknown L2: normal)."""
    return EVICT_FIRST if l2_bytes is not None and weight_bytes > l2_bytes // 2 else EVICT_NORMAL
