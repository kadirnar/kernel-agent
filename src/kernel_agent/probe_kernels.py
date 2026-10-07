"""Triton kernels of the ``doctor`` probes (:mod:`kernel_agent.probes`).

Kept in their own module: Triton reads a kernel's source and globals, and this module
imports ``triton`` at the top, which :mod:`kernel_agent.probes` must not (doctor runs
without Triton too).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def dot_scaled_kernel(
    a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """One ``BM x BN`` tile of e4m3 ``a [BM, BK] @ b [BK, BN]`` with ue8m0 scales per 32
    along K (``sa [BM, BK // 32]``, ``sb [BN, BK // 32]``): ``tl.dot_scaled``."""
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rs = tl.arange(0, BK // 32)
    a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
    sa = tl.load(sa_ptr + rm[:, None] * (BK // 32) + rs[None, :])
    sb = tl.load(sb_ptr + rn[:, None] * (BK // 32) + rs[None, :])
    c = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3")
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], c)


@triton.jit
def tma_copy_kernel(src_desc, dst_desc, BM: tl.constexpr, BN: tl.constexpr):
    """Copy one ``BM x BN`` tile through host-built TMA descriptors."""
    pid = tl.program_id(0)
    tile = src_desc.load([pid * BM, 0])
    dst_desc.store([pid * BM, 0], tile)
