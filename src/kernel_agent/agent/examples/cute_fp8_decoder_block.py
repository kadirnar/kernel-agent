"""Example candidate (CuTe DSL): a fused small-M decoder-layer building block with FP8
weights. ``RMSNorm -> gate|up -> silu(gate) * up`` in one launch and ``down (+ residual)``
in a second: the MLP half of a pre-norm decoder layer, ``x + down(silu(gate(n)) * up(n))``
with ``n = RMSNorm(x)``, at decode-like row counts (M <= 16).

Status: written for issue #133 and traced on the CPU; **not yet run on a GPU**.
``kernel-agent doctor --smoke`` and ``pytest -m gpu tests/test_cute_examples.py`` run its
selftest. Evaluate a copy with ``mode="quick"`` before relying on it.

Reduced precision: ``fp8_weights`` (weight-only; activations stay bf16): e4m3 weights with
one fp32 scale per output channel (``kernel_agent.kernels.quant.quantize_fp8``), quantised
once in ``build()``. At M <= 16 the weight bytes set the floor (memory bound: e.g. a
[16, 1024] x [1024, 8192] gate|up streams 8.4 MB of e4m3, ~11 us at 767 GB/s), so FP8
weight-only halves the bytes of bf16 while W8A8 would add an activation quantisation for no
speed (docs/FP8.md §3: at decode, weight-only is also the more accurate recipe).

``_RowsGemv`` (one ``@cute.kernel`` specialised by compile-time flags):

* ``norm``: every CTA computes the inverse RMS of each row (a warp per row, fp32 sum of
  squares, ``warp_reduction_sum``) and stages ``RMSNorm(x)`` chunk by chunk in shared
  memory with the reference's roundings (``(x * inv).to(bf16)``, then ``* weight`` and bf16
  again, as LlamaRMSNorm-style modules do). No normalised copy goes through global memory.
* weights stream once: a warp owns ``COLS_PER_WARP`` output channels for the whole K loop;
  each lane loads 16 e4m3 codes per channel and K chunk (one 128-bit load), converts them
  to fp32 in registers and multiplies them with the M staged rows (fp32 FMAs: at M <= 16
  the math is far below the bytes; tensor cores would not help a memory-bound GEMV).
* ``gated``: the weight is ``cat([gate, up])``; the warp accumulates both rows of its channel
  and the epilogue writes ``silu(g * s_g) * (u * s_u)`` in bf16 (one rounding);
* ``residual``: ``+ r[m, n]`` in the epilogue (the decoder layer's skip connection).
* rows are bucketed to a power of two (``M_MAX`` 1..16, one compilation each, cached across
  evaluations by ``kernel_agent.cute_dsl.compile_cached``); rows past M are computed on a
  copy of the last row and never stored. K must be a multiple of 16 (K chunks of 512 with a
  guarded tail). More than 16 rows: the dequantised weights through cuBLAS (a fallback with
  the same weight numerics; at M >= 128 use ``cute_fp8_blockscaled_gemm.py``).

``build(reference)`` accepts a gated MLP module (``gate_proj`` / ``up_proj`` / ``down_proj``
``nn.Linear`` without bias and a SiLU ``act_fn``, the common Hugging Face layout) and runs
it in two launches. For a decoder-layer target, call :func:`norm_mlp_residual` with the
layer's post-attention norm weight and epsilon: norm + MLP + residual in the same two
launches (``rmsnorm`` and the residual add disappear as separate kernels).

Policy (docs/RESEARCH-TRITON.md §5.1): small-M GEMVs go to CUDA C++ first
(``cuda_fp8_skinny_gemm.py``, bf16 ``mma.sync`` on upcast weights); this block shows the
fusion boundary in CuTe DSL, the second backend for fused decoder layers, and is the
starting point for a persistent CuTe layer.
"""

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from torch import nn

from kernel_agent.kernels.quant import dequantize_fp8, fp8_error, quantize_fp8

try:  # an on-disk cache of the compiled kernels across evaluations (kernel_agent/cute_dsl.py)
    from kernel_agent.cute_dsl import compile_cached
except ImportError:  # pragma: no cover - outside kernel-agent
    compile_cached = None

THREADS = 256
WARPS = THREADS // 32
COLS_PER_WARP = 2  # output channels a warp owns for the whole K loop
VEC = 16  # e4m3 codes per lane and channel per K chunk (one 128-bit load)
KC = 32 * VEC  # K chunk staged in shared memory (512)
MAX_ROWS = 16


class _RowsGemv:
    """``y[m, n] = epi(sum_k a[m, k] * w8[n, k] * s[n])`` for ``m < M <= m_max`` rows, where
    ``a`` is ``x`` or ``RMSNorm(x)`` (``norm``), ``epi`` is ``silu(g) * u`` over the gate and
    up rows (``gated``) and adds ``r[m, n]`` (``residual``)."""

    def __init__(self, *, m_max: int, k: int, n_out: int, norm: bool, gated: bool, residual: bool):
        self.m_max, self.k, self.n_out = m_max, k, n_out
        self.norm, self.gated, self.residual = norm, gated, residual

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,  # (M, K) bf16
        norm_weight: cute.Tensor,  # (K,) bf16 (unused without norm)
        w: cute.Tensor,  # (N_rows, K) uint8 e4m3 codes, N_rows = 2 n_out when gated
        w_scale: cute.Tensor,  # (N_rows,) fp32
        r: cute.Tensor,  # (R, n_out) bf16 (unused without residual)
        y: cute.Tensor,  # (M, n_out) bf16
        eps: cutlass.Float32,
        stream,
    ):
        w8 = cute.make_tensor(cute.recast_ptr(w.iterator, dtype=cutlass.Float8E4M3FN), w.layout)
        cols_per_cta = WARPS * COLS_PER_WARP
        grid = ((self.n_out + cols_per_cta - 1) // cols_per_cta, 1, 1)
        self.kernel(x, norm_weight, w8, w_scale, r, y, eps).launch(
            grid=grid, block=(THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gNW: cute.Tensor,
        gW: cute.Tensor,
        gS: cute.Tensor,
        gR: cute.Tensor,
        gY: cute.Tensor,
        eps: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        rows = gX.shape[0]
        k = self.k
        m_max = self.m_max

        smem = cutlass.utils.SmemAllocator()
        s_x = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((m_max, KC), stride=(KC, 1)), byte_alignment=16
        )
        s_inv = smem.allocate_tensor(cutlass.Float32, cute.make_layout(m_max))

        # 1. inverse RMS of every row (a warp per row)
        if cutlass.const_expr(self.norm):
            for row in range(warp, m_max, WARPS):
                src = cutlass.min(row, rows - 1)
                acc = cutlass.Float32(0.0)
                for c in range(lane, k, 32):
                    v = gX[src, c].to(cutlass.Float32)
                    acc += v * v
                acc = cute.arch.warp_reduction_sum(acc)
                if lane == 0:
                    s_inv[row] = cute.math.rsqrt(acc / k + eps)
            cute.arch.barrier()

        # 2. stream the weights of this warp's channels, K chunk by K chunk
        acc_g = cute.make_rmem_tensor((COLS_PER_WARP, m_max), cutlass.Float32)
        acc_u = cute.make_rmem_tensor((COLS_PER_WARP, m_max), cutlass.Float32)
        acc_g.fill(0.0)
        acc_u.fill(0.0)
        wr = cute.make_rmem_tensor((1, VEC), cutlass.Float8E4M3FN)
        wf = cute.make_rmem_tensor((1, VEC), cutlass.Float32)
        col0 = (bidx * WARPS + warp) * COLS_PER_WARP
        for kc in range(0, k, KC):
            # stage the (normalised) activations of this chunk; rows past M copy row M - 1
            for idx in range(tidx, m_max * KC, THREADS):
                row = idx // KC
                c = idx % KC
                kk = cutlass.min(kc + c, k - 1)
                v = gX[cutlass.min(row, rows - 1), kk]
                if cutlass.const_expr(self.norm):
                    nrm = (v.to(cutlass.Float32) * s_inv[row]).to(cutlass.BFloat16)
                    v = (nrm.to(cutlass.Float32) * gNW[kk].to(cutlass.Float32)).to(cutlass.BFloat16)
                s_x[row, c] = v
            cute.arch.barrier()
            k_lane = kc + lane * VEC
            if k_lane < k:
                for j in cutlass.range_constexpr(COLS_PER_WARP):
                    col = cutlass.min(col0 + j, self.n_out - 1)  # past the end: recomputed
                    cute.autovec_copy(cute.local_tile(gW, (1, VEC), (col, k_lane // VEC)), wr)
                    wf.store(wr.load().to(cutlass.Float32))
                    for row in cutlass.range_constexpr(m_max):
                        part = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(VEC):
                            part += wf[i] * s_x[row, lane * VEC + i].to(cutlass.Float32)
                        acc_g[j, row] = acc_g[j, row] + part
                    if cutlass.const_expr(self.gated):
                        cute.autovec_copy(
                            cute.local_tile(gW, (1, VEC), (col + self.n_out, k_lane // VEC)), wr
                        )
                        wf.store(wr.load().to(cutlass.Float32))
                        for row in cutlass.range_constexpr(m_max):
                            part = cutlass.Float32(0.0)
                            for i in cutlass.range_constexpr(VEC):
                                part += wf[i] * s_x[row, lane * VEC + i].to(cutlass.Float32)
                            acc_u[j, row] = acc_u[j, row] + part
            cute.arch.barrier()

        # 3. epilogue: reduce over the lanes, scales, silu(g) * u, residual, one bf16 rounding
        for j in cutlass.range_constexpr(COLS_PER_WARP):
            col = col0 + j
            colc = cutlass.min(col, self.n_out - 1)
            for row in cutlass.range_constexpr(m_max):
                g = cute.arch.warp_reduction_sum(acc_g[j, row]) * gS[colc]
                h = g
                if cutlass.const_expr(self.gated):
                    u = cute.arch.warp_reduction_sum(acc_u[j, row]) * gS[colc + self.n_out]
                    h = g / (cutlass.Float32(1.0) + cute.math.exp(-g)) * u
                if cutlass.const_expr(self.residual):
                    h = h + gR[cutlass.min(row, rows - 1), colc].to(cutlass.Float32)
                if lane == 0 and row < rows and col < self.n_out:
                    gY[row, col] = h.to(cutlass.BFloat16)


def _compile(fn, *args, key):
    options = "--enable-tvm-ffi"
    if compile_cached is not None:
        return compile_cached(fn, *args, key=key, options=options)
    return cute.compile(fn, *args, options=options)


def compile_rows_gemv(
    m_max: int,
    k: int,
    n_out: int,
    *,
    norm: bool = False,
    gated: bool = False,
    residual: bool = False,
):
    """The kernel for up to ``m_max`` rows of ``k`` features and ``n_out`` outputs."""
    m = cute.sym_int()
    n_rows = 2 * n_out if gated else n_out
    x = make_fake_compact_tensor(cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16)
    nw = make_fake_compact_tensor(cutlass.BFloat16, (k,), assumed_align=2)
    w = make_fake_compact_tensor(cutlass.Uint8, (n_rows, k), stride_order=(1, 0), assumed_align=16)
    s = make_fake_compact_tensor(cutlass.Float32, (n_rows,), assumed_align=4)
    r = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), n_out), stride_order=(1, 0), assumed_align=2
    )
    y = make_fake_compact_tensor(cutlass.BFloat16, (m, n_out), stride_order=(1, 0), assumed_align=2)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    op = _RowsGemv(m_max=m_max, k=k, n_out=n_out, norm=norm, gated=gated, residual=residual)
    key = ("rows_gemv", m_max, k, n_out, norm, gated, residual, COLS_PER_WARP, VEC)
    return _compile(op, x, nw, w, s, r, y, cutlass.Float32(1e-6), stream, key=key)


def _bucket(rows: int) -> int:
    m = 1
    while m < rows:
        m *= 2
    return m


def _is_silu(act) -> bool:
    if isinstance(act, nn.SiLU):
        return True
    try:
        t = torch.linspace(-4.0, 4.0, 17)
        return bool(torch.allclose(act(t), F.silu(t)))
    except Exception:
        return False


class Fp8GatedMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))`` with e4m3 weights (per-channel scales), two launches
    for up to 16 rows; ``norm_weight`` / ``residual`` turn it into the decoder layer's MLP
    half (:func:`norm_mlp_residual`)."""

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        gate, up, down = reference.gate_proj, reference.up_proj, reference.down_proj
        self.hidden, self.inter = gate.in_features, gate.out_features
        w_gu = torch.cat([gate.weight, up.weight], dim=0)
        self.gu_q, self.gu_s = quantize_fp8(w_gu)  # once, here: never per call
        self.down_q, self.down_s = quantize_fp8(down.weight)
        self.gu_u8, self.down_u8 = self.gu_q.view(torch.uint8), self.down_q.view(torch.uint8)
        self.quant_error = {  # for NOTES.md
            "gate_up": fp8_error(w_gu, self.gu_q, self.gu_s),
            "down": fp8_error(down.weight, self.down_q, self.down_s),
        }
        dev = gate.weight.device
        bf16 = torch.bfloat16
        # placeholders for the inputs a launch does not use (no norm, no residual)
        self._ones_h = torch.ones(self.hidden, device=dev, dtype=bf16)
        self._ones_i = torch.ones(self.inter, device=dev, dtype=bf16)
        self._zeros_h = torch.zeros((1, self.hidden), device=dev, dtype=bf16)
        self._zeros_i = torch.zeros((1, self.inter), device=dev, dtype=bf16)
        self._kernels: dict[tuple, object] = {}

    def _kernel(self, m_max: int, which: str, norm: bool, residual: bool):
        key = (m_max, which, norm, residual)
        fn = self._kernels.get(key)
        if fn is None:
            if which == "gate_up":
                fn = compile_rows_gemv(m_max, self.hidden, self.inter, norm=norm, gated=True)
            else:
                fn = compile_rows_gemv(m_max, self.inter, self.hidden, residual=residual)
            self._kernels[key] = fn
        return fn

    def _fallback(self, x, norm_weight, eps, residual):
        h = x
        if norm_weight is not None:
            v = x.float()
            h = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
            h = norm_weight * h
        gu = F.linear(h, dequantize_fp8(self.gu_q, self.gu_s, h.dtype))
        g, u = gu.split(self.inter, dim=-1)
        y = F.linear(F.silu(g) * u, dequantize_fp8(self.down_q, self.down_s, h.dtype))
        return y if residual is None else residual + y

    def run(self, x, norm_weight=None, eps: float = 1e-6, residual=None):
        shape = x.shape
        x2 = x.reshape(-1, self.hidden)
        rows = x2.shape[0]
        ok = x.is_cuda and x.dtype == torch.bfloat16 and 0 < rows <= MAX_ROWS
        if not ok:
            return self._fallback(x, norm_weight, eps, residual)
        x2 = x2.contiguous()
        m_max = _bucket(rows)
        h = torch.empty((rows, self.inter), device=x.device, dtype=torch.bfloat16)
        nw = self._ones_h if norm_weight is None else norm_weight.detach().contiguous()
        gate_up = self._kernel(m_max, "gate_up", norm_weight is not None, False)
        gate_up(x2, nw, self.gu_u8, self.gu_s, self._zeros_i, h, eps)
        y = torch.empty((rows, self.hidden), device=x.device, dtype=torch.bfloat16)
        res = self._zeros_h if residual is None else residual.reshape(-1, self.hidden).contiguous()
        down = self._kernel(m_max, "down", False, residual is not None)
        down(h, self._ones_i, self.down_u8, self.down_s, res, y, eps)
        return y.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(x)


def norm_mlp_residual(x, norm_weight, eps: float, mlp: Fp8GatedMLP) -> torch.Tensor:
    """``x + mlp(RMSNorm(x))`` for a decoder-layer candidate: two launches."""
    return mlp.run(x, norm_weight=norm_weight, eps=eps, residual=x)


def build(reference: nn.Module) -> nn.Module:
    lins = [getattr(reference, n, None) for n in ("gate_proj", "up_proj", "down_proj")]
    ok = (
        all(isinstance(m, nn.Linear) and m.bias is None for m in lins)
        and all(m.weight.dtype == torch.bfloat16 and m.weight.is_cuda for m in lins)
        and _is_silu(getattr(reference, "act_fn", None))
        and lins[0].in_features % VEC == 0
        and lins[0].out_features % VEC == 0
    )
    if not ok:
        return reference
    return Fp8GatedMLP(reference)


# ------------------------------------------------------------------ selftest (GPU)


def selftest(hidden: int = 1024, inter: int = 4096, rows: int = 16, seed: int = 0) -> dict:
    """Run on a GPU (sm_89+): the fused block against the same math in torch on the
    dequantised weights (norm + gate|up + silu·up + down + residual), for every row bucket.
    Returns the worst relative L2 per row count; raises AssertionError past 1e-2."""

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            kw = {"bias": False, "device": "cuda", "dtype": torch.bfloat16}
            self.gate_proj = nn.Linear(hidden, inter, **kw)
            self.up_proj = nn.Linear(hidden, inter, **kw)
            self.down_proj = nn.Linear(inter, hidden, **kw)
            self.act_fn = nn.SiLU()

    torch.manual_seed(seed)
    mlp = build(_MLP())
    assert isinstance(mlp, Fp8GatedMLP), "build() fell back to the reference"
    norm_w = (torch.randn(hidden, device="cuda") * 0.1 + 1).to(torch.bfloat16)
    out = {}
    for m in sorted({1, 2, 3, 8, rows}):
        x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
        got = norm_mlp_residual(x, norm_w, 1e-6, mlp).float()
        want = mlp._fallback(x, norm_w, 1e-6, x).float()
        out[m] = float((got - want).norm() / want.norm().clamp_min(1e-30))
        assert out[m] < 1e-2, f"{m} rows: relative L2 {out[m]:.3g}"
    return out
