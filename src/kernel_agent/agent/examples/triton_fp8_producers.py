"""Example candidate (Triton): producers that emit FP8. RMSNorm and ``silu(gate) * up`` write
e4m3 codes plus their scales in the epilogue, so the next W8A8 GEMM reads one byte per
activation and no separate quantisation pass runs.

Reduced precision: ``"precision": "fp8_w8a8"`` (a ``--quality near-lossless`` run,
knowledge/low_precision.md); the exact tier rejects it. The numerics are the class's: the
producer computes its output as eager does (bf16 roundings included), then quantises it per
token (``scale = amax / 448``, dynamic, every call; ``quant.quantize_fp8_activations``) or,
with ``mx=True``, per 32 elements with a power-of-two ue8m0 scale (MXFP8, the
non-saturating ``2^ceil(log2(amax / 448))`` rule of ``quant.quantize_mxfp8``; scales
row-major ``[rows, K / 32]`` for ``tl.dot_scaled``, or ``blocked=True``: cuBLASLt's 128 x 4
layout, ``quant.mx_scale_offset``). MXFP8 output feeds an MXFP8 GEMM: a ``"fp8_mx"`` target,
not ``fp8_w8a8``'s per-token contract.

Why (docs/FP8.md §4, RTX 5070 Ti, CUDA graph, M = 352): quantising in the producer costs
almost nothing, a separate pass does not. RMSNorm [352, 1024]: bf16 out 1.35 us, fused
per-token e4m3 1.48 us, bf16 + separate quantisation 2.67 us; ``silu(gate) * up``
[352, 8192] -> [352, 4096]: bf16 out 3.34 us, fused e4m3 3.03 us (half the bytes written),
separate 5.50 us; fused MXFP8 1.54 / 3.35 us. A per-token scale needs the whole row (the
amax), which a row-wise producer has; a GEMM epilogue tile or a per-head attention CTA does
not (use 1 x 32 / 1 x 128 group scales there).

* :func:`rmsnorm_fp8` (``x * rsqrt(mean(x^2) + eps)`` in fp32, cast to bf16, times the
  weight in bf16: the math of Llama / Qwen / Mistral / MiniCPM RMSNorm; adapt the two lines
  for other variants), :func:`silu_mul_fp8` (``F.silu(gate) * up`` in bf16 from separate or
  merged ``[..., 2F]`` gate|up), :func:`quantize_rows` (the plain pass, for inputs that
  arrive in bf16). One program per row, the row in registers (K <= :data:`MAX_K`).
* ``build()`` takes a gated MLP with ``gate_proj`` / ``up_proj`` / ``down_proj`` Linears
  (no bias) and a SiLU ``act_fn`` (LlamaMLP-style, any size): gate|up merged into one W8A8
  GEMM, the SiLU-mul producer emits e4m3 + per-token scales for ``down_proj``. The GEMMs
  are ``torch._scaled_mm`` (row-wise scales; swap in ``triton_fp8_w8a8_gemm.py``'s kernel or
  ``cuda_cublaslt_fp8.py``'s ``fp8_gemm`` as needed). The MLP's own input arrives in bf16
  (its RMSNorm sits outside the module), so it takes :func:`quantize_rows`; in a decoder
  layer target call :func:`rmsnorm_fp8` instead.

Not run on a GPU yet (written from the measured kernels of
docs/research-scripts/fp8-sm120/fusion_bench.py); verify with ``kernel-agent doctor
--smoke`` and ``pytest -m gpu tests/test_fp8_toolkit.py``.
"""

import re

import torch
import triton
import triton.language as tl
from torch import nn

from kernel_agent.kernels.quant import fp8_error, fp8_w8a8_linear, quantize_fp8

# One namespace per candidate file (the evaluator names the module after the file's hash).
_NS = re.sub(r"\W", "_", __name__)

#: Longest row one program keeps in registers (a longer row needs a two-pass kernel).
MAX_K = 16384


@triton.jit
def _ue8m0(amax):
    # smallest e with amax / 2^e <= 448, ceil(log2(amax / 448)), exact from the bits of
    # amax = m * 2^E (m in [1, 2)): E - 8, plus one when m > 1.75 (448 = 1.75 * 2^8); zero /
    # subnormal maxima -127; at most 126 (2^-e stays normal). quant.quantize_mxfp8's rule.
    bits = amax.to(tl.int32, bitcast=True)  # amax >= 0: no sign bit
    field = (bits >> 23) & 0xFF
    e = field - 127 - 8 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    e = tl.where(field == 0, -127, e)
    return tl.minimum(tl.maximum(e, -127), 126)


@triton.jit
def _emit_fp8(
    h, row, live, cols, K, q, s, MX: tl.constexpr, BLOCKED: tl.constexpr, BLOCK: tl.constexpr
):
    # h: the producer's fp32 row (values already rounded as eager rounds them), 0 past K.
    # Writes e4m3 codes q[row, :K] and the row's scales into s.
    mask = (cols < K) & live
    if MX:
        g = tl.reshape(h, (BLOCK // 32, 32))
        e = _ue8m0(tl.max(tl.abs(g), 1))
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)  # 2^-e, exact
        codes = tl.clamp(g * inv[:, None], -448.0, 448.0).to(tl.float8e4nv)
        tl.store(q + row * K + cols, tl.reshape(codes, (BLOCK,)), mask=mask)
        gcols = tl.arange(0, BLOCK // 32)
        byte = tl.where(live, e + 127, 0).to(tl.uint8)  # padding rows: 0, as swizzle_mx_scales
        if BLOCKED:  # cuBLASLt's 128 x 4 layout (rows padded to 128)
            ncb = (K // 32 + 3) // 4
            r = row % 128
            off = (
                ((row // 128) * ncb + gcols // 4) * 512 + (r % 32) * 16 + (r // 32) * 4 + gcols % 4
            )
        else:  # row-major [rows, K / 32], what tl.dot_scaled loads
            off = row * (K // 32) + gcols
        tl.store(s + off, byte, mask=gcols < K // 32)
    else:
        # IEEE divisions (div_rn), as quant.quantize_fp8_activations: the same codes
        amax = tl.max(tl.abs(h), 0)
        scale = tl.math.div_rn(amax, tl.full(amax.shape, 448.0, tl.float32))
        scale = tl.where(amax > 0, scale, 1.0)  # a row of zeros: scale 1, codes 0
        codes = tl.clamp(tl.math.div_rn(h, tl.zeros_like(h) + scale), -448.0, 448.0)
        tl.store(q + row * K + cols, codes.to(tl.float8e4nv), mask=mask)
        tl.store(s + row, scale, mask=live)


@triton.jit
def _quant_rows_kernel(
    x, q, s, M, K, ldx, MX: tl.constexpr, BLOCKED: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    live = row < M
    cols = tl.arange(0, BLOCK)
    h = tl.load(x + row * ldx + cols, mask=(cols < K) & live, other=0.0).to(tl.float32)
    _emit_fp8(h, row, live, cols, K, q, s, MX, BLOCKED, BLOCK)


@triton.jit
def _rmsnorm_fp8_kernel(
    x, w, q, s, M, K, ldx, eps, MX: tl.constexpr, BLOCKED: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    live = row < M
    cols = tl.arange(0, BLOCK)
    mask = (cols < K) & live
    v = tl.load(x + row * ldx + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 0) / K + eps)
    hn = (v * r).to(tl.bfloat16).to(tl.float32)  # .to(input_dtype), as eager
    wv = tl.load(w + cols, mask=cols < K, other=0.0).to(tl.float32)
    h = (hn * wv).to(tl.bfloat16).to(tl.float32)  # weight * h in bf16
    _emit_fp8(h, row, live, cols, K, q, s, MX, BLOCKED, BLOCK)


@triton.jit
def _silu_mul_fp8_kernel(
    g, u, q, s, M, K, ldg, ldu, MX: tl.constexpr, BLOCKED: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    live = row < M
    cols = tl.arange(0, BLOCK)
    mask = (cols < K) & live
    gv = tl.load(g + row * ldg + cols, mask=mask, other=0.0).to(tl.float32)
    uv = tl.load(u + row * ldu + cols, mask=mask, other=0.0).to(tl.float32)
    a = (gv / (1.0 + tl.exp(-gv))).to(tl.bfloat16).to(tl.float32)  # F.silu in bf16
    h = (a * uv).to(tl.bfloat16).to(tl.float32)  # * up in bf16
    _emit_fp8(h, row, live, cols, K, q, s, MX, BLOCKED, BLOCK)


def _rows(t: torch.Tensor) -> torch.Tensor:
    t2 = t.reshape(-1, t.shape[-1])
    return t2 if t2.stride(-1) == 1 else t2.contiguous()


def _outputs(
    m: int, k: int, device: torch.device, mx: bool, blocked: bool
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(codes [m, k] e4m3, scales, programs): scales fp32 [m] per token; MXFP8 uint8 [m, k / 32]
    row-major, or flat blocked (rows padded to 128: the padding programs write unit scales)."""
    if mx and k % 32:
        raise ValueError(f"MXFP8 needs K % 32 == 0, got {k}")
    q = torch.empty((m, k), device=device, dtype=torch.float8_e4m3fn)
    if not mx:
        return q, torch.empty((m,), device=device, dtype=torch.float32), m
    if not blocked:
        return q, torch.empty((m, k // 32), device=device, dtype=torch.uint8), m
    rows = -(-m // 128) * 128
    cols = -(-(k // 32) // 4) * 4
    return q, torch.empty((rows * cols,), device=device, dtype=torch.uint8), rows


def _launch(kernel, args, k: int, programs: int) -> None:
    block = triton.next_power_of_2(k)
    kernel[(programs,)](*args, BLOCK=block, num_warps=4 if block <= 4096 else 8)


def quantize_rows(
    x: torch.Tensor, mx: bool = False, blocked: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scales)`` of bf16 ``x [..., K]``: the plain quantisation pass (per token,
    or MXFP8 with ``mx``)."""
    x2 = _rows(x)
    m, k = x2.shape
    q, s, programs = _outputs(m, k, x.device, mx, blocked)
    if m:
        args = (x2, q, s, m, k, x2.stride(0), mx, blocked)
        _launch(_quant_rows_kernel, args, k, programs)
    return q, s


def rmsnorm_fp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float, mx: bool = False, blocked: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scales)`` of ``RMSNorm(x) * weight`` (bf16 ``x [..., K]``), quantised in the
    norm's epilogue: no bf16 output is written."""
    x2 = _rows(x)
    m, k = x2.shape
    q, s, programs = _outputs(m, k, x.device, mx, blocked)
    if m:
        args = (x2, weight, q, s, m, k, x2.stride(0), float(eps), mx, blocked)
        _launch(_rmsnorm_fp8_kernel, args, k, programs)
    return q, s


def silu_mul_fp8(
    gate: torch.Tensor, up: torch.Tensor | None = None, mx: bool = False, blocked: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scales)`` of ``F.silu(gate) * up`` (bf16), quantised in the epilogue. Without
    ``up``, ``gate`` is the merged ``[..., 2F]`` output of a gate|up GEMM (gate first)."""
    g2 = _rows(gate)
    if up is None:
        f = g2.shape[1] // 2
        g2, u2 = g2[:, :f], g2[:, f:]
    else:
        u2 = _rows(up)
    m, k = g2.shape
    q, s, programs = _outputs(m, k, gate.device, mx, blocked)
    if m:
        args = (g2, u2, q, s, m, k, g2.stride(0), u2.stride(0), mx, blocked)
        _launch(_silu_mul_fp8_kernel, args, k, programs)
    return q, s


def _scaled_mm(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor):
    # W8A8 with row-wise scales on the FP8 tensor cores (as quant.fp8_w8a8_linear)
    return torch._scaled_mm(
        xq, wq.t(), scale_a=xs[:, None], scale_b=ws[None, :], out_dtype=torch.bfloat16
    )


def _w8a8_mlp(
    x: torch.Tensor, w_gu: torch.Tensor, s_gu: torch.Tensor, w_d: torch.Tensor, s_d: torch.Tensor
) -> torch.Tensor:
    """down(silu(gate(x)) * up(x)), every GEMM W8A8; x [..., H] bf16, w_gu [2F, H], w_d [H, F]."""
    xq, xs = quantize_rows(x)  # in a decoder layer: rmsnorm_fp8 of the residual stream
    gu = _scaled_mm(xq, xs, w_gu, s_gu)  # [M, 2F] bf16, scaled
    hq, hs = silu_mul_fp8(gu)  # the producer: e4m3 + per-token scale, no bf16 h
    return _scaled_mm(hq, hs, w_d, s_d).reshape(*x.shape[:-1], w_d.shape[0])


w8a8_mlp = torch.library.custom_op(f"{_NS}::w8a8_mlp", _w8a8_mlp, mutates_args=())


@w8a8_mlp.register_fake
def _(x, w_gu, s_gu, w_d, s_d):
    return x.new_empty((*x.shape[:-1], w_d.shape[0]))


class Fp8ProducerMLP(nn.Module):
    """A gated SiLU MLP with W8A8 GEMMs and the SiLU-mul emitting e4m3 for ``down_proj``."""

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        gate, up, down = reference.gate_proj, reference.up_proj, reference.down_proj
        w_gu = torch.cat([gate.weight.detach(), up.weight.detach()])  # one GEMM for both
        q_gu, s_gu = quantize_fp8(w_gu)  # once, here: never per call
        q_d, s_d = quantize_fp8(down.weight.detach())
        self.register_buffer("gate_up_fp8", q_gu)
        self.register_buffer("gate_up_scale", s_gu)
        self.register_buffer("down_fp8", q_d)
        self.register_buffer("down_scale", s_d)
        self.quant_error = {  # for NOTES.md
            "gate_up": fp8_error(w_gu, q_gu, s_gu),
            "down": fp8_error(down.weight, q_d, s_d),
        }
        self._args = (q_gu, s_gu, q_d, s_d)  # plain attributes: no nn.Module lookups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda:  # the same numerics in torch
            q_gu, s_gu, q_d, s_d = self._args
            gu = fp8_w8a8_linear(x, q_gu, s_gu)
            g, u = gu.chunk(2, dim=-1)
            return fp8_w8a8_linear(nn.functional.silu(g) * u, q_d, s_d)
        launch = w8a8_mlp if torch.compiler.is_compiling() else _w8a8_mlp
        return launch(x, *self._args)


def _is_silu(act: object) -> bool:
    return act is nn.functional.silu or type(act).__name__ in ("SiLU", "SiLUActivation")


def build(reference: nn.Module) -> nn.Module:
    linears = [getattr(reference, n, None) for n in ("gate_proj", "up_proj", "down_proj")]
    if not all(isinstance(lin, nn.Linear) for lin in linears):
        return reference
    gate, up, down = linears
    hidden, inter = gate.in_features, gate.out_features
    ok = (
        _is_silu(getattr(reference, "act_fn", None))
        and all(lin.bias is None and lin.weight.dtype == torch.bfloat16 for lin in linears)
        and gate.weight.is_cuda
        and torch.cuda.get_device_capability(gate.weight.device) >= (8, 9)  # e4m3 MMA
        and (up.in_features, up.out_features) == (hidden, inter)
        and (down.in_features, down.out_features) == (inter, hidden)
        and hidden % 16 == 0
        and inter % 16 == 0  # torch._scaled_mm
        and max(hidden, inter) <= MAX_K
    )
    return Fp8ProducerMLP(reference) if ok else reference
