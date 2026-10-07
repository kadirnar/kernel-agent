"""Example candidate (Triton): cheap launches for an eager-timed module with several kernels
per call, here a pre-norm gated MLP block ``x + down(silu(gate(norm(x))) * up(norm(x)))``.

The module evaluator times eager calls, so every launch's host time counts. Triton's JIT
launcher (``kernel[grid](...)``) binds and specialises the arguments on every call, tens of
microseconds each; a candidate with a few Triton kernels per call becomes host bound
(docs/RESEARCH-TRITON.md §4.3: the same fused decoder layer measured 2.21x with 5 Triton
launches per call, 2.70x behind one C++ entry). ``CachedLaunch``
(``kernel_agent.kernels.triton_launch``) keeps the ``CompiledKernel`` of each
specialisation and calls its C launcher directly from the second call on: the kernels,
grids and arguments stay exactly as with ``kernel[grid](...)``.

The block itself:

* RMSNorm in one Triton kernel (the reference's eager RMSNorm launches about eight
  kernels: casts, pow, mean, add, rsqrt, two multiplies); normalised in fp32, cast to the
  input dtype, then multiplied by the weight, the reference's two roundings.
* gate and up as one cuBLAS GEMM on the concatenated weight (``torch.cat`` once in
  ``build()``), ``silu(gate) * up`` in one Triton kernel reading both halves of its output
  (two roundings, as ``F.silu(g) * u`` in bf16), down with cuBLAS and the residual added in
  place into its fresh output (``x + y`` rounds the same way).

So 2 Triton launches + 2 GEMMs + 1 add per call instead of ~14 eager ops. With
``fast_launch=False`` the same kernels go through the JIT launcher: compare both with
``sweep_candidate`` (``fast_launch``: true / false) to see what the launch path is worth
on your shapes. Under ``torch.compile`` the plain ``kernel[grid](...)`` is used (Dynamo
traces user Triton kernels; host time does not count in a compiled or CUDA-graph run).

Generic over hidden / intermediate sizes (any, masked), dtypes (fp16 / bf16) and leading
dimensions; another block structure gets the reference back (unsupported instance).
Contract: ``build(reference) -> nn.Module`` returning a drop-in replacement.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from kernel_agent.kernels.triton_launch import CachedLaunch

SILU_BLOCK = 1024


@triton.jit
def _rmsnorm_kernel(x, w, y, stride_x, stride_y, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    v = tl.load(x + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(v * v, axis=0) / n_cols
    h = (v * tl.rsqrt(var + eps)).to(x.dtype.element_ty)  # the reference's cast to x's dtype
    wt = tl.load(w + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y + row * stride_y + cols, (wt * h.to(tl.float32)).to(y.dtype.element_ty), mask=mask)


@triton.jit
def _silu_mul_kernel(gu, out, n, stride_gu, stride_out, BLOCK: tl.constexpr):
    # gu: [M, 2n] (gate | up), out: [M, n] = silu(gate) * up with the reference's roundings
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n
    g = tl.load(gu + row * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu + row * stride_gu + n + cols, mask=mask, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(out.dtype.element_ty)
    y = (s.to(tl.float32) * u).to(out.dtype.element_ty)
    tl.store(out + row * stride_out + cols, y, mask=mask)


_rmsnorm = CachedLaunch(_rmsnorm_kernel)
_silu_mul = CachedLaunch(_silu_mul_kernel)


class CheapLaunchMlpBlock(nn.Module):
    def __init__(self, reference: nn.Module, fast_launch: bool) -> None:
        super().__init__()
        norm = reference.norm
        self.norm_weight = norm.weight
        self.eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
        gate, up = reference.gate_proj.weight, reference.up_proj.weight
        self.gate_up = nn.Parameter(torch.cat([gate, up]).detach(), requires_grad=False)
        self.down = reference.down_proj.weight
        self.inter = gate.shape[0]
        self.fast_launch = fast_launch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.shape[-1]
        x2 = x.reshape(-1, hidden)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        rows = x2.shape[0]
        if rows == 0:
            return x.clone()
        h = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
        act = torch.empty((rows, self.inter), device=x.device, dtype=x.dtype)
        # compiled: the JIT path, which Dynamo traces; eager: the cached launch
        cached = self.fast_launch and not torch.compiler.is_compiling()
        norm = _rmsnorm if cached else _rmsnorm_kernel
        silu_mul = _silu_mul if cached else _silu_mul_kernel
        block = triton.next_power_of_2(hidden)
        norm[(rows,)](
            x2,
            self.norm_weight,
            h,
            x2.stride(0),
            h.stride(0),
            hidden,
            self.eps,
            BLOCK=block,
            num_warps=4 if block <= 2048 else 8,
        )
        gu = F.linear(h, self.gate_up)  # [rows, 2 * inter], one cuBLAS GEMM
        silu_mul[(rows, triton.cdiv(self.inter, SILU_BLOCK))](
            gu, act, self.inter, gu.stride(0), act.stride(0), BLOCK=SILU_BLOCK, num_warps=4
        )
        y = F.linear(act, self.down)
        return y.add_(x2).reshape(x.shape)  # in place into the fresh output


def build(reference: nn.Module, fast_launch: bool = True) -> nn.Module:
    """``fast_launch``: launch through ``CachedLaunch`` (False: the JIT launcher, to
    compare with ``sweep_candidate``)."""
    linears = [getattr(reference, n, None) for n in ("gate_proj", "up_proj", "down_proj")]
    norm = getattr(reference, "norm", None)
    ok = (
        all(isinstance(m, nn.Linear) and m.bias is None for m in linears)
        and norm is not None
        and getattr(norm, "weight", None) is not None
        and norm.weight.dim() == 1
        and norm.weight.is_cuda
        and norm.weight.dtype in (torch.float16, torch.bfloat16)
        and all(m.weight.dtype == norm.weight.dtype for m in linears)
    )
    if not ok:
        return reference  # another structure: keep the original
    return CheapLaunchMlpBlock(reference, fast_launch)
