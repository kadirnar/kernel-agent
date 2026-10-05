"""Example candidate (Triton) that stacks with ``torch.compile`` and CUDA graphs.

The kernel launcher is a PyTorch custom operator: ``torch.library.custom_op``
makes it one op that Dynamo, Inductor and CUDA graphs treat like an ATen op,
and ``register_fake`` tells the compiler the output's shape / dtype / device
without running the kernel.  Without the wrapper, a launcher Dynamo cannot
trace (a ``load_inline`` / pybind module, NVRTC or ``cuda.core`` handles, a CuTe
``cute.compile`` function, host-side logic on tensor values) is a graph break,
and an error under ``fullgraph=True`` (VoxCPM's ``model.optimize()``, the
compiled baseline of many runs).  The wrapper is the same for every backend:
only the body of ``rmsnorm`` changes.

Rules: the op takes tensors and plain values (int / float / bool / str / lists),
never modules; it returns fresh tensors (declare in-place writes with
``mutates_args``); the fake function only allocates.  Check it with
``kernel-agent eval CAPTURE CANDIDATE --compile-check`` or
``evaluate_candidate(..., compile_check=true)``: graph breaks + compiled outputs.

Contract: ``build(reference) -> nn.Module`` returning a drop-in replacement.
"""

import re

import torch
import triton
import triton.language as tl
from torch import nn

# One namespace per candidate file (the evaluator names the module after the file's
# hash), so two candidates copied from this example never replace each other's op.
_NS = re.sub(r"\W", "_", __name__)


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_row, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # Match the reference: normalise in fp32, cast to input dtype, then scale.
    y = y.to(w.dtype) * w
    tl.store(y_ptr + row * stride_row + cols, y, mask=mask)


@torch.library.custom_op(f"{_NS}::rmsnorm", mutates_args=())
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    n_cols = x.shape[-1]
    x2 = x.reshape(-1, n_cols).contiguous()
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)  # fresh, never a view of x
    block = triton.next_power_of_2(n_cols)
    _rmsnorm_kernel[(x2.shape[0],)](
        x2, weight, y, x2.stride(0), n_cols, eps, BLOCK=block, num_warps=4 if block <= 2048 else 8
    )
    return y


@rmsnorm.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)  # same as the real op


class TritonRMSNorm(nn.Module):
    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.weight = reference.weight
        self.eps = float(getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-6)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rmsnorm(hidden_states, self.weight, self.eps)


def build(reference: nn.Module) -> nn.Module:
    if reference.weight.dim() != 1:
        return reference  # unsupported instance: keep the original
    return TritonRMSNorm(reference)
