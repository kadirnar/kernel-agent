"""Example candidate (Triton): fused RMSNorm, one program per row.

Contract: ``build(reference) -> nn.Module`` returning a drop-in replacement.
"""

import torch
import triton
import triton.language as tl
from torch import nn


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


class TritonRMSNorm(nn.Module):
    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.weight = reference.weight
        self.eps = float(getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-6)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1]).contiguous()
        y = torch.empty_like(x)
        n_cols = x.shape[-1]
        block = triton.next_power_of_2(n_cols)
        num_warps = 4 if block <= 2048 else 8
        _rmsnorm_kernel[(x.shape[0],)](
            x, self.weight, y, x.stride(0), n_cols, self.eps, BLOCK=block, num_warps=num_warps
        )
        return y.view(shape)


def build(reference: nn.Module) -> nn.Module:
    if reference.weight.dim() != 1:
        return reference  # unsupported instance: keep the original
    return TritonRMSNorm(reference)
