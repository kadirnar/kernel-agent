"""Example candidate (TileLang): RMSNorm with a row tile per CTA.

TileLang kernels are Python functions that build a ``T.prim_func``;
``@tilelang.jit(out_idx=[-1])`` allocates the last argument as the output and
returns a torch-callable kernel.  Compile once per (rows, cols, dtype).
"""

import tilelang
import tilelang.language as T
import torch
from torch import nn

_DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


@tilelang.jit(out_idx=[-1])
def _rmsnorm(M, N, eps, dtype, block_m=1):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), W: T.Tensor((N,), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_m), threads=128) as bx:
            xs = T.alloc_fragment((block_m, N), "float32")
            sq = T.alloc_fragment((block_m, N), "float32")
            ssum = T.alloc_fragment((block_m,), "float32")
            T.copy(X[bx * block_m, 0], xs)
            for i, j in T.Parallel(block_m, N):
                sq[i, j] = xs[i, j] * xs[i, j]
            T.reduce_sum(sq, ssum, dim=1)
            for i in T.Parallel(block_m):
                ssum[i] = T.rsqrt(ssum[i] / N + eps)
            for i, j in T.Parallel(block_m, N):
                Y[bx * block_m + i, j] = T.Cast(dtype, T.Cast(dtype, xs[i, j] * ssum[i]) * W[j])

    return main


class TileLangRMSNorm(nn.Module):
    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.weight = reference.weight
        self.eps = float(getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-6)))
        self._kernels: dict[tuple, object] = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1]).contiguous()
        key = (x.shape[0], x.shape[1], x.dtype)
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = _rmsnorm(x.shape[0], x.shape[1], self.eps, _DTYPES[x.dtype])
            self._kernels[key] = kernel
        return kernel(x, self.weight).view(shape)


def build(reference: nn.Module) -> nn.Module:
    if reference.weight.shape[0] > 8192:
        return reference
    return TileLangRMSNorm(reference)
