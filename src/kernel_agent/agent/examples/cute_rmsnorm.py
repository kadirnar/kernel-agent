"""Example candidate (CuTe DSL, ``nvidia-cutlass-dsl``): RMSNorm, one CTA per row.

Pattern: ``@cute.kernel`` device code + ``@cute.jit`` host launcher, compiled
once with ``cute.compile(..., options="--enable-tvm-ffi")``.  With TVM-FFI the
compiled function takes torch tensors directly and launches on torch's current
stream (``make_fake_stream(use_tvm_ffi_env_stream=True)``), which cuts the
per-call host overhead ~3x versus calling ``from_dlpack`` on every forward.
``mark_layout_dynamic`` lets one compilation serve every row count.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.runtime import make_fake_stream
from torch import nn

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_80+"
ARCHS_WHY = "CuTe DSL targets sm_80+ (nvidia-cutlass-dsl 4.8 has no sm_75 target)"

THREADS = 256


@cute.kernel
def _rmsnorm_kernel(gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor, eps: cutlass.Float32):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    cols = gX.shape[1]
    smem = cutlass.utils.SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(THREADS // 32))

    acc = cutlass.Float32(0.0)
    for c in range(tidx, cols, THREADS):
        v = gX[row, c].to(cutlass.Float32)
        acc += v * v
    acc = cute.arch.warp_reduction_sum(acc)
    if tidx % 32 == 0:
        red[tidx // 32] = acc
    cute.arch.barrier()
    total = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(THREADS // 32):
        total += red[i]
    inv = cute.math.rsqrt(total / cols + eps)
    for c in range(tidx, cols, THREADS):
        n = (gX[row, c].to(cutlass.Float32) * inv).to(gX.element_type)
        gY[row, c] = (n.to(cutlass.Float32) * gW[c].to(cutlass.Float32)).to(gY.element_type)


@cute.jit
def _rmsnorm(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, eps: cutlass.Float32, stream):
    _rmsnorm_kernel(mX, mW, mY, eps).launch(
        grid=(mX.shape[0], 1, 1), block=(THREADS, 1, 1), stream=stream
    )


class CuteRMSNorm(nn.Module):
    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.weight = reference.weight
        self.eps = float(getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-6)))
        self._compiled: dict[tuple, object] = {}
        self.w = reference.weight.detach()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1]).contiguous()
        y = torch.empty_like(x)
        key = (x.shape[1], x.dtype)
        fn = self._compiled.get(key)
        if fn is None:
            mx = from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            my = from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            mw = from_dlpack(self.weight.detach(), assumed_align=16)
            stream = make_fake_stream(use_tvm_ffi_env_stream=True)
            fn = cute.compile(
                _rmsnorm, mx, mw, my, cutlass.Float32(self.eps), stream, options="--enable-tvm-ffi"
            )
            self._compiled[key] = fn
        fn(x, self.w, y, self.eps)
        return y.view(shape)


def build(reference: nn.Module) -> nn.Module:
    return CuteRMSNorm(reference)
