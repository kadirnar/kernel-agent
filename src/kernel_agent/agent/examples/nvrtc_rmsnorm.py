"""Example candidate (raw CUDA C++ compiled at runtime with NVRTC via ``cuda.core``).

No nvcc / no C++ extension build: the kernel source is compiled to a cubin in
milliseconds and launched on torch's current stream with raw pointers.
"""

import numpy as np
import torch
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
from torch import nn

from kernel_agent.toolchain import cuda_include_dirs

SRC = r"""
#include <cuda_bf16.h>
extern "C" __global__ void rmsnorm_bf16(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ y, int cols, float eps) {
  extern __shared__ float red[];
  const __nv_bfloat16* xr = x + (size_t)blockIdx.x * cols;
  __nv_bfloat16* yr = y + (size_t)blockIdx.x * cols;
  float acc = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = __bfloat162float(xr[c]);
    acc += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {
    acc = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (threadIdx.x == 0) red[0] = rsqrtf(acc / cols + eps);
  }
  __syncthreads();
  float inv = red[0];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    __nv_bfloat16 n = __float2bfloat16(__bfloat162float(xr[c]) * inv);
    yr[c] = __float2bfloat16(__bfloat162float(n) * __bfloat162float(w[c]));
  }
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        dev = Device(torch.cuda.current_device())
        dev.set_current()
        opts = ProgramOptions(arch=f"sm_{dev.arch}", std="c++17", include_path=cuda_include_dirs())
        _kernel = (
            Program(SRC, code_type="c++", options=opts).compile("cubin").get_kernel("rmsnorm_bf16")
        )
    return _kernel


class NvrtcRMSNorm(nn.Module):
    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.weight = reference.weight
        self.eps = float(getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-6)))
        self.kernel = _get_kernel()
        self.device = Device(torch.cuda.current_device())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.contiguous()
        y = torch.empty_like(x)
        cols = x.shape[-1]
        rows = x.numel() // cols
        threads = 256
        stream = self.device.create_stream(torch.cuda.current_stream())
        cfg = LaunchConfig(grid=rows, block=threads, shmem_size=(threads // 32) * 4)
        launch(
            stream,
            cfg,
            self.kernel,
            x.data_ptr(),
            self.weight.data_ptr(),
            y.data_ptr(),
            np.int32(cols),
            np.float32(self.eps),
        )
        return y


def build(reference: nn.Module) -> nn.Module:
    if reference.weight.dtype != torch.bfloat16:
        return reference  # this example only handles bf16
    return NvrtcRMSNorm(reference)
