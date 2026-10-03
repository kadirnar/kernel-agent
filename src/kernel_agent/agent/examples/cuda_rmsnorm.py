"""Example candidate (CUDA C++ via torch.utils.cpp_extension.load_inline).

Warp-per-row RMSNorm with vectorised loads.  ``load_inline`` caches the build
under ~/.cache/torch_extensions, so only the first evaluation pays for nvcc.
Give every new source version a new ``name`` (hash suffix) to avoid stale caches.
"""

import hashlib

import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f(float v) { return v; }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_f(float v) { return __float2half(v); }
template <> __device__ __forceinline__ float from_f(float v) { return v; }

template <typename T>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ y,
                               int rows, int cols, float eps) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if (warp >= rows) return;
  const T* xr = x + (size_t)warp * cols;
  T* yr = y + (size_t)warp * cols;
  float acc = 0.f;
  for (int c = lane; c < cols; c += 32) { float v = to_f(xr[c]); acc += v * v; }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
  float inv = rsqrtf(acc / cols + eps);
  for (int c = lane; c < cols; c += 32) {
    T normed = from_f<T>(to_f(xr[c]) * inv);
    yr[c] = from_f<T>(to_f(normed) * to_f(w[c]));
  }
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
  auto x2 = x.contiguous();
  auto y = torch::empty_like(x2);
  int cols = x2.size(-1);
  int rows = x2.numel() / cols;
  int threads = 256;
  int blocks = (rows * 32 + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto tag) {
    using T = decltype(tag);
    rmsnorm_kernel<T><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const T*>(x2.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
        reinterpret_cast<T*>(y.data_ptr()), rows, cols, (float)eps);
  };
  switch (x2.scalar_type()) {
    case at::kBFloat16: launch(__nv_bfloat16{}); break;
    case at::kHalf: launch(__half{}); break;
    case at::kFloat: launch(float{}); break;
    default: TORCH_CHECK(false, "unsupported dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""
CPP_SRC = "torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps);"

_ext = None


def _load():
    global _ext
    if _ext is None:
        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:8]
        _ext = load_inline(
            name=f"ka_rmsnorm_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["rmsnorm"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _ext


class CudaRMSNorm(nn.Module):
    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        self.weight = reference.weight
        self.eps = float(getattr(reference, "variance_epsilon", getattr(reference, "eps", 1e-6)))
        self.ext = _load()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ext.rmsnorm(hidden_states, self.weight, self.eps)


def build(reference: nn.Module) -> nn.Module:
    return CudaRMSNorm(reference)
