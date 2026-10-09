"""Example candidate (CUDA C++ via load_inline): FP8 e4m3 weight-only GEMV for ``nn.Linear``.

Reduced precision: only for a target whose spec says ``"precision": "fp8_weights"`` (a
``--quality near-lossless`` run, skill fp8-weights). The evaluator checks it in
the near-lossless tier; the exact tier rejects it (FP8 moves every output by ~2.5 %).

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_fp8``: e4m3,
  one fp32 scale per output channel) and keeps no bf16 copy: half the bytes to stream,
  half the memory. ``quant_error`` holds the error report of the stored weight.
* Decode (M <= 4 rows): one warp per output row, 128-bit loads of 16 weights streamed past
  L1, dequantised in registers (``cvt`` e4m3x2 -> f16x2 -> fp32), activations in the
  model's dtype (bf16 or fp16: the kernel is a template on it, :data:`DTYPES`), fp32
  accumulation, warp-shuffle reduction; the per-channel scale (and the bias) are applied
  once per output in the epilogue, one rounding to the activations' dtype: ``y = scale[n] *
  sum_k x[k] q[n, k] + bias[n]``.
* Other shapes and dtypes: the dequantised weight through cuBLAS (the same FP8 math; slow,
  a fallback).
* The forward is one pybind call (shape logic and fallback in C++): at decode sizes the host
  overhead is a large part of the latency.

``unroll`` (128-bit weight loads in flight per lane) is a ``build()`` keyword for
``sweep_candidate``. RTX 5070 Ti, weights streamed from DRAM: [1, 2048] x [2048, 6144] in
16.2 us (777 GB/s, 2.0x cuBLAS bf16). At M = 2..4 the per-row FMA work makes it ALU bound
when the weight sits in L2; ``cuda_fp8_skinny_gemm.py`` (tensor cores) is faster there.
Older GPUs (sm_75 / sm_86): bf16 and fp16 outputs verified through their PTX (``compute_75``,
``compute_86``) JIT-compiled on an RTX 5070 Ti; not timed on such a GPU yet.
"""

import hashlib

import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

from kernel_agent.kernels.quant import fp8_error, quantize_fp8

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_75+"
ARCHS_WHY = (
    "CUDA cores only (bf16 activations through cuda_bf16's software math before sm_80); e4m3 "
    "weights converted in registers (hardware cvt from sm_89, CUDA's software conversion "
    "before)"
)
#: Activation dtypes the kernel takes (a template on the type; the output keeps it).
DTYPES = (torch.bfloat16, torch.float16)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

typedef __nv_bfloat16 bf16;

// The activation type T: bf16 or __half (the model's dtype; the output keeps it)
template <typename T> struct Act;
template <> struct Act<bf16> {
  typedef __nv_bfloat162 T2;
  static __device__ __forceinline__ float2 f2(T2 v) { return __bfloat1622float2(v); }
  static __device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ bf16 from(float v) { return __float2bfloat16(v); }
};
template <> struct Act<__half> {
  typedef __half2 T2;
  static __device__ __forceinline__ float2 f2(T2 v) { return __half22float2(v); }
  static __device__ __forceinline__ float f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from(float v) { return __float2half(v); }
};

// 128-bit load that bypasses L1: every weight byte is read once
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// 16 e4m3 weights (one uint4) -> fp32 in registers (cvt e4m3x2 -> f16x2 is exact)
__device__ __forceinline__ void dequant16(const uint4& w, float* f) {
  const __nv_fp8x2_storage_t* w2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(w2[i], __NV_E4M3);
    float2 v = __half22float2(*reinterpret_cast<const __half2*>(&h));
    f[2 * i] = v.x;
    f[2 * i + 1] = v.y;
  }
}

// 16 fp32 weights . 16 activations of type T (two uint4), fp32 accumulation
template <typename T>
__device__ __forceinline__ float dot16(const float* f, const uint4& xa, const uint4& xb) {
  typedef typename Act<T>::T2 T2;
  const T2* x0 = reinterpret_cast<const T2*>(&xa);
  const T2* x1 = reinterpret_cast<const T2*>(&xb);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 xf = Act<T>::f2(i < 4 ? x0[i] : x1[i - 4]);
    s = fmaf(f[2 * i], xf.x, s);
    s = fmaf(f[2 * i + 1], xf.y, s);
  }
  return s;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// x: [M, K] T, w: [N, K] e4m3 codes, scale: [N] fp32, bias: [N] T or null, y: [M, N] T.
// One warp per output row, MT activation rows at a time, U loads in flight.
template <typename T, int MT, int U>
__global__ void __launch_bounds__(256) fp8_gemv_kernel(
    const T* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ scale,
    const T* __restrict__ bias, T* __restrict__ y, int M, int K, int N) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const uint8_t* wr = w + (size_t)row * K;
  const int chunks = K >> 4;  // 16 weights per chunk
  for (int m0 = 0; m0 < M; m0 += MT) {
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;
    for (int c = lane; c < chunks; c += 32 * U) {
      uint4 wv[U];
#pragma unroll
      for (int i = 0; i < U; ++i)
        wv[i] = c + 32 * i < chunks ? ld_stream(wr + 16 * (size_t)(c + 32 * i))
                                    : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int i = 0; i < U; ++i) {
        if (c + 32 * i >= chunks) break;
        float f[16];
        dequant16(wv[i], f);  // once per chunk, reused by the MT activation rows
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)min(m0 + m, M - 1) * K) +
                            2 * (c + 32 * i);
          acc[m] += dot16<T>(f, __ldg(xr), __ldg(xr + 1));
        }
      }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
      const float s = scale[row];
      const float b = bias ? Act<T>::f(bias[row]) : 0.f;
#pragma unroll
      for (int m = 0; m < MT; ++m)
        if (m0 + m < M) y[(size_t)(m0 + m) * N + row] = Act<T>::from(fmaf(acc[m], s, b));
    }
  }
}

template <typename T, int MT>
void launch(const T* x, const uint8_t* w, const float* s, const T* b, T* y, int M, int K, int N,
            int unroll, cudaStream_t stream) {
  const dim3 grid((N + 7) / 8), block(256);
  switch (unroll) {
    case 1: fp8_gemv_kernel<T, MT, 1><<<grid, block, 0, stream>>>(x, w, s, b, y, M, K, N); break;
    case 2: fp8_gemv_kernel<T, MT, 2><<<grid, block, 0, stream>>>(x, w, s, b, y, M, K, N); break;
    case 8: fp8_gemv_kernel<T, MT, 8><<<grid, block, 0, stream>>>(x, w, s, b, y, M, K, N); break;
    default: fp8_gemv_kernel<T, MT, 4><<<grid, block, 0, stream>>>(x, w, s, b, y, M, K, N); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void run(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& scale,
         const torch::Tensor& bias, torch::Tensor& y, int M, int K, int N, int unroll) {
  auto px = reinterpret_cast<const T*>(x.data_ptr());
  auto pw = reinterpret_cast<const uint8_t*>(w.data_ptr());
  auto ps = scale.data_ptr<float>();
  auto pb = bias.numel() ? reinterpret_cast<const T*>(bias.data_ptr()) : nullptr;
  auto py = reinterpret_cast<T*>(y.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) launch<T, 1>(px, pw, ps, pb, py, M, K, N, unroll, stream);
  else launch<T, 4>(px, pw, ps, pb, py, M, K, N, unroll, stream);
}

// y = x @ (q * scale)^T + bias for x [..., K]. One pybind call per forward: shape logic and
// the fallback live here, not in Python (host overhead is most of a decode GEMV's latency).
torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor bias,
                         int64_t unroll, int64_t max_rows) {
  const int64_t N = w.size(0), K = w.size(1);
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == K && x.device() == w.device(), "fp8_linear: bad input");
  const int64_t M = x.numel() / K;
  const auto dt = x.scalar_type();
  if (M < 1 || M > max_rows || (dt != at::kBFloat16 && dt != at::kHalf) ||
      (bias.numel() && bias.scalar_type() != dt)) {
    // other shapes and dtypes: the same FP8 math through cuBLAS (dequantised weight; a fallback)
    auto wd = (w.to(at::kFloat) * scale.unsqueeze(1)).to(dt);
    return bias.numel() ? at::linear(x, wd, bias.to(dt)) : at::linear(x, wd);
  }
  if (!x.is_contiguous() || (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15))
    x = x.clone(at::MemoryFormat::Contiguous);  // 128-bit loads need 16-byte alignment
  auto sizes = x.sizes().vec();
  sizes.back() = N;
  auto y = torch::empty(sizes, x.options());
  if (dt == at::kHalf) run<__half>(x, w, scale, bias, y, (int)M, (int)K, (int)N, (int)unroll);
  else run<bf16>(x, w, scale, bias, y, (int)M, (int)K, (int)N, (int)unroll);
  return y;
}
"""
CPP_SRC = (
    "torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor w, torch::Tensor scale, "
    "torch::Tensor bias, int64_t unroll, int64_t max_rows);"
)

MAX_ROWS = 4  # decode GEMV; larger M takes the dequantised fallback (a skinny GEMM kernel)

_ext = None


def _load():
    global _ext
    if _ext is None:
        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:8]
        _ext = load_inline(
            name=f"ka_fp8_gemv_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["fp8_linear"],
            extra_cuda_cflags=["-O3"],
        )
    return _ext


class Fp8Linear(nn.Module):
    """``nn.Linear`` with e4m3 weights and one fp32 scale per output channel."""

    def __init__(self, reference: nn.Linear, unroll: int = 4) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        q, scale = quantize_fp8(reference.weight)  # once, here: never per call
        self.register_buffer("weight_fp8", q)
        self.register_buffer("weight_scale", scale)
        self.register_parameter("bias", reference.bias)
        self.quant_error = fp8_error(reference.weight, q, scale)  # for NOTES.md
        none = torch.empty(0, dtype=reference.weight.dtype, device=q.device)
        bias = reference.bias if reference.bias is not None else none
        # plain attributes: the forward is one pybind call without nn.Module lookups
        self._args = (q, scale, bias, int(unroll), MAX_ROWS)
        self._fn = _load().fp8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x, *self._args)


def build(reference: nn.Module, unroll: int = 4) -> nn.Module:
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype in DTYPES
        and reference.weight.is_cuda
        and reference.in_features % 16 == 0  # 16 weights per 128-bit load
        and (reference.bias is None or reference.bias.dtype == reference.weight.dtype)
    )
    return Fp8Linear(reference, unroll) if ok else reference
