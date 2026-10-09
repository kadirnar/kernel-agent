"""Example candidate (CUDA C++ via load_inline): INT8 weight-only GEMV for ``nn.Linear``
(decode, M <= 4 rows): the weight streams at one byte per element, activations stay bf16.

Reduced precision: only for a target whose spec says ``"precision": "int8_weights"`` (a
``--quality near-lossless`` or ``relaxed`` run, skill int8-weights). The
evaluator checks it in the near-lossless tier; the exact tier rejects it.

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_int8``:
  symmetric int8 codes in [-127, 127], one fp32 scale ``amax / 127`` per output channel)
  and keeps no bf16 copy: half the bytes to stream, half the memory. ``quant_error`` holds
  the error report of the stored weight.
* Decode (M <= 4 rows): one warp per output row, 128-bit loads of 16 weights streamed past
  L1, converted in registers with two ops per weight (``prmt`` builds the fp32 ``2^23 +
  (code + 128)``, one subtraction of ``2^23 + 128`` gives the code exactly: no ``I2F``, and
  no e4m3 conversion, which before sm_89 runs in software), bf16 activations,
  fp32 accumulation, warp-shuffle reduction; the per-channel scale (and the bias) once per
  output in the epilogue: ``y = scale[n] * sum_k x[k] q[n, k] + bias[n]``.
* Other shapes: ``quant.int8_weights_linear`` (the same math; a fallback).
* The forward is one pybind call (shape logic in C++): at decode sizes the host overhead is
  a large part of the latency.

``unroll`` (128-bit weight loads in flight per lane) is a ``build()`` keyword for
``sweep_candidate``. Where it pays: every GPU from sm_80 (on Ampere the way to halve a
decode GEMV's bytes without FP8 hardware); int8 per channel is also about 2.5x more accurate
than e4m3 per channel on Gaussian-like weights (relative L2 ~0.01 vs 0.026), less on rows
with outliers (``quant_error["crest"]``).
"""

import hashlib

import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

from kernel_agent.kernels.quant import int8_error, int8_weights_linear, quantize_int8

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_75+"
ARCHS_WHY = (
    "bf16 activations (cuda_bf16 conversions, software bf16 math before sm_80); int8 codes "
    "converted with prmt + FADD"
)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

typedef __nv_bfloat16 bf16;

// 128-bit load that bypasses L1: every weight byte is read once
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// 16 int8 codes (one uint4) -> fp32, exact: byte b ^ 0x80 = b + 128 placed in the mantissa of
// 2^23 (prmt), minus 2^23 + 128
__device__ __forceinline__ void dequant16(const uint4& w, float* f) {
  const unsigned* w4 = reinterpret_cast<const unsigned*>(&w);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned u = w4[i] ^ 0x80808080u;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      f[4 * i + j] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7440u | j)) - 8388736.0f;
  }
}

// 16 fp32 weights . 16 bf16 activations (two uint4), fp32 accumulation
__device__ __forceinline__ float dot16(const float* f, const uint4& xa, const uint4& xb) {
  const __nv_bfloat162* x0 = reinterpret_cast<const __nv_bfloat162*>(&xa);
  const __nv_bfloat162* x1 = reinterpret_cast<const __nv_bfloat162*>(&xb);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 xf = __bfloat1622float2(i < 4 ? x0[i] : x1[i - 4]);
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

// x: [M, K] bf16, w: [N, K] int8 codes, scale: [N] fp32, bias: [N] bf16 or null,
// y: [M, N] bf16. One warp per output row, MT activation rows at a time, U loads in flight.
template <int MT, int U>
__global__ void __launch_bounds__(256) int8_gemv_kernel(
    const bf16* __restrict__ x, const int8_t* __restrict__ w, const float* __restrict__ scale,
    const bf16* __restrict__ bias, bf16* __restrict__ y, int M, int K, int N) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const int8_t* wr = w + (size_t)row * K;
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
          acc[m] += dot16(f, __ldg(xr), __ldg(xr + 1));
        }
      }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
      const float s = scale[row];
      const float b = bias ? __bfloat162float(bias[row]) : 0.f;
#pragma unroll
      for (int m = 0; m < MT; ++m)
        if (m0 + m < M) y[(size_t)(m0 + m) * N + row] = __float2bfloat16(fmaf(acc[m], s, b));
    }
  }
}

template <int MT>
void launch(const bf16* x, const int8_t* w, const float* s, const bf16* b, bf16* y, int M, int K,
            int N, int unroll, cudaStream_t stream) {
  const dim3 grid((N + 7) / 8), block(256);
  switch (unroll) {
    case 1: int8_gemv_kernel<MT, 1><<<grid, block, 0, stream>>>(x, w, s, b, y, M, K, N); break;
    case 2: int8_gemv_kernel<MT, 2><<<grid, block, 0, stream>>>(x, w, s, b, y, M, K, N); break;
    case 8: int8_gemv_kernel<MT, 8><<<grid, block, 0, stream>>>(x, w, s, b, y, M, K, N); break;
    default: int8_gemv_kernel<MT, 4><<<grid, block, 0, stream>>>(x, w, s, b, y, M, K, N); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y = x @ (q * scale)^T + bias for x [..., K]: one pybind call per forward. Other shapes: the
// same math through cuBLAS (dequantised weight; a fallback).
torch::Tensor int8_linear(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor bias,
                          int64_t unroll, int64_t max_rows) {
  const int64_t N = w.size(0), K = w.size(1);
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == K && x.device() == w.device(),
              "int8_linear: bad input");
  const int64_t M = x.numel() / K;
  if (M < 1 || M > max_rows || x.scalar_type() != at::kBFloat16) {
    auto wd = (w.to(at::kFloat) * scale.unsqueeze(1)).to(x.scalar_type());
    return bias.numel() ? at::linear(x, wd, bias.to(x.scalar_type())) : at::linear(x, wd);
  }
  if (!x.is_contiguous() || (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15))
    x = x.clone(at::MemoryFormat::Contiguous);  // 128-bit loads need 16-byte alignment
  auto sizes = x.sizes().vec();
  sizes.back() = N;
  auto y = torch::empty(sizes, x.options());
  auto px = reinterpret_cast<const bf16*>(x.data_ptr());
  auto pw = w.data_ptr<int8_t>();
  auto ps = scale.data_ptr<float>();
  auto pb = bias.numel() ? reinterpret_cast<const bf16*>(bias.data_ptr()) : nullptr;
  auto py = reinterpret_cast<bf16*>(y.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) launch<1>(px, pw, ps, pb, py, M, K, N, (int)unroll, stream);
  else launch<4>(px, pw, ps, pb, py, M, K, N, (int)unroll, stream);
  return y;
}
"""
CPP_SRC = (
    "torch::Tensor int8_linear(torch::Tensor x, torch::Tensor w, torch::Tensor scale, "
    "torch::Tensor bias, int64_t unroll, int64_t max_rows);"
)

MAX_ROWS = 4  # decode GEMV; larger M takes the dequantised fallback

_ext = None


def _load():
    global _ext
    if _ext is None:
        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:8]
        _ext = load_inline(
            name=f"ka_int8_gemv_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["int8_linear"],
            extra_cuda_cflags=["-O3"],
        )
    return _ext


class Int8Linear(nn.Module):
    """``nn.Linear`` with int8 weights and one fp32 scale per output channel (weight-only)."""

    def __init__(self, reference: nn.Linear, unroll: int = 4) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        q, scale = quantize_int8(reference.weight)  # once, here: never per call
        self.register_buffer("weight_int8", q)
        self.register_buffer("weight_scale", scale)
        self.register_parameter("bias", reference.bias)
        self.quant_error = int8_error(reference.weight, q, scale)  # for NOTES.md
        none = torch.empty(0, dtype=torch.bfloat16, device=q.device)
        bias = reference.bias if reference.bias is not None else none
        # plain attributes: the forward is one pybind call without nn.Module lookups
        self._args = (q, scale, bias, int(unroll), MAX_ROWS)
        self._ref = (q, scale, reference.bias)
        self._fn = _load().int8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return int8_weights_linear(x, *self._ref)
        return self._fn(x, *self._args)


def build(reference: nn.Module, unroll: int = 4) -> nn.Module:
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype == torch.bfloat16
        and reference.weight.is_cuda
        and reference.in_features % 16 == 0  # 16 weights per 128-bit load
        and (reference.bias is None or reference.bias.dtype == torch.bfloat16)
    )
    return Int8Linear(reference, unroll) if ok else reference
