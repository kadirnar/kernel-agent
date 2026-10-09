"""Example candidate (CUDA C++ via load_inline): NVFP4 weight-only GEMV for ``nn.Linear``.

Reduced precision: only for a target whose spec says ``"precision": "fp4_weights"`` (a
``--quality near-lossless`` run, skill fp4-weights). The evaluator checks it in
the near-lossless-fp4 tier; the exact and the FP8 near-lossless tier reject it (FP4 moves
every output by ~10 %).

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_fp4``: e2m1
  codes, two per byte, one e4m3 scale per 16 weights of a row and one fp32 scale per
  tensor) and keeps no bf16 copy: a quarter of the bytes to stream. ``quant_error`` holds
  the error report of the stored weight.
* Decode (M <= 4 rows): one warp per output row; each lane streams 16 bytes of codes (32
  weights, two blocks) past L1 and the two block scales, dequantises in registers (a
  ``prmt`` and a few bit operations put each code's sign, exponent and mantissa bits into
  an f16, ``x 2^14`` folded into the block scale), bf16 activations, fp32 accumulation:
  ``y = tensor_scale * sum_b scale[n, b] * sum_k x[k] e2m1[n, k] + bias[n]``.
* Other shapes: the dequantised weight through cuBLAS (the same FP4 math; slow, a fallback).
* The forward is one pybind call (shape logic and fallback in C++).

``unroll`` (16-byte code loads in flight per lane) is a ``build()`` keyword for
``sweep_candidate``. RTX 5070 Ti, weights streamed from DRAM: [1, 2048] x [2048, 6144] in
9.0 us (784 GB/s of codes + scales; FP8 GEMV 16.2 us, cuBLAS bf16 32.3 us). At M = 2..4
it is ALU bound (19.9 us at M = 4, slower than the FP8 GEMV): use tensor cores there.
"""

import hashlib

import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

from kernel_agent.kernels.quant import E2M1_VALUES, fp4_error, quantize_fp4

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_75+"
ARCHS_WHY = (
    "bf16 activations (cuda_bf16's software bf16 math before sm_80); e2m1 codes decoded with "
    "integer ops, e4m3 block scales converted in registers (software conversion before sm_89)"
)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

// Codes 2j and 2j+1 of a word (byte j: low and high nibble) -> two fp32 values / 2^14.
// The nibble's sign bit goes to the f16 sign (bit 15), its exponent and mantissa bits to
// f16 bits 11..9: the f16 is then e2m1 * 2^-14 exactly (code 1, 0.5, becomes a subnormal).
__device__ __forceinline__ float2 e2m1x2(uint32_t w, uint32_t w_hi, uint32_t sel) {
  const uint32_t t = __byte_perm(w, w_hi, sel);  // byte 0: code 2j, byte 2: code 2j+1
  const uint32_t h = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// 32 codes (one uint4) -> fp32 values / 2^14 in registers
__device__ __forceinline__ void dequant32(const uint4& wv, float* f) {
  const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t hi = ws[i] >> 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 v = e2m1x2(ws[i], hi, j | (j << 4) | ((4 + j) << 8) | ((4 + j) << 12));
      f[8 * i + 2 * j] = v.x;
      f[8 * i + 2 * j + 1] = v.y;
    }
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

// x: [M, K] bf16, w: [N, K / 2] packed e2m1 codes, s: [N, K / 16] e4m3 block scales,
// y: [M, N] bf16. One warp per output row, MT activation rows at a time, U loads in flight.
template <int MT, int U>
__global__ void __launch_bounds__(256) fp4_gemv_kernel(
    const bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
    float tensor_scale, const bf16* __restrict__ bias, bf16* __restrict__ y, int M, int K,
    int N) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const uint8_t* wr = w + (size_t)row * (K / 2);
  const __nv_fp8x2_storage_t* sr =
      reinterpret_cast<const __nv_fp8x2_storage_t*>(s + (size_t)row * (K / 16));
  const int chunks = K >> 5;  // 32 weights (16 bytes of codes, two blocks) per chunk
  for (int m0 = 0; m0 < M; m0 += MT) {
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.f;
    for (int c = lane; c < chunks; c += 32 * U) {
      uint4 wv[U];
      __nv_fp8x2_storage_t sv[U];
#pragma unroll
      for (int i = 0; i < U; ++i) {
        const bool in = c + 32 * i < chunks;
        wv[i] = in ? ld_stream(wr + 16 * (size_t)(c + 32 * i)) : make_uint4(0, 0, 0, 0);
        sv[i] = in ? sr[c + 32 * i] : (__nv_fp8x2_storage_t)0;
      }
#pragma unroll
      for (int i = 0; i < U; ++i) {
        if (c + 32 * i >= chunks) break;
        // the two block scales, x 2^14 (the codes are dequantised as e2m1 * 2^-14)
        __half2_raw hs = __nv_cvt_fp8x2_to_halfraw2(sv[i], __NV_E4M3);
        float2 sc = __half22float2(*reinterpret_cast<const __half2*>(&hs));
        sc.x *= 16384.f;
        sc.y *= 16384.f;
        float f[32];
        dequant32(wv[i], f);  // once per chunk, reused by the MT activation rows
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)min(m0 + m, M - 1) * K) +
                            4 * (c + 32 * i);
          const float p0 = dot16(f, __ldg(xr), __ldg(xr + 1));
          const float p1 = dot16(f + 16, __ldg(xr + 2), __ldg(xr + 3));
          acc[m] = fmaf(p0, sc.x, fmaf(p1, sc.y, acc[m]));
        }
      }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
      const float b = bias ? __bfloat162float(bias[row]) : 0.f;
#pragma unroll
      for (int m = 0; m < MT; ++m)
        if (m0 + m < M)
          y[(size_t)(m0 + m) * N + row] = __float2bfloat16(fmaf(acc[m], tensor_scale, b));
    }
  }
}

template <int MT>
void launch(const bf16* x, const uint8_t* w, const uint8_t* s, float ts, const bf16* b, bf16* y,
            int M, int K, int N, int unroll, cudaStream_t stream) {
  const dim3 grid((N + 7) / 8), block(256);
  switch (unroll) {
    case 1: fp4_gemv_kernel<MT, 1><<<grid, block, 0, stream>>>(x, w, s, ts, b, y, M, K, N); break;
    case 2: fp4_gemv_kernel<MT, 2><<<grid, block, 0, stream>>>(x, w, s, ts, b, y, M, K, N); break;
    case 8: fp4_gemv_kernel<MT, 8><<<grid, block, 0, stream>>>(x, w, s, ts, b, y, M, K, N); break;
    default: fp4_gemv_kernel<MT, 4><<<grid, block, 0, stream>>>(x, w, s, ts, b, y, M, K, N); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y = x @ W^T + bias with W = e2m1(codes) * block scale * tensor scale, for x [..., K]. One
// pybind call per forward: shape logic and the fallback live here, not in Python.
torch::Tensor fp4_linear(torch::Tensor x, torch::Tensor codes, torch::Tensor scales,
                         double tensor_scale, torch::Tensor bias, torch::Tensor lut,
                         int64_t unroll, int64_t max_rows) {
  const int64_t N = codes.size(0), K = 2 * codes.size(1);
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == K && x.device() == codes.device(),
              "fp4_linear: bad input");
  const int64_t M = x.numel() / K;
  if (M < 1 || M > max_rows || x.scalar_type() != at::kBFloat16) {
    // other shapes: the same FP4 math through cuBLAS (dequantised weight; a fallback)
    auto nib = at::stack({codes.bitwise_and(15), codes.bitwise_right_shift(4)}, -1);
    auto vals = lut.index({nib.reshape({N, K}).to(at::kLong)}).view({N, K / 16, 16});
    auto wd = (vals * scales.to(at::kFloat).unsqueeze(-1) * tensor_scale).view({N, K});
    wd = wd.to(x.scalar_type());
    return bias.numel() ? at::linear(x, wd, bias.to(x.scalar_type())) : at::linear(x, wd);
  }
  if (!x.is_contiguous() || (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15))
    x = x.clone(at::MemoryFormat::Contiguous);  // 128-bit loads need 16-byte alignment
  auto sizes = x.sizes().vec();
  sizes.back() = N;
  auto y = torch::empty(sizes, x.options());
  auto px = reinterpret_cast<const bf16*>(x.data_ptr());
  auto pw = reinterpret_cast<const uint8_t*>(codes.data_ptr());
  auto ps = reinterpret_cast<const uint8_t*>(scales.data_ptr());
  auto pb = bias.numel() ? reinterpret_cast<const bf16*>(bias.data_ptr()) : nullptr;
  auto py = reinterpret_cast<bf16*>(y.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  const float ts = (float)tensor_scale;
  if (M == 1) launch<1>(px, pw, ps, ts, pb, py, M, K, N, (int)unroll, stream);
  else launch<4>(px, pw, ps, ts, pb, py, M, K, N, (int)unroll, stream);
  return y;
}
"""
CPP_SRC = (
    "torch::Tensor fp4_linear(torch::Tensor x, torch::Tensor codes, torch::Tensor scales, "
    "double tensor_scale, torch::Tensor bias, torch::Tensor lut, int64_t unroll, "
    "int64_t max_rows);"
)

MAX_ROWS = 4  # decode GEMV; larger M takes the dequantised fallback

_ext = None


def _load():
    global _ext
    if _ext is None:
        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:8]
        _ext = load_inline(
            name=f"ka_fp4_gemv_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["fp4_linear"],
            extra_cuda_cflags=["-O3"],
        )
    return _ext


class Fp4Linear(nn.Module):
    """``nn.Linear`` with NVFP4 weights: e2m1 codes, an e4m3 scale per 16, an fp32 scale."""

    def __init__(self, reference: nn.Linear, unroll: int = 4) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        codes, scales, tensor_scale = quantize_fp4(reference.weight)  # once, here
        self.register_buffer("weight_codes", codes)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("weight_tensor_scale", tensor_scale)
        self.register_parameter("bias", reference.bias)
        self.quant_error = fp4_error(reference.weight, codes, scales, tensor_scale)
        none = torch.empty(0, dtype=torch.bfloat16, device=codes.device)
        bias = reference.bias if reference.bias is not None else none
        values = torch.tensor(E2M1_VALUES, device=codes.device)
        lut = torch.cat([values, -values])  # code -> value, for the fallback
        # plain attributes: the forward is one pybind call without nn.Module lookups
        self._args = (codes, scales, float(tensor_scale), bias, lut, int(unroll), MAX_ROWS)
        self._fn = _load().fp4_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x, *self._args)


def build(reference: nn.Module, unroll: int = 4) -> nn.Module:
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype == torch.bfloat16
        and reference.weight.is_cuda
        and reference.in_features % 32 == 0  # 32 weights (16 bytes of codes) per load
        and (reference.bias is None or reference.bias.dtype == torch.bfloat16)
    )
    return Fp4Linear(reference, unroll) if ok else reference
