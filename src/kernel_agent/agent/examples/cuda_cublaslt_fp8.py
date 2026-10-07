"""Example candidate (CUDA C++ via load_inline): FP8 GEMMs straight through cuBLASLt, with
descriptors, layouts and the algorithm cached per shape, for ``nn.Linear`` (and several
Linears on one input behind a single call).

Reduced precision: ``"precision": "fp8_w8a8"`` (a ``--quality near-lossless`` run,
knowledge/low_precision.md) in the default tensor-wise mode, ``"fp8_mx"`` in the MXFP8 mode
(``build(mxfp8=1)``); the exact tier rejects both.

Why: ``torch._scaled_mm`` costs 18-20 us of host time per eager call and ``at::_scaled_mm``
from C++ 19 us (ATen's scale checks, layout logic and a fresh output), against 6.1 us for a
cached ``cublasLtMatmul`` (5.5 us each with four behind one pybind11 call; docs/FP8.md §7,
RTX 5070 Ti, cuBLASLt 13.1). An eager-timed W8A8 decoder layer of the VoxCPM2 throughput run
went from 3.46x to 5.42x with this change alone (``dit_layer__fp8_w8a8``, slice 31). Inside
CUDA graphs host time does not count: then this is the same kernel as ``_scaled_mm``.

* Modes (``build(mxfp8=...)``):
  - **tensor-wise** (0, default; ``fp8_w8a8``): weights e4m3 with one fp32 scale per output
    channel (``quantize_fp8``, once in ``build()``), activations per token on every call
    (``_quant_token_k``), the GEMM with unit scalar scales (``SCALAR_32F``: cuBLASLt's nvjet
    kernels, 325-338 TFLOP/s at 8192^3 on sm_120) writes the unscaled product in bf16, and
    ``_scale_k`` applies ``sx[m] * sw[n]`` (+ bias) in fp32 with one more bf16 rounding
    (2^-9 relative, next to FP8's ~3 %). In a fused layer apply the scales where the output
    is read next instead (the SiLU-mul, the residual add): ``fp8_gemm`` is the GEMM alone.
    Split-K over K slices (a strided batch; ``_scale_k`` sums the partials) fills the SMs
    when N is small for M: chosen by timing (``splits=0``) or forced (1, 2).
  - **MXFP8** (1, ``fp8_mx``; ``VEC32_UE8M0``, sm_100+ / sm_120): e4m3 with one
    power-of-two ue8m0 scale per 32 elements along K on both operands, applied by the tensor
    core (``QMMA.SF``), so the output arrives scaled. The scale rule is ``fp8_mx``'s
    non-saturating ``2^ceil(log2(amax / 448))``, exact from the exponent bits in
    ``quant_mx_k`` as in ``quant.quantize_mxfp8`` (never the OCP floor rule, which
    saturates; the evaluator's scale-rule guard runs :func:`quantize_activations`). Weight
    scales are swizzled once into cuBLASLt's 128 x 4 layout (``quant.swizzle_mx_scales``);
    the activation kernel writes its scales there directly (``quant.mx_scale_offset``).
* Which mode and split pay depends on the shape and the GPU, so measure on the model at
  hand (the plan's ``us``, :func:`plan_info`). Evidence (RTX 5070 Ti, L2-cold CUDA graph,
  VoxCPM2 LocDiT shapes at M = 352, us; docs/FP8.md §3.1): MXFP8 beats tensor-wise on wide
  N (N = 8192, K = 1024: 24.7 vs 29.5; N = 2560: 9.3 vs 10.6), loses at N = 1024 where
  cuBLASLt has one MXFP8 algorithm and no split-K (K = 2048: 14.8 vs 8.8 tensor-wise with
  split-K 2; K = 4096: 26.6 vs 15.2); at M <= ~80 (memory bound) both stream the same weight
  bytes and MXFP8 is never faster. cuBLASLt on sm_120 has no ``OUTER_VEC_32F`` (row-wise),
  ``VEC128_32F`` or ``BLK128x128_32F`` (DeepSeek blockwise) mode: ``NOT_SUPPORTED``.
* Plans: one per (device, M, N, K, mode, split request). The heuristic's <= 8 algorithms
  (and the split options) are timed interleaved, 3 rounds, on scratch operands (sequential
  timing on a busy GPU once picked a 10 % slower kernel; the top-1 of the heuristic picked a
  slower ``sm89_xmma`` split-K kernel for one shape). Each new M is a new plan (~ms once):
  decode loops with growing M should bucket M or call :func:`plan_info` up front.
* CUDA graphs: the workspace is one fixed 32 MB buffer per device and outputs come from the
  caching allocator, so calls capture. A plan first needed during capture is not timed (the
  heuristic's first algorithm); call each shape once eagerly before capturing. One workspace
  per device: do not run these GEMMs on two streams at once.
* Fallback (other shapes, CPU, non-bf16): ``kernel_agent.kernels.quant.fp8_w8a8_linear``
  (tensor-wise) or ``quant.mxfp8_linear`` (MXFP8): the same numerics in torch.

Requirements: tensor-wise K and N multiples of 16 (sm_89+); MXFP8 K a multiple of 128 and N
of 16 (sm_100+). Not run on a GPU yet (written from the measured research code in
docs/research-scripts/fp8-sm120/lt_ext.py and the R3 layer); verify with ``kernel-agent
doctor --smoke`` and ``pytest -m gpu tests/test_fp8_toolkit.py``.
"""

import hashlib
import re

import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

from kernel_agent.kernels.quant import (
    fp8_error,
    fp8_w8a8_linear,
    mx_scale_offset,
    mxfp8_error,
    mxfp8_linear,
    quantize_fp8,
    quantize_mxfp8,
    swizzle_mx_scales,
)

# One namespace per candidate file (the evaluator names the module after the file's hash).
_NS = re.sub(r"\W", "_", __name__)

TENSORWISE, MXFP8 = 0, 1

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

typedef __nv_bfloat16 bf16;

#define LT_CHECK(x)                                                                     \
  do {                                                                                  \
    const cublasStatus_t st_ = (x);                                                     \
    TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, "cublasLt: " #x " returned ", (int)st_); \
  } while (0)

// set a matmul-descriptor / layout attribute from a variable
#define DESC_SET(desc, attr, v) \
  LT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_##attr, &(v), sizeof(v)))
#define LAYOUT_SET(l, attr, v) \
  LT_CHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_##attr, &(v), sizeof(v)))

enum { TENSORWISE = 0, MXFP8 = 1 };

// ------------------------------------------------------------------ activation quantisation

// max over a block of 256 threads (8 warps); every thread gets the result
__device__ __forceinline__ float block_max(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = lane < 8 ? red[lane] : 0.f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

__device__ __forceinline__ float amax8(const uint4& v, float m) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
  return m;
}

// 8 bf16 (one uint4) -> 8 e4m3 codes (one uint2): v / s rounded to nearest even, saturated
// to +-448 (the math of quant.quantize_fp8_activations: clamp, then round)
__device__ __forceinline__ uint2 to_e4m3x8(const uint4& v, float s) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
  uint2 out;
  __nv_fp8x2_storage_t* o = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    o[i] = __nv_cvt_float2_to_fp8x2(make_float2(f.x / s, f.y / s), __NV_SATFINITE, __NV_E4M3);
  }
  return out;
}

// Per token: one block per row; s[row] = amax / 448 (1 for a row of zeros), q = x / s.
__global__ void __launch_bounds__(256) quant_token_k(const bf16* __restrict__ x, int64_t ldx,
                                                     uint8_t* __restrict__ q,
                                                     float* __restrict__ s, int K) {
  __shared__ float red[8];
  const bf16* xr = x + (int64_t)blockIdx.x * ldx;
  float amax = 0.f;
  for (int c = threadIdx.x * 8; c < K; c += 256 * 8)
    amax = amax8(__ldg(reinterpret_cast<const uint4*>(xr + c)), amax);
  amax = block_max(amax, red);
  const float sc = amax > 0.f ? amax / 448.f : 1.f;
  if (threadIdx.x == 0) s[blockIdx.x] = sc;
  uint8_t* qr = q + (int64_t)blockIdx.x * K;
  for (int c = threadIdx.x * 8; c < K; c += 256 * 8)
    *reinterpret_cast<uint2*>(qr + c) =
        to_e4m3x8(__ldg(reinterpret_cast<const uint4*>(xr + c)), sc);
}

// ue8m0 exponent of a block: the smallest e with amax / 2^e <= 448, ceil(log2(amax / 448)),
// exact from the bits of amax = m * 2^E (m in [1, 2)): E - 8, plus one when m > 1.75 (448 =
// 1.75 * 2^8); zero / subnormal maxima -127; at most 126 (byte e + 127; 255 is NaN). The rule
// of kernel_agent.kernels.quant.quantize_mxfp8 (fp8_mx).
__device__ __forceinline__ int ue8m0_ceil(float amax) {
  const unsigned b = __float_as_uint(amax);
  const int field = (int)((b >> 23) & 0xff);
  if (field == 0) return -127;
  const int e = field - 127 - 8 + ((b & 0x7fffff) > 0x600000 ? 1 : 0);
  return max(-127, min(e, 126));
}

// Offset of scale (row, col) in cuBLASLt's 128 x 4 blocked layout (ncb = column blocks of 4):
// blocks of 128 rows x 4 columns, 512 bytes each, row-block major; inside a block row r sits
// at (r % 32) * 16 + (r / 32) * 4 + col % 4.
__device__ __forceinline__ int64_t blocked_offset(int row, int col, int ncb) {
  return ((int64_t)(row >> 7) * ncb + (col >> 2)) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 +
         (col & 3);
}

// MXFP8: one thread per 32 consecutive elements of a row: codes x / 2^e and the scale byte at
// its blocked place; the padding rows M..Mpad-1 get code 0 (as quant.swizzle_mx_scales).
__global__ void __launch_bounds__(256) quant_mx_k(const bf16* __restrict__ x, int64_t ldx,
                                                  uint8_t* __restrict__ q,
                                                  uint8_t* __restrict__ sf, int M, int Mpad,
                                                  int K) {
  const int groups = K >> 5, ncb = (groups + 3) >> 2;
  const int64_t g = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (g >= (int64_t)Mpad * groups) return;
  const int row = (int)(g / groups), col = (int)(g % groups);
  if (row >= M) {
    sf[blocked_offset(row, col, ncb)] = 0;
    return;
  }
  const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)row * ldx + col * 32);
  uint4 v[4];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = __ldg(xr + i);
    amax = amax8(v[i], amax);
  }
  const int e = ue8m0_ceil(amax);
  const float sc = ldexpf(1.f, e);  // a power of two: x / sc is exact before the rounding
  uint2* qr = reinterpret_cast<uint2*>(q + (int64_t)row * K + col * 32);
#pragma unroll
  for (int i = 0; i < 4; ++i) qr[i] = to_e4m3x8(v[i], sc);
  sf[blocked_offset(row, col, ncb)] = (uint8_t)(e + 127);
}

// Tensor-wise epilogue: y[m, n] = bf16((sum over splits of p[s, m, n]) * sx[m] * sw[n] + b[n]),
// 8 outputs per thread (N % 8 == 0). p may be y (one split: in place).
__global__ void __launch_bounds__(256) scale_k(const bf16* p, int splits, int64_t MN,
                                               const float* __restrict__ sx,
                                               const float* __restrict__ sw,
                                               const bf16* __restrict__ bias, bf16* y, int N) {
  const int64_t i = ((int64_t)blockIdx.x * 256 + threadIdx.x) * 8;
  if (i >= MN) return;
  const int m = (int)(i / N), n = (int)(i % N);
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  for (int s = 0; s < splits; ++s) {
    const uint4 v = *reinterpret_cast<const uint4*>(p + s * MN + i);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]);
      acc[2 * j] += f.x;
      acc[2 * j + 1] += f.y;
    }
  }
  const float a = sx[m];
  uint4 out;
  __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float r0 = acc[2 * j] * a * sw[n + 2 * j], r1 = acc[2 * j + 1] * a * sw[n + 2 * j + 1];
    if (bias) {
      r0 += __bfloat162float(bias[n + 2 * j]);
      r1 += __bfloat162float(bias[n + 2 * j + 1]);
    }
    o[j] = __floats2bfloat162_rn(r0, r1);
  }
  *reinterpret_cast<uint4*>(y + i) = out;
}

// ------------------------------------------------------------------ cuBLASLt plans

// y [M, N] (row-major) = x [M, K] . w [N, K]^T. In cuBLASLt's column-major terms D [N x M] =
// op(A) . B with A = w (K x N, ld K, transposed: the "TN" layout FP8 needs) and B = x (K x M,
// ld K): cuBLASLt's A scale is the weight's, its B scale the activations'.
struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  int pick = 0;     // the algorithm in use
  int splits = 1;   // split-K slices (a strided batch over K; D holds [splits, M, N] partials)
  float us = -1.f;  // timed per call (-1: untimed, the heuristic's first)
  void* ws = nullptr;
};

static constexpr size_t kWs = 32u << 20;
static std::mutex g_mu;
static std::map<int, void*> g_ws;  // one fixed workspace per device (graph-capturable)
static std::map<std::tuple<int, int64_t, int64_t, int64_t, int, int>, Plan> g_plans;

static cublasLtHandle_t handle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) LT_CHECK(cublasLtCreate(&h));
  return h;
}

static void* workspace(int dev) {
  auto it = g_ws.find(dev);
  if (it != g_ws.end()) return it->second;
  void* p = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&p, kWs));
  g_ws[dev] = p;
  return p;
}

static void destroy(Plan& p) {
  if (p.la) cublasLtMatrixLayoutDestroy(p.la);
  if (p.lb) cublasLtMatrixLayoutDestroy(p.lb);
  if (p.lc) cublasLtMatrixLayoutDestroy(p.lc);
  if (p.desc) cublasLtMatmulDescDestroy(p.desc);
  p = Plan();
}

// Descriptors, layouts and the heuristic's algorithms (<= 8; none if the mode is unsupported).
static Plan make_plan(int dev, int64_t M, int64_t N, int64_t K, int mode, int splits) {
  Plan p;
  p.splits = splits;
  p.ws = workspace(dev);
  LT_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  DESC_SET(p.desc, TRANSA, ta);
  DESC_SET(p.desc, TRANSB, tb);
  if (mode == MXFP8) {
    const cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    DESC_SET(p.desc, A_SCALE_MODE, sm);
    DESC_SET(p.desc, B_SCALE_MODE, sm);
    // the heuristic wants non-null, aligned scale pointers; the real ones are set per call
    const void* dummy = p.ws;
    DESC_SET(p.desc, A_SCALE_POINTER, dummy);
    DESC_SET(p.desc, B_SCALE_POINTER, dummy);
  } else {
    const int8_t fast = 0;  // plain fp32 accumulation; unit scalar scales (null pointers)
    DESC_SET(p.desc, FAST_ACCUM, fast);
  }
  const int64_t kb = K / splits;
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_8F_E4M3, kb, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_8F_E4M3, kb, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, N, M, N));
  if (splits > 1) {  // batch b reads K slice b of both operands and writes partial b of D
    const int32_t count = splits;
    const int64_t mn = M * N;
    for (cublasLtMatrixLayout_t l : {p.la, p.lb, p.lc}) LAYOUT_SET(l, BATCH_COUNT, count);
    LAYOUT_SET(p.la, STRIDED_BATCH_OFFSET, kb);
    LAYOUT_SET(p.lb, STRIDED_BATCH_OFFSET, kb);
    LAYOUT_SET(p.lc, STRIDED_BATCH_OFFSET, mn);
  }
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  const size_t ws = kWs;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &ws, sizeof(ws)));
  p.algos.resize(8);
  int got = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(handle(), p.desc, p.la, p.lb, p.lc, p.lc,
                                                           pref, 8, p.algos.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  p.algos.resize(st == CUBLAS_STATUS_SUCCESS ? got : 0);
  p.algos.erase(std::remove_if(p.algos.begin(), p.algos.end(),
                               [](const cublasLtMatmulHeuristicResult_t& r) {
                                 return r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize > kWs;
                               }),
                p.algos.end());
  return p;
}

static cublasStatus_t matmul(const Plan& p, int algo, const void* w, const void* x, const void* sw,
                             const void* sx, void* d, cudaStream_t stream) {
  if (sw) {  // MXFP8: this call's scale pointers (read at launch: graph capture records them)
    const cublasLtMatmulDescAttributes_t a = CUBLASLT_MATMUL_DESC_A_SCALE_POINTER;
    const cublasLtMatmulDescAttributes_t b = CUBLASLT_MATMUL_DESC_B_SCALE_POINTER;
    cublasLtMatmulDescSetAttribute(p.desc, a, &sw, sizeof(sw));
    cublasLtMatmulDescSetAttribute(p.desc, b, &sx, sizeof(sx));
  }
  const float alpha = 1.f, beta = 0.f;
  return cublasLtMatmul(handle(), p.desc, &alpha, w, p.la, x, p.lb, &beta, d, p.lc, d, p.lc,
                        &p.algos[algo].algo, p.ws, kWs, stream);
}

// The plan for a shape: every (split, algorithm) candidate timed interleaved (3 rounds of 10
// calls each, minimum per candidate) on scratch operands; inside a graph capture the
// heuristic's first algorithm, untimed.
static Plan choose(int dev, int64_t M, int64_t N, int64_t K, int mode, int splits, cudaStream_t st,
                   bool capturing) {
  std::vector<int> options;
  if (splits > 0) options = {splits};
  else if (mode == TENSORWISE && K % 64 == 0 && K >= 1024) options = {1, 2};
  else options = {1};
  std::vector<Plan> cands;
  for (int s : options) {
    Plan p = make_plan(dev, M, N, K, mode, s);
    if (p.algos.empty()) destroy(p);
    else cands.push_back(p);
  }
  TORCH_CHECK(!cands.empty(), "cublasLt has no FP8 algorithm for M=", M, " N=", N, " K=", K,
              " mode=", mode == MXFP8 ? "MXFP8" : "tensor-wise", " on this GPU");
  if (capturing) {
    for (size_t i = 1; i < cands.size(); ++i) destroy(cands[i]);
    return cands[0];
  }
  auto bytes = at::TensorOptions().device(at::kCUDA, dev).dtype(at::kByte);
  const int64_t kc = (K / 32 + 3) / 4 * 4;
  at::Tensor w = at::full({N * K}, 0x30, bytes), x = at::full({M * K}, 0x30, bytes);  // 0.5
  at::Tensor sw = at::full({(N + 127) / 128 * 128 * kc}, 127, bytes);
  at::Tensor sx = at::full({(M + 127) / 128 * 128 * kc}, 127, bytes);
  at::Tensor d = at::empty({2 * M * N * 2}, bytes);
  const void* psw = mode == MXFP8 ? sw.data_ptr() : nullptr;
  const void* psx = mode == MXFP8 ? sx.data_ptr() : nullptr;
  std::vector<std::pair<int, int>> ids;
  for (size_t c = 0; c < cands.size(); ++c)
    for (size_t a = 0; a < cands[c].algos.size(); ++a) ids.emplace_back((int)c, (int)a);
  std::vector<float> best(ids.size(), 1e30f);
  for (size_t i = 0; i < ids.size(); ++i)  // warm up; drop what fails to run
    for (int r = 0; r < 2; ++r)
      if (matmul(cands[ids[i].first], ids[i].second, w.data_ptr(), x.data_ptr(), psw, psx,
                 d.data_ptr(), st) != CUBLAS_STATUS_SUCCESS)
        best[i] = -1.f;
  cudaEvent_t e0, e1;
  C10_CUDA_CHECK(cudaEventCreate(&e0));
  C10_CUDA_CHECK(cudaEventCreate(&e1));
  for (int round = 0; round < 3; ++round)
    for (size_t i = 0; i < ids.size(); ++i) {
      if (best[i] < 0.f) continue;
      C10_CUDA_CHECK(cudaEventRecord(e0, st));
      for (int t = 0; t < 10; ++t)
        matmul(cands[ids[i].first], ids[i].second, w.data_ptr(), x.data_ptr(), psw, psx,
               d.data_ptr(), st);
      C10_CUDA_CHECK(cudaEventRecord(e1, st));
      C10_CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0.f;
      C10_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      best[i] = std::min(best[i], 100.f * ms);  // us per call
    }
  C10_CUDA_CHECK(cudaEventDestroy(e0));
  C10_CUDA_CHECK(cudaEventDestroy(e1));
  int arg = -1;
  for (size_t i = 0; i < ids.size(); ++i)
    if (best[i] >= 0.f && (arg < 0 || best[i] < best[arg])) arg = (int)i;
  TORCH_CHECK(arg >= 0, "cublasLt: no FP8 algorithm ran for M=", M, " N=", N, " K=", K);
  const int c = ids[arg].first;
  for (size_t i = 0; i < cands.size(); ++i)
    if ((int)i != c) destroy(cands[i]);
  Plan out = cands[c];
  out.pick = ids[arg].second;
  out.us = best[arg];
  return out;
}

static const Plan& get_plan(int64_t M, int64_t N, int64_t K, int mode, int splits,
                            cudaStream_t st) {
  const int dev = c10::cuda::current_device();
  const auto key = std::make_tuple(dev, M, N, K, mode, splits);
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(st, &cs));
  const bool capturing = cs != cudaStreamCaptureStatusNone;
  TORCH_CHECK(!capturing || g_ws.count(dev),
              "FP8 cuBLASLt helper: call it once outside CUDA graph capture first (workspace)");
  return g_plans.emplace(key, choose(dev, M, N, K, mode, splits, st, capturing)).first->second;
}

// ------------------------------------------------------------------ entry points

static at::Tensor rows_of(const at::Tensor& x, int64_t M, int64_t K) {
  at::Tensor x2 = x.reshape({M, K});
  if (x2.stride(1) != 1 || x2.stride(0) % 8 || (reinterpret_cast<uintptr_t>(x2.data_ptr()) & 15))
    x2 = x2.clone(at::MemoryFormat::Contiguous);  // 128-bit loads need 16-byte aligned rows
  return x2;
}

// y_i = x . w_i^T (+ bias_i) for every weight w_i [N_i, K] e4m3 (one pybind call): x [..., K]
// bf16 is quantised once (per token, or MXFP8), then one cached cuBLASLt matmul per weight.
// ws_i: fp32 [N_i] per-channel scales (tensor-wise) or uint8 blocked ue8m0 scales (MXFP8);
// bias_i: bf16 [N_i] or empty. splits: 0 timed, else forced (tensor-wise only).
std::vector<at::Tensor> fp8_linears(at::Tensor x, std::vector<at::Tensor> w,
                                    std::vector<at::Tensor> ws, std::vector<at::Tensor> bias,
                                    int64_t mode, int64_t splits) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() >= 1,
              "fp8_linears: x must be bf16 on a CUDA device");
  TORCH_CHECK(!w.empty() && w.size() == ws.size() && w.size() == bias.size(),
              "fp8_linears: one scale and one (possibly empty) bias per weight");
  TORCH_CHECK(mode == TENSORWISE || splits <= 1, "fp8_linears: MXFP8 has no split-K");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t K = x.size(-1), M = K ? x.numel() / K : 0;
  TORCH_CHECK(K % (mode == MXFP8 ? 128 : 16) == 0, "fp8_linears: K=", K,
              mode == MXFP8 ? " is not a multiple of 128" : " is not a multiple of 16");
  for (auto& wi : w)
    TORCH_CHECK(wi.size(0) % 16 == 0, "fp8_linears: N=", wi.size(0), " is not a multiple of 16");
  auto sizes = x.sizes().vec();
  std::vector<at::Tensor> ys;
  if (M == 0) {
    for (auto& wi : w) {
      sizes.back() = wi.size(0);
      ys.push_back(at::empty(sizes, x.options()));
    }
    return ys;
  }
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const at::Tensor x2 = rows_of(x, M, K);
  const auto bytes = x.options().dtype(at::kByte);
  const at::Tensor xq = at::empty({M, K}, bytes);
  at::Tensor xs;
  if (mode == MXFP8) {
    const int64_t Mpad = (M + 127) / 128 * 128, groups = K / 32;
    xs = at::empty({Mpad * ((groups + 3) / 4 * 4)}, bytes);
    quant_mx_k<<<(unsigned)((Mpad * groups + 255) / 256), 256, 0, st>>>(
        (const bf16*)x2.data_ptr(), x2.stride(0), xq.data_ptr<uint8_t>(), xs.data_ptr<uint8_t>(),
        (int)M, (int)Mpad, (int)K);
  } else {
    xs = at::empty({M}, x.options().dtype(at::kFloat));
    quant_token_k<<<(unsigned)M, 256, 0, st>>>((const bf16*)x2.data_ptr(), x2.stride(0),
                                                xq.data_ptr<uint8_t>(), xs.data_ptr<float>(),
                                                (int)K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for (size_t i = 0; i < w.size(); ++i) {
    const int64_t N = w[i].size(0);
    TORCH_CHECK(w[i].dim() == 2 && w[i].size(1) == K && w[i].is_contiguous(),
                "fp8_linears: weight ", i, " must be a contiguous [N, K] e4m3 tensor");
    const Plan& p = get_plan(M, N, K, (int)mode, (int)splits, st);
    sizes.back() = N;
    at::Tensor y = at::empty(sizes, x.options());
    at::Tensor part = p.splits > 1 ? at::empty({p.splits, M, N}, x.options()) : y;
    const void* sw = mode == MXFP8 ? ws[i].data_ptr() : nullptr;
    const void* sx = mode == MXFP8 ? xs.data_ptr() : nullptr;
    LT_CHECK(matmul(p, p.pick, w[i].data_ptr(), xq.data_ptr(), sw, sx, part.data_ptr(), st));
    if (mode == TENSORWISE) {
      const int64_t mn = M * N;
      scale_k<<<(unsigned)((mn / 8 + 255) / 256), 256, 0, st>>>(
          (const bf16*)part.data_ptr(), p.splits, mn, xs.data_ptr<float>(), ws[i].data_ptr<float>(),
          bias[i].numel() ? (const bf16*)bias[i].data_ptr() : nullptr, (bf16*)y.data_ptr(), (int)N);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (bias[i].numel()) {
      y.add_(bias[i]);
    }
    ys.push_back(y);
  }
  return ys;
}

// The GEMM alone on activations a producer already quantised (xq [M, K] e4m3 as uint8 or
// float8): tensor-wise writes the UNSCALED product into out ([splits, M, N] bf16; apply
// x_scale[m] * w_scale[n] where the output is read next), MXFP8 the scaled one into out
// [M, N] (xs, ws: blocked ue8m0 scales). splits: forced, >= 1.
void fp8_gemm(at::Tensor xq, at::Tensor xs, at::Tensor w, at::Tensor ws, at::Tensor out,
              int64_t mode, int64_t splits) {
  TORCH_CHECK(splits >= 1 && (mode == TENSORWISE || splits == 1), "fp8_gemm: bad splits");
  const c10::cuda::CUDAGuard guard(xq.device());
  const int64_t M = xq.size(0), K = xq.size(1), N = w.size(0);
  TORCH_CHECK(xq.is_contiguous() && w.is_contiguous() && w.size(1) == K, "fp8_gemm: bad operands");
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 &&
                  out.numel() == splits * M * N,
              "fp8_gemm: out must be contiguous bf16 [splits, M, N]");
  if (M == 0) return;
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const Plan& p = get_plan(M, N, K, (int)mode, (int)splits, st);
  LT_CHECK(matmul(p, p.pick, w.data_ptr(), xq.data_ptr(), mode == MXFP8 ? ws.data_ptr() : nullptr,
                  mode == MXFP8 ? xs.data_ptr() : nullptr, out.data_ptr(), st));
}

// The MXFP8 activation quantisation of fp8_linears alone: codes [M, K] (uint8) and the scales
// in the blocked layout (flat uint8): what the fp8_mx scale-rule guard checks.
std::vector<at::Tensor> quant_mx(at::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.size(-1) % 32 == 0,
              "quant_mx: bf16 CUDA x [..., K], K % 32 == 0");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t K = x.size(-1), M = x.numel() / K;
  const int64_t Mpad = (M + 127) / 128 * 128, groups = K / 32;
  const auto bytes = x.options().dtype(at::kByte);
  at::Tensor q = at::empty({M, K}, bytes);
  at::Tensor sf = at::empty({Mpad * ((groups + 3) / 4 * 4)}, bytes);
  if (M == 0) return {q, sf};
  const at::Tensor x2 = rows_of(x, M, K);
  quant_mx_k<<<(unsigned)((Mpad * groups + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)x2.data_ptr(), x2.stride(0), q.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(),
      (int)M, (int)Mpad, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, sf};
}

// [splits, algorithms the heuristic offered, us per call (-1: untimed)] of the plan for a
// shape and mode (timed now unless capturing): which mode and split run here, measured.
std::vector<double> plan_info(int64_t M, int64_t N, int64_t K, int64_t mode, int64_t splits) {
  const Plan& p = get_plan(M, N, K, (int)mode, (int)splits, at::cuda::getCurrentCUDAStream());
  return {(double)p.splits, (double)p.algos.size(), (double)p.us};
}
"""
CPP_SRC = """
std::vector<at::Tensor> fp8_linears(at::Tensor x, std::vector<at::Tensor> w,
                                    std::vector<at::Tensor> ws, std::vector<at::Tensor> bias,
                                    int64_t mode, int64_t splits);
void fp8_gemm(at::Tensor xq, at::Tensor xs, at::Tensor w, at::Tensor ws, at::Tensor out,
              int64_t mode, int64_t splits);
std::vector<double> plan_info(int64_t M, int64_t N, int64_t K, int64_t mode, int64_t splits);
std::vector<at::Tensor> quant_mx(at::Tensor x);
"""

_ext = None


def _load():
    global _ext
    if _ext is None:
        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:8]
        _ext = load_inline(
            name=f"ka_cublaslt_fp8_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["fp8_linears", "fp8_gemm", "plan_info", "quant_mx"],
            extra_cuda_cflags=["-O3"],
            extra_ldflags=["-lcublasLt"],
        )
    return _ext


def plan_info(m: int, n: int, k: int, mode: int = TENSORWISE, splits: int = 0) -> dict:
    """The cached plan of a shape (timed on first use): ``splits``, ``algorithms`` offered by
    the heuristic, ``us`` per call (-1: untimed). Compare modes and splits on the model's own
    shapes before choosing; an unsupported mode raises."""
    s, n_algos, us = _load().plan_info(m, n, k, mode, splits)
    return {"splits": int(s), "algorithms": int(n_algos), "us": us}


# ------------------------------------------------------------------ MXFP8 activations


def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 mode's activation quantisation of ``x [..., K]``: codes e4m3 ``[rows, K]``
    and unswizzled e8m0 scales ``[rows, K / 32]`` (what the ``fp8_mx`` scale-rule guard
    checks). CUDA bf16 with K % 128 == 0: this file's ``quant_mx_k`` (its blocked scales read
    back through ``quant.mx_scale_offset``); else ``quant.quantize_mxfp8``, the same rule."""
    x2 = x.reshape(-1, x.shape[-1])
    rows, k = x2.shape
    if not (x2.is_cuda and x2.dtype == torch.bfloat16 and k % 128 == 0):
        return quantize_mxfp8(x2)
    codes, flat = _load().quant_mx(x2)
    r = torch.arange(rows, device=x2.device)[:, None]
    c = torch.arange(k // 32, device=x2.device)[None, :]
    scales = flat[mx_scale_offset(r, c, k // 32)]
    return codes.view(torch.float8_e4m3fn), scales.view(torch.float8_e8m0fnu)


def shape_ok(n: int, k: int, mode: int) -> bool:
    """Whether the cuBLASLt path takes a weight ``[n, k]`` in ``mode`` (FP8 operands need
    multiples of 16; MXFP8 whole 128-wide scale blocks along K)."""
    return n % 16 == 0 and k % (128 if mode == MXFP8 else 16) == 0


# ------------------------------------------------------------------ modules


def _lt_linear(
    x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, bias: torch.Tensor, mode: int, splits: int
) -> torch.Tensor:
    return _load().fp8_linears(x, [w], [ws], [bias], mode, splits)[0]


# One opaque op for torch.compile / CUDA graphs (fresh output, no in-place writes).
lt_linear = torch.library.custom_op(f"{_NS}::lt_linear", _lt_linear, mutates_args=())


@lt_linear.register_fake
def _(x, w, ws, bias, mode, splits):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


class _Quantised:
    """One Linear's weight in ``mode``: (codes, scales for the kernel, bias or empty, fallback)."""

    def __init__(self, reference: nn.Linear, mode: int) -> None:
        w = reference.weight.detach()
        self.bias = reference.bias
        self.mode = mode
        if mode == MXFP8:  # e4m3 + e8m0 per 32 (ceil rule), scales swizzled once
            self.codes, self.exps = quantize_mxfp8(w)
            self.scales = swizzle_mx_scales(self.exps)
            self.error = mxfp8_error(w, self.codes, self.exps)
        else:
            self.codes, self.scales = quantize_fp8(w)  # once, here: never per call
            self.error = fp8_error(w, self.codes, self.scales)
        none = torch.empty(0, dtype=torch.bfloat16, device=w.device)
        self.bias_arg = reference.bias.detach() if reference.bias is not None else none

    def fallback(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == MXFP8:
            return mxfp8_linear(x, self.codes, self.exps, self.bias)
        return fp8_w8a8_linear(x, self.codes, self.scales, self.bias)


class Fp8LtLinear(nn.Module):
    """``nn.Linear`` on FP8 through a cached cuBLASLt call (one pybind call per forward)."""

    def __init__(self, reference: nn.Linear, mode: int, splits: int) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        q = _Quantised(reference, mode)
        self.register_buffer("weight_fp8", q.codes)
        self.register_buffer("weight_scale", q.scales)
        self.register_parameter("bias", reference.bias)
        self.quant_error = q.error  # for NOTES.md
        self._q = q
        # plain attributes: the forward does no nn.Module lookups
        self._args = (q.codes, q.scales, q.bias_arg, mode, splits)
        self._fn = _load().fp8_linears

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            return self._q.fallback(x)
        if torch.compiler.is_compiling():
            return lt_linear(x, *self._args)
        w, ws, b, mode, splits = self._args
        return self._fn(x, [w], [ws], [b], mode, splits)[0]


class Fp8LtGroup(nn.Module):
    """Several ``nn.Linear`` on the same input (q / k / v, gate / up) behind ONE pybind call:
    the input is quantised once and every GEMM uses its cached plan. ``forward(x)`` returns
    the outputs in order. Eager only (no custom op): inside a compiled region use
    :class:`Fp8LtLinear` per projection, or register this call as a list-returning op."""

    def __init__(self, references: list[nn.Linear], mode: int = TENSORWISE, splits: int = 0):
        super().__init__()
        qs = [_Quantised(r, mode) for r in references]
        self._qs = qs
        self._lists = ([q.codes for q in qs], [q.scales for q in qs], [q.bias_arg for q in qs])
        self._mode, self._splits = mode, splits
        self._fn = _load().fp8_linears

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            return tuple(q.fallback(x) for q in self._qs)
        return tuple(self._fn(x, *self._lists, self._mode, self._splits))


def build(reference: nn.Module, mxfp8: int = 0, splits: int = 0) -> nn.Module:
    """``mxfp8``: 0 tensor-wise W8A8 (``fp8_w8a8``), 1 MXFP8 (needs sm_100+); ``splits``: 0
    timed per shape, 1 or 2 forced (tensor-wise). Tuning keywords for ``sweep_candidate``."""
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype == torch.bfloat16
        and reference.weight.is_cuda
        and (reference.bias is None or reference.bias.dtype == torch.bfloat16)
    )
    if not ok:
        return reference
    mode = MXFP8 if mxfp8 else TENSORWISE
    capability = torch.cuda.get_device_capability(reference.weight.device)
    if capability < ((10, 0) if mode == MXFP8 else (8, 9)):
        return reference
    if not shape_ok(reference.out_features, reference.in_features, mode):
        return reference
    return Fp8LtLinear(reference, mode, 0 if mode == MXFP8 else int(splits))
