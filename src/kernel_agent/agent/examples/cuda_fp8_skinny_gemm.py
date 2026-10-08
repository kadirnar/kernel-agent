"""Example candidate (CUDA C++ via load_inline): FP8 e4m3 weight-only skinny GEMM (M <= 32)
for ``nn.Linear``, on bf16 tensor cores (``mma.sync`` m16n8k16).

Reduced precision: only for a target whose spec says ``"precision": "fp8_weights"`` (a
``--quality near-lossless`` run, skill fp8-weights); the exact tier rejects it.

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_fp8``: e4m3,
  one fp32 scale per output channel); no bf16 copy is kept.
* The weight is the A operand (16 output channels x k16, row-major as stored), the tokens
  the B operand (x is ``[M, K]``, the "col" layout), so fragments load straight from global
  memory: thread t of a quad loads 16 consecutive k (one 128-bit load of e4m3 codes per
  channel row) and both operands use the same k permutation (a dot product does not care
  about the order of k). The codes are upcast in registers (``cvt`` e4m3x2 -> f16x2 ->
  fp32 -> bf16x2, exact: every e4m3 value is a bf16 value) and fed to the bf16 MMA with
  fp32 accumulation; activations stay bf16. The per-channel scale (and the bias) are
  applied once per output in the epilogue.
* Block = ``16 * rows`` output channels; ``warps`` warps take interleaved 64-wide k chunks
  (``unroll`` chunks in flight per warp) and reduce through shared memory. Every warp
  computes all ``rows`` channel tiles of its block: the activations it loads serve
  ``16 * rows`` channels (activation traffic from L2 falls by that factor). By default
  both are chosen per call: 1 tile x 8 warps up to M = 24, 2 tiles x 4 warps above (when
  the grid keeps >= 96 blocks).
* ``mma.sync`` with e4m3 x e4m3 inputs (m16n8k32) also runs on sm_120, but it needs FP8
  activations (W8A8): not weight-only.
* 32 < M <= ``MAX_ROWS``: groups of 32 tokens (``blockIdx.y``) that re-read the weights from
  L2. Correct, but from M ~ 64 the GEMM is compute bound and cuBLAS bf16 is faster (M = 176:
  26.6 vs 20.0 us): weight-only FP8 pays for few rows only. Other shapes (more rows, other
  dtypes): the dequantised weight through cuBLAS (a fallback).

RTX 5070 Ti, weights streamed from DRAM, against cuBLAS bf16: LocDiT [22, 1024] x
[1024, 4096] in 6.9 us (1.75x), [22, 4096] x [4096, 1024] in 7.6 us (1.75x); batched LM
decode (M = 8 / 16) [M, 2048] x [2048, 6144] in 16.3 / 16.6 us (1.94x / 1.91x), [M, 6144] x
[6144, 2048] in 17.0 / 17.6 us (2.30x / 2.22x).
"""

import hashlib

import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

from kernel_agent.kernels.quant import fp8_error, quantize_fp8

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_80+"
ARCHS_WHY = (
    "bf16 mma.sync m16n8k16; e4m3 weights converted in registers (hardware cvt from sm_89, "
    "CUDA's software conversion before)"
)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// 4 e4m3 codes (k, k+1, k+2, k+3) -> bf16x2 (k, k+1) in lo and (k+2, k+3) in hi; exact
__device__ __forceinline__ void fp8x4_to_bf16(unsigned v, unsigned& lo, unsigned& hi) {
  __half2_raw h0 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xffffu), __NV_E4M3);
  __half2_raw h1 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v >> 16), __NV_E4M3);
  __nv_bfloat162 b0 = __float22bfloat162_rn(__half22float2(*reinterpret_cast<__half2*>(&h0)));
  __nv_bfloat162 b1 = __float22bfloat162_rn(__half22float2(*reinterpret_cast<__half2*>(&h1)));
  lo = *reinterpret_cast<unsigned*>(&b0);
  hi = *reinterpret_cast<unsigned*>(&b1);
}

__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                    unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// y[m, r] = scale[r] * sum_k x[m, k] q[r, k] + bias[r]; x: [M, K] bf16, q: [R, K] e4m3,
// y: [M, R] bf16. NT token tiles of 8, RB channel tiles of 16 per block, WARPS warps take
// interleaved 64-wide k chunks, U chunks in flight. GROUPED (M > 32): blockIdx.y picks a group
// of 8 * NT tokens, and the groups re-read the weight tile from L2 (a separate instantiation:
// in the M <= 32 kernels the group arithmetic cost ~30 % of their time). Fragments of mma j
// (0..3) of a chunk: thread (g = lane / 4, t = lane % 4) holds physical k = 16t + 4j + {0, 1}
// (logical k 2t, 2t+1 of the m16n8k16 layout) and 16t + 4j + {2, 3} (logical 2t+8, 2t+9), in
// A and in B.
template <int NT, int RB, int WARPS, int U, bool GROUPED>
__global__ void __launch_bounds__(WARPS * 32) fp8_skinny_kernel(
    const bf16* __restrict__ x_all, const uint8_t* __restrict__ q, const float* __restrict__ scale,
    const bf16* __restrict__ bias, bf16* __restrict__ y_all, int M_all, int K, int R) {
  __shared__ float red[WARPS][RB][NT][4][32];
  const int tok0 = GROUPED ? blockIdx.y * 8 * NT : 0;
  const bf16* __restrict__ x = GROUPED ? x_all + (size_t)tok0 * K : x_all;
  bf16* __restrict__ y = GROUPED ? y_all + (size_t)tok0 * R : y_all;
  const int M = GROUPED ? min(M_all - tok0, 8 * NT) : M_all;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * 16 * RB;
  const uint8_t* pa = q + (size_t)(row0 + g) * K + 16 * t;
  const bf16* px[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) px[n] = x + (size_t)min(n * 8 + g, M - 1) * K + 16 * t;

  float acc[RB][NT][4];
#pragma unroll
  for (int b = 0; b < RB; ++b)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[b][n][j] = 0.f;

  const int chunks = K >> 6;
  for (int c0 = warp; c0 < chunks; c0 += WARPS * U) {
    uint4 a[U][RB][2];  // rows g and g + 8 of every channel tile
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = c0 + u * WARPS;
      if (c < chunks) {
#pragma unroll
        for (int b = 0; b < RB; ++b) {
          a[u][b][0] = ld_stream(pa + (size_t)(16 * b) * K + 64 * (size_t)c);
          a[u][b][1] = ld_stream(pa + (size_t)(16 * b + 8) * K + 64 * (size_t)c);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = c0 + u * WARPS;
      if (c >= chunks) break;
      unsigned xb[NT][8];  // 16 bf16 of token tile n: k 16t .. 16t+15
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const uint4* xp = reinterpret_cast<const uint4*>(px[n] + 64 * (size_t)c);
        const uint4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
        xb[n][0] = x0.x; xb[n][1] = x0.y; xb[n][2] = x0.z; xb[n][3] = x0.w;
        xb[n][4] = x1.x; xb[n][5] = x1.y; xb[n][6] = x1.z; xb[n][7] = x1.w;
      }
#pragma unroll
      for (int b = 0; b < RB; ++b) {
        const unsigned* r0 = reinterpret_cast<const unsigned*>(&a[u][b][0]);
        const unsigned* r1 = reinterpret_cast<const unsigned*>(&a[u][b][1]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          unsigned a0, a1, a2, a3;
          fp8x4_to_bf16(r0[j], a0, a2);  // row g:     k 4j, 4j+1 | 4j+2, 4j+3
          fp8x4_to_bf16(r1[j], a1, a3);  // row g + 8
#pragma unroll
          for (int n = 0; n < NT; ++n)
            mma(acc[b][n], a0, a1, a2, a3, xb[n][2 * j], xb[n][2 * j + 1]);
        }
      }
    }
  }

#pragma unroll
  for (int b = 0; b < RB; ++b)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) red[warp][b][n][j][lane] = acc[b][n][j];
  __syncthreads();
  // c[j] of tile (b, n): channel 16b + g + 8 * (j / 2), token 8n + 2t + (j % 2)
  for (int idx = threadIdx.x; idx < RB * NT * 128; idx += WARPS * 32) {
    const int l = idx & 31, j = (idx >> 5) & 3, n = (idx >> 7) % NT, b = idx / (NT * 128);
    const int m = n * 8 + 2 * (l & 3) + (j & 1);
    if (m >= M) continue;
    const int r = row0 + 16 * b + (l >> 2) + (j >> 1) * 8;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) s += red[w][b][n][j][l];
    const float bv = bias ? __bfloat162float(bias[r]) : 0.f;
    y[(size_t)m * R + r] = __float2bfloat16(fmaf(s, scale[r], bv));
  }
}

template <int NT, int RB, int WARPS, bool G>
void launch_u(const bf16* x, const uint8_t* q, const float* s, const bf16* b, bf16* y, int M,
              int K, int R, int unroll, cudaStream_t stream) {
  const dim3 grid(R / (16 * RB), G ? (M + 8 * NT - 1) / (8 * NT) : 1), block(WARPS * 32);
  if (unroll >= 2)
    fp8_skinny_kernel<NT, RB, WARPS, 2, G><<<grid, block, 0, stream>>>(x, q, s, b, y, M, K, R);
  else
    fp8_skinny_kernel<NT, RB, WARPS, 1, G><<<grid, block, 0, stream>>>(x, q, s, b, y, M, K, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NT, bool G>
void launch(const bf16* x, const uint8_t* q, const float* s, const bf16* b, bf16* y, int M, int K,
            int R, int rows, int warps, int unroll, cudaStream_t stream) {
  if (rows >= 2 && warps <= 4) launch_u<NT, 2, 4, G>(x, q, s, b, y, M, K, R, unroll, stream);
  else if (rows >= 2) launch_u<NT, 2, 8, G>(x, q, s, b, y, M, K, R, unroll, stream);
  else if (warps <= 4) launch_u<NT, 1, 4, G>(x, q, s, b, y, M, K, R, unroll, stream);
  else launch_u<NT, 1, 8, G>(x, q, s, b, y, M, K, R, unroll, stream);
}

// y = x @ (q * scale)^T + bias for x [..., K]: one pybind call per forward. rows: channel tiles
// of 16 per block, warps: the k split (0: auto), unroll: chunks in flight; max_rows: larger M
// falls back.
torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor q, torch::Tensor scale, torch::Tensor bias,
                         int64_t rows_, int64_t warps_, int64_t unroll_, int64_t max_rows) {
  const int64_t R = q.size(0), K = q.size(1);
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == K && x.device() == q.device(), "fp8_linear: bad input");
  const int64_t M = x.numel() / K;
  // auto: every block re-reads its tokens' activations from L2, so at M > 24 two channel
  // tiles per block (half the activation traffic) win while the grid keeps >= 96 blocks
  const int64_t groups = (M + 31) / 32;  // M > 32: groups of 32 tokens (grid.y)
  int rows = (int)rows_, warps = (int)warps_;
  if (rows <= 0) rows = M > 24 && R % 32 == 0 && R / 32 * groups >= 96 ? 2 : 1;
  if (warps <= 0) warps = rows == 2 ? 4 : 8;
  const int unroll = (int)unroll_;
  if (M < 1 || M > max_rows || x.scalar_type() != at::kBFloat16 || R % (16 * rows)) {
    auto wd = (q.to(at::kFloat) * scale.unsqueeze(1)).to(x.scalar_type());  // fallback
    return bias.numel() ? at::linear(x, wd, bias.to(x.scalar_type())) : at::linear(x, wd);
  }
  if (!x.is_contiguous() || (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15))
    x = x.clone(at::MemoryFormat::Contiguous);  // 128-bit loads need 16-byte alignment
  auto sizes = x.sizes().vec();
  sizes.back() = R;
  auto y = torch::empty(sizes, x.options());
  auto px = reinterpret_cast<const bf16*>(x.data_ptr());
  auto pq = reinterpret_cast<const uint8_t*>(q.data_ptr());
  auto ps = scale.data_ptr<float>();
  auto pb = bias.numel() ? reinterpret_cast<const bf16*>(bias.data_ptr()) : nullptr;
  auto py = reinterpret_cast<bf16*>(y.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M <= 8) launch<1, false>(px, pq, ps, pb, py, M, K, R, rows, warps, unroll, stream);
  else if (M <= 16) launch<2, false>(px, pq, ps, pb, py, M, K, R, rows, warps, unroll, stream);
  else if (M <= 24) launch<3, false>(px, pq, ps, pb, py, M, K, R, rows, warps, unroll, stream);
  else if (M <= 32) launch<4, false>(px, pq, ps, pb, py, M, K, R, rows, warps, unroll, stream);
  else launch<4, true>(px, pq, ps, pb, py, M, K, R, rows, warps, unroll, stream);  // groups of 32
  return y;
}
"""
CPP_SRC = (
    "torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor q, torch::Tensor scale, "
    "torch::Tensor bias, int64_t rows, int64_t warps, int64_t unroll, int64_t max_rows);"
)

#: Up to 32 tokens: one token group (LocDiT at batch 1, decode at batch <= 32); up to this
#: many (batched LocDiT, short prefills): groups of 32 that re-read the weights from L2;
#: more: the dequantised fallback (a compute-bound GEMM wants a tiled kernel, not this one).
MAX_ROWS = 512

_ext = None


def _load():
    global _ext
    if _ext is None:
        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:8]
        _ext = load_inline(
            name=f"ka_fp8_skinny_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["fp8_linear"],
            extra_cuda_cflags=["-O3"],
        )
    return _ext


class Fp8SkinnyLinear(nn.Module):
    """``nn.Linear`` with e4m3 weights and one fp32 scale per output channel (M <= 32)."""

    def __init__(self, reference: nn.Linear, rows: int, warps: int, unroll: int) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        q, scale = quantize_fp8(reference.weight)  # once, here: never per call
        self.register_buffer("weight_fp8", q)
        self.register_buffer("weight_scale", scale)
        self.register_parameter("bias", reference.bias)
        self.quant_error = fp8_error(reference.weight, q, scale)  # for NOTES.md
        none = torch.empty(0, dtype=torch.bfloat16, device=q.device)
        bias = reference.bias if reference.bias is not None else none
        # plain attributes: the forward is one pybind call without nn.Module lookups
        self._args = (q, scale, bias, rows, warps, unroll, MAX_ROWS)
        self._fn = _load().fp8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x, *self._args)


def build(reference: nn.Module, rows: int = 0, warps: int = 0, unroll: int = 1) -> nn.Module:
    """``rows`` (1 or 2 channel tiles of 16 per block), ``warps`` (4 or 8: the k split; 0 for
    both: chosen per call from M) and ``unroll`` (1 or 2 chunks in flight) are tuning
    keywords for ``sweep_candidate``."""
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype == torch.bfloat16
        and reference.weight.is_cuda
        and reference.in_features % 64 == 0  # 64-wide k chunks
        and reference.out_features % 16 == 0  # 16 channels per tile
        and (reference.bias is None or reference.bias.dtype == torch.bfloat16)
    )
    if not ok:
        return reference
    while rows > 1 and reference.out_features % (16 * rows):
        rows //= 2
    return Fp8SkinnyLinear(reference, rows, warps, unroll)
