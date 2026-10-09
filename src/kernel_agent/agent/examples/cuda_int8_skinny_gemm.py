"""Example candidate (CUDA C++ via load_inline): INT8 W8A8 skinny GEMM (decode, M <= 32 rows
per weight read) for ``nn.Linear`` on the IMMA tensor cores (``mma.sync m16n8k32
.s32.s8.s8.s32``; Turing: four ``m8n8k16`` per product), bf16 or fp16 activations.

Reduced precision: only for a target whose spec says ``"precision": "int8_w8a8"`` (a
``--quality near-lossless`` or ``relaxed`` run, skill int8-w8a8); the exact tier
rejects it.

* ``build()`` quantises the weight once (``kernel_agent.kernels.quant.quantize_int8``:
  symmetric int8 codes, one fp32 scale ``amax / 127`` per output channel); no 16-bit copy
  is kept.
* Every call, one pybind call launches two kernels: ``quant_rows`` (one block per token:
  ``scale = amax * (1 / 127)``, codes ``round(x / scale)`` to nearest even with IEEE
  division, as ``quant.quantize_int8_activations``: the same codes) and the GEMM. Both are
  templates on the activations' type (the model's dtype, :data:`DTYPES`); the output keeps
  it.
* The weight is the A operand (16 output channels x k32, row-major as stored), the tokens
  the B operand (8 tokens per tile), so fragments load straight from global memory with no
  conversion at all: thread t of a quad loads 16 consecutive k bytes (one 128-bit load) of
  its weight rows and of its token, and both operands use the same k permutation (a dot
  product does not care about the order of k): one 64-wide k chunk is two ``m16n8k32``
  products (Turing, sm_75, has the ``m8n8k16`` form only: four of them on the same
  fragments, the same integers). Accumulation is exact int32 (the cross-warp reduction
  too); the epilogue applies ``acc * x_scale[m] * w_scale[r] + bias[r]`` in fp32 (no FMA
  contraction: the reference's roundings) and rounds to bf16 / fp16 once, so the output
  equals ``quant.int8_w8a8_linear``'s.
* Block = ``16 * rows`` output channels; ``warps`` warps take interleaved 64-wide k chunks
  and reduce through shared memory; every warp computes all ``rows`` channel tiles of its
  block, so the activations it loads serve ``16 * rows`` channels. 32 < M <= ``MAX_ROWS``:
  groups of 32 tokens (``blockIdx.y``) that re-read the weight from L2; more rows:
  ``quant.int8_w8a8_linear`` (a compute-bound GEMM wants a tiled kernel:
  ``triton_int8_w8a8_gemm.py``).

Against weight-only kernels at decode: the weight bytes are the same (one per weight), but no
per-element conversion runs (the bf16 skinny FP8 kernel converts every weight it loads): at a
few to 32 rows per call, where a weight-only GEMV turns ALU bound, IMMA keeps it memory bound.
The Turing (m8n8k16) and Ampere paths: outputs verified bit for bit through their PTX
(``compute_75``, ``compute_86``) JIT-compiled on an RTX 5070 Ti; not timed on such a GPU yet.
"""

import hashlib

import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

from kernel_agent.kernels.quant import int8_error, int8_w8a8_linear, quantize_int8

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_75+"
ARCHS_WHY = "int8 mma.sync (IMMA): m16n8k32 from sm_80, four m8n8k16 on Turing"
#: Activation dtypes the kernels take (templates on the type; the output keeps it).
DTYPES = (torch.bfloat16, torch.float16)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

typedef __nv_bfloat16 bf16;

// The activation type T: bf16 or __half (the model's dtype; the output keeps it)
template <typename T> struct Act;
template <> struct Act<bf16> {
  static __device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ bf16 from(float v) { return __float2bfloat16(v); }
};
template <> struct Act<__half> {
  static __device__ __forceinline__ float f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from(float v) { return __float2half(v); }
};

__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// c += A (16 x k32 s8) . B (k32 x 8 s8), exact int32: m16n8k32 (sm_80+). Turing (sm_75) has the
// m8n8k16 form only: four of them on the same fragments, rows g (c0, c1) from a0 (k 0..15) and
// a2 (k 16..31), rows g + 8 (c2, c3) from a1 and a3, against b0 (k 0..15) and b1 (k 16..31).
__device__ __forceinline__ void imma(int* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                     unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
#define KA_IMMA_K16(c0, c1, a, b)                                                             \
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" \
               : "+r"(c0), "+r"(c1)                                                          \
               : "r"(a), "r"(b))
  KA_IMMA_K16(c[0], c[1], a0, b0);
  KA_IMMA_K16(c[0], c[1], a2, b1);
  KA_IMMA_K16(c[2], c[3], a1, b0);
  KA_IMMA_K16(c[2], c[3], a3, b1);
#undef KA_IMMA_K16
#endif
}

// One block per token row: scale = amax * (1 / 127) (1 for zeros), codes = round(x / scale) to
// nearest even (IEEE division), clamped to +-127: quant.quantize_int8_activations, bit for bit.
template <typename T>
__global__ void __launch_bounds__(256) quant_rows(const T* __restrict__ x, int8_t* __restrict__ q,
                                                  float* __restrict__ s, int K) {
  __shared__ float red[8];
  const T* xr = x + (size_t)blockIdx.x * K;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += 256) amax = fmaxf(amax, fabsf(Act<T>::f(xr[k])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) amax = fmaxf(amax, red[w]);
  const float scale = amax > 0.f ? __fmul_rn(amax, 1.0f / 127.0f) : 1.f;
  if (threadIdx.x == 0) s[blockIdx.x] = scale;
  int8_t* qr = q + (size_t)blockIdx.x * K;
  for (int k = threadIdx.x; k < K; k += 256) {
    const float v = rintf(__fdiv_rn(Act<T>::f(xr[k]), scale));
    qr[k] = (int8_t)fminf(fmaxf(v, -127.f), 127.f);
  }
}

// y[m, r] = sx[m] * sw[r] * sum_k xq[m, k] q[r, k] + bias[r]; xq: [M, K] int8 tokens, q: [R, K]
// int8 weight, y: [M, R] T. NT token tiles of 8, RB channel tiles of 16 per block, WARPS
// warps take interleaved 64-wide k chunks, U chunks in flight. GROUPED (M > 32): blockIdx.y
// picks a group of 8 * NT tokens. Thread (g = lane / 4, t = lane % 4) loads physical k 16t ..
// 16t + 15 of its rows; m16n8k32 product j (0, 1) of a chunk takes the bytes 16t + 8j .. + 3 as
// logical k 4t .. 4t + 3 and 16t + 8j + 4 .. + 7 as logical k 16 + 4t .., in A and in B.
template <typename T, int NT, int RB, int WARPS, int U, bool GROUPED>
__global__ void __launch_bounds__(WARPS * 32) int8_skinny_kernel(
    const int8_t* __restrict__ xq_all, const float* __restrict__ sx_all,
    const int8_t* __restrict__ q, const float* __restrict__ sw, const T* __restrict__ bias,
    T* __restrict__ y_all, int M_all, int K, int R) {
  __shared__ int red[WARPS][RB][NT][4][32];
  const int tok0 = GROUPED ? blockIdx.y * 8 * NT : 0;
  const int8_t* __restrict__ xq = xq_all + (size_t)tok0 * K;
  const float* __restrict__ sx = sx_all + tok0;
  T* __restrict__ y = y_all + (size_t)tok0 * R;
  const int M = GROUPED ? min(M_all - tok0, 8 * NT) : M_all;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * 16 * RB;
  const int8_t* pa = q + (size_t)(row0 + g) * K + 16 * t;
  const int8_t* px[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) px[n] = xq + (size_t)min(n * 8 + g, M - 1) * K + 16 * t;

  int acc[RB][NT][4];
#pragma unroll
  for (int b = 0; b < RB; ++b)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[b][n][j] = 0;

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
      uint4 xb[NT];  // 16 int8 of token tile n: k 16t .. 16t + 15
#pragma unroll
      for (int n = 0; n < NT; ++n)
        xb[n] = __ldg(reinterpret_cast<const uint4*>(px[n] + 64 * (size_t)c));
#pragma unroll
      for (int b = 0; b < RB; ++b) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          imma(acc[b][n], a[u][b][0].x, a[u][b][1].x, a[u][b][0].y, a[u][b][1].y, xb[n].x, xb[n].y);
          imma(acc[b][n], a[u][b][0].z, a[u][b][1].z, a[u][b][0].w, a[u][b][1].w, xb[n].z, xb[n].w);
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
    int s = 0;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) s += red[w][b][n][j][l];
    float v = __fmul_rn(__fmul_rn((float)s, sx[m]), sw[r]);  // acc * x_scale * w_scale
    if (bias) v = __fadd_rn(v, Act<T>::f(bias[r]));
    y[(size_t)m * R + r] = Act<T>::from(v);
  }
}

template <typename T, int NT, int RB, int WARPS, bool G>
void launch_u(const int8_t* xq, const float* sx, const int8_t* q, const float* sw, const T* b,
              T* y, int M, int K, int R, int unroll, cudaStream_t stream) {
  const dim3 grid(R / (16 * RB), G ? (M + 8 * NT - 1) / (8 * NT) : 1), block(WARPS * 32);
  if (unroll >= 2)
    int8_skinny_kernel<T, NT, RB, WARPS, 2, G>
        <<<grid, block, 0, stream>>>(xq, sx, q, sw, b, y, M, K, R);
  else
    int8_skinny_kernel<T, NT, RB, WARPS, 1, G>
        <<<grid, block, 0, stream>>>(xq, sx, q, sw, b, y, M, K, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int NT, bool G>
void launch(const int8_t* xq, const float* sx, const int8_t* q, const float* sw, const T* b, T* y,
            int M, int K, int R, int rows, int warps, int unroll, cudaStream_t st) {
  if (rows >= 2 && warps <= 4) launch_u<T, NT, 2, 4, G>(xq, sx, q, sw, b, y, M, K, R, unroll, st);
  else if (rows >= 2) launch_u<T, NT, 2, 8, G>(xq, sx, q, sw, b, y, M, K, R, unroll, st);
  else if (warps <= 4) launch_u<T, NT, 1, 4, G>(xq, sx, q, sw, b, y, M, K, R, unroll, st);
  else launch_u<T, NT, 1, 8, G>(xq, sx, q, sw, b, y, M, K, R, unroll, st);
}

// The per-token quantisation and the GEMM for activations of type T.
template <typename T>
void run(const torch::Tensor& x, torch::Tensor& xq, torch::Tensor& sx, const torch::Tensor& q,
         const torch::Tensor& sw, const torch::Tensor& bias, torch::Tensor& y, int m, int k, int r,
         int rows, int w, int u) {
  auto st = at::cuda::getCurrentCUDAStream();
  quant_rows<T><<<(unsigned)m, 256, 0, st>>>(reinterpret_cast<const T*>(x.data_ptr()),
                                              xq.data_ptr<int8_t>(), sx.data_ptr<float>(), k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto pxq = xq.data_ptr<int8_t>();
  auto psx = sx.data_ptr<float>();
  auto pq = q.data_ptr<int8_t>();
  auto psw = sw.data_ptr<float>();
  auto pb = bias.numel() ? reinterpret_cast<const T*>(bias.data_ptr()) : nullptr;
  auto py = reinterpret_cast<T*>(y.data_ptr());
  if (m <= 8) launch<T, 1, false>(pxq, psx, pq, psw, pb, py, m, k, r, rows, w, u, st);
  else if (m <= 16) launch<T, 2, false>(pxq, psx, pq, psw, pb, py, m, k, r, rows, w, u, st);
  else if (m <= 24) launch<T, 3, false>(pxq, psx, pq, psw, pb, py, m, k, r, rows, w, u, st);
  else if (m <= 32) launch<T, 4, false>(pxq, psx, pq, psw, pb, py, m, k, r, rows, w, u, st);
  else launch<T, 4, true>(pxq, psx, pq, psw, pb, py, m, k, r, rows, w, u, st);
}

// y = W8A8(x) for x [..., K] bf16 or fp16 (1 <= M <= max rows, checked by the caller): the
// per-token quantisation and the GEMM, one pybind call. rows: channel tiles of 16 per block,
// warps: the k split (0: auto), unroll: chunks in flight.
torch::Tensor int8_linear(torch::Tensor x, torch::Tensor q, torch::Tensor sw, torch::Tensor bias,
                          int64_t rows_, int64_t warps_, int64_t unroll_) {
  const int64_t R = q.size(0), K = q.size(1);
  const auto dt = x.scalar_type();
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) == K && x.device() == q.device() &&
                  (dt == at::kBFloat16 || dt == at::kHalf) &&
                  (!bias.numel() || bias.scalar_type() == dt),
              "int8_linear: bad input");
  const int64_t M = x.numel() / K;
  TORCH_CHECK(M >= 1 && K % 64 == 0 && R % 16 == 0, "int8_linear: unsupported shape");
  const int64_t groups = (M + 31) / 32;
  int rows = (int)rows_, warps = (int)warps_;
  // auto (swept streamed on an RTX 5070 Ti): two channel tiles per block (half the activation
  // traffic) for token groups, and past 16 rows when R <= 4096 (wider outputs fill the GPU
  // with one tile), while the grid keeps >= 96 blocks; 8 warps, 4 for token groups
  const bool two = M > 32 || (M > 16 && R <= 4096);
  if (rows <= 0) rows = two && R % 32 == 0 && R / 32 * groups >= 96 ? 2 : 1;
  if (R % (16 * rows)) rows = 1;
  if (warps <= 0) warps = M > 32 && rows == 2 ? 4 : 8;
  const int unroll = (int)unroll_;
  x = x.contiguous();
  auto opts = x.options();
  auto xq = torch::empty({M, K}, opts.dtype(at::kChar));
  auto sx = torch::empty({M}, opts.dtype(at::kFloat));
  auto sizes = x.sizes().vec();
  sizes.back() = R;
  auto y = torch::empty(sizes, opts);
  const int m = (int)M, k = (int)K, r = (int)R;
  if (dt == at::kHalf) run<__half>(x, xq, sx, q, sw, bias, y, m, k, r, rows, warps, unroll);
  else run<bf16>(x, xq, sx, q, sw, bias, y, m, k, r, rows, warps, unroll);
  return y;
}
"""
CPP_SRC = (
    "torch::Tensor int8_linear(torch::Tensor x, torch::Tensor q, torch::Tensor sw, "
    "torch::Tensor bias, int64_t rows, int64_t warps, int64_t unroll);"
)

#: Up to 32 tokens: one token group (decode at batch <= 32); up to this many: groups of 32
#: that re-read the weight from L2; more: ``quant.int8_w8a8_linear`` (``torch._int_mm``).
MAX_ROWS = 128

_ext = None


def _load():
    global _ext
    if _ext is None:
        tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:8]
        _ext = load_inline(
            name=f"ka_int8_skinny_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["int8_linear"],
            extra_cuda_cflags=["-O3"],
        )
    return _ext


class Int8SkinnyLinear(nn.Module):
    """``nn.Linear`` with int8 weights and activations (per channel / per token scales),
    int32 accumulation on IMMA (M <= ``MAX_ROWS``)."""

    def __init__(self, reference: nn.Linear, rows: int, warps: int, unroll: int) -> None:
        super().__init__()
        self.in_features, self.out_features = reference.in_features, reference.out_features
        q, scale = quantize_int8(reference.weight)  # once, here: never per call
        self.register_buffer("weight_int8", q)
        self.register_buffer("weight_scale", scale)
        self.register_parameter("bias", reference.bias)
        self.quant_error = int8_error(reference.weight, q, scale)  # for NOTES.md
        none = torch.empty(0, dtype=reference.weight.dtype, device=q.device)
        bias = reference.bias if reference.bias is not None else none
        # plain attributes: the forward is one pybind call without nn.Module lookups
        self._args = (q, scale, bias, rows, warps, unroll)
        self._ref = (q, scale, reference.bias)
        self._dtype = reference.weight.dtype  # the kernels' activations (and bias) dtype
        self._fn = _load().int8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // self.in_features
        if not (x.is_cuda and x.dtype == self._dtype and 1 <= rows <= MAX_ROWS):
            return int8_w8a8_linear(x, *self._ref)  # the same numerics (torch._int_mm)
        return self._fn(x, *self._args)


def build(reference: nn.Module, rows: int = 0, warps: int = 0, unroll: int = 1) -> nn.Module:
    """``rows`` (1 or 2 channel tiles of 16 per block), ``warps`` (4 or 8: the k split; 0 for
    both: chosen per call from M) and ``unroll`` (1 or 2 chunks in flight) are tuning
    keywords for ``sweep_candidate``."""
    ok = (
        isinstance(reference, nn.Linear)
        and reference.weight.dtype in DTYPES
        and reference.weight.is_cuda
        and torch.cuda.get_device_capability(reference.weight.device) >= (7, 5)  # s8 mma.sync
        and reference.in_features % 64 == 0  # 64-wide k chunks
        and reference.out_features % 16 == 0  # 16 channels per tile
        and (reference.bias is None or reference.bias.dtype == reference.weight.dtype)
    )
    if not ok:
        return reference
    while rows > 1 and reference.out_features % (16 * rows):
        rows //= 2
    return Int8SkinnyLinear(reference, rows, warps, unroll)
