// The two baselines of the example, from the same math as the megakernel's GEMV opcode
// (RMSNorm fused into the projection's prologue, the residual into its epilogue):
//
// * chain_pdl: one kernel per layer, one warp per output row; the row's weights (and the
//   norm's gamma) load into registers before ka_pdl_wait(), so with PDL edges in a CUDA graph
//   layer l + 1 streams its weights while layer l drains (docs/PARALLEL.md §4.6: 77.7 us for
//   28 layers of [1024, 1024] on an RTX 5070 Ti, the DRAM floor 76.7 us);
// * chain_coop: one cooperative persistent kernel, a hand-rolled grid barrier after every
//   layer (§4.6: 136.7 us, ~2.2-2.4 us per barrier).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "mk_ops.cuh"
#include "ops.h"

namespace {

using mk::bf16;

// 128-bit weight load that bypasses L1: every weight byte is read once
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// y = x + W rmsnorm(x) for one layer, n = 256 * KI; lane l of the row's warp holds columns
// [(i * 32 + l) * 8, +8) for i < KI, so the warp alone covers the norm's sum of squares.
template <int KI>
__global__ void __launch_bounds__(256) layer_pdl(const bf16* __restrict__ w,
                                                 const bf16* __restrict__ g, const bf16* x,
                                                 bf16* y, int n, float eps) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  uint4 wr[KI], gr[KI];
  if (row < n) {
    const bf16* p = w + static_cast<size_t>(row) * n;
#pragma unroll
    for (int i = 0; i < KI; ++i) wr[i] = ld_stream(p + (i * 32 + lane) * 8);
  }
#pragma unroll
  for (int i = 0; i < KI; ++i) gr[i] = __ldg(reinterpret_cast<const uint4*>(g + (i * 32 + lane) * 8));
  ka_pdl_launch_dependents();  // the next layer may launch and start loading its weights
  ka_pdl_wait();               // the previous layer finished: x is complete and visible
  if (row >= n) return;
  uint4 xr[KI];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < KI; ++i) {
    xr[i] = *reinterpret_cast<const uint4*>(x + (i * 32 + lane) * 8);
    ss += mk::sumsq8(xr[i]);
  }
  const float inv = rsqrtf(mk::warp_sum(ss) / n + eps);
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < KI; ++i) acc = mk::dot8(wr[i], mk::norm8(xr[i], gr[i], inv), acc);
  acc = mk::warp_sum(acc);
  if (lane == 0)
    y[row] = __float2bfloat16(__bfloat162float(x[row]) +
                              __bfloat162float(__float2bfloat16(acc)));
}

// bar[0]: arrivals, bar[1]: generation. The last block to arrive resets the arrivals and
// releases the next generation.
__device__ __forceinline__ void grid_barrier(int* bar) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const int gen = ka_mk::ld_acquire(bar + 1);
    if (ka_mk::atom_add_acq_rel(bar, 1) == static_cast<int>(gridDim.x) - 1) {
      bar[0] = 0;
      ka_mk::red_release(bar + 1, 1);
    } else {
      while (ka_mk::ld_acquire(bar + 1) == gen) {
      }
    }
  }
  __syncthreads();
}

template <int KI>
__global__ void __launch_bounds__(256) chain_coop_kernel(const unsigned long long* wt,
                                                         const unsigned long long* gt, bf16* buf,
                                                         int layers, int n, float eps, int* bar) {
  const int lane = threadIdx.x & 31;
  const int warps = gridDim.x * 8;
  const int first = blockIdx.x * 8 + (threadIdx.x >> 5);
  for (int l = 0; l < layers; ++l) {
    const bf16* w = reinterpret_cast<const bf16*>(wt[l]);
    const bf16* g = reinterpret_cast<const bf16*>(gt[l]);
    const bf16* x = buf + static_cast<size_t>(l) * n;
    bf16* y = buf + static_cast<size_t>(l + 1) * n;
    uint4 hr[KI];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < KI; ++i) {
      hr[i] = __ldcg(reinterpret_cast<const uint4*>(x + (i * 32 + lane) * 8));
      ss += mk::sumsq8(hr[i]);
    }
    const float inv = rsqrtf(mk::warp_sum(ss) / n + eps);
#pragma unroll
    for (int i = 0; i < KI; ++i)
      hr[i] = mk::norm8(hr[i], __ldg(reinterpret_cast<const uint4*>(g + (i * 32 + lane) * 8)), inv);
    for (int row = first; row < n; row += warps) {
      const bf16* p = w + static_cast<size_t>(row) * n;
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < KI; ++i) acc = mk::dot8(ld_stream(p + (i * 32 + lane) * 8), hr[i], acc);
      acc = mk::warp_sum(acc);
      if (lane == 0)
        y[row] = __float2bfloat16(__bfloat162float(__ldcg(x + row)) +
                                  __bfloat162float(__float2bfloat16(acc)));
    }
    grid_barrier(bar);
  }
}

#define KA_CASES(X) \
  X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15) X(16)

template <typename F>
void dispatch(int n, F&& f) {
  TORCH_CHECK(n % 256 == 0 && n >= 256 && n <= 4096, "chain: n must be 256 * k, k <= 16");
  switch (n / 256) {
#define KA_CASE(K) \
  case K: f(std::integral_constant<int, K>{}); break;
    KA_CASES(KA_CASE)
#undef KA_CASE
    default: break;
  }
}

}  // namespace

namespace ka_native {

void chain_pdl(const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& gammas,
               const at::Tensor& buf, double eps, bool pdl) {
  TORCH_CHECK(buf.dim() == 2 && buf.is_contiguous() && buf.scalar_type() == at::kBFloat16 &&
                  buf.size(0) == static_cast<int64_t>(weights.size()) + 1,
              "chain_pdl: buf must be a contiguous bf16 [layers + 1, n]");
  TORCH_CHECK(gammas.size() == weights.size(), "chain_pdl: one gamma per layer");
  const int n = static_cast<int>(buf.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bf16* b = reinterpret_cast<bf16*>(buf.data_ptr());
  KaLaunch opt;
  opt.pdl = pdl;
  for (size_t l = 0; l < weights.size(); ++l) {
    const bf16* w = reinterpret_cast<const bf16*>(weights[l].data_ptr());
    const bf16* g = reinterpret_cast<const bf16*>(gammas[l].data_ptr());
    const bf16* x = b + l * static_cast<size_t>(n);
    bf16* y = b + (l + 1) * static_cast<size_t>(n);
    cudaError_t err = cudaSuccess;
    dispatch(n, [&](auto ki) {
      err = ka_launch(layer_pdl<decltype(ki)::value>, dim3((n + 7) / 8), dim3(256), 0, stream, opt,
                      w, g, x, y, n, static_cast<float>(eps));
    });
    C10_CUDA_CHECK(err);
  }
}

int64_t coop_blocks(int64_t n) {
  int blocks = 0;
  dispatch(static_cast<int>(n), [&](auto ki) {
    blocks = ka_coresident_blocks(chain_coop_kernel<decltype(ki)::value>, 256, 0,
                                  at::cuda::getCurrentCUDAStream());
  });
  return blocks;
}

void chain_coop(const at::Tensor& wtable, const at::Tensor& gtable, const at::Tensor& buf,
                const at::Tensor& bar, int64_t layers, double eps, int64_t grid) {
  TORCH_CHECK(bar.scalar_type() == at::kInt && bar.numel() >= 2, "chain_coop: int32 bar[2]");
  const int n = static_cast<int>(buf.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KaLaunch opt;
  opt.cooperative = true;
  cudaError_t err = cudaSuccess;
  dispatch(n, [&](auto ki) {
    err = ka_launch(chain_coop_kernel<decltype(ki)::value>, dim3(static_cast<int>(grid)), dim3(256),
                    0, stream, opt,
                    reinterpret_cast<const unsigned long long*>(wtable.data_ptr<int64_t>()),
                    reinterpret_cast<const unsigned long long*>(gtable.data_ptr<int64_t>()),
                    reinterpret_cast<bf16*>(buf.data_ptr()), static_cast<int>(layers), n,
                    static_cast<float>(eps), bar.data_ptr<int>());
  });
  C10_CUDA_CHECK(err);
}

}  // namespace ka_native
