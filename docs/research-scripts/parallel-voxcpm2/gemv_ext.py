"""bf16 GEMV chain kernels for the persistent-vs-launches microbenchmark (#136).

y = W x per layer, W [H, H] bf16 row-major, one warp per output row, weights streamed with
ld.global.nc.L1::no_allocate, fp32 accumulation. Variants:
* ``gemv(W, x, y, pdl)``: one launch per layer; with ``pdl`` the kernel loads its weights,
  lets the next grid launch (griddepcontrol.launch_dependents) and only then waits for the
  previous grid (griddepcontrol.wait) before it reads x (programmatic dependent launch).
* ``chain_persistent(W, buf, bar, L, prefetch, blocks)``: one cooperative launch for all L
  layers, a grid barrier between layers (``bar``: 3 int32, ``bar[2]`` set to 1 when a barrier
  waited > 2 s, i.e. the grid was not co-resident; the kernel then gives up instead of hanging); with ``prefetch`` every warp loads the weights of its
  first row of layer l+1 into registers before it waits at the barrier.
"""

from __future__ import annotations

import os

import torch

from kernel_agent import toolchain

toolchain.setup()  # nvcc / ninja as kernel-agent finds them, before cpp_extension reads CUDA_HOME
import torch.utils.cpp_extension as _ce  # noqa: E402

if _ce.CUDA_HOME is None and os.environ.get("CUDA_HOME"):
    _ce.CUDA_HOME = os.environ["CUDA_HOME"]
load_inline = _ce.load_inline

CUDA = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ uint4 ldw(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

template <int KI>
__device__ __forceinline__ void load_row(const bf16* __restrict__ wr, int lane, uint4 (&w)[KI]) {
#pragma unroll
  for (int i = 0; i < KI; ++i) w[i] = ldw(wr + (size_t)(i * 32 + lane) * 8);
}

template <int KI>
__device__ __forceinline__ float dot_row(const uint4 (&w)[KI], const bf16* __restrict__ x, int lane) {
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < KI; ++i) {
    uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)(i * 32 + lane) * 8);
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w[i]);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 a = __bfloat1622float2(wp[j]), b = __bfloat1622float2(xp[j]);
      acc = fmaf(a.x, b.x, acc);
      acc = fmaf(a.y, b.y, acc);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  return acc;
}

template <int KI, bool PDL>
__global__ void __launch_bounds__(256) gemv_kernel(const bf16* __restrict__ W, const bf16* x,
                                                   bf16* y, int N) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  uint4 w[KI];
  if (row < N) load_row<KI>(W + (size_t)row * KI * 256, lane, w);
  if (PDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  if (row >= N) return;
  float acc = dot_row<KI>(w, x, lane);
  if (lane == 0) y[row] = __float2bfloat16(acc);
}

__device__ __forceinline__ void grid_sync(unsigned* bar, unsigned nb) {
  __syncthreads();
  if (threadIdx.x == 0 && *(volatile unsigned*)(bar + 2) == 0u) {  // after a timeout: skip
    volatile unsigned* gen = bar + 1;
    unsigned g = *gen;
    __threadfence();
    if (atomicAdd(bar, 1u) == nb - 1) {
      bar[0] = 0;
      __threadfence();
      atomicAdd(bar + 1, 1u);
    } else {
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      while (*gen == g) {
        __nanosleep(32);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) { atomicExch(bar + 2, 1u); break; }  // not co-resident
      }
    }
    __threadfence();
  }
  __syncthreads();
}

template <int KI, bool PF>
__global__ void __launch_bounds__(256) chain_kernel(const bf16* __restrict__ W, bf16* buf, int N,
                                                    int L, unsigned* bar) {
  const int lane = threadIdx.x & 31;
  const int nwarps = gridDim.x * 8;
  const int gw = blockIdx.x * 8 + (threadIdx.x >> 5);
  const size_t layer = (size_t)N * KI * 256;
  uint4 wn[KI];
  bool have = false;
  for (int l = 0; l < L; ++l) {
    const bf16* x = buf + (size_t)(l & 1) * N;
    bf16* y = buf + (size_t)((l + 1) & 1) * N;
    const bf16* Wl = W + (size_t)l * layer;
    for (int row = gw; row < N; row += nwarps) {
      uint4 w[KI];
      if (PF && have && row == gw) {
#pragma unroll
        for (int i = 0; i < KI; ++i) w[i] = wn[i];
      } else {
        load_row<KI>(Wl + (size_t)row * KI * 256, lane, w);
      }
      float acc = dot_row<KI>(w, x, lane);
      if (lane == 0) y[row] = __float2bfloat16(acc);
    }
    have = false;
    if (PF && l + 1 < L && gw < N) {
      load_row<KI>(Wl + layer + (size_t)gw * KI * 256, lane, wn);
      have = true;
    }
    if (l + 1 < L) grid_sync(bar, gridDim.x);
  }
}

template <int KI>
void launch_gemv(const bf16* W, const bf16* x, bf16* y, int N, bool pdl, cudaStream_t s) {
  dim3 grid((N + 7) / 8), block(256);
  if (!pdl) {
    gemv_kernel<KI, false><<<grid, block, 0, s>>>(W, x, y, N);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0; cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, gemv_kernel<KI, true>, W, x, y, N));
}

void gemv(torch::Tensor W, torch::Tensor x, torch::Tensor y, bool pdl) {
  int N = W.size(0), K = W.size(1);
  auto s = at::cuda::getCurrentCUDAStream();
  auto pw = (const bf16*)W.data_ptr(); auto px = (const bf16*)x.data_ptr(); auto py = (bf16*)y.data_ptr();
  if (K == 1024) launch_gemv<4>(pw, px, py, N, pdl, s);
  else if (K == 2048) launch_gemv<8>(pw, px, py, N, pdl, s);
  else TORCH_CHECK(false, "K must be 1024 or 2048");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int KI, bool PF>
int launch_chain(const bf16* W, bf16* buf, int N, int L, unsigned* bar, int blocks, cudaStream_t s) {
  auto kern = chain_kernel<KI, PF>;
  int per_sm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 256, 0);
  int maxb = per_sm * prop.multiProcessorCount;
  if (blocks <= 0 || blocks > maxb) blocks = maxb;
  void* args[] = {(void*)&W, (void*)&buf, (void*)&N, (void*)&L, (void*)&bar};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kern, dim3(blocks), dim3(256), args, 0, s));
  return blocks;
}

int chain_persistent(torch::Tensor W, torch::Tensor buf, torch::Tensor bar, int64_t L, bool pf,
                     int64_t blocks) {
  int N = W.size(1), K = W.size(2);
  auto s = at::cuda::getCurrentCUDAStream();
  auto pw = (const bf16*)W.data_ptr(); auto pb = (bf16*)buf.data_ptr();
  auto pbar = (unsigned*)bar.data_ptr();
  if (K == 1024) return pf ? launch_chain<4, true>(pw, pb, N, L, pbar, blocks, s)
                           : launch_chain<4, false>(pw, pb, N, L, pbar, blocks, s);
  if (K == 2048) return pf ? launch_chain<8, true>(pw, pb, N, L, pbar, blocks, s)
                           : launch_chain<8, false>(pw, pb, N, L, pbar, blocks, s);
  TORCH_CHECK(false, "K must be 1024 or 2048");
  return 0;
}
"""

CPP = """
void gemv(torch::Tensor W, torch::Tensor x, torch::Tensor y, bool pdl);
int chain_persistent(torch::Tensor W, torch::Tensor buf, torch::Tensor bar, int64_t L, bool pf, int64_t blocks);
"""

_ext = None


def ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            "ka136_gemv_chain",
            cpp_sources=CPP,
            cuda_sources=CUDA,
            functions=["gemv", "chain_persistent"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
            verbose=False,
        )
    return _ext


def chain_ref(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    for l in range(W.shape[0]):
        x = (W[l].float() @ x.float()).to(torch.bfloat16)
    return x
