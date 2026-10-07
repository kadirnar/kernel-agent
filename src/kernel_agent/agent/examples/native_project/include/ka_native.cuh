// Device helpers shared by the .cu files of a native project.
//
// Launch helpers come from kernel-agent's toolkit header, on the include path of every
// project build: `#include "ka_launch.cuh"` for programmatic dependent launch (PDL) and
// cooperative grids (`ka_launch(kernel, grid, block, smem, stream, opt, args...)`,
// `ka_pdl_wait()`, `ka_pdl_launch_dependents()`, `ka_coresident_blocks(...)`).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ka {

// ------------------------------------------------------------------ conversions

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f(float v) { return v; }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_f(float v) { return __float2half(v); }
template <> __device__ __forceinline__ float from_f(float v) { return v; }

// ------------------------------------------------------------------ reductions

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

}  // namespace ka
