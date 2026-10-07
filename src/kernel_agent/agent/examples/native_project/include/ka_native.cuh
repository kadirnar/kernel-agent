// Device and launch helpers shared by the .cu files of a native project.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <utility>

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

// ------------------------------------------------------------------ programmatic dependent launch
// A kernel launched with launch_pdl() may start while the previous kernel of the stream is
// still running: issue the loads that do not depend on it (weights) first, then pdl_wait()
// before reading the previous kernel's output; pdl_launch_dependents() lets the next kernel
// start its own prologue. Both are no-ops without PDL (and before sm_90).

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <typename... Params, typename... Args>
inline cudaError_t launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem,
                              cudaStream_t stream, Args&&... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
}

}  // namespace ka
