// ka_launch.cuh: kernel launches with programmatic dependent launch (PDL) and cooperative
// grids, for CUDA C++ candidates built with torch.utils.cpp_extension.load_inline (#147).
//
//   load_inline(..., extra_include_paths=[str(kernel_agent.concurrency.include_dir())])
//   #include "ka_launch.cuh"
//
// Host side:
//
//   KaLaunch opt;
//   opt.pdl = true;  // may start while the previous kernel on the stream drains
//   C10_CUDA_CHECK(ka_launch(my_kernel, grid, block, smem, stream, opt, arg0, arg1));
//
// Device side, in a kernel launched with opt.pdl (the PDL idiom): first issue the work that
// does not depend on the previous kernel (weight loads, address math, shared-memory
// staging of constants), then
//
//   ka_pdl_launch_dependents();  // the next kernel on the stream may start launching
//   ka_pdl_wait();               // the previous kernel finished, its writes are visible
//
// and only then read what the previous kernel wrote. The wait waits for the whole previous
// grid (its completion and memory flush), not only for its launch_dependents, so writing
// after the wait is safe even to buffers the previous kernel read. Calling them in a kernel
// launched without PDL (or before sm_90) is a no-op.
//
// Where it pays (docs/PARALLEL.md §4.6, RTX 5070 Ti): a CUDA graph of 28 dependent GEMVs of
// 2 MB weights each, 101.2 us -> 77.7 us with PDL edges (the DRAM floor is 76.7 us): every
// graph kernel boundary costs ~0.9 us of drain + launch + ramp-up, PDL hides it when the
// kernel's prologue is its weight load. Eager launches gain little (host-bound). A wait
// placed before the independent loads hides nothing; cuBLAS / cuBLASLt kernels between two
// of yours do not take part.
//
// Cooperative launches (opt.cooperative = true, grid-wide sync) need every block resident:
// size the grid by ka_coresident_blocks(kernel, threads, smem, stream), which counts the SMs
// of the stream's green-context partition (ka_sm_count). cudaDevAttrMultiProcessorCount
// reports the whole device even inside a partition, and a cooperative grid sized from it is
// rejected there ("too many blocks in cooperative launch").
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <utility>

struct KaLaunch {
  bool pdl = false;          // cudaLaunchAttributeProgrammaticStreamSerialization (sm_90+)
  bool cooperative = false;  // cudaLaunchAttributeCooperative (grid-wide synchronisation)
};

// ---------------------------------------------------------------- device side

__device__ __forceinline__ void ka_pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ void ka_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// ---------------------------------------------------------------- host side

namespace ka_detail {

constexpr int kMaxDevices = 64;

inline int current_device() {
  int dev = 0;
  cudaGetDevice(&dev);
  return dev;
}

// Compute capability major of `dev`, cached (0 when unknown).
inline int cc_major(int dev) {
  static int cache[kMaxDevices] = {0};
  if (dev < 0 || dev >= kMaxDevices) return 0;
  if (cache[dev] == 0) {
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
      cudaGetLastError();
      return 0;
    }
    cache[dev] = major;
  }
  return cache[dev];
}

// A driver API function by name without linking libcuda, in the variant of the cuda.h this
// file is compiled against (its structs' layout); nullptr when the driver lacks it.
inline void* driver_fn(const char* name, unsigned int version = CUDA_VERSION) {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
  cudaError_t err = cudaGetDriverEntryPointByVersion(name, &fn, version, cudaEnableDefault, &status);
#else
  (void)version;
  cudaError_t err = cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault, &status);
#endif
  if (err != cudaSuccess || status != cudaDriverEntryPointSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  return fn;
}

}  // namespace ka_detail

// Whether PDL is available on the current device (sm_90 or newer).
inline bool ka_pdl_supported() { return ka_detail::cc_major(ka_detail::current_device()) >= 9; }

// SMs the kernels of `stream` can use: the SM count of the stream's green-context partition,
// else the device's multiprocessor count. Size persistent and cooperative grids by this.
inline int ka_sm_count(cudaStream_t stream) {
  const int dev = ka_detail::current_device();
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
    cudaGetLastError();
    sms = 0;
  }
#if CUDART_VERSION >= 12040
  using GetGreenCtx = CUresult (*)(CUstream, CUgreenCtx*);
  using GetResource = CUresult (*)(CUgreenCtx, CUdevResource*, CUdevResourceType);
  static GetGreenCtx get_green =
      reinterpret_cast<GetGreenCtx>(ka_detail::driver_fn("cuStreamGetGreenCtx"));
  static GetResource get_resource =
      reinterpret_cast<GetResource>(ka_detail::driver_fn("cuGreenCtxGetDevResource"));
  if (get_green != nullptr && get_resource != nullptr && stream != nullptr) {
    CUgreenCtx green = nullptr;
    if (get_green(reinterpret_cast<CUstream>(stream), &green) == CUDA_SUCCESS && green) {
      CUdevResource resource = {};
      if (get_resource(green, &resource, CU_DEV_RESOURCE_TYPE_SM) == CUDA_SUCCESS &&
          resource.sm.smCount > 0)
        return static_cast<int>(resource.sm.smCount);
    }
  }
#endif
  return sms;
}

// Blocks of `kernel` that can be resident at once on the SMs of `stream` (the largest grid a
// cooperative launch accepts there).
template <typename... KArgs>
inline int ka_coresident_blocks(void (*kernel)(KArgs...), int block_threads, size_t smem,
                                cudaStream_t stream) {
  int per_sm = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, block_threads, smem) !=
      cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return per_sm * ka_sm_count(stream);
}

// cudaLaunchKernelEx with the attributes of `opt`. PDL is dropped where the device does not
// have it (the launch is then an ordinary <<<>>> launch); a cooperative launch on a device
// without cooperative launches returns cudaErrorNotSupported. Returns the launch's error.
template <typename... KArgs, typename... Args>
inline cudaError_t ka_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem,
                             cudaStream_t stream, KaLaunch opt, Args&&... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attrs[2];
  unsigned int n = 0;
  if (opt.pdl && ka_pdl_supported()) {
    attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  if (opt.cooperative) {
    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, ka_detail::current_device());
    if (!coop) return cudaErrorNotSupported;
    attrs[n].id = cudaLaunchAttributeCooperative;
    attrs[n].val.cooperative = 1;
    ++n;
  }
  config.attrs = n ? attrs : nullptr;
  config.numAttrs = n;
  return cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
}
