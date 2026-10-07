// Warp-per-row RMSNorm with the reference's rounding: normalise in fp32, round to the input
// dtype, then multiply by the weight and round again (as `weight * x.to(dtype)` does).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "ka_native.cuh"
#include "ops.h"

namespace {

template <typename T>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                               T* __restrict__ y, int rows, int cols, float eps) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x % 32;
  if (warp >= rows) return;
  const T* xr = x + (size_t)warp * cols;
  T* yr = y + (size_t)warp * cols;
  float acc = 0.f;
  for (int c = lane; c < cols; c += 32) {
    float v = ka::to_f(xr[c]);
    acc += v * v;
  }
  float inv = rsqrtf(ka::warp_sum(acc) / cols + eps);
  for (int c = lane; c < cols; c += 32) {
    T normed = ka::from_f<T>(ka::to_f(xr[c]) * inv);
    yr[c] = ka::from_f<T>(ka::to_f(normed) * ka::to_f(w[c]));
  }
}

}  // namespace

namespace ka_native {

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "rmsnorm: CUDA tensors expected");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "rmsnorm: x and w must share a dtype");
  auto x2 = x.contiguous();
  auto w2 = w.contiguous();
  auto y = at::empty_like(x2);
  int cols = x2.size(-1);
  TORCH_CHECK(w2.numel() == cols, "rmsnorm: weight size ", w2.numel(), " != ", cols);
  int rows = cols ? x2.numel() / cols : 0;
  if (rows == 0) return y;
  int threads = 256;
  int blocks = (rows * 32 + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto tag) {
    using T = decltype(tag);
    rmsnorm_kernel<T><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const T*>(x2.data_ptr()), reinterpret_cast<const T*>(w2.data_ptr()),
        reinterpret_cast<T*>(y.data_ptr()), rows, cols, static_cast<float>(eps));
  };
  switch (x2.scalar_type()) {
    case at::kBFloat16: launch(__nv_bfloat16{}); break;
    case at::kHalf: launch(__half{}); break;
    case at::kFloat: launch(float{}); break;
    default: TORCH_CHECK(false, "rmsnorm: unsupported dtype ", x2.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace ka_native
