// The example's megakernels: kernel-agent's interpreter (ka_mk.cuh, on every project's include
// path) instantiated with the opcodes of include/mk_ops.cuh (mk::Ops: the chain) and with the
// decode-step opcodes too (mk::DecodeOps, include/mk_decode.cuh), and its host entry points.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "mk_ops.cuh"
#include "ops.h"

namespace ka_native {

namespace {

template <class O>
std::vector<int64_t> info() {
  const int pages = ka_mk::pool_pages<O>();
  TORCH_CHECK(pages > 0, "mk_info: not one ", O::kPageBytes, "-byte page fits shared memory");
  const int queues = ka_mk::queues<O>(pages, at::cuda::getCurrentCUDAStream());
  return {pages, queues, O::kPageBytes, mk::kMaxSlice};
}

}  // namespace

std::vector<int64_t> mk_info(bool decode) {
  return decode ? info<mk::DecodeOps>() : info<mk::Ops>();
}

void mk_run(const at::Tensor& program, const at::Tensor& counters, const at::Tensor& table,
            int64_t status_ptr, const std::optional<at::Tensor>& trace, int64_t timeout_ns,
            int64_t n_pages, int64_t n_queues, int64_t inflight, bool decode) {
  TORCH_CHECK(program.is_cuda() && program.scalar_type() == at::kInt, "mk_run: int32 program");
  TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == at::kInt, "mk_run: int32 counters");
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kLong, "mk_run: int64 tensor table");
  ka_mk::Params p;
  p.program = program.data_ptr<int>();
  p.counters = counters.data_ptr<int>();
  p.tensors = reinterpret_cast<const unsigned long long*>(table.data_ptr<int64_t>());
  p.n_tensors = static_cast<int>(table.numel());
  p.status = reinterpret_cast<int*>(status_ptr);
  p.trace = trace.has_value() ? reinterpret_cast<long long*>(trace->data_ptr<int64_t>()) : nullptr;
  p.timeout_ns = timeout_ns;
  p.n_pages = static_cast<int>(n_pages);
  p.inflight = static_cast<int>(inflight);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int queues = static_cast<int>(n_queues);
  C10_CUDA_CHECK(decode ? ka_mk::launch<mk::DecodeOps>(p, queues, stream)
                        : ka_mk::launch<mk::Ops>(p, queues, stream));
}

}  // namespace ka_native
