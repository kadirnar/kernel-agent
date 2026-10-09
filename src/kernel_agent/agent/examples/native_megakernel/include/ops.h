// Host entry points of the example (implemented in csrc/*.cu, bound in csrc/binding.cpp).
#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <vector>

namespace ka_native {

// The megakernel (csrc/megakernel.cu): pool pages and resident queues on this GPU, one launch.
std::vector<int64_t> mk_info();
void mk_run(const at::Tensor& program, const at::Tensor& counters, const at::Tensor& table,
            int64_t status_ptr, const std::optional<at::Tensor>& trace, int64_t timeout_ns,
            int64_t n_pages, int64_t n_queues, int64_t inflight);

// The baselines from the same math (csrc/baselines.cu): one kernel per layer with PDL edges,
// and one cooperative kernel with a grid barrier per layer.
void chain_pdl(const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& gammas,
               const at::Tensor& buf, double eps, bool pdl);
int64_t coop_blocks(int64_t n);
void chain_coop(const at::Tensor& wtable, const at::Tensor& gtable, const at::Tensor& buf,
                const at::Tensor& bar, int64_t layers, double eps, int64_t grid);

}  // namespace ka_native
