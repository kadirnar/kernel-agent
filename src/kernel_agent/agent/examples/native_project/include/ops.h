// Host entry points of the project (implemented in csrc/*.cu, bound in csrc/binding.cpp).
#pragma once

#include <ATen/ATen.h>

namespace ka_native {

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps);

}  // namespace ka_native
