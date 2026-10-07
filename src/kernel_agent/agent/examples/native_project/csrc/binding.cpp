// Python binding of the project's host entry points (one PYBIND11_MODULE per project).
#include <torch/extension.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &ka_native::rmsnorm, "RMSNorm (bf16 / fp16 / fp32)", pybind11::arg("x"),
        pybind11::arg("w"), pybind11::arg("eps"));
}
