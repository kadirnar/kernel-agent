// Python binding of the example's host entry points.
#include <torch/extension.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mk_info", &ka_native::mk_info,
        "[pool pages, resident queues, page bytes, max GEMV slice] of the megakernel here",
        pybind11::arg("decode") = false);
  m.def("mk_run", &ka_native::mk_run, "one launch of the megakernel", pybind11::arg("program"),
        pybind11::arg("counters"), pybind11::arg("table"), pybind11::arg("status_ptr"),
        pybind11::arg("trace"), pybind11::arg("timeout_ns"), pybind11::arg("n_pages"),
        pybind11::arg("n_queues"), pybind11::arg("inflight") = 0,
        pybind11::arg("decode") = false);
  m.def("chain_pdl", &ka_native::chain_pdl, "baseline: one kernel per layer, PDL edges");
  m.def("coop_blocks", &ka_native::coop_blocks, "resident blocks of the cooperative baseline");
  m.def("chain_coop", &ka_native::chain_coop, "baseline: one cooperative kernel, grid barriers");
}
