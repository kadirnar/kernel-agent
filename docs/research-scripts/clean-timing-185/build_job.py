"""A simulated agent's nvcc build, like a load_inline candidate: main.cpp + one .cu (both
include torch/extension.h), compiled by ninja (MAX_JOBS); ``variants`` > 1 builds that many
variants at once in separate processes, as agents do when they compare variants."""

import subprocess
import sys
import time
from pathlib import Path

root, name, variants = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if variants > 1:
    procs = [
        subprocess.Popen([sys.executable, __file__, str(root), f"{name}_v{i}", "1"])
        for i in range(variants)
    ]
    sys.exit(max(p.wait() for p in procs))

from torch.utils.cpp_extension import load  # noqa: E402

src = root / name
src.mkdir(parents=True, exist_ok=True)
(src / "k.cu").write_text(
    """#include <torch/extension.h>
#include <cuda_bf16.h>
template <int N> __global__ void k(float* x, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  #pragma unroll
  for (int r = 0; r < N; ++r) acc += __sinf(x[(j + r) % n]) * r;
  if (j < n) x[j] = acc;
}
void f(torch::Tensor x) {
  int n = x.numel();
  k<16><<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), n);
  k<32><<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), n);
}
"""
)
(src / "main.cpp").write_text(
    '#include <torch/extension.h>\nvoid f(torch::Tensor x);\n'
    'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("f", &f); }\n'
)
start = time.perf_counter()
load(name=name, sources=[str(src / "main.cpp"), str(src / "k.cu")], build_directory=str(src))
print(f"built {name} in {time.perf_counter() - start:.1f}s", flush=True)
