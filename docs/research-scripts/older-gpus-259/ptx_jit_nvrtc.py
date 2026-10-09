"""Run older-arch NVRTC code on this GPU through the driver's PTX JIT (issue #259). GPU.

The kernel is compiled to PTX for compute_75 / 80 / 86 / 89 and loaded with
``ObjectCode.from_ptx``: the driver JIT-compiles it for the real GPU, and the kernel takes
the older arch's code path (``__CUDA_ARCH__``). It records the arch it saw, whether the PTX
has the hardware e4m3 conversion (``cvt.rn.f16x2.e4m3x2``, sm_89+) and whether CUDA's
``__nv_cvt_fp8_to_halfraw`` (software routine below sm_89) equals torch's e4m3 decoding on
all 256 codes (NaN codes compared as NaN). Correctness only: the SASS is this GPU's. Run:

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 PYTHONPATH=src \
        python docs/research-scripts/older-gpus-259/ptx_jit_nvrtc.py \
        > docs/research-scripts/older-gpus-259/ptx_jit_nvrtc.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch  # noqa: E402
from cuda.core import Device, LaunchConfig, ObjectCode, Program, ProgramOptions, launch  # noqa: E402

from kernel_agent.toolchain import cuda_include_dirs  # noqa: E402

SRC = r"""
#include <cuda_fp8.h>
#include <cuda_fp16.h>
extern "C" __global__ void probe(int* arch, float* out, const unsigned char* codes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef __CUDA_ARCH__
  if (i == 0) arch[0] = __CUDA_ARCH__;
#endif
  __half_raw h = __nv_cvt_fp8_to_halfraw(codes[i], __NV_E4M3);
  out[i] = __half2float(__half(h));
}
"""


def main() -> None:
    dev = Device(torch.cuda.current_device())
    dev.set_current()
    stream = dev.create_stream(torch.cuda.current_stream())
    print(f"# GPU: {torch.cuda.get_device_name()} sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
    codes = torch.arange(256, dtype=torch.uint8, device="cuda")
    want = codes.view(torch.float8_e4m3fn).float()
    for arch in ("compute_75", "compute_80", "compute_86", "compute_89", "sm_120"):
        options = ProgramOptions(arch=arch, std="c++17", include_path=cuda_include_dirs())
        program = Program(SRC, code_type="c++", options=options)
        kind = "cubin" if arch.startswith("sm_") else "ptx"
        obj = program.compile(kind)
        ptx = obj.code.decode() if kind == "ptx" else ""
        if kind == "ptx":
            obj = ObjectCode.from_ptx(obj.code)  # the driver JIT-compiles it for this GPU
        kernel = obj.get_kernel("probe")
        seen = torch.zeros(1, dtype=torch.int32, device="cuda")
        out = torch.zeros(256, device="cuda")
        launch(stream, LaunchConfig(grid=2, block=128), kernel, seen.data_ptr(), out.data_ptr(), codes.data_ptr())
        torch.cuda.synchronize()
        equal = torch.equal(out.nan_to_num(nan=12345.0), want.nan_to_num(nan=12345.0))
        hw = "cvt.rn.f16x2.e4m3x2" in ptx if kind == "ptx" else "n/a (cubin)"
        print(
            f"{arch:11s} ran: __CUDA_ARCH__={int(seen.item())}; hardware e4m3 cvt in PTX: {hw}; "
            f"256 e4m3 codes equal torch's decoding: {equal}"
        )


if __name__ == "__main__":
    main()
