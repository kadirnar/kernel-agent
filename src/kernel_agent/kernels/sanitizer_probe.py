"""The programs :func:`kernel_agent.kernels.memcheck.selftest` runs under ``--tool racecheck``
and ``--tool synccheck`` (``kernel-agent doctor``, issue #225): CUDA C++ (:data:`SRC`)
compiled with NVRTC for this GPU (:func:`kernel_agent.toolchain.nvrtc_target`), each kernel
one block, a clean kernel first and then its deliberate hazard:

* ``race``: ``ka_ordered`` (two warps exchange shared-memory words behind a barrier), then
  ``ka_race`` (the same exchange without the barrier: a read-after-write hazard between the
  warps, which racecheck must report);
* ``sync``: ``ka_uniform`` (a barrier every thread reaches), then ``ka_divergent`` (a
  ``__syncthreads()`` only half of the warp reaches, which synccheck must report; the other
  half goes on and exits, so the barrier still completes without a sanitizer).

The clean kernels' outputs are checked here (a sanitizer that breaks a correct kernel is not
usable either); a report naming a clean kernel fails the self-test.

    python -m kernel_agent.kernels.sanitizer_probe race|sync
"""

from __future__ import annotations

import json
import os
import sys

from kernel_agent.kernels.memcheck import RESULT_MARKER

#: The probe's kernels: ``KERNELS[mode]`` = (clean kernel, its hazard, threads per block).
KERNELS = {"race": ("ka_ordered", "ka_race", 64), "sync": ("ka_uniform", "ka_divergent", 32)}
#: The file name the sanitizer's reports give (NVRTC's program name, with line info).
SOURCE_NAME = "ka_sanitizer_probe.cu"
SRC = r"""
extern "C" __global__ void ka_ordered(int* out) {
  __shared__ int s[64];
  const int t = threadIdx.x;
  s[t] = t;
  __syncthreads();
  out[t] = s[63 - t];  // the other warp's word, after the barrier
}

extern "C" __global__ void ka_race(int* out) {
  __shared__ int s[64];
  const int t = threadIdx.x;
  s[t] = t;
  out[t] = s[63 - t];  // no barrier: the other warp may not have written it yet
}

extern "C" __global__ void ka_uniform(int* out) {
  __shared__ int s[32];
  const int t = threadIdx.x;
  s[t] = t;
  __syncthreads();  // every thread of the block
  out[t] = s[31 - t];
}

extern "C" __global__ void ka_divergent(int* out) {
  __shared__ int s[32];
  const int t = threadIdx.x;
  s[t] = t;
  if (t < 16) __syncthreads();  // half of the warp: a divergent barrier
  out[t] = s[t];
}
"""


def run(mode: str) -> dict[str, object]:
    """Compile :data:`SRC`, run ``mode``'s clean kernel and then its hazard (one block each)."""
    import torch
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

    from kernel_agent import toolchain

    clean, hazard, threads = KERNELS[mode]
    index = torch.cuda.current_device()
    dev = Device(index)
    dev.set_current()
    arch, kind = toolchain.nvrtc_target(torch.cuda.get_device_capability(index))
    options = ProgramOptions(arch=arch, std="c++17", lineinfo=True, name=SOURCE_NAME)
    code = Program(SRC, code_type="c++", options=options).compile(kind)
    stream = dev.create_stream(torch.cuda.current_stream())
    out = torch.full((threads,), -1, device="cuda", dtype=torch.int32)
    config = LaunchConfig(grid=1, block=threads)
    launch(stream, config, code.get_kernel(clean), out.data_ptr())
    torch.cuda.synchronize()
    want = torch.arange(threads - 1, -1, -1, device="cuda", dtype=torch.int32)
    result: dict[str, object] = {"status": "ok", "clean": bool(torch.equal(out, want))}
    try:
        launch(stream, config, code.get_kernel(hazard), out.data_ptr())
        torch.cuda.synchronize()
    except Exception as exc:  # the sanitizer may stop the kernel: its report is the result
        result["hazard_error"] = repr(exc)[:300]
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    mode = args[0] if args else ""
    tag = sys.stdin.readline().strip()
    result: dict[str, object]
    if mode not in KERNELS:
        result = {"status": "error", "error": f"mode {mode!r}: one of {', '.join(KERNELS)}"}
    else:
        try:
            result = run(mode)
        except Exception as exc:
            result = {"status": "error", "error": repr(exc)[:500]}
    sys.stdout.write(f"{RESULT_MARKER}{tag}@@" + json.dumps(result) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
