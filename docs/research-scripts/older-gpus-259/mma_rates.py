"""Tensor-core ``mma.sync`` rates per instruction form on this GPU (issue #259). GPU.

The register-only rate kernel of ``kernel_agent.kernels.mma_peaks`` (``source``: CHAINS
independent accumulator chains per warp, 4 blocks of 256 threads per SM, best of TRIALS
CUDA-event timings) run for the forms the older families use: the Turing shapes
(m16n8k8 fp16, m8n8k16 s8), fp16 with fp32 vs fp16 accumulation, TF32, bf16, s8 m16n8k32
and the e4m3 forms. FLOPs per instruction = 2 m n k (integer forms in TOPS). On an older GPU
``kernel-agent doctor --remeasure-peaks`` measures the forms ``mma_peaks`` knows; this
script measures every form here. Run:

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 PYTHONPATH=src \
        python docs/research-scripts/older-gpus-259/mma_rates.py \
        > docs/research-scripts/older-gpus-259/mma_rates.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch  # noqa: E402

from kernel_agent.kernels import mma_peaks as mp  # noqa: E402

F4 = '"+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])'
H2 = '"+r"(h[c][0]), "+r"(h[c][1])'
I4 = '"+r"(n[c][0]), "+r"(n[c][1]), "+r"(n[c][2]), "+r"(n[c][3])'
I2 = '"+r"(n[c][0]), "+r"(n[c][1])'
A = ['"r"(a0)', '"r"(a1)', '"r"(a2)', '"r"(a3)']
B = ['"r"(b0)', '"r"(b1)']


def body(ptx: str, out: str, n_out: int, n_a: int, n_b: int) -> str:
    regs, k = [], 0
    for n in (n_out, n_a, n_b):
        regs.append("{" + ",".join(f"%{k + j}" for j in range(n)) + "}")
        k += n
    regs.append("{" + ",".join(f"%{j}" for j in range(n_out)) + "}")
    return f'asm volatile("{ptx} {",".join(regs)};" : {out} : {", ".join(A[:n_a] + B[:n_b])});'


#: label, (m, n, k), minimum cc, body, integer
FORMS = (
    ("f16 m16n8k16 fp32 acc", (16, 8, 16), 80,
     body("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", F4, 4, 4, 2), False),
    ("f16 m16n8k16 fp16 acc", (16, 8, 16), 80,
     body("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", H2, 2, 4, 2), False),
    ("bf16 m16n8k16 fp32 acc", (16, 8, 16), 80,
     body("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", F4, 4, 4, 2), False),
    ("tf32 m16n8k8 fp32 acc", (16, 8, 8), 80,
     body("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", F4, 4, 4, 2), False),
    ("f16 m16n8k8 fp32 acc (Turing form)", (16, 8, 8), 75,
     body("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32", F4, 4, 2, 1), False),
    ("f16 m16n8k8 fp16 acc (Turing form)", (16, 8, 8), 75,
     body("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16", H2, 2, 2, 1), False),
    ("s8 m16n8k32 s32 acc", (16, 8, 32), 80,
     body("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", I4, 4, 4, 2), True),
    ("s8 m8n8k16 s32 acc (Turing form)", (8, 8, 16), 75,
     body("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32", I2, 2, 1, 1), True),
    ("e4m3 m16n8k32 fp32 acc", (16, 8, 32), 89,
     body("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", F4, 4, 4, 2), False),
    ("e4m3 m16n8k32 fp16 acc", (16, 8, 32), 89,
     body("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", H2, 2, 4, 2), False),
)


def main() -> None:
    index = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(index)
    cc = cap[0] * 10 + cap[1]
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    dev = Device(index)
    dev.set_current()
    stream = dev.create_stream(torch.cuda.current_stream())
    out = torch.empty(sms * mp.BLOCKS_PER_SM * mp.THREADS, device="cuda")
    config = LaunchConfig(grid=sms * mp.BLOCKS_PER_SM, block=mp.THREADS)
    warps = sms * mp.BLOCKS_PER_SM * mp.THREADS / 32
    print(f"# {torch.cuda.get_device_name(index)} sm_{cc}, {sms} SMs; best of {mp.TRIALS}")
    for label, (m, n, k), need, text, integer in FORMS:
        if cc < need:
            print(f"{label:38s} needs sm_{need}")
            continue
        ins = mp.Instruction(label, label, "", k, need, "", text)
        program = Program(mp.source(ins), code_type="c++", options=ProgramOptions(arch=mp.arch(cap), std="c++17"))
        kernel = program.compile("cubin").get_kernel("mma_rate")
        best = float("inf")
        for trial in range(mp.TRIALS + 1):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            launch(stream, config, kernel, out.data_ptr(), np.uint32(trial + 1))
            end.record()
            end.synchronize()
            if trial:
                best = min(best, start.elapsed_time(end))
        ops = warps * mp.ITERS * mp.CHAINS * 2.0 * m * n * k
        print(f"{label:38s} {ops / (best * 1e-3) / 1e12:7.1f} {'TOPS' if integer else 'TFLOP/s'}")


if __name__ == "__main__":
    main()
