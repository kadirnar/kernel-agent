"""Tensor-core instruction rates: which ``mma.sync`` runs at full rate on this GPU (#146).

The GEMM peaks of :func:`~kernel_agent.kernels.roofline.measure_peaks` are what a library
GEMM reaches (cuBLAS / cuBLASLt). A hand-written kernel is capped by the instruction it
issues instead. These are the ``mma.sync`` rates: the full-rate path on Ampere, Ada and
GeForce Blackwell; on Hopper and datacenter Blackwell ``wgmma`` / ``tcgen05`` run faster
(``kernel_agent/gpu_arch.py``) and e4m3 ``mma.sync`` is emulated (:func:`available`). On
GeForce Blackwell (sm_120) the FP8 instructions differ by 2x
(docs/RESEARCH-TRITON.md §1.1, RTX 5070 Ti): ``HMMA.16816.F32.BF16`` 104 TFLOP/s,
``QMMA.16832.F32.E4M3`` (plain e4m3, fp32 accumulation; both the sm_89 form and
``kind::f8f6f4``) 208, the block-scaled ``QMMA.SF.16832.F32.E4M3.E4M3.E8``
(``kind::mxf8f6f4.block_scale``) 416, and e4m3 with fp16 accumulation 416. Row-wise
CUTLASS (``_scaled_mm`` with row-wise scales), Triton ``tl.dot`` on e4m3 and
DeepSeek-style blockwise kernels use ``QMMA.F32``; Triton ``tl.dot_scaled`` and MXFP8 use
``QMMA.SF``. The INT8 W8A8 kernels (``int8_w8a8``, issue #178: Triton ``tl.dot`` on int8,
cuBLASLt's int8 GEMMs behind ``torch._int_mm``, CUDA ``mma.sync`` s8) use ``IMMA``: s8 x s8
with int32 accumulation, ``m16n8k32`` from sm_80 (Turing's form is ``m8n8k16``), measured
as :data:`S8_S32` in TOPS (one multiply-add = 2 ops, like the FLOP rates).

:func:`measure` compiles one register-only kernel per instruction with NVRTC
(``cuda.core``; each warp runs :data:`CHAINS` independent accumulator chains for
:data:`ITERS` iterations, 4 blocks of 256 threads per SM) and times it with CUDA events
(best of :data:`TRIALS`). An instruction the GPU or the compiler lacks is reported in the
second dict with the reason, never assumed. The peaks cache stores the rates as
``mma_tflops`` (keys: :data:`INSTRUCTIONS`) and the ceilings table names the instruction a
W8A8 kernel needs (:func:`kernel_agent.profiling.ceilings.fp8_instruction`).

The kernel follows ``docs/research-scripts/triton-sm120/mma_rate.cu`` (measured with nvcc
there); this NVRTC version has not been run on a GPU yet (``kernel-agent doctor
--remeasure-peaks`` measures it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ITERS = 4096
CHAINS = 8
THREADS = 256
BLOCKS_PER_SM = 4
TRIALS = 5


@dataclass(frozen=True)
class Instruction:
    key: str  # ``peaks["mma_tflops"]`` key
    label: str  # for tables
    sass: str  # what ptxas makes of it on sm_120
    k: int  # K of one m16n8k<k> instruction
    capability: int  # minimum compute capability (major * 10 + minor)
    used_by: str
    body: str  # one accumulation, chain c (CUDA C++)


_F32_OUT = '"+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])'
_F16_OUT = '"+r"(h[c][0]), "+r"(h[c][1])'
_S32_OUT = '"+r"(n[c][0]), "+r"(n[c][1]), "+r"(n[c][2]), "+r"(n[c][3])'
_AB = '"r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)'


def _mma(ptx: str, out: str, f32: bool) -> str:
    regs = (
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}"
        if f32
        else ("{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1}")
    )
    return f'asm volatile("{ptx} {regs};" : {out} : {_AB});'


FP8_F32 = "e4m3_f32"
FP8_SF = "e4m3_sf_f32"
BF16_F32 = "bf16_f32"
S8_S32 = "s8_s32"
INSTRUCTIONS = (
    Instruction(
        BF16_F32,
        "bf16 HMMA.F32",
        "HMMA.16816.F32.BF16",
        16,
        80,
        "cuBLAS bf16, Triton tl.dot on bf16",
        _mma("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", _F32_OUT, True),
    ),
    Instruction(
        FP8_F32,
        "e4m3 QMMA.F32",
        "QMMA.16832.F32.E4M3.E4M3",
        32,
        89,
        "row-wise CUTLASS (_scaled_mm row-wise), Triton tl.dot on e4m3, DeepSeek-style blockwise",
        _mma("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", _F32_OUT, True),
    ),
    Instruction(
        FP8_SF,
        "e4m3 QMMA.SF",
        "QMMA.SF.16832.F32.E4M3.E4M3.E8",
        32,
        120,
        "Triton tl.dot_scaled (unit or MX scales), MXFP8 GEMMs",
        'asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.'
        "row.col.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        '{%10},{%11,%12},{%13},{%14,%15};" : '
        + _F32_OUT
        + " : "
        + _AB
        + ', "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));',
    ),
    Instruction(
        "e4m3_f16",
        "e4m3 QMMA.F16",
        "QMMA.16832.F16.E4M3.E4M3",
        32,
        89,
        "e4m3 with fp16 accumulation (fp16 partial sums)",
        _mma("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", _F16_OUT, False),
    ),
    Instruction(
        S8_S32,
        "s8 IMMA.S32",
        "IMMA.16832.S8.S8",
        32,
        80,
        "INT8 W8A8 (int8_w8a8): Triton tl.dot on int8, cuBLASLt int8 (torch._int_mm), mma s8",
        _mma("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", _S32_OUT, True),
    ),
)
BY_KEY = {i.key: i for i in INSTRUCTIONS}


def source(instruction: Instruction) -> str:
    """The CUDA C++ of one instruction's rate kernel (``mma_rate``)."""
    return f"""
extern "C" __global__ void __launch_bounds__({THREADS}) mma_rate(float* out, unsigned int seed) {{
  unsigned int a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u;
  unsigned int b0 = a0 * 11u, b1 = a0 * 13u;
  // small finite operands: no e4m3 NaN codes (0x7f / 0xff)
  a0 &= 0x3f3f3f3fu; a1 &= 0x3f3f3f3fu; a2 &= 0x3f3f3f3fu; a3 &= 0x3f3f3f3fu;
  b0 &= 0x3f3f3f3fu; b1 &= 0x3f3f3f3fu;
  float d[{CHAINS}][4];
  unsigned int h[{CHAINS}][2];
  int n[{CHAINS}][4];
#pragma unroll
  for (int c = 0; c < {CHAINS}; ++c) {{
    d[c][0] = d[c][1] = d[c][2] = d[c][3] = 0.f;
    h[c][0] = h[c][1] = 0u;
    n[c][0] = n[c][1] = n[c][2] = n[c][3] = 0;
  }}
  unsigned int sfa = 127u, sfb = 127u;
  unsigned short z = 0;
#pragma unroll 1
  for (int it = 0; it < {ITERS}; ++it) {{
#pragma unroll
    for (int c = 0; c < {CHAINS}; ++c) {{
      {instruction.body}
    }}
  }}
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < {CHAINS}; ++c)
    s += d[c][0] + d[c][1] + d[c][2] + d[c][3] + __uint_as_float(h[c][0]) +
         __uint_as_float(h[c][1]) + (float)(n[c][0] + n[c][1] + n[c][2] + n[c][3]);
  out[blockIdx.x * blockDim.x + threadIdx.x] = s + (float)(sfa + sfb + z);
}}
"""


def flops(instruction: Instruction, sms: int) -> float:
    """FLOPs of one launch on ``sms`` SMs."""
    warps = sms * BLOCKS_PER_SM * THREADS / 32
    return warps * ITERS * CHAINS * 2.0 * 16 * 8 * instruction.k


def arch(capability: tuple[int, int]) -> str:
    """NVRTC ``arch``: the architecture-specific target (``sm_120a``) from sm_90 on, where
    the block-scaled and ``kind::`` forms live."""
    major, minor = capability
    return f"sm_{major}{minor}" + ("a" if major >= 9 else "")


#: Instructions on e4m3 operands (``mma.sync`` m16n8k32): native on sm_89 and sm_12x only.
_E4M3 = (FP8_F32, FP8_SF, "e4m3_f16")


def available(instruction: Instruction, capability: tuple[int, int]) -> str | None:
    """Why ``instruction`` cannot run on a GPU of ``capability`` (None: it can). e4m3
    ``mma.sync`` compiles on sm_90 / sm_100 but runs there as fp16 upcasts + HMMA (Triton's
    ``AccelerateMatmul.cpp``): its rate would be mistaken for an FP8 one, so it is not
    measured; those GPUs reach their FP8 peak through ``wgmma`` / ``tcgen05`` (the FP8 GEMM
    peak of ``roofline.measure_peaks``). The s8 ``mma.sync`` (IMMA) is measured from sm_80
    on: on sm_90 / sm_100 it is the ``mma.sync`` rate, below the ``wgmma`` / ``tcgen05.mma
    kind::i8`` path that the INT8 GEMM peak (``torch._int_mm``) reaches."""
    cc = capability[0] * 10 + capability[1]
    if cc < instruction.capability:
        return f"needs sm_{instruction.capability}+, this GPU is sm_{cc}"
    if instruction.key in _E4M3 and capability[0] in (9, 10, 11):
        full = "wgmma" if capability[0] == 9 else "tcgen05.mma"
        return f"e4m3 mma.sync is emulated through fp16 on sm_{cc} (its FP8 rate is {full}'s)"
    if instruction.key == FP8_SF and capability[0] != 12:
        return "mma.sync block_scale exists on sm_120a / sm_121a only"
    return None


def measure() -> tuple[dict[str, float], dict[str, str]]:
    """``({key: TFLOP/s}, {key: why unavailable})`` on the current GPU (hold the GPU lock;
    about a second)."""
    import numpy as np
    import torch
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

    index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    dev = Device(index)
    dev.set_current()
    stream = dev.create_stream(torch.cuda.current_stream())
    out = torch.empty(sms * BLOCKS_PER_SM * THREADS, device="cuda")
    config = LaunchConfig(grid=sms * BLOCKS_PER_SM, block=THREADS)
    rates: dict[str, float] = {}
    missing: dict[str, str] = {}
    for instruction in INSTRUCTIONS:
        if (why := available(instruction, capability)) is not None:
            missing[instruction.key] = why
            continue
        try:
            options = ProgramOptions(arch=arch(capability), std="c++17")
            program = Program(source(instruction), code_type="c++", options=options)
            kernel = program.compile("cubin").get_kernel("mma_rate")
            best = float("inf")
            for trial in range(TRIALS + 1):  # the first launch warms up
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                launch(stream, config, kernel, out.data_ptr(), np.uint32(trial + 1))
                end.record()
                end.synchronize()
                if trial:
                    best = min(best, start.elapsed_time(end))
            rates[instruction.key] = round(flops(instruction, sms) / (best * 1e-3) / 1e12, 1)
        except Exception as exc:  # not compiled for / not runnable on this GPU
            missing[instruction.key] = f"{type(exc).__name__}: {exc}"[:200]
    return rates, missing


def describe(rates: dict[str, Any]) -> str:
    """``mma.sync bf16 104 · e4m3 QMMA.F32 208 · e4m3 QMMA.SF 416 TFLOP/s`` (integer rates,
    ``s8 IMMA.S32``, after them in TOPS)."""
    parts = [
        f"{BY_KEY[k].label} {v:.0f}" if k in BY_KEY else f"{k} {v:.0f}"
        for k, v in rates.items()
        if k != S8_S32
    ]
    text = "mma.sync " + " · ".join(parts) + " TFLOP/s" if parts else ""
    if rates.get(S8_S32):
        ints = f"{BY_KEY[S8_S32].label} {float(rates[S8_S32]):.0f} TOPS"
        text = f"{text} · {ints}" if text else f"mma.sync {ints}"
    return text
