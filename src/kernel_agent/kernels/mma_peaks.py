"""Tensor-core instruction rates: which ``mma.sync`` runs at full rate on this GPU (#146).

The GEMM peaks of :func:`~kernel_agent.kernels.roofline.measure_peaks` are what a library
GEMM reaches (cuBLAS / cuBLASLt). A hand-written kernel is capped by the instruction it
issues instead. These are the ``mma.sync`` rates: the full-rate path on Turing, Ampere, Ada
and GeForce Blackwell; on Hopper and datacenter Blackwell ``wgmma`` / ``tcgen05`` run faster
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
with int32 accumulation, ``m16n8k32`` from sm_80 (:data:`S8_S32`), Turing's ``m8n8k16``
(:data:`S8_S32_K16`), in TOPS (one multiply-add = 2 ops, like the FLOP rates).

The 16-bit forms (#257): fp16 with fp32 accumulation (:data:`F16_F32`) and with fp16
accumulation (:data:`F16_F16`), TF32 (:data:`TF32_F32`, ``m16n8k8``), all sm_80+, and
Turing's fp16 ``m16n8k8`` (:data:`F16_F32_K8`, :data:`F16_F16_K8`, sm_75+). GeForce Turing,
Ampere, Ada and Blackwell run fp32-accumulating HMMA at half the fp16-accumulating rate,
datacenter parts (T4, A100, A10, A40, L4, L40S) at the same one: :func:`accumulation_line`
says which this GPU does. Measured on an RTX 5070 Ti (sm_120): bf16 / fp16 ``HMMA.F32`` 104,
fp16 ``HMMA.F16`` 208, TF32 52 TFLOP/s; Turing's shapes run padded there (``m16n8k8`` at
half the ``m16n8k16`` rate: 52 / 104; ``m8n8k16`` becomes ``IMMA.16816``: 99 TOPS, a
quarter of ``m16n8k32``'s 412), so a kernel for sm_80+ issues the larger shapes.

:func:`measure` compiles one register-only kernel per instruction with NVRTC
(``cuda.core``; each warp runs :data:`CHAINS` independent accumulator chains for
:data:`ITERS` iterations, 4 blocks of 256 threads per SM; no kernel spills on sm_75 to
sm_120) and times it with CUDA events (best of :data:`TRIALS`). An instruction the GPU or
the compiler lacks is reported in the second dict with the reason, never assumed;
``measure(capability)`` runs an older GPU's kernels through its PTX (rates of this GPU).
The peaks cache stores the rates as ``mma_tflops`` (keys: :data:`INSTRUCTIONS`), the
toolchain summary shows them (:func:`describe`) with :func:`accumulation_line`, and the
ceilings table names the instruction a W8A8 kernel needs
(:func:`kernel_agent.profiling.ceilings.fp8_instruction`).

The kernel follows ``docs/research-scripts/triton-sm120/mma_rate.cu`` (measured with nvcc
there; ``kernel-agent doctor --remeasure-peaks`` measures this NVRTC version).
"""

from __future__ import annotations

from collections.abc import Mapping
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
    k: int  # K of one m<m>n<n>k<k> instruction
    capability: int  # minimum compute capability (major * 10 + minor)
    used_by: str
    body: str  # one accumulation, chain c (CUDA C++)
    m: int = 16  # M x N of one instruction: m16n8 but Turing's s8 m8n8k16
    n: int = 8
    integer: bool = False  # s8 x s8 -> int32: rated in TOPS (one multiply-add = 2 ops)


_F32_OUT = '"+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])'
_AB = '"r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)'


def _mma(ptx: str, acc: str, outs: int, a: int = 4, b: int = 2) -> str:
    """One ``mma.sync`` on chain ``c``: ``outs`` accumulator registers of ``acc`` (``d``
    fp32, ``h`` fp16x2, ``n`` int32; read and written), ``a`` / ``b`` operand registers (the
    m16n8k16 forms take 4 / 2, Turing's m16n8k8 fp16 2 / 1, m8n8k16 s8 1 / 1)."""
    out = ", ".join(f'"+{"f" if acc == "d" else "r"}"({acc}[c][{i}])' for i in range(outs))
    ins = ", ".join([f'"r"(a{i})' for i in range(a)] + [f'"r"(b{i})' for i in range(b)])

    def regs(first: int, count: int) -> str:
        return "{" + ",".join(f"%{first + i}" for i in range(count)) + "}"

    operands = ",".join((regs(0, outs), regs(outs, a), regs(outs + a, b), regs(0, outs)))
    return f'asm volatile("{ptx} {operands};" : {out} : {ins});'


FP8_F32 = "e4m3_f32"
FP8_SF = "e4m3_sf_f32"
BF16_F32 = "bf16_f32"
F16_F32 = "f16_f32"
F16_F16 = "f16_f16"
TF32_F32 = "tf32_f32"
F16_F32_K8 = "f16_f32_k8"
F16_F16_K8 = "f16_f16_k8"
S8_S32 = "s8_s32"
S8_S32_K16 = "s8_s32_k16"
INSTRUCTIONS = (
    Instruction(
        BF16_F32,
        "bf16 HMMA.F32",
        "HMMA.16816.F32.BF16",
        16,
        80,
        "cuBLAS bf16, Triton tl.dot on bf16",
        _mma("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", "d", 4),
    ),
    Instruction(
        F16_F32,
        "fp16 HMMA.F32",
        "HMMA.16816.F32",
        16,
        80,
        "cuBLAS fp16, Triton tl.dot on fp16 (fp32 accumulation)",
        _mma("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "d", 4),
    ),
    Instruction(
        F16_F16,
        "fp16 HMMA.F16",
        "HMMA.16816.F16",
        16,
        80,
        "fp16 accumulation: Triton tl.dot(out_dtype=tl.float16), cuBLAS with "
        "torch.backends.cuda.matmul.allow_fp16_accumulation, mma.sync .f16 accumulators",
        _mma("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", "h", 2),
    ),
    Instruction(
        TF32_F32,
        "tf32 HMMA.F32",
        "HMMA.1688.F32.TF32",
        8,
        80,
        "fp32 matmuls on TF32 tensor cores (torch allow_tf32, cuDNN TF32), Triton tl.dot on "
        "fp32 (input_precision tf32)",
        _mma("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", "d", 4),
    ),
    Instruction(
        F16_F32_K8,
        "fp16 HMMA.1688.F32",
        "HMMA.1688.F32",
        8,
        75,
        "Turing's fp16 HMMA shape (m16n8k16 needs sm_80): mma.sync fp16 on sm_75",
        _mma("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32", "d", 4, 2, 1),
    ),
    Instruction(
        F16_F16_K8,
        "fp16 HMMA.1688.F16",
        "HMMA.1688.F16",
        8,
        75,
        "Turing's fp16 HMMA with fp16 accumulation (m16n8k8)",
        _mma("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16", "h", 2, 2, 1),
    ),
    Instruction(
        FP8_F32,
        "e4m3 QMMA.F32",
        "QMMA.16832.F32.E4M3.E4M3",
        32,
        89,
        "row-wise CUTLASS (_scaled_mm row-wise), Triton tl.dot on e4m3, DeepSeek-style blockwise",
        _mma("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", "d", 4),
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
        _mma("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", "h", 2),
    ),
    Instruction(
        S8_S32,
        "s8 IMMA.S32",
        "IMMA.16832.S8.S8",
        32,
        80,
        "INT8 W8A8 (int8_w8a8): Triton tl.dot on int8, cuBLASLt int8 (torch._int_mm), mma s8",
        _mma("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", "n", 4),
        integer=True,
    ),
    Instruction(
        S8_S32_K16,
        "s8 IMMA.8816.S32",
        "IMMA.16816.S8.S8",  # padded; IMMA.8816.S8.S8 on sm_75 to sm_89
        16,
        75,
        "Turing's INT8 IMMA shape (m16n8k32 needs sm_80): mma.sync s8 on sm_75",
        _mma("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32", "n", 2, 1, 1),
        m=8,
        integer=True,
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
    """FLOPs (integer instructions: ops) of one launch on ``sms`` SMs: 2 × m × n × k per
    instruction."""
    warps = sms * BLOCKS_PER_SM * THREADS / 32
    shape = instruction.m * instruction.n * instruction.k
    return warps * ITERS * CHAINS * 2.0 * shape


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
    on (Turing's ``m8n8k16`` from sm_75): on sm_90 / sm_100 it is the ``mma.sync`` rate,
    below the ``wgmma`` / ``tcgen05.mma kind::i8`` path that the INT8 GEMM peak
    (``torch._int_mm``) reaches. Every other form needs its minimum capability only."""
    cc = capability[0] * 10 + capability[1]
    if cc < instruction.capability:
        return f"needs sm_{instruction.capability}+, this GPU is sm_{cc}"
    if instruction.key in _E4M3 and capability[0] in (9, 10, 11):
        full = "wgmma" if capability[0] == 9 else "tcgen05.mma"
        return f"e4m3 mma.sync is emulated through fp16 on sm_{cc} (its FP8 rate is {full}'s)"
    if instruction.key == FP8_SF and capability[0] != 12:
        return "mma.sync block_scale exists on sm_120a / sm_121a only"
    return None


def measure(
    capability: tuple[int, int] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """``({key: TFLOP/s}, {key: why unavailable})`` on the current GPU (hold the GPU lock;
    about a second). ``capability``: an older GPU's (``(7, 5)``): the instructions it has
    are compiled to its ``compute_XX`` PTX, which the driver JIT-compiles for this GPU. That
    runs an older GPU's rate kernels here (do they compile, load and run); the rates are
    this GPU's, never that GPU's."""
    import numpy as np
    import torch
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

    index = torch.cuda.current_device()
    native = torch.cuda.get_device_capability(index)
    capability = native if capability is None else capability
    if tuple(capability) > tuple(native):
        raise ValueError(f"sm_{capability[0]}{capability[1]} is newer than this GPU")
    jit = tuple(capability) != tuple(native)
    target = f"compute_{capability[0]}{capability[1]}" if jit else arch(capability)
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
            options = ProgramOptions(arch=target, std="c++17")
            program = Program(source(instruction), code_type="c++", options=options)
            kernel = program.compile("ptx" if jit else "cubin").get_kernel("mma_rate")
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


def describe(rates: Mapping[str, Any]) -> str:
    """``mma.sync bf16 HMMA.F32 104 · e4m3 QMMA.F32 208 · e4m3 QMMA.SF 416 TFLOP/s`` (the
    integer rates, ``s8 IMMA.S32``, after them in TOPS)."""
    floats: list[str] = []
    ints: list[str] = []
    for key, value in rates.items():
        instruction = BY_KEY.get(key)
        if not value:
            continue
        part = f"{instruction.label if instruction else key} {float(value):.0f}"
        (ints if instruction is not None and instruction.integer else floats).append(part)
    groups = [" · ".join(p) + unit for p, unit in ((floats, " TFLOP/s"), (ints, " TOPS")) if p]
    return "mma.sync " + " · ".join(groups) if groups else ""


#: fp16 HMMA pairs (fp32 accumulation, fp16 accumulation): sm_80's m16n8k16, else Turing's
#: m16n8k8. GeForce Turing, Ampere, Ada and Blackwell run fp32-accumulating HMMA at half
#: the fp16-accumulating rate (RTX 3090 71 vs 142 TFLOP/s dense, RTX 4090 165 vs 330:
#: NVIDIA's whitepapers; RTX 5070 Ti 104 vs 208 measured), datacenter parts at the same
#: rate (T4, A100, A10, A40, L4, L40S).
ACCUMULATION_PAIRS = ((F16_F32, F16_F16), (F16_F32_K8, F16_F16_K8))
#: fp32 accumulation below this share of the fp16-accumulating rate: the half-rate rule.
HALF_RATE = 0.75
#: ... at or above it: full rate.
FULL_RATE = 0.9


def accumulation(rates: Mapping[str, Any]) -> tuple[float, float] | None:
    """(fp32-accumulating, fp16-accumulating) fp16 HMMA rates measured here (the first pair
    of :data:`ACCUMULATION_PAIRS` with both), None without one."""
    for f32, f16 in ACCUMULATION_PAIRS:
        if rates.get(f32) and rates.get(f16):
            return float(rates[f32]), float(rates[f16])
    return None


def accumulation_line(rates: Mapping[str, Any]) -> str | None:
    """``fp32-accumulating HMMA at 50 % of fp16-accumulating here (...)``: what fp32
    accumulation costs on this GPU's tensor cores (None: the pair was not measured)."""
    found = accumulation(rates)
    if found is None:
        return None
    f32, f16 = found
    ratio = f32 / f16
    text = (
        f"fp32-accumulating HMMA at {ratio * 100:.0f} % of fp16-accumulating here (measured "
        f"fp16 `mma.sync` {f32:.0f} vs {f16:.0f} TFLOP/s)"
    )
    if ratio < HALF_RATE:
        return (
            f"{text}: a kernel that accumulates in fp32 (cuBLAS, Triton `tl.dot`, every bf16 "
            "HMMA) is capped at the lower rate; fp16 operands with fp16 accumulation reach the "
            'higher one where their error is acceptable (`precision: "reduced"`)'
        )
    if ratio >= FULL_RATE:
        return f"{text}: fp32 accumulation costs no tensor-core rate on this GPU"
    return text
