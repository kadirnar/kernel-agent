"""Kernel backends by target class: the policy, and which backend won where.

Agents default to one backend: in the runs studied for docs/RESEARCH-TRITON.md (§4) they
wrote 161 CUDA C++ candidates, 10 Triton, 7 hybrid and no CuTe DSL or TileLang kernel,
although the planner listed `cute` once and Triton first five times. This module gives the
planner and the engineers a policy per *target class* (RESEARCH-TRITON §5.1: what the op
is, its shape regime M / N / K, its bound and its precision; never a model's module
names) and records what each backend achieved, so the choice follows evidence rather than
habit:

* :func:`classify` reads a candidate's backend from what its source runs (``@triton.jit``,
  ``load_inline``, ``T.prim_func``, ``cutlass.cute``, ``@helion.kernel``), not from its
  imports alone: a kernel
  that imports ``tilelang`` only to find the CUTLASS headers it ships is CUDA C++ (§4.1).
  Without any such marker it falls back to the imports
  (:func:`kernel_agent.ledger.detect_backend`, the ledger's ``backend`` column).
* :func:`target_class` puts a target in one row of :data:`POLICY` from its module family,
  precision and captured shapes (rows ``M`` of the dominant case, sequence length);
  :func:`policy_text` is the planner's table, :func:`engineer_note` the target's row.
  Both follow the GPU (issue #165, :class:`kernel_agent.gpu_arch.Facts`): the rows that
  differ by architecture (:data:`ARCH_POLICY`: compute-bound FP8 on Ada, Hopper and
  datacenter Blackwell; on GeForce Blackwell with this GPU's measured FP8 instruction
  rates; compute-bound INT8 on Turing, Hopper and datacenter Blackwell; Turing's small-M
  GEMMs, short attention, decoder layers and bf16 GEMMs) replace
  :data:`POLICY`'s, whose evidence was measured on an RTX 5070 Ti (sm_120);
  a class whose precision the GPU cannot run is left out, and :data:`ARCH_RULES` adds what
  each family needs for the tensor-core peak. No row or rule names a bundled example whose
  ``ARCHS`` excludes the GPU (:func:`runnable_text`, #254).
* :func:`outcomes` / :func:`by_backend` tabulate a run's kernel evaluations per target and
  backend (library priors and library scout rows excluded, a re-evaluation replaces its
  snapshot's numbers);
  :func:`report_lines` and :func:`status_lines` show them in ``report.md`` and ``status``.
* :func:`record_run` appends the run's outcomes to the kernel library
  (``<library>/<sm_arch>/backends.jsonl``, one line per run and target: class, planned
  backends, per-backend evaluations / correct / kept / best, the winner);
  :func:`track_record` aggregates them per class and backend and
  :func:`track_record_note` tells the next planner which backend won where on this GPU.
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from kernel_agent import ledger
from kernel_agent.budget import not_agents
from kernel_agent.gpu_arch import Facts, example_requirement, supports
from kernel_agent.workspace import RunDir, read_json

#: What a candidate's source *runs*, per backend (first column of RESEARCH-TRITON §4.1).
_USES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cute", re.compile(r"\bcutlass\.cute\b|@cute\.(?:jit|kernel)\b|\bcute\.compile\s*\(")),
    (
        "tilelang",
        re.compile(r"\bT\.prim_func\b|@tilelang\.jit\b|\btilelang\.(?:compile|lower)\s*\("),
    ),
    (
        "nvrtc",
        re.compile(
            r"^\s*(?:import|from)\s+cuda\.core\b|\bProgram\s*\(|\bnvrtc\.nvrtc\w+\s*\(", re.M
        ),
    ),
    ("cuda", re.compile(r"\bload_inline\s*\(|\bcpp_extension\.load\s*\(|\bload\s*\(\s*name=")),
    ("triton", re.compile(r"@triton\.(?:jit|autotune|heuristics)\b|\btriton\.jit\s*\(")),
    # Helion ("PyTorch with tiles", compiled to Triton; issue #229): its kernel decorator, or
    # the kernel a candidate makes in build (kernels/helion_tune.py)
    (
        "helion",
        re.compile(
            r"@helion\.(?:kernel|jit)\b|\bhelion\.(?:kernel|jit)\s*\(|\bhelion_tune\.kernel\s*\("
        ),
    ),
)


def classify(source: str) -> str:
    """Backend(s) a candidate's source runs (``cuda+triton`` for a hybrid), from kernel
    decorators and compile calls; without any, from its imports
    (:func:`kernel_agent.ledger.detect_backend`; ``torch`` when there is no custom kernel)."""
    if (library := ledger.detect_backend(source)).startswith("library:"):
        return library  # a library scout candidate (libscout/): the library is its backend
    found = [name for name, pattern in _USES if pattern.search(source)]
    return "+".join(found) if found else ledger.detect_backend(source)


# ------------------------------------------------------------------ policy


@dataclass(frozen=True)
class TargetClass:
    """One row of the backend policy (RESEARCH-TRITON §5.1, sm_120 measurements)."""

    id: str
    label: str
    first: str  # first backend and recipe
    why: str  # the evidence
    second: str  # the fallback
    order: tuple[str, ...] = ()  # suggested `backends` order for the plan
    never: str = ""  # what not to do for this class


#: What a GeForce Blackwell FP8 GEMM kernel must beat (measured on an RTX 5070 Ti).
_GEFORCE_FP8_BASELINE = (
    "the baseline to beat is cuBLASLt tensor-wise (nvjet; RTX 5070 Ti: 325 TFLOP/s at 4096³, "
    "gate|up 30.2 us at M = 352) with the scales applied by the consumer"
)

POLICY: tuple[TargetClass, ...] = (
    TargetClass(
        "fp8_gemm",
        "Compute-bound FP8 GEMM (W8A8, M ≳ 128 rows)",
        "block-scaled MMA (`mma.sync kind::mxf8f6f4.block_scale`, SASS `QMMA.SF`): CuTe DSL "
        "`MmaMXF8Op` GEMM with the scales, bias / residual and e4m3 output fused in the "
        "epilogue (`examples/cute_fp8_blockscaled_gemm.py`, unit ue8m0 scales when the real "
        "scales are per token / channel); Triton `tl.dot_scaled` with unit scales for a "
        "quick win",
        "`QMMA.F32` (plain e4m3 `mma.sync`) is capped at 208 TFLOP/s on sm_120, `QMMA.SF` "
        "runs 416 with the same fp32 accumulation (RTX 5070 Ti); " + _GEFORCE_FP8_BASELINE,
        "direct cuBLASLt from C++ (descriptors cached, top-8 algorithms timed), the scales "
        "and glue in one fused kernel",
        ("cute", "triton", "cuda"),
        "never a plain `QMMA.F32` kernel (Triton `tl.dot` on e4m3, row-wise `_scaled_mm`, "
        "CUTLASS `SM120_16x8x32_TN`) for compute-bound FP8: half rate",
    ),
    TargetClass(
        "int8_gemm",
        "Compute-bound INT8 GEMM (INT8 W8A8, M ≳ 128 rows)",
        "Triton `tl.dot` on int8 tiles with an int32 accumulator, per-token quantisation in "
        "a one-pass row kernel (or fused into the producer) and both scales + bias in the "
        "epilogue (`examples/triton_int8_w8a8_gemm.py`)",
        "s8 `mma.sync` (`IMMA.16832`) runs 410 TOPS on sm_120, twice the plain e4m3 "
        "`QMMA.F32` and as fast as block-scaled `QMMA.SF` (RTX 5070 Ti; `torch._int_mm` 327 "
        "TOPS): the example's gate|up at M = 352 runs 25 us vs 68 bf16, 32 for the FP8 example",
        "cuBLASLt int8 (`torch._int_mm`) with the scales applied by the consumer, or CUDA C++ "
        "`mma.sync` s8 inside a fused layer",
        ("triton", "cuda", "cute"),
        "never per-tensor or static (calibrated) activation scales, and never plain per-token "
        "INT8 on activations with outlier channels (token crest > ~20): SmoothQuant, or keep "
        "those GEMMs in FP8 / bf16",
    ),
    TargetClass(
        "small_m_gemm",
        "Small-M GEMV / skinny GEMM (M < 128 rows, weights streamed)",
        "CUDA C++ (`load_inline`): the bundled FP8 GEMV / skinny GEMM examples (INT8: "
        "`cuda_int8_gemv.py` weight-only, `cuda_int8_skinny_gemm.py` W8A8 on IMMA)",
        "memory bound: the weight bytes set the floor and ~19 us host per launch matters; "
        "every kept small-M kernel of the studied runs was CUDA C++ (up to 10.5x)",
        "Triton `tl.dot` with BM = 16 inside CUDA graphs (774 GB/s)",
        ("cuda", "triton"),
    ),
    TargetClass(
        "short_attention",
        "Attention over ≤ 16 tokens",
        "Triton single-tile kernel (one program per sequence × head, Q/K/V in registers) "
        "as an SDPA custom_op",
        "latency bound: 5.7 vs 15.9 us for cuDNN at 11 tokens; FlashAttention's 128-row "
        "tiles are padding there",
        "CUDA WMMA inside a fused CUDA layer (4.8-6.8 us)",
        ("triton", "cuda"),
    ),
    TargetClass(
        "conv",
        "Conv / VAE (channels-last, large activations)",
        "Triton conv-as-GEMM with prologue / epilogue fusion",
        "7.88x on a VAE decoder and fast to write",
        "CUDA C++ for a fused stencil + GEMM with an smem halo (9.64x)",
        ("triton", "cuda"),
    ),
    TargetClass(
        "decoder_layer",
        "Fused decoder layer (norm, RoPE, attention, GEMMs)",
        "timed eagerly: CUDA C++ behind one launcher; inside CUDA graphs: Inductor glue + "
        "library or (on GPUs with them) block-scaled GEMMs",
        "host time decides an eager-timed layer: 2.21x with 5 Triton launches vs 2.70x for "
        "the same math behind one C++ launcher",
        "CuTe DSL persistent layer from `examples/cute_fp8_decoder_block.py` (one launch per "
        "fused block, TVM-FFI)",
        ("cuda", "cute"),
    ),
    TargetClass(
        "bf16_gemm",
        "Compute-bound bf16 GEMM / MLP (M ≳ 128 rows)",
        "keep cuBLAS(Lt) for the GEMM and fuse what is around it in CUDA C++ (one launcher)",
        "a Triton `tl.dot` sweep did not beat cuBLAS at M = 352 (25.9 vs 25.5 us, 50.5 vs "
        "39.2); the exact tier needs cuBLAS's summation order",
        "CuTe DSL / CUTLASS GEMM with a fused epilogue where the tier allows another "
        "summation order",
        ("cuda", "cute"),
    ),
    TargetClass(
        "elementwise",
        "Norm / activation / RoPE / small glue (launch- or bandwidth-bound)",
        "CUDA C++ `load_inline` (~19 us host per call) or CuTe DSL with TVM-FFI (~25 us)",
        "host overhead per call decides small ops (README, Backends: Triton ~43 us)",
        "Triton inside CUDA graphs (host time does not count there)",
        ("cuda", "cute", "triton"),
    ),
    TargetClass(
        "other",
        "Anything else",
        "the backend whose bundled example is closest to the op",
        "no class-specific evidence yet",
        "the next backend of the plan",
    ),
)
_BY_ID = {c.id: c for c in POLICY}

_SIG = re.compile(r"\[([0-9, ]*)\]")
#: Precisions whose GEMMs run on FP8 tensor cores (both operands e4m3): W8A8 with per-token /
#: per-channel scales and MXFP8 (one ue8m0 scale per 32 K-elements, ``fp8_mx``).
_FP8 = ("fp8_w8a8", "fp8_mx")
_FP8_LABEL = "Compute-bound FP8 GEMM (W8A8, M ≳ 128 rows)"
#: Precisions whose GEMMs run on the INT8 tensor cores (IMMA: s8 x s8 -> int32, #178).
_INT8 = ("int8_w8a8",)
_INT8_LABEL = "Compute-bound INT8 GEMM (INT8 W8A8, M ≳ 128 rows)"
_INT8_NEVER = (
    "never per-tensor or static (calibrated) activation scales, and never plain per-token "
    "INT8 on activations with outlier channels (token crest > ~20): SmoothQuant, or keep "
    "those GEMMs in FP8 / bf16"
)

#: :data:`POLICY` rows that differ by architecture family (``gpu_arch.Family.key``): they
#: replace the row of the same id on that family. :data:`POLICY`'s rows were measured on an
#: RTX 5070 Ti (GeForce Blackwell); these follow the families' documented instructions
#: (``gpu-architectures/gpus.md`` and its sources), and the GeForce Blackwell FP8 row the GPU's own
#: measured rates (:func:`_geforce_fp8`).
ARCH_POLICY: dict[str, dict[str, TargetClass]] = {
    "pre_ampere": {
        "int8_gemm": TargetClass(
            "int8_gemm",
            _INT8_LABEL,
            "cuBLASLt int8 (`torch._int_mm`: IMMA) with the per-token / per-channel scales in a "
            "fused epilogue kernel, or CUDA C++ `mma.sync.m8n8k16.s32.s8.s8.s32` (sm_75's form)",
            "Turing has IMMA (m8n8k16) but no bf16 tensor cores, and Triton's int8 `tl.dot` "
            "does not compile below sm_80 (Triton 3.8: `TritonGPUAccelerateMatmul` fails for "
            "sm_75; fp16 / bf16 dots run on FMA units there)",
            "CUTLASS sm_75 int8 GEMMs from C++",
            ("cuda", "triton"),
            "never Triton `tl.dot` on int8 tiles on sm_75 (it does not compile); never the "
            "`m16n8k32` s8 form (sm_80+)",
        ),
        # #254: the sm_120 rows name bf16 examples declared sm_80+ (and a CuTe DSL one,
        # which has no sm_75 target); these are the paths Turing has
        "small_m_gemm": TargetClass(
            "small_m_gemm",
            _BY_ID["small_m_gemm"].label,
            "CUDA C++ (`load_inline`) GEMV / skinny GEMM: 16-byte weight loads, CUDA-core FMA "
            "for a few rows, fp16 `mma.sync.m16n8k8` (Turing's HMMA form) once M fills a tile; "
            "weight-only INT8 / FP8 codes dequantised in registers",
            "memory bound: the weight bytes set the floor and the host time per launch matters "
            "(RTX 5070 Ti: every kept small-M kernel of the studied runs was CUDA C++, up to "
            "10.5x); sm_75 has no bf16 tensor cores, no `cp.async` and no `m16n8k16` / "
            "`m16n8k32` forms, and the bundled small-M examples are bf16 and sm_80+",
            "Triton `tl.dot` on fp16 with BM = 16 inside CUDA graphs (FMA units below sm_80: "
            "enough while the weight stream dominates; int8 `tl.dot` does not compile here)",
            ("cuda", "triton"),
            "never `mma.sync` `m16n8k16` / `m16n8k32` or `cp.async` on sm_75 (ptxas: sm_80+)",
        ),
        "short_attention": TargetClass(
            "short_attention",
            _BY_ID["short_attention"].label,
            "CUDA C++ single-tile kernel (one block per sequence × head, Q/K/V in shared "
            "memory, WMMA fp16 `m16n16k16` or FMA for the ≤ 16 × 16 scores) as an SDPA "
            "custom_op",
            "latency bound (RTX 5070 Ti: a single-tile kernel 5.7 vs 15.9 us for cuDNN at 11 "
            "tokens); on sm_75 SDPA has no flash or cuDNN backend (both sm_80+) and Triton's "
            "`tl.dot` runs on FMA units",
            "SDPA's memory-efficient backend in fp16 as the baseline; a Triton single-tile "
            "kernel in fp16 (FMA `tl.dot`: measure it, the tile is ≤ 16 × 16)",
            ("cuda", "triton"),
        ),
        "decoder_layer": TargetClass(
            "decoder_layer",
            _BY_ID["decoder_layer"].label,
            "timed eagerly: CUDA C++ behind one launcher (cuBLAS fp16 GEMMs or fp16 "
            "`mma.sync` m16n8k8, norm / RoPE / attention glue fused); inside CUDA graphs: "
            "Inductor glue + cuBLAS",
            "host time decides an eager-timed layer (RTX 5070 Ti: 2.21x with 5 Triton launches "
            "vs 2.70x for the same math behind one C++ launcher); sm_75 has fp16 tensor cores "
            "only, and CuTe DSL 4.8 has no sm_75 target",
            "Triton for the memory-bound glue inside CUDA graphs (its `tl.dot` runs on FMA "
            "units here: keep the GEMMs in cuBLAS)",
            ("cuda", "triton"),
        ),
        "bf16_gemm": TargetClass(
            "bf16_gemm",
            _BY_ID["bf16_gemm"].label,
            "keep cuBLAS(Lt) for the GEMM and fuse what is around it in CUDA C++ (one "
            "launcher); the run's dtype decides the rate here: fp16 GEMMs get the tensor "
            "cores (HMMA), bf16 ones run on CUDA cores",
            "sm_75 has fp16 `mma.sync` (m16n8k8) but no bf16 tensor cores (datasheet T4: 65 "
            "fp16 tensor TFLOP/s vs 8.1 fp32; the measured fp16 / bf16 peaks give this GPU's "
            "ratio); the exact tier needs cuBLAS's summation order",
            "where the tier allows another rounding: fp16 HMMA on operands converted from "
            "bf16 in the kernel (fp16 overflows past 65504: check the operands' amax), or "
            "CUTLASS sm_75 fp16 GEMMs with a fused epilogue",
            ("cuda", "triton"),
            "never Triton `tl.dot` for a compute-bound GEMM on sm_75 (FMA units)",
        ),
    },
    "ampere": {
        "int8_gemm": TargetClass(
            "int8_gemm",
            _INT8_LABEL,
            "Triton `tl.dot` on int8 tiles (IMMA `mma.sync` m16n8k32 s8, int32 accumulators, "
            "`cp.async` pipelines through `num_stages`) with per-token quantisation in a "
            "one-pass row kernel and the scales + bias in the epilogue "
            "(`examples/triton_int8_w8a8_gemm.py`); cuBLASLt int8 (`torch._int_mm`) as the "
            "baseline",
            "Ampere has no FP8 tensor cores: IMMA is its 8-bit compute path, 2x the bf16 rate "
            "(A100: 624 vs 312 TOPS; GeForce RTX 30xx, whose fp32-accumulating bf16 runs at "
            "half rate: 4x)",
            "CUTLASS sm_80 int8 GEMMs from C++ when the epilogue fuses more; CUDA C++ "
            "`mma.sync` s8 inside a fused layer",
            ("triton", "cuda", "cute"),
            _INT8_NEVER,
        ),
    },
    "ada": {
        "int8_gemm": TargetClass(
            "int8_gemm",
            _INT8_LABEL,
            "Triton `tl.dot` on int8 tiles (IMMA `mma.sync` m16n8k32 s8, int32 accumulators) "
            "with per-token quantisation in a one-pass row kernel and the scales + bias in "
            "the epilogue (`examples/triton_int8_w8a8_gemm.py`); cuBLASLt int8 "
            "(`torch._int_mm`) as the baseline",
            "Ada runs INT8 at the rate of FP8 with fp16 accumulation; GeForce Ada runs FP8 with "
            "fp32 accumulation at half of it (RTX 4090: 661 INT8 TOPS vs 330 FP8 TFLOP/s), so "
            "INT8 is the faster 8-bit path there where its accuracy holds",
            "FP8 W8A8 (`fp8_w8a8`) where the activations have outlier channels",
            ("triton", "cuda", "cute"),
            _INT8_NEVER,
        ),
        "fp8_gemm": TargetClass(
            "fp8_gemm",
            _FP8_LABEL,
            "plain FP8 `mma.sync` (QMMA, the only FP8 tensor-core instruction on sm_89): Triton "
            "`tl.dot` on e4m3 (`examples/triton_fp8_w8a8_gemm.py` takes this path below "
            "sm_100) or direct cuBLASLt tensor-wise (`examples/cuda_cublaslt_fp8.py`), the "
            "per-token / per-channel scales in the epilogue",
            "Ada has no block-scaled MMA: the plain e4m3 instruction is its full FP8 rate "
            "(the *Ceilings* table's W8A8 peak and the toolchain's measured `QMMA.F32` rate)",
            "CUTLASS's sm_89 FP8 GEMMs (blockwise scales: `examples/94_ada_fp8_blockwise`) "
            "from C++; `torch._scaled_mm` as the reference",
            ("triton", "cuda", "cute"),
            "never `tl.dot_scaled` / MXFP8 on sm_89 (no block-scaled MMA: Triton emulates it "
            "through bf16)",
        ),
    },
    "hopper": {
        "int8_gemm": TargetClass(
            "int8_gemm",
            _INT8_LABEL,
            "`wgmma` s8 (int32 accumulators): Triton `tl.dot` on int8 tiles (it emits wgmma on "
            "sm_90), cuBLASLt int8 (`torch._int_mm`) as the baseline, CUTLASS sm_90 int8 GEMMs "
            "when the epilogue fuses scales, bias or quantisation",
            "Hopper's INT8 peak equals its FP8 one (2x bf16) and needs `wgmma`: `mma.sync` s8 "
            "reaches only part of it",
            "FP8 W8A8 (`fp8_w8a8`) where the activations have outlier channels (same rate, "
            "relative steps)",
            ("triton", "cuda", "cute"),
            "never a `mma.sync` kernel for a compute-bound INT8 GEMM on sm_90",
        ),
        "fp8_gemm": TargetClass(
            "fp8_gemm",
            _FP8_LABEL,
            "`wgmma` on e4m3 (TMA loads, warp-specialised producer / consumer warpgroups, "
            "persistent tiles): Triton `tl.dot` on e4m3 (it emits wgmma on sm_90), cuBLASLt "
            "as the baseline (`torch._scaled_mm`, `examples/cuda_cublaslt_fp8.py`), CuTe DSL "
            "/ CUTLASS sm_90 GEMMs (`cute.nvgpu.warpgroup`) when the epilogue fuses scales, "
            "bias or quantisation (`examples/cute_sm90_gemm_ws.py`: TMA + wgmma, producer / "
            "consumer warpgroups, persistent, fused epilogue, swap-AB for M ≤ 64; the "
            "`cute-dsl` skill's `sm90-wgmma.md`)",
            "Hopper's FP8 peak (2x bf16) is reached by `wgmma` only: e4m3 `mma.sync` is "
            "emulated through fp16 HMMA on sm_90, bf16 `mma.sync` reaches ~2/3 of the wgmma "
            "peak, and there is no block-scaled MMA",
            "DeepSeek-style blockwise FP8 (1 x 128 activation, 128 x 128 weight scales, "
            "partial sums promoted to fp32 every 128 along K) with CUTLASS sm_90 blockwise "
            "GEMMs",
            ("triton", "cuda", "cute"),
            "never a `mma.sync` kernel for a compute-bound GEMM on sm_90 (below the wgmma "
            "rate); never `tl.dot_scaled` / MXFP8 (no block-scaled MMA: emulated)",
        ),
    },
    "blackwell": {
        "int8_gemm": TargetClass(
            "int8_gemm",
            _INT8_LABEL,
            "`tcgen05.mma kind::i8` (TMEM int32 accumulators): cuBLASLt int8 (`torch._int_mm`) "
            "as the baseline, CuTe DSL / CUTLASS sm_100 GEMMs for fused epilogues, Triton "
            "`tl.dot` on int8",
            "B200 runs INT8 at its FP8 rate, but Blackwell Ultra (sm_103, B300) cuts INT8 "
            "tensor throughput to ~1/30 of FP8 and has no `kind::i8` (NVIDIA's specifications, "
            "arXiv 2608.11693): the measured INT8 peak (*Ceilings*: *INT8 W8A8*) decides, else "
            "FP8 W8A8",
            "FP8 W8A8 (`fp8_w8a8`) or MXFP8 (`fp8_mx`) where the run allows them",
            ("triton", "cuda", "cute"),
            "never a `mma.sync` kernel for a compute-bound INT8 GEMM on sm_100",
        ),
        "fp8_gemm": TargetClass(
            "fp8_gemm",
            _FP8_LABEL,
            "`tcgen05.mma` (TMEM accumulators, TMA, 2-CTA pairs): cuBLASLt tensor-wise or "
            "MXFP8 as the baseline (`torch._scaled_mm`, `F.scaled_mm` BlockWise1x32, "
            "`examples/cuda_cublaslt_fp8.py`), Triton `tl.dot` on e4m3 (`tcgen05.mma "
            "kind::f8f6f4`: `examples/triton_fp8_w8a8_gemm.py` takes this path here), CuTe "
            "DSL Blackwell GEMMs (`cute.nvgpu.tcgen05`) when the epilogue fuses scales, bias "
            "or quantisation (`examples/cute_sm100_gemm_tcgen05.py`: TMA + tcgen05.mma into "
            "TMEM, persistent, optional 2-CTA pairs, fused epilogue; the `cute-dsl` skill's "
            "`sm100-tcgen05.md`)",
            "datacenter Blackwell reaches its FP8 peak (plain and block-scaled) through "
            "`tcgen05.mma` only: e4m3 `mma.sync` is emulated through fp16 HMMA on sm_100, "
            "`mma.sync` saturates near a quarter of the B200 peak, `wgmma` does not exist",
            "MXFP8 (`fp8_mx`) where the run allows it: block scales at the same MMA rate",
            ("triton", "cuda", "cute"),
            "never a `mma.sync` kernel for a compute-bound GEMM on sm_100 (the sm_120 CuTe "
            "example's `MmaMXF8Op` and the CUDA FP8 examples use it)",
        ),
    },
}
#: What each family needs for the tensor-core peak, and its Triton notes (policy_text /
#: engineer_note lines; GeForce Blackwell's were measured on an RTX 5070 Ti).
ARCH_RULES: dict[str, tuple[str, ...]] = {
    "pre_ampere": (
        "No bf16 tensor cores before sm_80 (fp16 is the tensor-core dtype; Turing's form is "
        "`mma.sync` m16n8k8, `m16n8k16` needs sm_80). Triton's `tl.dot` runs on FMA units "
        "below sm_80 for fp16 / bf16 / tf32 and does not compile on int8 (Triton 3.8): keep "
        "GEMMs in cuBLAS or CUDA C++ fp16 `mma.sync`, use Triton for memory-bound glue.",
        "CuTe DSL 4.8 has no target below sm_80 (the toolchain block's `backends "
        "unavailable` line says it for the installed version); CUTLASS C++ has sm_75 "
        "tensor-core GEMMs.",
    ),
    "ampere": (
        "No FP8 tensor cores (sm_80 / sm_86): `fp8_weights`, `fp4_weights` and `fp8_kv` "
        "targets dequantise to bf16 in registers (software e4m3 conversion: CUDA C++, or "
        "Triton on `uint8` codes); FP8 W8A8 / MXFP8 do not exist here. The 8-bit compute "
        "class is INT8 W8A8 (`int8_w8a8`: IMMA `mma.sync` m16n8k32 s8, Triton `tl.dot` on "
        "int8, `torch._int_mm`), and `int8_weights` halves weight streams with a cheap int8 "
        "conversion (no e4m3 emulation).",
        "No TMA, clusters or PDL: `cp.async` multi-stage pipelines (Triton `num_stages`).",
    ),
    "ada": (
        "Triton FP8: `tl.dot` on e4m3 (QMMA, the full FP8 rate on sm_89); `tl.dot_scaled` is "
        "emulated through bf16 here.",
        "No TMA, clusters or PDL on sm_89: `cp.async` pipelines.",
    ),
    "hopper": (
        "Compute-bound GEMM-like work (GEMMs, attention, convolutions as GEMMs) needs `wgmma` "
        "for the tensor-core peak: Triton `tl.dot` (wgmma on sm_90), CuTe DSL "
        "`cute.nvgpu.warpgroup`, CUTLASS sm_90 kernels or cuBLAS; a hand-written `mma.sync` "
        "kernel stops below it.",
        "Triton on sm_90: `tl.dot` on e4m3 for FP8; sweep TMA descriptors and "
        "`warp_specialize=True` (Hopper is what they were built for).",
        "CuTe DSL on sm_90: start from `examples/cute_sm90_gemm_ws.py` (the `cute-dsl` "
        "skill's `sm90-wgmma.md`: wgmma, TMA, mbarrier phases, `setmaxnreg`); none of it "
        "runs on sm_100 or sm_120.",
    ),
    "blackwell": (
        "Compute-bound GEMM-like work needs `tcgen05.mma` for the tensor-core peak: Triton "
        "`tl.dot` (tcgen05 on sm_100), CuTe DSL `cute.nvgpu.tcgen05`, CUTLASS sm_100 "
        "kernels or cuBLAS(Lt); a hand-written `mma.sync` kernel stops below it, and "
        "`wgmma` does not exist on sm_100.",
        "Triton FP8 on sm_100: `tl.dot` on e4m3 (`tcgen05.mma kind::f8f6f4`, full rate); "
        "`tl.dot_scaled` only with 128-row tiles (64-row tiles fall back to `kind::f16`, an "
        "upcast: check the PTX, Triton 3.8).",
        "Shared memory is ~227 KB per block: deeper pipelines and larger tiles than the "
        "sm_120 examples (99 KB) use.",
        "CuTe DSL on sm_100: start from `examples/cute_sm100_gemm_tcgen05.py` (the "
        "`cute-dsl` skill's `sm100-tcgen05.md`: TMEM 512-column budget, `tcgen05.alloc` / "
        "`dealloc`, 2-CTA pairs); Triton stays first for GEMM-shaped glue (`tl.dot` lowers "
        "to tcgen05 here).",
    ),
    "blackwell_geforce": (
        "Triton FP8: start from `tl.dot_scaled` with unit ue8m0 scales (`QMMA.SF`); never "
        "`warp_specialize` on sm_120 (slower, and it fails to compile `tl.dot_scaled`; "
        "measured on an RTX 5070 Ti).",
    ),
}


def _geforce_fp8(facts: Facts | None) -> TargetClass | None:
    """GeForce Blackwell's FP8 row with this GPU's measured instruction rates (None: not
    measured, :data:`POLICY`'s row with the RTX 5070 Ti numbers stays)."""
    from kernel_agent.kernels.mma_peaks import FP8_F32, FP8_SF

    mma = facts.mma if facts is not None else {}
    f32, sf = mma.get(FP8_F32), mma.get(FP8_SF)
    if not f32 or not sf:
        return None
    row = _BY_ID["fp8_gemm"]
    measured = (
        f"measured on this GPU: `QMMA.F32` (plain e4m3 `mma.sync`) {f32:.0f} TFLOP/s, "
        f"`QMMA.SF` {sf:.0f} with the same fp32 accumulation"
    )
    if sf >= 1.3 * f32:
        why = f"{measured}; {_GEFORCE_FP8_BASELINE}"
        return TargetClass(row.id, row.label, row.first, why, row.second, row.order, row.never)
    return TargetClass(
        row.id,
        row.label,
        row.first + "; plain e4m3 `mma.sync` is as fast on this GPU",
        f"{measured}: both run at the same rate here, either path; {_GEFORCE_FP8_BASELINE}",
        row.second,
        row.order,
    )


#: The bundled examples a row may name (``examples/<name>.py`` or ``<name>.py``).
EXAMPLES_DIR = Path(__file__).parent / "agent" / "examples"
_EXAMPLE = re.compile(r"\b[a-z0-9_]+\.py\b")
_PARENS = re.compile(r"\s*\([^()]*\)")


@functools.cache
def _declared(name: str) -> tuple[bool, str | None]:
    """Whether ``name`` is a bundled example file, and its ``ARCHS``."""
    path = EXAMPLES_DIR / name
    return (True, example_requirement(path)[0]) if path.is_file() else (False, None)


def unrunnable(text: str, capability: tuple[int, ...] | None) -> list[str]:
    """The bundled examples ``text`` names whose ``ARCHS`` exclude a GPU of ``capability``
    (none without a capability)."""
    if capability is None:
        return []
    out = []
    for name in dict.fromkeys(_EXAMPLE.findall(text)):
        bundled, spec = _declared(name)
        if bundled and not supports(spec, capability):
            out.append(name)
    return out


def _clauses(text: str) -> list[str]:
    """``text`` split at its ``"; "`` outside parentheses."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and text.startswith("; ", i):
            parts.append(text[start:i])
            start = i + 2
    return [*parts, text[start:]]


def runnable_text(text: str, capability: tuple[int, ...] | None) -> str:
    """``text`` without what names a bundled example a GPU of ``capability`` cannot run
    (:func:`unrunnable`): the parenthesis that names it, else its ``;`` clause ("" when
    nothing is left). A row never recommends an example that does not run here (#254)."""
    bad = unrunnable(text, capability)
    if not bad:
        return text
    names = re.compile("|".join(re.escape(n) for n in bad))
    kept = []
    for clause in _clauses(text):
        while names.search(clause):
            group = next((g for g in _PARENS.finditer(clause) if names.search(g.group())), None)
            if group is None:
                break
            clause = clause[: group.start()] + clause[group.end() :]
        if not names.search(clause):
            kept.append(clause)
    return "; ".join(kept)


def _runnable_row(row: TargetClass, capability: tuple[int, ...] | None) -> TargetClass:
    """``row`` without the bundled examples a GPU of ``capability`` cannot run; a first or
    second backend left empty becomes the ``other`` row's."""
    first = runnable_text(row.first, capability) or _BY_ID["other"].first
    second = runnable_text(row.second, capability) or _BY_ID["other"].second
    why, never = runnable_text(row.why, capability), runnable_text(row.never, capability)
    if (first, why, second, never) == (row.first, row.why, row.second, row.never):
        return row
    return replace(row, first=first, why=why, second=second, never=never)


def arch_rules(facts: Facts | None) -> tuple[str, ...]:
    """:data:`ARCH_RULES` of the GPU of ``facts`` (unknown: GeForce Blackwell's), without a
    rule that names an example this GPU cannot run."""
    fam = facts.family if facts is not None else None
    cap = facts.capability if facts is not None else None
    rules = ARCH_RULES.get(fam.key if fam else "blackwell_geforce", ())
    return tuple(r for r in rules if not unrunnable(r, cap))


def policy(class_id: str, facts: Facts | None = None) -> TargetClass:
    """The policy row of ``class_id`` for the GPU of ``facts`` (None: :data:`POLICY`'s),
    naming only bundled examples whose ``ARCHS`` hold this GPU."""
    fam = facts.family if facts is not None else None
    row = None
    if fam is not None and class_id == "fp8_gemm" and fam.key == "blackwell_geforce":
        row = _geforce_fp8(facts)  # this GPU's measured FP8 instruction rates
    elif fam is not None:
        row = ARCH_POLICY.get(fam.key, {}).get(class_id)
    row = row or _BY_ID.get(class_id, _BY_ID["other"])
    return _runnable_row(row, facts.capability if facts is not None else None)


def _fp8_runs(facts: Facts | None) -> bool:
    """Whether the GPU of ``facts`` has FP8 tensor cores (unknown GPU: assume so)."""
    return facts is None or facts.family is None or facts.has("fp8_tc")


def _int8_runs(facts: Facts | None) -> bool:
    """Whether the GPU of ``facts`` has INT8 tensor cores (IMMA, sm_75+; unknown: assume so)."""
    from kernel_agent.gpu_arch import precision_unsupported

    cap = facts.capability if facts is not None else None
    return precision_unsupported("int8_w8a8", cap) is None


def _shape(signature: str) -> tuple[int, ...]:
    """Shape of the first tensor of a case signature (``a0[32, 11, 1024]:bfloat16``)."""
    match = _SIG.search(signature)
    if not match or not match.group(1).strip():
        return ()
    return tuple(int(x) for x in match.group(1).split(",") if x.strip())


def dominant_shape(spec: dict[str, Any]) -> tuple[int, ...]:
    """First-tensor shape of the case with the most calls per run (correctness-only cases,
    0 calls, only when nothing else is captured)."""
    cases = list((spec.get("capture") or {}).get("cases") or [])
    if not cases:
        return ()
    best = max(cases, key=lambda c: int(c.get("count") or 0))
    return _shape(str(best.get("signature") or ""))


def rows_of(shape: tuple[int, ...]) -> int | None:
    """GEMM rows ``M`` of an input ``[..., K]``: the product of every dimension but the last."""
    return math.prod(shape[:-1]) if len(shape) >= 2 else None


def target_class(spec: dict[str, Any]) -> str:
    """The :data:`POLICY` row of a target (``spec.json``): module family, precision and the
    dominant case's rows ``M`` and sequence length."""
    from kernel_agent import library, precisions

    cls = str(spec.get("module_class") or spec.get("parent_class") or "")
    family = library.module_family(cls) if cls else ""
    name = cls.lower()
    shape = dominant_shape(spec)
    rows = rows_of(shape)
    seq = shape[-2] if len(shape) >= 3 else None
    fp8 = precisions.of_spec(spec) in _FP8
    int8 = precisions.of_spec(spec) in _INT8
    if family == "attention":
        return "short_attention" if seq is not None and seq <= 16 else "other"
    if family == "block" or spec.get("kind") == "region":
        return "decoder_layer"
    if family == "conv" or any(w in name for w in ("vae", "autoencoder")):
        return "conv"
    if family in ("linear", "mlp"):
        if rows is not None and rows < 128:
            return "small_m_gemm"
        return "fp8_gemm" if fp8 else "int8_gemm" if int8 else "bf16_gemm"
    if family in ("norm", "activation", "rope", "embedding"):
        return "elementwise"
    return "other"


def suggested_order(
    class_id: str, available: Iterable[str], facts: Facts | None = None
) -> list[str]:
    """The class's backend order (on the GPU of ``facts``) restricted to ``available``
    (then the rest)."""
    avail = list(dict.fromkeys(available))
    ordered = [b for b in policy(class_id, facts).order if b in avail]
    return ordered + [b for b in avail if b not in ordered]


def policy_text(available: Iterable[str] | None = None, facts: Facts | None = None) -> str:
    """The planner's backend policy for the GPU of ``facts`` (None: unknown, the sm_120
    table): the class table and the rules that go with it."""
    avail = set(available) if available is not None else None
    fam = facts.family if facts is not None else None
    where = (
        "measured on sm_120 / RTX 5070 Ti, docs/RESEARCH-TRITON.md §5.1"
        if fam is None
        else f"for this GPU, {facts.label if facts else ''}; rows not specific to "
        f"{fam.name} were measured on an RTX 5070 Ti (sm_120), docs/RESEARCH-TRITON.md §5.1"
    )
    lines = [
        f"**Backend policy by target class** ({where}). Classify each target, then order "
        "its `backends` like this (backends not available here are skipped):",
        "",
        "| target class | first backend | why (evidence) | second |",
        "|---|---|---|---|",
    ]
    for base in POLICY:
        c = policy(base.id, facts)
        if c.id == "other" or (c.id == "fp8_gemm" and not _fp8_runs(facts)):
            continue
        if c.id == "int8_gemm" and not _int8_runs(facts):
            continue
        if avail is not None and c.order and not set(c.order) & avail:
            continue
        lines.append(f"| {c.label} | {c.first} | {c.why} | {c.second} |")
    lines.append("")
    if _fp8_runs(facts) and (never := policy("fp8_gemm", facts).never):
        block = facts is None or facts.has("block_scaled")
        lines.append(
            "* Compute-bound FP8 GEMMs (W8A8 `fp8_w8a8`"
            + (" or MXFP8 `fp8_mx`" if block else "")
            + f", M ≳ 128): {never}."
            + (" The same block-scaled MMA runs real MXFP8 scales (`fp8_mx`)." if block else "")
        )
    lines.append(
        "* An eager-timed target with ≥ 3 kernel launches per call: one C++ launcher "
        "(host time, not GPU time, decides there)."
    )
    lines += [f"* {rule}" for rule in arch_rules(facts)]
    lines.append("* TileLang has no evidence either way on this GPU: list it, not first.")
    return "\n".join(lines)


def engineer_note(
    target: dict[str, Any], backends: Iterable[str] = (), facts: Facts | None = None
) -> str:
    """Engineer-prompt lines for the GPU of ``facts`` (None: unknown, the sm_120 rows): the
    target's class and its policy row (and, for a fused FP8 layer, the FP8 GEMM row for the
    GEMMs inside it)."""
    from kernel_agent import precisions

    planned = list(backends)
    fam = facts.family if facts is not None else None
    cid = target_class(target)
    c = policy(cid, facts)
    lines = [f"Target class: **{c.label}** (backend policy, docs/RESEARCH-TRITON.md §5.1)."]
    if cid != "other":
        lines += [f"* first: {c.first}", f"* why: {c.why}", f"* second: {c.second}"]
    if c.never:
        lines.append(f"* {c.never}.")
    rows = rows_of(dominant_shape(target))
    fp8 = precisions.of_spec(target) in _FP8
    int8 = precisions.of_spec(target) in _INT8
    for inside, wanted in (("fp8_gemm", fp8), ("int8_gemm", int8)):
        if cid != inside and wanted and (rows or 0) >= 128:
            gemm = policy(inside, facts)
            never = f" {gemm.never}." if gemm.never else ""
            lines.append(f"* The GEMMs inside (M = {rows}): {gemm.first}.{never}")
    rules = arch_rules(facts)
    if fp8 and "triton" in planned:  # the family's Triton FP8 rule
        lines += [f"* {r}" for r in rules if r.startswith("Triton")]
    if fam is not None:  # what this family needs for the tensor-core peak
        lines += [f"* {r}" for r in rules if not r.startswith("Triton")]
    lines.append(
        "* The module evaluator times eagerly: host time per launch counts (CUDA C++ "
        "`load_inline` ~19 us, CuTe DSL TVM-FFI ~25 us, Triton ~43 us on the development "
        "machine); with ≥ 3 launches per call, put them behind one C++ launcher."
    )
    order = suggested_order(cid, planned, facts)
    if c.order and planned and order[0] != planned[0]:
        lines.append(
            f"* The plan lists `{planned[0]}` first; the policy for this class would start "
            f"with `{order[0]}`. Follow the plan unless its evidence is weaker."
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ outcomes of a run


@dataclass
class Outcome:
    """What one backend achieved on one target of a run."""

    target: str
    backend: str
    evaluations: int = 0  # benchmark evaluations (quick checks, duplicates excluded)
    correct: int = 0
    kept: int = 0
    quick_ok: int = 0
    quick_fail: int = 0
    best: float | None = None  # best correct speedup (a re-evaluation replaces its row's)
    best_snapshot: str | None = None
    snapshots: list[str] = field(default_factory=list)


def _source(run: RunDir, target_id: str, snapshot: str) -> str | None:
    path = run.history_dir(target_id) / Path(snapshot).name
    try:
        return path.read_text(errors="replace") if snapshot and path.is_file() else None
    except OSError:
        return None


def outcomes(run: RunDir, rows: list[dict[str, Any]] | None = None) -> list[Outcome]:
    """Per target and backend (classified from each snapshot's source, :func:`classify`;
    the ledger's label when the snapshot is gone): evaluations, correct, kept, quick checks
    and the best correct speedup. Library priors (re-evaluations of an earlier run's
    winner) are left out; a ``re-evaluated`` row replaces its snapshot's correctness and
    speedup."""
    rows = ledger.rows(run) if rows is None else rows
    kernel_rows = [r for r in rows if r["target"] != ledger.E2E]
    reeval = {
        (r["target"], r["snapshot"]): r for r in kernel_rows if r["status"] == ledger.REEVALUATED
    }
    cache: dict[tuple[str, str], str] = {}
    out: dict[tuple[str, str], Outcome] = {}
    for r in kernel_rows:
        status = r["status"]
        if status in (ledger.REEVALUATED, ledger.DUPLICATE):
            continue
        if not_agents(r.get("hypothesis")):  # library priors and scout rows
            continue
        key = (r["target"], str(r["snapshot"] or ""))
        if key not in cache:
            src = _source(run, *key)
            cache[key] = classify(src) if src is not None else (r["backend"] or "torch")
        backend = cache[key]
        o = out.setdefault((r["target"], backend), Outcome(r["target"], backend))
        if status == ledger.QUICK_OK:
            o.quick_ok += 1
            continue
        if status == ledger.QUICK_FAIL:
            o.quick_fail += 1
            continue
        again = reeval.get(key)
        correct = bool(again["correct"] if again else r["correct"])
        speedup = (again or r).get("speedup")
        o.evaluations += 1
        o.correct += correct
        o.kept += status == ledger.KEEP
        if r["snapshot"]:
            o.snapshots.append(str(r["snapshot"]))
        if correct and speedup is not None and (o.best is None or speedup > o.best):
            o.best, o.best_snapshot = float(speedup), str(r["snapshot"])
    return sorted(out.values(), key=lambda o: (o.target, o.backend))


def winners(found: Iterable[Outcome]) -> dict[str, str]:
    """Target → the backend with its best correct speedup (only targets with one > 1)."""
    best: dict[str, Outcome] = {}
    for o in found:
        if o.best is None or o.best <= 1.0:
            continue
        if o.target not in best or o.best > (best[o.target].best or 0.0):
            best[o.target] = o
    return {t: o.backend for t, o in best.items()}


def by_backend(found: list[Outcome]) -> list[dict[str, Any]]:
    """Per backend over a run's targets (the table of RESEARCH-TRITON §4.1): targets tried,
    targets won, evaluations, correct, kept, quick checks failed, best speedup and where."""
    won = winners(found)
    out: dict[str, dict[str, Any]] = {}
    for o in found:
        agg = out.setdefault(
            o.backend,
            {
                "backend": o.backend,
                "targets": 0,
                "won": 0,
                "evaluations": 0,
                "correct": 0,
                "kept": 0,
                "quick_fail": 0,
                "best": None,
                "best_target": None,
            },
        )
        agg["targets"] += 1
        agg["won"] += won.get(o.target) == o.backend
        agg["evaluations"] += o.evaluations
        agg["correct"] += o.correct
        agg["kept"] += o.kept
        agg["quick_fail"] += o.quick_fail
        if o.best is not None and (agg["best"] is None or o.best > agg["best"]):
            agg["best"], agg["best_target"] = o.best, o.target
    return sorted(out.values(), key=lambda a: (-a["won"], -a["evaluations"], a["backend"]))


def _x(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}x"


def _pct(part: int, whole: int) -> str:
    return f"{part} ({100 * part / whole:.0f} %)" if whole else str(part)


def _specs(run: RunDir) -> dict[str, dict[str, Any]]:
    return {t: read_json(run.target(t) / "spec.json", {}) or {} for t in run.target_ids()}


def report_lines(run: RunDir, rows: list[dict[str, Any]] | None = None) -> list[str]:
    """``## Backends`` section of report.md: per backend, then per target (planned vs
    tried, by source). Empty without kernel evaluations."""
    found = outcomes(run, rows)
    if not found:
        return []
    specs = _specs(run)
    won = winners(found)
    lines = [
        "",
        "## Backends",
        "",
        "Kernel evaluations by backend, read from each snapshot's source (what it runs, not "
        "its imports); library priors left out, a re-evaluation replaces its snapshot's "
        "numbers. Policy per target class: docs/RESEARCH-TRITON.md §5.1.",
        "",
        "| backend | targets (won) | evaluations | correct | kept | quick checks failed "
        "| best module speedup |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in by_backend(found):
        where = f" (`{a['best_target']}`)" if a["best_target"] else ""
        lines.append(
            f"| {a['backend']} | {a['targets']} ({a['won']}) | {a['evaluations']} | "
            f"{_pct(a['correct'], a['evaluations'])} | {a['kept']} | {a['quick_fail']} | "
            f"{_x(a['best'])}{where} |"
        )
    lines += [
        "",
        "| target | class | planned | tried (by source) | best |",
        "|---|---|---|---|---|",
    ]
    per_target: dict[str, list[Outcome]] = {}
    for o in found:
        per_target.setdefault(o.target, []).append(o)
    for target_id, items in per_target.items():
        spec = specs.get(target_id, {})
        planned = ", ".join(spec.get("backends") or []) or "—"
        tried = ", ".join(
            f"{o.backend} {o.correct}/{o.evaluations}" + (f" {_x(o.best)}" if o.best else "")
            for o in items
        )
        winner = won.get(target_id)
        best = max((o.best for o in items if o.best is not None), default=None)
        lines.append(
            f"| `{target_id}` | {policy(target_class(spec)).label if spec else '—'} | "
            f"{planned} | {tried} | {_x(best)}{f' ({winner})' if winner else ''} |"
        )
    return lines


def status_lines(run: RunDir, rows: list[dict[str, Any]] | None = None) -> list[str]:
    """Compact per-backend lines for ``kernel-agent status``."""
    found = outcomes(run, rows)
    if not found:
        return []
    lines = ["backends (by source)"]
    for a in by_backend(found):
        where = f" ({a['best_target']})" if a["best_target"] else ""
        lines.append(
            f"  {a['backend']:<12} {a['targets']} target(s), {a['won']} won  ·  "
            f"{a['evaluations']} evals, {a['correct']} correct, {a['kept']} kept  ·  "
            f"best {_x(a['best'])}{where}"
        )
    return lines


# ------------------------------------------------------------------ the library's record

RECORD = "backends.jsonl"


def record_path(arch: str) -> Path:
    from kernel_agent import library

    return library.root() / re.sub(r"[^A-Za-z0-9_.-]+", "_", arch) / RECORD


def run_records(run: RunDir, rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One record per target of the run with kernel evaluations: class, planned backends,
    per-backend outcome and the winner."""
    found = outcomes(run, rows)
    specs = _specs(run)
    won = winners(found)
    card = (run.load().get("card") or {}) if run.run_json.exists() else {}
    out = []
    for target_id in dict.fromkeys(o.target for o in found):
        spec = specs.get(target_id, {})
        items = [o for o in found if o.target == target_id]
        out.append(
            {
                "run": str(run.root),
                "repo_id": card.get("repo_id"),
                "target": target_id,
                "module_class": spec.get("module_class"),
                "class": target_class(spec),
                "precision": str(spec.get("precision") or "exact"),
                "rows": rows_of(dominant_shape(spec)),
                "planned": list(spec.get("backends") or []),
                "winner": won.get(target_id),
                "backends": {
                    o.backend: {
                        "evaluations": o.evaluations,
                        "correct": o.correct,
                        "kept": o.kept,
                        "quick_fail": o.quick_fail,
                        "best": o.best,
                    }
                    for o in items
                },
            }
        )
    return out


def record_run(run: RunDir, arch: str, rows: list[dict[str, Any]] | None = None) -> Path | None:
    """Write the run's per-target backend outcomes into the library's record of ``arch``
    (this run's earlier lines are replaced: a re-integration records the run again)."""
    records = run_records(run, rows)
    if not records:
        return None
    path = record_path(arch)
    kept = [r for r in read_records(path) if r.get("run") != str(run.root)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in [*kept, *records]))
    tmp.replace(path)
    return path


def read_records(path: Path) -> list[dict[str, Any]]:
    out = []
    try:
        text = path.read_text()
    except OSError:
        return []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def track_record(arch: str, records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Per (class, backend) over every recorded run of ``arch``: targets tried and won,
    evaluations, correct, best speedup."""
    records = read_records(record_path(arch)) if records is None else records
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        for backend, o in (rec.get("backends") or {}).items():
            if not isinstance(o, dict):
                continue
            key = (str(rec.get("class") or "other"), str(backend))
            agg = out.setdefault(
                key,
                {
                    "class": key[0],
                    "backend": key[1],
                    "targets": 0,
                    "won": 0,
                    "evaluations": 0,
                    "correct": 0,
                    "best": None,
                },
            )
            agg["targets"] += 1
            agg["won"] += rec.get("winner") == backend
            agg["evaluations"] += int(o.get("evaluations") or 0)
            agg["correct"] += int(o.get("correct") or 0)
            best = o.get("best")
            if isinstance(best, int | float) and (agg["best"] is None or best > agg["best"]):
                agg["best"] = float(best)
    order = [c.id for c in POLICY]
    return sorted(
        out.values(),
        key=lambda a: (
            order.index(a["class"]) if a["class"] in order else len(order),
            -a["won"],
            -(a["best"] or 0.0),
        ),
    )


def track_record_note(arch: str | None, limit: int = 16) -> str:
    """Planner-prompt section: which backend won which target class in earlier runs on
    this GPU architecture ("" without a record)."""
    if not arch:
        return ""
    found = track_record(arch)
    if not found:
        return ""
    lines = [
        f"**Backend track record on {arch}** (kernel library, earlier runs; won = held the "
        "target's best correct kernel):",
        "",
        "| target class | backend | targets (won) | evaluations (correct) | best |",
        "|---|---|---|---|---|",
    ]
    for a in found[:limit]:
        lines.append(
            f"| {policy(a['class']).label} | {a['backend']} | {a['targets']} ({a['won']}) | "
            f"{a['evaluations']} ({a['correct']}) | {_x(a['best'])} |"
        )
    lines.append("Prefer a backend that won its class here unless the policy's evidence is newer.")
    return "\n".join(lines)
