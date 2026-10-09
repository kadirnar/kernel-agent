"""Speed of light (roofline) of every captured case, against peaks measured on this GPU.

* **Peaks** (:func:`measure_peaks`): device-to-device copy bandwidth from DRAM
  and from L2 (bytes read + bytes written), dense matmul TFLOP/s per dtype (best
  of a few large shapes; FP8 e4m3 via ``torch._scaled_mm`` and NVFP4 where torch
  has a kernel for the GPU, else ``tflops_unavailable`` says why) and the launch
  floor (median time of a module call that
  launches one tiny kernel, timed like a candidate), the INT8 GEMM peak (``int8``:
  ``torch._int_mm``, cuBLASLt's IMMA kernels, in TOPS), and the tensor-core
  instruction rates (``mma_tflops``, :mod:`.mma_peaks`: plain FP8 ``QMMA.F32``
  vs block-scaled ``QMMA.SF`` vs bf16 / fp16 (fp32 and fp16 accumulation) / TF32 ``HMMA``
  vs s8 ``IMMA``, Turing's shapes too). The bf16 / fp16 peaks are also measured
  under sustained load (``tflops_sustained``: :data:`SUSTAINED_S` of back-to-back
  GEMMs, :func:`sustain`), with the SM clock, power and clock-event reasons sampled
  meanwhile (``sustained``): a power-capped board slows down there, and the
  ceilings' floors use that rate (:func:`floor_tflops`, #253).  :func:`ensure_peaks`
  measures them once per GPU + torch version in a subprocess under the GPU lock
  and caches them in ``<cache>/peaks-<gpu>-torch<version>.json``; they are never
  measured inside a timed evaluation.
* **Work** (:func:`count_case`): FLOPs come from
  ``torch.utils.flop_counter.FlopCounterMode`` on the reference call, attributed
  to the dtype of each op's inputs; attention (SDPA) FLOPs only count the
  query/key pairs the mask (or ``is_causal``) allows.  ``min_bytes`` counts every
  byte range of the inputs, parameters and buffers the reference reads (each
  once), the outputs and new state it writes, and the elements of inputs it
  updates in place (KV-cache appends: the touched slice, not the whole cache).
* **Speed of light**: ``sol_ms = max(Σ flops_dtype / peak_dtype, min_bytes /
  bandwidth)``.  The bandwidth is the L2 one when ``min_bytes`` fits in L2 and
  the timing runs with a warm L2 (the evaluator's default), else DRAM.  ``bound``
  is ``compute`` or ``memory``, or ``launch`` when ``sol_ms`` is below the launch
  floor (one launch from Python costs more than the work itself).

:func:`annotate` adds per case ``flops``, ``min_bytes``, ``sol_ms``,
``pct_of_sol`` (100 × sol_ms / new_ms) and ``bound``, plus the weighted
``pct_of_sol`` of the whole target.  ``new_ms < 0.9 × sol_ms`` is physically
implausible and flagged ``suspicious_faster_than_sol``; when the *reference*
already beats 0.9 × sol_ms, the estimate is wrong and the case is flagged
``sol_unreliable`` instead.

Reduced precision (the capture's ``precision``): the weights of an ``fp8_weights`` or
``fp8_w8a8`` target count at one byte per element (:data:`WEIGHT_BITS`), and the GEMMs on
the weights of an ``fp8_w8a8`` target at the measured FP8 peak (:data:`MATH_DTYPE`); without
an FP8 peak (none measured on this GPU, or a cache older than :data:`PEAKS_VERSION`) such
cases are flagged ``sol_unreliable`` with a ``sol_note``. An ``fp8_mx`` (MXFP8) target's
weights count at one byte plus one scale byte per 32, its GEMMs at the measured MXFP8 peak
(:data:`MXFP8`: the block-scaled tensor cores through ``F.scaled_mm`` ``BlockWise1x32``).
INT8 (#178): an ``int8_weights`` or ``int8_w8a8`` target's weights count at one byte, and the
GEMMs on the weights of an ``int8_w8a8`` target at the measured INT8 peak (:data:`INT8`).
W4A4 (#233): an ``fp4_w4a4`` target's weights count as NVFP4 (4 bits plus a scale byte per
16), its GEMMs at the measured NVFP4 peak (:data:`FP4`).

Limitations: bytes are what the reference touches.  Reads through gather ops
(embedding, index, index_select, gather) count the gathered rows, and SDPA counts
only the key/value positions some query attends to, but other masked or
data-dependent reads (e.g. ``repeat_kv`` copies of a whole static cache, or
masks applied outside SDPA) count in full, so a kernel that skips such work can
legitimately beat the estimate and is flagged (it is never rejected).  Strided
views count their whole span.  FLOPs only cover ops ``FlopCounterMode`` knows
(matmul, conv, attention); element-wise math is treated as free.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernel_agent import toolchain

SUSPICIOUS_RATIO = 0.9  # new_ms below this share of sol_ms is faster than the hardware allows
MASKED = -1e4  # additive attention-mask values at or below this mask the position out
#: Bits per weight element of a reduced-precision target (its capture's ``precision``):
#: the 2-D floating-point parameters the reference reads count at this width plus one fp32
#: scale per output channel (row), so ``pct_of_sol`` of an FP8 kernel is measured against
#: the bytes it must stream, not the bf16 weights it replaced.
WEIGHT_BITS = {
    "fp8_weights": 8,
    "fp4_weights": 4,
    "fp8_w8a8": 8,
    "fp8_mx": 8,
    "int8_weights": 8,
    "int8_w8a8": 8,
    "fp4_w4a4": 4,
}
#: ... or, for block-scaled formats, plus one 1-byte scale per this many elements and one
#: fp32 scale per tensor (``fp4_weights`` and ``fp4_w4a4``: NVFP4, an e4m3 scale per 16;
#: ``fp8_mx``: MXFP8, an e8m0 scale per 32).
WEIGHT_SCALE_BLOCK = {"fp4_weights": 16, "fp8_mx": 32, "fp4_w4a4": 16}
#: ``peaks["tflops"]`` keys of the low-precision tensor-core peaks: FP8 e4m3 and NVFP4.
FP8, FP4 = "float8_e4m3fn", "float4_e2m1fn_x2"
#: ... and MXFP8 (e4m3 with an e8m0 scale per 32 along K on both operands: the block-scaled
#: MMA, ``kind::mxf8f6f4.block_scale``, sm_100 / sm_120; cuBLASLt ``VEC32_UE8M0``).
MXFP8 = "mxfp8"
#: ... and INT8 (s8 x s8 -> int32 on the IMMA tensor cores, ``torch._int_mm``; TOPS: one
#: multiply-add = 2 ops, like a FLOP).
INT8 = "int8"
#: Schema of the cached peaks; 2 adds the FP8 / FP4 peaks, 3 the MXFP8 one, 4 the
#: tensor-core instruction rates (``mma_tflops``), 5 the INT8 peak and the s8 IMMA rate, 6
#: the fp16 (fp32 and fp16 accumulation), TF32 and Turing-shape instruction rates (#257), 7
#: the sustained 16-bit matmul peaks with the SM clock and power under that load (#253).
#: :func:`ensure_peaks` measures an older cache again (once per process at most); until then
#: it stays in use.
PEAKS_VERSION = 7
#: Sustained 16-bit peaks (``tflops_sustained``, #253): seconds of back-to-back GEMMs per
#: dtype, in batches of about :data:`SUSTAINED_BATCH_MS`. A power-capped board (the 150 W
#: passively cooled A10, the 70 W T4 and L4) holds its boost clock for the ~60 ms bursts of
#: the other peaks but not under a model's sustained load.
SUSTAINED_S = 2.0
SUSTAINED_BATCH_MS = 100.0
#: A sustained peak at or below this share of the burst one replaces it in the floors of
#: the ceilings table (:func:`floor_tflops`): the board slows down under sustained load.
SUSTAINED_BELOW = 0.9
#: The tensor-core math of a reduced-precision target: the FLOPs of every op that reads one
#: of its narrowed weights count at this dtype's peak (W8A8: FP8), not at the reference's.
MATH_DTYPE = {"fp8_w8a8": FP8, "fp8_mx": MXFP8, "int8_w8a8": INT8, "fp4_w4a4": FP4}
_MiB = 1024**2

# Ops that look at a tensor argument's metadata only (no data read).
_NO_READ = {
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "rand_like",
    "randn_like",
    "randint_like",
    "new_empty",
    "new_empty_strided",
    "new_zeros",
    "new_ones",
    "new_full",
}
# Ops that read only the rows of their first argument selected by an index.
_GATHER = {"embedding", "index", "_unsafe_index", "index_select", "gather", "take"}
_ATTENTION = {
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_attention_math",
    "_scaled_dot_product_fused_attention_overrideable",
}


# ------------------------------------------------------------------ peaks


def _best_ms(fn: Any, iters: int, trials: int = 3) -> float:
    """Fastest per-iteration GPU time of ``fn`` over a few trials (milliseconds)."""
    import torch

    best = math.inf
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


def _copy_gbps(src: Any, dst: Any, *, graph_reps: int = 0) -> float:
    """Best bandwidth (read + write bytes) of ``dst <- src`` via memcpy or an element-wise kernel.

    ``graph_reps > 0`` replays that many copies from a CUDA graph, so host launch
    overhead cannot hide the bandwidth of small (L2-resident) buffers."""
    import torch

    nbytes = 2 * src.numel() * src.element_size()
    best = 0.0
    for op in (lambda: dst.copy_(src), lambda: torch.neg(src, out=dst)):
        op()
        if graph_reps:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                op()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_reps):
                    op()
            graph.replay()
            ms = _best_ms(graph.replay, iters=10) / graph_reps
        else:
            ms = _best_ms(op, iters=10)
        best = max(best, nbytes / ms / 1e6)
    return best


def _gemm_tflops(fn: Any, m: int, n: int, k: int) -> float:
    """TFLOP/s of ``fn``, one ``[m, k] x [k, n]`` matmul per call."""
    fn()
    est = _best_ms(fn, iters=2, trials=1)
    ms = _best_ms(fn, iters=max(3, int(60 / est)))
    return 2 * m * n * k / ms / 1e9


def _matmul_tflops(dtype: Any, shapes: list[tuple[int, int, int]], free: int) -> float | None:
    import torch

    best = None
    for m, n, k in shapes:
        need = (m * k + k * n + m * n) * torch.empty((), dtype=dtype).element_size()
        if need > free // 2:
            continue
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        c = torch.empty(m, n, device="cuda", dtype=dtype)
        tflops = _gemm_tflops(lambda a=a, b=b, c=c: torch.mm(a, b, out=c), m, n, k)
        best = tflops if best is None else max(best, tflops)
        del a, b, c
    return None if best is None else round(best, 1)


def _fp8_tflops(shapes: list[tuple[int, int, int]], free: int) -> float | None:
    """Dense FP8 e4m3 x e4m3 -> bf16 (fp32 accumulation) via ``torch._scaled_mm``."""
    import torch

    best = None
    for m, n, k in shapes:
        if m * k + k * n + 2 * m * n > free // 2:
            continue
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
        one = torch.ones((), device="cuda")  # tensor-wise scales; b is column-major

        def fp8_mm(a: Any = a, b: Any = b, one: Any = one) -> Any:
            return torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

        tflops = _gemm_tflops(fp8_mm, m, n, k)
        best = tflops if best is None else max(best, tflops)
        del a, b
    return None if best is None else round(best, 1)


def _fp4_tflops(shapes: list[tuple[int, int, int]], free: int) -> float | None:
    """Dense NVFP4 (e2m1, one e4m3 scale per 16 elements) x NVFP4 -> bf16 via
    ``torch.nn.functional.scaled_mm``; raises where torch or the GPU has no such kernel."""
    import torch
    import torch.nn.functional as F

    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    scaled_mm = getattr(F, "scaled_mm", None)
    if fp4 is None or scaled_mm is None:
        raise RuntimeError("this torch has no NVFP4 scaled_mm")
    blockwise, swizzle = F.ScalingType.BlockWise1x16, F.SwizzleType.SWIZZLE_32_4_4
    best = None
    for m, n, k in shapes:
        if (m * k + k * n) // 2 + 2 * m * n > free // 2:
            continue
        # Random codes (every e2m1 code is finite) and scales: the values do not matter.
        a = torch.randint(0, 256, (m, k // 2), device="cuda", dtype=torch.uint8).view(fp4)
        b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8).view(fp4).t()
        scale_a = torch.rand(m, k // 16, device="cuda").to(torch.float8_e4m3fn)
        scale_b = torch.rand(k // 16, n, device="cuda").to(torch.float8_e4m3fn)

        def fp4_mm(a: Any = a, b: Any = b, sa: Any = scale_a, sb: Any = scale_b) -> Any:
            return scaled_mm(
                a,
                b,
                scale_a=[sa],
                scale_recipe_a=[blockwise],
                scale_b=[sb],
                scale_recipe_b=[blockwise],
                swizzle_a=[swizzle],
                swizzle_b=[swizzle],
                output_dtype=torch.bfloat16,
            )

        tflops = _gemm_tflops(fp4_mm, m, n, k)
        best = tflops if best is None else max(best, tflops)
        del a, b, scale_a, scale_b
    return None if best is None else round(best, 1)


def _mxfp8_tflops(shapes: list[tuple[int, int, int]], free: int) -> float | None:
    """Dense MXFP8 (e4m3 + one e8m0 scale per 32 along K, both operands, swizzled 128 x 4
    scale blocks) -> bf16 via ``torch.nn.functional.scaled_mm`` ``BlockWise1x32``: the
    block-scaled tensor cores (cuBLASLt ``VEC32_UE8M0``); raises where torch or the GPU has
    no such kernel (before sm_100)."""
    import torch
    import torch.nn.functional as F

    from kernel_agent.kernels.quant import swizzle_mx_scales

    scaled_mm = getattr(F, "scaled_mm", None)
    if scaled_mm is None or not hasattr(torch, "float8_e8m0fnu"):
        raise RuntimeError("this torch has no MXFP8 scaled_mm")
    blockwise, swizzle = F.ScalingType.BlockWise1x32, F.SwizzleType.SWIZZLE_32_4_4
    best = None
    for m, n, k in shapes:
        if m * k + k * n + 2 * m * n > free // 2:
            continue
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = w.t()  # column-major [K, N]
        unit = torch.full((max(m, n), k // 32), 127, device="cuda", dtype=torch.uint8)  # 2^0
        scale_a, scale_b = swizzle_mx_scales(unit[:m]), swizzle_mx_scales(unit[:n])

        def mx_mm(a: Any = a, b: Any = b, sa: Any = scale_a, sb: Any = scale_b) -> Any:
            return scaled_mm(
                a,
                b,
                scale_a=sa,
                scale_recipe_a=blockwise,
                scale_b=sb,
                scale_recipe_b=blockwise,
                swizzle_a=swizzle,
                swizzle_b=swizzle,
                output_dtype=torch.bfloat16,
            )

        tflops = _gemm_tflops(mx_mm, m, n, k)
        best = tflops if best is None else max(best, tflops)
        del a, w, b, unit, scale_a, scale_b
    return None if best is None else round(best, 1)


def _int8_tops(shapes: list[tuple[int, int, int]], free: int) -> float | None:
    """Dense INT8 s8 x s8 -> int32 via ``torch._int_mm`` (cuBLASLt's IMMA kernels; the
    weight-like operand column-major, as ``quant.int8_matmul`` passes it), in TOPS."""
    import torch

    best = None
    for m, n, k in shapes:
        if m * k + k * n + 4 * m * n > free // 2:
            continue
        a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8).t()

        def int_mm(a: Any = a, b: Any = b) -> Any:
            return torch._int_mm(a, b)

        tops = _gemm_tflops(int_mm, m, n, k)
        best = tops if best is None else max(best, tops)
        del a, b
    return None if best is None else round(best, 1)


# ------------------------------------------------------------------ sustained peaks (#253)


def sustain(
    launch: Callable[[], Any],
    wait: Callable[[Any], float],
    seconds: float,
    monitor: Any = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[float, list[dict[str, Any]]]:
    """``(TFLOP/s, samples)`` of back-to-back batches of one GEMM for ``seconds``:
    ``launch()`` queues a batch and returns its handle, ``wait(handle)`` waits for it and
    returns its TFLOP/s. The next batch is queued before the monitor samples
    (:class:`kernel_agent.telemetry.Monitor`) and the previous one is waited for, so the GPU
    never idles between batches (an idle gap would let a power-capped clock recover). The
    rate is the median of the second half of the batches (the clock has settled there),
    the samples are those taken meanwhile."""
    begin = clock()
    pending = launch()
    rates: list[float] = []
    samples: list[dict[str, Any] | None] = []
    while True:
        queued = launch() if clock() - begin < seconds else None
        samples.append(monitor.sample("sustained") if monitor is not None else None)
        rates.append(float(wait(pending)))
        if queued is None:
            break
        pending = queued
    half = len(rates) // 2
    return round(statistics.median(rates[half:]), 1), [s for s in samples[half:] if s]


def _sustained_tflops(
    dtype: Any, free: int, monitor: Any, seconds: float = SUSTAINED_S
) -> tuple[float, list[dict[str, Any]]] | None:
    """:func:`sustain` on 8192³ ``torch.mm`` of ``dtype`` (4096³ when memory is short;
    None: not even that fits)."""
    import torch

    width = torch.empty((), dtype=dtype).element_size()
    size = next((s for s in (8192, 4096) if 3 * s * s * width <= free // 2), None)
    if size is None:
        return None
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    c = torch.empty(size, size, device="cuda", dtype=dtype)

    def gemm() -> None:
        torch.mm(a, b, out=c)

    gemm()
    per = max(1, int(SUSTAINED_BATCH_MS / _best_ms(gemm, iters=2, trials=1)))

    def launch() -> tuple[Any, Any]:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(per):
            gemm()
        end.record()
        return start, end

    def wait(events: tuple[Any, Any]) -> float:
        start, end = events
        end.synchronize()
        return 2 * size**3 * per / start.elapsed_time(end) / 1e9

    return sustain(launch, wait, seconds, monitor)  # a, b, c are freed with the closures


def sustained_peaks(
    peaks: dict[str, Any],
    monitor: Any,
    measure: Callable[[str, Any], tuple[float, list[dict[str, Any]]] | None],
) -> dict[str, Any]:
    """Add the sustained 16-bit peaks to ``peaks`` (``measure(dtype name, monitor)``:
    :func:`_sustained_tflops`): ``tflops_sustained`` per dtype and ``sustained``, the load
    they ran under: the median SM clock and power of the samples of the settled halves, the
    clock-event reasons seen there, the highest SM clock of every sample of ``monitor``
    (the burst peaks' samples too), the board's maximum clock and its enforced power limit.
    Returns ``peaks``."""
    rates: dict[str, float] = {}
    settled: list[dict[str, Any]] = []
    for name in ("bfloat16", "float16"):
        if not (peaks.get("tflops") or {}).get(name):
            continue
        found = measure(name, monitor)
        if found is not None:
            rates[name], samples = found
            settled += samples
    if not rates:
        return peaks
    peaks["tflops_sustained"] = rates
    info: dict[str, Any] = {"seconds": SUSTAINED_S}
    seen = list(getattr(monitor, "samples", None) or [])
    if settled:
        info["sm_mhz"] = round(statistics.median(s["sm_mhz"] for s in settled))
        info["power_w"] = round(statistics.median(s["power_w"] for s in settled), 1)
        info["reasons"] = sorted({r for s in settled for r in s.get("reasons") or []})
    if seen:
        info["burst_sm_mhz"] = max(s["sm_mhz"] for s in seen)
        if board := [s["sm_max_mhz"] for s in seen if s.get("sm_max_mhz")]:
            info["sm_max_mhz"] = max(board)
    limit = getattr(monitor, "power_limit_w", None)
    if callable(limit) and (watts := limit()) is not None:
        info["power_limit_w"] = watts
    peaks["sustained"] = info
    return peaks


def sustained_drops(peaks: Mapping[str, Any] | None) -> dict[str, tuple[float, float]]:
    """dtype → ``(burst, sustained)`` TFLOP/s of the matmul peaks whose sustained rate is at
    most :data:`SUSTAINED_BELOW` of the burst one (a power-capped board; {}: none)."""
    tflops = (peaks or {}).get("tflops") or {}
    out = {}
    for dtype, rate in ((peaks or {}).get("tflops_sustained") or {}).items():
        burst = tflops.get(dtype)
        if burst and rate and float(rate) <= SUSTAINED_BELOW * float(burst):
            out[dtype] = (float(burst), float(rate))
    return out


def floor_tflops(peaks: Mapping[str, Any] | None) -> dict[str, float]:
    """The matmul peaks a floor of the ceilings table runs at: ``tflops``, each sustained
    peak in place of its burst one where :func:`sustained_drops` says the board slows down
    (a model's end-to-end run is sustained load; a peak without a sustained measurement
    stays as measured: no ratio is assumed)."""
    tflops = {k: float(v) for k, v in ((peaks or {}).get("tflops") or {}).items() if v}
    tflops.update({k: rate for k, (_, rate) in sustained_drops(peaks).items()})
    return tflops


def measure_peaks() -> dict[str, Any]:
    """Measure the roofline peaks of the current GPU (takes ~15-35 s, the sustained peaks
    about 5 of them; hold the GPU lock)."""
    import torch
    from torch import nn

    from kernel_agent.kernels.bench import ensure_clocks, time_call, warm_gpu

    tc = toolchain.setup()
    if tc.gpu is None:
        raise RuntimeError("no CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 peak means real fp32 math
    warm_gpu()
    ensure_clocks()  # the memory clock too: these peaks are what its probes compare with
    free, _ = torch.cuda.mem_get_info()
    peaks: dict[str, Any] = {
        "version": PEAKS_VERSION,
        "gpu": tc.gpu.name,
        "arch": tc.gpu.arch,
        "torch": tc.torch_version,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "l2_mb": round(tc.gpu.l2_cache_mb, 1),
    }

    # Bandwidth: copies of 512 MiB buffers (DRAM) and of buffers a quarter of L2 each.
    n = int(min(512 * _MiB, free // 8)) // 16 * 16
    src = torch.ones(n // 2, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    peaks["dram_gbps"] = round(_copy_gbps(src, dst), 1)
    l2_elems = int(tc.gpu.l2_cache_mb * _MiB) // 4 // 2
    if 0 < l2_elems <= src.numel():
        peaks["l2_gbps"] = round(_copy_gbps(src[:l2_elems], dst[:l2_elems], graph_reps=50), 1)
    del src, dst
    torch.cuda.empty_cache()

    # Dense matmul (tensor cores for 16-bit types; fp32 with TF32 off). The clocks are
    # sampled right after each burst (telemetry: NVML or nvidia-smi; none without them).
    from kernel_agent.telemetry import Monitor

    monitor = Monitor()
    big = [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 8192, 8192)]
    peaks["tflops"] = {}
    for name, dtype, shapes in (
        ("bfloat16", torch.bfloat16, big),
        ("float16", torch.float16, big),
        ("float32", torch.float32, big[:2]),
    ):
        if (tflops := _matmul_tflops(dtype, shapes, free)) is not None:
            peaks["tflops"][name] = tflops
        monitor.sample(f"burst {name}")
    torch.cuda.empty_cache()
    # Low-precision tensor cores: measured where torch has a kernel for this GPU, else the
    # reason is recorded (a ratio to bf16 is never assumed).
    unavailable: dict[str, str] = {}
    for name, measure in (
        (FP8, _fp8_tflops),
        (FP4, _fp4_tflops),
        (MXFP8, _mxfp8_tflops),
        (INT8, _int8_tops),
    ):
        try:
            tflops = measure(big, free)
        except Exception as exc:  # no kernel for this GPU / torch build
            unavailable[name] = f"{type(exc).__name__}: {exc}"[:200]
        else:
            if tflops is not None:
                peaks["tflops"][name] = tflops
        torch.cuda.empty_cache()
    if unavailable:
        peaks["tflops_unavailable"] = unavailable

    # Sustained 16-bit peaks (#253), after every burst so that the heat of their load cannot
    # lower a burst peak; a failure here leaves the peaks above as they are.
    def sustained(name: str, mon: Any) -> tuple[float, list[dict[str, Any]]] | None:
        return _sustained_tflops(getattr(torch, name), free, mon)

    try:
        sustained_peaks(peaks, monitor, sustained)
    except Exception as exc:  # out of memory, a lost device, ...
        peaks["sustained"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    torch.cuda.empty_cache()

    # Launch floor: a module call that launches one tiny kernel, timed like a candidate.
    class _OneKernel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    x = torch.zeros(1, device="cuda")
    floor = min(time_call(_OneKernel(), (x,), {}, target_ms=50.0)["median_ms"] for _ in range(3))
    peaks["launch_floor_us"] = round(floor * 1000, 2)

    # Tensor-core instruction rates (which instruction a kernel needs, mma_peaks.py): last,
    # so that a failure there cannot spoil the peaks above.
    try:
        from kernel_agent.kernels import mma_peaks

        rates, missing = mma_peaks.measure()
    except Exception as exc:  # no cuda.core / NVRTC, or the GPU rejected a kernel
        peaks["mma_unavailable"] = {"all": f"{type(exc).__name__}: {exc}"[:200]}
    else:
        if rates:
            peaks["mma_tflops"] = rates
        if missing:
            peaks["mma_unavailable"] = missing
    return peaks


def _peaks_file() -> tuple[Any, Path] | None:
    """(toolchain, peaks cache file) of the current GPU, or None without one."""
    tc = toolchain.setup()
    name = getattr(getattr(tc, "gpu", None), "name", None)
    if not isinstance(name, str):
        return None
    return tc, toolchain.peaks_path(name, tc.torch_version)


def current_peaks() -> dict[str, Any] | None:
    """Cached peaks of the current GPU + torch version (None if not measured yet)."""
    found = _peaks_file()
    if found is None:
        return None
    tc, path = found
    if tc.peaks is None:
        tc.peaks = toolchain.load_peaks(path)
    return tc.peaks


_MEASURE_FAILED = False


def _up_to_date(peaks: dict[str, Any] | None) -> bool:
    return peaks is not None and int(peaks.get("version") or 1) >= PEAKS_VERSION


def ensure_peaks(
    *, remeasure: bool = False, timeout: float = 600.0, verbose: bool = False
) -> dict[str, Any] | None:
    """Cached peaks, measuring them first (subprocess, under the GPU lock) when missing
    or older than :data:`PEAKS_VERSION`.

    Never raises; returns None without a GPU or when the measurement fails (it is
    retried once per process at most; an older cache stays in use). The peaks are
    measured on the GPU of the pool this thread locks and filed under this process's
    GPU model: a pool of identical GPUs shares one file (mixed models: restrict the
    pool, ``KERNEL_AGENT_GPUS``)."""
    global _MEASURE_FAILED
    stale = None  # an older cache: kept when the measurement fails
    try:
        found = _peaks_file()
        if found is None:
            return None
        tc, path = found
        if not remeasure:
            peaks = stale = current_peaks()
            if _up_to_date(peaks) or _MEASURE_FAILED:
                return peaks
        from kernel_agent.gpulock import child_env, gpu_lock

        with gpu_lock():
            peaks = None if remeasure else toolchain.load_peaks(path)  # another process won
            if not _up_to_date(peaks):
                if verbose:
                    print(f"measuring GPU peaks for the roofline -> {path}", file=sys.stderr)
                cmd = [sys.executable, "-m", "kernel_agent.kernels.roofline", "--out", str(path)]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout, env=child_env()
                )
                if proc.returncode != 0 and verbose:
                    print(proc.stderr[-2000:], file=sys.stderr)
                peaks = toolchain.load_peaks(path) or stale
    except Exception as exc:  # OSError, TimeoutExpired, a broken toolchain, ...
        if verbose:
            print(f"peak measurement failed: {exc}", file=sys.stderr)
        _MEASURE_FAILED = True
        return stale
    _MEASURE_FAILED = not _up_to_date(peaks)
    tc.peaks = peaks
    return peaks


# ------------------------------------------------------------------ counting


@dataclass
class CaseCost:
    """Minimum work of one reference call."""

    flops: dict[str, int] = field(default_factory=dict)  # per dtype name
    read_bytes: int = 0
    write_bytes: int = 0

    @property
    def total_flops(self) -> int:
        return sum(self.flops.values())

    @property
    def min_bytes(self) -> int:
        return self.read_bytes + self.write_bytes


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _key(t: Any) -> tuple[str, int] | None:
    """Identity of a tensor's storage (None for tensors without data)."""
    try:
        storage = t.untyped_storage()
        ptr = storage.data_ptr()
        nbytes = storage.nbytes()
    except (RuntimeError, NotImplementedError, TypeError):
        return None
    return (str(t.device), ptr) if ptr and nbytes else None


def _span(t: Any) -> tuple[int, int]:
    """Byte range ``[start, end)`` of a view inside its storage."""
    item = t.element_size()
    start = t.storage_offset() * item
    if t.numel() == 0:
        return start, start
    if any(s < 0 for s in t.stride()):
        return start, start + t.numel() * item
    last = sum((n - 1) * s for n, s in zip(t.shape, t.stride(), strict=True))
    return start, start + (last + 1) * item


def _union(ranges: list[tuple[int, int]]) -> int:
    total, cur_start, cur_end = 0, 0, -1
    for start, end in sorted(ranges):
        if start > cur_end:
            total += max(cur_end - cur_start, 0)
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + max(cur_end - cur_start, 0)


def _flatten(
    obj: Any, path: str = "", out: dict[str, Any] | None = None, seen: set[int] | None = None
) -> dict[str, Any]:
    """``path -> tensor`` for every tensor inside containers and plain objects (caches)."""
    import torch

    out = {} if out is None else out
    seen = set() if seen is None else seen
    if isinstance(obj, torch.Tensor):
        out.setdefault(path, obj)
        return out
    if (
        obj is None
        or isinstance(obj, str | bytes | int | float | bool | torch.nn.Module)
        or id(obj) in seen
        or path.count(".") > 6
    ):
        return out
    seen.add(id(obj))
    if isinstance(obj, dict):
        items: Any = obj.items()
    elif isinstance(obj, list | tuple):
        items = enumerate(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        items = vars(obj).items()
    else:
        return out
    for name, value in items:
        _flatten(value, f"{path}.{name}", out, seen)
    return out


def _module_tensors(module: Any) -> list[Any]:
    import torch

    tensors = [*module.parameters(), *module.buffers()]
    for sub in module.modules():
        tensors += [v for v in vars(sub).values() if isinstance(v, torch.Tensor)]
    return tensors


def _bind(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Arguments of an aten op by schema name (defaults filled in)."""
    bound: dict[str, Any] = {}
    for i, arg in enumerate(func._schema.arguments):
        if i < len(args) and not arg.kwarg_only:
            bound[arg.name] = args[i]
        elif arg.name in kwargs:
            bound[arg.name] = kwargs[arg.name]
        else:
            bound[arg.name] = getattr(arg, "default_value", None)
    return bound


def attention_density(mask: Any, is_causal: bool, lq: int, lk: int) -> tuple[float, float]:
    """(share of query/key pairs computed, share of key/value positions read) for SDPA."""
    if lq <= 0 or lk <= 0:
        return 0.0, 0.0
    if is_causal:  # upper-left aligned, like torch's is_causal
        n = lq * (lq + 1) // 2 if lq <= lk else lk * (lk + 1) // 2 + (lq - lk) * lk
        return n / (lq * lk), min(lq, lk) / lk
    if mask is None:
        return 1.0, 1.0
    import torch

    for dim, stride in enumerate(mask.stride()):  # broadcast (expanded) dims change no share
        if stride == 0 and mask.shape[dim] > 1:
            mask = mask.narrow(dim, 0, 1)
    allowed = mask if mask.dtype == torch.bool else mask > MASKED
    pairs = int(allowed.count_nonzero()) / max(allowed.numel(), 1)
    needed = allowed.any(dim=-2) if allowed.dim() >= 2 else allowed
    return pairs, int(needed.count_nonzero()) / max(needed.numel(), 1)


def _tracker(
    counter: Any,
    external: dict[tuple[str, int], int],
    narrow_math: tuple[set[tuple[str, int]], str] | None = None,
) -> Any:
    """Dispatch mode that records which external bytes each op reads or writes.
    ``narrow_math``: ``(weight storage keys, dtype name)``: the FLOPs of ops that read one
    of those weights count as that dtype's (:data:`MATH_DTYPE`)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class Tracker(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.reads: dict[tuple[str, int], list[tuple[int, int]]] = collections.defaultdict(list)
            self.gathered: dict[tuple[str, int], int] = collections.defaultdict(int)
            self.writes: dict[tuple[str, int], list[tuple[int, int]]] = collections.defaultdict(
                list
            )
            self.flops: dict[str, int] = collections.defaultdict(int)
            self.depth = 0

        def _read(self, t: Any, share: float = 1.0) -> None:
            key = _key(t)
            if key in external:
                start, end = _span(t)
                self.reads[key].append((start, start + math.ceil((end - start) * share)))

        def _attention(self, func: Any, args: Any, kwargs: Any, bound: dict[str, Any]) -> Any:
            q, k, v = bound["query"], bound["key"], bound["value"]
            mask = bound.get("attn_mask")
            mask = bound.get("attn_bias") if mask is None else mask
            self.depth += 1
            try:
                out = func(*args, **kwargs)
            finally:
                self.depth -= 1
            lq, lk = q.shape[-2], k.shape[-2]
            pairs, kv_share = attention_density(mask, bool(bound.get("is_causal")), lq, lk)
            rows = q.numel() // max(lq * q.shape[-1], 1)  # batch x heads
            full = 2 * rows * lq * lk * (q.shape[-1] + v.shape[-1])
            self.flops[_dtype_name(q.dtype)] += round(full * pairs)
            self._read(q)
            self._read(k, kv_share)
            self._read(v, kv_share)
            if isinstance(mask, torch.Tensor):
                self._read(mask)
            return out

        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None):
            kwargs = kwargs or {}
            if self.depth:
                return func(*args, **kwargs)
            name = func._schema.name.split("::")[-1]
            bound = _bind(func, args, kwargs)
            if name in _ATTENTION:
                return self._attention(func, args, kwargs, bound)
            before = counter.get_total_flops()
            out = func(*args, **kwargs)
            inputs = [t for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
            delta = counter.get_total_flops() - before
            if delta and narrow_math and any(_key(t) in narrow_math[0] for t in inputs):
                self.flops[narrow_math[1]] += delta  # a GEMM on a narrowed weight (W8A8)
            elif delta:
                floating = [t for t in inputs if t.is_floating_point()] or inputs
                self.flops[_dtype_name(floating[0].dtype) if floating else "unknown"] += delta
            if name in _NO_READ:
                return out
            mutated = {
                id(t)
                for arg in func._schema.arguments
                if arg.alias_info is not None and arg.alias_info.is_write
                for t in tree_leaves(bound.get(arg.name))
                if isinstance(t, torch.Tensor)
            }
            returned = {_key(t) for t in tree_leaves(out) if isinstance(t, torch.Tensor)}
            for i, t in enumerate(inputs):
                key = _key(t)
                if key not in external:
                    continue
                if id(t) in mutated:
                    self.writes[key].append(_span(t))
                elif key in returned:
                    continue  # a view, or handed back as is (a no-op .to()): nothing read
                elif name in _GATHER and i == 0:
                    got = sum(
                        o.numel() * o.element_size()
                        for o in tree_leaves(out)
                        if isinstance(o, torch.Tensor)
                    )
                    start, end = _span(t)
                    self.gathered[key] += min(got, end - start)
                else:
                    self.reads[key].append(_span(t))
            return out

    return Tracker()


def _changed_elements(before: Any, after: Any) -> int:
    import torch

    diff = before != after
    if before.is_floating_point() or before.is_complex():
        diff &= ~(torch.isnan(before) & torch.isnan(after))
    return int(diff.sum())


def _weight_shares(module: Any, precision: str | None) -> dict[tuple[str, int], tuple[float, int]]:
    """``storage key -> (share of the bytes read, scale bytes added)`` of the weights a
    reduced-precision kernel streams narrower (:data:`WEIGHT_BITS`; none for exact)."""
    bits = WEIGHT_BITS.get(precision or "")
    block = WEIGHT_SCALE_BLOCK.get(precision or "")
    shares: dict[tuple[str, int], tuple[float, int]] = {}
    if not bits:
        return shares
    for p in module.parameters():
        width = 8 * p.element_size()
        if p.dim() == 2 and p.is_floating_point() and width > bits:
            key = _key(p)
            if key is not None:
                scales = -(-p.numel() // block) + 4 if block else 4 * int(p.shape[0])
                shares[key] = (bits / width, scales)
    return shares


def count_case(
    module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    method: str | None = None,
    precision: str | None = None,
) -> CaseCost:
    """FLOPs (per dtype) and minimum bytes of one reference call (inputs are not mutated).
    ``precision`` (``fp8_weights``, ``fp4_weights``, ``fp8_w8a8``, ``int8_w8a8``, ...): the
    weights count at their reduced width (:data:`WEIGHT_BITS`, :data:`WEIGHT_SCALE_BLOCK`),
    and the GEMMs on them at the FP8 peak for ``fp8_w8a8``, the INT8 one for ``int8_w8a8``
    (:data:`MATH_DTYPE`)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    a, k = copy.deepcopy(args), copy.deepcopy(kwargs)
    before = _flatten((args, kwargs))
    external: dict[tuple[str, int], int] = {}
    for t in [*_flatten((a, k)).values(), *_module_tensors(module)]:
        key = _key(t)
        if key is not None:
            external[key] = t.untyped_storage().nbytes()
    narrow = _weight_shares(module, precision)
    fn = module if method in (None, "forward") else getattr(module, method)
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        dtype = MATH_DTYPE.get(precision or "")
        tracker = _tracker(counter, external, (set(narrow), dtype) if dtype else None)
        with tracker:
            out = fn(*a, **k)
    if any(t.is_cuda for t in before.values()) or any(p.is_cuda for p in _module_tensors(module)):
        torch.cuda.synchronize()

    reads = 0
    for key in set(tracker.reads) | set(tracker.gathered):
        n = min(_union(tracker.reads.get(key, [])) + tracker.gathered.get(key, 0), external[key])
        if key in narrow:  # a reduced-precision weight: its codes + scales
            share, scales = narrow[key]
            n = math.ceil(n * share) + scales
        reads += n

    after = _flatten((a, k))
    produced: dict[tuple[str, int], list[tuple[int, int]]] = collections.defaultdict(list)
    for t in [*_flatten(out).values(), *after.values()]:
        key = _key(t)
        if key is not None and key not in external:
            produced[key].append(_span(t))  # outputs and new state (e.g. a concatenated cache)
    writes = sum(_union(spans) for spans in produced.values())

    # In-place updates of inputs: the elements that changed (the touched cache slice).
    diffed: set[tuple[str, int]] = set()
    seen: set[tuple[tuple[str, int] | None, tuple[int, int]]] = set()
    for path, t in after.items():
        key = _key(t)
        if key is None or key not in tracker.writes or (key, _span(t)) in seen:
            continue
        old = before.get(path)
        if old is None or old.shape != t.shape or old.dtype != t.dtype:
            continue
        seen.add((key, _span(t)))
        writes += _changed_elements(old, t) * t.element_size()
        diffed.add(key)
    for key, spans in tracker.writes.items():
        if key not in diffed:  # module-held state: count the views the ops wrote
            writes += _union(spans)
    return CaseCost(flops=dict(tracker.flops), read_bytes=reads, write_bytes=writes)


# ------------------------------------------------------------------ speed of light


def sol_time(cost: CaseCost, peaks: dict[str, Any], *, hot_l2: bool = True) -> dict[str, Any]:
    """Speed-of-light time (ms) and bound of one case. FP8 (or MXFP8) FLOPs without a
    measured peak of their own count at the fastest peak, and ``peak_missing`` says so: the
    estimate is then too slow, not a ceiling."""
    tflops = {k: float(v) for k, v in (peaks.get("tflops") or {}).items() if v}
    fastest = max(tflops.values(), default=0.0)
    missing = next((d for d in MATH_DTYPE.values() if cost.flops.get(d) and d not in tflops), None)
    compute_ms = sum(
        f / (tflops.get(dtype) or fastest) / 1e9
        for dtype, f in cost.flops.items()
        if f and (tflops.get(dtype) or fastest)
    )
    l2 = bool(
        hot_l2 and peaks.get("l2_gbps") and cost.min_bytes <= float(peaks.get("l2_mb") or 0) * _MiB
    )
    gbps = float(peaks["l2_gbps"] if l2 else peaks["dram_gbps"])
    memory_ms = cost.min_bytes / gbps / 1e6
    sol_ms = max(compute_ms, memory_ms)
    floor_ms = float(peaks.get("launch_floor_us") or 0.0) / 1000
    bound = "compute" if compute_ms > memory_ms else "memory"
    if sol_ms < floor_ms:
        bound = "launch"  # one launch from Python costs more than the work
    found = {"sol_ms": sol_ms, "bound": bound, "l2_resident": l2}
    return found | ({"peak_missing": missing} if missing else {})


def _sig(value: float, digits: int = 4) -> float:
    return float(f"{value:.{digits}g}")


def _pct(sol_ms: float, new_ms: float) -> float:
    """100 x sol / new: one decimal, or three significant digits below 1 %."""
    pct = 100 * sol_ms / new_ms
    return round(pct, 1) if pct >= 1 else _sig(pct, 3)


def apply_sol(
    result: dict[str, Any],
    costs: list[CaseCost],
    peaks: dict[str, Any],
    *,
    hot_l2: bool = True,
) -> None:
    """Add per-case and weighted SOL fields to a timed evaluation result (in place)."""
    sol_total = new_total = 0.0
    time_by_bound: dict[str, float] = collections.defaultdict(float)
    suspicious = unreliable = False
    for report, cost in zip(result["cases"], costs, strict=True):
        sol = sol_time(cost, peaks, hot_l2=hot_l2)
        sol_ms = sol["sol_ms"]
        report.update(
            flops=cost.total_flops,
            min_bytes=cost.min_bytes,
            sol_ms=_sig(sol_ms),
            bound=sol["bound"],
        )
        if sol["l2_resident"]:
            report["l2_resident"] = True
        if sol.get("peak_missing"):  # no stop advice from a ceiling that is not one
            result["sol_note"] = (
                f"no {sol['peak_missing']} peak in the GPU peaks (`tflops_unavailable` says "
                "why): its FLOPs count at the fastest measured one, not a ceiling"
            )
        new_ms, ref_ms = report.get("new_ms"), report.get("ref_ms")
        if not new_ms:
            continue
        report["pct_of_sol"] = _pct(sol_ms, new_ms)
        if sol.get("peak_missing"):
            report["sol_unreliable"] = True
            unreliable = True
        elif ref_ms is not None and ref_ms < SUSPICIOUS_RATIO * sol_ms:
            report["sol_unreliable"] = True  # the reference beats it: the estimate is wrong
            unreliable = True
        elif new_ms < SUSPICIOUS_RATIO * sol_ms:
            report["suspicious_faster_than_sol"] = True
            suspicious = True
        n = report.get("calls_per_run") or 1
        sol_total += n * sol_ms
        new_total += n * new_ms
        time_by_bound[sol["bound"]] += n * new_ms
    if new_total <= 0:
        return
    result["sol_ms_weighted"] = _sig(sol_total)
    result["pct_of_sol"] = _pct(sol_total, new_total)
    result["bound"] = max(time_by_bound, key=lambda b: time_by_bound[b])
    result["launch_floor_ms"] = round(float(peaks.get("launch_floor_us") or 0.0) / 1000, 5)
    if suspicious:
        result["suspicious_faster_than_sol"] = True
    if unreliable:
        result["sol_unreliable"] = True


def annotate(
    result: dict[str, Any],
    module: Any,
    cases: list[dict[str, Any]],
    *,
    peaks: dict[str, Any] | None = None,
    l2_flush: bool = False,
    precision: str | None = None,
    restore: Any = None,
) -> None:
    """Add SOL fields to a timed evaluation result in place; never raises. ``precision``:
    the capture's reduced precision (weights counted at :data:`WEIGHT_BITS`); ``restore``:
    sets a case's module state before its call (``restore(case)``, profiling/state.py)."""
    peaks = peaks or current_peaks()
    if not peaks:
        result["sol_note"] = "GPU peaks not measured yet (`kernel-agent doctor` measures them)"
        return
    try:
        costs = []
        for c in cases:
            if restore is not None:
                restore(c)
            costs.append(
                count_case(
                    module, c["args"], c["kwargs"], method=c.get("method"), precision=precision
                )
            )
        apply_sol(result, costs, peaks, hot_l2=not l2_flush)
    except Exception as exc:
        result["sol_error"] = f"{type(exc).__name__}: {exc}"[:300]


def sol_signal(result: dict[str, Any]) -> float | None:
    """Weighted ``pct_of_sol`` of a correct evaluation whose SOL estimate is trustworthy."""
    if (
        not result.get("correct")
        or result.get("suspicious_faster_than_sol")
        or result.get("sol_unreliable")
    ):
        return None
    pct = result.get("pct_of_sol")
    return float(pct) if isinstance(pct, int | float) else None


def main(argv: list[str] | None = None) -> int:
    from kernel_agent.gpulock import gpu_lock, pinned
    from kernel_agent.workspace import write_json

    parser = argparse.ArgumentParser(description="Measure and cache the GPU roofline peaks.")
    parser.add_argument("--out", default=None, help="JSON file (default: the peaks cache)")
    ns = parser.parse_args(argv)
    with gpu_lock() as gpu:
        os.environ.update(pinned(gpu))  # before CUDA starts: measure the GPU we locked
        peaks = measure_peaks()
    tc = toolchain.setup()
    assert tc.gpu is not None
    path = ns.out or toolchain.peaks_path(tc.gpu.name, tc.torch_version)
    write_json(Path(path), peaks)
    print(json.dumps(peaks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
