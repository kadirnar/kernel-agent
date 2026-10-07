"""Tensor-core instruction rates (#146): the microbenchmark's source and arithmetic, the
peaks line, and the ceilings table's FP8 instruction column; GPU-marked: the measurement."""

from __future__ import annotations

import pytest

from kernel_agent import toolchain
from kernel_agent.kernels import mma_peaks
from kernel_agent.profiling import ceilings

# The RTX 5070 Ti: GEMM peaks (test_ceilings.py) + instruction rates (RESEARCH-TRITON §1.1).
PEAKS = {
    "version": 3,
    "dram_gbps": 767.1,
    "launch_floor_us": 15.78,
    "tflops": {"bfloat16": 99.4, "float32": 34.4, "float8_e4m3fn": 332.6},
    "mma_tflops": {"bf16_f32": 104.0, "e4m3_f32": 208.0, "e4m3_sf_f32": 416.0, "e4m3_f16": 416.0},
}


def test_each_instruction_has_its_ptx_in_the_kernel():
    ptx = {
        "bf16_f32": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "e4m3_f32": "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
        "e4m3_sf_f32": "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32",
        "e4m3_f16": "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16",
    }
    for instruction in mma_peaks.INSTRUCTIONS:
        src = mma_peaks.source(instruction)
        assert ptx[instruction.key] in src
        assert 'extern "C" __global__' in src and "mma_rate" in src
        assert src.count("{") == src.count("}")
    sf = mma_peaks.source(mma_peaks.BY_KEY["e4m3_sf_f32"])
    assert '"r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z)' in sf


def test_flops_arch_and_availability():
    bf16 = mma_peaks.BY_KEY["bf16_f32"]
    fp8 = mma_peaks.BY_KEY["e4m3_f32"]
    warps = 70 * mma_peaks.BLOCKS_PER_SM * mma_peaks.THREADS / 32
    assert mma_peaks.flops(bf16, 70) == warps * 4096 * 8 * 2 * 16 * 8 * 16
    assert mma_peaks.flops(fp8, 70) == 2 * mma_peaks.flops(bf16, 70)
    assert mma_peaks.arch((12, 0)) == "sm_120a" and mma_peaks.arch((8, 9)) == "sm_89"
    sf = mma_peaks.BY_KEY["e4m3_sf_f32"]
    assert mma_peaks.available(sf, (12, 0)) is None
    assert "needs sm_120+" in str(mma_peaks.available(sf, (10, 0)))
    assert "sm_120a / sm_121a only" in str(mma_peaks.available(sf, (13, 0)))
    assert "sm_89+" in str(mma_peaks.available(fp8, (8, 6)))
    assert mma_peaks.available(fp8, (8, 9)) is None


def test_the_peaks_line_shows_the_instruction_rates():
    line = toolchain.format_peaks(PEAKS)
    assert "mma.sync bf16 HMMA.F32 104 · e4m3 QMMA.F32 208 · e4m3 QMMA.SF 416" in line


def row(flops: float, weights: float, calls: int = 1) -> dict:
    return {
        "flops": {"bfloat16": int(flops)},
        "weight_elems": int(weights),
        "weight_bytes": int(2 * weights),
        "io_bytes": 0,
        "calls": calls,
    }


def test_a_compute_bound_row_needs_the_block_scaled_instruction():
    gemm = row(2 * 352 * 8192 * 1024 * 540, 8192 * 1024)  # LocDiT-like gate|up, 540 calls
    found = ceilings.fp8_instruction(gemm, PEAKS)
    assert found is not None and found["needs"] == "QMMA.SF"
    w8a8 = ceilings.floor(gemm, ceilings.PRECISIONS["w8a8"], PEAKS)
    assert w8a8 is not None
    assert found["f32_floor_ms"] == pytest.approx(w8a8["compute_ms"] * 332.6 / 208.0, rel=1e-3)
    decode = row(2 * 16 * 12288 * 2048, 12288 * 2048)  # 16-row decode GEMV: memory bound
    assert ceilings.fp8_instruction(decode, PEAKS) == {"needs": "any"}
    without = {k: v for k, v in PEAKS.items() if k != "mma_tflops"}
    assert ceilings.fp8_instruction(gemm, without) is None


def profile(work: list[dict]) -> dict:
    return {
        "hooked_wall_ms": 100.0,
        "classes": [{"cls": "Mlp", "is_leaf": False, "work": work}],
    }


def entry(group: str, phase: str, flops: float, weights: int, ms: float, **extra: int) -> dict:
    return {
        "group": group,
        "phase": phase,
        "instances": 1,
        "calls": 1,
        "inclusive_ms": ms,
        "flops": {"bfloat16": int(flops)},
        "weight_elems": weights,
        "weight_bytes": 2 * weights,
        "io_bytes": 0,
        "signature": "",
        **extra,
    }


def test_the_ceilings_table_shows_the_fp8_instruction_column():
    work = [
        entry("model.dit.mlp", "prefill", 2 * 352 * 8192 * 1024 * 540, 8192 * 1024, 60.0),
        entry("model.lm.mlp", "decode", 2 * 16 * 12288 * 2048, 12288 * 2048, 30.0),
    ]
    table = ceilings.build(profile(work), PEAKS, 100.0)
    rows = {r["group"]: r for r in table["rows"]}
    assert rows["model.dit.mlp"]["fp8_mma"]["needs"] == "QMMA.SF"
    assert rows["model.lm.mlp"]["fp8_mma"] == {"needs": "any"}
    assert table["peaks"]["mma_tflops"]["e4m3_f32"] == 208.0
    md = ceilings.markdown(table)
    assert "| FP8 MMA |" in md and "| SF (" in md and "| any |" in md
    assert "runs at 208 TFLOP/s here, block-scaled `QMMA.SF`" in md
    # a run that does not allow W8A8 has no such column
    exact = ceilings.markdown(ceilings.build(profile(work), PEAKS, 100.0, allowed=["exact"]))
    assert "FP8 MMA" not in exact


@pytest.mark.gpu
def test_the_instruction_rates_are_measured():  # not run in #146: verify on the GPU
    rates, missing = mma_peaks.measure()
    assert rates.get("bf16_f32", 0) > 10, missing
    if "e4m3_f32" in rates and "e4m3_sf_f32" in rates:  # sm_120: the block-scaled form is 2x
        assert rates["e4m3_sf_f32"] > 1.5 * rates["e4m3_f32"]
