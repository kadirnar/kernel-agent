"""Tensor-core instruction rates (#146, #257): the microbenchmark's source and arithmetic,
which GPU has which instruction (NVRTC compiles and refuses them on the CPU), the peaks
line, and the ceilings table's FP8 instruction column; GPU-marked: the measurement, natively
and through an older GPU's PTX."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from kernel_agent import gpu_arch, toolchain
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

#: The forms of #257: PTX, minimum compute capability, the SASS ptxas makes of it there.
NEW = {
    "f16_f32": ("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", (8, 0), "HMMA.16816.F32"),
    "f16_f16": ("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", (8, 0), "HMMA.16816.F16"),
    "tf32_f32": (
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        (8, 0),
        "HMMA.1688.F32.TF32",
    ),
    "f16_f32_k8": ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32", (7, 5), "HMMA.1688.F32"),
    "f16_f16_k8": ("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16", (7, 5), "HMMA.1688.F16"),
    "s8_s32_k16": ("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32", (7, 5), "IMMA.8816.S8.S8"),
}
TURING = {"f16_f32_k8", "f16_f16_k8", "s8_s32_k16"}
AMPERE = TURING | {"bf16_f32", "f16_f32", "f16_f16", "tf32_f32", "s8_s32"}
ADA = AMPERE | {"e4m3_f32", "e4m3_f16"}


def test_each_instruction_has_its_ptx_in_the_kernel():
    ptx = {
        "bf16_f32": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "e4m3_f32": "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
        "e4m3_sf_f32": "kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32",
        "e4m3_f16": "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16",
        "s8_s32": "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
        **{key: ptx for key, (ptx, _, _) in NEW.items()},
    }
    for instruction in mma_peaks.INSTRUCTIONS:
        src = mma_peaks.source(instruction)
        assert ptx[instruction.key] in src
        assert 'extern "C" __global__' in src and "mma_rate" in src
        assert src.count("{") == src.count("}")
    sf = mma_peaks.source(mma_peaks.BY_KEY["e4m3_sf_f32"])
    assert '"r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z)' in sf


@pytest.mark.parametrize(
    ("capability", "runs"),
    [
        ((7, 5), TURING),
        ((8, 0), AMPERE),
        ((8, 6), AMPERE),
        ((8, 9), ADA),
        ((12, 0), ADA | {"e4m3_sf_f32"}),
    ],
)
def test_which_instructions_each_gpu_has(capability, runs):
    have = {i.key for i in mma_peaks.INSTRUCTIONS if mma_peaks.available(i, capability) is None}
    assert have == runs
    for key in set(mma_peaks.BY_KEY) - runs:
        assert mma_peaks.available(mma_peaks.BY_KEY[key], capability)


#: Compiles each form for its minimum arch (cubin: ptxas checks the target) and one below it,
#: with no GPU; prints {key: {arch: {"error", "ptx", "sass"}}}.
_COMPILE = """
import json, re, subprocess, sys, tempfile
from pathlib import Path
from cuda.core import Program, ProgramOptions
from kernel_agent.kernels import mma_peaks
from kernel_agent.kernels.sass import find_cuobjdump

tool, out = find_cuobjdump(), {}
for key, arch in json.loads(sys.argv[1]):
    src = mma_peaks.source(mma_peaks.BY_KEY[key])
    found = out.setdefault(key, {}).setdefault(arch, {})
    try:
        program = Program(src, code_type="c++", options=ProgramOptions(arch=arch, std="c++17"))
        found["ptx"] = program.compile("ptx").code.decode()
        program = Program(src, code_type="c++", options=ProgramOptions(arch=arch, std="c++17"))
        cubin = program.compile("cubin").code
    except Exception as exc:
        found["error"] = str(exc)
        continue
    if tool is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k.cubin"
            path.write_bytes(cubin)
            sass = subprocess.run([tool, "-sass", str(path)], capture_output=True, text=True)
        found["sass"] = sorted(set(re.findall(r"\\b([HIQ]MMA[\\w.]*)", sass.stdout)))
print(json.dumps(out))
"""


def test_nvrtc_compiles_each_new_form_for_its_arch_and_refuses_it_below():
    pytest.importorskip("cuda.core")
    jobs = []
    for key, (_, (major, minor), _) in NEW.items():
        jobs.append((key, f"sm_{major}{minor}"))
        if (major, minor) == (8, 0):  # NVRTC 13 targets sm_75 and newer: nothing below Turing
            jobs.append((key, "sm_75"))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}  # the CPU only
    proc = subprocess.run(
        [sys.executable, "-c", _COMPILE, json.dumps(jobs)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    found = json.loads(proc.stdout)
    for key, (ptx, (major, minor), sass) in NEW.items():
        own = found[key][f"sm_{major}{minor}"]
        assert "error" not in own, own.get("error")
        assert ptx in own["ptx"]
        if "sass" in own:  # a cuobjdump (the toolkit's or Triton's) to disassemble with
            assert sass in own["sass"], own["sass"]
        if (major, minor) == (8, 0):
            assert "requires .target sm_80" in found[key]["sm_75"]["error"]


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
    # m x n x k per instruction: Turing's shapes do a half / a quarter of sm_80's
    by = mma_peaks.BY_KEY
    assert mma_peaks.flops(by["f16_f32"], 70) == mma_peaks.flops(bf16, 70)
    assert mma_peaks.flops(by["tf32_f32"], 70) == mma_peaks.flops(bf16, 70) / 2
    assert mma_peaks.flops(by["f16_f32_k8"], 70) == mma_peaks.flops(bf16, 70) / 2
    assert mma_peaks.flops(by["s8_s32_k16"], 70) == mma_peaks.flops(by["s8_s32"], 70) / 4
    assert "needs sm_80+, this GPU is sm_75" in str(mma_peaks.available(by["f16_f16"], (7, 5)))


def test_the_operands_of_each_shape():
    by = mma_peaks.BY_KEY
    turing_f16 = mma_peaks.source(by["f16_f16_k8"])
    assert "{%0,%1},{%2,%3},{%4},{%0,%1}" in turing_f16  # fp16x2 D / C, 2 A, 1 B registers
    assert '"+r"(h[c][0]), "+r"(h[c][1]) : "r"(a0), "r"(a1), "r"(b0)' in turing_f16
    imma = mma_peaks.source(by["s8_s32_k16"])
    assert "{%0,%1},{%2},{%3},{%0,%1}" in imma and '"r"(a0), "r"(b0));' in imma
    assert "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3}" in mma_peaks.source(by["f16_f32_k8"])
    tf32 = mma_peaks.source(by["tf32_f32"])  # m16n8k8 tf32: 4 A, 2 B registers like m16n8k16
    assert "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}" in tf32


def test_the_peaks_line_has_every_form_and_reads_back():
    rates = {
        "bf16_f32": 104.0,
        "f16_f32": 104.0,
        "f16_f16": 208.0,
        "tf32_f32": 52.0,
        "f16_f32_k8": 52.0,
        "f16_f16_k8": 104.0,
        "e4m3_f32": 208.0,
        "s8_s32": 412.0,
        "s8_s32_k16": 99.0,
    }
    line = mma_peaks.describe(rates)
    assert line.startswith("mma.sync bf16 HMMA.F32 104 · fp16 HMMA.F32 104 · fp16 HMMA.F16 208")
    assert "e4m3 QMMA.F32 208 TFLOP/s · s8 IMMA.S32 412 · s8 IMMA.8816.S32 99 TOPS" in line
    summary = "GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, 70 SMs)\npeaks: " + line
    assert gpu_arch.from_summary(summary).mma == rates


@pytest.mark.parametrize(
    ("rates", "want"),
    [
        ({"f16_f32": 71.0, "f16_f16": 142.0}, "at 50 % of fp16-accumulating here"),  # RTX 3090
        ({"f16_f32": 125.0, "f16_f16": 125.0}, "costs no tensor-core rate"),  # A10
        ({"f16_f32_k8": 65.0, "f16_f16_k8": 65.0}, "at 100 % of"),  # T4: Turing's shape
        ({"f16_f32": 80.0, "f16_f16": 100.0}, "at 80 % of fp16-accumulating here (measured"),
    ],
)
def test_what_fp32_accumulation_costs(rates, want):
    line = mma_peaks.accumulation_line(rates)
    assert line is not None and want in line
    assert ("capped at the lower rate" in line) is (want.startswith("at 50"))
    assert mma_peaks.accumulation_line({"f16_f32": 71.0}) is None


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
def test_the_instruction_rates_are_measured():
    import torch

    capability = torch.cuda.get_device_capability()
    rates, missing = mma_peaks.measure()
    assert set(rates) | set(missing) == set(mma_peaks.BY_KEY)
    for key in missing:  # only what this GPU lacks, never a kernel that failed
        assert mma_peaks.available(mma_peaks.BY_KEY[key], capability), missing[key]
    assert rates.get("bf16_f32", 0) > 10, missing
    if "e4m3_f32" in rates and "e4m3_sf_f32" in rates:  # sm_120: the block-scaled form is 2x
        assert rates["e4m3_sf_f32"] > 1.5 * rates["e4m3_f32"]
    if "f16_f16" in rates:  # fp16 accumulation: at least the fp32-accumulating rate (2x on
        assert rates["f16_f16"] > 0.9 * rates["f16_f32"]  # GeForce); TF32 at half of it
        assert rates["tf32_f32"] < 0.6 * rates["f16_f32"]
        assert mma_peaks.accumulation_line(rates) is not None


@pytest.mark.gpu
def test_turings_rate_kernels_run_through_its_ptx():
    """The sm_75 kernels, JIT-compiled from compute_75 PTX for this GPU (#257): they load
    and run; their rates are this GPU's."""
    rates, missing = mma_peaks.measure((7, 5))
    assert set(rates) == TURING, missing
    assert all(rate > 1 for rate in rates.values())
    assert all("needs sm_" in why for why in missing.values())
