"""Boards of one architecture (#253): the NVIDIA A10 as a first-class target, told apart from
a GeForce RTX 3090 (both sm_86) by measured facts only, never by name. CPU only: faked GPUs
whose peaks are NVIDIA's datasheet numbers (labelled; nothing here was measured on them)."""

from __future__ import annotations

import itertools
import re

import pytest

from kernel_agent import backends, gpu_arch, precisions, toolchain
from kernel_agent.agent import prompts
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.profiling import ceilings

#: name, capability, memory GB, SMs, L2 MB, smem per block KB, smem per SM KB; then peaks
#: from NVIDIA's datasheets and whitepapers (dense math, boost clocks): NOT measurements.
#: ``mma_tflops`` ``f16_f32`` / ``f16_f16``: fp16 HMMA with fp32 / fp16 accumulation.
DATASHEET = "NVIDIA datasheet (dense), not measured"
SKUS = {
    "A10": (
        ("NVIDIA A10", (8, 6), 22.1, 72, 6.0, 99.0, 100.0),
        {
            "source": DATASHEET,
            "dram_gbps": 600.0,
            "tflops": {"bfloat16": 125.0, "float16": 125.0, "float32": 31.2, "int8": 250.0},
            "mma_tflops": {"bf16_f32": 125.0, "f16_f32": 125.0, "f16_f16": 125.0, "s8_s32": 250.0},
        },
    ),
    "RTX 3090": (
        ("NVIDIA GeForce RTX 3090", (8, 6), 23.7, 82, 6.0, 99.0, 100.0),
        {
            "source": DATASHEET,
            "dram_gbps": 936.0,
            "tflops": {"bfloat16": 71.0, "float16": 71.0, "float32": 35.6, "int8": 284.0},
            "mma_tflops": {"bf16_f32": 71.0, "f16_f32": 71.0, "f16_f16": 142.0, "s8_s32": 284.0},
        },
    ),
    "A100": (
        ("NVIDIA A100-SXM4-80GB", (8, 0), 79.3, 108, 40.0, 163.0, 164.0),
        {
            "source": DATASHEET,
            "dram_gbps": 2039.0,
            "tflops": {"bfloat16": 312.0, "float16": 312.0, "float32": 19.5, "int8": 624.0},
        },
    ),
    "T4": (
        ("Tesla T4", (7, 5), 14.6, 40, 4.0, 64.0, 64.0),
        {
            "source": DATASHEET,
            "dram_gbps": 320.0,
            "tflops": {"float16": 65.0, "bfloat16": 8.1, "float32": 8.1, "int8": 130.0},
        },
    ),
    "L4": (
        ("NVIDIA L4", (8, 9), 22.0, 58, 48.0, 99.0, 100.0),
        {
            "source": DATASHEET,
            "dram_gbps": 300.0,
            "tflops": {
                "bfloat16": 121.0,
                "float16": 121.0,
                "float32": 30.3,
                "float8_e4m3fn": 242.0,
                "int8": 242.0,
            },
        },
    ),
}


def peaks_of(sku: str, **extra) -> dict:
    gpu, sheet = SKUS[sku]
    arch = gpu_arch.arch_of(gpu[1])
    return {"version": 6, "arch": arch, "launch_floor_us": 10.0, **sheet, **extra}


def fake(sku: str, *, mma: bool = True, **extra) -> toolchain.Toolchain:
    gpu = toolchain.GPUInfo(*SKUS[sku][0])
    peaks = peaks_of(sku, **extra)
    if not mma:
        peaks.pop("mma_tflops", None)
    available = {"cuda": True, "triton": True, "cute": True, "nvrtc": True, "tilelang": False}
    return toolchain.Toolchain(gpu, "2.14", "13.0", None, "13.0", available, [], {}, peaks)


def facts(sku: str, **kw) -> gpu_arch.Facts:
    return gpu_arch.from_summary(fake(sku, **kw).summary())


# ------------------------------------------------------------------ the A10 is first-class


def test_the_a10_engineer_prompt_has_its_gpu_and_the_ampere_section():
    target = {"id": "mlp", "module_class": "Qwen3MLP", "precision": "int8_w8a8"}
    capture = {"tier": "relaxed", "cases": [{"signature": "a0[352, 2048]:bfloat16", "count": 9}]}
    text = prompts.engineer_prompt(
        target, capture, ["triton", "cuda"], "py", fake("A10").summary(), 4, None
    )
    assert "# This GPU: NVIDIA A10 (sm_86, Ampere)" in text
    assert gpu_arch.knowledge_section("ampere").splitlines()[0] in text
    assert "`skus.md`" in text  # the Ampere section points at the SKU classes
    # the INT8 GEMM row with this GPU's measured ratio (2x on the A10, not GeForce's 4x)
    assert "measured on this GPU: `s8 IMMA.S32` 250 TOPS vs `bf16 HMMA.F32` 125" in text
    assert "(2.0x;" in text


def test_the_a10_plan_schema_and_precisions():
    cap = SKUS["A10"][0][1]
    allowed = precisions.allowed("relaxed", None, cap)
    assert {"exact", "fp8_weights", "reduced", "int8_weights", "int8_w8a8"} <= set(allowed)
    assert not {"fp8_w8a8", "fp8_mx", "fp4_w4a4"} & set(allowed)
    enum = prompts.plan_schema(allowed)["properties"]["targets"]["items"]["properties"]
    assert "fp8_w8a8" not in enum["precision"]["enum"]
    assert "int8_w8a8" in enum["precision"]["enum"]
    summary = fake("A10").summary()
    assert "8-bit tensor-core math here is INT8" in summary and "software conversion" in summary
    assert "16-bit ridge (bf16, the 16-bit tensor-core dtype here): a GEMM turns compute " in (
        summary
    )
    assert "near M ≈ 208 rows per weight read (measured bf16 125 TFLOP/s / DRAM 600" in summary


def _profile(rows: int = 352) -> dict:
    work = {
        "group": "model.layers.*.mlp",
        "phase": "prefill",
        "instances": 1,
        "calls": 100,
        "inclusive_ms": 400.0,
        "flops": {"bfloat16": 2 * rows * 8192 * 4096 * 100},
        "weight_elems": 8192 * 4096,
        "weight_bytes": 2 * 8192 * 4096,
        "io_bytes": 0,
        "signature": "",
    }
    return {"hooked_wall_ms": 1000.0, "classes": [{"cls": "Mlp", "is_leaf": False, "work": [work]}]}


def test_the_a10_ceilings_show_int8_w8a8_and_hide_fp8_math():
    table = ceilings.build(_profile(), peaks_of("A10"), 1000.0)
    assert set(table["gpu_hidden"]) == {"w8a8", "mxfp8", "w4a4"}
    assert {"exact", "fp8_weights", "int8_w8a8"} <= set(table["columns"])
    md = ceilings.markdown(table)
    assert "| INT8 W8A8 |" in md and "| FP8 w |" in md and "| W8A8 |" not in md
    assert "| MXFP8 |" not in md and "| W4A4 |" not in md
    row = table["rows"][0]
    flops = 2 * 352 * 8192 * 4096 * 100
    assert row["floors"]["exact"] == pytest.approx(flops / 125e9, rel=1e-3)  # compute bound
    assert row["floors"]["int8_w8a8"] == pytest.approx(flops / 250e9, rel=1e-3)


# ------------------------------------------------------------------ the policy names what runs

_NAMES = re.compile(r"\b(\w+\.py)\b")


def _excluded(text: str, cap: tuple[int, int]) -> list[str]:
    """The bundled examples ``text`` names whose ``ARCHS`` excludes ``cap``."""
    names = {p.name for p in EXAMPLES_DIR.iterdir()}
    found = sorted(set(_NAMES.findall(text)) & names)
    return [
        n
        for n in found
        if not gpu_arch.supports(gpu_arch.example_requirement(EXAMPLES_DIR / n)[0], cap)
    ]


@pytest.mark.parametrize("sku", ["A10", "RTX 3090", "A100"])
def test_the_ampere_policy_names_no_example_the_gpu_cannot_run(sku):
    gpu = facts(sku)
    cap = SKUS[sku][0][1]
    text = backends.policy_text(["cuda", "triton", "cute", "tilelang"], gpu)  # the planner's
    classes = ("Qwen3DecoderLayer", "Qwen3MLP", "Qwen3Attention", "Qwen3RMSNorm")
    for cls, rows, precision in itertools.product(
        classes, (8, 352), ("exact", "int8_w8a8", "fp8_weights")
    ):
        case = {"signature": f"a0[1, {rows}, 2048]:bfloat16", "count": 1}
        spec = {"module_class": cls, "precision": precision, "capture": {"cases": [case]}}
        text += backends.engineer_note(spec, ["cuda", "triton", "cute"], gpu)
    named = set(_NAMES.findall(text))
    assert {"cuda_int8_skinny_gemm.py", "triton_int8_w8a8_gemm.py", "cuda_int8_gemv.py"} <= named
    assert _excluded(text, cap) == []
    # the fused layer's second is CUDA C++ with IMMA / bf16 GEMMs (the CuTe layer is sm_89+)
    layer = backends.policy("decoder_layer", gpu)
    assert "cuda_int8_skinny_gemm.py" in layer.second
    assert "cute_fp8_decoder_block.py" not in layer.second
    # the GPUs that run the CuTe layer keep it
    assert "cute_fp8_decoder_block.py" in backends.policy("decoder_layer", facts("L4")).second


# ------------------------------------------------------------------ told apart by measurements


def test_fp32_accumulation_rate_tells_a10_from_rtx_3090():
    a10, geforce = fake("A10").summary(), fake("RTX 3090").summary()
    assert "fp32-accumulating HMMA at 100 % of fp16-accumulating here" in a10  # full rate
    assert "fp32 accumulation costs no tensor-core rate on this GPU" in a10
    assert "fp32-accumulating HMMA at 50 % of fp16-accumulating here" in geforce  # half rate
    assert "capped at the lower rate" in geforce
    # the same family text: only the measured lines differ
    same = [line for line in a10.splitlines() if line.startswith(("arch:", "tensor cores:"))]
    assert same and all(line in geforce for line in same)
    # without the pair of rates: no line, nothing assumed from the name
    assert "fp32-accumulating HMMA" not in fake("A10", mma=False).summary()
    # read back by the prompts (gpu_arch.from_summary), the A10's measured pair
    assert {"f16_f32": 125.0, "f16_f16": 125.0}.items() <= facts("A10").mma.items()


def test_int8_row_uses_the_measured_ratio_and_labels_other_skus():
    a10 = backends.policy("int8_gemm", facts("A10")).why
    geforce = backends.policy("int8_gemm", facts("RTX 3090")).why
    assert "measured on this GPU" in a10 and "(2.0x;" in a10
    assert "measured on this GPU" in geforce and "(4.0x;" in geforce
    unmeasured = backends.policy("int8_gemm", facts("A10", mma=False)).why
    assert (
        "measured on this GPU" not in unmeasured and "2x to 4x the bf16 rate by SKU" in unmeasured
    )
    for why in (a10, geforce, unmeasured):  # the other SKUs' numbers stay, labelled
        assert "datasheets, dense: A100 624 vs 312 TOPS" in why and "RTX 3090 284 vs 71" in why
    # never keyed on the name: an A10 renamed like a GeForce board gets the A10's facts
    renamed = gpu_arch.Facts("NVIDIA GeForce RTX 3090", (8, 6), facts("A10").mma)
    assert backends.policy("int8_gemm", renamed).why == a10


def test_datasheet_numbers_are_labelled_and_not_in_the_prompts():
    for _gpu, sheet in SKUS.values():
        assert sheet["source"] == DATASHEET
    text = prompts.planner_prompt(
        {"repo_id": "org/model", "modality": "llm"},
        {"median_ms": 1.0},
        "# Profile",
        ["cuda", "triton"],
        3,
        "py",
        fake("A10").summary(),
        quality="relaxed",
        precisions=precisions.allowed("relaxed", None, (8, 6)),
    )
    assert DATASHEET not in text and "source" not in toolchain.format_peaks(peaks_of("A10"))
