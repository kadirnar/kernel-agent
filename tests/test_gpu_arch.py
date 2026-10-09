"""Any NVIDIA GPU (#165): architecture families, precisions the GPU cannot run, the backend
policy, the prompts' GPU facts, the ceilings columns, the examples' ``ARCHS`` and the doctor
probes, on faked GPUs (sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120). CPU only."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kernel_agent import backends, gpu_arch, precisions, probes, selftest, toolchain
from kernel_agent.agent import prompts, web
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels import mma_peaks
from kernel_agent.profiling import ceilings
from kernel_agent.report import _quality_lines

#: name, capability, memory GB, SMs, L2 MB, smem per block KB; a second GPU of an arch is
#: keyed ``<arch>/<GPU>`` (:func:`arch_of`)
GPUS = {
    "sm_75": ("Tesla T4", (7, 5), 14.6, 40, 4.0, 64.0),
    "sm_80": ("NVIDIA A100-SXM4-80GB", (8, 0), 80.0, 108, 40.0, 163.0),
    "sm_86": ("NVIDIA GeForce RTX 3090", (8, 6), 24.0, 82, 6.0, 99.0),
    "sm_86/A10": ("NVIDIA A10", (8, 6), 22.1, 72, 6.0, 99.0),
    "sm_89": ("NVIDIA L40S", (8, 9), 48.0, 142, 96.0, 99.0),
    "sm_89/L4": ("NVIDIA L4", (8, 9), 22.1, 58, 48.0, 99.0),
    "sm_90": ("NVIDIA H100 80GB HBM3", (9, 0), 80.0, 132, 50.0, 227.0),
    "sm_100": ("NVIDIA B200", (10, 0), 180.0, 148, 126.0, 227.0),
    "sm_120": ("NVIDIA GeForce RTX 5070 Ti", (12, 0), 15.5, 70, 48.0, 99.0),
}
#: The RTX 5070 Ti's measured peaks (test_mma_peaks.py), and a generic set for the others.
PEAKS_120 = {
    "version": 4,
    "arch": "sm_120",
    "dram_gbps": 767.1,
    "launch_floor_us": 15.78,
    "tflops": {"bfloat16": 99.4, "float32": 34.4, "float8_e4m3fn": 332.6, "mxfp8": 239.0},
    "mma_tflops": {"bf16_f32": 104.0, "e4m3_f32": 208.0, "e4m3_sf_f32": 416.0},
}


def arch_of(key: str) -> str:
    """The arch of a :data:`GPUS` key (``sm_86`` for ``sm_86/A10``)."""
    return key.split("/")[0]


def peaks_of(arch: str) -> dict:
    arch = arch_of(arch)
    if arch == "sm_120":
        return dict(PEAKS_120)
    if arch == "sm_75":  # T4 datasheet: fp16 tensor cores; bf16 at the fp32 rate
        tflops = {"bfloat16": 8.1, "float16": 65.0, "float32": 8.1}
        return {"version": 4, "arch": arch, "dram_gbps": 320.0, "tflops": tflops}
    tflops = {"bfloat16": 300.0, "float32": 60.0}
    if arch in ("sm_89", "sm_90", "sm_100"):
        tflops["float8_e4m3fn"] = 600.0
    if arch == "sm_100":
        tflops["mxfp8"] = 600.0
    return {
        "version": 4,
        "arch": arch,
        "dram_gbps": 2000.0,
        "launch_floor_us": 6.0,
        "tflops": tflops,
    }


def fake(arch: str, *, peaks: bool = True, smem: bool = True) -> toolchain.Toolchain:
    name, cap, mem, sms, l2, kb = GPUS[arch]
    gpu = toolchain.GPUInfo(name, cap, mem, sms, l2, kb if smem else 0.0, kb + 1 if smem else 0.0)
    backends_ = {"cuda": True, "triton": True, "cute": True, "nvrtc": True, "tilelang": False}
    return toolchain.Toolchain(
        gpu, "2.14", "13.0", None, "13.0", backends_, [], {}, peaks_of(arch) if peaks else None
    )


ALL = list(GPUS)


# ------------------------------------------------------------------ families and parsing


@pytest.mark.parametrize(
    ("cap", "key"),
    [
        ((7, 5), "pre_ampere"),
        ((8, 0), "ampere"),
        ((8, 6), "ampere"),
        ((8, 9), "ada"),
        ((9, 0), "hopper"),
        ((10, 0), "blackwell"),
        ((10, 3), "blackwell"),
        ((11, 0), "blackwell"),
        ((12, 0), "blackwell_geforce"),
        ((12, 1), "blackwell_geforce"),
        ((13, 0), "newer"),
    ],
)
def test_family_of_each_capability(cap, key):
    fam = gpu_arch.family(cap)
    assert fam is not None and fam.key == key
    assert gpu_arch.knowledge_section(key)  # every family has its section in gpus.md


def test_features_and_arch_names():
    assert gpu_arch.family(None) is None
    assert gpu_arch.has((9, 0), "wgmma") and not gpu_arch.has((10, 0), "wgmma")
    assert gpu_arch.has((10, 0), "tcgen05") and not gpu_arch.has((12, 0), "tcgen05")
    assert gpu_arch.has((12, 0), "mma_block_scale") and not gpu_arch.has((10, 0), "mma_block_scale")
    assert gpu_arch.has((8, 9), "fp8_tc") and not gpu_arch.has((8, 6), "fp8_tc")
    assert gpu_arch.has((9, 0), "pdl") and not gpu_arch.has((8, 9), "pdl")
    for arch, cap in (("sm_120a", (12, 0)), ("sm_90", (9, 0)), ("compute_100", (10, 0))):
        assert gpu_arch.capability_of(arch) == cap
    assert gpu_arch.capability_of("sm_8") is None and gpu_arch.capability_of("gfx90a") is None
    assert gpu_arch.arch_of((8, 6)) == "sm_86" and gpu_arch.arch_of(None) is None


def test_arch_specs_of_the_examples():
    assert gpu_arch.supports(None, None) and gpu_arch.supports("", (7, 5))
    assert gpu_arch.supports("sm_89+", (9, 0)) and not gpu_arch.supports("sm_89+", (8, 6))
    assert gpu_arch.supports("sm_12x", (12, 1)) and not gpu_arch.supports("sm_12x", (10, 0))
    assert gpu_arch.supports("sm_90", (9, 0)) and not gpu_arch.supports("sm_90", (10, 0))
    assert gpu_arch.supports("sm_80, sm_12x", (12, 0)) and not gpu_arch.supports("sm_89+", None)
    for bad in ("ampere", "sm_8+"):
        with pytest.raises(ValueError):
            gpu_arch.supports(bad, (8, 0))


def test_every_bundled_example_declares_parseable_archs():
    for path in sorted(EXAMPLES_DIR.iterdir()):
        if path.name.startswith(("_", ".")) or path.suffix not in (".py", ""):
            continue
        spec, why = gpu_arch.example_requirement(path)
        gpu_arch.supports(spec, (12, 0))  # raises on a malformed declaration
        if spec:
            assert why, path.name  # a declaration says why
    # the low-precision examples are the ones that need a declaration
    for name in ("triton_fp8_w8a8_gemm.py", "triton_mxfp8_gemm.py", "cute_fp8_blockscaled_gemm.py"):
        assert gpu_arch.example_requirement(EXAMPLES_DIR / name)[0], name


# ------------------------------------------------------------------ precisions


@pytest.mark.parametrize(
    ("arch", "w8a8", "mx"),
    [
        ("sm_80", False, False),
        ("sm_86", False, False),
        ("sm_89", True, False),
        ("sm_90", True, False),
        ("sm_100", True, True),
        ("sm_120", True, True),
    ],
)
def test_precisions_follow_the_gpu(arch, w8a8, mx):
    cap = GPUS[arch][1]
    allowed = precisions.allowed("near-lossless", None, cap)
    assert "fp8_weights" in allowed and "reduced" in allowed  # weight-only runs everywhere
    assert ("fp8_w8a8" in allowed) is w8a8 and ("fp8_mx" in allowed) is mx
    # an explicit --precisions cannot bring them back
    asked = precisions.allowed("near-lossless", ["fp8_w8a8", "fp8_mx", "fp8_kv"], cap)
    assert ("fp8_w8a8" in asked) is w8a8 and ("fp8_mx" in asked) is mx and "fp8_kv" in asked
    refused = precisions.gpu_refused("near-lossless", None, cap)
    assert set(refused) == {p for p, ok in (("fp8_w8a8", w8a8), ("fp8_mx", mx)) if not ok}
    for name, why in refused.items():
        msg = precisions.refusal(name, precisions.allowed("near-lossless"), cap)
        assert msg and "cannot run on this GPU" in msg and why in msg and arch in msg
    assert precisions.refusal("fp8_weights", allowed, cap) is None
    # no GPU known: nothing is filtered (CPU tests, simulated runs)
    assert "fp8_mx" in precisions.allowed("near-lossless", None, None)
    # the schema the planner fills offers only what this GPU can run
    enum = prompts.plan_schema(allowed)["properties"]["targets"]["items"]["properties"]
    assert ("fp8_mx" in enum["precision"]["enum"]) is mx
    if not w8a8:
        assert "not on " + arch in precisions.describe(allowed, cap)


def test_report_names_the_precisions_the_gpu_cannot_run():
    data = {"config": {"quality": "near-lossless", "precisions": ["exact", "fp8_weights"]}}
    line = _quality_lines(data, {}, {"capability": [8, 6]})[-1]
    assert "not on sm_86: fp8_w8a8, fp8_mx" in line
    assert "not on" not in _quality_lines(data, {}, {"capability": [12, 0]})[-1]


# ------------------------------------------------------------------ toolchain summary


@pytest.mark.parametrize("key", ALL)
def test_summary_carries_the_gpu_facts(key):
    tc = fake(key)
    text = tc.summary()
    arch = arch_of(key)
    fam = gpu_arch.family(GPUS[key][1])
    assert fam is not None
    assert f"arch: {fam.name} ({arch})" in text and "smem per block" in text
    short = "fp16" if arch == "sm_75" else "bf16"  # Turing's tensor cores have no bf16
    assert f"16-bit ridge ({short}, the 16-bit tensor-core dtype here)" in text
    assert "full rate needs" in text
    assert ("precisions this GPU cannot run" in text) is (arch not in ("sm_100", "sm_120"))
    assert ("software conversion" in text) is (arch in ("sm_75", "sm_80", "sm_86"))
    if arch != "sm_120":
        assert "5070" not in text and "QMMA.SF" not in text
    facts = gpu_arch.from_summary(text)
    assert facts.capability == GPUS[key][1] and facts.name == GPUS[key][0]
    assert facts == gpu_arch.from_toolchain(tc)  # the prompts read back what doctor shows


def _summary(name: str, cap: tuple[int, int], peaks: dict) -> str:
    gpu = toolchain.GPUInfo(name, cap, 16.0, 40, 4.0, 64.0, 64.0)
    return toolchain.Toolchain(gpu, "2.14", "13.0", None, "13.0", {}, [], {}, peaks).summary()


def test_the_16bit_ridge_is_the_tensor_core_dtypes():
    """A T4 (datasheet: fp16 tensor cores 65 TFLOP/s, 320 GB/s; bf16 without tensor cores at
    the fp32 rate) turns compute bound near M ≈ 203 on fp16, not 25 on bf16 (#257)."""
    t4 = {
        "arch": "sm_75",
        "dram_gbps": 320.0,
        "tflops": {"bfloat16": 8.1, "float16": 65.0, "float32": 8.1},
        "mma_tflops": {"f16_f32_k8": 65.0, "f16_f16_k8": 65.0, "s8_s32_k16": 130.0},
    }
    text = _summary("Tesla T4", (7, 5), t4)
    assert "16-bit ridge (fp16, the 16-bit tensor-core dtype here): a GEMM turns compute " in text
    assert "M ≈ 203 rows per weight read (measured fp16 65 TFLOP/s / DRAM 320 GB/s)" in text
    assert "bf16 matmuls run without tensor cores here (8 TFLOP/s: M ≈ 25)" in text
    assert "fp32-accumulating HMMA at 100 % of fp16-accumulating here" in text
    assert gpu_arch.from_summary(text).mma == t4["mma_tflops"]
    assert gpu_arch.tensor_core_16bit((7, 5)) == "float16"
    assert gpu_arch.tensor_core_16bit((8, 0)) == gpu_arch.tensor_core_16bit(None) == "bfloat16"


@pytest.mark.parametrize(
    ("name", "rates", "half"),
    [("NVIDIA GeForce RTX 3090", (71.0, 142.0), True), ("NVIDIA A10", (125.0, 125.0), False)],
)
def test_the_summary_says_what_fp32_accumulation_costs(name, rates, half):
    """GeForce Ampere runs fp32-accumulating HMMA at half rate, the A10 at full rate
    (NVIDIA's whitepaper / datasheet numbers, dense): only the measured rates tell."""
    peaks = {**peaks_of("sm_86"), "mma_tflops": {"f16_f32": rates[0], "f16_f16": rates[1]}}
    text = _summary(name, (8, 6), peaks)
    pct = 50 if half else 100
    assert f"fp32-accumulating HMMA at {pct} % of fp16-accumulating here" in text
    assert ("capped at the lower rate" in text) is half
    assert ("costs no tensor-core rate" in text) is not half
    assert "fp32-accumulating" not in _summary(name, (8, 6), peaks_of("sm_86"))


def test_builds_target_the_arch_specific_isa_where_the_peak_mma_needs_it():
    assert toolchain.cuda_arch_list((9, 0)) == "9.0a"
    assert toolchain.cuda_arch_list((10, 0)) == "10.0a"
    assert toolchain.cuda_arch_list((8, 6)) == "8.6"
    assert toolchain.cuda_arch_list((12, 0)) == "12.0"  # as verified on the RTX 5070 Ti


def test_summary_without_smem_names_the_documented_value():
    text = fake("sm_90", smem=False).summary()
    assert "not detected (227 KB for sm_90" in text
    assert gpu_arch.summary_lines(None, None) == []
    assert gpu_arch.from_summary("GPU: none detected") == gpu_arch.Facts()


def test_measured_instruction_rates_round_trip():
    facts = gpu_arch.from_summary(fake("sm_120").summary())
    assert facts.mma == {"bf16_f32": 104.0, "e4m3_f32": 208.0, "e4m3_sf_f32": 416.0}


# ------------------------------------------------------------------ backend policy


def _policy(arch: str, **kw) -> str:
    return backends.policy_text(
        ["cuda", "triton", "cute"], gpu_arch.from_summary(fake(arch, **kw).summary())
    )


def test_policy_on_ampere_has_no_fp8_class():
    for arch in ("sm_80", "sm_86"):
        text = _policy(arch)
        assert "Compute-bound FP8 GEMM" not in text and "QMMA" not in text
        assert "No FP8 tensor cores" in text and "warp_specialize" not in text


def test_policy_on_ada_uses_plain_fp8_mma():
    text = _policy("sm_89")
    assert "only FP8 tensor-core instruction on sm_89" in text
    assert "never `tl.dot_scaled` / MXFP8 on sm_89" in text and "QMMA.SF" not in text


def test_policy_on_hopper_needs_wgmma():
    text = _policy("sm_90")
    assert "`wgmma` on e4m3" in text and "never a `mma.sync` kernel" in text
    assert "QMMA.SF" not in text and "MmaMXF8Op" not in text
    assert "never `warp_specialize`" not in text and "warp_specialize=True" in text
    assert "MXFP8 `fp8_mx`" not in text  # no block-scaled MMA: not offered


def test_policy_on_datacenter_blackwell_needs_tcgen05():
    text = _policy("sm_100")
    assert "`tcgen05.mma`" in text and "MXFP8 `fp8_mx`" in text
    assert "QMMA.SF" not in text and "never `warp_specialize`" not in text


def test_policy_on_geforce_blackwell_uses_the_measured_rates():
    text = _policy("sm_120")
    assert "measured on this GPU: `QMMA.F32` (plain e4m3 `mma.sync`) 208 TFLOP/s" in text
    assert "never a plain `QMMA.F32` kernel" in text and "never `warp_specialize`" in text
    # a GeForce-class GPU whose FP8 instructions run at the same rate: no half-rate rule
    same = gpu_arch.Facts("RTX PRO", (12, 0), {"e4m3_f32": 400.0, "e4m3_sf_f32": 410.0})
    row = backends.policy("fp8_gemm", same)
    assert not row.never and "same rate" in row.why
    # unmeasured: the RTX 5070 Ti evidence, labelled
    unmeasured = backends.policy("fp8_gemm", gpu_arch.Facts("RTX 5090", (12, 0)))
    assert "RTX 5070 Ti" in unmeasured.why


def test_unknown_gpu_keeps_the_measured_sm120_table():
    text = backends.policy_text(["cuda", "triton", "cute"])
    assert "measured on sm_120 / RTX 5070 Ti" in text and "`QMMA.SF`" in text


def test_engineer_note_follows_the_gpu():
    spec = {
        "module_class": "Qwen3MLP",
        "precision": "fp8_w8a8",
        "capture": {"cases": [{"signature": "a0[4, 176, 1024]:bfloat16", "count": 10}]},
    }
    hopper = gpu_arch.from_summary(fake("sm_90").summary())
    note = backends.engineer_note(spec, ["triton", "cuda"], hopper)
    assert "wgmma" in note and "QMMA.SF" not in note and "tl.dot_scaled` with unit" not in note
    geforce = backends.engineer_note(spec, ["triton", "cuda"])  # unknown GPU: as before
    assert "`tl.dot_scaled` with unit ue8m0 scales" in geforce


# ------------------------------------------------------------------ prompts


def _planner(arch: str, quality: str = "near-lossless") -> str:
    tc = fake(arch)
    return prompts.planner_prompt(
        {"repo_id": "org/model", "modality": "llm"},
        {"median_ms": 1.0},
        "# Profile",
        ["cuda", "triton", "cute"],
        3,
        "py",
        tc.summary(),
        quality=quality,
        precisions=precisions.allowed(quality, None, GPUS[arch][1]),
    )


@pytest.mark.parametrize("key", ALL)
def test_planner_prompt_gets_this_gpus_facts(key):
    text = _planner(key)
    arch = arch_of(key)
    fam = gpu_arch.family(GPUS[key][1])
    assert fam is not None
    assert f"# This GPU: {GPUS[key][0]} ({arch}, {fam.name})" in text
    section = gpu_arch.knowledge_section(fam.key)
    assert section.splitlines()[0] in text  # the family's section of gpus.md
    others = [f for f in gpu_arch.FAMILIES if f.key != fam.key]
    for other in others:  # and only that one
        assert gpu_arch.knowledge_section(other.key).splitlines()[0] not in text
    if arch in ("sm_75", "sm_80", "sm_86"):
        assert "This GPU cannot run `fp8_w8a8`" in text and '"fp8_w8a8"' not in text
    if arch in ("sm_89", "sm_90"):
        assert "This GPU cannot run `fp8_mx`" in text and 'precision: "fp8_mx"' not in text
    if arch in ("sm_100", "sm_120"):
        assert "This GPU cannot run" not in text and 'precision: "fp8_mx"' in text


def test_prompts_without_a_gpu_have_no_gpu_section():
    assert "# This GPU" not in _planner_without_gpu()


def _planner_without_gpu() -> str:
    return prompts.planner_prompt(
        {"repo_id": "org/model", "modality": "llm"},
        {"median_ms": 1.0},
        "# Profile",
        ["cuda"],
        3,
        "py",
        "tc",
    )


def test_engineer_prompt_of_a_weight_only_target_on_ampere():
    target = {"id": "mlp", "module_class": "Qwen3MLP", "precision": "fp8_weights"}
    capture = {
        "tier": "near-lossless",
        "cases": [{"signature": "a0[1, 2048]:bfloat16", "count": 28}],
    }
    text = prompts.engineer_prompt(
        target, capture, ["cuda"], "py", fake("sm_86").summary(), 4, None
    )
    assert "On this GPU: this GPU has no hardware e4m3 conversion" in text
    assert "# This GPU: NVIDIA GeForce RTX 3090 (sm_86, Ampere)" in text
    assert "`ARCHS` names the GPUs it runs on" in text
    ada = prompts.engineer_prompt(target, capture, ["cuda"], "py", fake("sm_89").summary(), 4, None)
    assert "no hardware e4m3 conversion" not in ada


def test_research_and_dossier_prompts_get_the_gpu_section(tmp_path):
    summary = fake("sm_90").summary()
    target = {"id": "mlp", "module_class": "Qwen3MLP"}
    dossier = prompts.dossier_prompt(target, {}, tmp_path / "research.md", summary)
    assert "# This GPU: NVIDIA H100 80GB HBM3 (sm_90, Hopper)" in dossier


# ------------------------------------------------------------------ ceilings


def _profile() -> dict:
    work = {
        "group": "model.dit.mlp",
        "phase": "prefill",
        "instances": 1,
        "calls": 540,
        "inclusive_ms": 60.0,
        "flops": {"bfloat16": 2 * 352 * 8192 * 1024 * 540},
        "weight_elems": 8192 * 1024,
        "weight_bytes": 2 * 8192 * 1024,
        "io_bytes": 0,
        "signature": "",
    }
    return {"hooked_wall_ms": 100.0, "classes": [{"cls": "Mlp", "is_leaf": False, "work": [work]}]}


@pytest.mark.parametrize(
    ("arch", "hidden"),
    [
        ("sm_86", {"w8a8", "mxfp8", "w4a4"}),
        ("sm_89", {"mxfp8", "w4a4"}),
        ("sm_90", {"mxfp8", "w4a4"}),
        ("sm_100", set()),
        ("sm_120", set()),
    ],
)
def test_ceilings_omit_the_columns_the_gpu_cannot_run(arch, hidden):
    table = ceilings.build(_profile(), peaks_of(arch), 100.0)
    assert set(table["gpu_hidden"]) == hidden
    assert not hidden & set(table["columns"])
    md = ceilings.markdown(table)
    for column in hidden:
        assert f"| {ceilings.PRECISIONS[column].label} |" not in md
    assert ("not on this GPU" in md) is bool(hidden)


def test_fp8_instruction_column_only_where_two_fp8_instructions_exist():
    row = ceilings.build(_profile(), peaks_of("sm_120"), 100.0)["rows"][0]
    assert row["fp8_mma"]["needs"] == "QMMA.SF"
    ada = {**peaks_of("sm_89"), "mma_tflops": {"bf16_f32": 180.0, "e4m3_f32": 360.0}}
    assert "fp8_mma" not in ceilings.build(_profile(), ada, 100.0)["rows"][0]


def test_mma_peaks_skip_the_emulated_fp8_mma_sync():
    fp8 = mma_peaks.BY_KEY["e4m3_f32"]
    assert "emulated" in str(mma_peaks.available(fp8, (9, 0)))
    assert "tcgen05" in str(mma_peaks.available(fp8, (10, 0)))
    assert mma_peaks.available(fp8, (8, 9)) is None and mma_peaks.available(fp8, (12, 0)) is None
    assert mma_peaks.available(mma_peaks.BY_KEY["bf16_f32"], (9, 0)) is None


# ------------------------------------------------------------------ selftest and doctor


@pytest.mark.parametrize(
    ("arch", "runs", "skips"),
    [
        (
            "sm_75",  # as compiled for it on the CPU (test_arch_matrix.py)
            {"cuda_fp8_gemv.py", "cuda_int8_gemv.py", "cuda_fp4_gemv.py", "tilelang_rmsnorm.py"},
            {"cuda_int8_skinny_gemm.py", "triton_int8_w8a8_gemm.py", "cute_rmsnorm.py"},
        ),
        (
            "sm_80",
            {"cuda_fp8_gemv.py", "cuda_fp4_gemv.py", "triton_short_attention.py"},
            {"triton_fp8_w8a8_gemm.py", "cuda_pdl_gemv_chain.py", "cuda_cublaslt_fp8.py"},
        ),
        (
            "sm_89",
            {"triton_fp8_w8a8_gemm.py", "cuda_cublaslt_fp8.py", "cute_fp8_decoder_block.py"},
            {"cuda_pdl_gemv_chain.py", "triton_mxfp8_gemm.py", "cute_fp8_blockscaled_gemm.py"},
        ),
        (
            "sm_90",
            {"cuda_pdl_gemv_chain.py", "triton_fp8_kv_decode.py"},
            {"triton_mxfp8_gemm.py", "cute_fp8_blockscaled_gemm.py"},
        ),
        (
            "sm_100",
            {"triton_mxfp8_gemm.py", "cuda_pdl_gemv_chain.py"},
            {"cute_fp8_blockscaled_gemm.py"},
        ),
        ("sm_120", {"triton_mxfp8_gemm.py", "cute_fp8_blockscaled_gemm.py"}, set()),
    ],
)
def test_examples_run_where_they_declare(arch, runs, skips):
    tc = fake(arch)
    for name in runs:
        assert selftest.example_skip(name, tc) is None, name
    for name in skips:
        why = selftest.example_skip(name, tc)
        assert why and why.startswith("needs sm_") and f"this GPU is {arch}" in why, name
    assert selftest.w8a8_supported(tc) is (arch not in ("sm_75", "sm_80"))
    assert selftest.pdl_supported(tc) is (arch in ("sm_90", "sm_100", "sm_120"))
    # weight-only FP8: every GPU here but Turing (the skinny GEMM's bf16 m16n8k16 MMA)
    assert selftest.fp8_supported(tc) is (arch != "sm_75")
    assert selftest.cute_fp8_supported(tc) is (arch == "sm_120")


def test_a_missing_backend_is_a_reason_too():
    tc = fake("sm_120")
    tc.backends["cute"] = False
    assert selftest.example_skip("cute_fp8_blockscaled_gemm.py", tc, "cute") == (
        "the cute backend is not available"
    )


def test_doctor_probes_skip_features_the_gpu_lacks(monkeypatch, tmp_path):
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(toolchain, "setup", lambda apply_env=True: fake("sm_86", peaks=False))
    monkeypatch.setattr(toolchain, "_module_available", lambda name: True)
    fail = {name: pytest.fail for name in ("dot_scaled", "tma", "pdl")}
    result = probes.run(True, fail, capability=(8, 6))
    found = {p["name"]: p for p in result["probes"]}
    assert all(p["ok"] is None for p in found.values())
    assert "needs sm_12x" in found["dot_scaled"]["detail"]
    assert (
        "needs sm_90+" in found["tma"]["detail"] and "this GPU is sm_86" in found["pdl"]["detail"]
    )


def test_cute_dsl_names_each_familys_peak_mma():
    from kernel_agent import cute_dsl

    status = cute_dsl.Status(installed=True, version="4.8", arch="sm_90a", arch_known=True)
    status.block_scaled = False
    assert "warpgroup" in status.describe()[0]
    status.arch = "sm_100a"
    assert "tcgen05" in status.describe()[0]


# ------------------------------------------------------------------ knowledge


def test_gpus_knowledge_cites_fetchable_sources():
    text = gpu_arch.KNOWLEDGE.read_text()
    urls = re.findall(r"https://[^\s)>,]+", text)
    assert len(urls) >= 10
    for url in urls:
        assert web.allowed(url, web.domains()), url
    assert "## Sources" in text


def test_knowledge_sections_label_the_rtx_5070ti_evidence():
    for fam in gpu_arch.FAMILIES:
        section = gpu_arch.knowledge_section(fam.key)
        if fam.key != "blackwell_geforce":
            assert "5070" not in section, fam.key
    assert Path(gpu_arch.KNOWLEDGE).name == "gpus.md"
