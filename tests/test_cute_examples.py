"""The CuTe DSL FP8 examples (issue #133): ``cute_fp8_blockscaled_gemm.py`` (W8A8 GEMM on
sm_120's block-scaled MMA, persistent, warp-specialised, fused epilogue) and
``cute_fp8_decoder_block.py`` (fused small-M norm + gated MLP + residual, FP8 weights).

* CPU: every kernel compiles for ``sm_120a`` with no GPU visible (fake tensors, a
  subprocess with ``CUDA_VISIBLE_DEVICES=``), the GEMM's PTX uses the block-scaled
  ``mma.sync`` (``QMMA.SF``) and TMA, and the compile cache reloads what it stored.
* GPU (``-m gpu``, sm_120): the examples' selftests and the evaluator in the near-lossless
  tier (rejected by the exact tier), as ``kernel-agent doctor --smoke`` runs them.

The Hopper / datacenter Blackwell templates (#228): ``cute_sm90_gemm_ws.py`` (TMA + wgmma,
producer / consumer warpgroups, persistent, fused epilogue, swap-AB) and
``cute_sm100_gemm_tcgen05.py`` (TMA + tcgen05.mma into TMEM, persistent, 2-CTA pairs):

* CPU: they compile for ``sm_90a`` / ``sm_100a`` with no GPU visible and their PTX has the
  instructions of their family (``wgmma.mma_async`` + ``cp.async.bulk.tensor``;
  ``tcgen05.mma`` + ``tcgen05.alloc`` + ``tcgen05.ld``) and never ``mma.sync``; on faked
  sm_86 / sm_89 / sm_120 (and each other's) GPUs they are skipped with the reason.
* GPU (``-m gpu``): their selftests and the evaluator on an sm_90 / sm_100 GPU (skipped
  elsewhere: not run on those GPUs yet).
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kernel_agent import backends, gpu_arch, selftest, toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR

GEMM = EXAMPLES_DIR / "cute_fp8_blockscaled_gemm.py"
BLOCK = EXAMPLES_DIR / "cute_fp8_decoder_block.py"
SM90 = EXAMPLES_DIR / "cute_sm90_gemm_ws.py"
SM100 = EXAMPLES_DIR / "cute_sm100_gemm_tcgen05.py"
HAS_CUTE = importlib.util.find_spec("cutlass") is not None

COMPILE_ALL = r"""
import importlib.util, sys
from kernel_agent import cute_dsl

def load(path):
    spec = importlib.util.spec_from_file_location("ex_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

gemm, block = load(sys.argv[1]), load(sys.argv[2])
gemm.compile_quant(1024)
gemm.compile_gemm(1024, 1024, max_ctas=70)
gemm.compile_gemm(2560, 1024, has_bias=True, has_residual=True)
gemm.compile_gemm(1024, 2048, out_fp8=True)
block.compile_rows_gemv(16, 1024, 4096, norm=True, gated=True)
block.compile_rows_gemv(4, 4096, 1024, residual=True)
first = cute_dsl.STATS.misses
gemm.compile_gemm(1024, 1024, max_ctas=70)  # the same kernel again: from the cache
print("MISSES", first, "HITS", cute_dsl.STATS.hits, "ERRORS", cute_dsl.STATS.errors)
"""


def _cpu_env(tmp_path: Path, arch: str = "sm_120a") -> dict[str, str]:
    src = Path(__file__).resolve().parents[1] / "src"
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",  # nothing may run on a GPU
        "CUTE_DSL_ARCH": arch,
        "KERNEL_AGENT_CUTE_CACHE": str(tmp_path / "cache"),
        "CUTE_DSL_KEEP": "ptx",
        "CUTE_DSL_DUMP_DIR": str(tmp_path / "dump"),
        "PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")]),
    }


@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_examples_compile_for_sm120_without_a_gpu(tmp_path):
    (tmp_path / "dump").mkdir()
    out = subprocess.run(
        [sys.executable, "-c", COMPILE_ALL, str(GEMM), str(BLOCK)],
        env=_cpu_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "MISSES 6 HITS 1 ERRORS []" in out.stdout, out.stdout + out.stderr[-2000:]
    ptx = {p.name: p.read_text() for p in (tmp_path / "dump").glob("*.ptx")}
    gemms = [text for name, text in ptx.items() if "BlockScaledGemm" in name]
    assert len(gemms) == 3
    for text in gemms:
        # full-rate FP8 (QMMA.SF), never the half-rate plain e4m3 mma (QMMA.F32)
        assert "mma.sync.aligned.kind::mxf8f6f4.block_scale" in text
        plain = [
            line for line in text.splitlines() if "mma.sync" in line and "block_scale" not in line
        ]
        assert plain == []
        assert "cp.async.bulk.tensor" in text and "setmaxnreg" in text
    fp8_out = next(t for n, t in ptx.items() if "BlockScaledGemm" in n and "2048i64" in n)
    assert fp8_out.count("cvt.rn.satfinite.e4m3x2.f32") >= 16  # every accumulator converted


def _sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_blockscaled_gemm_selftest():
    if not _sm120():
        pytest.skip("the block-scaled MMA needs sm_120 / sm_121")
    errors = _load(GEMM).selftest(m=352, n=1024, k=1024)
    assert errors["bias"] < 1e-2 and errors["residual"] < 1e-2


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_decoder_block_selftest():
    # its ARCHS (sm_89+: the e4m3 -> f16 conversion); an A10 (sm_86) failed in NVVM here
    if (why := gpu_arch.example_skip(BLOCK, torch.cuda.get_device_capability())) is not None:
        pytest.skip(why)
    errors = _load(BLOCK).selftest()
    assert max(errors.values()) < 1e-2


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_cute_examples_through_the_evaluator(tmp_path):
    if not _sm120():
        pytest.skip("the block-scaled MMA needs sm_120 / sm_121")
    assert selftest.smoke_fp8(
        tmp_path, True, precision="fp8_w8a8", examples=selftest.CUTE_W8A8_EXAMPLES
    )
    assert selftest.smoke_fp8(
        tmp_path, True, examples=selftest.CUTE_BLOCK_EXAMPLES, capture=selftest.make_mlp_capture
    )


# ------------------------------------------------------------------ sm_90 / sm_100 templates

COMPILE_ARCH = r"""
import importlib.util, json, os, sys
from pathlib import Path
from kernel_agent import cute_dsl

spec = importlib.util.spec_from_file_location("ex_arch", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
configs, out = json.loads(sys.argv[2]), Path(sys.argv[3])
first = None
for name, cfg in [*configs.items(), ("quant", None)]:
    fn = mod.compile_quant(1024) if cfg is None else mod.compile_gemm(**cfg)
    # the dump's file name is truncated (configs collide): keep each PTX right away
    ptx = fn.__ptx__  # the text, or the dumped file's path
    (out / f"{name}.ptx").write_text(Path(ptx).read_text() if os.path.isfile(ptx) else ptx)
    first = first or cfg
mod.compile_gemm(**first)  # the same kernel again: from the cache
print("MISSES", cute_dsl.STATS.misses, "HITS", cute_dsl.STATS.hits, "ERRORS", cute_dsl.STATS.errors)
"""

_NK = {"n": 1024, "k": 1024}
_E4M3_OUT = {**_NK, "has_bias": True, "has_residual": True, "out_fp8": True}
_SM90_CONFIGS = {  # name: (compile_gemm keywords, what its PTX holds)
    "fp8": ({**_NK, "max_ctas": 132}, ["wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3"]),
    "bf16_bias": (
        {**_NK, "fp8": False, "has_bias": True},
        ["wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16"],
    ),
    # every accumulator converted to e4m3 with saturation
    "fp8_res_e4m3": (_E4M3_OUT, ["cvt.rn.satfinite.e4m3x2.f32"] * 16),
    "swap_ab": (
        {**_NK, "swap_ab": True, "has_bias": True},
        ["wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3"],
    ),
    "cluster": (
        {**_NK, "cluster_m": 2, "tile_n": 256, "max_ctas": 66},
        ["m64n256k32", "multicast::cluster"],
    ),
}
_SM100_CONFIGS = {
    "fp8": ({**_NK, "max_ctas": 148}, ["tcgen05.mma.cta_group::1.kind::f8f6f4"]),
    "bf16_bias": ({**_NK, "fp8": False, "has_bias": True}, ["tcgen05.mma.cta_group::1.kind::f16"]),
    "fp8_res_e4m3": (_E4M3_OUT, ["cvt.rn.satfinite.e4m3x2.f32"] * 16),
    "two_cta": (
        {**_NK, "two_cta": True, "max_ctas": 74},
        ["tcgen05.mma.cta_group::2.kind::f8f6f4", "multicast::cluster"],
    ),
    "bf16_two_cta": (
        {**_NK, "fp8": False, "two_cta": True, "tile_n": 256, "has_residual": True},
        ["tcgen05.mma.cta_group::2.kind::f16"],
    ),
}
_TCGEN05 = ("tcgen05.mma", "tcgen05.alloc", "tcgen05.ld", "tcgen05.dealloc", "cp.async.bulk.tensor")
#: Per CuTe DSL arch: the template, its configs, what every GEMM's PTX must hold and what it
#: must not (the other families' MMAs, the slow `mma.sync` path).
ARCH_TEMPLATES = {
    "sm_90a": (
        SM90,
        _SM90_CONFIGS,
        ("wgmma.mma_async", "cp.async.bulk.tensor", "setmaxnreg"),
        ("tcgen05", "mma.sync"),
    ),
    "sm_100a": (SM100, _SM100_CONFIGS, _TCGEN05, ("wgmma", "mma.sync")),
    # Blackwell Ultra (B300): the same template, its own arch-specific target
    "sm_103a": (
        SM100,
        {k: _SM100_CONFIGS[k] for k in ("fp8", "two_cta")},
        _TCGEN05,
        ("wgmma", "mma.sync"),
    ),
}


@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
@pytest.mark.parametrize("arch", sorted(ARCH_TEMPLATES))
def test_arch_templates_compile_for_their_arch_without_a_gpu(tmp_path, arch):
    path, configs, need, never = ARCH_TEMPLATES[arch]
    (tmp_path / "dump").mkdir()
    (tmp_path / "ptx").mkdir()
    keywords = json.dumps({name: cfg for name, (cfg, _) in configs.items()})
    out = subprocess.run(
        [sys.executable, "-c", COMPILE_ARCH, str(path), keywords, str(tmp_path / "ptx")],
        env=_cpu_env(tmp_path, arch),
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    misses = len(configs) + 1  # and the quantiser
    assert f"MISSES {misses} HITS 1 ERRORS []" in out.stdout, out.stdout + out.stderr[-2000:]
    ptx = {name: (tmp_path / "ptx" / f"{name}.ptx").read_text() for name in [*configs, "quant"]}
    assert f".target {arch}" in ptx["quant"]
    for name, (_, holds) in configs.items():
        text = ptx[name]
        assert f".target {arch}" in text, name
        for instruction in [*need, *holds]:  # a form listed n times: at least n of them
            assert text.count(instruction) >= max(1, holds.count(instruction)), (name, instruction)
        for instruction in never:
            assert instruction not in text, (name, instruction)


def _fake_toolchain(capability: tuple[int, int]) -> toolchain.Toolchain:
    gpu = toolchain.GPUInfo("GPU", capability, 80.0, 132, 50.0, 227.0, 228.0)
    on = {"cuda": True, "triton": True, "cute": True, "nvrtc": True, "tilelang": False}
    return toolchain.Toolchain(gpu, "2.14", "13.0", None, "13.0", on, [], {}, None)


@pytest.mark.parametrize(
    ("capability", "runs"),
    [
        ((8, 6), set()),
        ((8, 9), set()),
        ((9, 0), {SM90.name}),
        ((10, 0), {SM100.name}),
        ((10, 3), {SM100.name}),
        ((12, 0), set()),
    ],
)
def test_arch_templates_run_only_on_their_family(capability, runs):
    tc = _fake_toolchain(capability)
    arch = gpu_arch.arch_of(capability)
    assert set(selftest.CUTE_ARCH_EXAMPLES) == {SM90.name, SM100.name}
    for name in selftest.CUTE_ARCH_EXAMPLES:
        why = selftest.example_skip(name, tc, "cute")
        if name in runs:
            assert why is None, name
        else:  # doctor --smoke lists it as skipped, with the reason
            assert why and why.startswith("needs sm_") and f"this GPU is {arch}" in why, why
            assert ("wgmma" if name == SM90.name else "tcgen05") in why


def test_policy_and_knowledge_name_the_templates():
    hopper = backends.policy("fp8_gemm", gpu_arch.Facts("H100", (9, 0)))
    blackwell = backends.policy("fp8_gemm", gpu_arch.Facts("B200", (10, 0)))
    assert SM90.name in hopper.first and "sm90-wgmma.md" in hopper.first
    assert SM100.name in blackwell.first and "sm100-tcgen05.md" in blackwell.first
    geforce = backends.policy("fp8_gemm", gpu_arch.Facts("RTX 5090", (12, 0)))
    assert SM90.name not in geforce.first and SM100.name not in geforce.first
    assert "sm90-wgmma.md" in gpu_arch.knowledge_section("hopper")
    assert "sm100-tcgen05.md" in gpu_arch.knowledge_section("blackwell")
    assert "wgmma.md" not in gpu_arch.knowledge_section("blackwell_geforce")


def _capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major, minor


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_sm90_template_selftest():
    if _capability() != (9, 0):
        pytest.skip("wgmma needs sm_90 (not run on an H100 yet)")
    errors = _load(SM90).selftest()
    assert max(errors.values()) < 8e-2


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_sm100_template_selftest():
    if (_capability() or (0, 0))[0] != 10:
        pytest.skip("tcgen05 needs sm_100 / sm_103 (not run on a B200 yet)")
    errors = _load(SM100).selftest()
    assert max(errors.values()) < 8e-2


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_arch_templates_through_the_evaluator(tmp_path):
    cap = _capability()
    here = {
        name: spec
        for name, spec in selftest.CUTE_ARCH_EXAMPLES.items()
        if gpu_arch.example_skip(EXAMPLES_DIR / name, cap) is None
    }
    if not here:
        pytest.skip(f"no template for {gpu_arch.arch_of(cap)} (sm_90 / sm_100 only)")
    assert selftest.smoke_fp8(tmp_path, True, precision="fp8_w8a8", examples=here)
