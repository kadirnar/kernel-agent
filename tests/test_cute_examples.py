"""The CuTe DSL FP8 examples (issue #133): ``cute_fp8_blockscaled_gemm.py`` (W8A8 GEMM on
sm_120's block-scaled MMA, persistent, warp-specialised, fused epilogue) and
``cute_fp8_decoder_block.py`` (fused small-M norm + gated MLP + residual, FP8 weights).

* CPU: every kernel compiles for ``sm_120a`` with no GPU visible (fake tensors, a
  subprocess with ``CUDA_VISIBLE_DEVICES=``), the GEMM's PTX uses the block-scaled
  ``mma.sync`` (``QMMA.SF``) and TMA, and the compile cache reloads what it stored.
* GPU (``-m gpu``, sm_120): the examples' selftests and the evaluator in the near-lossless
  tier (rejected by the exact tier), as ``kernel-agent doctor --smoke`` runs them.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kernel_agent.agent.prompts import EXAMPLES_DIR

GEMM = EXAMPLES_DIR / "cute_fp8_blockscaled_gemm.py"
BLOCK = EXAMPLES_DIR / "cute_fp8_decoder_block.py"
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


def _cpu_env(tmp_path: Path) -> dict[str, str]:
    src = Path(__file__).resolve().parents[1] / "src"
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",  # nothing may run on a GPU
        "CUTE_DSL_ARCH": "sm_120a",
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
    errors = _load(BLOCK).selftest()
    assert max(errors.values()) < 1e-2


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUTE, reason="no CuTe DSL")
def test_cute_examples_through_the_evaluator(tmp_path):
    from kernel_agent import selftest

    if not _sm120():
        pytest.skip("the block-scaled MMA needs sm_120 / sm_121")
    assert selftest.smoke_fp8(
        tmp_path, True, precision="fp8_w8a8", examples=selftest.CUTE_W8A8_EXAMPLES
    )
    assert selftest.smoke_fp8(
        tmp_path, True, examples=selftest.CUTE_BLOCK_EXAMPLES, capture=selftest.make_mlp_capture
    )
