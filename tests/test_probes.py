"""``doctor`` probes (#146): results, failures and skips are reported, never raised; the
versions are recorded. GPU-marked: the probes themselves (not run in #146)."""

from __future__ import annotations

import pytest

from kernel_agent import probes, toolchain
from kernel_agent.probes import Probe


class FakeToolchain:
    gpu = None  # no CUDA lookups in a CPU test
    nvcc_version = "13.0"
    torch_version = "2.14"


@pytest.fixture
def no_gpu_lookup(monkeypatch, tmp_path):
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(toolchain, "setup", lambda apply_env=True: FakeToolchain())


def test_probe_results_failures_and_skips_are_reported(no_gpu_lookup):

    def broken() -> Probe:
        raise RuntimeError("no kernel image for this GPU")

    fake = {
        "dot_scaled": lambda: Probe("dot_scaled", True, "lowers to block_scale (SASS QMMA.SF)"),
        "pdl": broken,
    }
    result = probes.run(True, fake)
    assert result["probes"] == [
        {"name": "dot_scaled", "ok": True, "detail": "lowers to block_scale (SASS QMMA.SF)"},
        {"name": "pdl", "ok": False, "detail": "RuntimeError: no kernel image for this GPU"},
    ]
    assert result["versions"]["torch"] and "triton" in result["versions"]
    text = probes.describe(result)
    assert "  dot_scaled: ok: lowers to block_scale" in text
    assert "  pdl: FAILED: RuntimeError" in text and text.startswith("versions: torch ")
    skipped = probes.run(False, fake)
    assert {p["ok"] for p in skipped["probes"]} == {None}
    assert "skipped: no CUDA GPU" in probes.describe(skipped)


def test_a_probe_whose_package_is_missing_is_skipped(monkeypatch, no_gpu_lookup):
    monkeypatch.setattr(toolchain, "_module_available", lambda name: name != "cuda.core")
    result = probes.run(True, {"pdl": pytest.fail})
    assert result["probes"] == [{"name": "pdl", "ok": None, "detail": "cuda.core is not installed"}]


@pytest.mark.gpu
def test_the_probes_on_this_gpu():  # not run in #146: `kernel-agent doctor` runs them
    result = probes.run(True)
    found = {p["name"]: p for p in result["probes"]}
    assert set(found) == set(probes.PROBES)
    assert found["tma"]["ok"] in (True, None), found["tma"]
    assert found["green_contexts"]["ok"] in (True, None), found["green_contexts"]
