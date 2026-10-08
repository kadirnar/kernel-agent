"""The Helion backend (issue #229) on the CPU, with a stand-in for Helion.

Helion is not installed here: the stand-in (written to a temporary directory and put on the
path) runs a kernel's body eagerly as one tile and records what its autotuner is asked. It
checks the source classification, ``doctor``'s probe and status (missing, a requirement on
another torch refused), the sweep's ``strategy="helion"`` (the default config checked, the
autotuner run with the sweep's budget, seed and arch limits, the tuned config checked and
timed) and the config bound into the snapshot as text and built again.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from kernel_agent import backends, probes
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels import helion_tune
from kernel_agent.kernels import search as search_mod
from kernel_agent.kernels import sweep as sweep_mod
from kernel_agent.kernels.evaluate import load_candidate_module
from kernel_agent.profiling.capture import capture_calls
from kernel_agent.selftest import RMSNorm

FAKE_HELION = '''
"""A stand-in for Helion on the CPU (tests): a kernel runs its body eagerly, one tile."""
import dataclasses
import inspect
from collections.abc import Mapping

from . import language

#: Every autotune call: the kernel's name and its settings then
CALLS = []
#: What the stand-in's autotuner "finds"
TUNED = {
    "block_sizes": [8],
    "reduction_loops": [None],
    "num_warps": 2,
    "indexing": "pointer",
    "range_warp_specializes": [None],
}


class Config(Mapping):
    def __init__(self, **kwargs):
        self.config = dict(kwargs)

    def __getitem__(self, key):
        return self.config[key]

    def __iter__(self):
        return iter(self.config)

    def __len__(self):
        return len(self.config)


@dataclasses.dataclass
class Settings:
    static_shapes: bool = True
    autotune_effort: str = "full"
    autotune_budget_seconds: object = None
    autotune_random_seed: int = 0
    autotune_progress_bar: bool = True
    allow_warp_specialize: bool = True
    autotune_seed_configs: object = None


class Kernel:
    def __init__(self, fn, configs, settings):
        self.fn, self.name, self.configs, self.settings = fn, fn.__name__, configs, settings
        self.running = None  # what Helion's autotune leaves the kernel running
        self.resets = 0

    def reset(self):
        self.running = None
        self.resets += 1

    def normalize_args(self, *args, **kwargs):
        bound = inspect.signature(self.fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(bound.args)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def autotune(self, args, *, force=True, **options):
        seeds = self.settings.autotune_seed_configs
        CALLS.append(
            {
                "name": self.name,
                "force": force,
                "args": [tuple(a.shape) if hasattr(a, "shape") else a for a in args],
                **{k: v for k, v in vars(self.settings).items() if k != "autotune_seed_configs"},
                "seeds": [dict(c) for c in seeds or []],
            }
        )
        self.running = Config(**TUNED)
        return self.running


def kernel(fn=None, *, config=None, configs=None, **settings):
    configs = [config] if config is not None else list(configs or [])
    return Kernel(fn, configs, Settings(**settings))
'''

FAKE_LANGUAGE = """
import torch


def tile(sizes):
    if isinstance(sizes, (list, tuple)):
        yield tuple(slice(0, s) for s in sizes)
    else:
        yield slice(0, sizes)


def zeros(shape, dtype=torch.float32):
    return torch.zeros([s.stop - s.start for s in shape], dtype=dtype)
"""


@pytest.fixture
def fake_helion(tmp_path, monkeypatch):
    """The stand-in importable as ``helion`` (with its distribution's metadata when a test
    writes one); removed from ``sys.modules`` afterwards."""
    root = tmp_path / "site"
    (root / "helion").mkdir(parents=True)
    (root / "helion" / "__init__.py").write_text(FAKE_HELION)
    (root / "helion" / "language.py").write_text(FAKE_LANGUAGE)
    monkeypatch.syspath_prepend(str(root))
    for name in ("helion", "helion.language"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import helion

    yield root, helion
    for name in ("helion", "helion.language"):
        sys.modules.pop(name, None)


def dist_info(root: Path, *requires: str) -> None:
    info = root / "helion-9.9.0.dist-info"
    info.mkdir()
    lines = ["Metadata-Version: 2.1", "Name: helion", "Version: 9.9.0"]
    (info / "METADATA").write_text("\n".join(lines + [f"Requires-Dist: {r}" for r in requires]))


# ------------------------------------------------------------------ classification, status


def test_helion_sources_classify_as_helion():
    for name in ("helion_rmsnorm.py", "helion_gemm_epilogue.py"):
        source = (EXAMPLES_DIR / name).read_text()
        assert backends.classify(source) == "helion"  # kernels made in build (helion_tune)
    decorated = "import helion\n\n@helion.kernel(static_shapes=True)\ndef k(x):\n    return x\n"
    assert backends.classify(decorated) == "helion"
    assert backends.classify("k = helion.kernel(body, config=c)\n") == "helion"
    triton = "@triton.jit\ndef t(x):\n    pass\n"
    assert backends.classify(decorated + triton) == "triton+helion"
    assert backends.classify("import helion_tune_notes\n") == "torch"


def test_status_says_why_helion_cannot_run(fake_helion):
    root, _ = fake_helion
    with pytest.MonkeyPatch.context() as patch:  # the distribution's metadata is missing
        patch.setattr(helion_tune.importlib.metadata, "version", _versions({"torch": "2.14"}))
        assert helion_tune.status() == {"ok": False, "version": None, "why": helion_tune.MISSING}
    dist_info(root, "torch>=99", "numpy", 'jax; extra == "pallas"')
    found = helion_tune.status()
    assert not found["ok"] and found["version"] == "9.9.0"
    assert "helion requires torch>=99" in found["why"] and "must not change torch" in found["why"]
    installed = {"torch": "2.14.1+cu130", "triton": "3.8.0"}
    bad = helion_tune.mismatches(["triton<3", "torch>=2.0", "x", "not a requirement!"], installed)
    assert bad == ["helion requires triton<3, this environment has triton 3.8.0"]


def test_status_accepts_a_helion_that_keeps_this_torch(fake_helion):
    root, _ = fake_helion
    dist_info(root, "torch>=2.0", "numpy")
    assert helion_tune.status() == {"ok": True, "version": "9.9.0", "why": None}


@pytest.mark.skipif(importlib.util.find_spec("helion") is not None, reason="helion installed")
def test_doctor_skips_helion_with_the_reason_when_it_is_missing(monkeypatch):
    probe = probes.probe_helion()  # no GPU work: it stops at the missing package
    assert probe.ok is None and probe.detail == helion_tune.MISSING
    tc = type("Toolchain", (), {"gpu": None, "nvcc_version": None})()
    monkeypatch.setattr(probes.toolchain, "setup", lambda: tc)
    result = probes.run(True, {"helion": probes.probe_helion}, capability=(12, 0))
    assert result["probes"] == [{"name": "helion", "ok": None, "detail": helion_tune.MISSING}]
    assert "helion" in probes.PROBES and "helion" in result["versions"]
    assert "skipped: helion is not installed" in probes.describe(result)


def _versions(known: dict[str, str]) -> Any:
    def version(name: str) -> str:
        if name in known:
            return known[name]
        raise helion_tune.importlib.metadata.PackageNotFoundError(name)

    return version


# ------------------------------------------------------------------ the examples and the sweep


def test_the_examples_compute_the_reference_on_the_stand_in(fake_helion):
    del fake_helion  # the stand-in is importable as helion
    torch.manual_seed(0)
    norm = RMSNorm(64, eps=1e-5).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64) * 0.1 + 1)
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    module = load_candidate_module(EXAMPLES_DIR / "helion_rmsnorm.py")
    built = module.build(norm)
    assert torch.equal(built(x), norm(x))  # the reference's rounding, bit for bit
    assert built.kernel.configs == [] and built.kernel.settings.autotune_effort == "none"
    tuned = module.build(norm, helion_configs={"helion_rmsnorm": {"block_sizes": [4]}})
    assert dict(tuned.kernel.configs[0]) == {"block_sizes": [4]}  # its own kernel and config
    assert built.kernel.configs == []

    gemm = load_candidate_module(EXAMPLES_DIR / "helion_gemm_epilogue.py")
    linear = torch.nn.Linear(32, 48)
    with torch.no_grad():
        linear.bias.normal_()
    a = torch.randn(3, 7, 32)
    w_t, bias = linear.weight.t().contiguous(), linear.bias.detach()
    got = gemm.helion_linear(a.reshape(-1, 32), w_t, bias).view(3, 7, 48)
    torch.testing.assert_close(got, linear(a))
    assert torch.equal(gemm.build(linear)(a), linear(a))  # CPU: the reference math
    assert gemm.build(norm) is norm


def capture_of(path: Path) -> Path:
    torch.manual_seed(0)
    module = RMSNorm(64, eps=1e-5).to(torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(torch.randn(64) * 0.1 + 1)
    calls = [
        ((torch.randn(1, 16, 64, dtype=torch.bfloat16),), {}, 1),
        ((torch.randn(1, 1, 64, dtype=torch.bfloat16),), {}, 7),
    ]
    capture_calls(module.eval(), calls, path)
    return path


def cpu_timer(fn, args, kwargs, *, l2_flush=False, target_ms=60.0, keep=False):
    with torch.inference_mode():
        fn(*args, **kwargs)
    return {"median_ms": 1e-3}


@pytest.mark.parametrize(("arch", "warp_specialize"), [((12, 0), False), ((9, 0), True)])
def test_a_helion_sweep_tunes_with_helions_autotuner_and_binds_its_config(
    tmp_path, fake_helion, monkeypatch, arch, warp_specialize
):
    _, helion = fake_helion
    for name in sweep_mod._HELION_ENV:
        monkeypatch.delenv(name, raising=False)
    capture = capture_of(tmp_path / "c.pt")
    candidate = tmp_path / "helion_rmsnorm.py"
    candidate.write_text((EXAMPLES_DIR / "helion_rmsnorm.py").read_text())
    spec = search_mod.spec_from(None, None, "helion", 11)
    spec |= {"arch": {"capability": list(arch), "smem_per_block": 101_376}, "warm": False}
    helion.CALLS.clear()

    out = sweep_mod.sweep(
        capture, candidate, [], device="cpu", timer=cpu_timer, deadline=None, search=spec
    )
    (call,) = helion.CALLS  # one kernel, tuned once
    assert call["name"] == "helion_rmsnorm" and call["force"]
    assert call["autotune_effort"] == "full" and call["autotune_random_seed"] == 11
    assert call["allow_warp_specialize"] is warp_specialize and not call["autotune_progress_bar"]
    assert call["autotune_budget_seconds"] is None  # no deadline here: Helion's own budget
    assert call["args"] == [(16, 64), (64,), 1e-05]  # the case with the most work
    assert out["helion"]["kernels"] == ["helion_rmsnorm"] and "error" not in out["helion"]
    assert out["helion"]["case"] == "a0[1, 16, 64]:bfloat16"
    tuned = {"helion_configs": {"helion_rmsnorm": helion.TUNED}}
    configs = sorted((r["index"], r["config"]) for r in out["table"])
    assert configs == [(0, {}), (1, tuned)] and all(r["correct"] for r in out["table"])
    assert all(r.get("speedup") for r in out["table"])  # both timed against the reference
    assert "HELION_AUTOTUNE_EFFORT" not in os.environ  # set for the sweep only
    info = {"configs": 2, "passed": 2, "failed": 0, "skipped": 0, "seconds": 1.0}
    info |= {"helion": out["helion"] | {"seconds": 3.0}, "table": out["table"]}
    text = sweep_mod.format_table({"sweep": info, "evaluation": {"status": "ok"}, "config": {}})
    assert "helion: tuned helion_rmsnorm in 3.0 s (on case a0[1, 16, 64]:bfloat16)" in text

    # bound into the snapshot as text, built again: the tuned config, no tuning
    source = sweep_mod.bind_config(candidate.read_text(), tuned)
    assert "_KA_SWEEP_CONFIG = {'helion_configs': {'helion_rmsnorm': {" in source
    bound = tmp_path / "bound.py"
    bound.write_text(source)
    module = load_candidate_module(bound)
    built = module.build(RMSNorm(64).to(torch.bfloat16))
    assert dict(built.kernel.configs[0]) == helion.TUNED
    assert built.kernel.settings.autotune_effort == "none" and len(helion.CALLS) == 1
    assert "helion_configs=" in sweep_mod.label(tuned) and len(sweep_mod.label(tuned)) < 120


def test_seeds_and_errors_of_the_autotune(fake_helion):
    _, helion = fake_helion
    module = load_candidate_module(EXAMPLES_DIR / "helion_rmsnorm.py")
    built = module.build(RMSNorm(16))
    x = torch.randn(3, 16)
    helion.CALLS.clear()
    seeds = {"helion_rmsnorm": [{"block_sizes": [2]}]}
    found = helion_tune.autotune(
        helion_tune.kernels(vars(module), vars(built)),
        lambda: built(x),
        seconds=30,
        seed=1,
        seeds=seeds,
    )
    assert found["configs"] == {"helion_rmsnorm": helion.TUNED}
    assert found["kernels"] == ["helion_rmsnorm"]
    assert helion.CALLS[0]["seeds"] == [{"block_sizes": [2]}]
    assert 20 <= helion.CALLS[0]["autotune_budget_seconds"] <= 30  # the sweep's time left
    assert built.kernel.settings.autotune_effort == "none"  # restored after
    assert built.kernel.settings.autotune_seed_configs is None
    # reset: the candidate tuned on runs its own (default) config again, not the tuned one
    assert built.kernel.running is None and built.kernel.resets == 1
    assert "no Helion kernel" in helion_tune.autotune([], lambda: None)["error"]
    idle = helion_tune.autotune(helion_tune.kernels(vars(built)), lambda: None)
    assert "launched none" in idle["error"]
    assert helion_tune.kernels({"a": 1, "k": built.kernel}, {"k2": built.kernel}) == [built.kernel]
