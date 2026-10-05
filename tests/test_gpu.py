"""GPU tests: evaluator, side-effect checking, patcher, profiler, capture."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.hub import Modality
from kernel_agent.integrate.patcher import KernelPatch, apply_kernels
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import capture_calls, capture_module, load_capture
from kernel_agent.profiling.profiler import profile_workload, summarize
from kernel_agent.selftest import RMSNorm, make_rmsnorm_capture
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def rms_capture(tmp_path_factory):
    return make_rmsnorm_capture(tmp_path_factory.mktemp("cap") / "rms.pt", hidden=1024)


def test_triton_example_is_correct_and_timed(rms_capture):
    result = evaluate(rms_capture, EXAMPLES_DIR / "triton_rmsnorm.py")
    assert result["status"] == "ok", result
    assert result["correct"] and result["speedup"] > 0
    assert len(result["cases"]) == 2
    assert all(c["new_ms"] > 0 for c in result["cases"])


def test_wrong_candidate_is_rejected(rms_capture, tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch\n"
        "class M(torch.nn.Module):\n"
        "    def __init__(self, r):\n"
        "        super().__init__(); self.r = r\n"
        "    def forward(self, x):\n"
        "        return self.r(x) * 1.1\n"
        "def build(r):\n    return M(r)\n"
    )
    result = evaluate(rms_capture, bad)
    assert result["status"] == "incorrect" and not result["correct"]


def test_build_errors_are_reported(rms_capture, tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def build(r):\n    raise RuntimeError('nope')\n")
    result = evaluate(rms_capture, broken)
    assert result["status"] == "build_error" and "nope" in result["error"]
    same = tmp_path / "same.py"
    same.write_text("def build(r):\n    return r\n")
    assert evaluate(rms_capture, same)["status"] == "build_error"


class CacheBox:
    def __init__(self, n: int) -> None:
        self.state = torch.zeros(n, device="cuda")


class Accumulate(nn.Module):
    """Returns x*2 and adds x into a cache object in place (like a KV-cache append)."""

    def forward(self, x: torch.Tensor, cache: CacheBox) -> torch.Tensor:
        cache.state.add_(x)
        return x * 2


def test_in_place_side_effects_are_checked(tmp_path):
    capture = tmp_path / "acc.pt"
    x = torch.randn(64, device="cuda")
    capture_calls(Accumulate(), [((x,), {"cache": CacheBox(64)}, 1)], capture)
    forgets = tmp_path / "forgets.py"
    forgets.write_text(
        "import torch\n"
        "class M(torch.nn.Module):\n"
        "    def forward(self, x, cache):\n"
        "        return x * 2\n"
        "def build(r):\n    return M()\n"
    )
    result = evaluate(capture, forgets)
    assert result["status"] == "incorrect"
    assert any("kwargs" in f["name"] for f in result["cases"][0]["failures"])
    good = tmp_path / "good.py"
    # x * 2 through a kernel the reference does not launch (lerp): a candidate that
    # only re-runs the reference's ops is a `fallback`
    good.write_text(
        "import torch\n"
        "class M(torch.nn.Module):\n"
        "    def forward(self, x, cache):\n"
        "        cache.state += x\n"
        "        return torch.lerp(x, x * 3, 0.5)\n"
        "def build(r):\n    return M()\n"
    )
    assert evaluate(capture, good)["correct"]


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(RMSNorm(256), nn.Linear(256, 256)) for _ in range(3)
        )
        self.norm = RMSNorm(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)


class TinyWorkload(Workload):
    modality = Modality.LLM

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = TinyModel().cuda().to(torch.bfloat16).eval()

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> Any:
        return torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16)

    def run(self, inputs: Any) -> Any:
        return self.model(inputs).float().cpu()

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.allclose(reference, candidate, atol=0.05, rtol=0.05)))


@pytest.fixture
def tiny():
    w = TinyWorkload(WorkloadSpec(repo_id="tiny/model", modality="llm"))
    w.load()
    return w


def test_profiler_and_capture(tiny, tmp_path):
    inputs = tiny.make_inputs()
    tiny.run(inputs)
    profile = profile_workload(tiny, inputs)
    classes = {c["cls"]: c for c in profile["classes"]}
    assert classes["RMSNorm"]["instances"] == 4 and classes["RMSNorm"]["calls"] == 4
    assert classes["RMSNorm"]["is_leaf"]
    assert profile["kernel_view"]["kernel_launches"] > 0
    assert "RMSNorm" in summarize(profile, 1.0)
    info = capture_module(tiny, inputs, "RMSNorm", tmp_path / "cap.pt")
    assert info["qualname"] == "model.layers.0.0"
    cap = load_capture(tmp_path / "cap.pt", device="cuda")
    assert cap["instances"] == 4 and cap["cases"][0]["count"] == 1


def test_patcher_replaces_all_instances(tiny):
    inputs = tiny.make_inputs()
    ref = tiny.run(inputs)
    report = apply_kernels(
        tiny.roots(),
        [KernelPatch("rms", "RMSNorm", Path(EXAMPLES_DIR / "triton_rmsnorm.py"))],
    )
    assert report.replaced == {"rms": 4} and not report.errors
    assert not any(isinstance(m, RMSNorm) for m in tiny.model.modules())
    assert tiny.compare(ref, tiny.run(inputs)).passed
