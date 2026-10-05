"""Compile compatibility of kernel candidates: graph breaks + compiled outputs
(CPU with the ``aot_eager`` backend; GPU: the bundled Triton examples)."""

import pytest
import torch
from torch import nn

from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels.compile_check import check


class Breaks(nn.Module):
    """Stands in for an opaque launcher (pybind / NVRTC handle) between two ops."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        torch._dynamo.graph_break()
        return y + 1


@torch.library.custom_op("ka_test_compile_check::twice_plus_one", mutates_args=())
def twice_plus_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2 + 1


@twice_plus_one.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class Opaque(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return twice_plus_one(x)


class Accumulate(nn.Module):
    """In-place side effect on an argument (like a KV-cache write)."""

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        cache.add_(x)
        return x * 2 + 1


def _cases(*, cache: bool = False):
    x = torch.randn(4, 8)
    if not cache:
        return [
            {"args": (x,), "kwargs": {}, "output": x * 2 + 1, "post_args": (x,), "post_kwargs": {}}
        ]
    c = torch.zeros(4, 8)
    return [
        {
            "method": "forward",
            "args": (x,),
            "kwargs": {"cache": c},
            "output": x * 2 + 1,
            "post_args": (x,),
            "post_kwargs": {"cache": c + x},
        }
    ]


def test_graph_breaks_are_reported():
    result = check(Breaks, _cases())
    assert result["backend"] == "aot_eager"  # CPU cases: tracing only
    assert result["passed"] and result["graph_breaks"] == 1 and not result["fullgraph_ok"]
    assert "graph_break" in result["break_reasons"][0]
    assert "test_compile_check.py" in result["break_reasons"][0]


def test_custom_op_compiles_without_breaks():
    result = check(Opaque, _cases())
    assert result["passed"] and result["graph_breaks"] == 0 and result["fullgraph_ok"]
    assert result["graphs"] == 1 and result["cases"][0]["ok"]


def test_side_effects_and_wrong_outputs_are_checked():
    assert check(Accumulate, _cases(cache=True))["passed"]

    class Forgets(nn.Module):
        def forward(self, x, cache):
            return x * 2 + 1

    result = check(Forgets, _cases(cache=True))
    assert not result["passed"] and not result["cases"][0]["ok"]


def test_build_errors_are_recorded():
    def build():
        raise RuntimeError("nope")

    result = check(build, _cases())
    assert not result["passed"] and not result["fullgraph_ok"] and "nope" in result["error"]


@pytest.mark.gpu
def test_compile_check_on_the_triton_examples(tmp_path):
    from kernel_agent.kernels.evaluate import evaluate
    from kernel_agent.selftest import make_rmsnorm_capture

    capture = make_rmsnorm_capture(tmp_path / "rms.pt", hidden=1024)
    plain = evaluate(capture, EXAMPLES_DIR / "triton_rmsnorm.py", compile_check=True)
    assert plain["correct"] and plain["status"] == "ok", plain
    compat = plain["compile_check"]
    assert compat["backend"] == "inductor" and compat["passed"], compat
    assert len(compat["cases"]) == 2

    wrapped = evaluate(capture, EXAMPLES_DIR / "triton_rmsnorm_custom_op.py", compile_check=True)
    assert wrapped["correct"] and wrapped["speedup"] > 0, wrapped
    compat = wrapped["compile_check"]
    assert compat["passed"] and compat["graph_breaks"] == 0 and compat["fullgraph_ok"], compat
