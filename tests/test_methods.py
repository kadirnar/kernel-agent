"""Non-``forward`` entrypoints (``forward_step``): discovery, profiling, capture, evaluation.

The toy model mirrors VoxCPM's MiniCPM decode loop: ``Model.forward_step`` →
``Layer.forward_step`` → ``Attention.forward_step(x, position_id, kv_cache)``,
which writes into per-layer views of one big ``[2, layers, B, T, D]`` cache.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from torch import nn

from kernel_agent.agent import prompts
from kernel_agent.hub import Modality
from kernel_agent.integrate.export import APPLY_TEMPLATE
from kernel_agent.integrate.patcher import KernelPatch, PatchError, apply_kernels
from kernel_agent.kernels.compare import compare_side_effects
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import (
    _detach,
    _Recorder,
    capture_calls,
    capture_module,
    load_capture,
)
from kernel_agent.profiling.methods import (
    discover_entrypoints,
    entrypoints_of,
    instrument,
    parse_entrypoints,
    workload_entrypoints,
)
from kernel_agent.profiling.profiler import ModuleTimer, summarize
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec

D = 32
MAX_LEN = 4096  # 2 x 2 layers x 4096 x 32 fp32 = 2 MiB cache, 512 KiB per K/V view
LAYERS = 2
PREFIX = 5
STEPS = 6


class Attention(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-1, -2) / math.sqrt(x.size(-1))
        causal = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        return self.o(weights @ v), (k, v)

    def forward_step(
        self,
        x: torch.Tensor,
        position_id: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        key_cache, value_cache = kv_cache
        key_cache[:, position_id] = k.unsqueeze(1)
        value_cache[:, position_id] = v.unsqueeze(1)
        valid = torch.arange(key_cache.size(1), device=x.device) <= position_id
        scores = (key_cache @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(x.size(-1))
        weights = scores.masked_fill(~valid, float("-inf")).softmax(-1)
        return self.o((weights.unsqueeze(-1) * value_cache).sum(1))


class Layer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.attn = Attention(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, kv = self.attn(x)
        x = x + h
        return x + self.mlp(x), kv

    def forward_step(
        self, x: torch.Tensor, position_id: torch.Tensor, kv_cache: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        x = x + self.attn.forward_step(x, position_id, kv_cache)
        return x + self.mlp(x)


class Model(nn.Module):
    def __init__(self, d: int = D, layers: int = LAYERS) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer(d) for _ in range(layers))
        self.cache: torch.Tensor | None = None

    def setup_cache(self, batch: int) -> None:
        p = next(self.parameters())
        self.cache = torch.zeros(
            2, len(self.layers), batch, MAX_LEN, D, dtype=p.dtype, device=p.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.cache is not None
        self.cache.zero_()
        for i, layer in enumerate(self.layers):
            x, (k, v) = layer(x)
            self.cache[0, i, :, : x.size(1)] = k
            self.cache[1, i, :, : x.size(1)] = v
        return x

    def forward_step(self, x: torch.Tensor, position_id: torch.Tensor) -> torch.Tensor:
        assert self.cache is not None
        for i, layer in enumerate(self.layers):
            x = layer.forward_step(x, position_id, (self.cache[0, i], self.cache[1, i]))
        return x


class ToyWorkload(Workload):
    """Prefill ``PREFIX`` positions through ``forward``, then ``STEPS`` ``forward_step`` calls."""

    modality = Modality.LLM

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = Model().to(self.device).eval()
        self.model.setup_cache(1)

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(1, PREFIX + STEPS, D, device=self.device)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            h = self.model(inputs[:, :PREFIX])
            outs = [h[:, -1]]
            for step in range(STEPS):
                pos = torch.tensor([PREFIX + step], device=self.device)
                outs.append(self.model.forward_step(inputs[:, PREFIX + step], pos))
            return torch.stack(outs, 1).float().cpu()

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.allclose(reference, candidate, atol=1e-4, rtol=1e-4)))


@pytest.fixture
def toy() -> ToyWorkload:
    w = ToyWorkload(WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cpu"))
    w.load()
    return w


SAME_STEP = """
import copy
import math

import torch


def build(reference):
    base = type(reference)

    class Fast(base):
        def forward_step(self, x, position_id, kv_cache):
            q, k, v = self.q(x), self.k(x), self.v(x)
            key_cache, value_cache = kv_cache
            key_cache[:, position_id] = k.unsqueeze(1)
            value_cache[:, position_id] = {value}
            valid = torch.arange(key_cache.size(1), device=x.device) <= position_id
            scores = (key_cache @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(x.size(-1))
            weights = scores.masked_fill(~valid, float("-inf")).softmax(-1)
            return self.o((weights.unsqueeze(-1) * value_cache).sum(1)) * {scale}

    new = copy.copy(reference)
    new.__class__ = Fast
    return new
"""

FORWARD_ONLY = """
import torch


class OnlyForward(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.ref = ref

    def forward(self, x):
        return self.ref(x)


def build(reference):
    return OnlyForward(reference)
"""


def _candidate(tmp_path: Path, name: str, value: str = "v.unsqueeze(1)", scale: str = "1") -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(SAME_STEP.format(value=value, scale=scale))
    return path


# ------------------------------------------------------------------ discovery


def test_discover_entrypoints(toy):
    found = discover_entrypoints(toy.roots())
    assert found == {Model: ["forward_step"], Layer: ["forward_step"], Attention: ["forward_step"]}

    class Sub(Attention):
        pass

    class Odd(nn.Module):
        def _forward_impl(self) -> None: ...
        def generate(self) -> None: ...
        def prefill(self) -> None: ...
        def custom(self) -> None: ...

        def decode_stream(self):  # generators are not entrypoints
            yield 1

    assert entrypoints_of(Sub) == ["forward_step"]  # inherited from a user class
    assert entrypoints_of(nn.Linear) == []
    assert entrypoints_of(Odd) == ["prefill"]
    extra = {"Odd": ["custom", "missing", "forward"]}
    assert entrypoints_of(Odd, extra) == ["prefill", "custom"]


def test_parse_entrypoints(toy):
    assert parse_entrypoints("A.step_x, B.y") == {"A": ["step_x"], "B": ["y"]}
    assert parse_entrypoints({"A": "x"}) == {"A": ["x"]}
    assert parse_entrypoints(None) == {}
    with pytest.raises(ValueError):
        parse_entrypoints("no_class")

    class W(ToyWorkload):
        entrypoints: ClassVar[dict[str, list[str]]] = {"Attention": ["a"]}

    w = W(WorkloadSpec("x/y", "llm", options={"entrypoints": "Attention.b,Layer.c"}))
    assert workload_entrypoints(w) == {"Attention": ["a", "b"], "Layer": ["c"]}


def test_instrument_wraps_per_instance_and_skips_self_calls():
    class R(nn.Module):
        def forward(self, x):
            return x * 2

        def forward_step(self, x):
            return self(x) + 1  # inner call on the same instance: not reported

    r, other = R(), R()
    previous = r.forward_step
    events: list[tuple[str, str]] = []
    pre = lambda m, meth, a, k: events.append(("pre", meth))  # noqa: E731
    post = lambda m, meth, a, k, out: events.append(("post", meth))  # noqa: E731
    with instrument([r], {R: ["forward_step"]}, pre, post):
        assert "forward_step" in r.__dict__ and "forward_step" not in other.__dict__
        assert r.forward_step(torch.ones(1)).item() == 3
        r(torch.ones(1))
        other.forward_step(torch.ones(1))  # not instrumented
    assert events == [
        ("pre", "forward_step"),
        ("post", "forward_step"),
        ("pre", "forward"),
        ("post", "forward"),
    ]
    assert "forward_step" not in r.__dict__ and not r._forward_hooks and not r._forward_pre_hooks
    assert r.forward_step == previous

    # A pre-existing instance attribute (e.g. a torch.compile'd method) is restored.
    def compiled(x):
        return x

    r.forward_step = compiled
    with instrument([r], {R: ["forward_step"]}, pre, post):
        assert r.forward_step is not compiled
    assert r.forward_step is compiled


# ------------------------------------------------------------------ profiler


def test_profiler_attributes_forward_step(toy):
    inputs = toy.make_inputs()
    with torch.inference_mode(), ModuleTimer(toy.roots(), cuda=False) as timer:
        toy.run(inputs)
    stats = {s.cls: s for s in timer.class_stats()}
    attn = stats["Attention"]
    assert attn.methods["forward_step"]["calls"] == LAYERS * STEPS
    assert attn.methods["forward"]["calls"] == LAYERS
    assert attn.calls == LAYERS * (STEPS + 1)
    assert stats["Model"].methods["forward_step"]["calls"] == STEPS
    assert any(s["signature"].startswith("forward_step: a0[1, 32]") for s in attn.signatures)
    # Linear layers inside forward_step now have a parent call (they used to be orphans).
    for call in timer.calls:
        if call.cls == "Linear":
            assert call.parent >= 0
    # Hooks and wrappers are gone after the context.
    assert not any("forward_step" in m.__dict__ for m in toy.model.modules())

    from dataclasses import asdict

    profile = {
        "module_calls": len(timer.calls),
        "entrypoints": ["Attention.forward_step"],
        "classes": [asdict(s) for s in stats.values()],
        "kernel_view": {
            "gpu_busy_ms": 1.0,
            "gpu_busy_fraction": 0.5,
            "kernel_launches": 1,
            "avg_kernel_us": 1.0,
            "kernels": [],
            "aten_ops": [],
        },
    }
    text = summarize(profile, 2.0)
    assert f"forward_step×{LAYERS * STEPS}" in text and "`Attention.forward_step`" in text


# ------------------------------------------------------------------ capture


def test_capture_records_forward_step_cases(toy, tmp_path):
    inputs = toy.make_inputs()
    info = capture_module(toy, inputs, "Attention", tmp_path / "cap.pt")
    assert info["qualname"] == "model.layers.0.attn"
    assert info["methods"] == {"forward_step": STEPS, "forward": 1}
    assert info["method_instances"] == {"forward_step": LAYERS, "forward": LAYERS}
    assert [(c["method"], c["count"]) for c in info["cases"]] == [
        ("forward_step", STEPS),
        ("forward", 1),
    ]
    # Only the per-layer K/V views are stored, not the whole [2, layers, ...] cache.
    assert info["bytes"] < 3 * 2**20
    cap = load_capture(tmp_path / "cap.pt", device="cpu")
    assert cap["methods"] == {"forward_step": STEPS, "forward": 1}
    assert cap["method_instances"] == {"forward_step": LAYERS, "forward": LAYERS}
    step = next(c for c in cap["cases"] if c["method"] == "forward_step")
    assert step["signature"] == "forward_step: a0[1, 32]:float32"
    key_pre, value_pre = step["args"][2]
    key_post, _ = step["post_args"][2]
    assert key_pre.shape == (1, MAX_LEN, D)
    assert key_pre.untyped_storage().nbytes() == MAX_LEN * D * 4
    assert key_pre.untyped_storage().data_ptr() != value_pre.untyped_storage().data_ptr()
    assert int(step["args"][1]) == PREFIX
    assert torch.count_nonzero(key_pre[0, PREFIX]) == 0 and torch.count_nonzero(key_post[0, PREFIX])
    assert torch.equal(key_pre[0, :PREFIX], key_post[0, :PREFIX])


def test_recorder_keeps_one_case_per_method():
    class M(nn.Module):
        def forward(self, x):
            return x + 1

        def forward_step(self, x):
            return x - 1

    m, peer = M(), M()
    recorder = _Recorder(m, max_cases=2, methods=["forward_step"], peers=[peer])
    try:
        for n in (1, 2, 2, 3):
            m(torch.ones(n))
        for _ in range(4):
            m.forward_step(torch.ones(5))
        m(torch.ones(4))
        peer(torch.ones(7))  # peers are only counted
    finally:
        recorder.remove()
    keys = {(meth, c["count"]) for (meth, _), c in recorder.cases.items()}
    assert keys == {("forward", 2), ("forward_step", 4)}
    assert recorder.calls == {"forward": 5, "forward_step": 4}
    assert recorder.callers == {"forward": {id(m), id(peer)}, "forward_step": {id(m)}}


def test_side_effects_compare_only_the_update():
    pre = (torch.zeros(8192, 4), torch.ones(3))
    ref = (pre[0].clone(), pre[1].clone())
    ref[0][7] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    def check(new: tuple[torch.Tensor, ...]) -> list[dict[str, Any]]:
        return compare_side_effects(pre, ref, new, "args")

    good = check((ref[0] + 1e-6 * (ref[0] != 0), pre[1].clone()))
    assert all(c["ok"] for c in good) and good[0]["changed_elements"] == 4
    assert good[1]["changed_elements"] == 0
    forgot = check((pre[0].clone(), pre[1].clone()))  # 4 of 32768 elements: 0.01 %
    assert not forgot[0]["ok"] and forgot[0]["mismatch_frac"] == 1.0
    stray = ref[0].clone()
    stray[9] = 1.0
    assert not check((stray, pre[1].clone()))[0]["ok"]
    grown = compare_side_effects((torch.zeros(2),), (torch.zeros(3),), (torch.zeros(3),), "a")
    assert grown[0]["ok"] and "changed_elements" not in grown[0]


def test_detach_copies_views_of_large_storages_compactly():
    buffer = torch.randn(2, 8, 65536)  # 4 MiB; each [i, j] row is 256 KiB
    a, b = buffer[0, 1], buffer[1, 2]
    c = buffer[0, 1, :10]  # overlaps a
    d = buffer[1, 4:6, :8]  # strided view, spans one row + 8 elements
    small = torch.randn(4, 8)[1]
    out = _detach((a, b, c, {"d": d}, small))
    a2, b2, c2, d2, small2 = out[0], out[1], out[2], out[3]["d"], out[4]
    for old, new in ((a, a2), (b, b2), (c, c2), (d, d2), (small, small2)):
        assert torch.equal(old, new) and old.stride() == new.stride()
        assert new.untyped_storage().data_ptr() != old.untyped_storage().data_ptr()
    assert a2.untyped_storage().nbytes() == 65536 * 4
    assert c2.untyped_storage().data_ptr() == a2.untyped_storage().data_ptr()  # aliasing kept
    a2[0] = 123.0
    assert c2[0] == 123.0 and buffer[0, 1, 0] != 123.0
    assert d2.untyped_storage().nbytes() == (65536 + 8) * 4
    assert small2.untyped_storage().nbytes() == 4 * 8 * 4  # small storages: plain deep copy


# ------------------------------------------------------------------ evaluation


@pytest.fixture
def toy_capture(toy, tmp_path) -> Path:
    path = tmp_path / "attn.pt"
    capture_module(toy, toy.make_inputs(), "Attention", path)
    return path


def test_evaluator_calls_captured_methods(toy_capture, tmp_path):
    good = evaluate(toy_capture, _candidate(tmp_path, "same"), device="cpu")
    assert good["status"] == "ok" and good["correct"], good
    assert {c["method"] for c in good["cases"]} == {"forward", "forward_step"}
    assert good["timing"].startswith("skipped")

    # Forgets to write V into the cache: output and side effect differ.
    forgets = evaluate(toy_capture, _candidate(tmp_path, "forgets", value="0"), device="cpu")
    assert forgets["status"] == "incorrect"
    step = next(c for c in forgets["cases"] if c["method"] == "forward_step")
    assert not step["ok"] and any("args[2][1]" in f["name"] for f in step["failures"])

    scaled = evaluate(toy_capture, _candidate(tmp_path, "scaled", scale="1.1"), device="cpu")
    assert scaled["status"] == "incorrect"
    forward = next(c for c in scaled["cases"] if c["method"] == "forward")
    assert forward["ok"]  # forward is inherited unchanged

    only_forward = tmp_path / "only_forward.py"
    only_forward.write_text(FORWARD_ONLY)
    missing = evaluate(toy_capture, only_forward, device="cpu")
    assert missing["status"] == "build_error" and "forward_step" in missing["error"]


def test_capture_calls_with_methods(tmp_path):
    torch.manual_seed(0)
    attn = Attention(D).eval()
    cache = torch.zeros(2, 1, MAX_LEN, D)
    calls = [
        ((torch.randn(1, 3, D),), {}, 1),
        ((torch.randn(1, D), torch.tensor([3]), (cache[0], cache[1])), {}, 7, "forward_step"),
    ]
    capture_calls(attn, calls, tmp_path / "c.pt", instances=2)
    cap = load_capture(tmp_path / "c.pt", device="cpu")
    assert cap["methods"] == {"forward_step": 7, "forward": 1}
    assert [c["method"] for c in cap["cases"]] == ["forward", "forward_step"]
    assert evaluate(tmp_path / "c.pt", _candidate(tmp_path, "same"), device="cpu")["correct"]


# ------------------------------------------------------------------ integration + prompts


def test_patcher_requires_captured_methods(toy, tmp_path):
    inputs = toy.make_inputs()
    reference = toy.run(inputs)
    only_forward = tmp_path / "only_forward.py"
    only_forward.write_text(FORWARD_ONLY)
    methods = ["forward_step", "forward"]
    with pytest.raises(PatchError, match="forward_step"):
        apply_kernels(toy.roots(), [KernelPatch("attn", "Attention", only_forward, None, methods)])
    report = apply_kernels(
        toy.roots(), [KernelPatch("attn", "Attention", _candidate(tmp_path, "same"), None, methods)]
    )
    assert report.replaced == {"attn": LAYERS} and not report.errors
    assert all(type(layer.attn).__name__ == "Fast" for layer in toy.model.layers)
    assert toy.compare(reference, toy.run(inputs)).passed


def test_apply_template_checks_methods():
    source = APPLY_TEMPLATE.format(repo_id="org/m", root="/tmp/x")
    compile(source, "apply.py", "exec")
    assert 'entry.get("methods", [])' in source


def test_engineer_prompt_states_entrypoint_contract():
    target = {"id": "attn", "module_class": "MiniCPMAttention", "why": "w", "approach": "a"}
    info = {
        "qualname": "model.base_lm.layers.0.self_attn",
        "methods": {"forward_step": 60, "forward": 1},
        "method_instances": {"forward_step": 36, "forward": 60},
        "cases": [
            {"method": "forward_step", "signature": "forward_step: a0[1, 2048]", "count": 60},
            {"method": "forward", "signature": "a0[1, 75, 2048]", "count": 1},
        ],
    }
    text = prompts.engineer_prompt(target, info, ["triton"], "py", "tc", 4, None)
    assert "# Entrypoints" in text and "candidate.forward_step(*args, **kwargs)" in text
    assert "class Fast(MiniCPMAttention)" in text
    assert "`forward_step` (60 calls per run per instance, 36 instances)" in text
    assert "Other instances" not in text
    plain = {"qualname": "m.norm", "cases": [{"signature": "a0[1, 8]", "count": 3}]}
    assert "# Entrypoints" not in prompts.engineer_prompt(
        target, plain, ["triton"], "p", "t", 4, None
    )
    # The captured instance only runs forward, but other instances call forward_step.
    dit = {
        "qualname": "model.feat_decoder.estimator.decoder.layers.0.self_attn",
        "methods": {"forward": 180},
        "method_instances": {"forward": 60, "forward_step": 36},
        "cases": [{"method": "forward", "signature": "a0[2, 11, 1024]", "count": 180}],
    }
    text = prompts.engineer_prompt(target, dit, ["triton"], "p", "t", 4, None)
    assert "Other instances of the class also call `forward_step`" in text


# ------------------------------------------------------------------ GPU end to end


@pytest.mark.gpu
def test_forward_step_capture_and_evaluate_on_gpu(tmp_path):
    w = ToyWorkload(WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cuda"))
    w.load()
    inputs = w.make_inputs()
    w.run(inputs)
    info = capture_module(w, inputs, "Attention", tmp_path / "cap.pt")
    assert info["methods"] == {"forward_step": STEPS, "forward": 1}
    result = evaluate(tmp_path / "cap.pt", _candidate(tmp_path, "same"))
    assert result["status"] == "ok" and result["correct"], result
    step = next(c for c in result["cases"] if c["method"] == "forward_step")
    assert step["calls_per_run"] == STEPS and step["new_ms"] > 0 and step["ref_ms"] > 0
    assert result["speedup"] > 0
    expected = sum(
        c["calls_per_run"] * (c["ref_ms"] - c["new_ms"]) * LAYERS for c in result["cases"]
    )
    assert result["est_saved_ms_per_run"] == pytest.approx(expected, abs=1e-2)
    bad = evaluate(tmp_path / "cap.pt", _candidate(tmp_path, "forgets", value="0"))
    assert bad["status"] == "incorrect"
    only_forward = tmp_path / "only_forward.py"
    only_forward.write_text(FORWARD_ONLY)
    assert evaluate(tmp_path / "cap.pt", only_forward)["status"] == "build_error"
    # Reference module, cases and captured cache views survive the round trip unchanged.
    cap = load_capture(tmp_path / "cap.pt", device="cuda")
    assert copy.deepcopy(cap["cases"][0]["args"])[0].is_cuda
