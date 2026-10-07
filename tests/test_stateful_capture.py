"""Captures of stateful modules (issue #162): module state outside the call's arguments (a
KV-cache attribute, a step counter) is snapshotted per case, restored before every replayed
call, and a capture the unmodified reference fails is refused.

The toy decoder mirrors VoxCPM's ``MiniCPMModel``: the KV cache is an attribute (an object of
the model's own cache class), ``forward`` (prefill) returns the per-layer K/V that the caller
writes into the cache, and ``forward_step`` reads and writes it at an explicit position. A
correctness variant (another prompt length) runs after the main run, so the module is saved
holding the variant's cache while the decode cases were recorded with the main run's.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from kernel_agent import dryrun, ledger, orchestrator, truth, worker
from kernel_agent.hub import Modality
from kernel_agent.improve import native_digest
from kernel_agent.kernels import recheck
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.kernels.sweep import sweep
from kernel_agent.native import engine
from kernel_agent.native import project as proj
from kernel_agent.profiling import state
from kernel_agent.profiling.capture import (
    UnverifiableCapture,
    capture_calls,
    capture_module,
    load_capture,
    self_check,
)
from kernel_agent.scheduler import NATIVE, Policy, build_arms
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec
from kernel_agent.workspace import RunDir, read_json, write_json

D = 16
LAYERS = 2
MAX_LEN = 4096  # 2 x 2 layers x 4096 x 16 fp32: a 1 MiB cache
CACHE_BYTES = 2 * LAYERS * MAX_LEN * D * 4


class StaticCache:
    """A model's own cache class (like VoxCPM's ``StaticKVCache``): one ``[2, layers, B, T, D]``
    tensor and the length the caller advances."""

    def __init__(self, batch: int, device: torch.device) -> None:
        self.kv = torch.zeros(2, LAYERS, batch, MAX_LEN, D, device=device)
        self.length = 0

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv[0, i], self.kv[1, i]

    def fill(self, caches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.kv.zero_()
        n = caches[0][0].size(1)
        for i, (k, v) in enumerate(caches):
            self.kv[0, i, :, :n] = k
            self.kv[1, i, :, :n] = v
        self.length = n

    def step(self) -> int:
        self.length += 1
        return self.length - 1


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.o = nn.Linear(D, D, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)
        causal = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        return x + self.o(weights @ v), (k, v)

    def forward_step(
        self, x: torch.Tensor, position_id: torch.Tensor, cache: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        keys, values = cache
        keys[:, position_id] = k.unsqueeze(1)
        values[:, position_id] = v.unsqueeze(1)
        valid = torch.arange(keys.size(1), device=x.device) <= position_id
        scores = (keys @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(D)
        weights = scores.masked_fill(~valid, float("-inf")).softmax(-1)
        return x + self.o((weights.unsqueeze(-1) * values).sum(1))


class Decoder(nn.Module):
    """The KV cache is an attribute (``kv_cache``), not an argument."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.kv_cache: StaticCache | None = None

    def setup_cache(self, batch: int) -> None:
        self.kv_cache = StaticCache(batch, next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[Any]]:
        caches = []
        for block in self.blocks:
            x, kv = block(x)
            caches.append(kv)
        return x, caches

    def forward_step(self, x: torch.Tensor, position_id: torch.Tensor) -> torch.Tensor:
        assert self.kv_cache is not None, "KV cache is not set up"
        for i, block in enumerate(self.blocks):
            x = block.forward_step(x, position_id, self.kv_cache.layer(i))
        return x


class DecoderWorkload(Workload):
    """Prefill ``prefix`` positions, write the cache, then ``steps`` decode steps; one
    correctness variant with another prompt length (it runs last)."""

    modality = Modality.LLM
    defaults = {"prefix": 5, "steps": 6}

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = Decoder().to(self.device).eval()
        self.model.setup_cache(1)

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def variants(self) -> list[dict[str, Any]]:
        return [{"prefix": 3, "steps": 2}]

    def make_inputs(self) -> torch.Tensor:
        torch.manual_seed(int(self.options["prefix"]))  # another prompt: other contents
        n = int(self.options["prefix"]) + int(self.options["steps"])
        return torch.randn(1, n, D, device=self.device)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        prefix = int(self.options["prefix"])
        model = self.model
        with torch.inference_mode():
            h, caches = model(inputs[:, :prefix])
            assert model.kv_cache is not None
            model.kv_cache.fill(caches)
            outs = [h[:, -1]]
            for step in range(int(self.options["steps"])):
                pos = torch.tensor([model.kv_cache.step()], device=self.device)
                outs.append(model.forward_step(inputs[:, prefix + step], pos))
            return torch.stack(outs, 1).float().cpu()

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.allclose(reference, candidate, atol=1e-5, rtol=1e-5)))


def _workload(device: str = "cpu") -> DecoderWorkload:
    wl = DecoderWorkload(WorkloadSpec(repo_id="toy/stateful", modality="llm", device=device))
    wl.load()
    wl.run(wl.make_inputs())  # warm-up, as the worker does: the cache holds a previous run
    return wl


#: The decode step of the reference written out (``{read}``: how it gets a layer's K/V
#: cache views; ``{clone}``: copies them first, so the cache is never written).
STEP = """
    def forward_step(self, x, position_id):
        for i, block in enumerate({blocks}):
            keys, values = ({clone}t for t in {cache}.layer(i))
            q = block.q(x)
            keys[:, position_id] = block.k(x).unsqueeze(1)
            values[:, position_id] = block.v(x).unsqueeze(1)
            valid = torch.arange(keys.size(1), device=x.device) <= position_id
            scores = (keys @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(x.size(-1))
            weights = scores.masked_fill(~valid, float("-inf")).softmax(-1)
            x = x + block.o((weights.unsqueeze(-1) * values).sum(1))
        return x
"""
SUBCLASS = """
import copy
import math

import torch


def build(reference):
    class Step(type(reference)):
{step}
    new = copy.copy(reference)
    new.__class__ = Step
    return new
"""
#: The reference's math (the native agent's ``refcheck`` diagnostic of exp 85).
SAME = SUBCLASS.format(
    step=STEP.format(blocks="self.blocks", cache="self.kv_cache", clone="").replace("\n", "\n    ")
)
#: Right outputs from the restored cache, but the cache is never written: the next decode
#: step of the real model would read a stale position.
NO_CACHE_WRITE = SUBCLASS.format(
    step=STEP.format(blocks="self.blocks", cache="self.kv_cache", clone="1 * ").replace(
        "\n", "\n    "
    )
)
#: A wrapper module: the state lives in the reference copy it holds, not on itself.
WRAPPER = """
import math

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)
{step}

def build(reference):
    return Wrapper(reference)
""".format(step=STEP.format(blocks="self.inner.blocks", cache="self.inner.kv_cache", clone=""))


def _candidate(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    return path


@pytest.fixture
def captured(tmp_path) -> tuple[Path, dict[str, Any]]:
    wl = _workload()
    path = tmp_path / "decoder.pt"
    info = capture_module(wl, wl.make_inputs(), "Decoder", path, variants=wl.variants())
    return path, info


# ------------------------------------------------------------------ the bug and the fix


def test_reference_math_passes_a_capture_saved_after_a_variant(captured, tmp_path):
    """On main the module was saved with the variant's cache and the unmodified reference
    failed its own decode cases (VoxCPM2 ``native_residual_lm``, exp 85)."""
    path, info = captured
    signatures = [c["signature"] for c in info["cases"]]
    assert sum("decode step" in s for s in signatures) == 3
    assert any("correctness only: prefix=3, steps=2" in s for s in signatures)
    # the capture ends with the variant's cache: 3 prefill + 2 decode positions
    assert load_capture(path)["module"].kv_cache.length == 5
    result = evaluate(path, _candidate(tmp_path, "same", SAME), device="cpu")
    assert result["status"] == "ok" and result["correct"], result
    assert all(c["ok"] for c in result["cases"])


def test_capture_records_the_state_that_changes_and_little_else(captured):
    path, info = captured
    capture = load_capture(path)
    tracked = capture["state"]
    assert tracked == info["state"]
    # the cache tensor and its length change; weights and the cache's shape do not
    assert tracked["keys"] == ["kv_cache"]
    decode = [c for c in capture["cases"] if c["method"] == "forward_step"]
    assert all("state" in c and "post_state" in c for c in decode)
    first = decode[0]["state"]["kv_cache"]
    assert first["kind"] == "attrs" and set(first["items"]) == {"kv", "length"}
    assert first["items"]["length"] == {"kind": "value", "value": 6}  # advanced before the call
    # a few 8 KiB chunks per case (the positions that differ), not a 1 MiB cache per case
    assert tracked["bytes"] < CACHE_BYTES / 2, tracked
    assert info["self_check"]["cases"] == len(capture["cases"])
    post = decode[0]["post_state"]["kv_cache"]["items"]
    assert set(post) == {"kv"}  # the step wrote one position per layer: one chunk per K/V
    assert post["kv"]["index"].numel() == 2 * LAYERS


def test_state_is_restored_in_place_and_checked(captured, tmp_path):
    path, _ = captured
    capture = load_capture(path)
    module = capture["module"]
    replay = state.Replay(capture, module)
    kv = module.kv_cache.kv
    view = kv[0, 0]  # a view taken at build time stays valid
    case = next(c for c in capture["cases"] if c.get("bucket") == "middle")
    pos = case["args"][1]  # the 4th of 6 decode steps after a 5-token prefill
    assert int(pos) == 5 + 6 // 2
    replay.restore(case, module)
    assert module.kv_cache.kv is kv and module.kv_cache.length == int(pos) + 1
    pre, _ = replay.expected(case)
    assert torch.equal(view, pre["kv_cache"].kv[0, 0])
    with torch.inference_mode():
        out = replay.call(case, module)(*case["args"], **case["kwargs"])
    assert torch.equal(out, case["output"]) and not torch.equal(view, pre["kv_cache"].kv[0, 0])
    assert all(c["ok"] for c in replay.check(case, module))
    kv[0, 0, :, pos] = 0  # as if the call had not written its position
    bad = [c for c in replay.check(case, module) if not c["ok"]]
    assert [c["name"] for c in bad] == ["state.kv_cache.kv"]


def test_a_candidate_must_update_the_state_like_the_reference(captured, tmp_path):
    path, _ = captured
    result = evaluate(path, _candidate(tmp_path, "nowrite", NO_CACHE_WRITE), device="cpu")
    assert result["status"] == "incorrect", result
    failed = result["cases"][result["failed_check"]["case"]]
    assert failed["method"] == "forward_step"
    assert [f["name"] for f in failed["failures"]] == ["state.kv_cache.kv"]
    assert failed["failures"][0]["changed_elements"] > 0
    assert "KV-cache" in result["state_note"]
    # a wrapper keeps the state in the reference copy it was built from: restored there
    assert evaluate(path, _candidate(tmp_path, "wrap", WRAPPER), device="cpu")["correct"]


def test_sweep_and_recheck_restore_the_state(captured, tmp_path, monkeypatch):
    path, _ = captured
    same = _candidate(tmp_path, "same", SAME)
    out = sweep(path, same, [{}], device="cpu")
    assert out["table"][0]["correct"], out["table"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    work = tmp_path / "recheck"
    work.mkdir()
    ref = recheck.reference_main(
        path, work, seed=3, seeds=2, capture_sha256=None, timing=False, rounds=1
    )
    expected = torch.load(work / recheck.EXPECTED, weights_only=False)["entries"]
    new = recheck.candidate_main(path, same, work, capture_sha256=None, timing=False, rounds=1)
    assert ref["status"] == new["status"] == "ok", (ref, new)
    saved = torch.load(work / recheck.CANDIDATE, weights_only=False)["entries"]
    pairs = [(i, s) for s in range(2) for i in range(len(ref["cases"]))]
    assert not any(recheck.compare_entries(expected, saved, pairs, tier=ref["tier"]))


# ------------------------------------------------------------------ the self-check


TICKS = [0]


class Clocked(nn.Module):
    """Its output depends on state outside the module and its arguments (a global)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        TICKS[0] += 1
        return self.lin(x) * TICKS[0]


def test_an_unverifiable_capture_is_refused_with_the_reason(tmp_path):
    torch.manual_seed(0)
    x = torch.randn(2, D)
    path = tmp_path / "clocked.pt"
    calls = [((x,), {}, 3), ((x[:1],), {}, 1)]
    with pytest.raises(UnverifiableCapture) as err:
        capture_calls(Clocked(), calls, path, self_check=True)
    text = str(err.value)
    assert "the unmodified reference fails its own capture on 2 of 2 cases" in text
    assert "case 0, a0[2, 16]:float32: output:" in text
    assert "no module state changes" in text and "capture refused" in text
    assert not path.exists()
    capture_calls(Clocked(), calls, path)  # without the self-check: saved ...
    check = self_check(path)  # ... and the reference fails it
    assert not check["ok"] and [f["case"] for f in check["failures"]] == [0, 1]


def test_a_stateless_capture_is_unchanged(tmp_path):
    torch.manual_seed(0)
    lin = nn.Linear(D, D)
    path = tmp_path / "lin.pt"
    capture_calls(lin, [((torch.randn(3, D),), {}, 1)], path, self_check=True)
    capture = load_capture(path)
    assert "state" not in capture and "state" not in capture["cases"][0]
    assert "post_state" not in capture["cases"][0]
    assert not state.Replay(capture, capture["module"])
    assert self_check(path) == {**self_check(path), "ok": True, "cases": 1, "failures": []}


# ------------------------------------------------------------------ transformers' DynamicCache


class CachedAttention(nn.Module):
    """One-token decode with a ``transformers`` ``DynamicCache`` attribute that grows by
    concatenation, and a step counter."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import DynamicCache

        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.cache = DynamicCache()
        self.steps = 0

    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.steps = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, 1, D]
        q, k, v = self.qkv(x).chunk(3, -1)
        keys, values = self.cache.update(k[:, None], v[:, None], 0)
        self.steps += 1
        weights = (q[:, None] @ keys.transpose(-1, -2) / math.sqrt(D)).softmax(-1)
        return (weights @ values)[:, 0]


class CachedWorkload(Workload):
    modality = Modality.LLM
    defaults = {"steps": 7}

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = CachedAttention().to(self.device).eval()

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(1, int(self.options["steps"]), D, device=self.device)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.reset()
        with torch.inference_mode():
            return torch.cat([self.model(inputs[:, t : t + 1]) for t in range(inputs.size(1))])

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return Comparison(bool(torch.allclose(reference, candidate)))


#: ``CachedAttention.forward`` written out; ``{count}`` advances the step counter or not.
CACHED = """
import copy
import math


def build(reference):
    class Cached(type(reference)):
        def forward(self, x):
            q, k, v = self.qkv(x).chunk(3, -1)
            keys, values = self.cache.update(k[:, None], v[:, None], 0)
            {count}
            weights = (q[:, None] @ keys.transpose(-1, -2) / math.sqrt(x.size(-1))).softmax(-1)
            return (weights @ values)[:, 0]

    new = copy.copy(reference)
    new.__class__ = Cached
    return new
"""


def test_a_growing_dynamic_cache_attribute(tmp_path):
    pytest.importorskip("transformers")
    wl = CachedWorkload(WorkloadSpec(repo_id="toy/cached", modality="llm", device="cpu"))
    wl.load()
    path = tmp_path / "cached.pt"
    info = capture_module(wl, wl.make_inputs(), "CachedAttention", path)
    assert [c.get("bucket") for c in info["cases"]] == ["first", "middle", "last"]
    capture = load_capture(path)
    assert capture["state"]["keys"] == ["cache", "steps"]
    # each case's cache is shorter than the saved one (7 tokens): no layer yet, 3, 6 tokens
    replay = state.Replay(capture, capture["module"])
    pres = [replay.expected(c)[0] for c in capture["cases"]]
    caches = [pre["cache"].layers for pre in pres]
    assert [len(layers) and layers[0].keys.shape[2] for layers in caches] == [0, 3, 6]
    assert [pre["steps"] for pre in pres] == [0, 3, 6]
    assert capture["module"].cache.layers[0].keys.shape[2] == 7
    same = _candidate(tmp_path, "same", CACHED.format(count="self.steps += 1"))
    assert evaluate(path, same, device="cpu")["correct"]
    forgets = _candidate(tmp_path, "nocount", CACHED.format(count="pass"))
    result = evaluate(path, forgets, device="cpu")
    assert result["status"] == "incorrect"
    failures = result["cases"][0]["failures"]
    assert failures == [
        {"name": "state.steps", "ok": False, "error": "0 after the call, expected 1"}
    ]


# ------------------------------------------------------------------ native stage targets


def test_native_stage_target_of_a_stateful_module(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "toy/stateful")
    write_json(run.run_json, {"truth": truth.new_section()})  # sealed: capture in .truth/
    stage = engine.Stage(id="decoder", scope="stage", group="model", module_class="Decoder")
    spec = engine.target_spec(stage)
    assert spec is not None and spec["id"] == "native_decoder" and spec["native"]
    write_json(run.target(spec["id"]) / "spec.json", spec)
    wl = _workload()
    monkeypatch.setattr(worker, "_workload", lambda run: wl)
    info = worker.cmd_capture(run, argparse.Namespace(target=spec["id"], max_cases=4))
    assert info["state"]["keys"] == ["kv_cache"] and info["self_check"]["cases"] == 5
    assert read_json(run.target(spec["id"]) / "spec.json")["capture"]["state"] == info["state"]
    # the agent's inputs-only copy keeps each case's state, not the answer key
    agent_copy = load_capture(run.target(spec["id"]) / "capture_inputs.pt")
    assert all("post_state" not in c for c in agent_copy["cases"])
    assert sum("state" in c for c in agent_copy["cases"]) == info["state"]["cases"]
    same = _candidate(tmp_path, "same", SAME)
    assert evaluate(run.capture_file(spec["id"]), same, device="cpu")["correct"]


def test_refused_stage_capture_reaches_the_native_digest(tmp_path, monkeypatch):
    from test_native_engine import DIFFUSION, make

    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setenv(proj.CACHE_ENV, str(tmp_path / "native-cache"))
    orch, _ = make(tmp_path)
    run = orch.run
    write_json(run.profile_dir / "ceilings.json", DIFFUSION)
    status = engine.status(run, ledger.rows(run), ["exact"])
    assert status.stage is not None
    spec = engine.target_spec(status.stage)
    assert spec is not None
    error = (
        "Traceback (most recent call last):\n  ...\n"
        "kernel_agent.profiling.capture.UnverifiableCapture: the unmodified reference fails "
        "its own capture on 2 of 2 cases (...): capture refused.\n"
    )
    monkeypatch.setattr(orch, "_worker", lambda *a, **k: {"error": error})
    assert orch._capture([spec]) == []  # dropped, with its reason
    failed = read_json(run.target(spec["id"]) / "spec.failed.json")
    assert failed["capture_error"] == error and failed["native"]
    why = engine.capture_refusal(run, spec["id"])
    assert why is not None and why.startswith("the unmodified reference fails its own capture")
    assert engine.capture_refusal(run, "native_other") is None
    arms = build_arms(run, Policy(native=True), [])
    arm = next(a for a in arms if a.id == NATIVE)
    text = native_digest(run, arm, 7, 6, Policy(), status, arms)
    assert f"no teacher-forced target (capture refused: {why}): end to end only" in text


# ------------------------------------------------------------------ GPU

#: The decode step through SDPA: other kernels than the reference's (not a fallback).
SDPA_STEP = """
import copy

import torch
import torch.nn.functional as F


def build(reference):
    class Sdpa(type(reference)):
        def forward_step(self, x, position_id):
            for i, block in enumerate(self.blocks):
                keys, values = self.kv_cache.layer(i)
                keys[:, position_id] = block.k(x).unsqueeze(1)
                values[:, position_id] = block.v(x).unsqueeze(1)
                valid = torch.arange(keys.size(1), device=x.device) <= position_id
                q = block.q(x)[:, None, None]
                out = F.scaled_dot_product_attention(q, keys[:, None], values[:, None], valid)
                x = x + block.o(out[:, 0, 0])
            return x

    new = copy.copy(reference)
    new.__class__ = Sdpa
    return new
"""
#: ``CachedAttention.forward`` through SDPA.
SDPA_CACHED = """
import copy

import torch.nn.functional as F


def build(reference):
    class Sdpa(type(reference)):
        def forward(self, x):
            q, k, v = self.qkv(x).chunk(3, -1)
            keys, values = self.cache.update(k[:, None], v[:, None], 0)
            self.steps += 1
            return F.scaled_dot_product_attention(q[:, None], keys, values)[:, 0]

    new = copy.copy(reference)
    new.__class__ = Sdpa
    return new
"""


@pytest.mark.gpu
def test_stateful_cases_are_timed_from_their_state_on_gpu(tmp_path):
    """Timing, the timed-output check, the hidden-work and peak-memory passes and the
    profiled activity pass restore each case's state outside their measurements: a static
    cache attribute (idempotent per call) and a growing ``DynamicCache`` with a counter (every
    call would grow it further)."""
    wl = _workload("cuda")
    path = tmp_path / "decoder.pt"
    capture_module(wl, wl.make_inputs(), "Decoder", path, variants=wl.variants())
    result = evaluate(path, _candidate(tmp_path, "sdpa", SDPA_STEP))
    assert result["status"] == "ok" and result["correct"], result
    timed = [c for c in result["cases"] if c["calls_per_run"]]
    assert len(timed) == 4 and all(c["new_ms"] > 0 and c["ref_ms"] > 0 for c in timed)
    assert "timed_output" in result["checks"]

    pytest.importorskip("transformers")
    cached = CachedWorkload(WorkloadSpec(repo_id="toy/cached", modality="llm", device="cuda"))
    cached.load()
    path = tmp_path / "cached.pt"
    capture_module(cached, cached.make_inputs(), "CachedAttention", path)
    result = evaluate(path, _candidate(tmp_path, "cached", SDPA_CACHED))
    assert result["status"] == "ok" and result["correct"], result
    assert all(c["new_ms"] > 0 for c in result["cases"])
