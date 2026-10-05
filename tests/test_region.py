"""Region targets that fuse across module boundaries (kernel_agent/region.py).

CPU only, no Claude: the refactor agent is a fake that writes a fixture rewrite
(``toy_decoder.py``); the verification runs in its real subprocess.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import sys
from pathlib import Path

import pytest
import torch
import toy_decoder
from test_truth import force_write, sealed_run, tamper_events
from toy_decoder import (
    FUSED_CANDIDATE,
    GOOD_REWRITE,
    MISNAMED_REWRITE,
    UNCALLED_REWRITE,
    WRONG_ORDER_REWRITE,
    ToyDecoderLayer,
)

from kernel_agent import (
    dryrun,
    gpulock,
    ledger,
    orchestrator,
    program,
    region,
    scheduler,
    toolchain,
    worker,
)
from kernel_agent.agent import prompts
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.runner import AgentResult
from kernel_agent.agent.tools import record_candidate, snapshot
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import capture_calls, load_capture
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workspace import append_jsonl, read_json, write_json

TESTS = Path(__file__).parent
SRC = Path(region.__file__).parents[1]
TOY = TESTS / "toy_decoder.py"
TID = "resid_norm"
REGION = "Region_resid_norm"


@pytest.fixture
def cpu(monkeypatch):
    """CPU only, here and in the verification subprocess (which imports the toy too)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: dryrun.SimToolchain())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", f"{SRC}:{TESTS}")
    monkeypatch.setenv(gpulock.ENV, "1")  # no wait for the machine's GPU lock


def _layer() -> ToyDecoderLayer:
    torch.manual_seed(0)
    layer = ToyDecoderLayer(32, 4).eval()
    with torch.no_grad():
        layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    return layer


def _parent_capture(tmp_path: Path) -> Path:
    """A capture of one toy decoder layer: two calls of different lengths."""
    torch.manual_seed(1)
    calls = [((torch.randn(1, n, 32),), {}, 3) for n in (5, 9)]
    path = tmp_path / "parent.pt"
    capture_calls(_layer(), calls, path, instances=2)
    return path


def _rewrite(tmp_path: Path, source: str, name: str = "rewrite.py") -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


# ------------------------------------------------------------------ plan


def test_plan_validation_and_schema():
    known = {"ToyDecoderLayer", "ToyRMSNorm"}
    module = {"id": "rms", "module_class": "ToyRMSNorm", "kind": "module", "region": None}
    assert region.validate(module, known) is None
    assert module == {"id": "rms", "module_class": "ToyRMSNorm"}  # no region fields
    assert (
        region.validate({"id": "x", "module_class": "Nope"}, known) == "class Nope not in profile"
    )
    target = {"id": TID, "kind": "region", "parent_class": "ToyDecoderLayer", "region": "add+norm"}
    assert region.validate(target, known) is None
    assert target["module_class"] == REGION and region.is_region(target)
    # the parent may come as module_class; a region needs a description and a known parent
    alias = {"id": "r2", "kind": "region", "module_class": "ToyDecoderLayer", "region": "x"}
    assert region.validate(alias, known) is None and alias["parent_class"] == "ToyDecoderLayer"
    assert "description" in region.validate(
        {"id": "r3", "kind": "region", "parent_class": "ToyDecoderLayer"}, known
    )
    assert "not in profile" in region.validate(
        {"id": "r4", "kind": "region", "parent_class": "Nope", "region": "x"}, known
    )
    assert region.validate({"id": "r5", "kind": "fusion"}, known) == "unknown kind 'fusion'"

    item = prompts.PLAN_SCHEMA["properties"]["targets"]["items"]
    assert {"kind", "parent_class", "region"} <= set(item["properties"])
    assert "module_class" not in item["required"]
    text = prompts.planner_prompt(
        {"repo_id": "x/y", "modality": "llm"}, {}, "", ["triton"], 3, "py", "tc"
    )
    assert '`kind: "region"`' in text and "`Region_<id>`" in text


# ------------------------------------------------------------------ loading + verification


def test_rewrite_classes_pickle_by_value(tmp_path):
    module = region.load_rewrite(_rewrite(tmp_path, GOOD_REWRITE), TID)
    assert module.__name__ == region.module_name(TID) == "ka_region_resid_norm"
    new = module.rewrite(_layer())
    assert type(new.region).__name__ == REGION and not hasattr(new, "post_attention_layernorm")
    buf = io.BytesIO()
    torch.save({"module": new.region}, buf)
    del sys.modules[region.module_name(TID)]  # another process: the class is not importable
    loaded = torch.load(io.BytesIO(buf.getvalue()), weights_only=False)["module"]
    assert type(loaded).__name__ == REGION and region.module_name(TID) in sys.modules
    x, res = torch.randn(1, 3, 32), torch.randn(1, 3, 32)
    assert all(torch.equal(a, b) for a, b in zip(loaded(x, res), new.region(x, res), strict=True))
    clone = copy.deepcopy(loaded)  # deep copies (the evaluator's) rebuild it the same way
    assert (
        type(clone) is type(loaded)
        and clone.post_attention_layernorm is not loaded.post_attention_layernorm
    )

    class Sub(type(loaded)):  # a candidate subclassing the region class: pickled by reference
        pass

    sub = copy.copy(loaded)
    sub.__class__ = Sub
    assert type(copy.deepcopy(sub)) is Sub
    with pytest.raises(AttributeError, match="rewrite"):
        region.load_rewrite(_rewrite(tmp_path, "x = 1\n", "empty.py"), TID)


def test_verify_accepts_the_refactor_and_rejects_broken_ones(tmp_path):
    capture = _parent_capture(tmp_path)
    good = region.verify(capture, _rewrite(tmp_path, GOOD_REWRITE), TID, device="cpu")
    assert good["verified"] and good["bitwise"] and good["status"] == "ok"
    assert good["region_instances"] == 1 and good["region_calls"] == 2
    assert [c["region_calls"] for c in good["cases"]] == [1, 1]

    wrong = region.verify(capture, _rewrite(tmp_path, WRONG_ORDER_REWRITE), TID, device="cpu")
    assert not wrong["verified"] and wrong["status"] == "incorrect"
    failure = wrong["cases"][0]["failures"][0]
    assert failure["name"] == "output" and failure["outside"] > 0 and not failure["bitwise"]

    uncalled = region.verify(capture, _rewrite(tmp_path, UNCALLED_REWRITE), TID, device="cpu")
    assert not uncalled["verified"] and uncalled["region_calls"] == 0
    assert all(c["ok"] for c in uncalled["cases"]) and "never called" in uncalled["error"]

    misnamed = region.verify(capture, _rewrite(tmp_path, MISNAMED_REWRITE), TID, device="cpu")
    assert misnamed["status"] == "build_error" and f"`{REGION}` submodule" in misnamed["error"]
    crash = region.verify(capture, _rewrite(tmp_path, "def rewrite(p):\n    1 / 0\n"), TID)
    assert crash["status"] == "build_error" and "ZeroDivisionError" in crash["error"]


def test_strict_compare_tolerates_one_ulp_only():
    ref = torch.linspace(-3, 3, 64)
    up = torch.nextafter(ref, torch.full_like(ref, 10.0))  # one ulp up
    (one,) = region.strict_compare(ref, up, "out")
    assert one["ok"] and not one["bitwise"]
    (far,) = region.strict_compare(ref, ref * (1 + 8 * torch.finfo(ref.dtype).eps), "out")
    assert not far["ok"] and far["outside"] > 0
    (same,) = region.strict_compare(
        torch.tensor([1.0, float("nan")]), torch.tensor([1.0, float("nan")]), "o"
    )
    assert same["ok"] and same["bitwise"]
    (ints,) = region.strict_compare(torch.tensor([1, 2]), torch.tensor([1, 3]), "o")
    assert not ints["ok"] and ints["mismatch_frac"] == 0.5
    (dtype,) = region.strict_compare(ref, ref.half(), "o")
    assert not dtype["ok"] and "float16" in dtype["error"]
    (missing,) = region.strict_compare({"a": ref}, {}, "o")
    assert not missing["ok"] and missing["error"] == "missing"


# ------------------------------------------------------------------ the pipeline on the toy decoder


def _call_worker(capsys):
    def call(run, command, *args):
        worker.main([command, "--run-dir", str(run.root), *args])
        out = capsys.readouterr().out
        line = [x for x in out.splitlines() if x.startswith(worker.MARKER)][-1]
        result = json.loads(line[len(worker.MARKER) :])
        if command == "e2e_ab" and result.get("ab"):  # CPU timings are noise: B is 2x faster
            result["ab"]["b_ms"] = [t / 2 for t in result["ab"]["b_ms"]]
        return result

    return call


def _orchestrator(tmp_path, capsys, rewrite: str | None):
    """A sealed run of the toy decoder after analyze, with a fake refactor agent that
    writes ``rewrite`` (None: no session expected) and an in-process worker."""
    spec = WorkloadSpec(
        repo_id="toy/decoder", modality="llm", device="cpu", dtype="float32", harness=str(TOY)
    )
    cfg = OptimizeConfig(model_ref="toy/decoder", runs_dir=tmp_path, use_library=False)
    run = sealed_run(tmp_path, workload=spec.to_dict(), config=cfg.to_dict())
    orch = orchestrator.Orchestrator(run, cfg)
    orch.worker = _call_worker(capsys)
    baseline = orch._worker("analyze", "--no-profile", "--iters", "1")
    assert baseline["deterministic"], baseline
    # CPU timings are noise: a long recorded baseline lets integration keep what passes
    orch.truth.seal_baseline(1e6)
    sessions = []

    async def fake_agent(name, **kwargs):
        sessions.append((name, kwargs))
        if rewrite is None:
            raise AssertionError(f"unexpected agent session {name}")
        Path(kwargs["writable"][0]).write_text(rewrite)
        return AgentResult(name=name)

    orch.agent_runner = fake_agent
    return orch, sessions


def _target():
    target = {
        "id": TID,
        "kind": "region",
        "parent_class": "ToyDecoderLayer",
        "region": "the residual add after self_attn + post_attention_layernorm",
        "why": "two memory round trips per layer",
        "approach": "one fused add + RMSNorm kernel",
        "backends": ["triton"],
    }
    assert region.validate(target, {"ToyDecoderLayer"}) is None
    return target


def test_region_target_end_to_end(tmp_path, capsys, cpu):
    """Parent capture → refactor (fake agent) → verified rewrite → region capture →
    a fused candidate → integration (rewrite + kernel) → optimized/apply.py."""
    orch, sessions = _orchestrator(tmp_path, capsys, GOOD_REWRITE)
    run, keeper = orch.run, orch.truth
    assert asyncio.run(orch.capture_targets([_target()])) == [TID]

    ((name, kw),) = sessions
    target_dir = run.target(TID)
    assert name == f"refactor-{TID}" and kw["cwd"] == target_dir
    assert kw["tools"] == ["Read", "Glob", "Grep", "Write", "Edit"]
    assert kw["writable"] == [target_dir / "rewrite.py"]
    assert kw["mcp_tools"] == ["mcp__ka__verify_rewrite"]
    assert f"class {REGION}(nn.Module)" in kw["system_append"]
    assert "the residual add after self_attn" in kw["system_append"]

    spec = read_json(target_dir / "spec.json")
    assert spec["module_class"] == REGION and spec["parent_class"] == "ToyDecoderLayer"
    assert spec["rewrite"]["verified"] and spec["rewrite"]["bitwise"]
    assert spec["parent_capture"]["qualname"] == "model.layers.0"
    assert spec["capture"]["qualname"] == "model.layers.0.region"
    assert spec["capture"]["method_instances"] == {"forward": 2}
    parent = region.parent_capture(run, TID)
    assert parent.name == f"{TID}.parent.pt" and keeper.verify(parent)
    assert keeper.verify(run.capture_file(TID)) and keeper.verify(region.verified_rewrite(run, TID))
    assert "class ToyDecoderLayer" in (target_dir / "parent" / "reference_source.py").read_text()
    assert f"class {REGION}" in (target_dir / "reference_source.py").read_text()
    assert load_capture(target_dir / "parent" / "capture_inputs.pt")["inputs_only"]
    assert (target_dir / "parent" / "workload_profile.md").is_file()
    events = [e for e in ledger.events(run) if e["event"] == "rewrite"]
    assert events[-1]["verified"] and events[-1]["target"] == TID

    # the region capture loads without the rewrite on the import path (by value)
    sys.modules.pop(region.module_name(TID), None)
    full = load_capture(run.capture_file(TID), sha256=keeper.expect(run.capture_file(TID)))
    assert type(full["module"]).__name__ == REGION and len(full["cases"]) >= 2
    assert all(len(c["args"]) == 2 for c in full["cases"])  # (attention output, residual)

    cand = target_dir / "candidates" / "fused.py"
    cand.write_text(FUSED_CANDIDATE)
    capture = run.capture_file(TID)
    result = evaluate(capture, cand, device="cpu", capture_sha256=keeper.expect(capture))
    assert result["correct"], result
    snap = snapshot(run, cand, TID)
    record_candidate(run, TID, cand, snap, {**result, "speedup": 1.5}, hypothesis="fused")

    asyncio.run(orch.integrate())
    integration = keeper.load_json(run.root / "integration.json")
    assert len(integration["accepted"]) == 1, integration.get("history")  # shows why on a flake
    (accepted,) = integration["accepted"]
    assert accepted["kind"] == "kernel" and accepted["item"].startswith(f"{TID}=")
    final = integration["final"]
    assert final["passed"] and final["patches"]["rewritten"] == {TID: 2}
    assert final["patches"]["replaced"] == {TID: 2}

    out = run.optimized_dir
    manifest = json.loads((out / "manifest.json").read_text())
    (entry,) = manifest["kernels"]
    assert entry["module_class"] == REGION and "qualname_regex" not in entry
    assert entry["rewrite"] == {"file": f"rewrites/{TID}.py", "parent_class": "ToyDecoderLayer"}
    assert (out / "rewrites" / f"{TID}.py").read_text() == GOOD_REWRITE

    # optimized/apply.py on a fresh model: the rewrite, then the kernel; the same outputs
    sys.path.insert(0, str(out))
    try:
        apply = __import__("apply")
        workload = toy_decoder.ToyDecoderWorkload(
            WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cpu", dtype="float32")
        )
        workload.load()
        inputs = workload.make_inputs()
        with torch.inference_mode():
            before = workload.run(inputs)
        counts = apply.apply_kernels(workload.model)
        assert counts == {f"{TID}:rewrite": 2, TID: 2}
        layer = workload.model.layers[0]
        assert type(layer).__name__ == "RegionDecoderLayer"
        assert type(layer.region).__name__ == "FusedAddNorm"
        with torch.inference_mode():
            after = workload.run(inputs)
        assert torch.equal(before["tokens"], after["tokens"])
        assert torch.equal(before["first_logits"], after["first_logits"])
    finally:
        sys.path.remove(str(out))
        sys.modules.pop("apply", None)

    # a rewrite changed after its verification keeps the target out of the integration
    force_write(region.verified_rewrite(run, TID), b"\n# changed\n", append=True)
    assert orch._kernel_bests() == []
    assert any("rewrite.py" in e["file"] for e in tamper_events(run))


def test_broken_rewrite_is_rejected(tmp_path, capsys, cpu):
    orch, sessions = _orchestrator(tmp_path, capsys, WRONG_ORDER_REWRITE)
    run = orch.run
    assert asyncio.run(orch.capture_targets([_target()])) == []
    assert len(sessions) == 1
    target_dir = run.target(TID)
    assert not (target_dir / "spec.json").exists() and run.target_ids() == []
    failed = read_json(target_dir / "spec.failed.json")
    assert failed["rewrite"]["status"] == "incorrect" and not failed["rewrite"]["verified"]
    assert not region.verified_rewrite(run, TID).exists()
    assert not run.capture_file(TID).exists()  # the region was never captured
    events = [e for e in ledger.events(run) if e["event"] == "rewrite"]
    assert [e["verified"] for e in events] == [False]


def test_a_rewrite_already_written_needs_no_session(tmp_path, capsys, cpu):
    """A resumed run (or a rewrite written by hand) is verified before any agent starts."""
    orch, sessions = _orchestrator(tmp_path, capsys, None)
    orch.run.target(TID).mkdir(parents=True)
    _rewrite(orch.run.target(TID), GOOD_REWRITE)
    assert asyncio.run(orch.capture_targets([_target()])) == [TID] and sessions == []


# ------------------------------------------------------------------ prompts, tool, program


def test_refactor_prompt_tool_and_role(tmp_path, monkeypatch):
    target = _target() | {"qualname_regex": r"layers\.[01]$"}
    info = {
        "qualname": "model.layers.0",
        "method_instances": {"forward": 2, "forward_step": 2},
        "cases": [{"signature": "a0[1, 8, 32]", "count": 3}],
    }
    text = prompts.refactor_prompt(target, info)
    assert f"class {REGION}(nn.Module)" in text and "class RewrittenToyDecoderLayer" in text
    assert "`forward`, `forward_step`" in text and r"matches `layers\.[01]$`" in text
    assert 'verify_rewrite(target_id="resid_norm")' in text and "there is no shell" in text
    engineer = prompts.engineer_prompt(target, info, ["triton"], "py", "tc", 4, None)
    assert "region of `ToyDecoderLayer`" in engineer and "`ka_region_resid_norm`" in engineer
    assert program.role_of("refactor-resid_norm") == "refactor"
    assert "## refactor" in program.TEMPLATE.read_text()

    run = sealed_run(tmp_path)
    keeper = orchestrator.truth.of(run)
    write_json(run.target(TID) / "spec.json", target)
    capture = region.parent_capture(run, TID)
    capture.parent.mkdir(parents=True)
    capture.write_bytes(_parent_capture(tmp_path).read_bytes())
    keeper.seal(capture)
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    monkeypatch.setattr(
        region,
        "run_verify",
        lambda c, r, t, capture_sha256, timeout: region.verify(
            c, r, t, device="cpu", capture_sha256=capture_sha256
        ),
    )
    server = {t.name: t for t in tools_mod.build_server(run, keeper=keeper)}

    def call(**args):
        out = asyncio.run(server["verify_rewrite"].handler(args))
        return json.loads(out["content"][0]["text"])

    assert "write" in call(target_id=TID)["error"]  # no rewrite.py yet
    _rewrite(run.target(TID), GOOD_REWRITE)
    assert call(target_id=TID)["verified"]
    assert "not a captured region target" in call(target_id="other")["error"]
    force_write(capture, b"forged")
    assert call(target_id=TID)["status"] == "tampered"


def test_improve_estimates_a_region_from_its_parent_then_its_evaluations(tmp_path):
    """``Region_<id>`` is in no profile: the improve scheduler weighs a region target by
    its parent's time until an evaluation has timed the region itself."""
    run = sealed_run(tmp_path)
    spec = {"id": TID, "kind": "region", "module_class": REGION, "parent_class": "Layer"}
    spec["capture"] = {"method_instances": {"forward": 30}}
    profiles = [{"shares": {"Layer": (0.5, 30)}, "baseline_ms": 100.0, "applied": {}}]
    assert scheduler._ref_ms(TID, spec, profiles) == 0.0
    assert scheduler._region_ref_ms(run, TID, spec, profiles) == 50.0
    timed = [{"method": "forward", "calls_per_run": 20, "ref_ms": 0.01}, {"calls_per_run": 0}]
    append_jsonl(run.results_file(TID), {"cases": [{"calls_per_run": 20}]})  # CPU: untimed
    append_jsonl(run.results_file(TID), {"cases": timed})
    assert scheduler._region_ref_ms(run, TID, spec, profiles) == pytest.approx(6.0)
    assert scheduler._region_ref_ms(run, "rms", {"module_class": "Layer"}, profiles) == 0.0


# ------------------------------------------------------------------ GPU: transformers' Llama layer


@pytest.mark.gpu
def test_llama_region_on_the_gpu(tmp_path):
    """A small random LlamaDecoderLayer in bf16: the refactor of the residual add +
    post_attention_layernorm is bitwise on CUDA, the wrong order is rejected, and a
    Triton fused add + RMSNorm candidate for the region module is correct and faster."""
    from llama_region import FUSED, REWRITE, WRONG_ORDER
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_attention_heads=9,
        num_key_value_heads=3,
        num_hidden_layers=1,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    layer = LlamaDecoderLayer(config, 0).to("cuda", torch.bfloat16).eval()
    with torch.no_grad():
        layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    rope = LlamaRotaryEmbedding(config).to("cuda")
    calls = []
    for length in (1, 37):
        x = torch.randn(1, length, 576, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(length, device="cuda")[None]
        calls.append(((x,), {"position_embeddings": rope(x, pos), "position_ids": pos}, 10))
    parent = tmp_path / "parent.pt"
    capture_calls(layer, calls, parent)

    good = region.verify(parent, _rewrite(tmp_path, REWRITE), TID)
    assert good["verified"] and good["bitwise"] and good["region_calls"] == 2, good
    wrong = region.verify(parent, _rewrite(tmp_path, WRONG_ORDER, "wrong.py"), TID)
    assert not wrong["verified"] and wrong["status"] == "incorrect"

    new = region.load_rewrite(tmp_path / "rewrite.py", TID).rewrite(layer)
    region_calls = []
    for length in (1, 37):
        attn, residual = torch.randn(2, 1, length, 576, device="cuda", dtype=torch.bfloat16)
        region_calls.append(((attn, residual), {}, 10))
    capture = tmp_path / "region.pt"
    capture_calls(new.region, region_calls, capture)
    result = evaluate(capture, _rewrite(tmp_path, FUSED, "fused.py"))
    assert result["correct"] and result["speedup"] > 1.0, result
