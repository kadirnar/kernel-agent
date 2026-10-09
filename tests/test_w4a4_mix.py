"""Per-layer demotion of W4A4 targets (#233): the probe of a capture
(``kernels/mix_probe.py``) and the policy that moves a target's mix (``demotion.py``).

CPU, deterministic (seeded weights and inputs, no clock): the sensitivity probe groups the
layers that read the same activations; the ladder runs the reference math with the top 0, 1,
2, ... groups in 8-bit on every captured case in the capture's tier and finds the fewest that
pass; the worker stores it when it captures a W4A4 target. The policy starts from it, moves
one group to 8 bits per failed gate (the evaluator: precision-like tier failures with none
correct; the end-to-end gate: the integration's verdict on the target's kernel alone), and
with every group at 8 bits proposes the pivot to the 8-bit class once, never stopping the
arm; the engineer prompt, the improve digest and the re-plan show the mix."""

import asyncio
import json
import types
from pathlib import Path

import torch
from torch import nn

from kernel_agent import demotion, improve, ledger, pivot, precisions, worker
from kernel_agent.agent import prompts
from kernel_agent.kernels import mix_probe, quant
from kernel_agent.profiling.capture import capture_calls, load_capture
from kernel_agent.workspace import RunDir, read_json, write_json

W4A4 = "fp4_w4a4"
FP4A = "near-lossless-fp4a"
TOY = Path(__file__).with_name("perceptual_toy.py")


class Chain(nn.Module):
    """gate / up read the same input, then a chain of square layers: W4A4's error compounds
    through it (about 0.13 relative L2 per layer alone) and fails the tier everywhere."""

    def __init__(self, dim: int = 128, depth: int = 6) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, dim, bias=False)
        self.chain = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (self.gate(x) + self.up(x)) * 0.7071
        for layer in self.chain:
            h = layer(h)
        return h


def _chain_capture(tmp_path: Path, dim: int = 128) -> Path:
    gen = torch.Generator().manual_seed(0)
    module = Chain(dim)
    with torch.no_grad():
        for lin in module.modules():
            if isinstance(lin, nn.Linear):  # orthogonal: the signal keeps its scale
                lin.weight.copy_(torch.linalg.qr(torch.randn(dim, dim, generator=gen))[0])
    module = module.to(torch.bfloat16)
    calls = [
        ((torch.randn(4, 64, dim, generator=gen).to(torch.bfloat16),), {}, 100),
        ((torch.randn(2, 33, dim, generator=gen).to(torch.bfloat16),), {}, 0),
    ]
    path = tmp_path / "chain.pt"
    capture_calls(module, calls, path, tier=FP4A, precision=W4A4)
    return path


# ------------------------------------------------------------------ the probe (kernels)


def test_the_probe_groups_layers_and_finds_the_fewest_8bit_groups(tmp_path):
    capture = load_capture(_chain_capture(tmp_path))
    found = mix_probe.probe(capture, eight_bit="fp8_w8a8")
    assert found["tier"] == FP4A and found["eight_bit"] == "fp8_w8a8" and not found.get("note")
    names = [n for g in found["groups"] for n in g["layers"]]
    assert sorted(names) == sorted(["gate", "up", *(f"chain.{i}" for i in range(6))])
    assert len(found["groups"]) == 7  # gate and up read the same input: one group
    assert next(sorted(g["layers"]) for g in found["groups"] if "gate" in g["layers"]) == [
        "gate",
        "up",
    ]
    errors = [g["w4a4_rel_l2"] for g in found["groups"]]
    assert errors == sorted(errors, reverse=True)  # the most sensitive first
    assert all(g["w4a4_rel_l2"] > g["fp8_rel_l2"] for g in found["groups"])
    ladder, k = found["ladder"], found["passes_at"]
    assert [r["demoted"] for r in ladder] == list(range(8))
    assert not ladder[0]["ok"] and ladder[0]["max_rel_l2"] > 0.28  # the tier's bound
    assert "case 0 output" in ladder[0]["failed"]
    assert k is not None and 1 <= k < 7 and ladder[k]["ok"]
    assert all(not r["ok"] for r in ladder[:k]) and all(r["ok"] for r in ladder[k:])
    assert ladder[k]["layers"] == [n for g in found["groups"][:k] for n in g["layers"]]
    rel = [r["max_rel_l2"] for r in ladder]
    assert rel == sorted(rel, reverse=True)  # each group at 8 bits takes error away
    # the reference's own layers are back as they were (the probe patched a loaded copy)
    assert all("forward" not in m.__dict__ for m in capture["module"].modules())
    # the GPU's 8-bit class: INT8 W8A8 where FP8 is absent, and it passes from the same rung
    int8 = mix_probe.probe(capture, eight_bit="int8_w8a8")
    assert int8["ladder"][-1]["ok"] and int8["ladder"][-1]["max_rel_l2"] < rel[-1]


def test_the_8bit_class_follows_the_gpu_and_errors_are_recorded(tmp_path):
    assert mix_probe.eight_bit_class(None) == "fp8_w8a8"  # unknown GPU
    assert mix_probe.eight_bit_class((12, 0)) == mix_probe.eight_bit_class((10, 0)) == "fp8_w8a8"
    assert mix_probe.eight_bit_class((8, 6)) == "int8_w8a8"  # Ampere: no FP8 tensor cores
    assert mix_probe.eight_bit_class((7, 0)) == "exact"  # Volta: no IMMA either
    missing = mix_probe.probe_capture(tmp_path / "missing.pt", capability=(12, 0))
    assert missing["error"].startswith("FileNotFoundError")


def test_sensitivity_rows_name_their_input_group():
    torch.manual_seed(0)
    module = Chain(64, 2)
    xs = [torch.randn(8, 64) for _ in range(3)]
    rows = quant.fp4_w4a4_sensitivity(module, lambda: [module(x) for x in xs])
    group = {r["name"]: r["input_group"] for r in rows}
    assert group == {"gate": "gate", "up": "gate", "chain.0": "chain.0", "chain.1": "chain.1"}


# ------------------------------------------------------------------ the policy

GROUPS = [
    {
        "layers": ["mlp.gate_proj", "mlp.up_proj"],
        "w4a4_rel_l2": 0.088,
        "fp8_rel_l2": 0.019,
        "flop_share": 0.49,
    },
    {"layers": ["mlp.down_proj"], "w4a4_rel_l2": 0.07, "fp8_rel_l2": 0.018, "flop_share": 0.25},
    {
        "layers": ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        "w4a4_rel_l2": 0.05,
        "fp8_rel_l2": 0.015,
        "flop_share": 0.26,
    },
]
PROBE = {
    "version": 1,
    "eight_bit": "fp8_w8a8",
    "tier": FP4A,
    "groups": GROUPS,
    "ladder": [
        {"demoted": 0, "ok": False, "min_cosine": 0.97, "max_rel_l2": 0.21, "failed": "norm"},
        {"demoted": 1, "ok": True, "min_cosine": 0.991, "max_rel_l2": 0.12},
        {"demoted": 2, "ok": True, "min_cosine": 0.995, "max_rel_l2": 0.08},
        {"demoted": 3, "ok": True, "min_cosine": 0.998, "max_rel_l2": 0.05},
    ],
    "passes_at": 1,
}


def _run(tmp_path: Path, probe: dict | None = None, precision: str = W4A4) -> RunDir:
    root = tmp_path / f"run{len(list(tmp_path.glob('run*')))}"  # several runs per test
    root.mkdir()
    run = RunDir(root.resolve())
    capture = {"precision": precision, "tier": FP4A, "w4a4_probe": probe or PROBE}
    spec = {"id": "dit", "module_class": "DiTLayer", "precision": precision, "capture": capture}
    write_json(run.target("dit") / "spec.json", spec)
    return run


def _spec(run: RunDir) -> dict:
    return read_json(run.target("dit") / "spec.json")


def _failure(exp: int, cosine: float = 0.95, stage: str = "correctness", **kw) -> dict:
    return {
        "exp": exp,
        "status": "incorrect",
        "correct": False,
        "stage": stage,
        "tolerance_tier": FP4A,
        "cases": [{"min_cosine": cosine}],
        "error": "norm x0.94",
        "snapshot": f"history/c_{exp:04d}.py",
        **kw,
    }


def _events(run: RunDir) -> list[dict]:
    return [e for e in ledger.events(run) if e.get("kind", e.get("event")) == "w4a4_mix"]


def test_the_probe_sets_the_first_mix_and_it_is_recorded_once(tmp_path):
    run = _run(tmp_path)
    spec = _spec(run)
    assert demotion.applies(spec) and demotion.MIX not in spec
    mix = demotion.mix_of(spec)
    assert mix["demoted"] == 1 and mix["layers"] == ["mlp.gate_proj", "mlp.up_proj"]
    assert "with the top 1 of 3 groups in fp8_w8a8 it passes" in mix["steps"][0]["reason"]
    done = demotion.step(run, "dit", records=[], exp=4)
    assert done["trigger"] == "probe" and "pivot" not in done
    assert _spec(run)[demotion.MIX]["demoted"] == 1 and _spec(run)[demotion.MIX]["exp"] == 0
    assert len(_events(run)) == 1
    assert demotion.step(run, "dit", records=[], exp=5) == {}  # nothing new
    # everything passes with every layer in W4A4: the mix keeps them all
    passes = _run(tmp_path, {**PROBE, "passes_at": 0})
    assert demotion.mix_of(_spec(passes))["demoted"] == 0
    assert "passes near-lossless-fp4a" in demotion.mix_of(_spec(passes))["steps"][0]["reason"]
    # not a W4A4 target, or a probe that failed: no mix
    assert not demotion.applies(_spec(_run(tmp_path, precision="fp8_w8a8")))
    assert not demotion.applies(_spec(_run(tmp_path, {"error": "OutOfMemoryError"})))
    assert demotion.step(_run(tmp_path, {"error": "x"}), "dit", records=[], exp=1) == {}


def test_evaluator_failures_move_the_next_group_to_8bit(tmp_path):
    run = _run(tmp_path)
    demotion.step(run, "dit", records=[], exp=0)  # the probe's mix, from exp 0
    two = [_failure(1), _failure(2)]
    assert demotion.step(run, "dit", records=two, exp=3) == {}  # fewer than MODULE_FAILURES
    noise = [
        _failure(3, cosine=0.3),  # a broken kernel, not precision
        _failure(4, stage="scale_rule"),
        _failure(5, mode="quick"),
        _failure(6, ledger_status=ledger.DUPLICATE),
    ]
    assert demotion.step(run, "dit", records=two + noise, exp=7) == {}
    passed = [*two, _failure(7), {"exp": 8, "correct": True, "status": "discard"}]
    assert demotion.step(run, "dit", records=passed, exp=9) == {}  # the mix works
    three = [*two, _failure(9, stage="perturbed", cases=[{"min_cosine": 0.999}])]
    done = demotion.step(run, "dit", records=three, exp=10)
    assert done["trigger"] == "evaluator" and "3 evaluations since the mix" in done["reason"]
    mix = _spec(run)[demotion.MIX]
    assert mix["demoted"] == 2 and mix["exp"] == 10
    assert mix["layers"] == ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    assert [s["trigger"] for s in mix["steps"]] == ["probe", "evaluator"]
    assert demotion.step(run, "dit", records=three, exp=11) == {}  # those counted already
    event = _events(run)[-1]
    assert event["demoted"] == 2 and event["trigger"] == "evaluator"


def _history(snapshot: str, **kw) -> dict:
    entry = {
        "items": [f"dit=/run/.truth/targets/dit/history/{snapshot}"],
        "status": "ok",
        "passed": False,
        "reason": "perceptual: sample energy x0.910",
        "ab": {"a_items": []},
    }
    return {**entry, **kw}


def test_an_end_to_end_rejection_moves_the_next_group(tmp_path):
    run = _run(tmp_path)
    demotion.step(run, "dit", records=[], exp=5)
    records = [
        {"exp": 6, "correct": True, "snapshot": "history/c_0006.py"},
        {"exp": 3, "correct": True, "snapshot": "history/c_0003.py"},
    ]
    ignored = {
        "history": [
            _history("c_0006.py", status="oom", reason="out of GPU memory, not judged"),
            _history("c_0006.py", items=["dit=/h/c_0006.py", "transform=/t.py"]),
            _history("c_0006.py", ab={"a_items": ["other=/h/x.py"]}),  # a step of the set
            _history("c_0006.py", passed=True),
            {**_history("c_0006.py"), "items": ["other=/h/c_0006.py"]},
        ]
    }
    assert demotion.step(run, "dit", records=records, integration=ignored, exp=7) == {}
    rejected = {"history": [_history("c_0006.py")]}
    done = demotion.step(run, "dit", records=records, integration=rejected, exp=8)
    assert done["trigger"] == "e2e" and "`c_0006.py` alone: perceptual" in done["reason"]
    assert _spec(run)[demotion.MIX]["demoted"] == 2
    # a snapshot evaluated before the mix moved does not count against the new one
    older = {"history": [_history("c_0003.py"), _history("c_0006.py")]}
    assert demotion.step(run, "dit", records=records, integration=older, exp=9) == {}


def test_every_group_at_8bit_then_the_pivot_once_and_the_arm_goes_on(tmp_path):
    run = _run(tmp_path, {**PROBE, "passes_at": 3})
    demotion.step(run, "dit", records=[], exp=0)
    assert _spec(run)[demotion.MIX]["demoted"] == 3  # every group
    failures = [_failure(i) for i in (1, 2, 3)]
    done = demotion.step(run, "dit", records=failures, exp=4)
    mix = _spec(run)[demotion.MIX]
    assert mix["exhausted"] and mix["pivot"] == "proposed" and mix["demoted"] == 3
    proposal = done["pivot"]
    assert proposal["target"] == "dit" and proposal["precision"] == "fp8_w8a8"
    allowed = (*precisions.default("near-lossless"), W4A4)
    check = pivot.check(  # a proposal the pivot accepts (numbers in its why)
        _spec(run), proposal, quality="near-lossless", taken=set(), allowed=allowed
    )
    assert check is None
    only = ("exact", W4A4)  # a run that allows no 8-bit class: refused, with the reason
    refused = pivot.check(_spec(run), proposal, quality="near-lossless", taken=set(), allowed=only)
    assert "not allowed" in refused
    demotion.record_pivot(run, "dit", {"target": "dit__fp8_w8a8"})
    assert _spec(run)[demotion.MIX]["pivot"] == "to dit__fp8_w8a8"
    more = [_failure(i) for i in (5, 6, 7)]
    assert demotion.step(run, "dit", records=failures + more, exp=8) == {}  # proposed once
    lines = "\n".join(demotion.digest_lines(_spec(run)))
    assert "Every group is at 8 bits; the pivot to `fp8_w8a8`: to dit__fp8_w8a8" in lines
    assert "Still to try for W4A4: per-token outer scales" in lines
    assert "Every group is at 8 bits; still to try" in demotion.prompt_text(_spec(run))
    # no 8-bit class (bf16 for the demoted layers): no pivot, the ideas only
    flat = _run(tmp_path, {**PROBE, "passes_at": 3, "eight_bit": "exact"})
    demotion.step(flat, "dit", records=[], exp=0)
    assert "pivot" not in demotion.step(flat, "dit", records=failures, exp=4)
    assert _spec(flat)[demotion.MIX]["exhausted"]


def test_the_prompt_the_digest_and_the_replan_show_the_mix(tmp_path):
    run = _run(tmp_path)
    spec = _spec(run)
    text = demotion.prompt_text(spec)
    assert "keep `mlp.gate_proj`, `mlp.up_proj` (the top 1 of 3 groups" in text
    assert "in FP8 W8A8 (`fp8_w8a8`'s numerics), every other layer W4A4" in text
    assert "| 1 | `mlp.gate_proj`, `mlp.up_proj` | 0.088 | 0.019 | 49% |" in text
    lines = demotion.digest_lines(spec)
    assert lines[1] == "## Precision mix (W4A4)" and "1 of 3 groups at 8 bits" in lines[2]
    assert "moves `mlp.down_proj` to `fp8_w8a8` next" in "\n".join(lines)
    assert demotion.label(spec) == ", gate_proj / up_proj in fp8_w8a8 (1 of 3 groups)"
    other = {**spec, "precision": "fp8_w8a8"}
    assert demotion.prompt_text(other) == "" and demotion.digest_lines(other) == []
    assert demotion.label(other) == ""
    target = {**spec, "why": "w", "approach": "a", "backends": ["cute"], "precision_why": "M"}
    toolchain = "GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, 16 GB)"
    engineer = prompts.engineer_prompt(
        target,
        spec["capture"],
        ["cute"],
        "python",
        toolchain,
        10,
        None,
        precisions=(*precisions.default("near-lossless"), W4A4),
    )
    assert "# Precision: `fp4_w4a4`" in engineer and "Precision mix (kernel-agent's" in engineer
    assert "quantize_activations(x) -> (codes, scales, outer)" in engineer
    one = {**PROBE, "groups": [{**GROUPS[0], "layers": [""]}], "passes_at": 1}
    assert "keep `the target itself`" in demotion.prompt_text(_spec(_run(tmp_path, one)))
    # the improve section of report.md: each arm's precision, with its mix
    slices = [{"n": 1, "arm": "dit", "best_before": 1.0, "best_after": 1.2, "evals": 3}]
    write_json(run.root / improve.STATE, {"slices": slices, "rounds": [{"n": 1}]})
    row = next(x for x in improve.report_lines(run) if x.startswith("| `dit` |"))
    assert "| fp4_w4a4, gate_proj / up_proj in fp8_w8a8 (1 of 3 groups) |" in row


class _Truth:
    def __init__(self, records: list[dict], integration: dict) -> None:
        self._records, self._integration = records, integration

    def records(self, path: Path) -> list[dict]:
        return list(self._records)

    def load_json(self, path: Path) -> dict:
        return dict(self._integration)


def test_the_improve_hook_applies_the_policy_and_makes_the_pivot(tmp_path):
    run = _run(tmp_path, {**PROBE, "passes_at": 3})
    pivots = []

    async def fake_pivot(target_id, proposal, *, source):
        pivots.append((target_id, proposal, source))
        return {"target": f"{target_id}__{proposal['precision']}"}

    orch = types.SimpleNamespace(truth=_Truth([], {}), pivot=fake_pivot)
    me = types.SimpleNamespace(run=run, orch=orch)
    first = asyncio.run(improve.Improver._w4a4_mix(me, "dit"))
    assert first["trigger"] == "probe" and not pivots
    orch.truth = _Truth([_failure(i) for i in (1, 2, 3)], {})
    done = asyncio.run(improve.Improver._w4a4_mix(me, "dit"))
    assert done["pivot_result"] == {"target": "dit__fp8_w8a8"}
    assert pivots == [("dit", done["pivot"], demotion.SOURCE)]
    assert _spec(run)[demotion.MIX]["pivot"] == "to dit__fp8_w8a8"
    assert asyncio.run(improve.Improver._w4a4_mix(me, "dit")) == {} and len(pivots) == 1
    write_json(run.target("plain") / "spec.json", {"id": "plain", "precision": "fp8_w8a8"})
    assert asyncio.run(improve.Improver._w4a4_mix(me, "plain")) == {}


# ------------------------------------------------------------------ the worker's capture


def test_the_worker_captures_a_w4a4_target_with_its_probe(tmp_path, monkeypatch, capsys):
    from kernel_agent import toolchain
    from kernel_agent.workloads.base import WorkloadSpec

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    spec = WorkloadSpec(repo_id="toy/w4a4", modality="tts", device="cpu", harness=str(TOY))
    run = RunDir.create(tmp_path, "toy/w4a4")
    config = {"quality": "near-lossless", "precisions": ["exact", W4A4]}
    write_json(
        run.run_json,
        {"card": {"repo_id": "toy/w4a4"}, "workload": spec.to_dict(), "config": config},
    )
    target = {"module_class": "Linear", "qualname": "model.backbone", "precision": W4A4}
    write_json(run.target("lin") / "spec.json", target)
    worker.main(["capture", "--run-dir", str(run.root), "--target", "lin"])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)]
    info = json.loads(line[-1][len(worker.MARKER) :])
    assert info["tier"] == FP4A and info["precision"] == W4A4
    found = read_json(run.target("lin") / "spec.json")["capture"][demotion.PROBE]
    assert found["tier"] == FP4A and found["eight_bit"] == "fp8_w8a8"  # no GPU: unknown
    assert [g["layers"] for g in found["groups"]] == [[""]]  # the target is one nn.Linear
    assert [r["demoted"] for r in found["ladder"]] == [0, 1] and found["ladder"][1]["ok"]
    assert demotion.applies(read_json(run.target("lin") / "spec.json"))
