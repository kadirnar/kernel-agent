"""Correctness coverage (#8): the held-out e2e input and the memoisation probe,
KV-length buckets of decode signatures, correctness-only capture variants."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from test_methods import PREFIX, SAME_STEP, STEPS, ToyWorkload, _candidate
from test_truth import force_write, sealed_run, tamper_events
from test_voxcpm import FakeVoxCPM, _voxcpm2_cached
from torch import nn

from kernel_agent import toolchain, truth, worker
from kernel_agent.hub import Modality
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.profiling.capture import _split_calls, bucket_plan, capture_module, load_capture
from kernel_agent.workloads import create_workload, holdout
from kernel_agent.workloads.base import Comparison, Workload, WorkloadSpec, measure
from kernel_agent.workloads.diffusion import DiffusionWorkload
from kernel_agent.workloads.llm import LLMWorkload
from kernel_agent.workloads.quality import probe_messages, summary_section
from kernel_agent.workloads.stt import STTWorkload
from kernel_agent.workspace import RunDir, write_json

TOY = Path(__file__).with_name("chaotic_toy.py")

#: A transform that memoises ``workload.run`` across runs and computes for real only
#: under the teacher-forcing / probe hooks (so today's checks all pass).
MEMO = """
import time

CACHE = {{}}


def apply(workload):
    run = workload.run

    def memoised(inputs):
        if "forward" in vars(workload.model.sampler):  # teacher forcing: really compute
            return run(inputs)
        key = {key}
        if key not in CACHE:
            time.sleep(0.05)  # the real work of a bigger model
            CACHE[key] = run(inputs)
        return CACHE[key]

    workload.run = memoised
"""
FULL_KEY = '(float(inputs.sum()), workload.options["seed"])'  # input and seed
NO_KEY = "0"  # replays the first output it computed

BENIGN = """import torch


def apply(workload):
    with torch.no_grad():
        workload.model.backbone.weight.mul_(1 + 2**-10)
"""


def _call(capsys, run: RunDir, *argv: Any) -> dict[str, Any]:
    worker.main([str(a) for a in (argv[0], "--run-dir", run.root, *argv[1:])])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)]
    return json.loads(line[-1][len(worker.MARKER) :])


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)


# ------------------------------------------------------------------ held-out input


def test_with_options_restores_and_mutates_in_place():
    wl = create_workload(
        WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm", device="cpu")
    )
    alias = wl.options
    before = dict(wl.options)
    with wl.with_options(wl.holdout_options(1)):
        assert alias["text"] != before["text"] and alias["seed"] == before["seed"] + 1
    assert wl.options is alias and wl.options == before
    with pytest.raises(RuntimeError), wl.with_options({"seed": 7}):
        raise RuntimeError
    assert wl.options == before
    assert wl.variants() == [{"text": wl.variants()[0]["text"], "patches": 8}]


def test_builtin_holdout_inputs_change_content_not_shapes():
    llm = LLMWorkload(WorkloadSpec(repo_id="x/y", modality="llm", device="cpu"))
    llm.options["prompt_len"] = 50
    llm.tokenizer = lambda text, **_: SimpleNamespace(
        input_ids=torch.tensor([[ord(c) for c in text]])
    )
    main = llm.make_inputs()["input_ids"]
    held = {}
    for variant in (1, *range(2, 400, 7), 2 + 10**9):
        with llm.with_options(llm.holdout_options(variant)):
            held[variant] = llm.make_inputs()["input_ids"]
        assert held[variant].shape == main.shape and not torch.equal(held[variant], main)
        # fresh inputs (variants >= 2) never repeat the held-out input
        assert variant == 1 or not torch.equal(held[variant], held[1]), variant
    assert llm.options.get("prompt") is None  # restored
    assert [v["prompt_len"] for v in llm.variants()] == [37, 301]

    stt = STTWorkload(WorkloadSpec(repo_id="x/y", modality="stt", device="cpu"))
    stt.sampling_rate = 16000
    stt.options["audio_seconds"] = 2.0
    stt.processor = lambda batch, **_: {"input_features": torch.tensor(np.stack(batch))}
    audio = stt.make_inputs()["input_features"]
    held = []
    for variant in (1, 2):
        with stt.with_options(stt.holdout_options(variant)):
            held.append(stt.make_inputs()["input_features"])
    assert all(h.shape == audio.shape for h in held)
    assert not torch.equal(held[0], audio) and not torch.equal(held[0], held[1])

    diffusion = DiffusionWorkload(WorkloadSpec(repo_id="x/y", modality="diffusion"))
    assert diffusion.variants() == [{"height": 384, "width": 384, "steps": 2}]
    assert diffusion.holdout_options(2)["seed"] == 2


def test_record_baseline_on_fake_voxcpm():
    spec = WorkloadSpec(
        repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm", options={"patches": 20}
    )
    wl = create_workload(spec)
    wl.model, wl.sampling_rate = FakeVoxCPM(), 16000
    main = wl.run(wl.make_inputs())
    output, info = holdout.record_baseline(wl, main)
    assert info["status"] == "ok" and info["distinct"] and info["probe_distinct"]
    assert info["teacher_forcing"]["passed"], info  # replays the held-out seed exactly
    assert output["latents"].shape == main["latents"].shape  # same patches
    assert wl.model.calls[-1]["max_len"] == 20 and wl.options["seed"] == 0


class ShapeToy(Workload):
    """Deterministic, no teacher forcing; the held-out input has *other* shapes."""

    modality = Modality.LLM
    defaults = {"length": 6, "seed": 0}

    def load(self) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 8))

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def holdout_options(self, variant: int = 1) -> dict[str, Any] | None:
        return {"length": 9, "seed": int(self.options["seed"]) + variant}

    def make_inputs(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(int(self.options["seed"]))
        return torch.randn(1, int(self.options["length"]), 8, generator=gen)

    def run(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"out": self.model(inputs)}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        ref, new = reference["out"], candidate["out"]
        ok = ref.shape == new.shape and bool(torch.allclose(ref, new, atol=1e-5))
        return Comparison(ok, {}, "" if ok else "differs")


def _held_out_verdict(wl: Workload, wrap: Any) -> dict[str, Any]:
    """Baseline (main + held-out), then ``wrap`` the run and check the candidate."""
    inputs = wl.make_inputs()
    with torch.inference_mode():
        main_ref = wl.run(inputs)
    ref, info = holdout.record_baseline(wl, main_ref)
    wrap(wl)
    timing = measure(wl, inputs, warmup=1, iters=3)
    return holdout.check(
        wl,
        ref,
        main_ref,
        main_output=timing["output"],
        main_ms=timing["median_ms"],
        baseline={"holdout": info},
        chaotic=False,
    )


def test_compiling_per_shape_is_not_memoisation():
    """A first run at new shapes may be slow (compile, graph capture); the probe times
    a fresh input at the held-out shapes, which the held-out run warmed up."""

    def compile_per_shape(wl: Workload) -> None:
        run, seen = wl.run, set()

        def compiled(inputs: torch.Tensor) -> Any:
            if tuple(inputs.shape) not in seen:
                seen.add(tuple(inputs.shape))
                time.sleep(0.2)  # compile for a new shape
            return run(inputs)

        wl.run = compiled

    wl = ShapeToy(WorkloadSpec(repo_id="toy/shape", modality="llm", device="cpu"))
    wl.load()
    result = _held_out_verdict(wl, compile_per_shape)
    assert result["passed"], result
    assert result["run_ms"] > 200  # the held-out run compiled its shapes ...
    memo = result["memoisation"]
    assert not memo["flagged"] and memo["fresh_ms"] < 100  # ... the fresh input did not
    assert not result["equals_main_output"] and result["quality"]["passed"]


def test_memoising_run_fails_the_probe():
    def memoise(wl: Workload) -> None:
        run, cache = wl.run, {}

        def memoised(inputs: torch.Tensor) -> Any:
            key = inputs.numpy().tobytes()
            if key not in cache:
                time.sleep(0.05)
                cache[key] = run(inputs)
            return cache[key]

        wl.run = memoised

    wl = ShapeToy(WorkloadSpec(repo_id="toy/shape", modality="llm", device="cpu"))
    wl.load()
    result = _held_out_verdict(wl, memoise)
    assert result["quality"]["passed"]  # computed correctly the first time
    assert not result["passed"] and result["memoisation"]["flagged"]
    assert result["reason"].startswith("memoisation: first runs on two fresh inputs")
    assert result["memoisation"]["confirm_ms"] > 50
    assert result["memoisation"]["fresh_over_repeat"] > holdout.MEMO_RATIO


def test_one_slow_fresh_run_is_noise_not_memoisation():
    """A busy GPU slows one run: the probe confirms on a second fresh input."""

    def hiccup(wl: Workload) -> None:
        run, calls = wl.run, [0]

        def busy(inputs: torch.Tensor) -> Any:
            calls[0] += 1
            if calls[0] == 6:  # 4 timed main runs, the held-out run, then the probe
                time.sleep(0.2)
            return run(inputs)

        wl.run = busy

    wl = ShapeToy(WorkloadSpec(repo_id="toy/shape", modality="llm", device="cpu"))
    wl.load()
    result = _held_out_verdict(wl, hiccup)
    memo = result["memoisation"]
    assert result["passed"] and not memo["flagged"], result
    assert memo["fresh_ms"] > 200 and memo["confirm_ms"] < 100


def test_summary_text():
    ok = {"passed": True, "reason": "", "memoisation": {"fresh_over_repeat": 1.04}}
    assert holdout.summary_text(ok) == "held-out input passed (fresh input 1.04x the repeated runs)"
    bad = {"passed": False, "reason": "memoisation: ..."}
    assert holdout.summary_text(bad) == "held-out input FAILED: memoisation: ..."
    skipped = {"passed": True, "reason": "", "skipped": "none declared"}
    assert holdout.summary_text(skipped) == "held-out input skipped (none declared)"


def test_outputs_equal():
    a = {"x": torch.ones(3), "n": 4, "y": [torch.zeros(2)]}
    assert holdout.outputs_equal(a, {"x": torch.ones(3), "n": 5, "y": [torch.zeros(2)]})
    assert not holdout.outputs_equal(a, {"x": torch.ones(3), "y": [torch.ones(2)]})
    assert not holdout.outputs_equal(a, {"x": torch.ones(3).double(), "y": [torch.zeros(2)]})
    assert not holdout.outputs_equal(a, {"x": torch.ones(3)})
    assert holdout.outputs_equal("text", "text") and not holdout.outputs_equal("a", "b")


def test_worker_stores_verifies_and_checks_the_held_out_input(tmp_path, cpu, capsys):
    """analyze stores the held-out baseline in .truth/ (sealed, verified by e2e); e2e
    passes a benign transform and rejects memoising ones that pass every other check."""
    spec = WorkloadSpec(repo_id="toy/chaotic", modality="tts", device="cpu", harness=str(TOY))
    run = sealed_run(tmp_path, workload=spec.to_dict())
    keeper = truth.of(run)
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    held = baseline["holdout"]
    assert held["status"] == "ok" and held["options"] == {"seed": 1}
    assert held["distinct"] and held["probe_distinct"] and held["teacher_forcing"]["passed"]
    path = run.baseline_output_holdout()
    assert path == run.root / ".truth/baseline_output_holdout.pt" and path.is_file()
    keeper.seal_baseline(baseline["median_ms"])
    assert f".truth/baseline_output_holdout.pt={keeper.expect(path)}" in keeper.worker_args()
    assert any("held-out input" in m for m in probe_messages(baseline))
    assert "held-out input" in summary_section(baseline)

    def e2e(name: str, source: str) -> dict[str, Any]:
        transform = tmp_path / f"{name}.py"
        transform.write_text(source)
        return _call(
            capsys, run, "e2e", "--transform", transform, "--iters", 1, *keeper.worker_args()
        )

    ok = e2e("benign", BENIGN)
    assert ok["passed"], ok
    held = ok["metrics"]["holdout"]
    assert held["passed"] and held["quality"]["metrics"]["teacher_forced"]["passed"]
    assert not held["equals_main_output"] and not held["memoisation"]["flagged"]

    # Memoised per (input, seed): the main input is replayed, fresh inputs are computed.
    memo = e2e("memo_full", MEMO.format(key=FULL_KEY))
    assert memo["status"] == "ok" and not memo["passed"], memo
    assert memo["metrics"]["teacher_forced"]["passed"]  # today's checks all pass ...
    assert memo["metrics"]["holdout"]["quality"]["passed"]
    assert memo["reason"].startswith("held-out input: memoisation: first runs on two fresh")

    # Replays the first output it computed: the held-out output is the main output.
    replay = e2e("memo_none", MEMO.format(key=NO_KEY))
    assert not replay["passed"] and replay["metrics"]["teacher_forced"]["passed"]
    assert replay["metrics"]["holdout"]["equals_main_output"]
    assert "held-out output equals the main output" in replay["reason"]

    force_write(path, path.read_bytes() + b"\0")
    tampered = e2e("benign2", BENIGN)
    assert tampered["status"] == "tampered" and not tampered["passed"]
    path.unlink()
    missing = e2e("benign3", BENIGN)
    assert missing["status"] == "tampered"
    assert ".truth/baseline_output_holdout.pt" in {e["file"] for e in tamper_events(run)}


def test_runs_without_a_held_out_baseline_skip_the_check(tmp_path, cpu, capsys):
    spec = WorkloadSpec(repo_id="toy/chaotic", modality="tts", device="cpu", harness=str(TOY))
    run = RunDir(tmp_path / "run")
    write_json(run.run_json, {"workload": spec.to_dict()})  # old layout: no .truth/
    _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    run.baseline_output_holdout().unlink()  # analyzed before held-out inputs existed
    result = _call(capsys, run, "e2e", "--iters", 1)
    assert result["passed"] and "re-run analyze" in result["metrics"]["holdout"]["skipped"]


# ------------------------------------------------------------------ KV-length buckets


def test_bucket_plan():
    assert bucket_plan(0) == bucket_plan(1) == []
    assert bucket_plan(2) == [("first", 0, 1), ("last", 1, 1)]
    assert bucket_plan(3) == [("first", 0, 1), ("middle", 1, 1), ("last", 2, 1)]
    assert bucket_plan(60) == [("first", 0, 20), ("middle", 30, 20), ("last", 59, 20)]
    assert bucket_plan(64) == [("first", 0, 21), ("middle", 32, 22), ("last", 63, 21)]
    for n in range(2, 50):
        assert sum(c for _, _, c in bucket_plan(n)) == n


def test_bucket_counts_add_up_when_the_run_differs_from_the_survey():
    members = [
        {"signature": "s", "count": 7, "bucket": "first", "decode_step": 0},
        {"signature": "s", "count": 0, "bucket": "middle", "decode_step": 30},
    ]  # the survey saw 60 calls, the recording 40: no "last" case
    _split_calls(members, total=40)
    assert [c["count"] for c in members] == [13 + 14, 13]  # "middle" (call 20) -> step 0
    assert members[1]["signature"] == "s @ decode step 31/40"


SHORT_KV = """
import copy
import math

import torch


def build(reference):
    class ShortKV(type(reference)):
        def forward_step(self, x, position_id, kv_cache):
            q, k, v = self.q(x), self.k(x), self.v(x)
            key_cache, value_cache = kv_cache
            key_cache[:, position_id] = k.unsqueeze(1)
            value_cache[:, position_id] = v.unsqueeze(1)
            n = {length}  # the KV length of the first decode step, hard-coded
            scores = (key_cache[:, :n] @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(x.size(-1))
            weights = scores.softmax(-1)
            return self.o((weights.unsqueeze(-1) * value_cache[:, :n]).sum(1))

    new = copy.copy(reference)
    new.__class__ = ShortKV
    return new
"""


@pytest.fixture
def toy() -> ToyWorkload:
    w = ToyWorkload(WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cpu"))
    w.load()
    return w


def test_a_kernel_for_the_first_kv_length_fails_later_buckets(toy, tmp_path):
    short = tmp_path / "short_kv.py"
    short.write_text(SHORT_KV.format(length=PREFIX + 1))
    single = tmp_path / "single.pt"
    capture_module(toy, toy.make_inputs(), "Attention", single, decode_buckets=False)
    # one decode case per signature (the first step): the bug goes unnoticed
    assert evaluate(single, short, device="cpu")["correct"]

    bucketed = tmp_path / "bucketed.pt"
    capture_module(toy, toy.make_inputs(), "Attention", bucketed)
    result = evaluate(bucketed, short, device="cpu")
    assert result["status"] == "incorrect"
    oks = {c["signature"].rsplit("@ ", 1)[-1]: c["ok"] for c in result["cases"]}
    assert oks == {
        f"decode step 1/{STEPS}": True,
        f"decode step {STEPS // 2 + 1}/{STEPS}": False,
        f"decode step {STEPS}/{STEPS}": False,
        "a0[1, 5, 32]:float32": True,
    }
    assert evaluate(bucketed, _candidate(tmp_path, "same"), device="cpu")["correct"]


def test_buckets_do_not_crowd_out_other_signatures(toy, tmp_path):
    info = capture_module(toy, toy.make_inputs(), "Attention", tmp_path / "c.pt", max_cases=2)
    assert [c["method"] for c in info["cases"]] == ["forward_step"] * 3 + ["forward"]
    prefill = capture_module(
        toy, toy.make_inputs(), "Attention", tmp_path / "p.pt", phase="prefill"
    )
    assert [c["method"] for c in prefill["cases"]] == ["forward"]


# ------------------------------------------------------------------ extra capture settings


class VariantToy(ToyWorkload):
    """The toy decoder with a ``prefix`` option and an extra setting (``variants``)."""

    defaults = {"prefix": PREFIX, "steps": STEPS}

    def variants(self) -> list[dict[str, Any]]:
        return [{"prefix": 3, "steps": 2}, {"prefix": PREFIX}, {"prefix": -1}]

    def make_inputs(self) -> torch.Tensor:
        torch.manual_seed(1)
        n = int(self.options["prefix"]) + int(self.options["steps"])
        return torch.randn(1, n, 32, device=self.device)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        prefix = int(self.options["prefix"])
        if prefix < 0:
            raise ValueError("no such setting")
        with torch.inference_mode():
            h = self.model(inputs[:, :prefix])
            outs = [h[:, -1]]
            for step in range(int(self.options["steps"])):
                pos = torch.tensor([prefix + step], device=self.device)
                outs.append(self.model.forward_step(inputs[:, prefix + step], pos))
            return torch.stack(outs, 1).float().cpu()


ONLY_CAPTURED_LENGTH = """
import copy
import math


def build(reference):
    class Fixed(type(reference)):
        def forward(self, x):
            if x.size(1) == {length}:
                return super().forward(x)
            q, k, v = self.q(x), self.k(x), self.v(x)  # "no mask needed" for other lengths
            weights = (q @ k.transpose(-1, -2) / math.sqrt(x.size(-1))).softmax(-1)
            return self.o(weights @ v), (k, v)

    new = copy.copy(reference)
    new.__class__ = Fixed
    return new
"""


def test_variant_cases_are_correctness_only(tmp_path):
    wl = VariantToy(WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cpu"))
    wl.load()
    path = tmp_path / "cap.pt"
    info = capture_module(wl, wl.make_inputs(), "Attention", path, variants=wl.variants())
    extra = [c for c in info["cases"] if c.get("correctness_only")]
    # only the new prefill length is kept; the decode signature is already covered
    assert [(c["method"], c["count"]) for c in extra] == [("forward", 0)]
    assert extra[0]["signature"] == "a0[1, 3, 32]:float32 [correctness only: prefix=3, steps=2]"
    assert info["variants"][0] == {"options": {"prefix": 3, "steps": 2}, "cases": 1}
    assert info["variants"][1]["cases"] == 0  # the main setting again: nothing new
    assert "ValueError: no such setting" in info["variants"][2]["error"]
    assert info["methods"] == {"forward_step": STEPS, "forward": 1}  # the main run only
    assert wl.options == {"prefix": PREFIX, "steps": STEPS}
    cap = load_capture(path)
    assert cap["cases"][-1]["correctness_only"] and cap["cases"][-1]["count"] == 0

    same = tmp_path / "same.py"
    same.write_text(SAME_STEP.format(value="v.unsqueeze(1)", scale="1"))
    assert evaluate(path, same, device="cpu")["correct"]
    fixed = tmp_path / "fixed.py"
    fixed.write_text(ONLY_CAPTURED_LENGTH.format(length=PREFIX))
    result = evaluate(path, fixed, device="cpu")
    assert result["status"] == "incorrect"
    bad = [c for c in result["cases"] if not c["ok"]]
    assert (
        len(bad) == 1 and bad[0]["calls_per_run"] == 0 and "correctness only" in bad[0]["signature"]
    )


# ------------------------------------------------------------------ GPU


@pytest.mark.gpu
def test_correctness_only_cases_are_not_timed_on_gpu(tmp_path):
    wl = VariantToy(WorkloadSpec(repo_id="toy/decoder", modality="llm", device="cuda"))
    wl.load()
    path = tmp_path / "cap.pt"
    capture_module(wl, wl.make_inputs(), "Attention", path, variants=wl.variants())
    result = evaluate(path, _candidate(tmp_path, "same"))
    assert result["status"] == "ok" and result["correct"], result
    timed = [c for c in result["cases"] if c["calls_per_run"]]
    untimed = [c for c in result["cases"] if not c["calls_per_run"]]
    assert len(timed) == 4 and all(c["new_ms"] > 0 for c in timed)
    assert len(untimed) == 1 and "new_ms" not in untimed[0] and untimed[0]["ok"]


@pytest.mark.gpu
@pytest.mark.skipif(not _voxcpm2_cached(), reason="needs voxcpm + openbmb/VoxCPM2 in the HF cache")
def test_voxcpm2_held_out_input_and_memoisation_probe():
    """VoxCPM2 at 20 patches: the held-out baseline (other text + seed, same patches)
    replays exactly; the unmodified and the compiled model (reference optimisations:
    torch.compile reduce-overhead) pass the held-out check; a memoised run fails it."""
    wl = create_workload(
        WorkloadSpec(
            repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm", options={"patches": 20}
        )
    )
    wl.load()
    inputs = wl.make_inputs()
    with torch.inference_mode():
        main_ref = wl.run(inputs)
    ref, info = holdout.record_baseline(wl, main_ref)
    assert info["distinct"] and info["probe_distinct"], info
    assert info["teacher_forcing"]["metrics"]["min_step_cosine"] == 1.0  # exact replay
    assert ref["latents"].shape == main_ref["latents"].shape and ref["latents"].shape[0] == 20
    baseline = {"holdout": info}

    def verdict(warmup: int = 1) -> dict[str, Any]:
        timing = measure(wl, inputs, warmup=warmup, iters=3)
        return holdout.check(
            wl,
            ref,
            main_ref,
            main_output=timing["output"],
            main_ms=timing["median_ms"],
            baseline=baseline,
            chaotic=True,
        )

    plain = verdict()
    print("plain:", {k: plain["memoisation"][k] for k in ("fresh_ms", "main_median_ms")})
    assert plain["passed"], plain
    assert not plain["memoisation"]["flagged"] and plain["memoisation"]["fresh_over_repeat"] < 1.5

    original = wl.run
    cache: dict[Any, Any] = {}

    def memoised(text: str) -> Any:
        if "forward" in vars(wl.model.feat_decoder):  # teacher forcing: really compute
            return original(text)
        key = (text, wl.options["seed"])
        if key not in cache:
            cache[key] = original(text)
        return cache[key]

    wl.run = memoised
    memo = verdict()
    assert not memo["passed"] and memo["memoisation"]["flagged"], memo
    assert memo["quality"]["passed"]  # every other check passes
    print("memoised:", memo["reason"])
    wl.run = original

    assert wl.reference_optimizations()  # model.optimize(): compiled + CUDA graphs
    compiled = verdict(warmup=2)
    print("compiled:", {k: compiled["memoisation"][k] for k in ("fresh_ms", "main_median_ms")})
    assert compiled["passed"], compiled
    assert not compiled["memoisation"]["flagged"]
