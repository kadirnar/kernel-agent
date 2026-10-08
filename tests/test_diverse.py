"""Data-dependent speedups (#170): the diverse input set (judged and timed per input), the
data-dependent label in the ledger, status and report, decode counters, the natural LLM
prompts, the near-tie tolerance of greedy tokens and the teacher-forced LLM gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diverse_toy import FASTER, SPECULATIVE, TEXTS, WRONG_ON_CODE, Clock, DiverseToy
from test_truth import sealed_run

from kernel_agent import abtest, diversity, ledger, status, toolchain, truth, worker
from kernel_agent.workloads import base, diverse, llm, perceptual, texts
from kernel_agent.workloads.base import WorkloadSpec, compare_tokens, decode_stats, measure
from kernel_agent.workloads.llm import CONTEXTS, LLMWorkload
from kernel_agent.workspace import RunDir

TOY = Path(__file__).with_name("diverse_toy.py")


@pytest.fixture
def clock(monkeypatch) -> Clock:
    """CPU only, and the toy's simulated time in place of ``time`` in ``workloads.base``."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(toolchain, "setup", lambda *a, **k: None)
    fake = Clock()
    monkeypatch.setattr(base, "time", fake)
    return fake


def _toy() -> DiverseToy:
    toy = DiverseToy(WorkloadSpec(repo_id="toy/diverse", modality="llm", device="cpu"))
    toy.load()
    return toy


def _apply(toy: DiverseToy, source: str) -> None:
    scope: dict[str, Any] = {}
    exec(compile(source, "<transform>", "exec"), scope)
    scope["apply"](toy)


def _baseline(toy: DiverseToy) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs, info = diverse.record_baseline(toy)
    return outputs, {"diverse": info}


# ------------------------------------------------------------------ the diverse set


def test_baseline_records_every_input_with_its_output(clock):
    toy = _toy()
    outputs, info = diverse.record_baseline(toy)
    assert info["status"] == "ok" and set(outputs) == set(TEXTS)
    rows = {r["label"]: r for r in info["inputs"]}
    assert all(r["ms"] == 160.0 and r["spread"] == 0.0 for r in rows.values())  # 16 x 10 ms
    assert rows["code"]["key"] == diverse.key_of({"text": TEXTS["code"]})
    assert toy.options["text"] == DiverseToy.defaults["text"]  # options restored
    assert any("3 of 3 inputs recorded" in m for m in diverse.messages({"diverse": info}))

    class Plain(DiverseToy):
        def diverse_inputs(self) -> dict[str, dict[str, Any]]:
            return {}

    _, none = diverse.record_baseline(Plain(toy.spec))
    assert none["status"] == "none"
    skipped = diverse.check(toy, None, {"diverse": none}, chaotic=False)
    assert skipped["passed"] and "declares no diverse input set" in skipped["skipped"]
    old = diverse.check(toy, None, {}, chaotic=False)  # analyze before #170
    assert old["passed"] and "re-run analyze" in old["skipped"]


def test_a_uniform_speedup_is_not_data_dependent(clock):
    toy = _toy()
    outputs, baseline = _baseline(toy)
    _apply(toy, FASTER)
    result = diverse.check(toy, outputs, baseline, chaotic=False)
    assert result["passed"] and result["status"] == "ok"
    assert result["median_speedup"] == result["min_speedup"] == result["max_speedup"] == 2.0
    assert not result[diversity.DATA_DEPENDENT] and not result["steps_vary"]
    assert all(r["quality"]["passed"] for r in result["inputs"])


def test_speculative_decoding_is_data_dependent_and_reports_its_counters(clock):
    toy = _toy()
    outputs, baseline = _baseline(toy)
    _apply(toy, SPECULATIVE)
    result = diverse.check(toy, outputs, baseline, chaotic=False)
    assert result["passed"]  # exact: the same tokens on every input
    speedups = {r["label"]: r["speedup"] for r in result["inputs"]}
    assert speedups["prose"] < 1.1 and speedups["code"] > 3 and speedups["table"] > 3
    assert result[diversity.DATA_DEPENDENT] and result["steps_vary"]
    assert result["min_input"] == "prose" and "decode steps per token change" in result["why"]
    code = next(r for r in result["inputs"] if r["label"] == "code")
    stats = code["decode_stats"]
    assert stats["tokens"] == 16 and stats["acceptance_rate"] == pytest.approx(
        (16 - stats["steps"]) / (4 * stats["steps"]), abs=1e-3
    )
    assert stats["tokens_per_verify"] > 1 and stats["tokens_per_step"] == round(
        16 / stats["steps"], 3
    )
    # the benchmark input's counters land in metric_detail
    timing = measure(toy, toy.make_inputs(), warmup=0, iters=2)
    assert timing["metric_detail"]["decode_stats"]["steps"] == 15  # little repeats
    assert diversity.stats_text(stats).startswith("acceptance ")


def test_a_content_dependent_bug_fails_on_the_diverse_set(clock):
    toy = _toy()
    outputs, baseline = _baseline(toy)
    main_ref = toy.run(toy.make_inputs())
    _apply(toy, WRONG_ON_CODE)
    assert toy.compare(main_ref, toy.run(toy.make_inputs())).passed  # the benchmark misses it
    result = diverse.check(toy, outputs, baseline, chaotic=False)
    assert not result["passed"] and result["reason"].startswith("code: tokens differ")
    assert [r["quality"]["passed"] for r in result["inputs"]] == [True, False, True]
    assert result["median_speedup"] == 2.0  # still timed


def test_inputs_that_changed_or_crash(clock):
    toy = _toy()
    outputs, baseline = _baseline(toy)
    baseline["diverse"]["inputs"][0]["key"] = "0" * 12  # the workload's set changed
    original = toy.run

    def crash(text: str) -> Any:
        if "1,2,3" in text:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return original(text)

    toy.run = crash  # type: ignore[method-assign]
    result = diverse.check(toy, outputs, baseline, chaotic=False)
    rows = {r["label"]: r for r in result["inputs"]}
    assert "changed since analyze" in rows["prose"]["skipped"] and "error" in rows["table"]
    assert not result["passed"] and result["reason"].startswith("table failed: OutOfMemoryError")
    assert result["status"] == "skipped"  # one input timed: no spread
    assert "diverse" in abtest.checks_out_of_memory({"diverse": result})  # no verdict: oom


def test_spread_within_the_noise_is_not_data_dependent():
    def row(label: str, speedup: float, spread: float) -> dict[str, Any]:
        return {"label": label, "speedup": speedup, "spread": spread, "baseline_spread": spread}

    calm = diversity.summarize([row("a", 2.0, 0.01), row("b", 2.15, 0.01)])
    assert calm["spread"] == pytest.approx(0.0723, abs=1e-3) and not calm["data_dependent"]
    noisy = diversity.summarize([row("a", 2.0, 0.05), row("b", 2.5, 0.05)])
    assert noisy["threshold"] == 0.3 and not noisy["data_dependent"]  # 3 x (5 % + 5 %)
    wide = diversity.summarize([row("a", 2.0, 0.01), row("b", 2.5, 0.01)])
    assert wide["data_dependent"] and "spread 22% > 10%" in wide["why"]
    steps = [{**row("a", 2.0, 0.0), "decode_stats": {"steps": 8, "tokens": 16}}]
    steps += [{**row("b", 2.0, 0.0), "decode_stats": {"steps": 9, "tokens": 16}}]
    assert diversity.summarize(steps)["data_dependent"]  # same speed, other step counts
    assert diversity.summarize([row("a", 2.0, 0.0)])["status"] == "skipped"


def test_decode_stats():
    assert decode_stats([{}, {}]) is None
    stats = decode_stats([{"steps": 4, "verifies": 3, "drafted": 30, "accepted": 15, "tokens": 19}])
    assert stats is not None
    assert stats["acceptance_rate"] == 0.5 and stats["tokens_per_verify"] == 6.0
    assert stats["tokens_per_step"] == 4.75
    text = diversity.stats_text(stats)
    assert text == (
        "acceptance 50% (15 of 30 drafted), 6.00 tokens per verification, 0.21 steps per token"
    )


# ------------------------------------------------------------------ worker, ledger, reports


def _call(capsys, run: RunDir, *argv: Any) -> dict[str, Any]:
    worker.main([str(a) for a in (argv[0], "--run-dir", run.root, *argv[1:])])
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith(worker.MARKER)]
    return json.loads(line[-1][len(worker.MARKER) :])


def test_worker_judges_times_and_labels_the_diverse_set(tmp_path, clock, capsys):
    spec = WorkloadSpec(repo_id="toy/diverse", modality="llm", device="cpu", harness=str(TOY))
    run = sealed_run(tmp_path, workload=spec.to_dict())
    keeper = truth.of(run)
    baseline = _call(capsys, run, "analyze", "--no-profile", "--iters", 1)
    assert baseline["diverse"]["status"] == "ok" and len(baseline["diverse"]["inputs"]) == 3
    path = run.baseline_output_diverse()
    assert path == run.root / ".truth/baseline_output_diverse.pt" and path.is_file()
    keeper.seal_baseline(baseline["median_ms"])
    assert f".truth/baseline_output_diverse.pt={keeper.expect(path)}" in keeper.worker_args()

    def e2e(name: str, source: str, *extra: str) -> dict[str, Any]:
        transform = tmp_path / f"{name}.py"
        transform.write_text(source)
        argv = ("e2e", "--transform", transform, "--iters", 1, *extra, *keeper.worker_args())
        return _call(capsys, run, *argv)

    spec_result = e2e("speculative", SPECULATIVE)
    assert spec_result["passed"], spec_result
    varied = spec_result["metrics"]["diverse"]
    assert varied["data_dependent"] and varied["median_speedup"] > 3
    assert spec_result["speedup"] < 1.1  # the benchmark text hardly repeats
    assert spec_result["metric_detail"]["decode_stats"]["steps"] == 15

    row = ledger.record_e2e(
        run, spec_result, backend="transform", snapshot="spec.py", hypothesis="prompt lookup"
    )
    assert row["flags"] == diversity.DATA_DEPENDENT and row["diverse_speedup"] > 3
    (read,) = ledger.rows(run)
    assert read["flags"] == "data_dependent" and read["diverse_speedup"] == row["diverse_speedup"]
    assert "keep (data-dependent)" in status.render(run, width=200)
    lines = "\n".join(diversity.report_lines(spec_result, "Diverse inputs"))
    assert "**data-dependent**" in lines and "| code | 160.0 |" in lines
    assert "diverse-set median" in diversity.headline(spec_result)

    wrong = e2e("wrong", WRONG_ON_CODE)
    assert not wrong["passed"] and "diverse input code: tokens differ" in wrong["reason"]
    assert diversity.flags(wrong) == ""  # a uniform 2x: not data-dependent, just wrong

    skipped = e2e("faster", FASTER, "--no-diverse")
    assert skipped["passed"] and "--no-diverse" in skipped["metrics"]["diverse"]["skipped"]
    assert diversity.of(skipped) is None


# ------------------------------------------------------------------ LLM inputs


class CharTokenizer:
    def __call__(self, text: str, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) for c in text]]))


def _llm(version: int | None = None, **options: Any) -> LLMWorkload:
    data = {"repo_id": "x/y", "modality": "llm", "device": "cpu", "options": options}
    if version is not None:
        data["inputs_version"] = version
    wl = LLMWorkload(WorkloadSpec.from_dict(data) if version is None else WorkloadSpec(**data))
    wl.tokenizer = CharTokenizer()
    return wl


def _ids(wl: LLMWorkload, overrides: dict[str, Any] | None = None) -> list[int]:
    with wl.with_options(overrides):
        return wl.make_inputs()["input_ids"][0].tolist()


def test_llm_prompts_are_long_non_repeating_text():
    new = _llm(2, prompt_len=2000)
    main = _ids(new)
    assert main == [ord(c) for c in CONTEXTS["essay"][:2000]]  # no repeated paragraph
    assert len(CONTEXTS["essay"]) > 2000 and texts.ESSAY in CONTEXTS["essay"]
    held = _ids(new, new.holdout_options(1))
    assert held == [ord(c) for c in CONTEXTS["story"][:2000]] and held != main
    custom = _ids(new, {**new.holdout_options(1), "prompt": None})
    assert custom == held  # -o prompt= does not leak into the held-out input

    # version 1 (runs recorded before #170): one paragraph repeated, unchanged
    old = _llm(None, prompt_len=2000)
    assert old.spec.inputs_version == 1 and old.legacy_inputs
    tiled = _ids(old)
    period = len(llm.PROMPT)
    assert tiled[:period] == tiled[period : 2 * period]
    assert old.diverse_inputs() == {} and old.perceptual_samples() == []
    assert old.holdout_options(1) == {"prompt": llm.HOLDOUT_PROMPT, "prompt_shift": 0}
    assert WorkloadSpec(repo_id="x", modality="llm").inputs_version == base.INPUTS_VERSION


def test_llm_diverse_inputs_keep_the_shapes_and_end_with_the_request():
    wl = _llm(2, prompt_len=300)
    inputs = wl.diverse_inputs()
    assert list(inputs) == list(texts.REQUESTS) and len(inputs) == 12
    seen = set()
    for label, options in inputs.items():
        ids = _ids(wl, options)
        request = [ord(c) for c in "\n\n" + texts.REQUESTS[label]][-300:]
        assert len(ids) == 300 and ids[-len(request) :] == request
        seen.add(tuple(ids))
    assert len(seen) == 12
    samples = wl.perceptual_samples()
    assert [s["sample"] for s in samples[:2]] == ["main", "held-out"] and len(samples) == 14


# ------------------------------------------------------------------ greedy tokens: near-ties


def test_a_divergence_at_a_near_tie_of_the_baseline_passes():
    ref, logits = torch.tensor([[5, 6, 7, 8]]), torch.ones(1, 10)
    new = torch.tensor([[5, 9, 9, 9]])
    margins = torch.tensor([[2.0, 0.25, 1.0, 1.0]])

    def check(margins: torch.Tensor | None, near_tie: float) -> Any:
        return compare_tokens(
            ref, new, logits, logits, min_prefix=4, min_cosine=0.9,
            ref_margins=margins, near_tie=near_tie,
        )  # fmt: skip

    strict = check(None, 0.5)
    assert not strict.passed and strict.reason == "tokens diverge at position 1"
    tied = check(margins, 0.5)
    assert tied.passed and tied.metrics["divergence_margin"] == 0.25 and tied.metrics["tolerated"]
    clear = check(margins * 4, 0.5)
    assert not clear.passed and "margin 1.000 there" in clear.reason


# ------------------------------------------------------------------ LLM gate: teacher forcing


class TinyLM(torch.nn.Module):
    """Next-token logits from the last token only (a bigram table)."""

    def __init__(self, vocab: int = 50, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.table = torch.nn.Parameter(torch.randn(vocab, vocab, generator=gen) * 3)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        logits = self.table[input_ids]
        if keep := kwargs.get("logits_to_keep"):
            logits = logits[:, -keep:]
        return SimpleNamespace(logits=logits)


def _greedy(model: TinyLM, prompt: torch.Tensor, n: int) -> torch.Tensor:
    ids = prompt.tolist()
    for _ in range(n):
        ids.append(int(model.table[ids[-1]].argmax()))
    return torch.tensor(ids[len(prompt) :])


def _score(model: TinyLM, prompts: list[torch.Tensor], reference: list[Any] | None) -> list[Any]:
    items = [
        {"prompt": p, "tokens": _greedy(model, p, 12), "reference": r}
        for p, r in zip(prompts, reference or [None] * len(prompts), strict=True)
    ]
    return perceptual.score_llm(model, items)


def test_teacher_forced_llm_gate_passes_small_changes_and_fails_broken_ones():
    model = TinyLM()
    prompts = [
        torch.randint(0, 50, (6,), generator=torch.Generator().manual_seed(i)) for i in range(4)
    ]
    eager = _score(model, prompts, None)
    assert eager[0]["topk_ids"].shape == (12, 32) and "kl" not in eager[0]

    same = _score(model, prompts, eager)
    cmp = perceptual.compare_llm(eager, same)
    assert cmp.passed and cmp.metrics["kl"] == 0 and cmp.metrics["top1"] == 1.0

    with torch.no_grad():
        model.table.mul_(1 + 2**-8)  # a rounding-sized change
    assert perceptual.compare_llm(eager, _score(model, prompts, eager)).passed

    broken = TinyLM(seed=1)  # another model: every distribution differs
    bad = perceptual.compare_llm(eager, _score(broken, prompts, eager))
    assert not bad.passed and "mean KL" in bad.reason and "top-1 agreement" in bad.reason

    # a decode loop that writes other text: the forward is fine, its own text is unlikely
    model = TinyLM()
    loop = [
        {"prompt": p, "tokens": torch.roll(_greedy(model, p, 12), 1), "reference": r}
        for p, r in zip(prompts, eager, strict=True)
    ]
    shifted = perceptual.compare_llm(eager, perceptual.score_llm(model, loop))
    assert shifted.metrics["kl"] == 0 and not shifted.passed
    assert "less likely than eager's" in shifted.reason
    assert not perceptual.compare_llm(eager, eager).passed  # not teacher forced on eager's


def test_llm_perceptual_quality_runs_the_samples_teacher_forced():
    wl = _llm(2, prompt_len=24, new_tokens=12)
    wl.model = TinyLM(vocab=256)
    samples = wl.perceptual_samples()[:3]
    generated = []
    for options in samples:
        with wl.with_options(options):
            prompt = wl.make_inputs()["input_ids"][0]
        generated.append(
            {"options": options, "output": {"tokens": _greedy(wl.model, prompt, 12)[None]}}
        )
    eager = wl.perceptual_quality(generated)
    for g, ref in zip(generated, eager, strict=True):
        g["reference"] = ref
    again = wl.perceptual_quality(generated)
    cmp = wl.compare_perceptual(eager, again)
    assert cmp.passed and cmp.metrics["samples"] == 3 and cmp.metrics["kl"] == 0
    assert "kl=0.0" in perceptual.summary_text({"passed": True, **cmp.metrics})
    assert LLMWorkload.near_lossless_options["min_prefix"] == 0


def test_report_and_status_show_the_diverse_median(tmp_path):
    from synthetic_run import make_run

    from kernel_agent.report import write_report
    from kernel_agent.workspace import read_json, write_json

    run = make_run(tmp_path)
    path = run.root / "integration.json"
    integration = read_json(path, {})
    rows = [
        {"label": "prose", "speedup": 11.0, "ms": 100.0, "baseline_ms": 1100.0},
        {"label": "code", "speedup": 14.0, "ms": 80.0, "baseline_ms": 1120.0},
        {"label": "poem", "speedup": 21.0, "ms": 53.0, "baseline_ms": 1113.0},
    ]
    varied = {"passed": True, "reason": "", **diversity.summarize(rows)}
    final = integration["final"]
    final["metrics"] = {**(final.get("metrics") or {}), "diverse": varied}
    final["metric_detail"] = {"decode_stats": {"steps": 9, "tokens": 64, "verifies": 6}}
    item = integration["accepted"][0]["item"]
    integration["data_dependent"] = [{"item": item, "accepted": True, "why": varied["why"]}]
    write_json(path, integration)
    text = status.render(run, width=240)
    assert "diverse-set median 14.00x (11.00x .. 21.00x), data-dependent" in text
    pytest.importorskip("matplotlib")
    report = write_report(run).read_text()
    assert f"**{final['speedup']}x** (diverse-set median 14.00x (11.00x .. 21.00x)" in report
    assert "## Diverse inputs (integrated result)" in report and "| poem | 1,113.0 |" in report
    assert "* benchmark input: 0.14 steps per token" in report
    assert f"* data-dependent alone (accepted): `{ledger.item_label(item)}`" in report
    assert "diverse inputs passed (0 judged)" in report
