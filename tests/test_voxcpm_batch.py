"""Batched VoxCPM2 throughput workload (``-o batch_size=N -o metric=throughput``): the
batched loop on a tiny random VoxCPM2 (voxcpm's own modules, CPU) against VoxCPM's own
batch-1 generate, per-request quality checks, per-request stop flags and the throughput
metric; batch 4 against batch 1 on the real VoxCPM2 (GPU)."""

from typing import Any

import pytest
import torch
import voxcpm_tiny
from test_voxcpm import _voxcpm2_cached
from torch import nn

from kernel_agent.workloads import create_workload, perceptual, stopping, validate_metric
from kernel_agent.workloads.base import Comparison, WorkloadSpec, measure, timed_run
from kernel_agent.workloads.voxcpm import SPEAKER_WAV, TEXT, VoxCPMWorkload
from kernel_agent.workloads.voxcpm_batch import VoxCPMBatchWorkload, request_texts

pytestmark = pytest.mark.skipif(not voxcpm_tiny.available(), reason="needs voxcpm")

PATCHES, SAMPLES_PER_PATCH = 12, 128  # the tiny AudioVAE: 128 samples (8 ms) per patch


def _spec(**options: Any) -> WorkloadSpec:
    return WorkloadSpec("openbmb/VoxCPM2", "tts", family="voxcpm", options=options, device="cpu")


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU timing, no GPU
    wl = create_workload(_spec(batch_size=4, patches=PATCHES))
    assert isinstance(wl, VoxCPMBatchWorkload)
    wl.model = voxcpm_tiny.tiny_voxcpm2()
    wl.sampling_rate = wl.model.sample_rate
    return wl


def _stock(wl: VoxCPMBatchWorkload) -> VoxCPMWorkload:
    """VoxCPM's own batch-1 path on the same model and options."""
    stock = VoxCPMWorkload(_spec(**{k: v for k, v in wl.options.items() if k != "metric"}))
    stock.model, stock.sampling_rate = wl.model, wl.sampling_rate
    return stock


def test_routing_options_and_texts():
    wl = create_workload(_spec(batch_size=4))
    assert isinstance(wl, VoxCPMBatchWorkload) and wl.metric == "throughput"
    assert wl.chaotic and wl.supports_teacher_forcing
    wl = create_workload(_spec(metric="throughput"))
    assert isinstance(wl, VoxCPMBatchWorkload) and wl.batch_size == 8
    validate_metric(_spec(metric="throughput", batch_size=16))
    with pytest.raises(ValueError, match="VoxCPMBatchWorkload cannot time metric=latency"):
        create_workload(_spec(batch_size=4, metric="latency"))
    assert type(create_workload(_spec())) is VoxCPMWorkload  # batch 1, latency: unchanged

    texts = request_texts(TEXT, 40)
    assert texts[0] == TEXT and len(set(texts)) == 40
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        request_texts(TEXT, 0)
    assert wl.variants()[1]["batch_size"] == 4  # capture: another batch size too


def test_batched_run_and_the_throughput_metric(batch):
    original_randn = torch.randn
    inputs = batch.make_inputs()
    lengths = {len(batch.model.text_tokenizer(t)) for t in inputs}
    assert len(lengths) == 4  # different prompt lengths: padding and per-request positions
    out, ms, detail = timed_run(batch, inputs)
    assert out["latents"].shape == (PATCHES, 4, 4, 2)  # [patches, batch, feat_dim, patch]
    assert out["audio"].shape == (4, PATCHES * SAMPLES_PER_PATCH)
    assert out["steps"].tolist() == [PATCHES] * 4 and "stop_margins" not in out
    audio_s = 4 * PATCHES * SAMPLES_PER_PATCH / 16000
    assert detail["requests"] == 4 and detail["audio_s"] == pytest.approx(audio_s)
    # the value is the wall time per second of audio: lower is better, speedups are ratios
    assert ms == pytest.approx(detail["run_ms"] / audio_s)
    assert detail["throughput"] == pytest.approx(audio_s * 1000 / detail["run_ms"])
    assert 0 < detail["request_ms"] <= detail["run_ms"]
    assert torch.randn is original_randn and "forward" not in vars(batch.model.feat_decoder)

    timing = measure(batch, inputs, warmup=0, iters=2)
    assert timing["metric"] == "throughput" and timing["metric_detail"]["requests"] == 4
    # a fixed-length run: the audio is fixed by the options, not by what a run reports
    batch.chunk_marks[:] = [(0.0, 1e6)]
    assert batch.output_seconds() == pytest.approx(audio_s)


def test_every_request_is_voxcpms_batch_1_run(batch):
    inputs = batch.make_inputs()
    out = batch.run(inputs)
    stock = _stock(batch)
    for b in (0, 2):  # free running: request b is VoxCPM's own generate(text_b, seed + b)
        with stock.with_options({"seed": b}):
            single = stock.run(inputs[b])
        torch.testing.assert_close(out["latents"][:, b : b + 1], single["latents"])
        torch.testing.assert_close(out["audio"][b], single["audio"], atol=1e-5, rtol=1e-4)
    check = batch.self_check(inputs, out)  # analyze: every request, teacher forced
    assert check["passed"] and check["requests"] == 4, check
    assert check["min_step_cosine"] > 0.9999 and "excused_steps" not in check
    assert "batch-1 generate" in check["check"]


def test_a_reference_voice_prompt_is_batched_too(batch):
    with batch.with_options({"reference_wav": SPEAKER_WAV}):
        inputs = batch.make_inputs()
        out = batch.run(inputs)
        check = batch.self_check(inputs, out)
    assert check["passed"] and check["min_step_cosine"] > 0.9999, check
    assert not torch.equal(out["latents"], batch.run(inputs)["latents"])  # the voice counts


class RowScaled(nn.Module):
    """A sampler that gets one request of the batch wrong."""

    def __init__(self, ref: nn.Module, row: int) -> None:
        super().__init__()
        self.ref, self.row = ref, row

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.ref(*args, **kwargs).clone()
        out[self.row] *= 1.3
        return out


def test_a_batch_with_one_wrong_request_fails(batch):
    inputs = batch.make_inputs()
    ref = batch.run(inputs)
    cmp = batch.compare_teacher_forced(ref, batch.run_teacher_forced(inputs, ref))
    assert cmp.passed and cmp.metrics["requests"] == 4 and cmp.metrics["min_step_cosine"] == 1.0

    batch.model.feat_decoder = RowScaled(batch.model.feat_decoder, row=2)
    cmp = batch.compare_teacher_forced(ref, batch.run_teacher_forced(inputs, ref))
    assert not cmp.passed and cmp.metrics["failed_requests"] == "2", cmp
    assert cmp.reason.startswith("request 2: RMS x1.3")


def test_the_decode_step_is_forward_step_per_request(batch):
    """``lm_step`` (a position per request) against VoxCPM's ``forward_step`` at batch 1:
    right-padded prompts of 5, 9 and 7 positions, prefilled once, then two decode steps."""
    lm, lengths, hidden = batch.model.base_lm, [5, 9, 7], batch.model.config.lm_config.hidden_size
    torch.manual_seed(0)
    prompts = [torch.randn(1, n, hidden) for n in lengths]
    steps = torch.randn(2, 3, hidden)
    with torch.inference_mode():
        want = []
        for b, prompt in enumerate(prompts):  # VoxCPM: batch 1, its own cache
            _, kv = lm(inputs_embeds=prompt, is_causal=True)
            lm.kv_cache.fill_caches(kv)
            want.append(
                [lm.forward_step(x[b : b + 1], torch.tensor([lm.kv_cache.step()])) for x in steps]
            )
        padded = torch.zeros(3, 9, hidden)
        for b, prompt in enumerate(prompts):
            padded[b, : lengths[b]] = prompt[0]
        positions = torch.tensor(lengths)

        def batched(step: Any) -> list[torch.Tensor]:
            cache, _ = batch._caches(3, 9 + 2)
            cache.fill_caches(lm(inputs_embeds=padded, is_causal=True)[1])
            return [step(lm, cache, x, positions + i) for i, x in enumerate(steps)]

        got = batched(batch.lm_step)
        for b in range(3):
            for i in range(2):
                torch.testing.assert_close(got[i][b : b + 1], want[b][i], atol=1e-5, rtol=1e-4)
        # a step written for batch 1 (request 0's position for every request) is wrong
        wrong = batched(
            lambda lm, cache, h, pos: batch.lm_step(lm, cache, h, pos[:1].expand_as(pos))
        )
    assert torch.allclose(wrong[1][0], want[0][1], atol=1e-5)
    assert not torch.allclose(wrong[1][1], want[1][1], atol=1e-3)


def test_one_ill_conditioned_step_per_request_is_excused(batch):
    torch.manual_seed(0)
    ref = {"latents": torch.randn(PATCHES, 3, 4, 2), "audio": torch.randn(3, 4096)}
    new = {"latents": ref["latents"].clone(), "audio": ref["audio"].clone()}
    new["latents"][5, 1] = torch.randn(4, 2)  # request 1, step 5: a flipped LocDiT step
    cmp = batch.compare_teacher_forced(ref, new)
    assert cmp.passed and cmp.metrics["excused_steps"].startswith("request 1 step 5 ("), cmp
    with batch.with_options({"outlier_steps": 0}):
        cmp = batch.compare_teacher_forced(ref, new)
    assert not cmp.passed and cmp.reason.startswith("request 1: mean step cosine"), cmp
    new["latents"][9, 1] = torch.randn(4, 2)  # a second one is not an outlier any more
    cmp = batch.compare_teacher_forced(ref, new)
    assert not cmp.passed and cmp.metrics["failed_requests"] == "1", cmp
    other = {"latents": ref["latents"][:, :2], "audio": ref["audio"][:2]}
    assert "requests" in batch.compare_teacher_forced(ref, other).reason  # another batch


class ClockStop(nn.Module):
    """``[continue, stop]`` logits: request *b* wants to stop from its ``targets[b]``-th
    call on (calls since the test reset ``calls``). ``first_only``: every request gets
    request 0's logits (a stop check written for batch 1)."""

    def __init__(self, targets: list[int]) -> None:
        super().__init__()
        self.targets, self.calls, self.first_only = torch.tensor(targets), 0, False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        stop = 4.0 * (self.calls >= self.targets[: h.shape[0]]).float() - 2.0
        self.calls += 1
        logits = torch.stack([torch.zeros_like(stop), stop], -1)
        return logits[:1].expand_as(logits) if self.first_only else logits


def test_every_request_stops_on_its_own(batch):
    head = batch.model.stop_head = ClockStop([6, 9, 4, 7])
    ref = batch.natural_length_run()
    assert ref["request_steps"] == [7, 10, 5, 8] and ref["steps"] == 10, ref["request_steps"]
    assert [len(m) for m in ref["request_stop_margins"]] == [7, 10, 5, 8]
    assert ref["request_stop_margins"][2][-1] == 2.0 and ref["max_steps"] == 100
    assert stopping.stop_summary(ref)["steps"] == 10

    head.calls = 0
    forced = batch.natural_length_run(ref)  # teacher forced: the same stops
    assert batch.compare_natural_length(ref, forced).passed

    head.calls, head.first_only = 0, True  # every request stops when request 0 does
    cmp = batch.compare_natural_length(ref, batch.natural_length_run(ref))
    assert not cmp.passed and cmp.metrics["failed_requests"] == "1,2,3", cmp
    assert "fires 3 step(s) early" in cmp.reason and "fires 2 step(s) late" in cmp.reason

    head.calls, head.first_only = 0, False
    assert batch.run(batch.make_inputs())["steps"].tolist() == [PATCHES] * 4  # fixed length


class OwnNoise(nn.Module):
    """A sampler that draws its noise without ``torch.randn`` (cannot be per request)."""

    def forward(self, mu, patch_size, cond, n_timesteps, cfg_value):
        return torch.randn_like(cond)


def test_the_noise_must_be_drawn_per_request(batch):
    original = torch.randn
    batch.model.feat_decoder = OwnNoise()
    with pytest.raises(RuntimeError, match=r"drew none with `torch.randn\(\(batch, \.\.\.\)\)`"):
        batch.run(batch.make_inputs())
    assert torch.randn is original


def test_perceptual_samples_run_batched_and_score_request_0(batch, monkeypatch):
    batch.model.stop_head = ClockStop([5, 3, 8, 4])
    scored: list[dict[str, Any]] = []

    def score_tts(items, device):
        scored.extend(items)
        return [{"wer": 0.0} for _ in items]

    monkeypatch.setattr(perceptual, "score_tts", score_tts)
    sample = {**batch.perceptual_samples()[0], "patches": 20}
    sample.pop("reference_wav")  # the tiny model clones a voice in its own test above
    generated, _ = perceptual.generate(batch, [sample])
    assert generated[0]["output"]["audio"].shape[0] == 4  # the whole batch ran
    assert batch.perceptual_quality(generated) == [{"wer": 0.0}]
    assert scored[0]["text"] == sample["text"]
    assert scored[0]["audio"].shape == (6 * SAMPLES_PER_PATCH,)  # request 0 stopped at 6


def test_the_self_check_is_logged_and_summarised():
    from kernel_agent.workloads import quality

    check = {"passed": True, "reason": "", "check": "each of 4 requests vs batch 1"}
    assert "analyze: self-check ok (each of 4 requests vs batch 1)" in quality.probe_messages(
        {"self_check": check}
    )
    failed = {**check, "passed": False, "reason": "request 2: RMS x1.3"}
    assert any(
        m.startswith("WARNING: self-check FAILED") and "request 2" in m
        for m in quality.probe_messages({"self_check": failed})
    )
    summary = quality.summary_section(
        {"sensitivity": {"free_running_passed": True}, "self_check": failed}
    )
    assert "* workload self-check (each of 4 requests vs batch 1): **FAILS** (request 2" in summary


def test_compare_per_request_names_the_requests():
    from kernel_agent.workloads.voxcpm_batch import per_request

    ok, bad = Comparison(True, {"x": 0.9}), Comparison(False, {"x": 0.1}, "broken")
    cmp = per_request([ok, bad, ok, bad, bad, bad], "x")
    assert not cmp.passed and cmp.metrics["worst_request"] == 1
    assert cmp.reason == (
        "request 1: broken; request 3: broken; request 4: broken; and 1 more of 6 requests"
    )
    assert per_request([ok, Comparison(True, {"x": 0.5})], "x").metrics["worst_request"] == 1


# ---------------------------------------------------------------- GPU


@pytest.mark.gpu
@pytest.mark.skipif(not _voxcpm2_cached(), reason="needs voxcpm + openbmb/VoxCPM2 in the HF cache")
def test_voxcpm2_batch_4_is_batch_1():
    wl = create_workload(
        WorkloadSpec("openbmb/VoxCPM2", "tts", family="voxcpm", options={"batch_size": 4})
    )
    wl.load()
    inputs = wl.make_inputs()
    timing = measure(wl, inputs, warmup=1, iters=2)
    out, detail = timing.pop("output"), timing["metric_detail"]
    assert out["latents"].shape == (60, 4, 64, 4) and out["audio"].shape == (4, 60 * 7680)
    assert detail["audio_s"] == pytest.approx(4 * 9.6)
    # eager VoxCPM2 is launch-bound at batch 1 (~1.8 s of audio per second): batch 4 is
    # ~4x that (7.3 measured on the RTX 5070 Ti)
    assert detail["throughput"] > 5.0, detail

    # every request against VoxCPM's own batch-1 generate, teacher forced
    with torch.inference_mode():
        check = wl.self_check(inputs, out)
    assert check["passed"] and check["requests"] == 4, check
    assert check["mean_step_cosine"] > 0.99, check

    # ... and one wrong request fails: a decode step written for batch 1 (one position)
    step = wl.lm_step
    wl.lm_step = lambda lm, cache, h, pos: step(lm, cache, h, pos[:1].expand_as(pos))
    with torch.inference_mode():
        forced = wl.run_teacher_forced(inputs, out)
    cmp = wl.compare_teacher_forced(out, forced)
    assert not cmp.passed and cmp.reason.startswith("request "), cmp
