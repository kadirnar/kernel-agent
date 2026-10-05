"""Built-in VoxCPM workload: family routing, teacher-forcing and natural-length
mechanics on a fake model (CPU), threshold calibration and the stop-condition check
on the real VoxCPM2 (GPU)."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from kernel_agent.workloads import create_workload, stopping
from kernel_agent.workloads.base import WorkloadSpec, measure, timed_run
from kernel_agent.workloads.voxcpm import NATURAL_TEXT, VoxCPMWorkload


class FakeDecoder(nn.Module):
    """Stands in for UnifiedCFM: fresh noise every call."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, mu, patch_size, cond, n_timesteps, cfg_value):
        mean = torch.tanh(mu).unsqueeze(-1).expand(-1, -1, patch_size)
        return self.scale * mean + 0.1 * torch.randn(mean.shape)


class FakeStopHead(nn.Module):
    """``[continue, stop]`` logits from a clock feature (the last one): stop at patch 12."""

    def forward(self, h):
        stop = 2.0 * (h[:, -1] - 11.5) + 0.1 * torch.tanh(h[:, 0])
        return torch.stack([torch.zeros_like(stop), stop], -1)


class FakeVoxCPM(nn.Module):
    sample_rate = 16000

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lm = nn.Linear(4, 4)
        self.feat_decoder = FakeDecoder()
        self.stop_head = FakeStopHead()
        self.calls: list[dict] = []
        self.bypass = False
        self.ignore_stop = False

    def _patches(self, min_len, max_len):
        x = torch.zeros(1, 4)
        for i in range(max_len):
            if self.bypass:
                pred = torch.tanh(x).unsqueeze(-1).expand(-1, -1, 2)
            else:
                pred = self.feat_decoder(
                    mu=self.lm(x) * 3, patch_size=2, cond=None, n_timesteps=10, cfg_value=2.0
                )
            yield pred
            # as VoxCPM: the stop head on the state that produced this patch, every patch
            logits = self.stop_head(torch.cat([x, torch.full((1, 1), float(i))], -1))
            if i > min_len and int(logits.argmax(-1)[0]) == 1 and not self.ignore_stop:
                break
            x = pred.mean(-1)

    def generate(self, target_text, min_len, max_len, **kwargs):
        self.calls.append({"text": target_text, "min_len": min_len, "max_len": max_len, **kwargs})
        feats = list(self._patches(min_len, max_len))
        return torch.cat(feats, -1).flatten().repeat(64).unsqueeze(0)  # "wav" [1, T]

    # metric=ttfa: VoxCPM's streaming path, one audio chunk per patch
    stream_all_at_once = False  # an `_inference` that ignores streaming=True
    vae_broken = False  # a decoder that `streaming_decode()` cannot drive
    loop_broken = False  # an `_inference` whose streaming branch crashes

    def decode_chunk(self, pred):
        if self.vae_broken:
            raise RuntimeError("Expected 3-D tensors, but got 4-D for tensor number 1")
        return pred.flatten().repeat(64).unsqueeze(0)  # [1, 512] per patch

    def generate_streaming(self, target_text, min_len, max_len, **kwargs):
        self.calls.append(
            {"text": target_text, "min_len": min_len, "max_len": max_len, **kwargs, "stream": 1}
        )
        patches = self._patches(min_len, max_len)
        if self.stream_all_at_once:
            yield self.decode_chunk(torch.cat(list(patches), -1))
            return
        for i, pred in enumerate(patches):
            if self.loop_broken and i == 3:
                raise IndexError("streaming branch lost")
            yield self.decode_chunk(pred)


@pytest.fixture
def fake():
    spec = WorkloadSpec(
        repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm", options={"patches": 20}
    )
    wl = create_workload(spec)
    assert isinstance(wl, VoxCPMWorkload)
    wl.model = FakeVoxCPM()
    wl.sampling_rate = 16000
    return wl


def test_family_routing_and_flags():
    spec = WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm")
    wl = create_workload(spec)
    assert isinstance(wl, VoxCPMWorkload) and wl.chaotic and wl.supports_teacher_forcing
    assert WorkloadSpec.from_dict({"repo_id": "a/b", "modality": "tts"}).family is None


def test_run_records_fixed_length_trajectory(fake):
    out = fake.run(fake.make_inputs())
    assert out["latents"].shape == (20, 1, 4, 2)
    assert out["sampling_rate"] == 16000 and out["audio"].dim() == 1
    call = fake.model.calls[-1]
    assert call["min_len"] == call["max_len"] == 20 and call["retry_badcase"] is False
    assert call["retry_badcase_ratio_threshold"] >= 20
    assert "forward" not in vars(fake.model.feat_decoder)  # hook removed


def test_teacher_forcing_replays_exactly(fake):
    inputs = fake.make_inputs()
    ref = fake.run(inputs)
    forced = fake.run_teacher_forced(inputs, ref)
    cmp = fake.compare_teacher_forced(ref, forced)
    assert cmp.passed, cmp.reason
    assert cmp.metrics["min_step_cosine"] == 1.0 and cmp.metrics["decoded_spectral_cosine"] > 0.999
    assert "forward" not in vars(fake.model.feat_decoder)


def test_teacher_forcing_hooks_the_current_decoder_instance(fake):
    inputs = fake.make_inputs()
    ref = fake.run(inputs)
    fake.model.feat_decoder = FakeDecoder(scale=1.5)  # e.g. a kernel replaced the sampler
    forced = fake.run_teacher_forced(inputs, ref)
    assert forced["latents"].shape == ref["latents"].shape
    cmp = fake.compare_teacher_forced(ref, forced)
    assert not cmp.passed


def test_teacher_forcing_detects_a_bypassed_decoder(fake):
    inputs = fake.make_inputs()
    ref = fake.run(inputs)
    fake.model.bypass = True  # e.g. the whole step captured in one CUDA graph
    assert fake.run(inputs)["latents"].numel() == 0
    with pytest.raises(RuntimeError, match="bypasses"):
        fake.run_teacher_forced(inputs, ref)
    with pytest.raises(ValueError, match="no recorded latents"):
        fake.run_teacher_forced(inputs, {"audio": ref["audio"]})


def test_teacher_forcing_respects_transforms_that_wrap_run(fake):
    inputs = fake.make_inputs()
    ref = fake.run(inputs)
    seen = []
    original = fake.run

    def wrapped(x):
        seen.append(x)
        return original(x)

    fake.run = wrapped  # what a model-level transform may do
    fake.run_teacher_forced(inputs, ref)
    assert seen == [inputs]


def test_natural_length_run_lets_the_stop_head_decide(fake):
    before = dict(fake.options)
    ref = fake.natural_length_run()
    call = fake.model.calls[-1]
    assert call["text"] == NATURAL_TEXT and call["retry_badcase"] is False
    assert call["min_len"] == 2 and call["max_len"] == 100
    assert call["retry_badcase_ratio_threshold"] >= 100  # generate() never caps max_len
    assert fake.options == before and "min_patches" not in fake.options
    assert ref["steps"] == 13 == ref["latents"].shape[0] and ref["max_steps"] == 100
    margins = ref["stop_margins"]
    assert len(margins) == 13 and margins[-1] > 0 > max(margins[3:-1])
    assert ref["output_length"] == ref["audio"].numel() > 0
    summary = stopping.stop_summary(ref)
    assert summary["stopped"] and summary["stop_step"] == 12

    forced = fake.natural_length_run(ref)  # teacher forced: the same patches, the same stop
    assert "stop_margins" not in forced
    assert fake.compare_natural_length(ref, forced).passed
    assert fake.run(fake.make_inputs())["latents"].shape[0] == 20  # the main run: fixed

    fake.model.ignore_stop = True  # a transform that never consults the stop head
    never = fake.natural_length_run(ref)
    cmp = fake.compare_natural_length(ref, never)
    assert never["steps"] == 100 and not cmp.passed and "never fires" in cmp.reason
    assert fake.run(fake.make_inputs())["latents"].shape[0] == 20  # ... passes the main run


# ---------------------------------------------------------------- metric=ttfa (streaming)


@pytest.fixture
def streaming(fake, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU timing
    fake.options["metric"] = "ttfa"
    return fake


def test_ttfa_run_streams_and_marks_every_chunk(streaming):
    inputs = streaming.make_inputs()
    out, ms, detail = timed_run(streaming, inputs)
    call = streaming.model.calls[-1]
    assert call["stream"] == 1 and call["min_len"] == call["max_len"] == 20
    assert call["retry_badcase"] is False and call["retry_badcase_ratio_threshold"] >= 20
    assert out["latents"].shape == (20, 1, 4, 2) and out["audio"].numel() == 20 * 512
    assert detail["chunks"] == 20 and detail["steady_chunks"] == 8
    assert 0 < ms <= detail["run_ms"]
    assert detail["rtf"] == pytest.approx(detail["chunk_ms"] / 32.0)  # 512 samples at 16 kHz
    assert "forward" not in vars(streaming.model.feat_decoder)  # every hook removed

    # quality: the full streamed output, teacher forced like the non-streaming run
    forced = streaming.run_teacher_forced(inputs, out)
    cmp = streaming.compare_teacher_forced(out, forced)
    assert cmp.passed and cmp.metrics["min_step_cosine"] == 1.0

    timing = measure(streaming, inputs, warmup=0, iters=2)
    assert timing["metric"] == "ttfa" and timing["median_ms"] < timing["metric_detail"]["run_ms"]
    assert timing["metric_detail"]["chunks"] == 20


def test_ttfa_metric_window_stops_after_the_first_chunk(streaming):
    with streaming.metric_window():
        out = streaming.run(streaming.make_inputs())
    assert out["latents"].shape[0] == 1 and out["audio"].numel() == 512
    assert streaming.run(streaming.make_inputs())["latents"].shape[0] == 20  # the full run again


def test_ttfa_natural_length_run_streams(streaming):
    ref = streaming.natural_length_run()
    assert ref["steps"] == 13 and streaming.model.calls[-1]["stream"] == 1
    assert streaming.compare_natural_length(ref, streaming.natural_length_run(ref)).passed


def test_ttfa_rejects_a_broken_streaming_path(streaming):
    inputs = streaming.make_inputs()
    model = streaming.model
    model.stream_all_at_once = True  # an `_inference` that ignores streaming=True
    with pytest.raises(RuntimeError, match=r"1 audio chunk\(s\) for 20 generated patches"):
        streaming.run(inputs)
    model.stream_all_at_once, model.vae_broken = False, True  # e.g. a channels-last decoder
    with pytest.raises(RuntimeError, match=r"AudioVAE decode failed \(RuntimeError: .*4-D"):
        streaming.run(inputs)
    model.vae_broken, model.loop_broken = False, True
    with pytest.raises(RuntimeError, match=r"`_inference\(streaming=True\)`\) failed \(IndexError"):
        streaming.run(inputs)
    assert "forward" not in vars(model.feat_decoder)
    model.loop_broken = False
    assert streaming.run(inputs)["latents"].shape[0] == 20

    def oom(**kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")
        yield

    model.generate_streaming = oom  # the GPU, not the streaming path: not reworded
    with pytest.raises(torch.OutOfMemoryError, match=r"^CUDA out of memory$"):
        streaming.run(inputs)


# ---------------------------------------------------------------- GPU calibration

BROKEN_RMSNORM = """
import torch
from torch import nn


class RMS(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.weight = ref.weight
        self.eps = {eps}

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        y = (x * torch.rsqrt(var + self.eps)).to(x.dtype)
        return y{weight}


def build(ref):
    return RMS(ref)
"""

BROKEN_ATTENTION = """
import torch
import torch.nn.functional as F
from torch import nn

_SDPA = F.scaled_dot_product_attention


def _drop_kv_head(q, k, v, *args, **kwargs):
    # every query group attends to KV head 0: KV head 1 is dropped
    return _SDPA(q, k[:, :1].expand_as(k), v[:, :1].expand_as(v), *args, **kwargs)


class Broken(nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.ref = ref

    def _call(self, fn, *args, **kwargs):
        torch.nn.functional.scaled_dot_product_attention = _drop_kv_head
        try:
            return fn(*args, **kwargs)
        finally:
            torch.nn.functional.scaled_dot_product_attention = _SDPA

    def forward(self, *args, **kwargs):
        return self._call(self.ref, *args, **kwargs)

    def forward_step(self, *args, **kwargs):
        return self._call(self.ref.forward_step, *args, **kwargs)


def build(ref):
    return Broken(ref)
"""


def _voxcpm2_cached() -> bool:
    if importlib.util.find_spec("voxcpm") is None:
        return False
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache("openbmb/VoxCPM2", "config.json"), str)


@pytest.mark.gpu
@pytest.mark.skipif(not _voxcpm2_cached(), reason="needs voxcpm + openbmb/VoxCPM2 in the HF cache")
def test_voxcpm2_teacher_forcing_calibration(tmp_path):
    from kernel_agent.agent.prompts import EXAMPLES_DIR
    from kernel_agent.integrate.patcher import KernelPatch, _set_child, apply_kernels
    from kernel_agent.workloads.quality import assess, probe

    wl = create_workload(WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm"))
    wl.load()
    inputs = wl.make_inputs()
    with torch.inference_mode():
        ref = wl.run(inputs)
    assert ref["latents"].shape[0] == 60

    probes = probe(wl, inputs, ref)
    assert probes["chaotic"] and probes["sensitivity"]["free_running_passed"] is False
    assert probes["teacher_forcing"]["metrics"]["min_step_cosine"] == 1.0  # exact replay

    def verdict(cls_name, candidate):
        originals = [(n, m) for n, m in wl.model.named_modules() if type(m).__name__ == cls_name]
        report = apply_kernels(wl.roots(), [KernelPatch("t", cls_name, candidate)])
        assert report.replaced["t"] > 0 and not report.errors
        try:
            with torch.inference_mode():
                out = wl.run(inputs)
            return assess(wl, inputs, ref, out, chaotic=True)
        finally:
            for name, module in originals:
                _set_child(wl.model, name, module)

    good = verdict("MiniCPMRMSNorm", EXAMPLES_DIR / "triton_rmsnorm.py")
    assert good["passed"], good
    assert good["metrics"]["free_running"]["passed"] is False  # diverges, informational

    eps = tmp_path / "rms_eps.py"
    eps.write_text(BROKEN_RMSNORM.format(eps="1e-2", weight=" * self.weight"))
    no_weight = tmp_path / "rms_no_weight.py"
    no_weight.write_text(BROKEN_RMSNORM.format(eps="ref.variance_epsilon", weight=""))
    drop_kv = tmp_path / "attn_drop_kv.py"
    drop_kv.write_text(BROKEN_ATTENTION)
    for cls_name, candidate in (
        ("MiniCPMRMSNorm", eps),
        ("MiniCPMRMSNorm", no_weight),
        ("MiniCPMAttention", drop_kv),
    ):
        bad = verdict(cls_name, candidate)
        assert not bad["passed"] and bad["reason"].startswith("teacher-forced"), bad


#: Verbatim copies of transforms by the systems agent of a live VoxCPM2 run
#: (runs/openbmb--VoxCPM2/20261005-042829/transforms/): two decode loops, which keep the stop
#: semantics and the streaming branch, and vae_channels_last (a channels-last AudioVAE
#: decoder, which `audio_vae.streaming_decode()` cannot drive).
STOP_TRANSFORMS = Path(__file__).with_name("voxcpm_transforms")
#: async_stop_loop's stop check, and two broken versions of it.
STOP_CHECK = (
    "        stop_flag = stop_host.item()\n        if i > min_len and stop_flag == 1:\n"
    "            break\n"
)
NEVER_STOP = "        stop_host.item()  # computed, never consulted\n"
LATE_STOP = (  # the flag of step i is only acted on at step i + 1: one patch too many
    "        if i > min_len + 1 and prev_flag == 1:\n            break\n"
    "        prev_flag = stop_host.item()\n"
)


@pytest.mark.gpu
@pytest.mark.skipif(not _voxcpm2_cached(), reason="needs voxcpm + openbmb/VoxCPM2 in the HF cache")
def test_voxcpm2_natural_length_checks_the_stop_condition(tmp_path):
    from kernel_agent.integrate.patcher import PatchReport, apply_transforms
    from kernel_agent.integrate.undo import Undo
    from kernel_agent.workloads.quality import perturb_linears

    wl = create_workload(WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm"))
    wl.load()
    with torch.inference_mode():
        ref = wl.natural_length_run()
    info = stopping.stop_summary(ref)
    assert info["stopped"] and 25 <= info["steps"] <= 60, info
    near_tie = float(wl.options["stop_near_tie"])  # a clear decision at the stop and before it
    assert info["stop_margin"] > 2 * near_tie and info["closest_margin"] > 2 * near_tie, info

    async_src = (STOP_TRANSFORMS / "async_stop_loop.py").read_text()
    assert async_src.count(STOP_CHECK) == 1
    loop = "    for i in tqdm(range(max_len)):\n"
    variants = {
        "never_stop": async_src.replace(STOP_CHECK, NEVER_STOP),
        "late_stop": async_src.replace(STOP_CHECK, LATE_STOP).replace(
            loop, "    prev_flag = 0\n" + loop
        ),
    }
    for name, source in variants.items():
        (tmp_path / f"{name}.py").write_text(source)

    def verdict(transform: Path | None = None) -> Any:
        handles: list[Undo] = []
        try:
            if transform is not None:
                apply_transforms(wl, [transform], PatchReport(), handles=handles)
            with torch.inference_mode():
                out = wl.natural_length_run(ref)
            return out, wl.compare_natural_length(ref, out)
        finally:
            for handle in reversed(handles):
                handle.undo()

    for transform in (
        STOP_TRANSFORMS / "async_stop_loop.py",
        STOP_TRANSFORMS / "skip_dead_work.py",
    ):
        _, cmp = verdict(transform)
        assert cmp.passed, (transform.name, cmp)
    with perturb_linears(wl.roots()):  # a benign numerical change keeps the stop step
        _, cmp = verdict()
    assert cmp.passed, cmp

    out, cmp = verdict(tmp_path / "never_stop.py")
    assert not cmp.passed and out["steps"] == wl.options["natural_max_patches"], cmp
    assert "never fires" in cmp.reason
    out, cmp = verdict(tmp_path / "late_stop.py")
    assert not cmp.passed and out["steps"] == info["steps"] + 1, cmp
    assert "fires 1 step(s) late" in cmp.reason and cmp.metrics["differing_step_margin"] > near_tie
    with wl.with_options({"stop_tolerance": 1}):  # ±1 only at a near-tie: still rejected
        assert not wl.compare_natural_length(ref, out).passed
    _, cmp = verdict()  # late_stop was undone
    assert cmp.passed, cmp


#: A transform whose `_inference` ignores `streaming=True`: every patch comes at the end.
IGNORE_STREAMING = """
import types


def apply(workload):
    model = workload.model
    original = type(model)._inference

    def _inference(self, *args, streaming=False, **kwargs):
        yield from original(self, *args, streaming=False, **kwargs)

    model._inference = types.MethodType(_inference, model)
"""


@pytest.mark.gpu
@pytest.mark.skipif(not _voxcpm2_cached(), reason="needs voxcpm + openbmb/VoxCPM2 in the HF cache")
def test_voxcpm2_time_to_first_audio(tmp_path):
    from kernel_agent.integrate.patcher import PatchReport, apply_transforms
    from kernel_agent.integrate.undo import Undo
    from kernel_agent.workloads.base import compare_audio
    from kernel_agent.workloads.quality import assess

    patches = 16
    options = {"metric": "ttfa", "patches": patches}
    wl = create_workload(WorkloadSpec("openbmb/VoxCPM2", "tts", family="voxcpm", options=options))
    wl.load()
    inputs = wl.make_inputs()
    timing = measure(wl, inputs, warmup=1, iters=3)
    ref, detail = timing.pop("output"), timing["metric_detail"]
    assert timing["metric"] == "ttfa" and detail["chunks"] == patches == ref["latents"].shape[0]
    assert ref["audio"].numel() == patches * 7680  # 160 ms of 48 kHz audio per patch
    # the first chunk comes after one patch (+ the prefill), not after all of them
    assert timing["median_ms"] < detail["run_ms"] / 5, timing
    assert detail["steady_chunks"] == 8 and 0 < detail["rtf"] < 1.5, detail  # eager: 0.6

    # the streamed audio is the non-streaming audio: the same trajectory, a stateful decoder
    with wl.with_options({"metric": "latency"}), torch.inference_mode():
        full = wl.run(inputs)
    assert torch.equal(full["latents"], ref["latents"])
    cmp = compare_audio(full["audio"], ref["audio"], min_spec_cosine=0.999)
    assert cmp.passed and float(cmp.metrics["waveform_cosine"]) > 0.9999, cmp

    # quality of the streamed output: the unmodified model replays itself exactly
    verdict = assess(wl, inputs, ref, ref, chaotic=True)
    assert verdict["passed"] and verdict["metrics"]["teacher_forced"]["min_step_cosine"] == 1.0

    def streamed(transform: Path) -> dict[str, Any]:
        handles: list[Undo] = []
        try:
            apply_transforms(wl, [transform], PatchReport(), handles=handles)
            with torch.inference_mode():
                out = wl.run(inputs)
            return assess(wl, inputs, ref, out, chaotic=True)
        finally:
            for handle in reversed(handles):
                handle.undo()

    # the live run's decode loops implement the streaming branch
    for name in ("async_stop_loop.py", "skip_dead_work.py"):
        verdict = streamed(STOP_TRANSFORMS / name)
        assert verdict["passed"], (name, verdict)
    (tmp_path / "ignore_streaming.py").write_text(IGNORE_STREAMING)
    with pytest.raises(RuntimeError, match=rf"1 audio chunk\(s\) for {patches} generated patches"):
        streamed(tmp_path / "ignore_streaming.py")
    assert streamed(STOP_TRANSFORMS / "skip_dead_work.py")["passed"]  # undone: works again

    # its channels-last AudioVAE decoder breaks `streaming_decode()`: rejected with the reason
    # (last: the transform removes the weight norm in place)
    apply_transforms(wl, [STOP_TRANSFORMS / "vae_channels_last.py"], PatchReport())
    with pytest.raises(RuntimeError, match="the streaming AudioVAE decode failed"):
        wl.run(inputs)
