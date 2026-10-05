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
from kernel_agent.workloads.base import WorkloadSpec
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

    def generate(self, target_text, min_len, max_len, **kwargs):
        self.calls.append({"text": target_text, "min_len": min_len, "max_len": max_len, **kwargs})
        x = torch.zeros(1, 4)
        feats = []
        for i in range(max_len):
            if self.bypass:
                pred = torch.tanh(x).unsqueeze(-1).expand(-1, -1, 2)
            else:
                pred = self.feat_decoder(
                    mu=self.lm(x) * 3, patch_size=2, cond=None, n_timesteps=10, cfg_value=2.0
                )
            feats.append(pred)
            # as VoxCPM: the stop head on the state that produced this patch, every patch
            logits = self.stop_head(torch.cat([x, torch.full((1, 1), float(i))], -1))
            if i > min_len and int(logits.argmax(-1)[0]) == 1 and not self.ignore_stop:
                break
            x = pred.mean(-1)
        return torch.cat(feats, -1).flatten().repeat(64).unsqueeze(0)  # "wav" [1, T]


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


#: Verbatim copies of two decode-loop transforms by the systems agent of a live VoxCPM2
#: run (runs/openbmb--VoxCPM2/20261005-042829/transforms/); both keep the stop semantics.
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
