"""VoxCPM2 (and VoxCPM 1.x) text-to-speech through the ``voxcpm`` package.

VoxCPM is a tokenizer-free diffusion-autoregressive TTS.  For every audio patch
a MiniCPM LM produces a hidden state, the LocDiT flow-matching head
(``model.feat_decoder``: CFG + fresh Gaussian noise per patch) samples a latent
patch, and the LocEnc feeds that latent back into the LM.  The trajectory is
chaotic: a one-ulp change anywhere changes the sampled patches, so the
free-running audio of every numerically-correct kernel diverges from the
baseline.  Quality is therefore judged with teacher forcing on the latents.

Teacher forcing wraps ``forward`` of the *current* ``model.feat_decoder``
instance at run time, i.e. after kernel replacements and transforms: the
wrapper runs the decoder (which draws its noise exactly as in the free run),
records its prediction for patch *i* and hands the reference patch *i* back to
the loop, so every step sees the baseline history and the baseline noise.  The
audio of that run is decoded from the reference latents, which validates the
(deterministic, non-chaotic) AudioVAE decoder as well.

Limitations: the candidate must keep calling ``model.feat_decoder`` from
Python once per patch, with the noise drawn from the global RNG as in the
original.  A transform that captures the whole step (LM + decoder) in one CUDA
graph, or calls ``generate`` more than once per run, cannot be teacher-forced
and is rejected with a clear reason.  The model runs in its checkpoint dtype
(``config.json``, bfloat16 for VoxCPM2); ``--dtype`` is ignored.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, compare_audio, compare_steps

TEXT = "Kernel level optimisation makes speech synthesis fast enough for real time conversation."


class VoxCPMWorkload(Workload):
    """Fixed-length zero-shot generation: exactly ``patches`` latent patches."""

    modality = Modality.TTS
    chaotic = True
    supports_teacher_forcing = True
    teacher_forcing_note = (
        "VoxCPM teacher forcing wraps `model.feat_decoder.forward` (the LocDiT/CFM sampler) "
        "at run time: candidates must call the current `model.feat_decoder` from Python "
        "once per patch and draw its noise with `torch.randn` as the original does."
    )
    # Thresholds calibrated on VoxCPM2 (60 patches, bf16, two texts): correct changes
    # (bundled Triton RMSNorm, fp32 RMSNorm, MATH-backend SDPA, every nn.Linear x
    # (1 ± 2^-8)) reach mean step cosine >= 0.9959, min >= 0.868 and RMS ratio within
    # ±0.2 %; broken RMSNorm (eps=1e-2, no weight) and attention (scale x1.25, one KV
    # head dropped) reach mean <= 0.974.
    defaults = {
        "text": TEXT,
        "patches": 60,
        "timesteps": 10,
        "cfg": 2.0,
        "seed": 0,
        "compile": False,
        "min_step_cosine": 0.7,
        "min_mean_step_cosine": 0.99,
        # Audio decoded from the teacher-forced (reference) latents; also reported,
        # informationally, for the free-running audio.
        "min_spec_cosine": 0.97,
    }
    # A diverged (but correct) free run is another plausible sample: its audio
    # loudness varies more than its latents (different seeds: x0.52 .. x1.54).
    sanity_rms_tolerance = {"audio": 0.6}

    def load(self) -> None:
        from huggingface_hub import snapshot_download

        path = snapshot_download(self.spec.repo_id, revision=self.spec.revision)
        with open(os.path.join(path, "config.json")) as fh:
            arch = str(json.load(fh).get("architecture", "voxcpm")).lower()
        from voxcpm.model.voxcpm import VoxCPMModel
        from voxcpm.model.voxcpm2 import VoxCPM2Model

        model_cls: Any = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
        self.model = model_cls.from_local(
            path, optimize=bool(self.options["compile"]), device=self.spec.device
        )
        self.sampling_rate = int(getattr(self.model, "sample_rate", 0))

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def reference_optimizations(self) -> str | None:
        """VoxCPM's own fast path, ``model.optimize()``: ``torch.compile(mode=
        "reduce-overhead", fullgraph=True)`` of both LMs' ``forward_step``, the
        LocEnc and the LocDiT estimator.  ``model.feat_decoder`` itself stays a
        Python call, so teacher forcing still applies."""
        if self.options["compile"]:
            return None  # `-o compile=true`: the measured baseline is already compiled
        self.model.optimize()
        # optimize() only prints a warning when it cannot compile; never time that as compiled.
        if not hasattr(self.model, "_feat_encoder_raw"):
            raise RuntimeError("model.optimize() did not compile the model (see its stderr)")
        return (
            "VoxCPM `model.optimize()`: torch.compile(mode='reduce-overhead', fullgraph=True) "
            "of base_lm/residual_lm.forward_step, feat_encoder and feat_decoder.estimator"
        )

    def make_inputs(self) -> str:
        return str(self.options["text"])

    @contextlib.contextmanager
    def _decoder_hook(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
        """Route every output of the *current* ``model.feat_decoder`` through ``fn``."""
        decoder = self.model.feat_decoder
        had_own = "forward" in vars(decoder)
        forward = decoder.forward

        def hooked(*args: Any, **kwargs: Any) -> torch.Tensor:
            return fn(forward(*args, **kwargs))

        vars(decoder)["forward"] = hooked
        try:
            yield
        finally:
            if had_own:
                vars(decoder)["forward"] = forward
            else:
                vars(decoder).pop("forward", None)

    def _generate(self, text: str) -> torch.Tensor:
        n = int(self.options["patches"])
        torch.manual_seed(int(self.options["seed"]))
        wav: torch.Tensor = self.model.generate(
            target_text=text,
            min_len=n,
            max_len=n,
            inference_timesteps=int(self.options["timesteps"]),
            cfg_value=float(self.options["cfg"]),
            retry_badcase=False,
            # generate() caps max_len at len(text tokens) * ratio + 10: never below n.
            retry_badcase_ratio_threshold=float(n),
        )
        return wav

    def run(self, inputs: str) -> dict[str, Any]:
        latents: list[torch.Tensor] = []

        def record(pred: torch.Tensor) -> torch.Tensor:
            latents.append(pred.detach().clone())
            return pred

        with self._decoder_hook(record):
            wav = self._generate(inputs)
        return {
            "audio": wav.float().flatten().cpu(),
            # [patches, batch, feat_dim, patch_size]; empty if the decoder was bypassed.
            "latents": torch.stack(latents).float().cpu() if latents else torch.empty(0),
            "sampling_rate": self.sampling_rate,
        }

    def run_teacher_forced(self, inputs: str, reference: Any) -> dict[str, Any]:
        ref = reference.get("latents") if isinstance(reference, dict) else None
        if ref is None or ref.numel() == 0:
            raise ValueError(
                "the baseline output has no recorded latents; re-run `analyze` to record them"
            )
        preds: list[torch.Tensor] = []

        def force(pred: torch.Tensor) -> torch.Tensor:
            step = len(preds)
            preds.append(pred.detach().float().cpu())
            if step >= ref.shape[0]:
                return pred  # longer than the reference; reported as a step-count mismatch
            return ref[step].to(device=pred.device, dtype=pred.dtype)

        # Through `self.run` so that transforms wrapping `workload.run` stay active.
        with self._decoder_hook(force):
            out = self.run(inputs)
        if not preds:
            raise RuntimeError(
                "teacher forcing saw no `model.feat_decoder` call: the candidate bypasses the "
                "Python decoder call (e.g. one CUDA graph over the whole step), so it cannot "
                "be validated"
            )
        # The audio is decoded from the reference latents, so it validates the
        # (deterministic) AudioVAE decoder path as well.
        return {"latents": torch.stack(preds), "audio": out["audio"]}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return compare_audio(
            reference["audio"],
            candidate["audio"],
            min_spec_cosine=float(self.options["min_spec_cosine"]),
        )

    def compare_teacher_forced(self, reference: Any, candidate: Any) -> Comparison:
        steps = compare_steps(
            reference["latents"],
            candidate["latents"],
            min_step_cosine=float(self.options["min_step_cosine"]),
            min_mean_step_cosine=float(self.options["min_mean_step_cosine"]),
        )
        decoded = compare_audio(
            reference["audio"],
            candidate["audio"],
            min_spec_cosine=float(self.options["min_spec_cosine"]),
        )
        metrics = {**steps.metrics, **{f"decoded_{k}": v for k, v in decoded.metrics.items()}}
        reasons = [steps.reason] if not steps.passed else []
        if not decoded.passed:
            reasons.append(f"audio decoded from the reference latents: {decoded.reason}")
        return Comparison(steps.passed and decoded.passed, metrics, "; ".join(reasons))
