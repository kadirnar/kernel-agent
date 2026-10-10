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

The fixed length (``min_len = max_len``) keeps runs comparable but never lets the
stop head decide, so a natural-length run (``natural_length_run``,
:mod:`kernel_agent.workloads.stopping`) checks the stop condition: ``natural_text``
with VoxCPM's own ``min_len`` (2), recorded at analyze time; every candidate replays
it teacher forced and must stop at the same patch.

Limitations: the candidate must keep calling ``model.feat_decoder`` from
Python once per patch, with the noise drawn from the global RNG as in the
original.  A transform that captures the whole step (LM + decoder) in one CUDA
graph, or calls ``generate`` more than once per run, cannot be teacher-forced
and is rejected with a clear reason.  The model runs in the run's dtype (``--dtype``, by
default the checkpoint's, bfloat16 for VoxCPM2; float16 or float32 on a GPU without bf16
tensor cores, ``workloads/dtypes.py``), which :meth:`VoxCPMWorkload.set_dtype` hands to
VoxCPM's config; a run from before #255 keeps the checkpoint's.

``-o metric=ttfa`` (:mod:`kernel_agent.objective`) optimises the time to first audio:
``run`` goes through VoxCPM's streaming path, ``generate_streaming``
(``_inference(streaming=True)`` yields every patch latent as it is generated, the
stateful ``audio_vae.streaming_decode()`` decodes it to one audio chunk), and marks
every chunk on arrival. The output is the concatenated chunks plus the latents, so
teacher forcing, the held-out input and the natural-length run judge the full streamed
output exactly as they judge the non-streaming one. A transform that breaks the
streaming path (an ``_inference`` without its streaming branch, an AudioVAE decoder
that ``streaming_decode()`` cannot drive) is rejected with a clear reason.

``-o lora=<name>`` runs a fine-tuned model: VoxCPM's own LoRA (``lora_weights.safetensors``
+ ``lora_config.json``, the format its fine-tuning writes) applied on the checkpoint through
``from_local(lora_config=...)`` and ``load_lora_weights``, unmerged (``LoRALinear``: the base
GEMM plus the two low-rank GEMMs, as a server holding several adapters runs it). ``name`` is
an adapter directory under the repo's ``adapters/`` (fetched alone; the base checkpoint's
download skips ``adapters/``) or a local directory.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent import objective
from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, compare_audio, compare_steps

TEXT = "Kernel level optimisation makes speech synthesis fast enough for real time conversation."
#: Held-out e2e input (``holdout_options``): another sentence and seed, the same patches.
HOLDOUT_TEXT = (
    "A held out sentence proves that every patch is computed, never replayed from memory."
)
#: Extra capture setting (``variants``): another LM prefill length, few patches.
SHORT_TEXT = "Short sentences matter too."
#: Diverse input set (``diverse_inputs``): texts of other kinds and languages, the main
#: input's patches and seed.
DIVERSE_TEXTS = {
    "question": "Could you tell me how late the museum stays open on public holidays?",
    "numbers": "Your booking code is four seven two nine; the train leaves from platform 11.",
    "chinese": "今天下午三点，我们在图书馆门口见面，然后一起去看电影。",
    "german": "Bitte schließen Sie alle Fenster, bevor Sie am Abend das Haus verlassen.",
}
#: Natural-length run (``natural_length_run``): VoxCPM2 stops it after about 30 patches
#: (seed 0), with a clear stop logit margin at the stop and before it.
NATURAL_TEXT = "When the sentence is finished, the speaker stops talking and waits for the reply."
#: ``min_len`` of the natural-length run: VoxCPM's own default.
NATURAL_MIN_PATCHES = 2
#: Perceptual gate (``--quality near-lossless``, ``perceptual_samples``): held-out
#: sentences (language, text) x seeds at natural length, cloned from ``SPEAKER_WAV`` (a
#: VoxCPM2 zero-shot voice, 16 kHz) so that every sample has the same speaker.
PERCEPTUAL_TEXTS = (
    (
        "en",
        "Please remember to bring your passport, a warm jacket and the charger for your laptop.",
    ),
    (
        "en",
        "The quick brown fox jumps over the lazy dog while the orchestra tunes its instruments.",
    ),
    ("en", "Every morning she walks along the river and feeds the ducks before going to work."),
    ("de", "Der Zug nach Berlin fährt heute eine Stunde später ab, weil es stark geschneit hat."),
)
PERCEPTUAL_SEEDS = (0, 1)
SPEAKER_WAV = "voxcpm_speaker.wav"
ASSETS = Path(__file__).with_name("assets")
#: ``-o lora=<name>``: the repo's adapter directories (``adapters/<name>/``).
ADAPTERS_DIR = "adapters"


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
        # metric=ttfa: the teacher-forced audio is streamed from the reference latents, so it
        # depends on the AudioVAE path only and a correct decode matches the reference (fp32,
        # waveform cosine ~1.0). A decode that loses its causal-convolution state between
        # 160 ms chunks clicks at every boundary: on VoxCPM2, waveform cosine 0.93 while the
        # spectral cosine stays at 0.970.
        "min_stream_waveform_cosine": 0.98,
        # Natural-length run (stop condition): the candidate stops at the baseline's patch;
        # `stop_tolerance=1` accepts ±1 patch at a near-tie of the baseline's stop logits.
        "natural_text": NATURAL_TEXT,
        "natural_max_patches": 100,
        "stop_tolerance": 0,
        "stop_near_tie": 0.5,
        # A LoRA adapter (`adapters/<name>` of the repo, or a directory); None: the base model.
        "lora": None,
    }
    # A diverged (but correct) free run is another plausible sample: its audio
    # loudness varies more than its latents (different seeds: x0.52 .. x1.54).
    sanity_rms_tolerance = {"audio": 0.6}
    # `-o metric=ttfa`: the streaming path (`_stream`), time to the first audio chunk.
    metrics = (objective.LATENCY, objective.TTFA)

    #: :meth:`set_dtype`'s dtype, passed to VoxCPM's config; None: the checkpoint's (a run
    #: from before #255, whose ``run.json`` has no ``dtype``).
    config_dtype: str | None = None

    def set_dtype(self, dtype: str) -> None:
        """VoxCPM reads its dtype from the checkpoint's ``config.json``: :meth:`load` hands
        ``from_local`` a copy of the checkpoint whose config says ``dtype``."""
        super().set_dtype(dtype)
        self.config_dtype = self.spec.dtype

    def load(self) -> None:
        from huggingface_hub import snapshot_download

        # adapters are fetched one at a time (`lora`), never with the base checkpoint
        path = snapshot_download(
            self.spec.repo_id, revision=self.spec.revision, ignore_patterns=[f"{ADAPTERS_DIR}/*"]
        )
        with open(os.path.join(path, "config.json")) as fh:
            arch = str(json.load(fh).get("architecture", "voxcpm")).lower()
        from voxcpm.model.voxcpm import VoxCPMModel
        from voxcpm.model.voxcpm2 import VoxCPM2Model

        model_cls: Any = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
        adapter = lora_directory(self.options.get("lora"), self.spec.repo_id, self.spec.revision)
        extra = {"lora_config": lora_config(model_cls, adapter)} if adapter is not None else {}
        with _with_config_dtype(path, self.config_dtype) as checkpoint:
            self.model = model_cls.from_local(
                checkpoint,
                optimize=bool(self.options["compile"]),
                device=self.spec.device,
                **extra,
            )
        if adapter is not None:
            loaded, skipped = self.model.load_lora_weights(str(adapter))
            if skipped or not loaded:  # a partly loaded adapter would be another model
                raise ValueError(
                    f"lora={self.options['lora']}: {len(skipped)} of {len(loaded) + len(skipped)} "
                    f"tensors match no LoRA module of the model (e.g. {(skipped or ['none'])[0]}); "
                    "its lora_config.json must name the targets it was trained with"
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

    def holdout_options(self, variant: int = 1) -> dict[str, Any] | None:
        """Another sentence and seed, the same number of patches; variants ``>= 2``
        change only the seed (the shapes of variant 1)."""
        return {"text": HOLDOUT_TEXT, "seed": int(self.options["seed"]) + variant}

    def variants(self) -> list[dict[str, Any]]:
        return [{"text": SHORT_TEXT, "patches": min(int(self.options["patches"]), 8)}]

    def diverse_inputs(self) -> dict[str, dict[str, Any]]:
        """:data:`DIVERSE_TEXTS`: a question, numbers, Chinese, German; the same patches."""
        return {label: {"text": text} for label, text in DIVERSE_TEXTS.items()}

    def natural_length_run(self, reference: Any = None) -> dict[str, Any]:
        """``natural_text`` with the stop head live (``min_len`` 2, ``max_len``
        ``natural_max_patches``), teacher forced on ``reference`` when given.  The stop
        logit margins come from a hook on ``model.stop_head``: the unmodified loop calls
        it once per patch (a candidate may call it differently; only its patch count
        counts)."""
        max_patches = int(self.options["natural_max_patches"])
        options = {
            "text": self.options["natural_text"],
            "patches": max_patches,
            "min_patches": NATURAL_MIN_PATCHES,
        }
        logits: list[torch.Tensor] = []
        hook = self.model.stop_head.register_forward_hook(
            lambda module, args, output: logits.append(output.detach().float().reshape(-1, 2)[0])
        )
        try:
            with self.with_options(options):
                inputs = self.make_inputs()
                if reference is None:
                    out = self.run(inputs)
                else:
                    out = self.run_teacher_forced(inputs, reference)
        finally:
            hook.remove()
        steps = int(out["latents"].shape[0])
        result: dict[str, Any] = {
            "steps": steps,
            "min_steps": NATURAL_MIN_PATCHES,
            "max_steps": max_patches,
            "output_length": int(out["audio"].numel()),
            "latents": out["latents"],
            "audio": out["audio"],
        }
        if reference is None and len(logits) == steps:
            result["stop_margins"] = [float(x[1] - x[0]) for x in torch.stack(logits).cpu()]
        return result

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
        # `min_patches` (natural_length_run only): the stop head may end the run early.
        min_len = self.options.get("min_patches")
        torch.manual_seed(int(self.options["seed"]))
        generate = self._stream if self.metric == objective.TTFA else self.model.generate
        wav: torch.Tensor = generate(
            target_text=text,
            min_len=n if min_len is None else int(min_len),
            max_len=n,
            inference_timesteps=int(self.options["timesteps"]),
            cfg_value=float(self.options["cfg"]),
            retry_badcase=False,
            # generate() caps max_len at len(text tokens) * ratio + 10: never below n.
            retry_badcase_ratio_threshold=float(n),
            **self._reference_wav(),
        )
        return wav

    def _reference_wav(self) -> dict[str, str]:
        """``reference_wav`` option (perceptual samples): clone the voice of that file (a
        name in ``assets/`` or a path), on the streaming path (``metric=ttfa``) as well."""
        name = self.options.get("reference_wav")
        return {"reference_wav_path": str(ASSETS / name)} if name else {}

    def _stream(self, **kwargs: Any) -> torch.Tensor:
        """``metric=ttfa``: ``model.generate_streaming(**kwargs)``, every chunk marked on
        arrival (:meth:`mark_chunk`; VoxCPM hands it over on the host). Returns the chunks
        concatenated. The streaming path must yield one chunk per generated patch. Inside
        :meth:`metric_window` (the timed requests) the stream is closed after its first
        chunk, as a client that stops listening closes it."""
        chunks: list[torch.Tensor] = []
        patches = 0

        def count(pred: torch.Tensor) -> torch.Tensor:
            nonlocal patches
            patches += 1
            return pred

        with self._decoder_hook(count):
            stream = self.model.generate_streaming(**kwargs)
            try:
                for chunk in stream:
                    chunks.append(chunk)
                    self.mark_chunk(audio_ms=1000.0 * chunk.shape[-1] / self.sampling_rate)
                    if self.in_window:
                        break
            except torch.OutOfMemoryError:
                raise  # the GPU, not the streaming path
            except Exception as exc:
                raise RuntimeError(_streaming_failure(exc)) from exc
            finally:
                stream.close()
        if not chunks:
            raise RuntimeError("metric=ttfa: the streaming path yielded no audio chunk")
        if patches and len(chunks) != patches and not self.in_window:
            raise RuntimeError(
                f"metric=ttfa: the streaming path yielded {len(chunks)} audio chunk(s) for "
                f"{patches} generated patches: `_inference(streaming=True)` must yield every "
                "patch latent as soon as it is generated (an `_inference` replacement that "
                "ignores `streaming=True` hands over all patches at the end of the run)"
            )
        return torch.cat(chunks, dim=-1)

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
        wave = decoded.metrics.get("waveform_cosine")
        floor = float(self.options["min_stream_waveform_cosine"])
        if self.metric == objective.TTFA and isinstance(wave, float) and wave < floor:
            reasons.append(
                f"audio streamed from the reference latents: waveform cosine {wave:.4f} < "
                f"{floor} (does the AudioVAE decode keep its state between chunks?)"
            )
        return Comparison(not reasons, metrics, "; ".join(reasons))

    # ------------------------------------------------- perceptual gate (near-lossless)

    #: ``--quality near-lossless``: the teacher-forced sanity floor and ±1 patch at a
    #: near-tie of the stop logits. Calibrated on VoxCPM2 (main input, 60 patches): FP8
    #: weight-only (e4m3 per output channel, every nn.Linear of both LMs and the LocDiT)
    #: reaches mean step cosine 0.987, min 0.65 (the exact thresholds reject it);
    #: `model.optimize()` 0.998 / 0.96; broken RMSNorm (eps 1e-2), a dropped KV head and
    #: int4 per-tensor weights reach mean <= 0.70.
    near_lossless_options = {
        "min_step_cosine": 0.2,
        "min_mean_step_cosine": 0.95,
        "stop_tolerance": 1,
    }
    #: ``--quality relaxed`` (#175): twice the mean-step budget (1 - 0.95 -> 1 - 0.90) and
    #: ±2 patches at a near-tie up to a margin of 1.0. NVFP4 weights everywhere (0.948) and
    #: MXFP4 (0.913) pass it; broken RMSNorm, a dropped KV head and int4 weights (<= 0.70)
    #: still fail. The min step cosine stays 0.2 (broken RMSNorm reaches 0.10).
    relaxed_options = {"min_mean_step_cosine": 0.90, "stop_tolerance": 2, "stop_near_tie": 1.0}

    def perceptual_samples(self) -> list[dict[str, Any]]:
        """``PERCEPTUAL_TEXTS`` x ``PERCEPTUAL_SEEDS``, natural length (the stop head
        decides, ``perceptual_max_patches`` at most), in the voice of ``SPEAKER_WAV``.
        They run through :meth:`run`, i.e. the path the run's metric times: ``generate``
        by default, ``generate_streaming`` with ``metric=ttfa`` (the gate scores the whole
        streamed audio, chunk-wise AudioVAE decode included)."""
        n = int(self.options.get("perceptual_max_patches", 120))
        return [
            {
                "text": text,
                "language": language,
                "seed": seed,
                "patches": n,
                "min_patches": NATURAL_MIN_PATCHES,
                "reference_wav": SPEAKER_WAV,
            }
            for language, text in PERCEPTUAL_TEXTS
            for seed in PERCEPTUAL_SEEDS
        ]

    def perceptual_quality(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from kernel_agent.workloads.perceptual import score_tts

        return score_tts(
            [
                {
                    "audio": s["output"]["audio"],
                    "sampling_rate": s["output"]["sampling_rate"],
                    "text": s["options"]["text"],
                    "language": s["options"].get("language", "en"),
                }
                for s in samples
            ],
            device=self.spec.device,
        )

    def compare_perceptual(
        self, reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
    ) -> Comparison:
        from kernel_agent.workloads import perceptual as p

        opt = self.options  # thresholds: -o max_error_increase=... (calibration: README)
        return p.compare_tts(
            reference,
            candidate,
            max_error_increase=float(opt.get("max_error_increase", p.MAX_ERROR_INCREASE)),
            min_speaker_similarity=float(
                opt.get("min_speaker_similarity", p.MIN_SPEAKER_SIMILARITY)
            ),
            min_speaker_similarity_worst=float(
                opt.get("min_speaker_similarity_worst", p.MIN_SPEAKER_SIMILARITY_WORST)
            ),
            max_mos_drop=float(opt.get("max_mos_drop", p.MAX_MOS_DROP)),
        )


def lora_directory(lora: Any, repo_id: str, revision: str | None) -> Path | None:
    """``-o lora=``: a local directory, or the name of an adapter directory under the repo's
    ``adapters/`` (only that adapter is downloaded); None without the option."""
    if not lora:
        return None
    path = Path(str(lora)).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            repo_id, revision=revision, allow_patterns=[f"{ADAPTERS_DIR}/{lora}/*"]
        )
        path = Path(root) / ADAPTERS_DIR / str(lora)
    if not any(
        (path / name).is_file() for name in ("lora_weights.safetensors", "lora_weights.ckpt")
    ):
        raise ValueError(
            f"lora={lora}: no lora_weights.safetensors in {path} (give a directory, or the name "
            f"of an adapter under {ADAPTERS_DIR}/ of {repo_id})"
        )
    return path


def lora_config(model_cls: type, directory: Path) -> Any:
    """VoxCPM's ``LoRAConfig`` (of ``model_cls``'s module) from the adapter's
    ``lora_config.json``: its ``lora_config`` object, as VoxCPM's fine-tuning writes it (keys
    the installed VoxCPM does not know, e.g. ``base_model``, are left out)."""
    config_cls = getattr(sys.modules[model_cls.__module__], "LoRAConfig", None)
    if config_cls is None:
        raise ValueError(f"lora: {model_cls.__name__}'s VoxCPM module has no LoRAConfig")
    with open(directory / "lora_config.json") as fh:
        raw = json.load(fh)
    fields = raw.get("lora_config", raw)
    return config_cls(**{k: v for k, v in fields.items() if k in config_cls.model_fields})


@contextlib.contextmanager
def _with_config_dtype(path: str, dtype: str | None) -> Iterator[str]:
    """The checkpoint directory ``path`` for ``from_local``; with a ``dtype`` its config does
    not say, a temporary directory of links to its files with a ``config.json`` that does
    (``from_local`` takes the model's dtype from there only)."""
    with open(os.path.join(path, "config.json")) as fh:
        config = json.load(fh)
    if dtype is None or config.get("dtype") == dtype:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="voxcpm-dtype-") as tmp:
        for name in os.listdir(path):
            if name != "config.json":
                os.symlink(os.path.join(path, name), os.path.join(tmp, name))
        with open(os.path.join(tmp, "config.json"), "w") as fh:
            json.dump({**config, "dtype": dtype}, fh)
        yield tmp


def _streaming_failure(exc: BaseException) -> str:
    """A clear reason for a failure of VoxCPM's streaming path (``metric=ttfa``)."""
    frames = traceback.extract_tb(exc.__traceback__)
    message = " ".join(str(exc).split())
    what = f"{type(exc).__name__}: {message}"[:500]
    if any(f.name == "decode_chunk" or "audiovae" in f.filename for f in frames):
        return (
            f"metric=ttfa: the streaming AudioVAE decode failed ({what}). "
            "`audio_vae.streaming_decode()` decodes one patch at a time and carries the "
            "causal-convolution state between chunks by replacing the `forward` of every "
            "CausalConv1d / CausalTransposeConv1d in `audio_vae.decoder` with one that takes "
            "[B, C, T] tensors: a transform that changes the decoder's tensor layout or the "
            "forwards of those modules (a channels-last 4-D decoder, say) must keep that path "
            "working"
        )
    return (
        f"metric=ttfa: VoxCPM's streaming path (`generate_streaming` -> "
        f"`_inference(streaming=True)`) failed ({what}); a transform that replaces "
        "`_inference` must keep its streaming branch working (yield every patch latent as "
        "soon as it is generated)"
    )
