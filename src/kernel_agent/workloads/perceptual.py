"""``--quality near-lossless`` / ``relaxed``: the perceptual gate for numerics-changing
optimisations.

``exact`` accepts changes whose numerics stay within rounding noise of the baseline: module
tolerances (:mod:`kernel_agent.kernels.compare`) and, end to end, the workload's own
comparison or teacher forcing (:mod:`kernel_agent.workloads.quality`). Once a model runs
near the memory-bandwidth floor of its bf16 weights, the next big gains (FP8 weights, ...)
change numerics by design. ``near-lossless`` accepts them when the *perceptual* quality
stays within the noise of eager; ``relaxed`` (the default of new runs, #175) with about
twice the error budgets (:data:`RELAXED_GATE`, :meth:`Workload.relaxed_options`, the
``relaxed`` tiers of :mod:`kernel_agent.kernels.compare`): small measured drops pass,
broken kernels still fail.

* **Perceptual gate.** The workload declares held-out samples
  (:meth:`~kernel_agent.workloads.base.Workload.perceptual_samples`; VoxCPM: four
  sentences in two languages x two seeds, natural length). ``analyze`` generates them with
  the eager model, scores them
  (:meth:`~kernel_agent.workloads.base.Workload.perceptual_quality`; TTS: Whisper-large-v3
  transcript and word / character error rate against the text, a WavLM speaker embedding,
  a UTMOS22 MOS when that predictor is cached) and stores outputs and scores as
  ``baseline_output_perceptual.pt`` (hashed and read-only in ``.truth/``). ``e2e``
  generates the same samples with the candidate (untimed, free running), scores them and
  compares them paired with eager's
  (:meth:`~kernel_agent.workloads.base.Workload.compare_perceptual`). Samples run through
  ``Workload.run`` under the run's metric (:mod:`kernel_agent.objective`), so the gate
  judges the audio of the code path the objective times: with ``-o metric=ttfa`` VoxCPM's
  streaming path (the whole streamed output), otherwise ``generate``.
* **LLM gate** (:func:`score_llm`, :func:`compare_llm`). On natural prompts greedy text
  diverges within a few tokens under FP8 weights although every distribution stays close,
  so free-running tokens cannot judge it. Each sample (the main, held-out and diverse
  prompts) is scored teacher forced, one forward of the candidate over the prompt and
  eager's continuation: per token the KL from eager's distribution and whether eager's top
  token stays the candidate's; and the likelihood of the candidate's own continuation,
  which catches a decode loop that writes wrong text while its forward is fine.
  Thresholds: :data:`LLM_MAX_KL` and the lines after it (calibration: README).
* **Sanity floor.** Teacher forcing, the held-out input and the stop check stay, with the
  workload's looser :attr:`~kernel_agent.workloads.base.Workload.near_lossless_options`
  (thresholds that FP8-weight variants pass and broken kernels still fail; the stop check
  accepts ±1 step at a near-tie); in ``relaxed`` mode with
  :attr:`~kernel_agent.workloads.base.Workload.relaxed_options` on top (:func:`floor_options`).
  Options the user set (``-o``) win.
* **Module tolerance tier.** Targets whose spec allows reduced precision
  (``"precision": "reduced"``) are captured with the ``near-lossless`` (``relaxed``) tier of
  :mod:`kernel_agent.kernels.compare` (whole-tensor cosine and relative L2 error).

Without a perceptual baseline (the workload declares no samples, or ``analyze`` ran in
``exact`` mode or its perceptual run failed) a near-lossless or relaxed run keeps the exact
checks.
The scoring models load lazily, in the worker process and only when the gate runs, one at
a time, and are freed afterwards; in the paired A/B of the integration after A's state is
freed (``check(before_scoring=...)``). Out of GPU memory in the gate is no verdict: the
worker makes the step ``oom`` (``abtest.CHECKS``, #137).
"""

from __future__ import annotations

import contextlib
import gc
import statistics
import time
import traceback
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

from kernel_agent.workloads.base import Comparison, Workload

EXACT = "exact"
NEAR_LOSSLESS = "near-lossless"
RELAXED = "relaxed"
MODES = (EXACT, NEAR_LOSSLESS, RELAXED)
#: The modes with a perceptual gate (and reduced-precision targets).
GATED = (NEAR_LOSSLESS, RELAXED)

#: Scoring models of the TTS gate (Hugging Face Hub; downloaded once when not cached).
ASR_MODEL = "openai/whisper-large-v3"
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
#: The MOS predictor, UTMOS22 strong (``torch.hub.load("tarepan/SpeechMOS:v1.2.0",
#: "utmos22_strong")``), used only when its repository and checkpoint are in the torch hub
#: cache: the gate never downloads code.
MOS_HUB_PREFIX = "tarepan_SpeechMOS"
MOS_ENTRY = "utmos22_strong"
MOS_CHECKPOINT = "utmos22_strong_step7459_v1.pt"
#: Sampling rate of every scoring model.
SCORE_RATE = 16000
#: Languages written without spaces between words: character error rate instead of WER.
CHAR_LANGUAGES = frozenset({"zh", "ja", "th", "lo", "km", "my"})

#: TTS gate thresholds, calibrated on VoxCPM2 (8 cloned samples; README): eager with
#: other seeds (a fully diverged, correct trajectory), ``model.optimize()`` and FP8
#: weight-only fake quantisation reach error increase 0.000, speaker similarity >= 0.972
#: (worst sample >= 0.961) and MOS change >= -0.049 (an improvement); broken RMSNorm
#: (eps 1e-2), a dropped KV head and int4 weights reach +0.74 / 0.91 / 0.71 / -0.42 at
#: best. The mean error rate of the candidate's samples may exceed eager's by this much,
MAX_ERROR_INCREASE = 0.05
#: ... the speaker embedding of each sample has at least this cosine with eager's sample of
#: the same text and seed (mean over the samples, worst sample), ...
MIN_SPEAKER_SIMILARITY = 0.93
MIN_SPEAKER_SIMILARITY_WORST = 0.85
#: ... and the mean MOS may drop by at most this much.
MAX_MOS_DROP = 0.3

#: LLM gate (teacher forcing on eager's continuations, :func:`score_llm`): eager's
#: distribution at every position is kept as its top this-many tokens plus the rest.
LLM_TOPK = 32
#: LLM gate thresholds, calibrated on Qwen3-0.6B (the 14 natural prompts of
#: ``LLMWorkload.perceptual_samples``, 64 tokens each; README): FP8 e4m3 weight-only fake
#: quantisation of every decoder Linear (per output channel / per tensor) reaches a mean KL
#: of 0.011 / 0.009 (worst sample 0.043 / 0.022), top-1 agreement 0.95 and an NLL change
#: of -0.07 / +0.03 nats per token; one bf16 rounding step per Linear output 0.005; int4
#: weights (group 128, per channel), FP8 scales x1.2, RMSNorm eps 1e-2 and a dropped KV
#: head reach a mean KL >= 0.34, worst sample >= 0.57 and top-1 <= 0.76, and a decode loop
#: whose continuation is shifted by one token +0.38 nats per token. The mean KL over the
#: samples may be at most this, ...
LLM_MAX_KL = 0.05
#: ... the worst sample's at most this, ...
LLM_MAX_KL_WORST = 0.15
#: ... eager's most likely token is the candidate's at this share of positions or more, ...
LLM_MIN_TOP1 = 0.85
#: ... and the candidate's own continuations are at most this many nats per token less
#: likely (under the candidate) than eager's own (under eager), on average.
LLM_MAX_NLL_INCREASE = 0.25

#: ``--quality relaxed`` (#175): the gate's thresholds as option overrides (the option names
#: of :meth:`Workload.compare_perceptual`; ``-o`` wins), about twice the near-lossless error
#: budgets where the broken variants of the calibrations above keep a clear margin
#: (README, "Quality modes"). TTS: the error rate may rise by 0.10 (broken: >= +0.74); the
#: speaker similarity (mean 0.93 -> 0.90, worst sample 0.85 -> 0.80) and the MOS drop (0.3 ->
#: 0.45) move by less than twice, as RMSNorm eps 1e-2 reaches 0.912 / 0.710 / -0.42 there.
#: LLM: mean KL 0.10 (broken: >= 0.34), worst sample 0.30 (>= 0.57), top-1 0.80 (<= 0.76),
#: NLL increase 0.30 nats per token (a decode loop shifted by one token: +0.38).
RELAXED_GATE: dict[str, float] = {
    "max_error_increase": 0.10,
    "min_speaker_similarity": 0.90,
    "min_speaker_similarity_worst": 0.80,
    "max_mos_drop": 0.45,
    "max_kl": 0.10,
    "max_kl_worst": 0.30,
    "min_top1": 0.80,
    "max_nll_increase": 0.30,
}


def mode_of(value: Any) -> str:
    """A validated quality mode (``None`` / empty: ``exact``)."""
    mode = str(value or EXACT)
    if mode not in MODES:
        raise ValueError(f"unknown quality mode {mode!r} (one of {', '.join(MODES)})")
    return mode


def gated(mode: Any) -> bool:
    """Whether quality mode ``mode`` judges numerics changes with the perceptual gate
    (:data:`GATED`: ``near-lossless``, ``relaxed``)."""
    return mode_of(mode) in GATED


def floor_options(workload: Workload, mode: str = NEAR_LOSSLESS) -> dict[str, Any]:
    """The workload's option overrides in quality mode ``mode``, without the ones the user
    set: ``near_lossless_options`` (the sanity floor); in ``relaxed`` mode also the gate's
    :data:`RELAXED_GATE` and the workload's ``relaxed_options`` on top."""
    found = dict(workload.near_lossless_options)
    if mode_of(mode) == RELAXED:
        found.update(RELAXED_GATE)
        found.update(workload.relaxed_options)
    return {k: v for k, v in found.items() if k not in workload.spec.options}


# ------------------------------------------------------------------ text


def tokens(text: str, *, chars: bool = False) -> list[str]:
    """Lower-cased words (characters with ``chars``), without punctuation and symbols."""
    text = unicodedata.normalize("NFKC", text).lower()
    cleaned = "".join(" " if unicodedata.category(c)[0] in "PSZC" else c for c in text)
    return [c for c in cleaned if not c.isspace()] if chars else cleaned.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance (substitutions, insertions, deletions) of two token lists."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[len(hyp)]


def error_rate(reference: str, hypothesis: str, *, language: str = "en") -> float:
    """Word error rate of ``hypothesis`` against ``reference`` (character error rate for
    :data:`CHAR_LANGUAGES`)."""
    chars = language.split("-")[0].lower() in CHAR_LANGUAGES
    ref, hyp = tokens(reference, chars=chars), tokens(hypothesis, chars=chars)
    return edit_distance(ref, hyp) / max(len(ref), 1)


# ------------------------------------------------------------------ TTS scoring


def _free() -> None:
    """Give the memory of a scoring model the caller dropped back to the GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pretrained(cls: Any, repo: str, **kwargs: Any) -> Any:
    """``cls.from_pretrained`` from the local cache (no Hub round trip), else downloaded."""
    try:
        return cls.from_pretrained(repo, local_files_only=True, **kwargs)
    except OSError:
        return cls.from_pretrained(repo, **kwargs)


def resample(wave: torch.Tensor, rate: int, target: int = SCORE_RATE) -> torch.Tensor:
    """A mono float waveform at ``target`` Hz."""
    wave = wave.detach().float().flatten().cpu()
    if rate == target:
        return wave
    try:
        import torchaudio.functional as taf

        return taf.resample(wave, rate, target)
    except ImportError:
        import librosa

        return torch.from_numpy(librosa.resample(wave.numpy(), orig_sr=rate, target_sr=target))


def mos_location() -> Path | None:
    """The cached UTMOS22 hub repository (``None`` when it or its checkpoint is missing:
    the gate then skips MOS)."""
    hub = Path(torch.hub.get_dir())
    repos = sorted(p for p in hub.glob(f"{MOS_HUB_PREFIX}*") if (p / "hubconf.py").is_file())
    if not repos or not (hub / "checkpoints" / MOS_CHECKPOINT).is_file():
        return None
    return repos[-1]


def _transcribe(waves: list[torch.Tensor], languages: list[str], device: torch.device) -> list[str]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = _pretrained(WhisperProcessor, ASR_MODEL)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # eager attention: one cold pass over a few clips; SDPA's first calls in a fresh
    # process cost ~9 s more on the RTX 5070 Ti (sm_120) than they save
    model = _pretrained(
        WhisperForConditionalGeneration, ASR_MODEL, dtype=dtype, attn_implementation="eager"
    )
    model = model.to(device).eval()
    texts = [""] * len(waves)
    try:
        for language in dict.fromkeys(languages):  # one batch per language
            idx = [i for i, lang in enumerate(languages) if lang == language]
            features = processor(
                [waves[i].numpy() for i in idx], sampling_rate=SCORE_RATE, return_tensors="pt"
            ).input_features.to(device, dtype)
            with torch.inference_mode():
                ids = model.generate(features, language=language, task="transcribe")
            for i, text in zip(
                idx, processor.batch_decode(ids, skip_special_tokens=True), strict=True
            ):
                texts[i] = text.strip()
    finally:
        del model
        _free()
    return texts


def _embed(waves: list[torch.Tensor], device: torch.device) -> list[list[float]]:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    extractor = _pretrained(AutoFeatureExtractor, SPEAKER_MODEL)
    model = _pretrained(WavLMForXVector, SPEAKER_MODEL).to(device).eval()
    out = []
    try:
        for wave in waves:
            inputs = extractor(wave.numpy(), sampling_rate=SCORE_RATE, return_tensors="pt")
            with torch.inference_mode():
                emb = model(**inputs.to(device)).embeddings[0].float()
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().tolist())
    finally:
        del model
        _free()
    return out


def _mos(waves: list[torch.Tensor], device: torch.device) -> list[float] | None:
    repo = mos_location()
    if repo is None:
        return None
    model = torch.hub.load(str(repo), MOS_ENTRY, source="local", trust_repo=True, progress=False)
    model = model.to(device).eval()
    try:
        with torch.inference_mode():
            return [float(model(w.to(device).unsqueeze(0), SCORE_RATE)[0]) for w in waves]
    finally:
        del model
        _free()


def score_tts(
    items: list[dict[str, Any]], device: str | torch.device | None = None
) -> list[dict[str, Any]]:
    """Perceptual scores of generated speech, one dict per item (``audio``: a waveform,
    ``sampling_rate``, ``text``: what it should say, ``language``: ISO 639-1):
    ``transcript`` (Whisper-large-v3), ``error_rate`` against ``text``, ``embedding``
    (WavLM-base-plus-SV, unit length), ``mos`` (UTMOS22, ``None`` when not cached) and
    ``audio_s``. The models are loaded one after the other and freed."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _free()  # what the generating runs left cached: the scoring models need the room
    waves = [resample(torch.as_tensor(it["audio"]), int(it["sampling_rate"])) for it in items]
    languages = [str(it.get("language") or "en") for it in items]
    texts = _transcribe(waves, languages, dev)
    embeddings = _embed(waves, dev)
    mos = _mos(waves, dev)
    return [
        {
            "transcript": text,
            "error_rate": round(error_rate(str(it["text"]), text, language=lang), 4),
            "embedding": emb,
            "mos": None if mos is None else round(mos[i], 4),
            "audio_s": round(wave.numel() / SCORE_RATE, 3),
        }
        for i, (it, text, lang, emb, wave) in enumerate(
            zip(items, texts, languages, embeddings, waves, strict=True)
        )
    ]


def speaker_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of two speaker embeddings."""
    x, y = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    denom = float(x.norm() * y.norm())
    return float(x @ y) / denom if denom > 0 else 0.0


def compare_tts(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    max_error_increase: float = MAX_ERROR_INCREASE,
    min_speaker_similarity: float = MIN_SPEAKER_SIMILARITY,
    min_speaker_similarity_worst: float = MIN_SPEAKER_SIMILARITY_WORST,
    max_mos_drop: float = MAX_MOS_DROP,
) -> Comparison:
    """Paired comparison of :func:`score_tts` results, sample *i* of the candidate with
    sample *i* of eager (the same text and seed): the mean error rate may rise by at most
    ``max_error_increase``, the speaker similarity (mean, worst sample) stays at or above
    its limits and the mean MOS may drop by at most ``max_mos_drop`` (when both have it)."""
    if len(reference) != len(candidate) or not reference:
        return Comparison(
            False,
            {"samples": len(candidate), "reference_samples": len(reference)},
            f"{len(candidate)} candidate samples for {len(reference)} eager samples",
        )
    ref_err = statistics.fmean(float(r["error_rate"]) for r in reference)
    new_err = statistics.fmean(float(c["error_rate"]) for c in candidate)
    sims = [
        speaker_similarity(r["embedding"], c["embedding"])
        for r, c in zip(reference, candidate, strict=True)
    ]
    worst = min(range(len(sims)), key=sims.__getitem__)
    mean_sim = statistics.fmean(sims)
    metrics: dict[str, float | int | str] = {
        "samples": len(candidate),
        "error_rate": round(new_err, 4),
        "eager_error_rate": round(ref_err, 4),
        "error_increase": round(new_err - ref_err, 4),
        "speaker_similarity": round(mean_sim, 4),
        "speaker_similarity_worst": round(sims[worst], 4),
        "worst_sample": worst,
    }
    problems = []
    if new_err - ref_err > max_error_increase:
        problems.append(
            f"error rate {new_err:.3f} vs eager {ref_err:.3f} (+{new_err - ref_err:.3f}, "
            f"allowed +{max_error_increase:g})"
        )
    if mean_sim < min_speaker_similarity:
        problems.append(f"speaker similarity {mean_sim:.3f} < {min_speaker_similarity:g}")
    if sims[worst] < min_speaker_similarity_worst:
        problems.append(
            f"speaker similarity of sample {worst} {sims[worst]:.3f} < "
            f"{min_speaker_similarity_worst:g}"
        )
    ref_mos = [r["mos"] for r in reference if r.get("mos") is not None]
    new_mos = [c["mos"] for c in candidate if c.get("mos") is not None]
    if len(ref_mos) == len(new_mos) == len(reference):
        ref_mean, new_mean = statistics.fmean(ref_mos), statistics.fmean(new_mos)
        drop = ref_mean - new_mean
        metrics.update(
            mos=round(new_mean, 3), eager_mos=round(ref_mean, 3), mos_drop=round(drop, 3)
        )
        if drop > max_mos_drop:
            problems.append(
                f"MOS {new_mean:.3f} vs eager {ref_mean:.3f} (-{drop:.3f}, allowed "
                f"-{max_mos_drop:g})"
            )
    else:
        metrics["mos"] = "unavailable"  # no UTMOS22 in the torch hub cache (mos_location)
    return Comparison(not problems, metrics, "; ".join(problems))


# ------------------------------------------------------------------ LLM: teacher forcing


def forced_logprobs(model: Any, prompt: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Log-probabilities ``[N, vocab]`` (float32) of the next token at every position of a
    continuation ``tokens`` (``N``) of ``prompt``, teacher forced: one forward of the model
    over prompt + continuation (``model(input_ids)``, the patched model of the process), so
    position *i* sees the continuation's first *i* tokens."""
    device = next(model.parameters()).device
    ids = torch.cat([prompt.flatten().to(device), tokens.flatten().to(device)])[None]
    n = int(tokens.numel())
    try:  # only the logits that are needed (most transformers models take it)
        out = model(input_ids=ids, use_cache=False, logits_to_keep=n + 1)
        logits = out.logits[0, :n]
    except TypeError:
        logits = model(input_ids=ids, use_cache=False).logits[0, -n - 1 : -1]
    return logits.float().log_softmax(-1)


@torch.inference_mode()
def score_llm(
    model: Any, items: list[dict[str, Any]], *, topk: int = LLM_TOPK
) -> list[dict[str, Any]]:
    """Teacher-forced scores of generated continuations, one dict per item (``prompt``,
    ``tokens``: 1-D token ids, ``reference``: the baseline's scores of the same sample or
    None).

    Every item: ``tokens``, ``nll`` (mean negative log-likelihood of its own continuation
    under ``model``) and the ``topk`` most likely tokens of every position (``topk_ids``,
    ``topk_logprobs``): for the baseline, the reference distributions. With a ``reference``
    the model is also teacher forced on the *reference* continuation: ``kl`` (mean over
    positions of KL(eager || model), eager's distribution as its top-k tokens plus the rest
    of its mass, a lower bound of the full KL), ``kl_max`` (worst position), ``top1``
    (positions whose most likely token is eager's) and ``reference_nll``."""
    out = []
    for it in items:
        tokens = torch.as_tensor(it["tokens"]).flatten()
        logp = forced_logprobs(model, torch.as_tensor(it["prompt"]), tokens)
        dev = logp.device
        own = logp.gather(1, tokens.to(dev)[:, None])[:, 0]
        top = logp.topk(min(topk, logp.shape[-1]), dim=-1)
        scores: dict[str, Any] = {
            "tokens": tokens.tolist(),
            "nll": round(float(-own.mean()), 5),
            "topk_ids": top.indices.cpu(),
            "topk_logprobs": top.values.cpu(),
        }
        ref = it.get("reference")
        if ref is not None:
            ref_tokens = torch.as_tensor(ref["tokens"]).flatten()
            logq = forced_logprobs(model, torch.as_tensor(it["prompt"]), ref_tokens)
            ids = torch.as_tensor(ref["topk_ids"]).to(logq.device)
            p_logp = torch.as_tensor(ref["topk_logprobs"]).float().to(logq.device)
            q_logp = logq.gather(1, ids)
            p, q = p_logp.exp(), q_logp.exp()
            p_rest = (1.0 - p.sum(1)).clamp_min(1e-12)
            q_rest = (1.0 - q.sum(1)).clamp_min(1e-12)
            kl = (p * (p_logp - q_logp)).sum(1) + p_rest * (p_rest.log() - q_rest.log())
            kl = kl.clamp_min(0.0)
            agree = q_logp[:, 0] >= logq.max(-1).values  # eager's top token is the model's (ties)
            scores.update(
                kl=round(float(kl.mean()), 6),
                kl_max=round(float(kl.max()), 5),
                top1=round(float(agree.float().mean()), 4),
                reference_nll=round(
                    float(-logq.gather(1, ref_tokens.to(logq.device)[:, None]).mean()), 5
                ),
            )
        out.append(scores)
    return out


def compare_llm(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    max_kl: float = LLM_MAX_KL,
    max_kl_worst: float = LLM_MAX_KL_WORST,
    min_top1: float = LLM_MIN_TOP1,
    max_nll_increase: float = LLM_MAX_NLL_INCREASE,
) -> Comparison:
    """Paired comparison of :func:`score_llm` results (the candidate's with a
    ``reference``): teacher forced on eager's continuations, the mean KL over the samples
    and the worst sample's stay within ``max_kl`` / ``max_kl_worst`` and the share of
    positions whose most likely token is eager's at or above ``min_top1``; free running,
    the negative log-likelihood of the candidate's own continuations (under the candidate)
    may exceed eager's of its own by at most ``max_nll_increase`` nats per token on average
    (a broken decode loop writes text no model finds likely)."""
    if len(reference) != len(candidate) or not reference:
        return Comparison(
            False,
            {"samples": len(candidate), "reference_samples": len(reference)},
            f"{len(candidate)} candidate samples for {len(reference)} eager samples",
        )
    if any(c.get("kl") is None for c in candidate):
        return Comparison(False, {"samples": len(candidate)}, "not teacher forced on eager's")
    kls = [float(c["kl"]) for c in candidate]
    top1 = [float(c["top1"]) for c in candidate]
    nll = [float(c["nll"]) - float(r["nll"]) for r, c in zip(reference, candidate, strict=True)]
    worst = max(range(len(kls)), key=kls.__getitem__)
    kl, agree, increase = statistics.fmean(kls), statistics.fmean(top1), statistics.fmean(nll)
    metrics: dict[str, float | int | str] = {
        "samples": len(candidate),
        "kl": round(kl, 6),
        "kl_worst": round(kls[worst], 6),
        "worst_sample": worst,
        "top1": round(agree, 4),
        "top1_worst": round(min(top1), 4),
        "nll_increase": round(increase, 4),
        "nll_increase_worst": round(max(nll), 4),
    }
    problems = []
    if kl > max_kl:
        problems.append(f"mean KL {kl:.4f} > {max_kl:g}")
    if kls[worst] > max_kl_worst:
        problems.append(f"KL of sample {worst} {kls[worst]:.4f} > {max_kl_worst:g}")
    if agree < min_top1:
        problems.append(f"top-1 agreement {agree:.3f} < {min_top1:g}")
    if increase > max_nll_increase:
        problems.append(
            f"own continuations {increase:+.3f} nats/token less likely than eager's "
            f"(allowed +{max_nll_increase:g})"
        )
    return Comparison(not problems, metrics, "; ".join(problems))


# ------------------------------------------------------------------ generate + score


def generate(
    workload: Workload, samples: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    """Every sample (option overrides) run once, free running and untimed:
    ``[{"options", "output"}]`` and the seconds it took."""
    from kernel_agent.workloads.holdout import run_variant

    start = time.perf_counter()
    generated = []
    for options in samples:
        _, output, _ = run_variant(workload, options)
        generated.append({"options": options, "output": output})
    return generated, time.perf_counter() - start


def score(
    workload: Workload, generated: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    """:meth:`Workload.perceptual_quality` of generated samples and the seconds it took."""
    start = time.perf_counter()
    scores = workload.perceptual_quality(generated)
    if len(scores) != len(generated):
        raise ValueError(
            f"perceptual_quality returned {len(scores)} scores for {len(generated)} samples"
        )
    return scores, time.perf_counter() - start


def _brief(options: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    """One sample for ``baseline.json`` / the e2e metrics: scalar scores only."""
    what = " ".join(str(options[k]) for k in ("sample", "language", "seed") if k in options)
    return {"sample": what, **{k: v for k, v in scores.items() if isinstance(v, int | float | str)}}


def _means(scores: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        [k for k, v in scores[0].items() if isinstance(v, int | float) and not isinstance(v, bool)]
        if scores
        else []
    )
    return {
        k: round(statistics.fmean(float(s[k]) for s in scores), 4)
        for k in keys
        if all(isinstance(s.get(k), int | float) for s in scores)
    }


# ------------------------------------------------------------------ analyze


def record_baseline(workload: Workload) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The baseline's perceptual samples with their scores (``None`` when the workload
    declares none) and ``baseline.json`` ``perceptual``. Raises when a run fails."""
    samples = workload.perceptual_samples()
    if not samples:
        return None, {"status": "none", "reason": "the workload declares no perceptual samples"}
    generated, gen_s = generate(workload, samples)
    scores, score_s = score(workload, generated)
    data = {
        "metric": workload.metric,  # the code path the samples ran through (objective.py)
        "samples": [{**g, "scores": s} for g, s in zip(generated, scores, strict=True)],
    }
    info = {
        "status": "ok",
        "metric": workload.metric,
        "samples": len(samples),
        "generate_s": round(gen_s, 1),
        "score_s": round(score_s, 1),
        "mean": _means(scores),
        "per_sample": [_brief(g["options"], s) for g, s in zip(generated, scores, strict=True)],
    }
    return data, info


def save_baseline(workload: Workload, path: Path, *, quality: str) -> dict[str, Any]:
    """:func:`record_baseline` in ``near-lossless`` and ``relaxed`` mode, saved to ``path``
    (none: no file), the mode recorded (``quality``). Failures are recorded, not raised:
    ``e2e`` then keeps the exact checks."""
    path.unlink(missing_ok=True)  # never leave a stale perceptual baseline behind
    mode = mode_of(quality)
    if mode not in GATED:
        return {"status": "none", "reason": f"--quality {mode}"}
    try:
        data, info = record_baseline(workload)
    except Exception:
        return {"status": "error", "quality": mode, "error": traceback.format_exc()[-2000:]}
    if data is not None:
        torch.save(data, path)
    return {**info, "quality": mode}


# ------------------------------------------------------------------ e2e


def check(
    workload: Workload,
    reference: dict[str, Any],
    *,
    before_scoring: Callable[[], None] | None = None,
    quality: str = NEAR_LOSSLESS,
) -> dict[str, Any]:
    """``metrics.perceptual`` of an ``e2e`` run: the stored baseline samples
    (``reference``) generated by the candidate, scored and compared paired with the
    baseline's scores: ``{"passed", "reason", ...}``, with the thresholds of quality mode
    ``quality`` (:func:`floor_options`: ``relaxed`` loosens them). Both ran through
    :meth:`Workload.run` under the run's metric (``metric=ttfa``: the streaming path), so
    the gate judges the audio of the code path the objective times; a baseline recorded
    under another metric fails (re-run ``analyze``). ``before_scoring`` runs between the
    model's last run (the samples generated) and the scoring models: the paired A/B frees
    A's state there."""
    samples = reference["samples"]
    recorded = str(reference.get("metric") or workload.metric)
    if recorded != workload.metric:
        return {
            "passed": False,
            "reason": f"the perceptual baseline was generated with metric={recorded}, this "
            f"run times metric={workload.metric}: re-run analyze",
            "metric": workload.metric,
        }
    try:
        generated, gen_s = generate(workload, [s["options"] for s in samples])
        for g, s in zip(generated, samples, strict=True):  # paired scoring (LLM: forced on it)
            g["reference"] = s["scores"]
        if before_scoring is not None:
            before_scoring()
        scores, score_s = score(workload, generated)
        with workload.with_options(floor_options(workload, quality)):  # the mode's thresholds
            cmp = workload.compare_perceptual([s["scores"] for s in samples], scores)
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"the perceptual gate failed: {type(exc).__name__}: {exc}"[:500],
            "error": traceback.format_exc()[-3000:],
        }
    return {
        "passed": cmp.passed,
        "reason": cmp.reason,
        **cmp.metrics,
        "quality": mode_of(quality),
        "metric": workload.metric,
        "generate_s": round(gen_s, 1),
        "score_s": round(score_s, 1),
        "per_sample": [_brief(g["options"], s) for g, s in zip(generated, scores, strict=True)],
    }


@contextlib.contextmanager
def judging(workload: Workload, quality: str, reference: Any) -> Iterator[bool]:
    """Around the exact checks of an ``e2e`` verdict: yields whether the perceptual gate
    decides (``near-lossless`` or ``relaxed`` with a perceptual baseline), with the
    workload's options of the mode (the sanity floor, :func:`floor_options`) applied while
    it does."""
    gate = gated(quality) and reference is not None
    with workload.with_options(floor_options(workload, quality) if gate else None):
        yield gate


def skipped(quality: str, baseline: dict[str, Any]) -> dict[str, Any] | None:
    """``metrics.perceptual`` when the gate cannot run (``None`` in ``exact`` mode)."""
    if not gated(quality):
        return None
    info = baseline.get("perceptual") or {}
    why = {
        "none": info.get("reason") or "no perceptual samples",
        "error": "its baseline run failed",
    }.get(str(info.get("status")), f"analyze ran without --quality {mode_of(quality)}")
    return {
        "passed": True,
        "reason": "",
        "skipped": f"no perceptual baseline ({why}): the exact checks apply",
    }


# ------------------------------------------------------------------ reporting


def messages(baseline: dict[str, Any]) -> list[str]:
    """``analyze`` log lines about the perceptual baseline."""
    info = baseline.get("perceptual") or {}
    status = info.get("status")
    if status == "error":
        return [
            "WARNING: the perceptual baseline run failed; e2e evaluations keep the exact "
            f"checks ({str(info.get('error', ''))[-300:]})"
        ]
    if status != "ok":
        return []
    return [
        f"analyze: perceptual baseline: {info.get('samples')} samples "
        f"({_fmt(info.get('mean'))}; generated in {info.get('generate_s')} s, scored in "
        f"{info.get('score_s')} s)"
    ]


def summary_lines(baseline: dict[str, Any]) -> list[str]:
    """Bullets for the ``End-to-end quality check`` section of ``profile/summary.md``."""
    info = baseline.get("perceptual") or {}
    if info.get("status") != "ok":
        return []
    mode = str(info.get("quality") or NEAR_LOSSLESS)  # recorded since #175
    within = (
        "stays within about twice the noise of eager (small measured drops pass)"
        if mode == RELAXED
        else "stays within the noise of eager"
    )
    return [
        f"* **quality mode: {mode}.** Numerics-changing optimisations (FP8 weights, "
        f"...) are allowed when the perceptual quality {within}: "
        f"every candidate also generates {info.get('samples')} held-out samples (free "
        "running, untimed), scored and compared paired with eager's (TTS: transcript error "
        "rate, speaker similarity, MOS; LLM: teacher forced on eager's continuation, KL and "
        "top-1 agreement per token, and the likelihood of its own continuation; eager: "
        f"{_fmt(info.get('mean'))}). Teacher forcing, the held-out input and the stop check "
        "stay, with looser thresholds (a sanity floor that catches broken kernels).",
    ]


def summary_text(result: dict[str, Any]) -> str:
    """One phrase for reports: the ``metrics.perceptual`` of an ``e2e`` run."""
    if result.get("skipped"):
        return f"perceptual gate skipped ({result['skipped']})"
    keys = (
        *("error_rate", "eager_error_rate", "speaker_similarity", "mos", "eager_mos"),
        *("kl", "kl_worst", "top1", "nll_increase"),  # LLM: teacher forced (score_llm)
    )
    found = {k: result[k] for k in keys if k in result}
    if not result.get("passed"):
        return f"perceptual gate FAILED: {result.get('reason')} ({_fmt(found)})"
    return f"perceptual gate passed ({_fmt(found)})"


def _fmt(values: Any) -> str:
    if not isinstance(values, dict):
        return str(values)
    return ", ".join(f"{k}={v}" for k, v in values.items())
