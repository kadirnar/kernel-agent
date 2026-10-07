"""``--quality near-lossless``: the perceptual gate for numerics-changing optimisations.

``exact`` (the default) accepts changes whose numerics stay within rounding noise of the
baseline: module tolerances (:mod:`kernel_agent.kernels.compare`) and, end to end, the
workload's own comparison or teacher forcing (:mod:`kernel_agent.workloads.quality`).
Once a model runs near the memory-bandwidth floor of its bf16 weights, the next big gains
(FP8 weights, ...) change numerics by design. ``near-lossless`` accepts them when the
*perceptual* quality stays within the noise of eager:

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
* **Sanity floor.** Teacher forcing, the held-out input and the stop check stay, with the
  workload's looser :attr:`~kernel_agent.workloads.base.Workload.near_lossless_options`
  (thresholds that FP8-weight variants pass and broken kernels still fail; the stop check
  accepts ±1 step at a near-tie). Options the user set (``-o``) win.
* **Module tolerance tier.** Targets whose spec allows reduced precision
  (``"precision": "reduced"``) are captured with the ``near-lossless`` tier of
  :mod:`kernel_agent.kernels.compare` (whole-tensor cosine and relative L2 error).

Without a perceptual baseline (the workload declares no samples, or ``analyze`` ran in
``exact`` mode or its perceptual run failed) a near-lossless run keeps the exact checks.
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
MODES = (EXACT, NEAR_LOSSLESS)

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


def mode_of(value: Any) -> str:
    """A validated quality mode (``None`` / empty: ``exact``)."""
    mode = str(value or EXACT)
    if mode not in MODES:
        raise ValueError(f"unknown quality mode {mode!r} (one of {', '.join(MODES)})")
    return mode


def floor_options(workload: Workload) -> dict[str, Any]:
    """The workload's near-lossless option overrides, without the ones the user set."""
    return {
        k: v for k, v in workload.near_lossless_options.items() if k not in workload.spec.options
    }


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
    what = " ".join(str(options[k]) for k in ("language", "seed") if k in options)
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
    """:func:`record_baseline` in ``near-lossless`` mode, saved to ``path`` (none: no file).
    Failures are recorded, not raised: ``e2e`` then keeps the exact checks."""
    path.unlink(missing_ok=True)  # never leave a stale perceptual baseline behind
    if mode_of(quality) != NEAR_LOSSLESS:
        return {"status": "none", "reason": f"--quality {mode_of(quality)}"}
    try:
        data, info = record_baseline(workload)
    except Exception:
        return {"status": "error", "error": traceback.format_exc()[-2000:]}
    if data is not None:
        torch.save(data, path)
    return info


# ------------------------------------------------------------------ e2e


def check(
    workload: Workload,
    reference: dict[str, Any],
    *,
    before_scoring: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """``metrics.perceptual`` of an ``e2e`` run: the stored baseline samples
    (``reference``) generated by the candidate, scored and compared paired with the
    baseline's scores: ``{"passed", "reason", ...}``. Both ran through :meth:`Workload.run`
    under the run's metric (``metric=ttfa``: the streaming path), so the gate judges the
    audio of the code path the objective times; a baseline recorded under another metric
    fails (re-run ``analyze``). ``before_scoring`` runs between the model's last run (the
    samples generated) and the scoring models: the paired A/B frees A's state there."""
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
        if before_scoring is not None:
            before_scoring()
        scores, score_s = score(workload, generated)
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
        "metric": workload.metric,
        "generate_s": round(gen_s, 1),
        "score_s": round(score_s, 1),
        "per_sample": [_brief(g["options"], s) for g, s in zip(generated, scores, strict=True)],
    }


@contextlib.contextmanager
def judging(workload: Workload, quality: str, reference: Any) -> Iterator[bool]:
    """Around the exact checks of an ``e2e`` verdict: yields whether the perceptual gate
    decides (``near-lossless`` with a perceptual baseline), with the workload's
    near-lossless options (the sanity floor) applied while it does."""
    gate = mode_of(quality) == NEAR_LOSSLESS and reference is not None
    with workload.with_options(floor_options(workload) if gate else None):
        yield gate


def skipped(quality: str, baseline: dict[str, Any]) -> dict[str, Any] | None:
    """``metrics.perceptual`` when the gate cannot run (``None`` in ``exact`` mode)."""
    if mode_of(quality) != NEAR_LOSSLESS:
        return None
    info = baseline.get("perceptual") or {}
    why = {
        "none": info.get("reason") or "no perceptual samples",
        "error": "its baseline run failed",
    }.get(str(info.get("status")), "analyze ran without --quality near-lossless")
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
            "WARNING: the perceptual baseline run failed; near-lossless e2e evaluations keep "
            f"the exact checks ({str(info.get('error', ''))[-300:]})"
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
    return [
        "* **quality mode: near-lossless.** Numerics-changing optimisations (FP8 weights, "
        "...) are allowed when the perceptual quality stays within the noise of eager: "
        f"every candidate also generates {info.get('samples')} held-out samples (free "
        "running, natural length, untimed), scored and compared paired with eager's "
        f"(eager: {_fmt(info.get('mean'))}). Teacher forcing, the held-out input and the "
        "stop check stay, with looser thresholds (a sanity floor that catches broken "
        "kernels).",
    ]


def summary_text(result: dict[str, Any]) -> str:
    """One phrase for reports: the ``metrics.perceptual`` of an ``e2e`` run."""
    if result.get("skipped"):
        return f"perceptual gate skipped ({result['skipped']})"
    keys = ("error_rate", "eager_error_rate", "speaker_similarity", "mos", "eager_mos")
    found = {k: result[k] for k in keys if k in result}
    if not result.get("passed"):
        return f"perceptual gate FAILED: {result.get('reason')} ({_fmt(found)})"
    return f"perceptual gate passed ({_fmt(found)})"


def _fmt(values: Any) -> str:
    if not isinstance(values, dict):
        return str(values)
    return ", ".join(f"{k}={v}" for k, v in values.items())
