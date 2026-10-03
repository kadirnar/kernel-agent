"""Speech recognition (Whisper-style seq2seq and CTC) via ``transformers``."""

from __future__ import annotations

import math
import wave
from typing import Any

import numpy as np
import torch
from torch import nn

from kernel_agent.hub import Modality
from kernel_agent.workloads.base import Comparison, Workload, compare_tokens


def load_wav(path: str, target_sr: int) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError("only 16-bit PCM WAV is supported")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        t_old = np.arange(audio.size) / sr
        t_new = np.arange(int(audio.size * target_sr / sr)) / target_sr
        audio = np.interp(t_new, t_old, audio).astype(np.float32)
    return audio


def synthetic_speechlike(seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    """Deterministic voiced-sound-like signal (harmonics + formant-ish AM + noise).

    Only used when no ``audio`` option is given; it exercises the same kernels
    as real speech (the decoder is forced to a fixed token count)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 30 * np.sin(2 * math.pi * 0.5 * t)
    phase = 2 * math.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = 0.5 * (1 + np.sin(2 * math.pi * 3 * t))
    sig = sig * env + 0.02 * rng.standard_normal(t.size)
    return (0.3 * sig / np.abs(sig).max()).astype(np.float32)


class STTWorkload(Workload):
    modality = Modality.STT
    defaults = {
        "audio": None,
        "audio_seconds": 20.0,
        "new_tokens": 96,
        "batch_size": 1,
        "min_prefix": 16,
        "min_cosine": 0.99,
    }

    def load(self) -> None:
        import transformers as tf

        kwargs: dict[str, Any] = {
            "revision": self.spec.revision,
            "trust_remote_code": self.spec.trust_remote_code,
        }
        self.processor = tf.AutoProcessor.from_pretrained(self.spec.repo_id, **kwargs)
        try:
            self.model = tf.AutoModelForSpeechSeq2Seq.from_pretrained(
                self.spec.repo_id, dtype=self.dtype, **kwargs
            )
            self.ctc = False
        except ValueError:
            self.model = tf.AutoModelForCTC.from_pretrained(
                self.spec.repo_id, dtype=self.dtype, **kwargs
            )
            self.ctc = True
        self.model.to(self.device).eval()
        fe = getattr(self.processor, "feature_extractor", self.processor)
        self.sampling_rate = int(getattr(fe, "sampling_rate", 16000))

    def roots(self) -> dict[str, nn.Module]:
        return {"model": self.model}

    def make_inputs(self) -> dict[str, torch.Tensor]:
        path = self.options.get("audio")
        if path:
            audio = load_wav(str(path), self.sampling_rate)
        else:
            audio = synthetic_speechlike(float(self.options["audio_seconds"]), self.sampling_rate)
        batch = [audio] * int(self.options["batch_size"])
        feats = self.processor(batch, sampling_rate=self.sampling_rate, return_tensors="pt")
        return {
            k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in feats.items()
        }

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.ctc:
            logits = self.model(**inputs).logits
            return {"tokens": logits.argmax(-1).cpu(), "first_logits": logits[:, :8].float().cpu()}
        n = int(self.options["new_tokens"])
        out = self.model.generate(
            **inputs,
            max_new_tokens=n,
            min_new_tokens=n,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        return {"tokens": out.sequences.cpu(), "first_logits": out.scores[0].float().cpu()}

    def compare(self, reference: Any, candidate: Any) -> Comparison:
        return compare_tokens(
            reference["tokens"],
            candidate["tokens"],
            reference.get("first_logits"),
            candidate.get("first_logits"),
            min_prefix=int(self.options["min_prefix"]),
            min_cosine=float(self.options["min_cosine"]),
        )
