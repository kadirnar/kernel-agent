"""Use a kernel-agent result with VoxCPM2's own API and compare it with eager PyTorch.

    python examples/voxcpm2_optimized.py runs/openbmb--VoxCPM2/<run>/optimized --out samples/

Generates the same texts with the eager model and with the optimised one
(kernels + model-level transforms from the run's ``optimized/`` package) through
``VoxCPM2Model.generate`` at natural length (the model decides when to stop),
reports latency and real-time factor, writes the WAV files, and transcribes
them with Whisper to compare the word error rate against the input text.
Generation is autoregressive with sampling noise, so the two models produce
different (equally valid) audio for the same seed; WER and duration are the
quality signal here, the strict per-step check is kernel-agent's teacher forcing.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch

TEXTS = [
    "Kernel level optimisation makes speech synthesis fast enough for real time conversation.",
    "The quick brown fox jumps over the lazy dog while the orchestra tunes its instruments.",
    "Please remember to bring your passport, a warm jacket and the charger for your laptop.",
]
SEED = 0


def load_model() -> torch.nn.Module:
    from huggingface_hub import snapshot_download
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    return VoxCPM2Model.from_local(snapshot_download("openbmb/VoxCPM2"), optimize=False)


def apply_optimized(model: torch.nn.Module, optimized: Path) -> dict[str, int]:
    """Kernels, then the model-level transforms, through the run's exported apply.py."""
    sys.path.insert(0, str(optimized))
    from apply import apply_kernels, apply_transforms  # type: ignore[import-not-found]

    counts = apply_kernels(model)
    apply_transforms(model)
    return counts


def generate(model: torch.nn.Module, text: str) -> tuple[torch.Tensor, float]:
    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    start = time.perf_counter()
    wav = model.generate(target_text=text, inference_timesteps=10, cfg_value=2.0)
    torch.cuda.synchronize()
    return wav.float().flatten().cpu(), time.perf_counter() - start


def run(name: str, model: torch.nn.Module, out: Path) -> list[dict[str, object]]:
    for text in TEXTS[:1]:  # warm-up: lazy CUDA graphs / compilation happen here
        generate(model, text)
    rows = []
    for i, text in enumerate(TEXTS):
        wav, seconds = generate(model, text)
        audio_s = wav.numel() / 48000
        path = out / f"{name}_{i}.wav"
        import soundfile as sf

        sf.write(path, wav.numpy(), 48000)
        rows.append(
            {"model": name, "text": i, "seconds": seconds, "audio_s": audio_s, "wav": str(path)}
        )
        print(
            f"{name:9s} text {i}: {seconds:6.2f} s for {audio_s:5.2f} s of audio "
            f"(RTF {seconds / audio_s:.3f})",
            flush=True,
        )
    return rows


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def transcribe(rows: list[dict[str, object]]) -> None:
    import librosa
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        dtype=torch.float16,
        device="cuda",
    )
    for row in rows:
        audio, _ = librosa.load(str(row["wav"]), sr=16000)
        text = asr(audio, generate_kwargs={"language": "en"})["text"]
        row["transcript"] = text.strip()
        row["wer"] = wer(TEXTS[int(row["text"])], text)  # type: ignore[call-overload]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("optimized", type=Path, help="the run's optimized/ directory")
    parser.add_argument("--out", type=Path, default=Path("voxcpm2_samples"))
    ns = parser.parse_args()
    ns.out.mkdir(parents=True, exist_ok=True)

    rows = run("eager", load_model(), ns.out)
    gc.collect()
    torch.cuda.empty_cache()

    model = load_model()
    print("optimised:", apply_optimized(model, ns.optimized.resolve()), flush=True)
    rows += run("optimised", model, ns.out)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    transcribe(rows)
    for row in rows:
        print(f'{row["model"]:9s} text {row["text"]}: WER {row["wer"]:.2f}  "{row["transcript"]}"')
    totals = {
        name: sum(float(r["seconds"]) for r in rows if r["model"] == name)  # type: ignore[arg-type]
        for name in ("eager", "optimised")
    }
    print(
        f"total: eager {totals['eager']:.2f} s, optimised {totals['optimised']:.2f} s, "
        f"speedup {totals['eager'] / totals['optimised']:.2f}x"
    )
    (ns.out / "results.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
