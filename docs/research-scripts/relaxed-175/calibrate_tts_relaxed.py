"""The TTS gate and sanity floor at near-lossless and relaxed thresholds (#175), VoxCPM2.

Each variant generates the 8 perceptual samples (``VoxCPMWorkload.perceptual_samples``),
scored by Whisper-large-v3 / WavLM-SV / UTMOS22 and compared paired with eager's
(``perceptual.compare_tts``) at the near-lossless thresholds and the relaxed ones
(``perceptual.RELAXED_GATE``); teacher forcing on the 60-patch main input gives the
sanity floor's mean / min step cosine (near-lossless 0.95 / 0.2, relaxed 0.90 / 0.2).
FP8 = every ``nn.Linear`` of both LMs and the LocDiT quantised and back to bf16 (343).

    python calibrate_tts_relaxed.py out.json
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable

import torch
from torch import nn

from kernel_agent import toolchain
from kernel_agent.kernels import quant
from kernel_agent.workloads import create_workload, perceptual
from kernel_agent.workloads.base import WorkloadSpec

toolchain.setup()
wl = create_workload(WorkloadSpec(repo_id="openbmb/VoxCPM2", modality="tts", family="voxcpm"))
wl.load()
reference, info = perceptual.record_baseline(wl)
assert reference is not None
ref_scores = [s["scores"] for s in reference["samples"]]
inputs = wl.make_inputs()
with torch.inference_mode():
    ref_out = wl.run(inputs)
print("eager", json.dumps(info["mean"]), flush=True)
model = wl.model
linears = {
    id(m): m
    for root in (model.base_lm, model.residual_lm, model.feat_decoder)
    for m in root.modules()
    if isinstance(m, nn.Linear)
}
saved = {k: m.weight.detach().cpu().clone() for k, m in linears.items()}  # GPU memory: scorers
norms = [m for m in model.modules() if type(m).__name__ == "MiniCPMRMSNorm"]
eps = {id(m): m.variance_epsilon for m in norms}
GATE = ("max_error_increase", "min_speaker_similarity", "min_speaker_similarity_worst")
RELAXED = {k: perceptual.RELAXED_GATE[k] for k in (*GATE, "max_mos_drop")}
FLOOR = {"near-lossless": (0.95, 0.2), "relaxed": (0.90, 0.2)}


def restore() -> None:
    with torch.no_grad():
        for k, m in linears.items():
            m.weight.copy_(saved[k])
    for m in norms:
        m.variance_epsilon = eps[id(m)]


def weights(fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    with torch.no_grad():
        for m in linears.values():
            m.weight.copy_(fn(m.weight).to(m.weight.dtype))


def fp8(w: torch.Tensor) -> torch.Tensor:
    return quant.dequantize_fp8(*quant.quantize_fp8(w), w.dtype)


def fp4(fmt: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda w: quant.dequantize_fp4(*quant.quantize_fp4(w, fmt), w.dtype)


def int4(w: torch.Tensor) -> torch.Tensor:
    s = w.float().abs().amax() / 7
    return (w.float() / s).round().clamp(-8, 7) * s


def broken_norm() -> None:
    for m in norms:
        m.variance_epsilon = 1e-2


def evaluate(label: str, setup: Callable[[], None]) -> dict:
    t = time.time()
    setup()
    generated, _ = perceptual.generate(wl, [s["options"] for s in reference["samples"]])
    with torch.inference_mode():
        forced = wl.run_teacher_forced(inputs, ref_out)
    steps = wl.compare_teacher_forced(ref_out, forced).metrics
    restore()
    scores, _ = perceptual.score(wl, generated)
    near = perceptual.compare_tts(ref_scores, scores)
    relaxed = perceptual.compare_tts(ref_scores, scores, **RELAXED)
    mean, low = float(steps["mean_step_cosine"]), float(steps["min_step_cosine"])
    floor = {q: mean >= m and low >= n for q, (m, n) in FLOOR.items()}
    out = {
        "variant": label,
        **{k: near.metrics.get(k) for k in ("error_increase", "speaker_similarity")},
        "speaker_similarity_worst": near.metrics.get("speaker_similarity_worst"),
        "mos_drop": near.metrics.get("mos_drop"),
        "mean_step_cosine": round(mean, 4),
        "min_step_cosine": round(low, 4),
        "near_lossless": near.passed and floor["near-lossless"],
        "relaxed": relaxed.passed and floor["relaxed"],
        "near_lossless_reason": near.reason or ("" if floor["near-lossless"] else "floor"),
        "relaxed_reason": relaxed.reason or ("" if floor["relaxed"] else "floor"),
        "s": round(time.time() - t, 1),
    }
    print(json.dumps(out), flush=True)
    return out


results = [
    evaluate(label, fn)
    for label, fn in [
        ("FP8 weights (343 nn.Linear)", lambda: weights(fp8)),
        ("NVFP4 weights (343)", lambda: weights(fp4("nvfp4"))),
        ("MXFP4 weights (343)", lambda: weights(fp4("mxfp4"))),
        ("RMSNorm eps 1e-2", broken_norm),
        ("int4 per tensor (343)", lambda: weights(int4)),
    ]
]
json.dump({"eager": info["mean"], "results": results}, open(sys.argv[1], "w"), indent=1)
