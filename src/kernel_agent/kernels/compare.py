"""Structural comparison of nested module outputs (tensors, tuples, dicts, caches).

A floating-point tensor matches its reference when

* it is a plain ``torch.Tensor`` (no subclass: a ``__torch_function__`` override
  could compute lazily, after the timer stopped) of the same shape, dtype and
  device type;
* NaN, +inf and -inf sit at exactly the same positions;
* at most :data:`MAX_MISMATCH` of the finite elements are outside ``atol + rtol ·
  |ref|`` (:data:`TOLERANCES`), and none of them by more than :data:`MAX_OUTLIER`
  times that tolerance;
* tensors with more than :data:`GLOBAL_MIN_NUMEL` elements whose reference has
  signal above the absolute tolerance (``‖ref‖ > atol · √n``) also match as a
  whole: cosine similarity and relative L2 error ``‖new − ref‖ / ‖ref‖`` within
  :data:`GLOBAL_TOLERANCES`.

Integer and boolean tensors must match exactly. In the ``near-lossless`` tier
(:data:`TIERS`: ``--quality near-lossless`` and a target whose spec allows reduced
precision) whole-tensor bounds replace the per-element tolerances; FP4 weights
(``fp4_weights``) get the wider bounds of the ``near-lossless-fp4`` tier. ``--quality
relaxed`` (the default of new runs, :data:`DEFAULT_QUALITY`) has the same structure with
about twice the error budgets: the ``relaxed``, ``relaxed-fp4`` and ``relaxed-kv`` tiers
(:data:`QUALITY_TIERS`). On redrawn
inputs (``perturbed``) these tiers use their own bounds (:data:`PERTURBED_BOUNDS`), with
the element bound scaled per channel. On inputs scaled by a factor (``input_scale``: the
evaluator's ×3 / ×0.01 / ×−1 checks, :mod:`kernel_agent.kernels.verify`) the absolute
tolerance grows with a factor above 1 and the signal threshold shrinks with one below 1.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch

#: (atol, rtol) per dtype.  Fused kernels change accumulation order, so these
#: are looser than ``torch.testing`` defaults but tight enough to catch bugs.
TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}
#: Fraction of elements allowed outside tolerance (rounding-boundary flips).
MAX_MISMATCH = 1e-3
#: ... and none of them by more than this multiple of its tolerance.
MAX_OUTLIER = 10.0
#: Tensors with more elements than this are also compared as a whole.
GLOBAL_MIN_NUMEL = 1024
#: (min cosine, max relative L2 error) per dtype for that whole-tensor check.  On
#: VoxCPM2 modules, fp32-accumulating and MATH-SDPA variants of bf16 attention and
#: MLP stay below a relative L2 error of 0.0035 (cosine >= 0.99999), also on
#: perturbed inputs.
GLOBAL_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (0.99999, 0.005),
    torch.float16: (0.9995, 0.01),
    torch.bfloat16: (0.999, 0.02),
}
#: Tolerance tiers. ``exact``: the checks above. ``near-lossless``
#: (``--quality near-lossless``, a target whose spec allows reduced precision,
#: :data:`REDUCED_PRECISIONS`; recorded in its capture as ``tier``): reduced-precision
#: weights (FP8) move every element of a GEMM's output by a few per cent of the output's
#: RMS, so the per-element (atol, rtol) checks are replaced by whole-tensor bounds. Tensors
#: without signal keep the exact checks. Calibrated on FP8 (e4m3, per output channel)
#: weight-only nn.Linear GEMVs / GEMMs, MLPs and the LocDiT estimator of VoxCPM2 on real
#: inputs: cosine >= 0.9985, relative L2 error <= 0.055, norm within 0.7 %, every element
#: within 0.47 of its bound below (README, "Quality modes"). FP8 W8A8 (``fp8_w8a8``: e4m3
#: weights per output channel x e4m3 activations per token, fp32 accumulation) fits the
#: same bounds on real inputs of the compute-bound GEMMs it is for: the VoxCPM2 LocDiT
#: layer at M = 352 and its MLP / q_proj alone reach cosine >= 0.9997, relative L2 error
#: <= 0.021, norm within 0.35 %, every element within 0.23 of its bound (fake quant, and on
#: the GPU with torch._scaled_mm); an LM decode layer and its MLP <= 0.015. Only the LM's
#: q_proj alone at M = 1 (activation crest ~30) fails, on the norm (2.4 %): memory bound,
#: fp8_weights' job (README, "FP8 W8A8"). MXFP8 W8A8 (``fp8_mx``: e4m3 + one power-of-two
#: ue8m0 scale per 32 along K on both operands) with the non-saturating scale rule
#: ``2^ceil(log2(amax / 448))`` has the same error and bounds (docs/FP8.md §3.2, fake quant
#: on the VoxCPM2 LocDiT layer at M = 352: relative L2 0.0205 vs 0.0204 per token, element
#: ratio 0.17, norm 0.92 %; 30 redrawn draws: none fails, relative L2 <= 0.038, element ratio
#: <= 0.44). The OCP rule ``2^(floor(log2 amax) - 8)`` saturates block maxima above 448 x
#: scale and fails on massive activations (o_proj: relative L2 0.041, norm 4.0 %); the
#: evaluator's scale-rule guard names it (:mod:`kernel_agent.kernels.scale_guard`).
#: INT8 (#178: symmetric int8, ``amax / 127`` per output channel; ``int8_w8a8`` also per
#: token, int32 accumulation) shares these bounds too (README, "INT8 W8A8"; real VoxCPM2 /
#: Qwen3 captures, ``torch._int_mm`` on the GPU, 12-90 redrawn and the three scaled draws per
#: case). INT8 weight-only is the most accurate 8-bit class (LocDiT layer relative L2 0.0054,
#: LM decode layer 0.0018, Qwen3 MLP 0.014; FP8 weights 0.015 / 0.004 / 0.038). INT8 W8A8
#: matches FP8 W8A8 where the activations have no outlier channels (LM decode layer 0.0046-
#: 0.0070, Qwen3 MLP 0.060 vs 0.050) and fails the tier where they do: on the LocDiT MLP
#: (token crest 29-49) small activations underflow to 0, a biased loss (norm -2.1 % at
#: M = 352, and the x0.01 check). SmoothQuant (``quant.smoothquant_factors``, alpha 0.4,
#: factors from another captured call) passes there (relative L2 0.0082, norm 0.24 %, 0 of 120
#: redrawn draws fail); alpha >= 0.5 fits the factors to the captured outliers and fails
#: redrawn draws (1 of 60 at 0.5, 60 of 60 at 0.7), as do factors from a single decode token.
#: Broken INT8 kernels fail: weight scales x 1.05, a neighbour channel's scale, the first
#: token's scale, a zeroed output channel, a static (calibrated) activation scale, activations
#: clipped at their 99.9th percentile, and a per-tensor activation scale (the x0.01 check).
#: In the ``relaxed`` tier every correct INT8 recipe passes, plain per-token INT8 W8A8 on the
#: LocDiT too (norm 2.1 % captured, 4.1 % at x0.01, within 4 % / 6 %), and each broken one
#: still fails at least one capture (a static calibrated activation scale passes the VoxCPM2
#: LM decode layer there, as it fails the LocDiT and Qwen3 ones).
EXACT_TIER = "exact"
NEAR_LOSSLESS_TIER = "near-lossless"
#: ``near-lossless-fp4``: the near-lossless checks with the wider bounds of block-scaled FP4
#: weights (``fp4_weights``, :data:`NEAR_LOSSLESS_BOUNDS`).
NEAR_LOSSLESS_FP4_TIER = "near-lossless-fp4"
#: ``near-lossless-kv``: an FP8 KV cache (``fp8_kv``): the near-lossless bounds on captured
#: inputs, wider ones on redrawn and scaled inputs (:data:`PERTURBED_BOUNDS`): scaling q and
#: k by f scales the attention logits by f², and e4m3 K's relative rounding error moves a
#: sharper softmax further.
NEAR_LOSSLESS_KV_TIER = "near-lossless-kv"
#: ``relaxed`` (``--quality relaxed``, #175): the near-lossless checks with about twice their
#: error budgets (:data:`RELAXED_BOUNDS`); ``relaxed-fp4`` and ``relaxed-kv`` likewise for
#: ``fp4_weights`` and ``fp8_kv``.
RELAXED_TIER = "relaxed"
RELAXED_FP4_TIER = "relaxed-fp4"
RELAXED_KV_TIER = "relaxed-kv"
TIERS = (
    EXACT_TIER,
    NEAR_LOSSLESS_TIER,
    NEAR_LOSSLESS_FP4_TIER,
    NEAR_LOSSLESS_KV_TIER,
    RELAXED_TIER,
    RELAXED_FP4_TIER,
    RELAXED_KV_TIER,
)
#: The quality modes (``--quality``): ``exact`` (numerics within rounding noise of eager),
#: ``near-lossless`` and ``relaxed`` (numerics-changing optimisations judged end to end by
#: the perceptual gate, :mod:`kernel_agent.workloads.perceptual`; ``relaxed`` with about
#: twice near-lossless's error budgets). A mode is named after its main tier.
QUALITIES = (EXACT_TIER, NEAR_LOSSLESS_TIER, RELAXED_TIER)
#: The quality modes that allow reduced precision (a target's tier follows its precision).
REDUCED_QUALITIES = (NEAR_LOSSLESS_TIER, RELAXED_TIER)
#: The quality mode of a new run (``optimize``, ``improve``, ``analyze``); a run keeps the
#: mode recorded in its ``run.json`` (one without any: ``exact``).
DEFAULT_QUALITY = RELAXED_TIER
#: ``"precision"`` of a target spec that ``--quality near-lossless`` captures in the
#: near-lossless tier: ``fp8_weights`` (FP8 weight-only storage, per-channel scales, bf16
#: activations; skill fp8-weights), ``fp8_w8a8`` (FP8 tensor-core math:
#: e4m3 weights per output channel and activations per token, fp32 accumulation; for
#: compute-bound GEMMs), ``fp8_mx`` (MXFP8 W8A8: e4m3 + ue8m0 per 32 along K on both
#: operands, block-scaled tensor cores; compute-bound GEMMs with wide N) or ``reduced``
#: (another numerics-changing kernel); ``fp4_weights`` (block-scaled FP4 weights, bf16
#: activations) in the near-lossless-fp4 tier (:data:`PRECISION_TIERS`). ``fp8_kv``: an FP8
#: (e4m3) KV cache, one scale per token and KV head, bf16 weights and math
#: (kernels/kv_quant.py); opt-in (``precisions.OPT_IN``), in the near-lossless-kv tier: e4m3 K / V
#: moved a decoder layer's output by <= 0.0011 relative L2 on real captures (docs/FP8.md §5).
#: ``int8_weights`` (INT8 weight-only: int8 codes per output channel, bf16 activations) and
#: ``int8_w8a8`` (INT8 tensor-core math, IMMA: int8 weights per output channel and activations
#: per token, int32 accumulation; the 8-bit compute class of GPUs without FP8 tensor cores),
#: #178: the near-lossless tier. Anything else (``exact``, none) is the exact tier.
REDUCED_PRECISIONS = (
    "fp8_weights",
    "reduced",
    "fp4_weights",
    "fp8_w8a8",
    "fp8_mx",
    "fp8_kv",
    "int8_weights",
    "int8_w8a8",
)
#: The tier of a reduced precision other than the near-lossless tier.
PRECISION_TIERS = {"fp4_weights": NEAR_LOSSLESS_FP4_TIER, "fp8_kv": NEAR_LOSSLESS_KV_TIER}
#: ... and other than the relaxed tier.
RELAXED_PRECISION_TIERS = {"fp4_weights": RELAXED_FP4_TIER, "fp8_kv": RELAXED_KV_TIER}
#: Per quality mode that allows reduced precision: (the tier of its reduced precisions, the
#: precisions with a tier of their own).
QUALITY_TIERS: dict[str, tuple[str, dict[str, str]]] = {
    NEAR_LOSSLESS_TIER: (NEAR_LOSSLESS_TIER, PRECISION_TIERS),
    RELAXED_TIER: (RELAXED_TIER, RELAXED_PRECISION_TIERS),
}
PRECISIONS = (EXACT_TIER, *REDUCED_PRECISIONS)
NEAR_LOSSLESS_MIN_COSINE = 0.996
NEAR_LOSSLESS_MAX_REL_L2 = 0.08
#: ``‖new‖ / ‖ref‖`` within this of 1: rounding noise is unbiased, a wrong scale is not.
NEAR_LOSSLESS_MAX_NORM_CHANGE = 0.02
#: Every element within ``a * RMS(ref) + r * |ref|`` (``(a, r)``): a corrupted element or
#: row fails, a massive activation keeps FP8's relative rounding step.
NEAR_LOSSLESS_ELEMENT = (0.5, 0.125)
#: The ``near-lossless-fp4`` tier's bounds (the same checks). Calibrated on NVFP4 weight-only
#: nn.Linear calls (147), MLPs and the LocDiT estimator of VoxCPM2 on real inputs: cosine
#: >= 0.9778, relative L2 error <= 0.21 (a 256-wide decode output; mean 0.055), norm within
#: 2.1 % (noise adds energy: ``sqrt(1 + rel_l2²)``), every element within 0.81 of its bound
#: below. Bugs: of the 147 calls, nibble-swapped codes and weights x 1.05 fail 139, block
#: scales shifted by one block 85, int4 per tensor 86, a zeroed output channel 34; of the
#: 21 MLP / LocDiT calls, nibble swaps and int4 fail all, shifted block scales 19 (README,
#: "Quality modes").
NEAR_LOSSLESS_FP4_MIN_COSINE = 0.96
NEAR_LOSSLESS_FP4_MAX_REL_L2 = 0.28
NEAR_LOSSLESS_FP4_MAX_NORM_CHANGE = 0.04
NEAR_LOSSLESS_FP4_ELEMENT = (1.25, 0.25)
#: The ``relaxed`` tier's bounds (#175): twice the near-lossless budgets of cosine, relative
#: L2 error and norm, 1.5 times its element bound (at 1.0 an output row a GEMM never writes
#: passes the captured inputs of the VoxCPM2 LocDiT and base-LM layers: element ratio 0.91 /
#: 0.92; at 0.75 1.17 / 1.18). Calibrated with the reference math of every 8-bit precision
#: on real captures (the VoxCPM2 LocDiT layer at M = 352 / 176, the VoxCPM2 base-LM and the
#: Qwen3-0.6B decoder layers at decode; docs/research-scripts/relaxed-175, README "Quality
#: modes"): cosine >= 0.9985, relative L2 <= 0.055, norm within 3.0 % (MXFP8 at M = 1,
#: which near-lossless rejects; else <= 1.3 %), element ratio <= 0.27. Weight scales x 1.05
#: / x 1.2, int4 per tensor, a dropped KV head and the unwritten row still fail.
RELAXED_MIN_COSINE = 0.99
RELAXED_MAX_REL_L2 = 0.16
RELAXED_MAX_NORM_CHANGE = 0.04
RELAXED_ELEMENT = (0.75, 0.125)
#: The ``relaxed-fp4`` tier's bounds: 1.5 times the near-lossless-fp4 budgets (at 2x the
#: element bound int4 per tensor passes the LocDiT layer's captured inputs). NVFP4 / MXFP4
#: weights reach cosine >= 0.989, relative L2 <= 0.146, norm within 4.9 % (MXFP4 on the
#: LocDiT layer, which near-lossless-fp4 rejects), element ratio <= 0.28; swapped nibbles,
#: block scales x 1.2 and int4 per tensor still fail.
RELAXED_FP4_MIN_COSINE = 0.94
RELAXED_FP4_MAX_REL_L2 = 0.36
RELAXED_FP4_MAX_NORM_CHANGE = 0.06
RELAXED_FP4_ELEMENT = (1.75, 0.25)
#: (min cosine, max relative L2 error, max norm change, element bound) of each tier that
#: replaces the per-element checks with whole-tensor bounds.
NEAR_LOSSLESS_BOUNDS: dict[str, tuple[float, float, float, tuple[float, float]]] = {
    NEAR_LOSSLESS_TIER: (
        NEAR_LOSSLESS_MIN_COSINE,
        NEAR_LOSSLESS_MAX_REL_L2,
        NEAR_LOSSLESS_MAX_NORM_CHANGE,
        NEAR_LOSSLESS_ELEMENT,
    ),
    NEAR_LOSSLESS_FP4_TIER: (
        NEAR_LOSSLESS_FP4_MIN_COSINE,
        NEAR_LOSSLESS_FP4_MAX_REL_L2,
        NEAR_LOSSLESS_FP4_MAX_NORM_CHANGE,
        NEAR_LOSSLESS_FP4_ELEMENT,
    ),
    NEAR_LOSSLESS_KV_TIER: (
        NEAR_LOSSLESS_MIN_COSINE,
        NEAR_LOSSLESS_MAX_REL_L2,
        NEAR_LOSSLESS_MAX_NORM_CHANGE,
        NEAR_LOSSLESS_ELEMENT,
    ),
    RELAXED_TIER: (
        RELAXED_MIN_COSINE,
        RELAXED_MAX_REL_L2,
        RELAXED_MAX_NORM_CHANGE,
        RELAXED_ELEMENT,
    ),
    RELAXED_FP4_TIER: (
        RELAXED_FP4_MIN_COSINE,
        RELAXED_FP4_MAX_REL_L2,
        RELAXED_FP4_MAX_NORM_CHANGE,
        RELAXED_FP4_ELEMENT,
    ),
    RELAXED_KV_TIER: (
        RELAXED_MIN_COSINE,
        RELAXED_MAX_REL_L2,
        RELAXED_MAX_NORM_CHANGE,
        RELAXED_ELEMENT,
    ),
}
#: Redrawn inputs (``perturbed``: the evaluator's perturbed-input check,
#: :mod:`kernel_agent.kernels.verify`, its timed-output check,
#: :func:`kernel_agent.kernels.bench.check_timed_output`, and the integration's re-check,
#: :mod:`kernel_agent.kernels.recheck`, compare the candidate with the reference called on
#: redrawn inputs) were drawn from each tensor's own mean and std, without outlier channels
#: (since #198 only a single token or a short cache is, the rest per channel:
#: :func:`kernel_agent.kernels.verify.redraw_stats`), so bounds calibrated on real inputs do
#: not carry over (#109). A weight row that writes a massive
#: activation (VoxCPM2 LocDiT o_proj / down_proj row 497: 10x / 7x the median row norm; real
#: outputs up to 8576 there) gives its output channel that many times the rounding error of
#: the others. On real inputs that channel stays massive and the ``r * |ref|`` term covers
#: it; on redrawn ones its values spread around zero, and where they are small the error is
#: up to 2.8 x the tensor's RMS. With the bounds above, the reference math (fake quant;
#: W8A8: ``quant.fp8_w8a8_linear``) of the LocDiT layer at M = 352 failed the element bound
#: in 98 of 200 W8A8 draws (element ratio up to 2.4 at cosine >= 0.99988, relative L2 error
#: <= 0.014; the run's W8A8 kernels: 1.42 and 1.65), and its attention, MLP, o_proj and
#: down_proj alone with every precision (FP8 weights up to 2.6, W8A8 5.9, NVFP4 3.4). So on
#: redrawn inputs the RMS in the element bound is the larger of the tensor's and the
#: element's channel's (one position of the last dimension, over at least CHANNEL_MIN_ROWS
#: rows; for fewer, the tensor's), and a tensor without signal gets the larger of the exact
#: tolerance and the tier's element bound at RMS = atol.
CHANNEL_MIN_ROWS = 16
#: (min cosine, max relative L2 error, max norm change, element bound) of each tier on
#: redrawn inputs, with the per-channel RMS above. Calibrated with the reference math of each
#: precision on real VoxCPM2 captures, 30-100 seeds of both redraws (normal; uniform /
#: Laplace / log-normal): the LocDiT layer (M = 352, 176, 22), its attention, MLP and seven
#: nn.Linear, and a base-LM decode layer (M = 1, KV cache), its attention, MLP and seven
#: nn.Linear (README, "Quality modes"); the GPU (``torch._scaled_mm``) and the run's W8A8
#: kernels give the LocDiT layer's numbers:
#:
#: * ``near-lossless``: FP8 weights reach cosine >= 0.9987, relative L2 <= 0.051, norm within
#:   1.1 %, element ``a`` (at r = 0.125) <= 0.22 with a channel RMS and 0.43 at decode (one
#:   row: the LM down_proj GEMV, 0.86 of the 0.5 above). W8A8 on the compute-bound GEMMs it
#:   is for (M >= 22): cosine >= 0.9987, relative L2 <= 0.050, norm within 1.9 % (the
#:   attention's o_proj output: rows 497 and 247 hold 12 % of its weight energy, so their
#:   noise does not average out), ``a`` <= 0.36; at decode (M = 1) 0.9977 / 0.071 / 3.2 % /
#:   0.60. Hence norm 3 % and ``a`` 0.75. W8A8 at M = 1 can still fail the norm of a
#:   256-value KV-cache slot (1 of 600 draws), as it fails on real decode inputs (#91).
#: * ``near-lossless-fp4``: NVFP4 reaches cosine >= 0.9506, relative L2 <= 0.333 and norm
#:   within 9.7 % (the V-cache slot of an LM decode step: 256 values near the signal
#:   threshold, 0.20 relative L2 on real inputs; everywhere else <= 0.18 and 4.0 %), ``a``
#:   (at r = 0.25) <= 1.05 with a channel RMS (the LocDiT attention's k output) and 2.06 at
#:   decode (LM down_proj). Its no-signal V-cache slots failed the exact tolerance in 6 of
#:   598 draws, none with the floor.
#:
#: Broken weights fail on captured inputs as before, and on every redrawn draw of the LocDiT
#: layer: weight scales x 1.05 (norm 5 %), a neighbour channel's scale, the first token's
#: activation scale, a zeroed output channel (FP8; FP4: 18 of 32, it passes FP4's captured
#: check), swapped FP4 nibbles, block scales shifted by one block; FP4's x 1.05 and int4 per
#: tensor fail only on captured inputs. Activation scales cached from the first call pass
#: the captured cases and fail 24 of 32 redrawn draws (a candidate gets four).
#:
#: * ``near-lossless-kv`` (``fp8_kv``): the reference math (``kv_quant.fp8_kv_attention``,
#:   per-token e4m3 K / V) of a decode step (GQA, head dim 64-128, 77-4096 cached tokens,
#:   12 seeds x 4 shapes) on inputs x 3 reaches cosine >= 0.9922, relative L2 <= 0.125,
#:   norm within 1.0 %, element ``a`` (at r = 0.125) <= 1.84 at 0.75 (x 0.01, x -1:
#:   <= 0.040 relative L2). Broken caches fail at x 3 on every draw: scales from the
#:   captured inputs (static: cosine <= 0.34), the first token's scale for every token
#:   (cosine <= 0.961, element 4.6), V scales x 1.05 (norm 5.2 %, also at x 0.01 and x -1).
#:
#: * ``relaxed`` / ``relaxed-fp4`` / ``relaxed-kv`` (#175): about twice the budgets above (the
#:   FP4 one 1.2-1.5 times). Redrawn and scaled inputs of the LocDiT and base-LM layers: every
#:   8-bit precision and NVFP4 / MXFP4 pass every draw (MXFP8 at decode reaches norm 5.0 %,
#:   failing 5 of 60 near-lossless draws); ``fp8_kv`` on 192 redraws: cosine >= 0.995,
#:   relative L2 <= 0.10, norm within 2.9 %, element ratio 0.31 (of ``a`` = 2.5). Scales x 1.2
#:   fail every draw, V scales x 1.05 43 of 192 (and every captured case).
#:
#: Re-measured with the per-channel redraw of #198 (``kernels.verify.redraw_stats``; rotary
#: tables kept; docs/research-scripts/per-channel-198), the bounds unchanged: on the
#: Qwen3-0.6B decoder layer at decode (K cache channels of RMS 225 against a median of 1.6)
#: the reference math of FP8 weights fails 4 / 0 of 80 redraws in near-lossless / relaxed
#: (before: 70 / 54), NVFP4 0 / 0 (71 / 59), MXFP4 6 / 0 (77 / 68), INT8 weights 0 / 0
#: (27 / 2), INT8 W8A8 4 / 0 (64 / 38), FP8 W8A8 and MXFP8 at M = 1 65 / 0-1 (74-75 /
#: 66-70); the VoxCPM2 LocDiT and base-LM layers as before (base-LM MXFP8
#: at decode 11 of 60 near-lossless draws, from 5), ``fp8_kv`` 5 of 192 near-lossless draws
#: on the norm (3.7-4.0 %; relaxed 0). Every broken variant of that calibration is rejected
#: where it was before (activation scales cached from the first call: on 40 of 40 LocDiT
#: draws in both modes, from 37 / 14), except FP4 weights with an unwritten output row on the
#: LocDiT layer, which only the global redraw caught (23 of 40 near-lossless draws):
#: realistic draws keep it within FP4's noise, as the captured inputs do.
PERTURBED_BOUNDS: dict[str, tuple[float, float, float, tuple[float, float]]] = {
    NEAR_LOSSLESS_TIER: (0.996, 0.08, 0.03, (0.75, 0.125)),
    NEAR_LOSSLESS_FP4_TIER: (0.94, 0.40, 0.12, (2.5, 0.25)),
    NEAR_LOSSLESS_KV_TIER: (0.985, 0.16, 0.03, (1.5, 0.125)),
    RELAXED_TIER: (0.99, 0.16, 0.06, (1.5, 0.125)),
    RELAXED_FP4_TIER: (0.90, 0.55, 0.18, (3.0, 0.25)),
    RELAXED_KV_TIER: (0.97, 0.25, 0.06, (2.5, 0.125)),
}
#: The tier of this process's comparisons when a call passes none. The evaluator sets it
#: from the capture before the candidate is imported, so the integrity snapshot
#: (:mod:`kernel_agent.kernels.integrity`) watches it like the constants above.
TIER = EXACT_TIER
_PLAIN = (torch.Tensor, torch.nn.Parameter)


def tier_of(capture: dict[str, Any] | None) -> str:
    """The tolerance tier recorded in a capture (``exact`` when none or unknown)."""
    name = (capture or {}).get("tier")
    return name if name in TIERS else EXACT_TIER


def tier_for(quality: str | None, precision: str | None) -> str:
    """The tier of a target: ``near-lossless`` when the run's quality mode is
    ``near-lossless`` and the target's spec allows reduced precision
    (:data:`REDUCED_PRECISIONS`; ``fp4_weights``: ``near-lossless-fp4``); ``relaxed`` (and
    ``relaxed-fp4`` / ``relaxed-kv``) likewise in a ``relaxed`` run."""
    if quality not in QUALITY_TIERS or precision not in REDUCED_PRECISIONS:
        return EXACT_TIER
    main, own = QUALITY_TIERS[str(quality)]
    return own.get(str(precision), main)


def allows_reduced(quality: str | None) -> bool:
    """Whether a run of quality mode ``quality`` allows reduced precision
    (:data:`REDUCED_QUALITIES`: ``near-lossless``, ``relaxed``)."""
    return quality in REDUCED_QUALITIES


def flatten(value: Any, prefix: str = "out", depth: int = 0) -> dict[str, torch.Tensor]:
    """Flatten nested containers / dataclasses / cache objects into named tensors."""
    found: dict[str, torch.Tensor] = {}
    if depth > 6 or value is None:
        return found
    if isinstance(value, torch.Tensor):
        found[prefix] = value
    elif isinstance(value, dict):
        for k, v in value.items():
            found.update(flatten(v, f"{prefix}.{k}", depth + 1))
    elif isinstance(value, tuple | list):
        for i, v in enumerate(value):
            found.update(flatten(v, f"{prefix}[{i}]", depth + 1))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            found.update(flatten(getattr(value, f.name, None), f"{prefix}.{f.name}", depth + 1))
    elif hasattr(value, "__dict__") and not isinstance(value, torch.nn.Module):
        for k, v in vars(value).items():
            if not k.startswith("__"):
                found.update(flatten(v, f"{prefix}.{k}", depth + 1))
    return found


def type_error(new: Any) -> str | None:
    """Why ``new`` cannot stand in for a reference tensor (None if it can)."""
    if type(new) in _PLAIN:
        return None
    if isinstance(new, torch.Tensor):
        return (
            f"got a {type(new).__name__} (a torch.Tensor subclass); return plain tensors "
            "(a subclass can defer the computation until the result is read)"
        )
    return f"expected tensor, got {type(new).__name__}"


def _non_finite_error(a: torch.Tensor, b: torch.Tensor) -> str | None:
    """Positions of NaN / +inf / -inf must be identical in reference and candidate."""
    problems = []
    for label, in_ref, in_new in (
        ("NaN", torch.isnan(a), torch.isnan(b)),
        ("+inf", a == math.inf, b == math.inf),
        ("-inf", a == -math.inf, b == -math.inf),
    ):
        differ = int((in_ref != in_new).sum())
        if differ:
            problems.append(
                f"{label} at {differ} positions that differ "
                f"(candidate {int(in_new.sum())}, reference {int(in_ref.sum())})"
            )
    return "; ".join(problems) or None


def _channel_rms(ref: torch.Tensor, rms: float) -> torch.Tensor | float:
    """``max(rms, RMS of the element's channel)`` per element of ``ref`` (a channel: one
    position of the last dimension, its RMS over all the other dimensions, non-finite
    values as 0); ``rms`` for tensors with fewer than :data:`CHANNEL_MIN_ROWS` rows."""
    if ref.dim() < 2 or ref.numel() < CHANNEL_MIN_ROWS * ref.shape[-1]:
        return rms
    values = torch.where(torch.isfinite(ref), ref, torch.zeros_like(ref))
    channel = values.pow(2).mean(dim=tuple(range(ref.dim() - 1)), keepdim=True).sqrt()
    return channel.clamp_min(rms).expand_as(ref)


def compare_tensors(
    name: str,
    ref: torch.Tensor,
    new: torch.Tensor,
    tol: tuple[float, float] | None = None,
    *,
    tier: str | None = None,
    perturbed: bool = False,
    input_scale: float = 1.0,
) -> dict[str, Any]:
    """One tensor against its reference (module docstring); ``tier``: the tolerance tier
    (default :data:`TIER`); ``perturbed``: on redrawn inputs (:data:`PERTURBED_BOUNDS`);
    ``input_scale``: the factor ``f`` the inputs were scaled by (the evaluator's ×3, ×0.01
    and ×−1 checks, :data:`kernel_agent.kernels.verify.SCALED`). A module's outputs, and
    their rounding errors, grow with ``|f| > 1``, so the absolute tolerance is
    ``atol × |f|`` (honest per-element flips near zero at ×3 are not a bug); with
    ``|f| < 1`` they shrink, so a tensor has signal from ``|f| × atol × √n`` on and gets
    the whole-tensor checks (cosine and relative L2 error; the near-lossless tiers' bounds,
    which are relative). Without that a ×0.01 output below the absolute tolerance would
    pass anything, e.g. an FP8 kernel with an activation scale calibrated on the captured
    inputs. The per-element tolerance itself never shrinks: intermediate rounding (a
    residual add, a normalisation) does not scale with the inputs."""
    result: dict[str, Any] = {"name": name, "ok": False}
    error = type_error(new)
    if error is not None:
        result["error"] = error
        return result
    if ref.shape != new.shape:
        result["error"] = f"shape {tuple(new.shape)} != reference {tuple(ref.shape)}"
        return result
    if ref.dtype != new.dtype:
        result["error"] = f"dtype {new.dtype} != reference {ref.dtype}"
        return result
    if new.layout != ref.layout or new.device.type != ref.device.type:
        result["error"] = (
            f"{new.layout} tensor on {new.device.type} != reference "
            f"{ref.layout} tensor on {ref.device.type}"
        )
        return result
    if not ref.is_floating_point():
        mismatches = (ref.to(new.device) != new).float().mean().item() if ref.numel() else 0.0
        result.update(ok=mismatches == 0, mismatch_frac=mismatches)
        return result
    atol, rtol = tol or TOLERANCES.get(ref.dtype, (1e-3, 1e-3))
    factor = abs(float(input_scale)) or 1.0
    floor = atol * min(1.0, factor)  # the signal threshold per element
    atol *= max(1.0, factor)
    a = full = ref.detach().to(new.device).float()
    b = new.detach().float()
    error = _non_finite_error(a, b)
    if error is not None:
        result["error"] = error
        return result
    finite = torch.isfinite(a)
    masked = not bool(finite.all())
    if masked:  # identical non-finite positions: compare the rest
        a, b = a[finite], b[finite]
    diff = (a - b).abs()
    ref_norm, new_norm = float(a.norm()), float(b.norm())
    signal = a.numel() > 0 and ref_norm > floor * math.sqrt(a.numel())
    near = (PERTURBED_BOUNDS if perturbed else NEAR_LOSSLESS_BOUNDS).get(tier or TIER)
    allowed = atol + rtol * a.abs()
    if perturbed and near is not None and not signal:  # the tier's element bound at RMS atol
        allowed = torch.maximum(allowed, near[3][0] * atol + near[3][1] * a.abs())
    mismatch = (diff > allowed).float().mean().item() if a.numel() else 0.0
    outlier = float((diff / allowed.clamp_min(1e-30)).max()) if a.numel() else 0.0
    denom = ref_norm * new_norm
    cos = float((a.flatten() @ b.flatten()) / denom) if denom > 0 else 1.0
    rel_l2 = float(diff.norm()) / ref_norm if ref_norm > 0 else 0.0
    result.update(
        max_abs_err=float(diff.max()) if diff.numel() else 0.0,
        mean_abs_err=float(diff.mean()) if diff.numel() else 0.0,
        mismatch_frac=round(mismatch, 6),
        max_err_ratio=round(outlier, 3),
        cosine=round(cos, 7),
        rel_l2=float(f"{rel_l2:.4g}"),
        atol=atol,
        rtol=rtol,
    )
    problems = []
    min_cos, max_rel = GLOBAL_TOLERANCES.get(ref.dtype, GLOBAL_TOLERANCES[torch.float32])
    if near is not None and signal:
        min_cos, max_rel, max_norm_change, (e_atol, e_rtol) = near
        rms = ref_norm / math.sqrt(a.numel())
        scale: torch.Tensor | float = rms
        if perturbed:  # the larger of the tensor's and the element's channel's RMS
            scale = _channel_rms(full, rms)
            if masked and isinstance(scale, torch.Tensor):
                scale = scale[finite]
        element = float((diff / (e_atol * scale + e_rtol * a.abs())).max())
        norm_ratio = new_norm / ref_norm
        result.update(
            tier=tier or TIER,
            max_err_over_rms=round(float(diff.max()) / rms, 4),
            element_ratio=round(element, 4),
            norm_ratio=round(norm_ratio, 6),
        )
        if element > 1.0:
            where = " on redrawn inputs" if perturbed else ""
            of = " (the larger of the tensor's and its channel's)" if perturbed else ""
            problems.append(
                f"an element is {element:.3g}x its {tier or TIER} tolerance away{where} "
                f"({e_atol:g} x RMS{of} + {e_rtol:g} x |reference|)"
            )
        if abs(norm_ratio - 1.0) > max_norm_change:
            problems.append(
                f"norm x{norm_ratio:.4f} (allowed ±{max_norm_change:.0%}): "
                "a systematic error, not rounding noise"
            )
        whole = True
    else:
        if mismatch > MAX_MISMATCH:
            problems.append(
                f"{mismatch:.4%} of elements outside tolerance (max {MAX_MISMATCH:.2%})"
            )
        if outlier > MAX_OUTLIER:
            problems.append(
                f"an element is {outlier:.3g}x its tolerance away (max {MAX_OUTLIER:g}x per "
                "element)"
            )
        whole = a.numel() > GLOBAL_MIN_NUMEL and signal
    if whole:
        if cos < min_cos:
            problems.append(f"cosine {cos:.6f} < {min_cos}")
        if rel_l2 > max_rel:
            problems.append(f"relative L2 error {rel_l2:.4g} > {max_rel}")
    if problems:
        result["error"] = "; ".join(problems)
    result["ok"] = not problems
    return result


def compare_structures(
    ref: Any,
    new: Any,
    prefix: str = "out",
    *,
    tier: str | None = None,
    perturbed: bool = False,
    input_scale: float = 1.0,
) -> list[dict[str, Any]]:
    ref_flat = flatten(ref, prefix)
    new_flat = flatten(new, prefix)
    kw: dict[str, Any] = {"tier": tier, "perturbed": perturbed, "input_scale": input_scale}
    results = []
    for name, tensor in ref_flat.items():
        if name not in new_flat:
            results.append({"name": name, "ok": False, "error": "missing in candidate output"})
            continue
        results.append(compare_tensors(name, tensor, new_flat[name], **kw))
    return results


def compare_side_effects(
    pre: Any,
    ref_post: Any,
    new_post: Any,
    prefix: str = "args",
    *,
    tier: str | None = None,
    perturbed: bool = False,
    input_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Compare the post-call state of a call's arguments (in-place side effects).

    For argument tensors that keep their shape and dtype, only the elements that
    the reference *or* the candidate changed are compared, so the mismatch
    allowance (:data:`MAX_MISMATCH`) is relative to the update.  Otherwise
    writing one position of an 8192-long KV cache (0.01 % of its elements), or
    forgetting to, would vanish inside the allowance.  Other tensors (e.g. caches
    that grow by concatenation) are compared whole."""
    return compare_side_effects_flat(
        flatten(pre, prefix),
        flatten(ref_post, prefix),
        flatten(new_post, prefix),
        tier=tier,
        perturbed=perturbed,
        input_scale=input_scale,
    )


def compare_side_effects_flat(
    pre_flat: dict[str, torch.Tensor],
    ref_flat: dict[str, torch.Tensor],
    new_flat: dict[str, torch.Tensor],
    *,
    tier: str | None = None,
    perturbed: bool = False,
    input_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """:func:`compare_side_effects` on already flattened ``{name: tensor}`` states."""
    kw: dict[str, Any] = {"tier": tier, "perturbed": perturbed, "input_scale": input_scale}
    results: list[dict[str, Any]] = []
    for name, ref in ref_flat.items():
        new = new_flat.get(name)
        if new is None:
            results.append({"name": name, "ok": False, "error": "missing in candidate arguments"})
            continue
        before = pre_flat.get(name)
        if (
            before is None
            or type_error(new) is not None
            or not (before.shape == ref.shape == new.shape)
            or not (before.dtype == ref.dtype == new.dtype)
        ):
            results.append(compare_tensors(name, ref, new, **kw))
            continue
        before, ref = before.to(new.device), ref.to(new.device)
        changed = (ref != before) | (new != before)
        count = int(changed.sum())
        if count == 0:
            results.append({"name": name, "ok": True, "changed_elements": 0, "max_abs_err": 0.0})
            continue
        result = compare_tensors(name, ref[changed], new[changed], **kw)
        result["changed_elements"] = count
        results.append(result)
    return results
