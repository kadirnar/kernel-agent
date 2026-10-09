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
(``fp4_weights``) get the wider bounds of the ``near-lossless-fp4`` tier, W4A4
(``fp4_w4a4``) those of ``near-lossless-fp4a``. ``--quality
relaxed`` (the default of new runs, :data:`DEFAULT_QUALITY`) has the same structure with
about twice the error budgets: the ``relaxed``, ``relaxed-fp4``, ``relaxed-fp4a`` and
``relaxed-kv`` tiers (:data:`QUALITY_TIERS`). On redrawn
inputs (``perturbed``) these tiers use their own bounds (:data:`PERTURBED_BOUNDS`), with
the element bound scaled per channel. On inputs scaled by a factor (``input_scale``: the
evaluator's ×3 / ×0.01 / ×−1 checks, :mod:`kernel_agent.kernels.verify`) the absolute
tolerance grows with a factor above 1 and the signal threshold shrinks with one below 1.

Caches are compared relative to the update. An argument a call updates in place is compared
on the elements the reference or the candidate changed (:func:`compare_side_effects`). A
cache the call grows (:func:`grown_dim`: an argument, or an output whose leading part along
one dimension equals an input tensor, e.g. a KV cache ``torch.cat``-ed with the new token's
K / V) is compared in two parts (:func:`compare_grown`): the appended rows on their own
(the tier's checks with their own RMS), and the part of the input's shape, which must stay
bit for bit where the reference kept it. Compared whole, one wrong new row of a 4096-token
cache was 0.02 % of the elements, inside :data:`MAX_MISMATCH`, and at ×0.01 the scaled old
rows set the element bound of a new row that is not scaled (#202). An output of an input's
shape that the reference wrote a few rows into (:func:`written_box`: a static KV cache
returned by a functional ``index_copy`` / ``scatter``) likewise (:func:`compare_written`): the
written rows on their own, the rest bit for bit (#206).
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
#: ``near-lossless-fp4a`` / ``relaxed-fp4a`` (#233): W4A4 (``fp4_w4a4``: block-scaled FP4
#: weights and activations), whose two quantised operands move a GEMM's output about 1.4x as
#: far as FP4 weights alone (:data:`NEAR_LOSSLESS_BOUNDS`).
NEAR_LOSSLESS_FP4A_TIER = "near-lossless-fp4a"
RELAXED_FP4A_TIER = "relaxed-fp4a"
TIERS = (
    EXACT_TIER,
    NEAR_LOSSLESS_TIER,
    NEAR_LOSSLESS_FP4_TIER,
    NEAR_LOSSLESS_KV_TIER,
    RELAXED_TIER,
    RELAXED_FP4_TIER,
    RELAXED_KV_TIER,
    NEAR_LOSSLESS_FP4A_TIER,
    RELAXED_FP4A_TIER,
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
#: #178: the near-lossless tier. ``fp4_w4a4`` (W4A4, #233: NVFP4 / MXFP4 weights and
#: activations on the block-scaled FP4 tensor cores, fp32 accumulation; opt-in like every
#: 4-bit precision): the near-lossless-fp4a tier. Anything else (``exact``, none) is the
#: exact tier.
REDUCED_PRECISIONS = (
    "fp8_weights",
    "reduced",
    "fp4_weights",
    "fp8_w8a8",
    "fp8_mx",
    "fp8_kv",
    "int8_weights",
    "int8_w8a8",
    "fp4_w4a4",
)
#: The tier of a reduced precision other than the near-lossless tier.
PRECISION_TIERS = {
    "fp4_weights": NEAR_LOSSLESS_FP4_TIER,
    "fp8_kv": NEAR_LOSSLESS_KV_TIER,
    "fp4_w4a4": NEAR_LOSSLESS_FP4A_TIER,
}
#: ... and other than the relaxed tier.
RELAXED_PRECISION_TIERS = {
    "fp4_weights": RELAXED_FP4_TIER,
    "fp8_kv": RELAXED_KV_TIER,
    "fp4_w4a4": RELAXED_FP4A_TIER,
}
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
#: The ``near-lossless-fp4a`` / ``relaxed-fp4a`` tiers' bounds (W4A4, #233), calibrated with
#: the reference math (``quant.fp4_w4a4_linear``: NVFP4 weights and per-token NVFP4
#: activations, ``F.scaled_mm`` on the GPU) on real captures (docs/research-scripts/w4a4-233):
#: the VoxCPM2 LocDiT layer (M = 352 / 176, compute bound: what W4A4 is for) reaches cosine
#: >= 0.9968, relative L2 <= 0.080, norm within 3.4 %, element ``a`` (at r = 0.25) <= 0.40;
#: the 12-layer VoxCPM2 LocEnc at M = 80 0.9929 / 0.119 / 0.2 % / 0.75; the VoxCPM2 base-LM
#: and Qwen3-0.6B decode layers (M = 1) >= 0.9839 / <= 0.179 / 2.3 % / 0.75. Quantised
#: activations shrink a GEMM's output by up to ~1 % (e2m1's grid, not the scale rounding:
#: unrounded block scales give the same; LocDiT gate_proj -0.9 %, down_proj -1.1 %, q_proj
#: 0.0 %), so the LocDiT MLP (gate / up, then down) loses 3.4 % of the hidden state's norm
#: (gate / up in FP8: 0.9 %). Hence the FP4 weight tier's cosine and relative L2 with norm
#: ±5 % and ``a`` 1.5. MXFP4 W4A4 fails it on the LocDiT layer (norm 6.5 %), as MXFP4 weights fail
#: near-lossless-fp4 there, and passes ``relaxed-fp4a`` (about 1.3-1.6x the budgets). Broken
#: kernels fail on the single-layer captures: weight tensor scale x 1.05 (norm 5.7-13.6 %)
#: and x 1.2, swapped nibbles, activation block scales shifted by one block, the first token's
#: outer scale for every token, activation codes truncated instead of rounded, activation
#: scales cached from the first call (redrawn inputs), int4 per tensor, a dropped KV head; an
#: unwritten output row and swapped q heads pass the LocDiT layer within W4A4's noise (as with
#: FP4 weights), and the LocEnc's final RMSNorm hides scale bugs (x 1.2 and truncated codes
#: pass it): the perceptual gate judges those end to end.
NEAR_LOSSLESS_FP4A_MIN_COSINE = 0.96
NEAR_LOSSLESS_FP4A_MAX_REL_L2 = 0.28
NEAR_LOSSLESS_FP4A_MAX_NORM_CHANGE = 0.05
NEAR_LOSSLESS_FP4A_ELEMENT = (1.5, 0.25)
RELAXED_FP4A_MIN_COSINE = 0.94
RELAXED_FP4A_MAX_REL_L2 = 0.36
RELAXED_FP4A_MAX_NORM_CHANGE = 0.08
RELAXED_FP4A_ELEMENT = (2.0, 0.25)
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
    NEAR_LOSSLESS_FP4A_TIER: (
        NEAR_LOSSLESS_FP4A_MIN_COSINE,
        NEAR_LOSSLESS_FP4A_MAX_REL_L2,
        NEAR_LOSSLESS_FP4A_MAX_NORM_CHANGE,
        NEAR_LOSSLESS_FP4A_ELEMENT,
    ),
    RELAXED_FP4A_TIER: (
        RELAXED_FP4A_MIN_COSINE,
        RELAXED_FP4A_MAX_REL_L2,
        RELAXED_FP4A_MAX_NORM_CHANGE,
        RELAXED_FP4A_ELEMENT,
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
#:
#: Re-measured with grown caches compared in parts (#202, :func:`compare_grown`;
#: docs/research-scripts/grown-cache-202), the bounds unchanged: on the Qwen3 layer the
#: x 0.01 checks (kept V rows scaled, the new row not) no longer fail the reference math
#: (near-lossless / relaxed, of 12: FP8 weights 3 / 0 -> 0 / 0, NVFP4 and MXFP4 4 / 4 ->
#: 0 / 0, FP8 W8A8 6 / 1 -> 2 / 0, MXFP8 5 / 2 -> 1 / 0, INT8 W8A8 4 / 1 -> 0 / 0); the new
#: rows' own norm, cosine and relative L2 checks add redrawn failures of the noisiest
#: numerics (of 80: FP8 weights 4 / 0 -> 5 / 0, MXFP8 65 / 1 -> 70 / 1, MXFP4 6 / 0 ->
#: 8 / 1, the key row's norm). Every broken variant is still rejected in both modes; the
#: VoxCPM2 captures (in-place caches) are unchanged.
#:
#: Re-measured with returned same-shape caches compared in parts (#206,
#: :func:`compare_written`; docs/research-scripts/same-shape-cache-206), the bounds unchanged:
#: the captures above give byte for byte the same results (none returns a written cache); the
#: Qwen3 layer with a returned static cache (its cache in the first half of a static one) as
#: the grown one: x 0.01 no longer fails the reference math (of 12: FP8 weights 4 / 3 -> 0 / 0,
#: NVFP4 and MXFP4 4 / 4 -> 0 / 0, FP8 W8A8 6 / 4 -> 2 / 0, MXFP8 5 / 4 -> 1 / 0, INT8 weights
#: 1 / 0 -> 0 / 0, INT8 W8A8 4 / 3 -> 0 / 0); redrawn (of 80) FP8 weights 8 / 0 -> 9 / 0, FP8
#: W8A8 63 / 3 -> 63 / 4, MXFP8 68 / 4 -> 68 / 6, MXFP4 5 / 1 -> 6 / 1 (the key row's norm).
#: Every broken variant is still rejected in both modes.
#:
#: * ``near-lossless-fp4a`` / ``relaxed-fp4a`` (W4A4, #233; per-channel redraws, 10 seeds of
#:   both kinds): NVFP4 W4A4 on the LocDiT layer reaches cosine >= 0.9885, relative L2 <=
#:   0.151, norm within 3.2 %, ``a`` <= 0.84 (redrawn and x 3 / x 0.01 / x -1), the LocEnc
#:   stack 0.9868 / 0.162 / 0.4 %, the base-LM decode layer 0.9684 / 0.256 / 2.8 % (x 0.01);
#:   none fails. The Qwen3-0.6B decode layer's one-token output reaches 0.893 / 0.479 / 15.4 %
#:   and fails 4 of 80 near-lossless draws on the cosine (none relaxed): W4A4 at M = 1 is
#:   memory bound anyway (``fp4_weights``' job). MXFP4 W4A4 there: 32 / 6 of 80. The redrawn
#:   norm bounds stay below 20 % (relaxed-fp4a at relaxed-fp4's 18 %): a KV-cache row written
#:   x 1.2 must fail every tier (``tests/test_grown_cache.py``).
PERTURBED_BOUNDS: dict[str, tuple[float, float, float, tuple[float, float]]] = {
    NEAR_LOSSLESS_TIER: (0.996, 0.08, 0.03, (0.75, 0.125)),
    NEAR_LOSSLESS_FP4_TIER: (0.94, 0.40, 0.12, (2.5, 0.25)),
    NEAR_LOSSLESS_KV_TIER: (0.985, 0.16, 0.03, (1.5, 0.125)),
    RELAXED_TIER: (0.99, 0.16, 0.06, (1.5, 0.125)),
    RELAXED_FP4_TIER: (0.90, 0.55, 0.18, (3.0, 0.25)),
    RELAXED_KV_TIER: (0.97, 0.25, 0.06, (2.5, 0.125)),
    NEAR_LOSSLESS_FP4A_TIER: (0.90, 0.50, 0.15, (3.0, 0.25)),
    RELAXED_FP4A_TIER: (0.85, 0.65, 0.18, (3.5, 0.25)),
}
#: An output of an input's shape is that input with rows written into it (:func:`written_box`,
#: #206) when the elements the reference changed fill at least WRITTEN_MIN_DENSITY of their box
#: (along each dimension, the indices where one changed) and the box is at most
#: WRITTEN_MAX_FRACTION of the tensor. A cache write fills its box (a new value equals the
#: slot's old one by chance: ~0.1 % of bf16 elements over random old rows); the rounding of a
#: residual add that leaves some elements of a short output as they were scatters them (two
#: tokens changed with probability p each: 1 / (2 - p) <= 0.59 of the box when the box is at
#: most half the tensor; 0.58 measured), and one token has no whole dimension of more than one
#: element. On the captures of VoxCPM2 and Qwen3-0.6B no residual output keeps more than
#: 0.88 % of its elements (docs/research-scripts/same-shape-cache-206, ``scan.md``).
WRITTEN_MAX_FRACTION = 0.5
WRITTEN_MIN_DENSITY = 0.75
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
    (:data:`REDUCED_PRECISIONS`; ``fp4_weights``: ``near-lossless-fp4``, ``fp4_w4a4``:
    ``near-lossless-fp4a``); ``relaxed`` (and ``relaxed-fp4`` / ``relaxed-fp4a`` /
    ``relaxed-kv``) likewise in a ``relaxed`` run."""
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


def _channels(ref: torch.Tensor) -> torch.Tensor | None:
    """The RMS of each channel of ``ref`` (one position of the last dimension, over all the
    other dimensions, non-finite values as 0; shape ``[1, ..., 1, C]``), None for fewer than
    :data:`CHANNEL_MIN_ROWS` rows."""
    if ref.dim() < 2 or ref.numel() < CHANNEL_MIN_ROWS * ref.shape[-1]:
        return None
    values = torch.where(torch.isfinite(ref), ref, torch.zeros_like(ref))
    return values.pow(2).mean(dim=tuple(range(ref.dim() - 1)), keepdim=True).sqrt()


def _channel_rms(
    ref: torch.Tensor, rms: float, floor: torch.Tensor | None = None
) -> torch.Tensor | float:
    """``max(rms, RMS of the element's channel)`` per element of ``ref`` (:func:`_channels`);
    ``rms`` for tensors with fewer than :data:`CHANNEL_MIN_ROWS` rows. ``floor``: a
    per-channel RMS (broadcastable to ``ref``) the element's is at least: the RMS of the
    channels of a grown cache for its appended rows (:func:`compare_grown`)."""
    channel = _channels(ref)
    if floor is not None:
        floor = floor.to(ref.device, torch.float32)
        channel = floor if channel is None else torch.maximum(channel, floor)
    if channel is None:
        return rms
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
    channel_rms: torch.Tensor | None = None,
) -> dict[str, Any]:
    """One tensor against its reference (module docstring); ``tier``: the tolerance tier
    (default :data:`TIER`); ``perturbed``: on redrawn inputs (:data:`PERTURBED_BOUNDS`;
    ``channel_rms``: a per-channel RMS, broadcastable to ``ref``, that the element bound's
    channel RMS is at least: a grown cache's for its appended rows, :func:`compare_grown`);
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
            scale = _channel_rms(full, rms, channel_rms)
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


def grown_dim(before: Any, after: Any) -> int | None:
    """The dimension along which ``after`` is ``before`` grown by a call (a cache extended by
    concatenation): floating-point tensors of one dtype and rank, strided, ``before`` not
    empty, the same size in every dimension but this one, where ``after`` is larger. None
    otherwise (the same shape too: an in-place update, :func:`compare_side_effects`)."""
    if not (isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor)):
        return None
    if (
        not before.is_floating_point()
        or before.dtype != after.dtype
        or before.dim() != after.dim()
        or before.layout != torch.strided
        or after.layout != torch.strided
        or before.numel() == 0
    ):
        return None
    dims = [d for d in range(before.dim()) if before.shape[d] != after.shape[d]]
    if len(dims) != 1 or after.shape[dims[0]] < before.shape[dims[0]]:
        return None
    return dims[0]


def _same(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise: ``a`` and ``b`` hold the same value (NaN the same as NaN)."""
    return (a == b) | (torch.isnan(a) & torch.isnan(b))


def compare_grown(
    name: str,
    before: torch.Tensor,
    ref: torch.Tensor,
    new: torch.Tensor,
    dim: int,
    *,
    tier: str | None = None,
    perturbed: bool = False,
    input_scale: float = 1.0,
) -> dict[str, Any]:
    """``new`` against ``ref``, both ``before`` grown by the call along ``dim``
    (:func:`grown_dim`; a KV cache ``torch.cat``-ed with the new token's K / V), relative to
    the update as in-place caches are (:func:`compare_side_effects`):

    * the appended rows (past ``before``'s size along ``dim``) on their own, with the tier's
      checks (:func:`compare_tensors`: their own RMS, mismatch fraction, norm, cosine and
      relative L2 error; on redrawn inputs an element's channel RMS is at least its
      channel's in the whole grown reference: a decode step appends one token, too few rows
      for a channel RMS of its own, and Qwen3-0.6B's ``k_norm`` makes key channel 50 ~10x
      the row's RMS);
    * the kept part (``before``'s shape): where the reference left it as ``before`` (a
      concatenation copies it), the candidate must too, bit for bit; where the reference
      changed it, the elements either side changed are compared.

    Compared whole, the kept rows diluted the new ones: one wrong row of a 4096-token
    cache was 0.02 % of the elements (within :data:`MAX_MISMATCH`), and at ×0.01 the
    scaled old rows set the element bound (their RMS) of a new row computed from
    normalised activations, which is not scaled (#202). A candidate whose tensor does not
    match ``ref``'s type, shape, dtype, layout or device is compared whole (the error)."""
    kw: dict[str, Any] = {"tier": tier, "perturbed": perturbed, "input_scale": input_scale}
    if (
        type_error(new) is not None
        or new.shape != ref.shape
        or new.dtype != ref.dtype
        or new.layout != ref.layout
        or new.device.type != ref.device.type
    ):
        return compare_tensors(name, ref, new, **kw)
    kept, total = before.shape[dim], ref.shape[dim]
    ref, before = ref.detach().to(new.device), before.detach().to(new.device)
    new = new.detach()
    appended = total - kept
    channels = _channels(ref.float()) if perturbed else None
    if channels is not None and dim == ref.dim() - 1:  # grown along the channels themselves
        channels = channels.narrow(dim, kept, appended)
    result = compare_tensors(
        name,
        ref.narrow(dim, kept, appended),
        new.narrow(dim, kept, appended),
        channel_rms=channels,
        **kw,
    )
    result["grown"] = {"dim": dim, "kept": kept, "appended": appended}
    rows = f"{appended} appended row{'s' if appended != 1 else ''} along dim {dim}"
    problems = [f"the {rows} (after {kept} kept): {result['error']}"] if "error" in result else []
    ref_old, new_old = ref.narrow(dim, 0, kept), new.narrow(dim, 0, kept)
    unchanged = _same(ref_old, before)
    if bool(unchanged.all()):  # the reference only appended: the kept part stays bit for bit
        moved = int((~_same(new_old, before)).sum())
        result["kept_changed"] = moved
        if moved:
            problems.append(
                f"{moved} of the {before.numel()} elements of the {kept} kept rows along dim "
                f"{dim} changed; the reference keeps them as they were (it only appends)"
            )
    else:  # the reference changed the kept part too: compare the elements either side changed
        changed = ~unchanged | ~_same(new_old, before)
        count = int(changed.sum())
        old = compare_tensors(name, ref_old[changed], new_old[changed], **kw)
        result["changed_elements"] = count
        if not old["ok"]:
            problems.append(f"the {count} changed elements of the {kept} kept rows: {old['error']}")
    result.pop("error", None)
    if problems:
        result["error"] = "; ".join(problems)
    result["ok"] = not problems
    return result


def written_box(before: Any, after: Any) -> list[torch.Tensor] | None:
    """Per dimension, the indices of the box of elements a call wrote into ``before`` to give
    ``after`` of the same shape (a static KV cache returned by a functional ``index_copy`` /
    ``scatter`` of the new token's K / V: the token's slot along the sequence dimension, every
    KV head and channel). None unless both are floating-point strided tensors of one dtype
    and shape, they differ (NaN as NaN) and the elements that differ are written rows: they
    fill at least :data:`WRITTEN_MIN_DENSITY` of their box (along each dimension, the indices
    where one differs), which spans a whole dimension of more than one element (rows are
    vectors, not the few scalars of a short output that a residual add's rounding left as
    they were) and at most :data:`WRITTEN_MAX_FRACTION` of the tensor."""
    if not (isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor)):
        return None
    if (
        not before.is_floating_point()
        or before.dtype != after.dtype
        or before.shape != after.shape
        or before.layout != torch.strided
        or after.layout != torch.strided
        or before.dim() == 0
        or before.numel() == 0
    ):
        return None
    changed = ~_same(after.detach(), before.detach().to(after.device))
    count = int(changed.sum())
    if count == 0:  # the input itself: compared whole
        return None
    hits = [changed.movedim(d, 0).reshape(n, -1).any(1) for d, n in enumerate(changed.shape)]
    sizes = [int(h.sum()) for h in hits]
    box = math.prod(sizes)
    if (
        box > WRITTEN_MAX_FRACTION * changed.numel()
        or count < WRITTEN_MIN_DENSITY * box
        or not any(s == n > 1 for s, n in zip(sizes, changed.shape, strict=True))
    ):
        return None
    return [h.nonzero().flatten() for h in hits]


def compare_written(
    name: str,
    before: torch.Tensor,
    ref: torch.Tensor,
    new: torch.Tensor,
    box: list[torch.Tensor],
    *,
    tier: str | None = None,
    perturbed: bool = False,
    input_scale: float = 1.0,
) -> dict[str, Any]:
    """``new`` against ``ref``, both ``before`` with rows written into it (:func:`written_box`:
    ``box``, the indices per dimension of the elements the reference changed; a static KV
    cache returned by a functional ``index_copy``), as grown caches are
    (:func:`compare_grown`):

    * the written part (the box) on its own, with the tier's checks (its own RMS, mismatch
      fraction, norm, cosine and relative L2 error; on redrawn inputs an element's channel
      RMS is at least its channel's in the whole reference);
    * the rest, which the reference kept as it was, bit for bit.

    Compared whole, one wrong written row of a 4096-slot cache was 0.02 % of the elements,
    inside :data:`MAX_MISMATCH` (#206). A candidate whose tensor does not match ``ref``'s
    type, shape, dtype, layout or device is compared whole (the error)."""
    kw: dict[str, Any] = {"tier": tier, "perturbed": perturbed, "input_scale": input_scale}
    if (
        type_error(new) is not None
        or new.shape != ref.shape
        or new.dtype != ref.dtype
        or new.layout != ref.layout
        or new.device.type != ref.device.type
    ):
        return compare_tensors(name, ref, new, **kw)
    ref, before = ref.detach().to(new.device), before.detach().to(new.device)
    new = new.detach()
    box = [index.to(new.device) for index in box]

    def part(t: torch.Tensor) -> torch.Tensor:
        for d, index in enumerate(box):
            if index.numel() < t.shape[d]:
                t = t.index_select(d, index)
        return t

    channels = _channels(ref.float()) if perturbed else None
    if channels is not None and box[-1].numel() < ref.shape[-1]:  # written along the channels
        channels = channels.index_select(-1, box[-1])
    result = compare_tensors(name, part(ref), part(new), channel_rms=channels, **kw)
    sizes = [int(index.numel()) for index in box]
    result["written"] = {"box": sizes, "of": list(ref.shape)}
    written = f"written {sizes} of {list(ref.shape)}"
    where = f"the {written} (where the reference changed the input)"
    problems = [f"{where}: {result['error']}"] if "error" in result else []
    inside = torch.ones((), dtype=torch.bool, device=new.device)
    for d, index in enumerate(box):
        hit = torch.zeros(ref.shape[d], dtype=torch.bool, device=new.device)
        hit[index] = True
        inside = inside & hit.view([-1 if i == d else 1 for i in range(ref.dim())])
    moved = int((~_same(new, before) & ~inside).sum())
    result["kept_changed"] = moved
    if moved:
        problems.append(
            f"{moved} of the {ref.numel() - math.prod(sizes)} elements outside the {written} "
            "changed; the reference keeps them as they were (it only writes the box)"
        )
    result.pop("error", None)
    if problems:
        result["error"] = "; ".join(problems)
    result["ok"] = not problems
    return result


def compare_output(
    name: str,
    ref: torch.Tensor,
    new: Any,
    inputs: Any = (),
    *,
    tier: str | None = None,
    perturbed: bool = False,
    input_scale: float = 1.0,
) -> dict[str, Any]:
    """One output tensor against its reference: :func:`compare_grown` when the reference is
    one of the call's input tensors ``inputs`` (pre-call) grown along one dimension (its
    leading part there equal to that input, NaN as NaN: a returned ``torch.cat`` of a cache
    and the new rows), :func:`compare_written` when it is one of them with rows written into
    it (:func:`written_box`: a returned functional ``index_copy`` into a static cache), else
    :func:`compare_tensors`."""
    kw: dict[str, Any] = {"tier": tier, "perturbed": perturbed, "input_scale": input_scale}
    for before in inputs:
        dim = grown_dim(before, ref)
        if dim is None:
            continue
        lead = ref.detach().narrow(dim, 0, before.shape[dim])
        if bool(_same(lead, before.detach().to(lead.device)).all()):
            return compare_grown(name, before, ref, new, dim, **kw)
    for before in inputs:
        box = written_box(before, ref)
        if box is not None:
            return compare_written(name, before, ref, new, box, **kw)
    return compare_tensors(name, ref, new, **kw)


def compare_structures(
    ref: Any,
    new: Any,
    prefix: str = "out",
    *,
    inputs: Any = None,
    tier: str | None = None,
    perturbed: bool = False,
    input_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """The candidate's output ``new`` against the reference's ``ref``, tensor by tensor
    (:func:`compare_output`); ``inputs``: the call's inputs before the call (e.g.
    ``(args, kwargs)``), for outputs that grow one of them (:func:`compare_grown`) or write
    rows into one (:func:`compare_written`)."""
    ref_flat = flatten(ref, prefix)
    new_flat = flatten(new, prefix)
    sources = list(flatten(inputs, "in").values())
    kw: dict[str, Any] = {"tier": tier, "perturbed": perturbed, "input_scale": input_scale}
    results = []
    for name, tensor in ref_flat.items():
        if name not in new_flat:
            results.append({"name": name, "ok": False, "error": "missing in candidate output"})
            continue
        results.append(compare_output(name, tensor, new_flat[name], sources, **kw))
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
    forgetting to, would vanish inside the allowance.  Caches that the call grows
    along one dimension (concatenation, :func:`grown_dim`) likewise: the appended
    rows on their own, the kept part unchanged where the reference kept it
    (:func:`compare_grown`).  Other tensors are compared whole."""
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
        if before is not None and (dim := grown_dim(before, ref)) is not None:
            results.append(compare_grown(name, before, ref, new, dim, **kw))
            continue
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
