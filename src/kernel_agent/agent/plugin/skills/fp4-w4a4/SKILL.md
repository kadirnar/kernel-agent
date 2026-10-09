---
name: fp4-w4a4
description: W4A4 kernels (precision fp4_w4a4, NVFP4 / MXFP4 weights and activations, opt-in) on block-scaled FP4 tensor cores — when they pay, the per-token activation quantiser, the epilogue, sensitive layers in FP8, the near-lossless-fp4a tier, F.scaled_mm and CuTe MmaMXF4NVF4Op. Use when a target's spec says fp4_w4a4.
---

# W4A4 (`precision: fp4_w4a4`)

Read the `precision-tiers` skill first; the weight format is the one of `fp4-weights`.

**Opt-in**: a run allows `fp4_w4a4` only when `--precisions` names it (no quality mode
allows a 4-bit class by default). **GPU**: block-scaled FP4 tensor cores, sm_100+
(`tcgen05.mma kind::mxf4nvf4` on sm_100 / sm_103, `mma.sync ... kind::mxf4nvf4.block_scale`
on sm_120 / sm_121); `gpu_arch.PRECISION_NEEDS` refuses it elsewhere with the reason
(Ada sm_89, Hopper sm_90, Ampere): use `fp8_w8a8` / `int8_w8a8` there.

**When**: compute-bound GEMMs only (many rows per weight read, past the FP8 ridge), where the
*Ceilings* table's *W4A4* floor is well below the *W8A8* / *MXFP8* one: the NVFP4 tensor
cores run at about twice FP8's rate (measured on an RTX 5070 Ti: `F.scaled_mm` NVFP4 661
TFLOP/s against cuBLASLt FP8 332, bf16 101). At a few rows per call a GEMM is memory bound:
quantising activations saves nothing there (`fp4_weights`, `fp8_weights`).

**Format** (`kernel_agent.kernels.quant`):

* weights once in `build()`: `quantize_fp4(weight)` (NVFP4: e2m1 codes, two per byte, even
  k in the low nibble; an e4m3 scale per 16 along K; an fp32 tensor scale);
* activations on every call, per token: `quantize_fp4_activations(x)` →
  `(codes [M, K/2], scales [M, K/16] e4m3, outer [M] fp32)`: `outer = amax(|row|) *
  NVFP4_OUTER_STEP` (`1 / 2688` in fp32), block scale `e4m3(min(bmax / (outer * 6), 448))`,
  codes `e2m1(x / (scale * outer))` to nearest even, saturated at ±6 (IEEE divisions, the
  hardware's `cvt.rn.satfinite.e2m1x2.f32`); a block whose scale is 0 gets codes 0. Per
  token needs no grid-wide reduction, so a producer (RMSNorm, `silu(gate) * up`) can quantise
  its own rows; `granularity="tensor"` (one outer scale per call) measured as accurate and
  lets two-level `F.scaled_mm` (`[BlockWise1x16, TensorWise]`) write bf16 itself;
* math: e2m1 x e2m1 with both block scales applied by the tensor core, fp32 accumulation,
  then `acc * outer[m] * tensor_scale (+ bias[n])` once per output, one rounding to bf16:
  `fp4_w4a4_linear` (the reference and fallback; `F.scaled_mm` `BlockWise1x16` with fp32 out
  where it applies: bit for bit the fp32 math of the same codes on sm_120);
* scale layout for the MMA: cuBLASLt's 128 x 4 blocks (`swizzle_fp4_scales`, offsets
  `mx_scale_offset`), the same as CuTe's `tile_atom_to_shape_SF`.

**The scale-rule guard**: define a module-level `quantize_activations(x) -> (codes, scales,
outer)` with the quantiser your kernels use (both examples do): codes packed `[rows, K /
2]`, scales **unswizzled** `[rows, K / 16]` e4m3 (MXFP4 `[rows, K / 32]` e8m0), outer
`[rows]` or one per call. The evaluator rejects the candidate (`stage: scale_rule`) when, on
the captured or a stress input, a block maximum exceeds 6 x scale x outer beyond e4m3's
rounding (6.375: a block scale rounded down, an outer scale below amax / 2688, MXFP4's OCP
floor rule) or a non-zero block gets scale 0 where rounding to nearest gives a subnormal
one. Details (the stress input, a rotating quantiser): [calibration.md](calibration.md).

**Accuracy**: W4A4 moves a GEMM's output about 1.4x as far as FP4 weights alone, and
quantised activations can shrink it by up to ~1 % (e2m1's grid, not the scale rounding:
unrounded block scales give the same; LocDiT gate_proj -0.9 %, q_proj 0.0 %). On the VoxCPM2 LocDiT layer at M = 352
the layer output reaches relative L2 0.080 and norm -3.4 % (its MLP), within the
**near-lossless-fp4a** tier: per output tensor cosine >= 0.96, relative L2 <= 0.28, norm
within ±5 %, every element within 1.5 x RMS + 0.25 x |reference|; on redrawn inputs 0.90 /
0.50 / ±15 % / 3.0 x RMS. `relaxed-fp4a`: 0.94 / 0.36 / ±8 % / 2.0 (redrawn 0.85 / 0.65 / ±18
% / 3.5). Per-layer numbers, MXFP4, rotations and the broken kernels the tiers reject:
[calibration.md](calibration.md).

**Sensitive layers stay FP8**: `fp4_w4a4_sensitivity(module, run)` quantises every
`nn.Linear` of the target alone on its captured inputs and ranks them (W4A4 output relative
L2, norm change, FP8 W8A8's for comparison, FLOP share). On the VoxCPM2 LocDiT the MLP's
gate / up rank first (0.088 vs FP8's 0.019, 49 % of the FLOPs); FP8 for them alone takes the
hidden output from 0.036 / -3.4 % to 0.014 / -0.9 %. **kernel-agent picks the mix**: it
probes every W4A4 target when it captures it (layers that read the same activations, q / k
/ v or gate / up, form one group; the reference math is run on the captured cases with the
top 0, 1, 2, ... groups in 8-bit until the tier passes) and your prompt names the layers to
keep in the GPU's 8-bit class (FP8 W8A8; INT8 W8A8 without FP8 tensor cores), the others
W4A4 (`spec.json` → `w4a4_mix`). Each failed gate moves the next group to 8 bits: 3
evaluations failing the tier near its bounds (cosine >= 0.9) with none passing, or the
end-to-end gate rejecting one of your kernels alone. Follow the mix in your prompt (the
same tier: 8-bit layers are within it); with every group at 8 bits the target's pivot to
its 8-bit class opens a new arm, and per-token outer scales, a rotation, a norm-bias
correction or bf16 for the top group remain for W4A4.

**MXFP4 and rotations**: MXFP4 W4A4 (`fmt="mxfp4"`, e8m0 per 32) fails near-lossless-fp4a on
the LocDiT layer (norm 6.5 %) and is not in torch 2.14's `F.scaled_mm` on sm_120 (B200 /
B300 only): CuTe `MmaMXF4Op` or Triton `tl.dot_scaled` there. A block Hadamard rotation
(`hadamard_rotate`, `rotate=16`: QuaRot / QuTLASS style, fused into the producer of the
activations and applied to the weight once) made NVFP4 worse on the captures measured (LocDiT
gate_proj 0.088 -> 0.126) and helped MXFP4 only on some layers: measure before using one.

**Examples**: `cute_nvfp4_w4a4_gemm.py` (CuTe DSL GEMM on `MmaMXF4NVF4Op`, sm_120a; the
per-token scale and bias in the epilogue) and `triton_nvfp4_w4a4_gemm.py` (a Triton
quantiser writing packed codes and swizzled scales, then `F.scaled_mm` NVFP4; sm_100+).
Speeds and shapes: [measured.md](measured.md).

**Report** `fp4_w4a4_error(weight, codes, scales, tensor_scale, x)` on captured activations
(its `activation_saturation` share is normal: NVFP4 rounds block scales to nearest) and the
evaluator's per-case numbers in `NOTES.md`.

## Examples and sources

* Examples: `cute_nvfp4_w4a4_gemm.py`, `triton_nvfp4_w4a4_gemm.py`, `cute_fp8_blockscaled_gemm.py` (the FP8 kernel it is built from). All in kernel-agent's examples directory (`kernel_agent/agent/examples/`; a session's prompt gives the directory): copy their structure; an example's `ARCHS` names the GPUs it runs on.
* Sources: the `documentation-sources` skill's `sources.md`, sections "Low precision (formats, scaling, accuracy)"; "CUTLASS / CuTe".
* Code: `kernel_agent.kernels.quant` (`quantize_fp4`, `quantize_fp4_activations`, `fp4_w4a4_linear`, `fp4_w4a4_error`, `fp4_w4a4_sensitivity`, `swizzle_fp4_scales`, `hadamard_rotate`); research scripts and raw results: `docs/research-scripts/w4a4-233`.
