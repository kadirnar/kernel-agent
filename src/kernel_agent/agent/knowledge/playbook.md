# Optimisation playbook

## 1. Diagnose before you write code

Classify every hot spot before choosing a technique:

| Regime | Symptom | What wins |
|---|---|---|
| **Launch / CPU bound** | GPU kernel time is a small share of latency; thousands of kernels of 1-5 us; decode loops with batch 1 | Fuse many small ops into one kernel, fewer Python ops per step, CUDA graphs, static KV cache, remove host syncs (`.item()`, `.cpu()`, Python control flow on tensors) |
| **Memory bound** | Elementwise / norm / softmax / GEMV (M=1..16) / attention decode; arithmetic intensity < ~50 FLOP/byte | Read every byte once: fusion, vectorised 128-bit loads, keep intermediates in registers/SMEM, lower precision weights, split-K / split-KV for parallelism |
| **Compute bound** | Large GEMM / conv / prefill attention; tensor-core kernels dominate | Tensor cores with the right MMA, tiling + pipelining (cp.async / TMA), epilogue fusion (bias, activation, residual, quantisation), FlashAttention-style tiling |

Roofline numbers: the evaluator does this for you. Peak copy bandwidth (DRAM
and L2), dense matmul TFLOP/s per dtype and the launch floor are measured on
this GPU (see "measured peaks" in the toolchain section), and every timed
result reports per case `flops`, `min_bytes` (what the reference must read and
write at least once), `sol_ms` = max(FLOPs / peak, bytes / bandwidth),
`pct_of_sol` and `bound`. Achieved bandwidth = `min_bytes` / `new_ms`. A
`memory` case at ≥ 80 % of SOL is done: fuse it with its neighbours instead.
For a `launch` case the work is smaller than one launch from Python: only
fewer launches and less host work per call help. Cases that fit in L2
(`l2_resident`) are compared with the L2 bandwidth, because the benchmark runs
them with a warm cache.

## 2. Hot paths by model family

**LLM (and LLM-style TTS like Orpheus / CSM / Qwen-TTS talkers, STT decoders)**
* Decode (batch 1, 1 token) is launch + memory bound. Per layer HF eager runs
  ~40 kernels: RMSNorm (≈6 kernels), q/k/v/o GEMV, RoPE (cat/neg/mul/add),
  attention, residual adds, SwiGLU (silu, mul), down GEMV.
* Biggest wins: fused `RMSNorm(+residual add)` (1 kernel instead of ~7);
  fused RoPE applied to q and k in one kernel; fused `silu(gate) * up`; merged
  QKV and gate/up projections (one GEMV instead of 3 / 2 — build fused weights in
  `build()` once); decode attention with split-KV (flash-decoding); whole
  attention block or whole MLP block as one module replacement.
* Prefill is compute bound: cuBLAS / FlashAttention are already strong; win by
  fusing epilogues and avoiding layout copies, not by rewriting plain GEMMs.

**Diffusion (UNet / DiT / Flux-style transformers + VAE)**
* Attention at long sequence lengths (FlashAttention-style kernels, fused QK-norm
  + RoPE), GroupNorm + SiLU fusion, GEGLU / GELU-tanh fusion with the projection
  epilogue, AdaLN modulation (`x * (1 + scale) + shift`) fused into the norm,
  conv epilogues, VAE decoder convs (channels_last).
* Steps repeat identical shapes → CUDA graphs on the denoiser are very effective.

**STT (Whisper, Parakeet, wav2vec2)**
* Encoder = prefill-like (conv stem + attention + MLP, GELU), decoder = LLM decode.
* Fuse conv1d + GELU, LayerNorm (+ residual), attention; static KV cache for the
  decoder with cross-attention K/V computed once.

**Diffusion-autoregressive TTS (VoxCPM2 measurements)**
RTX 5070 Ti, bf16, 60 patches (9.6 s of audio), 10 flow-matching steps, CFG
2.0 (docs/RESEARCH.md §5). Per audio patch: AR LM decode (MiniCPM base +
residual LM, batch 1, one position, through `forward_step`), the LocDiT
flow-matching sampler (Euler; the guided steps run cond + uncond as batch 2,
`[2, 11, 1024]` per DiT attention call), the LocEnc that feeds the patch back,
and once per run the AudioVAE vocoder.
* 5.46 s eager, 3.80 s with VoxCPM's own `torch.compile(reduce-overhead)`;
  GPU busy 47 %, 507k kernel launches per run (5 µs average): launch bound
  before anything else.
* Decode attention is the top kernel: `fmha_cutlassF` 1151 ms, 45 % of GPU
  time. `MiniCPMAttention.forward_step` attends over all 8192 slots of the
  static KV cache with a mask while ~70 hold data (prompt + generated
  patches: 17..36 of 8192 slots with 20 patches). Attend over the valid
  length only (`position_id + 1` slots: slice the cache or use a split-KV
  decode kernel that stops there). `workload_profile.md` gives the valid range.
* The LocDiT runs 9 guided Euler steps per patch (the first of the 10 is
  zero-initialised), each a batch-2 pass through 12 layers: 108 attention
  calls on `[2, 11, 1024]` per patch, 68 % of all `MiniCPMAttention` calls
  in a 20-patch capture (LM decode: 23 %, 36 layers × 1 position). Such
  tiny problems are launch bound: fuse the whole layer (norm, QKV, RoPE,
  attention, o-proj, MLP) into a few launches and keep cond + uncond in one
  batch-2 call. One class serves both: split `MiniCPMAttention` into a
  `decode` target (`forward_step`) and a `prefill` target (`forward`, the DiT).
* CUDA graphs per step remove most launches, but teacher forcing (the only
  valid quality check: one-ulp changes diverge the free run after ~18 steps,
  spectral cosine 0.69 for a verified RMSNorm) wraps `model.feat_decoder`
  from Python once per patch and needs its noise from `torch.randn`. Graph
  the LM step and the DiT estimator separately, never the whole step (LM +
  decoder): such a transform cannot be validated and is rejected.
* Host syncs per patch: the loop reads the stop flag with `.cpu().item()` on
  every patch (even before `min_len`) and builds positions with
  `torch.tensor([kv_cache.step()], device=...)` twice per patch. Keep
  positions on the device and drop the sync where the loop does not need it.

**TTS vocoders / codecs (HiFi-GAN, Vocos, DAC, SNAC, Mimi)**
* Conv1d / ConvTranspose1d with small channel counts, Snake / LeakyReLU
  activations, residual stacks: fuse activation + conv, use channels-last 1D
  layouts, fuse iSTFT post-processing. Many tiny kernels → launch bound.

## 3. Algorithm-level changes (model transforms)

These change *how* the model runs, not the math, and often beat any kernel:
* `model.generation_config.cache_implementation = "static"` + `torch.compile` /
  CUDA graphs for decode loops (removes launch overhead almost entirely).
* `torch.backends.cuda.matmul.allow_tf32`, SDPA backend choice
  (`torch.nn.attention.sdpa_kernel`), `channels_last` for conv nets.
* Precompute constant tensors (rotary tables, masks, positional embeddings).
* Fusing linear layers that share an input (QKV, gate/up) — done in `build()`.
* Avoid recomputation (cache encoder output, cross-attention K/V).
* Exact speculative / lookahead decoding keeps greedy outputs identical.
Quality gates still apply end to end; anything that changes outputs beyond the
workload's tolerance is rejected.

## 4. Engineering loop

1. Read the captured module's source (`spec.json → source_file`) and the
   captured shapes/dtypes. Write down the exact math, including dtype casts.
2. First candidate: simplest correct fused kernel. Evaluate.
3. Then optimise with evidence: run `evaluate_candidate` with `profile=true`
   to see which kernels remain and how long each takes.
4. Cover every captured case (prefill AND decode shapes). Specialise per shape
   inside `forward` if needed (e.g. GEMV path for M ≤ 16, tensor-core path for
   larger M). Fall back to the reference math for shapes you do not handle.
5. Keep the best version; record what you learned in `NOTES.md`.

## 5. Host overhead matters for tiny kernels

At decode shapes a kernel runs for 2-10 us, so Python-side launch cost is
visible in the measurement (it is real latency too). Measured on this machine
for a 2048-wide RMSNorm: CUDA C++ via `load_inline` ≈ 19 us, NVRTC ≈ 28 us,
CuTe DSL with TVM-FFI ≈ 25 us, TileLang ≈ 32 us, Triton ≈ 43 us, HF eager
(6 kernels) ≈ 74 us. Therefore: fuse *more* work per launch (a whole block, not
one op), avoid per-call Python work (no `from_dlpack`, dict lookups, shape
math or allocations you can hoist), and cache compiled kernels per shape.
