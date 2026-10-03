# Optimisation playbook

## 1. Diagnose before you write code

Classify every hot spot before choosing a technique:

| Regime | Symptom | What wins |
|---|---|---|
| **Launch / CPU bound** | GPU kernel time is a small share of latency; thousands of kernels of 1-5 us; decode loops with batch 1 | Fuse many small ops into one kernel, fewer Python ops per step, CUDA graphs, static KV cache, remove host syncs (`.item()`, `.cpu()`, Python control flow on tensors) |
| **Memory bound** | Elementwise / norm / softmax / GEMV (M=1..16) / attention decode; arithmetic intensity < ~50 FLOP/byte | Read every byte once: fusion, vectorised 128-bit loads, keep intermediates in registers/SMEM, lower precision weights, split-K / split-KV for parallelism |
| **Compute bound** | Large GEMM / conv / prefill attention; tensor-core kernels dominate | Tensor cores with the right MMA, tiling + pipelining (cp.async / TMA), epilogue fusion (bias, activation, residual, quantisation), FlashAttention-style tiling |

Roofline numbers: achieved bandwidth = bytes moved / time; compare with the GPU
peak (consumer Blackwell RTX 50xx ≈ 0.9 TB/s on a 5070 Ti, Hopper ≈ 3.3 TB/s).
If a memory-bound kernel already reaches > 80 % of peak bandwidth, stop
optimising it and fuse it with its neighbours instead.

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
