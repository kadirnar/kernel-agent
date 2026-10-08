# Hot paths by model family

Part of the `optimisation-playbook` skill.

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
  time (20 patches: 383 ms for 720 calls, while the DiT and LocEnc
  flash-attention kernels take 12.5 ms for 2412 calls).
  `MiniCPMAttention.forward_step` attends over all 8192 slots of the
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
