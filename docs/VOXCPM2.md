# VoxCPM2 on an RTX 5070 Ti: 7.4× faster with kernel-agent

The motivating case for the roadmap ([ROADMAP.md](ROADMAP.md), tracking issue
[#24](https://github.com/kadirnar/kernel-agent/issues/24)): make
`openbmb/VoxCPM2` (voxcpm 2.0.3, bf16) fast on one RTX 5070 Ti (16 GB, sm_120,
torch 2.14.1+cu130) with the continuous `kernel-agent improve` loop.

## Result

Benchmark workload (built-in VoxCPM workload: 60 patches = 9.6 s of 48 kHz
audio, 10 flow-matching steps, CFG 2.0, seed 0):

| configuration | latency per run | vs eager | vs VoxCPM's own `torch.compile` |
|---|---|---|---|
| eager PyTorch (`optimize=False`) | 5,407 ms | 1.00× | 0.70× |
| VoxCPM `optimize()` (`torch.compile`, reduce-overhead) | 3,798 ms | 1.42× | 1.00× |
| kernel-agent, round 1 (integrated) | 891 ms | 6.07× | 4.26× |
| **kernel-agent, final (paired A/B integration)** | **731 ms** | **7.40×** | **5.20×** |

Real use through VoxCPM2's own `generate()` at natural length
(`examples/voxcpm2_optimized.py`, three English sentences, the exported
`optimized/` package applied to a freshly loaded model):

| | total time | real-time factor | Whisper-large-v3 WER |
|---|---|---|---|
| eager | 8.44 s | 0.57 | 0.08 / 0.00 / 0.00 |
| optimised | 1.13 s | 0.077 | 0.08 / 0.00 / 0.00 |

(The 0.08 is Whisper spelling "optimisation" as "optimization".) Audio lengths
match (4.6–5.0 s per sentence).

Cost: 19 agent sessions, **$31.67** of Claude usage, 4 h 41 min wall clock
(two `improve` rounds), 113 ledger rows.

## What made it fast

The final integration (paired in-process A/B, each step ≥ 80 % of 8 rounds
won and a bootstrap 95 % CI of the gain > 1 %) accepted four items:

| item | kind | what it does | gain when added |
|---|---|---|---|
| `decoder_layer_fused` | CUDA C++ kernel (agent-written) | the whole MiniCPM decoder layer's `forward_step` as one cooperative kernel per layer with 4 grid syncs: RMSNorm, QKV GEMV, RoPE + KV-cache write + split-KV attention over the valid positions only, o-proj + residual, RMSNorm, gate/up GEMV with SiLU, down-proj + residual; bf16 rounding after each op like the reference; 97 % of the speed of light at module level | −4,399 ms |
| `hoist_cfm_invariants` | transform | moves loop-invariant work out of the flow-matching Euler solver and captures the solver as a CUDA graph | −247 ms |
| `skip_dead_work` | transform | skips work whose results are never used (LM step after the last patch, stop head before `min_len`, LocEnc over an all-zero prefill mask — checked at run time) | −22 ms |
| `vae_channels_last` | transform | AudioVAE decoder in a channels-last layout (no cuDNN layout transposes), `weight_norm` folded once instead of every decode, depthwise causal convs + Snake as fused stencils | −16 ms |

The decisive insight came from the profile kernel-agent produced once
`forward_step` entrypoints were visible (#1): 45 % of the eager run's GPU time
was a masked SDPA over the whole 8,192-slot static KV cache while ~70 slots
were valid, and the run was launch-bound (47 % GPU busy, 507k launches).

![integration](images/voxcpm2-integration.png)

![progress](images/voxcpm2-progress.png)

## What the roadmap work caught along the way

* **Every correct kernel was rejected at first.** Free-running TTS output is
  chaotic under bf16 rounding changes (the bundled Triton RMSNorm scored
  spectral cosine 0.69 against a 0.97 gate). Teacher-forced validation (#2)
  fixed that; a natural-length run now also checks the stop condition (#55).
* **The decode path was invisible** to module hooks (`forward_step`, #1). The
  same work found that the side-effect check missed KV-cache writes entirely.
* **Integration bugs found on this run**: the systems arm's gains on top of
  kernels were not credited (#40) and transforms evaluated only with kernels
  never reached integration (#43; 929 → 847 ms once fixed). Noisy single
  measurements were replaced by paired A/B (#11).
* **The evaluator was gameable** (12 of 18 known exploit classes in v0.1.0);
  #5–#8 close them, and a memoising transform that would have reported
  285,656× is now rejected. Re-checking the winners with the hardened
  evaluator showed the attention kernel's module speedup is 18.8×, not the
  46.9× the old evaluator recorded. Stale records are now re-evaluated by the
  current evaluator before the re-check (#58): the attention kernel's record is
  18.3×.

## Reproduce

```bash
uv sync --extra all --extra voxcpm
uv pip install --no-deps "voxcpm>=2.0.3"
uv pip install --no-deps torchaudio --index-url https://download.pytorch.org/whl/cu130
uv run --no-sync kernel-agent improve openbmb/VoxCPM2 --max-hours 6 --max-usd 40 \
    --slice 4 --agent-minutes 60 --rounds 2 --speedup-goal 4
uv run --no-sync kernel-agent watch runs/openbmb--VoxCPM2/<run>      # live dashboard
uv run --no-sync python examples/voxcpm2_optimized.py runs/openbmb--VoxCPM2/<run>/optimized
```

Results depend on the GPU, driver and torch version; the agents' kernels are
written for the GPU they run on.
