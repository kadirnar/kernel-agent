---
name: correctness-and-anti-gaming
description: What the evaluator checks before a kernel or transform counts (cases, side effects, state, redrawn and scaled inputs, held-out input, memoisation, hidden work, patching, fallbacks, memcheck, re-check) and the reward hacks it rejects. Use to review a candidate or understand a failed check.
---

# Correctness and anti-gaming: what counts, what is rejected

A result counts only when the evaluation tool reports it correct; nothing here is optional
or negotiable. Use the list as a review checklist before spending an evaluation, and to read
a failure (`stage`, `failed_check`, `integrity_violation`, `fallback`). Details and the
fixtures that prove each check: the repository README ("What correct means", "Anti-gaming
guards", "Independent re-check of winners", "Out-of-bounds accesses: memcheck").

## The rules every agent gets

* Never call the reference module's `forward` (or an inherited / `super()` forward) for the
  captured cases, never return cached or captured outputs, never skip work the reference
  does, never lower the precision beyond the target's tier (a reduced precision only where
  the target's spec says so: the `precision-tiers` skill).
* Fallbacks only for shapes / dtypes the kernel does not support, and they must be the
  reference math; the main captured cases run the kernel.
* Never write code that targets the benchmark instead of inference: clock burn-in loops,
  outputs cached across runs, work skipped when inputs repeat, constants calibrated on the
  captured values.

## Module kernels (`evaluate_candidate`)

* **Captured cases**: outputs and in-place side effects (e.g. KV-cache writes) within the
  dtype's tolerances (bf16 2e-2, fp16 1e-2, fp32 1e-4; at most 0.1 % of elements outside,
  none by more than 10×), whole-tensor relative L2 and cosine for tensors above 1024
  elements, NaN / inf at the same positions, plain `torch.Tensor` outputs, aliasing of
  outputs and inputs exactly as the reference's. Handing back the reference unchanged is
  rejected.
* **Several KV lengths and settings**: the first, middle and last decode step of every
  decode signature are separate cases, and `variants()` add correctness-only cases (other
  lengths and settings): a kernel that only handles the captured length fails them.
* **Module state** outside the arguments (a KV-cache attribute, a step counter): each case
  runs from the state its call saw and the state it changes is compared (`state.*`); keep
  state where and in the format the reference keeps it.
* **Beyond the captured values**: inputs rotate between copies during timing and one timed
  call runs on redrawn values (`incorrect_timed_output`); after timing every case runs at
  fresh addresses, with its floating-point inputs redrawn in place (normal, uniform,
  Laplace, log-normal) and × 3, × 0.01 and × −1 (`scaled_x3`, `scaled_x0.01`,
  `sign_flipped`), each against the live reference (`incorrect_perturbed`). Outputs cached
  by address, shape or call count, skipped work, reads of unused cache slots, absolute
  epsilons and static activation scales fail here.
* **Integrity** (`integrity_violation`): patching the timer, the comparator, torch
  functions, `nn` module forwards, the reference's methods or weights, or backend flags
  (TF32, SDPA backends, matmul precision, default dtype, current stream, dispatch modes);
  threads running candidate code; **hidden work** (GPU work on streams or threads the
  timer does not see: wall time above event time, a side stream never joined, launches
  from another thread). Joined side streams are fine; declare them with
  `kernel_agent.concurrency` (the `cuda-graphs-streams-pdl` skill).
* **Fallback detection** (`fallback`): on the dominant case, running the reference's
  entrypoint code with less than half of the GPU time in own kernels, or no own kernels
  while launching every kernel of the reference. Pure-torch restructurings that launch
  fewer kernels (merged q/k/v or gate/up weights built once in `build()`) pass.
* **Peak memory**: a warning when a case's peak rises above 25 % of the reference's and
  16 MiB.
* The reference time is checked against a candidate-free subprocess, and outputs are
  compared again by the parent process.

## Model level (`evaluate_e2e`, integration)

* The workload's own comparison (identical greedy tokens with near-tie tolerance and a
  first-step logits cosine for LLM / STT, spectral cosine for TTS, PSNR for diffusion;
  teacher forcing for chaotic workloads; the perceptual gate in near-lossless and relaxed
  runs).
* **Held-out input** (another prompt / text / seed, untimed) and a **memoisation probe**
  (a fresh input timed after warm-up must not be over 3× slower than the repeated runs;
  equal outputs where the baseline's differ fail).
* **Stop condition**: a teacher-forced natural-length run must stop at the baseline's step.
* **Diverse input set**: every passing candidate also runs other content at the same
  shapes; a speedup that varies with the content is labelled `data_dependent`, never
  rejected for it (the `speculative-decoding` skill).
* **Concurrency**: no GPU work may outlive `run()`; side streams joined before it returns.
* **Undo**: transforms rebind weights (`param.data = new`), never mutate them in place, so
  the paired A/B can switch them off.

## Winners

* **Memcheck** at integration: every kept kernel runs under compute-sanitizer memcheck
  with odd-size variants; unmasked tail loads (rows past `M % BM`) are refused: clamp or
  mask every row-indexed load, epilogues included.
* **Re-check**: winners are re-verified in fresh processes on fresh inputs, reference and
  candidate apart (`kernel-agent recheck`), and timed again.

## Reward hacks seen in kernel-generation agents

Delegating to torch / the reference, returning constants or the identity, caching outputs,
lazy outputs computed after the timer stops, side streams the timer does not see, patched
timers or tolerances, shrinking the problem, skipping masked ops that test inputs happen not
to need (an all-positive input hides a missing ReLU), calibrating on the test inputs. The
papers and checkers: the `documentation-sources` skill's `sources.md`, section
"Kernel-generation agents: failures and reward hacks (for reviews of a candidate)".

## Examples and sources

* Code: `kernel_agent.kernels.evaluate`, `kernel_agent.kernels.integrity`,
  `kernel_agent.kernels.compare`, `kernel_agent.kernels.recheck`,
  `kernel_agent.kernels.memcheck`; exploit fixtures in `tests/test_evaluator_exploits.py`.
