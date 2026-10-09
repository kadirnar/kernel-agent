# The reference's own rounding spread (#250)

RTX 5070 Ti (sm_120), torch 2.14.1+cu130. Chains from `kernel_agent.selftest`, hidden 1024,
one row per call: `NormGemvChain` ("norm": RMSNorm → GEMV → residual, the megakernel
example's reference) and `GemvChain` ("plain": the PDL example's).

## What changed

A check on redrawn inputs that fails the plain exact tolerance (the timed-output check, the
perturbed and scaled inputs, the integration's re-check) is judged again:

1. The reference runs once more on the same inputs under `verify.Rerounding`. This
   dispatch mode multiplies every rounded aten result by `1 + u·ξ` (`u` the dtype's unit
   roundoff, `ξ` uniform in [-1, 1]) and rounds it back. That moves about a quarter of the
   elements by one ulp. Views, copies and indexing stay exact.
2. `spread` is the RMS of `rerounded − ref`: the larger of the tensor's and the element's
   channel's.
3. If the candidate's RMS error is at most `ROUNDING_SPREAD_RMS = 2` × the spread, each
   element's tolerance grows by `ROUNDING_SPREAD_K = 4` × the spread. The growth is capped
   at `ROUNDING_SPREAD_CAP = 0.125` × the reference's RMS.
4. The 0.1 % mismatch budget, the 10× outlier rule, the whole-tensor checks and the
   captured-input checks are unchanged.

## Calibration (`calibrate.py` → `summarize.py` → `calibrate.txt`; 200 seeded draws per row)

Columns:

* **spread / RMS**: the median, relative to the output's RMS.
* **eager vs fp32**: eager's own error against an fp32 recompute.
* **candidate error / spread**: the maximum RMS error, in units of the spread.
* **K for every element**: the maximum `K` with every element within `tol + K × spread`.
* **past the plain tolerance**: draws with more than 0.1 % of elements outside.

| chain, layers | candidate | spread / RMS | eager vs fp32 | candidate error / spread | K for every element | past the plain tolerance |
|---|---|---|---|---|---|---|
| norm 2 | megakernel / `torch.compile` | 0.65 % | 0.37 % | 0.09 / 0.69 | ≤ 0 / 0.20 | 0 / 0 |
| norm 4 | megakernel / `torch.compile` | 0.90 % | 0.48 % | 0.34 / 0.63 | ≤ 0 / 0.84 | 0 / 102 |
| norm 8 | megakernel / `torch.compile` | 1.22 % | 0.63 % | 0.47 / 0.60 | 0.26 / 1.43 | 3 / 200 |
| norm 16 | megakernel / `torch.compile` | 1.64 % | 0.83 % | 0.50 / 0.62 | 0.96 / 1.75 | 15 / 200 |
| norm 28 | megakernel / `torch.compile` | 2.10 % | 1.04 % | 0.55 / 0.60 | 1.42 / 2.16 | 35 / 200 |
| plain 28 | PDL example / `torch.compile` | 1.86 % | 0.88 % | 0.77 / 0.58 | 1.41 / 0.77 | 64 / 101 |

The graph + PDL and grid-barrier modes are within the megakernel's numbers (identical to
each other).

What the check must keep rejecting:

* **Spread units (`K`)**: the `K` at which the output would pass.
* **RMS units**: the widening, in units of the output's RMS, at which it would pass. The
  cap is 0.125.

| output, minimum over all rows | spread units | RMS units |
|---|---|---|
| the previous draw's output | 161 | 3.2 |
| a stale 16-element tile | 53 | 1.0 |
| the last layer skipped | 22 (norm 28) | 0.46 (norm 28) |

## The timed-output check, before and after (`timed_check.py`, `timed_*.json`)

* **Before**: `bench.check_timed_output` with the spread off.
* **After**: as merged. A draw is only judged again when it fails before.
* **Cheats**: the previous draw's output, a stale 16-row tile and the last layer skipped,
  tried on every 10th draw.

| chain, layers, draws | megakernel | graph + PDL | grid barrier | `torch.compile` | PDL example | cheats rejected |
|---|---|---|---|---|---|---|
| norm 4, 1,000 | 0 → 0 | 0 → 0 | 0 → 0 | 490 → 0 | | 297 / 297 |
| norm 8, 3,000 | 53 → **0** | 49 → **0** | 49 → **0** | 3,000 → **0** | | 897 / 897 |
| norm 16, 1,000 | 84 → **0** | 87 → **0** | 87 → **0** | 1,000 → **0** | | 297 / 297 |
| norm 28, 1,000 | 171 → **0** | 168 → **0** | 168 → **0** | 1,000 → **0** | | 297 / 297 |
| plain 8, 1,000 | | | | 0 → 0 | 0 → 0 | 297 / 297 |
| plain 28, 1,000 | | | | 447 → **0** | 342 → **0** | 297 / 297 |

## The evaluator, before and after (`eval_loop.py`, `eval_*.txt`)

The megakernel example, on the example's capture with 8 layers instead of 4. Before is
`origin/main` at 12ddbc2.

| | evaluations | refused |
|---|---|---|
| before | 100 | 4: 2 `incorrect_timed_output` (1.46 % and 0.20 % outside), 2 `incorrect_perturbed` |
| after | 100 | **0** |

## Decoder blocks (`blocks.py`, `blocks.txt`; 200 draws each)

The blocks are RMSNorm → qkv → SDPA with an in-place KV cache → o → residual, then RMSNorm
→ SiLU MLP → residual. The honest candidate is `torch.compile`. "Spread" is the spread
relative to the RMS; "error" is the candidate's relative L2 error.

| layers, tokens, cache | rejected before → after | spread | error | previous draw's output rejected |
|---|---|---|---|---|
| 1, 1, 512 | 0 → 0 | 0.65–0.69 % | 0.40–0.42 % | 39 / 39 |
| 1, 64, 64 | 0 → 0 | 0.77–0.79 % | 0.51–0.52 % | 39 / 39 |
| 4, 1, 512 | 200 → **0** | 1.43–1.55 % | 0.83–0.93 % | 39 / 39 |
| 4, 64, 64 | 200 → **0** | 1.61–1.63 % | 1.02–1.05 % | 39 / 39 |

## What is newly accepted (`systematic.py`, `systematic.txt`; 100 draws)

Each draw takes the reference's output and adds one systematic error (or random noise of
that RMS). The table gives draws rejected before → after, with one row / 64 rows per call
where they differ, then the median error / spread.

| error | 1 layer | 8 layers | 28 layers |
|---|---|---|---|
| bias 1 % of the RMS | 0 → 0 (2.3) | 100 → 0 (0.84) | 100 → 0 (0.50) |
| bias 2 % | 100 → 100 (4.6) | 100 → 0 / 100 (1.7) | 100 → 0 (0.95) |
| bias 4 % | 100 → 100 | 100 → 100 (3.3) | 100 → 11 / 100 (1.9) |
| scale 1.03 | 100 → 100 | 100 → 100 (2.4) | 100 → 0 / 100 (1.4) |
| scale 1.05 | 100 → 100 | 100 → 100 | 100 → 100 (2.4) |
| noise 1 % | 100 → 99 / 100 (2.2) | 100 → 0 (0.83) | 100 → 0 (0.48) |
| noise 2 % | 100 → 100 (4.4) | 100 → 36 / 98 (1.6) | 100 → 0 / 87 (0.96) |
| the last layer skipped | 100 → 100 (155) | 100 → 100 (27) | 100 → 100 (8.9) |
| the previous output (99 draws) | 99 → 99 (311) | 99 → 99 (115) | 99 → 99 (66) |

Only errors within about twice the spread pass. At 8 layers that is a bias up to ~2 % of the
RMS. Eager's own error against fp32 is 0.63 % there.
