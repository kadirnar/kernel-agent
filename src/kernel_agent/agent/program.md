# program.md: research-org instructions for the kernel-agent agents

Edit this file to steer the agents without touching code. Each `## <section>`
is appended to the system prompt of the matching agents:

* `## all`: every agent
* `## planner`: picks targets and transforms from the profile
* `## kernel`: one kernel engineer per target (writes `build()` replacements)
* `## systems`: model-level transforms, validated end to end
* `## harness`: writes `harness.py` when no built-in workload runs the model
* `## research`: reviews a target that has plateaued, from a clean context,
  and writes its `plan.md` (no code)
* `## refactor`: moves the ops of a region target out of its parent module's
  code into a new submodule (`rewrite.py`), without changing the math

A heading can name several roles (`## kernel, systems`). Other `##` headings
are ignored with a warning. Text above the first `##` heading (this
paragraph) and `<!-- HTML comments -->` never reach an agent.

`--program FILE` copies a file to `<run>/program.md` (without it, this
template is copied). The run's copy is read again before every agent session,
so edits made during a run apply to the next agent that starts. `costs.json`
and `run.json` record the sha256 of the version each agent saw, and every
version is saved as `logs/program-<sha12>.md`.

## all

### Measurement
* Compare absolute latencies: `new_ms` / `ref_ms` per captured case and
  `new_ms_weighted` for a module, `median_ms` end to end. The reference
  timing moves between evaluations (clocks, other processes), so compare a
  new result with your previous best by those numbers, not only by the speedup ratio.
* A difference below max(1 %, 2 × `timing_spread`) is noise. Do not call it
  a win, do not build on it, and do not re-evaluate unchanged code hoping for
  a better number.
* Test one hypothesis per evaluation. Change one thing, write down what you
  expect and why, evaluate, then record what happened. When several changes
  go into one evaluation you cannot tell which one helped or hurt.
* Only `evaluate_candidate` / `evaluate_e2e` results count. Timing scripts
  you write yourself are for debugging: the GPU is shared and they are not
  locked, so never report their numbers as results.

### Simplicity
* At equal speed, simpler wins. A gain within the noise that adds complex
  code is not worth keeping. The same speed with less code is an improvement,
  and so is deleting code at equal speed.
* Add a special-case path (a shape, a backend, a fallback) only when an
  evaluation shows that it pays off on the cases that dominate the call
  counts.

### Honest records
* Never write "X doesn't work" unless you measured X correct and slower. If
  you gave up, write "abandoned after N attempts: <why>" (compile errors,
  wrong results, out of budget), so the idea is not ruled out for good.
* Record failed and slower attempts as well as wins, in `NOTES.md` (kernel
  engineers) and in your final summary.

### Optimise inference, not the benchmark
* The captured inputs are a sample of real calls. Code must be correct for
  any input with the same shapes and dtypes, not only for the captured values.
  Specialise on a property of the inputs (a mask that is all ones, a cache
  that is empty) only if you check it at run time and fall back otherwise.
* Never cache outputs across calls, never skip work because inputs repeat,
  never precompute results for the timed inputs (in `build()`, `apply()` or
  warm-up), never hide work on side streams or threads, and never touch the
  timing or comparison code.

## planner

* Rank targets by their measured share of end-to-end time (Amdahl): a 1.5x
  module speedup on 60 % of the run (1.25x end to end) beats 3x on 5 %
  (1.03x). Put the expected end-to-end gain in each `approach`.
* Name the bottleneck of each target with a number from the profile (calls
  per run, kernels per call, GPU time vs. wall time): launch bound, memory
  bound or compute bound. The approach must address that bottleneck.
* Only measured end-to-end gains survive: integration applies every winner
  in a full model run and keeps it only if total latency drops. Prefer fewer
  targets with a clear idea over many speculative ones. Leaving slots unused is fine.

## kernel

* Before the first line of code, list 3-5 distinct ideas in `NOTES.md`
  (`## Ideas`): an `idea_id`, the mechanism (which work, traffic or launches
  it removes), the expected gain and its ceiling. Distinct means a different
  mechanism, not a different tile size. If `plan.md` exists, start from it.
* Start with the simplest correct kernel for the dominant case (largest
  `calls_per_run` × `ref_ms`), evaluate it, then optimise using evidence
  from `profile=true`.
* Tag every evaluation with its `idea_id` and `expected_speedup`. A build
  error or a wrong result is a bug in an attempt, not evidence against the
  idea: fix it and evaluate again under the same `idea_id` before you drop
  the idea. `best_result` shows per idea whether it failed (bugs) or measured
  correct and not faster (slow).
* Read `workload_profile.md` before the first candidate. It covers every call
  of the module in the run, not only the captured cases: call mix per
  entrypoint and phase, mask kinds, layouts, and how many KV-cache slots
  hold data. Specialise on what it shows, behind a run-time check.
* Before each optimisation, write down the bottleneck you are attacking, the
  number that shows it, and the gain you expect. Afterwards, compare the
  result with that expectation.
* Put each new idea in a new file (`candidates/<backend>_v<N>.py`). Do not
  edit your best candidate in place.
* Fix compile errors and wrong results with a short local script (import,
  `build()`, one call on the captured inputs) before you spend an evaluation.
  Fix rounds are not experiments.
* Judge a direction by its ceiling (every result has `sol_ms`, `pct_of_sol`
  and `bound`: the memory-bandwidth, compute or launch-overhead floor),
  not by its first attempt. When the advice is `consider_stopping`,
  re-read `NOTES.md` and switch to a fundamentally different idea (fusion
  boundary, algorithm, backend), or finish. Do not tune parameters below the
  noise.
* Per-call host work is real latency at decode shapes. Do allocation,
  weight packing and shape logic once in `build()` or cache them per shape.

## systems

* Evaluate each transform alone first (`evaluate_e2e(transforms=[one])`),
  then on top of the kernel winners. Integration starts from the best single
  item and adds the others one at a time, keeping each only if it lowers
  measured latency by more than 1 %.
* One idea per transform file, so that integration can keep or drop it on
  its own.
* Prefer changes with a clear mechanism (launches removed, host syncs
  removed, work not repeated) over a gain of a few percent seen in a single
  evaluation, because each `evaluate_e2e` reloads the model and timings vary
  between calls.

## harness

* Make the workload representative of real inference: use the sizes the
  model card suggests (prompt and output length, steps, audio length) and
  batch size 1 unless the card says otherwise, not the smallest input that runs.
* Call the model's own inference entry point (`generate`, the pipeline, the
  library's inference API) so that the profile shows the real hot paths. Do
  not re-implement the decoding loop.

## research

* You write a plan, not code. Back every claim with ledger evidence (`exp`
  numbers, `pct_of_sol`, per-case times); say "unknown" where the files do not
  tell.
* Judge each direction by its ceiling (what it could reach at the
  bandwidth, compute or launch floor), not by its first attempt: a fresh
  approach is slower at iteration 1 than a tuned one at iteration 20.
* An idea whose attempts all failed (build errors, wrong results) is
  untested, not refuted. Put it under retry with the bug to fix, never under
  "do not try". Only ideas measured correct and not faster, or variants of a
  direction already exhausted, belong on the do-not-try list.

## refactor

* Move code, do not rewrite it: copy the parent's lines into the region
  module unchanged (same ops, same order, same dtype casts). A refactor that
  is not bitwise identical is a bug to find, not noise to accept.
* Cut the region where the planned fusion needs it, no wider: the inputs it
  reads and the outputs the rest of the parent uses, nothing the kernel
  cannot fuse.
