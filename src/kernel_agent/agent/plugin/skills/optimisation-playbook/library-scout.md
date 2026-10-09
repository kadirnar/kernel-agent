# The library bar (library scout)

Before a target's first engineer session kernel-agent runs the **library scout**
(`kernel_agent/libscout/`, no agent): it traces what the reference calls on one captured
case (SDPA, RMSNorm as `F.rms_norm` or written out, `F.linear`, softmax, sampling, RoPE,
gated MLP), and every library adapter that applies on this GPU becomes a candidate swept
through `sweep_candidate`'s machinery and the full evaluator.

## What you get

* Ledger rows with `backend` `library:<package>@<version>`, `session` `libscout` and the
  hypothesis `library scout: ...`; their snapshots are in `history/` like yours. They set
  the target's best when they are faster; they never count against your evaluations or
  your streak.
* `## Library bar` in your digest and first prompt, and `targets/<id>/libscout.json`:
  * the best scout candidate (module speedup, % of SOL);
  * every adapter's result (`fallback` when the library launches only the reference's own
    kernels: torch already picked that kernel);
  * **op bars**: each library op alone on the reference's recorded inputs, against the
    reference's op, as kernel time (calls replayed from a CUDA graph) and as eager time
    per call (host launch cost included);
  * the adapters not run, with the reason (not installed, architecture, dtype, mask).

## How to use it

* The bar is a **floor**, never a reason to stop: start from it, beat it.
* A library op that wins its op bar but loses at module level usually loses to host
  overhead (a scout candidate runs the reference's TorchDynamo graph eagerly and pays the
  guard check per call; cuDNN SDPA's eager path costs tens of microseconds of host time).
  Call the library inside your own fused module, or keep it for a CUDA-graph transform.
  Measured on the RTX 5070 Ti (sm_120): VoxCPM2's 11-token GQA attention, cuDNN 15.8 us vs
  the reference's flash 38.6 us as kernel time, but 33 to 122 us per eager call against
  the reference's 37 to 43 us (a busy CPU).
* A written-out RMSNorm folded into `F.rms_norm` must keep the reference's roundings
  (normalised, rounded, then the weight: the scout's `FUSE=0`): with one rounding fewer a
  decoder layer whose residual stream cancels failed the tolerance (VoxCPM2 LocDiT layer).
  The same holds for a residual add fused into a GEMM epilogue.
* Copy a scout snapshot as a starting point: it is an ordinary candidate (`build`, the
  config bound in `_KA_SWEEP_CONFIG`), with its library and licence declared
  (`KA_LIBRARY`, `KA_LICENCE`; the export writes `requirements.txt` from them).
* An adapter's template marked "not verified on a GPU" was written from the library's
  documentation: treat its failure as a template bug, not as evidence against the library.
