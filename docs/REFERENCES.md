# References: what kernel-agent took from which project or paper

A map from sources to features. The analysis behind it (feature comparison,
reward-hacking audit, literature table) is in [RESEARCH.md](RESEARCH.md); the
issues are listed in [ROADMAP.md](ROADMAP.md).

## Repositories read at source level

| repository | what kernel-agent adopted |
|---|---|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | the never-stopping experiment loop (`kernel-agent improve`, #13); the experiment ledger `results.tsv` and the running-best `progress.png` (#3); a human-edited `program.md` that steers the agents (#14); "at equal speed, simpler code wins" |
| [RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel) and its paper ([arXiv 2603.21331](https://arxiv.org/abs/2603.21331)) | Amdahl-ranked targets and move-on rules (5 non-improving evaluations, 90 % of speed of light; #13); multi-stage correctness checks (#6, #8); a KernelBench bridge (#21) |
| [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent) | parallel isolated workers per target and duplicate detection (#17); region targets across module boundaries, after its Fuser (#22); speed-of-light / roofline feedback (#9; Nsight Compute metrics are #10); per-GPU locks (#23) |
| [Dogacel/auto-gpu-kernel](https://github.com/Dogacel/auto-gpu-kernel) | hashing the evaluator's files to detect tampering (#5); "a delta below ~5 % is noise unless a paired A/B confirms it" (#11); a clean-context research agent on plateaus (#15); a live dashboard (#16); workload statistics (#20) |

Deliberately **not** copied from AutoKernel: synthetic canonical test shapes,
profiler rows that count an aten op and its kernel twice, and end-to-end
verification by a single prefill forward. kernel-agent's captured real inputs
and measured integration are stricter.

## Papers and reports that became features

| source | feature in kernel-agent |
|---|---|
| KernelBench, Stanford ([arXiv 2502.10517](https://arxiv.org/abs/2502.10517)) | fast_p metric and `kernel-agent bench-suite` (#21) |
| METR, [Measuring automated kernel engineering](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering) | side-stream / hidden-work detection (#7); parallel search (#17) |
| Sakana AI CUDA Engineer → robust-kbench ([arXiv 2509.14279](https://arxiv.org/abs/2509.14279)) | perturbed inputs and unseen shapes (#6, #8) |
| SOL-ExecBench, NVIDIA ([arXiv 2603.19173](https://arxiv.org/abs/2603.19173)) | the exploit checklist (#6, #7); speed-of-light per case (#9) |
| CUDA-L1, Kevin-32B, TritonRL, Dr.Kernel | the reward-hacking taxonomy; reference-fallback detection (#7) |
| CudaForge ([arXiv 2511.01884](https://arxiv.org/abs/2511.01884)) | one hypothesis per evaluation (#3, #15); curated profiler metrics (#10) |
| STARK, K-Search | idea ledger; strategy separated from implementation (#15) |
| KernelBand ([arXiv 2511.18868](https://arxiv.org/abs/2511.18868)) | UCB scheduler over targets (#13) |
| InferenceBench | the sweep / autotune tool (#19) |
| AccelOpt, KernelBlaster, AdaExplore, GEAK v3 | cross-run kernel library and distilled lessons (#18) |
| LLM4LLM | phase-specific (prefill / decode) targets (#20); in-model validation |
| KernelOPT, CUDA Agent, PIKE | reporting speedups against `torch.compile` (#12) |
| FlashInfer-Bench, MEP | confirmed the capture + `apply.py` design |

No published agent work targets TTS or audio models; the closest is the
hand-optimised Faster IndexTTS-2. Teacher-forced validation (#2) and the
natural-length stop check (#55), both developed for VoxCPM2, have no
counterpart in this literature.

## How these were read

The four repositories were cloned and their source read. Most papers were
read from their arXiv abstract / HTML pages rather than full PDFs. Numbers in
these sources are the authors' own and were not re-measured here. Some 2026
affiliations are inferred and marked "(?)" in [RESEARCH.md](RESEARCH.md).
