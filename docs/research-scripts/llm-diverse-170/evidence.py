"""#170 evidence on Qwen3-0.6B (RTX 5070 Ti): the library's own diverse-set and LLM-gate code
on the new natural prompts, for two prompt-lookup speculative decoding transforms:
an eager one written for this check (pld_eager.py, reports its counters) and the Qwen3 run's
FP8-megakernel + prompt-lookup transform (runs/Qwen--Qwen3-0.6B/20261008-021850,
transforms/history/033_prompt_lookup_decode_gqa_verify_797d15c5.py with the bundle it loads,
028_decode_step_megakernel_2c930eef.py, copied to run033/ next to this script).

    flock ~/.cache/kernel-agent/gpu.lock env KERNEL_AGENT_LOCK_HELD=1 \\
        python evidence.py evidence.json      # about 4 minutes; calibrate_llm.py: about 4 too
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from kernel_agent import diversity, toolchain
from kernel_agent.workloads import create_workload, diverse, llm, perceptual
from kernel_agent.workloads.base import WorkloadSpec, measure

HERE = Path(__file__).parent
toolchain.setup()
wl = create_workload(WorkloadSpec(repo_id="Qwen/Qwen3-0.6B", modality="llm"))
wl.load()
inputs = wl.make_inputs()
tiled = {"prompt": llm.PROMPT}  # the version-1 benchmark prompt: one paragraph repeated
with wl.with_options(tiled):
    tiled_inputs = wl.make_inputs()


def timing(x: dict) -> dict:
    t = measure(wl, x, warmup=1, iters=3)
    return {"ms": round(t["median_ms"], 2), **(t.get("metric_detail") or {})}


base = timing(inputs)
base_tiled = timing(tiled_inputs)
outputs, info = diverse.record_baseline(wl)
baseline = {"median_ms": base["ms"], "diverse": info}
gate_ref, gate_info = perceptual.record_baseline(wl)
print(json.dumps({"baseline_ms": base["ms"], "tiled_baseline_ms": base_tiled["ms"]}), flush=True)


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


results = []
for label, path in [
    ("eager prompt lookup (K=10, report_stats)", HERE / "pld_eager.py"),
    ("run 033: FP8 megakernel + prompt lookup", HERE / "run033/033_prompt_lookup_decode_gqa_verify_797d15c5.py"),
]:
    load(path).apply(wl)
    main = timing(inputs)
    old = timing(tiled_inputs)
    floor = perceptual.floor_options(wl)  # near-lossless, as in the Qwen3 run
    with wl.with_options(floor):
        varied = diverse.check(wl, outputs, baseline, chaotic=False)
    exact = diverse.check(wl, outputs, baseline, chaotic=False)  # exact tier: token identity
    gate = perceptual.check(wl, gate_ref)
    result = {
        "transform": label,
        "benchmark_speedup": round(base["ms"] / main["ms"], 2),
        "benchmark_decode_stats": main.get("decode_stats"),
        "tiled_v1_prompt_speedup": round(base_tiled["ms"] / old["ms"], 2),
        "tiled_decode_stats": old.get("decode_stats"),
        "diverse": diversity.compact({"metrics": {"diverse": varied}}),
        "diverse_quality_near_lossless": {"passed": varied["passed"], "reason": varied["reason"]},
        "diverse_quality_exact": {"passed": exact["passed"], "reason": exact["reason"][:600]},
        "gate": {k: gate.get(k) for k in ("passed", "reason", "kl", "kl_worst", "top1", "nll_increase", "nll_increase_worst")},
        "rows": [
            {k: r.get(k) for k in ("label", "baseline_ms", "ms", "speedup", "decode_stats", "quality")}
            for r in varied["inputs"]
        ],
    }
    results.append(result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    for r in result["rows"]:
        print("   ", r["label"], r["speedup"], diversity.stats_text(r.get("decode_stats")), r.get("quality"), flush=True)
    if "run" in vars(wl):
        del wl.run  # undo: both transforms only wrap workload.run
    torch.cuda.empty_cache()
json.dump({"baseline": base, "tiled": base_tiled, "results": results}, open(sys.argv[1], "w"), indent=1, default=str)
