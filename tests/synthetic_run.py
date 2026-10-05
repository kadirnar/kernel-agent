"""A realistic, fully synthetic run directory (no GPU, no Claude).

It goes through the same recording code as a real run (``record_candidate``,
``record_e2e_result``, ``ledger.record_e2e``, ``ledger.event``), in wall-clock
order, so results.jsonl, results.tsv and events.jsonl look like the real thing.
The story: Qwen3-0.6B decoding 128 tokens; four kernel targets written by two
parallel kernel agents, then model-level transforms, then the greedy
integration (baseline 1532 ms, final 889 ms).

``python tests/synthetic_run.py OUT_DIR`` writes one and renders its charts.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kernel_agent import ledger
from kernel_agent.agent.tools import _snapshot, record_candidate, record_e2e_result
from kernel_agent.config import OptimizeConfig
from kernel_agent.workspace import RunDir, write_json

REPO = "Qwen/Qwen3-0.6B"
CREATED = "2026-10-05 09:00:00"
BASELINE_MS = 1532.4

HEADERS = {
    "triton": "import torch\nimport triton\nimport triton.language as tl\n",
    "cuda": "import torch\nfrom torch.utils.cpp_extension import load_inline\n",
    "cute": "import cutlass\nimport cutlass.cute as cute\nimport torch\n",
    "tilelang": "import tilelang\nimport tilelang.language as T\nimport torch\n",
    "nvrtc": "import torch\nfrom cuda.core import Device, Program, ProgramOptions\n",
}

# (target, class, instances, isolated module ms per run, backends, start min, minutes per eval)
TARGETS: list[tuple[str, str, int, float, list[str], float, float]] = [
    ("attn", "Qwen3Attention", 28, 505.0, ["triton", "cuda"], 9.0, 4.4),
    ("mlp", "Qwen3MLP", 28, 343.0, ["cuda", "triton"], 9.5, 3.9),
    ("rmsnorm", "Qwen3RMSNorm", 113, 189.0, ["cuda", "cute"], 64.0, 3.2),
    ("rope", "Qwen3RotaryEmbedding", 1, 25.0, ["triton", "cuda"], 79.5, 3.3),
]

# Per target: (backend, candidate, hypothesis, outcome); outcome = speedup or a failure status.
SCRIPTS: dict[str, list[tuple[str, str, str, float | str]]] = {
    "attn": [
        (
            "triton",
            "triton_v1",
            "fuse q/k RMSNorm and RoPE into one Triton kernel before SDPA",
            1.18,
        ),
        ("triton", "triton_v2", "also write K/V straight into the cache slot", "build_error"),
        ("triton", "triton_v2", "write K/V into the cache slot (fixed strides)", 1.31),
        ("triton", "triton_v3", "split-K flash-decode for the 1-token decode case", "incorrect"),
        ("triton", "triton_v3", "split-K flash-decode with the causal mask fixed", 1.52),
        ("triton", "triton_v4", "num_warps 4→8 and BLOCK_N 64→128", 1.49),
        ("triton", "triton_v5", "persistent kernel looping over heads", "runtime_error"),
        ("cuda", "cuda_v1", "CUDA decode attention, warp per head, bf16x8 loads", 1.44),
        ("cuda", "cuda_v2", "cp.async double-buffered KV tiles", 1.71),
        ("cuda", "cuda_v3", "pre-pack q/k/v weights in build(): one GEMM instead of three", 1.93),
        ("cuda", "cuda_v4", "fp32 accumulation in registers, no smem reduction", 1.95),
        ("cuda", "cuda_v5", "tensor-core mma.sync path for prefill", "incorrect"),
        ("cuda", "cuda_v6", "cache cos/sin per position across layers", 1.90),
        ("cuda", "cuda_v7", "__launch_bounds__(128) and unrolled head loop", "timeout"),
        ("cuda", "cuda_v8", "fold the o_proj epilogue into the attention kernel", 2.12),
        ("cuda", "cuda_v9", "separate static-shape prefill and decode kernels", 2.08),
    ],
    "mlp": [
        ("cuda", "cuda_v1", "merge gate/up into one pre-packed GEMM with fused SiLU*mul", 1.22),
        ("cuda", "cuda_v2", "vectorised bf16x8 SiLU*mul kernel", 1.23),
        ("triton", "triton_v1", "Triton GEMM with fused SiLU*mul epilogue", 0.91),
        ("triton", "triton_v2", "autotune BLOCK_M/N/K for M=1 decode", 1.08),
        ("cuda", "cuda_v3", "GEMV for M=1 decode: warp per 8 rows, weights streamed once", 1.46),
        ("cuda", "cuda_v4", "split-K for the down_proj GEMV", "incorrect"),
        ("cuda", "cuda_v5", "split-K with an fp32 scratch reduction", 1.58),
        ("cuda", "cuda_v6", "down_proj in the same launch with a grid sync", "runtime_error"),
        ("cuda", "cuda_v7", "128-bit loads and an L2 prefetch hint on weights", 1.66),
        ("cuda", "cuda_v8", "cuBLAS for prefill, custom GEMV only for M<=4", 1.67),
        ("cuda", "cuda_v9", "int4 weight-only quantisation of gate/up", "incorrect"),
        ("cuda", "cuda_v10", "unroll the K loop by 4", 1.68),
        ("cuda", "cuda_v11", "persistent CTAs, one per SM", 1.79),
        ("cuda", "cuda_v12", "overlap the gate/up GEMV with the SiLU", "build_error"),
    ],
    "rmsnorm": [
        ("cuda", "cuda_v1", "warp-per-row RMSNorm, bf16x8 loads, fp32 accumulation", 1.64),
        ("cuda", "cuda_v2", "fuse the residual add (in place) into the norm", "incorrect"),
        ("cuda", "cuda_v3", "two rows per warp, residual add kept separate", 1.71),
        ("cute", "cute_v1", "CuTe DSL, one CTA per row", 1.38),
        ("cute", "cute_v2", "TVM-FFI calling convention to cut host overhead", 1.69),
        ("cuda", "cuda_v4", "block per row with a shared-memory reduction", 1.55),
        ("cuda", "cuda_v5", "skip .contiguous() and reuse the output buffer", 1.92),
        ("cuda", "cuda_v6", "one launch for q_norm and k_norm", "build_error"),
        ("cuda", "cuda_v7", "one launch for q_norm and k_norm, head_dim 128 rows", 2.05),
        ("cuda", "cuda_v8", "__launch_bounds__(256), no shared memory", 2.06),
        ("cuda", "cuda_v9", "CUDA-graph capture inside the module", "incorrect"),
    ],
    "rope": [
        ("triton", "triton_v1", "cos/sin in one Triton kernel from position ids", 1.12),
        ("triton", "triton_v2", "precompute the cos/sin table once, gather by position", 1.85),
        ("triton", "triton_v3", "store the table in bf16", "incorrect"),
        ("triton", "triton_v4", "vectorised 2×bf16 gather", 1.88),
        ("cuda", "cuda_v1", "load_inline gather kernel, one thread per pair", 2.31),
        ("cuda", "cuda_v2", "return views of the table instead of copies", "incorrect"),
    ],
}

# (minute, transform file or None, kernels, hypothesis, outcome ms or failure status)
TRANSFORMS: list[tuple[float, str | None, list[str], str, float | str]] = [
    (104.0, "static_cache", [], "static KV cache, no per-step reallocation", 1389.0),
    (109.0, "cuda_graph_decode", [], "static cache + CUDA graph for the decode step", 968.0),
    (
        115.0,
        "cuda_graph_decode",
        ["attn", "mlp", "rmsnorm"],
        "CUDA graph on top of the kernels",
        "incorrect",
    ),
    (120.0, "sdpa_flash", [], "force the flash SDPA backend for prefill", 1521.0),
    (126.0, "merged_qkv", [], "merge q/k/v projections into one Linear", 1490.0),
    (131.0, "no_item_sync", [], "drop the per-step .item() sync in the stop check", "patch_error"),
]

# Greedy integration: (minute, items, outcome ms or failure reason)
SINGLES = [
    ("attn", 1392.0),
    ("mlp", 1431.0),
    ("rmsnorm", 1468.0),
    ("rope", 1530.0),
    ("cuda_graph_decode", 968.0),
    ("static_cache", 1389.0),
]
GREEDY = [
    ("static_cache", 975.0),
    ("attn", 912.0),
    ("mlp", "greedy tokens diverge at step 41"),
    ("rmsnorm", 889.0),
    ("rope", 887.0),
]

COSTS = {
    "planner": (0.42, 9, 2.4),
    "kernel-attn": (6.85, 96, 70.1),
    "kernel-mlp": (5.10, 81, 54.6),
    "kernel-rmsnorm": (3.20, 58, 35.8),
    "kernel-rope": (1.15, 27, 19.7),
    "systems": (3.90, 44, 35.0),
}
PHASES = [
    ("analyze", 0.0, 4.5),
    ("plan", 4.5, 7.0),
    ("capture", 7.0, 9.0),
    ("kernels", 9.0, 100.5),
    ("transforms", 100.5, 135.0),
    ("integrate", 135.0, 151.0),
    ("report", 151.0, 151.2),
]


def _profile() -> dict[str, Any]:
    def cls(name: str, inclusive: float, instances: int, calls: int, leaf: bool = False) -> dict:
        return {
            "root": "model",
            "cls": name,
            "module_path": f"transformers.models.qwen3.modeling_qwen3.{name}",
            "source_file": "/site-packages/transformers/models/qwen3/modeling_qwen3.py",
            "instances": instances,
            "calls": calls,
            "inclusive_ms": inclusive,
            "self_ms": round(inclusive * 0.2, 3),
            "params": 0,
            "is_leaf": leaf,
            "example_qualname": f"model.{name}",
            "signatures": [
                {
                    "signature": "a0[1, 1, 1024]:bfloat16",
                    "calls": calls - instances,
                    "inclusive_ms": inclusive * 0.9,
                },
                {
                    "signature": "a0[1, 512, 1024]:bfloat16",
                    "calls": instances,
                    "inclusive_ms": inclusive * 0.1,
                },
            ],
        }

    return {
        "hooked_wall_ms": 2190.0,
        "module_calls": 63_744,
        "classes": [
            cls("Qwen3ForCausalLM", 2086.0, 1, 128),
            cls("Qwen3Model", 2010.0, 1, 128),
            cls("Qwen3DecoderLayer", 1931.0, 28, 3584),
            cls("Linear", 905.0, 197, 25_216, leaf=True),
            cls("Qwen3Attention", 856.0, 28, 3584),
            cls("Qwen3MLP", 588.0, 28, 3584),
            cls("Qwen3RMSNorm", 321.0, 113, 14_464, leaf=True),
            cls("SiLU", 54.0, 28, 3584, leaf=True),
            cls("Qwen3RotaryEmbedding", 42.0, 1, 128),
            cls("Embedding", 6.0, 1, 128, leaf=True),
        ],
        "kernel_view": {
            "wall_ms": 1551.0,
            "gpu_busy_ms": 702.0,
            "gpu_busy_fraction": 0.453,
            "kernel_launches": 61_440,
            "avg_kernel_us": 11.4,
            "kernels": [
                {
                    "name": "ampere_bf16_s16816gemm_bf16_64x64_ldg8",
                    "calls": 25_216,
                    "total_ms": 388.0,
                }
            ],
            "aten_ops": [{"name": "aten::linear", "calls": 25_216, "device_ms": 401.0}],
        },
    }


def _kernel_result(speedup: float | str, iso_ms: float, instances: int) -> dict[str, Any]:
    if isinstance(speedup, str):
        error = {"incorrect": None}.get(speedup, f"{speedup}: synthetic failure")
        result: dict[str, Any] = {"status": speedup, "correct": False}
        if error:
            result["error"] = error
        if speedup == "incorrect":
            result["cases"] = [
                {
                    "case": 0,
                    "signature": "a0[1, 1, 1024]:bfloat16",
                    "ok": False,
                    "calls_per_run": 127,
                    "max_abs_err": 0.31,
                    "min_cosine": 0.97,
                }
            ]
        return result
    cases = []
    ref_total = new_total = 0.0
    for sig, count, frac, s in (
        ("a0[1, 1, 1024]:bfloat16", 127, 0.88, speedup),
        ("a0[1, 512, 1024]:bfloat16", 1, 0.12, max(speedup * 0.82, 0.8)),
    ):
        ref = iso_ms * frac / (instances * count)
        new = ref / s
        ref_total += count * ref
        new_total += count * new
        cases.append(
            {
                "case": len(cases),
                "signature": sig,
                "calls_per_run": count,
                "ok": True,
                "max_abs_err": 0.0078,
                "min_cosine": 0.99999,
                "ref_ms": round(ref, 5),
                "new_ms": round(new, 5),
                "speedup": round(s, 3),
                "timing_spread": 0.006 if count > 1 else 0.02,
            }
        )
    return {
        "status": "ok",
        "correct": True,
        "cases": cases,
        "speedup": round(ref_total / new_total, 3),
        "est_saved_ms_per_run": round((ref_total - new_total) * instances, 3),
        "ref_ms_weighted": round(ref_total, 4),
        "new_ms_weighted": round(new_total, 4),
        "eval_seconds": 41.0,
    }


def _e2e_result(outcome: float | str) -> dict[str, Any]:
    if isinstance(outcome, str) and outcome in ("patch_error", "runtime_error", "crash"):
        return {"status": outcome, "passed": False, "error": f"{outcome}: synthetic failure"}
    if isinstance(outcome, str):  # quality failure
        ms = 905.0
        return {
            "status": "ok",
            "passed": False,
            "reason": outcome,
            "median_ms": ms,
            "times_ms": [ms - 3, ms, ms + 4],
            "baseline_ms": BASELINE_MS,
            "speedup": round(BASELINE_MS / ms, 4),
            "metrics": {"token_match": 0.32},
        }
    return {
        "status": "ok",
        "passed": True,
        "reason": None,
        "median_ms": outcome,
        "times_ms": [outcome - 2.0, outcome, outcome + 3.5],
        "baseline_ms": BASELINE_MS,
        "speedup": round(BASELINE_MS / outcome, 4),
        "metrics": {"token_match": 1.0, "logits_cosine": 0.9998},
    }


def make_run(base: Path, *, integrate: bool = True) -> RunDir:
    """Write the synthetic run under ``base`` and return it."""
    t0 = time.mktime(time.strptime(CREATED, "%Y-%m-%d %H:%M:%S"))

    def at(minute: float) -> float:
        return t0 + minute * 60

    run = RunDir.create(base, REPO)
    cfg = OptimizeConfig(model_ref=REPO, parallel=2, evaluations_per_target=16)
    write_json(
        run.run_json,
        {
            "card": {
                "repo_id": REPO,
                "revision": "main",
                "modality": "llm",
                "architectures": ["Qwen3ForCausalLM"],
                "params": 596_049_920,
                "size_gb": 1.4,
            },
            "workload": {
                "repo_id": REPO,
                "modality": "llm",
                "dtype": "bfloat16",
                "options": {"prompt_len": 512, "new_tokens": 128},
            },
            "config": cfg.to_dict(),
            "phases": {},
            "created": CREATED,
        },
    )
    write_json(
        run.toolchain_json,
        {
            "gpu": {"name": "NVIDIA GeForce RTX 5070 Ti", "arch": "sm_120"},
            "torch_version": "2.14.1+cu130",
        },
    )
    write_json(
        run.baseline_json,
        {
            "workload": "llm: prompt 512 tokens, 128 new tokens, batch 1",
            "median_ms": BASELINE_MS,
            "times_ms": [1529.8, 1532.4, 1536.1],
            "compiled_ms": 1104.6,
            "deterministic": True,
        },
    )
    write_json(run.profile_dir / "profile.json", _profile())
    (run.profile_dir / "summary.md").write_text("# Profile summary\n\n(synthetic)\n")
    plan = {
        "analysis": "Decode is launch bound (45 % GPU busy): fuse small ops, then CUDA graphs.",
        "targets": [
            {
                "id": t,
                "module_class": c,
                "backends": b,
                "why": "hot in decode",
                "approach": "fuse and vectorise",
            }
            for t, c, _, _, b, _, _ in TARGETS
        ],
        "transforms": [
            {"id": "cuda_graph_decode", "idea": "CUDA graph for decode", "why": "launch bound"}
        ],
    }
    write_json(run.plan_json, plan)
    for t in plan["targets"]:
        (run.target(t["id"]) / "candidates").mkdir(parents=True, exist_ok=True)
        write_json(run.target(t["id"]) / "spec.json", t)
    run.transforms_dir.mkdir(parents=True, exist_ok=True)

    # Every action at its wall-clock minute, executed in time order.
    actions: list[tuple[float, int, Callable[[], None]]] = []

    def add(minute: float, fn: Callable[[], None]) -> None:
        actions.append((minute, len(actions), fn))

    for phase, a, b in PHASES:
        add(a, lambda p=phase, a=a: ledger.event(run, "phase_start", when=at(a), phase=p))
        add(b, lambda p=phase, b=b: ledger.event(run, "phase_done", when=at(b), phase=p))
    agent_spans = {"planner": (4.6, 6.9), "systems": (100.6, 134.8)}
    for target, _, instances, iso_ms, _, start, step in TARGETS:
        script = SCRIPTS[target]
        agent_spans[f"kernel-{target}"] = (start, start + step * (len(script) + 0.5))
        for k, (backend, name, hypothesis, outcome) in enumerate(script):
            minute = start + step * (k + 1) + (0.6 if k % 3 == 1 else 0.0)

            def evaluate(
                target: str = target,
                backend: str = backend,
                name: str = name,
                hypothesis: str = hypothesis,
                outcome: float | str = outcome,
                iso_ms: float = iso_ms,
                instances: int = instances,
                minute: float = minute,
            ) -> None:
                src = run.target(target) / "candidates" / f"{name}.py"
                src.write_text(
                    f'{HEADERS[backend]}\n"""{hypothesis}"""\n\n\ndef build(reference):\n'
                    "    return reference\n"
                )
                snap = _snapshot(src, run.target(target) / "history")
                record_candidate(
                    run,
                    target,
                    src,
                    snap,
                    _kernel_result(outcome, iso_ms, instances),
                    hypothesis=hypothesis,
                    eval_s=41.0,
                    when=at(minute),
                )

            add(minute, evaluate)
    for name, (a, b) in agent_spans.items():
        add(a, lambda n=name, a=a: ledger.event(run, "agent_start", when=at(a), agent=n))
        add(
            b,
            lambda n=name, b=b: ledger.event(
                run, "agent_done", when=at(b), agent=n, usd=COSTS[n][0]
            ),
        )
    for minute, transform, kernels, hypothesis, outcome in TRANSFORMS:

        def transform_eval(
            transform: str | None = transform,
            kernels: list[str] = kernels,
            hypothesis: str = hypothesis,
            outcome: float | str = outcome,
            minute: float = minute,
        ) -> None:
            snaps = []
            if transform:
                src = run.transforms_dir / f"{transform}.py"
                src.write_text(f'"""{hypothesis}"""\n\n\ndef apply(workload):\n    pass\n')
                snaps.append(_snapshot(src, run.transforms_dir / "history"))
            entries = [f"{k}={run.target(k)}/history/x.py" for k in kernels]
            record_e2e_result(
                run,
                _e2e_result(outcome),
                snaps,
                entries,
                hypothesis=hypothesis,
                eval_s=95.0,
                when=at(minute),
            )

        add(minute, transform_eval)
    actions.sort(key=lambda a: (a[0], a[1]))
    for _, _, fn in actions:
        fn()

    if integrate:
        _integrate(run, at)
    data = run.load()
    data["phases"] = {p: {"done": True} for p, _, _ in PHASES if integrate or p != "integrate"}
    write_json(run.run_json, data)
    write_json(
        run.root / "costs.json",
        {
            name: {
                "usd": usd,
                "turns": turns,
                "minutes": minutes,
                "tools": {"evaluate_candidate": len(SCRIPTS.get(name.removeprefix("kernel-"), []))}
                if name.startswith("kernel-")
                else {},
            }
            for name, (usd, turns, minutes) in COSTS.items()
        },
    )
    return run


def _integrate(run: RunDir, at: Callable[[float], float]) -> None:
    def item(name: str) -> str:
        if name in SCRIPTS:
            return f"{name}={run.target(name)}/history/best.py"
        return str(run.transforms_dir / "history" / f"002_{name}_0a1b2c3d.py")

    history: list[dict[str, Any]] = []
    minute = 136.0

    def measure(names: list[str], outcome: float | str) -> dict[str, Any]:
        nonlocal minute
        result = _e2e_result(outcome)
        history.append(
            {
                "items": [item(n) for n in names],
                **{
                    k: result.get(k)
                    for k in ("status", "passed", "reason", "median_ms", "speedup", "metrics")
                },
            }
        )
        label = " + ".join(names)
        ledger.record_e2e(
            run,
            result,
            backend="integrate",
            snapshot="+".join(names),
            hypothesis=f"integration: {label}" + (" alone" if len(names) == 1 else ""),
            eval_s=70.0,
            when=at(minute),
        )
        minute += 1.3
        return result

    for name, ms in SINGLES:
        measure([name], ms)
    accepted = ["cuda_graph_decode"]
    final = _e2e_result(968.0)
    for name, outcome in GREEDY:
        r = measure([*accepted, name], outcome)
        if r.get("passed") and r["median_ms"] < final["median_ms"] * 0.99:
            accepted.append(name)
            final = r
    write_json(
        run.root / "integration.json",
        {
            "baseline_ms": BASELINE_MS,
            "accepted": [
                {"kind": "kernel" if n in SCRIPTS else "transform", "item": item(n)}
                for n in accepted
            ],
            "final": final,
            "history": history,
        },
    )


if __name__ == "__main__":
    from kernel_agent.dashboard import refresh
    from kernel_agent.report import write_report

    out = make_run(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic-runs"))
    refresh(out)
    write_report(out)
    print(out.root)
