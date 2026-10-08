"""``improve --dry-run`` with 1, 2, 3 and 4 agent sessions at once on the same simulated run
(issue #183, docs/MULTIAGENT.md §3.9): evaluations per simulated hour, GPU busy share,
GPU queue waits and the final simulated speedup.

Every row runs in virtual time (``dryrun.VirtualClock``): think time is simulated, and every
evaluation, A/B step, capture and re-profile holds the GPU through the real GPU job queue for
its simulated seconds. ``1 (sequential)`` is today's loop (``--agents 1``) with that time
accounting; ``1 (coordinator)`` is the concurrent coordinator with one slot (its
re-integration runs in the background); 2-4 are ``--agents N``. Seeds 0-4 of the simulation,
two budgets: ``--max-hours 8`` (what N buys in a fixed time) and ``--rounds 2`` without a
time limit (how long the same work takes).

    PYTHONPATH=src python docs/research-scripts/agents-183/agents_table.py OUT_DIR
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from kernel_agent import charts, dryrun, gpuqueue, ledger, orchestrator
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.workspace import read_json, read_jsonl

orchestrator.toolchain.setup = dryrun.SimToolchain  # type: ignore[assignment]
charts.available = lambda: False  # type: ignore[assignment]

SEEDS = range(5)
ROWS = [("1 (sequential)", 1, False), ("1 (coordinator)", 1, True)] + [
    (str(n), n, True) for n in (2, 3, 4)
]
BUDGETS = {"max-hours 8": ({"max_hours": 8.0}, {}), "rounds 2": ({}, {"rounds": 2})}


def one(tmp: Path, seed: int, agents: int, coordinated: bool, cfg: dict, icfg: dict) -> dict:
    config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp, dossier=False, **cfg)
    orch = orchestrator.Orchestrator(dryrun.create_run(config, seed), config)
    world = dryrun.World(orch, virtual=True)
    improver = Improver(
        orch, ImproveConfig(agents=agents, **icfg), require_capture=False, live_charts=False
    )
    improver.concurrent = coordinated  # one slot through the coordinator too

    async def main() -> str:
        with world.driving():
            return await improver.improve()

    with world.installed():
        asyncio.run(main())
    run, t0 = orch.run, world.clock.t0
    state = read_json(run.root / "improve.json")
    loop_s = float(state["finished"]["budget"]["hours"]) * 3600  # until the loop stopped
    rows = ledger.rows(run)
    evals = [
        r
        for r in rows
        if r["backend"] != "integrate" and r["status"] not in ledger.UNMEASURED and r["session"]
    ]
    log = read_jsonl(run.root / gpuqueue.FILE)
    starts = {e["id"]: e for e in log if e["state"] == "start"}
    hold = sum(
        e["hold_s"] for e in log if e["state"] == "done" and starts[e["id"]]["ts"] - t0 < loop_s
    )
    waits = sorted(
        float(e.get("wait_s") or 0.0)
        for e in starts.values()
        if e.get("session") and e["kind"] in ("eval", "e2e")
    )
    final = (read_json(run.root / "integration.json", {}) or {}).get("final") or {}
    costs = read_json(run.root / "costs.json", {}) or {}
    return {
        "loop_h": loop_s / 3600,
        "end_h": (state["finished"]["at"] - t0) / 3600,
        "evals": len(evals),
        "evals_per_h": len(evals) / (loop_s / 3600),
        "gpu_busy": hold / loop_s,
        "wait_mean": statistics.fmean(waits) if waits else 0.0,
        "wait_p95": waits[int(0.95 * (len(waits) - 1))] if waits else 0.0,
        "speedup": float(final.get("speedup") or 1.0),
        "rounds": len(state["rounds"]),
        "peak": (state.get("coordinator") or {}).get("peak", 1),
        "usd": sum(float(c.get("usd") or 0.0) for c in costs.values()),
    }


def table(results: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    out = []
    for budget, by_row in results.items():
        out += [
            f"### `--dry-run --{budget}` (seeds {SEEDS.start}-{SEEDS.stop - 1}, mean)",
            "",
            "| --agents | evaluations / simulated h | GPU busy | eval wait mean / p95 | "
            "evaluations | loop h | done at h | final speedup | rounds | $ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, runs in by_row.items():

            def m(key: str, runs: list[dict[str, Any]] = runs) -> float:
                return statistics.fmean(r[key] for r in runs)

            out.append(
                f"| {name} | {m('evals_per_h'):.1f} | {m('gpu_busy'):.0%} | "
                f"{m('wait_mean'):.0f} s / {m('wait_p95'):.0f} s | {m('evals'):.0f} | "
                f"{m('loop_h'):.2f} | {m('end_h'):.2f} | {m('speedup'):.3f}x | "
                f"{m('rounds'):.1f} | {m('usd'):.0f} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for budget, (cfg, icfg) in BUDGETS.items():
        for name, agents, coordinated in ROWS:
            for seed in SEEDS:
                with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
                    r = one(Path(tmp), seed, agents, coordinated, cfg, icfg)
                results.setdefault(budget, {}).setdefault(name, []).append(r)
                print(budget, name, seed, {k: round(v, 3) for k, v in r.items()}, flush=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    (out_dir / "results.md").write_text(table(results))
    print(table(results))


if __name__ == "__main__":
    main()
