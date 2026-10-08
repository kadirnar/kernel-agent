"""``improve --dry-run --agents 3`` with 1, 2 and 3 islands per kernel target (issue #189,
docs/MULTIAGENT.md §3.5): evaluations, the final simulated speedup, agent hours and what the
migrations and culls did.

The simulated targets have a ceiling per backend (``dryrun.SimTarget.ceilings``): the
planned first backend's is the target's ceiling (what one session per target approaches,
as before), the others are higher for some targets (attention in CUDA, RMSNorm in CuTe) and
lower for the rest. An island starts in its own direction (its first backend) and builds on
its own lineage; it builds on another island's result only when its digest offers it as an
inspiration (``--migrate-every``) or when it is reseeded from the target's best
(``--cull-gap``). ``independent`` rows have neither: parallel sampling. Every row runs in
virtual time (``dryrun.VirtualClock``), seeds 0-4, two budgets: ``--max-hours 8`` (what
islands buy in a fixed time) and ``--rounds 2`` (until every arm has stopped twice).

    PYTHONPATH=src python docs/research-scripts/islands-189/islands_table.py OUT_DIR [PROCS]
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import multiprocessing
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from kernel_agent import charts, dryrun, ledger, orchestrator, workers
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.workspace import read_json

orchestrator.toolchain.setup = dryrun.SimToolchain  # type: ignore[assignment]
charts.available = lambda: False  # type: ignore[assignment]

SEEDS = range(5)
AGENTS = 3
INDEPENDENT = {"migrate_every": 0, "cull_gap": 0.0}
ROWS: dict[str, tuple[int | None, dict[str, Any]]] = {
    "1 (default)": (None, {}),
    "2": (2, {}),
    "2, independent": (2, INDEPENDENT),
    "3": (3, {}),
    "3, independent": (3, INDEPENDENT),
}
BUDGETS = {"max-hours 8": ({"max_hours": 8.0}, {}), "rounds 2": ({}, {"rounds": 2})}


def one(job: tuple[str, str, int]) -> tuple[str, str, dict[str, Any]]:
    budget, name, seed = job
    cfg, icfg = BUDGETS[budget]
    islands, extra = ROWS[name]
    with tempfile.TemporaryDirectory() as tmp:
        config = OptimizeConfig(
            model_ref="Qwen/Qwen3-0.6B",
            runs_dir=Path(tmp),
            dossier=False,
            seeds_per_target=islands,
            **cfg,
        )
        orch = orchestrator.Orchestrator(dryrun.create_run(config, seed), config)
        world = dryrun.World(orch, virtual=True)
        improver = Improver(
            orch,
            ImproveConfig(agents=AGENTS, **icfg, **extra),
            require_capture=False,
            live_charts=False,
        )

        async def main() -> str:
            with world.driving():
                return await improver.improve()

        with world.installed(), contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(main())
        return budget, name, measure(orch.run)


def measure(run: Any) -> dict[str, Any]:
    state = read_json(run.root / "improve.json")
    rows = ledger.rows(run)
    kernel = [
        r
        for r in ledger.measured(rows)
        if r["target"] != ledger.E2E and r["session"] and r["backend"] != "integrate"
    ]
    final = (read_json(run.root / "integration.json", {}) or {}).get("final") or {}
    speedup = float(final.get("speedup") or 1.0)
    sessions = [*state["slices"], *state.get("research", [])]
    agent_h = sum(float(s.get("seconds") or 0.0) for s in sessions) / 3600
    targets = sorted({r["target"] for r in kernel})
    return {
        "kernel_evals": len(kernel),
        "speedup": speedup,
        "agent_h": agent_h,
        "per_agent_h": math.log(speedup) / agent_h if agent_h else 0.0,  # log gain per h
        "loop_h": float(state["finished"]["budget"]["hours"]),
        "rounds": len(state["rounds"]),
        "offered": len(state.get("migrations") or []),
        "used": sum(workers.adoptions([r for r in rows if r["target"] == t]) for t in targets),
        "culls": len(state.get("culls") or []),
        "best": {
            t: max(float(r["speedup"] or 0.0) for r in kernel if r["target"] == t) for t in targets
        },
    }


def table(results: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    out = []
    for budget, by_row in results.items():
        out += [
            f"### `--dry-run --agents {AGENTS} --{budget}` (seeds {SEEDS.start}-"
            f"{SEEDS.stop - 1}, mean)",
            "",
            "| --islands | kernel evaluations | final speedup | agent h | log gain / agent h | "
            "loop h | rounds | migrations offered / used | reseeded |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, runs in by_row.items():

            def m(key: str, runs: list[dict[str, Any]] = runs) -> float:
                return statistics.fmean(r[key] for r in runs)

            out.append(
                f"| {name} | {m('kernel_evals'):.0f} | {m('speedup'):.3f}x | "
                f"{m('agent_h'):.1f} | {m('per_agent_h'):.4f} | {m('loop_h'):.2f} | "
                f"{m('rounds'):.1f} | {m('offered'):.1f} / {m('used'):.1f} | {m('culls'):.1f} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    out_dir = Path(sys.argv[1])
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(b, name, seed) for b in BUDGETS for name in ROWS for seed in SEEDS]
    with multiprocessing.get_context("spawn").Pool(procs) as pool:
        done = pool.map(one, jobs)
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (budget, name, seed), (_, _, r) in zip(jobs, done, strict=True):
        results.setdefault(budget, {}).setdefault(name, []).append(r)
        print(budget, name, seed, json.dumps(r, default=str), flush=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    (out_dir / "results.md").write_text(table(results))
    print(table(results))


if __name__ == "__main__":
    main()
