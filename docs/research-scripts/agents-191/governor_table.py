"""``improve --dry-run`` with fixed ``--agents N`` against ``--agents auto`` (the governor,
issue #191) on a subscription's simulated usage windows: evaluations, how full the 5-hour
window got, how long the account was spent (the rate gate closed: every session waits, and
so does the user's own Claude) and the sessions stopped at the limit.

Every run is ``--max-hours 10`` in virtual time (``dryrun.VirtualClock``), the coordinator
with up to N sessions, the simulated usage windows of ``dryrun.SimUsage``: each simulated
session's work fills them (the 5-hour window resets 4 h into the run, then every 5 h), and
its stream carries the rate-limit events the governor reads. Scenarios:

* ``median``: the measured median rate (docs/MULTIAGENT-DATA.md §7: one session about 10
  points of the 5-hour window and 2.3 of the 7-day one per hour), the 5-hour window 10 % used;
* ``used40``: the same with the 5-hour window 40 % used before the run (interactive use);
* ``p90``: the measured p90 rate of the 5-hour window (17.6 points per session-hour).

``+async`` rows add ``--async-evals`` (the simulated engineer writes its next candidate while
the last one is evaluated). Seeds 0-4, means.

    PYTHONPATH=src python docs/research-scripts/agents-191/governor_table.py OUT_DIR
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from kernel_agent import charts, dryrun, ledger, orchestrator
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.workspace import read_json

orchestrator.toolchain.setup = dryrun.SimToolchain  # type: ignore[assignment]
charts.available = lambda: False  # type: ignore[assignment]

SEEDS = range(5)
HOURS = 10.0
SCENARIOS: dict[str, Callable[[], dryrun.SimUsage]] = {
    "median": lambda: dryrun.SimUsage.subscription(0.1, 0.3, five_hour_reset_h=4.0),
    "used40": lambda: dryrun.SimUsage.subscription(0.4, 0.3, five_hour_reset_h=4.0),
    "p90": lambda: dryrun.SimUsage.subscription(
        0.1, 0.3, five_hour_reset_h=4.0, per_hour=(0.21, 0.028)
    ),
}
ROWS = [("1", False), ("2", False), ("3", False), ("4", False), ("auto", False)]
ROWS += [("3", True), ("auto", True)]


def one(job: tuple[str, str, bool, int, str]) -> dict[str, Any]:
    scenario, agents, async_evals, seed, out_dir = job
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        config = OptimizeConfig(
            model_ref="Qwen/Qwen3-0.6B", runs_dir=Path(tmp), dossier=False, max_hours=HOURS
        )
        orch = orchestrator.Orchestrator(dryrun.create_run(config, seed), config)
        world = dryrun.World(orch, virtual=True)
        world.usage = SCENARIOS[scenario]()
        auto = agents == "auto"
        icfg = ImproveConfig(agents=6 if auto else int(agents), governor=auto)
        icfg.async_evals = async_evals
        improver = Improver(orch, icfg, require_capture=False, live_charts=False)
        improver.concurrent = True  # one slot through the coordinator too

        async def main() -> str:
            with world.driving():
                return await improver.improve()

        with world.installed():
            asyncio.run(main())
        state = read_json(orch.run.root / "improve.json")
        loop_h = float(state["finished"]["budget"]["hours"])
        evals = [
            r
            for r in ledger.rows(orch.run)
            if r["backend"] != "integrate" and r["session"] and r["status"] not in ledger.UNMEASURED
        ]
        closed, since = 0.0, None
        for e in ledger.events(orch.run):
            if e["event"] == "rate_gate" and e["state"] == "closed":
                since = e["ts"]
            elif e["event"] == "rate_gate" and e["state"] == "open" and since is not None:
                closed, since = closed + e["ts"] - since, None
        final = (read_json(orch.run.root / "integration.json", {}) or {}).get("final") or {}
        coord = state.get("coordinator") or {}
        return {
            "scenario": scenario,
            "row": agents + (" +async" if async_evals else ""),
            "seed": seed,
            "evals": len(evals),
            "evals_per_h": len(evals) / loop_h,
            "five_hour": world.usage.peak["five_hour"],
            "seven_day": world.usage.peak["seven_day"],
            "spent_h": closed / 3600,
            "stopped": world.usage.limited,
            "speedup": float(final.get("speedup") or 1.0),
            "mean_k": (coord.get("governor") or {}).get("mean"),
            "peak": coord.get("peak"),
        }


def table(results: list[dict[str, Any]]) -> str:
    out = []
    for scenario in SCENARIOS:
        out += [
            f"### `{scenario}` (seeds {SEEDS.start}-{SEEDS.stop - 1}, mean)",
            "",
            "| --agents | evaluations | per h | 5-hour window max | account spent | "
            "sessions stopped at the limit | final speedup | mean k |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in dict.fromkeys(r["row"] for r in results):
            runs = [r for r in results if r["scenario"] == scenario and r["row"] == name]

            def m(key: str, runs: list[dict[str, Any]] = runs) -> float:
                return statistics.fmean(float(r[key] or 0.0) for r in runs)

            k = f"{m('mean_k'):.1f}" if runs[0]["mean_k"] is not None else "—"
            out.append(
                f"| {name} | {m('evals'):.0f} | {m('evals_per_h'):.1f} | {m('five_hour'):.0%} | "
                f"{m('spent_h'):.2f} h | {m('stopped'):.1f} | {m('speedup'):.2f}x | {k} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (scenario, agents, async_evals, seed, str(out_dir))
        for scenario in SCENARIOS
        for agents, async_evals in ROWS
        for seed in SEEDS
    ]
    with ProcessPoolExecutor(6) as pool:
        results = list(pool.map(one, jobs))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    (out_dir / "results.md").write_text(table(results))
    print(table(results))


if __name__ == "__main__":
    main()
