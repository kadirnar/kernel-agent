"""End-to-end optimisation pipeline.

    analyze → (harness) → plan → capture → kernels → transforms → integrate → report

Deterministic steps (profiling, capture, evaluation, integration) run in GPU
worker subprocesses; creative steps (harness writing, planning, kernel writing,
model transforms) are Claude agents.  Each phase is recorded in ``run.json`` so
an interrupted run can be resumed.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from kernel_agent import hub, ledger, program, toolchain
from kernel_agent.agent import prompts
from kernel_agent.agent.runner import AgentResult, agent_env, run_agent
from kernel_agent.agent.tools import best_for_target, build_server, tool_names
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.dashboard import refresh
from kernel_agent.integrate.export import export_optimized
from kernel_agent.report import write_report
from kernel_agent.worker import call_worker
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workloads.quality import probe_messages
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

PHASES = ["analyze", "plan", "capture", "kernels", "transforms", "integrate", "report"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Orchestrator:
    def __init__(self, run: RunDir, cfg: OptimizeConfig) -> None:
        self.run = run
        self.cfg = cfg
        self.tc = toolchain.setup()
        self.budget = Budget.from_config(run, cfg)
        self.server = build_server(run, self.budget)
        self.env = agent_env(self.tc.env)
        self.python = sys.executable
        self.agent_results: list[AgentResult] = []
        self.phase = PHASES[0]

    # ------------------------------------------------------------ creation

    @classmethod
    def create(cls, cfg: OptimizeConfig) -> Orchestrator:
        tc = toolchain.setup()
        if tc.gpu is None:
            raise SystemExit("no CUDA GPU detected; kernel-agent needs one")
        if cfg.program and not Path(cfg.program).expanduser().is_file():
            raise SystemExit(f"program file not found: {cfg.program}")
        log(f"resolving {cfg.model_ref}")
        card = hub.resolve(cfg.model_ref, token=cfg.hf_token, modality=cfg.modality)
        log(
            f"{card.repo_id}: modality={card.modality.value} arch={card.architectures} "
            f"params={card.params} size={card.size_gb} GB"
        )
        run = RunDir.create(cfg.runs_dir, card.repo_id)
        spec = WorkloadSpec(
            repo_id=card.repo_id,
            revision=card.revision,
            modality=card.modality.value,
            dtype=cfg.dtype,
            trust_remote_code=cfg.trust_remote_code,
            harness=str(Path(cfg.harness).resolve()) if cfg.harness else None,
            options=cfg.workload_options,
            family=card.family,
        )
        write_json(
            run.run_json,
            {
                "card": card.to_dict(),
                "workload": spec.to_dict(),
                "config": cfg.to_dict(),
                "phases": {},
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        from kernel_agent.kernels.roofline import ensure_peaks

        ensure_peaks(verbose=True)  # roofline peaks for toolchain.json + prompts (cached)
        write_json(run.toolchain_json, tc.to_dict())
        program.install(run, cfg.program)
        log(f"run directory: {run.root}")
        return cls(run, cfg)

    @classmethod
    def resume(cls, root: Path, overrides: dict[str, Any] | None = None) -> Orchestrator:
        run = RunDir(root.resolve())
        data = run.load()
        cfg = OptimizeConfig.from_dict({**data["config"], **(overrides or {})})
        program.install(run, (overrides or {}).get("program"))  # keeps the run's edited copy
        return cls(run, cfg)

    # ------------------------------------------------------------ helpers

    def _phase_done(self, name: str) -> bool:
        return bool(self.run.load().get("phases", {}).get(name, {}).get("done"))

    def _mark(self, name: str, **info: Any) -> None:
        data = self.run.load()
        phase = data.setdefault("phases", {}).setdefault(name, {})  # keep budget notes
        phase.update(done=True, at=time.strftime("%H:%M:%S"), **info)
        write_json(self.run.run_json, data)

    def _available_backends(self) -> list[str]:
        avail = [b for b in self.cfg.backends if self.tc.backends.get(b)]
        if not avail:
            raise SystemExit(f"none of the requested backends {self.cfg.backends} is available")
        return avail

    async def _agent(self, name: str, **kwargs: Any) -> AgentResult:
        prog = program.for_agent(self.run, name, log)  # re-read: humans may edit it mid-run
        ledger.event(self.run, "agent_start", agent=name, program_sha256=prog.sha256)
        result = AgentResult(name=name)
        timeout = self.budget.start_agent(name)
        cfg = self.budget.agent_config(self.cfg)
        kwargs["system_append"] += self.budget.prompt_note(name, cfg, kwargs["mcp_tools"])
        kwargs["system_append"] += prog.prompt_note(name)
        timer = asyncio.timeout(timeout)
        try:
            async with timer:
                result = await run_agent(
                    name,
                    cfg=cfg,
                    mcp_server=self.server,
                    env=self.env,
                    log_dir=self.run.root / "logs",
                    result=result,
                    **kwargs,
                )
        except TimeoutError:
            if not timer.expired():
                raise
            result.timed_out = True
            log(f"agent {name}: stopped at its {(timeout or 0) / 60:.1f} min limit")
            self.budget.note(
                self.phase, "timed_out", {"agent": name, "session_id": result.session_id}
            )
        finally:
            self.budget.end_agent(name)
        self.agent_results.append(result)
        ledger.event(
            self.run,
            "agent_done",
            agent=name,
            usd=round(result.cost_usd, 4),
            minutes=round(result.seconds / 60, 1),
            error=result.is_error,
        )
        costs = read_json(self.run.root / "costs.json", {})
        costs[name] = {
            "usd": round(result.cost_usd, 4),
            "turns": result.turns,
            "minutes": round(result.seconds / 60, 1),
            "tools": result.tool_calls,
            "session_id": result.session_id,
            "program_sha256": prog.sha256,
            **({"timed_out": True} if result.timed_out else {}),
        }
        write_json(self.run.root / "costs.json", costs)
        return result

    # ------------------------------------------------------------ phases

    async def analyze(self) -> None:
        log("analyze: loading model, measuring baseline, profiling")
        result = call_worker(self.run, "analyze", "--iters", "3")
        if "error" in result and self.cfg.allow_harness_agent:
            log("analyze: built-in workload failed; asking Claude to write a harness")
            await self.write_harness(result["error"])
            result = call_worker(self.run, "analyze", "--iters", "3")
        if "error" in result:
            raise SystemExit(f"analyze failed:\n{result['error']}")
        log(
            f"analyze: baseline {result['median_ms']:.1f} ms, deterministic="
            f"{result['deterministic']}, peak {result['peak_mem_gb']:.2f} GB"
        )
        if not result["deterministic"]:
            log("WARNING: workload output is not deterministic; quality checks may be noisy")
        for message in probe_messages(result):  # sensitivity probe / teacher forcing
            log(message)
        self._mark("analyze", baseline_ms=result["median_ms"])

    async def write_harness(self, error: str) -> None:
        card = self.run.load()["card"]
        await self._agent(
            "harness",
            prompt="Write and validate harness.py for this model.",
            system_append=prompts.harness_prompt(card, error, self.python, self.tc.summary()),
            cwd=self.run.root,
            mcp_tools=tool_names("check_harness"),
            add_dirs=[prompts.WORKLOADS_DIR],
        )
        if not self.run.harness.exists():
            raise SystemExit("harness agent did not produce harness.py")
        data = self.run.load()
        data["workload"]["harness"] = str(self.run.harness)
        write_json(self.run.run_json, data)

    async def plan(self) -> None:
        data = self.run.load()
        baseline = read_json(self.run.baseline_json, {})
        summary = (self.run.profile_dir / "summary.md").read_text()
        backends = self._available_backends()
        log(
            f"plan: asking the planner ({self.cfg.claude_model}) for up to "
            f"{self.cfg.max_targets} targets"
        )
        result = await self._agent(
            "planner",
            prompt="Produce the optimisation plan for this run.",
            system_append=prompts.planner_prompt(
                data["card"],
                baseline,
                summary,
                backends,
                self.cfg.max_targets,
                self.python,
                self.tc.summary(),
            ),
            cwd=self.run.root,
            mcp_tools=[],
            output_format={"type": "json_schema", "schema": prompts.PLAN_SCHEMA},
        )
        plan = result.structured
        if not isinstance(plan, dict):
            plan = _extract_json(result.text)
        if not isinstance(plan, dict) or "targets" not in plan:
            raise SystemExit(f"planner returned no usable plan:\n{result.text[:2000]}")
        profile = read_json(self.run.profile_dir / "profile.json", {})
        known = {c["cls"] for c in profile.get("classes", [])}
        targets = []
        for t in plan["targets"][: self.cfg.max_targets]:
            if t["module_class"] not in known:
                log(f"plan: dropping {t['id']}: class {t['module_class']} not in profile")
                continue
            t["backends"] = [b for b in t.get("backends", []) if b in backends] or backends[:2]
            targets.append(t)
        plan["targets"] = targets
        write_json(self.run.plan_json, plan)
        for t in targets:
            log(
                f"plan: target {t['id']} = {t['module_class']} via {t['backends']}: "
                f"{t['approach'][:120]}"
            )
        for t in plan.get("transforms", []):
            log(f"plan: transform {t['id']}: {t['idea'][:120]}")
        self._mark("plan", targets=[t["id"] for t in targets])

    async def capture(self) -> None:
        plan = read_json(self.run.plan_json, {})
        kept = []
        for t in plan.get("targets", []):
            target_dir = self.run.target(t["id"])
            (target_dir / "candidates").mkdir(parents=True, exist_ok=True)
            spec = {**t}
            write_json(target_dir / "spec.json", spec)
            log(f"capture: {t['id']} ({t['module_class']})")
            info = call_worker(self.run, "capture", "--target", t["id"])
            if "error" in info:
                log(f"capture: {t['id']} failed, dropping target:\n{info['error'][-800:]}")
                (target_dir / "spec.json").rename(target_dir / "spec.failed.json")
                continue
            log(f"capture: {t['id']} cases={[(c['signature'], c['count']) for c in info['cases']]}")
            kept.append(t["id"])
        self._mark("capture", targets=kept)

    async def kernels(self) -> None:
        profile = read_json(self.run.profile_dir / "profile.json", {})
        stats = {c["cls"]: c for c in profile.get("classes", [])}
        ids = self.run.target_ids()
        done = set(self.run.load().get("phases", {}).get("kernels", {}).get("finished", []))
        pending = [t for t in ids if t not in done]
        sem = asyncio.Semaphore(max(1, self.cfg.parallel))

        async def one(target_id: str) -> None:
            async with sem:
                if reason := self.budget.exhausted():
                    log(f"kernels: skipping {target_id}: {reason}")
                    skip = {"agent": f"kernel-{target_id}", "reason": reason}
                    self.budget.note("kernels", "budget_skipped", skip)
                    return
                target_dir = self.run.target(target_id)
                spec = read_json(target_dir / "spec.json")
                system = prompts.engineer_prompt(
                    spec,
                    spec.get("capture", {}),
                    spec["backends"],
                    self.python,
                    self.tc.summary(),
                    self.cfg.evaluations_per_target,
                    stats.get(spec["module_class"]),
                )
                (target_dir / "NOTES.md").touch()
                await self._agent(
                    f"kernel-{target_id}",
                    prompt=(
                        f"Optimise target `{target_id}`. Start by reading reference_source.py "
                        "and spec.json, then write and evaluate candidates."
                    ),
                    system_append=system,
                    cwd=target_dir,
                    mcp_tools=tool_names("evaluate_candidate", "best_result"),
                    add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
                )
                best = best_for_target(self.run, target_id)
                log(
                    f"kernels: {target_id} best = "
                    + (f"{best['speedup']}x ({best['snapshot']})" if best else "none correct")
                )
                data = self.run.load()
                fin = (
                    data.setdefault("phases", {})
                    .setdefault("kernels", {})
                    .setdefault("finished", [])
                )
                fin.append(target_id)
                write_json(self.run.run_json, data)

        await asyncio.gather(*(one(t) for t in pending))
        self._mark("kernels")  # `finished` lists the targets whose agent ran

    async def transforms(self) -> None:
        if not self.cfg.do_transforms:
            self._mark("transforms", skipped=True)
            return
        if reason := self.budget.exhausted():
            log(f"transforms: skipped: {reason}")
            self.budget.note("transforms", "budget_skipped", {"agent": "systems", "reason": reason})
            self._mark("transforms", skipped=True)
            return
        data = self.run.load()
        plan = read_json(self.run.plan_json, {})
        baseline = read_json(self.run.baseline_json, {})
        summary = (self.run.profile_dir / "summary.md").read_text()
        self.run.transforms_dir.mkdir(parents=True, exist_ok=True)
        await self._agent(
            "systems",
            prompt="Design, write and evaluate model-level transforms.",
            system_append=prompts.systems_prompt(
                data["card"],
                baseline,
                summary,
                plan.get("transforms", []),
                self.python,
                self.tc.summary(),
                self.cfg.transform_evaluations,
                kernels=self._kernel_winners(),
            ),
            cwd=self.run.transforms_dir,
            mcp_tools=tool_names("evaluate_e2e", "run_info"),
            add_dirs=[prompts.WORKLOADS_DIR, prompts.KNOWLEDGE_DIR],
        )
        self._mark("transforms")

    def _kernel_winners(self) -> list[tuple[str, str, float]]:
        """(target_id, snapshot path, module speedup) of every kernel worth integrating."""
        winners = []
        for target_id in self.run.target_ids():
            best = best_for_target(self.run, target_id)
            if best and best["speedup"] >= self.cfg.min_speedup:
                path = self.run.target(target_id) / best["snapshot"]
                winners.append((target_id, str(path), float(best["speedup"])))
        return winners

    async def integrate(self) -> None:
        """Measure every candidate alone, then grow the best combination greedily.

        Ordering by *measured* end-to-end gain (not module-level estimates)
        matters: a model-level transform can beat every kernel on its own and
        be incompatible with them, so the best single item seeds the search.
        """
        baseline = read_json(self.run.baseline_json, {})
        base_ms = float(baseline["median_ms"])
        items: list[tuple[str, str]] = [
            ("kernel", f"{tid}={path}") for tid, path, _ in self._kernel_winners()
        ]
        best_tf: dict[str, dict[str, Any]] = {}
        for rec in read_jsonl(self.run.transforms_dir / "results.jsonl"):
            if rec.get("passed") and not rec.get("kernels") and len(rec.get("transforms", [])) == 1:
                stem = re.sub(r"^\d+_|_[0-9a-f]{8}$", "", Path(rec["transforms"][0]).stem)
                if rec["speedup"] > 1.0 and (
                    stem not in best_tf or rec["speedup"] > best_tf[stem]["speedup"]
                ):
                    best_tf[stem] = rec
        items += [
            ("transform", str(self.run.transforms_dir / rec["transforms"][0]))
            for rec in best_tf.values()
        ]
        log(f"integrate: {len(items)} candidate optimisations")

        history: list[dict[str, Any]] = []

        def e2e(combo: list[tuple[str, str]]) -> dict[str, Any]:
            cli = ["--warmup", "2", "--iters", "5"]
            for kind, arg in combo:
                cli += ["--kernel" if kind == "kernel" else "--transform", arg]
            start = time.perf_counter()
            r = call_worker(self.run, "e2e", *cli)
            history.append({"items": [a for _, a in combo], **_short(r)})
            names = [ledger.item_label(a) for _, a in combo]
            ledger.record_e2e(
                self.run,
                r,
                backend="integrate",
                snapshot="+".join(names),
                hypothesis="integration: "
                + " + ".join(names)
                + (" alone" if len(names) == 1 else ""),
                eval_s=round(time.perf_counter() - start, 1),
            )
            return r

        singles: list[tuple[tuple[str, str], dict[str, Any]]] = []
        for item in items:
            r = e2e([item])
            if r.get("passed") and r["median_ms"] < base_ms:
                singles.append((item, r))
                log(f"integrate: alone {Path(item[1]).name}: {r['median_ms']:.1f} ms")
            else:
                log(f"integrate: drop {Path(item[1]).name} ({r.get('reason') or r.get('status')})")
        singles.sort(key=lambda s: s[1]["median_ms"])

        accepted: list[tuple[str, str]] = []
        final: dict[str, Any] | None = None
        if singles:
            accepted, final = [singles[0][0]], singles[0][1]
            for item, _ in singles[1:]:
                r = e2e([*accepted, item])
                if r.get("passed") and r["median_ms"] < final["median_ms"] * 0.99:
                    accepted.append(item)
                    final = r
                    log(f"integrate: + {Path(item[1]).name} -> {r['median_ms']:.1f} ms")
                else:
                    reason = r.get("reason") or r.get("status")
                    if r.get("passed"):
                        reason = f"no gain ({r['median_ms']:.1f} ms)"
                    log(f"integrate: - {Path(item[1]).name} ({reason})")
        result = {
            "baseline_ms": base_ms,
            "accepted": [{"kind": k, "item": a} for k, a in accepted],
            "final": final,
            "history": history,
        }
        write_json(self.run.root / "integration.json", result)
        export_optimized(self.run, [(k, a, 0.0) for k, a in accepted])
        if final:
            log(
                f"integrate: final {final['median_ms']:.1f} ms vs {base_ms:.1f} ms "
                f"= {final['speedup']}x"
            )
        else:
            log("integrate: no optimisation survived end-to-end validation")
        self._mark("integrate", speedup=final["speedup"] if final else 1.0)

    async def report(self) -> None:
        path = write_report(self.run)
        self._mark("report")
        log(f"report: {path}")

    async def run_all(self, until: str | None = None) -> RunDir:
        for phase in PHASES:
            if self._phase_done(phase):
                continue
            self.phase = phase
            ledger.event(self.run, "phase_start", phase=phase)
            try:
                await getattr(self, phase)()
            except BaseException as exc:  # SystemExit / Ctrl-C too: status must not say "running"
                ledger.event(self.run, "phase_failed", phase=phase, error=repr(exc)[:300])
                raise
            ledger.event(self.run, "phase_done", phase=phase)
            refresh(self.run)
            if phase == until:
                break
        return self.run


def _short(r: dict[str, Any]) -> dict[str, Any]:
    return {
        k: r.get(k)
        for k in ("status", "passed", "reason", "median_ms", "speedup", "metrics", "patches")
    }


def _extract_json(text: str) -> Any:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.search(
        r"(\{.*\})", text, re.S
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


async def optimize(cfg: OptimizeConfig, until: str | None = None) -> RunDir:
    orch = Orchestrator.create(cfg)
    return await orch.run_all(until=until)
