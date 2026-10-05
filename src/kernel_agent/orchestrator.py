"""End-to-end optimisation pipeline.

    analyze → (harness) → plan → capture → kernels → transforms → integrate → report

Deterministic steps (profiling, capture, evaluation, integration) run in GPU
worker subprocesses; creative steps (harness writing, planning, kernel writing,
model transforms) are Claude agents.  Each phase is recorded in ``run.json`` so
an interrupted run can be resumed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from kernel_agent import (
    abtest,
    hub,
    ledger,
    library,
    program,
    region,
    research,
    scheduler,
    strong_baseline,
    telemetry,
    toolchain,
    truth,
    workers,
)
from kernel_agent.agent import prompts
from kernel_agent.agent.runner import READ_TOOLS, AgentResult, agent_env, run_agent
from kernel_agent.agent.tools import best_for_target, build_server, tool_names
from kernel_agent.budget import MIN_AGENT_USD, SOL_STOP_PCT, Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.dashboard import refresh
from kernel_agent.integrate.export import export_optimized
from kernel_agent.phases import PHASES as CALL_PHASES
from kernel_agent.report import write_report
from kernel_agent.worker import call_worker
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workloads.quality import probe_messages
from kernel_agent.workspace import RunDir, read_json, write_json

PHASES = ["analyze", "plan", "capture", "kernels", "transforms", "integrate", "report"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Orchestrator:
    def __init__(self, run: RunDir, cfg: OptimizeConfig) -> None:
        self.run = run
        self.cfg = cfg
        self.tc = toolchain.setup()
        self.budget = Budget.from_config(run, cfg)
        self.truth = truth.of(run)  # digests of the evaluator's ground truth (truth.py)
        self.server = build_server(run, self.budget, self.truth)
        self.env = agent_env(self.tc.env)
        self.python = sys.executable
        self.agent_results: list[AgentResult] = []
        self.phase = PHASES[0]
        # Replaced by `improve --dry-run` (simulated agent / GPU worker); None = the real ones.
        self.agent_runner: Callable[..., Awaitable[AgentResult]] | None = None
        self.worker: Callable[..., dict[str, Any]] | None = None

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
                "truth": truth.new_section(),  # ground truth in .truth/, hashed
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

    def _worker(self, command: str, *args: str) -> dict[str, Any]:
        return (self.worker or call_worker)(self.run, command, *args)

    async def _agent(
        self,
        name: str,
        label: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run an agent session; ``label`` keys its ``costs.json`` entry (default: ``name``),
        ``config`` overrides fields of the run's config for this session (e.g. the model),
        ``mcp_server`` (in ``kwargs``) replaces the run's tools (a worker's, ``workers.py``)."""
        server = kwargs.pop("mcp_server", None) or self.server
        tag: dict[str, Any] = {"label": label} if label else {}
        prog = program.for_agent(self.run, name, log)  # re-read: humans may edit it mid-run
        ledger.event(self.run, "agent_start", agent=name, program_sha256=prog.sha256, **tag)
        result = AgentResult(name=name)
        timeout = self.budget.start_agent(name)
        cfg = self.budget.agent_config(dataclasses.replace(self.cfg, **(config or {})))
        kwargs["system_append"] += self.budget.prompt_note(name, cfg, kwargs["mcp_tools"])
        kwargs["system_append"] += prog.prompt_note(name)
        timer = asyncio.timeout(timeout)
        try:
            async with timer:
                result = await (self.agent_runner or run_agent)(
                    name,
                    cfg=cfg,
                    mcp_server=server,
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
            **tag,
        )
        costs = read_json(self.run.root / "costs.json", {})
        costs[label or name] = {
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
        if (compiled := strong_baseline.describe(result)) is not None:
            log("analyze: " + compiled.replace("**", ""))
        self.truth.seal_baseline(result["median_ms"])  # baseline.json + baseline_output.pt
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
            if problem := region.validate(t, known) or _scope(t):
                log(f"plan: dropping {t['id']}: {problem}")
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
        self._mark("capture", targets=await self.capture_targets(plan.get("targets", [])))

    async def capture_targets(self, targets: list[dict[str, Any]]) -> list[str]:
        """:meth:`_capture`, then the refactor step of the region targets; returns the ids."""
        kept = self._capture(targets)
        for t in targets:
            if region.is_region(t) and t["id"] in kept and not await self.refactor(t["id"]):
                kept.remove(t["id"])
        return kept

    def _capture(self, targets: list[dict[str, Any]]) -> list[str]:
        """Create ``targets/<id>/spec.json`` and capture each target (of a region target: its
        parent class, see ``region.py``); returns the captured ids."""
        kept = []
        for t in targets:
            target_dir = self.run.target(t["id"])
            (target_dir / "candidates").mkdir(parents=True, exist_ok=True)
            spec = {**t}
            write_json(target_dir / "spec.json", spec)
            parent = ["--parent"] if region.is_region(t) else []
            log(f"capture: {t['id']} ({t['parent_class'] if parent else t['module_class']})")
            info = self._worker("capture", "--target", t["id"], *parent)
            if "error" in info:
                log(f"capture: {t['id']} failed, dropping target:\n{info['error'][-800:]}")
                (target_dir / "spec.json").rename(target_dir / "spec.failed.json")
                continue
            capture = self.run.capture_file(t["id"])
            if parent:
                capture = region.parent_capture(self.run, t["id"])
            if capture.exists():
                self.truth.seal(capture)
            log(f"capture: {t['id']} cases={[(c['signature'], c['count']) for c in info['cases']]}")
            kept.append(t["id"])
        return kept

    async def refactor(self, target_id: str) -> bool:
        """The refactor step of a region target (``region.py``): a ``refactor-<id>`` session
        writes ``rewrite.py`` (read-only tools + ``verify_rewrite``; it may write that file
        only), a sealed copy is verified on the parent's capture, and the ``Region_<id>``
        module it adds is captured. False (the target is dropped) when a step fails."""
        target_dir = self.run.target(target_id)
        spec = read_json(target_dir / "spec.json")
        rewrite = target_dir / region.REWRITE_FILE

        def install() -> dict[str, Any]:
            timeout = self.budget.eval_timeout_s
            return region.install(self.run, target_id, self.truth, timeout=timeout)

        # a resumed run, or a rewrite written by hand: no session when it verifies
        result = await asyncio.to_thread(install) if rewrite.is_file() else {}
        if not result.get("verified") and (reason := self.budget.exhausted()):
            result = {"status": "budget_skipped", "error": reason}
        elif not result.get("verified"):
            await self._agent(
                f"refactor-{target_id}",
                prompt=(
                    f"Isolate the region of target `{target_id}` in a new submodule: write "
                    f"{rewrite} and check it with verify_rewrite."
                ),
                system_append=prompts.refactor_prompt(spec, spec.get("parent_capture", {})),
                cwd=target_dir,
                mcp_tools=tool_names("verify_rewrite"),
                tools=[*READ_TOOLS, "Write", "Edit"],
                writable=[rewrite],
            )
            result = await asyncio.to_thread(install)
        spec["rewrite"] = region.summary(result)
        write_json(target_dir / "spec.json", spec)
        if result.get("verified"):
            how = "bitwise" if result.get("bitwise") else "within one ulp"
            log(f"refactor: {target_id}: rewrite verified ({how}); capture {spec['module_class']}")
            info = self._worker("capture", "--target", target_id)
            if "error" not in info:
                self.truth.seal(self.run.capture_file(target_id))
                cases = [(c["signature"], c["count"]) for c in info["cases"]]
                log(f"capture: {target_id} cases={cases}")
                return True
            result = {"status": "capture_failed", "error": info["error"]}
        log(f"refactor: {target_id} dropped ({result.get('status')}):\n{_why(result)[-800:]}")
        (target_dir / "spec.json").rename(target_dir / "spec.failed.json")
        return False

    async def kernels(self) -> None:
        profile = read_json(self.run.profile_dir / "profile.json", {})
        stats = {c["cls"]: c for c in profile.get("classes", [])}
        ids = self.run.target_ids()
        done = set(self.run.load().get("phases", {}).get("kernels", {}).get("finished", []))
        pending = [t for t in ids if t not in done]
        sem = asyncio.Semaphore(max(1, self.cfg.parallel))

        async def one(target_id: str) -> None:
            team: list[workers.Seed] = []
            async with sem:
                if reason := self.budget.exhausted():
                    log(f"kernels: skipping {target_id}: {reason}")
                    skip = {"agent": f"kernel-{target_id}", "reason": reason}
                    self.budget.note("kernels", "budget_skipped", skip)
                    return
                await self.seed_library([target_id])  # prior winners first, no LLM cost
                if reason := self._prior_suffices(target_id):
                    log(f"kernels: {target_id}: no agent session needed: {reason}")
                    return
                first, _ = workers.rounds(self.cfg.evaluations_per_target, self.cfg.reseed_workers)
                team = self.kernel_seeds(target_id, first)
                if not team:  # one engineer session in the target directory
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
                    ) + self._library_note(target_id, spec)
                    (target_dir / "NOTES.md").touch()
                    await self._agent(
                        f"kernel-{target_id}",
                        prompt=(
                            f"Optimise target `{target_id}`. Start by reading "
                            "reference_source.py and spec.json, then write and evaluate "
                            "candidates."
                        ),
                        system_append=system,
                        cwd=target_dir,
                        mcp_tools=tool_names(
                            "evaluate_candidate", "sweep_candidate", "best_result"
                        ),
                        add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
                    )
            if team:  # every worker session takes its own --parallel slot
                await self._kernel_workers(target_id, team, sem)
            best = best_for_target(self.run, target_id, self.truth)  # across the workers
            log(
                f"kernels: {target_id} best = "
                + (f"{best['speedup']}x ({best['snapshot']})" if best else "none correct")
            )
            data = self.run.load()
            fin = data.setdefault("phases", {}).setdefault("kernels", {}).setdefault("finished", [])
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

    # ------------------------------------------------------------ workers (workers.py)

    def _share(self, spec: dict[str, Any]) -> float:
        """A target's share of the profiled time (``--seeds-per-target auto``)."""
        profile = read_json(self.run.profile_dir / "profile.json", {}) or {}
        shares = scheduler.class_shares(profile)
        share, instances = shares.get(str(spec.get("module_class")), (0.0, 1))
        return share / instances if spec.get("qualname") and instances > 1 else share

    def kernel_seeds(self, target_id: str, total: int) -> list[workers.Seed]:
        """The first-round worker sessions of a target for ``total`` evaluations, or [] when
        it has a single worker (the classic session in the target directory)."""
        spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
        k = workers.count(self.cfg.seeds_per_target, self._share(spec))
        if k <= 1:
            return []
        return workers.seeds(spec, k, total, self._available_backends())

    async def worker_session(
        self,
        target_id: str,
        seed: workers.Seed,
        team: list[workers.Seed],
        *,
        prompt: str,
        digest: str = "",
        label: str | None = None,
    ) -> AgentResult:
        """One worker session of a target: its own directory and tools bound to it (paths,
        the ``worker`` of its ledger rows, its evaluation budget), the engineer prompt with
        the worker's approach and backends, and a ``# Worker`` section (``team``: the round)."""
        profile = read_json(self.run.profile_dir / "profile.json", {})
        stats = {c["cls"]: c for c in profile.get("classes", [])}
        spec = read_json(self.run.target(target_id) / "spec.json")
        cwd = workers.prepare(self.run, target_id, seed.worker)
        name = workers.agent_name(target_id, seed.worker)
        system = (
            prompts.engineer_prompt(
                {**spec, "approach": seed.approach},
                spec.get("capture", {}),
                list(seed.backends) or spec["backends"],
                self.python,
                self.tc.summary(),
                seed.evaluations,
                stats.get(spec["module_class"]),
            )
            + self._library_note(target_id, spec)
            + workers.prompt_note(target_id, seed, team)
        )
        binding = workers.Binding(target_id, seed.worker, name, seed.evaluations)
        return await self._agent(
            name,
            label,
            prompt=prompt,
            system_append=system + digest,
            cwd=cwd,
            mcp_tools=tool_names("evaluate_candidate", "sweep_candidate", "best_result"),
            add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
            mcp_server=build_server(self.run, self.budget, self.truth, binding),
        )

    async def _kernel_workers(
        self, target_id: str, team: list[workers.Seed], sem: asyncio.Semaphore
    ) -> None:
        """The ``kernels`` phase of a target with workers: ``team`` (round 1), then with
        ``--reseed-workers`` round 2 from its two best snapshots. Each session waits for a
        ``--parallel`` slot; finished sessions are skipped by a resumed run."""
        finished = set(self.run.load().get("phases", {}).get("kernels", {}).get("workers", []))

        async def job(seed: workers.Seed, members: list[workers.Seed], key: str) -> None:
            if key in finished:
                return
            async with sem:
                if reason := self.budget.exhausted():
                    log(f"kernels: skipping {key}: {reason}")
                    agent = workers.agent_name(target_id, seed.worker)
                    self.budget.note(
                        "kernels", "budget_skipped", {"agent": agent, "reason": reason}
                    )
                    return
                await self.worker_session(
                    target_id,
                    seed,
                    members,
                    prompt=f"Optimise target `{target_id}` as worker {seed.worker} (see the "
                    "`# Worker` section). Start by reading reference_source.py and spec.json, "
                    "then write and evaluate candidates in your candidates/.",
                )
            data = self.run.load()
            kernels = data.setdefault("phases", {}).setdefault("kernels", {})
            kernels.setdefault("workers", []).append(key)
            write_json(self.run.run_json, data)

        await asyncio.gather(*(job(s, team, f"{target_id}/w{s.worker}") for s in team))
        _, second = workers.rounds(self.cfg.evaluations_per_target, self.cfg.reseed_workers)
        again = workers.reseeds(self.run, target_id, team, second, self.truth) if second else []
        if again:
            starts = ", ".join(f"{s.parent} ({s.parent_speedup:.2f}x)" for s in again)
            log(f"kernels: {target_id}: round 2 of its workers from {starts}")
            await asyncio.gather(*(job(s, again, f"{target_id}/r2w{s.worker}") for s in again))

    def _kernel_bests(self) -> list[tuple[str, dict[str, Any]]]:
        """(target_id, verified best record) of every kernel worth integrating."""
        bests = []
        for target_id in self.run.target_ids():
            best = best_for_target(self.run, target_id, self.truth)
            # a region target's kernel needs its verified rewrite, unchanged (region.py)
            ok = region.rewrite_ok(self.run, target_id, self.truth)
            if best and best["speedup"] >= self.cfg.min_speedup and ok:
                bests.append((target_id, best))
        return bests

    def _kernel_winners(self) -> list[tuple[str, str, float]]:
        """(target_id, snapshot path, module speedup) of every kernel worth integrating,
        for prompts: the snapshot copies in the targets' own ``history/``."""
        winners = []
        for target_id, best in self._kernel_bests():
            path = self.run.target(target_id) / "history" / Path(best["snapshot"]).name
            winners.append((target_id, str(path), float(best["speedup"])))
        return winners

    async def integrate(self, reuse: bool = False) -> None:
        """Measure every candidate alone, then grow the best combination greedily.

        Ordering by *measured* end-to-end gain (not module-level estimates)
        matters: a model-level transform can beat every kernel on its own and
        be incompatible with them, so the best single item seeds the search,
        unless the best combination the systems agent measured (its transforms
        on top of kernels, ``integration.json`` ``composite``) is faster again
        now: then that combination seeds it as a whole.

        Every measurement is a paired A/B (``abtest.py``): A and B alternate in
        one process (``worker e2e_ab``). An item alone is B against the
        unmodified model (a candidate when it passes the quality checks and its
        gain is positive; ordered by that gain). B is accepted when it passes
        the quality checks, wins at least ``ab_min_win_rate`` of the rounds and
        the 95 % CI of its gain is above ``ab_min_gain``: the seed (the first
        candidate accepted against the unmodified model, or the combination
        against the best single item), then each step B = the accepted set A +
        the next candidate. Sets that cannot be undone in-process are measured in
        two processes back to back instead (``abtest.SEPARATE_ITERS`` runs each,
        A measured again). ``integration.json`` keeps each step's ``ab`` record
        and the ``projection`` (baseline − Σ est. saved ms) of every accepted set.

        ``reuse`` (re-integrations of the improve loop) takes the measurements
        of the same snapshot files (the same A and B for an A/B) from the
        previous ``integration.json`` instead of measuring them again.

        Everything comes from the verified truth (``truth.py``): the baseline
        latency the orchestrator recorded, records and snapshots whose digests
        match, and a reuse cache only from an unmodified ``integration.json``.
        """
        base_ms = self.truth.baseline_ms()
        items, digests = self._integration_items()
        composite = self._integration_composite(digests)
        log(
            f"integrate: {len(items)} candidate optimisations"
            + (f" + the combination of exp {composite[1].get('exp')}" if composite else "")
        )

        history: list[dict[str, Any]] = []
        integration = self.run.root / "integration.json"
        previous = self.truth.load_json(integration) if reuse else {}
        known = (previous or {}).get("history", [])
        paired = {_ab_key(h): h for h in known if h.get("ab")}
        # items whose state cannot be undone in-process (snapshot paths: content-addressed)
        irreversible = set((previous or {}).get("irreversible") or [])

        def ab(
            a: list[tuple[str, str]], b: list[tuple[str, str]], note: str = ""
        ) -> dict[str, Any]:
            """B's result with its ``ab`` record (decided) against A."""
            a_items = [x for _, x in a]
            hit = paired.get((tuple(a_items), tuple(x for _, x in b)))
            r = dict(hit) if hit is not None else self._paired(a, b, note, irreversible)
            record = {**(r.get("ab") or {}), "a_items": a_items}
            if record.get("a_ms") and record.get("b_ms"):
                record = abtest.judge(
                    record, min_win_rate=self.cfg.ab_min_win_rate, min_gain=self.cfg.ab_min_gain
                )
            else:  # B failed before any timed round
                why = record.get("why") or r.get("reason") or r.get("status")
                record.update(accepted=False, why=why)
            r["ab"] = record
            history.append({"items": [x for _, x in b], **_short(r), "ab": record})
            return r

        singles: list[tuple[tuple[str, str], dict[str, Any]]] = []
        for item in items:  # against the unmodified model of the same session
            r = ab([], [item])
            if r.get("passed") and float(r["ab"].get("gain") or 0.0) > 0.0:
                singles.append((item, r))
                log(
                    f"integrate: alone {Path(item[1]).name}: {r['median_ms']:.1f} ms "
                    f"({abtest.describe(r['ab'])})"
                )
            else:
                reason = r.get("reason") or r.get("status")
                if r.get("passed"):
                    reason = f"not faster than the baseline: {abtest.describe(r['ab'])}"
                log(f"integrate: drop {Path(item[1]).name} ({reason})")
        singles.sort(key=lambda s: -float(s[1]["ab"]["gain"]))

        accepted: list[tuple[str, str]] = []
        final: dict[str, Any] | None = None
        seed: dict[str, Any] | None = None
        sets: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []  # each accepted set
        if composite is not None:  # measured again, like everything else
            combo, rec = composite
            best = [singles[0][0]] if singles else []
            r = ab(best, combo, f" (the combination of exp {rec.get('exp')})")
            seeded = bool(r.get("passed")) and r["ab"]["accepted"]
            seed = {"items": [a for _, a in combo], "exp": rec.get("exp"), "seeded": seeded}
            names = " + ".join(ledger.item_label(a) for _, a in combo)
            if seeded:
                accepted, final = list(combo), r
                log(
                    f"integrate: seed {names} (exp {rec.get('exp')}): {r['median_ms']:.1f} ms "
                    f"({abtest.describe(r['ab'])})"
                )
            else:
                reason = r.get("reason") or r.get("status")
                if r.get("passed"):
                    than = ledger.item_label(best[0][1]) if best else "the baseline"
                    reason = f"not faster than {than}: {abtest.describe(r['ab'])}; {r['ab']['why']}"
                log(f"integrate: no seed {names} (exp {rec.get('exp')}): {reason}")
        if final is None and singles:  # the fastest item that passes the rule on its own
            first = next(((i, r) for i, r in singles if r["ab"]["accepted"]), None)
            if first is not None:
                accepted, final = [first[0]], first[1]
            else:
                log("integrate: no item alone passes the A/B rule against the unmodified model")
        if final is not None:
            sets.append((list(accepted), final))
        for item, _ in singles:
            if final is None or item in accepted:  # part of the seed
                continue
            if any(_item_key(item) == _item_key(a) for a in accepted):
                log(f"integrate: = {Path(item[1]).name} (a version of it is accepted)")
                continue
            r = ab(accepted, [*accepted, item])
            if r.get("passed") and r["ab"]["accepted"]:
                accepted.append(item)
                final = r
                sets.append((list(accepted), r))
                log(
                    f"integrate: + {Path(item[1]).name} -> {r['median_ms']:.1f} ms "
                    f"({abtest.describe(r['ab'])})"
                )
            else:
                reason = r.get("reason") or r.get("status")
                if r.get("passed"):
                    reason = f"no significant gain: {abtest.describe(r['ab'])}; {r['ab']['why']}"
                log(f"integrate: - {Path(item[1]).name} ({reason})")
        projection = self._projection(base_ms, sets, history)
        result = {
            "baseline_ms": base_ms,
            "accepted": [{"kind": k, "item": a} for k, a in accepted],
            "final": final,
            "history": history,
            "projection": projection,
        }
        if seed is not None:
            result["composite"] = seed
        if irreversible:
            result["irreversible"] = sorted(irreversible)
        baseline = self.truth.load_json(self.run.baseline_json)  # its compiled_ms
        reference = self._with_reference(baseline, accepted, previous or {})
        if reference is not None:
            result["reference"] = reference
        write_json(integration, result)
        self.truth.seal(integration)
        export_optimized(self.run, [(k, a, 0.0) for k, a in accepted], digests=digests)
        self._library_store([a for k, a in accepted if k == "kernel"], final)
        if final:
            _, vs_compiled = strong_baseline.speedups(baseline, final["median_ms"])
            projected = projection[-1]["projected_ms"]
            log(
                f"integrate: final {final['median_ms']:.1f} ms vs {base_ms:.1f} ms "
                f"= {final['speedup']}x"
                + (f" ({vs_compiled:.2f}x vs compiled)" if vs_compiled else "")
                + f"; projected {projected:.1f} ms"
            )
        else:
            log("integrate: no optimisation survived end-to-end validation")
        self._mark("integrate", speedup=final["speedup"] if final else 1.0)

    def _integration_call(
        self, command: str, combo: list[tuple[str, str]], cli: list[str], note: str
    ) -> dict[str, Any]:
        """One integration measurement (``e2e`` / ``e2e_ab`` of ``combo``) and its ledger row
        (none when an A/B could not run in-process: nothing was measured)."""
        start = time.perf_counter()
        r = self._worker(command, *cli, *self.truth.worker_args())
        if command == "e2e_ab" and r.get("status") in abtest.FALLBACK:
            return r
        names = [ledger.item_label(a) for _, a in combo] or ["baseline"]
        ledger.record_e2e(
            self.run,
            r,
            backend="integrate",
            snapshot="+".join(names),
            hypothesis="integration: "
            + " + ".join(names)
            + (" alone" if len(names) == 1 and not note else note),
            eval_s=round(time.perf_counter() - start, 1),
        )
        gpu = r.get("gpu") or (r.get("ab") or {}).get("gpu")
        if message := telemetry.warning(gpu):
            log(f"integrate: WARNING {message}")
        return r

    def _paired(
        self, a: list[tuple[str, str]], b: list[tuple[str, str]], note: str, irreversible: set[str]
    ) -> dict[str, Any]:
        """B's result with an ``ab`` record of its timings against A: from one process
        (``e2e_ab``), or from two back to back when a state cannot be undone in-process
        (``irreversible`` collects such items)."""
        why = "an item cannot be undone in-process"
        if not irreversible.intersection(x for _, x in [*a, *b]):
            cli = [*_cli(a, warmup=2), *_cli(b, prefix="--b-")]
            cli += ["--rounds", str(self.cfg.ab_rounds)]
            r = self._integration_call("e2e_ab", b, cli, note)
            if r.get("status") not in abtest.FALLBACK:
                return r
            irreversible.update(r.get("irreversible") or [])
            why = str(r.get("reason") or r.get("status"))
            log(f"integrate: no in-process A/B ({why[:300]}); separate processes")
        iters = abtest.SEPARATE_ITERS
        ra = self._integration_call(
            "e2e", a, _cli(a, iters=iters), " (A of an A/B in separate processes)"
        )
        rb = self._integration_call("e2e", b, _cli(b, iters=iters), note)
        rb["ab"] = {
            "mode": "separate",
            "a_ms": ra.get("times_ms") or [],
            "b_ms": rb.get("times_ms") or [],
            "fallback": why,
        }
        if not ra.get("times_ms"):
            rb["ab"]["why"] = f"A measured again: {ra.get('reason') or ra.get('status')}"
        return rb

    def _projection(
        self,
        base_ms: float,
        sets: list[tuple[list[tuple[str, str]], dict[str, Any]]],
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Projected (baseline − Σ est. saved ms of its items) vs measured latency of every
        accepted set: a kernel's saving is its module-level estimate, a transform's its
        measured gain alone (paired against the unmodified model)."""
        alone = {
            h["items"][0]: base_ms * float(h["ab"]["gain"])
            for h in history
            if len(h["items"]) == 1
            and not (h.get("ab") or {}).get("a_items")
            and h.get("passed")
            and (h.get("ab") or {}).get("gain") is not None
        }
        out = []
        for combo, r in sets:
            saved: dict[str, float | None] = {}
            for kind, arg in combo:
                est = None
                if kind == "kernel":
                    target_id, _, path = arg.partition("=")
                    est = self._kernel_saving(target_id, Path(path).name)
                elif arg in alone:
                    est = alone[arg]
                saved[arg] = None if est is None else round(est, 3)
            known = [v for v in saved.values() if v is not None]
            out.append(
                {
                    "items": [a for _, a in combo],
                    "projected_ms": round(base_ms - sum(known), 3),
                    "measured_ms": r.get("median_ms"),
                    "est_saved_ms": saved,
                }
            )
        return out

    def _kernel_saving(self, target_id: str, snapshot: str) -> float | None:
        """``est_saved_ms_per_run`` of the verified evaluation of a kernel snapshot."""
        try:
            records = self.truth.records(self.run.results_file(target_id))
        except truth.TamperError:
            return None
        for rec in records:
            if Path(str(rec.get("snapshot", ""))).name == snapshot:
                est = rec.get("est_saved_ms_per_run")
                return float(est) if est is not None else None
        return None

    def _integration_items(self) -> tuple[list[tuple[str, str]], dict[str, str | None]]:
        """Kernel winners + the best transform per idea, and their snapshots' sha256.

        A transform counts when it is part of a passing ``evaluate_e2e`` record that
        beat the baseline, alone or with other transforms and kernels (the systems
        agent evaluates its transforms on top of the kernel winners); of every idea
        (file stem) the version of the fastest such record. Only records kernel-agent
        wrote and snapshots that are still the evaluated files count; anything else
        is ignored (and reported as tampering)."""
        items: list[tuple[str, str]] = []
        digests: dict[str, str | None] = {}
        for target_id, best in self._kernel_bests():
            kernel = str(self.run.history_dir(target_id) / Path(best["snapshot"]).name)
            items.append(("kernel", f"{target_id}={kernel}"))
            digests[kernel] = best.get("snapshot_sha256")
            if (rewrite := region.verified_rewrite(self.run, target_id)).exists():
                digests[str(rewrite)] = self.truth.expect(rewrite)  # the export checks it
        best_tf: dict[str, tuple[float, str, str | None]] = {}
        for rec, snaps in self._e2e_records():
            for snap in snaps:
                if snap is None:
                    continue
                stem = ledger.snapshot_stem(snap[0])
                if stem not in best_tf or rec["speedup"] > best_tf[stem][0]:
                    best_tf[stem] = (rec["speedup"], *snap)
        for _, transform, digest in best_tf.values():
            items.append(("transform", transform))
            digests[transform] = digest
        return items, digests

    def _integration_composite(
        self, digests: dict[str, str | None]
    ) -> tuple[list[tuple[str, str]], dict[str, Any]] | None:
        """The fastest passing combination the systems agent measured (two or more
        transforms and kernels) as integration items, and its record (None: there is
        none); its snapshots' sha256 go to ``digests``. A combination with a file that
        is not a verified snapshot (a changed transform, a kernel candidate) is skipped."""
        best: tuple[list[tuple[str, str]], dict[str, Any]] | None = None
        files: dict[str, str | None] = {}
        for rec, snaps in self._e2e_records():
            kernels = [self._kernel_snapshot(k) for k in rec.get("kernels") or []]
            parts = [*snaps, *kernels]
            if len(parts) < 2 or None in parts:
                continue
            if best is None or rec["speedup"] > best[1]["speedup"]:  # first of equals wins
                combo = [("transform", s[0]) for s in snaps if s]
                combo += [("kernel", f"{k[0]}={k[1]}") for k in kernels if k]
                files = {s[0]: s[1] for s in snaps if s} | {k[1]: k[2] for k in kernels if k}
                best = (combo, rec)
        digests.update(files)
        return best

    def _e2e_records(self) -> list[tuple[dict[str, Any], list[tuple[str, str | None] | None]]]:
        """Passing ``evaluate_e2e`` records faster than the baseline, each with the snapshot
        path and sha256 of its transforms (None: no longer the evaluated file)."""
        try:
            records = self.truth.records(self.run.results_file())
        except truth.TamperError:
            return []
        history = self.run.history_dir()
        out = []
        for rec in records:
            if not rec.get("passed") or float(rec.get("speedup") or 0.0) <= 1.0:
                continue
            shas = rec.get("transforms_sha256") or []
            snaps: list[tuple[str, str | None] | None] = []
            for i, name in enumerate(rec.get("transforms") or []):
                snap, digest = history / Path(name).name, (shas[i] if i < len(shas) else None)
                snaps.append((str(snap), digest) if self.truth.snapshot_ok(snap, digest) else None)
            out.append((rec, snaps))
        return out

    def _kernel_snapshot(self, item: str) -> tuple[str, str, str | None] | None:
        """(target, snapshot, sha256) of a kernel of an ``evaluate_e2e`` record
        (``target=path``, usually the agent's copy of a snapshot): the verified
        snapshot of a correct evaluation with that file name, else None."""
        target_id, _, path = item.partition("=")
        if target_id not in self.run.target_ids():
            return None
        if not region.rewrite_ok(self.run, target_id, self.truth):  # a region: its rewrite
            return None
        try:
            records = self.truth.records(self.run.results_file(target_id))
        except truth.TamperError:
            return None
        snap = self.run.history_dir(target_id) / Path(path).name
        for rec in records:
            if rec.get("correct") and Path(str(rec.get("snapshot", ""))).name == snap.name:
                digest = rec.get("snapshot_sha256")
                if self.truth.snapshot_ok(snap, digest):
                    return target_id, str(snap), digest
        return None

    def _with_reference(
        self, baseline: dict[str, Any], accepted: list[tuple[str, str]], previous: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The accepted kernels under the workload's reference optimisations
        (``strong_baseline.py``): do they survive its torch.compile path, and beat it?
        Measured only when ``analyze`` measured a compiled baseline; informational
        (``accepted`` and ``final`` stay the greedy result)."""
        compiled = strong_baseline.compiled_ms(baseline)
        kernels = [a for k, a in accepted if k == "kernel"]
        if compiled is None or not kernels:
            return None
        items = [*kernels, strong_baseline.REFERENCE_LABEL]
        known = previous.get("reference") or {}
        if known.get("items") == items:  # re-integration of the same kernels
            return dict(known)
        cli = ["--warmup", "2", "--iters", "5"]
        for arg in kernels:
            cli += ["--kernel", arg]
        cli += ["--transform", str(strong_baseline.REFERENCE_TRANSFORM)]
        r = self._worker("e2e", *cli, *self.truth.worker_args())
        record = {"items": items, **strong_baseline.combination(r, compiled)}
        log(
            f"integrate: reference optimisations + {len(kernels)} kernel(s): "
            + strong_baseline.combination_text(record)
        )
        ledger.event(
            self.run,
            "reference_combination",
            verdict=record["verdict"],
            median_ms=record.get("median_ms"),
        )
        return record

    # ------------------------------------------------------------ improve loop (improve.py)

    async def kernel_slice(
        self, target_id: str, *, evaluations: int, digest: str, label: str
    ) -> AgentResult:
        """A fresh kernel-engineer session for ``target_id`` seeded with ``digest``."""
        profile = read_json(self.run.profile_dir / "profile.json", {})
        stats = {c["cls"]: c for c in profile.get("classes", [])}
        await self.seed_library([target_id])  # no-op once the target was seeded
        target_dir = self.run.target(target_id)
        spec = read_json(target_dir / "spec.json")
        (target_dir / "NOTES.md").touch()
        self.budget.kernel_evals = evaluations  # evaluation advice says `stop` after these
        system = prompts.engineer_prompt(
            spec,
            spec.get("capture", {}),
            spec["backends"],
            self.python,
            self.tc.summary(),
            evaluations,
            stats.get(spec["module_class"]),
        ) + self._library_note(target_id, spec)
        return await self._agent(
            f"kernel-{target_id}",
            label,
            prompt=(
                f"Continue optimising target `{target_id}`. Read the `# Improve slice` section "
                "first: it says where the previous sessions left off."
            ),
            system_append=system + digest,
            cwd=target_dir,
            mcp_tools=tool_names("evaluate_candidate", "sweep_candidate", "best_result"),
            add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
        )

    async def systems_slice(self, *, evaluations: int, digest: str, label: str) -> AgentResult:
        """A fresh systems-engineer session seeded with ``digest``."""
        plan = read_json(self.run.plan_json, {})
        self.run.transforms_dir.mkdir(parents=True, exist_ok=True)
        self.budget.transform_evals = evaluations
        system = prompts.systems_prompt(
            self.run.load()["card"],
            read_json(self.run.baseline_json, {}),
            (self.run.profile_dir / "summary.md").read_text(),
            plan.get("transforms", []),
            self.python,
            self.tc.summary(),
            evaluations,
            kernels=self._kernel_winners(),
        )
        return await self._agent(
            "systems",
            label,
            prompt=(
                "Continue designing and evaluating model-level transforms. Read the "
                "`# Improve slice` section first: it says where the previous sessions left off."
            ),
            system_append=system + digest,
            cwd=self.run.transforms_dir,
            mcp_tools=tool_names("evaluate_e2e", "run_info"),
            add_dirs=[prompts.WORKLOADS_DIR, prompts.KNOWLEDGE_DIR],
        )

    async def research(
        self, target_id: str, *, reason: str, label: str | None = None
    ) -> AgentResult:
        """A clean-context research session on a plateaued target (``research.py``).

        Read-only tools plus ``best_result``; it may write ``targets/<id>/plan.md``
        and nothing else (``run_agent(writable=...)``)."""
        target_dir = self.run.target(target_id)
        spec = read_json(target_dir / "spec.json")
        plan = research.plan_path(self.run, target_id)
        system = prompts.research_prompt(
            spec,
            spec.get("capture", {}),
            research.evidence(self.run, target_id, reason, self.truth),
            plan,
            self.tc.summary(),
        )
        return await self._agent(
            f"research-{target_id}",
            label,
            prompt=(
                f"Target `{target_id}` has plateaued ({reason}). Diagnose why from the ledger "
                f"and the files, then write the plan to {plan}."
            ),
            system_append=system,
            cwd=target_dir,
            mcp_tools=tool_names("best_result"),
            add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
            tools=[*READ_TOOLS, "Write"],
            writable=[plan],
        )

    def reprofile(self, out_dir: Path, accepted: list[dict[str, Any]]) -> dict[str, Any]:
        """Baseline + profile of the model with ``accepted`` integration items applied.

        Written to ``out_dir`` (``baseline.json``, ``profile/``); the run's own
        baseline stays the reference for every comparison.
        """
        cli = ["--iters", "3", "--out-dir", str(out_dir)]
        for item in accepted:
            cli += ["--kernel" if item["kind"] == "kernel" else "--transform", item["item"]]
        return self._worker("analyze", *cli)

    async def replan(self, round_dir: Path, context: str, label: str) -> list[dict[str, Any]]:
        """Planner session on the re-profile in ``round_dir``; returns the new targets.

        The plan is written to ``round_dir/plan.json``. Targets whose id or module
        class already exists, or whose class is not in the re-profile, are dropped.
        """
        baseline = read_json(round_dir / "baseline.json", {})
        backends = self._available_backends()
        result = await self._agent(
            "planner",
            label,
            prompt="Re-plan this run: pick new targets in the optimised model.",
            system_append=prompts.planner_prompt(
                self.run.load()["card"],
                baseline,
                (round_dir / "profile" / "summary.md").read_text(),
                backends,
                self.cfg.max_targets,
                self.python,
                self.tc.summary(),
            )
            + context,
            cwd=round_dir,
            mcp_tools=[],
            output_format={"type": "json_schema", "schema": prompts.PLAN_SCHEMA},
        )
        plan = result.structured
        if not isinstance(plan, dict):
            plan = _extract_json(result.text)
        if not isinstance(plan, dict):
            log(f"replan: planner returned no usable plan:\n{result.text[:1000]}")
            return []
        profile = read_json(round_dir / "profile" / "profile.json", {})
        known = {c["cls"] for c in profile.get("classes", [])}
        specs = [read_json(self.run.target(t) / "spec.json", {}) for t in self.run.target_ids()]
        # (class, phase) pairs already targeted; None = all phases
        taken = {(s.get("module_class"), s.get("phase")) for s in specs}
        new = []
        for t in plan.get("targets", [])[: self.cfg.max_targets]:
            problem = region.validate(t, known) or _scope(t)
            cls = t.get("module_class")
            phase = t.get("phase")
            overlap = any(c == cls and (None in (p, phase) or p == phase) for c, p in taken)
            if overlap or problem or self.run.target(t["id"]).exists():
                log(f"replan: dropping {t['id']} ({cls}): " + (problem or "already a target"))
                continue
            t["backends"] = [b for b in t.get("backends", []) if b in backends] or backends[:2]
            new.append(t)
        plan["targets"] = new
        write_json(round_dir / "plan.json", plan)
        return new

    async def report(self) -> None:
        path = write_report(self.run)
        self._mark("report")
        log(f"report: {path}")
        await self.librarian()

    # ------------------------------------------------------------ kernel library (library.py)

    def _library_arch(self) -> str | None:
        """GPU architecture whose library entries this run reads and writes (None: off)."""
        gpu = getattr(self.tc, "gpu", None)
        return gpu.arch if self.cfg.use_library and gpu is not None else None

    async def seed_library(self, target_ids: list[str]) -> None:
        """Evaluate matching library entries on targets not seeded yet (zero LLM cost)."""
        if (arch := self._library_arch()) is None:
            return
        for target_id in target_ids:
            if library.seeded(self.run, target_id) is not None:
                continue
            tried = await asyncio.to_thread(
                library.seed_target,
                self.run,
                target_id,
                arch=arch,
                keeper=self.truth,
                timeout=self.budget.eval_timeout_s,
            )
            library.remember_seed(self.run, target_id, tried)

    def _prior_suffices(self, target_id: str) -> str | None:
        """Why a target needs no agent: a prior winner already reaches the speed-of-light stop."""
        from kernel_agent.kernels.roofline import sol_signal

        if not library.seeded(self.run, target_id):
            return None
        best = best_for_target(self.run, target_id, self.truth)
        pct = sol_signal(best) if best else None
        if best is None or pct is None or pct < SOL_STOP_PCT:
            return None
        return f"prior winner {best['snapshot']} reaches {pct:.0f} % of its speed of light"

    def _library_note(self, target_id: str, spec: dict[str, Any]) -> str:
        """Engineer-prompt section: priors evaluated on the target + lessons ("" when off)."""
        if self._library_arch() is None:
            return ""
        return library.prompt_note(self.run, target_id, spec)

    def _library_store(self, accepted: list[str], final: dict[str, Any] | None) -> None:
        """Store the verified module winners and accepted kernels (never fails the run)."""
        if (arch := self._library_arch()) is None:
            return
        try:
            stored = library.store_run(
                self.run,
                self.truth,
                arch=arch,
                gpu=getattr(self.tc.gpu, "name", None),
                torch_version=getattr(self.tc, "torch_version", None),
                accepted=accepted,
                final=final,
                min_speedup=self.cfg.min_speedup,
            )
        except Exception as exc:
            log(f"library: storing this run's kernels failed: {exc!r}")
            return
        if stored:
            library.remember_store(self.run, stored)
            ledger.event(self.run, "library_store", entries=[s["entry"] for s in stored])
            log(f"library: stored {', '.join(s['entry'] for s in stored)} in {library.root()}")

    async def librarian(self) -> None:
        """Distil the run's notes + ledger into the library's lessons (a cheap agent)."""
        arch = self._library_arch()
        if arch is None or not self.cfg.librarian or not library.librarian_due(self.run):
            return
        usd = self.budget.usd_left()
        if usd is not None and usd < MIN_AGENT_USD:
            log("librarian: skipped: the USD budget is spent")
            return
        names = library.lesson_names(self.run)
        try:
            result = await self._agent(
                "librarian",
                config={
                    "claude_model": self.cfg.librarian_model or self.cfg.claude_model,
                    "effort": self.cfg.librarian_effort,
                    "max_turns_per_agent": 8,
                },
                prompt="Distil this run into the library's lessons files.",
                system_append=library.librarian_prompt(self.run, names, arch=arch),
                cwd=self.run.root,
                mcp_tools=[],
                output_format={"type": "json_schema", "schema": library.LESSONS_SCHEMA},
            )
        except Exception as exc:  # lessons are a bonus: never fail a finished run
            log(f"librarian: failed: {exc!r}")
            return
        answer = result.structured
        written = library.write_lessons(
            answer if isinstance(answer, dict) else _extract_json(result.text), names
        )
        library.remember_librarian(self.run, written)
        log(f"librarian: lessons {', '.join(p.name for p in written) or 'unchanged'}")

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


def _item_key(item: tuple[str, str]) -> str:
    """What an integration item optimises (its kernel target or its transform's idea):
    one version of each can be accepted."""
    kind, arg = item
    if kind == "kernel":
        return f"kernel {arg.partition('=')[0]}"
    return f"transform {ledger.snapshot_stem(arg)}"


def _why(result: dict[str, Any]) -> str:
    """The error of a failed step, else its first failing cases."""
    failed = [c for c in result.get("cases") or [] if not c.get("ok")]
    return str(result.get("error") or json.dumps(failed[:2], default=str))


def _cli(
    combo: list[tuple[str, str]], *, prefix: str = "--", warmup: int | None = 2, iters: int = 0
) -> list[str]:
    """Worker flags applying ``combo`` (``prefix`` ``--b-``: state B of ``e2e_ab``)."""
    cli = ["--warmup", str(warmup)] if warmup is not None and prefix == "--" else []
    cli += ["--iters", str(iters)] if iters else []
    for kind, arg in combo:
        cli += [f"{prefix}{kind}", arg]
    return cli


def _ab_key(h: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(A items, B items) of an A/B history entry: what a re-integration may reuse."""
    return tuple(h["ab"].get("a_items") or []), tuple(h["items"])


def _short(r: dict[str, Any]) -> dict[str, Any]:
    return {
        k: r.get(k)
        for k in ("status", "passed", "reason", "median_ms", "speedup", "metrics", "patches")
    }


def _scope(target: dict[str, Any]) -> str | None:
    """Normalise a planned target's ``phase`` / ``qualname_regex`` in place; returns why the
    target is unusable, or None. ``phase: all`` (or none) means a whole-class target."""
    phase = target.pop("phase", None)
    if phase in CALL_PHASES:
        target["phase"] = phase
    elif phase not in (None, "", "all"):
        return f"unknown phase {phase!r}"
    regex = target.pop("qualname_regex", None)
    if regex:
        try:
            re.compile(regex)
        except re.error as exc:
            return f"bad qualname_regex {regex!r}: {exc}"
        target["qualname_regex"] = regex
    return None


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
