"""End-to-end optimisation pipeline.

    analyze → (harness) → plan → capture → kernels → transforms → integrate → report

Deterministic steps (profiling, capture, evaluation, integration) run in GPU
worker subprocesses; creative steps (harness writing, planning, kernel writing,
model transforms) are Claude agents.  Each phase is recorded in ``run.json`` so
an interrupted run can be resumed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import re
import sys
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

from kernel_agent import (
    abtest,
    board,
    diversity,
    gpuqueue,
    hub,
    interrupt,
    ledger,
    library,
    objective,
    precisions,
    program,
    projection,
    region,
    research,
    roles,
    scheduler,
    sessions,
    strong_baseline,
    telemetry,
    toolchain,
    truth,
    workers,
)
from kernel_agent.agent import auth, prompts, web
from kernel_agent.agent.runner import AgentResult, agent_env, run_agent
from kernel_agent.agent.tools import (
    ASYNC_TOOLS,
    SessionBinding,
    best_for_target,
    build_server,
    current_records,
    ranked_for_target,
    record_candidate,
    settle,
    tool_names,
)
from kernel_agent.budget import MIN_AGENT_SECONDS, MIN_AGENT_USD, SOL_STOP_PCT, Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.dashboard import refresh
from kernel_agent.integrate import export as export_mod
from kernel_agent.integrate import owners as owners_mod
from kernel_agent.integrate import reuse as reuse_cache
from kernel_agent.integrate.export import export_optimized
from kernel_agent.kernels import evaluate, memcheck, recheck
from kernel_agent.kernels.compare import allows_reduced
from kernel_agent.native import engine as native_engine
from kernel_agent.phases import PHASES as CALL_PHASES
from kernel_agent.report import write_report
from kernel_agent.worker import call_worker
from kernel_agent.workloads import validate_metric
from kernel_agent.workloads.base import WorkloadSpec
from kernel_agent.workloads.quality import probe_messages
from kernel_agent.workspace import RunDir, coordinator_lock, read_json, write_json

PHASES = ["analyze", "plan", "capture", "kernels", "transforms", "integrate", "report"]
#: The GPU job kind (``gpuqueue.py``) of the worker commands the coordinator runs
WORKER_JOBS = {
    "capture": "capture",
    "analyze": "reprofile",
    "e2e": "integration",
    "e2e_ab": "integration",
}
#: The kinds of the integration's GPU jobs (``Orchestrator._gpu_job``)
INTEGRATION_JOBS = frozenset({"integration", "recheck", "reevaluate", "memcheck"})


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Orchestrator:
    def __init__(self, run: RunDir, cfg: OptimizeConfig) -> None:
        self.run = run
        self.cfg = cfg
        self.tc = toolchain.setup()
        self.budget = Budget.from_config(run, cfg)
        self.truth = truth.of(run)  # digests of the evaluator's ground truth (truth.py)
        # The running agent sessions by agent name (unique among running sessions): what the
        # tools of each are bound to (its own MCP server, :meth:`_agent`).
        self.bindings: dict[str, SessionBinding] = {}
        self.env = agent_env(self.tc.env, cfg.auth)  # --auth subscription: no API key vars
        self.python = sys.executable
        self.agent_results: list[AgentResult] = []
        self.phase = PHASES[0]
        # Replaced by `improve --dry-run` (simulated agent / GPU worker); None = the real ones.
        self.agent_runner: Callable[..., Awaitable[AgentResult]] | None = None
        # Wall clock and sleep of the waits for a usage limit to reset (fakes in tests).
        self.clock: Callable[[], float] = time.time
        self.sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self.worker: Callable[..., dict[str, Any]] | None = None
        # The integration's re-check of a kernel (kernels/recheck.py); None = the real one,
        # which a simulated run (dryrun.py: no captures) skips.
        self.rechecker: Callable[..., dict[str, Any]] | None = None
        # Its re-evaluation of a stale or disagreeing record (kernels/evaluate.py
        # run_evaluation); None = the real one.
        self.reevaluator: Callable[..., dict[str, Any]] | None = None
        # The memcheck of a kernel that passed its re-check (kernels/memcheck.py); None = the
        # real one, which a simulated run skips.
        self.memchecker: Callable[..., dict[str, Any]] | None = None
        # The self-test of the exported optimized/ (integrate/export.py check_export); None =
        # the real one, which a simulated run (or a fake GPU worker) skips.
        self.export_checker: Callable[..., dict[str, Any]] | None = None
        # (target, snapshot name) -> the conservative speedup of a kernel whose re-check
        # disagrees with its record (recheck.speed_warning): what ranks and projects it.
        self.speed_caps: dict[tuple[str, str], float] = {}
        self.simulated = "dry_run" in run.load()
        # Concurrent sessions (improve --agents N, coordinator.py): every engineer session may
        # write only in its own directory (:meth:`_owned`, docs/MULTIAGENT.md §3.6).
        self.ownership = False
        # improve --async-evals (issue #191): the sessions with evaluate_candidate also get
        # submit_evaluation / evaluation_result (agent/tools.py)
        self.async_evals = False
        # What every agent session is doing and the GPU's queue (sessions.py, issue #184):
        # sessions.jsonl, session_state events, improve.json sessions / gpu, costs.json time
        self.observer = sessions.Observer(
            run, reserved=lambda label: self.budget.reservations.get(label)
        )

    # ------------------------------------------------------------ creation

    @classmethod
    def create(cls, cfg: OptimizeConfig) -> Orchestrator:
        tc = toolchain.setup()
        if tc.gpu is None:
            raise SystemExit("no CUDA GPU detected; kernel-agent needs one")
        if cfg.program and not Path(cfg.program).expanduser().is_file():
            raise SystemExit(f"program file not found: {cfg.program}")
        if problem := precisions.check(cfg.quality, cfg.precisions):
            raise SystemExit(problem)
        cap = getattr(tc.gpu, "capability", None)  # run.json: what this GPU allows too (#165)
        for name, why in precisions.gpu_refused(cfg.quality, cfg.precisions, cap).items():
            log(f"precision {name}: not on this GPU: {why}")
        cfg.precisions = list(precisions.allowed(cfg.quality, cfg.precisions, cap))
        log(f"quality: {cfg.quality}; precisions: {precisions.describe(cfg.precisions, cap)}")
        log(auth.preflight(cfg.auth))  # --auth: a login / API key exists (presence only)
        log(f"resolving {cfg.model_ref}")
        card = hub.resolve(cfg.model_ref, token=cfg.hf_token, modality=cfg.modality)
        log(
            f"{card.repo_id}: modality={card.modality.value} arch={card.architectures} "
            f"params={card.params} size={card.size_gb} GB"
        )
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
        try:  # -o metric= (objective.py): one the workload can time, before a run starts
            validate_metric(spec)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        run = RunDir.create(cfg.runs_dir, card.repo_id)
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
        overrides = dict(overrides or {})
        if (wanted := overrides.pop("precisions", None)) is not None:  # recorded in run.json
            quality = data["config"].get("quality")
            if problem := precisions.check(quality, wanted):
                raise SystemExit(problem)
            cap = getattr(getattr(toolchain.setup(), "gpu", None), "capability", None)  # #165
            chosen = data["config"]["precisions"] = list(precisions.allowed(quality, wanted, cap))
            run.update(lambda d: d["config"].__setitem__("precisions", chosen))
            log(f"precisions: {precisions.describe(data['config']['precisions'])} (run.json)")
        for key in ("role_models", "role_efforts"):  # --role-model / --role-effort: per role
            if key in overrides:
                own = getattr(OptimizeConfig.from_dict(data["config"]), key)
                overrides[key] = {**own, **overrides[key]}
        cfg = OptimizeConfig.from_dict({**data["config"], **overrides})
        if "dry_run" not in data:  # simulated runs have no agents
            log(auth.preflight(cfg.auth))
        program.install(run, (overrides or {}).get("program"))  # keeps the run's edited copy
        return cls(run, cfg)

    # ------------------------------------------------------------ helpers

    def _phase_done(self, name: str) -> bool:
        return bool(self.run.load().get("phases", {}).get(name, {}).get("done"))

    def _mark(self, name: str, **info: Any) -> None:
        done = {"done": True, "at": time.strftime("%H:%M:%S"), **info}

        def mark(data: dict[str, Any]) -> None:  # keep budget notes
            data.setdefault("phases", {}).setdefault(name, {}).update(done)

        self.run.update(mark)  # under run.json's lock (workspace.update_json): threads write it

    def _phase_list(self, phase: str, key: str, item: str) -> None:
        """Append ``item`` to ``run.json`` ``phases[phase][key]`` (under the file's lock)."""

        def add(data: dict[str, Any]) -> None:
            data.setdefault("phases", {}).setdefault(phase, {}).setdefault(key, []).append(item)

        self.run.update(add)

    def allowed_precisions(self) -> tuple[str, ...]:
        """The target precisions this run allows (``--precisions``, ``precisions.py``) on
        its GPU (#165)."""
        return precisions.allowed(self.cfg.quality, self.cfg.precisions, self._capability())

    def _capability(self) -> tuple[int, int] | None:
        """The compute capability of this machine's GPU (None: none detected)."""
        cap = getattr(getattr(getattr(self, "tc", None), "gpu", None), "capability", None)
        return (int(cap[0]), int(cap[1])) if cap else None

    def _refused(self, target_id: str) -> str | None:
        """Why the kernels of a target may not be used in this run (its precision is not
        allowed, or this GPU cannot run it: ``precisions.py``), or None."""
        spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
        allowed = self.allowed_precisions()
        return precisions.refusal(precisions.of_spec(spec), allowed, self._capability())

    def _available_backends(self) -> list[str]:
        avail = [b for b in self.cfg.backends if self.tc.backends.get(b)]
        if not avail:
            raise SystemExit(f"none of the requested backends {self.cfg.backends} is available")
        return avail

    def _worker(self, command: str, *args: str) -> dict[str, Any]:
        if command == "capture":  # a target's tolerance tier follows the run's quality mode
            allowed = ",".join(self.allowed_precisions())  # and its precision must be allowed
            args = (*args, "--quality", self.cfg.quality, "--precisions", allowed)
        # its kind in the GPU job queue; its class: the integration's (_gpu_job) inside one
        with gpuqueue.tagged(WORKER_JOBS.get(command, "capture"), self.run):
            return (self.worker or call_worker)(self.run, command, *args)

    def _gpu_job(self, kind: str) -> contextlib.AbstractContextManager[gpuqueue.Job]:
        """Tag the coordinator's GPU work of ``kind`` for the GPU job queue (``gpuqueue.py``):
        background work, except the integration's jobs (:data:`INTEGRATION_JOBS`): class
        ``deadline`` once the run is in the time kept for its final integration, ``eval``
        while the final integration's estimate fills the share of ``--max-hours`` kept for it
        (a re-integration's measurements then progress; the final one reuses them)."""
        job_class = None
        if kind in INTEGRATION_JOBS:
            budget = self.budget
            left = budget.agent_seconds_left()
            share = scheduler.INTEGRATION_SHARE * (budget.max_hours or 0.0) * 3600
            if left is not None and left < MIN_AGENT_SECONDS:
                job_class = gpuqueue.DEADLINE
            elif share and budget.final_reserve_s >= share:
                job_class = gpuqueue.EVAL
        return gpuqueue.tagged(kind, self.run, job_class=job_class)

    @contextlib.contextmanager
    def gpu_access(self, mode: str) -> Iterator[None]:
        """How the agents of the sessions started inside reach the GPU (``--agent-gpu``,
        #185): their environment (``runner.agent_env``: ``tool`` hides the GPU from their
        Bash commands; with clean timing on, ``hygiene.py``, their builds' ``MAX_JOBS``) and
        what their prompts say (``prompts.gpu_access``)."""
        before = self.env
        self.env = agent_env(self.tc.env, self.cfg.auth, gpu=mode)
        try:
            with prompts.gpu_access(mode):
                yield
        finally:
            self.env = before

    def _session_config(
        self, role: str | None, config: dict[str, Any] | None, label: str | None = None
    ) -> OptimizeConfig:
        """The config of one session of ``role``: the run's with ``config`` and the role's
        model, effort and turns (``roles.session_config``), its USD capped (``Budget``; with
        concurrent sessions, minus what the others reserved: ``label`` is the session's)."""
        cfg = dataclasses.replace(self.cfg, **(config or {}))
        return self.budget.agent_config(roles.session_config(role, cfg), label)

    async def _agent(
        self,
        name: str,
        label: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        role: str | None = None,
        context: str = "",
        **kwargs: Any,
    ) -> AgentResult:
        """Run an agent session; ``label`` keys its ``costs.json`` entry (default: ``name``),
        ``role`` (default: the one of ``name``) decides its model, effort, turns and tools
        (``roles.py``; ``config`` overrides fields of the run's config before that),
        ``binding`` (in ``kwargs``, a :class:`~kernel_agent.agent.tools.SessionBinding`) is
        what the session's own tools are bound to (its target, worker, evaluation budget;
        labelled ``label``): every session gets its own MCP server (``build_server``).

        The prompt in cache order (#181, docs/MULTIAGENT.md §3.12.2): the system prompt is
        ``system_append`` (a split role's ``prompts.stable_prefix``) and the role's notes
        (program, documentation), the same for every session of the role; the first message
        is ``context`` (the target block and the digest), the budget note, then ``prompt``.
        A session stopped at a usage limit is resumed once the limit resets
        (:meth:`_wait_for_limit`). A run that is stopping (Ctrl-C) starts none."""
        interrupt.check()
        role = roles.role_of(name) if role is None else role
        spec = roles.get(role)
        binding = kwargs.pop("binding", None) or SessionBinding()
        binding = dataclasses.replace(binding, label=label or name, role=binding.role or role or "")
        # --async-evals (#191): a session that evaluates kernels may submit them, too
        async_evals = self.async_evals and "evaluate_candidate" in spec.mcp_tools
        server = build_server(
            self.run, self.budget, self.truth, binding, env=self.env, async_evals=async_evals
        )
        tag: dict[str, Any] = {"label": label} if label else {}
        prog = program.for_agent(self.run, name, log)  # re-read: humans may edit it mid-run
        ledger.event(self.run, "agent_start", agent=name, program_sha256=prog.sha256, **tag)
        result = AgentResult(name=name)
        timeout = self.budget.start_agent(name, label=label or name)
        cfg = self._session_config(role, config, label or name)
        kwargs.setdefault("mcp_tools", roles.mcp_tools(role))  # the role's tools (roles.py)
        if spec.builtin_tools is not None:
            kwargs.setdefault("tools", list(spec.builtin_tools))
        kwargs["system_append"] += prog.prompt_note(name)
        if cfg.allow_web:  # when to look things up, where, citations (issue #125)
            kwargs["system_append"] += prompts.web_note(name, web.domains(cfg.web_domains))
        kwargs["system_append"] += prompts.docs_note(name)  # doc_search / doc_read (#177)
        if board.active(self.run) is not None and role in board.ROLES:  # the blackboard (#187)
            kwargs["mcp_tools"] = [*kwargs["mcp_tools"], *tool_names(*board.TOOLS)]
            kwargs["system_append"] += prompts.board_note(name)
        if async_evals:  # submit_evaluation / evaluation_result and their advice semantics
            kwargs["mcp_tools"] = [*kwargs["mcp_tools"], *tool_names(*ASYNC_TOOLS)]
            kwargs["system_append"] += prompts.async_note()
        budget = self.budget.prompt_note(name, cfg, kwargs["mcp_tools"])
        # the session's own part, after everything its role's sessions share
        kwargs["prompt"] = prompts.first_message((context, budget), kwargs["prompt"])
        if self.agent_runner is None:  # real sessions: build the doc library meanwhile, once
            from kernel_agent import doclib

            hosts = web.domains(cfg.web_domains)
            doclib.prepare_in_background(fetch=cfg.allow_web, hosts=hosts, log=log)
        waits = 0
        # the time its GPU jobs wait behind other jobs is not the session's (gpuqueue.py)
        clock = gpuqueue.SessionClock(
            label or name,
            cap=self.budget.agent_seconds_left,
            extend=functools.partial(self.budget.extend_deadline, name),
        )
        # its states and time split (sessions.py): tool spans, GPU jobs, turns
        tracker = self.observer.open(
            label or name, agent=name, role=role or "", arm=binding.target_id
        )
        while True:
            timer = asyncio.timeout(timeout)
            self.bindings[name] = binding
            try:
                async with tracker.running(result), timer, clock.running(timer):
                    result = await (self.agent_runner or run_agent)(
                        name,
                        role=role,
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
            except auth.AuthError as exc:  # --auth: the session would bill the wrong account
                refused = {"agent": name, "api_key_source": result.api_key_source}
                self.budget.note(self.phase, "auth_refused", {**refused, "reason": str(exc)})
                raise
            finally:
                self.budget.end_agent(name)
                self.bindings.pop(name, None)
            if result.usage_limit is None or result.timed_out:
                break
            if not await self._wait_for_limit(name, result, waits):
                break
            waits += 1
            timeout = self.budget.start_agent(name, worked_s=result.seconds, label=label or name)
            cfg = self._session_config(role, config, label or name)
            if result.session_id:  # continue it (else the same prompt in a new session)
                kwargs.update(prompt=auth.RESUME_PROMPT, resume=result.session_id)
        if async_evals and (left := await settle(self.run, label or name)):
            log(f"agent {name}: {left} submitted evaluation(s) finished after it ended")
        timing = tracker.close(result)
        self.agent_results.append(result)
        if self.budget.gate is not None and result.usage_limit is None:
            self.budget.gate.progress()  # past the limit: the next one is a first wait again
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
            # its helpers (#186): delegations, tool calls, seconds, models; their USD is in "usd"
            **({"subagents": result.subagents} if result.subagents else {}),
            # the role, its model and effort, its tokens (#181: report "Usage per role")
            **({"role": role} if role else {}),
            "model": result.model or cfg.claude_model,
            "effort": cfg.effort,
            **({"usage": result.usage} if result.usage else {}),
            **({"first_usage": result.first_usage} if result.first_usage else {}),
            **({"models": result.model_usage} if result.model_usage else {}),
            "session_id": result.session_id,
            "program_sha256": prog.sha256,
            "auth": cfg.auth,
            "api_key_source": result.api_key_source,
            "billing": auth.billing(result.api_key_source, self.env),
            **({"timed_out": True} if result.timed_out else {}),
            **({"gpu_wait_s": round(clock.waited, 1)} if clock.waited else {}),
            "time": timing,  # seconds per part of its time split (sessions.PARTS)
            **({"usage_limit_waits": waits} if waits else {}),
            **({"usage_limit": result.usage_limit.to_dict()} if result.usage_limit else {}),
            **({"web": web.summary(result.web)} if result.web else {}),
        }
        write_json(self.run.root / "costs.json", costs)
        web.record(self.run.root, label or name, result.web)  # research/sources.jsonl
        return result

    async def _wait_for_limit(self, name: str, result: AgentResult, waits: int) -> bool:
        """Sleep until the usage limit ``result`` stopped at resets (``auth.wait_seconds``)
        and return True: the session is resumed. False when it resets only after the time
        budget ends, or after ``auth.MAX_LIMIT_WAITS`` waits: the budget is then ``blocked``,
        so no further agent starts and integrate + report run on what exists."""
        limit = result.usage_limit
        assert limit is not None
        if self.budget.gate is not None:  # concurrent sessions: one wait for all of them
            return await self._wait_on_gate(name, result)
        wait = auth.wait_seconds(limit, waits, self.clock())
        left = self.budget.agent_seconds_left()
        resume_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.clock() + wait))
        item = {"agent": name, "session_id": result.session_id, **limit.to_dict()}
        item.update(wait_min=round(wait / 60, 1), resume_at=resume_at)
        if waits >= auth.MAX_LIMIT_WAITS:
            why = f"still limited after {waits} waits"
        elif left is not None and wait > left - MIN_AGENT_SECONDS:
            why = f"it resets at {resume_at}, after the time budget ends"
        else:
            log(f"agent {name}: usage limit ({limit.message[:120]}); resuming at {resume_at}")
            self.budget.note(self.phase, "usage_limit", item)
            await self.sleep(wait)
            return True
        self.budget.blocked = f"usage limit: {why}"
        log(f"agent {name}: {self.budget.blocked}; no further agent starts")
        self.budget.note(self.phase, "usage_limit_stop", {**item, "reason": self.budget.blocked})
        return False

    async def _wait_on_gate(self, name: str, result: AgentResult) -> bool:
        """:meth:`_wait_for_limit` with concurrent sessions (``coordinator.RateGate``): a usage
        limit is the account's, so the first session stopped at it closes the shared gate
        until it resets (no session starts meanwhile) and every session stopped at it waits
        on the gate and is resumed when it opens. One wait for all of them: its back-off and
        ``auth.MAX_LIMIT_WAITS`` count closings in a row, not each session's waits."""
        gate, limit = self.budget.gate, result.usage_limit
        assert gate is not None and limit is not None
        item = {"agent": name, "session_id": result.session_id, **limit.to_dict()}
        if (closed := gate.closed()) is None:  # the first session at this limit
            wait = auth.wait_seconds(limit, gate.waits, self.clock())
            left = self.budget.agent_seconds_left()
            resume_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.clock() + wait))
            item.update(wait_min=round(wait / 60, 1), resume_at=resume_at, shared=True)
            why = None
            if gate.waits >= auth.MAX_LIMIT_WAITS:
                why = f"still limited after {gate.waits} waits"
            elif left is not None and wait > left - MIN_AGENT_SECONDS:
                why = f"it resets at {resume_at}, after the time budget ends"
            if why is not None:
                self.budget.blocked = f"usage limit: {why}"
                log(f"agent {name}: {self.budget.blocked}; no further agent starts")
                note = {**item, "reason": self.budget.blocked}
                self.budget.note(self.phase, "usage_limit_stop", note)
                return False
            gate.close(wait, limit.message)
            log(
                f"agent {name}: usage limit ({limit.message[:120]}); no session starts until "
                f"{resume_at}, and every session stopped at it resumes then"
            )
            self.budget.note(self.phase, "usage_limit", item)
        else:  # a limit that resets later (another window) keeps the gate closed longer
            gate.close(auth.wait_seconds(limit, max(gate.waits - 1, 0), self.clock()), closed)
            log(f"agent {name}: usage limit; waiting with the other sessions ({closed})")
        await gate.wait()
        return self.budget.blocked is None

    # ------------------------------------------------------------ phases

    async def analyze(self) -> None:
        log("analyze: loading model, measuring baseline, profiling")
        # GPU worker calls run off the event loop (as every one in a coroutine): other
        # sessions and their tools keep going meanwhile
        result = await asyncio.to_thread(call_worker, self.run, "analyze", "--iters", "3")
        if "error" in result and self.cfg.allow_harness_agent:
            log("analyze: built-in workload failed; asking Claude to write a harness")
            await self.write_harness(result["error"])
            result = await asyncio.to_thread(call_worker, self.run, "analyze", "--iters", "3")
        if "error" in result:
            raise SystemExit(f"analyze failed:\n{result['error']}")
        log(
            f"analyze: baseline {result['median_ms']:.1f} ms, deterministic="
            f"{result['deterministic']}, peak {result['peak_mem_gb']:.2f} GB"
        )
        if (metric := objective.describe(result)) is not None:  # -o metric=ttfa, say
            log(f"analyze: {metric}")
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
            add_dirs=[prompts.WORKLOADS_DIR],
        )
        if not self.run.harness.exists():
            raise SystemExit("harness agent did not produce harness.py")
        harness = str(self.run.harness)
        self.run.update(lambda data: data["workload"].__setitem__("harness", harness))

    async def plan(self) -> None:
        data = self.run.load()
        baseline = read_json(self.run.baseline_json, {})
        summary = (self.run.profile_dir / "summary.md").read_text()
        backends = self._available_backends()
        log(
            f"plan: asking the planner ({roles.model_for('planner', self.cfg)}) for up to "
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
                quality=self.cfg.quality,
                precisions=self.allowed_precisions(),
                backend_record=self._backend_record(),
            ),
            cwd=self.run.root,
            output_format={"type": "json_schema", "schema": self._plan_schema()},
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
            if problem := (
                region.validate(t, known)
                or _scope(t)
                or _precision(t, self.cfg.quality, self.allowed_precisions())
            ):
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
                + (f" [{t['precision']}: {t.get('precision_why', '')}]" if "precision" in t else "")
            )
        for t in plan.get("transforms", []):
            log(f"plan: transform {t['id']}: {t['idea'][:120]}")
        self._mark("plan", targets=[t["id"] for t in targets])

    def _plan_schema(self) -> dict[str, Any]:
        """The planner's output schema: its precisions are the ones the run allows."""
        return prompts.plan_schema(self.allowed_precisions())

    async def capture(self) -> None:
        plan = read_json(self.run.plan_json, {})
        self._mark("capture", targets=await self.capture_targets(plan.get("targets", [])))

    async def capture_targets(self, targets: list[dict[str, Any]]) -> list[str]:
        """:meth:`_capture`, then the refactor step of the region targets; returns the ids.
        The captures (GPU worker processes) run in a thread: not on the event loop."""
        kept = await asyncio.to_thread(self._capture, targets)
        for t in targets:
            if region.is_region(t) and t["id"] in kept and not await self.refactor(t["id"]):
                kept.remove(t["id"])
        return kept

    def _capture(self, targets: list[dict[str, Any]]) -> list[str]:
        """Create ``targets/<id>/spec.json`` and capture each target (of a region target: its
        parent class, see ``region.py``); returns the captured ids."""
        kept = []
        for t in targets:
            allowed, cap = self.allowed_precisions(), self._capability()
            if why := precisions.refusal(t.get("precision"), allowed, cap):
                log(f"capture: {t['id']} refused: {why}")  # planned before the run's list
                ledger.event(self.run, "capture_refused", target=t["id"], why=why)
                continue
            target_dir = self.run.target(t["id"])
            (target_dir / "candidates").mkdir(parents=True, exist_ok=True)
            spec = {**t}
            write_json(target_dir / "spec.json", spec)
            parent = ["--parent"] if region.is_region(t) else []
            log(f"capture: {t['id']} ({t['parent_class'] if parent else t['module_class']})")
            info = self._worker("capture", "--target", t["id"], *parent)
            if "error" in info:
                log(f"capture: {t['id']} failed, dropping target:\n{info['error'][-800:]}")
                # the reason stays with it (an unverifiable capture: native/engine.py digest)
                failed = read_json(target_dir / "spec.json", {}) or spec
                write_json(
                    target_dir / "spec.failed.json", {**failed, "capture_error": info["error"]}
                )
                (target_dir / "spec.json").unlink(missing_ok=True)
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
                writable=[rewrite],
            )
            result = await asyncio.to_thread(install)
        spec["rewrite"] = region.summary(result)
        write_json(target_dir / "spec.json", spec)
        if result.get("verified"):
            how = "bitwise" if result.get("bitwise") else "within one ulp"
            log(f"refactor: {target_id}: rewrite verified ({how}); capture {spec['module_class']}")
            info = await asyncio.to_thread(self._worker, "capture", "--target", target_id)
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
            if why := self._refused(target_id):  # captured before the run's --precisions
                log(f"kernels: skipping {target_id}: {why}")
                return
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
                await self.dossier(target_id)  # research.md first (web on, issue #125)
                first, _ = workers.rounds(self.cfg.evaluations_per_target, self.cfg.reseed_workers)
                team = self.kernel_seeds(target_id, first)
                if not team:  # one engineer session in the target directory
                    target_dir = self.run.target(target_id)
                    spec = read_json(target_dir / "spec.json")
                    evals = self.cfg.evaluations_per_target
                    (target_dir / "NOTES.md").touch()
                    await self._agent(
                        f"kernel-{target_id}",
                        prompt=(
                            f"Optimise target `{target_id}`. Start by reading "
                            "reference_source.py and spec.json, then write and evaluate "
                            "candidates."
                        ),
                        system_append=self._stable("kernel"),
                        context=self._engineer_target(spec, spec["backends"], evals, stats),
                        cwd=target_dir,
                        add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
                        binding=SessionBinding(
                            role="kernel",
                            target_id=target_id,
                            evaluations=self.cfg.evaluations_per_target,
                        ),
                    )
            if team:  # every worker session takes its own --parallel slot
                await self._kernel_workers(target_id, team, sem)
            best = best_for_target(self.run, target_id, self.truth)  # across the workers
            log(
                f"kernels: {target_id} best = "
                + (f"{best['speedup']}x ({best['snapshot']})" if best else "none correct")
            )
            self._phase_list("kernels", "finished", target_id)

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
            system_append=self._stable("systems"),
            context=prompts.systems_target(
                data["card"],
                baseline,
                summary,
                plan.get("transforms", []),
                self.cfg.transform_evaluations,
                kernels=self._kernel_winners(),
            ),
            cwd=self.run.transforms_dir,
            add_dirs=[prompts.WORKLOADS_DIR, prompts.KNOWLEDGE_DIR],
            binding=SessionBinding(role="systems", evaluations=self.cfg.transform_evaluations),
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

    def island_count(self, target_id: str) -> int:
        """The islands of a target in ``improve`` (``--islands``, an alias of
        ``--seeds-per-target``; workers.py, issue #189): 1 (its classic session) unless set."""
        if self.cfg.seeds_per_target is None:
            return 1
        spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
        return workers.count(self.cfg.seeds_per_target, self._share(spec))

    async def worker_session(
        self,
        target_id: str,
        seed: workers.Seed,
        team: list[workers.Seed],
        *,
        prompt: str,
        digest: str = "",
        label: str | None = None,
        note: str | None = None,
    ) -> AgentResult:
        """One worker session of a target: its own directory and tools bound to it (paths,
        the ``worker`` of its ledger rows, its evaluation budget), the engineer prompt with
        the worker's approach and backends, and a ``# Worker`` section (``team``: the round;
        ``note``: an island's ``# Island`` section instead, ``workers.island_note``)."""
        profile = read_json(self.run.profile_dir / "profile.json", {})
        stats = {c["cls"]: c for c in profile.get("classes", [])}
        spec = read_json(self.run.target(target_id) / "spec.json")
        cwd = workers.prepare(self.run, target_id, seed.worker)
        name = workers.agent_name(target_id, seed.worker)
        backends = list(seed.backends) or spec["backends"]
        context = self._engineer_target(
            spec, backends, seed.evaluations, stats, approach=seed.approach
        ) + (workers.prompt_note(target_id, seed, team) if note is None else note)
        return await self._agent(
            name,
            label,
            prompt=prompt,
            system_append=self._stable("kernel"),
            context=context + digest,
            cwd=cwd,
            add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
            binding=SessionBinding(
                role="kernel",
                target_id=target_id,
                worker=seed.worker,
                agent=name,
                evaluations=seed.evaluations,
            ),
            **self._owned(cwd),
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
            self._phase_list("kernels", "workers", key)

        await asyncio.gather(*(job(s, team, f"{target_id}/w{s.worker}") for s in team))
        _, second = workers.rounds(self.cfg.evaluations_per_target, self.cfg.reseed_workers)
        again = workers.reseeds(self.run, target_id, team, second, self.truth) if second else []
        if again:
            starts = ", ".join(f"{s.parent} ({s.parent_speedup:.2f}x)" for s in again)
            log(f"kernels: {target_id}: round 2 of its workers from {starts}")
            await asyncio.gather(*(job(s, again, f"{target_id}/r2w{s.worker}") for s in again))

    def _kernel_best(self, target_id: str) -> dict[str, Any] | None:
        """The verified best record of a target when its kernel is worth integrating. A
        snapshot with a speed cap (``speed_caps``) ranks by it: its ``speedup`` is the
        conservative one, ``evaluator_speedup`` the record's."""
        best = None
        allowed = self.allowed_precisions()
        for rec in ranked_for_target(self.run, target_id, self.truth):
            if not precisions.tier_allowed(rec.get("tolerance_tier"), allowed):
                continue  # evaluated in the tier of a precision the run does not allow
            if best is not None and rec["speedup"] <= best["speedup"]:
                break  # ranked by the records' speedups, which a cap only lowers
            cap = self.speed_caps.get((target_id, Path(str(rec["snapshot"])).name))
            if cap is not None and cap < rec["speedup"]:
                rec = {**rec, "speedup": cap, "evaluator_speedup": rec["speedup"]}
            if best is None or rec["speedup"] > best["speedup"]:
                best = rec
        # a region target's kernel needs its verified rewrite, unchanged (region.py)
        ok = region.rewrite_ok(self.run, target_id, self.truth)
        return best if best and best["speedup"] >= self.cfg.min_speedup and ok else None

    def _kernel_bests(self) -> list[tuple[str, dict[str, Any]]]:
        """(target_id, verified best record) of every kernel worth integrating (none of a
        target at a precision the run does not allow: :meth:`skipped_targets`)."""
        ids = [t for t in self.run.target_ids() if self._refused(t) is None]
        bests = [(t, self._kernel_best(t)) for t in ids]
        return [(t, best) for t, best in bests if best is not None]

    def skipped_targets(self) -> list[dict[str, str]]:
        """``{"target", "precision", "reason"}`` of every target the integration leaves
        out: its precision is not one the run allows (``--precisions``, ``precisions.py``)."""
        skipped = []
        for target_id in self.run.target_ids():
            if why := self._refused(target_id):
                spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
                skipped.append(
                    {"target": target_id, "precision": precisions.of_spec(spec), "reason": why}
                )
        return skipped

    def _kernel_winners(self) -> list[tuple[str, str, float]]:
        """(target_id, snapshot path, module speedup) of every kernel worth integrating,
        for prompts: the snapshot copies in the targets' own ``history/``."""
        winners = []
        for target_id, best in self._kernel_bests():
            path = self.run.target(target_id) / "history" / Path(best["snapshot"]).name
            winners.append((target_id, str(path), float(best["speedup"])))
        return winners

    async def integrate(self, reuse: bool = False) -> None:
        """:meth:`_integrate_sync` in a thread: its A/B steps (GPU worker processes, minutes
        each, up to hours in all) do not hold the event loop, so other agent sessions and
        their tools keep going meanwhile."""
        await asyncio.to_thread(self._integrate_sync, reuse)

    def _integrate_sync(self, reuse: bool = False) -> None:
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
        the next candidate, then the version swaps (:meth:`_swaps`): B = A with
        one accepted item replaced by another version of it (the target's best
        kernel, say, when the combination holds an older one). An item that
        overlaps accepted items (:mod:`kernel_agent.integrate.owners`: it changes
        the modules they changed, or replaces a module they changed something
        inside, as a LocEnc kernel and a CUDA graph of the LocEnc do) is also
        tried in their place, after its addition: B = A with those items removed
        and it in their place (``kind: replace``). Sets that cannot be undone
        in-process, or did not fit in one process together (``oom``: out of GPU
        memory anywhere in the step, the quality checks and the perceptual gate
        included, never a quality verdict), are measured in two processes back to
        back instead (``abtest.SEPARATE_ITERS`` runs each, A measured again; after
        an ``oom`` with ``--expandable-segments``); a step that still runs out of
        memory is recorded as ``oom``, which a re-integration measures again.
        ``integration.json`` keeps each step's ``ab`` record (a swap's with
        ``kind: swap``, ``old`` and ``new``), the ``projection`` (baseline −
        Σ est. saved ms, nested kernels and items of overlapping modules counted
        once; a later set from the set before it) of every accepted set and
        the ``owners`` (touched and owned modules) of the accepted items.

        ``reuse`` (re-integrations of the improve loop) takes the measurements
        of the same content (:mod:`kernel_agent.integrate.reuse`: the same A and
        B for an A/B, by the sha256 of every file their items load, under the
        same evaluator schema, baseline and A/B rounds) from the previous
        ``integration.json`` instead of measuring them again, whatever the
        snapshot names (each evaluation snapshots its files anew), and tries
        the versions it swapped in or accepted again first. ``reuse`` counts
        the reused and the measured steps.

        Everything comes from the verified truth (``truth.py``): the baseline
        latency the orchestrator recorded, records and snapshots whose digests
        match, and a reuse cache only from an unmodified ``integration.json``.

        The kernels of a target at a precision the run does not allow
        (``--precisions``, ``precisions.py``: 4-bit unless asked for), and kernels
        evaluated in the tolerance tier of one, are left out; ``skipped`` says which
        targets and why (:meth:`skipped_targets`).
        """
        base_ms = self.truth.baseline_ms()
        skipped = self.skipped_targets()  # at a precision the run does not allow
        for s in skipped:
            log(f"integrate: skipping {s['target']}: {s['reason']}")
        items, digests = self._integration_items()
        composite = self._integration_composite(digests)
        log(
            f"integrate: {len(items)} candidate optimisations"
            + (f" + the combination of exp {composite[1].get('exp')}" if composite else "")
        )

        history: list[dict[str, Any]] = []
        integration = self.run.root / "integration.json"
        previous = self.truth.load_json(integration) if reuse else {}
        # by content, not by snapshot name: every evaluation snapshots its files anew (#93)
        keys = reuse_cache.Keys(self.run, self._reuse_context(base_ms))
        known, migrated = self._reusable(previous or {}, keys)  # a pre-#93 file: migrated
        if migrated:
            log(f"integrate: {migrated}")
        paired = {  # a crash or timeout can be transient (another process, OOM): measure again
            h["reuse_key"]: h for h in known if h.get("ab") and not _transient(h)
        }
        counts = {"reused": 0, "measured": 0}
        irreversible = set((previous or {}).get("irreversible") or [])  # not undone in-process
        crowded: list[set[str]] = []  # held by both states of an A/B that ran out of memory
        owners = owners_mod.Owners()  # the modules each item changes (integrate/owners.py)
        versions = self._previous_versions(previous or {}, digests)
        rechecks: list[dict[str, Any]] = []
        if self.cfg.recheck:  # fresh inputs, separate processes; failing kernels are refused
            items, composite, rechecks = self._recheck_kernels(
                items, composite, previous or {}, digests, versions
            )
            refused = {r["item"] for r in rechecks if not r.get("passed")}
            versions = [v for v in versions if v[1] not in refused]
        # and this integration's items with the content of such an item (another snapshot)
        stuck = {k for x in irreversible if (k := keys.arg(x)) is not None}
        everything = [*items, *(composite[0] if composite else []), *versions]
        irreversible.update(x for k, x in everything if stuck and keys.item((k, x)) in stuck)

        def ab(
            a: list[tuple[str, str]], b: list[tuple[str, str]], note: str = "", **swap: Any
        ) -> dict[str, Any]:
            """B's result with its ``ab`` record (decided) against A (``swap``: the
            ``kind``, ``old`` and ``new`` item of a version swap or a replacement, kept in
            its history entry)."""
            a_items = [x for _, x in a]
            key = keys.step(a, b)
            hit = paired.get(key) if key is not None else None
            counts["measured" if hit is None else "reused"] += 1
            if hit is not None:
                names = " + ".join(ledger.item_label(x) for _, x in b)
                log(f"integrate: reused {names}{note}: measured before with the same content")
            r = dict(hit) if hit is not None else self._paired(a, b, note, irreversible, crowded)
            owners.note(a, (r.get("ab") or {}).get("a_patches"))
            owners.note(b, r.get("patches"))
            record = {**(r.get("ab") or {}), "a_items": a_items}
            if record.get("a_ms") and record.get("b_ms"):
                record = abtest.judge(
                    record, min_win_rate=self.cfg.ab_min_win_rate, min_gain=self.cfg.ab_min_gain
                )
            else:  # B failed before any timed round
                why = record.get("why") or r.get("reason") or r.get("status")
                record.update(accepted=False, why=why)
            r["ab"] = record
            entry = {**swap, "items": [x for _, x in b], **_short(r), "ab": record}
            history.append({**entry, "reuse_key": key} if key is not None else entry)
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
        replaced: set[tuple[str, str]] = set()  # an item in their place won their A/B
        for item, _ in singles:
            if final is None or item in accepted or item in replaced:  # seed, or lost its place
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
            overlaps = owners.overlapping(item, [a for a in accepted if a != item])
            if not overlaps:
                continue
            olds = [o for o, _ in overlaps]
            b = _in_place(accepted, olds, item)  # the alternative: it instead of them
            names = ", ".join(ledger.item_label(o[1]) for o in olds)
            what = f"{ledger.item_label(item[1])} instead of {names}"
            shared = sorted({m for _, where in overlaps for m in where})
            log(f"integrate: {what}? they change the same modules: " + _few(shared))
            r = ab(accepted, b, f" ({what})", kind="replace", old=[o[1] for o in olds], new=item[1])
            if r.get("passed") and r["ab"]["accepted"]:
                accepted, final = b, r
                sets.append((list(accepted), r))
                replaced.update(olds)
                log(f"integrate: {what}: {r['median_ms']:.1f} ms ({abtest.describe(r['ab'])})")
            else:
                reason = r.get("reason") or r.get("status")
                if r.get("passed"):
                    reason = f"no significant gain: {abtest.describe(r['ab'])}; {r['ab']['why']}"
                log(f"integrate: keep {names}, not {Path(item[1]).name} instead ({reason})")
        for key, new in self._swaps(accepted, singles, versions, _alone(base_ms, history)):
            old = next(a for a in accepted if _item_key(a) == key)
            if _same_file(old, new):  # another snapshot of the same file: nothing to measure
                continue
            b = [new if a == old else a for a in accepted]  # in place: the same order
            what = f"{ledger.item_label(new[1])} {Path(old[1]).name} -> {Path(new[1]).name}"
            r = ab(accepted, b, f" (swap {what})", kind="swap", old=old[1], new=new[1])
            if r.get("passed") and r["ab"]["accepted"]:
                accepted, final = b, r
                sets.append((list(accepted), r))
                log(f"integrate: swap {what}: {r['median_ms']:.1f} ms ({abtest.describe(r['ab'])})")
            else:
                reason = r.get("reason") or r.get("status")
                if r.get("passed"):
                    reason = f"no significant gain: {abtest.describe(r['ab'])}; {r['ab']['why']}"
                log(f"integrate: keep {Path(old[1]).name}, not {Path(new[1]).name} ({reason})")
        projection = self._projection(base_ms, sets, history, owners.changes())
        result = {
            "baseline_ms": base_ms,
            "accepted": [{"kind": k, "item": a} for k, a in accepted],
            "final": final,
            "history": history,
            "projection": projection,
        }
        if seed is not None:
            result["composite"] = seed
        if dependent := _data_dependent(history, accepted):  # diversity.py: labelled, kept
            result["data_dependent"] = dependent
        result["reuse"] = counts  # A/B steps taken from the previous integration / measured
        if irreversible:
            result["irreversible"] = sorted(irreversible)
        if found := owners.record(accepted):  # the re-plan of a round sees what they own
            result["owners"] = found
        if rechecks:
            result["recheck"] = rechecks
        if skipped:
            result["skipped"] = skipped
        baseline = self.truth.load_json(self.run.baseline_json)  # its compiled_ms
        reference = self._with_reference(baseline, accepted, previous or {})
        if reference is not None:
            result["reference"] = reference
        write_json(integration, result)
        self.truth.seal(integration)
        if reuse:
            log(
                f"integrate: {counts['reused']} of {len(history)} measurements reused "
                f"(unchanged content), {counts['measured']} measured"
            )
        export_optimized(self.run, [(k, a, 0.0) for k, a in accepted], digests=digests)
        self._check_export(final)
        self._library_store([a for k, a in accepted if k == "kernel"], final)
        if final:
            _, vs_compiled = strong_baseline.speedups(baseline, final["median_ms"])
            projected = projection[-1]["projected_ms"]
            log(
                f"integrate: final {final['median_ms']:.1f} ms vs {base_ms:.1f} ms "
                f"= {final['speedup']}x"
                + (f" ({vs_compiled:.2f}x vs compiled)" if vs_compiled else "")
                + (f"; {spread}" if (spread := diversity.headline(final)) else "")
                + (
                    f"; projected {projected:.1f} ms"
                    if projected is not None
                    else f"; not projected ({projection[-1].get('not_additive') or 'no A/B'})"
                )
            )
        else:
            log("integrate: no optimisation survived end-to-end validation")
        self._mark("integrate", speedup=final["speedup"] if final else 1.0)

    def _check_export(self, final: dict[str, Any] | None) -> dict[str, Any] | None:
        """The self-test of the exported ``optimized/`` (``integrate/export.py``, #171): it is
        imported and applied in a fresh process outside the run directory. A failure fails
        the export: a log line naming the files the package lacks, an ``export_failed`` event
        and ``optimized/export_check.json``; the run goes on. None: not checked (nothing
        accepted, ``--no-export-check``, a simulated run)."""
        if final is None or not self.cfg.export_check:
            return None
        checker = self.export_checker
        if checker is None:
            if self.simulated or self.worker is not None:  # no GPU worker to import it in
                return None
            checker = export_mod.check_export
        expected = final.get("patches")
        with self._gpu_job("integration"):
            result = checker(self.run, call_worker, self.truth.worker_args(), expected=expected)
        if result.get("passed"):
            again = " (the same package passed before)" if result.get("reused") else ""
            log(f"integrate: export self-test passed{again}: {result.get('reason')}")
            ledger.event(self.run, "export_check", passed=True, reused=bool(result.get("reused")))
        else:
            log(f"integrate: export FAILED its self-test: {result.get('reason')}")
            missing = result.get("missing") or []
            ledger.event(self.run, "export_failed", reason=result.get("reason"), missing=missing)
        return result

    def _reuse_context(self, base_ms: float) -> dict[str, Any]:
        """What an integration measurement depends on besides its items, part of every
        reuse key (:mod:`kernel_agent.integrate.reuse`): the evaluator schema (a bump
        invalidates every earlier measurement, as in :meth:`_recheck_kernels`), the
        baseline the A/B ran against (its latency, ``baseline.json`` and the digests and
        quality mode the worker verifies) and the A/B rounds."""
        baseline = self.run.baseline_json
        return {
            "evaluator_schema": evaluate.EVALUATOR_SCHEMA,
            "baseline_ms": base_ms,
            "baseline_sha256": truth.sha256_file(baseline) if baseline.exists() else None,
            "worker_args": self.truth.worker_args(),
            "ab_rounds": self.cfg.ab_rounds,
        }

    def reusable(self) -> tuple[list[dict[str, Any]], str]:
        """What a re-integration now could take from the last integration
        (:meth:`_reusable`; the improve loop's estimate of its final integration)."""
        previous = self.truth.load_json(self.run.root / "integration.json") or {}
        if not previous.get("history"):
            return [], ""
        keys = reuse_cache.Keys(self.run, self._reuse_context(self.truth.baseline_ms()))
        return self._reusable(previous, keys)

    def _reusable(
        self, previous: dict[str, Any], keys: reuse_cache.Keys
    ) -> tuple[list[dict[str, Any]], str]:
        """The history entries of the integration ``previous`` with a ``reuse_key``, and the
        log line of a migration ("" when none). A file from before content keys (#93) has
        none: its entries get the key their step has now (``keys``) where the run proves
        what they ran with (:func:`reuse.migrate <kernel_agent.integrate.reuse.migrate>`)."""
        history = previous.get("history") or []
        if not history or any(h.get("reuse_key") for h in history):
            return [h for h in history if h.get("reuse_key")], ""
        transforms = {  # snapshots the integration measured: verified against their records
            snap[0] for _, snaps in self._e2e_records() for snap in snaps if snap is not None
        }

        def verified(arg: str) -> bool:
            if reuse_cache.as_item(arg)[0] == "transform":
                return arg in transforms
            kernel = self._kernel_snapshot(arg)
            return kernel is not None and Path(kernel[1]) == Path(arg.partition("=")[2])

        migration = reuse_cache.migrate(
            previous,
            keys,
            schema=evaluate.EVALUATOR_SCHEMA,
            baseline_ms=self.truth.baseline_ms(),
            ab_rounds=self.cfg.ab_rounds,
            perceptual=allows_reduced(self.truth.quality),
            verified=verified,
        )
        return migration.entries, migration.note()

    def _recheck_kernels(
        self,
        items: list[tuple[str, str]],
        composite: tuple[list[tuple[str, str]], dict[str, Any]] | None,
        previous: dict[str, Any],
        digests: dict[str, str | None] | None = None,
        versions: list[tuple[str, str]] | None = None,
    ) -> tuple[
        list[tuple[str, str]],
        tuple[list[tuple[str, str]], dict[str, Any]] | None,
        list[dict[str, Any]],
    ]:
        """The independent re-check (``kernels/recheck.py``) of every kernel the integration
        considers (each target's best, the kernels of the measured combination and the
        ``versions`` a re-integration tries as swaps): fresh inputs of the captured shapes
        against a freshly computed reference, reference and kernel timed in processes of
        their own. A kernel that passes it runs once under memcheck (:meth:`_memcheck`).
        A kernel that fails either is refused (a log line and a ``recheck_failed``
        event with the reason; a refused version is a record with ``passed: false``), and
        so is a combination with it. A re-integration reuses the result of the same
        snapshot (``previous``), and runs a memcheck that did not decide (skipped, failed)
        again.

        A correct kernel whose speedup disagrees with its record (``speed_disagrees``) is a
        warning (a log line and a ``recheck_speed_disagrees`` event), not a refusal: it is
        ranked and projected by its conservative speedup (``speed_caps``) and its A/B
        decides. A re-evaluation or a speed cap (:meth:`_recheck_one`) can rank another
        snapshot of the target first: that one is re-checked too and, when it passes,
        integrated instead (its sha256 goes to ``digests``)."""
        # Reuse a re-check only when it was made by the current evaluator: a schema bump
        # (e.g. timing at full clocks, #81) invalidates every earlier measurement.
        known = {
            (r.get("item"), r.get("sha256")): r
            for r in previous.get("recheck") or []
            if r.get("evaluator_schema") == evaluate.EVALUATOR_SCHEMA
        }
        records: list[dict[str, Any]] = []
        refused: set[str] = set()
        checked: dict[str, dict[str, Any]] = {}

        def check(arg: str) -> dict[str, Any]:
            if arg in checked:
                return checked[arg]
            target_id, _, path = arg.partition("=")
            snap = Path(path)
            rec = self._kernel_record(target_id, snap.name)
            sha = (rec or {}).get("snapshot_sha256")
            hit = known.get((arg, sha)) if sha else None
            result = dict(hit) if hit is not None else self._recheck_one(target_id, snap, rec)
            if memcheck.pending(result):  # passed, and no memcheck verdict yet
                result = self._memcheck(target_id, snap, result)
            checked[arg] = result
            records.append(
                {
                    **result,
                    "item": arg,
                    "target": target_id,
                    "snapshot": snap.name,
                    "sha256": sha,
                    "evaluator_schema": evaluate.EVALUATOR_SCHEMA,
                }
            )
            warn = "WARNING " if result.get("status") == recheck.SPEED_DISAGREES else ""
            log(f"integrate: {warn}recheck {target_id} ({snap.name}): {recheck.describe(result)}")
            if warn:
                self.speed_caps[(target_id, snap.name)] = float(result["conservative_speedup"])
                ledger.event(
                    self.run,
                    "recheck_speed_disagrees",
                    target=target_id,
                    snapshot=snap.name,
                    evaluator=(result.get("evaluator") or {}).get("speedup"),
                    recheck=result.get("speedup"),
                    conservative=result.get("conservative_speedup"),
                )
            if not result.get("passed"):
                refused.add(arg)
                ledger.event(
                    self.run,
                    "recheck_failed",
                    target=target_id,
                    snapshot=snap.name,
                    status=result.get("status"),
                    reason=str(result.get("reason"))[:500],
                )
            return result

        items = list(items)
        for i, (kind, arg) in enumerate(items):
            if kind != "kernel":
                continue
            current, digest = arg, None
            while _reranks(check(current)):  # the target's ranking may have changed
                top = self._kernel_item(arg.partition("=")[0])
                if top is None or top[0] == current or check(top[0]).get("passed") is not True:
                    break
                current, digest = top
            if current != arg:
                log(
                    f"integrate: {Path(current).name} ranks first after the re-check; "
                    f"it replaces {Path(arg).name}"
                )
                items[i] = (kind, current)
                if digests is not None:
                    digests[current.partition("=")[2]] = digest
        for kind, arg in [*(composite[0] if composite else []), *(versions or [])]:
            if kind == "kernel":
                check(arg)
        if refused:
            items = [(k, a) for k, a in items if not (k == "kernel" and a in refused)]
            if composite is not None and refused.intersection(a for _, a in composite[0]):
                exp = composite[1].get("exp")
                log(f"integrate: no seed from exp {exp}: one of its kernels failed the recheck")
                composite = None
        return items, composite, records

    def _memcheck(self, target_id: str, snap: Path, result: dict[str, Any]) -> dict[str, Any]:
        """A kernel snapshot that passed its re-check (``result``) once on its target's
        captured cases and their odd-size variants under ``compute-sanitizer --tool
        memcheck`` (``kernels/memcheck.py``; ``memcheck`` in the record: status, seconds,
        the sanitizer's first report). Memory errors refuse it (status ``memcheck``); a
        sanitizer that is missing or fails is recorded with its reason, and the kernel kept.
        A ``memcheck`` ledger event records each run."""
        if self.memchecker is None and self.simulated:  # nothing runs: no log line, no event
            return {**result, "memcheck": {"status": "skipped", "reason": "simulated run"}}
        capture = self.run.capture_file(target_id)
        if not capture.exists():
            check = {"status": "skipped", "reason": f"no capture file {capture}"}
        else:
            try:
                with self._gpu_job("memcheck"):
                    check = (self.memchecker or memcheck.run_memcheck)(
                        capture,
                        snap,
                        capture_sha256=self.truth.verify(capture),
                        timeout=2 * self.budget.eval_timeout_s,
                    )
            except Exception as exc:  # the check broke: recorded, the kernel is kept
                check = {"status": "error", "reason": repr(exc)[:500]}
        log(f"integrate: memcheck {target_id} ({snap.name}): {memcheck.describe(check)}")
        ledger.event(
            self.run,
            "memcheck",
            target=target_id,
            snapshot=snap.name,
            status=check.get("status"),
            seconds=check.get("seconds"),
            errors=check.get("errors"),
            reason=str(check.get("reason") or "")[:500],
        )
        result = {**result, "memcheck": check}
        return memcheck.refuse(result, check) if check.get("status") == memcheck.STATUS else result

    def _recheck_one(
        self, target_id: str, snap: Path, rec: dict[str, Any] | None
    ) -> dict[str, Any]:
        """:func:`kernels.recheck.run_recheck` of one kernel snapshot on its target's capture,
        against the verified evaluation record ``rec``. In a run with ``.truth/`` a missing or
        changed capture fails it; a run from before that layout without one skips it.

        A record from an older evaluator (:func:`kernels.evaluate.stale`) is re-evaluated
        first, and so is one whose speedup the re-check disagrees with when the kernel is
        correct on the fresh inputs (:meth:`_reevaluate`): the current evaluator's result
        is then the verdict (``reevaluated``: old and new speedup). A speedup that still
        disagrees is re-checked once more, in new processes, and the run that agrees
        better counts (``rechecks``: both). If it still disagrees, the kernel is kept with
        a warning (:func:`kernels.recheck.speed_warning`). It is refused when it is wrong on
        fresh inputs or violates integrity in either re-check, when the re-evaluation
        fails, or when a stale record's re-evaluation disagrees with a re-check that
        measures no speedup at all (≤ 1x)."""
        if self.rechecker is None and self.simulated:
            return {"status": "skipped", "passed": True, "reason": "simulated run (no captures)"}
        capture = self.run.capture_file(target_id)
        try:
            capture_sha256 = self.truth.verify(capture)
        except truth.TamperError as exc:
            return {"status": "tampered", "passed": False, "reason": str(exc)}
        if capture_sha256 is None and not capture.exists():
            return {"status": "skipped", "passed": True, "reason": f"no capture file {capture}"}
        fresh: dict[str, Any] | None = None
        why = evaluate.stale(rec) if rec is not None else None
        stale = bool(why)
        if rec is not None and why:
            fresh = self._reevaluate(target_id, snap, rec, capture_sha256, why)
            if not fresh.get("correct"):
                return _reevaluation_failed({}, rec, fresh, why)

        def run_recheck(verdict: dict[str, Any] | None) -> dict[str, Any]:
            try:
                with self._gpu_job("recheck"):
                    return (self.rechecker or recheck.run_recheck)(
                        capture,
                        snap,
                        verdict=verdict,
                        capture_sha256=capture_sha256,
                        timeout=2 * self.budget.eval_timeout_s,
                    )
            except Exception as exc:  # the re-check broke: the kernel is not confirmed
                return {"status": "error", "passed": False, "reason": repr(exc)[:500]}

        result = run_recheck(_verdict(fresh or rec))
        if fresh is None and rec is not None and result.get("status") == recheck.DISAGREES:
            why = f"the re-check measured {result.get('speedup')}x"
            fresh = self._reevaluate(target_id, snap, rec, capture_sha256, why)
            if not fresh.get("correct"):
                return _reevaluation_failed(result, rec, fresh, why)
            result = recheck.judge(result, _verdict(fresh))
        if result.get("status") == recheck.DISAGREES:  # once more, in new processes
            again = run_recheck(_verdict(fresh or rec))
            if again.get("status") not in ("ok", recheck.DISAGREES):
                result = {**again, "reason": f"on a second re-check: {again.get('reason')}"}
            else:
                runs = [result, again]  # an agreeing run first, else the closer one
                result = min(runs, key=lambda r: (r["status"] != "ok", recheck.disagreement(r)))
                result["rechecks"] = [
                    {k: r.get(k) for k in ("speedup", "timing_spread", "seed", "speedup_ratio")}
                    for r in runs
                ]
        if result.get("status") == recheck.DISAGREES:
            if stale and float(result.get("speedup") or 0.0) <= 1.0:
                result["reason"] += "; no speedup at all in separate processes"
            else:
                result = recheck.speed_warning(result)
        if rec is not None and fresh is not None and why:
            result["reevaluated"] = _reevaluation(rec, fresh, why)
        return result

    def _reevaluate(
        self,
        target_id: str,
        snap: Path,
        rec: dict[str, Any],
        capture_sha256: str | None,
        why: str,
    ) -> dict[str, Any]:
        """The current evaluator on a kernel snapshot whose record ``rec`` is stale or
        disagrees with its re-check (``why``), appended to the target's records as a
        ``re-evaluated`` ledger row that replaces ``rec`` (``reevaluates``,
        :func:`agent.tools.current_records`); returns the new record. A snapshot that is
        not (or no longer) the file ``rec`` measured is ``tampered`` and not recorded."""
        sha = rec.get("snapshot_sha256")
        changed = {"status": "tampered", "correct": False, "error": f"{snap.name} changed"}
        if not self.truth.snapshot_ok(snap, sha):
            return changed
        sha = sha or truth.sha256_file(snap)
        start = time.perf_counter()
        with self._gpu_job("reevaluate") as job:
            try:
                result = (self.reevaluator or evaluate.run_evaluation)(
                    self.run.capture_file(target_id),
                    snap,
                    timeout=self.budget.eval_timeout_s,
                    capture_sha256=capture_sha256,
                )
            except Exception as exc:
                result = {"status": "error", "correct": False, "error": repr(exc)[:500]}
        if truth.sha256_file(snap) != sha:
            self.truth.alarm(snap, "snapshot changed during its re-evaluation")
            return changed
        src = self.run.target(target_id) / str(rec.get("candidate") or f"history/{snap.name}")
        record, _ = record_candidate(
            self.run,
            target_id,
            src,
            snap,
            result,
            hypothesis=f"re-evaluation of exp {rec.get('exp')} ({rec.get('speedup')}x): {why}",
            parent=rec.get("parent"),
            eval_s=round(time.perf_counter() - start - job.wait_s, 1),
            snapshot_sha256=sha,
            keeper=self.truth,
            idea=str(rec.get("idea") or ""),
            queue_s=job.queue_s,
            reevaluates={
                "exp": rec.get("exp"),
                "speedup": rec.get("speedup"),
                "evaluator_version": rec.get("evaluator_version"),
                "why": why,
            },
        )
        return record

    def _kernel_record(self, target_id: str, snapshot: str) -> dict[str, Any] | None:
        """The verified correct evaluation record of a kernel snapshot (None: there is none);
        of a re-evaluated snapshot its re-evaluation (:func:`agent.tools.current_records`)."""
        try:
            records = current_records(self.truth.records(self.run.results_file(target_id)))
        except truth.TamperError:
            return None
        for rec in records:
            if rec.get("correct") and Path(str(rec.get("snapshot", ""))).name == snapshot:
                return rec
        return None

    def _kernel_item(self, target_id: str) -> tuple[str, str | None] | None:
        """The integration item (``target=snapshot``) and sha256 of a target's best kernel."""
        best = self._kernel_best(target_id)
        if best is None:
            return None
        snap = self.run.history_dir(target_id) / Path(best["snapshot"]).name
        return f"{target_id}={snap}", best.get("snapshot_sha256")

    def _integration_call(
        self, command: str, combo: list[tuple[str, str]], cli: list[str], note: str
    ) -> dict[str, Any]:
        """One integration measurement (``e2e`` / ``e2e_ab`` of ``combo``) and its ledger row
        (none when an A/B could not run in-process: nothing was measured)."""
        start = ledger.clock()  # simulated in a dry run, like the worker's measurements
        with self._gpu_job("integration") as job:  # one A/B step: one turn in the GPU queue
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
            # improve.py's integration estimate: the hold, without its queue wait
            eval_s=round(ledger.clock() - start - job.wait_s, 1),
            queue_s=job.queue_s,
        )
        gpu = r.get("gpu") or (r.get("ab") or {}).get("gpu")
        if message := telemetry.warning(gpu):
            log(f"integrate: WARNING {message}")
        return r

    def _paired(
        self,
        a: list[tuple[str, str]],
        b: list[tuple[str, str]],
        note: str,
        irreversible: set[str],
        crowded: list[set[str]] | None = None,
    ) -> dict[str, Any]:
        """B's result with an ``ab`` record of its timings against A: from one process
        (``e2e_ab``), or from two back to back when a state cannot be undone in-process
        (``irreversible`` collects such items) or A and B do not fit in one process
        together (``oom``). After an ``oom`` each process runs with
        ``--expandable-segments``; ``crowded`` collects what both states of such a pair
        held (each state applies it afresh), and a later pair whose states both hold one of
        those sets goes to two processes at once."""
        why = "an item cannot be undone in-process"
        a_items, b_items = {x for _, x in a}, {x for _, x in b}
        full = next((c for c in crowded or [] if c <= a_items and c <= b_items), None)
        oom = full is not None
        if full is not None:
            why = f"out of GPU memory before with {len(full)} of these items in both states"
            log(f"integrate: no in-process A/B ({why}); separate processes")
        elif not irreversible.intersection(a_items | b_items):
            cli = [*_cli(a, warmup=2), *_cli(b, prefix="--b-")]
            cli += ["--rounds", str(self.cfg.ab_rounds)]
            r = self._integration_call("e2e_ab", b, cli, note)
            if r.get("status") not in abtest.FALLBACK:
                return r
            irreversible.update(r.get("irreversible") or [])
            why = str(r.get("reason") or r.get("status"))
            if r.get("status") == abtest.OOM:
                oom = True
                if crowded is not None and a_items & b_items:
                    crowded.append(a_items & b_items)
            log(f"integrate: no in-process A/B ({why[:300]}); separate processes")
        iters = abtest.SEPARATE_ITERS
        flags = ["--expandable-segments"] if oom else []
        ra = self._integration_call(  # A's diverse set: measured when it was B
            "e2e",
            a,
            [*_cli(a, iters=iters), *flags, "--no-diverse"],
            " (A of an A/B in separate processes)",
        )
        rb = self._integration_call("e2e", b, [*_cli(b, iters=iters), *flags], note)
        rb["ab"] = {
            "mode": "separate",
            "a_ms": ra.get("times_ms") or [],
            "b_ms": rb.get("times_ms") or [],
            "fallback": why,
            **({"expandable_segments": True} if oom else {}),
            **({"a_patches": ra["patches"]} if ra.get("patches") else {}),
        }
        if not ra.get("times_ms"):
            rb["ab"]["why"] = f"A measured again: {ra.get('reason') or ra.get('status')}"
        return rb

    def _projection(
        self,
        base_ms: float,
        sets: list[tuple[list[tuple[str, str]], dict[str, Any]]],
        history: list[dict[str, Any]],
        changes: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """Projected (baseline − Σ est. saved ms of its items) vs measured latency of every
        accepted set: a kernel's saving is its module-level estimate in the metric's ms
        (``projection.Units``: per second of audio for ``metric=throughput``), a transform's
        its measured gain alone (paired against the unmodified model). Nested kernels count
        once (:func:`projection.of_set`: a decoder layer's kernel replaces the attention
        kernel inside it), and so do items whose modules overlap (``changes``: what each
        item touched or owns, #121); ``counted_ms`` is the part of each item's saving that
        counts. A set after the first is projected from the set before it, measured in the
        A/B of its step, minus its step's estimated gain (:func:`projection.of_sets`)."""
        alone = _alone(base_ms, history)
        tree, units = projection.tree(self.run), self._units()
        rows = []
        for combo, r in sets:
            saved: dict[str, float | None] = {}
            for item in combo:
                est = self._saving(item, alone, units)
                saved[item[1]] = None if est is None else round(est, 3)
            a_ms = (r.get("ab") or {}).get("a_median_ms")  # the set before it, same A/B
            rows.append(
                {
                    "items": [a for _, a in combo],
                    "est_saved_ms": saved,
                    "measured_ms": r.get("median_ms"),
                    "from_ms": a_ms,
                }
            )
        return projection.of_sets(tree, rows, base_ms, changes)  # in the metric's ms (#114)

    def _units(self) -> projection.Units:
        """A kernel's est. saved ms per run → the metric's ms (#114), from the sealed
        ``baseline.json``."""
        return projection.units_of(self.run, baseline=self.truth.load_json(self.run.baseline_json))

    def _saving(
        self, item: tuple[str, str], alone: dict[str, float], units: projection.Units
    ) -> float | None:
        """Est. saved ms of an integration item in the metric's ms: a kernel's module-level
        estimate (:meth:`_kernel_saving`, per run) converted with ``units``, a transform's
        measured gain alone (``alone``)."""
        kind, arg = item
        if kind == "kernel":
            target_id, _, path = arg.partition("=")
            return units(target_id, self._kernel_saving(target_id, Path(path).name))
        return alone.get(arg)

    def _swaps(
        self,
        accepted: list[tuple[str, str]],
        singles: list[tuple[tuple[str, str], dict[str, Any]]],
        versions: list[tuple[str, str]],
        alone: dict[str, float],
    ) -> list[tuple[str, tuple[str, str]]]:
        """The version swaps to try on the accepted set, as (item key, new version): the
        seed (the measured combination above all) can hold an older version of a kernel
        target or transform idea than its best one, which the greedy additions skip.

        Of every accepted target and idea, first the ``versions`` a previous integration
        swapped in or accepted, in its order (a re-integration of the same files takes the
        same steps: the reuse cache), then its best version measured alone (``singles``: the
        target's best verified and re-checked kernel, the idea's transform of the fastest
        passing record) ordered by expected gain: the new version's est. saved ms minus the
        accepted one's (:meth:`_saving`: a kernel's module-level estimate at its
        conservative speedup, a transform's paired gain alone; unknown last)."""
        current = {_item_key(a): a for a in accepted}
        seen = set(accepted)
        units = self._units()  # a kernel's saving in the metric's ms
        swaps: list[tuple[str, tuple[str, str]]] = []
        for group, rank in ((versions, False), ([i for i, _ in singles], True)):
            ranked: list[tuple[float | None, str, tuple[str, str]]] = []
            for item in group:
                key = _item_key(item)
                if key not in current or item in seen:
                    continue
                seen.add(item)
                new = self._saving(item, alone, units)
                old = self._saving(current[key], alone, units)
                ranked.append((None if new is None else new - (old or 0.0), key, item))
            if rank:
                ranked.sort(key=lambda g: (g[0] is None, -(g[0] or 0.0)))
            swaps += [(key, item) for _, key, item in ranked]
        return swaps

    def _previous_versions(
        self, previous: dict[str, Any], digests: dict[str, str | None]
    ) -> list[tuple[str, str]]:
        """The versions a previous integration swapped in (tried, in order) or accepted that
        are still verified snapshots: a kernel of a correct record of its target, a
        transform of a passing ``evaluate_e2e`` record faster than the baseline (their
        sha256 go to ``digests``). A re-integration tries them as swaps first (:meth:`_swaps`)."""
        news = [h.get("new") for h in previous.get("history") or [] if h.get("kind") == "swap"]
        olds = [a.get("item") for a in previous.get("accepted") or []]
        transforms = {
            Path(s[0]).name: s for _, snaps in self._e2e_records() for s in snaps if s is not None
        }
        versions: list[tuple[str, str]] = []
        for arg in map(str, filter(None, [*news, *olds])):
            target_id, sep, _ = arg.partition("=")
            if sep and "/" not in target_id:  # target=path
                if (kernel := self._kernel_snapshot(arg)) is None:
                    continue
                item, (path, digest) = ("kernel", f"{kernel[0]}={kernel[1]}"), kernel[1:]
            elif (snap := transforms.get(Path(arg).name)) is not None:
                item, (path, digest) = ("transform", snap[0]), snap
            else:
                continue
            if item not in versions:
                versions.append(item)
                digests[path] = digest
        return versions

    def _kernel_saving(self, target_id: str, snapshot: str) -> float | None:
        """``est_saved_ms_per_run`` of the verified evaluation of a kernel snapshot; with a
        speed cap (``speed_caps``) the saving at that conservative speedup."""
        try:
            records = current_records(self.truth.records(self.run.results_file(target_id)))
        except truth.TamperError:
            return None
        for rec in records:
            if Path(str(rec.get("snapshot", ""))).name == snapshot:
                est = rec.get("est_saved_ms_per_run")
                if est is None:
                    return None
                speedup = float(rec.get("speedup") or 0.0)
                cap = self.speed_caps.get((target_id, snapshot))
                if cap is not None and 0.0 < cap < speedup and speedup > 1.0:  # ∝ 1 - 1/speedup
                    return float(est) * (1 - 1 / cap) / (1 - 1 / speedup)
                return float(est)
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
        if target_id not in self.run.target_ids() or self._refused(target_id):
            return None
        if not region.rewrite_ok(self.run, target_id, self.truth):  # a region: its rewrite
            return None
        try:
            records = current_records(self.truth.records(self.run.results_file(target_id)))
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
        cli = ["--warmup", "2", "--iters", "5", "--no-diverse"]
        for arg in kernels:
            cli += ["--kernel", arg]
        cli += ["--transform", str(strong_baseline.REFERENCE_TRANSFORM)]
        with self._gpu_job("integration"):
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
        target = self._engineer_target(spec, spec["backends"], evaluations, stats)
        return await self._agent(
            f"kernel-{target_id}",
            label,
            prompt=(
                f"Continue optimising target `{target_id}`. Read the `# Improve slice` section "
                "first: it says where the previous sessions left off."
            ),
            system_append=self._stable("kernel"),
            context=target + digest,
            cwd=target_dir,
            add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
            # evaluation advice says `stop` after these
            binding=SessionBinding(role="kernel", target_id=target_id, evaluations=evaluations),
            **self._owned(target_dir, [target_dir / workers.DIR]),
        )

    async def systems_slice(self, *, evaluations: int, digest: str, label: str) -> AgentResult:
        """A fresh systems-engineer session seeded with ``digest``."""
        plan = read_json(self.run.plan_json, {})
        self.run.transforms_dir.mkdir(parents=True, exist_ok=True)
        context = prompts.systems_target(
            self.run.load()["card"],
            read_json(self.run.baseline_json, {}),
            (self.run.profile_dir / "summary.md").read_text(),
            plan.get("transforms", []),
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
            system_append=self._stable("systems"),
            context=context + digest,
            cwd=self.run.transforms_dir,
            add_dirs=[prompts.WORKLOADS_DIR, prompts.KNOWLEDGE_DIR],
            binding=SessionBinding(role="systems", evaluations=evaluations),
            **self._owned(self.run.transforms_dir, [native_engine.native_dir(self.run)]),
        )

    def _owned(self, root: Path, excluded: list[Path] | None = None) -> dict[str, Any]:
        """The write policy of an engineer session when sessions run concurrently
        (:attr:`ownership`, docs/MULTIAGENT.md §3.6): its file-writing tools may touch only
        ``root`` without ``excluded`` (``runner.write_guard``); {} with one session at a time."""
        if not self.ownership:
            return {}
        return {"roots": [root], "excluded": list(excluded or [])}

    def native_minutes(self) -> float | None:
        """Session length of the native agent: ``--native-minutes``, else a multiple of
        ``--agent-minutes`` (its ``session_factor``, ``roles.py``; None: no per-session
        limit). Its turns are the same multiple of ``--max-turns``."""
        if self.cfg.native_minutes:
            return float(self.cfg.native_minutes)
        factor = roles.get("native").session_factor
        return self.cfg.agent_minutes * factor if self.cfg.agent_minutes else None

    async def native_slice(self, *, evaluations: int, digest: str, label: str) -> AgentResult:
        """A fresh systems-native session (``native/engine.py``, issue #134) seeded with
        ``digest``: a longer session (:meth:`native_minutes`, more turns), the native-engine
        contract, this run's and the library's best kernels as building blocks, and the
        tools for its stage targets and end-to-end runs, working in ``transforms/native/``
        (its project directories)."""
        cwd = native_engine.native_dir(self.run)
        cwd.mkdir(parents=True, exist_ok=True)
        (cwd / "NOTES.md").touch()
        if (minutes := self.native_minutes()) is not None:
            self.budget.minutes_by_agent["native"] = minutes
        why = native_engine.enabled(self.cfg.native, native_engine.plan_entry(self.run))
        blocks = native_engine.building_blocks(self._kernel_winners(), self._library_arch())
        context = prompts.native_target(
            self.run.load()["card"],
            read_json(self.run.baseline_json, {}),
            (self.run.profile_dir / "summary.md").read_text(),
            evaluations,
            why=f"{why or 'opened by the scheduler'}; every module arm has plateaued",
            blocks=blocks,
        )
        return await self._agent(
            "native",
            label,
            prompt=(
                "Continue the native engine of this model. Read the `# Improve slice` section "
                "first: it says which stage is current and where the previous sessions left off."
            ),
            system_append=self._stable("native"),
            context=context + digest,
            cwd=cwd,
            add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR, prompts.WORKLOADS_DIR],
            binding=SessionBinding(role="native", agent="native", evaluations=evaluations, cwd=cwd),
            **self._owned(cwd),
        )

    async def research(
        self, target_id: str, *, reason: str, label: str | None = None
    ) -> AgentResult:
        """A clean-context research session on a plateaued target (``research.py``).

        Read-only tools plus ``best_result``; it may write ``targets/<id>/plan.md``
        and, in a near-lossless or relaxed run, a precision pivot proposal ``pivot.json``
        (``pivot.py``), nothing else (``run_agent(writable=...)``)."""
        from kernel_agent import pivot

        target_dir = self.run.target(target_id)
        spec = read_json(target_dir / "spec.json")
        plan = research.plan_path(self.run, target_id)
        near = allows_reduced(self.cfg.quality)
        allowed = self.allowed_precisions()  # a pivot only to another one of them
        others = [p for p in precisions.reduced(allowed) if p != precisions.of_spec(spec)]
        proposal = pivot.proposal_path(self.run, target_id) if near and others else None
        evidence = research.evidence(self.run, target_id, reason, self.truth)
        if near:
            evidence += research.transforms_section(self.run)
        dossier = research.dossier_path(self.run, target_id) if self.cfg.allow_web else None
        context = prompts.research_target(
            spec,
            spec.get("capture", {}),
            evidence,
            plan,
            pivot=proposal,
            dossier=dossier,
            precisions=allowed,
        )
        return await self._agent(
            f"research-{target_id}",
            label,
            prompt=(
                f"Target `{target_id}` has plateaued ({reason}). Diagnose why from the ledger "
                f"and the files, then write the plan to {plan}."
            ),
            system_append=self._stable("research"),
            context=context,
            cwd=target_dir,
            add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
            writable=[plan] + ([proposal] if proposal else []) + ([dossier] if dossier else []),
            binding=SessionBinding(role="research", target_id=target_id),  # its board (#187)
        )

    async def dossier(self, target_id: str, *, label: str | None = None) -> AgentResult | None:
        """The research dossier of a target before its first engineer session (issue #125).

        A short ``dossier-<id>`` session (effort and turns capped, read-only tools, the web
        tools) looks up the documentation, reference code and papers for the target and
        writes ``targets/<id>/research.md``, nothing else. None (no session) with
        ``--no-web`` / ``--no-dossier``, when the file exists or the budget is spent, and
        when the session failed: a dossier is a bonus, never a reason to stop a target."""
        path = research.dossier_path(self.run, target_id)
        if not (self.cfg.allow_web and self.cfg.dossier) or path.is_file():
            return None
        if reason := self.budget.exhausted(label or f"dossier-{target_id}"):
            log(f"dossier: {target_id}: skipped: {reason}")
            return None
        target_dir = self.run.target(target_id)
        spec = read_json(target_dir / "spec.json")
        log(f"dossier: {target_id}: documentation and reference code before its first session")
        try:
            return await self._agent(  # cheap: its model, effort and turns are the registry's
                f"dossier-{target_id}",
                label,
                prompt=(
                    f"Look up what the sources say about making target `{target_id}` fast and "
                    f"write the dossier to {path}."
                ),
                system_append=prompts.dossier_prompt(
                    spec, spec.get("capture", {}), path, self.tc.summary()
                ),
                cwd=target_dir,
                add_dirs=[prompts.EXAMPLES_DIR, prompts.KNOWLEDGE_DIR],
                writable=[path],
            )
        except auth.AuthError:  # the wrong billing stops the run, as in any session
            raise
        except Exception as exc:
            if interrupt.requested():
                raise interrupt.Interrupted from exc
            log(f"dossier: {target_id}: agent session failed: {exc!r}")
            return None

    async def pivot(
        self, target_id: str, proposal: dict[str, Any], *, source: str
    ) -> dict[str, Any]:
        """Move a target to another precision tier (``pivot.py``): check the proposal,
        write the spec of the new target ``<id>__<precision>`` and capture it in its tier.
        Returns ``{"target": new id}``, or ``{"refused": why}``; the old target is left as
        it is. ``source``: who proposed it (a research session, a round's re-plan)."""
        from kernel_agent import pivot

        spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
        taken = pivot.taken(self.run)
        allowed = self.allowed_precisions()
        problem = pivot.check(
            spec, proposal, quality=self.cfg.quality, taken=taken, allowed=allowed
        )
        if problem is not None:
            log(f"pivot: {target_id}: not moved to {proposal.get('precision')!r} ({problem})")
            ledger.event(self.run, "pivot_refused", target=target_id, why=problem, source=source)
            return {"refused": problem}
        new = pivot.pivot_spec(spec, proposal)
        log(f"pivot: {target_id} -> {new['id']} ({source}): {new['precision_why'][:200]}")
        if not await self.capture_targets([new]):
            ledger.event(self.run, "pivot_failed", target=target_id, new=new["id"], source=source)
            return {"refused": f"the capture of {new['id']} failed"}
        plan = research.plan_path(self.run, target_id)
        if plan.is_file():  # the diagnosis behind the pivot, for the new arm's first slice
            (self.run.target(new["id"]) / research.PLAN_FILE).write_text(plan.read_text())
        ledger.event(
            self.run,
            "pivot",
            target=target_id,
            new=new["id"],
            precision=new["precision"],
            source=source,
        )
        return {"target": new["id"]}

    def reprofile(self, out_dir: Path, accepted: list[dict[str, Any]]) -> dict[str, Any]:
        """Baseline + profile of the model with ``accepted`` integration items applied.

        Written to ``out_dir`` (``baseline.json``, ``profile/``); the run's own
        baseline stays the reference for every comparison.
        """
        cli = ["--iters", "3", "--out-dir", str(out_dir)]
        for item in accepted:
            cli += ["--kernel" if item["kind"] == "kernel" else "--transform", item["item"]]
        return self._worker("analyze", *cli)

    def e2e_items(self, snapshot: str) -> list[dict[str, Any]] | None:
        """The items of the newest passing ``evaluate_e2e`` record of a ledger ``snapshot``
        (its transforms and kernels) as :meth:`reprofile` takes them (the native arm's best
        run, issue #164); None: no such record, or a file that is not a verified snapshot."""
        for rec, snaps in reversed(self._e2e_records()):
            measured = ledger.e2e_snapshot(rec.get("transforms") or [], rec.get("kernels") or [])
            if measured != snapshot:
                continue
            kernels = [self._kernel_snapshot(k) for k in rec.get("kernels") or []]
            if None in snaps or None in kernels:
                return None
            return [{"kind": "transform", "item": s[0]} for s in snaps if s] + [
                {"kind": "kernel", "item": f"{k[0]}={k[1]}"} for k in kernels if k
            ]
        return None

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
                quality=self.cfg.quality,
                precisions=self.allowed_precisions(),
                backend_record=self._backend_record(),
            )
            + context,
            cwd=round_dir,
            output_format={"type": "json_schema", "schema": self._plan_schema()},
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
        allowed = self.allowed_precisions()
        for t in plan.get("targets", [])[: self.cfg.max_targets]:
            problem = (
                region.validate(t, known) or _scope(t) or _precision(t, self.cfg.quality, allowed)
            )
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
            if library.seeded(self.run, target_id) is not None or self._refused(target_id):
                continue  # seeded, or at a precision the run does not allow
            with self._gpu_job("seed"):  # background work in the GPU queue
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

    def _stable(self, role: str) -> str:
        """The system prompt of every session of a split ``role`` on this run (#181): its
        ``prompts.stable_prefix``; the run's precision policy for the systems and native
        engineers (the same for each of their sessions)."""
        text = prompts.stable_prefix(role, self.python, self.tc.summary())
        if role in ("systems", "native"):
            text += prompts.precision_note(self.cfg.quality, self.allowed_precisions())
        return text

    def _engineer_target(
        self,
        spec: dict[str, Any],
        backends: list[str],
        evaluations: int,
        stats: dict[str, dict[str, Any]],
        approach: str | None = None,
    ) -> str:
        """The target block of a kernel engineer session (``prompts.engineer_target``; a
        worker's ``approach``) and the library's priors and lessons for the target."""
        target = prompts.engineer_target(
            {**spec, "approach": approach} if approach is not None else spec,
            spec.get("capture", {}),
            backends,
            self.tc.summary(),
            evaluations,
            stats.get(spec["module_class"]),
            precisions=self.allowed_precisions(),
        )
        return target + self._library_note(spec["id"], spec)

    def _backend_record(self) -> str:
        """Planner-prompt section: which backend won which target class on this GPU in
        earlier runs (kernel_agent/backends.py; "" without a library or a record)."""
        from kernel_agent import backends

        try:
            return backends.track_record_note(self._library_arch())
        except Exception as exc:  # advice only: never fails the plan
            log(f"library: backend track record unreadable: {exc!r}")
            return ""

    def _library_store(self, accepted: list[str], final: dict[str, Any] | None) -> None:
        """Store the verified module winners and accepted kernels (never fails the run)."""
        if (arch := self._library_arch()) is None:
            return
        from kernel_agent import backends

        try:  # which backend each target tried and which won (by source), per target class
            backends.record_run(self.run, arch)
        except Exception as exc:
            log(f"library: recording the backends of this run failed: {exc!r}")
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
                prompt="Distil this run into the library's lessons files.",
                system_append=library.librarian_prompt(self.run, names, arch=arch),
                cwd=self.run.root,
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
    one version of each can be accepted. A target's precision pivots (``pivot.py``)
    optimise the same modules: their kernels are versions of the original target's."""
    from kernel_agent.pivot import family

    kind, arg = item
    if kind == "kernel":
        return f"kernel {family(arg.partition('=')[0])}"
    return f"transform {ledger.snapshot_stem(arg)}"


def _same_file(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """Whether two integration items apply files of the same content."""
    try:
        paths = [Path(arg.partition("=")[2] if kind == "kernel" else arg) for kind, arg in (a, b)]
        return truth.sha256_file(paths[0]) == truth.sha256_file(paths[1])
    except OSError:
        return False


def _reranks(result: dict[str, Any]) -> bool:
    """Whether a re-check changed how its target's snapshots rank (a re-evaluation, a
    speed cap)."""
    return bool(result.get("reevaluated")) or result.get("status") == recheck.SPEED_DISAGREES


def _verdict(rec: dict[str, Any] | None) -> dict[str, Any] | None:
    """What a re-check compares with: an evaluation's correctness, speedup and timing spread."""
    if rec is None:
        return None
    verdict = {"correct": rec.get("correct"), "speedup": rec.get("speedup")}
    if (spread := recheck.spread(rec)) is not None:
        verdict["timing_spread"] = spread
    return verdict


def _reevaluation(rec: dict[str, Any], fresh: dict[str, Any], why: str) -> dict[str, Any]:
    """``reevaluated`` of a re-check: the record it replaced and the current evaluator's."""
    return {
        "why": why,
        "old_exp": rec.get("exp"),
        "old_speedup": rec.get("speedup"),
        "new_exp": fresh.get("exp"),
        "new_status": fresh.get("status"),
        "new_correct": bool(fresh.get("correct")),
        "new_speedup": fresh.get("speedup"),
    }


def _reevaluation_failed(
    result: dict[str, Any], rec: dict[str, Any], fresh: dict[str, Any], why: str
) -> dict[str, Any]:
    """``result`` refused: the current evaluator's re-evaluation of its record failed."""
    error = fresh.get("error") or fresh.get("failed_check") or fresh.get("stage") or ""
    return {
        **result,
        "status": "reevaluation_failed",
        "passed": False,
        "reason": f"the current evaluator's re-evaluation: {fresh.get('status')}: "
        + str(error)[-500:],
        "reevaluated": _reevaluation(rec, fresh, why),
    }


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


def _alone(base_ms: float, history: list[dict[str, Any]]) -> dict[str, float]:
    """Est. saved ms of every item that passed alone: its paired gain against the
    unmodified model × the baseline."""
    return {
        h["items"][0]: base_ms * float(h["ab"]["gain"])
        for h in history
        if len(h["items"]) == 1
        and not (h.get("ab") or {}).get("a_items")
        and h.get("passed")
        and (h.get("ab") or {}).get("gain") is not None
    }


#: Statuses of an integration measurement a re-integration measures again, not reuses
#: (``oom``: what else the process held, not the items).
_TRANSIENT = ("crash", "error", "harness_error", "timeout", abtest.OOM)


def _transient(entry: dict[str, Any]) -> bool:
    """Whether a re-integration measures an ``integration.json`` history entry again: a
    :data:`_TRANSIENT` status, or out of GPU memory anywhere (``abtest.step_out_of_memory``:
    also a check that a kernel-agent before #137 recorded as failed, ``perceptual: the
    perceptual gate failed: OutOfMemoryError: ...``)."""
    return entry.get("status") in _TRANSIENT or abtest.step_out_of_memory(entry) is not None


def _few(names: list[str], n: int = 3) -> str:
    """The first ``n`` of ``names``, and how many more."""
    return ", ".join(names[:n]) + (f" (+{len(names) - n} more)" if len(names) > n else "")


def _in_place(
    accepted: list[tuple[str, str]], olds: list[tuple[str, str]], new: tuple[str, str]
) -> list[tuple[str, str]]:
    """``accepted`` without ``olds``, ``new`` in the place of the first of them (where it
    is already, when it is accepted)."""
    out = []
    for item in accepted:
        if item not in olds:
            out.append(item)
        elif new not in out and new not in accepted:
            out.append(new)
    return out


def _short(r: dict[str, Any]) -> dict[str, Any]:
    return {
        k: r.get(k)
        for k in ("status", "passed", "reason", "median_ms", "speedup", "metrics", "patches")
    }


def _data_dependent(
    history: list[dict[str, Any]], accepted: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """The items whose A/B alone against the unmodified model measured a speedup that changes
    with the input (:mod:`kernel_agent.diversity`), with their diverse-set speedups and
    whether they were accepted: labelled in ``integration.json``, never dropped for it."""
    kept = {a for _, a in accepted}
    out = []
    for h in history:
        if len(h.get("items") or []) != 1 or (h.get("ab") or {}).get("a_items"):
            continue
        if diversity.is_data_dependent(h) and (info := diversity.compact(h)) is not None:
            item = h["items"][0]
            out.append({"item": item, "accepted": item in kept, **info})
    return out


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


def _precision(
    target: dict[str, Any], quality: str, allowed: tuple[str, ...] | None = None
) -> str | None:
    """Normalise a planned target's ``precision`` / ``precision_why`` in place; returns why
    the target is refused, or None. A reduced precision (``fp8_weights``, ``fp8_w8a8``,
    ``reduced``, ``fp4_weights``: ``kernels.compare.REDUCED_PRECISIONS``) needs ``--quality
    near-lossless`` or ``relaxed`` and must be one the run allows (``allowed``; None: the
    default of the quality mode, without the 4-bit ones: ``precisions.py``)."""
    from kernel_agent.kernels.compare import PRECISIONS, REDUCED_PRECISIONS

    precision = target.pop("precision", None) or "exact"
    why = str(target.pop("precision_why", None) or "").strip()
    if precision not in PRECISIONS:
        return f"unknown precision {precision!r}"
    if precision not in REDUCED_PRECISIONS:
        return None
    if not allows_reduced(quality):
        return (
            f"precision {precision!r} needs --quality near-lossless or relaxed "
            f"(this run: {quality})"
        )
    if problem := precisions.refusal(
        precision, precisions.default(quality) if allowed is None else allowed
    ):
        return problem
    target["precision"] = precision
    if why:
        target["precision_why"] = why
    else:
        log(f"plan: {target.get('id')}: precision {precision} without a precision_why")
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
    with coordinator_lock(orch.run):  # one process per run (workspace.coordinator_lock)
        return await orch.run_all(until=until)
