"""``kernel-agent`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from kernel_agent.config import (
    ALL_BACKENDS,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    QUALITIES,
    OptimizeConfig,
)


def _parse_value(raw: str) -> Any:
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"none", "null"}:
        return None
    return raw


def _options(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        out[key.strip()] = _parse_value(value.strip())
    return out


def cmd_doctor(ns: argparse.Namespace) -> int:
    from kernel_agent import toolchain

    tc = toolchain.setup()
    if tc.gpu is not None and not ns.no_peaks:
        from kernel_agent.kernels.roofline import ensure_peaks

        ensure_peaks(remeasure=ns.remeasure_peaks, verbose=True)  # cached per GPU + torch
    print(tc.summary())
    from kernel_agent import cute_dsl

    # CuTe DSL: version, import, the arch it compiles for, block-scaled MMA, compile cache
    print("\n".join(cute_dsl.check(tc.gpu.capability if tc.gpu else None).describe()))
    from kernel_agent.gpulock import describe

    print(describe())  # the GPU lock pool (nvidia-smi, CUDA_VISIBLE_DEVICES, KERNEL_AGENT_GPUS)
    _doctor_probes(ns, gpu=tc.gpu is not None)
    _doctor_libraries(tc)
    _doctor_docs()
    if not _doctor_sanitizer(ns, gpu=tc.gpu is not None):
        return 1
    if ns.smoke:
        from kernel_agent.selftest import smoke_backends

        ok = smoke_backends(verbose=True)
        return 0 if ok else 1
    return 0


def _doctor_probes(ns: argparse.Namespace, *, gpu: bool) -> None:
    """Nsight Compute for ``profile="ncu"`` (kernels/ncu.py), the versions and the feature
    probes (probes.py: ``tl.dot_scaled`` → block-scaled MMA, TMA, PDL, green contexts,
    conditional graph nodes); informative, never a failure."""
    from kernel_agent import probes
    from kernel_agent.kernels import ncu, sass

    print(ncu.describe(ncu.availability()))
    print(sass.describe())  # the opcode census of profile=true (#230)
    if ns.no_probes:
        return
    print(probes.describe(probes.run(gpu)))


def _doctor_libraries(tc: Any) -> None:
    """The library scout's libraries (libscout/, #227): installed versions, licences and the
    adapters that can run on this GPU, each skipped one with the reason; never a failure."""
    from kernel_agent.libscout import scout as libscout

    capability = tuple(tc.gpu.capability) if tc.gpu is not None else None
    try:
        print("\n".join(libscout.doctor_lines(capability, tc.backends)))
    except Exception as exc:
        print(f"library scout: probe failed: {exc!r}")


def _doctor_docs() -> None:
    """The doc library the agents search (doclib, issue #177): its shelves per library and
    version, built first from the installed packages if needed (no network; `kernel-agent
    docs build` fetches the web docs); informative, never a failure."""
    from kernel_agent import doclib

    try:
        doclib.ensure()
    except Exception as exc:
        print(f"doc library: not built: {exc!r}")
    print("\n".join(doclib.describe()))


def _doctor_sanitizer(ns: argparse.Namespace, *, gpu: bool) -> bool:
    """``compute-sanitizer`` (the integration's memcheck, kernels/memcheck.py): where it is
    (``--fetch-sanitizer``: install NVIDIA's first) and, with a GPU, the self-test that it
    reports a deliberate out-of-bounds read. False: a fetch or the self-test failed."""
    from kernel_agent import toolchain

    if ns.fetch_sanitizer:
        try:
            print(f"installed {toolchain.fetch_sanitizer()}")
        except Exception as exc:
            print(f"compute-sanitizer: fetch failed: {exc}")
            return False
    tool = toolchain.sanitizer()
    print(tool.describe())
    for why in tool.rejected if tool.path else tool.rejected[1:]:  # [0]: in the reason
        print(f"  not usable: {why}")
    if not tool.path or not gpu:
        return True
    from kernel_agent.kernels.memcheck import selftest

    check = selftest(tool)
    verdict = "ok" if check.get("ok") else "FAILED"
    print(f"memcheck self-test: {verdict} ({check.get('seconds')} s): {check.get('reason')}")
    return bool(check.get("ok"))


def _seeds(raw: str) -> int | str | None:
    """``--seeds-per-target``: a count or ``auto``."""
    from kernel_agent import workers

    try:
        return workers.parse(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _integration_reserve(raw: str) -> float | None:
    """``--integration-reserve``: ``auto`` (None) or minutes (``0``: none)."""
    if raw.strip().lower() == "auto":
        return None
    try:
        minutes = float(raw)
    except ValueError:
        minutes = -1.0
    if not 0.0 <= minutes < float("inf"):
        raise argparse.ArgumentTypeError(f"expected auto or minutes (>= 0), got {raw!r}")
    return minutes


def _agents(raw: str) -> int | str:
    """``--agents``: concurrent agent sessions (a positive count), or ``auto`` / ``auto:N``
    (the governor decides, up to N: ``governor.parse_agents``)."""
    from kernel_agent.governor import parse_agents

    try:
        n = parse_agents(raw)[0]
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive number of sessions, auto or auto:N, got {raw!r}"
        )
    return raw.strip().lower() if raw.strip().lower().startswith("auto") else n


def _count(raw: str) -> int:
    """A count >= 0 (``--migrate-every``)."""
    if not raw.strip().isdigit():
        raise argparse.ArgumentTypeError(f"expected a count >= 0, got {raw!r}")
    return int(raw)


def _share(raw: str) -> float:
    """A share in [0, 1) (``--cull-gap``)."""
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"expected a share in [0, 1), got {raw!r}")
    return value


def _role_max(raw: str) -> dict[str, int]:
    """``--role-max kernel=2,research=1``: sessions of a role at once (the roles not named
    keep the role registry's ``max_concurrent``, ``roles.REGISTRY``)."""
    roles = ("kernel", "systems", "native", "research", "dossier")
    out: dict[str, int] = {}
    for part in (p.strip() for p in raw.split(",") if p.strip()):
        role, _, value = part.partition("=")
        if role.strip() not in roles or not value.strip().isdigit():
            raise argparse.ArgumentTypeError(
                f"expected ROLE=N with ROLE one of {', '.join(roles)}, got {part!r}"
            )
        out[role.strip()] = int(value)
    return out


def _precisions(raw: str) -> list[str]:
    """``--precisions``: a comma list of target precisions (``exact`` is always added)."""
    from kernel_agent import precisions

    try:
        return precisions.parse(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _role_model(raw: str) -> dict[str, str | None]:
    """``--role-model ROLE=MODEL`` (``roles.parse_settings``)."""
    from kernel_agent import roles

    try:
        return roles.parse_settings([raw])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _role_effort(raw: str) -> dict[str, str | None]:
    """``--role-effort ROLE=LEVEL`` (``roles.parse_settings``)."""
    from kernel_agent import roles

    try:
        return roles.parse_settings([raw], effort=True)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _role_settings(ns: argparse.Namespace, key: str, defaults: bool = True) -> dict[str, Any]:
    """``role_models`` / ``role_efforts`` of the config: the defaults (``config.py``; without
    ``defaults`` none) with every ``--role-model`` / ``--role-effort`` over them, and
    ``--librarian-model`` for the librarian."""
    from kernel_agent.config import ROLE_EFFORTS, ROLE_MODELS

    out: dict[str, Any] = {}
    if defaults:
        out.update(ROLE_MODELS if key == "role_models" else ROLE_EFFORTS)
    if key == "role_models" and getattr(ns, "librarian_model", None):
        out["librarian"] = ns.librarian_model
    for given in getattr(ns, key, None) or []:
        out.update(given)
    return out


def _config(ns: argparse.Namespace) -> OptimizeConfig:
    return OptimizeConfig(
        model_ref=ns.model,
        runs_dir=Path(ns.runs_dir),
        modality=ns.modality,
        dtype=ns.dtype,
        trust_remote_code=ns.trust_remote_code,
        workload_options=_options(ns.option),
        harness=ns.harness,
        backends=[b.strip() for b in ns.backends.split(",") if b.strip()],
        max_targets=ns.max_targets,
        evaluations_per_target=ns.evaluations,
        parallel=ns.parallel,
        seeds_per_target=ns.seeds_per_target,
        reseed_workers=ns.reseed_workers,
        do_transforms=not ns.no_transforms,
        native=ns.native or "plan",
        native_minutes=ns.native_minutes,
        native_evaluations=ns.native_evaluations,
        claude_model=ns.claude_model,
        effort=ns.effort,
        max_turns_per_agent=ns.max_turns,
        budget_usd_per_agent=ns.budget,
        permission_mode=ns.permission_mode,
        allow_web=not ns.no_web,
        web_domains=ns.web_domain or [],
        dossier=not ns.no_dossier,
        auth=ns.auth,
        max_hours=ns.max_hours,
        max_usd=ns.max_usd,
        max_sessions=ns.max_sessions,
        agent_minutes=ns.agent_minutes,
        budget_reserve=ns.budget_reserve,
        eval_timeout_s=ns.eval_timeout,
        program=ns.program,
        compile_baseline=ns.compile_baseline,
        ab_rounds=ns.ab_rounds,
        ab_min_win_rate=ns.ab_min_win_rate,
        ab_min_gain=ns.ab_min_gain,
        early_stop=ns.early_stop != "off",
        recheck=not ns.no_recheck,
        export_check=not ns.no_export_check,
        quality=ns.quality,
        precisions=ns.precisions,
        use_library=not ns.no_library,
        librarian=not ns.no_librarian,
        library_scout=not ns.no_library_scout,
        role_models=_role_settings(ns, "role_models"),
        role_efforts=_role_settings(ns, "role_efforts"),
        hf_token=os.environ.get("HF_TOKEN"),
        verbose=ns.verbose,
    )


def cmd_optimize(ns: argparse.Namespace) -> int:
    from kernel_agent.orchestrator import optimize

    run = asyncio.run(optimize(_config(ns), until=ns.until))
    report = run.report
    if report.exists():
        print("\n" + report.read_text())
    print(f"\nrun directory: {run.root}")
    return 0


def cmd_analyze(ns: argparse.Namespace) -> int:
    from kernel_agent.orchestrator import optimize

    ns.until = "analyze"
    cfg = _config(ns)
    cfg.allow_harness_agent = not ns.no_harness_agent
    run = asyncio.run(optimize(cfg, until="analyze"))
    print((run.profile_dir / "summary.md").read_text())
    print(f"\nrun directory: {run.root}")
    return 0


def cmd_resume(ns: argparse.Namespace) -> int:
    from kernel_agent.orchestrator import Orchestrator

    overrides: dict[str, Any] = {}
    if ns.claude_model:
        overrides["claude_model"] = ns.claude_model
    if ns.program:
        overrides["program"] = ns.program
    if ns.auth:
        overrides["auth"] = ns.auth
    if ns.precisions:  # recorded in run.json (precisions.py)
        overrides["precisions"] = ns.precisions
    for key in ("role_models", "role_efforts"):  # over the run's, per role (#181)
        if given := _role_settings(ns, key, defaults=False):
            overrides[key] = given
    from kernel_agent.workspace import RunDir, coordinator_lock

    with coordinator_lock(RunDir(Path(ns.run_dir).resolve())):  # one process per run
        orch = Orchestrator.resume(Path(ns.run_dir), overrides)
        if ns.redo:
            from kernel_agent.orchestrator import PHASES

            def redo(data: dict[str, Any]) -> None:
                phases = data.get("phases", {})
                for phase in PHASES[PHASES.index(ns.redo) :]:
                    phases.pop(phase, None)

            orch.run.update(redo)
        run = asyncio.run(orch.run_all(until=ns.until))
    print(run.report.read_text() if run.report.exists() else "")
    return 0


def cmd_integrate(ns: argparse.Namespace) -> int:
    """Integrate a run again (the measurements of unchanged content reused, as the improve
    loop's re-integrations) and rewrite its report; no agent session."""
    from kernel_agent import interrupt
    from kernel_agent.orchestrator import Orchestrator
    from kernel_agent.report import write_report
    from kernel_agent.workspace import RunDir, coordinator_lock

    overrides = {"precisions": ns.precisions} if ns.precisions else {}
    with coordinator_lock(RunDir(Path(ns.run_dir).resolve())):  # one process per run
        orch = Orchestrator.resume(Path(ns.run_dir), overrides)  # --precisions: into run.json
        if orch.simulated:
            raise SystemExit(f"{ns.run_dir} is a dry-run run (improve --dry-run integrates it)")

        async def integrate() -> Path:
            await orch.integrate(reuse=not ns.no_reuse)
            path = write_report(orch.run)
            orch._mark("report")
            return path

        try:
            report = interrupt.run(integrate())
        except KeyboardInterrupt:
            return interrupt.EXIT_CODE
    print(f"\nreport: {report}\nrun directory: {orch.run.root}")
    return 0


def cmd_improve(ns: argparse.Namespace) -> int:
    from kernel_agent import interrupt
    from kernel_agent.governor import parse_agents
    from kernel_agent.improve import ImproveConfig, improve
    from kernel_agent.scheduler import Policy

    agents, governed = parse_agents(ns.agents)  # --agents N | auto | auto:N
    icfg = ImproveConfig(
        slice=ns.slice,
        rounds=ns.rounds,
        integrate_every=ns.integrate_every,
        max_slices=ns.max_slices,
        research_every=ns.research_every,
        integration_reserve=ns.integration_reserve,
        agents=agents,
        governor=governed,
        async_evals=ns.async_evals,
        **({"role_max": ns.role_max} if ns.role_max else {}),
        overlap=ns.overlap,
        board=ns.board,
        critic=ns.critic,
        critic_wait=ns.critic_wait,
        migrate_every=ns.migrate_every,
        cull_gap=ns.cull_gap,
        agent_gpu=ns.agent_gpu,
        timing_cores=ns.timing_cores,
        policy=Policy(
            patience=ns.patience,
            sol_stop=ns.sol_stop or None,
            target_hours=ns.target_hours or None,
            speedup_goal=ns.speedup_goal or None,
        ),
    )
    try:  # Ctrl-C / SIGTERM: stop the run's work and processes, record it, exit 130
        run = interrupt.run(improve(ns.model, _config(ns), icfg, dry_run=ns.dry_run, seed=ns.seed))
    except KeyboardInterrupt:
        return interrupt.EXIT_CODE
    print(f"\nreport: {run.report}\nrun directory: {run.root}")
    return 0


def cmd_eval(ns: argparse.Namespace) -> int:
    from kernel_agent.kernels.evaluate import run_evaluation

    # the timing context (#226): eager or CUDA graph, warm or cold L2
    timing = {"context": ns.context, "l2_flush": ns.l2 == "cold"}

    if ns.sweep:  # many configs of build(reference, **config), the best fully evaluated
        from kernel_agent.kernels import search, sweep

        found: dict[str, Any] | None = None
        configs: list[dict[str, Any]] = []
        notes: list[str] = []
        try:
            text = Path(ns.sweep).read_text()
            spec = json.loads(text) if text.lstrip().startswith("{") else None
            if isinstance(spec, dict) and ("space" in spec or "strategy" in spec):  # a search
                args = (spec.get(k) for k in ("space", "constraints", "strategy", "seed"))
                found = search.spec_from(*args)
            else:
                configs, notes = sweep.configs_from(text, ns.max_configs)
        except (OSError, ValueError) as exc:
            print(f"--sweep {ns.sweep}: {exc}", file=sys.stderr)
            return 2
        data = sweep.run_sweep(
            Path(ns.capture),
            Path(ns.candidate),
            configs,
            timeout=ns.timeout,
            compile_check=ns.compile_check,
            search=found,
            **timing,
        )
        data["sweep"]["notes"] = notes
        print(sweep.format_table(data), file=sys.stderr)
        print(json.dumps(data, indent=2, default=str))
        return 0 if data["evaluation"].get("correct") else 1
    result = run_evaluation(
        Path(ns.capture),
        Path(ns.candidate),
        profile=ns.profile,
        compile_baseline=ns.compile_baseline,
        timeout=ns.timeout,
        compile_check=ns.compile_check,
        quick=ns.quick,
        **timing,
    )
    if ns.profile and result.get("sass"):  # what to change, with its numbers (#230)
        from kernel_agent.kernels import directives

        result["directives"] = directives.texts(directives.build(result, result))
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("correct") else 1


def cmd_recheck(ns: argparse.Namespace) -> int:
    from kernel_agent.kernels.recheck import describe, run_recheck

    capture, candidate = Path(ns.capture), Path(ns.candidate)
    verdict: dict[str, Any] | None = None
    if ns.speedup is not None:
        verdict = {"correct": True, "speedup": ns.speedup}
    elif not ns.no_evaluate:  # the verdict to compare with: the evaluator's
        from kernel_agent.kernels.evaluate import run_evaluation

        verdict = run_evaluation(capture, candidate, timeout=ns.timeout)
        print(f"evaluator: {verdict.get('status')}, speedup {verdict.get('speedup')}", flush=True)
    result = run_recheck(
        capture, candidate, seeds=ns.seeds, seed=ns.seed, verdict=verdict, timeout=ns.timeout
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"recheck: {describe(result)}")
    return 0 if result.get("passed") else 1


def cmd_memcheck(ns: argparse.Namespace) -> int:
    from kernel_agent.kernels.memcheck import STATUS, describe, run_memcheck

    result = run_memcheck(
        Path(ns.capture), Path(ns.candidate), timeout=ns.timeout, variants=not ns.no_variants
    )
    print(json.dumps(result, indent=2, default=str))
    if result.get("report"):
        print(result["report"])
    print(describe(result))
    return 1 if result.get("status") == STATUS else 0


def cmd_install_claude_code(ns: argparse.Namespace) -> int:
    """``/optimize-model``, the agent definitions of every role and the skills (#176) into
    ``<project>/.claude/`` (``commands/``, ``agents/``, ``skills/<name>/``)."""
    import shutil

    from kernel_agent import roles, skills

    dst = Path(ns.project) / ".claude"
    commands = sorted((Path(__file__).parent / "claude_code" / "commands").glob("*.md"))
    copies = [(file, dst / "commands" / file.name) for file in commands]
    copies += [(r.path, dst / "agents" / r.path.name) for r in roles.definitions().values()]
    for skill in skills.index().values():
        for file in (skill.path, *skill.resources):
            copies.append((file, dst / "skills" / skill.name / file.relative_to(skill.dir)))
    for file, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    print(f"installed into {dst}:")
    print(f"  commands: {', '.join('/' + f.stem for f in commands)}")
    print(f"  agents:   {', '.join(roles.definitions())}")
    print(f"  skills:   {', '.join(skills.index())}")
    return 0


def cmd_program_init(ns: argparse.Namespace) -> int:
    from kernel_agent import program

    path = program.init(Path(ns.path), force=ns.force)
    print(f"wrote {path}; edit it and pass `--program {path}` to optimize, analyze or resume")
    return 0


def cmd_report(ns: argparse.Namespace) -> int:
    from kernel_agent.report import write_report
    from kernel_agent.workspace import RunDir

    path = write_report(RunDir(Path(ns.run_dir).resolve()))
    print(path.read_text())
    return 0


def cmd_status(ns: argparse.Namespace) -> int:
    import time

    from kernel_agent.status import render
    from kernel_agent.workspace import RunDir

    run = RunDir(Path(ns.run_dir).resolve())
    if not run.run_json.exists():
        raise SystemExit(f"{run.root} is not a run directory (no run.json)")
    if not ns.watch:
        print(render(run))
        return 0
    try:
        while True:
            text = render(run)
            print("\033[2J\033[H" + text, flush=True)  # clear the screen, cursor home
            time.sleep(ns.watch)
    except KeyboardInterrupt:
        return 0


def cmd_watch(ns: argparse.Namespace) -> int:
    from kernel_agent.watch import serve

    run_dir = Path(ns.run_dir).resolve()
    if not (run_dir / "run.json").exists():
        raise SystemExit(f"{run_dir} is not a run directory (no run.json)")
    serve(run_dir, host=ns.host, port=ns.port)
    return 0


def _add_auth_arg(p: argparse.ArgumentParser, default: str | None = "auto") -> None:
    p.add_argument(
        "--auth",
        choices=["auto", "subscription", "api"],
        default=default,
        help="subscription: agents run on your Claude Code login only (API key variables "
        "ignored, the login checked before the run); api: an API key or cloud provider "
        "only; auto: whatever Claude Code finds (default)",
    )


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="Hugging Face URL or repo id (org/name[@revision])")
    p.add_argument(
        "--modality", choices=["llm", "stt", "tts", "diffusion"], help="override detection"
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--option",
        "-o",
        action="append",
        metavar="KEY=VALUE",
        help="workload option, e.g. -o prompt_len=1024 -o new_tokens=128 -o steps=20",
    )
    p.add_argument("--harness", help="use your own harness.py (create(spec) -> Workload)")
    p.add_argument("--program", metavar="FILE", help="program.md with instructions for the agents")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument(
        "--backends",
        default=",".join(ALL_BACKENDS),
        help=f"comma list from {','.join(ALL_BACKENDS)} (and helion, opt in: when installed)",
    )
    p.add_argument("--max-targets", type=int, default=4)
    p.add_argument("--evaluations", type=int, default=12, help="evaluation budget per target")
    p.add_argument("--parallel", type=int, default=1, help="kernel agents running concurrently")
    p.add_argument(
        "--seeds-per-target",
        "--islands",
        dest="seeds_per_target",
        type=_seeds,
        metavar="K|auto",
        help="isolated workers per target, each from another approach or backend (auto: 2 "
        "for targets with >= 20%% of the profile; default 1). optimize splits the target's "
        "evaluation budget across them; improve keeps them as islands: each session goes to "
        "the island that pays most, with migration and culling (--migrate-every, --cull-gap)",
    )
    p.add_argument(
        "--reseed-workers",
        action="store_true",
        help="optimize, with several workers: a second round of sessions from each target's "
        "two best snapshots (half of the budget; improve: migration does this)",
    )
    p.add_argument("--no-transforms", action="store_true", help="skip model-level transforms")
    p.add_argument("--claude-model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    _add_role_args(p)
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--budget", type=float, default=None, help="USD cap per agent")
    p.add_argument(
        "--permission-mode",
        default="bypassPermissions",
        choices=["bypassPermissions", "acceptEdits", "default"],
    )
    p.add_argument("--no-web", action="store_true", help="disable WebFetch/WebSearch for agents")
    p.add_argument(
        "--web-domain",
        action="append",
        metavar="HOST",
        help="also let WebFetch reach this host and its subdomains (repeatable; the "
        "default list: documentation, code and paper sites, agent/web.py)",
    )
    p.add_argument(
        "--no-dossier",
        action="store_true",
        help="no research dossier session (research.md) before a target's first session",
    )
    _add_auth_arg(p)
    p.add_argument("--max-hours", type=float, help="wall-clock budget for the whole run")
    p.add_argument(
        "--max-usd",
        type=float,
        help="USD budget for all agents together (notional with --auth subscription)",
    )
    p.add_argument(
        "--max-sessions", type=int, help="agent sessions this invocation may start (all agents)"
    )
    p.add_argument("--agent-minutes", type=float, help="time limit per agent session")
    p.add_argument(
        "--native",
        choices=["off", "plan", "on"],
        help="improve: the native-engine arm (issue #134, docs/NATIVE.md): a systems-native "
        "agent rewrites stages as multi-file CUDA projects once every module arm has "
        "plateaued; plan (default): only when the plan asks for it",
    )
    p.add_argument(
        "--native-minutes",
        type=float,
        help="time limit per native session (default: 3 x --agent-minutes)",
    )
    p.add_argument("--native-evaluations", type=int, default=6, help="evaluations per native slice")
    p.add_argument(
        "--budget-reserve",
        type=float,
        default=0.15,
        help="share of --max-hours kept for integrate + report",
    )
    p.add_argument(
        "--eval-timeout", type=float, default=300.0, help="seconds per evaluate_candidate"
    )
    p.add_argument(
        "--compile-baseline",
        action="store_true",
        help="analyze: time a generic torch.compile when the workload has no "
        "reference_optimizations() hook",
    )
    p.add_argument(
        "--no-library",
        action="store_true",
        help="do not reuse or store kernels and lessons of the cross-run library",
    )
    p.add_argument("--no-librarian", action="store_true", help="skip the lessons agent")
    p.add_argument(
        "--no-library-scout",
        action="store_true",
        help="do not sweep library kernels (SDPA backends, cuBLASLt, installed libraries) on "
        "each target before its first agent session",
    )
    p.add_argument(
        "--ab-rounds", type=int, default=8, help="integration: timed rounds per paired A/B"
    )
    p.add_argument(
        "--ab-min-win-rate",
        type=float,
        default=0.8,
        help="integration: share of the A/B rounds an item must win",
    )
    p.add_argument(
        "--ab-min-gain",
        type=float,
        default=0.01,
        help="integration: the 95%% CI of an item's A/B gain must start above this",
    )
    p.add_argument(
        "--early-stop",
        choices=["on", "off"],
        default="on",
        help="early termination of hopeless measurements (on, default): an evaluation stops "
        "timing a correct candidate that cannot beat the best so far, a sweep stops timing "
        "configs clearly slower than its leader, an A/B stops once its verdict is decided; "
        "correctness is always checked in full. off: every measurement runs all its rounds",
    )
    p.add_argument(
        "--librarian-model", help="model of the librarian (= --role-model librarian=MODEL)"
    )
    p.add_argument(
        "--no-recheck",
        action="store_true",
        help="integration: do not re-check kernels on fresh inputs in separate processes",
    )
    p.add_argument(
        "--no-export-check",
        action="store_true",
        help="integration: do not self-test optimized/ in a fresh process outside the run",
    )
    p.add_argument(
        "--quality",
        default=DEFAULT_QUALITY,
        choices=list(QUALITIES),
        help="relaxed (default): numerics-changing optimisations (FP8 weights, W8A8, fusions "
        "that round differently, ...) pass module bounds about twice near-lossless's and a "
        "perceptual gate that allows small measured drops; near-lossless: the same within the "
        "noise of eager (WER, speaker similarity, MOS for TTS; teacher-forced KL for LLMs; "
        "workloads/perceptual.py); exact: numerics within rounding noise of eager. Recorded "
        "in run.json: a run keeps its mode",
    )
    _add_precisions_arg(p)
    p.add_argument("--verbose", "-v", action="store_true")


def _add_role_args(p: argparse.ArgumentParser) -> None:
    from kernel_agent.config import ROLE_EFFORTS, ROLE_MODELS

    models = ", ".join(f"{r}={m}" for r, m in ROLE_MODELS.items() if m != "inherit")
    efforts = ", ".join(f"{r}={e}" for r, e in ROLE_EFFORTS.items() if e != "inherit")
    p.add_argument(
        "--role-model",
        action="append",
        type=_role_model,
        dest="role_models",
        metavar="ROLE=MODEL",
        help="the model of one role (repeatable; inherit: --claude-model). Default: "
        f"--claude-model, except {models}. Recorded in run.json; given to a run that exists "
        "(improve, resume), it replaces that role's",
    )
    p.add_argument(
        "--role-effort",
        action="append",
        type=_role_effort,
        dest="role_efforts",
        metavar="ROLE=LEVEL",
        help="the effort of one role (repeatable; low, medium, high, xhigh, max; inherit: "
        f"--effort; none: the model's default). Default: --effort, except {efforts}",
    )


def _add_precisions_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--precisions",
        type=_precisions,
        metavar="P,P,...",
        help="target precisions the run allows, recorded in run.json (exact is always "
        "allowed): exact, fp8_weights, fp8_w8a8, fp8_mx, int8_weights, int8_w8a8, reduced, "
        "fp4_weights, fp8_kv (needs --quality relaxed or near-lossless). Default: relaxed and "
        "near-lossless allow all but the 4-bit fp4_weights and fp8_kv, which are opt-in; exact "
        "allows exact. Given to a run that exists (improve, resume, integrate), it replaces "
        "the run's list",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kernel-agent",
        description="Claude-powered kernel-level optimisation of Hugging Face models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="show GPU / compiler / backend status")
    p.add_argument("--smoke", action="store_true", help="compile and run a kernel per backend")
    p.add_argument(
        "--remeasure-peaks", action="store_true", help="measure the roofline peaks again"
    )
    p.add_argument("--no-peaks", action="store_true", help="do not measure missing peaks")
    p.add_argument(
        "--no-probes",
        action="store_true",
        help="skip the feature probes (tl.dot_scaled lowering, TMA, PDL, green contexts)",
    )
    p.add_argument(
        "--fetch-sanitizer",
        action="store_true",
        help="install compute-sanitizer from NVIDIA's CUDA redistributables (the pip wheel "
        "cannot launch anything)",
    )
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("optimize", help="full pipeline: profile, plan, write kernels, integrate")
    _add_run_args(p)
    p.add_argument(
        "--until", choices=["analyze", "plan", "capture", "kernels", "transforms", "integrate"]
    )
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("analyze", help="baseline + profile only (no kernel writing)")
    _add_run_args(p)
    p.add_argument("--no-harness-agent", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("resume", help="continue an interrupted run")
    p.add_argument("run_dir")
    p.add_argument(
        "--redo", choices=["plan", "capture", "kernels", "transforms", "integrate", "report"]
    )
    p.add_argument("--until", choices=["plan", "capture", "kernels", "transforms", "integrate"])
    p.add_argument("--claude-model")
    _add_role_args(p)
    p.add_argument("--program", metavar="FILE", help="replace the run's program.md with FILE")
    _add_auth_arg(p, default=None)  # None: the run's
    _add_precisions_arg(p)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser(
        "integrate",
        help="integrate a run again (unchanged measurements reused) and rewrite its report",
    )
    p.add_argument("run_dir")
    p.add_argument("--no-reuse", action="store_true", help="measure every A/B again")
    _add_precisions_arg(p)
    p.set_defaults(func=cmd_integrate)

    p = sub.add_parser(
        "improve",
        help="continuous loop: slices for the targets that pay most, until budget or plateau",
        description="MODEL is a run directory to continue or a Hugging Face URL / repo id to "
        "start (analyze, plan and capture first). --max-hours / --max-usd are the budget of "
        "this invocation, and it uses all of it: when every target has stopped it starts a "
        "new round (re-profile, re-plan); without them it runs until a round brings no gain.",
    )
    _add_run_args(p)
    p.add_argument("--slice", type=int, default=4, help="evaluations per slice (one session)")
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="re-profile + re-plan rounds (default: as many as the budget allows; without "
        "--max-hours / --max-usd / --max-sessions while each round brings a gain; 1: none)",
    )
    p.add_argument(
        "--integrate-every",
        type=int,
        default=4,
        help="re-integrate after this many kept results (0: only the final integration)",
    )
    p.add_argument(
        "--integration-reserve",
        type=_integration_reserve,
        default=None,
        metavar="auto|MINUTES",
        help="time --max-hours keeps for the final integration (auto: its estimate, at most "
        "a third of --max-hours; a longer one runs past it; 0: none)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="retire a target for the round after this many evals w/o a gain",
    )
    p.add_argument(
        "--sol-stop",
        type=float,
        default=0.9,
        help="retire a target for the round at this share of its recipe's roofline (0: off)",
    )
    p.add_argument(
        "--target-hours", type=float, default=2.0, help="time per target and round (0: off)"
    )
    p.add_argument(
        "--speedup-goal",
        type=float,
        default=2.0,
        help="retire a target for the round once it gained this speedup in it (0: off)",
    )
    p.add_argument("--max-slices", type=int, help="stop after this many slices")
    p.add_argument(
        "--research-every",
        type=int,
        default=3,
        help="research session on a plateaued target at most once per this many of its "
        "slices (0: never)",
    )
    p.add_argument(
        "--agents",
        type=_agents,
        default=3,
        metavar="N|auto",
        help="agent sessions at once (default 3; 1: one at a time, the sequential loop). "
        "With N > 1 the slots go to the arms that pay (slices, research sessions, dossiers), "
        "the GPU evaluations take turns in the GPU job queue with clean timing (the agents' "
        "GPU scripts through run_on_gpu, timing cores, builds before the lock, dirty "
        "timings re-run), the re-integration runs in the background and a usage limit "
        "pauses every session once (docs/MULTIAGENT.md); k = 3-4 is the measured sweet "
        "spot on one GPU. auto (auto:N: at most N, default 6): the governor keeps as many "
        "as make the run fastest: up to the GPU's measured knee, and no more than the usage "
        "windows sustain until they reset (a spent window pauses every session); the "
        "host's memory too, and USD only with --max-usd",
    )
    p.add_argument(
        "--async-evals",
        action="store_true",
        help="kernel and native sessions also get submit_evaluation / evaluation_result: "
        "an evaluation runs while the agent writes its next candidate",
    )
    p.add_argument(
        "--role-max",
        type=_role_max,
        default=None,
        metavar="ROLE=N,...",
        help="--agents N: sessions of a role at once (default kernel=4,systems=1,native=1)",
    )
    p.add_argument(
        "--overlap",
        choices=["allow", "warn", "avoid"],
        default="warn",
        help="--agents N: sessions working on the same modules at once: warn (default: their "
        "digests name each other), avoid (one at a time) or allow",
    )
    p.add_argument(
        "--board",
        choices=["auto", "on", "off"],
        default="auto",
        help="the blackboard (board.jsonl): the sessions post their conclusions (post_note: "
        "wins, traps, insights, claims, questions) and read the others' in their digests and "
        "on their evaluation results; kernel-agent posts every new best, integration and "
        "round. auto (default): with --agents above 1",
    )
    p.add_argument(
        "--critic",
        choices=["off", "static", "model"],
        default="static",
        help="the critic of every full evaluation: static (default) checks the candidate for "
        "known ways to game the evaluator (fallback to the reference, try/except fallbacks, "
        "outputs cached by address, unjoined streams, threads, frame reads, patches) and "
        "withdraws a confident reject before it reaches the GPU (force=true overrides); "
        "model also asks a cheap model (the critic role: Haiku, escalated to Sonnet) about "
        "inconclusive ones while their job waits for the GPU (with --agents above 1)",
    )
    p.add_argument(
        "--critic-wait",
        type=float,
        default=30.0,
        metavar="S",
        help="--critic model: ask the model only when the job is expected to wait this long",
    )
    p.add_argument(
        "--migrate-every",
        type=_count,
        default=4,
        metavar="M",
        help="--islands: every M evaluations of a target an island's next digest lists the "
        "best results of the other islands as inspirations (0: no migration)",
    )
    p.add_argument(
        "--cull-gap",
        type=_share,
        default=0.15,
        metavar="F",
        help="--islands: an island this share below its target's best after 8 evaluations "
        "and 2 sessions without a new island best is reseeded from that best (0: never)",
    )
    p.add_argument(
        "--agent-gpu",
        choices=["tool", "bash"],
        default=None,
        help="how the agents' own scripts reach the GPU: tool (default with --agents > 1: "
        "their Bash commands see no GPU and run GPU scripts with run_on_gpu, through the GPU "
        "job queue, so they never run during a timed evaluation) or bash (they see it)",
    )
    p.add_argument(
        "--timing-cores",
        type=int,
        default=None,
        metavar="N",
        help="physical CPU cores the timed GPU jobs get to themselves while builds and the "
        "agents run on the others at a lower priority (default 2 with --agents > 1, off below "
        "8 CPUs; 0: no CPU isolation)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="simulated agents and GPU (no Claude, no GPU)"
    )
    p.add_argument("--seed", type=int, default=0, help="--dry-run: seed of the simulation")
    p.set_defaults(func=cmd_improve)

    p = sub.add_parser("eval", help="evaluate a candidate against a capture file")
    p.add_argument("capture")
    p.add_argument("candidate")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--compile-baseline", action="store_true", help="also time torch.compile")
    p.add_argument(
        "--compile-check",
        action="store_true",
        help="also torch.compile the candidate: graph breaks + compiled outputs",
    )
    p.add_argument(
        "--quick", action="store_true", help="correctness on the smallest + largest case, untimed"
    )
    p.add_argument("--timeout", type=float, default=300.0, help="seconds before giving up")
    p.add_argument(
        "--sweep",
        metavar="CONFIGS_JSON",
        help="sweep build() keyword arguments: a JSON list of configs (or a dict of lists), "
        'or a search: {"space": {...}, "constraints": [...], "strategy": "auto", "seed": 0}; '
        "the best one is fully evaluated",
    )
    p.add_argument("--max-configs", type=int, default=None, help="--sweep: at most this many")
    p.add_argument(
        "--context",
        choices=("eager", "graph"),
        default="eager",
        help="time eager calls, or calls captured in a CUDA graph (as in a graphed stage)",
    )
    p.add_argument(
        "--l2",
        choices=("warm", "cold"),
        default="warm",
        help="cold: evict the L2 before every timed call",
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser(
        "recheck",
        help="re-check a winner on fresh inputs, reference and candidate in separate processes",
    )
    p.add_argument("capture")
    p.add_argument("candidate")
    p.add_argument("--seeds", type=int, default=3, help="fresh input draws per captured case")
    p.add_argument("--seed", type=int, help="first input seed (default: random)")
    p.add_argument(
        "--speedup",
        type=float,
        help="the evaluator's speedup to compare with (default: run the evaluator first)",
    )
    p.add_argument("--no-evaluate", action="store_true", help="no evaluator verdict to compare")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per subprocess")
    p.set_defaults(func=cmd_recheck)

    p = sub.add_parser(
        "memcheck",
        help="run a candidate on a capture's cases under compute-sanitizer memcheck",
    )
    p.add_argument("capture")
    p.add_argument("candidate")
    p.add_argument("--no-variants", action="store_true", help="the captured shapes only")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds")
    p.set_defaults(func=cmd_memcheck)

    p = sub.add_parser(
        "install-claude-code",
        help="add /optimize-model, kernel-agent's subagents and skills to a project's .claude/",
    )
    p.add_argument("project", nargs="?", default=".")
    p.set_defaults(func=cmd_install_claude_code)

    p = sub.add_parser("program", help="program.md: human-editable instructions for the agents")
    program_sub = p.add_subparsers(dest="program_command", required=True)
    p = program_sub.add_parser("init", help="write the default program.md for editing")
    p.add_argument("path", nargs="?", default="program.md", help="file or directory")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(func=cmd_program_init)

    p = sub.add_parser("report", help="(re)write report.md, charts and dashboard.html for a run")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("status", help="per-target progress, e2e, cost and the latest evaluations")
    p.add_argument("run_dir")
    p.add_argument("--watch", type=float, metavar="SECONDS", help="refresh every SECONDS")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("watch", help="live dashboard of a run in the browser (HTTP + SSE)")
    p.add_argument("run_dir")
    p.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (default: localhost)")
    p.set_defaults(func=cmd_watch)

    from kernel_agent import library

    library.add_parser(sub).set_defaults(func=library.main)

    from kernel_agent import doclib

    doclib.add_parser(sub).set_defaults(func=doclib.main)  # docs: the doc library (#177)

    from kernel_agent import suite

    suite.add_parser(sub).set_defaults(func=suite.main)  # bench-suite (KernelBench)

    from kernel_agent import experiments

    experiments.add_parser(sub).set_defaults(func=experiments.main)  # exp: the ledger (#222)

    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
