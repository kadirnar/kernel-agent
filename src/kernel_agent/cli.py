"""``kernel-agent`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from kernel_agent.config import ALL_BACKENDS, DEFAULT_MODEL, OptimizeConfig


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
    from kernel_agent.gpulock import describe

    print(describe())  # the GPU lock pool (nvidia-smi, CUDA_VISIBLE_DEVICES, KERNEL_AGENT_GPUS)
    if ns.smoke:
        from kernel_agent.selftest import smoke_backends

        ok = smoke_backends(verbose=True)
        return 0 if ok else 1
    return 0


def _seeds(raw: str) -> int | str | None:
    """``--seeds-per-target``: a count or ``auto``."""
    from kernel_agent import workers

    try:
        return workers.parse(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


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
        claude_model=ns.claude_model,
        effort=ns.effort,
        max_turns_per_agent=ns.max_turns,
        budget_usd_per_agent=ns.budget,
        permission_mode=ns.permission_mode,
        allow_web=not ns.no_web,
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
        recheck=not ns.no_recheck,
        quality=ns.quality,
        use_library=not ns.no_library,
        librarian=not ns.no_librarian,
        librarian_model=ns.librarian_model,
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
    orch = Orchestrator.resume(Path(ns.run_dir), overrides)
    if ns.redo:
        data = orch.run.load()
        phases = data.get("phases", {})
        from kernel_agent.orchestrator import PHASES

        for phase in PHASES[PHASES.index(ns.redo) :]:
            phases.pop(phase, None)
        from kernel_agent.workspace import write_json

        write_json(orch.run.run_json, data)
    run = asyncio.run(orch.run_all(until=ns.until))
    print(run.report.read_text() if run.report.exists() else "")
    return 0


def cmd_improve(ns: argparse.Namespace) -> int:
    from kernel_agent.improve import ImproveConfig, improve
    from kernel_agent.scheduler import Policy

    icfg = ImproveConfig(
        slice=ns.slice,
        rounds=ns.rounds,
        integrate_every=ns.integrate_every,
        max_slices=ns.max_slices,
        research_every=ns.research_every,
        policy=Policy(
            patience=ns.patience,
            sol_stop=ns.sol_stop or None,
            target_hours=ns.target_hours or None,
            speedup_goal=ns.speedup_goal or None,
        ),
    )
    try:
        run = asyncio.run(improve(ns.model, _config(ns), icfg, dry_run=ns.dry_run, seed=ns.seed))
    except KeyboardInterrupt:
        return 130
    print(f"\nreport: {run.report}\nrun directory: {run.root}")
    return 0


def cmd_eval(ns: argparse.Namespace) -> int:
    from kernel_agent.kernels.evaluate import run_evaluation

    if ns.sweep:  # many configs of build(reference, **config), the best fully evaluated
        from kernel_agent.kernels import sweep

        try:
            configs, notes = sweep.configs_from(Path(ns.sweep).read_text(), ns.max_configs)
        except (OSError, ValueError) as exc:
            print(f"--sweep {ns.sweep}: {exc}", file=sys.stderr)
            return 2
        data = sweep.run_sweep(
            Path(ns.capture),
            Path(ns.candidate),
            configs,
            timeout=ns.timeout,
            compile_check=ns.compile_check,
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
    )
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


def cmd_install_claude_code(ns: argparse.Namespace) -> int:
    import shutil

    src = Path(__file__).parent / "claude_code"
    dst = Path(ns.project) / ".claude"
    for kind in ("commands", "agents"):
        (dst / kind).mkdir(parents=True, exist_ok=True)
        for file in (src / kind).glob("*.md"):
            shutil.copy2(file, dst / kind / file.name)
            print(f"installed {dst / kind / file.name}")
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
        help=f"comma list from {','.join(ALL_BACKENDS)}",
    )
    p.add_argument("--max-targets", type=int, default=4)
    p.add_argument("--evaluations", type=int, default=12, help="evaluation budget per target")
    p.add_argument("--parallel", type=int, default=1, help="kernel agents running concurrently")
    p.add_argument(
        "--seeds-per-target",
        type=_seeds,
        metavar="K|auto",
        help="isolated workers per target, each from another approach or backend (auto: 2 "
        "for targets with >= 20%% of the profile); the target's evaluation budget is split "
        "across them (default 1)",
    )
    p.add_argument(
        "--reseed-workers",
        action="store_true",
        help="with several workers: a second round of sessions from each target's two best "
        "snapshots (half of the budget)",
    )
    p.add_argument("--no-transforms", action="store_true", help="skip model-level transforms")
    p.add_argument("--claude-model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--budget", type=float, default=None, help="USD cap per agent")
    p.add_argument(
        "--permission-mode",
        default="bypassPermissions",
        choices=["bypassPermissions", "acceptEdits", "default"],
    )
    p.add_argument("--no-web", action="store_true", help="disable WebFetch/WebSearch for agents")
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
    p.add_argument("--librarian-model", help="model of the librarian (default: --claude-model)")
    p.add_argument(
        "--no-recheck",
        action="store_true",
        help="integration: do not re-check kernels on fresh inputs in separate processes",
    )
    p.add_argument(
        "--quality",
        default="exact",
        choices=["exact", "near-lossless"],
        help="exact: numerics within rounding noise of eager; near-lossless: numerics-changing "
        "optimisations (FP8 weights, ...) pass a perceptual gate instead (WER, speaker "
        "similarity, MOS for TTS; workloads/perceptual.py)",
    )
    p.add_argument("--verbose", "-v", action="store_true")


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
    p.add_argument("--program", metavar="FILE", help="replace the run's program.md with FILE")
    _add_auth_arg(p, default=None)  # None: the run's
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser(
        "improve",
        help="continuous loop: slices for the targets that pay most, until budget or plateau",
        description="MODEL is a run directory to continue or a Hugging Face URL / repo id to "
        "start (analyze, plan and capture first). --max-hours / --max-usd are the budget of "
        "this invocation; without them it runs until every target has stopped.",
    )
    _add_run_args(p)
    p.add_argument("--slice", type=int, default=4, help="evaluations per slice (one session)")
    p.add_argument("--rounds", type=int, default=1, help="re-profile + re-plan rounds (1 = none)")
    p.add_argument(
        "--integrate-every",
        type=int,
        default=4,
        help="re-integrate after this many kept results (0: only the final integration)",
    )
    p.add_argument(
        "--patience", type=int, default=5, help="stop a target after this many evals w/o a gain"
    )
    p.add_argument("--sol-stop", type=float, default=0.9, help="stop at this share of SOL (0: off)")
    p.add_argument("--target-hours", type=float, default=2.0, help="time cap per target (0: off)")
    p.add_argument(
        "--speedup-goal", type=float, default=2.0, help="stop a target at this speedup (0: off)"
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
        help="sweep build() keyword arguments: a JSON list of configs (or a dict of lists); "
        "the best one is fully evaluated",
    )
    p.add_argument("--max-configs", type=int, default=None, help="--sweep: at most this many")
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
        "install-claude-code", help="add /optimize-model + kernel-engineer subagent to a project"
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

    from kernel_agent import suite

    suite.add_parser(sub).set_defaults(func=suite.main)  # bench-suite (KernelBench)

    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
