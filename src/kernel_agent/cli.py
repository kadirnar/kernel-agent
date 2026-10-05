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
    print(tc.summary())
    if ns.smoke:
        from kernel_agent.selftest import smoke_backends

        ok = smoke_backends(verbose=True)
        return 0 if ok else 1
    return 0


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
        do_transforms=not ns.no_transforms,
        claude_model=ns.claude_model,
        effort=ns.effort,
        max_turns_per_agent=ns.max_turns,
        budget_usd_per_agent=ns.budget,
        permission_mode=ns.permission_mode,
        allow_web=not ns.no_web,
        max_hours=ns.max_hours,
        max_usd=ns.max_usd,
        agent_minutes=ns.agent_minutes,
        budget_reserve=ns.budget_reserve,
        eval_timeout_s=ns.eval_timeout,
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


def cmd_eval(ns: argparse.Namespace) -> int:
    from kernel_agent.kernels.evaluate import run_evaluation

    result = run_evaluation(
        Path(ns.capture),
        Path(ns.candidate),
        profile=ns.profile,
        compile_baseline=ns.compile_baseline,
        timeout=ns.timeout,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("correct") else 1


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
    p.add_argument("--max-hours", type=float, help="wall-clock budget for the whole run")
    p.add_argument("--max-usd", type=float, help="USD budget for all agents together")
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
    p.add_argument("--verbose", "-v", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kernel-agent",
        description="Claude-powered kernel-level optimisation of Hugging Face models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="show GPU / compiler / backend status")
    p.add_argument("--smoke", action="store_true", help="compile and run a kernel per backend")
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
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("eval", help="evaluate a candidate against a capture file")
    p.add_argument("capture")
    p.add_argument("candidate")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--compile-baseline", action="store_true", help="also time torch.compile")
    p.add_argument("--timeout", type=float, default=300.0, help="seconds before giving up")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser(
        "install-claude-code", help="add /optimize-model + kernel-engineer subagent to a project"
    )
    p.add_argument("project", nargs="?", default=".")
    p.set_defaults(func=cmd_install_claude_code)

    p = sub.add_parser("report", help="(re)write report.md, charts and dashboard.html for a run")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("status", help="per-target progress, e2e, cost and the latest evaluations")
    p.add_argument("run_dir")
    p.add_argument("--watch", type=float, metavar="SECONDS", help="refresh every SECONDS")
    p.set_defaults(func=cmd_status)

    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
