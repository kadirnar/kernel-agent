"""``kernel-agent bench-suite``: KernelBench as a regression suite for the agent (issue #21).

To A/B-test a ``program.md``, prompt or evaluator change, run the same problems before
and after it and compare fast_p:

1. the problems of ``--kernelbench-level`` (the first ``--n`` by id, or ``--problems``)
   become the targets of one suite run (:mod:`kernel_agent.kernelbench`: problem files
   fetched at run time, sizes fitted to ``--max-input-mb``, sealed module copies and
   captures with one timed case and two correctness-only seeds);
2. :meth:`Orchestrator.kernels` runs the normal engineer agent on every problem with
   ``--evaluations`` evaluations, the cross-run library off (no prior winners, nothing
   stored), no transforms and no integration. ``--dry-run`` replaces Claude with
   :class:`FakeEngineer` (no Claude, no cost);
3. :func:`score` evaluates each problem's best snapshot again with
   ``compile_baseline=True``: speedup vs eager and vs ``torch.compile``
   (``max-autotune-no-cudagraphs``) on the timed case, weighted by calls per run;
4. ``suite.json``, ``suite.md`` and ``fast_p.png`` in the run directory: fast_p at
   p = 1.0, 1.1, 1.25, 1.5 and 2 against both baselines (KernelBench's metric: the share
   of *all* problems whose kernel is correct and more than p times faster), a row per
   problem and the agents' cost (``costs.json``).

The suite run is an ordinary run directory: ``kernel-agent status`` / ``watch`` work on it
while it runs, ``targets/<id>/`` has every candidate, record and note.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import importlib.util
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel_agent import charts, kernelbench, ledger, program, toolchain, truth
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import ALL_BACKENDS, DEFAULT_MODEL, OptimizeConfig
from kernel_agent.workspace import RunDir, read_json, write_json

if TYPE_CHECKING:
    from kernel_agent.orchestrator import Orchestrator

#: Speedup thresholds of fast_p.
THRESHOLDS = (1.0, 1.1, 1.25, 1.5, 2.0)
SUITE_JSON, SUITE_MD, CHART = "suite.json", "suite.md", "fast_p.png"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class SuiteConfig:
    level: int = 1
    n: int | None = 20
    problems: list[int] | None = None  # ids; replaces n
    evaluations: int = 4  # per problem
    kernelbench_dir: Path | None = None  # a local checkout instead of the download
    kernelbench_ref: str = "main"
    runs_dir: Path = Path("runs")
    max_input_mb: float = 16.0  # inputs + outputs of one call (kernelbench.fit_sizes)
    dry_run: bool = False
    backends: list[str] = field(default_factory=lambda: list(ALL_BACKENDS))
    claude_model: str = DEFAULT_MODEL
    effort: str | None = "high"
    max_usd: float | None = None
    agent_minutes: float | None = 30.0
    parallel: int = 1
    program: str | None = None
    eval_timeout_s: float = 300.0
    device: str | None = None  # default: cuda when available

    def optimize_config(self) -> OptimizeConfig:
        """The run configuration: kernels only, no library, no transforms, no integration."""
        return OptimizeConfig(
            model_ref=f"KernelBench/level{self.level}",
            runs_dir=self.runs_dir,
            backends=list(self.backends),
            evaluations_per_target=self.evaluations,
            parallel=self.parallel,
            do_transforms=False,
            allow_harness_agent=False,
            recheck=False,
            use_library=False,
            librarian=False,
            claude_model=self.claude_model,
            effort=self.effort,
            max_usd=self.max_usd,
            agent_minutes=self.agent_minutes,
            eval_timeout_s=self.eval_timeout_s,
            program=self.program,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["runs_dir"] = str(self.runs_dir)
        data["kernelbench_dir"] = str(self.kernelbench_dir) if self.kernelbench_dir else None
        return data


# ------------------------------------------------------------------ fast_p


def fast_p(
    rows: list[dict[str, Any]], key: str = "speedup", thresholds: tuple[float, ...] = THRESHOLDS
) -> dict[str, float]:
    """Share of all ``rows`` that are correct with ``row[key]`` above each threshold."""
    n = len(rows)

    def share(p: float) -> float:
        hits = sum(1 for r in rows if r.get("correct") and float(r.get(key) or 0.0) > p)
        return round(hits / n, 4) if n else 0.0

    return {f"{p:g}": share(p) for p in thresholds}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """fast_p against both baselines, correctness, geometric mean speedups and cost."""
    correct = [r for r in rows if r.get("correct")]

    def geomean(key: str) -> float | None:
        values = [float(r[key]) for r in correct if r.get(key)]
        return round(math.exp(statistics.fmean(map(math.log, values))), 3) if values else None

    return {
        "problems": len(rows),
        "correct": len(correct),
        "fast_p": {"eager": fast_p(rows, "speedup"), "compile": fast_p(rows, "speedup_vs_compile")},
        "geomean_speedup": geomean("speedup"),
        "geomean_speedup_vs_compile": geomean("speedup_vs_compile"),
        "usd": round(sum(float(r.get("usd") or 0.0) for r in rows), 4),
        "evaluations": sum(int(r.get("evaluations") or 0) for r in rows),
    }


# ------------------------------------------------------------------ the dry-run engineer

RMSNORM_CANDIDATE = '''"""dry run: KernelBench RMSNorm (over dim 1, no weight) on the kernel of the
bundled Triton RMSNorm example: features moved last, a weight of ones."""
import importlib.util

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location("ka_example_triton_rmsnorm", {example!r})
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


class FeatureRMSNorm(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.eps = float(reference.eps)
        self.weight = None
        self.norm = None

    def forward(self, x):
        rows = x.movedim(1, -1)
        if self.norm is None or self.weight.device != x.device or self.weight.dtype != x.dtype:
            self.weight = torch.ones(rows.shape[-1], device=x.device, dtype=x.dtype)
            self.norm = example.TritonRMSNorm(self)
        return self.norm(rows).movedim(-1, 1)


def build(reference):
    return FeatureRMSNorm(reference)
'''

REWRITE_BUILD = '''

def build(reference):
    """dry run: the problem's own torch code as a class of this file (an honest rewrite)."""
    new = copy.copy(reference)  # shares the reference's parameters and buffers
    new.__class__ = Rewrite
    return new
'''


def dry_run_candidate(spec: dict[str, Any], source: str, *, triton: bool) -> tuple[str, str, str]:
    """``(file name, source, hypothesis)`` of the fake engineer's candidate for a problem: the
    bundled Triton RMSNorm example for RMSNorm (with Triton), otherwise the problem's own
    torch code in a class of the candidate file."""
    name = str((spec.get("kernelbench") or {}).get("name", "")).lower()
    if triton and "rmsnorm" in name:
        text = RMSNORM_CANDIDATE.format(example=str(EXAMPLES_DIR / "triton_rmsnorm.py"))
        return "triton_rmsnorm_v1.py", text, "the bundled Triton RMSNorm kernel, features last"
    text = "import copy\n\n" + re.sub(r"\bModel\b", "Rewrite", source) + REWRITE_BUILD
    return "torch_rewrite_v1.py", text, "the reference's torch code as an honest rewrite"


class FakeEngineer:
    """``--dry-run``: the engineer agent without Claude. One candidate per problem
    (:func:`dry_run_candidate`), evaluated and recorded with the evaluation tool's own
    code (snapshot, :func:`kernels.evaluate.run_evaluation`, ``record_candidate``)."""

    def __init__(self, orch: Orchestrator, device: str) -> None:
        self.orch = orch
        triton = importlib.util.find_spec("triton") is not None
        self.triton = triton and device.startswith("cuda")

    async def run(
        self,
        name: str,
        *,
        cwd: Path,
        result: AgentResult | None = None,
        **_: Any,
    ) -> AgentResult:
        from kernel_agent.agent.tools import record_candidate, snapshot
        from kernel_agent.kernels.evaluate import run_evaluation

        run, keeper = self.orch.run, self.orch.truth
        result = result or AgentResult(name=name)
        target_id = name.removeprefix("kernel-")
        spec = read_json(run.target(target_id) / "spec.json", {}) or {}
        source = (run.target(target_id) / "reference_source.py").read_text()
        file, text, hypothesis = dry_run_candidate(spec, source, triton=self.triton)
        src = Path(cwd) / "candidates" / file
        src.write_text(text)
        start = time.perf_counter()
        snap = snapshot(run, src, target_id)
        capture = run.capture_file(target_id)
        evaluation = await asyncio.to_thread(
            run_evaluation,
            capture,
            snap,
            timeout=self.orch.budget.eval_timeout_s,
            capture_sha256=keeper.expect(capture),
        )
        _, row = record_candidate(
            run,
            target_id,
            src,
            snap,
            evaluation,
            hypothesis=hypothesis,
            eval_s=round(time.perf_counter() - start, 1),
            keeper=keeper,
            idea="dry_run",
        )
        result.tool_calls = {"evaluate_candidate": 1}
        result.turns = 1
        result.seconds = time.perf_counter() - start
        result.text = f"dry run: {file}: {evaluation.get('status')} ({row['status']})"
        return result


# ------------------------------------------------------------------ the run


def _source(cfg: SuiteConfig) -> tuple[Path, dict[str, Any]]:
    """The KernelBench directory and where it came from."""
    if cfg.kernelbench_dir is not None:
        root = kernelbench.find_root(cfg.kernelbench_dir)
        if root is None:
            raise SystemExit(f"{cfg.kernelbench_dir}: no KernelBench level<N>/ folders found")
        return root, {"dir": str(root)}
    log(f"bench-suite: KernelBench {cfg.kernelbench_ref} (cached after the first download)")
    root = kernelbench.fetch(cfg.kernelbench_ref)
    return root, {
        "ref": cfg.kernelbench_ref,
        "url": kernelbench.TARBALL.format(ref=cfg.kernelbench_ref),
        "dir": str(root),
    }


def create_run(
    cfg: SuiteConfig,
    opt: OptimizeConfig,
    source: dict[str, Any],
    problems: list[kernelbench.Problem],
) -> RunDir:
    """A run directory for the suite (``.truth/`` layout, analyze/plan/capture marked done)."""
    repo = f"KernelBench/level{cfg.level}"
    run = RunDir.create(cfg.runs_dir, repo)
    write_json(
        run.run_json,
        {
            "card": {
                "repo_id": repo,
                "revision": source.get("ref", "local"),
                "modality": "kernelbench",
                "architectures": [],
                "params": None,
                "size_gb": None,
            },
            "config": opt.to_dict(),
            "phases": {p: {"done": True, "by": "bench-suite"} for p in ("analyze", "plan")},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "suite": {
                **cfg.to_dict(),
                "source": source,
                "problems": [p.to_dict() for p in problems],
            },
            "truth": truth.new_section(),
        },
    )
    tc = toolchain.setup()
    to_dict = getattr(tc, "to_dict", None)
    write_json(run.toolchain_json, to_dict() if callable(to_dict) else {})
    program.install(run, opt.program)
    return run


def _free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_all(
    run: RunDir, problems: list[kernelbench.Problem], cfg: SuiteConfig, device: str
) -> dict[str, dict[str, Any]]:
    """Every problem as a target (:func:`kernelbench.prepare`, under the GPU lock); returns
    the rows of the problems that could not be captured."""
    from kernel_agent.gpulock import gpu_lock

    keeper = truth.of(run)
    failed: dict[str, dict[str, Any]] = {}
    for problem in problems:
        try:
            with gpu_lock() if device.startswith("cuda") else contextlib.nullcontext():
                spec = kernelbench.prepare(
                    run,
                    problem,
                    keeper,
                    device=device,
                    max_mb=cfg.max_input_mb,
                    backends=list(cfg.backends),
                )
            scaled = spec["kernelbench"]["scaled"]
            log(
                f"bench-suite: {problem.target_id}: captured"
                + (
                    f" (scaled {', '.join(f'{k} {a}->{b}' for k, (a, b) in scaled.items())})"
                    if scaled
                    else ""
                )
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            log(f"bench-suite: {problem.target_id}: capture failed: {error}")
            failed[problem.target_id] = {
                "problem": problem.to_dict(),
                "target": problem.target_id,
                "correct": False,
                "status": "capture_failed",
                "error": error,
            }
        finally:
            _free_gpu()
    data = run.load()
    data.setdefault("phases", {})["capture"] = {"done": True, "by": "bench-suite"}
    write_json(run.run_json, data)
    return failed


def _weighted(cases: list[dict[str, Any]], key: str) -> float | None:
    """Σ calls per run × ``key`` over the timed cases (None if one has no number)."""
    timed = [c for c in cases if c.get("ref_ms") is not None]
    values = [c.get(key) for c in timed]
    if not timed or not all(isinstance(v, int | float) for v in values):
        return None
    return sum(float(c.get("calls_per_run") or 0) * float(c[key]) for c in timed)


def score(
    run: RunDir, keeper: truth.Truth, problem: kernelbench.Problem, *, timeout: float
) -> dict[str, Any]:
    """The problem's row: its best snapshot evaluated again with ``compile_baseline``."""
    from kernel_agent.agent.tools import best_for_target
    from kernel_agent.kernels.evaluate import run_evaluation

    target_id = problem.target_id
    records = keeper.records(run.results_file(target_id))
    row: dict[str, Any] = {
        "problem": problem.to_dict(),
        "target": target_id,
        "correct": False,
        "evaluations": len(records),
        "speedup": None,
        "speedup_vs_compile": None,
    }
    best = best_for_target(run, target_id, keeper)
    if best is None:  # without CUDA nothing is timed: the last correct snapshot
        history = run.history_dir(target_id)
        best = next(
            (
                r
                for r in records[::-1]
                if r.get("correct")
                and keeper.snapshot_ok(
                    history / Path(str(r.get("snapshot"))).name, r.get("snapshot_sha256")
                )
            ),
            None,
        )
    if best is None:
        found = Counter(str(r.get("status")) for r in records)
        row.update(status="no_correct_kernel", attempts=dict(found))
        return row
    snap = run.history_dir(target_id) / Path(str(best["snapshot"])).name
    capture = run.capture_file(target_id)
    result = run_evaluation(
        capture, snap, compile_baseline=True, capture_sha256=keeper.expect(capture), timeout=timeout
    )
    row.update(
        snapshot=snap.name,
        agent_speedup=best.get("speedup"),
        status=result.get("status"),
        correct=bool(result.get("correct")),
    )
    if not row["correct"]:
        row["error"] = str(result.get("error") or "")[:500]
        return row
    cases = result.get("cases") or []
    eager, new = _weighted(cases, "ref_ms"), _weighted(cases, "new_ms")
    compiled = _weighted(cases, "torch_compile_ms")
    row.update(
        speedup=result.get("speedup"),
        eager_ms=eager,
        new_ms=new,
        compile_ms=compiled,
        speedup_vs_compile=round(compiled / new, 3) if compiled and new else None,
        pct_of_sol=max((c.get("pct_of_sol") or 0 for c in cases), default=None) or None,
    )
    if result.get("timing"):
        row["timing"] = result["timing"]
    return row


def _costs(run: RunDir, rows: list[dict[str, Any]]) -> None:
    """``usd`` and agent minutes of every row, from ``costs.json``."""
    costs = read_json(run.root / "costs.json", {}) or {}
    for row in rows:
        mine = [
            c
            for name, c in costs.items()
            if name == f"kernel-{row['target']}" or name.startswith(f"kernel-{row['target']}-")
        ]
        row["usd"] = round(sum(float(c.get("usd") or 0.0) for c in mine), 4)
        row["agent_minutes"] = round(sum(float(c.get("minutes") or 0.0) for c in mine), 1)


async def run_suite(cfg: SuiteConfig) -> RunDir:
    """Run the suite (see the module docstring); returns its run directory."""
    import torch

    from kernel_agent.orchestrator import Orchestrator

    root, source = _source(cfg)
    problems = kernelbench.problems(root, cfg.level, n=cfg.n, ids=cfg.problems)
    opt = cfg.optimize_config()
    run = create_run(cfg, opt, source, problems)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"bench-suite: {len(problems)} problems of level {cfg.level} -> {run.root}")
    failed = prepare_all(run, problems, cfg, device)
    orch = Orchestrator(run, opt)
    orch.phase = "kernels"
    if cfg.dry_run:
        orch.agent_runner = FakeEngineer(orch, device).run
    ledger.event(run, "phase_start", phase="kernels")
    await orch.kernels()
    ledger.event(run, "phase_done", phase="kernels")
    timeout = 2 * cfg.eval_timeout_s  # torch.compile max-autotune of the baseline too
    rows = []
    for problem in problems:
        row = failed.get(problem.target_id)
        if row is None:
            log(f"bench-suite: {problem.target_id}: scoring (with the torch.compile baseline)")
            row = await asyncio.to_thread(score, run, orch.truth, problem, timeout=timeout)
        rows.append(row)
    _costs(run, rows)
    write_results(run, cfg, source, rows)
    return run


# ------------------------------------------------------------------ outputs


def _x(value: Any) -> str:
    return f"{value:.2f}x" if isinstance(value, int | float) else "—"


def _ms(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else "—"


def write_results(
    run: RunDir, cfg: SuiteConfig, source: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """``suite.json``, ``suite.md`` and ``fast_p.png`` in the run directory."""
    summary = summarise(rows)
    data = {"config": cfg.to_dict(), "source": source, "summary": summary, "rows": rows}
    write_json(run.root / SUITE_JSON, data)
    chart = fast_p_chart(run.root / CHART, summary, cfg.level)
    engineer = "fake engineer (--dry-run)" if cfg.dry_run else f"Claude `{cfg.claude_model}`"
    prog = (run.load().get("program") or {}).get("sha256")
    lines = [
        f"# KernelBench level {cfg.level} suite",
        "",
        f"* problems: {summary['problems']} ({', '.join(str(r['problem']['id']) for r in rows)}), "
        f"from {source.get('ref') or source.get('dir')}; inputs + outputs fitted to "
        f"{cfg.max_input_mb:g} MB per call",
        f"* engineer: {engineer}, {cfg.evaluations} evaluations per problem, library off"
        + (f", program.md `{str(prog)[:12]}`" if prog else ""),
        f"* correct: {summary['correct']}/{summary['problems']}; geometric mean speedup of the "
        f"correct ones: {_x(summary['geomean_speedup'])} vs eager, "
        f"{_x(summary['geomean_speedup_vs_compile'])} vs torch.compile",
        f"* cost: ${summary['usd']:.2f}, {summary['evaluations']} evaluations",
        "",
        "fast_p: share of all problems whose kernel is correct and more than p times faster.",
        "",
        "| fast_p | " + " | ".join(f"p = {p:g}" for p in THRESHOLDS) + " |",
        "|---|" + "---|" * len(THRESHOLDS),
    ]
    for key, label in (("eager", "vs eager"), ("compile", "vs torch.compile")):
        shares = summary["fast_p"][key]
        lines.append(
            f"| {label} | " + " | ".join(f"{shares[f'{p:g}']:.0%}" for p in THRESHOLDS) + " |"
        )
    lines.append("")
    if chart is not None:
        lines += [f"![fast_p]({CHART})", ""]
    lines += [
        "| problem | status | vs eager | vs torch.compile | eager ms | compile ms | kernel ms "
        "| evals | $ | best snapshot |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = r["problem"]
        status = str(r.get("status"))
        if r.get("error"):
            status += f": {(str(r['error']).splitlines() or [''])[0][:60]}"
        if r.get("attempts"):  # no correct kernel: what the evaluations ended in
            status += " (" + ", ".join(f"{k} ×{n}" for k, n in r["attempts"].items()) + ")"
        lines.append(
            f"| {p['id']} `{p['name']}` | {status} | {_x(r.get('speedup'))} "
            f"| {_x(r.get('speedup_vs_compile'))} | {_ms(r.get('eager_ms'))} "
            f"| {_ms(r.get('compile_ms'))} | {_ms(r.get('new_ms'))} | {r.get('evaluations', 0)} "
            f"| {float(r.get('usd') or 0):.2f} | {r.get('snapshot') or '—'} |"
        )
    lines += [
        "",
        f"Run directory: `{run.root}` (targets/<id>/ has every candidate, record and note; "
        "`kernel-agent status` and `watch` work on it).",
        "",
    ]
    (run.root / SUITE_MD).write_text("\n".join(lines))
    return summary


def fast_p_chart(path: Path, summary: dict[str, Any], level: int) -> Path | None:
    """Grouped bars: fast_p per threshold, against eager and against torch.compile."""
    if not charts.available():
        return None

    def draw(fig: Any, ax: Any) -> None:
        from matplotlib.patches import Patch
        from matplotlib.ticker import PercentFormatter

        width, gap = 0.36, 0.04
        series = (
            ("eager", "vs eager", charts.PROJECTED_COLOR),
            ("compile", "vs torch.compile", charts.COMPILE_COLOR),
        )
        for k, (key, _, color) in enumerate(series):
            values = [summary["fast_p"][key][f"{p:g}"] for p in THRESHOLDS]
            xs = [i + (k - 0.5) * (width + gap) for i in range(len(THRESHOLDS))]
            ax.bar(xs, values, width, color=color, zorder=3)
            for x, v in zip(xs, values, strict=True):
                ax.annotate(
                    f"{v:.0%}",
                    (x, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=charts.INK_2,
                )
        ax.set_xticks(range(len(THRESHOLDS)), [f"{p:g}×" for p in THRESHOLDS])
        ax.set_xlabel("speedup threshold p")
        ax.set_ylabel("fast_p (share of problems)")
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
        charts._header(
            ax,
            f"KernelBench level {level}: fast_p",
            f"{summary['problems']} problems, {summary['correct']} correct, ${summary['usd']:.2f}",
        )
        handles = [Patch(color=color, label=label) for _, label, color in series]
        charts._legend(ax, handles)

    return charts._render(path, (7.0, 3.6), draw)


# ------------------------------------------------------------------ command line


def _ids(raw: str) -> list[int]:
    try:
        return [int(x) for x in raw.replace(" ", "").split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected problem ids like 1,19,36, got {raw!r}"
        ) from None


def add_parser(sub: Any) -> argparse.ArgumentParser:
    """The ``kernel-agent bench-suite`` command (``sub``: the CLI's subparsers)."""
    p: argparse.ArgumentParser = sub.add_parser(
        "bench-suite",
        help="KernelBench regression suite: fast_p vs eager and torch.compile",
        description="Run the engineer agent on KernelBench problems with a small budget and "
        "report fast_p (A/B-test program.md, prompt or evaluator changes). KernelBench is "
        "downloaded at run time into the cache (or --kernelbench-dir).",
    )
    p.add_argument("--kernelbench-level", type=int, default=1, help="KernelBench level")
    p.add_argument("--n", type=int, default=20, help="the first N problems by id")
    p.add_argument("--problems", type=_ids, help="problem ids instead, e.g. 1,19,36")
    p.add_argument("--evaluations", type=int, default=4, help="evaluations per problem")
    p.add_argument("--dry-run", action="store_true", help="a fake engineer instead of Claude")
    p.add_argument("--kernelbench-dir", type=Path, help="a local KernelBench checkout")
    p.add_argument("--kernelbench-ref", default="main", help="branch, tag or commit to fetch")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument(
        "--max-input-mb",
        type=float,
        default=16.0,
        help="scale problem sizes so one call's inputs + outputs fit this many MB",
    )
    p.add_argument("--backends", default=",".join(ALL_BACKENDS), help="comma list")
    p.add_argument("--claude-model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-usd", type=float, help="USD budget for all agents together")
    p.add_argument("--agent-minutes", type=float, default=30.0, help="time limit per problem")
    p.add_argument("--parallel", type=int, default=1, help="problems worked on concurrently")
    p.add_argument("--program", metavar="FILE", help="program.md for the agents")
    p.add_argument("--eval-timeout", type=float, default=300.0, help="seconds per evaluation")
    return p


def main(ns: argparse.Namespace) -> int:
    cfg = SuiteConfig(
        level=ns.kernelbench_level,
        n=ns.n,
        problems=ns.problems,
        evaluations=ns.evaluations,
        kernelbench_dir=ns.kernelbench_dir,
        kernelbench_ref=ns.kernelbench_ref,
        runs_dir=Path(ns.runs_dir),
        max_input_mb=ns.max_input_mb,
        dry_run=ns.dry_run,
        backends=[b.strip() for b in ns.backends.split(",") if b.strip()],
        claude_model=ns.claude_model,
        effort=ns.effort,
        max_usd=ns.max_usd,
        agent_minutes=ns.agent_minutes,
        parallel=ns.parallel,
        program=ns.program,
        eval_timeout_s=ns.eval_timeout,
    )
    run = asyncio.run(run_suite(cfg))
    print((run.root / SUITE_MD).read_text())
    summary = json.loads((run.root / SUITE_JSON).read_text())["summary"]
    print(f"fast_p vs eager: {summary['fast_p']['eager']}")
    print(f"run directory: {run.root}")
    return 0
