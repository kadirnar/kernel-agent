"""``kernel-agent improve --dry-run``: a simulated model, agents and GPU worker.

No GPU and no Claude. The simulated agents write the files a real run has
(candidates, snapshots, ``NOTES.md`` with open ideas, transforms) and record
every evaluation through the same code as the evaluation tools
(``record_candidate``, ``record_e2e_result``), so the ledger, the budget
advice, the charts, the dashboard and the report are exercised end to end.
The simulated worker runs the integration's end-to-end measurements, the
re-profile of a round and the capture of new targets.

The model is a Qwen3-0.6B decode workload (1532 ms baseline, launch bound).
Each kernel target has a hidden ceiling that it approaches in noisy,
diminishing steps, with failures (build errors, wrong results, timeouts) and
regressions on the way, so targets plateau. Some targets report a
speed-of-light estimate (``pct_of_sol``), others do not; one target only shows
up in the re-profile of round 2. The systems agent finds a static cache and a
CUDA graph, then plateaus; the graph is incompatible with the MLP kernel, which
the measured integration catches.

Every draw is seeded by (seed, arm, evaluation index), so a dry run is
reproducible and a restarted one continues like the original. Time is
simulated as well (``ledger.clock``, :class:`SimBudget`), so the charts show
hours of work and ``--max-hours`` stops the loop in simulated hours.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import random
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel_agent import ledger, program, truth
from kernel_agent.agent.runner import AgentResult
from kernel_agent.agent.tools import record_candidate, record_e2e_result, snapshot
from kernel_agent.budget import Budget
from kernel_agent.config import OptimizeConfig
from kernel_agent.kernels.roofline import sol_signal
from kernel_agent.scheduler import class_shares, snapshot_record, systems_rows
from kernel_agent.workspace import RunDir, read_json, read_jsonl, write_json

if TYPE_CHECKING:
    from kernel_agent.orchestrator import Orchestrator

BASELINE_MS = 1532.4
HOOK_OVERHEAD = 2086.0 / BASELINE_MS  # profiled (hooked) time / real time
GPU_BUSY = 0.453
KERNEL_EFFICIENCY = 0.9  # share of a module's saving that shows up end to end

HEADERS = {
    "triton": "import torch\nimport triton\nimport triton.language as tl\n",
    "cuda": "import torch\nfrom torch.utils.cpp_extension import load_inline\n",
    "cute": "import cutlass\nimport cutlass.cute as cute\nimport torch\n",
}

# (class, inclusive ms, instances, calls per run, leaf)
CLASSES: list[tuple[str, float, int, int, bool]] = [
    ("Qwen3ForCausalLM", 2086.0, 1, 128, False),
    ("Qwen3Model", 2010.0, 1, 128, False),
    ("Qwen3DecoderLayer", 1931.0, 28, 3584, False),
    ("Linear", 905.0, 197, 25_216, True),
    ("Qwen3Attention", 856.0, 28, 3584, False),
    ("Qwen3MLP", 588.0, 28, 3584, False),
    ("Qwen3RMSNorm", 321.0, 113, 14_464, True),
    ("SiLU", 54.0, 28, 3584, True),
    ("Qwen3RotaryEmbedding", 42.0, 1, 128, False),
    ("Embedding", 6.0, 1, 128, True),
]
CONTAINERS = ("Qwen3ForCausalLM", "Qwen3Model", "Qwen3DecoderLayer")


@dataclass(frozen=True)
class SimTarget:
    id: str
    cls: str
    ceiling: float  # module speedup the target approaches
    p_fail: float  # chance that an evaluation fails
    sol_at: float | None  # module speedup at 100 % of the speed of light (None: no estimate)
    backends: tuple[str, ...]
    hypotheses: tuple[str, ...]
    round: int = 1  # the round whose plan has it


TARGETS = (
    SimTarget(
        "attn",
        "Qwen3Attention",
        2.3,
        0.25,
        2.65,
        ("triton", "cuda"),
        (
            "fuse q/k RMSNorm and RoPE into one kernel before SDPA",
            "write K/V straight into the cache slot",
            "split-K flash-decode for the 1-token decode case",
            "cp.async double-buffered KV tiles",
            "pre-pack q/k/v weights in build(): one GEMM instead of three",
            "fold the o_proj epilogue into the attention kernel",
            "fp32 accumulation in registers, no shared-memory reduction",
            "persistent kernel looping over heads",
            "separate static-shape prefill and decode kernels",
            "bf16x8 vector loads for K and V",
        ),
    ),
    SimTarget(
        "mlp",
        "Qwen3MLP",
        1.8,
        0.2,
        1.95,
        ("cuda", "triton"),
        (
            "merge gate/up into one pre-packed GEMM with a fused SiLU*mul",
            "GEMV for M=1 decode: a warp per 8 rows, weights streamed once",
            "split-K for the down_proj GEMV with an fp32 scratch reduction",
            "128-bit loads and an L2 prefetch hint on the weights",
            "persistent CTAs, one per SM",
            "cuBLAS for prefill, the custom GEMV only for M<=4",
            "unroll the K loop by 4",
            "overlap the gate/up GEMV with the SiLU",
        ),
    ),
    SimTarget(
        "rmsnorm",
        "Qwen3RMSNorm",
        1.9,
        0.3,
        None,
        ("cuda", "cute"),
        (
            "warp-per-row RMSNorm, bf16x8 loads, fp32 accumulation",
            "two rows per warp",
            "skip .contiguous() and reuse the output buffer",
            "one launch for q_norm and k_norm",
            "__launch_bounds__(256), no shared memory",
            "CuTe DSL with the TVM-FFI calling convention",
        ),
    ),
    SimTarget(
        "rope",
        "Qwen3RotaryEmbedding",
        2.3,
        0.2,
        None,
        ("triton", "cuda"),
        (
            "precompute the cos/sin table once, gather by position",
            "load_inline gather kernel, one thread per pair",
            "vectorised 2×bf16 gather",
            "return views of the table instead of copies",
        ),
        round=2,
    ),
)

# (file stem, hypothesis, family, e2e speedup alone or a failure status). Transforms of
# one family do not stack (a static cache is part of the CUDA-graph transforms).
SYSTEM_IDEAS: tuple[tuple[str, str, str, float | str], ...] = (
    ("static_cache", "static KV cache, no per-step reallocation", "graph", 1.10),
    ("cuda_graph_decode", "static cache + CUDA graph for the decode step", "graph", 1.52),
    ("sdpa_flash", "force the flash SDPA backend for prefill", "sdpa", 1.004),
    ("no_item_sync", "drop the per-step .item() sync in the stop check", "sync", "patch_error"),
    ("cuda_graph_v2", "CUDA graph + pinned sampling buffers, no host sync", "graph", 1.58),
    ("merged_qkv", "merge q/k/v projections into one Linear", "fusion", 1.02),
)
SYSTEM_TWEAKS = (
    "share one CUDA-graph memory pool across prompt buckets",
    "torch.compile the sampler",
    "pre-allocate the logits buffer",
    "prefill in two chunks to overlap with graph capture",
    "pin the KV cache layout to [layer, head, pos, dim]",
)
FAMILIES = {stem: family for stem, _, family, _ in SYSTEM_IDEAS}
INCOMPATIBLE = {("graph", "mlp"): "greedy tokens diverge at step 41"}


def _rng(seed: int, *key: Any) -> random.Random:
    return random.Random("|".join(map(str, (seed, *key))))


def sim_target(spec: dict[str, Any]) -> SimTarget:
    """The simulated behaviour of a target (made up from its id when it is not in TARGETS)."""
    for sim in TARGETS:
        if sim.cls == spec.get("module_class"):
            return sim
    rng = _rng(0, "target", spec.get("id"))
    return SimTarget(
        str(spec.get("id")),
        str(spec.get("module_class")),
        rng.uniform(1.4, 2.4),
        rng.uniform(0.15, 0.35),
        None,
        tuple(spec.get("backends") or ("triton",)),
        ("fuse the elementwise ops into one kernel", "vectorised loads", "tune the block size"),
    )


# ------------------------------------------------------------------ clock and budget


class SimClock:
    """Simulated wall clock (seconds since the epoch)."""

    def __init__(self, start: float) -> None:
        self.t0 = self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


@dataclass
class SimBudget(Budget):
    """A :class:`Budget` whose time is the simulated clock's."""

    clock: SimClock | None = None

    @classmethod
    def of(cls, budget: Budget, clock: SimClock) -> SimBudget:
        fields = {f.name: getattr(budget, f.name) for f in dataclasses.fields(Budget)}
        return cls(**fields, clock=clock)

    def elapsed_s(self) -> float:
        assert self.clock is not None
        return self.clock.now() - self.clock.t0


class SimToolchain:
    gpu = None
    backends = {b: True for b in ("cuda", "triton", "cute", "tilelang", "nvrtc")}
    env: dict[str, str] = {}

    def summary(self) -> str:
        return "GPU: simulated (kernel-agent improve --dry-run)"


# ------------------------------------------------------------------ the run


def _repo_id(ref: str) -> str:
    ref = re.sub(r"^https?://huggingface\.co/", "", ref.strip()).strip("/")
    return ref.split("@")[0] or "Qwen/Qwen3-0.6B"


def _profile(scale: dict[str, float] | None = None, busy: float = GPU_BUSY) -> dict[str, Any]:
    """Module profile; ``scale`` maps a class to a factor on its time (0: not in the model)."""
    classes: list[dict[str, Any]] = []
    for cls, ms, instances, calls, leaf in CLASSES:
        factor = 1.0 if scale is None else scale.get(cls, scale.get("*", 1.0))
        if factor <= 0:
            continue
        classes.append(
            {
                "root": "model",
                "cls": cls,
                "module_path": f"transformers.models.qwen3.modeling_qwen3.{cls}",
                "source_file": "/site-packages/transformers/models/qwen3/modeling_qwen3.py",
                "instances": instances,
                "calls": calls,
                "inclusive_ms": round(ms * factor, 3),
                "self_ms": round(ms * factor * 0.2, 3),
                "is_leaf": leaf,
                "example_qualname": f"model.{cls}",
                "signatures": [],
            }
        )
    return {
        "hooked_wall_ms": round(classes[0]["inclusive_ms"] * 1.05, 1),
        "classes": classes,
        "kernel_view": {"gpu_busy_fraction": round(busy, 3)},
    }


def _summary(profile: dict[str, Any], ms: float) -> str:
    rows = "\n".join(
        f"| {c['cls']} | {c['instances']} | {c['calls']} | {c['inclusive_ms']:.1f} |"
        for c in profile["classes"]
    )
    busy = profile["kernel_view"]["gpu_busy_fraction"]
    return (
        f"# Profile summary (simulated)\n\nwall {ms:.1f} ms per run, GPU busy {busy:.0%}\n\n"
        f"| class | instances | calls | inclusive ms |\n|---|---|---|---|\n{rows}\n"
    )


def _target_spec(sim: SimTarget) -> dict[str, Any]:
    return {
        "id": sim.id,
        "module_class": sim.cls,
        "qualname": None,
        "why": f"{sim.cls} is hot in decode",
        "approach": sim.hypotheses[0],
        "backends": list(sim.backends),
    }


def _capture_info(sim: SimTarget) -> dict[str, Any]:
    return {
        "qualname": f"model.layers.0.{sim.id}",
        "cases": [
            {"signature": "a0[1, 1, 1024]:bfloat16", "count": 127},
            {"signature": "a0[1, 512, 1024]:bfloat16", "count": 1},
        ],
    }


def _write_target(run: RunDir, spec: dict[str, Any]) -> None:
    target_dir = run.target(spec["id"])
    (target_dir / "candidates").mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "spec.json", spec)
    (target_dir / "reference_source.py").write_text(f"# {spec['module_class']} (simulated)\n")
    (target_dir / "NOTES.md").touch()


def create_run(cfg: OptimizeConfig, seed: int = 0) -> RunDir:
    """A simulated run with analyze, plan and capture done (no GPU, no Claude, no network)."""
    repo = _repo_id(cfg.model_ref)
    run = RunDir.create(cfg.runs_dir, repo)
    t0 = time.time()
    profile = _profile()
    plan: dict[str, Any] = {
        "analysis": "Decode is launch bound (45 % GPU busy): fuse the small ops of attention, "
        "MLP and norms; a static cache + CUDA graph for the decode step.",
        "targets": [_target_spec(sim) for sim in TARGETS if sim.round == 1],
        "transforms": [
            {"id": "cuda_graph_decode", "idea": "CUDA graph for decode", "why": "launch bound"}
        ],
    }
    write_json(
        run.run_json,
        {
            "card": {
                "repo_id": repo,
                "revision": "main",
                "modality": "llm",
                "architectures": ["Qwen3ForCausalLM"],
                "params": 596_049_920,
                "size_gb": 1.4,
            },
            "workload": {"repo_id": repo, "modality": "llm", "dtype": cfg.dtype, "options": {}},
            "config": cfg.to_dict(),
            "phases": {p: {"done": True} for p in ("analyze", "plan", "capture")},
            "created": ledger.stamp(t0),
            "dry_run": {"seed": seed},
            "truth": truth.new_section(),
        },
    )
    write_json(run.toolchain_json, {"gpu": {"name": "simulated", "arch": "sm_120"}})
    write_json(
        run.baseline_json,
        {
            "workload": "llm: prompt 512 tokens, 128 new tokens, batch 1 (simulated)",
            "median_ms": BASELINE_MS,
            "times_ms": [1529.8, 1532.4, 1536.1],
            "deterministic": True,
        },
    )
    truth.of(run).seal_baseline(BASELINE_MS)
    write_json(run.profile_dir / "profile.json", profile)
    (run.profile_dir / "summary.md").write_text(_summary(profile, BASELINE_MS))
    write_json(run.plan_json, plan)
    for spec in plan["targets"]:
        _write_target(run, {**spec, "capture": _capture_info(sim_target(spec))})
    run.transforms_dir.mkdir(parents=True, exist_ok=True)
    program.install(run)
    for phase, a, b in (("analyze", 0.0, 4.5), ("plan", 4.5, 7.0), ("capture", 7.0, 9.0)):
        ledger.event(run, "phase_start", when=t0 + a * 60, phase=phase)
        ledger.event(run, "phase_done", when=t0 + b * 60, phase=phase)
    return run


# ------------------------------------------------------------------ the world


class World:
    """Simulated agents and worker for one run; :meth:`installed` plugs them into ``orch``."""

    def __init__(self, orch: Orchestrator, hook: Callable[[str, int], None] | None = None) -> None:
        self.orch = orch
        self.run = orch.run
        self.seed = int((self.run.load().get("dry_run") or {}).get("seed", 0))
        last = [float(e["ts"]) for e in ledger.events(self.run) if "ts" in e]
        self.clock = SimClock(max(last, default=time.time()) + 30)
        self.hook = hook  # called after every simulated evaluation: (agent, evals so far)
        self.sessions: list[dict[str, Any]] = []
        profile = read_json(self.run.profile_dir / "profile.json", {}) or {}
        self.shares = class_shares(profile)

    @contextmanager
    def installed(self) -> Iterator[World]:
        orch = self.orch
        saved = (orch.agent_runner, orch.worker, orch.budget, orch.tc, ledger.clock)
        orch.agent_runner, orch.worker = self.run_agent, self.worker
        orch.budget = SimBudget.of(orch.budget, self.clock)
        orch.tc = SimToolchain()  # type: ignore[assignment]
        ledger.clock = self.clock.now
        try:
            yield self
        finally:
            orch.agent_runner, orch.worker, orch.budget, orch.tc, ledger.clock = saved

    def ref_ms(self, cls: str | None) -> float:
        share, _ = self.shares.get(str(cls), (0.0, 1))
        return share * BASELINE_MS

    # -------------------------------------------------------- agents

    async def run_agent(
        self,
        name: str,
        *,
        prompt: str,
        system_append: str,
        cwd: Path,
        result: AgentResult | None = None,
        **_: Any,
    ) -> AgentResult:
        result = result or AgentResult(name=name)
        self.sessions.append({"name": name, "prompt": prompt, "system": system_append})
        result.session_id = f"dry-{name}-{len(self.sessions)}"
        start = self.clock.now()
        rng = _rng(self.seed, "session", name, len(ledger.rows(self.run)))
        self.clock.advance(rng.uniform(40, 90))  # reading the digest and the files
        evals = 0
        if name == "planner":
            result.structured = self._plan(Path(cwd))
            result.cost_usd = 0.4
        elif name == "systems":
            evals = self._systems()
            result.cost_usd = 0.3 + 0.5 * evals * rng.uniform(0.8, 1.2)
        elif name.startswith("kernel-"):
            evals = self._kernel(name.removeprefix("kernel-"))
            result.cost_usd = 0.25 + 0.35 * evals * rng.uniform(0.8, 1.2)
        result.tool_calls = {"evaluate": evals} if evals else {}
        result.turns = 4 + 5 * evals
        result.seconds = self.clock.now() - start
        result.text = f"simulated session: {evals} evaluations"
        await asyncio.sleep(0)
        return result

    def _advice(
        self,
        agent: str,
        results: Path,
        evals: int | None,
        rng: random.Random,
        pct_of_sol: float | None = None,
    ) -> bool:
        """Whether the simulated agent stops after this evaluation (it follows the advice)."""
        budget = self.orch.budget
        ok_key = "passed" if agent == "systems" else "correct"
        feedback = budget.feedback(agent, results, evals, ok_key=ok_key, pct_of_sol=pct_of_sol)
        advice = feedback["advice"]
        if advice == "stop" or budget.exhausted():
            return True
        return advice == "consider_stopping" and rng.random() < 0.5

    def _kernel(self, target_id: str) -> int:
        target_dir = self.run.target(target_id)
        spec = read_json(target_dir / "spec.json", {}) or {}
        sim = sim_target(spec)
        ref_ms = self.ref_ms(sim.cls) or 10.0
        instances = self.shares.get(sim.cls, (0.0, 1))[1]
        used = 0
        while True:
            rows = [r for r in ledger.rows(self.run) if r["target"] == target_id]
            k = len(rows)
            rng = _rng(self.seed, "kernel", target_id, k)
            kept = [r for r in rows if r["status"] == ledger.KEEP]
            best = ledger.best_kept(rows)
            backend = sim.backends[(k // 5) % len(sim.backends)]
            hypothesis = sim.hypotheses[k % len(sim.hypotheses)]
            if k >= len(sim.hypotheses):
                hypothesis += f" (variant {k // len(sim.hypotheses) + 1})"
            outcome = self._kernel_outcome(sim, best, rng)
            self.clock.advance(rng.uniform(150, 330))
            src = target_dir / "candidates" / f"{backend}_v{k + 1}.py"
            header = HEADERS.get(backend, "import torch\n")
            src.write_text(
                f'{header}\n"""{hypothesis}"""\n\n\ndef build(reference):\n    return reference\n'
            )
            snap = snapshot(self.run, src, target_id)
            result = kernel_result(outcome, ref_ms, instances, sim)
            _, row = record_candidate(
                self.run,
                target_id,
                src,
                snap,
                result,
                hypothesis=hypothesis,
                parent=f"history/{kept[-1]['snapshot']}" if kept else None,
                eval_s=round(rng.uniform(25, 70), 1),
                when=self.clock.now(),
            )
            used += 1
            _note(target_dir / "NOTES.md", row, sim.hypotheses[(k + 1) % len(sim.hypotheses) :])
            if self.hook:
                self.hook(f"kernel-{target_id}", used)
            results = self.run.results_file(target_id)
            evals = self.orch.budget.kernel_evals
            if self._advice(f"kernel-{target_id}", results, evals, rng, sol_signal(result)):
                return used

    @staticmethod
    def _kernel_outcome(sim: SimTarget, best: float, rng: random.Random) -> float | str:
        if rng.random() < sim.p_fail:
            return rng.choice(("build_error", "incorrect", "incorrect", "runtime_error", "timeout"))
        gap = max(sim.ceiling - best, 0.0)
        progress = gap / max(sim.ceiling - 1.0, 1e-6)  # 1 at the start, 0 at the ceiling
        if rng.random() < 0.3 + 0.5 * progress:
            return best + gap * rng.uniform(0.2, 0.6)
        return best * rng.uniform(0.84, 1.006)

    def _systems(self) -> int:
        used = 0
        while True:
            rows = systems_rows(ledger.rows(self.run))
            k = len(rows)
            rng = _rng(self.seed, "systems", k)
            best = max(
                [1.0, *(r["speedup"] for r in rows if r["correct"] and r["speedup"])],
            )
            if k < len(SYSTEM_IDEAS):
                stem, hypothesis, _, outcome = SYSTEM_IDEAS[k]
            else:
                j = k - len(SYSTEM_IDEAS)
                stem, hypothesis = f"tweak_{j + 1}", SYSTEM_TWEAKS[j % len(SYSTEM_TWEAKS)]
                outcome = best * rng.uniform(0.93, 1.008)
            self.clock.advance(rng.uniform(240, 420))
            src = self.run.transforms_dir / f"{stem}.py"
            src.write_text(f'"""{hypothesis}"""\n\n\ndef apply(workload):\n    pass\n')
            snap = snapshot(self.run, src)
            if isinstance(outcome, str):
                result: dict[str, Any] = {
                    "status": outcome,
                    "passed": False,
                    "error": f"{outcome}: simulated failure",
                }
            else:
                result = _e2e_result(BASELINE_MS / (outcome * rng.uniform(0.996, 1.004)))
            _, row = record_e2e_result(
                self.run,
                result,
                [snap],
                [],
                hypothesis=hypothesis,
                eval_s=round(rng.uniform(60, 110), 1),
                when=self.clock.now(),
            )
            used += 1
            _note(self.run.transforms_dir / "NOTES.md", row, SYSTEM_TWEAKS[k % 3 :][:2])
            if self.hook:
                self.hook("systems", used)
            results = self.run.results_file()
            if self._advice("systems", results, self.orch.budget.transform_evals, rng):
                return used

    def _plan(self, cwd: Path) -> dict[str, Any]:
        profile = read_json(cwd / "profile" / "profile.json", {}) or {}
        present = {c["cls"] for c in profile.get("classes", [])}
        taken = {
            (read_json(self.run.target(t) / "spec.json", {}) or {}).get("module_class")
            for t in self.run.target_ids()
        }
        targets = [
            _target_spec(sim) for sim in TARGETS if sim.cls in present and sim.cls not in taken
        ]
        self.clock.advance(120)
        return {
            "analysis": "Re-profile: attention, MLP and norms are fast now; RoPE is next.",
            "targets": targets,
            "transforms": [
                {
                    "id": "graph_prefill",
                    "idea": "capture prefill in a CUDA graph per prompt bucket",
                    "why": "prefill is launch bound now",
                }
            ],
        }

    # -------------------------------------------------------- worker

    def worker(self, run: RunDir, command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--kernel", action="append", default=[])
        parser.add_argument("--transform", action="append", default=[])
        parser.add_argument("--out-dir", type=Path)
        parser.add_argument("--target")
        ns, _ = parser.parse_known_args(list(args))
        if command == "e2e":
            return self._e2e(ns.kernel, ns.transform)
        if command == "analyze" and ns.out_dir:
            return self._reprofile(ns.out_dir, ns.kernel, ns.transform)
        if command == "capture":
            return self._capture(ns.target)
        return {"status": "error", "error": f"dry run: worker {command} {' '.join(args)}"}

    def _model(
        self, kernels: list[str], transforms: list[str]
    ) -> tuple[float, dict[str, float], dict[str, float], str | None]:
        """End-to-end ms, kernel speedups, transform factor per family, failure reason."""
        speedups: dict[str, float] = {}
        for item in kernels:
            target_id, _, path = item.partition("=")
            rec = snapshot_record(self.run, target_id, path) or {}
            speedups[target_id] = float(rec.get("speedup") or 1.0)
        families: dict[str, float] = {}
        records = read_jsonl(self.run.results_file())
        for path in transforms:
            name = Path(path).name
            rec = next((r for r in records if Path(r["transforms"][0]).name == name), {})
            stem = re.sub(r"^\d+_|_[0-9a-f]{8}$", "", Path(path).stem)
            family = FAMILIES.get(stem, "graph")
            families[family] = max(families.get(family, 1.0), float(rec.get("speedup") or 1.0))
        reason = next(
            (
                why
                for (fam, tid), why in INCOMPATIBLE.items()
                if fam in families and tid in speedups
            ),
            None,
        )
        ms = BASELINE_MS
        for target_id, s in speedups.items():
            spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
            ms -= self.ref_ms(spec.get("module_class")) * (1 - 1 / s) * KERNEL_EFFICIENCY
        if families:
            factors = sorted(families.values(), reverse=True)
            factor = factors[0]
            for f in factors[1:]:
                factor *= 1 + 0.15 * (f - 1)
            if speedups:  # the kernels already removed part of the launch overhead
                factor = 1 + (factor - 1) * 0.8
            ms /= factor
        return ms, speedups, families, reason

    def _e2e(self, kernels: list[str], transforms: list[str]) -> dict[str, Any]:
        ms, _, _, reason = self._model(kernels, transforms)
        key = "+".join(sorted(ledger.item_label(i) for i in [*kernels, *transforms]))
        rng = _rng(self.seed, "e2e", key, len(ledger.rows(self.run)))
        self.clock.advance(rng.uniform(65, 95))
        result = _e2e_result(ms * rng.uniform(0.997, 1.003))
        if reason:
            result.update(passed=False, reason=reason, metrics={"token_match": 0.32})
        return result

    def _reprofile(
        self, out_dir: Path, kernels: list[str], transforms: list[str]
    ) -> dict[str, Any]:
        ms, speedups, families, reason = self._model(kernels, transforms)
        if reason:
            return {"status": "error", "error": f"the optimised model fails: {reason}"}
        self.clock.advance(300)
        replaced = {
            (read_json(self.run.target(t) / "spec.json", {}) or {}).get("module_class")
            for t in speedups
        }
        factor = max(families.values(), default=1.0)
        scale: dict[str, float] = {cls: 0.0 for cls in replaced if cls}
        scale["*"] = 1 / factor
        for cls in CONTAINERS:
            scale[cls] = ms / BASELINE_MS
        profile = _profile(scale, busy=min(0.95, GPU_BUSY * factor))
        out = RunDir(out_dir)
        baseline = {
            "workload": "llm: prompt 512 tokens, 128 new tokens, batch 1 (simulated, optimised)",
            "median_ms": round(ms, 3),
            "times_ms": [round(ms * 0.998, 3), round(ms, 3), round(ms * 1.003, 3)],
            "deterministic": True,
        }
        write_json(out.baseline_json, baseline)
        write_json(out.profile_dir / "profile.json", profile)
        (out.profile_dir / "summary.md").write_text(_summary(profile, ms))
        return baseline

    def _capture(self, target_id: str) -> dict[str, Any]:
        spec = read_json(self.run.target(target_id) / "spec.json", {}) or {}
        info = _capture_info(sim_target(spec))
        _write_target(self.run, {**spec, "capture": info})
        self.clock.advance(45)
        return info


# ------------------------------------------------------------------ results


def kernel_result(
    outcome: float | str, ref_ms: float, instances: int, sim: SimTarget
) -> dict[str, Any]:
    """An ``evaluate_candidate`` result (``ref_ms``: the target's time per model run)."""
    if isinstance(outcome, str):
        result: dict[str, Any] = {"status": outcome, "correct": False}
        if outcome == "incorrect":
            result["cases"] = [
                {"signature": "a0[1, 1, 1024]:bfloat16", "ok": False, "max_abs_err": 0.31}
            ]
        else:
            result["error"] = f"{outcome}: simulated failure"
        return result
    s = outcome
    ref_w = ref_ms / max(instances, 1)  # per instance and run, over the captured cases
    cases = []
    for sig, count, frac, case_s in (
        ("a0[1, 1, 1024]:bfloat16", 127, 0.9, s),
        ("a0[1, 512, 1024]:bfloat16", 1, 0.1, s),
    ):
        ref = ref_w * frac / count
        cases.append(
            {
                "signature": sig,
                "calls_per_run": count,
                "ok": True,
                "max_abs_err": 0.0078,
                "ref_ms": round(ref, 6),
                "new_ms": round(ref / case_s, 6),
                "speedup": round(case_s, 3),
                "timing_spread": 0.006,
            }
        )
    result = {
        "status": "ok",
        "correct": True,
        "cases": cases,
        "speedup": round(s, 4),
        "est_saved_ms_per_run": round(ref_ms * (1 - 1 / s), 3),
        "ref_ms_weighted": round(ref_w, 5),
        "new_ms_weighted": round(ref_w / s, 5),
        "eval_seconds": 41.0,
    }
    if sim.sol_at:  # the evaluator's speed-of-light estimate (issue #9)
        result["pct_of_sol"] = round(100 * s / sim.sol_at, 1)
        result["bound"] = "memory"
    return result


def _e2e_result(ms: float) -> dict[str, Any]:
    return {
        "status": "ok",
        "passed": True,
        "reason": None,
        "metrics": {"token_match": 1.0, "logits_cosine": 0.9998},
        "median_ms": round(ms, 3),
        "times_ms": [round(ms * 0.997, 3), round(ms, 3), round(ms * 1.004, 3)],
        "baseline_ms": BASELINE_MS,
        "speedup": round(BASELINE_MS / ms, 4),
        "peak_mem_gb": 2.1,
        "patches": {},
    }


def _note(path: Path, row: dict[str, Any], ideas: tuple[str, ...] | list[str]) -> None:
    """Append the evaluation to NOTES.md and rewrite its ``## Open ideas`` section."""
    text = path.read_text() if path.exists() else ""
    log = text.split("\n## Open ideas", 1)[0].rstrip()
    speedup = "" if row["speedup"] is None else f" {row['speedup']:.3f}x"
    log += f"\n- exp {row['exp']}: {row['hypothesis']} → {row['status']}{speedup}"
    todo = "\n".join(f"- {idea}" for idea in list(ideas)[:3]) or "- (none)"
    path.write_text(f"{log.strip()}\n\n## Open ideas\n{todo}\n")
