"""Run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALL_BACKENDS = ("cuda", "triton", "cute", "tilelang", "nvrtc")
DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class OptimizeConfig:
    model_ref: str
    runs_dir: Path = Path("runs")
    modality: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    workload_options: dict[str, Any] = field(default_factory=dict)
    harness: str | None = None

    backends: list[str] = field(default_factory=lambda: list(ALL_BACKENDS))
    max_targets: int = 4
    evaluations_per_target: int = 12
    transform_evaluations: int = 6
    min_speedup: float = 1.03
    parallel: int = 1
    #: Isolated workers per target (workers.py): a count, "auto" (2 for targets with >= 20 %
    #: of the profile) or None (1); the target's evaluation budget is split across them.
    seeds_per_target: int | str | None = None
    reseed_workers: bool = False  # a second round of worker sessions from the 2 best snapshots
    do_transforms: bool = True
    #: The native arm of `improve` (native/engine.py, issue #134): "off", "plan" (only when
    #: the plan asks for it) or "on" (once every module arm has plateaued).
    native: str = "plan"
    native_minutes: float | None = None  # per native session (None: 3 × agent_minutes)
    native_evaluations: int = 6  # evaluations per native slice
    allow_harness_agent: bool = True
    #: analyze: also time a generic torch.compile when the workload has no
    #: reference_optimizations() hook (strong_baseline.py).
    compile_baseline: bool = False
    #: Integration (abtest.py): timed rounds of each paired A/B, and the acceptance rule:
    #: B wins >= ab_min_win_rate of them and the 95 % CI of its gain starts above ab_min_gain.
    ab_rounds: int = 8
    ab_min_win_rate: float = 0.8
    ab_min_gain: float = 0.01
    #: Integration: re-check every kernel on fresh inputs, reference and kernel timed in
    #: processes of their own (kernels/recheck.py); a kernel that fails it is refused.
    recheck: bool = True
    #: "exact" (numerics within rounding noise) or "near-lossless": numerics-changing
    #: optimisations pass when the perceptual quality stays within the noise of eager
    #: (workloads/perceptual.py).
    quality: str = "exact"
    #: The target precisions the run allows (``--precisions``, precisions.py, issue #131);
    #: None: the default of ``quality`` (near-lossless: every reduced precision but 4-bit).
    precisions: list[str] | None = None

    claude_model: str = DEFAULT_MODEL
    effort: str | None = "high"
    max_turns_per_agent: int = 120
    budget_usd_per_agent: float | None = None
    permission_mode: str = "bypassPermissions"
    allow_web: bool = True
    #: Documentation lookups (agent/web.py, issue #125): extra hosts WebFetch may reach on
    #: top of web.DOMAINS, and the research dossier of a target before its first session.
    web_domains: list[str] = field(default_factory=list)
    dossier: bool = True
    #: How sessions authenticate (kernel_agent/agent/auth.py): "subscription" (the Claude
    #: Code login only), "api" (an API key / cloud provider only) or "auto" (either).
    auth: str = "auto"

    # Run budgets (kernel_agent/budget.py); None = unlimited.
    max_hours: float | None = None
    max_usd: float | None = None  # notional with --auth subscription
    max_sessions: int | None = None  # agent sessions started by this process
    agent_minutes: float | None = None
    budget_reserve: float = 0.15  # share of max_hours kept for integrate + report
    eval_timeout_s: float = 300.0  # per evaluate_candidate subprocess

    program: str | None = None  # program.md for the agents (kernel_agent/program.py)

    # Cross-run kernel library + lessons (kernel_agent/library.py).
    use_library: bool = True  # reuse prior winners and lessons, store this run's winners
    librarian: bool = True  # distil lessons after the report (a cheap agent)
    librarian_model: str | None = None  # None: claude_model
    librarian_effort: str | None = "low"

    hf_token: str | None = None
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["runs_dir"] = str(self.runs_dir)
        data.pop("hf_token", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizeConfig:
        data = dict(data)
        data["runs_dir"] = Path(data.get("runs_dir", "runs"))
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})
