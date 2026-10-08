"""Run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALL_BACKENDS = ("cuda", "triton", "cute", "tilelang", "nvrtc")
DEFAULT_MODEL = "claude-opus-5-5"
#: Other model ids for `--role-model ROLE=MODEL` (every role runs on DEFAULT_MODEL by default).
SONNET_MODEL = "claude-sonnet-5-5"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
#: A role's model or effort that is the session's: ``--claude-model`` / ``--effort``.
INHERIT = "inherit"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
#: The model of each role (#181, docs/MULTIAGENT.md §3.12.1; ``roles.py`` is the registry):
#: the creative work and the planning on ``--claude-model``, high-volume retrieval and
#: distillation on Sonnet, the critic's triage on Haiku and its escalation on Sonnet.
#: ``--role-model ROLE=MODEL`` changes one (``inherit``: back to ``--claude-model``).
ROLE_MODELS = {
    "planner": INHERIT,
    "kernel": INHERIT,
    "systems": INHERIT,
    "native": INHERIT,
    "research": INHERIT,
    "refactor": INHERIT,
    "harness": INHERIT,
    "dossier": INHERIT,
    "librarian": INHERIT,
    "critic": INHERIT,
    "critic-escalation": INHERIT,
    "doc-lookup": INHERIT,
    "profile-analyst": INHERIT,
    "compile-triage": INHERIT,
    "reviewer": INHERIT,
}
#: The effort of each role (``--role-effort ROLE=LEVEL``; ``inherit``: ``--effort``; None:
#: none set, the model's default). Every role inherits ``--effort`` (high) by default.
ROLE_EFFORTS: dict[str, str | None] = {role: INHERIT for role in ROLE_MODELS}
#: ``--quality`` (kernels/compare.py ``QUALITIES``, without importing torch) and the mode of
#: a new run (#175).
QUALITIES = ("exact", "near-lossless", "relaxed")
DEFAULT_QUALITY = "relaxed"
#: One line per quality mode for ``status`` (the numbers: kernels/compare.py, perceptual.py).
QUALITY_NOTES = {
    "exact": "numerics within rounding noise of eager",
    "near-lossless": "reduced precision within the noise of eager (per tensor cosine >= "
    "0.996, relative L2 <= 0.08, norm ±2 %)",
    "relaxed": "about twice near-lossless's error budgets (per tensor cosine >= 0.99, "
    "relative L2 <= 0.16, norm ±4 %; small measured perceptual drops pass)",
}


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
    #: Integration: import and apply the exported optimized/ in a fresh process outside the
    #: run directory (integrate/export.py, issue #171); a failure names the missing files.
    export_check: bool = True
    #: "exact" (numerics within rounding noise), "near-lossless" (numerics-changing
    #: optimisations pass when the perceptual quality stays within the noise of eager,
    #: workloads/perceptual.py) or "relaxed" (the default, #175: the same checks with about
    #: twice near-lossless's error budgets, kernels/compare.py). A run.json without it was an
    #: exact run (:meth:`from_dict`).
    quality: str = DEFAULT_QUALITY
    #: The target precisions the run allows (``--precisions``, precisions.py, issue #131);
    #: None: the default of ``quality`` (near-lossless, relaxed: every reduced precision but
    #: the opt-in 4-bit and fp8_kv).
    precisions: list[str] | None = None

    claude_model: str = DEFAULT_MODEL
    effort: str | None = "high"
    #: Model and effort per role (:data:`ROLE_MODELS`, :data:`ROLE_EFFORTS`; ``roles.py``).
    role_models: dict[str, str] = field(default_factory=lambda: dict(ROLE_MODELS))
    role_efforts: dict[str, str | None] = field(default_factory=lambda: dict(ROLE_EFFORTS))
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
    librarian: bool = True  # distil lessons after the report (a cheap agent: role_models)

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
        data.setdefault("quality", "exact")  # made before --quality: an exact run
        if "role_models" not in data:  # made before #181: every role on --claude-model
            data["role_models"] = {role: INHERIT for role in ROLE_MODELS}
            data["role_models"]["librarian"] = data.get("librarian_model") or INHERIT
            efforts = data["role_efforts"] = {role: INHERIT for role in ROLE_EFFORTS}
            efforts.update(dossier="low", librarian=data.get("librarian_effort", "low"))
        # roles added since the run was made get their defaults
        data["role_models"] = {**ROLE_MODELS, **data["role_models"]}
        data["role_efforts"] = {**ROLE_EFFORTS, **(data.get("role_efforts") or {})}
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})
