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
    do_transforms: bool = True
    allow_harness_agent: bool = True

    claude_model: str = DEFAULT_MODEL
    effort: str | None = "high"
    max_turns_per_agent: int = 120
    budget_usd_per_agent: float | None = None
    permission_mode: str = "bypassPermissions"
    allow_web: bool = True
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
