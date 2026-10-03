"""On-disk layout of one optimisation run.

```
runs/<org>--<name>/<timestamp>/
  run.json              run configuration (model card, workload spec, options)
  toolchain.json        GPU + compiler + backend availability
  harness.py            (optional) agent-written workload for unusual models
  baseline.json         end-to-end baseline latency + reference outputs summary
  profile/              profile.json, summary.md, kernels.txt
  plan.json             planner output: targets + model-level transforms
  targets/<id>/         spec.json, capture.pt, candidates/*.py, results.jsonl, NOTES.md
  transforms/           *.py model-level transforms + results.jsonl
  optimized/            exported winners + apply.py
  report.md             final report
```
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class RunDir:
    root: Path

    @classmethod
    def create(cls, base: Path, repo_id: str) -> RunDir:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = base / slug(repo_id) / stamp
        root.mkdir(parents=True, exist_ok=False)
        return cls(root.resolve())

    @classmethod
    def latest(cls, base: Path, repo_id: str) -> RunDir | None:
        parent = base / slug(repo_id)
        runs = sorted(p for p in parent.glob("*") if p.is_dir()) if parent.exists() else []
        return cls(runs[-1].resolve()) if runs else None

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def toolchain_json(self) -> Path:
        return self.root / "toolchain.json"

    @property
    def harness(self) -> Path:
        return self.root / "harness.py"

    @property
    def baseline_json(self) -> Path:
        return self.root / "baseline.json"

    @property
    def profile_dir(self) -> Path:
        return self.root / "profile"

    @property
    def plan_json(self) -> Path:
        return self.root / "plan.json"

    @property
    def targets_dir(self) -> Path:
        return self.root / "targets"

    @property
    def transforms_dir(self) -> Path:
        return self.root / "transforms"

    @property
    def optimized_dir(self) -> Path:
        return self.root / "optimized"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    def target(self, target_id: str) -> Path:
        return self.targets_dir / target_id

    def target_ids(self) -> list[str]:
        if not self.targets_dir.exists():
            return []
        return sorted(p.name for p in self.targets_dir.iterdir() if (p / "spec.json").exists())

    def load(self) -> dict[str, Any]:
        data = read_json(self.run_json, {})
        assert isinstance(data, dict)
        return data
