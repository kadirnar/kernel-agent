"""On-disk layout of one optimisation run.

```
runs/<org>--<name>/<timestamp>/
  run.json              run configuration (model card, workload spec, options) and the
                        sha256 of every truth file (``truth``, see truth.py)
  toolchain.json        GPU + compiler + backend availability
  harness.py            (optional) agent-written workload for unusual models
  baseline.json         end-to-end baseline latency + reference outputs summary
  profile/              profile.json, summary.md, kernels.txt
  plan.json             planner output: targets + model-level transforms
  .truth/               what the evaluator trusts (read-only, hashed): baseline_output.pt,
                        baseline_output_holdout.pt (held-out input),
                        baseline_output_natural.pt (natural-length run, stop condition),
                        baseline_output_perceptual.pt (--quality near-lossless),
                        captures/<id>.pt, targets/<id>/{results.jsonl,history/},
                        transforms/{results.jsonl,history/}
  targets/<id>/         spec.json, capture_inputs.pt (no outputs), reference_source.py,
                        workload_profile.md, candidates/*.py, NOTES.md, progress.png,
                        copies of history/ and results.jsonl for the agent
  transforms/           *.py model-level transforms (+ copies of history/, results.jsonl)
  results.tsv           experiment ledger: one row per evaluation (kernel, transform, integration)
  events.jsonl          phase changes, agent start/stop, evaluations
  progress.png  amdahl.png  integration.png  dashboard.html   charts (see charts.py)
  optimized/            exported winners + apply.py
  improve.json          `kernel-agent improve`: slices, re-integrations, rounds (+ improve.png)
  rounds/<n>/           improve --rounds: re-profile (baseline.json, profile/) + plan.json
  report.md             final report
```

Runs created before ``.truth/`` existed (no ``truth`` in ``run.json``) keep
``capture.pt``, ``results.jsonl`` and ``history/`` in ``targets/<id>/`` and
``baseline_output.pt`` in the root; the path methods below return those.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRUTH_DIR = ".truth"


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

    @property
    def ledger(self) -> Path:
        return self.root / "results.tsv"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def dashboard(self) -> Path:
        return self.root / "dashboard.html"

    def target(self, target_id: str) -> Path:
        return self.targets_dir / target_id

    def target_ids(self) -> list[str]:
        if not self.targets_dir.exists():
            return []
        return sorted(p.name for p in self.targets_dir.iterdir() if (p / "spec.json").exists())

    # ground truth (``.truth/`` in runs that have it, the old places otherwise)

    @property
    def truth_dir(self) -> Path:
        return self.root / TRUTH_DIR

    def sealed(self) -> bool:
        """Whether the run keeps its ground truth in ``.truth/`` (``run.json`` has ``truth``)."""
        return isinstance(self.load().get("truth"), dict)

    def baseline_output(self) -> Path:
        """Output of the baseline run that ``e2e`` compares against."""
        return (self.truth_dir if self.sealed() else self.root) / "baseline_output.pt"

    def baseline_output_holdout(self) -> Path:
        """Baseline output of the held-out input (``workloads/holdout.py``)."""
        return (self.truth_dir if self.sealed() else self.root) / "baseline_output_holdout.pt"

    def baseline_output_natural(self) -> Path:
        """Baseline of the natural-length run (stop condition, ``workloads/stopping.py``)."""
        return (self.truth_dir if self.sealed() else self.root) / "baseline_output_natural.pt"

    def baseline_output_perceptual(self) -> Path:
        """Baseline samples + scores of the perceptual gate (``workloads/perceptual.py``)."""
        return (self.truth_dir if self.sealed() else self.root) / "baseline_output_perceptual.pt"

    def capture_file(self, target_id: str) -> Path:
        """The full capture of a target (module, inputs, reference outputs)."""
        if self.sealed():
            return self.truth_dir / "captures" / f"{target_id}.pt"
        return self.target(target_id) / "capture.pt"

    def results_file(self, target_id: str | None = None) -> Path:
        """Evaluation records of a target (``None``: of the transforms)."""
        return self._evals(target_id) / "results.jsonl"

    def history_dir(self, target_id: str | None = None) -> Path:
        """Snapshots of the evaluated files of a target (``None``: of the transforms)."""
        return self._evals(target_id) / "history"

    def _evals(self, target_id: str | None) -> Path:
        base = self.truth_dir if self.sealed() else self.root
        return base / "targets" / target_id if target_id else base / "transforms"

    def load(self) -> dict[str, Any]:
        data = read_json(self.run_json, {})
        assert isinstance(data, dict)
        return data
