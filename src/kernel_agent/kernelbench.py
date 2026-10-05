"""KernelBench problems as targets of a suite run (``kernel-agent bench-suite``, issue #21).

A KernelBench problem (github.com/ScalingIntelligence/KernelBench) is one file
``KernelBench/level<N>/<id>_<Name>.py`` defining ``class Model(nn.Module)``,
``get_inputs()`` and ``get_init_inputs()``. :func:`prepare` turns one into a target:

* the files come from the repository tarball, fetched at run time into the cache
  (:func:`fetch`, ``~/.cache/kernel-agent/kernelbench/<ref>/``), or from a local
  checkout (:func:`find_root`); the dataset is never vendored;
* sizes: current KernelBench inputs are sized for 80 GB GPUs (``19_ReLU`` takes a
  6.4 GB tensor). :func:`fit_sizes` halves the integer constants ``get_inputs`` reads
  (first those only it reads, then those it shares with ``get_init_inputs``; a value
  the model rejects is put back) until inputs + outputs, measured on the meta device,
  fit the budget; the new values are appended to the source as assignments, with
  the original ones in comments;
* the (scaled) source is copied to ``.truth/kernelbench/<module>.py`` (sealed) and
  imported from there with its ``nn.Module`` classes pickling by value (as region
  rewrites do, :mod:`kernel_agent.region`): the pickled ``Model`` of a capture loads in
  the evaluator's and the re-check's subprocesses without any path setup, and its
  source is covered by the capture's digest;
* the capture (:func:`profiling.capture.capture_calls`): ``Model(*get_init_inputs())``
  with weights drawn under :data:`SEEDS` [0], one timed case from ``get_inputs()``
  under the same seed and correctness-only cases (``count`` 0) from the other seeds.
"""

from __future__ import annotations

import re
import shutil
import tarfile
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent import truth
from kernel_agent.toolchain import CACHE_DIR
from kernel_agent.workspace import RunDir, write_json

TARBALL = "https://github.com/ScalingIntelligence/KernelBench/archive/{ref}.tar.gz"
#: Seed of the weights and the timed case, then of each correctness-only case.
SEEDS = (0, 1, 2)
_FILE = re.compile(r"^(\d+)_(.+)\.py$")
_LEVEL = re.compile(r"^level\d+$")


@dataclass(frozen=True)
class Problem:
    level: int
    pid: int
    name: str  # the file name without its id, e.g. "ReLU"
    path: Path

    @property
    def target_id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")[:40]
        return f"l{self.level}_{self.pid:03d}_{slug}"

    @property
    def module_name(self) -> str:
        return f"ka_kernelbench_{self.target_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "id": self.pid, "name": self.name, "file": self.path.name}


# ------------------------------------------------------------------ the problem files


def find_root(path: Path) -> Path | None:
    """The directory with the ``level<N>/`` folders: ``path`` itself, or ``KernelBench/``
    in it (a checkout of the repository)."""
    for root in (Path(path), Path(path) / "KernelBench"):
        if root.is_dir() and any(_LEVEL.match(p.name) for p in root.iterdir() if p.is_dir()):
            return root
    return None


def fetch(ref: str = "main", *, cache: Path | None = None, url: str | None = None) -> Path:
    """The problem files of KernelBench at ``ref`` (branch, tag or commit), downloaded once
    into ``cache``: only ``KernelBench/level<N>/*.py`` of the tarball is kept."""
    dest = (cache or CACHE_DIR / "kernelbench") / re.sub(r"[^\w.-]+", "_", ref)
    if (found := find_root(dest)) is not None:
        return found
    part = dest.with_name(dest.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    with (
        urllib.request.urlopen(url or TARBALL.format(ref=ref), timeout=120) as response,
        tarfile.open(fileobj=response, mode="r|gz") as tar,
    ):
        for member in tar:
            parts = Path(member.name).parts
            if not (member.isfile() and len(parts) == 4 and parts[1] == "KernelBench"):
                continue
            if not (_LEVEL.match(parts[2]) and _FILE.match(parts[3])):
                continue
            target = part / "KernelBench" / parts[2] / parts[3]
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is not None:
                target.write_bytes(source.read())
    if find_root(part) is None:
        shutil.rmtree(part, ignore_errors=True)
        raise RuntimeError(f"no KernelBench/level*/ problem files in the tarball of {ref!r}")
    if not dest.exists():  # another process may have won the race
        part.rename(dest)
    shutil.rmtree(part, ignore_errors=True)
    found = find_root(dest)
    assert found is not None
    return found


def problems(
    root: Path, level: int, *, n: int | None = None, ids: list[int] | None = None
) -> list[Problem]:
    """Problems of ``level`` by id: the given ``ids``, else the first ``n`` (None: all)."""
    folder = root / f"level{level}"
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} does not exist")
    found: dict[int, Problem] = {}
    for path in folder.glob("*.py"):
        if match := _FILE.match(path.name):
            pid = int(match[1])
            found[pid] = Problem(level, pid, match[2], path.resolve())
    if ids:
        missing = [i for i in ids if i not in found]
        if missing:
            raise KeyError(f"level {level} has no problem {', '.join(map(str, missing))}")
        return [found[i] for i in ids]
    ordered = [found[i] for i in sorted(found)]
    return ordered if n is None else ordered[:n]


# ------------------------------------------------------------------ module + sizes


def load_module(source: str, name: str, filename: str) -> types.ModuleType:
    """Import a problem's source as module ``name`` (its classes pickle by value)."""
    from kernel_agent.region import _exec

    module = _exec(source, name, filename)
    for attr in ("Model", "get_inputs", "get_init_inputs"):
        if not hasattr(module, attr):
            raise AttributeError(f"{filename} defines no {attr} (not a KernelBench problem)")
    return module


def _nbytes(value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.numel()) * value.element_size()
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    if isinstance(value, list | tuple):
        return sum(_nbytes(v) for v in value)
    return 0


def footprint(module: types.ModuleType) -> int:
    """Bytes of one call's inputs + outputs, on the meta device (nothing is allocated;
    an output the meta device cannot compute counts as large as the inputs)."""
    import torch

    with torch.device("meta"), torch.no_grad():
        inputs = list(module.get_inputs())
        model = module.Model(*module.get_init_inputs())
        try:
            output = model(*inputs)
        except Exception:
            output = inputs
    return _nbytes(inputs) + _nbytes(output)


def fit_sizes(module: types.ModuleType, max_bytes: int) -> tuple[dict[str, list[int]], int]:
    """Halve the size constants of ``module`` (see the module docstring) until
    :func:`footprint` fits ``max_bytes``; returns ``({name: [original, new]}, bytes)``."""
    reads = set(module.get_inputs.__code__.co_names)
    shared = reads & set(module.get_init_inputs.__code__.co_names)

    def sizes(names: set[str]) -> list[str]:
        values = vars(module)
        return sorted(n for n in names if type(values.get(n)) is int and values[n] > 1)

    tiers = [sizes(reads - shared), sizes(shared)]
    changed: dict[str, list[int]] = {}
    frozen: set[str] = set()
    size = footprint(module)
    while size > max_bytes:
        pool: list[str] = []
        for tier in tiers:  # the first tier with a constant left to halve
            pool = [n for n in tier if n not in frozen and getattr(module, n) > 1]
            if pool:
                break
        if not pool:
            break
        name = max(pool, key=lambda n: getattr(module, n))
        old = getattr(module, name)
        setattr(module, name, old // 2)
        try:
            size = footprint(module)
        except Exception:  # e.g. a channel count the model's groups no longer divide
            setattr(module, name, old)
            frozen.add(name)
            continue
        changed.setdefault(name, [old, old])[1] = old // 2
    return changed, size


def scaled_source(source: str, changed: dict[str, list[int]], max_mb: float) -> str:
    """``source`` with the scaled constants appended as assignments."""
    if not changed:
        return source
    lines = [
        "",
        f"# kernel-agent bench-suite: sizes scaled to fit {max_mb:g} MB of inputs + outputs",
        *(f"{name} = {new}  # KernelBench: {old}" for name, (old, new) in changed.items()),
    ]
    return source.rstrip("\n") + "\n" + "\n".join(lines) + "\n"


# ------------------------------------------------------------------ capture + target


def _to(value: Any, device: str) -> Any:
    import torch

    return value.to(device) if isinstance(value, torch.Tensor) else value


def build_capture(module: types.ModuleType, path: Path, *, device: str) -> list[dict[str, Any]]:
    """Capture ``Model`` (see the module docstring); returns the cases' summaries."""
    import torch

    from kernel_agent.profiling.capture import capture_calls

    torch.manual_seed(SEEDS[0])
    model = module.Model(*module.get_init_inputs()).to(device).eval()
    calls: list[tuple[Any, ...]] = []
    for k, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        inputs = tuple(_to(x, device) for x in module.get_inputs())
        calls.append((inputs, {}, 1 if k == 0 else 0))
    capture_calls(model, calls, path)
    data = torch.load(path, weights_only=False)
    for k, case in enumerate(data["cases"]):
        if k:  # another seed: checked by the evaluator, never timed
            case.update(
                correctness_only=True,
                variant=f"seed {SEEDS[k]}",
                signature=f"{case['signature']} [correctness only: seed {SEEDS[k]}]",
            )
    torch.save(data, path)
    keys = ("method", "signature", "count", "correctness_only")
    return [{k: c[k] for k in keys if k in c} for c in data["cases"]]


def prepare(
    run: RunDir,
    problem: Problem,
    keeper: truth.Truth,
    *,
    device: str,
    max_mb: float,
    backends: list[str],
) -> dict[str, Any]:
    """Make ``problem`` a target of ``run``: the sealed module copy and capture, and the
    target directory (``spec.json``, ``reference_source.py``, ``capture_inputs.pt``,
    ``NOTES.md``, ``candidates/``); returns the spec."""
    source = problem.path.read_text()
    module = load_module(source, problem.module_name, str(problem.path))
    changed, size = fit_sizes(module, int(max_mb * 2**20))
    final = scaled_source(source, changed, max_mb)
    sealed = truth.replace(run.truth_dir / "kernelbench" / f"{problem.module_name}.py")
    sealed.write_text(final)
    keeper.seal(sealed)
    module = load_module(final, problem.module_name, str(sealed))
    target_id = problem.target_id
    capture = truth.replace(run.capture_file(target_id))
    cases = build_capture(module, capture, device=device)
    keeper.seal(capture)
    target_dir = run.target(target_id)
    (target_dir / "candidates").mkdir(parents=True, exist_ok=True)
    truth.write_inputs_capture(capture, target_dir / "capture_inputs.pt")
    scaled = ", ".join(f"{n} {old} -> {new}" for n, (old, new) in changed.items())
    (target_dir / "reference_source.py").write_text(
        f"# KernelBench level {problem.level} problem {problem.pid}: {problem.path.name}\n"
        f"# module `{problem.module_name}` (the captured instance is its `Model`)\n"
        + (f"# sizes scaled to fit {max_mb:g} MB: {scaled}\n" if scaled else "")
        + "\n"
        + final
    )
    (target_dir / "NOTES.md").touch()
    (target_dir / "workload_profile.md").write_text(
        f"# Workload: KernelBench level {problem.level} problem {problem.pid}\n\n"
        f"* one `forward` call per run on `get_inputs()` at seed {SEEDS[0]}: the timed case\n"
        f"* seeds {', '.join(map(str, SEEDS[1:]))}: correctness-only cases (not timed)\n"
        f"* inputs + outputs of one call: {size / 2**20:.2f} MB\n"
    )
    spec = {
        "id": target_id,
        "module_class": "Model",
        "backends": backends,
        "why": f"KernelBench level {problem.level} problem {problem.pid} ({problem.name}): "
        "the whole module is the target",
        "approach": "beat the PyTorch reference on these shapes with your own kernels",
        "kernelbench": {**problem.to_dict(), "scaled": changed, "bytes_per_call": size},
        "capture": {
            "qualname": "Model",
            "methods": {"forward": 1},
            "method_instances": {"forward": 1},
            "cases": cases,
            "bytes": capture.stat().st_size,
        },
    }
    write_json(target_dir / "spec.json", spec)
    return spec
