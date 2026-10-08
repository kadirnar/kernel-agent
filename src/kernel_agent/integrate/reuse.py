"""Content keys of integration measurements: what a re-integration may reuse (issue #93).

Every evaluation snapshots its files under a new name (``history/034_merge_..._cc6df165.py``
holds the same bytes as ``history/021_merge_..._cc6df165.py``), so the integration's reuse
cache matches a measurement by what it applied, not by file name. An item's key is the
sha256 of everything it loads (:func:`loaded_files`), a kernel's also of the ``spec.json``
fields and the region rewrite the worker applies it with (:func:`item_key`); a measurement's
key adds the context it ran in (the evaluator schema, the baseline, the A/B rounds) and the
keys of its A and B sets in order (:class:`Keys`). A file that cannot be read has no key,
and a measurement without a key is never reused.

An ``integration.json`` from before content keys has none; :func:`migrate` gives its
measurements the keys they would have had where the run proves what they ran with
(issue #108).
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import kernel_agent
from kernel_agent import ledger, region, truth
from kernel_agent.workspace import RunDir, read_json

#: The ``kernel_agent`` package: the modules an item imports from it are part of its key.
PACKAGE = Path(kernel_agent.__file__).resolve().parent
#: What a string literal may name for its file to count as loaded (a ``.cu`` next to a
#: kernel, a kernel snapshot a transform loads by path).
SOURCES = (".py", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".ptx")
#: The ``spec.json`` fields ``worker._kernel_patches`` applies a kernel with.
SPEC_FIELDS = ("module_class", "qualname_regex", "phase", "kind", "parent_class")


def digest(value: Any) -> str:
    """sha256 of a JSON value (keys sorted)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def loaded_files(path: Path, root: Path) -> dict[str, str]:
    """sha256 of a transform or kernel file (key ``""``) and of the files it loads besides
    the installed packages, by their place (relative to the run directory ``root``):

    * the modules it imports from its own directory, which is on ``sys.path`` when it is
      loaded (``patcher.load_transform``, ``evaluate.load_candidate_module``), and theirs;
    * the ``kernel_agent`` modules it or they import by name (the rest of kernel-agent is
      the evaluator, versioned by ``EVALUATOR_SCHEMA``);
    * the existing source files their string literals name (:data:`SOURCES`, relative to
      the importing file, its item or the run directory), followed like imports.

    Raises ``OSError`` when the file cannot be read."""
    path = Path(path).resolve()
    base, root = path.parent, Path(root).resolve()
    files = {"": truth.sha256_file(path)}
    todo, seen = [path], {path}
    while todo:
        current = todo.pop()
        for dep, follow in _loads(current, base, root):
            dep = dep.resolve()
            if dep in seen:
                continue
            seen.add(dep)
            files[_place(dep, root)] = truth.sha256_file(dep)
            if follow and dep.suffix == ".py":
                todo.append(dep)
    return files


def item_key(run: RunDir, kind: str, arg: str) -> str | None:
    """Content key of an integration item: a transform path, or a kernel ``target=path``
    with what the worker applies it with (``spec.json``, a region target's rewrite). None
    when one of its files cannot be read."""
    try:
        if kind != "kernel":
            return digest({"kind": kind, "files": loaded_files(Path(arg), run.root)})
        target_id, _, path = arg.partition("=")
        spec = read_json(run.target(target_id) / "spec.json", {}) or {}
        what: dict[str, Any] = {
            "kind": kind,
            "target": target_id,
            "files": loaded_files(Path(path), run.root),
            "spec": {k: spec.get(k) for k in SPEC_FIELDS},
            "methods": (spec.get("capture") or {}).get("method_instances"),
        }
        if region.is_region(spec):
            what["rewrite"] = loaded_files(region.verified_rewrite(run, target_id), run.root)
        return digest(what)
    except OSError:
        return None


class Keys:
    """Content keys of one integration's items and measurements (each item hashed once).
    ``context``: what every measurement depends on besides its items."""

    def __init__(self, run: RunDir, context: dict[str, Any]) -> None:
        self.run = run
        self.context = digest(context)
        self._items: dict[tuple[str, str], str | None] = {}

    def item(self, item: tuple[str, str]) -> str | None:
        if item not in self._items:
            self._items[item] = item_key(self.run, *item)
        return self._items[item]

    def arg(self, arg: str) -> str | None:
        """The key of an item given by its argument alone (``target=path``: a kernel)."""
        return self.item(as_item(arg))

    def step(self, a: list[tuple[str, str]], b: list[tuple[str, str]]) -> str | None:
        """Key of an A/B measurement of set B against set A (in their order: transforms
        apply in it); None when an item has none."""
        sets = [[self.item(i) for i in a], [self.item(i) for i in b]]
        if any(k is None for keys in sets for k in keys):
            return None
        return digest([self.context, *sets])

    def entry(self, h: dict[str, Any]) -> str | None:
        """The key the step of an ``integration.json`` history entry has now."""
        a_items = (h.get("ab") or {}).get("a_items") or []
        return self.step([as_item(x) for x in a_items], [as_item(x) for x in h.get("items") or []])


def as_item(arg: str) -> tuple[str, str]:
    """(kind, argument) of an integration item given by its argument (``target=path``: a
    kernel, else a transform's path)."""
    target, sep, _ = arg.partition("=")
    return ("kernel" if sep and "/" not in target else "transform", arg)


# ------------------------------------------------------------------ migration (issue #108)


@dataclass
class Migration:
    """The measurements of an ``integration.json`` from before content keys that
    :func:`migrate` keyed (``entries``), of its ``total``; ``why`` the others were not."""

    total: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)
    why: list[str] = field(default_factory=list)
    proof: str = ""  # what showed the context they ran in

    def note(self) -> str:
        """The log line ("" when there was nothing to migrate)."""
        if not self.total:
            return ""
        head = "integration.json predates content keys (#93): "
        if not self.entries:
            return head + f"none of its {self.total} measurements reused: " + "; ".join(self.why)
        line = head + f"{len(self.entries)} of {self.total} measurements migrated ({self.proof})"
        rest = self.total - len(self.entries)
        return line + (f"; {rest} not: " + "; ".join(self.why) if rest else "")


def migrate(
    previous: dict[str, Any],
    keys: Keys,
    *,
    schema: int,
    baseline_ms: float,
    ab_rounds: int,
    perceptual: bool,
    verified: Callable[[str], bool],
) -> Migration:
    """The history entries of ``previous`` (an ``integration.json`` without ``reuse_key``s)
    with the key :meth:`Keys.entry` gives them now, where the run proves that they ran with
    what that key covers; anything not proven is measured again:

    * the evaluator schema: every ``recheck`` record of the file carries
      ``evaluator_schema`` (the integration that wrote it stamped them), equal to
      ``schema``; a file without re-check records proves none;
    * the baseline: its ``baseline_ms`` is ``baseline_ms``. ``analyze`` measures the latency
      anew and seals it together with ``baseline.json`` and the baseline outputs
      (``Truth.seal_baseline``), so the digests the worker verifies are today's too; its
      measurements carry ``metrics.perceptual`` exactly when the quality mode has the
      perceptual gate (near-lossless, relaxed: ``perceptual``);
    * the A/B rounds: every paired A/B with timed rounds ran ``ab_rounds`` of them;
    * each item: its file is still the snapshot a verified record of the run evaluated
      (``verified``: the sha256 the record holds; the integration measured only such
      snapshots), and it loads nothing else. A ``kernel_agent`` module it imports may have
      changed with kernel-agent since (an upgrade is what made the file pre-#93), and its
      content then is not recorded; a region kernel's rewrite neither. A kernel's
      ``spec.json`` fields are today's: kernel-agent writes them once, when it captures the
      target."""
    history = previous.get("history") or []
    out = Migration()
    if not history or any(h.get("reuse_key") for h in history):
        return out  # nothing, or a file with content keys
    out.total = len(history)
    rechecks = previous.get("recheck") or []
    schemas = {r.get("evaluator_schema") for r in rechecks}
    rounds = {
        ab.get("rounds")
        for h in history
        if (ab := h.get("ab") or {}).get("mode") == "paired" and ab.get("a_ms")
    }
    modes = {
        "perceptual" in h["metrics"]
        for h in history
        if h.get("status") == "ok" and isinstance(h.get("metrics"), dict)
    }
    if not rechecks or None in schemas:
        out.why.append("its evaluator schema is not recorded (no re-check records)")
    elif schemas != {schema}:
        out.why.append(f"measured by evaluator schema {sorted(schemas)} (now {schema})")
    if previous.get("baseline_ms") != baseline_ms:
        then = previous.get("baseline_ms")
        out.why.append(f"measured against another baseline ({then} ms, now {baseline_ms} ms)")
    if rounds - {ab_rounds}:
        ran = ", ".join(map(str, sorted(rounds, key=str)))
        out.why.append(f"its paired A/B steps ran {ran} rounds (now --ab-rounds {ab_rounds})")
    if modes - {perceptual}:
        out.why.append("measured in another --quality mode")
    if out.why:
        return out
    out.proof = (
        f"evaluator schema {schema} from its re-checks, baseline {baseline_ms:.2f} ms, "
        f"{ab_rounds} A/B rounds, the same snapshots"
    )
    problems: dict[str, str | None] = {}

    def problem(arg: str) -> str | None:
        """Why the content of an item then is not proven (None: it is)."""
        if arg not in problems:
            problems[arg] = _unproven(keys.run, arg, verified)
        return problems[arg]

    missed: dict[str, set[str]] = {}  # why -> labels of the items
    for h in history:
        args = [*((h.get("ab") or {}).get("a_items") or []), *(h.get("items") or [])]
        bad = [(why, arg) for arg in args if (why := problem(arg)) is not None]
        for why, arg in bad:
            missed.setdefault(why, set()).add(ledger.item_label(arg))
        if not bad and (key := keys.entry(h)) is not None:
            out.entries.append({**h, "reuse_key": key})
    out.why = [f"{', '.join(sorted(labels))}: {why}" for why, labels in missed.items()]
    return out


def _unproven(run: RunDir, arg: str, verified: Callable[[str], bool]) -> str | None:
    """Why a migrated measurement of an item cannot count as one of its content today."""
    kind, _ = as_item(arg)
    path = Path(arg.partition("=")[2] if kind == "kernel" else arg)
    if not verified(arg):
        return "not a verified snapshot of the run (missing, changed or no record)"
    try:
        files = loaded_files(path, run.root)
    except OSError:
        return "cannot be read"
    if others := sorted(f for f in files if f):
        return f"loads {', '.join(others)}, whose content then is not recorded"
    if kind == "kernel":
        spec = read_json(run.target(arg.partition("=")[0]) / "spec.json", {}) or {}
        if region.is_region(spec):
            return "a region kernel, applied with a rewrite whose content then is not recorded"
    return None


# ------------------------------------------------------------------ what a file loads


def _loads(path: Path, base: Path, root: Path) -> list[tuple[Path, bool]]:
    """(file, follow its own loads) of what ``path`` imports or names (see
    :func:`loaded_files`); nothing for a file that does not parse."""
    try:
        tree = ast.parse(path.read_bytes())
    except (SyntaxError, ValueError):
        return []
    dirs = list(dict.fromkeys([path.parent, base]))
    out: list[tuple[Path, bool]] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out += [(f, True) for f in _named_file(node.value, [*dirs, root])]
        for name in names:
            out += _module_files(name, dirs)
    return out


def _module_files(name: str, dirs: list[Path]) -> list[tuple[Path, bool]]:
    """The files of an imported module: a ``kernel_agent`` module's own file (not
    followed), else the module or package of its first name in one of ``dirs`` (followed)."""
    parts = name.split(".")
    if parts[0] == PACKAGE.name:
        stem = PACKAGE.parent.joinpath(*parts)
        found = [p for p in (stem.with_suffix(".py"), stem / "__init__.py") if p.is_file()]
        return [(p, False) for p in found[:1]]
    for d in dirs:
        if (module := d / f"{parts[0]}.py").is_file():
            return [(module, True)]
        if (package := d / parts[0]).is_dir():  # a package, or a namespace package
            return [(p, True) for p in sorted(package.rglob("*.py"))]
    return []


def _named_file(text: str, dirs: list[Path]) -> list[Path]:
    """The existing source file a string literal names, if any."""
    if not text.endswith(SOURCES) or "\n" in text or len(text) > 1024:
        return []
    named = Path(text)
    for p in [named] if named.is_absolute() else [d / named for d in dirs]:
        if p.is_file():
            return [p]
    return []


def _place(path: Path, root: Path) -> str:
    """Where a loaded file is: relative to the run directory or the package's parent."""
    for parent in (root, PACKAGE.parent):
        if path.is_relative_to(parent):
            return str(path.relative_to(parent))
    return str(path)
