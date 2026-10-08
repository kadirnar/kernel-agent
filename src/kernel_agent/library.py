"""Cross-run kernel library and distilled lessons.

Without it everything a run learned is discarded with the run. The library keeps
the kernels that won, per GPU architecture, and short lessons distilled from the
agents' notes and ledgers, so later runs start from them (AccelOpt, KernelBlaster,
AdaExplore: slow→fast examples and validity rules; FlashInfer-Bench: a serialisable
definition + solution + evaluation per kernel)::

    ~/.cache/kernel-agent/library/            ($KERNEL_AGENT_LIBRARY overrides it)
      <sm_arch>/<module_class>/<entry-id>/    one kernel that won a run on this GPU architecture
        kernel.py     the evaluated snapshot
        spec.json     the target: module class, backends, approach, phase, capture info
        result.json   its evaluation: speedup and per-case numbers, % of SOL, backend
        NOTES.md      the kernel engineer's notes of that run
        entry.json    model repo, torch version, dates, signature summary (method +
                      shapes/dtypes of the captured cases), flags, precision (``exact``,
                      or e.g. ``fp8_weights``), and the sha256 of every file above
      lessons/<backend>.md  lessons/<module_family>.md   rules distilled by the librarian

* **Store** (:func:`store_run`, after every integration): the verified best kernel
  of every target that reached ``min_speedup`` (``module_winner``) and every kernel
  the integration accepted end to end (``accepted``). The entry id is
  ``<signature hash>-<code hash>``: the same kernel stored again for the same shapes
  updates its entry (``runs`` lists every run that stored it).
* **Reuse** (:func:`seed_target`, before a target's first agent session): up to
  :data:`MAX_PRIORS` entries of the same module class and GPU architecture whose
  entrypoints and dtypes cover the target's (and whose precision the target allows: a
  reduced-precision kernel serves only a target of that precision, and only in a run whose
  ``--precisions`` allow it: no FP4 prior unless 4-bit is asked for) are copied to
  ``candidates/prior_<entry-id>.py`` and evaluated like any candidate (verified
  capture, snapshot, ``results.jsonl``, a ledger row with the hypothesis ``prior
  winner from <repo>``), at no LLM cost. The engineer prompt lists them slow → fast
  with their numbers (:func:`prompt_note`), followed by the lessons for the target's
  backends and module family (at most :data:`LESSONS_CHARS` characters).
* **Lessons** (:func:`librarian_prompt`, :func:`write_lessons`): after the report a
  cheap agent turns the run's ``NOTES.md`` files and ledger rows into short validity
  rules and merges them into ``lessons/``, pruning duplicates and stale rules.
* **Claude Code memory** (:func:`cmd_import_memory`): agent sessions run without Claude
  Code's auto memory (``runner.SESSION_ENV``), so what they learn lands here, not in the
  user's ``~/.claude/projects/<project>/memory/``. ``kernel-agent library import-memory DIR``
  turns notes earlier sessions left there into lessons rules (a dry run unless ``--write``);
  the memory files are only read.
* **Safety.** Entries are code that will run. Every entry records the sha256 of its
  files; it is reused only when every digest matches and its ``sm_arch`` is this
  GPU's (other architectures' directories are never searched), and the reused
  kernel must pass the target's correctness checks like any candidate. The digests
  are tamper evidence, not a signature: like a run directory, the library is within
  reach of an agent's Bash tool.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent import ledger, truth, workers
from kernel_agent.budget import PRIOR_HYPOTHESIS
from kernel_agent.toolchain import CACHE_DIR
from kernel_agent.truth import TamperError, Truth, read_verified, sha256_bytes, sha256_file
from kernel_agent.workspace import RunDir, read_json, write_json

ENV = "KERNEL_AGENT_LIBRARY"
VERSION = 1
KERNEL = "kernel.py"
MAX_PRIORS = 3  # library entries evaluated per target before its first agent session
LESSONS_CHARS = 1500  # lessons in one engineer prompt
MAX_RULES = 20  # rules per lessons file
RULE_CHARS = 240
NOTES_CHARS = 20_000  # tail of NOTES.md kept per entry
MAX_RUNS = 20  # runs remembered per entry
LIBRARIAN_NOTES_CHARS = 3000  # tail of each target's NOTES.md in the librarian prompt
LIBRARIAN_ROWS = 40  # last ledger rows per target in the librarian prompt

#: Coarse module families for the lessons files (first match wins).
FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("norm", ("norm",)),
    ("attention", ("attention", "attn")),
    ("rope", ("rotary", "rope")),
    ("mlp", ("mlp", "feedforward", "ffn", "glu")),
    ("conv", ("conv",)),
    ("embedding", ("embed",)),
    ("activation", ("silu", "gelu", "relu", "snake", "activation", "tanh")),
    ("linear", ("linear", "proj")),
    ("block", ("layer", "block")),
)

LESSONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "rules": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["file", "rules"],
            },
        }
    },
    "required": ["lessons"],
}

_DTYPE = re.compile(r":([A-Za-z][A-Za-z0-9_]*)")
_SHAPE = re.compile(r"\[([0-9, ]*)\]")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,40}$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] library: {msg}", flush=True)


def root() -> Path:
    """The library directory: ``$KERNEL_AGENT_LIBRARY``, else ``<cache>/library``."""
    env = os.environ.get(ENV)
    return Path(env).expanduser() if env else CACHE_DIR / "library"


def lessons_dir() -> Path:
    return root() / "lessons"


def module_family(cls: str) -> str:
    """Coarse family of a module class for the lessons (``Qwen3RMSNorm`` → ``norm``)."""
    name = cls.lower()
    for family, words in FAMILIES:
        if any(w in name for w in words):
            return family
    return re.sub(r"[^a-z0-9_]+", "_", name).strip("_") or "module"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip(".") or "_"


# ------------------------------------------------------------------ signatures


def _method_of(case: dict[str, Any]) -> str:
    if case.get("method"):
        return str(case["method"])
    head, sep, _ = str(case.get("signature", "")).partition(": ")
    return head if sep and re.fullmatch(r"\w+", head) else "forward"


def signature_summary(capture_info: dict[str, Any]) -> dict[str, Any]:
    """Entrypoints, dtypes and case signatures of a target's capture (``spec["capture"]``)."""
    cases = [
        {
            "method": _method_of(c),
            "signature": str(c.get("signature", "")),
            "count": int(c.get("count") or 0),
        }
        for c in capture_info.get("cases") or []
    ]
    methods = {c["method"] for c in cases} | set(capture_info.get("method_instances") or {})
    out: dict[str, Any] = {
        "methods": sorted(methods),
        "dtypes": sorted({d for c in cases for d in _DTYPE.findall(str(c["signature"]))}),
        "cases": cases,
    }
    if capture_info.get("phase"):
        out["phase"] = capture_info["phase"]
    return out


def _dims(sig: dict[str, Any]) -> set[int]:
    """Last dimension of each case's first tensor (typically the hidden size)."""
    dims = set()
    for case in sig.get("cases") or []:
        match = _SHAPE.search(str(case.get("signature", "")))
        if match and match.group(1).strip():
            dims.add(int(match.group(1).split(",")[-1]))
    return dims


def compatible(entry_sig: dict[str, Any], target_sig: dict[str, Any], source: str = "") -> bool:
    """Whether a library kernel can serve a target: it implements every entrypoint the
    target needs besides ``forward`` (by its own capture or a ``def`` in ``source``) and was
    verified on the target's dtypes. Shapes may differ: kernels read sizes from the module."""
    needed = set(target_sig.get("methods") or []) - {"forward"}
    have = set(entry_sig.get("methods") or [])
    missing = [m for m in needed - have if not re.search(rf"\bdef {re.escape(m)}\b", source)]
    dtypes = set(target_sig.get("dtypes") or [])
    return not missing and dtypes <= set(entry_sig.get("dtypes") or [])


def closeness(entry_sig: dict[str, Any], target_sig: dict[str, Any]) -> tuple[int, int]:
    """(identical case signatures, shared hidden sizes): higher is closer."""
    same = {c["signature"] for c in target_sig.get("cases") or []} & {
        c["signature"] for c in entry_sig.get("cases") or []
    }
    return len(same), len(_dims(target_sig) & _dims(entry_sig))


def entry_id(signature: dict[str, Any], code: bytes) -> str:
    """``<signature hash>-<code hash>`` (8 hex digits each)."""
    key = json.dumps(
        {
            "methods": signature.get("methods") or [],
            "cases": sorted(c["signature"] for c in signature.get("cases") or []),
            "phase": signature.get("phase"),
        },
        sort_keys=True,
    )
    return f"{sha256_bytes(key.encode())[:8]}-{sha256_bytes(code)[:8]}"


# ------------------------------------------------------------------ entries


@dataclass(frozen=True)
class Entry:
    path: Path
    meta: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.meta.get("id") or self.path.name)

    @property
    def speedup(self) -> float:
        return float(self.meta.get("speedup") or 0.0)

    @property
    def kernel_sha256(self) -> str | None:
        digest = (self.meta.get("files") or {}).get(KERNEL)
        return str(digest) if digest else None

    def problem(self, arch: str | None = None) -> str | None:
        """Why the entry must not be reused (None: right architecture, every digest matches)."""
        if self.meta.get("broken"):
            return str(self.meta["broken"])
        if arch is not None and self.meta.get("sm_arch") != arch:
            return f"built for {self.meta.get('sm_arch')}, this GPU is {arch}"
        files = self.meta.get("files") or {}
        if KERNEL not in files:
            return f"no sha256 recorded for {KERNEL}"
        for name, digest in files.items():
            if Path(name).name != name or name.startswith("."):
                return f"bad file name {name!r} in entry.json"
            try:
                found = sha256_file(self.path / name)
            except OSError:
                return f"{name} is missing"
            if found != digest:
                return f"{name} was modified (sha256 {found[:12]}…, recorded {str(digest)[:12]}…)"
        return None

    def kernel(self) -> bytes:
        """The kernel source, read once and checked against its sha256 (:class:`TamperError`)."""
        digest = self.kernel_sha256
        if digest is None:
            raise TamperError(f"{self.path / KERNEL}: no sha256 recorded")
        return read_verified(self.path / KERNEL, digest)

    def source(self) -> str:
        """The kernel source without the digest check (for matching and display only)."""
        try:
            return (self.path / KERNEL).read_text(errors="replace")
        except OSError:
            return ""


def entries(
    arch: str | None = None, module_class: str | None = None, *, base: Path | None = None
) -> list[Entry]:
    """Library entries (of one architecture / module class); unreadable ones are ``broken``."""
    base = root() if base is None else base
    pattern = f"{_safe(arch) if arch else '*'}/{_safe(module_class) if module_class else '*'}"
    out = []
    for path in sorted(base.glob(f"{pattern}/*/entry.json")):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            meta = None
        if not isinstance(meta, dict):
            meta = {"id": path.parent.name, "broken": "entry.json is unreadable"}
        out.append(Entry(path.parent, meta))
    return out


def find(entry_id_or_prefix: str, *, base: Path | None = None) -> list[Entry]:
    """Entries whose id is (or starts with) the argument."""
    found = [e for e in entries(base=base) if e.id == entry_id_or_prefix]
    return found or [e for e in entries(base=base) if e.id.startswith(entry_id_or_prefix)]


def precision_of(capture_info: dict[str, Any]) -> str:
    """The precision a target's kernels may use: its capture's reduced precision
    (``fp8_weights``; recorded only in the near-lossless tier), else ``exact``."""
    return str(capture_info.get("precision") or "exact")


def matches(
    arch: str,
    spec: dict[str, Any],
    *,
    limit: int = MAX_PRIORS,
    precisions: Iterable[str] | None = None,
) -> list[Entry]:
    """Entries of ``arch`` that can serve the target ``spec``: same module class, compatible
    entrypoints and dtypes, and a precision the target allows (an ``exact`` kernel serves
    any target, a reduced-precision one only a target of that precision) and the run allows
    (``precisions``, ``kernel_agent/precisions.py``; None: no further limit); closest shapes,
    accepted end to end and fastest first; one entry per kernel source."""
    target_sig = signature_summary(spec.get("capture") or {})
    allowed = {"exact", precision_of(spec.get("capture") or {})}
    if precisions is not None:
        allowed &= set(precisions)
    found = []
    for entry in entries(arch, str(spec.get("module_class") or "")):
        if entry.meta.get("broken") or entry.meta.get("module_class") != spec.get("module_class"):
            continue
        if str(entry.meta.get("precision") or "exact") not in allowed:
            continue
        if compatible(entry.meta.get("signature") or {}, target_sig, entry.source()):
            found.append(entry)
    found.sort(
        key=lambda e: (
            closeness(e.meta.get("signature") or {}, target_sig),
            bool(e.meta.get("accepted")),
            e.speedup,
        ),
        reverse=True,
    )
    seen: set[str | None] = set()
    unique = []
    for entry in found:
        if entry.kernel_sha256 not in seen:
            seen.add(entry.kernel_sha256)
            unique.append(entry)
    return unique[:limit]


# ------------------------------------------------------------------ store


def _record_of(run: RunDir, keeper: Truth, target_id: str, snapshot: str) -> dict[str, Any] | None:
    """The verified correct record of a target's snapshot (file name), or None."""
    try:
        records = keeper.records(run.results_file(target_id))
    except TamperError:
        return None
    for rec in reversed(records):
        if Path(str(rec.get("snapshot", ""))).name == snapshot and rec.get("correct"):
            history = run.history_dir(target_id) / snapshot
            return rec if keeper.snapshot_ok(history, rec.get("snapshot_sha256")) else None
    return None


def _result(rec: dict[str, Any], backend: str) -> dict[str, Any]:
    """The evaluation record of an entry (``result.json``)."""
    from kernel_agent.agent.tools import compact

    return {
        **compact(rec),
        "backend": backend,
        "snapshot": rec.get("snapshot"),
        "hypothesis": rec.get("hypothesis"),
        "exp": rec.get("exp"),
    }


def _write_files(dst: Path, blobs: dict[str, bytes]) -> dict[str, str]:
    """Write the entry's files (read-only) and return their sha256."""
    dst.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, data in blobs.items():
        path = truth.replace(dst / name)
        path.write_bytes(data)
        truth.read_only(path)
        digests[name] = sha256_bytes(data)
    return digests


def store_run(
    run: RunDir,
    keeper: Truth,
    *,
    arch: str,
    gpu: str | None,
    torch_version: str | None,
    accepted: Iterable[str],
    final: dict[str, Any] | None,
    min_speedup: float,
) -> list[dict[str, Any]]:
    """Add the run's verified module winners and accepted kernels (``TARGET=path`` items of
    the integration) to the library; returns one summary per stored entry."""
    from kernel_agent.agent.tools import best_for_target

    picks: dict[tuple[str, str], dict[str, Any]] = {}
    for target_id in run.target_ids():
        best = best_for_target(run, target_id, keeper)
        if best and float(best.get("speedup") or 0.0) >= min_speedup:
            picks[(target_id, Path(best["snapshot"]).name)] = {"rec": best, "winner": True}
    for item in accepted:
        target_id, _, path = item.partition("=")
        key = (target_id, Path(path).name)
        if key not in picks and (rec := _record_of(run, keeper, *key)) is not None:
            picks[key] = {"rec": rec, "winner": False}
        if key in picks:
            picks[key]["accepted"] = True
    data = run.load()
    card = data.get("card") or {}
    stored = []
    for (target_id, name), pick in picks.items():
        rec = pick["rec"]
        try:
            code = read_verified(run.history_dir(target_id) / name, rec.get("snapshot_sha256"))
        except (OSError, TamperError) as exc:
            log(f"not storing {target_id}/{name}: {exc}")
            continue
        target_dir = run.target(target_id)
        spec = read_json(target_dir / "spec.json", {}) or {}
        cls = str(spec.get("module_class") or "")
        sig = signature_summary(spec.get("capture") or {})
        eid = entry_id(sig, code)
        dst = root() / _safe(arch) / _safe(cls) / eid
        old = read_json(dst / "entry.json", {}) if (dst / "entry.json").exists() else {}
        old = old if isinstance(old, dict) else {}
        backend = ledger.detect_backend(code.decode(errors="replace"))
        notes_path = workers.notes_file(run, target_id, rec.get("worker"))  # its worker's
        notes = notes_path.read_bytes()[-NOTES_CHARS:] if notes_path.exists() else b""
        accepted_now = bool(pick.get("accepted"))
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        speedup = float(rec.get("speedup") or 0.0)
        e2e = float(final["speedup"]) if accepted_now and final and final.get("speedup") else None
        this_run = {
            "repo_id": card.get("repo_id"),
            "run": str(run.root),
            "date": now,
            "speedup": speedup,
            "accepted": accepted_now,
        }
        runs = [r for r in old.get("runs") or [] if r.get("run") != str(run.root)]
        digest = sha256_bytes(code)
        same = [e for e in entries(arch, cls) if e.id != eid and e.kernel_sha256 == digest]
        origin = old.get("origin") or (same[0].id if same else None)
        hypothesis = rec.get("hypothesis")
        if str(hypothesis).startswith(PRIOR_HYPOTHESIS) and same:  # a prior won again
            hypothesis = same[0].meta.get("hypothesis") or hypothesis
        files = _write_files(
            dst,
            {
                KERNEL: code,
                "spec.json": json.dumps(spec, indent=2, default=str).encode(),
                "result.json": json.dumps(_result(rec, backend), indent=2, default=str).encode(),
                "NOTES.md": notes,
            },
        )
        meta = {
            "version": VERSION,
            "id": eid,
            "sm_arch": arch,
            "gpu": gpu,
            "module_class": cls,
            "family": module_family(cls),
            "repo_id": card.get("repo_id"),
            "revision": card.get("revision"),
            "dtype": (data.get("workload") or {}).get("dtype"),
            "target_id": target_id,
            "phase": spec.get("phase"),
            "precision": precision_of(spec.get("capture") or {}),  # exact | fp8_weights | ...
            "backend": backend,
            "speedup": speedup,
            "pct_of_sol": rec.get("pct_of_sol"),
            "module_winner": bool(pick.get("winner") or old.get("module_winner")),
            "accepted": accepted_now or bool(old.get("accepted")),
            "e2e_speedup": e2e if e2e is not None else old.get("e2e_speedup"),
            "hypothesis": hypothesis,
            "torch_version": torch_version,
            "created": old.get("created") or now,
            "updated": now,
            "run": str(run.root),
            "origin": origin,
            "signature": sig,
            "runs": [*runs, this_run][-MAX_RUNS:],
            "files": files,
        }
        write_json(dst / "entry.json", meta)
        stored.append(
            {
                "entry": eid,
                "target": target_id,
                "module_class": cls,
                "speedup": speedup,
                "accepted": meta["accepted"],
                "module_winner": meta["module_winner"],
            }
        )
    return stored


# ------------------------------------------------------------------ reuse


def refuse(run: RunDir, entry: Entry, problem: str) -> None:
    """Report a library entry that is not reused: a loud line and a ``library_rejected`` event."""
    print(
        f"[{time.strftime('%H:%M:%S')}] LIBRARY: entry {entry.id} ({entry.path}): {problem} "
        "(refused)",
        file=sys.stderr,
        flush=True,
    )
    ledger.event(run, "library_rejected", entry=entry.id, problem=problem[:300])


def evaluate_file(
    run: RunDir,
    target_id: str,
    src: Path,
    *,
    hypothesis: str,
    keeper: Truth,
    timeout: float,
    sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate ``src`` on a target exactly as the ``evaluate_candidate`` tool does: verified
    capture, snapshot, evaluator subprocess, ``results.jsonl`` + ledger. ``sha256``: refuse a
    snapshot without this digest (:class:`TamperError`)."""
    from kernel_agent.agent.tools import record_candidate, snapshot
    from kernel_agent.kernels import evaluate

    capture = run.capture_file(target_id)
    capture_sha256 = keeper.expect(capture)
    snap = snapshot(run, src, target_id)
    snap_sha256 = sha256_file(snap)
    if sha256 is not None and snap_sha256 != sha256:
        raise TamperError(f"{snap.name}: snapshot does not match the library's sha256")
    start = time.perf_counter()
    result = evaluate.run_evaluation(capture, snap, timeout=timeout, capture_sha256=capture_sha256)
    if result.get("status") == "tampered":
        keeper.alarm(capture, str(result.get("error")))
    elif sha256_file(snap) != snap_sha256:
        keeper.alarm(snap, "snapshot changed during its evaluation")
        result = {
            "status": "tampered",
            "correct": False,
            "error": f"{snap.name} changed while it was evaluated; result discarded",
        }
    return record_candidate(
        run,
        target_id,
        src,
        snap,
        result,
        hypothesis=hypothesis,
        eval_s=round(time.perf_counter() - start, 1),
        snapshot_sha256=snap_sha256,
        keeper=keeper,
    )


def _short_error(record: dict[str, Any]) -> str:
    error = str(record.get("error") or "").strip().splitlines()
    return error[-1][:160] if error else ""


def seed_target(
    run: RunDir,
    target_id: str,
    *,
    arch: str,
    keeper: Truth,
    timeout: float = 300.0,
    say: Callable[[str], None] = log,
) -> list[dict[str, Any]]:
    """Evaluate the library entries that match a target before its first agent session.

    Returns what was tried (also rejected entries); the caller records it with
    :func:`remember_seed` so this runs once per target."""
    from kernel_agent import precisions

    target_dir = run.target(target_id)
    spec = read_json(target_dir / "spec.json", {}) or {}
    if not run.capture_file(target_id).exists():
        return []
    allowed = precisions.of_run(run)  # --precisions: no 4-bit prior unless the run allows it
    if why := precisions.refusal(precisions.of_spec(spec), allowed):
        say(f"{target_id}: no library priors: {why}")
        return []
    tried: list[dict[str, Any]] = []
    for entry in matches(arch, spec, precisions=allowed):
        info: dict[str, Any] = {
            "entry": entry.id,
            "repo_id": entry.meta.get("repo_id"),
            "backend": entry.meta.get("backend"),
            "there": entry.speedup,
            "there_pct_of_sol": entry.meta.get("pct_of_sol"),
            "approach": entry.meta.get("hypothesis"),
        }
        problem = entry.problem(arch)
        code = b""
        if problem is None:
            try:
                code = entry.kernel()
            except (OSError, TamperError) as exc:
                problem = str(exc)
        if problem is not None:
            refuse(run, entry, problem)
            tried.append({**info, "status": "rejected", "reason": problem})
            continue
        src = truth.replace(target_dir / "candidates" / f"prior_{entry.id}.py")
        src.write_bytes(code)
        hypothesis = (
            f"{PRIOR_HYPOTHESIS}{entry.meta.get('repo_id')} "
            f"(library entry {entry.id}, {entry.speedup:.2f}x there)"
        )
        try:
            record, row = evaluate_file(
                run,
                target_id,
                src,
                hypothesis=hypothesis,
                keeper=keeper,
                timeout=timeout,
                sha256=entry.kernel_sha256,
            )
        except TamperError as exc:
            refuse(run, entry, str(exc))
            tried.append({**info, "status": "rejected", "reason": str(exc)})
            continue
        tried.append(
            {
                **info,
                "candidate": f"candidates/{src.name}",
                "snapshot": record["snapshot"],
                "status": row["status"],
                "correct": bool(record.get("correct")),
                "speedup": record.get("speedup"),
                "pct_of_sol": record.get("pct_of_sol"),
                "error": _short_error(record),
                "exp": row["exp"],
            }
        )
        here = f"{record['speedup']:.2f}x" if record.get("correct") else row["status"]
        say(f"{target_id}: prior {entry.id} from {entry.meta.get('repo_id')}: {here}")
    ledger.event(run, "library_seed", target=target_id, entries=[t["entry"] for t in tried])
    return tried


def remember_seed(run: RunDir, target_id: str, tried: list[dict[str, Any]]) -> None:
    """Record the seeding of a target in ``run.json`` → ``library.seeded``."""

    def put(data: dict[str, Any]) -> None:
        data.setdefault("library", {}).setdefault("seeded", {})[target_id] = tried

    run.update(put)  # library.seed_target runs in a thread: run.json's lock (update_json)


def remember_store(run: RunDir, stored: list[dict[str, Any]]) -> None:
    """Record stored entries in ``run.json`` → ``library.stored`` (latest per entry)."""

    def put(data: dict[str, Any]) -> None:
        section = data.setdefault("library", {})
        known = {s["entry"]: s for s in section.get("stored") or []}
        known.update({s["entry"]: s for s in stored})
        section["stored"] = list(known.values())

    run.update(put)


def seeded(run: RunDir, target_id: str) -> list[dict[str, Any]] | None:
    """What :func:`seed_target` tried for a target (None: not seeded yet)."""
    tried = ((run.load().get("library") or {}).get("seeded") or {}).get(target_id)
    return tried if isinstance(tried, list) else None


# ------------------------------------------------------------------ prompts


def _x(value: Any) -> str:
    return f"{float(value):.2f}x" if isinstance(value, int | float) else "?"


def _pct(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.2g}"


def _sol(value: Any) -> str:
    return f" at {_pct(float(value))} % of SOL" if isinstance(value, int | float) else ""


def priors_text(tried: list[dict[str, Any]], limit: int = MAX_PRIORS) -> list[str]:
    """The evaluated priors, slow → fast (failures first), at most ``limit``."""
    evaluated = [t for t in tried if t.get("status") != "rejected"]
    evaluated.sort(key=lambda t: (bool(t.get("correct")), float(t.get("speedup") or 0.0)))
    lines = []
    for i, t in enumerate(evaluated[-limit:], 1):
        if t.get("correct"):
            here = f"{_x(t.get('speedup'))}{_sol(t.get('pct_of_sol'))}, `{t.get('snapshot')}`"
        else:
            here = f"{t.get('status')}" + (f" ({t['error']})" if t.get("error") else "")
        approach = str(t.get("approach") or "").strip()
        lines.append(
            f"{i}. `{t.get('candidate')}` ({t.get('backend')}, from `{t.get('repo_id')}`): "
            f"{_x(t.get('there'))}{_sol(t.get('there_pct_of_sol'))} there; here {here}"
            + (f". Approach: {approach[:200]}" if approach else "")
        )
    return lines


def rules(name: str) -> list[str]:
    """The rules (bullet lines) of ``lessons/<name>.md``."""
    path = lessons_dir() / f"{name}.md"
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    return [line[2:].strip() for line in text.splitlines() if line[:2] in ("- ", "* ")]


def lessons_note(backends: Iterable[str], family: str, *, limit: int = LESSONS_CHARS) -> str:
    """Lessons of the module family, then of each backend, cut to ``limit`` characters
    (whole rules only)."""
    lines: list[str] = []
    used = 0
    for name in dict.fromkeys([family, *backends]):
        header: str | None = f"`{name}`:"
        for rule in rules(name):
            line = f"* {rule}"
            extra = len(line) + 1 + (0 if header is None else len(header) + 1)
            if used + extra > limit:
                break
            if header is not None:
                lines.append(header)
                header = None
            lines.append(line)
            used += extra
    return "\n".join(lines)


def prompt_note(run: RunDir, target_id: str, spec: dict[str, Any]) -> str:
    """Engineer-prompt section: the priors evaluated on this target and the lessons."""
    lines: list[str] = []
    shown = priors_text(seeded(run, target_id) or [])
    if shown:
        lines += [
            "",
            "",
            "# Prior kernels from the library",
            f"Kernels for `{spec.get('module_class')}` that won earlier runs on this GPU "
            "architecture were evaluated on this target before you started (ledger rows "
            "`prior winner from ...`). Slow → fast:",
            *shown,
            "Build on the fastest correct one (`parent` = its snapshot) unless you have a "
            "better plan, and use the progression to see which ideas paid off. A prior that "
            "failed here still shows an approach; do not evaluate it again unchanged.",
        ]
    note = lessons_note(spec.get("backends") or [], module_family(spec.get("module_class") or ""))
    if note:
        lines += [
            "",
            "",
            "# Lessons from earlier runs",
            "Validity rules the librarian distilled from earlier runs' notes and ledgers. "
            "They are evidence, not law: re-check a rule when your situation differs.",
            note,
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------ librarian


def lesson_names(run: RunDir) -> dict[str, str]:
    """Lessons files this run has evidence for: name → kind (backend / module family)."""
    names: dict[str, str] = {}
    for target_id in run.target_ids():
        spec = read_json(run.target(target_id) / "spec.json", {}) or {}
        if spec.get("module_class"):
            names[module_family(spec["module_class"])] = "module family"
        for backend in spec.get("backends") or []:
            names.setdefault(str(backend), "backend")
    for row in ledger.rows(run):
        if row["target"] != ledger.E2E:
            for backend in str(row["backend"] or "").split("+"):
                if backend and backend != "torch":
                    names.setdefault(backend, "backend")
    return {k: v for k, v in names.items() if _NAME.match(k)}


def _rows_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| exp | status | speedup | % SOL | backend | hypothesis |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        speedup = "" if r["speedup"] is None else f"{r['speedup']:.3f}x"
        sol = "" if r.get("pct_of_sol") is None else f"{r['pct_of_sol']:.0f}"
        hypothesis = str(r["hypothesis"] or "")[:220].replace("|", "/")
        lines.append(
            f"| {r['exp']} | {r['status']} | {speedup} | {sol} | {r['backend']} | {hypothesis} |"
        )
    return lines


def librarian_prompt(run: RunDir, names: dict[str, str], *, arch: str | None = None) -> str:
    """System prompt of the librarian: the run's evidence and the current lessons files."""
    data = run.load()
    card = data.get("card") or {}
    rows = ledger.rows(run)
    lines = [
        "You are the librarian of kernel-agent's cross-run kernel library. Distil what this "
        "optimisation run learned into short validity rules, so that later runs on other models "
        "avoid its dead ends and repeat its wins sooner.",
        "",
        "# This run",
        f"* model `{card.get('repo_id')}` ({card.get('modality')}), GPU architecture {arch}",
    ]
    integration = read_json(run.root / "integration.json", {}) or {}
    final = integration.get("final") or {}
    if final:
        accepted = ", ".join(
            f"`{ledger.item_label(i['item'])}`" for i in integration.get("accepted", [])
        )
        lines.append(f"* end to end: {final.get('speedup')}x with {accepted or 'nothing'}")
    for target_id in run.target_ids():
        spec = read_json(run.target(target_id) / "spec.json", {}) or {}
        cls = str(spec.get("module_class") or "")
        cases = ", ".join(
            f"`{c.get('signature')}` ×{c.get('count')}"
            for c in (spec.get("capture") or {}).get("cases", [])[:4]
        )
        trows = [r for r in rows if r["target"] == target_id][-LIBRARIAN_ROWS:]
        notes = "\n\n".join(  # the target's and its workers' (workers.py)
            (f"[{label}]\n" if label else "") + path.read_text(errors="replace").strip()
            for label, path in workers.all_notes(run, target_id)
        ).strip()
        if len(notes) > LIBRARIAN_NOTES_CHARS:
            notes = "…" + notes[-LIBRARIAN_NOTES_CHARS:]
        lines += [
            "",
            f"## Target `{target_id}`: `{cls}` (module family `{module_family(cls)}`)",
            f"* backends: {', '.join(spec.get('backends') or [])}; cases: {cases or '?'}",
            f"* approach planned: {spec.get('approach', '')}",
            "",
            *(_rows_table(trows) if trows else ["(no evaluations)"]),
            "",
            "NOTES.md of its kernel engineer (most recent part):",
            "```",
            notes or "(empty)",
            "```",
        ]
    lines += ["", "# Current lessons files (merge into these)"]
    for name in names:
        current = rules(name)
        lines += ["", f"## `{name}.md`", *(f"- {r}" for r in current)] if current else []
    if not any(rules(name) for name in names):
        lines.append("(none yet)")
    listed = ", ".join(f"`{name}` ({kind})" for name, kind in names.items())
    lines += [
        "",
        "# Task",
        f"Return the COMPLETE new rule list of every lessons file this run has evidence for. "
        f"Files: {listed}. For each file:",
        "* Keep the existing rules that still hold, add what this run showed, merge "
        "duplicates, and drop rules this run's evidence contradicts.",
        f"* One line per rule, at most {RULE_CHARS} characters, specific and actionable: "
        '"do X when Y" or "Z fails because W (abandoned after N attempts)". Name the shapes, '
        "dtypes, sizes or GPU when they matter, and the measured effect.",
        "* Only what the ledger and the notes show (statuses, speedups, % of SOL, errors). "
        "No generic advice.",
        f"* At most 12 rules per file, most useful first (hard cap {MAX_RULES}).",
        "* Backend-specific rules go to the backend's file, rules about the kind of module "
        "to the module family's file. Leave out files with nothing to add or change.",
        "Answer with the structured output only; do not edit any file.",
    ]
    return "\n".join(lines)


def librarian_due(run: RunDir) -> bool:
    """Whether the run has kernel evaluations the librarian has not distilled yet."""
    rows = ledger.rows(run)
    done = (run.load().get("library") or {}).get("librarian") or {}
    return any(r["target"] != ledger.E2E for r in rows) and done.get("rows") != len(rows)


def remember_librarian(run: RunDir, written: list[Path]) -> None:
    """Record the librarian's session in ``run.json`` → ``library.librarian``."""
    record = {
        "rows": len(ledger.rows(run)),
        "files": [str(p) for p in written],
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    run.update(lambda data: data.setdefault("library", {}).__setitem__("librarian", record))


def _clean(rule: Any) -> str:
    text = " ".join(str(rule).split()).lstrip("-*• ").strip()
    return text if len(text) <= RULE_CHARS else text[: RULE_CHARS - 1].rstrip() + "…"


def write_lessons(structured: Any, names: Iterable[str]) -> list[Path]:
    """Write the librarian's answer to ``lessons/<name>.md`` (only the ``names`` it was asked
    about; duplicates removed, at most :data:`MAX_RULES` rules); returns the files written."""
    allowed = set(names)
    items = structured.get("lessons") if isinstance(structured, dict) else None
    written = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("file", "")).strip().removesuffix(".md").lower()
        if name not in allowed:
            continue
        keep: dict[str, str] = {}
        for rule in item.get("rules") or []:
            text = _clean(rule)
            key = re.sub(r"[^a-z0-9]+", "", text.lower())
            if key and key not in keep:
                keep[key] = text
        path = lessons_dir() / f"{name}.md"
        if not keep:
            continue  # an empty answer never wipes a file
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"- {r}" for r in list(keep.values())[:MAX_RULES])
        path.write_text(
            f"# Lessons: {name}\n\n<!-- maintained by the kernel-agent librarian (library.py); "
            f"edit freely -->\n\n{body}\n"
        )
        written.append(path)
    return written


# ------------------------------------------------------------------ Claude Code memory

MEMORY_INDEX = "MEMORY.md"  # the index of an auto-memory directory, not a note
#: Words of a memory note's name or description that name a backend's lessons file, most
#: specific first (a Triton note that mentions CUDA graphs is a Triton note).
MEMORY_BACKEND_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("triton", ("triton",)),
    ("tilelang", ("tilelang",)),
    ("cute", ("cute", "cutedsl")),
    ("nvrtc", ("nvrtc",)),
    ("cuda", ("cuda", "cublas", "cublaslt", "cutlass", "nvcc", "ptx")),
)
WRITERS = ("Write", "Edit", "MultiEdit")  # the tools that write a memory note
SIMILAR = 0.5  # word overlap above which an imported rule is shown next to an existing one
_STOP = {"a", "an", "and", "at", "for", "in", "is", "it", "of", "on", "or", "the", "to", "with"}


@dataclass
class MemoryNote:
    """One note of a Claude Code auto-memory directory: ``<name>.md`` with a front matter
    (``name``, ``description``, ``type``, and ``originSessionId`` when recorded)."""

    path: Path
    name: str
    description: str
    kind: str  # user, feedback, project or reference
    origin: str  # the Claude Code session that wrote it ("" when not recorded)
    body: str


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    """The ``key: value`` lines of a note's front matter (nested ones too, first wins) and
    the text after it."""
    m = re.match(r"---\r?\n(.*?)\r?\n---[ \t]*\r?\n?", text, re.S)
    if not m:
        return {}, text
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        key, sep, value = line.strip().partition(":")
        if sep and key and key not in fields:
            fields[key] = _scalar(value.strip())
    return fields, text[m.end() :]


def _scalar(value: str) -> str:
    """A front-matter value without its YAML quotes (``"a \\"b\\""`` → ``a "b"``)."""
    if len(value) > 1 and value[0] == value[-1] == '"':
        try:
            return str(json.loads(value))
        except ValueError:
            return value[1:-1]
    if len(value) > 1 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def memory_notes(directory: Path) -> list[MemoryNote]:
    """The notes of an auto-memory directory (``MEMORY.md`` left out). Only read."""
    notes = []
    for path in sorted(directory.glob("*.md")):
        if path.name == MEMORY_INDEX or not path.is_file():
            continue
        fields, body = _front_matter(path.read_text(errors="replace"))
        notes.append(
            MemoryNote(
                path=path,
                name=fields.get("name") or path.stem,
                description=fields.get("description", ""),
                kind=fields.get("type", ""),
                origin=fields.get("originSessionId", ""),
                body=body.strip(),
            )
        )
    return notes


def memory_rule(note: MemoryNote) -> str:
    """The lesson rule of a note: its description (the one-line summary Claude Code keeps
    for it), else its first paragraph; at most :data:`RULE_CHARS` characters."""
    text = note.description or note.body.split("\n\n")[0]
    return _clean(text)


def memory_lesson(note: MemoryNote) -> str | None:
    """The lessons file a note most likely belongs to: a backend its name or description
    mentions (:data:`MEMORY_BACKEND_WORDS`), else a module family in its name; None when
    neither (``--to`` decides)."""
    words = set(re.findall(r"[a-z0-9]+", f"{note.name} {note.description}".lower()))
    for backend, keys in MEMORY_BACKEND_WORDS:
        if words & set(keys):
            return backend
    for token in re.split(r"[^a-z0-9]+", note.name.lower()):
        for family, keys in FAMILIES:
            if token and any(k in token for k in keys):
                return family
    return None


def agent_writers(runs: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The run directory of every agent session in the agent logs (``logs/agent-*.jsonl``)
    of the runs under ``runs``, by session id, and of every auto-memory note those sessions
    wrote or edited (a Write or Edit of ``~/.claude/projects/<project>/memory/<file>``), by
    file name."""
    sessions: dict[str, str] = {}
    notes: dict[str, str] = {}
    for log_path in sorted(runs.rglob("agent-*.jsonl")):
        run = str(log_path.parent.parent)
        for line in log_path.read_text(errors="replace").splitlines():
            for sid in re.findall(r'"session_id": "([0-9A-Fa-f-]{8,})"', line):
                sessions.setdefault(sid, run)
            if "/.claude/projects/" not in line or not line.startswith('{"type": "Assistant'):
                continue
            try:
                blocks = json.loads(line)["data"]["content"]
            except (ValueError, KeyError, TypeError):
                continue
            for block in blocks if isinstance(blocks, list) else []:
                tool_input = block.get("input") if isinstance(block, dict) else None
                if not isinstance(tool_input, dict) or block.get("name") not in WRITERS:
                    continue
                path = Path(str(tool_input.get("file_path") or ""))
                if path.parent.name == "memory" and "/.claude/projects/" in str(path):
                    notes.setdefault(path.name, run)
    return sessions, notes


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP


def similar_rule(rule: str, existing: Iterable[str]) -> str | None:
    """The existing rule that shares most words with ``rule`` (at least :data:`SIMILAR` of
    the shorter one's), or None."""
    words = _words(rule)
    best, score = None, SIMILAR
    for other in existing:
        theirs = _words(other)
        if words and theirs:
            overlap = len(words & theirs) / min(len(words), len(theirs))
            if overlap >= score:
                best, score = other, overlap
    return best


def _key(rule: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", rule.lower())  # as write_lessons deduplicates


def cmd_import_memory(ns: argparse.Namespace) -> int:
    """Turn Claude Code auto-memory notes into lessons rules. Prints the plan; writes the
    lessons files only with ``--write``. The memory files are only read."""
    source = Path(ns.memory_dir).expanduser()
    if not source.is_dir():
        raise SystemExit(
            f"{source}: not a directory (a Claude Code memory directory is "
            "~/.claude/projects/<project>/memory)"
        )
    if ns.to and not _NAME.match(ns.to):
        raise SystemExit(f"--to {ns.to!r}: not a lessons file name (e.g. cuda, triton, norm)")
    notes = memory_notes(source)
    wanted = set(ns.note or [])
    unknown = wanted - {n.name for n in notes} - {n.path.stem for n in notes}
    if unknown:
        raise SystemExit(f"no note named {', '.join(sorted(unknown))} in {source}")
    sessions, written = agent_writers(Path(ns.runs).expanduser()) if ns.runs else ({}, {})
    print(f"memory: {source} ({len(notes)} notes; {MEMORY_INDEX}, the index, is not imported)")
    print(f"lessons: {lessons_dir()}")
    added: dict[str, list[str]] = {}
    for note in notes:
        if wanted and not wanted & {note.name, note.path.stem}:
            continue
        run = sessions.get(note.origin) or written.get(note.path.name)
        origin = f", session {note.origin[:8]}" if note.origin else ""
        origin += f", by an agent of {run}" if run else ""
        head = f"  {note.name} ({note.kind or 'no type'}{origin})"
        name = ns.to or memory_lesson(note)
        rule = memory_rule(note)
        why = None
        if note.kind == "user" and not wanted:
            why = "a note about you, not a kernel lesson (select it with --note)"
        elif ns.runs and run is None:
            why = f"no agent session of the runs under {ns.runs} wrote it"
        elif not rule:
            why = "no description or text"
        elif name is None:
            why = "no backend or module family in its name or description (--to NAME)"
        elif _key(rule) in {_key(r) for r in rules(name) + added.get(name, [])}:
            why = f"already in {name}.md"
        if why:
            print(f"{head}: skipped, {why}")
            continue
        assert name is not None
        print(f"{head} -> {name}.md")
        print(f"      + {rule}")
        if like := similar_rule(rule, rules(name) + added.get(name, [])):
            print(f"      ~ similar to: {like}")
        added.setdefault(name, []).append(rule)
    total = sum(len(v) for v in added.values())
    files = ", ".join(f"{k}.md +{len(v)}" for k, v in added.items())
    for name, new in added.items():
        over = len(rules(name)) + len(new) - MAX_RULES
        if over > 0:
            print(f"  {name}.md: {over} rule(s) over the cap of {MAX_RULES} would be dropped")
    if not ns.write:
        print(
            f"dry run: {total} rule(s) would be added ({files or 'none'}). Nothing was "
            "written; --write writes them. The memory files are never changed."
        )
        return 0
    for name, new in added.items():
        write_lessons({"lessons": [{"file": name, "rules": rules(name) + new}]}, [name])
    print(
        f"added {total} rule(s) ({files or 'none'}) to {lessons_dir()}; the memory files "
        "are unchanged (delete them yourself if you no longer want them)."
    )
    return 0


# ------------------------------------------------------------------ report


def report_lines(run: RunDir) -> list[str]:
    """``## Kernel library`` section of report.md (empty when the library was not used)."""
    section = run.load().get("library") or {}
    tried = [(t, item) for t, items in (section.get("seeded") or {}).items() for item in items]
    stored = section.get("stored") or []
    lessons = (section.get("librarian") or {}).get("files") or []
    if not (tried or stored or lessons):
        return []
    lines = ["", "## Kernel library", ""]
    for target_id, t in tried:
        if t.get("status") == "rejected":
            here = f"rejected: {t.get('reason')}"
        elif t.get("correct"):
            here = _x(t.get("speedup"))
        else:
            here = str(t.get("status"))
        lines.append(
            f"* `{target_id}`: prior `{t.get('entry')}` from `{t.get('repo_id')}` "
            f"({_x(t.get('there'))} there) → {here} here"
        )
    for s in stored:
        flags = [
            flag
            for flag, on in (
                ("accepted end to end", s.get("accepted")),
                ("module winner", s.get("module_winner")),
            )
            if on
        ]
        lines.append(
            f"* stored `{s.get('entry')}`: `{s.get('target')}` (`{s.get('module_class')}`), "
            f"{_x(s.get('speedup'))}, {', '.join(flags)}"
        )
    if lessons:
        lines.append("* lessons updated: " + ", ".join(f"`{Path(p).name}`" for p in lessons))
    return [*lines, ""]


# ------------------------------------------------------------------ CLI


def _age_days(meta: dict[str, Any]) -> float | None:
    stamp = str(meta.get("updated") or meta.get("created") or "")
    when = ledger.epoch(stamp)
    return None if when is None else (time.time() - when) / 86400


def cmd_list(ns: argparse.Namespace) -> int:
    found = entries(ns.arch, ns.module_class)
    print(f"library: {root()} ({len(found)} entries)")
    if not found:
        return 0
    header = ("id", "arch", "module class", "backend", "speedup", "% SOL", "flags", "model", "date")
    rows = [header]
    for e in found:
        m = e.meta
        problem = e.problem()
        flags = "accepted" if m.get("accepted") else "winner" if m.get("module_winner") else ""
        if m.get("precision") not in (None, "exact"):  # e.g. fp8_weights
            flags = f"{flags} {m['precision']}".strip()
        sol = m.get("pct_of_sol")
        rows.append(
            (
                e.id,
                str(m.get("sm_arch") or e.path.parent.parent.name),
                str(m.get("module_class") or e.path.parent.name),
                str(m.get("backend") or ""),
                _x(m.get("speedup")) if m.get("speedup") else "",
                _pct(float(sol)) if isinstance(sol, int | float) else "",
                "BROKEN" if problem else flags,
                str(m.get("repo_id") or ""),
                str(m.get("updated") or "")[:10],
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for r in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)).rstrip())
    return 0


def cmd_show(ns: argparse.Namespace) -> int:
    found = find(ns.entry_id)
    if len(found) != 1:
        what = "no entry" if not found else f"{len(found)} entries"
        raise SystemExit(f"{what} match {ns.entry_id!r} (see `kernel-agent library list`)")
    entry = found[0]
    m = entry.meta
    problem = entry.problem()
    print(f"entry {entry.id}  ({entry.path})")
    for key in (
        "sm_arch",
        "gpu",
        "module_class",
        "family",
        "repo_id",
        "target_id",
        "phase",
        "precision",
        "backend",
        "speedup",
        "pct_of_sol",
        "module_winner",
        "accepted",
        "e2e_speedup",
        "torch_version",
        "created",
        "updated",
        "origin",
        "hypothesis",
    ):
        if m.get(key) is not None:
            print(f"  {key}: {m[key]}")
    print(f"  integrity: {'OK (every sha256 matches)' if problem is None else problem}")
    sig = m.get("signature") or {}
    print(f"  methods: {', '.join(sig.get('methods') or [])}  dtypes: {sig.get('dtypes')}")
    for case in sig.get("cases") or []:
        print(f"    {case.get('signature')}  ×{case.get('count')}")
    result = read_json(entry.path / "result.json", {}) or {}
    for case in result.get("cases") or []:
        print(
            f"    measured {case.get('signature')}: {_x(case.get('speedup'))}"
            f"{_sol(case.get('pct_of_sol'))}"
        )
    for r in m.get("runs") or []:
        print(f"  run {r.get('date')}: {r.get('repo_id')} {_x(r.get('speedup'))} ({r.get('run')})")
    notes = entry.path / "NOTES.md"
    if notes.exists() and notes.stat().st_size:
        print("  NOTES.md (head):")
        for line in notes.read_text(errors="replace").splitlines()[:30]:
            print(f"    {line}")
    print(f"  kernel: {entry.path / KERNEL}")
    return 0


def cmd_prune(ns: argparse.Namespace) -> int:
    removed = 0
    for entry in entries():
        problem = entry.problem()
        age = _age_days(entry.meta)
        why = problem
        if why is None and ns.older_than is not None and age is not None and age > ns.older_than:
            why = f"last stored {age:.0f} days ago"
        if why is None:
            continue
        print(f"{'would remove' if ns.dry_run else 'removed'} {entry.id} ({entry.path}): {why}")
        if not ns.dry_run:
            shutil.rmtree(entry.path, ignore_errors=True)
        removed += 1
    noun = "entry" if removed == 1 else "entries"
    print(f"{removed} {noun} {'would be removed' if ns.dry_run else 'removed'}")
    return 0


def cmd_path(ns: argparse.Namespace) -> int:
    print(root())
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "list": cmd_list,
    "show": cmd_show,
    "prune": cmd_prune,
    "path": cmd_path,
    "import-memory": cmd_import_memory,
}


def add_parser(sub: Any) -> argparse.ArgumentParser:
    """The ``kernel-agent library`` subcommands (``sub``: the CLI's subparsers)."""
    p: argparse.ArgumentParser = sub.add_parser(
        "library",
        help="cross-run kernel library: list, show, prune, path, import-memory",
        description=f"Kernels that won earlier runs, per GPU architecture, and distilled "
        f"lessons. Location: ${ENV} or ~/.cache/kernel-agent/library.",
    )
    lib = p.add_subparsers(dest="library_command", required=True)
    q = lib.add_parser("list", help="entries of every GPU architecture")
    q.add_argument("--arch", help="only this architecture, e.g. sm_120")
    q.add_argument("--module-class", help="only this module class")
    q = lib.add_parser("show", help="one entry: metadata, integrity, cases, runs, notes")
    q.add_argument("entry_id", help="entry id or a unique prefix")
    q = lib.add_parser("prune", help="remove broken entries (and with --older-than, old ones)")
    q.add_argument("--older-than", type=float, metavar="DAYS")
    q.add_argument("--dry-run", action="store_true", help="only list what would go")
    lib.add_parser("path", help="print the library directory")
    q = lib.add_parser(
        "import-memory",
        help="turn Claude Code auto-memory notes into lessons (dry run unless --write)",
        description="Turn the notes of a Claude Code auto-memory directory into rules of "
        "lessons/<backend or module family>.md. Prints what it would add; --write adds it. "
        "The memory files are only read, never changed or deleted.",
    )
    q.add_argument("memory_dir", help="e.g. ~/.claude/projects/<project>/memory")
    q.add_argument("--note", action="append", metavar="NAME", help="only this note (repeat)")
    q.add_argument("--to", metavar="NAME", help="lessons file for every selected note")
    q.add_argument(
        "--runs",
        metavar="DIR",
        help="only notes that agent sessions of the runs under DIR wrote (from their "
        "logs/agent-*.jsonl: a Write or Edit of the note, or its originSessionId)",
    )
    q.add_argument("--write", action="store_true", help="write the lessons files")
    return p


def main(ns: argparse.Namespace) -> int:
    return COMMANDS[ns.library_command](ns)
