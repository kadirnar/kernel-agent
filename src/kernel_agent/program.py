"""``program.md``: human-editable research-org instructions per agent role.

The human steers the agents by editing a Markdown file instead of code (the
``program.md`` idea of karpathy/autoresearch). Level-2 headings select who
reads a section:

* ``## all`` – every agent
* ``## planner``, ``## kernel`` (``kernel-<target>`` agents), ``## systems``,
  ``## harness``, ``## research`` (``research-<target>``: plateau reviews),
  ``## refactor`` (``refactor-<target>``: rewrites of region targets) – that
  role only; ``## kernel, systems`` names several roles.

Other ``##`` headings are ignored with a warning; text above the first ``##``
heading and ``<!-- comments -->`` are for humans and never reach an agent.

``--program FILE`` (default: the packaged template ``agent/program.md``) is
copied to ``<run>/program.md`` when the run is created. That copy is re-read
before every agent session (:func:`for_agent`), so a human can edit it while
the run is going; the sha256 each agent saw goes into ``costs.json`` and
``run.json`` → ``program``, and every version is kept as
``logs/program-<sha12>.md``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from kernel_agent.workspace import RunDir, write_json

ROLES = ("planner", "kernel", "systems", "harness", "research", "refactor")
SECTIONS = ("all", *ROLES)
FILENAME = "program.md"
TEMPLATE = Path(__file__).parent / "agent" / "program.md"
DEFAULT_SOURCE = "packaged default"

_HEADING = re.compile(r"^##(?!#)(?:[ \t]+(.*?))?[ \t#]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_COMMENT = re.compile(r"<!--.*?-->", re.S)


def role_of(agent: str) -> str | None:
    """Role of an orchestrator agent name (``kernel-rmsnorm`` → ``kernel``), or None."""
    role = agent.split("-", 1)[0].lower()
    return role if role in ROLES else None


def parse(text: str) -> tuple[dict[str, str], list[str]]:
    """Section name → body (sections repeated or shared between roles are joined) + warnings.

    ``##`` lines inside fenced code blocks are content, not headings; HTML comments
    are removed first, so commenting out a whole section disables it.
    """
    parts: dict[str, list[str]] = {}
    warnings: list[str] = []
    current: list[str] = []  # names the lines below go to ([] = preamble or unknown section)
    in_fence = False
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            for name in current:
                parts.setdefault(name, []).append(body)
        lines.clear()

    for line in _COMMENT.sub("", text).splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match is None:
            lines.append(line)
            continue
        flush()
        title = match.group(1) or ""
        names = [n.strip().lower() for n in re.split(r"[,/]", title) if n.strip()]
        current = [n for n in names if n in SECTIONS]
        unknown = [n for n in names if n not in SECTIONS]
        if unknown or not names:
            what = f"unknown name {', '.join(unknown)}" if unknown else "no name"
            where = f"only {', '.join(current)} receive it" if current else "its text is ignored"
            warnings.append(
                f"program.md: `## {title}`: {what} (known: {', '.join(SECTIONS)}); {where}"
            )
    flush()
    return {name: "\n\n".join(bodies) for name, bodies in parts.items()}, warnings


@dataclass(frozen=True)
class Program:
    sections: dict[str, str] = field(default_factory=dict)
    sha256: str | None = None  # None: there is no program file
    warnings: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path) -> Program:
        if not path.is_file():
            return cls()
        data = path.read_bytes()
        sections, warnings = parse(data.decode("utf-8", errors="replace"))
        return cls(sections, hashlib.sha256(data).hexdigest(), tuple(warnings))

    def for_role(self, role: str | None) -> str:
        """The ``## all`` section, then the role's own ("" when both are missing)."""
        names = ["all"] if role is None else ["all", role]
        return "\n\n".join(
            f"## {name}\n{self.sections[name]}" for name in names if self.sections.get(name)
        )

    def prompt_note(self, agent: str) -> str:
        """``# Program`` section appended to an agent's system prompt ("" when empty)."""
        body = self.for_role(role_of(agent))
        if not body:
            return ""
        return (
            "\n\n# Program (program.md, written by the human running this optimisation)\n"
            "Follow these instructions. Where they conflict with the guidance above they "
            "take precedence, except for the correctness and no-cheating rules and the tool "
            "contracts, which always hold. `program.md` in the run directory is read-only "
            "for you.\n\n" + body
        )


def install(run: RunDir, source: str | Path | None = None) -> Path:
    """Copy ``source`` (default: the packaged template) to ``<run>/program.md``.

    Without ``source`` an existing copy is kept (a resumed run keeps the human's
    edits), and so is a deliberately deleted one: the template is only installed
    into runs that never had a program (new runs, runs from before program.md).
    """
    dst = run.root / FILENAME
    data = run.load()
    if source is None and (dst.exists() or "program" in data):
        return dst
    src = TEMPLATE if source is None else Path(source).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"program file not found: {source}")
    shutil.copyfile(src, dst)
    data["program"] = {
        "source": DEFAULT_SOURCE if source is None else str(src),
        "installed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": Program.load(dst).sha256,
        "versions": data.get("program", {}).get("versions", []),
    }
    write_json(run.run_json, data)
    return dst


def for_agent(run: RunDir, agent: str, log: Callable[[str], None] = print) -> Program:
    """Re-read ``<run>/program.md`` for an agent that is about to start and record its version.

    A version that differs from the previous agent's is appended to
    ``run.json`` → ``program.versions``, snapshotted to ``logs/program-<sha12>.md``
    and logged together with its parse warnings.
    """
    path = run.root / FILENAME
    program = Program.load(path)
    data = run.load()
    record = data.setdefault("program", {"source": None, "sha256": None, "versions": []})
    versions = record.setdefault("versions", [])
    previous = versions[-1]["sha256"] if versions else record.get("sha256")
    if versions and previous == program.sha256:
        return program
    versions.append({"sha256": program.sha256, "agent": agent, "at": time.strftime("%H:%M:%S")})
    write_json(run.run_json, data)
    if program.sha256 is None:
        log(f"program: {path} is missing; agents get no program.md instructions")
        return program
    snapshot = run.root / "logs" / f"program-{program.sha256[:12]}.md"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copyfile(path, snapshot)
    if previous is not None and previous != program.sha256:
        log(f"program: program.md changed (sha256 {program.sha256[:12]}); used from {agent} on")
    for warning in program.warnings:
        log(warning)
    return program


def init(path: Path, *, force: bool = False) -> Path:
    """Write the packaged template to ``path`` for editing."""
    if path.is_dir():
        path = path / FILENAME
    if path.exists() and not force:
        raise SystemExit(f"{path} exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, path)
    return path
