"""kernel-agent's know-how as Agent Skills (issue #176).

The skills live in a Claude Code plugin inside the package, :data:`PLUGIN_DIR`: the manifest
``.claude-plugin/plugin.json`` (plugin name ``kernel-agent``) and ``skills/<name>/SKILL.md``
with the resource files it links. Every agent session loads that plugin by explicit path
(``ClaudeAgentOptions.plugins``) and enables exactly its skills (``skills=``), while its
filesystem settings stay off (``setting_sources=[]``, #126): a session sees kernel-agent's
skills and nothing of the user's ``~/.claude`` (skills, settings, memory), ``CLAUDE.md`` or
``AGENTS.md``.

Progressive disclosure: a session's context holds each skill's name and description; the
Skill tool loads a ``SKILL.md`` when the task needs it (its directory comes with it), and
the files it links are read on demand. A prompt names the skills its role should load
first (:func:`for_role`, ``prompts.py``).

The skills replace ``agent/knowledge/*.md``: their text moved into the skills section by
section, and :data:`LEGACY` maps each former file to the skills that hold it now
(:func:`legacy_text`). :func:`problems` validates the tree (frontmatter, links, resources,
example names); ``tests/test_skills.py`` runs it.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk.types import SdkPluginConfig

AGENT_DIR = Path(__file__).parent / "agent"
PLUGIN_DIR = AGENT_DIR / "plugin"
SKILLS_DIR = PLUGIN_DIR / "skills"
EXAMPLES_DIR = AGENT_DIR / "examples"
#: ``name`` of ``.claude-plugin/plugin.json``: a session lists the skills as ``kernel-agent:<name>``
PLUGIN_NAME = "kernel-agent"
NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_DESCRIPTION = 1024
SKILL_FILE = "SKILL.md"

#: The former ``agent/knowledge/<file>`` → the skills that hold its text now.
LEGACY: dict[str, tuple[str, ...]] = {
    "playbook.md": ("optimisation-playbook",),
    "triton.md": ("triton-kernels",),
    "cuda.md": ("cuda-kernels", "cuda-graphs-streams-pdl"),
    "cute_dsl.md": ("cute-dsl",),
    "tilelang.md": ("tilelang-kernels",),
    "low_precision.md": (
        "precision-tiers",
        "fp8-weights",
        "fp8-w8a8",
        "mxfp8",
        "fp8-kv-cache",
        "fp4-weights",
    ),
    "systems.md": ("systems-patterns", "speculative-decoding"),
    "native.md": ("native-engines",),
    "gpus.md": ("gpu-architectures",),
    "sources.md": ("documentation-sources",),
}

#: The skill of each backend.
BACKEND_SKILLS = {
    "triton": "triton-kernels",
    "cuda": "cuda-kernels",
    "nvrtc": "cuda-kernels",
    "cute": "cute-dsl",
    "tilelang": "tilelang-kernels",
}
#: The skill of each reduced precision (``precision-tiers``: what they share).
PRECISION_SKILLS = {
    "fp8_weights": "fp8-weights",
    "fp8_w8a8": "fp8-w8a8",
    "fp8_mx": "mxfp8",
    "fp8_kv": "fp8-kv-cache",
    "fp4_weights": "fp4-weights",
    "reduced": "precision-tiers",
}
#: The skills a role loads first, before those of its target's backends and precision.
ROLE_SKILLS = {
    "planner": ("optimisation-playbook", "profiling-and-roofline"),
    "kernel": ("optimisation-playbook",),
    "systems": ("systems-patterns", "speculative-decoding", "optimisation-playbook"),
    "native": ("native-engines", "cuda-kernels", "cute-dsl", "cuda-graphs-streams-pdl"),
    "research": ("optimisation-playbook", "profiling-and-roofline"),
    "dossier": ("documentation-sources",),
}

_LINK = re.compile(r"\]\(([\w./-]+\.md)(?:#[^)]*)?\)")  # markdown links to files of the tree
_EXAMPLE = re.compile(r"`(?:examples/)?((?:cuda|nvrtc|triton|cute|tilelang)_[a-z0-9_]+\.py)`")
#: Backticked names that look like examples and are not (Unsloth's / kernels/triton_launch.py).
_NOT_EXAMPLES = {"triton_launch.py"}


@dataclass(frozen=True)
class Skill:
    """One skill: ``skills/<name>/SKILL.md`` and the resource files next to it."""

    name: str
    description: str
    dir: Path

    @property
    def qualified(self) -> str:
        """The name a session's Skill tool takes: ``kernel-agent:<name>``."""
        return f"{PLUGIN_NAME}:{self.name}"

    @property
    def path(self) -> Path:
        return self.dir / SKILL_FILE

    @property
    def resources(self) -> list[Path]:
        """The files besides ``SKILL.md``, sorted."""
        return sorted(p for p in self.dir.rglob("*") if p.is_file() and p.name != SKILL_FILE)

    def text(self) -> str:
        """``SKILL.md`` and every resource file, joined: everything the skill holds."""
        return "\n\n".join(p.read_text() for p in [self.path, *self.resources])


def frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """The YAML frontmatter of a ``SKILL.md`` or an agent file and the body after it.

    The subset they use: ``key: value`` plain scalars (an int when all digits) and
    ``key:`` followed by ``  - item`` lines. A plain scalar may not contain ``": "`` (YAML
    would read a mapping there and Claude Code would drop the field): ``ValueError``."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError("no frontmatter (the file must start with ---)")
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        raise ValueError("frontmatter not closed with ---") from None
    data: dict[str, Any] = {}
    key = None
    for line in lines[1:end]:
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) and line.strip().startswith("- ") and key:
            if not isinstance(data[key], list):
                raise ValueError(f"{key}: a list item after a value")
            data[key].append(line.strip()[2:].strip())
            continue
        match = re.match(r"^([A-Za-z][\w-]*):(?:\s+(.*))?$", line)
        if not match:
            raise ValueError(f"not a `key: value` line: {line!r}")
        key, value = match.group(1), (match.group(2) or "").strip()
        if key in data:
            raise ValueError(f"{key}: given twice")
        if not value:
            data[key] = []
            continue
        if ": " in value or value.endswith(":") or value.startswith(("'", '"', "[", "{", "#")):
            raise ValueError(f"{key}: not a plain YAML scalar: {value!r}")
        data[key] = int(value) if value.isdigit() else value
    return data, "\n".join(lines[end + 1 :]).lstrip("\n")


@functools.cache
def index() -> dict[str, Skill]:
    """Every skill of the plugin by name (its frontmatter ``name``), sorted."""
    found = {}
    for path in sorted(SKILLS_DIR.glob(f"*/{SKILL_FILE}")):
        meta, _ = frontmatter(path.read_text())
        name = str(meta.get("name") or path.parent.name)
        found[name] = Skill(name, str(meta.get("description") or ""), path.parent)
    return found


def get(name: str) -> Skill:
    """A skill by its name or its qualified name (``kernel-agent:<name>``)."""
    return index()[name.removeprefix(f"{PLUGIN_NAME}:")]


def qualified(name: str) -> str:
    """``kernel-agent:<name>`` of a skill (KeyError when there is none)."""
    return get(name).qualified


def qualified_names() -> list[str]:
    """The names a session enables (``ClaudeAgentOptions.skills``): every skill's."""
    return [s.qualified for s in index().values()]


def for_role(
    role: str,
    *,
    backends: Iterable[str] = (),
    precision: str | None = None,
    quality: str = "exact",
) -> list[str]:
    """The skills (qualified, in a fixed order) a session of ``role`` loads first: the
    role's own (:data:`ROLE_SKILLS`), the skill of each of its target's ``backends`` (kernel,
    research and dossier sessions), and for a reduced ``precision`` ``precision-tiers`` and
    its skill; a planner of a run that allows reduced precision (``quality`` not exact) gets
    ``precision-tiers``. The prompts name them (``prompts.py``); a coordinator may pass them
    as a session's ``skills=`` (#174)."""
    names = list(ROLE_SKILLS.get(role, ()))
    if role in ("kernel", "research", "dossier"):
        names += [BACKEND_SKILLS[b] for b in backends if b in BACKEND_SKILLS]
    if precision is not None:
        names += ["precision-tiers", PRECISION_SKILLS.get(precision, "precision-tiers")]
    elif role == "planner" and quality != "exact":
        names.append("precision-tiers")
    return [qualified(n) for n in dict.fromkeys(names)]


#: Claude Code's own skills a session keeps: ``workflow-authoring`` is the reference of its
#: Workflow tool (with the skill hidden, Claude Code inlines it into that tool's description:
#: +17K characters per request). Every other bundled skill is hidden.
BUNDLED = ("workflow-authoring",)


def session_skills() -> list[str]:
    """``ClaudeAgentOptions.skills`` of every session: kernel-agent's and :data:`BUNDLED`."""
    return [*qualified_names(), *BUNDLED]


def plugin() -> SdkPluginConfig:
    """The ``ClaudeAgentOptions.plugins`` entry of kernel-agent's plugin."""
    return {"type": "local", "path": str(PLUGIN_DIR)}


def legacy_text(filename: str) -> str:
    """The text the former ``agent/knowledge/<filename>`` held, from the skills it moved to."""
    return "\n\n".join(get(name).text() for name in LEGACY[filename])


def problems() -> list[str]:
    """What is wrong with the skills tree ([] when nothing): frontmatter (name = directory,
    a description of at most :data:`MAX_DESCRIPTION` characters), every relative link
    resolves, every resource file is linked from its ``SKILL.md`` (reachable on demand), every
    example named exists in :data:`EXAMPLES_DIR`, and the plugin manifest names the plugin."""
    out = []
    manifest = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
    if f'"name": "{PLUGIN_NAME}"' not in (manifest.read_text() if manifest.is_file() else ""):
        out.append(f"{manifest}: missing or not named {PLUGIN_NAME}")
    for path in sorted(SKILLS_DIR.glob(f"*/{SKILL_FILE}")):
        where = path.relative_to(SKILLS_DIR)
        try:
            meta, body = frontmatter(path.read_text())
        except ValueError as exc:
            out.append(f"{where}: {exc}")
            continue
        name, description = meta.get("name"), meta.get("description")
        if name != path.parent.name or not NAME.match(str(name)) or len(str(name)) > 64:
            out.append(f"{where}: name {name!r} must be the directory name, lowercase-hyphenated")
        if not isinstance(description, str) or not 0 < len(description) <= MAX_DESCRIPTION:
            out.append(f"{where}: description missing or longer than {MAX_DESCRIPTION}")
        if unknown := set(meta) - {"name", "description"}:
            out.append(f"{where}: unexpected frontmatter keys {sorted(unknown)}")
        if not body.lstrip().startswith("# "):
            out.append(f"{where}: the body starts with a # title")
        linked = set()
        for file in sorted(path.parent.rglob("*.md")):
            for target in _LINK.findall(file.read_text()):
                resolved = (file.parent / target).resolve()
                if not resolved.exists():
                    out.append(f"{file.relative_to(SKILLS_DIR)}: broken link {target}")
                elif file == path:
                    linked.add(resolved)
            for example in sorted(set(_EXAMPLE.findall(file.read_text())) - _NOT_EXAMPLES):
                if not (EXAMPLES_DIR / example).is_file():
                    out.append(f"{file.relative_to(SKILLS_DIR)}: no example {example}")
        for resource in Skill(str(name), "", path.parent).resources:
            if resource.resolve() not in linked:
                out.append(f"{resource.relative_to(SKILLS_DIR)}: not linked from {SKILL_FILE}")
    return out
