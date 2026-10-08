"""Agent definitions of kernel-agent's roles (issue #176).

``agent/agents/<name>.md`` are Claude Code subagent files: frontmatter (``name``,
``description``, ``tools``, ``model``, optional ``maxTurns`` and ``skills``) and the role's
prompt as the body. One file per pipeline role (planner, kernel engineer, systems, native,
research, dossier, refactor, harness author, librarian) and per helper a session may
delegate to (:data:`HELPERS`: ``doc-lookup``, ``profile-analyst``, ``reviewer``). They serve
three ways:

* the session runner (``runner.run_agent``) looks up the definition of a session's role
  (:func:`for_session`: ``kernel-mlp-w2`` → ``kernel-engineer``) and passes the helpers its
  ``tools`` list as ``Agent(...)`` to the SDK as agent definitions (``agents=``), so the
  session can hand a documentation lookup, a profile analysis or a review to a subagent
  with its own context; the Skill tool and kernel-agent's skills come with every session;
* a coordinator (#174) starts any role uniformly from :func:`agent_definition` (and
  ``skills.for_role`` for the skills it loads first);
* ``kernel-agent install-claude-code`` copies them into a project's ``.claude/agents/`` for
  ``/optimize-model``.

A pipeline session's own system prompt still comes from ``prompts.py`` (it holds the run's
target, profile and budget); a definition's body is the role's prompt when it runs as a
subagent. ``model: inherit`` everywhere: the session's model (``--claude-model``) decides.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import AgentDefinition

from kernel_agent import skills
from kernel_agent.agent.web import DOC_TOOLS

AGENTS_DIR = Path(__file__).parent / "agent" / "agents"
#: Session role (the part of a session name before the first ``-``) → its definition.
ROLE_AGENTS = {
    "planner": "planner",
    "kernel": "kernel-engineer",
    "systems": "systems-engineer",
    "native": "native-engineer",
    "research": "researcher",
    "dossier": "dossier-researcher",
    "refactor": "refactor-engineer",
    "harness": "harness-author",
    "librarian": "librarian",
}
#: Subagents a role may delegate to (``Agent(...)`` in its ``tools``).
HELPERS = ("doc-lookup", "profile-analyst", "reviewer")
WEB_TOOLS = ("WebFetch", "WebSearch")


@dataclass(frozen=True)
class Role:
    """One agent definition (``agents/<name>.md``)."""

    name: str
    description: str
    prompt: str
    tools: tuple[str, ...]  # without the Agent(...) entry
    delegates: tuple[str, ...]  # the subagents its Agent(...) entry names
    model: str
    skills: tuple[str, ...]  # skill names (unqualified) preloaded when it runs as a subagent
    max_turns: int | None
    path: Path


def split_tools(value: str) -> list[str]:
    """``Read, Agent(a, b), Grep`` → ``["Read", "Agent(a, b)", "Grep"]`` (commas inside
    parentheses do not split)."""
    out, depth, current = [], 0, ""
    for char in value:
        depth += char == "("
        depth -= char == ")"
        if char == "," and depth == 0:
            out.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        out.append(current.strip())
    return out


def _role(path: Path) -> Role:
    meta, body = skills.frontmatter(path.read_text())
    tools, delegates = [], []
    for tool in split_tools(str(meta.get("tools") or "")):
        if tool.startswith("Agent(") and tool.endswith(")"):
            delegates += [d.strip() for d in tool[6:-1].split(",") if d.strip()]
        else:
            tools.append(tool)
    turns = meta.get("maxTurns")
    return Role(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description") or ""),
        prompt=body.strip() + "\n",
        tools=tuple(tools),
        delegates=tuple(delegates),
        model=str(meta.get("model") or "inherit"),
        skills=tuple(meta.get("skills") or ()),
        max_turns=turns if isinstance(turns, int) else None,
        path=path,
    )


@functools.cache
def definitions() -> dict[str, Role]:
    """Every agent definition by name, sorted."""
    return {r.name: r for r in map(_role, sorted(AGENTS_DIR.glob("*.md")))}


def role_of(session: str) -> str | None:
    """The role of a session name (``kernel-mlp`` → ``kernel``), or None."""
    role = session.split("-", 1)[0].lower()
    return role if role in ROLE_AGENTS else None


def for_session(session: str) -> Role | None:
    """The definition of a session's role (None for a name of no known role)."""
    role = role_of(session)
    return definitions().get(ROLE_AGENTS[role]) if role else None


def agent_definition(name: str, *, web: bool = True) -> AgentDefinition:
    """The SDK agent definition of ``name``: its tools (``Agent`` when it may delegate
    further; without the web tools when ``web`` is off), its skills qualified
    (``kernel-agent:<name>``, preloaded into the subagent's context) and its prompt."""
    role = definitions()[name]
    tools = [t for t in role.tools if web or t not in WEB_TOOLS]
    if role.delegates:
        tools.append("Agent")
    return AgentDefinition(
        description=role.description,
        prompt=role.prompt,
        tools=tools,
        model=role.model,
        skills=[skills.qualified(s) for s in role.skills] or None,
        maxTurns=role.max_turns,
    )


def subagents(session: str, *, web: bool = True) -> dict[str, AgentDefinition]:
    """The helpers the role of ``session`` may delegate to, as SDK agent definitions ({} for
    a role without any or an unknown session name)."""
    role = for_session(session)
    return {d: agent_definition(d, web=web) for d in role.delegates} if role else {}


def delegation_note(helpers: Mapping[str, AgentDefinition]) -> str:
    """The ``# Helpers`` section of a session's prompt ("" without helpers): which subagents
    it may hand work to and how (the Agent tool lists their descriptions)."""
    if not helpers:
        return ""
    names = ", ".join(f"`{name}`" for name in helpers)
    return f"""

# Helpers (the Agent tool)
Subagents you may delegate to: {names} (their descriptions are listed with the Agent tool).
Each works in its own context and returns a short report, so hand them what would fill
yours: a documentation question, a long profile or result history, a review of a candidate
before you spend an evaluation on it. They see nothing of this session: give each a
self-contained task (the question, the file paths, what the answer is for). Pass
`run_in_background: false` when your next step needs the answer. Their reports are advice:
only the evaluation tools decide what is correct and faster.
"""


def problems() -> list[str]:
    """What is wrong with the agent definitions ([] when nothing): every role and helper has
    one, names match file names, descriptions exist, delegates are helpers, skills exist,
    tools are known."""
    known_tools = {"Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill", *WEB_TOOLS}
    known_tools |= set(DOC_TOOLS)  # the doc library of every session (#177)
    out = []
    defs = definitions()
    for name in (*ROLE_AGENTS.values(), *HELPERS):
        if name not in defs:
            out.append(f"no agent definition {name}.md")
    for role in defs.values():
        where = role.path.name
        if role.path.stem != role.name or not skills.NAME.match(role.name):
            out.append(f"{where}: name {role.name!r} must be the file name")
        if not role.description or len(role.description) > skills.MAX_DESCRIPTION:
            out.append(f"{where}: description missing or too long")
        if not role.prompt.strip():
            out.append(f"{where}: empty prompt")
        if unknown := set(role.tools) - known_tools:
            out.append(f"{where}: unknown tools {sorted(unknown)}")
        if stray := set(role.delegates) - set(HELPERS):
            out.append(f"{where}: delegates to {sorted(stray)}, which are not helpers")
        for skill in role.skills:
            if skill not in skills.index():
                out.append(f"{where}: no skill {skill}")
    return out
