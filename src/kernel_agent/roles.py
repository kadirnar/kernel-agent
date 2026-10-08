"""kernel-agent's roles: the registry (#181) and the agent definitions (#176).

**The registry** (:data:`REGISTRY`, a :class:`RoleSpec` per role) holds what each kind of
session gets: its model and effort (``--role-model`` / ``--role-effort`` over the defaults
of ``config.ROLE_MODELS`` / ``ROLE_EFFORTS``, docs/MULTIAGENT.md §3.12.1), its turns, its
built-in and kernel-agent tools, whether it needs the GPU, the helpers it may delegate to
and the skills it loads first. ``Orchestrator._agent`` builds every session from it
(:func:`session_config`: model, effort, turns; the tools), ``runner.run_agent`` its SDK
options (:func:`options_for`), ``program.py`` its ``program.md`` sections and the report its
usage per role (:func:`usage_lines`); a coordinator (#174) will fill its slots by
``needs_gpu`` and ``max_concurrent``. A session's role is the part of its name before the
first ``-`` (:func:`role_of`: ``kernel-mlp-w2`` → ``kernel``).

**The agent definitions**: ``agent/agents/<name>.md`` are Claude Code subagent files:
frontmatter (``name``, ``description``, ``tools``, ``model``, optional ``maxTurns`` and
``skills``) and the role's prompt as the body. One file per pipeline role (planner, kernel
engineer, systems, native, research, dossier, refactor, harness author, librarian) and per
helper a session may delegate to (:data:`HELPERS`: ``doc-lookup``, ``profile-analyst``,
``compile-triage``, ``reviewer``; read-only, :data:`HELPER_TOOLS`). They serve three ways:

* the session runner (``runner.run_agent``) looks up the definition of a session's role
  (:func:`for_session`: ``kernel-mlp-w2`` → ``kernel-engineer``) and passes the helpers its
  ``tools`` list as ``Agent(...)`` to the SDK as agent definitions (``agents=``, with each
  helper's model and effort from the registry), so the session can hand a documentation
  lookup, a profile analysis, a long compiler error or a review to a subagent with its own
  context (:func:`delegation_note` says when that pays, #186); the Skill tool
  and kernel-agent's skills come with every session;
* a coordinator (#174) starts any role uniformly from :func:`agent_definition` (and
  ``skills.for_role`` for the skills it loads first);
* ``kernel-agent install-claude-code`` copies them into a project's ``.claude/agents/`` for
  ``/optimize-model``.

A pipeline session's own prompt still comes from ``prompts.py`` (it holds the run's target,
profile and budget); a definition's body is the role's prompt when it runs as a subagent.
The files say ``model: inherit`` (in the user's Claude Code the user's session decides); in
kernel-agent's own sessions the registry does.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import AgentDefinition

from kernel_agent import skills
from kernel_agent.agent.web import DOC_TOOLS
from kernel_agent.config import EFFORTS, INHERIT, ROLE_EFFORTS, ROLE_MODELS, OptimizeConfig

AGENTS_DIR = Path(__file__).parent / "agent" / "agents"
#: The built-in tools of a session that is not restricted to fewer (``RoleSpec.builtin_tools``).
BASE_TOOLS = ("Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite")
READ_TOOLS = ("Read", "Glob", "Grep")
WEB_TOOLS = ("WebFetch", "WebSearch")
#: The only tools a helper may have (docs/MULTIAGENT.md §3.12.3): it reads, looks up and
#: answers; it never writes, runs commands or GPU work, so it needs no ownership or GPU
#: accounting and makes no evaluation the engineer does not see.
HELPER_TOOLS = (*READ_TOOLS, "Skill", *WEB_TOOLS, *DOC_TOOLS)
#: kernel-agent's evaluation tools of the kernel and the end-to-end sessions (short names).
KERNEL_TOOLS = ("evaluate_candidate", "sweep_candidate", "best_result")
E2E_TOOLS = ("evaluate_e2e", "run_info")
#: An agent's own GPU scripts through the GPU job queue (#185): every role whose Bash may need
#: the GPU (with ``--agent-gpu tool`` its Bash commands see none).
DEV_TOOLS = ("run_on_gpu",)


@dataclass(frozen=True)
class RoleSpec:
    """One role: what every session of it gets (docs/MULTIAGENT.md §3.13).

    ``agent``: its agent definition (``agents/<agent>.md``; None: none yet); ``helper``: a
    subagent sessions delegate to, never a session of its own; ``max_turns``: its turns
    (None: ``--max-turns`` × ``session_factor``); ``session_factor``: how many times longer
    than the others its sessions are (turns, and without ``--native-minutes`` the native
    session's minutes); ``builtin_tools``: the only built-in tools it has (None:
    :data:`BASE_TOOLS`; the web tools come with ``--allow-web``); ``mcp_tools``:
    kernel-agent's tools it may call (short names; the doc library's come with every
    session); ``needs_gpu``: its tools run GPU work (a coordinator gives free slots to the
    roles that do not while the GPU is saturated); ``max_concurrent``: its sessions at once
    under a coordinator (#174 §3.2; None: no cap); ``program``: ``program.md`` has a
    ``## <name>`` section for it."""

    name: str
    agent: str | None = None
    helper: bool = False
    max_turns: int | None = None
    session_factor: int = 1
    builtin_tools: tuple[str, ...] | None = None
    mcp_tools: tuple[str, ...] = ()
    needs_gpu: bool = False
    max_concurrent: int | None = None
    program: bool = True

    @property
    def model(self) -> str:
        """Its default model (``config.ROLE_MODELS``; ``inherit``: ``--claude-model``)."""
        return ROLE_MODELS.get(self.name, INHERIT)

    @property
    def effort(self) -> str | None:
        """Its default effort (``config.ROLE_EFFORTS``; ``inherit``: ``--effort``)."""
        return ROLE_EFFORTS.get(self.name, INHERIT)

    @property
    def subagents(self) -> tuple[str, ...]:
        """The helpers it may delegate to: the ``Agent(...)`` entry of its definition."""
        role = definitions().get(self.agent) if self.agent else None
        return role.delegates if role else ()

    def skills(
        self, *, backends: Iterable[str] = (), precision: str | None = None, quality: str = "exact"
    ) -> list[str]:
        """The skills a session of it loads first (``skills.for_role``; its prompt names
        them). Every session enables all of kernel-agent's skills (``options_for``): one
        listing for every role, so the listing never breaks a shared prompt-cache prefix."""
        return skills.for_role(self.name, backends=backends, precision=precision, quality=quality)


#: Every role: the pipeline's sessions (in ``program.md``'s order), then the helpers.
REGISTRY: dict[str, RoleSpec] = {
    spec.name: spec
    for spec in (
        RoleSpec("planner", "planner"),
        RoleSpec(
            "kernel",
            "kernel-engineer",
            mcp_tools=(*KERNEL_TOOLS, *DEV_TOOLS),
            needs_gpu=True,
            max_concurrent=4,
        ),
        RoleSpec(
            "systems",
            "systems-engineer",
            mcp_tools=(*E2E_TOOLS, *DEV_TOOLS),
            needs_gpu=True,
            max_concurrent=1,
        ),
        RoleSpec(
            "native",
            "native-engineer",
            session_factor=3,
            mcp_tools=(*KERNEL_TOOLS, *E2E_TOOLS, *DEV_TOOLS),
            needs_gpu=True,
            max_concurrent=1,
        ),
        RoleSpec(
            "harness", "harness-author", mcp_tools=("check_harness", *DEV_TOOLS), needs_gpu=True
        ),
        RoleSpec(
            "research",
            "researcher",
            builtin_tools=(*READ_TOOLS, "Write"),
            mcp_tools=("best_result",),
        ),
        RoleSpec(
            "refactor",
            "refactor-engineer",
            builtin_tools=(*READ_TOOLS, "Write", "Edit"),
            mcp_tools=("verify_rewrite",),
            needs_gpu=True,
        ),
        # short and cheap: it delays a target's first engineer session by minutes at most
        RoleSpec(
            "dossier", "dossier-researcher", max_turns=20, builtin_tools=(*READ_TOOLS, "Write")
        ),
        RoleSpec("librarian", "librarian", max_turns=8),
        RoleSpec("critic", max_turns=1, builtin_tools=READ_TOOLS, program=False),  # #174 PR 8
        RoleSpec("critic-escalation", max_turns=1, builtin_tools=READ_TOOLS, program=False),
        RoleSpec("doc-lookup", "doc-lookup", helper=True, program=False),
        RoleSpec("profile-analyst", "profile-analyst", helper=True, program=False),
        RoleSpec("compile-triage", "compile-triage", helper=True, program=False),  # #186
        RoleSpec("reviewer", "reviewer", helper=True, program=False),
    )
}
#: The role of a session name of no known role: the run's model, effort and tools.
NO_ROLE = RoleSpec("", program=False)
#: Session role → its agent definition.
ROLE_AGENTS = {r.name: r.agent for r in REGISTRY.values() if r.agent and not r.helper}
#: Subagents a role may delegate to (``Agent(...)`` in its ``tools``).
HELPERS = tuple(r.name for r in REGISTRY.values() if r.helper)
#: The roles with a ``## <role>`` section in ``program.md`` (``program.ROLES``).
PROGRAM_ROLES = tuple(r.name for r in REGISTRY.values() if r.program)


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
    """The role of a session name or label (``kernel-mlp`` and ``systems#3`` → ``kernel``
    and ``systems``), or None."""
    role = session.split("#", 1)[0].split("-", 1)[0].lower()
    return role if role in REGISTRY and not REGISTRY[role].helper else None


def get(role: str | None) -> RoleSpec:
    """The registry entry of a role or helper (:data:`NO_ROLE` for None or an unknown name)."""
    return REGISTRY.get(role or "", NO_ROLE)


def for_session(session: str) -> Role | None:
    """The definition of a session's role (None for a name of no known role)."""
    agent = get(role_of(session)).agent
    return definitions().get(agent) if agent else None


def agent_definition(
    name: str, *, web: bool = True, cfg: OptimizeConfig | None = None
) -> AgentDefinition:
    """The SDK agent definition of ``name``: its tools (``Agent`` when it may delegate
    further; without the web tools when ``web`` is off), its skills qualified
    (``kernel-agent:<name>``, preloaded into the subagent's context), its prompt, and its
    model and effort from the registry (``cfg``'s ``role_models`` / ``role_efforts``, else
    the defaults; ``inherit``: the session's)."""
    role = definitions()[name]
    tools = [t for t in role.tools if web or t not in WEB_TOOLS]
    if role.delegates:
        tools.append("Agent")
    model = (cfg.role_models if cfg else ROLE_MODELS).get(name, role.model)
    effort = (cfg.role_efforts if cfg else ROLE_EFFORTS).get(name, INHERIT)
    return AgentDefinition(
        description=role.description,
        prompt=role.prompt,
        tools=tools,
        model=model,
        skills=[skills.qualified(s) for s in role.skills] or None,
        maxTurns=role.max_turns,
        effort=None if effort == INHERIT else effort,  # type: ignore[arg-type]
    )


def subagents(
    session: str, *, web: bool = True, cfg: OptimizeConfig | None = None
) -> dict[str, AgentDefinition]:
    """The helpers the role of ``session`` (a session name or a role) may delegate to, as SDK
    agent definitions ({} for a role without any or an unknown session name)."""
    spec = get(role_of(session))
    return {d: agent_definition(d, web=web, cfg=cfg) for d in spec.subagents}


def delegation_note(helpers: Mapping[str, AgentDefinition]) -> str:
    """The ``# Helpers`` section of a session's prompt ("" without helpers): which subagents
    it may hand work to, how, and when that pays (:data:`WHEN_TO_DELEGATE` of those it has,
    #186; the Agent tool lists their descriptions). The same for every session of a role:
    part of its prompt-cache prefix."""
    if not helpers:
        return ""
    names = ", ".join(f"`{name}`" for name in helpers)
    when = "".join(
        f"* `{name}`: {WHEN_TO_DELEGATE[name]}\n" for name in helpers if name in WHEN_TO_DELEGATE
    )
    return f"""

# Helpers (the Agent tool)
Subagents you may delegate to: {names} (their descriptions are listed with the Agent tool).
Each works in its own context and returns a short report, so hand them what would fill
yours: a documentation question, a long profile or result history, a review of a candidate
before you spend an evaluation on it. They see nothing of this session: give each a
self-contained task (the question, the file paths, what the answer is for). Pass
`run_in_background: false` when your next step needs the answer. Their reports are advice:
only the evaluation tools decide what is correct and faster. Helpers only read: they never
edit files, run commands or GPU work.

When delegating pays (a helper costs a session of its own: do yourself what takes one or
two tool calls):
{when}"""


#: When each helper is worth its session (:func:`delegation_note`).
WHEN_TO_DELEGATE = {
    "doc-lookup": "an API, instruction, layout or limit you have not verified that needs "
    "more than one `doc_search` + `doc_read` (several pages, a version or architecture "
    "detail); a doc id you already have: `doc_read` it yourself.",
    "profile-analyst": 'a `profile=true` / `"ncu"` result: give it the result\'s '
    "`profile.file` (the full tables) and your question (which kernel to attack, what bounds "
    "it); also a long `results.jsonl`, `profile/summary.md` or ceilings table.",
    "compile-triage": "a build or runtime error longer than a screen (about 40 lines) or "
    "one you cannot place at once: write the full output to a file (`... > build.log 2>&1; "
    "tail -n 20 build.log`) and give it the candidate and the log's path, or the "
    "evaluation's `error`; a one-line error you understand: fix it yourself.",
    "reviewer": "before the full evaluation of a large change, or a failed check you cannot "
    "explain: give it the candidate, the reference and the failed result.",
}


def problems() -> list[str]:
    """What is wrong with the agent definitions ([] when nothing): every role and helper has
    one, names match file names, descriptions exist, delegates are helpers, skills exist,
    tools are known, helpers are read-only (:data:`HELPER_TOOLS`) and the prompt says when
    each pays."""
    known_tools = {"Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill", *WEB_TOOLS}
    known_tools |= set(DOC_TOOLS)  # the doc library of every session (#177)
    out = []
    defs = definitions()
    for name in (*ROLE_AGENTS.values(), *HELPERS):
        if name not in defs:
            out.append(f"no agent definition {name}.md")
    if names := set(HELPERS) ^ set(WHEN_TO_DELEGATE):  # the prompt says when each one pays
        out.append(f"roles.WHEN_TO_DELEGATE and the helpers differ in {sorted(names)}")
    for table in ("ROLE_MODELS", "ROLE_EFFORTS"):  # config.py's defaults: one per role
        if names := set(REGISTRY) ^ set(ROLE_MODELS if table == "ROLE_MODELS" else ROLE_EFFORTS):
            out.append(f"config.{table} and the registry differ in {sorted(names)}")
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
        if role.name in HELPERS and (writes := set(role.tools) - set(HELPER_TOOLS)):
            out.append(f"{where}: a helper is read-only, not {sorted(writes)}")
        if role.name in HELPERS and role.delegates:
            out.append(f"{where}: a helper does not delegate")
        for skill in role.skills:
            if skill not in skills.index():
                out.append(f"{where}: no skill {skill}")
    return out


# ------------------------------------------------------------------ sessions (#181)

#: The prefix of kernel-agent's MCP tools in a session (``agent/tools.py`` ``SERVER_NAME``).
MCP_PREFIX = "mcp__ka__"


def model_for(role: str | None, cfg: OptimizeConfig) -> str:
    """The model of a session of ``role``: ``cfg.role_models`` (``inherit``, or a role it
    does not name: ``cfg.claude_model``)."""
    model = cfg.role_models.get(role or "", get(role).model)
    return cfg.claude_model if model == INHERIT else model


def effort_for(role: str | None, cfg: OptimizeConfig) -> str | None:
    """The effort of a session of ``role``: ``cfg.role_efforts`` (``inherit``: ``cfg.effort``;
    None: none set)."""
    effort = cfg.role_efforts.get(role or "", get(role).effort)
    return cfg.effort if effort == INHERIT else effort


def max_turns_for(role: str | None, cfg: OptimizeConfig) -> int:
    """The turns of a session of ``role``: its own, else ``--max-turns`` × its factor."""
    spec = get(role)
    return spec.max_turns or cfg.max_turns_per_agent * spec.session_factor


def session_config(role: str | None, cfg: OptimizeConfig) -> OptimizeConfig:
    """The run's ``cfg`` for one session of ``role``: its model, effort and turns
    (``claude_model``, ``effort``, ``max_turns_per_agent``). Apply it once: a session config
    is not a run config (its turns are multiplied already)."""
    return dataclasses.replace(
        cfg,
        claude_model=model_for(role, cfg),
        effort=effort_for(role, cfg),
        max_turns_per_agent=max_turns_for(role, cfg),
    )


def mcp_tools(role: str | None) -> list[str]:
    """kernel-agent's tools a session of ``role`` may call, as the session names them."""
    return [MCP_PREFIX + name for name in get(role).mcp_tools]


def options_for(
    role: str | None,
    cfg: OptimizeConfig,
    *,
    tools: Iterable[str] | None = None,
    mcp: Iterable[str] | None = None,
    extra_tools: Iterable[str] = (),
) -> dict[str, Any]:
    """The ``ClaudeAgentOptions`` fields of a session of ``role`` that the registry decides
    (docs/MULTIAGENT.md §3.13): ``model``, ``effort`` and ``max_turns`` (``cfg``: the
    session's, :func:`session_config`); the tools (built-in: ``tools``, else the role's, plus
    the web tools with ``cfg.allow_web``; kernel-agent's: ``mcp``, else the role's; the doc
    library's always; a restricted session also has Skill, and Agent when it has helpers);
    its helpers (``agents``); kernel-agent's skills by explicit plugin path (#176) and no
    filesystem settings (``setting_sources=[]``, #126). Tools, skills and helpers are the
    same for every session of a role: a byte-identical prefix for the prompt cache."""
    spec = get(role)
    restricted = list(tools) if tools is not None else spec.builtin_tools
    builtin = list(BASE_TOOLS if restricted is None else restricted)
    builtin += (list(WEB_TOOLS) if cfg.allow_web else []) + list(extra_tools)
    helpers = subagents(role or "", web=cfg.allow_web, cfg=cfg)
    out: dict[str, Any] = {
        "model": cfg.claude_model,
        "max_turns": cfg.max_turns_per_agent,
        "allowed_tools": builtin + list(mcp_tools(role) if mcp is None else mcp) + DOC_TOOLS,
        "setting_sources": [],  # no user/project settings, hooks or CLAUDE.md files (#126)
        "plugins": [skills.plugin()],  # kernel-agent's skills, by explicit path (#176)
        "skills": skills.session_skills(),  # those (+ Workflow's): no other bundled or user skill
        "agents": helpers or None,
    }
    if cfg.effort:
        out["effort"] = cfg.effort
    if restricted is not None:  # a restricted session: no other built-in tool exists at all
        out["tools"] = [*builtin, "Skill", *(["Agent"] if helpers else [])]
    return out


def parse_settings(values: Iterable[str], *, effort: bool = False) -> dict[str, str | None]:
    """``--role-model ROLE=MODEL`` / ``--role-effort ROLE=LEVEL`` values → ``{role: value}``
    (``ValueError`` says what is wrong). A role is any name of the registry; a level is one
    of ``config.EFFORTS``, ``inherit`` or ``none`` (no effort set)."""
    out: dict[str, str | None] = {}
    for value in values:
        role, sep, setting = (part.strip() for part in value.partition("="))
        if not sep or not role or not setting:
            raise ValueError(f"{value!r}: expected ROLE=VALUE")
        if role not in REGISTRY:
            raise ValueError(f"{role!r}: not a role ({', '.join(REGISTRY)})")
        if effort and setting not in (*EFFORTS, INHERIT, "none"):
            raise ValueError(f"{setting!r}: not an effort ({', '.join(EFFORTS)}, inherit, none)")
        out[role] = None if effort and setting == "none" else setting
    return out


def changed(table: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """The entries of ``table`` that differ from ``defaults``: what ``--role-model`` /
    ``--role-effort`` set, which a resumed run takes over its own (``improve``)."""
    return {k: v for k, v in table.items() if k not in defaults or defaults[k] != v}


# ------------------------------------------------------------------ usage (#181)

#: The token counts of ``ResultMessage.usage`` a session records (``costs.json`` ``usage``):
#: input after the last cache breakpoint, written to the cache, read from it, output.
USAGE_KEYS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def usage_of(raw: Mapping[str, Any] | None) -> dict[str, int]:
    """The :data:`USAGE_KEYS` of an SDK ``usage`` dict ({} without one)."""
    if not raw:
        return {}
    return {key: int(raw.get(key) or 0) for key in USAGE_KEYS}


def add_usage(*usages: Mapping[str, int] | None) -> dict[str, int]:
    """The sum of usage dicts ({} when none has a count)."""
    out: dict[str, int] = {}
    for usage in usages:
        for key, count in (usage or {}).items():
            if key in USAGE_KEYS:
                out[key] = out.get(key, 0) + int(count or 0)
    return out


def cache_hit_rate(usage: Mapping[str, int] | None) -> float | None:
    """The share of the input tokens read from the prompt cache (None without input)."""
    usage = usage or {}
    read = usage.get("cache_read_input_tokens", 0)
    total = read + usage.get("cache_creation_input_tokens", 0) + usage.get("input_tokens", 0)
    return read / total if total else None


def usage_by_role(costs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``costs.json`` summed per role (a session's ``role``, else its label's): sessions, USD,
    the models and efforts it ran on, its usage and its first requests' usage."""
    out: dict[str, dict[str, Any]] = {}
    for label, cost in costs.items():
        if not isinstance(cost, dict):
            continue
        role = cost.get("role") or role_of(label) or "other"
        row = out.setdefault(
            role, {"sessions": 0, "usd": 0.0, "models": {}, "efforts": {}, "usage": {}, "first": {}}
        )
        row["sessions"] += 1
        row["usd"] += float(cost.get("usd") or 0.0)
        for key, value in (("models", cost.get("model")), ("efforts", cost.get("effort"))):
            if value:
                row[key][value] = row[key].get(value, 0) + 1
        row["usage"] = add_usage(row["usage"], cost.get("usage"))
        row["first"] = add_usage(row["first"], cost.get("first_usage"))
    return out


def _mtok(count: int | None) -> str:
    return f"{count / 1e6:.2f}" if count else "-"


def _pct(share: float | None) -> str:
    return f"{share * 100:.0f} %" if share is not None else "-"


def usage_lines(costs: Mapping[str, Any]) -> list[str]:
    """The report's usage per role ([] before any session recorded tokens): sessions, model,
    effort, USD, input tokens and the share of them read from the prompt cache, over whole
    sessions and over their first requests (what a session reads of a cache that other
    sessions wrote: the stable prefix of its role, #181), and output tokens."""
    by_role = usage_by_role(costs)
    if not any(row["usage"] for row in by_role.values()):
        return []
    lines = [
        "",
        "### Usage per role",
        "",
        "Input: uncached + written to + read from the prompt cache. Cached: the share read "
        "from the cache; first request: the same over each session's first request (what it "
        "reads of the prefix its role's earlier sessions wrote).",
        "",
        "| role | sessions | model | effort | $ | $ / session | input Mtok | cached "
        "| first request cached | output Mtok |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for role, row in sorted(by_role.items(), key=lambda item: -item[1]["usd"]):
        usage, usd = row["usage"], row["usd"]
        total = sum(usage.get(key, 0) for key in USAGE_KEYS[:3])
        models = ", ".join(f"`{m}`" for m in row["models"]) or "-"
        efforts = ", ".join(str(e) for e in row["efforts"]) or "-"
        lines.append(
            f"| {role} | {row['sessions']} | {models} | {efforts} | {usd:.2f} "
            f"| {usd / row['sessions']:.2f} | {_mtok(total)} | {_pct(cache_hit_rate(usage))} "
            f"| {_pct(cache_hit_rate(row['first']))} | {_mtok(usage.get('output_tokens'))} |"
        )
    return lines
