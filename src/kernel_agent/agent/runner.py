"""Thin wrapper around ``claude_agent_sdk.query`` with logging and cost tracking."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    HookJSONOutput,
    HookMatcher,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from claude_agent_sdk.types import HookEvent

from kernel_agent import hygiene, roles, sessions
from kernel_agent.agent import auth, web
from kernel_agent.config import OptimizeConfig

#: ``--agent-gpu`` (#185): the agents' Bash commands see the GPU (``bash``, one session at a
#: time) or not, and run their GPU scripts through the GPU job queue (``tool``: ``run_on_gpu``)
GPU_BASH = "bash"
GPU_TOOL = "tool"
GPU_MODES = (GPU_TOOL, GPU_BASH)
BASE_TOOLS = list(roles.BASE_TOOLS)
READ_TOOLS = list(roles.READ_TOOLS)
WEB_TOOLS = list(roles.WEB_TOOLS)
WRITE_TOOLS = "Write|Edit|MultiEdit|NotebookEdit"
#: Set in every session (#126): no Claude Code auto memory, which would read and write the
#: user's own ``~/.claude/projects/<repo>/memory/`` (lessons belong in library.py), and no
#: claude.ai connectors of the login (they act on the user's account).
SESSION_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
INSTRUCTION_FILES = ("CLAUDE.md", "CLAUDE.local.md", "AGENTS.md")
#: Called with every message of the sessions this context runs (a coordinator's listener,
#: ``coordinator.py``: a role's first streamed message, the rate-limit events); see
#: :func:`listening`. The session's task copies the context, so each session has its own.
_listener: ContextVar[Callable[[Any], None] | None] = ContextVar(
    "kernel_agent_session_listener", default=None
)


@contextlib.contextmanager
def listening(listener: Callable[[Any], None]) -> Iterator[None]:
    """The agent sessions started in this context hand every message to ``listener``."""
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


def heard(message: Any) -> None:
    """Hand ``message`` of a session to the listener of this context, if any (a simulated
    session, ``dryrun.py``, calls it too); a listener's error never ends the session."""
    if (listener := _listener.get()) is not None:
        with contextlib.suppress(Exception):
            listener(message)


@dataclass
class AgentResult:
    name: str
    text: str = ""
    structured: Any = None
    cost_usd: float = 0.0
    turns: int = 0
    seconds: float = 0.0
    is_error: bool = False
    session_id: str | None = None
    tool_calls: dict[str, int] = field(default_factory=dict)
    timed_out: bool = False
    api_key_source: str | None = None  # the init message's; "none" = no API key in use
    usage_limit: auth.UsageLimit | None = None  # the session stopped at a usage limit
    web: list[dict[str, Any]] = field(default_factory=list)  # WebFetch / WebSearch / doc_*
    # Tokens (#181, roles.USAGE_KEYS): the session's main thread (ResultMessage.usage) and
    # its first request (the first AssistantMessage's: what it read of a cache other sessions
    # wrote), the model it ran on, and per model the tokens and USD of everything it ran
    # (helpers, Claude Code's own calls; ResultMessage.model_usage)
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    first_usage: dict[str, int] = field(default_factory=dict)
    model_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    # In-session helpers (#186): ``tool_calls`` and ``turns`` are the main thread's; per
    # helper (the Agent call's ``subagent_type``) its delegations, tool calls (its messages:
    # ``parent_tool_use_id`` set), seconds and models (:class:`Delegations`). Their tokens
    # and USD are in ``cost_usd`` and ``model_usage``.
    subagents: dict[str, dict[str, Any]] = field(default_factory=dict)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _brief(block_input: Any, limit: int = 110) -> str:
    if isinstance(block_input, dict):
        keys = ("candidate", "file_path", "command", "pattern", "url", "transforms", "skill")
        for key in (*keys, "subagent_type"):
            if key in block_input:
                return f"{key}={str(block_input[key])[:limit]}"
    return str(block_input)[:limit]


def agent_env(
    extra: dict[str, str], mode: str = auth.AUTO, *, gpu: str = GPU_BASH
) -> dict[str, str]:
    """Environment for the agent's Bash tool: same Python env + toolchain variables; with
    ``mode`` subscription, the API key / cloud provider variables blanked (auth.py). With
    ``gpu`` :data:`GPU_TOOL` (``--agent-gpu tool``, the default with several sessions) its
    commands see no GPU (``CUDA_VISIBLE_DEVICES=""``; builds still work: the architectures
    come from ``TORCH_CUDA_ARCH_LIST``) and run their GPU scripts with ``run_on_gpu``, through
    the GPU job queue. With clean timing on (``hygiene.py``) its builds get ``MAX_JOBS``."""
    venv_bin = str(Path(sys.executable).parent)
    env = {
        "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONUNBUFFERED": "1",
        **extra,
        **hygiene.background_env(),
    }
    if gpu == GPU_TOOL:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if "VIRTUAL_ENV" not in os.environ and (Path(venv_bin).parent / "pyvenv.cfg").exists():
        env["VIRTUAL_ENV"] = str(Path(venv_bin).parent)
    return auth.scrub(env) if mode == auth.SUBSCRIPTION else env


def write_guard(
    writable: list[Path],
    cwd: Path,
    *,
    roots: list[Path] | None = None,
    excluded: list[Path] | None = None,
) -> dict[HookEvent, list[HookMatcher]]:
    """PreToolUse hook that lets the file-writing tools touch only ``writable`` and the
    files under ``roots`` that are not under ``excluded`` (a session's own directory among
    concurrent sessions, docs/MULTIAGENT.md §3.6: the systems agent's ``transforms/`` without
    the native agent's ``transforms/native/``).

    A hook, not ``can_use_tool``: with ``bypassPermissions`` the CLI never asks."""
    allowed = {p.resolve() for p in writable}
    tops = [p.resolve() for p in roots or []]
    outs = [p.resolve() for p in excluded or []]

    def permitted(target: Path) -> bool:
        if target in allowed:
            return True
        inside = any(target.is_relative_to(top) for top in tops)
        return inside and not any(target.is_relative_to(out) for out in outs)

    where = [str(p) for p in sorted(allowed)] + [f"{p}/" for p in tops]
    if outs:
        where[-1] += " (not " + ", ".join(f"{p}/" for p in outs) + ")"

    async def guard(data: Any, tool_use_id: str | None, context: Any) -> HookJSONOutput:
        tool_input = data.get("tool_input") or {}
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if path and permitted(_resolve(cwd, path)):
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "this session may write only " + ", ".join(where),
            }
        }

    return {"PreToolUse": [HookMatcher(matcher=WRITE_TOOLS, hooks=[guard])]}


def claude_files_guard(cwd: Path) -> HookMatcher:
    """PreToolUse hook of every session (#126): the file-writing tools may not touch Claude
    Code's instruction files (:data:`INSTRUCTION_FILES`, anywhere) or its config directory
    (``~/.claude``: settings, auto memory, skills), which later sessions, the user's own
    among them, would load. With auto memory off, a session asked to remember something
    wrote the repository's ``CLAUDE.md`` instead. Bash is not covered."""
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    config = config.expanduser().resolve()

    async def guard(data: Any, tool_use_id: str | None, context: Any) -> HookJSONOutput:
        tool_input = data.get("tool_input") or {}
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        target = _resolve(cwd, path) if path else None
        if target is None or not (
            target.name in INSTRUCTION_FILES or target.is_relative_to(config)
        ):
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"{target}: Claude Code's instruction and config "
                "files are off limits; keep notes in the run directory",
            }
        }

    return HookMatcher(matcher=WRITE_TOOLS, hooks=[guard])


def _resolve(cwd: Path, path: str) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else cwd / p).resolve()


#: The tool a session delegates to a helper with (Claude Code's ``Agent``, once ``Task``).
AGENT_TOOLS = ("Agent", "Task")


class Delegations:
    """The helpers of one session's messages (#186): a main-thread ``Agent`` call names its
    helper (``subagent_type``), the helper's own messages carry that call's id as their
    ``parent_tool_use_id``, and the call's result says how it went. :meth:`see` keeps per
    helper (``AgentResult.subagents``) its delegations (``calls``), its tool calls
    (``tools``, counted by the runner), its seconds, the models it ran on and the
    delegations that did not complete (``failed``). Its tokens and USD are in the
    session's: ``ResultMessage.total_cost_usd`` and ``model_usage`` include them (checked
    against the real CLI, tests/test_subagents.py)."""

    def __init__(self) -> None:
        self.helpers: dict[str, str] = {}  # an Agent call's id -> its helper

    @staticmethod
    def _row(out: dict[str, dict[str, Any]], helper: str) -> dict[str, Any]:
        return out.setdefault(helper, {"calls": 0, "tools": {}, "seconds": 0.0, "models": []})

    def see(self, message: Message, out: dict[str, dict[str, Any]]) -> str | None:
        """The helper whose message ``message`` is (None: the main thread's or not an
        assistant message), counted into ``out``."""
        if isinstance(message, UserMessage) and message.parent_tool_use_id is None:
            done = message.tool_use_result if isinstance(message.tool_use_result, dict) else {}
            for block in message.content if isinstance(message.content, list) else []:
                if not isinstance(block, ToolResultBlock) or block.tool_use_id not in self.helpers:
                    continue
                row = self._row(out, self.helpers[block.tool_use_id])
                row["seconds"] = round(row["seconds"] + (done.get("totalDurationMs") or 0) / 1e3, 1)
                if (model := done.get("resolvedModel")) and model not in row["models"]:
                    row["models"].append(model)
                if block.is_error or done.get("status") not in (
                    None,
                    "completed",
                    "async_launched",
                ):
                    row["failed"] = row.get("failed", 0) + 1
            return None
        if not isinstance(message, AssistantMessage):
            return None
        if message.parent_tool_use_id is None:
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name in AGENT_TOOLS:
                    helper = str(block.input.get("subagent_type") or "general-purpose")
                    self.helpers[block.id] = helper
                    self._row(out, helper)["calls"] += 1
            return None
        helper = self.helpers.get(message.parent_tool_use_id, "subagent")
        self._row(out, helper)
        return helper


async def run_agent(
    name: str,
    *,
    prompt: str,
    system_append: str,
    cwd: Path,
    cfg: OptimizeConfig,
    mcp_server: Any,
    mcp_tools: list[str],
    env: dict[str, str],
    log_dir: Path,
    add_dirs: list[Path] | None = None,
    output_format: dict[str, Any] | None = None,
    extra_tools: list[str] | None = None,
    result: AgentResult | None = None,
    tools: list[str] | None = None,
    writable: list[Path] | None = None,
    resume: str | None = None,
    role: str | None = None,
    roots: list[Path] | None = None,
    excluded: list[Path] | None = None,
) -> AgentResult:
    """Run one agent session to completion.

    ``result`` is filled in place while messages arrive, so a caller that
    cancels the session (e.g. a timeout) still has its session id, turns and
    tool calls so far. Cancelling terminates the Claude Code subprocess.
    ``role`` (default: the one of ``name``, ``roles.role_of``) decides the session's tools
    and helpers (``roles.options_for``; ``cfg``: the session's config, whose model, effort
    and turns it runs on); ``tools`` replaces the role's built-in tools (None:
    :data:`BASE_TOOLS` for most roles) as the ones the session has at all; with
    ``writable`` the file-writing tools may touch only those files, and with ``roots`` only
    the files under them that are not under ``excluded`` (:func:`write_guard`).

    A session that stops at a usage limit returns with ``usage_limit`` set instead of
    raising (auth.py); ``resume`` (its session id) continues it, and the USD, tokens, turns
    and seconds of ``result`` then add up over both runs. :class:`~kernel_agent.agent.auth.
    AuthError` stops a session whose API key source ``cfg.auth`` does not allow.
    """
    role = roles.role_of(name) if role is None else role
    fields = roles.options_for(role, cfg, tools=tools, mcp=mcp_tools, extra_tools=extra_tools or ())
    helpers = fields["agents"] or {}  # the role's delegates (#176)
    # clean timing (hygiene.py): the CLI, found by this mark, moves off the timing cores
    mark = {hygiene.SESSION_ENV: f"{os.getpid()}-{name}"} if hygiene.current() else {}
    cli: list[int] = []  # its pid, once moved
    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_append + roles.delegation_note(helpers),
        },
        cwd=str(cwd),
        add_dirs=[str(d) for d in (add_dirs or [])],
        mcp_servers={"ka": mcp_server},
        permission_mode=cfg.permission_mode,  # type: ignore[arg-type]
        max_budget_usd=cfg.budget_usd_per_agent,
        env={**env, **SESSION_ENV, **mark},
        output_format=output_format,
        **fields,  # model, effort, turns, tools, helpers, skills, no settings (roles.py)
    )
    guarded = writable is not None or roots is not None
    hooks = write_guard(writable or [], cwd, roots=roots, excluded=excluded) if guarded else {}
    hooks.setdefault("PreToolUse", []).append(claude_files_guard(cwd))
    options.hooks = hooks
    tracker = sessions.current()  # the session's states (sessions.py, #184): its tool spans
    if tracker is not None:
        tracker.hooks(hooks)
    if resume:
        options.resume = resume

    result = result or AgentResult(name=name)
    lookups = web.Lookups(result.web, cfg.web_domains, cfg.allow_web)  # + the doc library
    if cfg.allow_web:  # WebFetch to documentation domains only, every lookup recorded
        hooks["PreToolUse"].append(lookups.guard())
    result.is_error, result.usage_limit = False, None
    usd, turns_before, seconds = result.cost_usd, result.turns, result.seconds  # resumed: > 0
    usage, by_model = dict(result.usage), dict(result.model_usage)  # resumed: a first run's
    watch = auth.LimitWatch()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"agent-{name}.jsonl"
    start = time.perf_counter()
    _log(f"agent {name}: started (cwd={cwd})")
    turns: set[str] = set()
    delegations = Delegations()
    finished = False
    stream = query(prompt=prompt, options=options)
    try:
        with log_path.open("a") as log:
            try:
                async for message in stream:
                    _write(log, message)
                    watch.see(message)
                    lookups.see(message)
                    heard(message)  # the coordinator's listener (rate limits, first message)
                    if tracker is not None:  # its turns; tool results close denied calls
                        tracker.see(message)
                    helper = delegations.see(message, result.subagents)  # None: main (#186)
                    if isinstance(message, AssistantMessage):
                        result.session_id = result.session_id or message.session_id
                        if message.parent_tool_use_id is None:
                            turns.add(message.message_id or f"#{len(turns)}")
                            result.turns = turns_before + len(turns)
                            result.model = result.model or message.model
                            if not result.first_usage:  # the session's first request
                                result.first_usage = roles.usage_of(message.usage)
                        who = f"{name}/{helper}" if helper else name
                        # the main thread's tool calls, and each helper's apart
                        calls = result.subagents[helper]["tools"] if helper else result.tool_calls
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                short = block.name.removeprefix("mcp__ka__")
                                calls[short] = calls.get(short, 0) + 1
                                _log(f"agent {who}: {short} {_brief(block.input)}")
                            elif isinstance(block, TextBlock) and cfg.verbose:
                                _log(f"agent {who}: {block.text[:300]}")
                    elif isinstance(message, SystemMessage) and message.subtype == "init":
                        if mark:  # before its first command: what it starts inherits it
                            cli[:] = hygiene.move_session(mark[hygiene.SESSION_ENV])[:1]
                        result.session_id = message.data.get("session_id") or result.session_id
                        result.api_key_source = message.data.get("apiKeySource")
                        if why := auth.session_problem(cfg.auth, result.api_key_source, env):
                            raise auth.AuthError(why)  # before the session's first request
                    elif isinstance(message, ResultMessage):
                        # a session whose background subagent or task finishes after its
                        # turn ends gets another turn and another result: the USD covers
                        # the session so far, the turns and the output only that turn (#176)
                        result.text = message.result or result.text
                        if message.structured_output is not None:
                            result.structured = message.structured_output
                        result.cost_usd = usd + _session_usd(message)
                        # so do the tokens (#181): the session so far, on top of a first run's
                        result.usage = roles.add_usage(usage, roles.usage_of(message.usage))
                        result.model_usage = _model_usage(by_model, message.model_usage)
                        result.turns = max(result.turns, turns_before + message.num_turns)
                        result.is_error = message.is_error
                        result.session_id = message.session_id
            except ClaudeSDKError as exc:  # Claude Code exits 1 after an error result
                watch.failed(exc)
                if watch.limit() is None:
                    raise
                result.is_error = True
        result.usage_limit = watch.limit()
        finished = True
    finally:
        # `async for` does not close the generator when the loop is left by an
        # exception (e.g. a timeout cancelling this task); closing it runs the
        # SDK's cleanup, which ends the Claude Code subprocess.
        await stream.aclose()  # type: ignore[attr-defined]
        for pid in cli:
            hygiene.done(pid)
        result.seconds = seconds + time.perf_counter() - start
        how = "done" if finished else "stopped"
        if result.usage_limit:
            how = f"stopped at a usage limit ({result.usage_limit.message[:120]})"
        _log(
            f"agent {name}: {how} in {result.seconds / 60:.1f} "
            f"min, {result.turns} turns, ${result.cost_usd:.2f}"
            f"{' (error)' if result.is_error else ''}"
        )
    return result


#: ``ResultMessage.model_usage`` (camelCase, per model) → the names of ``roles.USAGE_KEYS``
_MODEL_USAGE = {
    "inputTokens": "input_tokens",
    "cacheCreationInputTokens": "cache_creation_input_tokens",
    "cacheReadInputTokens": "cache_read_input_tokens",
    "outputTokens": "output_tokens",
}


def _session_usd(message: ResultMessage) -> float:
    """A result's USD with its helpers': ``total_cost_usd``, which Claude Code 2.1.286 sums
    over the main thread and its subagents (#186, tests/test_subagents.py), or the sum of
    ``model_usage``'s per-model USD should a CLI leave the helpers out of the total."""
    models = (message.model_usage or {}).values()
    by_model = [c.get("costUSD") for c in models if isinstance(c, Mapping)]
    helpers_in = sum(float(usd) for usd in by_model if isinstance(usd, int | float))
    return max(float(message.total_cost_usd or 0.0), helpers_in)


def _model_usage(
    before: dict[str, dict[str, Any]], raw: Mapping[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """``before`` (a first run's, when resumed) plus a result's tokens and USD per model."""
    out = {model: dict(counts) for model, counts in before.items()}
    for model, counts in (raw or {}).items():
        if not isinstance(counts, Mapping):
            continue
        row = out.setdefault(str(model), {})
        for key, name in _MODEL_USAGE.items():
            row[name] = row.get(name, 0) + int(counts.get(key) or 0)
        row["usd"] = round(row.get("usd", 0.0) + float(counts.get("costUSD") or 0.0), 6)
    return out


def _write(log: IO[str], message: Message) -> None:
    """One message of the session as a line of its ``agent-<name>.jsonl`` log."""
    payload: Any
    try:
        payload = (
            dataclasses.asdict(message) if dataclasses.is_dataclass(message) else repr(message)
        )
    except Exception:
        payload = repr(message)
    log.write(json.dumps({"type": type(message).__name__, "data": payload}, default=str) + "\n")
    log.flush()
