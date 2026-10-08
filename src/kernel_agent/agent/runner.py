"""Thin wrapper around ``claude_agent_sdk.query`` with logging and cost tracking."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
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
    ToolUseBlock,
    query,
)
from claude_agent_sdk.types import HookEvent

from kernel_agent import roles, skills
from kernel_agent.agent import auth, web
from kernel_agent.config import OptimizeConfig

BASE_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite"]
READ_TOOLS = ["Read", "Glob", "Grep"]
WEB_TOOLS = ["WebFetch", "WebSearch"]
WRITE_TOOLS = "Write|Edit|MultiEdit|NotebookEdit"
#: Set in every session (#126): no Claude Code auto memory, which would read and write the
#: user's own ``~/.claude/projects/<repo>/memory/`` (lessons belong in library.py), and no
#: claude.ai connectors of the login (they act on the user's account).
SESSION_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
INSTRUCTION_FILES = ("CLAUDE.md", "CLAUDE.local.md", "AGENTS.md")


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


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _brief(block_input: Any, limit: int = 110) -> str:
    if isinstance(block_input, dict):
        keys = ("candidate", "file_path", "command", "pattern", "url", "transforms", "skill")
        for key in (*keys, "subagent_type"):
            if key in block_input:
                return f"{key}={str(block_input[key])[:limit]}"
    return str(block_input)[:limit]


def agent_env(extra: dict[str, str], mode: str = auth.AUTO) -> dict[str, str]:
    """Environment for the agent's Bash tool: same Python env + toolchain variables; with
    ``mode`` subscription, the API key / cloud provider variables blanked (auth.py)."""
    venv_bin = str(Path(sys.executable).parent)
    env = {
        "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONUNBUFFERED": "1",
        **extra,
    }
    if "VIRTUAL_ENV" not in os.environ and (Path(venv_bin).parent / "pyvenv.cfg").exists():
        env["VIRTUAL_ENV"] = str(Path(venv_bin).parent)
    return auth.scrub(env) if mode == auth.SUBSCRIPTION else env


def write_guard(writable: list[Path], cwd: Path) -> dict[HookEvent, list[HookMatcher]]:
    """PreToolUse hook that lets the file-writing tools touch only ``writable``.

    A hook, not ``can_use_tool``: with ``bypassPermissions`` the CLI never asks."""
    allowed = {p.resolve() for p in writable}

    async def guard(data: Any, tool_use_id: str | None, context: Any) -> HookJSONOutput:
        tool_input = data.get("tool_input") or {}
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if path and _resolve(cwd, path) in allowed:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "this session may write only "
                + ", ".join(str(p) for p in sorted(allowed)),
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
) -> AgentResult:
    """Run one agent session to completion.

    ``result`` is filled in place while messages arrive, so a caller that
    cancels the session (e.g. a timeout) still has its session id, turns and
    tool calls so far. Cancelling terminates the Claude Code subprocess.
    ``tools`` replaces :data:`BASE_TOOLS` as the built-in tools the session has at
    all; with ``writable`` the file-writing tools may touch only those files.

    A session that stops at a usage limit returns with ``usage_limit`` set instead of
    raising (auth.py); ``resume`` (its session id) continues it, and the USD, turns and
    seconds of ``result`` then add up over both runs. :class:`~kernel_agent.agent.auth.
    AuthError` stops a session whose API key source ``cfg.auth`` does not allow.
    """
    builtin = list(BASE_TOOLS if tools is None else tools)
    builtin += (WEB_TOOLS if cfg.allow_web else []) + list(extra_tools or [])
    helpers = roles.subagents(name, web=cfg.allow_web)  # the role's delegates (#176)
    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_append + roles.delegation_note(helpers),
        },
        cwd=str(cwd),
        add_dirs=[str(d) for d in (add_dirs or [])],
        allowed_tools=builtin + mcp_tools + web.DOC_TOOLS,  # doc library: every session
        mcp_servers={"ka": mcp_server},
        permission_mode=cfg.permission_mode,  # type: ignore[arg-type]
        model=cfg.claude_model,
        max_turns=cfg.max_turns_per_agent,
        max_budget_usd=cfg.budget_usd_per_agent,
        setting_sources=[],  # no user/project settings, hooks or CLAUDE.md files (#126)
        plugins=[skills.plugin()],  # kernel-agent's skills, by explicit path (#176)
        skills=skills.session_skills(),  # those (+ Workflow's): no other bundled or user skill
        agents=helpers or None,
        env={**env, **SESSION_ENV},
        output_format=output_format,
    )
    if cfg.effort:
        options.effort = cfg.effort  # type: ignore[assignment]
    if tools is not None:  # a restricted session: no other built-in tool exists at all
        options.tools = [*builtin, "Skill", *(["Agent"] if helpers else [])]
    hooks = write_guard(writable, cwd) if writable is not None else {}
    hooks.setdefault("PreToolUse", []).append(claude_files_guard(cwd))
    options.hooks = hooks
    if resume:
        options.resume = resume

    result = result or AgentResult(name=name)
    lookups = web.Lookups(result.web, cfg.web_domains, cfg.allow_web)  # + the doc library
    if cfg.allow_web:  # WebFetch to documentation domains only, every lookup recorded
        hooks["PreToolUse"].append(lookups.guard())
    result.is_error, result.usage_limit = False, None
    usd, turns_before, seconds = result.cost_usd, result.turns, result.seconds  # resumed: > 0
    watch = auth.LimitWatch()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"agent-{name}.jsonl"
    start = time.perf_counter()
    _log(f"agent {name}: started (cwd={cwd})")
    turns: set[str] = set()
    finished = False
    stream = query(prompt=prompt, options=options)
    try:
        with log_path.open("a") as log:
            try:
                async for message in stream:
                    _write(log, message)
                    watch.see(message)
                    lookups.see(message)
                    if isinstance(message, AssistantMessage):
                        result.session_id = result.session_id or message.session_id
                        if message.parent_tool_use_id is None:
                            turns.add(message.message_id or f"#{len(turns)}")
                            result.turns = turns_before + len(turns)
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                short = block.name.removeprefix("mcp__ka__")
                                result.tool_calls[short] = result.tool_calls.get(short, 0) + 1
                                _log(f"agent {name}: {short} {_brief(block.input)}")
                            elif isinstance(block, TextBlock) and cfg.verbose:
                                _log(f"agent {name}: {block.text[:300]}")
                    elif isinstance(message, SystemMessage) and message.subtype == "init":
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
                        result.cost_usd = usd + (message.total_cost_usd or 0.0)
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
