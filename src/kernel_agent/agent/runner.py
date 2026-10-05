"""Thin wrapper around ``claude_agent_sdk.query`` with logging and cost tracking."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookJSONOutput,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    query,
)
from claude_agent_sdk.types import HookEvent

from kernel_agent.config import OptimizeConfig

BASE_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite"]
READ_TOOLS = ["Read", "Glob", "Grep"]
WEB_TOOLS = ["WebFetch", "WebSearch"]
WRITE_TOOLS = "Write|Edit|MultiEdit|NotebookEdit"


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


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _brief(block_input: Any, limit: int = 110) -> str:
    if isinstance(block_input, dict):
        for key in ("candidate", "file_path", "command", "pattern", "url", "transforms"):
            if key in block_input:
                return f"{key}={str(block_input[key])[:limit]}"
    return str(block_input)[:limit]


def agent_env(extra: dict[str, str]) -> dict[str, str]:
    """Environment for the agent's Bash tool: same Python env + toolchain variables."""
    venv_bin = str(Path(sys.executable).parent)
    env = {
        "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONUNBUFFERED": "1",
        **extra,
    }
    if "VIRTUAL_ENV" not in os.environ and (Path(venv_bin).parent / "pyvenv.cfg").exists():
        env["VIRTUAL_ENV"] = str(Path(venv_bin).parent)
    return env


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
) -> AgentResult:
    """Run one agent session to completion.

    ``result`` is filled in place while messages arrive, so a caller that
    cancels the session (e.g. a timeout) still has its session id, turns and
    tool calls so far. Cancelling terminates the Claude Code subprocess.
    ``tools`` replaces :data:`BASE_TOOLS` as the built-in tools the session has at
    all; with ``writable`` the file-writing tools may touch only those files.
    """
    builtin = list(BASE_TOOLS if tools is None else tools)
    builtin += (WEB_TOOLS if cfg.allow_web else []) + list(extra_tools or [])
    options = ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code", "append": system_append},
        cwd=str(cwd),
        add_dirs=[str(d) for d in (add_dirs or [])],
        allowed_tools=builtin + mcp_tools,
        mcp_servers={"ka": mcp_server},
        permission_mode=cfg.permission_mode,  # type: ignore[arg-type]
        model=cfg.claude_model,
        max_turns=cfg.max_turns_per_agent,
        max_budget_usd=cfg.budget_usd_per_agent,
        setting_sources=[],
        env=env,
        output_format=output_format,
    )
    if cfg.effort:
        options.effort = cfg.effort  # type: ignore[assignment]
    if tools is not None:  # a restricted session: no other built-in tool exists at all
        options.tools = builtin
    if writable is not None:
        options.hooks = write_guard(writable, cwd)

    result = result or AgentResult(name=name)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"agent-{name}.jsonl"
    start = time.perf_counter()
    _log(f"agent {name}: started (cwd={cwd})")
    turns: set[str] = set()
    finished = False
    stream = query(prompt=prompt, options=options)
    try:
        with log_path.open("a") as log:
            async for message in stream:
                payload: Any
                try:
                    payload = (
                        dataclasses.asdict(message)
                        if dataclasses.is_dataclass(message)
                        else repr(message)
                    )
                except Exception:
                    payload = repr(message)
                log.write(
                    json.dumps({"type": type(message).__name__, "data": payload}, default=str)
                    + "\n"
                )
                log.flush()
                if isinstance(message, AssistantMessage):
                    result.session_id = result.session_id or message.session_id
                    if message.parent_tool_use_id is None:
                        turns.add(message.message_id or f"#{len(turns)}")
                        result.turns = len(turns)
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            short = block.name.removeprefix("mcp__ka__")
                            result.tool_calls[short] = result.tool_calls.get(short, 0) + 1
                            _log(f"agent {name}: {short} {_brief(block.input)}")
                        elif isinstance(block, TextBlock) and cfg.verbose:
                            _log(f"agent {name}: {block.text[:300]}")
                elif isinstance(message, SystemMessage) and message.subtype == "init":
                    result.session_id = message.data.get("session_id") or result.session_id
                elif isinstance(message, ResultMessage):
                    result.text = message.result or ""
                    result.structured = message.structured_output
                    result.cost_usd = message.total_cost_usd or 0.0
                    result.turns = message.num_turns
                    result.is_error = message.is_error
                    result.session_id = message.session_id
        finished = True
    finally:
        # `async for` does not close the generator when the loop is left by an
        # exception (e.g. a timeout cancelling this task); closing it runs the
        # SDK's cleanup, which ends the Claude Code subprocess.
        await stream.aclose()  # type: ignore[attr-defined]
        result.seconds = time.perf_counter() - start
        _log(
            f"agent {name}: {'done' if finished else 'stopped'} in {result.seconds / 60:.1f} "
            f"min, {result.turns} turns, ${result.cost_usd:.2f}"
            f"{' (error)' if result.is_error else ''}"
        )
    return result
