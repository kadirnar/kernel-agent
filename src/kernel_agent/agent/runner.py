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
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from kernel_agent.config import OptimizeConfig

BASE_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "TodoWrite"]
WEB_TOOLS = ["WebFetch", "WebSearch"]


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
) -> AgentResult:
    tools = BASE_TOOLS + (WEB_TOOLS if cfg.allow_web else []) + list(extra_tools or [])
    options = ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code", "append": system_append},
        cwd=str(cwd),
        add_dirs=[str(d) for d in (add_dirs or [])],
        allowed_tools=tools + mcp_tools,
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

    result = AgentResult(name=name)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"agent-{name}.jsonl"
    start = time.perf_counter()
    _log(f"agent {name}: started (cwd={cwd})")
    with log_path.open("a") as log:
        async for message in query(prompt=prompt, options=options):
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
                json.dumps({"type": type(message).__name__, "data": payload}, default=str) + "\n"
            )
            log.flush()
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        short = block.name.removeprefix("mcp__ka__")
                        result.tool_calls[short] = result.tool_calls.get(short, 0) + 1
                        _log(f"agent {name}: {short} {_brief(block.input)}")
                    elif isinstance(block, TextBlock) and cfg.verbose:
                        _log(f"agent {name}: {block.text[:300]}")
            elif isinstance(message, ResultMessage):
                result.text = message.result or ""
                result.structured = message.structured_output
                result.cost_usd = message.total_cost_usd or 0.0
                result.turns = message.num_turns
                result.is_error = message.is_error
                result.session_id = message.session_id
    result.seconds = time.perf_counter() - start
    _log(
        f"agent {name}: done in {result.seconds / 60:.1f} min, {result.turns} turns, "
        f"${result.cost_usd:.2f}{' (error)' if result.is_error else ''}"
    )
    return result
