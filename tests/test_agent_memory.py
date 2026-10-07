"""#126: agent sessions without Claude Code auto memory, and `library import-memory`.

CPU only: a fake SDK stream / a fake Claude Code CLI, synthetic memory notes."""

import asyncio
import functools
import json
import sys
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from test_auth import _init, _result, run_stream

from kernel_agent import cli, library
from kernel_agent.agent import auth, runner
from kernel_agent.agent import tools as tools_mod
from kernel_agent.config import OptimizeConfig
from kernel_agent.workspace import RunDir

# ------------------------------------------------------------------ the sessions


def test_every_session_runs_without_auto_memory_and_connectors(tmp_path, monkeypatch):
    _, seen = run_stream(tmp_path, monkeypatch, [_init(), _result()], env={"X": "1"})
    options = seen["options"]
    assert options.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert options.env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert options.env["X"] == "1"  # the caller's environment is kept
    assert options.setting_sources == []  # no user/project settings, hooks or CLAUDE.md
    assert [m.matcher for m in options.hooks["PreToolUse"]] == [runner.WRITE_TOOLS]

    # a restricted session keeps its write guard, and gets the Claude files guard too
    _, seen = run_stream(
        tmp_path, monkeypatch, [_init(), _result()], writable=[tmp_path / "plan.json"]
    )
    assert len(seen["options"].hooks["PreToolUse"]) == 2


def _decision(matcher, cwd: Path, path: str, tool: str = "Write") -> str | None:
    data = {"tool_name": tool, "tool_input": {"file_path": path}}
    out = asyncio.run(matcher.hooks[0](data, "tu1", None))
    return (out.get("hookSpecificOutput") or {}).get("permissionDecision")


def test_claude_files_guard(tmp_path, monkeypatch):
    config = tmp_path / "config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    cwd = tmp_path / "runs" / "r" / "targets" / "t"
    cwd.mkdir(parents=True)
    guard = runner.claude_files_guard(cwd)
    for path in (
        "CLAUDE.md",  # the session's own directory
        str(tmp_path / "CLAUDE.md"),  # the repository root above the run
        "../../CLAUDE.local.md",
        str(tmp_path / "AGENTS.md"),
        str(config / "projects" / "-p" / "memory" / "note.md"),  # auto memory
        str(config / "settings.json"),
    ):
        assert _decision(guard, cwd, path) == "deny", path
    assert _decision(guard, cwd, str(config / "CLAUDE.md"), tool="Edit") == "deny"
    for path in ("NOTES.md", "candidates/v1.py", str(tmp_path / "runs" / "r" / "plan.json")):
        assert _decision(guard, cwd, path) is None, path
    assert asyncio.run(guard.hooks[0]({"tool_input": {}}, None, None)) == {}


FAKE_CLI = """#!{python}
import json, os, sys
if sys.argv[1:] == ["-v"]:
    print("2.1.286 (Claude Code)")
    sys.exit(0)
seen = {{"argv": sys.argv[1:], "env": {{k: os.environ.get(k) for k in {names!r}}}}}
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        if msg["request"].get("subtype") == "initialize":
            seen["hooks"] = msg["request"].get("hooks")
        emit({{"type": "control_response", "response": {{"subtype": "success",
              "request_id": msg["request_id"], "response": {{}}}}}})
    elif msg.get("type") == "user":
        open({record!r}, "w").write(json.dumps(seen))
        emit({{"type": "system", "subtype": "init", "session_id": "sess-1",
              "apiKeySource": "none", "tools": ["Read", "Write"]}})
        emit({{"type": "result", "subtype": "success", "is_error": False, "duration_ms": 5,
              "duration_api_ms": 0, "num_turns": 1, "session_id": "sess-1",
              "total_cost_usd": 0, "result": "ok"}})
        break
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake CLI")
def test_claude_code_gets_the_switches_through_the_real_sdk(tmp_path, monkeypatch):
    """What the Claude Code process is started with, even when the user's shell turns
    auto memory on (``CLAUDE_CODE_DISABLE_AUTO_MEMORY=0`` forces it on)."""
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "0")
    record = tmp_path / "seen.json"
    fake = tmp_path / "claude"
    names = sorted(runner.SESSION_ENV)
    fake.write_text(FAKE_CLI.format(python=sys.executable, names=names, record=str(record)))
    fake.chmod(0o755)
    monkeypatch.setattr(
        runner, "ClaudeAgentOptions", functools.partial(ClaudeAgentOptions, cli_path=str(fake))
    )
    result = asyncio.run(
        runner.run_agent(
            "k",
            prompt="remember this",
            system_append="",
            cwd=tmp_path,
            cfg=OptimizeConfig(model_ref="org/m", auth="subscription"),
            mcp_server=tools_mod.build_server(RunDir.create(tmp_path, "org/m")),
            mcp_tools=[],
            env=runner.agent_env({}, auth.SUBSCRIPTION),
            log_dir=tmp_path / "logs",
        )
    )
    assert result.text == "ok" and not result.is_error
    seen = json.loads(record.read_text())
    assert seen["env"] == runner.SESSION_ENV
    assert "--setting-sources=" in seen["argv"]
    matchers = seen["hooks"]["PreToolUse"]
    assert [m["matcher"] for m in matchers] == [runner.WRITE_TOOLS]


# ------------------------------------------------------------------ import-memory


NOTES = {
    "MEMORY.md": "- [Triton tail rows](triton-tail-rows.md) — clamp tail rows\n",
    "triton-tail-rows.md": (
        "---\nname: triton-tail-rows\ndescription: Integration memcheck refuses Triton kernels "
        "whose row-indexed epilogue loads are unmasked; clamp tail rows\nmetadata:\n"
        "  node_type: memory\n  type: feedback\n  originSessionId: aaaaaaaa-0000-4000-8000-"
        "000000000001\n---\n\nUnmasked epilogue loads read past the buffer.\n"
    ),
    "graph-wrapper-strides.md": (
        "---\nname: graph-wrapper-strides\ndescription: CUDA-graph wrappers must return copies "
        "with the eager strides\nmetadata:\n  type: project\n---\nbody\n"
    ),
    "rmsnorm-warp-per-row.md": (
        "---\nname: rmsnorm-warp-per-row\ndescription: one warp per row beats a block per row "
        "at H=1024 (3.4 -> 1.9 us)\ntype: reference\n---\n"
    ),
    "git-push-workaround.md": (
        "---\nname: git-push-workaround\ndescription: the global gh credential helper path is "
        "broken; use a per-command -c helper\nmetadata:\n  type: reference\n---\n"
    ),
    "about-me.md": "---\nname: about-me\ndescription: prefers Triton\ntype: user\n---\n",
    "quoted.md": (
        '---\nname: quoted\ndescription: "a \\"L2-warm\\" eval of 17 MB weights is DRAM '
        'bound"\nmetadata:\n  type: project\n---\n'
    ),
}


@pytest.fixture
def memory(tmp_path, monkeypatch):
    """A synthetic auto-memory directory and an empty library."""
    monkeypatch.setenv(library.ENV, str(tmp_path / "lib"))
    directory = tmp_path / "memory"
    directory.mkdir()
    for name, text in NOTES.items():
        (directory / name).write_text(text)
    return directory


def _state(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


def test_memory_notes_parse(memory):
    notes = {n.name: n for n in library.memory_notes(memory)}
    assert "MEMORY" not in notes and len(notes) == len(NOTES) - 1
    tail = notes["triton-tail-rows"]
    assert (tail.kind, tail.origin) == ("feedback", "aaaaaaaa-0000-4000-8000-000000000001")
    assert tail.body == "Unmasked epilogue loads read past the buffer."
    assert notes["rmsnorm-warp-per-row"].kind == "reference"  # a top-level `type:` too
    assert library.memory_rule(notes["quoted"]).startswith('a "L2-warm" eval')
    guesses = {name: library.memory_lesson(n) for name, n in notes.items()}
    assert guesses == {
        "triton-tail-rows": "triton",
        "graph-wrapper-strides": "cuda",
        "rmsnorm-warp-per-row": "norm",  # a module family in its name
        "git-push-workaround": None,
        "about-me": "triton",
        "quoted": None,
    }


def test_import_memory_is_a_dry_run_by_default(memory, capsys):
    library.write_lessons(
        {
            "lessons": [
                {
                    "file": "triton",
                    "rules": [
                        "Triton epilogues with row-indexed loads "
                        "must clamp tail rows or memcheck refuses"
                    ],
                }
            ]
        },
        ["triton"],
    )
    lessons_before = _state(library.lessons_dir())
    notes_before = _state(memory)
    assert cli.main(["library", "import-memory", str(memory)]) == 0
    out = capsys.readouterr().out
    assert "triton-tail-rows (feedback, session aaaaaaaa) -> triton.md" in out
    assert "~ similar to: Triton epilogues with row-indexed loads" in out
    assert "graph-wrapper-strides (project) -> cuda.md" in out
    assert "rmsnorm-warp-per-row (reference) -> norm.md" in out
    assert "git-push-workaround (reference): skipped, no backend or module family" in out
    assert "about-me (user): skipped, a note about you" in out
    assert "dry run: 3 rule(s) would be added (cuda.md +1, norm.md +1, triton.md +1)" in out
    assert _state(library.lessons_dir()) == lessons_before  # nothing written
    assert _state(memory) == notes_before  # the memory files are only read


def test_import_memory_write(memory, capsys):
    notes_before = _state(memory)
    library.write_lessons({"lessons": [{"file": "cuda", "rules": ["an old rule"]}]}, ["cuda"])
    assert cli.main(["library", "import-memory", str(memory), "--write"]) == 0
    assert "added 3 rule(s)" in capsys.readouterr().out
    assert library.rules("cuda") == [
        "an old rule",
        "CUDA-graph wrappers must return copies with the eager strides",
    ]
    assert library.rules("triton")[0].startswith("Integration memcheck refuses Triton")
    text = (library.lessons_dir() / "norm.md").read_text()
    assert text.startswith("# Lessons: norm\n\n<!-- maintained by the kernel-agent librarian")
    assert _state(memory) == notes_before

    # again: nothing new
    assert cli.main(["library", "import-memory", str(memory), "--write"]) == 0
    out = capsys.readouterr().out
    assert "skipped, already in cuda.md" in out and "added 0 rule(s)" in out
    assert len(library.rules("cuda")) == 2


def test_import_memory_selection(memory, capsys, tmp_path):
    argv = ["library", "import-memory", str(memory), "--note", "quoted", "--to", "cuda"]
    assert cli.main([*argv, "--note", "about-me"]) == 0
    out = capsys.readouterr().out
    assert "quoted (project) -> cuda.md" in out and "about-me (user) -> cuda.md" in out
    assert "triton-tail-rows" not in out  # not selected: not listed
    assert "dry run: 2 rule(s) would be added (cuda.md +2)" in out
    with pytest.raises(SystemExit, match="no note named nope"):
        cli.main(["library", "import-memory", str(memory), "--note", "nope"])
    with pytest.raises(SystemExit, match="not a lessons file name"):
        cli.main(["library", "import-memory", str(memory), "--to", "../x"])
    with pytest.raises(SystemExit, match="not a directory"):
        cli.main(["library", "import-memory", str(tmp_path / "missing")])


def test_import_memory_from_agent_sessions(memory, capsys, tmp_path):
    """--runs: notes whose originSessionId is an agent session of a run, or that an agent
    session wrote with Write/Edit (Claude Code does not always record the origin)."""
    logs = tmp_path / "runs" / "org--m" / "20261007-000000" / "logs"
    logs.mkdir(parents=True)
    write = {
        "type": "AssistantMessage",
        "data": {
            "content": [
                {
                    "id": "tu1",
                    "name": "Write",
                    "input": {
                        "file_path": "/home/u/.claude/projects/-p/memory/graph-wrapper-strides.md",
                        "content": "...",
                    },
                }
            ],
            "session_id": "bbbbbbbb-0000-4000-8000-000000000002",
        },
    }
    init = {
        "type": "SystemMessage",
        "data": {"subtype": "init", "data": {"session_id": "aaaaaaaa-0000-4000-8000-000000000001"}},
    }
    (logs / "agent-kernel-t.jsonl").write_text(json.dumps(init) + "\n" + json.dumps(write) + "\n")
    sessions, written = library.agent_writers(tmp_path / "runs")
    run = str(logs.parent)
    assert sessions["aaaaaaaa-0000-4000-8000-000000000001"] == run
    assert written == {"graph-wrapper-strides.md": run}

    assert (
        cli.main(["library", "import-memory", str(memory), "--runs", str(tmp_path / "runs")]) == 0
    )
    out = capsys.readouterr().out
    assert f"triton-tail-rows (feedback, session aaaaaaaa, by an agent of {run}) -> triton" in out
    assert f"graph-wrapper-strides (project, by an agent of {run}) -> cuda.md" in out
    assert "rmsnorm-warp-per-row (reference): skipped, no agent session of the runs" in out
    assert "dry run: 2 rule(s)" in out
