"""--auth modes and usage limits: CPU only, a fake SDK stream / fake agents, no Claude."""

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    RateLimitInfo,
    ResultError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from test_budget import make_orchestrator

from kernel_agent import charts, cli, dryrun, improve, orchestrator, status
from kernel_agent.agent import auth, runner
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import OptimizeConfig
from kernel_agent.improve import ImproveConfig, Improver
from kernel_agent.workspace import RunDir, read_json, read_jsonl

SECRET = "sk-ant-test-0000-not-a-real-key"  # must never show up in any output
LIMIT_TEXT = "You've hit your session limit · resets 3pm (Europe/Istanbul)"
NOW = 1_800_000_000.0


@pytest.fixture
def clean_env(monkeypatch):
    """No credential variables from the developer's shell."""
    for name in (*auth.SCRUBBED, auth.OAUTH_TOKEN_VAR, "CLAUDE_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ environment and login


def test_subscription_scrubs_api_key_variables(monkeypatch, clean_env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    env = runner.agent_env({"TORCH_CUDA_ARCH_LIST": "12.0"}, auth.SUBSCRIPTION)
    assert env["ANTHROPIC_API_KEY"] == "" and env["CLAUDE_CODE_USE_BEDROCK"] == ""
    assert "ANTHROPIC_AUTH_TOKEN" not in env  # not set: nothing to blank
    assert env["TORCH_CUDA_ARCH_LIST"] == "12.0" and SECRET not in env.values()
    # the SDK starts Claude Code with this process's environment updated by `env`
    assert auth.present({**os.environ, **env}, auth.SCRUBBED) == []
    assert auth.billing("none", env) == auth.SUBSCRIPTION
    for mode in (auth.AUTO, auth.API):  # today's behaviour: the key reaches Claude Code
        assert "ANTHROPIC_API_KEY" not in runner.agent_env({}, mode)
        assert auth.billing("ANTHROPIC_API_KEY", runner.agent_env({}, mode)) == auth.API


def test_login_presence_only(tmp_path, monkeypatch, clean_env):
    home = tmp_path / "home"
    assert auth.login({}, home=home) is None
    with pytest.raises(SystemExit, match="no Claude Code login found"):
        auth.preflight(auth.SUBSCRIPTION, {}, home=home)
    credentials = home / ".claude" / auth.CREDENTIALS_FILE
    credentials.parent.mkdir(parents=True)
    credentials.write_text(f'{{"claudeAiOauth": {{"accessToken": "{SECRET}"}}}}')
    credentials.chmod(0)  # existence is all that is checked: it is never opened
    try:
        assert auth.login({}, home=home) == str(credentials)
        line = auth.preflight(auth.SUBSCRIPTION, {"ANTHROPIC_API_KEY": SECRET}, home=home)
    finally:
        credentials.chmod(0o600)
    assert line.startswith("auth: subscription (Claude Code login: ")
    assert "ignoring ANTHROPIC_API_KEY" in line and SECRET not in line

    other = tmp_path / "config"
    (other / auth.CREDENTIALS_FILE).parent.mkdir()
    (other / auth.CREDENTIALS_FILE).touch()
    assert auth.login({"CLAUDE_CONFIG_DIR": str(other)}, home=tmp_path) == str(
        other / auth.CREDENTIALS_FILE
    )
    assert auth.login({auth.OAUTH_TOKEN_VAR: SECRET}, home=tmp_path) == auth.OAUTH_TOKEN_VAR

    # macOS: the keychain item's existence (exit status), never its secret (-w)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(auth.sys, "platform", "darwin")
    monkeypatch.setattr(auth.subprocess, "run", fake_run)
    assert auth.login({}, home=tmp_path) == "macOS keychain"
    cmd, kwargs = calls[0]
    assert cmd[:2] == ["security", "find-generic-password"] and "-w" not in cmd
    assert kwargs["stdout"] is subprocess.DEVNULL and kwargs["stderr"] is subprocess.DEVNULL


def test_api_and_auto_preflight(tmp_path, clean_env):
    with pytest.raises(SystemExit, match="--auth api: none of ANTHROPIC_API_KEY"):
        auth.preflight(auth.API, {}, home=tmp_path)
    line = auth.preflight(auth.API, {"ANTHROPIC_API_KEY": SECRET})
    assert line == "auth: api (ANTHROPIC_API_KEY)"
    assert "the Claude Code login" in auth.preflight(auth.AUTO, {}, home=tmp_path)
    with pytest.raises(SystemExit, match="--auth must be one of"):
        auth.preflight("oauth", {})


def test_create_checks_the_login_before_the_run_starts(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(orchestrator.toolchain, "setup", lambda: SimpleNamespace(gpu=object()))
    cfg = OptimizeConfig(model_ref="org/m", runs_dir=tmp_path / "runs", auth="subscription")
    with pytest.raises(SystemExit, match="no Claude Code login found"):
        orchestrator.Orchestrator.create(cfg)
    assert not (tmp_path / "runs").exists()  # no run directory, no model download


def test_cli_auth_and_session_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_optimize", lambda ns: seen.append(cli._config(ns)) or 0)
    assert cli.main(["optimize", "org/m", "--auth", "subscription", "--max-sessions", "12"]) == 0
    assert cli.main(["optimize", "org/m"]) == 0
    cfg, default = seen
    assert (cfg.auth, cfg.max_sessions) == ("subscription", 12)
    assert (default.auth, default.max_sessions) == ("auto", None)
    assert OptimizeConfig.from_dict(cfg.to_dict()).auth == "subscription"


# ------------------------------------------------------------------ fake SDK stream


def _init(source: str | None = "none") -> SystemMessage:
    data = {"type": "system", "subtype": "init", "session_id": "sess-1"}
    return SystemMessage("init", {**data, **({"apiKeySource": source} if source else {})})


def _assistant(*blocks, error=None, msg_id="msg_1") -> AssistantMessage:
    return AssistantMessage(
        list(blocks), "claude-opus-5-5", error=error, message_id=msg_id, session_id="sess-1"
    )


def _rate_limit(status: str, resets_at: int | None = None, overage: str | None = None):
    info = RateLimitInfo(status, resets_at, "five_hour", overage_status=overage)  # type: ignore[arg-type]
    return RateLimitEvent(rate_limit_info=info, uuid="u1", session_id="sess-1")


def _result(*, is_error=False, text="done", status=None, cost=0.2, turns=2) -> ResultMessage:
    return ResultMessage(
        "success",
        1000,
        900,
        is_error,
        turns,
        "sess-1",
        total_cost_usd=cost,
        result=text,
        api_error_status=status,
    )


TOOL = ToolUseBlock("tu1", "mcp__ka__evaluate_candidate", {"candidate": "candidates/v1.py"})


def run_stream(
    tmp_path, monkeypatch, messages, *, mode="auto", fail=None, env=None, seen=None, **kwargs
):
    """``run_agent`` on a fake ``query`` that yields ``messages`` and then raises ``fail``
    (as the SDK does after an error result: Claude Code exits 1); ``seen``: what the
    stream saw (its options, the messages taken, whether it was closed)."""
    seen = {} if seen is None else seen
    seen.update(consumed=0, closed=False)

    async def fake_query(*, prompt, options):
        seen["options"], seen["prompt"] = options, prompt
        try:
            for message in messages:
                seen["consumed"] += 1
                yield message
            if fail is not None:
                raise fail
        finally:
            seen["closed"] = True

    monkeypatch.setattr(runner, "query", fake_query)
    cfg = OptimizeConfig(model_ref="org/m", auth=mode)
    common = {"prompt": "go", "system_append": "", "cwd": tmp_path, "cfg": cfg}
    common |= {"mcp_server": None, "mcp_tools": [], "log_dir": tmp_path / "logs"}
    result = asyncio.run(runner.run_agent("k", **common, env=env or {}, **kwargs))
    return result, seen


def test_api_key_source_aborts_a_subscription_session(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    stream = [_init("ANTHROPIC_API_KEY"), _assistant(TOOL), _result()]
    env, seen = runner.agent_env({}, auth.SUBSCRIPTION), {}
    with pytest.raises(auth.AuthError, match="API key source 'ANTHROPIC_API_KEY'") as err:
        run_stream(tmp_path, monkeypatch, stream, mode="subscription", env=env, seen=seen)
    assert isinstance(err.value, SystemExit)  # the run stops, not just this session
    assert SECRET not in str(err.value)
    # stopped at the init message, before any request; the CLI subprocess is closed
    assert seen["consumed"] == 1 and seen["closed"]
    assert seen["options"].env["ANTHROPIC_API_KEY"] == ""  # what Claude Code was given
    log = read_jsonl(tmp_path / "logs" / "agent-k.jsonl")
    assert [line["type"] for line in log] == ["SystemMessage"]
    for source in ("/login managed key", "apiKeyHelper"):
        with pytest.raises(auth.AuthError):
            run_stream(tmp_path, monkeypatch, [_init(source)], mode="subscription")

    result, seen = run_stream(tmp_path, monkeypatch, stream)  # auto: as before, recorded
    assert seen["consumed"] == 3 and seen["closed"]
    assert result.api_key_source == "ANTHROPIC_API_KEY" and result.cost_usd == 0.2
    result, _ = run_stream(tmp_path, monkeypatch, [_init("none"), _result()], mode="subscription")
    assert result.api_key_source == "none" and not result.is_error
    _, seen = run_stream(tmp_path, monkeypatch, [_init(None), _result()], mode="subscription")
    assert seen["consumed"] == 2  # an older CLI without apiKeySource: nothing to check


def test_api_mode_refuses_the_claude_code_login(tmp_path, monkeypatch, clean_env):
    with pytest.raises(auth.AuthError, match="--auth api"):
        run_stream(tmp_path, monkeypatch, [_init("none"), _result()], mode="api")
    token = {"ANTHROPIC_AUTH_TOKEN": SECRET}  # a bearer token: apiKeySource "none" too
    result, _ = run_stream(tmp_path, monkeypatch, [_init("none"), _result()], mode="api", env=token)
    assert auth.billing(result.api_key_source, token) == auth.API


def test_usage_limit_result_is_detected_not_raised(tmp_path, monkeypatch):
    stream = [
        _init(),
        _assistant(TOOL),
        _rate_limit("rejected", resets_at=int(NOW) + 7200, overage="rejected"),
        _assistant(TextBlock(LIMIT_TEXT), error="rate_limit", msg_id="msg_2"),
        _result(is_error=True, text=LIMIT_TEXT, status=429),
    ]
    data = {"subtype": "success", "result": LIMIT_TEXT, "api_error_status": 429}
    fail = ResultError(f"Claude Code returned an error result: {LIMIT_TEXT}", data, 1)
    result, seen = run_stream(tmp_path, monkeypatch, stream, fail=fail)
    assert seen["closed"] and result.is_error and result.session_id == "sess-1"
    limit = result.usage_limit
    assert limit is not None and limit.message == LIMIT_TEXT
    assert (limit.resets_at, limit.kind) == (NOW + 7200, "five_hour")
    assert result.tool_calls == {"evaluate_candidate": 1} and result.cost_usd == 0.2

    # the CLI's older text, the reset time in it, and no exception after the result
    legacy = "Claude AI usage limit reached|1800003600"
    result, _ = run_stream(tmp_path, monkeypatch, [_init(), _result(is_error=True, text=legacy)])
    assert result.usage_limit is not None and result.usage_limit.resets_at == 1800003600.0
    # a 429 without the text (a server-side rate limit): no reset time, back off
    result, _ = run_stream(tmp_path, monkeypatch, [_result(is_error=True, text="", status=429)])
    assert result.usage_limit is not None and result.usage_limit.resets_at is None


def test_no_usage_limit_without_an_error(tmp_path, monkeypatch):
    # the VoxCPM2 run's events: within the limit, extra usage (overage) disabled
    allowed = _rate_limit("allowed", resets_at=int(NOW), overage="rejected")
    result, _ = run_stream(tmp_path, monkeypatch, [_init(), allowed, _result()])
    assert result.usage_limit is None and not result.is_error
    # rejected, but the session finished (extra usage covered it)
    rejected = _rate_limit("rejected", resets_at=int(NOW), overage="allowed")
    result, _ = run_stream(tmp_path, monkeypatch, [_init(), rejected, _result()])
    assert result.usage_limit is None
    # any other error still fails the session
    fail = ResultError("Claude Code returned an error result: error_max_turns", {}, 1)
    with pytest.raises(ResultError):
        run_stream(tmp_path, monkeypatch, [_init(), _result(is_error=True, text="")], fail=fail)


LIMIT_CLI = """#!{python}
import json, sys
if sys.argv[1:] == ["-v"]:
    print("2.1.286 (Claude Code)")
    sys.exit(0)
def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "control_request":
        emit({{"type": "control_response", "response": {{"subtype": "success",
              "request_id": msg["request_id"], "response": {{}}}}}})
    elif msg.get("type") == "user":
        emit({{"type": "system", "subtype": "init", "session_id": "sess-1",
              "apiKeySource": "none"}})
        emit({{"type": "rate_limit_event", "uuid": "u1", "session_id": "sess-1",
              "rate_limit_info": {{"status": "rejected", "resetsAt": {resets},
                                  "rateLimitType": "five_hour", "overageStatus": "rejected"}}}})
        emit({{"type": "assistant", "session_id": "sess-1", "error": "rate_limit",
              "message": {{"id": "msg_1", "model": "<synthetic>",
                          "content": [{{"type": "text", "text": {text!r}}}]}}}})
        emit({{"type": "result", "subtype": "success", "is_error": True, "duration_ms": 5,
              "duration_api_ms": 0, "num_turns": 1, "session_id": "sess-1",
              "total_cost_usd": 0, "result": {text!r}, "api_error_status": 429}})
        sys.exit(1)  # as Claude Code does after an error result
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fake CLI")
def test_usage_limit_through_the_real_sdk(tmp_path, monkeypatch):
    """The SDK parses the CLI's frames and raises ResultError after the error result;
    run_agent returns the limit instead."""
    import functools

    from claude_agent_sdk import ClaudeAgentOptions

    from kernel_agent.agent import tools as tools_mod

    fake = tmp_path / "claude"
    fake.write_text(LIMIT_CLI.format(python=sys.executable, resets=int(NOW), text=LIMIT_TEXT))
    fake.chmod(0o755)
    monkeypatch.setattr(
        runner, "ClaudeAgentOptions", functools.partial(ClaudeAgentOptions, cli_path=str(fake))
    )
    result = asyncio.run(
        runner.run_agent(
            "k",
            prompt="go",
            system_append="",
            cwd=tmp_path,
            cfg=OptimizeConfig(model_ref="org/m", auth="subscription"),
            mcp_server=tools_mod.build_server(RunDir.create(tmp_path, "org/m")),
            mcp_tools=[],
            env={},
            log_dir=tmp_path / "logs",
        )
    )
    assert result.api_key_source == "none" and result.is_error
    assert result.usage_limit == auth.UsageLimit(LIMIT_TEXT, NOW, "five_hour")


def test_resume_adds_up_cost_turns_and_seconds(tmp_path, monkeypatch):
    before = AgentResult(name="k", cost_usd=0.5, turns=3, seconds=60.0, session_id="sess-1")
    before.usage_limit = auth.UsageLimit("limit")
    result, seen = run_stream(
        tmp_path,
        monkeypatch,
        [_init(), _assistant(TOOL), _result(cost=0.25, turns=2)],
        result=before,
        resume="sess-1",
    )
    assert seen["options"].resume == "sess-1"
    assert result is before and result.usage_limit is None and not result.is_error
    assert result.cost_usd == 0.75 and result.turns == 5 and result.seconds >= 60.0


def test_wait_seconds():
    limit = auth.UsageLimit("limit", resets_at=NOW + 3600)
    assert auth.wait_seconds(limit, 0, NOW) == 3600 + auth.RESET_SLACK_S
    assert auth.wait_seconds(limit, 0, NOW + 7200) == auth.BACKOFF_S  # stale reset time
    unknown = auth.UsageLimit("limit")
    waits = [auth.wait_seconds(unknown, n, NOW) for n in range(7)]
    assert waits == [60, 120, 240, 480, 960, 1800, 1800]


def test_usd_note():
    sub = {"usd": 1.0, "billing": "subscription"}
    assert auth.usd_note({"a": sub, "b": sub}) == "notional (subscription)"
    mixed = auth.usd_note({"a": sub, "b": {"usd": 1.0, "billing": "api"}})
    assert mixed == "notional (subscription) for 1 of 2 sessions"
    assert auth.usd_note({"a": {"usd": 1.0}}) == ""


# ------------------------------------------------------------------ the orchestrator waits


def fake_clock(orch, start=NOW):
    now = [start]
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    orch.clock, orch.sleep = (lambda: now[0]), sleep
    return now, sleeps


def test_optimize_waits_for_the_reset_and_resumes(tmp_path, monkeypatch, clean_env):
    orch, _ = make_orchestrator(tmp_path, monkeypatch, [], do_transforms=False, auth="subscription")
    now, sleeps = fake_clock(orch)
    sessions = []

    async def agent(name, *, prompt, result=None, resume=None, **_):
        result = result or AgentResult(name=name)
        sessions.append((name, prompt, resume))
        result.session_id, result.api_key_source = f"sess-{name}", "none"
        result.is_error, result.usage_limit = False, None
        result.cost_usd += 0.25
        if name == "kernel-t1" and resume is None:  # stops at the 5-hour limit
            result.is_error = True
            result.usage_limit = auth.UsageLimit(LIMIT_TEXT, now[0] + 3 * 3600, "five_hour")
        return result

    orch.agent_runner = agent
    asyncio.run(orch.run_all())

    assert sleeps == [3 * 3600 + auth.RESET_SLACK_S]
    assert [s[0] for s in sessions] == ["kernel-t1", "kernel-t1", "kernel-t2"]
    assert sessions[1][1:] == (auth.RESUME_PROMPT, "sess-kernel-t1")  # the same session
    assert orch.budget.sessions == 2  # a resumed session counts once
    costs = read_json(orch.run.root / "costs.json")
    t1 = costs["kernel-t1"]
    assert t1["usd"] == 0.5 and t1["usage_limit_waits"] == 1 and "usage_limit" not in t1
    assert (t1["auth"], t1["api_key_source"]) == ("subscription", "none")
    assert t1["billing"] == "subscription"
    events = [e for e in read_jsonl(orch.run.root / "events.jsonl") if e["event"] == "usage_limit"]
    assert [(e["agent"], e["kind"], e["wait_min"]) for e in events] == [
        ("kernel-t1", "five_hour", 181.0)
    ]
    phases = orch.run.load()["phases"]
    assert phases["kernels"]["finished"] == ["t1", "t2"] and phases["report"]["done"]
    assert "## Agent usage (total $0.75, notional (subscription))" in orch.run.report.read_text()
    assert "cost $0.75 notional (subscription)" in status.render(orch.run, width=200)


def test_a_reset_after_the_time_budget_stops_new_agents(tmp_path, monkeypatch):
    orch, _ = make_orchestrator(tmp_path, monkeypatch, [], do_transforms=False, max_hours=1.0)
    now, sleeps = fake_clock(orch)
    sessions = []

    async def agent(name, *, result=None, **_):
        result = result or AgentResult(name=name)
        sessions.append(name)
        result.is_error = True
        result.usage_limit = auth.UsageLimit(LIMIT_TEXT, now[0] + 10 * 3600, "seven_day")
        return result

    orch.agent_runner = agent
    asyncio.run(orch.run_all())
    assert sleeps == [] and sessions == ["kernel-t1"]
    assert orch.budget.blocked and "after the time budget ends" in orch.budget.blocked
    kernels = orch.run.load()["phases"]["kernels"]
    assert kernels["usage_limit_stop"][0]["agent"] == "kernel-t1"
    assert kernels["budget_skipped"][0]["reason"].startswith("usage limit: it resets at")
    t1 = read_json(orch.run.root / "costs.json")["kernel-t1"]
    assert t1["usage_limit"]["kind"] == "seven_day"
    report = orch.run.report.read_text()  # integrate + report still ran
    assert "## Budget stops" in report and "usage limit stop `kernel-t1`" in report


# ------------------------------------------------------------------ the improve loop


@pytest.fixture
def sim(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setattr(charts, "available", lambda: False)

    def make(**cfg):
        cfg = {"dossier": False, **cfg}  # sessions counted: no dossier (test_web.py)
        config = OptimizeConfig(model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, **cfg)
        orch = orchestrator.Orchestrator(dryrun.create_run(config), config)
        return orch, dryrun.World(orch)

    return make


def _loop(orch, world, **icfg):
    improver = Improver(orch, ImproveConfig(**icfg), require_capture=False, live_charts=False)
    with world.installed():
        return asyncio.run(improver.improve())


def limit_on(world, session: int, hours: float):
    """The dry run's agents, with session number ``session`` stopping at a usage limit
    that resets ``hours`` later (simulated time)."""
    real, calls = world.run_agent, []

    async def run_agent(name, *, prompt, result=None, resume=None, **kwargs):
        calls.append((name, prompt, resume))
        result = await real(name, prompt=prompt, result=result, **kwargs)
        result.is_error, result.usage_limit = False, None
        if len(calls) == session:
            result.is_error = True
            reset = world.clock.now() + hours * 3600
            result.usage_limit = auth.UsageLimit(LIMIT_TEXT, reset, "five_hour")
        return result

    world.run_agent = run_agent
    return calls


def test_improve_survives_a_usage_limit(sim):
    orch, world = sim()
    calls = limit_on(world, 2, hours=2.0)
    _, sleeps = fake_clock(orch)
    orch.clock = world.clock.now
    advance = orch.sleep

    async def sleep(seconds):  # the simulated time budget sees the wait
        world.clock.advance(seconds)
        await advance(seconds)

    orch.sleep = sleep
    reason = _loop(orch, world, max_slices=4)
    assert reason == "--max-slices 4 reached"
    assert sleeps == [pytest.approx(2 * 3600 + auth.RESET_SLACK_S)]
    name, _, resume = calls[2]
    assert (name, resume) == (calls[1][0], f"dry-{name}-2") and calls[2][1] == auth.RESUME_PROMPT
    state = read_json(orch.run.root / "improve.json")
    assert [s["status"] for s in state["slices"]] == ["done"] * 4  # not failed, not stopped
    assert state["slices"][1]["seconds"] >= 2 * 3600
    costs = read_json(orch.run.root / "costs.json")
    assert costs[state["slices"][1]["label"]]["usage_limit_waits"] == 1
    assert orch.run.report.exists()


def test_improve_stops_when_the_limit_outlasts_the_budget(sim):
    orch, world = sim(max_hours=3.0)
    limit_on(world, 2, hours=10.0)
    _, sleeps = fake_clock(orch)
    orch.clock = world.clock.now
    reason = _loop(orch, world)
    assert reason.startswith("usage limit: it resets at") and sleeps == []
    state = read_json(orch.run.root / "improve.json")
    assert len(state["slices"]) == 2 and state["slices"][1]["status"] == "error"
    assert [i["why"] for i in state["integrations"]] == ["final integration"]
    assert orch.run.report.exists()


def test_max_sessions_is_the_budget_of_this_invocation(sim, tmp_path):
    orch, world = sim(max_sessions=3)
    reason = _loop(orch, world)
    assert reason == "session budget spent (3 of 3 agent sessions)"
    assert len(world.sessions) == 3
    # `kernel-agent improve <run_dir> --max-sessions 2`: two more sessions
    run = RunDir(orch.run.root)
    cfg = OptimizeConfig(model_ref=str(run.root), runs_dir=tmp_path, max_sessions=2)
    asyncio.run(improve.improve(str(run.root), cfg, ImproveConfig(), dry_run=True))
    state = read_json(run.root / "improve.json")
    assert state["finished"]["reason"] == "session budget spent (2 of 2 agent sessions)"
    assert len(state["slices"]) + len(state["research"]) == 5
