"""#181: the role registry, per-role models and efforts, cache-friendly prompt order, token
and cache accounting.

CPU only, no Claude: sessions run on fake runners and a fake SDK stream against
``kernel_agent.dryrun`` runs, and once through the bundled Claude Code CLI against a local
fake Messages API (no model call, no network beyond localhost)."""

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from test_gpu_arch import fake
from test_skills import _bundled_cli, _FakeAPI, _texts

from kernel_agent import cli, dryrun, orchestrator, program, roles, skills
from kernel_agent.agent import prompts, runner
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import (
    DEFAULT_MODEL,
    HAIKU_MODEL,
    INHERIT,
    ROLE_EFFORTS,
    ROLE_MODELS,
    SONNET_MODEL,
    OptimizeConfig,
)
from kernel_agent.improve import _role_overrides
from kernel_agent.workspace import read_json


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No real toolchain probing, no doc library build thread."""
    monkeypatch.setattr(orchestrator.toolchain, "setup", dryrun.SimToolchain)
    monkeypatch.setenv("KERNEL_AGENT_DOCS_PREPARE", "0")


def make(tmp_path, **cfg):
    """An orchestrator on a simulated run (analyze, plan and capture done)."""
    cfg.setdefault("quality", "exact")
    config = OptimizeConfig(
        model_ref="Qwen/Qwen3-0.6B", runs_dir=tmp_path, use_library=False, **cfg
    )
    return orchestrator.Orchestrator(dryrun.create_run(config, 0), config)


def record(orch):
    """Fake sessions: every call's keyword arguments, in order."""
    seen = []

    async def agent(name, *, result=None, **kwargs):
        seen.append({"name": name, **kwargs})
        return result or AgentResult(name=name)

    orch.agent_runner = agent
    return seen


# ------------------------------------------------------------------ the registry


def test_registry_and_the_options_of_each_role():
    assert roles.problems() == []
    assert set(roles.REGISTRY) == set(ROLE_MODELS) == set(ROLE_EFFORTS)
    assert program.ROLES == roles.PROGRAM_ROLES and {"dossier", "librarian"} <= set(program.ROLES)
    assert not set(program.ROLES) & {*roles.HELPERS, "critic"}
    assert roles.role_of("systems#3") == "systems" and roles.role_of("kernel-mlp-w2#4") == "kernel"
    assert roles.role_of("doc-lookup") is None and roles.get("chef") is roles.NO_ROLE
    assert [r for r, spec in roles.REGISTRY.items() if spec.needs_gpu] == [
        "kernel",
        "systems",
        "native",
        "harness",
        "refactor",
    ]

    cfg = OptimizeConfig(model_ref="m")  # --claude-model opus 5.5, --effort high, 120 turns
    expect = {  # role: model, effort, turns (§3.12.1)
        "planner": (DEFAULT_MODEL, "high", 120),
        "kernel": (DEFAULT_MODEL, "high", 120),
        "systems": (DEFAULT_MODEL, "high", 120),
        "native": (DEFAULT_MODEL, "high", 360),  # 3x longer sessions
        "research": (DEFAULT_MODEL, "high", 120),
        "dossier": (SONNET_MODEL, "low", 20),
        "librarian": (SONNET_MODEL, "low", 8),
        "critic": (HAIKU_MODEL, None, 1),
    }
    for role, wanted in expect.items():
        options = roles.options_for(role, roles.session_config(role, cfg))
        assert (options["model"], options.get("effort"), options["max_turns"]) == wanted, role
        assert options["setting_sources"] == [] and options["plugins"] == [skills.plugin()]
        assert options["skills"] == skills.session_skills()  # one listing for every role
        assert set(roles.mcp_tools(role)) <= set(options["allowed_tools"]), role

    kernel = roles.options_for("kernel", roles.session_config("kernel", cfg))
    assert "tools" not in kernel  # every built-in tool
    assert {"Bash", "WebFetch", "mcp__ka__sweep_candidate", "mcp__ka__doc_search"} <= set(
        kernel["allowed_tools"]
    )
    helpers = kernel["agents"]
    assert set(helpers) == set(roles.HELPERS)
    assert (helpers["doc-lookup"].model, helpers["doc-lookup"].effort) == (SONNET_MODEL, "low")
    analyst = helpers["profile-analyst"]
    assert (analyst.model, analyst.effort) == (SONNET_MODEL, "medium")
    assert (helpers["reviewer"].model, helpers["reviewer"].effort) == (INHERIT, None)

    research = roles.options_for("research", roles.session_config("research", cfg))
    assert research["tools"] == [*roles.READ_TOOLS, "Write", *roles.WEB_TOOLS, "Skill", "Agent"]
    assert set(research["agents"]) == {"doc-lookup", "profile-analyst"}
    offline = dataclasses.replace(cfg, allow_web=False)
    dossier = roles.options_for("dossier", roles.session_config("dossier", offline))
    assert dossier["tools"] == [*roles.READ_TOOLS, "Write", "Skill"] and dossier["agents"] is None
    assert dossier["allowed_tools"] == [*roles.READ_TOOLS, "Write", *runner.web.DOC_TOOLS]
    native = roles.options_for("native", roles.session_config("native", cfg))
    assert {"mcp__ka__evaluate_candidate", "mcp__ka__evaluate_e2e"} <= set(native["allowed_tools"])

    # --role-model / --role-effort, inherit and none
    cfg.role_models.update(kernel=SONNET_MODEL, dossier=INHERIT)
    cfg.role_efforts.update(kernel="xhigh", dossier=INHERIT, librarian=None)
    assert roles.model_for("kernel", cfg) == SONNET_MODEL
    assert roles.model_for("dossier", cfg) == DEFAULT_MODEL  # inherit: --claude-model
    assert roles.effort_for("kernel", cfg) == "xhigh" and roles.effort_for("dossier", cfg) == "high"
    assert "effort" not in roles.options_for("librarian", roles.session_config("librarian", cfg))
    assert roles.model_for("chef", cfg) == cfg.claude_model  # a session of no known role
    cfg.role_models["doc-lookup"] = INHERIT
    lookup = roles.options_for("kernel", roles.session_config("kernel", cfg))["agents"]
    assert lookup["doc-lookup"].model == INHERIT  # the session's (here Sonnet)


def test_the_cli_command_keeps_settings_off_with_skills_and_plugins(tmp_path, monkeypatch):
    """#126 isolation holds with the registry's options: the Claude Code command line
    carries an empty ``--setting-sources=`` beside kernel-agent's plugin and skills, and the
    role's model, effort and turns."""
    seen = {}

    async def fake_query(*, prompt, options):
        seen["options"] = options
        for message in ():
            yield message

    monkeypatch.setattr(runner, "query", fake_query)
    for name in ("kernel-mlp", "dossier-mlp", "librarian"):
        role = roles.role_of(name)
        cfg = roles.session_config(role, OptimizeConfig(model_ref="m"))
        common = {"prompt": "go", "system_append": "P", "cwd": tmp_path, "cfg": cfg, "env": {}}
        common |= {"mcp_server": None, "mcp_tools": roles.mcp_tools(role)}
        asyncio.run(runner.run_agent(name, log_dir=tmp_path / "logs", **common))
        options = dataclasses.replace(seen["options"], cli_path="/bin/true", mcp_servers={})
        cmd = SubprocessCLITransport(prompt="go", options=options)._build_command()
        assert "--setting-sources=" in cmd, name  # empty: no user / project settings
        assert cmd[cmd.index("--plugin-dir") + 1] == str(skills.PLUGIN_DIR)
        assert cmd[cmd.index("--model") + 1] == roles.model_for(role, OptimizeConfig("m"))
        assert cmd[cmd.index("--max-turns") + 1] == str(cfg.max_turns_per_agent)
        assert cmd[cmd.index("--effort") + 1] == cfg.effort
        assert any(
            a.startswith("Skill(kernel-agent:")
            for a in cmd[cmd.index("--allowedTools") + 1].split(",")
        )


def test_role_flags_and_their_resume(tmp_path, monkeypatch):
    seen = {}

    def optimize(ns):
        seen["cfg"] = cli._config(ns)
        return 0

    monkeypatch.setattr(cli, "cmd_optimize", optimize)
    flags = ["--role-model", "kernel=claude-sonnet-5-5", "--role-model", "critic=inherit"]
    flags += ["--role-effort", "dossier=medium", "--role-effort", "librarian=none"]
    assert cli.main(["optimize", "org/m", *flags, "--librarian-model", HAIKU_MODEL]) == 0
    cfg = seen["cfg"]
    models = {"kernel": SONNET_MODEL, "critic": INHERIT, "librarian": HAIKU_MODEL}
    assert cfg.role_models == {**ROLE_MODELS, **models}
    assert cfg.role_efforts == {**ROLE_EFFORTS, "dossier": "medium", "librarian": None}
    for bad in (
        ["--role-model", "chef=x"],
        ["--role-effort", "kernel=extreme"],
        ["--role-model", "kernel"],
    ):
        with pytest.raises(SystemExit):
            cli.main(["optimize", "org/m", *bad])
    assert cli.main(["optimize", "org/m"]) == 0 and seen["cfg"].role_models == ROLE_MODELS

    # improve on a run that exists: only the flags given, over the run's own
    overrides = _role_overrides(cfg)
    assert overrides == {
        "role_models": models,
        "role_efforts": {"dossier": "medium", "librarian": None},
    }
    assert _role_overrides(OptimizeConfig(model_ref="m")) == {}
    orch = make(tmp_path, role_models={**ROLE_MODELS, "dossier": INHERIT, "systems": SONNET_MODEL})
    resumed = orchestrator.Orchestrator.resume(orch.run.root, overrides)
    own = {"dossier": INHERIT, "systems": SONNET_MODEL}  # the run's, kept
    assert resumed.cfg.role_models == {**ROLE_MODELS, **models, **own}
    assert resumed.cfg.role_efforts == {**ROLE_EFFORTS, **overrides["role_efforts"]}
    assert orchestrator.Orchestrator.resume(orch.run.root, {}).cfg.role_models["dossier"] == INHERIT

    # a run made before the registry: every role on --claude-model, the librarian its own
    old = {k: v for k, v in cfg.to_dict().items() if k not in ("role_models", "role_efforts")}
    legacy = OptimizeConfig.from_dict({**old, "librarian_model": HAIKU_MODEL})
    assert legacy.role_models == {**{r: INHERIT for r in ROLE_MODELS}, "librarian": HAIKU_MODEL}
    assert legacy.role_efforts == {
        **{r: INHERIT for r in ROLE_EFFORTS},
        "dossier": "low",
        "librarian": "low",
    }


# ------------------------------------------------------------------ cache-friendly prompts


def _target(tid, backends, precision=None):
    spec = {"id": tid, "module_class": f"M_{tid}", "backends": backends, "why": f"why {tid}"}
    spec |= {"approach": f"approach {tid}", **({"precision": precision} if precision else {})}
    info = {"qualname": f"model.{tid}", "cases": [{"signature": f"a0[{len(tid)}]", "count": 3}]}
    if precision:
        info["tier"] = "near-lossless"
    return spec, info


def test_each_split_prompt_starts_with_its_role_s_stable_prefix():
    """Byte compare: two targets (other backends, precision, budget, evidence) of one role
    share the stable prefix, which names none of them."""
    tc = fake("sm_120").summary()
    (a, a_info), (b, b_info) = (
        _target("alpha", ["triton"]),
        _target("beta", ["cuda", "cute"], "fp8_w8a8"),
    )
    texts = {
        "kernel": (
            prompts.engineer_prompt(a, a_info, a["backends"], "py", tc, 4, None),
            prompts.engineer_prompt(b, b_info, b["backends"], "py", tc, 9, {"calls": 7}),
        ),
        "research": (
            prompts.research_prompt(a, a_info, "EVIDENCE A", Path("/r/alpha/plan.md"), tc),
            prompts.research_prompt(
                b, b_info, "EVIDENCE B", Path("/r/beta/plan.md"), tc, pivot=Path("/r/p.json")
            ),
        ),
    }
    card = {"repo_id": "org/model", "modality": "llm"}
    texts["systems"] = tuple(
        prompts.systems_prompt(
            card, {"median_ms": ms}, f"# Profile {ms}", [], "py", tc, n, kernels=k
        )
        for ms, n, k in ((10.0, 4, None), (8.0, 2, [("alpha", "/h/1.py", 1.5)]))
    )
    texts["native"] = tuple(
        prompts.native_prompt(card, {"median_ms": 9.0}, "# P", "py", tc, n, why=w, blocks=b)
        for n, w, b in ((3, "plateau", []), (6, "plan", ["* block"]))
    )
    for role, (one, two) in texts.items():
        stable = prompts.stable_prefix(role, "py", tc)
        assert one.startswith(stable) and two.startswith(stable), role
        assert one != two and len(stable) > 2000, role
        for name in ("`alpha`", "`beta`", "EVIDENCE", "/r/", "org/model", "# Profile"):
            assert name not in stable, (role, name)
        assert "# This GPU: NVIDIA GeForce RTX 5070 Ti" in stable  # the run's GPU: stable
    kernel = prompts.stable_prefix("kernel", "py", tc)
    for skill in ("cuda-kernels", "cute-dsl", "fp8-w8a8"):  # the target's skills: its block
        assert f"`kernel-agent:{skill}`" not in kernel
        assert f"`kernel-agent:{skill}`" in texts["kernel"][1]
    assert "`kernel-agent:triton-kernels`" in texts["kernel"][0]
    assert "`kernel-agent:optimisation-playbook`" in kernel
    with pytest.raises(KeyError):
        prompts.stable_prefix("planner", "py", tc)


def test_sessions_of_a_role_share_a_byte_identical_system_prompt(tmp_path):
    """Through ``Orchestrator._agent``: the system prompt of every session of a split role
    is the same bytes (stable prefix + the role's notes); the target block, the digest and
    the budget go into the first message, in that order, before the task."""
    orch = make(tmp_path)
    seen = record(orch)
    a, b = orch.run.target_ids()[:2]
    for target in (a, b, a):
        n = len(seen)
        digest = f"\n\n# Improve slice\nDIGEST {target} {n}"
        asyncio.run(
            orch.kernel_slice(target, evaluations=3, digest=digest, label=f"kernel-{target}#{n}")
        )
    for target in (a, b):
        asyncio.run(orch.research(target, reason="plateau", label=f"research-{target}#1"))
    for n in (1, 2):
        asyncio.run(orch.systems_slice(evaluations=n, digest=f"\nDIGEST {n}", label=f"systems#{n}"))
    by_role: dict[str, list[dict]] = {}
    for session in seen:
        by_role.setdefault(roles.role_of(session["name"]), []).append(session)
    assert set(by_role) == {"kernel", "research", "systems"}
    for role, group in by_role.items():
        (system,) = {s["system_append"] for s in group}  # one prefix: byte-identical
        assert system.startswith(orch._stable(role)), role
        assert "# Documentation library" in system and "# Program" in system  # the role's notes
        assert len({s["prompt"] for s in group}) == len(group)  # each its own first message
        assert "# Budget" not in system and str(orch.run.root) not in system
    for session, target in zip(by_role["kernel"], (a, b, a), strict=True):
        brief = session["prompt"]
        assert f"`{target}`" not in session["system_append"]
        assert brief.startswith(prompts.BRIEF_HEADING)
        assert brief.index(f"# Target `{target}`") < brief.index(f"DIGEST {target}")
        assert brief.index("DIGEST") < brief.index("# Budget") < brief.index("# Task\nContinue")
        assert session["mcp_tools"] == roles.mcp_tools("kernel") and "tools" not in session
        assert (session["cfg"].claude_model, session["cfg"].effort) == (DEFAULT_MODEL, "high")
    research = by_role["research"][0]
    assert research["tools"] == [*roles.READ_TOOLS, "Write"]
    assert research["mcp_tools"] == ["mcp__ka__best_result"] and research["role"] == "research"


# ------------------------------------------------------------------ tokens and cache


USAGE = {
    "input_tokens": 20,
    "cache_creation_input_tokens": 3000,
    "cache_read_input_tokens": 9000,
    "output_tokens": 500,
}
FIRST = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 2000,
    "cache_read_input_tokens": 1000,
    "output_tokens": 50,
}


def test_runner_records_the_usage_of_the_stream(tmp_path, monkeypatch):
    def assistant(model, usage, parent=None):
        return AssistantMessage(
            content=[TextBlock("x")],
            model=model,
            parent_tool_use_id=parent,
            usage=usage,
            message_id=f"m-{model}-{parent}",
            session_id="s",
        )

    result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="s",
        total_cost_usd=0.25,
        usage={**USAGE, "cache_creation": {"ephemeral_1h_input_tokens": 3000}},
        model_usage={
            DEFAULT_MODEL: {
                "inputTokens": 20,
                "outputTokens": 500,
                "cacheReadInputTokens": 9000,
                "cacheCreationInputTokens": 3000,
                "costUSD": 0.2,
            },
            SONNET_MODEL: {"inputTokens": 5, "outputTokens": 9, "costUSD": 0.05},
        },
    )
    stream = [assistant(SONNET_MODEL, {"input_tokens": 99}, parent="toolu_1")]
    stream += [assistant(DEFAULT_MODEL, FIRST), assistant(DEFAULT_MODEL, USAGE), result]

    async def fake_query(*, prompt, options):
        for message in stream:
            yield message

    monkeypatch.setattr(runner, "query", fake_query)
    common = {"prompt": "go", "system_append": "P", "cwd": tmp_path, "env": {}, "mcp_tools": []}
    common |= {"cfg": OptimizeConfig(model_ref="m"), "mcp_server": None, "log_dir": tmp_path}
    out = asyncio.run(runner.run_agent("kernel-mlp", **common))
    assert out.model == DEFAULT_MODEL and out.usage == USAGE
    assert out.first_usage == FIRST  # the main thread's first request, not the helper's
    assert out.model_usage[SONNET_MODEL] == {
        "input_tokens": 5,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 9,
        "usd": 0.05,
    }
    # resumed after a usage limit: the tokens add up, the first request stays the first
    again = asyncio.run(runner.run_agent("kernel-mlp", result=out, resume="s", **common))
    assert again.usage == {k: 2 * v for k, v in USAGE.items()} and again.first_usage == FIRST
    assert again.model_usage[DEFAULT_MODEL]["usd"] == 0.4 and again.cost_usd == 0.5


def test_costs_and_the_report_have_tokens_and_cache_hits_per_role(tmp_path):
    orch = make(tmp_path)

    async def agent(name, *, cfg, result=None, **kwargs):
        result = result or AgentResult(name=name)
        result.cost_usd, result.model = 0.5, cfg.claude_model
        result.usage, result.first_usage = dict(USAGE), dict(FIRST)
        result.model_usage = {cfg.claude_model: {**USAGE, "usd": 0.5}}
        return result

    orch.agent_runner = agent
    a = orch.run.target_ids()[0]
    asyncio.run(orch.kernel_slice(a, evaluations=2, digest="", label=f"kernel-{a}#1"))
    asyncio.run(orch.dossier(a, label=f"dossier-{a}"))
    costs = read_json(orch.run.root / "costs.json")
    kernel, dossier = costs[f"kernel-{a}#1"], costs[f"dossier-{a}"]
    assert (kernel["role"], kernel["model"], kernel["effort"]) == ("kernel", DEFAULT_MODEL, "high")
    assert kernel["usage"] == USAGE and kernel["first_usage"] == FIRST
    assert kernel["models"][DEFAULT_MODEL]["usd"] == 0.5
    assert (dossier["role"], dossier["model"], dossier["effort"]) == (
        "dossier",
        SONNET_MODEL,
        "low",
    )

    costs["planner"] = {"usd": 1.0, "turns": 3}  # a session from before #181: no tokens
    by_role = roles.usage_by_role(costs)
    assert by_role["kernel"]["sessions"] == 1 and by_role["planner"]["usage"] == {}
    assert roles.cache_hit_rate(USAGE) == pytest.approx(9000 / 12020)
    assert roles.cache_hit_rate({}) is None
    text = "\n".join(roles.usage_lines(costs))
    assert "### Usage per role" in text
    assert (
        f"| kernel | 1 | `{DEFAULT_MODEL}` | high | 0.50 | 0.50 | 0.01 | 75 % | 33 % | 0.00 |"
        in text
    )
    assert (
        f"| dossier | 1 | `{SONNET_MODEL}` | low |" in text
        and "| planner | 1 | - | - | 1.00 |" in text
    )
    assert roles.usage_lines({"planner": {"usd": 1.0}}) == []  # no tokens recorded: no table


@pytest.mark.skipif(_bundled_cli() is None, reason="no bundled Claude Code CLI")
def test_a_real_session_sends_one_cacheable_system_prompt_for_two_targets(tmp_path):
    """The bundled Claude Code CLI against a local fake API, through ``Orchestrator._agent``:
    the kernel sessions of two targets (other directories, digests and budgets) send the same
    tools and byte-identical system blocks with a cache breakpoint at their end, so the second
    session reads the prefix the first one wrote; the target, the digest and the working
    directory come after it. costs.json records the session's model and tokens."""
    api = _FakeAPI(main=[], sub=[], sub_marker="NO_HELPER_IS_CALLED")
    config = tmp_path / "config"  # the user's ~/.claude: empty
    config.mkdir()
    orch = make(tmp_path / "runs", auth="api")
    orch.agent_runner = None  # the real runner
    orch.env = {
        **orch.env,
        "ANTHROPIC_BASE_URL": api.url,
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "CLAUDE_CONFIG_DIR": str(config),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
    }
    a, b = orch.run.target_ids()[:2]
    try:
        for target, evals in ((a, 2), (b, 5)):
            digest = f"\n\n# Improve slice\nDIGEST {target}"
            session = orch.kernel_slice(target, evaluations=evals, digest=digest, label=target)
            result = asyncio.run(asyncio.wait_for(session, timeout=120))
            assert not result.is_error and result.text == "main done", result
    finally:
        api.server.shutdown()
    stable = orch._stable("kernel")
    first = [
        body
        for _, body in api.requests
        if any(stable in (block.get("text") or "") for block in body.get("system") or [])
    ]
    assert len(first) == 2  # one request per session (the fake ends each at once)
    one, two = (body["system"] for body in first)
    # block 0: Claude Code's billing header, per request (outside the cached prompt)
    assert one[0]["text"].startswith("x-anthropic-billing-header")
    assert one[1:] == two[1:] and len(one) > 2  # byte-identical system prompt
    assert one[-1]["cache_control"] == {"type": "ephemeral"}  # the shared entry ends here
    assert stable in one[-1]["text"]
    assert first[0]["tools"] == first[1]["tools"]
    for body, target in zip(first, (a, b), strict=True):
        system = json.dumps(body["system"])
        assert f"`{target}`" not in system and str(orch.run.root) not in system
        messages = _texts(body)
        assert f"# Target `{target}`" in messages and f"DIGEST {target}" in messages
        assert str(orch.run.target(target)) in messages  # the working directory, after it
    costs = read_json(orch.run.root / "costs.json")
    assert costs[a]["model"] == DEFAULT_MODEL and costs[a]["role"] == "kernel"
    assert costs[a]["usage"]["output_tokens"] >= 1 and costs[a]["first_usage"]
