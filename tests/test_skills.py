"""#176: skills (progressive disclosure) and agent definitions of the roles.

CPU only: the skills tree and the definitions are validated as files; the sessions are run
on a fake SDK stream, and once through the real bundled Claude Code CLI against a local fake
Messages API (no model call, no network beyond localhost)."""

import asyncio
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from claude_agent_sdk import ResultMessage, SystemMessage

from kernel_agent import cli, program, roles, skills
from kernel_agent.agent import prompts, runner
from kernel_agent.agent import tools as tools_mod
from kernel_agent.config import OptimizeConfig
from kernel_agent.workspace import RunDir

#: The section headings of the former agent/knowledge/*.md files (at their removal, without
#: the "N. " numbers): each is a heading of the skills tree, so no section was dropped.
KNOWLEDGE_HEADINGS = {
    "playbook.md": (
        "Diagnose before you write code",
        "Hot paths by model family",
        "Algorithm-level changes (model transforms)",
        "Engineering loop",
        "Host overhead matters for tiny kernels",
    ),
    "triton.md": (
        "sm_120 (RTX 50xx, consumer Blackwell), measured with Triton 3.8",
        "When Triton is the right backend (VoxCPM2 ledgers, docs/RESEARCH-TRITON.md §4-5)",
    ),
    "cuda.md": (
        "A. `cuda` — `torch.utils.cpp_extension.load_inline` (nvcc)",
        "B. `nvrtc` — runtime compilation with `cuda.core`",
        "Architecture cheat sheet",
        "Streams, PDL and SM partitions (`kernel_agent.concurrency`)",
    ),
    "cute_dsl.md": (
        "Structure and compilation",
        "Language facts",
        "sm_120 (GeForce Blackwell) specifics",
        "Pipelining, warp specialisation, persistence (the GEMM example)",
        "Porting the CUTLASS sm_120 C++ examples",
        "Checking without a GPU",
        "Plan amnesia and false infeasibility",
    ),
    "low_precision.md": (
        "When it is allowed",
        "The contract",
        "Formats",
        "Scale granularity and outliers",
        "Dequantise in registers (CUDA; hardware conversion from sm_89)",
        "Tensor cores and libraries on sm_120 (RTX 50xx), verified here",
        "Measured (RTX 5070 Ti, DRAM 767 GB/s copy peak, L2 48 MB)",
        "FP8 W8A8 (`precision: fp8_w8a8`): compute-bound GEMMs",
        "FP8 scaling recipes on sm_120 (measured, docs/FP8.md)",
        "MXFP8 W8A8 (`precision: fp8_mx`)",
        "FP8 KV cache (`fp8_kv`, opt-in)",
        "FP4 weights (`fp4_weights`)",
    ),
    "systems.md": (
        "Host synchronisation",
        "A post-processing stage on a side stream",
        "Serving: continuous batching",
        "Building on another file of the run",
        "Data-dependent speedups: speculative decoding, early exit",
    ),
    "native.md": (
        "Scopes (what you may replace)",
        "The staged plan",
        "Interface to the PyTorch model",
        "Correctness",
        "Integration and ownership",
        "Projects",
        "Timing",
        "Rules",
    ),
    "gpus.md": (
        "[pre_ampere] Volta / Turing (sm_70, sm_75: V100, T4, RTX 20xx)",
        "[ampere] Ampere (sm_80: A100, A30; sm_86 / sm_87: A10, A40, RTX A6000, RTX 30xx, Orin)",
        "[ada] Ada Lovelace (sm_89: RTX 40xx, RTX 6000 Ada, L4, L40S)",
        "[hopper] Hopper (sm_90: H100, H200, GH200, H20)",
        "[blackwell] Blackwell, datacenter (sm_100 / sm_103 / sm_110: B200, GB200, B300, GB300)",
        "[blackwell_geforce] Blackwell GeForce / RTX PRO / DGX Spark (sm_120 / sm_121)",
        "[newer] Newer than this table",
        "Sources",
    ),
    "sources.md": (
        'Local reference code (paths: the prompt\'s "Local sources")',
        "CUDA C++ and PTX",
        "cuBLASLt (FP8 / FP4 GEMMs, epilogues, heuristics)",
        "CUTLASS / CuTe",
        "Triton",
        "Triton libraries (reference kernels and techniques)",
        "TileLang",
        "PyTorch (2.14)",
        "Attention and LLM kernels (reference code)",
        "Low precision (formats, scaling, accuracy)",
        "Attention and sampling algorithms (papers)",
        "Kernel-generation agents: failures and reward hacks (for reviews of a candidate)",
    ),
}

#: Facts of the former files, one or two each, that must stay in the skills they moved to.
FACTS = {
    "playbook.md": ("Triton ≈ 43 us", "InferenceBench: 11.5x vs 8.1x", "`fmha_cutlassF` 1151 ms"),
    "triton.md": ("CachedLaunch", "101 376 B per block", "tl.math.div_rn", "5.7 us vs cuDNN SDPA"),
    "cuda.md": ("AT_DISPATCH_FLOATING_TYPES", "77.7 us", "VEC32_UE8M0", "ka_coresident_blocks"),
    "cute_dsl.md": ("compile_cached", "MmaMXF8Op", "Error Internal", "make_fake_compact_tensor"),
    "tilelang.md": ("T.Pipelined", "≈ 30 us"),
    "low_precision.md": ("0.0205 vs 0.0204", "mx_scale_offset", "kv_cache_share", "868 GB/s"),
    "systems.md": ("AsyncFlags", "65x on the repeated paragraph", "artifacts.load("),
    "native.md": ("`native_<id>`", "200 files / 4 MB", "PDL"),
    "gpus.md": ("FP22", "flashinfer #5963", "sm_121"),
    "sources.md": ("LtFp8CustomFind", "arxiv.org/abs/2607.16241"),
}


def _headings(text: str) -> set[str]:
    out, fence = set(), False
    for line in text.splitlines():
        if re.match(r"^\s*```", line):
            fence = not fence
        elif not fence and (m := re.match(r"^#+\s+(?:\d+\.\s+)?(.*?)\s*$", line)):
            out.add(m[1])
    return out


# ------------------------------------------------------------------ the skills tree


def test_the_skills_tree_is_valid():
    assert skills.problems() == []
    index = skills.index()
    assert len(index) >= 19 and all(skills.NAME.match(n) for n in index)
    assert skills.qualified_names() == [f"kernel-agent:{n}" for n in index]
    assert skills.plugin() == {"type": "local", "path": str(skills.PLUGIN_DIR)}
    manifest = json.loads((skills.PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == skills.PLUGIN_NAME == "kernel-agent"
    for skill in index.values():
        meta, body = skills.frontmatter(skill.path.read_text())
        assert meta == {"name": skill.name, "description": skill.description}
        assert 50 < len(skill.description) <= 400  # the listing every session carries
        assert "Use " in skill.description  # says when to load it
        assert body.startswith("# ")
    # every skill a prompt or a definition names exists
    named = {*prompts.BACKEND_GUIDES.values(), *prompts.PRECISION_SKILLS.values()}
    named |= {s for legacy in skills.LEGACY.values() for s in legacy}
    assert named <= set(index)


def test_frontmatter_is_plain_yaml():
    yaml = pytest.importorskip("yaml")
    for path in [*skills.SKILLS_DIR.glob("*/SKILL.md"), *roles.AGENTS_DIR.glob("*.md")]:
        text = path.read_text()
        meta, _ = skills.frontmatter(text)
        assert yaml.safe_load(text.split("---\n")[1]) == meta, path  # what Claude Code reads
    with pytest.raises(ValueError, match="plain YAML scalar"):
        skills.frontmatter("---\nname: x\ndescription: a: b\n---\n# t\n")
    with pytest.raises(ValueError, match="no frontmatter"):
        skills.frontmatter("# t\n")
    meta, body = skills.frontmatter("---\nname: x\nmaxTurns: 3\nskills:\n  - a\n  - b\n---\nB")
    assert meta == {"name": "x", "maxTurns": 3, "skills": ["a", "b"]} and body == "B"


def test_problems_catch_a_broken_tree(tmp_path, monkeypatch):
    plugin = tmp_path / "plugin"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "kernel-agent"}')
    skill = plugin / "skills" / "good-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: other\ndescription: d. Use it.\n---\n# T\n[a](a.md) [gone](gone.md) "
        "`examples/cuda_nothing.py`\n"
    )
    (skill / "a.md").write_text("# A\n")
    (skill / "orphan.md").write_text("# O\n")
    monkeypatch.setattr(skills, "PLUGIN_DIR", plugin)
    monkeypatch.setattr(skills, "SKILLS_DIR", plugin / "skills")
    found = "\n".join(skills.problems())
    assert "name 'other' must be the directory name" in found
    assert "broken link gone.md" in found and "no example cuda_nothing.py" in found
    assert "orphan.md: not linked from SKILL.md" in found and "a.md" not in found.split("orphan")[0]


def test_every_knowledge_section_and_fact_is_reachable():
    """Every section of the former knowledge files is a heading of the skill it moved to, its
    facts are there, and every file of a skill is linked from its SKILL.md (problems())."""
    for name, headings in KNOWLEDGE_HEADINGS.items():
        text = skills.legacy_text(name)
        assert set(headings) <= _headings(text), (name, set(headings) - _headings(text))
        for fact in FACTS[name]:
            assert fact in text, (name, fact)
    assert set(KNOWLEDGE_HEADINGS) | {"tilelang.md"} == set(skills.LEGACY)
    # the compatibility reader of the old names
    assert prompts.knowledge("low_precision.md") == skills.legacy_text("low_precision.md")
    assert prompts.knowledge("triton-kernels") == skills.get("triton-kernels").text()
    assert prompts.SOURCES.is_file() and prompts.SOURCES.parent.name == "documentation-sources"
    assert not (prompts.AGENT_DIR / "knowledge").exists()


def test_skills_name_their_examples_and_sources():
    for skill in skills.index().values():
        text = skill.path.read_text()
        if skill.name == "documentation-sources":
            continue
        assert "## Examples and sources" in text, skill.name


# ------------------------------------------------------------------ prompts reference skills


def _fp8_target():
    target = {"id": "mlp", "module_class": "Qwen3MLP", "why": "w", "approach": "a"}
    target |= {"backends": ["triton", "cuda"], "precision": "fp8_w8a8", "precision_why": "M=352"}
    capture = {"cases": [{"signature": "a0[352, 1024]", "count": 9}], "tier": "near-lossless"}
    return target, capture


def test_prompts_name_skills_instead_of_inlining_them():
    target, capture = _fp8_target()
    text = prompts.engineer_prompt(target, capture, target["backends"], "py", "tc", 10, None)
    for name in ("optimisation-playbook", "triton-kernels", "cuda-kernels", "fp8-w8a8"):
        assert f"`kernel-agent:{name}`" in text, name
        body = skills.get(name).path.read_text().split("---\n", 2)[2]
        assert body[200:400] not in text, name  # named, not inlined
    assert "`kernel-agent:precision-tiers`" in text and str(skills.SKILLS_DIR) in text
    assert len(text) < 25_000  # was ~70K characters with the guides inlined
    exact = prompts.engineer_prompt(
        {**target, "precision": None}, capture, ["cute"], "p", "t", 4, None
    )
    assert "`kernel-agent:cute-dsl`" in exact and "fp8-w8a8" not in exact

    card = {"repo_id": "o/m", "modality": "llm", "architectures": ["X"], "params": 1}
    plan = prompts.planner_prompt(card, {"median_ms": 1.0}, "# Profile", ["triton"], 3, "p", "t")
    assert "`kernel-agent:optimisation-playbook`" in plan and "model-families.md" in plan
    assert "`kernel-agent:precision-tiers`" not in plan  # an exact run
    near = prompts.planner_prompt(
        card, {"median_ms": 1.0}, "# Profile", ["triton"], 3, "p", "t", quality="near-lossless"
    )
    assert "`kernel-agent:precision-tiers`" in near
    hot = skills.get("optimisation-playbook").path.read_text()
    assert "| **Launch / CPU bound** |" in hot and "| **Launch / CPU bound** |" not in plan

    systems = prompts.systems_prompt(card, {"median_ms": 2.0}, "", [], "py", "tc", 4)
    assert "`kernel-agent:systems-patterns`" in systems and "AsyncFlags" not in systems
    native = prompts.native_prompt(card, {"median_ms": 2.0}, "", "py", "tc", 4, why="w", blocks=[])
    for prompt, role in ((plan, "planner"), (systems, "systems"), (native, "native")):
        assert all(f"`{n}`" in prompt for n in skills.for_role(role)), role
    assert skills.for_role("kernel", backends=["cuda", "nvrtc", "x"], precision="fp8_w8a8") == [
        "kernel-agent:optimisation-playbook",
        "kernel-agent:cuda-kernels",
        "kernel-agent:precision-tiers",
        "kernel-agent:fp8-w8a8",
    ]
    assert "kernel-agent:precision-tiers" in skills.for_role("planner", quality="near-lossless")
    assert skills.for_role("librarian") == []
    review = prompts.research_prompt(target, capture, "E", Path("/r/plan.md"), "tc")
    assert "`kernel-agent:fp8-w8a8`" in review and "`kernel-agent:triton-kernels`" in review
    dossier = prompts.dossier_prompt(target, capture, Path("/r/research.md"), "tc")
    assert "`kernel-agent:cuda-kernels`" in dossier and "`kernel-agent:fp8-w8a8`" in dossier
    assert "kernel-agent:documentation-sources" in prompts.web_note("kernel-mlp", ["x.org"])


# ------------------------------------------------------------------ agent definitions


def test_agent_definitions():
    assert roles.problems() == []
    defs = roles.definitions()
    assert set(roles.HELPERS) <= set(defs) and set(roles.ROLE_AGENTS.values()) <= set(defs)
    assert {r for r in program.ROLES} <= set(roles.ROLE_AGENTS)  # program.md's roles
    for role in defs.values():
        assert role.model == "inherit"  # the session's model decides (model-agnostic)
    working = ("planner", "kernel-engineer", "systems-engineer", "native-engineer", "researcher")
    for name in (*working, "profile-analyst"):  # never self-limit: they end with what is next
        assert re.search(r"next\s+(idea|direction)", defs[name].prompt), name
    kernel = roles.for_session("kernel-mlp-w2")
    assert kernel is not None and kernel.name == "kernel-engineer"
    assert set(kernel.delegates) == set(roles.HELPERS)
    assert roles.for_session("refactor-x").delegates == ()  # type: ignore[union-attr]
    assert roles.for_session("k") is None and roles.subagents("k") == {}
    assert roles.split_tools("Read, Agent(a, b), Grep") == ["Read", "Agent(a, b)", "Grep"]

    helpers = roles.subagents("research-mlp", web=False)
    assert set(helpers) == {"doc-lookup", "profile-analyst"}
    lookup = helpers["doc-lookup"]
    assert "WebFetch" not in (lookup.tools or []) and "Skill" in (lookup.tools or [])
    assert lookup.skills == ["kernel-agent:documentation-sources"] and lookup.maxTurns == 20
    assert "WebFetch" in (roles.agent_definition("doc-lookup").tools or [])
    planner = roles.agent_definition("planner")
    assert "Agent" in (planner.tools or []) and planner.prompt.startswith("You decide")
    note = roles.delegation_note(helpers)
    assert "`doc-lookup`" in note and "run_in_background: false" in note
    assert roles.delegation_note({}) == ""


# ------------------------------------------------------------------ the sessions


def _session(tmp_path, monkeypatch, name, *, messages=None, **kwargs):
    seen = {}

    async def fake_query(*, prompt, options):
        seen["options"] = options
        for message in messages or []:
            yield message

    monkeypatch.setattr(runner, "query", fake_query)
    cfg = kwargs.pop("cfg", OptimizeConfig(model_ref="org/m"))
    common = {"prompt": "go", "system_append": "PROMPT", "cwd": tmp_path, "cfg": cfg}
    common |= {"mcp_server": None, "mcp_tools": [], "env": {}, "log_dir": tmp_path / "logs"}
    result = asyncio.run(runner.run_agent(name, **common, **kwargs))
    return result, seen["options"]


def test_sessions_load_the_skills_and_their_helpers_isolated(tmp_path, monkeypatch):
    _, options = _session(tmp_path, monkeypatch, "kernel-mlp")
    assert options.setting_sources == []  # #126: no user / project settings or CLAUDE.md
    assert options.plugins == [skills.plugin()]  # kernel-agent's skills by explicit path
    assert options.skills == [*skills.qualified_names(), *skills.BUNDLED]
    assert set(options.agents) == set(roles.HELPERS)
    assert options.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    append = Path(options.extra_args[runner.APPEND_PROMPT_FILE]).read_text()
    assert append.startswith("PROMPT") and "# Helpers (the Agent tool)" in append

    # a restricted session gets the Skill tool, and Agent only with helpers
    _, options = _session(
        tmp_path, monkeypatch, "research-mlp", tools=[*runner.READ_TOOLS, "Write"]
    )
    assert options.tools[-2:] == ["Skill", "Agent"]
    assert set(options.agents) == {"doc-lookup", "profile-analyst"}
    _, options = _session(tmp_path, monkeypatch, "refactor-x", tools=[*runner.READ_TOOLS])
    assert options.tools[-1] == "Skill" and options.agents is None
    assert "# Helpers" not in Path(options.extra_args[runner.APPEND_PROMPT_FILE]).read_text()


def test_a_background_subagent_result_keeps_the_structured_output(tmp_path, monkeypatch):
    """A session whose background subagent finishes after its turn gets a second turn and a
    second result: the plan of the first stays, the turns add up."""

    def result(turns, structured, text, cost):
        return ResultMessage(
            "success",
            10,
            5,
            False,
            turns,
            "s1",
            total_cost_usd=cost,
            result=text,
            structured_output=structured,
        )

    init = SystemMessage("init", {"type": "system", "subtype": "init", "session_id": "s1"})
    stream = [init, result(4, {"targets": []}, "plan", 0.5), result(1, None, "", 0.6)]
    res, _ = _session(tmp_path, monkeypatch, "planner", messages=stream)
    assert res.structured == {"targets": []} and res.text == "plan"
    assert res.turns == 4 and res.cost_usd == pytest.approx(0.6)


# ------------------------------------------------------------------ the real CLI, a fake API


def _bundled_cli() -> Path | None:
    import claude_agent_sdk

    path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    return path if path.is_file() else None


class _FakeAPI:
    """A local Messages API that plays a script: the main thread's turns call the tools of
    ``main`` in order (by the number of tool results it has seen), a subagent's (its system
    prompt holds ``sub_marker``) those of ``sub``; then each ends with a text. Every request
    body is kept, with whose it was."""

    def __init__(self, main: list[tuple[str, dict]], sub: list[tuple[str, dict]], sub_marker):
        self.requests: list[tuple[str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
                if "count_tokens" in self.path:
                    return self._send(b'{"input_tokens": 1}', "application/json")
                who = "sub" if sub_marker in json.dumps(body.get("system")) else "main"
                outer.requests.append((who, body))
                done = sum(
                    b.get("type") == "tool_result"
                    for m in body.get("messages", [])
                    if isinstance(m.get("content"), list)
                    for b in m["content"]
                )
                script = sub if who == "sub" else main
                if done < len(script):
                    name, args = script[done]
                    block = {"type": "tool_use", "id": f"toolu_{who}{done}", "name": name}
                    block["input"] = {}
                    delta = {"type": "input_json_delta", "partial_json": json.dumps(args)}
                    stop = "tool_use"
                else:
                    block = {"type": "text", "text": ""}
                    delta = {"type": "text_delta", "text": f"{who} done"}
                    stop = "end_turn"
                message = {"id": "m", "type": "message", "role": "assistant", "content": []}
                message |= {"model": body.get("model"), "stop_reason": None}
                message["usage"] = {"input_tokens": 1, "output_tokens": 1}
                events = [
                    {"type": "message_start", "message": message},
                    {"type": "content_block_start", "index": 0, "content_block": block},
                    {"type": "content_block_delta", "index": 0, "delta": delta},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {}},
                    {"type": "message_stop"},
                ]
                data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
                self._send(data.encode(), "text/event-stream")

            def _send(self, data: bytes, kind: str):
                self.send_response(200)
                self.send_header("content-type", kind)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _texts(request: dict) -> str:
    out = []
    for m in request.get("messages", []):
        content = m.get("content")
        blocks = content if isinstance(content, list) else [{"text": content}]
        out += [b.get("text", "") for b in blocks if isinstance(b, dict)]
        out += [json.dumps(b.get("content")) for b in blocks if b.get("type") == "tool_result"]
    return "\n".join(out)


@pytest.mark.skipif(_bundled_cli() is None, reason="no bundled Claude Code CLI")
def test_a_real_session_sees_the_skills_and_nothing_of_the_user(tmp_path):
    """The bundled Claude Code CLI against a local fake API. The session lists kernel-agent's
    skills and helpers, loads a skill's SKILL.md through the Skill tool and delegates to a
    helper, whose tool calls the session's hooks still guard; it sees none of the user's
    skills, CLAUDE.md or AGENTS.md (#126) although the run sits inside a repository that
    has them."""
    config = tmp_path / "config"  # the user's ~/.claude
    (config / "skills" / "user-skill").mkdir(parents=True)
    (config / "skills" / "user-skill" / "SKILL.md").write_text(
        "---\nname: user-skill\ndescription: MARKER_USER_SKILL\n---\n# U\n"
    )
    (config / "CLAUDE.md").write_text("MARKER_USER_CLAUDE_MD\n")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "CLAUDE.md").write_text("MARKER_REPO_CLAUDE_MD\n")
    (repo / "AGENTS.md").write_text("MARKER_REPO_AGENTS_MD\n")
    (repo / ".claude" / "skills" / "proj").mkdir(parents=True)
    (repo / ".claude" / "skills" / "proj" / "SKILL.md").write_text(
        "---\nname: proj\ndescription: MARKER_PROJECT_SKILL\n---\n# P\n"
    )
    run = RunDir.create(repo / "runs", "org/m")
    question = {"subagent_type": "doc-lookup", "description": "look up", "prompt": "Q?"}
    api = _FakeAPI(
        main=[
            ("Skill", {"skill": "kernel-agent:triton-kernels"}),
            ("Agent", {**question, "run_in_background": False}),
        ],
        sub=[("WebFetch", {"url": "https://evil.example.com/page", "prompt": "q"})],
        sub_marker="You answer one documentation question",  # doc-lookup's prompt
    )
    env = {
        "ANTHROPIC_BASE_URL": api.url,
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "CLAUDE_CONFIG_DIR": str(config),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
    }
    try:
        result = asyncio.run(
            asyncio.wait_for(
                runner.run_agent(
                    "kernel-mlp",
                    prompt="Start.",
                    system_append="MARKER_PROMPT",
                    cwd=run.root,
                    cfg=OptimizeConfig(model_ref="org/m", auth="api", max_turns_per_agent=6),
                    mcp_server=tools_mod.build_server(run),
                    mcp_tools=[],
                    env=env,
                    log_dir=tmp_path / "logs",
                ),
                timeout=120,
            )
        )
    finally:
        api.server.shutdown()
    assert not result.is_error and result.text == "main done", result
    assert result.tool_calls.get("Skill") == 1 and result.tool_calls.get("Agent") == 1
    main = [body for who, body in api.requests if who == "main"]
    sub = [body for who, body in api.requests if who == "sub"]
    sent = json.dumps(main[0])
    assert "MARKER_PROMPT" in sent and "# Helpers (the Agent tool)" in sent
    for marker in (
        "USER_SKILL",
        "USER_CLAUDE_MD",
        "REPO_CLAUDE_MD",
        "REPO_AGENTS_MD",
        "PROJECT_SKILL",
    ):
        assert f"MARKER_{marker}" not in json.dumps(api.requests), marker
    listing = _texts(main[0])
    for skill in skills.index().values():
        assert f"{skill.qualified}: {skill.description}" in listing, skill.name
    for helper in roles.HELPERS:
        assert f"- {helper}: " in listing, helper
    body = "tl.cdiv(K, BLOCK_K)"  # in the SKILL.md body only: loaded on demand
    assert body not in sent
    loaded = _texts(main[1])  # after the Skill tool: SKILL.md with its directory
    assert str(skills.get("triton-kernels").dir) in loaded and body in loaded

    # the helper: its own prompt, its preloaded skill, its tools; the session's web guard
    # denied its fetch outside the documentation domains
    assert sub and "MARKER_PROMPT" not in json.dumps(sub[0])
    assert sorted(t["name"] for t in sub[0]["tools"]) == sorted(
        roles.agent_definition("doc-lookup").tools or []
    )
    assert "Base directory for this skill" in _texts(sub[0])  # documentation-sources
    assert "not https://evil.example.com/page" in _texts(sub[-1])
    assert any(w.get("url") == "https://evil.example.com/page" for w in result.web)
    assert "sub done" in _texts(main[-1])  # the report came back to the session


# ------------------------------------------------------------------ install-claude-code


def test_install_claude_code(tmp_path, capsys):
    assert cli.main(["install-claude-code", str(tmp_path)]) == 0
    claude = tmp_path / ".claude"
    assert (claude / "commands" / "optimize-model.md").is_file()
    for name in roles.definitions():
        assert (claude / "agents" / f"{name}.md").is_file(), name
    for skill in skills.index().values():
        for file in (skill.path, *skill.resources):
            target = claude / "skills" / skill.name / file.relative_to(skill.dir)
            assert target.read_text() == file.read_text()
    for role in roles.definitions().values():  # its skills are installed next to it
        assert all((claude / "skills" / s / "SKILL.md").is_file() for s in role.skills)
    command = (claude / "commands" / "optimize-model.md").read_text()
    for agent in ("planner", "kernel-engineer", "reviewer"):
        assert f"`{agent}`" in command
    out = capsys.readouterr().out
    assert "skills:" in out and "triton-kernels" in out and "kernel-engineer" in out
