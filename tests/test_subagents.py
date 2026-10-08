"""#186: in-session helpers (doc lookup, compile triage, profile analysis, review), the
profile tables of an evaluation in a file, and "Docs to read first" in the digests.

CPU only, no model call: the helpers' definitions and the delegation note as data, the
runner on a fake SDK stream and once through the real bundled Claude Code CLI against a
local fake Messages API (as ``test_skills.py``), the tools with a fake evaluator (as
``test_workers.py``) and the digests on the doc library's fixture corpus (``test_doclib``).
"""

import asyncio
import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from test_budget import make_orchestrator
from test_doclib import corpus, library  # noqa: F401  (fixtures: the fixture corpus)
from test_native_engine import DIFFUSION
from test_native_engine import make as make_native
from test_skills import _bundled_cli, _texts
from test_workers import _server, _target

from kernel_agent import doclib, improve, ledger, roles
from kernel_agent.agent import runner
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import SONNET_MODEL, OptimizeConfig
from kernel_agent.doclib import reading, store
from kernel_agent.native import engine
from kernel_agent.scheduler import KERNEL, NATIVE, Arm, Policy, build_arms
from kernel_agent.workspace import RunDir, read_json, write_json

WRITERS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "Agent", "Task"}

# ------------------------------------------------------------------ the helpers


def test_engineers_get_read_only_helpers():
    """Kernel, native and systems sessions get every helper as an SDK agent definition with
    read-only tools, compile-triage on the cheap model at medium effort; the research
    session only its two."""
    assert roles.problems() == []
    assert set(roles.HELPERS) == {"doc-lookup", "profile-analyst", "compile-triage", "reviewer"}
    cfg = OptimizeConfig(model_ref="m")
    for role in ("kernel", "native", "systems"):
        helpers = roles.options_for(role, roles.session_config(role, cfg))["agents"]
        assert set(helpers) == set(roles.HELPERS), role
        for name, helper in helpers.items():
            assert set(helper.tools or ()) <= set(roles.HELPER_TOOLS), name
            assert not WRITERS & set(helper.tools or ()), name
    triage = roles.agent_definition("compile-triage", cfg=cfg)
    assert (triage.model, triage.effort, triage.maxTurns) == (SONNET_MODEL, "medium", 15)
    doc_tools = tools_mod.tool_names("doc_search", "doc_read")
    assert set(triage.tools or ()) == {"Read", "Glob", "Grep", "Skill", *doc_tools}
    assert "## Cause" in triage.prompt and "## Patch" in triage.prompt
    research = roles.options_for("research", roles.session_config("research", cfg))
    assert set(research["agents"]) == {"doc-lookup", "profile-analyst"}


def test_problems_catch_a_helper_that_writes_or_delegates(monkeypatch):
    defs = dict(roles.definitions())
    triage = defs["compile-triage"]
    defs["compile-triage"] = dataclasses.replace(
        triage, tools=(*triage.tools, "Write", "Bash"), delegates=("doc-lookup",)
    )
    monkeypatch.setattr(roles, "definitions", lambda: defs)
    found = roles.problems()
    assert "compile-triage.md: a helper is read-only, not ['Bash', 'Write']" in found
    assert "compile-triage.md: a helper does not delegate" in found
    monkeypatch.setattr(roles, "WHEN_TO_DELEGATE", {"doc-lookup": "x"})
    assert any("WHEN_TO_DELEGATE and the helpers differ" in p for p in roles.problems())


def test_the_prompt_says_when_delegating_pays():
    """The helpers' section of a session's system prompt (its stable prefix): when each
    helper pays and when to do it yourself; the same for every session of a role."""
    note = roles.delegation_note(roles.subagents("kernel-attn"))
    assert note == roles.delegation_note(roles.subagents("kernel-mlp"))  # cache prefix
    assert "When delegating pays" in note and "do yourself what takes one or" in note
    for helper in roles.HELPERS:
        assert f"* `{helper}`: {roles.WHEN_TO_DELEGATE[helper]}" in note
    assert "`profile.file`" in note and "build.log" in note and "Helpers only read" in note
    research = roles.delegation_note(roles.subagents("research-attn"))
    assert "`doc-lookup`:" in research and "`compile-triage`" not in research


# ------------------------------------------------------------------ the runner


def _session(tmp_path, monkeypatch, messages):
    async def fake_query(*, prompt, options):
        for message in messages:
            yield message

    monkeypatch.setattr(runner, "query", fake_query)
    common = {"prompt": "go", "system_append": "P", "cwd": tmp_path, "mcp_server": None}
    common |= {"mcp_tools": [], "env": {}, "log_dir": tmp_path / "logs"}
    cfg = OptimizeConfig(model_ref="m")
    return asyncio.run(runner.run_agent("kernel-t", cfg=cfg, **common))


def test_the_runner_counts_main_and_helper_calls_apart(tmp_path, monkeypatch):
    """Turns and ``tool_calls`` are the main thread's; each helper's (its messages carry the
    Agent call's id as ``parent_tool_use_id``) are counted under its name with its seconds
    and model, and its doc lookups are recorded like the session's own."""

    def assistant(mid, *blocks, parent=None, model="claude-opus-5-5"):
        return AssistantMessage(list(blocks), model, parent, message_id=mid)

    call = {"subagent_type": "compile-triage", "description": "d", "prompt": "why?"}
    done = {"status": "completed", "totalDurationMs": 1500, "resolvedModel": SONNET_MODEL}
    stream = [
        assistant("m1", ToolUseBlock("toolu_a", "Agent", call)),
        assistant(
            "s1",
            ToolUseBlock("toolu_b", "Read", {"file_path": "v1.py"}),
            ToolUseBlock("toolu_c", "mcp__ka__doc_search", {"query": "tl.dot"}),
            parent="toolu_a",
            model=SONNET_MODEL,
        ),
        UserMessage(
            [ToolResultBlock("toolu_c", json.dumps({"results": [{"id": "triton:1"}]}))],
            parent_tool_use_id="toolu_a",
        ),
        UserMessage([ToolResultBlock("toolu_a", "the cause")], tool_use_result=done),
        assistant("m2", ToolUseBlock("toolu_d", "mcp__ka__evaluate_candidate", {"x": 1})),
        assistant("m2", ToolUseBlock("toolu_e", "Read", {"file_path": "NOTES.md"})),
        assistant("m3", ToolUseBlock("toolu_f", "Agent", {**call, "subagent_type": "nope"})),
        UserMessage([ToolResultBlock("toolu_f", "no agent nope", is_error=True)]),
        ResultMessage("success", 1, 1, False, 3, "s", total_cost_usd=0.5),
    ]
    result = _session(tmp_path, monkeypatch, stream)
    assert result.turns == 3 and result.cost_usd == 0.5
    assert result.tool_calls == {"Agent": 2, "evaluate_candidate": 1, "Read": 1}
    assert result.subagents == {
        "compile-triage": {
            "calls": 1,
            "tools": {"Read": 1, "doc_search": 1},
            "seconds": 1.5,
            "models": [SONNET_MODEL],
        },
        "nope": {"calls": 1, "tools": {}, "seconds": 0.0, "models": [], "failed": 1},
    }
    assert [(w["tool"], w["status"], w["hits"]) for w in result.web] == [
        ("doc_search", "ok", ["triton:1"])
    ]

    # costs.json: the main thread's tools, and its helpers'
    calls: list[dict] = []
    orch, _ = make_orchestrator(tmp_path / "orch", monkeypatch, calls)

    async def agent(name, *, result=None, **_):
        out = result or AgentResult(name=name)
        out.tool_calls, out.subagents = {"Agent": 1}, {"compile-triage": {"calls": 1}}
        return out

    orch.agent_runner = agent
    kwargs = {"prompt": "p", "system_append": "", "cwd": orch.run.root, "mcp_tools": []}
    asyncio.run(orch._agent("kernel-t1", **kwargs))
    costs = read_json(orch.run.root / "costs.json")["kernel-t1"]
    assert costs["tools"] == {"Agent": 1} and costs["subagents"] == {"compile-triage": {"calls": 1}}


def test_the_session_usd_has_its_helpers():
    """Claude Code's total includes the helpers (the real CLI test below); should a CLI
    leave them out, the per-model USD (main and helper models) still counts them."""

    def result(total, models):
        usage = {m: {"costUSD": usd, "inputTokens": 1} for m, usd in models.items()}
        return ResultMessage("success", 1, 1, False, 1, "s", total, model_usage=usage)

    split = {"claude-opus-5-5": 0.3, SONNET_MODEL: 0.4}
    assert runner._session_usd(result(0.7, split)) == pytest.approx(0.7)
    assert runner._session_usd(result(0.3, split)) == pytest.approx(0.7)  # helpers left out
    assert runner._session_usd(result(None, {})) == 0.0


# ------------------------------------------------------------------ the real CLI, a fake API


class _FakeAPI:
    """A local Messages API that plays a script per thread: a request whose system prompt
    holds a marker of ``scripts`` is that helper's, any other the main thread's (``main``).
    Each thread calls its script's tools in order (by the tool results it has seen), then
    ends with a text. Every request body is kept with whose it was; each response reports
    ``usage[who]`` tokens."""

    def __init__(self, scripts: dict[str, list[tuple[str, dict]]], usage: dict[str, dict]):
        self.requests: list[tuple[str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
                if "count_tokens" in self.path:
                    return self._send(b'{"input_tokens": 1}', "application/json")
                system = json.dumps(body.get("system"))
                who = next((m for m in scripts if m != "main" and m in system), "main")
                outer.requests.append((who, body))
                done = sum(
                    b.get("type") == "tool_result"
                    for m in body.get("messages", [])
                    if isinstance(m.get("content"), list)
                    for b in m["content"]
                )
                script = scripts[who]
                if done < len(script):
                    name, args = script[done]
                    block = {"type": "tool_use", "id": f"toolu_{len(outer.requests)}_{done}"}
                    block |= {"name": name, "input": {}}
                    delta = {"type": "input_json_delta", "partial_json": json.dumps(args)}
                    stop = "tool_use"
                else:
                    block = {"type": "text", "text": ""}
                    delta = {"type": "text_delta", "text": f"{who[:12]} done"}
                    stop = "end_turn"
                message = {"id": f"msg_{len(outer.requests)}", "type": "message"}
                message |= {"role": "assistant", "content": [], "stop_reason": None}
                message |= {"model": body.get("model"), "usage": usage[who]}
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


TRIAGE = "You triage one failed build or run"  # compile-triage's prompt
SCRIBE = "MARKER_SCRIBE: you write notes"  # a helper that may write (only in this test)


@pytest.mark.skipif(_bundled_cli() is None, reason="no bundled Claude Code CLI")
def test_a_real_session_delegates_to_read_only_helpers(tmp_path, library, monkeypatch):  # noqa: F811
    """The bundled Claude Code CLI against a local fake API. The kernel session hands a
    compile error to ``compile-triage``, which runs on its own model with exactly its
    read-only tools, reads the candidate and searches the doc library (the session's MCP
    server, the fixture corpus). A helper that has Write (only in this test) is stopped by
    the session's write guard and its instruction-file guard: the session's hooks apply to
    its helpers' tool calls. The runner counts the main thread's and each helper's calls
    apart, and the session's USD includes its helpers'."""
    base, _, _ = library
    monkeypatch.setenv(store.ENV, str(base))
    run = RunDir.create(tmp_path / "runs", "org/m")
    tdir = run.target("t")
    (tdir / "candidates").mkdir(parents=True)
    candidate = tdir / "candidates" / "v1.py"
    candidate.write_text("def build(reference):\n    return reference.nope()\n")
    scribe = AgentDefinition(
        description="Writes notes.", prompt=f"{SCRIBE}.", tools=["Read", "Write"], model="inherit"
    )
    real = roles.subagents
    monkeypatch.setattr(roles, "subagents", lambda s, **kw: {**real(s, **kw), "scribe": scribe})
    triage = {"subagent_type": "compile-triage", "description": "triage", "prompt": "Why?"}
    note = {"subagent_type": "scribe", "description": "notes", "prompt": "Write it."}
    scripts = {
        "main": [
            ("Agent", {**triage, "run_in_background": False}),
            ("Agent", {**note, "run_in_background": False}),
        ],
        TRIAGE: [
            ("Read", {"file_path": str(candidate)}),
            ("mcp__ka__doc_search", {"query": "dot_scaled", "k": 2}),
        ],
        SCRIBE: [
            ("Write", {"file_path": str(tdir / "CLAUDE.md"), "content": "x"}),
            ("Write", {"file_path": str(run.root / "plan.json"), "content": "x"}),
            ("Write", {"file_path": str(tdir / "NOTES.md"), "content": "ok"}),
        ],
    }
    usage = {
        "main": {"input_tokens": 1000, "output_tokens": 100},
        TRIAGE: {"input_tokens": 3000, "output_tokens": 300},
        SCRIBE: {"input_tokens": 10, "output_tokens": 1},
    }
    api = _FakeAPI(scripts, usage)
    env = {
        "ANTHROPIC_BASE_URL": api.url,
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "CLAUDE_CONFIG_DIR": str(tmp_path / "config"),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
    }
    cfg = OptimizeConfig(model_ref="org/m", auth="api", max_turns_per_agent=6)
    try:
        result = asyncio.run(
            asyncio.wait_for(
                runner.run_agent(
                    "kernel-t",
                    prompt="Start.",
                    system_append="MARKER_PROMPT",
                    cwd=tdir,
                    cfg=roles.session_config("kernel", cfg),
                    mcp_server=tools_mod.build_server(run),
                    mcp_tools=roles.mcp_tools("kernel"),
                    env=env,
                    log_dir=tmp_path / "logs",
                    writable=[tdir / "NOTES.md"],
                ),
                timeout=120,
            )
        )
    finally:
        api.server.shutdown()
    assert not result.is_error and result.text == "main done", result
    main = [body for who, body in api.requests if who == "main"]
    sub = [body for who, body in api.requests if who == TRIAGE]
    notes = [body for who, body in api.requests if who == SCRIBE]
    assert "When delegating pays" in json.dumps(main[0]["system"])

    # compile-triage: its model, exactly its tools, the candidate and the doc library
    assert {body["model"] for body in sub} == {SONNET_MODEL}
    assert sorted(t["name"] for t in sub[0]["tools"]) == sorted(
        roles.agent_definition("compile-triage").tools or []
    )
    assert "reference.nope()" in _texts(sub[-1])
    assert "fakelang.core.dot_scaled" in _texts(sub[-1])  # doc_search of the fixture corpus

    # the session's hooks on the helper's calls: only the session's own file was written
    later = _texts(notes[-1])
    assert "Claude Code's instruction and config files are off limits" in later
    assert "this session may write only" in later
    assert not (tdir / "CLAUDE.md").exists() and not (run.root / "plan.json").exists()
    assert (tdir / "NOTES.md").read_text() == "ok"

    # the counts: the main thread's and each helper's apart
    assert result.tool_calls == {"Agent": 2} and result.turns == len(main)
    triaged = result.subagents["compile-triage"]
    assert triaged["calls"] == 1 and triaged["tools"] == {"Read": 1, "doc_search": 1}
    assert triaged["models"] == [SONNET_MODEL]
    assert result.subagents["scribe"]["tools"] == {"Write": 3}
    assert [(w["tool"], w["results"]) for w in result.web] == [("doc_search", 2)]

    # the session's USD includes its helpers' (Claude Code's total_cost_usd, per model)
    sonnet = result.model_usage[SONNET_MODEL]
    assert sonnet["input_tokens"] == 3000 * len(sub) and sonnet["usd"] > 0
    opus = result.model_usage[cfg.claude_model]
    assert opus["input_tokens"] == 1000 * len(main) + 10 * len(notes)
    assert result.cost_usd == pytest.approx(sonnet["usd"] + opus["usd"])


# ------------------------------------------------------------------ profile tables in a file


def _profiled(snap, kwargs):
    rows = [{"kernel": f"k{i}_" + "x" * 200, "calls": 2, "us": 100.0 - 10 * i} for i in range(6)]
    out = {"status": "ok", "correct": True, "speedup": 1.3, "cases": []}
    if kwargs.get("profile"):
        out["kernels_candidate"], out["kernels_reference"] = rows, rows[:2]
        spill = "k0: 64 bytes spilled (registers 255): spills go to local memory"
        out["compiler_stats"] = {"triton": [{"kernel": "k0", "spills": 64}], "warnings": [spill]}
    return out


def test_profile_tables_go_to_a_file_with_a_summary(tmp_path, monkeypatch):
    run = RunDir.create(tmp_path, "org/m")
    tdir = _target(run)
    call, seen = _server(run, monkeypatch, _profiled)
    (tdir / "candidates" / "v1.py").write_text("def build(r):\n    return r\n")
    args = {"target_id": "t", "candidate": "candidates/v1.py", "hypothesis": "h"}

    out = call("evaluate_candidate", **args, profile=True)
    assert seen[-1]["profile"] and out["status"] == "ok"
    assert not {"kernels_candidate", "kernels_reference", "compiler_stats"} & set(out)
    profile = out["profile"]
    snap = out["best_so_far"]["snapshot"]  # history/<snapshot>.py
    path = tdir / "profiles" / (snap.removeprefix("history/").removesuffix(".py") + ".json")
    assert profile["file"] == str(path)
    tables = json.loads(path.read_text())
    assert len(tables["kernels_candidate"]) == 6 and tables["snapshot"] in snap
    assert tables["compiler_stats"]["triton"] == [{"kernel": "k0", "spills": 64}]
    total = sum(100.0 - 10 * i for i in range(6))
    assert profile["candidate"]["kernels"] == 6 and profile["candidate"]["gpu_us"] == total
    top = profile["candidate"]["top"]
    assert len(top) == tools_mod.PROFILE_TOP and top[0].endswith(f"({100 / total:.0%})")
    assert len(top[0]) < 120  # names cut: the file has them whole
    assert profile["reference"]["kernels"] == 2 and "spilled" in profile["warnings"][0]
    assert "profile-analyst" in profile["note"]
    assert len(json.dumps(out)) < len(path.read_text())

    # Nsight Compute: the full metrics to the file, each top kernel's bound in the summary
    from kernel_agent.kernels import ncu

    report = {"status": "ok", "kernels": [{"kernel": "k0", "bound": "memory", "why": "dram"}]}
    report["kernels"][0]["metrics"] = {"sm_pct": 20.0, "memory_pct": 85.0}
    monkeypatch.setattr(ncu, "profile_candidate", lambda *a, **k: report)
    (tdir / "candidates" / "v2.py").write_text("def build(r):\n    return r.eval()\n")
    out = call("evaluate_candidate", **{**args, "candidate": "candidates/v2.py"}, profile="ncu")
    assert out["profile"]["ncu"] == ["k0: memory (dram)"] and "ncu" not in out
    stored = json.loads((tdir / "profiles" / out["profile"]["file"].rsplit("/", 1)[-1]).read_text())
    assert stored["ncu"]["kernels"][0]["metrics"]["memory_pct"] == 85.0

    # without profile: nothing written, nothing summarised; a session's own directory
    (tdir / "candidates" / "v3.py").write_text("def build(r):\n    return r.float()\n")
    out = call("evaluate_candidate", **{**args, "candidate": "candidates/v3.py"})
    assert "profile" not in out and len(list((tdir / "profiles").iterdir())) == 2
    own = tmp_path / "native"
    (own / "candidates").mkdir(parents=True)
    (own / "candidates" / "v4.py").write_text("def build(r):\n    return r.half()\n")
    binding = tools_mod.SessionBinding(cwd=own)
    call, _ = _server(run, monkeypatch, _profiled, binding=binding)
    out = call("evaluate_candidate", **{**args, "candidate": "candidates/v4.py"}, profile=True)
    assert out["profile"]["file"].startswith(str(own / "profiles"))


def test_profile_summary_keeps_errors_and_ncu_unavailable():
    tables = {"profile_error": "x" * 1000, "ncu": {"status": "unavailable", "reason": "perm"}}
    summary = tools_mod.profile_summary(tables)
    assert len(summary["error"]) == 300 and summary["ncu"] == {
        "status": "unavailable",
        "reason": "perm",
    }
    assert tools_mod.profile_summary({"kernels_candidate": []})["candidate"] == {
        "kernels": 0,
        "gpu_us": 0.0,
        "top": [],
    }


# ------------------------------------------------------------------ docs to read first


def test_queries_of_a_target():
    found = reading.queries(
        module_class="Qwen3Attention",
        text="merged FP8 GEMV, RMSNorm epilogue, then PDL",
        precision="fp8_w8a8",
        arch="sm_120",
    )
    assert found == [
        "fp8 e4m3 scaled matmul",  # the precision first
        "attention softmax",  # then in the order the target names them
        "matrix vector multiply",
        "layer normalization",
        "warp shuffle reduction",
        "programmatic dependent launch",
        "sm120 mma",  # the GPU's tensor-core instructions last
    ]
    assert reading.queries(module_class="CausalDecoder", pattern="other", arch="sm_120") == [
        "convolution implicit gemm",
        "sm120 mma",
    ]
    # no tensor-core work, no mma query
    norm = reading.queries(module_class="Qwen3RMSNorm", text="warp per row", arch="sm_120")
    assert norm == ["layer normalization", "warp shuffle reduction"]
    assert reading.queries() == []
    assert reading.libraries(["triton", "cuda", "nvrtc", "nope"]) == ["triton", "cuda", "ptx"]


def test_queries_search_only_their_libraries(library, monkeypatch):  # noqa: F811
    """Each query in the target's libraries that are built; a query whose words mean
    something else elsewhere only in its own (:data:`reading.QUERY_LIBRARIES`)."""
    base, _, _ = library
    seen = []
    search = doclib.bm25.Index.search

    def spy(self, query, libraries=None, k=8):
        seen.append((query, libraries))
        return search(self, query, libraries, k)

    monkeypatch.setattr(doclib.bm25.Index, "search", spy)
    found = ["warp shuffle reduction", "dot_scaled"]
    reads = reading.first_reads(found, ["triton", "fakelang", "cuda"], base=base)
    assert seen == [("warp shuffle reduction", ["cuda"]), ("dot_scaled", ["fakelang", "cuda"])]
    assert reads[0]["title"] == "fakelang.core.dot_scaled"
    seen.clear()
    assert reading.first_reads(["warp shuffle reduction"], ["fakelang"], base=base) == []
    assert seen == []  # not searched in every library instead
    assert reading.first_reads(["dot_scaled"], ["triton"], base=base) == []  # not built


def _spec(run, **spec):
    spec = {"id": "t", "module_class": "FusedLinear", "backends": ["cuda"], **spec}
    write_json(run.target("t") / "spec.json", spec)
    return spec


def test_the_kernel_digest_lists_docs_to_read_first(tmp_path, library, monkeypatch):  # noqa: F811
    """BM25 alone over the doc library (the fixture corpus): the sections for the target's
    precision, what it computes and its GPU, in its backends' libraries, each once; none
    that only shares a word with a query; no section before the library is built."""
    base, _, _ = library
    run = RunDir.create(tmp_path / "runs", "org/m")
    write_json(run.toolchain_json, {"gpu": {"arch": "sm_120"}})
    _spec(run, approach="one kernel per layer, chained with PDL", why="launch bound")
    arm = Arm("t", KERNEL, 1.0)
    assert "## Docs to read first" not in improve.kernel_digest(run, arm, 1, 4, Policy())

    monkeypatch.setenv(store.ENV, str(base))
    digest = improve.kernel_digest(run, arm, 1, 4, Policy())
    pdl = doclib.search("Programmatic Dependent Launch", library="cuda", base=base)["results"]
    ids = [r["id"] for r in pdl if "Programmatic Dependent Launch" in r["title"]]
    lines = digest.split("## Docs to read first\n", 1)[1].split("\n\n")[0].splitlines()[1:]
    assert len(lines) == reading.FIRST_READS and all(f"`{i}`" in digest for i in ids)
    assert "Fake Guide › 4.5. Programmatic Dependent Launch (cuda 9.9, web)" in lines[0]
    # only the PDL query matched more than half of its words in the target's libraries
    assert all(line.endswith("for `programmatic dependent launch`") for line in lines)
    assert len({line.split("`")[1] for line in lines}) == len(lines)  # each section once
    assert "fakelang" not in digest  # not a library of the target's backends
    assert digest.index("## Docs to read first") < digest.index("## Before you finish")

    # the precision first, in each backend's library; a query's words, not one of them
    monkeypatch.setitem(reading.LIBRARIES, "fake", ("fakelang",))
    _spec(run, backends=["fake", "cuda"], precision="fp8_mx", approach="PDL")
    digest = improve.kernel_digest(run, arm, 1, 4, Policy())
    first = digest.split("## Docs to read first\n", 1)[1].splitlines()[1]
    assert "fakelang.core.dot_scaled" in first and "mxfp8 e8m0 block scaled" in first
    reads = reading.first_reads(["layer normalization"], ["cuda"], base=base)
    assert reads == []  # the headers say "layer", nothing says both words

    monkeypatch.setattr(reading, "first_reads", lambda *a, **k: 1 / 0)  # never fails a slice
    assert "## Docs to read first" not in improve.kernel_digest(run, arm, 1, 4, Policy())


def test_the_native_digest_lists_docs_of_its_stage(tmp_path, library, monkeypatch):  # noqa: F811
    base, _, _ = library
    monkeypatch.setenv(store.ENV, str(base))
    monkeypatch.setitem(reading.PATTERN_QUERIES, "solver", "programmatic dependent launch")
    orch, _ = make_native(tmp_path)
    run = orch.run
    arms = build_arms(run, Policy(native=True), [])
    arm = next(a for a in arms if a.id == NATIVE)

    # no stage graph yet: the docs of the module arms' classes (the modules it fuses)
    status = engine.status(run, ledger.rows(run), ["exact"])
    assert status.stage is None and "Qwen3Attention" in {a.module_class for a in arms}
    topics = ((r"attention", "programmatic dependent launch"),)
    with monkeypatch.context() as patched:
        patched.setattr(reading, "TOPICS", topics)
        text = improve.native_digest(run, arm, 7, 6, Policy(), status, arms)
    assert "## Docs to read first" in text and "Programmatic Dependent Launch" in text

    write_json(run.profile_dir / "ceilings.json", DIFFUSION)
    status = engine.status(run, ledger.rows(run), ["exact"])
    assert status.stage is not None and status.stage.pattern == "solver"
    text = improve.native_digest(run, arm, 7, 6, Policy(), status, arms)
    section = text.split("## Docs to read first\n", 1)[1]
    assert "Programmatic Dependent Launch" in section.split("\n\n")[0]
    assert "for `programmatic dependent launch`" in section  # its stage's pattern
