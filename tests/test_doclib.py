"""The local doc library (issue #177): chunking, the BM25 index and its ranking, versioned
shelves, the polite web fetch (allowlist, robots.txt, cache, offline), the agents'
doc_search / doc_read tools, their citations in research/sources.jsonl / costs / report,
the prompts and the CLI.

CPU only, no network: a small fixture corpus (tests/fixtures/docs) stands in for the
installed packages, the CUDA headers and the web docs.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ToolResultBlock, ToolUseBlock, UserMessage
from test_budget import make_orchestrator

from kernel_agent import cli, doclib
from kernel_agent.agent import prompts, runner, web
from kernel_agent.agent import tools as tools_mod
from kernel_agent.agent.runner import AgentResult
from kernel_agent.config import OptimizeConfig
from kernel_agent.doclib import bm25, chunks, sources, store
from kernel_agent.workspace import RunDir, read_json, read_jsonl

FIX = Path(__file__).parent / "fixtures" / "docs"
BASE_URL = "https://docs.nvidia.com/fake/"
ROBOTS = b"User-agent: *\nDisallow: /fake/private.html\n"


@dataclass(frozen=True)
class FakeInstalled(sources.Installed):
    """The fixture package ``fakelang`` at a version of the test's choice."""

    fake_version: str | None = "1.0"

    def root(self) -> Path | None:
        return FIX / "pkg" / "fakelang"

    def version(self) -> str | None:
        return self.fake_version


def fake_package(version: str | None = "1.0") -> FakeInstalled:
    return FakeInstalled(
        "fakelang",
        "fakelang",
        "fakelang",
        "fakelang",
        "fakelang",
        include=("*.py", "*.md"),
        fake_version=version,
    )


HEADERS = sources.Headers(
    "cuda-headers",
    "cuda",
    "CUDA headers",
    files=("cuda_runtime_api.h", "cuda.h"),
    version_file="cuda.h",
    version_macros=("CUDA_VERSION",),
)
SITE = sources.Site(
    "fake-guide",
    "cuda",
    "Fake Guide",
    (BASE_URL + "index.html",),
    prefix=BASE_URL,
    max_pages=10,
    skip=("/_",),
)


def fake_fetch(calls: list[str]):
    def get(url: str) -> tuple[int, str, bytes]:
        calls.append(url)
        if url == "https://docs.nvidia.com/robots.txt":
            return 200, url, ROBOTS
        path = FIX / "web" / url.removeprefix(BASE_URL)
        if url.startswith(BASE_URL) and path.is_file():
            return 200, url, path.read_bytes()
        return 404, url, b""

    return get


def offline(url: str) -> tuple[int, str, bytes]:
    raise OSError("Network is unreachable")


@pytest.fixture
def corpus(monkeypatch):
    """The fixture corpus as the library's sources (no installed package, header or site
    of this machine)."""
    monkeypatch.setattr(sources, "INSTALLED", (fake_package(),))
    monkeypatch.setattr(sources, "HEADERS", (HEADERS,))
    monkeypatch.setattr(sources, "sites", lambda: (SITE,))
    monkeypatch.setattr(sources, "include_dirs", lambda: [FIX / "include"])


@pytest.fixture
def library(tmp_path, corpus):
    """A built library: the fixture package and headers, and the fake site fetched."""
    calls: list[str] = []
    stats = store.fetch_web(tmp_path, fetch=fake_fetch(calls), delay_s=0, log=lambda m: None)
    store.ensure(tmp_path, log=lambda m: None)
    return tmp_path, calls, stats


# ------------------------------------------------------------------ tokens and chunks


def test_tokens_split_identifiers_like_the_code_names_them():
    def toks(text):
        return set(bm25.tokens(text))

    assert {"tl.dot_scaled", "tl", "dot_scaled", "dot", "scaled"} <= toks("tl.dot_scaled")
    assert {"cudalaunchkernelex", "cuda", "launch", "kernel", "ex"} <= toks("cudaLaunchKernelEx")
    assert {"cp.async", "cp.async.bulk", "cp.async.bulk.tensor"} <= toks("cp.async.bulk.tensor")
    assert {"cute.make_tensor", "make_tensor", "make", "tensor"} <= toks("cute::make_tensor")
    assert "9.7.14" in toks("§9.7.14") and toks("the a of x") == set()
    assert bm25.api_name("triton.language.core.dot_scaled (2/3)") == "dot_scaled"
    assert bm25.api_name("cublasLt.h: cublasLtMatmul") == "cublasltmatmul"
    assert bm25.api_name("PyTorch › cpp_extension › torch.utils.load_inline(name, x)") == (
        "load_inline"
    )


def test_python_chunks_one_per_public_api():
    path = FIX / "pkg" / "fakelang" / "core.py"
    found = {c["title"]: c for c in chunks.python_chunks(path, "fakelang.core")}
    assert set(found) == {
        "fakelang.core.builtin",  # short and undocumented: its code
        "fakelang.core.dot_scaled",
        "fakelang.core.dot",
        "fakelang.core.program_id",
        "fakelang.core.Tensor",
        "fakelang.core.Tensor.reshape",  # a documented method of its own
    }  # no module chunk (short docstring), no _private_helper, no long undocumented code
    dot = found["fakelang.core.dot_scaled"]
    assert dot["text"].startswith("@builtin\ndef dot_scaled(lhs, lhs_scale, lhs_format,")
    assert "one e8m0 exponent per 32 elements" in dot["text"]
    assert dot["source"] == f"{path}:8" and dot["doc"] == str(path)
    assert "return axis" in found["fakelang.core.program_id"]["text"]
    tensor = found["fakelang.core.Tensor"]["text"]
    assert "A block of values" in tensor and "- __init__(self, shape, dtype)" in tensor
    assert "- reshape(self, shape): Returns a tensor" in tensor and "- to(self, dtype)" in tensor
    module = chunks.python_chunks(FIX / "pkg" / "fakelang" / "__init__.py", "fakelang")
    assert [c["title"] for c in module] == ["fakelang (module)"]
    assert chunks.python_chunks(path, "x", text="def broken(:\n") == []


def test_header_chunks_per_doxygen_comment():
    path = FIX / "include" / "cuda_runtime_api.h"
    found = {c["title"]: c for c in chunks.header_chunks(path)}
    assert set(found) == {
        "cuda_runtime_api.h: Execution Control",  # \defgroup
        "cuda_runtime_api.h: cudaLaunchKernelExC",
        "cuda_runtime_api.h: cudaLaunchAttributeID",
        "cuda_runtime_api.h: CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F",  # enum members
        "cuda_runtime_api.h: CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3",
    }
    launch = found["cuda_runtime_api.h: cudaLaunchKernelExC"]
    assert launch["text"].startswith("\\brief Launches a CUDA function")
    assert "```c\nextern __host__ cudaError_t CUDARTAPI cudaLaunchKernelExC(" in launch["text"]
    assert launch["source"] == f"{path}:10"
    enum = found["cuda_runtime_api.h: cudaLaunchAttributeID"]["text"]
    assert "cudaLaunchAttributeProgrammaticStreamSerialization = 6" in enum
    assert "} cudaLaunchAttributeID;" in enum
    vec16 = found["cuda_runtime_api.h: CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3"]["text"]
    assert "16-element block" in vec16 and "= 1," in vec16
    assert sources.header_version("#define CUDA_VERSION 13020\n", ("CUDA_VERSION",)) == "13.2"
    macros = ("A", "B", "C")
    assert sources.header_version("#define A 13\n#define B 1\n#define C 1\n", macros) == "13.1.1"


def test_html_sections_keep_headings_anchors_code_and_skip_navigation():
    url = BASE_URL + "guide.html"
    page = chunks.html_sections((FIX / "web" / "guide.html").read_text())
    assert page["title"] == "4.5. Programmatic Dependent Launch — Fake Guide"
    found = {c["title"]: c for c in chunks.section_chunks(page, url, "Fake Guide")}
    assert list(found) == [
        "Fake Guide › 4.5. Programmatic Dependent Launch",
        "Fake Guide › 4.5. Programmatic Dependent Launch › 4.5.2. API Description",
    ]  # the heading-only section has no chunk
    top, api = found.values()
    assert top["source"] == url + "#programmatic-dependent-launch"  # the section id
    assert "allows a dependent secondary kernel" in top["text"] and top["doc"] == url
    assert api["source"] == url + "#api-description"
    text = api["text"]
    assert "```\n__global__ void secondary_kernel() {\n    cudaGridDependencySynchronize();" in (
        text
    )
    assert "- Set `cudaLaunchAttributeProgrammaticStreamSerialization` on the launch." in text
    assert "| attribute | value" in text and "¶" not in text
    assert "Site header" not in json.dumps(found)

    api_page = chunks.html_sections((FIX / "web" / "api.html").read_text())
    assert api_page["title"] == "API — Fake Guide"  # not the <title> of an inline SVG
    found = {c["title"]: c for c in chunks.section_chunks(api_page, BASE_URL + "api.html", "G")}
    launch = found["G › API › fake.launch(kernel, grid, pdl=False)"]
    assert launch["source"] == BASE_URL + "api.html#fake.launch"
    assert "with `pdl` the launch overlaps" in launch["text"]
    assert "G › API › fake.wait()" in found

    index = chunks.html_sections((FIX / "web" / "index.html").read_text())
    assert index["meta"]["docs-version"] == "9.9" and "guide.html" in index["links"]
    text = json.dumps(index["sections"])
    assert "fakelang" in text and "not content" not in text and "Outside" not in text


def test_markdown_chunks_and_long_sections_in_parts():
    path = FIX / "pkg" / "fakelang" / "NOTES.md"
    found = chunks.markdown_chunks(path, "fakelang NOTES.md")
    assert [c["title"] for c in found] == ["fakelang NOTES.md › fakelang notes › Pipelining"]
    assert "# a comment in a code block, not a heading" in found[0]["text"]

    para = "word " * 150  # 750 characters
    text = "\n\n".join([para] * 6 + ["```\n" + "code line\n" * 40 + "```"] + [para] * 2)
    parts = chunks.split_long(text, limit=1000)
    assert len(parts) > 3 and all(len(p) <= 1200 for p in parts)
    assert "".join(parts).count("word") == text.count("word")
    assert chunks.split_long("short") == ["short"]
    big = chunks._parts("T", "x\n\n" * 10 + "y" * 4000, "s", "d")
    assert [c["title"] for c in big] == ["T (1/2)", "T (2/2)"]


# ------------------------------------------------------------------ build, search, read


def test_build_search_and_read(library):
    base, calls, stats = library
    manifest = read_json(base / store.MANIFEST)
    shelves = manifest["shelves"]
    assert set(shelves) == {"fakelang", "cuda-headers", "web-fake-guide"}
    assert manifest["missing"] == {}
    assert (shelves["fakelang"]["version"], shelves["fakelang"]["origin"]) == ("1.0", "installed")
    assert shelves["cuda-headers"]["version"] == "13.0"  # CUDA_VERSION 13000
    web_shelf = shelves["web-fake-guide"]
    assert (web_shelf["version"], web_shelf["origin"], web_shelf["sources"]) == ("9.9", "web", 3)
    assert web_shelf["fetched"] == shelves["web-fake-guide"]["built"][:10]
    assert (base / "shelves" / "fakelang-1.0.jsonl").is_file()
    assert stats == {"fetched": 3, "offline": False, "busy": False}  # missing.html: a 404

    out = doclib.search("dot_scaled", base=base)
    first = out["results"][0]
    assert first["title"] == "fakelang.core.dot_scaled" and first["id"].startswith("fakelang:")
    assert set(first) == {"id", "title", "library", "version", "origin", "source", "snippet"} | {
        "score"
    }
    assert first["source"].endswith("core.py:8") and "e8m0" in first["snippet"]

    out = doclib.search("programmatic dependent launch", library="cuda", k=3, base=base)
    assert len(out["results"]) == 3 and {r["library"] for r in out["results"]} == {"cuda"}
    assert out["results"][0]["source"] == BASE_URL + "guide.html#programmatic-dependent-launch"
    hit = doclib.search("cudaLaunchAttributeProgrammaticStreamSerialization", base=base)
    titles = [r["title"] for r in hit["results"][:2]]  # the header and the guide's page
    assert "cuda_runtime_api.h: cudaLaunchAttributeID" in titles
    hit = doclib.search("T.reshape", "fakelang, cuda", base=base)  # several libraries
    assert hit["results"][0]["title"] == "fakelang.core.Tensor.reshape"
    bad = doclib.search("dot", library="nope", base=base)
    assert "no library nope" in bad["error"] and set(bad["libraries"]) == {"cuda", "fakelang"}
    none = doclib.search("zzzqqq", base=base)
    assert none["results"] == [] and "hint" in none

    page = doclib.search("Programmatic Dependent Launch", library="cuda", base=base)
    top = next(r for r in page["results"] if r["source"].endswith("#programmatic-dependent-launch"))
    read = doclib.read(top["id"], base=base)
    assert read["read"][0] == top["id"] and len(read["read"]) == 2  # read on: 4.5.2 too
    assert (
        "## Fake Guide › 4.5. Programmatic Dependent Launch › 4.5.2. API Description"
        in (read["text"])
    )
    assert read["next"] is None and read["prev"] is None and read["origin"] == "web"
    assert read["fetched"] == web_shelf["fetched"] and read["version"] == "9.9"
    short = doclib.read(top["id"], max_chars=10, base=base)  # at least 500 characters
    assert short["read"] == [top["id"]] and short["next"] == read["read"][1]
    assert "no chunk 'nope'" in doclib.read("nope", base=base)["error"]

    # the crawl: robots.txt, the prefix, the skip list and the allowlist were respected
    assert BASE_URL + "private.html" not in calls and BASE_URL + "_static/theme.html" not in calls
    assert not any("evil" in c or "/other/" in c for c in calls)
    pages = sources.read_pages(base)
    assert pages[BASE_URL + "private.html"]["error"] == "disallowed by robots.txt"
    assert pages[BASE_URL + "missing.html"]["error"] == "HTTP 404"
    ok = pages[BASE_URL + "guide.html"]
    assert ok["status"] == "ok" and ok["site"] == "fake-guide" and len(ok["sha256"]) == 64


def test_versions_rebuild_and_missing_sources(library, monkeypatch):
    base, _, _ = library
    first = read_json(base / store.MANIFEST)["shelves"]["fakelang"]
    old_ids = set(store.load(base).by_id)
    assert store.ensure(base)["shelves"]["fakelang"]["built"] == first["built"]  # unchanged

    monkeypatch.setattr(sources, "INSTALLED", (fake_package("2.0"),))  # an upgrade
    manifest = store.ensure(base, log=lambda m: None)
    assert manifest["shelves"]["fakelang"]["version"] == "2.0"
    assert (base / "shelves" / "fakelang-2.0.jsonl").is_file()
    assert not (base / "shelves" / "fakelang-1.0.jsonl").exists()  # the old version dropped
    hit = doclib.search("dot_scaled", base=base)["results"][0]
    assert hit["version"] == "2.0" and hit["id"] in old_ids  # ids are stable across versions

    monkeypatch.setattr(sources, "INSTALLED", (fake_package(None),))  # uninstalled
    manifest = store.ensure(base, log=lambda m: None)
    assert "fakelang" not in manifest["shelves"]
    assert manifest["missing"]["fakelang"] == "fakelang is not installed"
    assert not (base / "shelves" / "fakelang-2.0.jsonl").exists()


def test_offline_uses_the_cache_and_lazy_build(library, tmp_path_factory):
    base, _, _ = library
    pages = sources.read_pages(base)
    out = store.fetch_web(base, fetch=offline, refresh=True, delay_s=0, log=lambda m: None)
    assert out["offline"] is True and out["fetched"] == 0
    assert sources.read_pages(base)[BASE_URL + "guide.html"] == pages[BASE_URL + "guide.html"]
    assert "web-fake-guide" in store.ensure(base)["shelves"]  # still built from the cache

    fresh = tmp_path_factory.mktemp("fresh")  # never fetched, offline: installed shelves only
    store.fetch_web(fresh, fetch=offline, delay_s=0, log=lambda m: None)
    lib = store.load(fresh)  # the first search builds what is missing
    assert set(lib.libraries) == {"fakelang", "cuda"}
    assert lib.manifest["missing"] == {"web-fake-guide": "not fetched (kernel-agent docs build)"}
    lines = store.describe(fresh)
    assert (
        "chunks in 2 shelves" in lines[0] and "missing: web-fake-guide: not fetched" in (lines[-1])
    )
    assert any(line.strip().startswith("fakelang  fakelang 1.0 (installed)") for line in lines)


def test_fetch_respects_the_allowlist_and_follows_moved_pages(tmp_path):
    calls: list[str] = []
    old = BASE_URL + "old.html"
    moved = b'<meta http-equiv="refresh" content="0; url=guide.html">'

    def get(url: str) -> tuple[int, str, bytes]:
        calls.append(url)
        return (200, url, moved) if url == old else fake_fetch([])(url)

    cache = sources.PageCache(tmp_path, fetch=get, hosts=web.domains(), delay_s=0)
    cache.log = lambda m: None
    assert cache.get("https://evil.example.com/x.html", "s") is None and calls == []
    html = cache.get(old, "s")
    assert html is not None and "Programmatic Dependent Launch" in html
    assert cache.index[old]["final_url"] == BASE_URL + "guide.html"
    assert cache.cached(old) == cache.cached(BASE_URL + "guide.html")
    calls.clear()
    assert cache.get(old, "s") == html and calls == []  # cached: no request


def test_prepare_in_background_is_off_under_test(monkeypatch):
    assert store.prepare_in_background(fetch=True) is None  # KERNEL_AGENT_DOCS_PREPARE=0
    monkeypatch.setenv(store.PREPARE_ENV, "1")
    ran = []
    monkeypatch.setattr(store, "ensure", lambda **kw: ran.append("ensure"))
    monkeypatch.setattr(store, "fetch_web", lambda **kw: ran.append("fetch"))
    monkeypatch.setattr(store, "_background", None)
    thread = store.prepare_in_background(fetch=True, log=lambda m: None)
    assert thread is not None
    thread.join(5)
    assert ran == ["ensure", "fetch", "ensure"]
    assert store.prepare_in_background(fetch=True) is None  # once per process


# ------------------------------------------------------------------ the agents' tools


def test_doc_tools_schema_and_results(library, monkeypatch, tmp_path_factory):
    base, _, _ = library
    monkeypatch.setenv(store.ENV, str(base))
    monkeypatch.setattr(tools_mod, "create_sdk_mcp_server", lambda n, version, tools: tools)
    run = RunDir.create(tmp_path_factory.mktemp("run"), "org/m")
    server = {t.name: t for t in tools_mod.build_server(run)}
    search, read = server["doc_search"], server["doc_read"]
    assert search.input_schema["required"] == ["query"]
    assert set(search.input_schema["properties"]) == {"query", "library", "k"}
    assert read.input_schema["required"] == ["id"]
    assert set(read.input_schema["properties"]) == {"id", "max_chars"}
    assert "no network" in search.description and "cite" in read.description

    def call(tool, **args):
        out = asyncio.run(tool.handler(args))
        return json.loads(out["content"][0]["text"])

    found = call(search, query="tl.dot_scaled", library="fakelang", k=2)
    assert len(found["results"]) == 2
    assert found["results"][0]["title"] == "fakelang.core.dot_scaled"
    chunk = call(read, id=found["results"][0]["id"])
    assert "e8m0 exponent" in chunk["text"] and chunk["source"].endswith("core.py:8")
    assert "error" in call(read, id="nope")

    monkeypatch.setattr(doclib, "search", lambda *a: 1 / 0)  # a broken library: an error
    assert "not available" in call(search, query="x")["error"]


def doc_call(tool_use_id, tool, **args):
    block = ToolUseBlock(id=tool_use_id, name=f"mcp__ka__{tool}", input=args)
    return AssistantMessage(content=[block], model="m")


def doc_result(tool_use_id, data, *, is_error=None):
    block = ToolResultBlock(tool_use_id=tool_use_id, content=json.dumps(data), is_error=is_error)
    return UserMessage(content=[block])


def test_lookups_record_doc_searches_reads_and_citations(tmp_path):
    items: list[dict] = []
    lookups = web.Lookups(items)
    hits = [{"id": "triton:abc", "title": "t"}, {"id": "ptx:def", "title": "p"}]
    lookups.see(doc_call("a", "doc_search", query="tl.dot_scaled", library="triton"))
    lookups.see(doc_result("a", {"query": "tl.dot_scaled", "results": hits}))
    chunk = {
        "id": "triton:abc",
        "title": "triton.language.core.dot_scaled",
        "library": "triton",
        "version": "3.8.0",
        "origin": "installed",
        "source": "/site-packages/triton/language/core.py:2427",
        "text": "Returns the matrix product",
        "read": ["triton:abc", "triton:abd"],
    }
    lookups.see(doc_call("b", "doc_read", id="triton:abc"))
    lookups.see(doc_result("b", chunk))
    lookups.see(doc_call("c", "doc_read", id="nope"))
    lookups.see(doc_result("c", {"error": "no chunk 'nope'"}))

    search, read, failed = items
    assert search["tool"] == "doc_search" and search["status"] == "ok"
    assert (search["query"], search["library"], search["results"]) == ("tl.dot_scaled", "triton", 2)
    assert search["hits"] == ["triton:abc", "ptx:def"]
    assert (read["tool"], read["id"], read["version"], read["chars"]) == (
        "doc_read",
        "triton:abc",
        "3.8.0",
        26,
    )
    assert read["source"].endswith("core.py:2427") and read["read"] == chunk["read"]
    assert (failed["status"], failed["reason"]) == ("error", "no chunk 'nope'")
    assert web.summary(items) == {
        "fetches": 0,
        "searches": 0,
        "denied": 0,
        "pages": 0,
        "doc_searches": 1,
        "doc_reads": 2,
        "doc_chunks": 2,
    }

    web.record(tmp_path, "kernel-t1", items)
    rows = read_jsonl(tmp_path / web.SOURCES_FILE)
    assert [r["tool"] for r in rows] == ["doc_search", "doc_read", "doc_read"]
    notes = tmp_path / "targets" / "t1" / "NOTES.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("[source] doc:triton:abc core.py — e8m0 scales per 32 elements\n")
    text = "\n".join(web.report_lines(tmp_path))
    assert "## Sources used" in text and "1 searches, 1 reads of 1 chunks in 1 agent" in text
    assert "| `triton:abc` triton.language.core.dot_scaled" in text
    assert "| triton 3.8.0 (installed) | `kernel-t1` | `targets/t1/NOTES.md` |" in text
    assert "“tl.dot_scaled”" in text and "pages fetched" not in text  # no web lookups


def test_every_session_gets_the_doc_tools_and_records_them(tmp_path, monkeypatch):
    seen = []

    async def fake_query(*, prompt, options):
        seen.append(options)
        yield doc_call("a", "doc_search", query="cp.async.bulk")
        yield doc_result("a", {"results": []})

    monkeypatch.setattr(runner, "query", fake_query)
    for allow in (True, False):
        common = {"prompt": "go", "system_append": "", "cwd": tmp_path, "mcp_server": None}
        common |= {"env": {}, "log_dir": tmp_path / "logs", "mcp_tools": []}
        cfg = OptimizeConfig(model_ref="m", allow_web=allow)
        out = asyncio.run(runner.run_agent("a", cfg=cfg, **common, tools=["Read"]))
        assert set(web.DOC_TOOLS) <= set(seen[-1].allowed_tools)
        assert [(i["tool"], i["results"]) for i in out.web] == [("doc_search", 0)]
        assert out.tool_calls == {"doc_search": 1}


def test_prompts_say_look_it_up_and_dossiers_start_from_the_library(tmp_path, monkeypatch):
    note = prompts.docs_note("kernel-attn-w2")
    assert "# Documentation library (doc_search / doc_read)" in note
    assert "Look an API up before you use it" in note and "`tl.dot_scaled`" in note
    assert "`NOTES.md`" in note and "[source] doc:<id> <source>" in note
    assert "Use WebFetch only for what the library does not have" in note
    assert "`plan.md` (and `research.md`)" in prompts.docs_note("research-attn")
    assert prompts.docs_note("refactor-x") == prompts.docs_note("librarian") == ""
    spec = {"id": "a", "module_class": "M", "backends": ["triton"]}
    dossier = prompts.dossier_prompt(spec, {}, Path("/r/research.md"), "t")
    assert "from the doc library first (`doc_search` / `doc_read`" in dossier

    for allow in (True, False):  # every session, with or without the web tools
        calls: list[dict] = []
        orch, _ = make_orchestrator(tmp_path / str(allow), monkeypatch, calls, allow_web=allow)
        asyncio.run(orch.run_all(until="kernels"))
        assert [c["name"] for c in calls] == ["kernel-t1", "kernel-t2"]
        assert all("# Documentation library (doc_search" in c["system"] for c in calls)


def test_orchestrator_counts_doc_lookups_in_costs(tmp_path, monkeypatch):
    calls: list[dict] = []
    orch, _ = make_orchestrator(tmp_path, monkeypatch, calls)
    items = [
        {"time": "t", "tool": "doc_search", "query": "q", "status": "ok", "results": 3},
        {"time": "t", "tool": "doc_read", "id": "ptx:1", "status": "ok", "read": ["ptx:1"]},
    ]

    async def agent(name, *, result=None, **_):
        result = result or AgentResult(name=name)
        result.web = [dict(i) for i in items]
        return result

    orch.agent_runner = agent
    kwargs = {"prompt": "p", "system_append": "", "cwd": orch.run.root, "mcp_tools": []}
    asyncio.run(orch._agent("kernel-t1", **kwargs))
    costs = read_json(orch.run.root / "costs.json")["kernel-t1"]
    assert costs["web"]["doc_searches"] == 1 and costs["web"]["doc_chunks"] == 1
    rows = read_jsonl(orch.run.root / web.SOURCES_FILE)
    assert [r["tool"] for r in rows] == ["doc_search", "doc_read"]


# ------------------------------------------------------------------ CLI


def test_cli_docs_commands(library, monkeypatch, capsys):
    base, _, _ = library
    monkeypatch.setenv(store.ENV, str(base))
    assert cli.main(["docs", "path"]) == 0 and capsys.readouterr().out.strip() == str(base)
    assert cli.main(["docs", "search", "dot_scaled", "--library", "fakelang", "-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "[fakelang 1.0] fakelang.core.dot_scaled" in out
    cid = out.split()[0]
    assert cli.main(["docs", "read", cid]) == 0
    assert "e8m0 exponent" in capsys.readouterr().out
    assert cli.main(["docs", "read", "nope"]) == 1
    assert cli.main(["docs", "search", "x", "--library", "nope"]) == 1
    capsys.readouterr()
    assert cli.main(["docs", "build", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "chunks in 3 shelves" in out and "Fake Guide 9.9 (web)" in out
    assert cli.main(["docs", "status"]) == 0 and "doc library:" in capsys.readouterr().out
