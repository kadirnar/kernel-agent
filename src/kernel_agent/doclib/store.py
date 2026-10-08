"""The doc library on disk, its search and its CLI (issue #177).

Layout of :func:`root` (``$KERNEL_AGENT_DOCS`` or ``~/.cache/kernel-agent/docs``)::

    manifest.json                 shelves (library, version, origin, chunks, key), missing
    shelves/<name>-<version>.jsonl         the chunks of a shelf, one JSON per line
    shelves/<name>-<version>.index.json    its BM25 postings (bm25.postings)
    pages.json  pages/<sha1>.html.gz       the web page cache (sources.PageCache)

A shelf is one source at one version: an installed package or header set
(``origin: installed``, keyed by the installed version and path) or one web site
(``origin: web``, keyed by the sha256 of its cached pages). :func:`ensure` (re)builds the
shelves whose key changed, without network: it runs lazily on the first search, so a
version change rebuilds that shelf and drops the old one. :func:`fetch_web` fetches the
web pages (``kernel-agent docs build``, or :func:`prepare_in_background` at the start of
a run with the web tools); offline it does nothing and the cache is used as it is.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_agent.doclib import bm25, chunks, sources

FORMAT = 1  # of the chunks and the index: a change rebuilds every shelf
ENV = "KERNEL_AGENT_DOCS"
PREPARE_ENV = "KERNEL_AGENT_DOCS_PREPARE"  # "0": no build / fetch when a run starts
MANIFEST = "manifest.json"
READ_CHARS = 8000  # doc_read default
MAX_READ_CHARS = 20000
MAX_K = 20
INSTALLED, WEB = "installed", "web"
Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def root() -> Path:
    """The library directory: ``$KERNEL_AGENT_DOCS``, else ``docs/`` in kernel-agent's
    cache directory."""
    if env := os.environ.get(ENV):
        return Path(env).expanduser()
    from kernel_agent.toolchain import CACHE_DIR

    return CACHE_DIR / "docs"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_manifest(base: Path) -> dict[str, Any]:
    try:
        data = json.loads((base / MANIFEST).read_text())
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        data = {"format": FORMAT, "shelves": {}, "missing": {}}
    data.setdefault("shelves", {})
    data.setdefault("missing", {})
    return data


def _write_json(path: Path, data: Any, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, separators=None if indent else (",", ":")))
    tmp.replace(path)


@contextlib.contextmanager
def _locked(path: Path, *, wait: bool = True) -> Iterator[bool]:
    """An exclusive ``flock`` on ``path`` (between processes and threads); yields False
    when ``wait`` is off and another holder has it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ------------------------------------------------------------------ plans: what to build


@dataclass
class Plan:
    """A shelf as the installed versions and the page cache say it should be."""

    name: str
    library: str
    origin: str
    title: str
    version: str = ""
    key: str = ""  # "" : nothing to build from (``missing`` says why)
    missing: str = ""
    build: Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]] | None = None

    @property
    def file(self) -> str:
        return f"{self.name}-{re.sub(r'[^A-Za-z0-9_.+-]', '_', self.version)}"


def _installed_plan(spec: sources.Installed) -> Plan:
    plan = Plan(spec.name, spec.library, INSTALLED, spec.title)
    pkg, version = spec.root(), spec.version()
    if pkg is None or version is None:
        plan.missing = f"{spec.dist} is not installed"
        return plan
    plan.version, plan.key = version, f"{FORMAT}|{version}|{pkg}"

    def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        out: list[dict[str, Any]] = []
        files = spec.files(pkg)
        for path in files:
            rel = path.relative_to(pkg)
            if path.suffix == ".py":
                out += chunks.python_chunks(path, sources.module_name(path, pkg, spec.package))
            elif path.suffix in (".md", ".rst"):
                out += chunks.markdown_chunks(path, f"{spec.title} {rel.as_posix()}")
        return out, {"sources": len(files), "path": str(pkg)}

    plan.build = build
    return plan


@functools.lru_cache(maxsize=32)
def _header_version(path: str, mtime_ns: int, macros: tuple[str, ...]) -> str | None:
    try:
        return sources.header_version(Path(path).read_text(errors="replace"), macros)
    except OSError:
        return None


def _headers_plan(spec: sources.Headers) -> Plan:
    plan = Plan(spec.name, spec.library, INSTALLED, spec.title)
    found = spec.located()
    vpath = spec.version_path()
    version = None
    if found and vpath is not None:
        version = _header_version(str(vpath), vpath.stat().st_mtime_ns, spec.version_macros)
    if not found or version is None:
        plan.missing = "CUDA headers not found (CUDA_HOME, nvcc, the nvidia pip wheels)"
        return plan
    files = [found[n] for n in spec.files if n in found]
    plan.version = version
    plan.key = f"{FORMAT}|{version}|" + ",".join(str(p) for p in files)

    def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in files:
            out += chunks.header_chunks(path)
        return out, {"sources": len(files), "path": str(files[0].parent)}

    plan.build = build
    return plan


def _web_plan(site: sources.Site, base: Path, pages: dict[str, dict[str, Any]]) -> Plan:
    plan = Plan(f"web-{site.name}", site.library, WEB, site.title)
    key = sources.site_key(site, pages.items())
    if not key:
        plan.missing = "not fetched (kernel-agent docs build)"
        return plan
    plan.key = f"{FORMAT}|{key}"
    plan.version = site.version or "?"  # read from the pages when built

    def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cache = sources.PageCache(base)  # offline: the cached pages only
        out: list[dict[str, Any]] = []
        version = ""
        fetched: list[str] = []
        crawled = sources.crawl(site, cache)
        for url, html in crawled:
            page = chunks.html_sections(html)
            version = version or sources.page_version(site, page)
            when = str(cache.index.get(url, {}).get("fetched", ""))[:10]
            fetched.append(when)
            for chunk in chunks.section_chunks(page, url, site.title):
                chunk["fetched"] = when
                out.append(chunk)
        plan.version = version or "latest"
        dates = sorted(d for d in fetched if d)
        return out, {
            "sources": len(crawled),
            "fetched": dates[-1] if dates else "",
            "seeds": list(site.seeds),
        }

    plan.build = build
    return plan


def plans(base: Path) -> list[Plan]:
    """Every shelf the library should have now (installed versions, cached pages)."""
    pages = sources.read_pages(base)
    out = [_installed_plan(s) for s in sources.INSTALLED]
    out += [_headers_plan(s) for s in sources.HEADERS]
    out += [_web_plan(s, base, pages) for s in sources.sites()]
    return out


def _finish(plan: Plan, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ids, library, version and origin on a shelf's chunks."""
    seen: set[str] = set()
    for chunk in items:
        digest = hashlib.sha1(f"{plan.name}|{chunk['source']}|{chunk['title']}".encode())
        cid = f"{plan.library}:{digest.hexdigest()[:10]}"
        n = 1
        while cid in seen:
            n += 1
            cid = f"{plan.library}:{digest.hexdigest()[:10]}-{n}"
        seen.add(cid)
        chunk.update(id=cid, library=plan.library, version=plan.version, origin=plan.origin)
    return items


def _current(base: Path, manifest: dict[str, Any], plan: Plan) -> bool:
    shelf = manifest["shelves"].get(plan.name)
    return bool(
        shelf
        and shelf.get("key") == plan.key
        and (base / "shelves" / f"{shelf['file']}.jsonl").is_file()
        and (base / "shelves" / f"{shelf['file']}.index.json").is_file()
    )


def _remove(base: Path, shelf: dict[str, Any] | None) -> None:
    if not shelf:
        return
    for suffix in (".jsonl", ".index.json"):
        (base / "shelves" / f"{shelf['file']}{suffix}").unlink(missing_ok=True)


def ensure(base: Path | None = None, *, log: Log = _log) -> dict[str, Any]:
    """Build the shelves whose installed version or cached pages changed (no network) and
    drop the shelves of versions no longer installed. Returns the manifest."""
    base = base or root()
    manifest = read_manifest(base)
    wanted = plans(base)
    if all(_current(base, manifest, p) for p in wanted if p.key) and {
        p.name for p in wanted if p.key
    } == set(manifest["shelves"]):
        return manifest
    with _locked(base / ".build.lock"):
        manifest = read_manifest(base)  # another process may have built it meanwhile
        for plan in wanted:
            old = manifest["shelves"].get(plan.name)
            if not plan.key:
                manifest["missing"][plan.name] = plan.missing
                _remove(base, manifest["shelves"].pop(plan.name, None))
                continue
            manifest["missing"].pop(plan.name, None)
            if _current(base, manifest, plan):
                continue
            assert plan.build is not None
            start = time.perf_counter()
            items, extra = plan.build()
            _finish(plan, items)
            (base / "shelves").mkdir(parents=True, exist_ok=True)
            path = base / "shelves" / f"{plan.file}.jsonl"
            tmp = path.with_suffix(".tmp")
            with tmp.open("w") as out:
                for item in items:
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
            tmp.replace(path)
            _write_json(base / "shelves" / f"{plan.file}.index.json", bm25.postings(items))
            if old and old.get("file") != plan.file:
                _remove(base, old)
            seconds = round(time.perf_counter() - start, 1)
            manifest["shelves"][plan.name] = {
                "library": plan.library,
                "origin": plan.origin,
                "title": plan.title,
                "version": plan.version,
                "chunks": len(items),
                "file": plan.file,
                "key": plan.key,
                "built": _now(),
                "seconds": seconds,
                **extra,
            }
            log(
                f"docs: {plan.name} {plan.version}: {len(items)} chunks from "
                f"{_sources(manifest['shelves'][plan.name])} ({seconds} s)"
            )
        for name in set(manifest["shelves"]) - {p.name for p in wanted}:
            _remove(base, manifest["shelves"].pop(name))
        manifest["updated"] = _now()
        _write_json(base / MANIFEST, manifest, indent=1)
    return manifest


def fetch_web(
    base: Path | None = None,
    *,
    hosts: list[str] | None = None,
    refresh: bool = False,
    retry_failed: bool = False,
    log: Log = _log,
    fetch: sources.Fetch | None = None,
    delay_s: float = sources.DELAY_S,
) -> dict[str, Any]:
    """Fetch the web docs that are not cached yet (``refresh``: all of them again) over
    the allowlist (``hosts``, default ``web.domains()``). Returns ``fetched`` (pages),
    ``offline`` and ``busy`` (another process is fetching)."""
    from kernel_agent.agent.web import domains

    base = base or root()
    hosts = hosts or domains()
    with _locked(base / ".fetch.lock", wait=False) as mine:
        if not mine:
            log("docs: another process is fetching the web docs; skipped")
            return {"fetched": 0, "offline": False, "busy": True}
        cache = sources.PageCache(
            base,
            fetch=fetch or sources.http_get(hosts),
            hosts=hosts,
            log=log,
            refresh=refresh,
            retry_failed=retry_failed,
            delay_s=delay_s,
        )
        for site in sources.sites():
            before = cache.fetched
            pages = sources.crawl(site, cache)
            cache.save()
            if cache.fetched > before:
                log(f"docs: {site.name}: {len(pages)} pages ({cache.fetched - before} fetched)")
        return {"fetched": cache.fetched, "offline": cache._offline, "busy": False}


def build(
    base: Path | None = None,
    *,
    fetch: bool = True,
    refresh: bool = False,
    hosts: list[str] | None = None,
    log: Log = _log,
) -> dict[str, Any]:
    """``kernel-agent docs build``: the installed shelves, then (``fetch``) the web pages,
    then their shelves. Returns the manifest."""
    base = base or root()
    manifest = ensure(base, log=log)
    if fetch:
        fetch_web(base, hosts=hosts, refresh=refresh, retry_failed=True, log=log)
        manifest = ensure(base, log=log)
    return manifest


_background: threading.Thread | None = None


def prepare_in_background(
    *, fetch: bool, hosts: list[str] | None = None, log: Log = _log
) -> threading.Thread | None:
    """Build the library in a daemon thread (once per process): the installed shelves,
    and with ``fetch`` the web pages not cached yet. Search waits for nothing but a build
    in progress; a failure is logged, never raised. ``$KERNEL_AGENT_DOCS_PREPARE=0``: no
    thread (searches still build the installed shelves on first use)."""
    global _background
    if _background is not None or os.environ.get(PREPARE_ENV, "1") == "0":
        return None

    def work() -> None:
        try:
            ensure(log=log)
            if fetch:
                fetch_web(hosts=hosts, log=log)
                ensure(log=log)
        except Exception as exc:  # documentation is a bonus: never stop a run for it
            log(f"docs: building the doc library failed: {exc!r}")

    _background = threading.Thread(target=work, name="doclib", daemon=True)
    _background.start()
    return _background


# ------------------------------------------------------------------ search and read


@dataclass
class Library:
    """The loaded shelves: their BM25 index and the chunks by id."""

    manifest: dict[str, Any]
    index: bm25.Index
    by_id: dict[str, tuple[list[dict[str, Any]], int]]

    @property
    def libraries(self) -> dict[str, list[str]]:
        """Library → its shelves as ``<version> (<origin>)``."""
        out: dict[str, list[str]] = {}
        for shelf in self.manifest["shelves"].values():
            out.setdefault(shelf["library"], []).append(f"{shelf['version']} ({shelf['origin']})")
        return dict(sorted(out.items()))


_loaded: dict[Path, tuple[int, Library]] = {}
_load_lock = threading.Lock()


def load(base: Path | None = None, *, build_missing: bool = True) -> Library:
    """The library at ``base`` (built first when ``build_missing``), cached per process
    until its manifest changes."""
    base = base or root()
    manifest = ensure(base) if build_missing else read_manifest(base)
    path = base / MANIFEST
    stamp = path.stat().st_mtime_ns if path.is_file() else 0
    with _load_lock:
        cached = _loaded.get(base)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        index = bm25.Index()
        by_id: dict[str, tuple[list[dict[str, Any]], int]] = {}
        for shelf in manifest["shelves"].values():
            stem = base / "shelves" / shelf["file"]
            try:
                with (stem.parent / f"{stem.name}.jsonl").open() as handle:
                    items = [json.loads(line) for line in handle if line.strip()]
                postings = json.loads((stem.parent / f"{stem.name}.index.json").read_text())
            except (OSError, ValueError):
                continue
            index.add(shelf["library"], items, postings)
            for i, item in enumerate(items):
                by_id[item["id"]] = (items, i)
        library = Library(manifest, index, by_id)
        _loaded[base] = (stamp, library)
        return library


def _libraries(raw: str | list[str] | None) -> list[str]:
    if not raw:
        return []
    items = raw if isinstance(raw, list) else re.split(r"[,\s]+", raw)
    return [i.strip().lower() for i in items if i and i.strip()]


def search(
    query: str,
    library: str | list[str] | None = None,
    k: int = 8,
    *,
    base: Path | None = None,
) -> dict[str, Any]:
    """``doc_search``: the ``k`` best chunks for ``query``, each with its id, title,
    library, version, origin, source and a snippet. ``library``: one name or several
    (comma-separated); an unknown one returns the names there are."""
    lib = load(base)
    wanted = _libraries(library)
    if unknown := [w for w in wanted if w not in lib.libraries]:
        return {
            "error": f"no library {', '.join(unknown)} in the doc library",
            "libraries": lib.libraries,
        }
    k = max(1, min(int(k or 8), MAX_K))
    hits = lib.index.search(str(query or ""), wanted or None, k)
    out: dict[str, Any] = {
        "query": query,
        "results": [
            {
                "id": c["id"],
                "title": c["title"],
                "library": c["library"],
                "version": c["version"],
                "origin": c["origin"],
                "source": c["source"],
                "snippet": bm25.snippet(c["text"], str(query or "")),
                "score": round(score, 2),
            }
            for score, c in hits
        ],
    }
    if not hits:
        out["libraries"] = lib.libraries
        out["hint"] = "no match: try the identifier alone, or another library"
    return out


def read(id_: str, max_chars: int = READ_CHARS, *, base: Path | None = None) -> dict[str, Any]:
    """``doc_read``: the chunk ``id_`` and, while under ``max_chars``, the chunks after it
    in the same page or file. ``next``: the id to read on from (None at the end)."""
    lib = load(base)
    found = lib.by_id.get(str(id_ or "").strip())
    if found is None:
        return {"error": f"no chunk {id_!r} (ids come from doc_search)"}
    items, i = found
    limit = max(500, min(int(max_chars or READ_CHARS), MAX_READ_CHARS))
    first = items[i]
    parts = [first["text"]]
    used = [first["id"]]
    size = len(first["text"])
    j = i + 1
    while j < len(items) and items[j]["doc"] == first["doc"]:
        more = items[j]
        piece = f"## {more['title']}\n(source: {more['source']})\n\n{more['text']}"
        if size + len(piece) > limit:
            break
        parts.append(piece)
        used.append(more["id"])
        size += len(piece) + 2
        j += 1
    text = "\n\n".join(parts)
    if len(text) > limit:
        text = text[:limit] + "\n… (cut: read again with a larger max_chars)"
    out: dict[str, Any] = {
        "id": first["id"],
        "title": first["title"],
        "library": first["library"],
        "version": first["version"],
        "origin": first["origin"],
        "source": first["source"],
        **({"fetched": first["fetched"]} if first.get("fetched") else {}),
        "text": text,
        "read": used,
        "next": items[j]["id"] if j < len(items) and items[j]["doc"] == first["doc"] else None,
        "prev": items[i - 1]["id"] if i > 0 and items[i - 1]["doc"] == first["doc"] else None,
    }
    return out


# ------------------------------------------------------------------ coverage


def _sources(shelf: dict[str, Any]) -> str:
    n = shelf.get("sources", "?")
    word = "page" if shelf.get("origin") == WEB else "file"
    return f"{n} {word}{'' if n == 1 else 's'}"


def describe(base: Path | None = None) -> list[str]:
    """``doctor``'s lines: the library's chunks per library and version, and what is
    missing (not installed, web docs not fetched)."""
    base = base or root()
    manifest = read_manifest(base)
    shelves = manifest["shelves"]
    total = sum(int(s.get("chunks", 0)) for s in shelves.values())
    lines = [
        f"doc library: {base}: {total} chunks in {len(shelves)} shelves "
        "(agents: doc_search / doc_read)"
    ]
    by_library: dict[str, list[str]] = {}
    for shelf in sorted(shelves.values(), key=lambda s: (s["library"], s["origin"], s["title"])):
        when = f", fetched {shelf['fetched']}" if shelf.get("fetched") else ""
        by_library.setdefault(shelf["library"], []).append(
            f"{shelf['title']} {shelf['version']} ({shelf['origin']}): {shelf['chunks']} chunks "
            f"from {_sources(shelf)}{when}"
        )
    for library, rows in sorted(by_library.items()):
        lines.append(f"  {library:9s} " + "; ".join(rows))
    missing = manifest.get("missing") or {}
    if not shelves:
        lines.append("  not built yet: it builds on the first search, or `kernel-agent docs build`")
    if missing:
        lines.append(
            "  missing: " + "; ".join(f"{name}: {why}" for name, why in sorted(missing.items()))
        )
    return lines


# ------------------------------------------------------------------ CLI


def add_parser(sub: Any) -> argparse.ArgumentParser:
    """The ``kernel-agent docs`` subcommands (``sub``: the CLI's subparsers)."""
    p: argparse.ArgumentParser = sub.add_parser(
        "docs",
        help="local doc library the agents search: build, status, search, read, path",
        description="The documentation of the installed Triton, CuTe DSL, TileLang, "
        "PyTorch and CUDA headers, plus the official CUDA / PTX / cuBLAS / CUTLASS / Triton "
        f"/ TileLang web docs, chunked and indexed (BM25). Location: ${ENV} or "
        "~/.cache/kernel-agent/docs.",
    )
    docs = p.add_subparsers(dest="docs_command", required=True)
    q = docs.add_parser("build", help="build the installed shelves and fetch the web docs")
    q.add_argument("--offline", action="store_true", help="no network: installed + cached")
    q.add_argument("--refresh", action="store_true", help="fetch every web page again")
    q.add_argument(
        "--web-domain",
        action="append",
        default=[],
        metavar="HOST",
        help="also allow HOST (besides the WebFetch allowlist)",
    )
    docs.add_parser("status", help="coverage per library and version")
    q = docs.add_parser("search", help="search the library like doc_search")
    q.add_argument("query")
    q.add_argument("--library", help="e.g. triton, cutlass, ptx, cuda, cublas, tilelang, torch")
    q.add_argument("-k", type=int, default=8)
    q = docs.add_parser("read", help="read a chunk like doc_read")
    q.add_argument("id")
    q.add_argument("--max-chars", type=int, default=READ_CHARS)
    docs.add_parser("path", help="print the library directory")
    return p


def main(ns: argparse.Namespace) -> int:
    command = ns.docs_command
    if command == "path":
        print(root())
        return 0
    if command == "build":
        from kernel_agent.agent.web import domains

        build(fetch=not ns.offline, refresh=ns.refresh, hosts=domains(ns.web_domain))
        print("\n".join(describe()))
        return 0
    if command == "status":
        print("\n".join(describe()))
        return 0
    if command == "search":
        result = search(ns.query, ns.library, ns.k)
        if "error" in result:
            print(json.dumps(result, indent=1))
            return 1
        for hit in result["results"]:
            print(f"{hit['id']}  [{hit['library']} {hit['version']}] {hit['title']}")
            print(f"    {hit['source']}")
            print(f"    {hit['snippet']}")
        return 0
    result = read(ns.id, ns.max_chars)
    if "error" in result:
        print(result["error"])
        return 1
    print(f"# {result['title']}  [{result['library']} {result['version']}]")
    print(f"source: {result['source']}\n")
    print(result["text"])
    if result.get("next"):
        print(f"\n(next: {result['next']})")
    return 0
