"""What the doc library is built from (issue #177): the documentation of the installed
packages and headers (:data:`INSTALLED`, :data:`HEADERS`), and the official web docs
(:func:`sites`) fetched once, politely, into a page cache (:class:`PageCache`).

* Installed: the Python sources of ``triton.language`` (and Gluon, the autotuner, the
  JIT), the CuTe DSL of ``nvidia-cutlass-dsl`` (``cutlass.cute``, pipelines, utils),
  TileLang's language / JIT / layouts, ``torch.utils.cpp_extension`` and ``torch.cuda``;
  the CUDA headers' Doxygen comments (runtime and driver API, launch attributes, FP8 / FP4
  conversions) and ``cublasLt.h``. Their version is the installed one.
* Web: the CUDA Programming Guide, Best Practices and Blackwell tuning guides, the PTX
  ISA, the cuBLAS reference (with cuBLASLt), the CUTLASS / CuTe DSL docs, the Triton docs
  and tutorials, the TileLang docs and the PyTorch pages of the installed torch version.
  Every URL (redirects too) must pass the WebFetch allowlist (``agent/web.py``) and the
  host's ``robots.txt``; requests carry :data:`USER_AGENT`, one at a time, at most one per
  :data:`DELAY_S` per host. A page is fetched once and kept (URL, fetch time, HTTP code,
  sha256); a failed fetch is retried by ``kernel-agent docs build`` or after a day.
  Offline, nothing is fetched and the cache is used as it is.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

USER_AGENT = "kernel-agent-docs/1 (+https://github.com/kadirnar/kernel-agent)"
DELAY_S = 0.5  # between two requests to one host
TIMEOUT_S = 60.0
RETRY_S = 86400.0  # a failed page is fetched again after this long (or by `docs build`)
PAGES_FILE = "pages.json"


# ------------------------------------------------------------------ installed packages


@dataclass(frozen=True)
class Installed:
    """The Python modules (and Markdown files) of an installed package, chunked per API."""

    name: str  # shelf name
    library: str
    title: str
    dist: str  # distribution whose version the shelf has
    package: str  # import name
    include: tuple[str, ...]  # globs under the package directory
    exclude: tuple[str, ...] = ()

    def root(self) -> Path | None:
        try:
            spec = importlib.util.find_spec(self.package)
        except (ImportError, ValueError):
            return None
        locations = list(spec.submodule_search_locations or []) if spec else []
        return Path(locations[0]) if locations else None

    def version(self) -> str | None:
        try:
            return importlib.metadata.version(self.dist)
        except importlib.metadata.PackageNotFoundError:
            return None

    def files(self, root: Path) -> list[Path]:
        found: dict[Path, None] = {}
        for pattern in self.include:
            for path in sorted(root.glob(pattern)):
                rel = path.relative_to(root).as_posix()
                if path.is_file() and not any(_match(rel, x) for x in self.exclude):
                    found[path] = None
        return list(found)


def _match(rel: str, pattern: str) -> bool:
    return Path(rel).match(pattern) or rel.startswith(pattern.rstrip("*"))


def module_name(path: Path, root: Path, package: str) -> str:
    """``cutlass/cute/nvgpu/warp/mma.py`` → ``cutlass.cute.nvgpu.warp.mma``."""
    parts = [package, *path.relative_to(root).with_suffix("").parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


INSTALLED = (
    Installed(
        "triton",
        "triton",
        "Triton",
        "triton",
        "triton",
        include=(
            "language/*.py",
            "language/extra/*.py",
            "language/extra/cuda/*.py",
            "experimental/gluon/language/*.py",
            "experimental/gluon/language/nvidia/**/*.py",
            "runtime/autotuner.py",
            "runtime/jit.py",
            "testing.py",
            "tools/tensor_descriptor.py",
            "tools/mxfp.py",
            "**/*.md",
        ),
        exclude=("language/semantic.py", "experimental/gluon/language/_semantic.py"),
    ),
    Installed(
        "cutlass-dsl",
        "cutlass",
        "CuTe DSL",
        "nvidia-cutlass-dsl",
        "cutlass",
        include=(
            "__init__.py",
            "cute/**/*.py",
            "pipeline/*.py",
            "utils/**/*.py",
            "cutlass_dsl/cutlass.py",
            "base_dsl/typing.py",
            "**/*.md",
            "**/*.rst",
        ),
        exclude=("cute/experimental/iket/*", "_mlir*"),
    ),
    Installed(
        "tilelang",
        "tilelang",
        "TileLang",
        "tilelang",
        "tilelang",
        include=(
            "__init__.py",
            "language/**/*.py",
            "jit/*.py",
            "autotuner/*.py",
            "layout/*.py",
            "tileop/**/*.py",
            "intrinsics/*.py",
            "*.md",
            "carver/*.md",
            "backend/*.md",
        ),
        exclude=("3rdparty/*", "language/ast/*", "language/parser/*", "layout/ascend.py"),
    ),
    Installed(
        "torch",
        "torch",
        "PyTorch",
        "torch",
        "torch",
        include=(
            "utils/cpp_extension.py",
            "cuda/__init__.py",
            "cuda/graphs.py",
            "cuda/streams.py",
            "cuda/memory.py",
            "cuda/nvtx.py",
            "cuda/green_contexts.py",
            "library.py",
            "_library/custom_ops.py",
            "compiler/__init__.py",
        ),
    ),
)


@dataclass(frozen=True)
class Headers:
    """C headers of the CUDA toolkit the kernels compile against, chunked per Doxygen
    comment."""

    name: str
    library: str
    title: str
    files: tuple[str, ...]
    version_file: str
    version_macros: tuple[str, ...]  # e.g. CUDA_VERSION (13000 → 13.0) or MAJOR, MINOR, PATCH

    def located(self) -> dict[str, Path]:
        """Header name → path, in the first include directory that has it."""
        out: dict[str, Path] = {}
        dirs = include_dirs()
        for name in self.files:
            for directory in dirs:
                if (directory / name).is_file():
                    out[name] = directory / name
                    break
        return out

    def version_path(self) -> Path | None:
        return next(
            (d / self.version_file for d in include_dirs() if (d / self.version_file).is_file()),
            None,
        )


def header_version(text: str, macros: tuple[str, ...]) -> str | None:
    """The version the ``#define``s of ``macros`` give: one (``CUDA_VERSION 13000`` →
    13.0) or major, minor, patch (→ 13.1.1)."""
    values = []
    for macro in macros:
        m = re.search(rf"#\s*define\s+{macro}\s+(\d+)", text)
        if not m:
            return None
        values.append(int(m.group(1)))
    if len(values) == 1:
        return f"{values[0] // 1000}.{values[0] % 1000 // 10}"
    return ".".join(map(str, values))


def include_dirs() -> list[Path]:
    """CUDA include directories, in the order the toolchain uses them (``CUDA_HOME``, the
    ``nvcc`` on PATH, the pip wheels, ``/usr/local/cuda``), without importing torch."""
    dirs: list[Path] = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if home := os.environ.get(env):
            dirs.append(Path(home) / "include")
    if nvcc := shutil.which("nvcc"):
        dirs.append(Path(nvcc).resolve().parent.parent / "include")
    try:
        from kernel_agent.toolchain import _pip_cuda_root

        if (root := _pip_cuda_root()) is not None:
            dirs.append(root / "include")
    except Exception:
        pass
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    for base in list(spec.submodule_search_locations or []) if spec else []:
        dirs += sorted(Path(base).glob("*/include"))
    dirs.append(Path("/usr/local/cuda/include"))
    return [d for d in dict.fromkeys(dirs) if d.is_dir()]


HEADERS = (
    Headers(
        "cuda-headers",
        "cuda",
        "CUDA headers",
        files=(
            "cuda_runtime_api.h",
            "driver_types.h",
            "cuda.h",
            "cuda_fp8.h",
            "cuda_fp6.h",
            "cuda_fp4.h",
            "cuda_bf16.h",
            "cuda_pipeline_primitives.h",
            "cuda_awbarrier_primitives.h",
        ),
        version_file="cuda.h",
        version_macros=("CUDA_VERSION",),
    ),
    Headers(
        "cublas-headers",
        "cublas",
        "cuBLAS headers",
        files=("cublasLt.h",),
        version_file="cublas_api.h",
        version_macros=("CUBLAS_VER_MAJOR", "CUBLAS_VER_MINOR", "CUBLAS_VER_PATCH"),
    ),
)


# ------------------------------------------------------------------ web docs


@dataclass(frozen=True)
class Site:
    """Official docs fetched once: the ``seeds``, and with a ``prefix`` the pages below it
    they link to (at most ``max_pages``, none whose URL contains a ``skip`` string)."""

    name: str
    library: str
    title: str
    seeds: tuple[str, ...]
    prefix: str = ""
    max_pages: int = 0  # 0: the seeds only
    skip: tuple[str, ...] = ()
    version: str = ""  # "" : read from the page (meta docs-version, the title), else "latest"


_SPHINX_SKIP = ("/_", "genindex", "search.html", "py-modindex", "/notices", "/contents.html")


def torch_docs_version() -> str:
    """``2.14`` for torch 2.14.1+cu130 (the versioned PyTorch docs); ``stable`` without
    torch."""
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return "stable"
    m = re.match(r"(\d+)\.(\d+)", version)
    return f"{m.group(1)}.{m.group(2)}" if m else "stable"


def sites() -> tuple[Site, ...]:
    """The web docs of the library."""
    nv = "https://docs.nvidia.com/cuda"
    torch_v = torch_docs_version()
    torch_docs = f"https://docs.pytorch.org/docs/{torch_v}"
    return (
        Site(
            "cuda-programming-guide",
            "cuda",
            "CUDA Programming Guide",
            (f"{nv}/cuda-programming-guide/index.html",),
            prefix=f"{nv}/cuda-programming-guide/",
            max_pages=80,
            skip=(*_SPHINX_SKIP, "/part"),
        ),
        Site(
            "cuda-best-practices",
            "cuda",
            "CUDA C++ Best Practices Guide",
            (f"{nv}/cuda-c-best-practices-guide/index.html",),
        ),
        Site(
            "blackwell-tuning-guide",
            "cuda",
            "Blackwell Tuning Guide",
            (f"{nv}/blackwell-tuning-guide/index.html",),
        ),
        Site("ptx-isa", "ptx", "PTX ISA", (f"{nv}/parallel-thread-execution/index.html",)),
        Site(
            "inline-ptx-assembly",
            "ptx",
            "Inline PTX Assembly",
            (f"{nv}/inline-ptx-assembly/index.html",),
        ),
        Site("cublas", "cublas", "cuBLAS", (f"{nv}/cublas/index.html",)),
        Site(
            "cutlass",
            "cutlass",
            "CUTLASS",
            ("https://docs.nvidia.com/cutlass/latest/index.html",),
            prefix="https://docs.nvidia.com/cutlass/latest/media/docs/",
            max_pages=200,
            skip=_SPHINX_SKIP,
        ),
        Site(
            "triton",
            "triton",
            "Triton",
            ("https://triton-lang.org/main/index.html",),
            prefix="https://triton-lang.org/main/",
            max_pages=120,
            # the API pages document the installed sources, which the library indexes
            skip=(*_SPHINX_SKIP, "/python-api/generated/", "/gluon/api/", "/dialects/"),
            version="main",
        ),
        Site(
            "tilelang",
            "tilelang",
            "TileLang",
            ("https://tilelang.com/index.html",),
            prefix="https://tilelang.com/",
            max_pages=80,
            skip=(*_SPHINX_SKIP, "/autoapi/", "privacy", "/developer_guide/", "metal_"),
        ),
        Site(
            "pytorch",
            "torch",
            "PyTorch",
            (
                f"{torch_docs}/notes/cuda.html",
                f"{torch_docs}/cpp_extension.html",
                f"{torch_docs}/notes/custom_operators.html",
                f"{torch_docs}/torch.compiler_cudagraph_trees.html",
            ),
            version=torch_v,
        ),
    )


def page_version(site: Site, parsed: dict[str, Any]) -> str:
    """A web shelf's version: the site's, a ``docs-version`` meta tag, a version in the
    page title (``PTX ISA 9.4``), else ``latest``."""
    if site.version:
        return site.version
    meta = parsed.get("meta") or {}
    if v := (meta.get("docs-version") or meta.get("docsearch:version") or "").strip():
        return v
    if m := re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", parsed.get("title") or ""):
        return m.group(1)
    return "latest"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class _Guarded(urllib.request.HTTPRedirectHandler):
    """Redirects only to allowed URLs (the WebFetch allowlist)."""

    def __init__(self, hosts: list[str]) -> None:
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # urllib's signature
        from kernel_agent.agent.web import allowed

        if not allowed(newurl, self.hosts):
            why = f"redirect off the allowlist: {newurl}"
            raise urllib.error.HTTPError(newurl, code, why, headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


Fetch = Callable[[str], tuple[int, str, bytes]]  # url -> (HTTP code, final URL, body)


def http_get(hosts: list[str]) -> Fetch:
    """A GET with :data:`USER_AGENT` that follows redirects on ``hosts`` only."""
    opener = urllib.request.build_opener(_Guarded(hosts))

    def get(url: str) -> tuple[int, str, bytes]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,*/*"},
        )
        try:
            with opener.open(request, timeout=TIMEOUT_S) as response:
                body: bytes = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return int(response.status), str(response.geturl()), body
        except urllib.error.HTTPError as exc:
            return int(exc.code), url, b""

    return get


@dataclass
class PageCache:
    """Fetched pages under ``root/pages/`` (gzip), indexed by URL in ``pages.json``:
    file, site, fetch time, HTTP code, size, sha256, or the error of a failed fetch."""

    root: Path
    fetch: Fetch | None = None  # None: offline, the cache only
    hosts: list[str] = field(default_factory=list)
    log: Callable[[str], None] = print
    refresh: bool = False  # fetch again pages fetched before (``docs build --refresh``)
    retry_failed: bool = False  # fetch again pages that failed less than RETRY_S ago
    delay_s: float = DELAY_S
    index: dict[str, dict[str, Any]] = field(default_factory=dict)
    fetched: int = 0
    _last: dict[str, float] = field(default_factory=dict)
    _robots: dict[str, urllib.robotparser.RobotFileParser | None] = field(default_factory=dict)
    _offline: bool = False

    def __post_init__(self) -> None:
        self.index = read_pages(self.root)

    def save(self) -> None:
        path = self.root / PAGES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, indent=1, sort_keys=True))
        tmp.replace(path)

    def cached(self, url: str) -> str | None:
        entry = self.index.get(url)
        if not entry or entry.get("status") != "ok":
            return None
        try:
            return gzip.decompress((self.root / "pages" / entry["file"]).read_bytes()).decode(
                "utf-8", "replace"
            )
        except (OSError, EOFError, gzip.BadGzipFile):
            return None

    def _wait(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None and (pause := self.delay_s - (time.monotonic() - last)) > 0:
            time.sleep(pause)
        self._last[host] = time.monotonic()

    def _robots_ok(self, url: str) -> bool:
        parts = urlsplit(url)
        host = parts.netloc
        if host not in self._robots:
            robots: urllib.robotparser.RobotFileParser | None = None
            assert self.fetch is not None
            self._wait(host)
            try:
                code, _, body = self.fetch(f"{parts.scheme}://{host}/robots.txt")
                if code == 200:
                    robots = urllib.robotparser.RobotFileParser()
                    robots.parse(body.decode("utf-8", "replace").splitlines())
            except (OSError, ValueError):
                robots = None
            self._robots[host] = robots
        robots = self._robots[host]
        return robots is None or robots.can_fetch(USER_AGENT, url)

    def get(self, url: str, site: str, hops: int = 0) -> str | None:
        """The page at ``url``: cached, else fetched (when online, allowed by the allowlist
        and robots.txt); None when it cannot be had. ``hops``: meta refreshes followed."""
        from kernel_agent.agent.web import allowed

        entry = self.index.get(url) or {}
        if not self.refresh and (html := self.cached(url)) is not None:
            return html
        if self.fetch is None or self._offline:
            return self.cached(url)
        failed_lately = time.time() - float(entry.get("at", 0)) < RETRY_S
        if entry.get("status") == "error" and failed_lately and not self.retry_failed:
            return None
        if not allowed(url, self.hosts):
            self.log(f"docs: {url}: not on the allowlist, skipped")
            return None
        try:
            if not self._robots_ok(url):
                self._record(url, site, error="disallowed by robots.txt")
                return None
            self._wait(urlsplit(url).netloc)
            code, final, body = self.fetch(url)
        except (OSError, ValueError) as exc:  # offline, DNS, timeout: stop fetching
            self._offline = True
            self.log(f"docs: {url}: {exc!r}; offline? using cached pages only")
            return self.cached(url)
        if code != 200 or not body:
            self._record(url, site, error=f"HTTP {code}", code=code)
            return self.cached(url)
        html = body.decode("utf-8", "replace")
        refresh = _meta_refresh(html, final)  # a redirect page (moved docs)
        if refresh and refresh != url and hops < 3 and allowed(refresh, self.hosts):
            target = self.get(refresh, site, hops + 1)
            if target is not None:  # the page under both URLs (the crawl asks for url)
                self.index[url] = {**self.index[refresh], "final_url": refresh}
            return target
        name = hashlib.sha1(url.encode()).hexdigest()[:20] + ".html.gz"
        (self.root / "pages").mkdir(parents=True, exist_ok=True)
        (self.root / "pages" / name).write_bytes(gzip.compress(body))
        self.index[url] = {
            "file": name,
            "site": site,
            "status": "ok",
            "fetched": _now(),
            "at": time.time(),
            "code": code,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            **({"final_url": final} if final != url else {}),
        }
        self.fetched += 1
        return html

    def _record(self, url: str, site: str, *, error: str, code: int | None = None) -> None:
        old = self.index.get(url) or {}
        if old.get("status") == "ok":  # keep the page we have; note the failed refresh
            old["refresh_error"] = error
            return
        self.index[url] = {"site": site, "status": "error", "error": error, "fetched": _now()}
        self.index[url]["at"] = time.time()
        if code is not None:
            self.index[url]["code"] = code
        self.log(f"docs: {url}: {error}")


def _meta_refresh(html: str, base: str) -> str | None:
    if len(html) > 20000:
        return None
    m = re.search(r'http-equiv="refresh"\s+content="\d+;\s*url=([^"]+)"', html, re.I)
    return urljoin(base, m.group(1).strip()) if m else None


def read_pages(root: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads((root / PAGES_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


_HREF = re.compile(r'href="([^"]+)"')


def links(html: str, base: str, site: Site) -> list[str]:
    """The pages below ``site.prefix`` that ``html`` links to (no fragment or query)."""
    out: list[str] = []
    for href in _HREF.findall(html):
        url = urljoin(base, href.replace("&amp;", "&")).split("#", 1)[0].split("?", 1)[0]
        if not url.startswith(site.prefix) or any(s in url for s in site.skip):
            continue
        if url.endswith("/"):
            url += "index.html"
        if url.endswith(".html"):
            out.append(url)
    return list(dict.fromkeys(out))


def crawl(site: Site, cache: PageCache) -> list[tuple[str, str]]:
    """``(url, html)`` of a site's pages: the seeds, then (with a prefix) the pages they
    link to, breadth first, at most ``max_pages`` (from the cache, or fetched)."""
    queue = list(site.seeds)
    seen = set(queue)
    pages: list[tuple[str, str]] = []
    limit = max(site.max_pages, len(site.seeds))
    while queue and len(pages) < limit:
        url = queue.pop(0)
        html = cache.get(url, site.name)
        if html is None:
            continue
        pages.append((url, html))
        if site.prefix:
            for link in links(html, url, site):
                if link not in seen:
                    seen.add(link)
                    queue.append(link)
    return pages


def site_key(site: Site, pages: Iterable[tuple[str, dict[str, Any]]]) -> str:
    """What a web shelf was built from: the sha256 of each cached page of the site."""
    rows = sorted(f"{u} {e.get('sha256')}" for u, e in pages if e.get("site") == site.name)
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()[:16] if rows else ""
