"""Documents → searchable chunks of the doc library (issue #177).

A chunk is a dict: ``title``, ``text``, ``source`` (``path:line`` of an installed file,
or the URL of a web page with its section anchor) and ``doc`` (the file or page it comes
from: :mod:`kernel_agent.doclib.store` reads on through the chunks of one ``doc``). The
store adds the id, library, version and origin.

* :func:`python_chunks`: one chunk per public function, class and documented method of
  an installed module (signature + docstring; the code of a short undocumented one).
* :func:`header_chunks`: one chunk per Doxygen comment of a C header and the
  declaration after it (``cuda_runtime_api.h``, ``cublasLt.h``, ...).
* :func:`html_sections` + :func:`section_chunks`: one chunk per section of a Sphinx
  page (headings, and the ``dt`` of an API object), titled by its heading path.
* :func:`markdown_chunks`: one chunk per section of a Markdown file shipped in a package.

Long sections are split into parts of about :data:`MAX_CHARS` (:func:`split_long`).
"""

from __future__ import annotations

import ast
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

MAX_CHARS = 3000  # of a chunk's text; longer sections become parts
MIN_CHARS = 40  # a section with less text (a heading of headings) is not a chunk
SHORT_SOURCE_LINES = 25  # an undocumented function this short is indexed with its code


def split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """``text`` in parts of at most about ``limit`` characters, cut at blank lines outside
    code fences when possible, else at line ends."""
    if len(text) <= limit * 1.2:
        return [text]
    blocks: list[str] = []
    fence = False
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            fence = not fence
        if not line.strip() and not fence and current:
            blocks.append("\n".join(current))
            current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    parts: list[str] = []
    buf = ""
    for block in blocks:
        pieces = [block]
        if len(block) > limit:  # one huge block (a table, a long listing): cut at lines
            pieces, cut = [], ""
            for line in block.split("\n"):
                if cut and len(cut) + len(line) + 1 > limit:
                    pieces.append(cut)
                    cut = ""
                cut = f"{cut}\n{line}" if cut else line[: limit * 2]
            pieces.append(cut)
        for piece in pieces:
            if buf and len(buf) + len(piece) + 2 > limit:
                parts.append(buf)
                buf = ""
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        parts.append(buf)
    return [p.strip("\n") for p in parts if p.strip()]


def _parts(title: str, text: str, source: str, doc: str) -> list[dict[str, Any]]:
    pieces = split_long(text.strip())
    n = len(pieces)
    return [
        {
            "title": title if n == 1 else f"{title} ({i}/{n})",
            "text": piece,
            "source": source,
            "doc": doc,
        }
        for i, piece in enumerate(pieces, 1)
    ]


# ------------------------------------------------------------------ Python modules


def _public(name: str) -> bool:
    return not name.startswith("_")


def _line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """Where a definition starts: its first decorator, else its ``def`` / ``class``."""
    return min([d.lineno for d in node.decorator_list] + [node.lineno])


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    decorators = "".join(f"@{ast.unparse(d)}\n" for d in node.decorator_list)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{decorators}{kind} {node.name}({ast.unparse(node.args)}){returns}"


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n", 1)[0][:160]


def python_chunks(path: Path, module: str, text: str | None = None) -> list[dict[str, Any]]:
    """The chunks of an installed Python module ``module`` (its dotted name) at ``path``:
    its docstring, every public top-level function and class, and every public method with
    a docstring. [] when the file does not parse."""
    try:
        source = path.read_text(errors="replace") if text is None else text
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return []
    doc = str(path)
    out: list[dict[str, Any]] = []
    if (mod_doc := ast.get_docstring(tree)) and len(mod_doc) >= 2 * MIN_CHARS:
        out += _parts(f"{module} (module)", mod_doc, f"{path}:1", doc)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(node.name):
            body = ast.get_docstring(node)
            if body is None:
                lines = (node.end_lineno or node.lineno) - node.lineno + 1
                if lines > SHORT_SOURCE_LINES:
                    continue
                body = ast.get_source_segment(source, node) or ""
                text_ = body
            else:
                text_ = f"{_signature(node)}\n\n{body}"
            out += _parts(f"{module}.{node.name}", text_, f"{path}:{_line(node)}", doc)
        elif isinstance(node, ast.ClassDef) and _public(node.name):
            out += _class_chunks(node, module, path, doc)
    return out


def _class_chunks(node: ast.ClassDef, module: str, path: Path, doc: str) -> list[dict[str, Any]]:
    name = f"{module}.{node.name}"
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    lines = [f"class {node.name}({bases})" if bases else f"class {node.name}"]
    if class_doc := ast.get_docstring(node):
        lines += ["", class_doc]
    methods: list[str] = []
    out: list[dict[str, Any]] = []
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (_public(item.name) or item.name in ("__init__", "__call__")):
            continue
        method_doc = ast.get_docstring(item)
        sig = _signature(item).split("\n")[-1].removeprefix("def ").removeprefix("async def ")
        methods.append(f"- {sig}" + (f": {_first_line(method_doc)}" if method_doc else ""))
        if method_doc and len(method_doc) >= 1.5 * MIN_CHARS:
            out += _parts(
                f"{name}.{item.name}",
                f"{_signature(item)}\n\n{method_doc}",
                f"{path}:{_line(item)}",
                doc,
            )
    if methods:
        lines += ["", "Methods:", *methods]
    if class_doc is None and not methods:
        return out
    return _parts(name, "\n".join(lines), f"{path}:{_line(node)}", doc) + out


# ------------------------------------------------------------------ C headers

_DOC_START = re.compile(r"^[ \t]*/\*\*(?!<)", re.M)
_GROUP = re.compile(r"\\defgroup\s+\w+\s+(.+)")
_TITLE_PATTERNS = (  # of the declaration, first match wins
    re.compile(r"^typedef\b.*}\s*(\w+)\s*;\s*$", re.S),
    re.compile(r"\benum\s+(\w+)"),
    re.compile(r"\bstruct\s+(\w+)"),
    re.compile(r"(\w+)\s*\("),
    re.compile(r"typedef[^;]*?(\w+)\s*;"),
    re.compile(r"#\s*define\s+(\w+)"),
    re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:=[^\n]*)?,?\s*$", re.M),  # an enum member
)


def _clean_comment(comment: str) -> str:
    lines = []
    for line in comment.split("\n"):
        line = line.strip()
        line = re.sub(r"^/\*\*+", "", line)
        line = re.sub(r"\*+/$", "", line)
        line = re.sub(r"^\*+ ?", "", line)
        lines.append(re.sub(r"\s{2,}", " ", line).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _declaration(code: str, max_lines: int = 300) -> str:
    """The declaration after a Doxygen comment: lines up to the ``;`` or ``}`` that ends it
    at brace depth 0 (alignment runs of spaces collapsed)."""
    out: list[str] = []
    depth = 0
    for line in code.split("\n"):
        if not out and not line.strip():
            continue
        if not line.strip() and depth <= 0:
            break
        out.append(re.sub(r"(?<=\S)[ \t]{2,}", " ", line.rstrip()))
        depth += line.count("{") - line.count("}")
        if depth <= 0 and (line.rstrip().endswith((";", "}")) or ";" in line):
            break
        if len(out) >= max_lines:
            break
    return "\n".join(out)


def header_chunks(path: Path, text: str | None = None) -> list[dict[str, Any]]:
    """One chunk per Doxygen ``/** ... */`` comment of a C header and the declaration that
    follows it, titled ``<header>: <declared name>``."""
    try:
        source = path.read_text(errors="replace") if text is None else text
    except OSError:
        return []
    starts = [m.start() for m in _DOC_START.finditer(source)]
    out: list[dict[str, Any]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(source)
        segment = source[start:end]
        close = segment.find("*/")
        if close < 0:
            continue
        comment = _clean_comment(segment[: close + 2])
        code = _declaration(segment[close + 2 :])
        if len(comment) < 25 and not code:
            continue
        name = m.group(1).strip() if (m := _GROUP.search(comment)) else ""
        for pattern in _TITLE_PATTERNS if not name else ():
            if m := pattern.search(code):
                name = m.group(1).strip()
                break
        if not name:
            name = _first_line(comment)[:80] or "(comment)"
        body = comment + (f"\n\n```c\n{code}\n```" if code else "")
        line = source.count("\n", 0, start) + 1
        out += _parts(f"{path.name}: {name}", body, f"{path}:{line}", str(path))
    return out


# ------------------------------------------------------------------ Markdown


def markdown_chunks(path: Path, title: str, text: str | None = None) -> list[dict[str, Any]]:
    """One chunk per section of a Markdown file (headings outside code fences)."""
    try:
        source = path.read_text(errors="replace") if text is None else text
    except OSError:
        return []
    sections: list[tuple[list[str], list[str], int]] = [([], [], 1)]
    stack: list[tuple[int, str]] = []
    fence = False
    for n, line in enumerate(source.split("\n"), 1):
        if line.strip().startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^(#{1,6})\s+(.*\S)", line)
        if m:
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip("# ")))
            sections.append(([h for _, h in stack], [], n))
        else:
            sections[-1][1].append(line)
    out: list[dict[str, Any]] = []
    for heads, lines, n in sections:
        body = "\n".join(lines).strip()
        if len(body) < MIN_CHARS:
            continue
        name = " › ".join([title, *heads[-2:]])
        out += _parts(name, body, f"{path}:{n}", str(path))
    return out


# ------------------------------------------------------------------ HTML (Sphinx pages)

_SKIP_TAGS = {
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "svg",
    "noscript",
    "template",
    "select",
    "head",
}
_SKIP_CLASSES = (
    "headerlink",
    "sphx-glr-download",
    "sphx-glr-signature",
    "toctree-wrapper",
    "prev-next",
    "related-pages",
    "bd-sidebar",
    "copybtn",
    "viewcode-link",
    "admonition-title-hide",
)
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "wbr"}
_BLOCK = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "ul",
    "ol",
    "dl",
    "dd",
    "table",
    "thead",
    "tbody",
    "tr",
    "blockquote",
    "figure",
    "figcaption",
    "caption",
    "details",
    "summary",
}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
#: The element that holds a page's content, by theme (first found wins).
_CONTAINERS = (
    ("itemprop", "articlebody"),
    ("role", "main"),
    ("tag", "article"),
    ("tag", "main"),
)


class _Page(HTMLParser):
    """A Sphinx page as sections: ``{"level", "title", "anchor", "text"}``. Only the
    content container (:data:`_CONTAINERS`) is read; navigation, scripts and header links
    are skipped; ``pre`` keeps its layout; a ``dt`` with an id (an API object) starts a
    section of its own."""

    def __init__(self, container: tuple[str, str] | None) -> None:
        super().__init__(convert_charrefs=True)
        self.container = container
        self.inside = container is None
        self.container_tag = ""
        self.container_depth = 0
        self.skip_tag = ""
        self.skip_depth = 0
        self.pre = 0
        self.heading: dict[str, Any] | None = None
        self.last_id = ""
        self.title = ""
        self.in_title = False
        self.in_body = False
        self.meta: dict[str, str] = {}
        self.links: list[str] = []
        self.sections: list[dict[str, Any]] = []
        self.cur: dict[str, Any] = {"level": 0, "title": "", "anchor": "", "parts": []}

    # -- helpers
    def _emit(self, text: str) -> None:
        if self.heading is not None:
            self.heading["parts"].append(text)
        else:
            self.cur["parts"].append(text)

    def _matches(self, tag: str, attrs: dict[str, str]) -> bool:
        if self.container is None:
            return False
        key, value = self.container
        if key == "tag":
            return tag == value
        return attrs.get(key, "").lower() == value

    def _flush(self) -> None:
        text = _tidy("".join(self.cur.pop("parts")))
        self.cur["text"] = text
        self.sections.append(self.cur)
        self.cur = {"level": 0, "title": "", "anchor": "", "parts": []}

    def _start_heading(self, level: int, anchor: str) -> None:
        self._flush()
        self.heading = {"level": level, "anchor": anchor, "parts": []}

    # -- parser callbacks
    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: v or "" for k, v in attrs_list}
        if tag == "body":
            self.in_body = True
        elif tag == "title" and not self.in_body:  # not the <title> of an inline SVG
            self.in_title = True
        elif tag == "meta" and attrs.get("name"):
            self.meta[attrs["name"]] = attrs.get("content", "")
        elif tag == "meta" and attrs.get("property"):
            self.meta[attrs["property"]] = attrs.get("content", "")
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if self.skip_tag:
            if tag == self.skip_tag:
                self.skip_depth += 1
            return
        if not self.inside:
            if self._matches(tag, attrs):
                self.inside, self.container_tag, self.container_depth = True, tag, 1
            return
        if tag == self.container_tag:
            self.container_depth += 1
        classes = attrs.get("class", "")
        if tag in _SKIP_TAGS or any(c in classes for c in _SKIP_CLASSES):
            if tag not in _VOID:
                self.skip_tag, self.skip_depth = tag, 1
            return
        if attrs.get("id") and tag in ("section", "div"):  # not the spans of old ids
            self.last_id = attrs["id"]
        if tag in _HEADINGS and self.heading is None:
            self._start_heading(_HEADINGS[tag], attrs.get("id") or self.last_id)
        elif tag == "dt" and attrs.get("id") and self.heading is None:
            self._start_heading(7, attrs["id"])
        elif tag == "pre":
            self.pre += 1
            self._emit("\n\n```\n")
        elif tag in ("code", "tt", "kbd") and not self.pre:
            self._emit("`")
        elif tag == "li":
            self._emit("\n- ")
        elif tag in ("td", "th"):
            self._emit(" | ")
        elif tag == "br":
            self._emit("\n")
        elif tag in _BLOCK or tag == "dt":
            self._emit("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if self.skip_tag:
            if tag == self.skip_tag:
                self.skip_depth -= 1
                if self.skip_depth == 0:
                    self.skip_tag = ""
            return
        if not self.inside:
            return
        if tag == self.container_tag:
            self.container_depth -= 1
            if self.container_depth == 0:
                self.inside = self.container is None
                return
        if self.heading is not None and (
            (tag in _HEADINGS and self.heading["level"] == _HEADINGS[tag])
            or (tag == "dt" and self.heading["level"] == 7)
        ):
            title = _tidy("".join(self.heading["parts"])).replace("\n", " ")
            title = re.sub(r"\s*[¶#]\s*$", "", title).strip()
            self.cur.update(level=self.heading["level"], title=title[:200])
            self.cur["anchor"] = self.heading["anchor"]
            self.heading = None
        elif tag == "pre" and self.pre:
            self.pre -= 1
            self._emit("\n```\n\n")
        elif tag in ("code", "tt", "kbd") and not self.pre:
            self._emit("`")
        elif tag in _BLOCK or tag == "dt":
            self._emit("\n\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
            return
        if self.skip_tag or not self.inside:
            return
        if self.pre:
            self._emit(data)
        else:
            self._emit(re.sub(r"\s+", " ", data))

    def close(self) -> None:
        super().close()
        if self.heading is not None:  # an unclosed heading: its text is the title
            self.cur.update(level=self.heading["level"], anchor=self.heading["anchor"])
            self.cur["title"] = _tidy("".join(self.heading["parts"]))[:200]
            self.heading = None
        self._flush()


def _tidy(text: str) -> str:
    out: list[str] = []
    fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            fence = not fence
            out.append(line.strip())
            continue
        out.append(line.rstrip() if fence else line.strip())
    text = re.sub(r"(^|\n)- *\n+(?=[^\n`-])", r"\1- ", "\n".join(out))  # <li><p>: one line
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _container(html: str) -> tuple[str, str] | None:
    lower = html.lower()
    for key, value in _CONTAINERS:
        if key == "tag" and f"<{value}" in lower:
            return key, value
        if key != "tag" and re.search(rf'{key}\s*=\s*["\']{value}["\']', lower):
            return key, value
    return None


def html_sections(html: str) -> dict[str, Any]:
    """A page's ``title``, ``meta`` (name → content), ``links`` (hrefs as written) and
    ``sections`` (``level`` 1-6 for headings, 7 for an API object's ``dt``; ``title``,
    ``anchor``, ``text``)."""
    page = _Page(_container(html))
    page.feed(html)
    page.close()
    return {
        "title": " ".join(page.title.split()),
        "meta": page.meta,
        "links": page.links,
        "sections": page.sections,
    }


def section_chunks(page: dict[str, Any], url: str, name: str) -> list[dict[str, Any]]:
    """Chunks of a page from :func:`html_sections`: one per section with text, titled
    ``<name> › <parent heading> › <heading>``, its source the URL with the anchor."""
    out: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    base = url.split("#", 1)[0]
    for section in page["sections"]:
        level, title = section["level"], section["title"]
        if title:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        if len(section["text"]) < MIN_CHARS:
            continue
        heads = [h for _, h in stack][-2:] or [page["title"] or base]
        source = f"{base}#{section['anchor']}" if section["anchor"] else base
        out += _parts(" › ".join([name, *heads]), section["text"], source, base)
    return out
