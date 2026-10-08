"""Okapi BM25 over the doc library's chunks, pure Python (issue #177).

:func:`tokens` splits code-aware: an identifier counts whole and in its parts
(``dot_scaled`` → ``dot_scaled``, ``dot``, ``scaled``; ``cudaLaunchKernelEx`` →
``cudalaunchkernelex``, ``cuda``, ``launch``, ``kernel``, ``ex``), and a dotted or ``::``
name also by its prefixes (``cp.async.bulk.tensor`` → ``cp.async``, ``cp.async.bulk``,
``cp.async.bulk.tensor``), so a query names an API the way the code does.

A shelf's postings (:func:`postings`) are computed when the shelf is built and stored next
to its chunks; :class:`Index` combines the shelves that are loaded and scores a query
without any network or service (a chunk's title counts :data:`TITLE_WEIGHT` times).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

K1, B = 1.2, 0.75
TITLE_WEIGHT = 3
NAME_BOOST = 2.0
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::)[A-Za-z0-9_]+)*|\d+(?:\.\d+)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP_WORDS = (
    "a an and are as at be by can for from has have if in into is it its may must not of "
    "on or that the their then there these this to was when which will with you your"
)
_STOP = frozenset(_STOP_WORDS.split())
MAX_PREFIX = 4  # dotted prefixes indexed: cp.async, cp.async.bulk, cp.async.bulk.tensor


def tokens(text: str) -> Iterator[str]:
    """The index terms of ``text`` (lower case; stop words and 1-character words left out)."""
    for match in _WORD.finditer(text):
        word = match.group()
        if len(word) > 80:
            continue
        parts = re.split(r"\.|::", word)
        if len(parts) > 1 and not word[0].isdigit():
            yield ".".join(parts).lower()
            for n in range(2, min(len(parts), MAX_PREFIX + 1)):
                yield ".".join(parts[:n]).lower()
        elif word[0].isdigit():
            yield word
            continue
        for part in parts:
            low = part.lower()
            if len(low) > 1 and low not in _STOP:
                yield low
            pieces = [p for p in part.split("_") if p]
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece.lower() not in _STOP:
                        yield piece.lower()
            for piece in pieces:
                camel = _CAMEL.findall(piece)
                if len(camel) > 1:
                    yield from (c.lower() for c in camel if len(c) > 1)


def terms(chunk: dict[str, Any]) -> Counter[str]:
    """Term frequencies of a chunk: its text plus its title :data:`TITLE_WEIGHT` times."""
    counts = Counter(tokens(chunk.get("text", "")))
    for term in tokens(chunk.get("title", "")):
        counts[term] += TITLE_WEIGHT
    return counts


def postings(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """A shelf's index: ``lengths`` (terms per chunk) and ``postings`` (term → flat list
    ``[chunk, tf, chunk, tf, ...]``), JSON-ready."""
    lengths: list[int] = []
    table: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        counts = terms(chunk)
        lengths.append(sum(counts.values()))
        for term, tf in counts.items():
            table.setdefault(term, []).extend((i, tf))
    return {"lengths": lengths, "postings": table}


def section(title: str) -> str:
    """A chunk's title without its part number (``... (2/5)``)."""
    return re.sub(r"\s*\(\d+/\d+\)$", "", title)


def api_name(title: str) -> str:
    """The name a chunk documents, lower case: the last component of its title before any
    argument list (``triton.language.core.dot_scaled (1/2)`` → ``dot_scaled``,
    ``cublasLt.h: cublasLtMatmul`` → ``cublasltmatmul``, ``PyTorch › ... ›
    torch.utils.cpp_extension.load_inline(name, ...)`` → ``load_inline``)."""
    head = section(title).split(" › ")[-1].split("(", 1)[0]
    parts = [p.strip("`") for p in re.split(r"[.:\s]+", head) if p.strip("`")]
    return parts[-1].lower() if parts else ""


class Index:
    """BM25 over several shelves: ``add`` a shelf's chunks with its :func:`postings`, then
    :meth:`search` (every shelf, or only those of some libraries). A chunk that documents
    a name the query spells out (``dot_scaled``, ``T.gemm``: :func:`api_name`) gains
    :data:`NAME_BOOST` times that term's full weight, so the API's own entry comes before
    the pages that mention it; of the parts of one section only the best is returned
    (``doc_read`` reads on through the others)."""

    def __init__(self) -> None:
        self.shelves: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        self.names: list[list[str]] = []
        self.n = 0
        self.total_length = 0

    def add(self, library: str, chunks: list[dict[str, Any]], index: dict[str, Any]) -> None:
        self.shelves.append((library, chunks, index))
        self.names.append([api_name(c.get("title", "")) for c in chunks])
        self.n += len(chunks)
        self.total_length += sum(index["lengths"])

    def search(
        self, query: str, libraries: Iterable[str] | None = None, k: int = 8
    ) -> list[tuple[float, dict[str, Any]]]:
        """The ``k`` best chunks for ``query`` as ``(score, chunk)``, best first."""
        wanted = set(dict.fromkeys(tokens(query)))
        if not wanted or not self.n:
            return []
        allow = {lib.lower() for lib in libraries} if libraries else None
        avg = self.total_length / self.n
        scores: dict[tuple[int, int], float] = {}
        idfs: dict[str, float] = {}
        for term in wanted:
            df = sum(len(ix["postings"].get(term, ())) // 2 for _, _, ix in self.shelves)
            if not df:
                continue
            idf = idfs[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for s, (library, _, ix) in enumerate(self.shelves):
                if allow is not None and library not in allow:
                    continue
                post = ix["postings"].get(term)
                if not post:
                    continue
                lengths = ix["lengths"]
                for j in range(0, len(post), 2):
                    i, tf = post[j], post[j + 1]
                    norm = tf + K1 * (1 - B + B * lengths[i] / avg)
                    key = (s, i)
                    scores[key] = scores.get(key, 0.0) + idf * tf * (K1 + 1) / norm
        # names the query spells: identifiers, and the last part of `tl.x` / `T.x`
        named = {w: w for w in wanted if "." not in w}
        named |= {w.split(".")[1]: w.split(".")[1] for w in wanted if w.count(".") == 1}
        for key in scores:
            name = self.names[key[0]][key[1]]
            if name in named:
                scores[key] += NAME_BOOST * idfs.get(name, 0.0) * (K1 + 1)
        out: list[tuple[float, dict[str, Any]]] = []
        seen: set[tuple[int, str, str]] = set()
        for (s, i), score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            chunk = self.shelves[s][1][i]
            part_of = (s, str(chunk.get("doc")), section(str(chunk.get("title"))))
            if part_of in seen:
                continue
            seen.add(part_of)
            out.append((score, chunk))
            if len(out) >= max(1, k):
                break
        return out


def snippet(text: str, query: str, width: int = 240) -> str:
    """About ``width`` characters of ``text`` around the first place a query term occurs
    (the longest term first), whitespace collapsed."""
    flat = " ".join(text.split())
    lower = flat.lower()
    at = -1
    for term in sorted(set(tokens(query)), key=len, reverse=True):
        at = lower.find(term)
        if at >= 0:
            break
    start = max(0, at - width // 3) if at >= 0 else 0
    piece = flat[start : start + width]
    return ("…" if start else "") + piece + ("…" if start + width < len(flat) else "")
