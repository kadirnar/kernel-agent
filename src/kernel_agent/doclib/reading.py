"""Docs to read first: the doc library's best sections for a target, without a model (#186).

A fresh engineer session starts from its digest (``improve.kernel_digest`` /
``native_digest``). Its ``## Docs to read first`` lists the ids of the library's
(:mod:`~kernel_agent.doclib.store`) best sections for the target, found by BM25 alone:
one short query per thing the target is about (:func:`queries`: its precision, then the
operations its module class, approach and why name, in the order they name them), each
searched in the libraries of the target's backends (:data:`LIBRARIES`), the queries' best
sections taken in turn until :data:`FIRST_READS` distinct ones. Deterministic and cheap
(about 40 ms a digest once the index is loaded, 0.6 s to load it, measured on 16 shelves),
it is part of the session's first message, after the prompt-cache prefix. It reads the
shelves that are built and never builds or fetches one: before the library exists the
digest has no such section.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kernel_agent.doclib import bm25, store

FIRST_READS = 5
#: The doc library's libraries of each backend (``spec.json`` ``backends``).
LIBRARIES = {
    "triton": ("triton",),
    "cuda": ("cuda", "ptx"),
    "nvrtc": ("cuda", "ptx"),
    "cute": ("cutlass",),
    "tilelang": ("tilelang",),
}
#: The libraries of a native engine (CUDA C++ / CuTe projects, ``native/engine.py``).
NATIVE_LIBRARIES = ("cuda", "cutlass", "ptx")
#: What to look a reduced precision up by (``kernels.compare.REDUCED_PRECISIONS``).
PRECISION_QUERIES = {
    "fp8_weights": "fp8 e4m3 conversion scale",
    "fp8_w8a8": "fp8 e4m3 scaled matmul",
    "fp8_mx": "mxfp8 e8m0 block scaled dot_scaled",
    "fp8_kv": "fp8 e4m3 conversion scale",
    "fp4_weights": "fp4 e2m1 block scaled",
    "int8_weights": "int8 dot",
    "int8_w8a8": "int8 dot",
}
#: A word of a target's module class, approach or why (a regular expression, matched case
#: insensitively) → what to look it up by.
TOPICS = (
    (r"attention|attn|sdpa|flash", "attention softmax"),
    (r"gemv", "matrix vector multiply"),
    (r"gemm|matmul|linear|mlp|proj|qkv", "matrix multiplication"),
    (r"rms_?norm|layer_?norm|group_?norm", "layer normalization"),
    (r"rms_?norm|layer_?norm|group_?norm|softmax|reduc", "warp shuffle reduction"),
    (r"conv|im2col", "convolution implicit gemm"),
    (r"softmax", "softmax"),
    (r"persistent|weight streaming", "persistent kernel"),
    (r"cuda.?graph", "cuda graphs"),
    (r"\bpdl\b|programmatic", "programmatic dependent launch"),
    (r"\btma\b|tensor memory accelerator", "tensor memory accelerator"),
    (r"warp.?speciali", "warp specialization"),
)
#: Queries whose words mean something else in the other libraries: searched only in these.
QUERY_LIBRARIES = {"warp shuffle reduction": ("cuda", "ptx", "cutlass")}
#: The queries of tensor-core work: a target with one of them (or a reduced precision) also
#: gets its GPU's ``mma`` query; an RMSNorm does not.
MMA_QUERIES = (
    "attention softmax",
    "matrix vector multiply",
    "matrix multiplication",
    "convolution implicit gemm",
)
#: What a native stage pattern (``native_engine.PATTERNS``) is about.
PATTERN_QUERIES = {
    "stack": "persistent kernel",
    "solver": "persistent kernel",
    "other": "convolution implicit gemm",
}


_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def _words(module_class: str | None) -> str:
    """``MiniCPMDecoderLayer`` → ``Mini CPM Decoder Layer`` (:data:`TOPICS` match words)."""
    return " ".join(_CAMEL.findall(module_class or ""))


def queries(
    *,
    module_class: str | None = None,
    text: str = "",
    precision: str | None = None,
    arch: str | None = None,
    pattern: str | None = None,
) -> list[str]:
    """The queries of a target, most specific first: its precision's, then one per
    :data:`TOPICS` entry its module class and ``text`` (approach, why, idea) name, in the
    order they name them, a native stage's ``pattern``, and for tensor-core work
    (:data:`MMA_QUERIES`) the GPU architecture's instructions (``sm_120`` → ``sm120 mma``)."""
    out: list[str] = []
    if precision in PRECISION_QUERIES:
        out.append(PRECISION_QUERIES[precision])
    words = f"{_words(module_class)} {module_class or ''} {text}"
    found = []
    for i, (regex, query) in enumerate(TOPICS):
        if match := re.search(regex, words, re.I):
            found.append((match.start(), i, query))
    out += [query for _, _, query in sorted(found)]
    if pattern in PATTERN_QUERIES:
        out.append(PATTERN_QUERIES[pattern])
    if arch and (precision in PRECISION_QUERIES or any(q in MMA_QUERIES for q in out)):
        out.append(f"{arch.replace('_', '')} mma")
    return list(dict.fromkeys(out))


def libraries(backends: Iterable[str]) -> list[str]:
    """The doc library's libraries of ``backends`` (in their order, each once)."""
    return list(dict.fromkeys(lib for b in backends for lib in LIBRARIES.get(b, ())))


def _covers(query: str, chunk: dict[str, Any]) -> bool:
    """Whether ``chunk`` contains more than half of the terms of ``query``: not a section
    of a library that only shares a word with it (the ``layer`` of a CUDA array for
    ``layer normalization``)."""
    wanted = set(bm25.tokens(query))
    have = set(bm25.tokens(f"{chunk.get('title', '')} {chunk.get('text', '')}"))
    return 2 * len(wanted & have) > len(wanted)


def first_reads(
    found: list[str],
    libs: Iterable[str] = (),
    *,
    k: int = FIRST_READS,
    base: Path | None = None,
) -> list[dict[str, Any]]:
    """The ``k`` sections to read first for the queries ``found`` (:func:`queries`) in the
    libraries ``libs`` (all when empty): the best section of each query in turn, then the
    second best, ..., each section once, and only sections with more than half of their
    query's terms. ``[]`` without queries or a library built. Each: ``id``, ``title``,
    ``library``, ``version``, ``origin``, ``source`` and its ``query``."""
    libs = list(libs)
    if not found:
        return []
    lib = store.load(base, build_missing=False)
    wanted = [name for name in libs or lib.libraries if name in lib.libraries]

    def matches(query: str) -> list[dict[str, Any]]:
        only = QUERY_LIBRARIES.get(query)
        where = [name for name in wanted if only is None or name in only]
        if not where:  # none of its libraries is built (search() would take every one)
            return []
        return [c for _, c in lib.index.search(query, where, 2 * k) if _covers(query, c)]

    ranked = [matches(query) for query in found]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for rank in range(2 * k):
        for query, hits in zip(found, ranked, strict=True):
            if rank >= len(hits) or len(out) >= k:
                continue
            chunk = hits[rank]
            key = (str(chunk.get("doc")), bm25.section(str(chunk.get("title"))))
            if key in seen:
                continue
            seen.add(key)
            keys = ("id", "title", "library", "version", "origin", "source")
            out.append({name: chunk.get(name) for name in keys} | {"query": query})
    return out


def section(reads: list[dict[str, Any]]) -> list[str]:
    """``## Docs to read first`` of a digest ([] without reads)."""
    if not reads:
        return []
    lines = [
        "",
        "## Docs to read first",
        "The doc library's best sections for this target (its precision, backends and what "
        "it computes; BM25, no model): `doc_read(id)` one before you use its API, or hand "
        "the ids to `doc-lookup` with your question.",
    ]
    for read in reads:
        lines.append(
            f"* `{read['id']}` {read['title']} ({read['library']} {read['version']}, "
            f"{read['origin']}): for `{read['query']}`"
        )
    return lines
