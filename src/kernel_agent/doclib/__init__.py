"""A local, versioned documentation library the agents search (issue #177).

Agents must read the real documentation of the tools they program, at the versions
installed here, not only curated notes and what a WebFetch of a 4 MB page shows. The
library holds:

* the installed packages' API documentation: Triton's ``triton.language`` (and Gluon),
  the CuTe DSL of ``nvidia-cutlass-dsl``, TileLang, ``torch.utils.cpp_extension`` and
  ``torch.cuda`` (signatures + docstrings), and the Doxygen comments of the CUDA headers
  (runtime / driver API, launch attributes, FP8 / FP4 conversions) and ``cublasLt.h``;
* the official web docs, fetched once over the WebFetch allowlist: the CUDA Programming
  Guide, Best Practices and Blackwell tuning guides, the whole PTX ISA, the cuBLAS
  reference with cuBLASLt, the CUTLASS / CuTe DSL docs, Triton's docs and tutorials,
  TileLang's docs and the PyTorch pages of the installed version.

Each source is chunked per API object or section (:mod:`~kernel_agent.doclib.chunks`) and
indexed with BM25 (:mod:`~kernel_agent.doclib.bm25`); :mod:`~kernel_agent.doclib.store`
keeps the shelves under ``~/.cache/kernel-agent/docs`` keyed by version, rebuilds one
when its version changes, and serves the agents' ``doc_search`` / ``doc_read`` tools
(``agent/tools.py``) without network. Every lookup is recorded with the WebFetch ones in
``research/sources.jsonl`` (``agent/web.py``).
"""

from kernel_agent.doclib.store import (
    add_parser,
    build,
    describe,
    ensure,
    fetch_web,
    load,
    main,
    prepare_in_background,
    read,
    root,
    search,
)

__all__ = [
    "add_parser",
    "build",
    "describe",
    "ensure",
    "fetch_web",
    "load",
    "main",
    "prepare_in_background",
    "read",
    "root",
    "search",
]
