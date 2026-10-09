"""Persistent tuned-config cache: tile / warp / stage choices of Triton kernels, CUTLASS
configs and cuBLASLt algorithms, kept across evaluations and runs (issue #148; FlagGems'
``LibTuner``, docs/RESEARCH-TRITON.md §2.1 and §5.3 item 8).

A kernel that picks its config by timing a few candidates pays that once per GPU, library
versions and shape class instead of once per process (every evaluation, sweep config,
re-check and A/B is a fresh process). Any kernel, any model: the caller names the op and
the shape it tunes for.

* **Key**: ``(gpu, backend, op, bucket)``. ``gpu`` is the device name and its compute
  capability (``NVIDIA GeForce RTX 5070 Ti sm_120``); ``backend`` the library whose choice
  it is (``triton``, ``cutlass``, ``cublaslt``, ...); ``op`` a name the kernel chooses
  (one per kernel and dtype, e.g. ``"my_gemm/e4m3"``); ``bucket`` the shape class
  (:func:`shape_bucket`: every dimension rounded up to a power of two except the ones
  passed as ``exact``, e.g. a weight's N and K; strings such as dtype names as given).
* **Versions**: an entry records the versions it was tuned with (:func:`library_versions`:
  torch, its CUDA, the NVIDIA driver and the backend's library). A lookup under other
  versions drops the entry and tunes again: a Triton, CUTLASS, cuBLAS or driver upgrade
  changes code generation and timings.
* **Store**: SQLite at ``<cache>/tuned-configs.sqlite`` (``KERNEL_AGENT_TUNED_DB``
  overrides the path), shared by concurrent processes; within a process lookups are
  memoised, so a kernel may consult it on every call (no I/O after the first).
* **Configs** are JSON values (dicts of numbers / strings / lists); a tuple comes back as a
  list. Every candidate config must give a correct result: the evaluator checks whichever
  one the cache picks.
* **Points** (issue #229): every config a search (``kernels/search.py``, ``sweep_candidate``
  with a ``space``) measured, with its score, under the same key and versions
  (:meth:`TunedConfigs.add_points`); a later search starts from the best ones at the same or
  a neighbouring bucket (:meth:`TunedConfigs.points`, :func:`signature_bucket` of the
  captured cases, :func:`bucket_distance`). They are starting points, measured again.

API: :func:`best_config` (look up, or time the candidates with the caller's ``bench`` and
store the fastest; never tunes while a CUDA graph is being captured), :func:`lookup`,
:func:`store`, :func:`forget`; :class:`TunedConfigs` for another database or explicit
versions; ``python -m kernel_agent.kernels.tuned [--purge-stale | --clear]`` lists them.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.metadata
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Environment variable with the database path (tests and sandboxes point it elsewhere).
DB_ENV = "KERNEL_AGENT_TUNED_DB"
#: Environment variable that turns tuning off (``0``): lookups still work, a miss returns
#: the default config without timing anything.
TUNE_ENV = "KERNEL_AGENT_TUNE"
#: The installed distributions whose version an entry of each backend records (the first
#: one installed of each group); torch, its CUDA and the driver are always recorded.
LIBRARIES: dict[str, tuple[tuple[str, ...], ...]] = {
    "triton": (("triton",),),
    "cutlass": (("nvidia-cutlass-dsl", "nvidia-cutlass"),),
    "cute": (("nvidia-cutlass-dsl", "nvidia-cutlass"),),
    "cublaslt": (("nvidia-cublas", "nvidia-cublas-cu13", "nvidia-cublas-cu12"),),
    "cublas": (("nvidia-cublas", "nvidia-cublas-cu13", "nvidia-cublas-cu12"),),
    "tilelang": (("tilelang",),),
    "helion": (("helion",), ("triton",)),  # Helion generates Triton
    "cuda": (("nvidia-cuda-nvcc", "nvidia-cuda-nvcc-cu13", "nvidia-cuda-nvcc-cu12"),),
}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS configs (
    gpu TEXT NOT NULL,
    backend TEXT NOT NULL,
    op TEXT NOT NULL,
    bucket TEXT NOT NULL,
    versions TEXT NOT NULL,
    config TEXT NOT NULL,
    ms REAL,
    tried INTEGER,
    source TEXT,
    updated REAL,
    PRIMARY KEY (gpu, backend, op, bucket)
)
"""
#: Every measured point of a search (kernels/search.py, issue #229): its score (a speedup,
#: higher is better; NULL: the config failed), the latest measurement of each config.
_POINTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS points (
    gpu TEXT NOT NULL,
    backend TEXT NOT NULL,
    op TEXT NOT NULL,
    bucket TEXT NOT NULL,
    config TEXT NOT NULL,
    versions TEXT NOT NULL,
    score REAL,
    status TEXT,
    updated REAL,
    PRIMARY KEY (gpu, backend, op, bucket, config)
)
"""


def default_path() -> Path:
    """The database file: ``$KERNEL_AGENT_TUNED_DB`` or ``<cache>/tuned-configs.sqlite``."""
    if path := os.environ.get(DB_ENV):
        return Path(path)
    from kernel_agent.toolchain import CACHE_DIR

    return CACHE_DIR / "tuned-configs.sqlite"


# ------------------------------------------------------------------ keys


def bucket(value: int) -> int:
    """``value`` rounded up to a power of two (0 and 1 stay)."""
    value = int(value)
    return value if value <= 1 else 1 << (value - 1).bit_length()


def shape_bucket(shape: Mapping[str, Any], *, exact: Iterable[str] = ()) -> str:
    """The shape class of ``shape`` (``{"M": 352, "N": 1024, "K": 4096, "dtype": "bf16"}``):
    ``name=value`` pairs in the given order, integers rounded up to a power of two
    (:func:`bucket`) unless their name is in ``exact``, anything else (dtype names, flags)
    as given. ``shape_bucket({"M": 352, "N": 2560}, exact=["N"])`` -> ``"M=512,N=2560"``."""
    keep = set(exact)
    parts = []
    for name, value in shape.items():
        if isinstance(value, bool) or not isinstance(value, int) or name in keep:
            parts.append(f"{name}={value}")
        else:
            parts.append(f"{name}={bucket(value)}")
    return ",".join(parts)


_DIMS = re.compile(r"\[([0-9, ]*)\]")
_NUMBER = re.compile(r"\d+")


def signature_bucket(signatures: Iterable[str]) -> str:
    """The shape class of captured cases (their signatures, ``a0[1, 352, 1024]:bfloat16``):
    every dimension inside brackets rounded up to a power of two (:func:`bucket`), the
    distinct results sorted and joined by ``;``."""

    def one(signature: str) -> str:
        return _DIMS.sub(
            lambda m: "[" + ", ".join(str(bucket(int(d))) for d in _NUMBER.findall(m[1])) + "]",
            str(signature),
        )

    return ";".join(sorted({one(s) for s in signatures}))


def bucket_distance(a: str, b: str) -> float | None:
    """How far apart two buckets are: the largest ``|log2(x / y)|`` over their numbers in
    the same places; None when they differ elsewhere (another op structure, dtype or
    number of cases)."""
    if _NUMBER.sub("#", a) != _NUMBER.sub("#", b):
        return None
    far = 0.0
    for x, y in zip(_NUMBER.findall(a), _NUMBER.findall(b), strict=True):
        x_, y_ = int(x), int(y)
        if x_ != y_:
            if min(x_, y_) == 0:
                return None
            far = max(far, abs(math.log2(x_ / y_)))
    return far


def _dist_version(names: Sequence[str]) -> str | None:
    for name in names:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            return importlib.metadata.version(name)
    return None


def driver_version() -> str | None:
    """The NVIDIA kernel driver's version (``/proc/driver/nvidia/version``), else the CUDA
    version the driver supports, else None."""
    with contextlib.suppress(OSError, IndexError):
        text = Path("/proc/driver/nvidia/version").read_text()
        words = text.split("Kernel Module", 1)[1].split()
        for word in words:
            if word[:1].isdigit() and "." in word:
                return word
    from kernel_agent.toolchain import driver_cuda_version

    cuda = driver_cuda_version()
    return f"cuda {cuda[0]}.{cuda[1]}" if cuda else None


@functools.cache
def library_versions(backend: str) -> dict[str, str | None]:
    """The versions an entry of ``backend`` is valid for: torch, the CUDA torch was built
    with, the driver, and the backend's library (:data:`LIBRARIES`; an unknown backend:
    torch, CUDA and the driver only)."""
    import torch

    versions: dict[str, str | None] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": driver_version(),
    }
    for part in backend.split("+"):  # a hybrid (``cuda+triton``): each backend's library
        for group in LIBRARIES.get(part, ()):
            versions[group[0]] = _dist_version(group)
    return versions


@functools.cache
def gpu_name(device: int | None = None) -> str:
    """``"<device name> sm_<major><minor>"`` of ``device`` (default: the current one);
    ``"cpu"`` without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    index = torch.cuda.current_device() if device is None else device
    props = torch.cuda.get_device_properties(index)
    return f"{props.name} sm_{props.major}{props.minor}"


def _emulated() -> bool:
    """Whether this process emulates an older GPU (``KERNEL_AGENT_EMULATE_ARCH``): nothing it
    times is stored."""
    from kernel_agent.emulate import active

    return active()


def _normal(config: Any) -> Any:
    """``config`` as it comes back from the database (JSON: tuples become lists)."""
    return json.loads(json.dumps(config))


# ------------------------------------------------------------------ the store


class TunedConfigs:
    """One tuned-config database (module docstring). ``gpu`` / ``versions`` override what
    this process would detect (tests, a cache prepared for another machine): ``versions``
    is then the version dict of every backend."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        gpu: str | None = None,
        versions: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._gpu = gpu
        self._versions = dict(versions) if versions is not None else None
        self._memo: dict[tuple[str, str, str, str], Any] = {}
        self._untuned: dict[tuple[str, str, str, str], Any] = {}  # tuning turned off
        #: Entries dropped by this instance because their versions differ (op, bucket, old).
        self.invalidated: list[dict[str, Any]] = []

    # -- environment

    @property
    def gpu(self) -> str:
        return self._gpu if self._gpu is not None else gpu_name()

    def versions(self, backend: str) -> dict[str, Any]:
        """The versions entries of ``backend`` must match."""
        if self._versions is not None:
            return dict(self._versions)
        return dict(library_versions(backend))

    # -- database

    @contextlib.contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30.0)
        try:
            with contextlib.suppress(sqlite3.DatabaseError):
                conn.execute("PRAGMA journal_mode=WAL")  # readers never wait for a writer
            conn.execute(_SCHEMA)
            conn.execute(_POINTS_SCHEMA)
            with conn:  # one transaction: committed on success
                yield conn
        finally:
            conn.close()

    def _key(self, backend: str, op: str, bucket: str) -> tuple[str, str, str, str]:
        return (self.gpu, backend, op, bucket)

    def get(
        self,
        op: str,
        shape: Mapping[str, Any] | str,
        *,
        backend: str = "triton",
        exact: Iterable[str] = (),
    ) -> Any | None:
        """The stored config of ``op`` for ``shape``'s bucket (a :func:`shape_bucket` string
        or the shape itself), or None: not tuned yet, or tuned under other versions (that
        entry is deleted and recorded in :attr:`invalidated`)."""
        key = self._key(backend, op, _bucket_of(shape, exact))
        if key in self._memo:
            return self._memo[key]
        current = self.versions(backend)
        with self._db() as db:
            row = db.execute(
                "SELECT versions, config FROM configs WHERE gpu=? AND backend=? AND op=? "
                "AND bucket=?",
                key,
            ).fetchone()
            if row is None:
                return None
            stored = json.loads(row[0])
            if stored != current:
                db.execute(
                    "DELETE FROM configs WHERE gpu=? AND backend=? AND op=? AND bucket=?", key
                )
                self.invalidated.append(
                    {"op": op, "bucket": key[3], "backend": backend, "versions": stored}
                )
                return None
        config = json.loads(row[1])
        self._memo[key] = config
        return config

    def put(
        self,
        op: str,
        shape: Mapping[str, Any] | str,
        config: Any,
        *,
        backend: str = "triton",
        exact: Iterable[str] = (),
        ms: float | None = None,
        tried: int | None = None,
        source: str | None = None,
    ) -> Any:
        """Store ``config`` for ``op`` at ``shape``'s bucket under the current versions
        (replacing what was there); returns it as :meth:`get` will (JSON round trip). Under
        emulation (``emulate.py``) only for this process: its timings are not that GPU's."""
        key = self._key(backend, op, _bucket_of(shape, exact))
        text = json.dumps(config, sort_keys=True)
        versions = json.dumps(self.versions(backend), sort_keys=True)
        if _emulated():
            self._memo[key] = json.loads(text)
            return self._memo[key]
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO configs (gpu, backend, op, bucket, versions, config, "
                "ms, tried, source, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*key, versions, text, ms, tried, source, time.time()),
            )
        self._memo[key] = json.loads(text)
        return self._memo[key]

    def forget(
        self,
        op: str | None = None,
        shape: Mapping[str, Any] | str | None = None,
        *,
        backend: str | None = None,
        exact: Iterable[str] = (),
    ) -> int:
        """Delete this GPU's entries (of ``op``, of ``backend``, at ``shape``'s bucket: each
        when given); returns how many."""
        where, args = ["gpu=?"], [self.gpu]
        if backend is not None:
            where.append("backend=?")
            args.append(backend)
        if op is not None:
            where.append("op=?")
            args.append(op)
        if shape is not None:
            where.append("bucket=?")
            args.append(_bucket_of(shape, exact))
        with self._db() as db:
            n = db.execute(f"DELETE FROM configs WHERE {' AND '.join(where)}", args).rowcount
        self._memo.clear()
        return int(n)

    def entries(self, *, all_gpus: bool = False) -> list[dict[str, Any]]:
        """Stored entries (this GPU's unless ``all_gpus``), each with ``stale`` (tuned under
        other versions than this process has)."""
        query = "SELECT gpu, backend, op, bucket, versions, config, ms, tried, source, updated "
        query += "FROM configs" + ("" if all_gpus else " WHERE gpu=?")
        with self._db() as db:
            rows = db.execute(query, () if all_gpus else (self.gpu,)).fetchall()
        out = []
        for gpu, backend, op, bucket_, versions, config, ms, tried, source, updated in rows:
            stored = json.loads(versions)
            out.append(
                {
                    "gpu": gpu,
                    "backend": backend,
                    "op": op,
                    "bucket": bucket_,
                    "config": json.loads(config),
                    "ms": ms,
                    "tried": tried,
                    "source": source,
                    "updated": updated,
                    "versions": stored,
                    "stale": gpu == self.gpu and stored != self.versions(backend),
                }
            )
        return out

    # -- a search's points (kernels/search.py)

    def add_points(
        self,
        op: str,
        shape: Mapping[str, Any] | str,
        points: Iterable[Mapping[str, Any]],
        *,
        backend: str = "triton",
        exact: Iterable[str] = (),
    ) -> int:
        """Store measured points of ``op`` at ``shape``'s bucket under the current versions:
        each ``{"config", "score" (higher is better; None: it failed), "status"}``, replacing
        an earlier measurement of the same config. Returns how many (none under emulation,
        ``emulate.py``: its timings are not that GPU's)."""
        if _emulated():
            return 0
        key = self._key(backend, op, _bucket_of(shape, exact))
        versions = json.dumps(self.versions(backend), sort_keys=True)
        now = time.time()
        rows = []
        for p in points:
            score = p.get("score")
            score = float(score) if score is not None and math.isfinite(float(score)) else None
            config = json.dumps(p["config"], sort_keys=True)
            rows.append((*key, config, versions, score, p.get("status"), now))
        if not rows:
            return 0
        with self._db() as db:
            db.executemany(
                "INSERT OR REPLACE INTO points (gpu, backend, op, bucket, config, versions, "
                "score, status, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def points(
        self,
        op: str,
        shape: Mapping[str, Any] | str,
        *,
        backend: str = "triton",
        exact: Iterable[str] = (),
        near: float = 1.0,
        limit: int | None = 32,
    ) -> list[dict[str, Any]]:
        """The points of ``op`` measured on this GPU under the current versions at ``shape``'s
        bucket or a neighbouring one (:func:`bucket_distance` at most ``near``: a factor 2 per
        dimension by default), the nearest bucket first, then the best score; failed points
        (score None) after the measured ones. Each ``{"config", "score", "status", "bucket",
        "distance"}``."""
        want = _bucket_of(shape, exact)
        current = self.versions(backend)
        with self._db() as db:
            rows = db.execute(
                "SELECT bucket, config, versions, score, status FROM points WHERE gpu=? AND "
                "backend=? AND op=?",
                (self.gpu, backend, op),
            ).fetchall()
        out = []
        for bucket_, config, versions, score, status in rows:
            far = bucket_distance(want, bucket_)
            if far is None or far > near or json.loads(versions) != current:
                continue
            out.append(
                {
                    "config": json.loads(config),
                    "score": score,
                    "status": status,
                    "bucket": bucket_,
                    "distance": far,
                }
            )
        out.sort(key=lambda p: (p["score"] is None, p["distance"], -(p["score"] or 0.0)))
        return out if limit is None else out[:limit]

    def purge_stale(self) -> int:
        """Delete this GPU's entries tuned under other versions; returns how many."""
        stale = [e for e in self.entries() if e["stale"]]
        with self._db() as db:
            for e in stale:
                db.execute(
                    "DELETE FROM configs WHERE gpu=? AND backend=? AND op=? AND bucket=?",
                    (e["gpu"], e["backend"], e["op"], e["bucket"]),
                )
        self._memo.clear()
        return len(stale)

    # -- tuning

    def best_config(
        self,
        op: str,
        shape: Mapping[str, Any],
        candidates: Sequence[Any],
        bench: Callable[[Any], float],
        *,
        backend: str = "triton",
        exact: Iterable[str] = (),
        default: Any = None,
        source: str | None = None,
    ) -> Any:
        """The config for ``op`` at ``shape``: the stored one, else the fastest of
        ``candidates`` by ``bench(config) -> ms`` (a candidate that raises, e.g. out of
        shared memory, is skipped), stored for every later process. Without tuning (a CUDA
        graph is being captured, ``KERNEL_AGENT_TUNE=0``, or every candidate failed) it
        returns ``default`` (else the first candidate) and stores nothing."""
        exact = tuple(exact)
        key = self._key(backend, op, _bucket_of(shape, exact))
        if key in self._untuned:
            return self._untuned[key]
        found = self.get(op, shape, backend=backend, exact=exact)
        if found is not None:
            return found
        fallback = _normal(default if default is not None or not candidates else candidates[0])
        why = _no_tuning()
        if why == "disabled":  # for the whole process: no database query per call
            self._untuned[key] = fallback
        if not candidates or why:
            return fallback
        timings: list[tuple[float, int]] = []
        for i, config in enumerate(candidates):
            try:
                ms = float(bench(config))
            except Exception:
                continue
            if math.isfinite(ms):
                timings.append((ms, i))
        if not timings:
            return fallback
        ms, i = min(timings)
        return self.put(
            op,
            shape,
            candidates[i],
            backend=backend,
            exact=exact,
            ms=round(ms, 6),
            tried=len(candidates),
            source=source,
        )


def _bucket_of(shape: Mapping[str, Any] | str, exact: Iterable[str]) -> str:
    return shape if isinstance(shape, str) else shape_bucket(shape, exact=exact)


def _no_tuning() -> str | None:
    """Why nothing may be timed now: ``disabled`` (:data:`TUNE_ENV`), ``capturing`` (a CUDA
    graph capture on the current stream: timing would break it); None: tuning is allowed."""
    if os.environ.get(TUNE_ENV, "1").strip().lower() in ("0", "false", "no", "off"):
        return "disabled"
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return "capturing"
    return None


def time_ms(fn: Callable[[], Any], *, warmup: int = 3, reps: int = 20) -> float:
    """Median CUDA-event time of ``fn()`` (ms) on the current stream, for ``bench``
    functions: ``tuned.best_config(..., bench=lambda c: tuned.time_ms(lambda: run(c)))``."""
    import torch

    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


# ------------------------------------------------------------------ the default store

_DEFAULT: dict[Path, TunedConfigs] = {}


def default() -> TunedConfigs:
    """This process's store at :func:`default_path` (one instance per path)."""
    path = default_path()
    if path not in _DEFAULT:
        _DEFAULT[path] = TunedConfigs(path)
    return _DEFAULT[path]


def lookup(op: str, shape: Mapping[str, Any] | str, **kwargs: Any) -> Any | None:
    """:meth:`TunedConfigs.get` on the default store."""
    return default().get(op, shape, **kwargs)


def store(op: str, shape: Mapping[str, Any] | str, config: Any, **kwargs: Any) -> Any:
    """:meth:`TunedConfigs.put` on the default store."""
    return default().put(op, shape, config, **kwargs)


def forget(op: str | None = None, shape: Mapping[str, Any] | str | None = None, **kw: Any) -> int:
    """:meth:`TunedConfigs.forget` on the default store."""
    return default().forget(op, shape, **kw)


def best_config(
    op: str,
    shape: Mapping[str, Any],
    candidates: Sequence[Any],
    bench: Callable[[Any], float],
    **kwargs: Any,
) -> Any:
    """:meth:`TunedConfigs.best_config` on the default store."""
    return default().best_config(op, shape, candidates, bench, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List or prune the tuned-config cache.")
    parser.add_argument("--db", type=Path, help=f"database (default: ${DB_ENV} or the cache)")
    parser.add_argument("--all-gpus", action="store_true", help="list every GPU's entries")
    parser.add_argument("--purge-stale", action="store_true", help="drop outdated entries")
    parser.add_argument("--clear", action="store_true", help="drop this GPU's entries")
    ns = parser.parse_args(argv)
    configs = TunedConfigs(ns.db)
    if ns.clear:
        print(f"removed {configs.forget()} entries")
    elif ns.purge_stale:
        print(f"removed {configs.purge_stale()} stale entries")
    rows = configs.entries(all_gpus=ns.all_gpus)
    print(f"{configs.path}: {len(rows)} entries")
    for e in rows:
        ms = f" {e['ms']:.4g} ms" if e["ms"] is not None else ""
        stale = " [stale]" if e["stale"] else ""
        print(f"  {e['gpu']} {e['backend']} {e['op']} {e['bucket']}: {e['config']}{ms}{stale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
