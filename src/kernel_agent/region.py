"""Region targets: fusions across module boundaries.

A target is one ``nn.Module`` class whose instances are replaced, so a fusion
whose ops live in a parent's ``forward`` between child modules (the residual add
after attention + the next RMSNorm of a decoder layer, an output projection +
residual + norm) cannot be one kernel. The planner marks such a fusion
``kind: "region"`` with a ``parent_class`` and a ``region`` (a description of
the ops to isolate). Before the normal kernel pipeline it goes through:

1. ``worker capture --parent``: the parent class (every phase) is captured to
   ``.truth/captures/<id>.parent.pt``; ``targets/<id>/parent/`` gets the
   inputs-only copy, the source and the workload profile;
2. a ``refactor-<id>`` agent (read-only tools + ``verify_rewrite``, may write only
   ``targets/<id>/rewrite.py``) writes ``rewrite(parent) -> nn.Module``: a drop-in
   parent whose entrypoints call a new submodule of class ``Region_<id>`` for the
   region, with the same math;
3. the orchestrator seals a copy of ``rewrite.py`` (:func:`verified_rewrite`) and
   checks it on every captured call of the parent (:func:`verify`): outputs and
   in-place side effects bitwise identical, or within about one unit in the last
   place (a refactor, not an optimisation), and ``Region_<id>`` must be called;
4. ``worker capture``: the workload runs with the rewrite applied to every parent
   instance, and ``Region_<id>`` is captured as the target's ``module_class``.

From then on it is a regular target. Wherever its kernel is applied (``e2e``,
re-profiles, ``optimized/apply.py``) the rewrite goes first, on every instance of
the parent class (``qualname_regex`` selects parent instances), and the kernels
then replace the ``Region_<id>`` modules it added.

The ``nn.Module`` classes a rewrite file defines pickle by value
(:func:`load_rewrite`): a capture of ``Region_<id>`` loads in any process (the
evaluator, the agent's own scripts) without the rewrite file on the import path.

    python -m kernel_agent.region CAPTURE REWRITE --target ID [--capture-sha256 S]
"""

from __future__ import annotations

import argparse
import copy
import json
import linecache
import re
import shutil
import subprocess
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kernel_agent import ledger, truth
from kernel_agent.gpulock import child_env, gpu_lock
from kernel_agent.kernels.compare import flatten
from kernel_agent.truth import TamperError, Truth
from kernel_agent.workspace import RunDir, read_json

KIND = "region"
PARENT_DIR = "parent"  # targets/<id>/parent/: the parent capture's files for the refactor agent
REWRITE_FILE = "rewrite.py"
MARKER = "@@KA_REWRITE@@"


def is_region(spec: dict[str, Any]) -> bool:
    return spec.get("kind") == KIND


def region_class(target_id: str) -> str:
    """Class name of a target's region module (unique per target: patches match by name)."""
    return f"Region_{target_id}"


def module_name(target_id: str) -> str:
    """Import name of a target's rewrite file (``from ka_region_<id> import Region_<id>``)."""
    return f"ka_region_{target_id}"


def parent_capture(run: RunDir, target_id: str) -> Path:
    """The capture of a region target's parent class (next to the target's capture)."""
    capture = run.capture_file(target_id)
    return capture.with_name(f"{capture.stem}.parent.pt")


def verified_rewrite(run: RunDir, target_id: str) -> Path:
    """The sealed copy of the verified ``rewrite.py`` that integration applies."""
    return run.history_dir(target_id) / REWRITE_FILE


def validate(target: dict[str, Any], known: set[str]) -> str | None:
    """Check a planned target against the profiled classes ``known``; returns why it is
    unusable, or None. A region target gets ``module_class`` = ``Region_<id>``; a module
    target (the default ``kind``) loses the region fields."""
    kind = target.pop("kind", None) or "module"
    if kind == "module":
        target.pop("parent_class", None)
        target.pop("region", None)
        if target.get("module_class") not in known:
            return f"class {target.get('module_class')} not in profile"
        return None
    if kind != KIND:
        return f"unknown kind {kind!r}"
    parent = target.get("parent_class") or target.get("module_class")
    if parent not in known:
        return f"parent class {parent} not in profile"
    if not str(target.get("region") or "").strip():
        return "a region target needs a `region` description"
    target.update(kind=KIND, parent_class=parent, module_class=region_class(target["id"]))
    return None


@dataclass
class Rewrite:
    """The verified rewrite of a region target, applied before any kernel."""

    target_id: str
    parent_class: str
    path: Path
    qualname_regex: str | None = None


def rewrite_of(run: RunDir, target_id: str, spec: dict[str, Any]) -> Rewrite | None:
    """The :class:`Rewrite` of a region target's ``spec`` (None for a module target)."""
    if not is_region(spec):
        return None
    return Rewrite(
        target_id,
        spec["parent_class"],
        verified_rewrite(run, target_id),
        spec.get("qualname_regex"),
    )


# ------------------------------------------------------------------ loading + applying


def load_rewrite(path: Path, target_id: str) -> types.ModuleType:
    """Import a rewrite file as ``ka_region_<id>``; the module classes it defines pickle
    (and deep-copy) by value."""
    path = Path(path).resolve()
    module = _exec(path.read_text(), module_name(target_id), str(path))
    if not callable(getattr(module, "rewrite", None)):
        raise AttributeError(f"{path.name} must define rewrite(parent) -> nn.Module")
    return module


def _exec(source: str, name: str, filename: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__dict__["__ka_source__"] = source
    # tracebacks and inspect.getsource() work even where the file is not on disk
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    sys.modules[name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    for value in list(vars(module).values()):
        if isinstance(value, type) and issubclass(value, nn.Module) and value.__module__ == name:
            value.__reduce_ex__ = _reducer(value, source, name, filename)  # type: ignore[method-assign]
    return module


def _reducer(cls: type, source: str, name: str, filename: str) -> Any:
    def __reduce_ex__(self: Any, protocol: Any) -> Any:
        if type(self) is not cls:  # a subclass defined elsewhere (a kernel candidate)
            return object.__reduce_ex__(self, protocol)
        return (_rebuild, (source, name, filename, cls.__qualname__), self.__getstate__())

    return __reduce_ex__


def _rebuild(source: str, name: str, filename: str, qualname: str) -> Any:
    """Unpickle a rewrite-file class: import its source (once per process) and return an
    empty instance (pickle then restores its state)."""
    module = sys.modules.get(name)
    if module is None or module.__dict__.get("__ka_source__") != source:
        module = _exec(source, name, filename)
    cls: Any = module
    for part in qualname.split("."):
        cls = getattr(cls, part)
    return cls.__new__(cls)


def apply_rewrites(roots: dict[str, nn.Module], rewrites: list[Rewrite]) -> dict[str, int]:
    """Replace every instance of each rewrite's parent class (whose qualname matches its
    ``qualname_regex``) with ``rewrite(instance)``; returns the instances per target."""
    from kernel_agent.integrate.patcher import _set_child

    counts: dict[str, int] = {}
    for rw in rewrites:
        module = load_rewrite(rw.path, rw.target_id)
        pattern = re.compile(rw.qualname_regex) if rw.qualname_regex else None
        n = 0
        for root_name, root in roots.items():
            for name, parent in list(root.named_modules()):
                if not name or type(parent).__name__ != rw.parent_class:
                    continue
                if pattern is not None and not pattern.search(f"{root_name}.{name}"):
                    continue
                new = module.rewrite(parent)
                if new is not parent:
                    _set_child(root, name, new)
                n += 1
        counts[rw.target_id] = n
    return counts


# ------------------------------------------------------------------ verification


def _short_tb(limit: int = 4000) -> str:
    text = traceback.format_exc()
    return text if len(text) <= limit else "...\n" + text[-limit:]


def strict_compare(ref: Any, new: Any, prefix: str) -> list[dict[str, Any]]:
    """Per tensor of ``ref``: equal bit for bit (``bitwise``), else floating point within
    ``eps * (|ref| + max|ref|)`` everywhere (about one unit in the last place, no
    mismatch allowance); other tensors must be equal."""
    new_flat = flatten(new, prefix)
    checks: list[dict[str, Any]] = []
    for name, a in flatten(ref, prefix).items():
        b = new_flat.get(name)
        check: dict[str, Any] = {"name": name, "ok": False, "bitwise": False}
        checks.append(check)
        if not isinstance(b, torch.Tensor):
            check["error"] = "missing" if b is None else f"expected tensor, got {type(b).__name__}"
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            check["error"] = f"{tuple(b.shape)} {b.dtype} != reference {tuple(a.shape)} {a.dtype}"
            continue
        a = a.detach().to(b.device)
        same = a == b
        if a.is_floating_point():
            same |= a.isnan() & b.isnan()
        if bool(same.all()):
            check.update(ok=True, bitwise=True, max_abs_err=0.0)
            continue
        if not a.is_floating_point():
            check["mismatch_frac"] = round(float((~same).float().mean()), 6)
            continue
        x, y = a.double(), b.detach().double()
        diff = torch.where(same, 0.0, (x - y).abs())
        finite = x.abs()[x.isfinite()]
        scale = float(finite.max()) if finite.numel() else 0.0
        tol = torch.finfo(a.dtype).eps * (x.abs() + scale)
        within = same | (diff <= tol)
        check.update(
            ok=bool(within.all()),
            max_abs_err=float(diff.nan_to_num(float("inf")).max()),
            outside=int((~within).sum()),
        )
    return checks


def verify(
    capture_path: Path,
    rewrite_path: Path,
    target_id: str,
    *,
    device: str | None = None,
    capture_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply ``rewrite()`` to a copy of the captured parent and replay every captured
    call: outputs and side effects must match the reference (:func:`strict_compare`),
    and the parent must call its ``Region_<id>`` module (through ``__call__``)."""
    from kernel_agent.profiling.capture import load_capture
    from kernel_agent.profiling.state import Replay
    from kernel_agent.workloads.base import synchronize

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        from kernel_agent import toolchain

        toolchain.setup()
    cls = region_class(target_id)
    result: dict[str, Any] = {"rewrite": str(rewrite_path), "status": "error", "verified": False}
    try:
        capture = load_capture(capture_path, device=device, sha256=capture_sha256)
    except TamperError as exc:
        result.update(status="tampered", error=str(exc))
        return result
    if capture.get("inputs_only"):
        result["error"] = f"{capture_path} is an inputs-only capture (no reference outputs)"
        return result
    reference = capture["module"].eval()
    cases = capture["cases"]
    replay = Replay(capture, reference)  # each case's module state (profiling/state.py)
    try:
        new = load_rewrite(rewrite_path, target_id).rewrite(copy.deepcopy(reference))
        if not isinstance(new, nn.Module):
            raise TypeError(f"rewrite() returned {type(new).__name__}, not an nn.Module")
        new.eval()
    except Exception:
        result.update(status="build_error", error=_short_tb())
        return result
    regions = [m for m in new.modules() if type(m).__name__ == cls]
    methods = {c.get("method", "forward") for c in cases}
    missing = [m for m in methods if m != "forward" and not callable(getattr(new, m, None))]
    if not regions or missing:
        result.update(
            status="build_error",
            error=f"rewrite() returned a {type(new).__name__} without "
            + (
                f"{', '.join(f'`{m}()`' for m in sorted(missing))}, which the model calls"
                if missing
                else f"a `{cls}` submodule: define `class {cls}(nn.Module)` in "
                f"{REWRITE_FILE} and attach an instance (e.g. `new.region = {cls}(...)`)"
            ),
        )
        return result
    calls = [0]

    def count(*_: Any) -> None:
        calls[0] += 1

    handles = [m.register_forward_pre_hook(count) for m in regions]
    reports = []
    try:
        for i, case in enumerate(cases):
            args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
            before = calls[0]
            try:
                with torch.inference_mode():
                    out = replay.call(case, new)(*args, **kwargs)
                synchronize()
            except Exception:
                result.update(status="runtime_error", error=_short_tb(), failed_case=i)
                return result
            checks = strict_compare(case["output"], out, "output")
            checks += strict_compare(case["post_args"], args, "args")
            checks += strict_compare(case["post_kwargs"], kwargs, "kwargs")
            bad = [c for c in checks if not c["ok"]]
            reports.append(
                {
                    "case": i,
                    "method": case.get("method", "forward"),
                    "signature": case.get("signature"),
                    "ok": not bad,
                    "bitwise": all(c["bitwise"] for c in checks),
                    "region_calls": calls[0] - before,
                    "max_abs_err": max((c.get("max_abs_err") or 0.0 for c in checks), default=0.0),
                    "failures": bad[:5],
                }
            )
    finally:
        for handle in handles:
            handle.remove()
    correct = all(r["ok"] for r in reports)
    result.update(
        status="ok" if correct and calls[0] else "incorrect",
        verified=correct and calls[0] > 0,
        bitwise=all(r["bitwise"] for r in reports),
        region_class=cls,
        region_instances=len(regions),
        region_calls=calls[0],
        cases=reports,
    )
    if correct and not calls[0]:
        result["error"] = (
            f"the rewritten parent never called its `{cls}` module on the captured calls: "
            "call it as a module (`self.region(...)`) where the region's ops were"
        )
    return result


def run_verify(
    capture: Path,
    rewrite: Path,
    target_id: str,
    *,
    capture_sha256: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """:func:`verify` in a fresh subprocess under the GPU lock."""
    cmd = [sys.executable, "-m", "kernel_agent.region", str(capture), str(rewrite)]
    cmd += ["--target", target_id]
    if capture_sha256:
        cmd += ["--capture-sha256", capture_sha256]
    with gpu_lock():
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=child_env()
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "verified": False, "error": f"exceeded {timeout:.0f}s"}
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith(MARKER):
            data: dict[str, Any] = json.loads(line[len(MARKER) :])
            return data
    tail = (proc.stderr or proc.stdout)[-4000:]
    return {"status": "crash", "verified": False, "returncode": proc.returncode, "error": tail}


# ------------------------------------------------------------------ orchestrator side


def check(run: RunDir, target_id: str, keeper: Truth, *, timeout: float) -> dict[str, Any]:
    """The ``verify_rewrite`` tool: the agent's ``rewrite.py`` on the parent's capture."""
    capture = parent_capture(run, target_id)
    src = run.target(target_id) / REWRITE_FILE
    if not capture.exists():
        return {"status": "error", "error": f"{target_id} is not a captured region target"}
    if not src.is_file():
        return {"status": "error", "error": f"write {src} first"}
    try:
        sha256 = keeper.expect(capture)
    except TamperError as exc:
        return {"status": "error", "error": str(exc)}
    return run_verify(capture, src, target_id, capture_sha256=sha256, timeout=timeout)


def install(run: RunDir, target_id: str, keeper: Truth, *, timeout: float) -> dict[str, Any]:
    """Seal a copy of the agent's ``rewrite.py`` and verify that copy on the parent's
    capture; the copy (:func:`verified_rewrite`) is kept only when it passes."""
    src = run.target(target_id) / REWRITE_FILE
    if not src.is_file():
        return {"status": "missing", "verified": False, "error": f"no {REWRITE_FILE} written"}
    dst = truth.replace(verified_rewrite(run, target_id))
    shutil.copyfile(src, dst)
    keeper.seal(dst)
    capture = parent_capture(run, target_id)
    try:
        sha256 = keeper.expect(capture)
        result = run_verify(capture, dst, target_id, capture_sha256=sha256, timeout=timeout)
        keeper.verify(dst)  # unchanged while it was verified
    except TamperError as exc:
        result = {"status": "tampered", "verified": False, "error": str(exc)}
    if not result.get("verified"):
        dst.unlink(missing_ok=True)
    ledger.event(
        run,
        "rewrite",
        target=target_id,
        verified=bool(result.get("verified")),
        bitwise=result.get("bitwise"),
        status=result.get("status"),
    )
    return result


def summary(result: dict[str, Any]) -> dict[str, Any]:
    """The verification result in ``spec.json`` (``rewrite``)."""
    keys = ("status", "verified", "bitwise", "region_instances", "region_calls", "error")
    out = {k: result[k] for k in keys if k in result}
    if isinstance(out.get("error"), str):
        out["error"] = out["error"][-1500:]
    cases = result.get("cases") or []
    out["cases"] = [{k: c.get(k) for k in ("signature", "ok", "bitwise")} for c in cases]
    return out


def rewrite_ok(run: RunDir, target_id: str, keeper: Truth) -> bool:
    """Whether a target's kernel may be integrated: a module target, or a region target
    whose verified rewrite is still the sealed file."""
    if not is_region(read_json(run.target(target_id) / "spec.json", {}) or {}):
        return True
    path = verified_rewrite(run, target_id)
    try:
        keeper.verify(path)
    except TamperError:
        return False
    return path.is_file()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a region rewrite on a parent capture.")
    parser.add_argument("capture", type=Path)
    parser.add_argument("rewrite", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--capture-sha256", help="refuse a capture without this digest")
    ns = parser.parse_args(argv)
    try:
        result = verify(ns.capture, ns.rewrite, ns.target, capture_sha256=ns.capture_sha256)
    except Exception:
        result = {"status": "harness_error", "verified": False, "error": _short_tb()}
    print(MARKER + json.dumps(result, default=str), flush=True)
    return 0 if result.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
