"""Compile a candidate's ``load_inline`` extensions before its evaluation, outside the GPU lock
(docs/MULTIAGENT.md §3.3 "CPU-only work off the queue", issue #185).

nvcc builds are about a third of a kernel session. A build under the GPU lock keeps every
other session's evaluation waiting, so with clean timing on (``hygiene.py``, several
sessions at once) ``run_evaluation`` and ``run_sweep`` first run :func:`prebuild`: the
candidate's builds in a subprocess that sees no GPU (``CUDA_VISIBLE_DEVICES=""``; the
architectures from ``TORCH_CUDA_ARCH_LIST``, as the toolchain sets it for the evaluator too)
at the background CPU share (``hygiene.background_env``: nice, the other cores,
``MAX_JOBS``), into the same ``TORCH_EXTENSIONS_DIR``. The evaluator's own ``load_inline``
then finds the build up to date (ninja has no work) and only loads it.

What the subprocess runs, each step's errors ignored (the evaluator reports them):

1. the candidate's import: a ``load_inline`` at module level compiles;
2. its module-level functions that call ``load_inline`` / ``cpp_extension.load`` and need no
   argument (the ``_load()`` getters of most candidates);
3. when a loader takes arguments (its configuration comes from ``build``) and a capture of
   the target's inputs is given (``--capture``: the inputs-only copy, never the answer key,
   :func:`inputs_capture`): ``build(reference, **config)`` for every config of a sweep (else
   with its defaults), on a reference whose weights are fake CUDA tensors (:func:`_fake_cuda`:
   a build that checks ``is_cuda`` gets to its compile; what it launches then fails here).

Triton, CuTe DSL and TileLang kernels compile at their first launch, on the GPU: they still
compile under the lock. A native project (``kernel_project.toml``) has its own prebuild
(``native/project.py``).

Run as a subprocess::

    python -m kernel_agent.kernels.prebuild CANDIDATE [--capture INPUTS] [--configs JSON]
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kernel_agent import hygiene, interrupt

RESULT_MARKER = "@@KA_PREBUILD@@"
BUILT_MARKER = "@@KA_PREBUILT@@"  # one line per extension, as it is built
TIMEOUT_S = 600.0  # a prebuild that takes longer is stopped; the evaluator builds the rest
_MARK = re.compile(r"\bload_inline\b|\bcpp_extension\b")
_LOADERS = ("load_inline", "load")  # torch.utils.cpp_extension's JIT builds


# ------------------------------------------------------------------ the source


def _cpp_names(tree: ast.Module) -> dict[str, str]:
    """Local name -> ``torch.utils.cpp_extension`` function of the loaders imported by name."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("cpp_extension"):
            for alias in node.names:
                if alias.name in _LOADERS:
                    names[alias.asname or alias.name] = alias.name
    return names


def _calls_loader(node: ast.AST, names: dict[str, str]) -> bool:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Name) and func.id in names:
            return True
        if isinstance(func, ast.Attribute) and (
            func.attr == "load_inline"
            or (func.attr == "load" and ast.unparse(func.value).endswith("cpp_extension"))
        ):
            return True
    return False


def _no_arguments(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    a = fn.args
    required = len(a.posonlyargs) + len(a.args) - len(a.defaults)
    return required == 0 and all(d is not None for d in a.kw_defaults)


def loaders(source: str) -> tuple[list[str], bool]:
    """The module-level functions of ``source`` that call a loader and need no argument, and
    whether a loader is called elsewhere (a function with arguments, a method, ``build``)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], False
    names = _cpp_names(tree)
    plain: list[str] = []
    other = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _calls_loader(node, names):
            if _no_arguments(node):
                plain.append(node.name)
            else:
                other = True
        elif isinstance(node, ast.ClassDef | ast.AsyncFunctionDef) and _calls_loader(node, names):
            other = True
    return plain, other


def inputs_capture(capture: Path) -> Path | None:
    """The capture a prebuild may give the candidate for :func:`prebuild`'s step 3: for a
    run's full capture (``.truth/captures/<id>.pt``) the inputs-only copy in the target's
    directory (``capture_inputs.pt``); a run without ``.truth/`` keeps its capture in the
    agent's directory anyway; None for any other file."""
    from kernel_agent.workspace import TRUTH_DIR

    capture = Path(capture)
    if capture.parent.name == "captures" and capture.parent.parent.name == TRUTH_DIR:
        inputs = capture.parent.parent.parent / "targets" / capture.stem / "capture_inputs.pt"
        return inputs if inputs.exists() else None
    if capture.name in ("capture.pt", "capture_inputs.pt") and capture.parent.parent.name == (
        "targets"
    ):
        return capture
    return None


# ------------------------------------------------------------------ the parent


def prebuild(
    candidate: Path,
    capture: Path | None = None,
    configs: list[dict[str, Any]] | None = None,
    *,
    timeout: float = TIMEOUT_S,
) -> dict[str, Any]:
    """Build ``candidate``'s ``load_inline`` extensions in a subprocess without a GPU (see the
    module docstring; ``capture``: an inputs-only capture for ``build``, ``configs``: a
    sweep's). ``status``: ``ok`` (``built``: each extension's name, seconds and error, if
    any; ``seconds``), ``skipped`` (``reason``), ``timeout`` or ``error``. It never fails
    an evaluation: whatever it could not build, the evaluator builds."""
    path = Path(candidate)
    if path.is_dir() or path.suffix != ".py":
        return {"status": "skipped", "reason": "not a single-file candidate"}
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError):
        return {"status": "skipped", "reason": "unreadable"}
    if not _MARK.search(source):
        return {"status": "skipped", "reason": "no load_inline"}
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return {"status": "skipped", "reason": "TORCH_CUDA_ARCH_LIST is not set"}
    plain, other = loaders(source)
    if not plain and not (other and capture is not None):
        return {"status": "skipped", "reason": "no loader to call without a capture"}
    cmd = [sys.executable, "-m", "kernel_agent.kernels.prebuild", str(path)]
    if other and capture is not None:
        cmd += ["--capture", str(capture)]
        if configs:
            cmd += ["--configs", json.dumps(configs)]
    interrupt.check()  # a stopping run starts nothing
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        interrupt.PARENT_ENV: str(os.getpid()),  # it dies with this process
        **hygiene.background_env(),
    }
    start = time.perf_counter()
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    ) as proc:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            interrupt.kill(proc.pid)  # with ninja and nvcc
            proc.communicate()
            return {"status": "timeout", "seconds": round(time.perf_counter() - start, 1)}
    seconds = round(time.perf_counter() - start, 1)
    lines = out.splitlines()
    for line in lines[::-1]:
        if line.startswith(RESULT_MARKER):
            return {**json.loads(line[len(RESULT_MARKER) :]), "seconds": seconds}
    built = [json.loads(x[len(BUILT_MARKER) :]) for x in lines if x.startswith(BUILT_MARKER)]
    return {"status": "error", "built": built, "error": (err or out)[-2000:], "seconds": seconds}


# ------------------------------------------------------------------ in the subprocess


def _recording(built: list[dict[str, Any]]) -> None:
    """Record every ``load_inline`` / ``load`` of this process in ``built`` (name, seconds,
    error); the candidate imports them after this, so it calls the recording ones."""
    from torch.utils import cpp_extension

    def wrap(original: Any) -> Any:
        def recorded(*args: Any, **kwargs: Any) -> Any:
            entry: dict[str, Any] = {"name": str(kwargs.get("name") or (args or ["?"])[0])}
            built.append(entry)
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[-2000:]
                raise
            finally:
                entry["seconds"] = round(time.perf_counter() - start, 2)
                # also on its own line: what a crash later in the build (a kernel launched on
                # a fake tensor) leaves of the result
                print(BUILT_MARKER + json.dumps(entry), flush=True)

        return recorded

    for name in _LOADERS:
        setattr(cpp_extension, name, wrap(getattr(cpp_extension, name)))


def _fake_cuda(module: Any) -> tuple[Any, Any]:
    """``module`` with every parameter and buffer a ``FakeTensor`` on the GPU (same shape,
    strides and dtype; no memory, no CUDA) and the mode its build runs in: a build that
    checks its weights are CUDA tensors before it compiles gets to its ``load_inline`` in
    this process without a GPU (its kernels' own launches then fail: the evaluator runs
    them). None for the mode when ``torch`` has no fake tensors: the weights stay on the CPU."""
    import torch

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        return module, None
    mode = FakeTensorMode(allow_non_fake_inputs=True)

    def fake(t: torch.Tensor) -> torch.Tensor:
        with mode:
            return torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="cuda")

    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is not None:
                sub._parameters[name] = torch.nn.Parameter(fake(param), requires_grad=False)
        for name, buffer in list(sub._buffers.items()):
            if buffer is not None:
                sub._buffers[name] = fake(buffer)
    return module, mode


def _build(path: Path, capture: Path | None, configs: list[dict[str, Any]]) -> dict[str, Any]:
    from kernel_agent import toolchain
    from kernel_agent.kernels.evaluate import load_candidate_module

    toolchain.setup()  # nvcc, its flags and ninja, as in the evaluator
    built: list[dict[str, Any]] = []
    errors: list[str] = []
    _recording(built)
    plain, other = loaders(path.read_text())
    try:
        module = load_candidate_module(path)
    except Exception as exc:
        return {"status": "ok", "built": built, "errors": [f"import: {exc!r}"[:500]]}
    for name in plain:
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # after its build, or no GPU here: the evaluator tells
                errors.append(f"{name}(): {exc!r}"[:500])
    build = getattr(module, "build", None)
    if other and capture is not None and callable(build):
        from kernel_agent.profiling.capture import load_capture

        reference = load_capture(capture, device="cpu")["module"].eval()
        reference, mode = _fake_cuda(reference)
        for config in configs or [{}]:
            try:
                with mode if mode is not None else contextlib.nullcontext():
                    build(copy.deepcopy(reference), **config)
            except Exception as exc:
                errors.append(f"build({config}): {exc!r}"[:500])
    return {"status": "ok", "built": built, **({"errors": errors[:5]} if errors else {})}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--capture", type=Path, help="an inputs-only capture for build()")
    parser.add_argument("--configs", help="a JSON list of build() keyword arguments")
    ns = parser.parse_args(argv)
    configs = json.loads(ns.configs) if ns.configs else []
    try:
        result = _build(ns.candidate.resolve(), ns.capture, configs)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[-2000:]}
    print(RESULT_MARKER + json.dumps(result, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
