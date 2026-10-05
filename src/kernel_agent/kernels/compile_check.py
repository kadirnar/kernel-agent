"""Compile compatibility of a kernel candidate (optional evaluator stage).

Models are often run under ``torch.compile`` (VoxCPM's ``model.optimize()``,
transformers' static-cache decode, a compiled denoiser), so a kernel that only
works eagerly may be lost on the compiled baseline.  ``evaluate(...,
compile_check=True)`` / ``kernel-agent eval --compile-check`` runs this on a
fresh ``build()`` of the candidate:

* ``torch._dynamo.explain`` per captured entrypoint: graphs and **graph
  breaks** with their reasons.  Launchers Dynamo cannot trace (pybind
  ``load_inline`` modules, NVRTC / ``cuda.core`` handles, CuTe ``cute.compile``
  functions) break the graph; under ``fullgraph=True`` that is an error.
* ``torch.compile(entrypoint, fullgraph=False)`` (Inductor on CUDA, tracing
  only via ``aot_eager`` on CPU) on every captured case, outputs and in-place
  side effects checked like the eager check.

Wrapping the launcher in ``torch.library.custom_op`` + ``register_fake``
(``examples/triton_rmsnorm_custom_op.py``) makes the kernel one opaque op for
Dynamo / Inductor / CUDA graphs instead of a graph break.
"""

from __future__ import annotations

import copy
import time
import traceback
from collections.abc import Callable
from typing import Any

MAX_REASONS = 5


def _reasons(explanation: Any) -> list[str]:
    out = []
    for reason in getattr(explanation, "break_reasons", []) or []:
        text = str(getattr(reason, "reason", reason)).strip().splitlines()[0][:240]
        stack = getattr(reason, "user_stack", None) or []
        if stack:
            frame = stack[-1]
            text += f" ({frame.filename.rsplit('/', 1)[-1]}:{frame.lineno} in {frame.name})"
        if text not in out:
            out.append(text)
    return out[:MAX_REASONS]


def check(
    build: Callable[[], Any], cases: list[dict[str, Any]], *, backend: str | None = None
) -> dict[str, Any]:
    """Graph breaks and compiled correctness of the candidate made by ``build()``
    (a fresh instance: tracing must not touch the one being timed) on the captured
    ``cases``.  ``passed``: every case runs compiled and matches the reference
    outputs; ``fullgraph_ok``: no graph break (it also works under ``fullgraph=True``)."""
    import torch
    import torch._dynamo as dynamo

    from kernel_agent.kernels.compare import compare_side_effects, compare_structures
    from kernel_agent.profiling.methods import entrypoint
    from kernel_agent.workloads.base import synchronize

    on_cuda = any(
        isinstance(t, torch.Tensor) and t.is_cuda
        for case in cases
        for t in [*case["args"], *case["kwargs"].values()]
    )
    backend = backend or ("inductor" if on_cuda else "aot_eager")
    result: dict[str, Any] = {
        "passed": False,
        "backend": backend,
        "graphs": 0,
        "graph_breaks": 0,
        "break_reasons": [],
    }
    t0 = time.perf_counter()
    dynamo.reset()
    try:
        candidate = build()
        compiled: dict[str, Any] = {}
        reasons: list[str] = []
        for case in cases:
            method = case.get("method", "forward")
            if method in compiled:
                continue
            fn = entrypoint(candidate, method)
            args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
            with torch.inference_mode():
                explanation = dynamo.explain(fn)(*args, **kwargs)
            result["graphs"] += int(explanation.graph_count)
            result["graph_breaks"] += int(explanation.graph_break_count)
            reasons += [r for r in _reasons(explanation) if r not in reasons]
            compiled[method] = torch.compile(fn, backend=backend, fullgraph=False)
        result["break_reasons"] = reasons[:MAX_REASONS]
        cases_out = []
        for i, case in enumerate(cases):
            args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
            with torch.inference_mode():
                out = compiled[case.get("method", "forward")](*args, **kwargs)
            synchronize()
            checks = compare_structures(case["output"], out, "output")
            checks += compare_side_effects(case["args"], case["post_args"], args, "args")
            checks += compare_side_effects(case["kwargs"], case["post_kwargs"], kwargs, "kwargs")
            bad = [c for c in checks if not c.get("ok")]
            cases_out.append({"case": i, "ok": not bad, "failures": bad[:3]})
        result["cases"] = cases_out
        result["passed"] = all(c["ok"] for c in cases_out)
    except Exception:
        text = traceback.format_exc()
        result["error"] = text if len(text) <= 3000 else "...\n" + text[-3000:]
    finally:
        dynamo.reset()  # drop the compiled graphs before anything else runs
    result["fullgraph_ok"] = "error" not in result and result["graph_breaks"] == 0
    result["seconds"] = round(time.perf_counter() - t0, 1)
    return result
