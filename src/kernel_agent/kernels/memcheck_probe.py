"""The program :func:`kernel_agent.kernels.memcheck.selftest` runs under the sanitizer: a
Triton copy of 1000 floats within bounds, then one that reads a whole 1024-element block of
them (24 elements past the end; the output is masked, so it is still correct).

    python -m kernel_agent.kernels.memcheck_probe
"""

from __future__ import annotations

import json
import os
import sys

import torch
import triton
import triton.language as tl

from kernel_agent.kernels.memcheck import RESULT_MARKER


@triton.jit
def _ka_memcheck_inbounds(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0), mask=mask)


@triton.jit
def _ka_memcheck_overrun(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs), mask=offs < n)  # the load is not masked


def main() -> int:
    tag = sys.stdin.readline().strip()
    result: dict[str, object] = {"status": "ok"}
    try:
        x = torch.arange(1000, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        _ka_memcheck_inbounds[(1,)](x, y, x.numel(), BLOCK=1024)
        torch.cuda.synchronize()
        result["inbounds"] = bool(torch.equal(x, y))
    except Exception as exc:
        result.update(status="error", error=repr(exc)[:500])
    else:
        try:
            _ka_memcheck_overrun[(1,)](x, y, x.numel(), BLOCK=1024)
            torch.cuda.synchronize()  # a memory error under the sanitizer ends the context
        except Exception as exc:
            result["overrun_error"] = repr(exc)[:300]
    sys.stdout.write(f"{RESULT_MARKER}{tag}@@" + json.dumps(result) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
