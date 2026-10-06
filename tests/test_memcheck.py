"""Kernels under compute-sanitizer memcheck (issue #115).

CPU: finding a usable sanitizer (the pip wheel's layout cannot launch anything), installing
NVIDIA's, the log parser, the odd-size variants, :func:`memcheck.run_memcheck` with a stub
sanitizer and the integration with a fake one. GPU (``gpu``, skipped without a usable
sanitizer): the self-test and Triton kernels that overrun, one only on a partial tile.
"""

import asyncio
import hashlib
import io
import json
import os
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest
import torch
from test_recheck import kernel, make, simulated, worker  # noqa: F401 (fixtures)

from kernel_agent import improve, ledger, toolchain
from kernel_agent.agent.prompts import EXAMPLES_DIR
from kernel_agent.kernels import memcheck, recheck
from kernel_agent.report import write_report
from kernel_agent.scheduler import KERNEL, Arm, Policy
from kernel_agent.workspace import read_json

#: A memcheck log of kernel 004 (issue #115) on an odd-size variant, shortened.
LOG_004 = """========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 16 bytes
=========     at _gemm_kernel+0x3440 in 004_triton_bf16_v3_6902171c.py:42
=========     by thread (32,0,0) in block (453,0,0)
=========     Access to 0x1001c3ecc00 is out of bounds
=========     and is 1.536 bytes before the nearest allocation at 0x1001c3ed200 of size 1.024 bytes
=========         Device Frame: _gemm_kernel+0x12c0 in 004_triton_bf16_v3_6902171c.py:122
=========
========= Invalid __global__ read of size 16 bytes
=========     at _gemm_kernel+0x3440 in 004_triton_bf16_v3_6902171c.py:42
=========
========= ERROR SUMMARY: 2752 errors
========= ERROR SUMMARY: 2742 errors were not printed. Use --print-limit option to adjust
"""
#: What every launch of the pip wheel's sanitizer ends with.
LOG_NO_LAUNCH = """========= COMPUTE-SANITIZER
========= Error: Target application terminated before first instrumented API call
========= Error: couldn't find exit code.
"""


def executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def fake_tool(folder: Path, parts=toolchain.SANITIZER_PARTS) -> Path:
    """A sanitizer directory: a binary that reports a version, and ``parts`` beside it."""
    binary = executable(folder / "compute-sanitizer", "#!/bin/sh\necho 'Version 2026.3.0.0'\n")
    for part in parts:
        (folder / part).write_text("")
    return binary


# ------------------------------------------------------------------ finding the sanitizer


def test_the_pip_wheel_layout_is_not_usable_a_complete_install_is(tmp_path):
    wheel = fake_tool(tmp_path / "nvidia" / "cu13" / "bin", parts=())  # libraries in lib/
    for part in toolchain.SANITIZER_PARTS[1:]:
        (tmp_path / "nvidia" / "cu13" / "lib").mkdir(exist_ok=True)
        (tmp_path / "nvidia" / "cu13" / "lib" / part).write_text("")
    found = toolchain.find_sanitizer([wheel])
    assert found.path is None and len(found.rejected) == 1
    assert "TreeLauncherSubreaper" in found.reason and "--fetch-sanitizer" in found.reason
    assert "unavailable" in found.describe()

    # NVIDIA's archive (and a toolkit): bin/compute-sanitizer runs the real binary
    real = fake_tool(tmp_path / "archive" / "compute-sanitizer")
    wrapper = executable(
        tmp_path / "archive" / "bin" / "compute-sanitizer",
        '#!/bin/sh\nexec "$(dirname "$0")"/../compute-sanitizer/compute-sanitizer "$@"\n',
    )
    found = toolchain.find_sanitizer([wheel, wrapper])
    assert found.path == str(real.resolve()) and found.version == "2026.3.0.0"
    assert len(found.rejected) == 1  # the wheel, listed by doctor
    assert "2026.3.0.0" in found.describe()

    lame = tmp_path / "lame" / "compute-sanitizer"
    lame.parent.mkdir()
    lame.write_text("")
    assert "not executable" in toolchain.find_sanitizer([lame]).rejected[0]
    assert toolchain.find_sanitizer([tmp_path / "nowhere"]).reason.startswith("not found")


def test_where_the_sanitizer_is_looked_for(tmp_path, monkeypatch):
    monkeypatch.setenv(toolchain.SANITIZER_ENV, str(tmp_path / "mine"))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path / "cache")
    for version in ("13.4.92", "13.10.1"):
        fake_tool(tmp_path / "cache" / "compute-sanitizer" / f"a-{version}" / "compute-sanitizer")
    places = [str(p) for p in toolchain._sanitizer_places()]
    assert places[0] == str(tmp_path / "mine")
    assert places[1] == str(tmp_path / "cuda" / "compute-sanitizer" / "compute-sanitizer")
    fetched = [p for p in places if "cache" in p]
    assert ["13.10.1" in fetched[0], "13.4.92" in fetched[1]] == [True, True]  # newest first


def _archive(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:xz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def test_fetch_installs_nvidias_sanitizer_for_the_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    root = "cuda_sanitizer_api-linux-x86_64-13.4.92-archive/compute-sanitizer/"
    files = {root + "compute-sanitizer": b"#!/bin/sh\necho 'Version 2026.3.0.0'\n"}
    files |= {root + part: b"" for part in toolchain.SANITIZER_PARTS}
    archive = _archive(files)
    path = "cuda_sanitizer_api/linux-x86_64/x.tar.xz"
    entry = {"relative_path": path, "sha256": hashlib.sha256(archive).hexdigest(), "size": "1"}
    web = {
        toolchain.REDIST_URL: b"redistrib_13.3.0.json redistrib_13.4.2.json redistrib_13.5.0.json",
        toolchain.REDIST_URL + "redistrib_13.4.2.json": json.dumps(
            {"cuda_sanitizer_api": {"linux-x86_64": entry}}
        ).encode(),
        toolchain.REDIST_URL + path: archive,
    }
    asked: list[str] = []

    def fetch(url):
        asked.append(url)
        return web[url]

    binary = toolchain.fetch_sanitizer((13, 4), fetch=fetch, log=lambda _: None)
    assert toolchain.REDIST_URL + "redistrib_13.5.0.json" not in asked  # newer than the driver
    assert binary == tmp_path / "cache" / "compute-sanitizer" / root / "compute-sanitizer"
    monkeypatch.delenv(toolchain.SANITIZER_ENV, raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    assert toolchain.find_sanitizer().path == str(binary)

    web[toolchain.REDIST_URL + path] = archive + b"x"
    with pytest.raises(RuntimeError, match="sha256"):
        toolchain.fetch_sanitizer((13, 4), fetch=fetch, log=lambda _: None)
    with pytest.raises(RuntimeError, match="no CUDA redistributable"):
        toolchain.fetch_sanitizer((12, 0), fetch=fetch, log=lambda _: None)


# ------------------------------------------------------------------ the log and the variants


def test_parse_log():
    errors, report = memcheck.parse_log(LOG_004)
    assert errors == 2752
    lines = report.splitlines()
    assert lines[0] == "Invalid __global__ read of size 16 bytes"
    assert lines[1] == "at _gemm_kernel+0x3440 in 004_triton_bf16_v3_6902171c.py:42"
    assert len(lines) == 6 and "Device Frame" in lines[-1]  # the first report only
    assert memcheck.parse_log(LOG_NO_LAUNCH) == (0, "")
    clean = "========= COMPUTE-SANITIZER\nhello\n========= ERROR SUMMARY: 0 errors\n"
    assert memcheck.parse_log(clean) == (0, "")


def test_odd_variants_shrink_the_sizes_that_differ_between_cases():
    torch.manual_seed(0)
    x0 = torch.randn(16, 240, 64).transpose(1, 2)  # [B, C, T], channels-last in memory
    cases = [
        {"args": (x0, torch.tensor([2])), "kwargs": {}, "method": "forward"},
        {"args": (torch.randn(16, 64, 32), torch.tensor([2])), "kwargs": {}},
        {"args": (torch.randn(8, 64, 32),), "kwargs": {"sr": torch.tensor([1])}},
        {"args": (torch.randn(4, 4),), "kwargs": {}, "method": "step"},  # alone of its kind
    ]
    found = memcheck.odd_variants(cases)
    assert [(i, shapes) for i, shapes, _, _ in found] == [
        (0, [[15, 64, 239], [1]]),
        (1, [[15, 64, 31], [1]]),
        (2, [[7, 64, 31], [1]]),
    ]
    args = found[0][2]
    assert torch.equal(args[0], x0[:15, :, :239]) and args[1] is cases[0]["args"][1]
    assert args[0].stride() == (239 * 64, 1, 64)  # the captured dimension order, no padding
    assert found[2][3]["sr"] is cases[2]["kwargs"]["sr"]
    assert cases[0]["args"][0] is x0  # the captured case is not changed

    same = torch.randn(6, 8)  # one tensor passed twice stays one tensor
    twice = [{"args": (same, same), "kwargs": {}}, {"args": (same[:2], same[:2]), "kwargs": {}}]
    (_, _, a, _), (_, _, b, _) = memcheck.odd_variants(twice)
    assert a[0] is a[1] and a[0].shape == (5, 8) and b[0].shape == (1, 8)


# ------------------------------------------------------------------ run_memcheck (stub sanitizer)

#: A stand-in for compute-sanitizer: records its arguments and environment, writes the log
#: of $STUB_LOG to --log-file and either runs the command or answers for it ($STUB_CHILD).
STUB = """#!{python}
import json, os, subprocess, sys, time
args = sys.argv[1:]
log = args[args.index("--log-file") + 1]
cmd = args[args.index("--log-file") + 2:]
with open(os.environ["STUB_RECORD"], "w") as f:
    env = {{"PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")}}
    json.dump({{"args": args, "env": env}}, f)
if os.environ.get("STUB_GRANDCHILD"):  # the checked process runs below the launcher
    g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open(os.environ["STUB_GRANDCHILD"], "w").write(str(g.pid))
time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
open(log, "w").write(os.environ.get("STUB_LOG", ""))
child = os.environ.get("STUB_CHILD")
if child is None:
    sys.exit(subprocess.run(cmd).returncode)
nonce = sys.stdin.readline().strip()
if child:
    print("@@KA_MEMCHECK@@" + nonce + "@@" + child)
"""


@pytest.fixture
def stub(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # the child on the CPU
    monkeypatch.setenv("KERNEL_AGENT_LOCK_HELD", "1")
    monkeypatch.setenv("STUB_RECORD", str(tmp_path / "record.json"))
    binary = executable(tmp_path / "stub" / "compute-sanitizer", STUB.format(python=sys.executable))
    return toolchain.Sanitizer(str(binary), "stub")


def _record(tmp_path):
    return json.loads((tmp_path / "record.json").read_text())


def test_run_memcheck_with_a_stub_sanitizer(tmp_path, stub, monkeypatch):
    capture, candidate = tmp_path / "c.pt", tmp_path / "k.py"
    torch.save({"module": torch.nn.Identity(), "cases": []}, capture)
    candidate.write_text("def build(reference):\n    return reference\n")
    run = lambda **kw: memcheck.run_memcheck(capture, candidate, tool=stub, **kw)  # noqa: E731

    result = run()  # the real checked process, on the CPU: nothing to check
    assert result["status"] == "skipped" and result["reason"] == "no CUDA device"
    record = _record(tmp_path)
    assert record["args"][:2] == ["--tool", "memcheck"] and "--report-api-errors" in record["args"]
    assert record["env"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
    assert "kernel_agent.kernels.memcheck" in record["args"]
    assert record["args"][-2:] == [str(capture.resolve()), str(candidate.resolve())]

    ok = {"status": "ok", "cases": 3, "variants": [{"case": 0, "shapes": [[15]]}]}
    monkeypatch.setenv("STUB_CHILD", json.dumps(ok))
    monkeypatch.setenv(
        "STUB_LOG", "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
    )
    result = run()
    assert result["status"] == "ok" and result["cases"] == 3 and "seconds" in result
    assert memcheck.describe(result).startswith("memcheck clean on 3 case(s) + 1 odd-size")

    monkeypatch.setenv("STUB_CHILD", json.dumps({"status": "runtime_error", "error": "CUDA"}))
    monkeypatch.setenv("STUB_LOG", LOG_004)
    result = run()
    assert result["status"] == memcheck.STATUS and result["errors"] == 2752
    assert result["report"].startswith("Invalid __global__ read of size 16 bytes\nat _gemm_kernel")
    assert "2752 memory error(s)" in result["reason"] and "_gemm_kernel" in result["reason"]
    assert memcheck.describe(result).startswith("memcheck FAILED")

    monkeypatch.setenv("STUB_CHILD", "")  # the pip wheel: the target never starts
    monkeypatch.setenv("STUB_LOG", LOG_NO_LAUNCH)
    result = run()
    assert result["status"] == "error" and "couldn't find exit code" in result["reason"]

    monkeypatch.setenv("STUB_SLEEP", "30")
    monkeypatch.setenv("STUB_GRANDCHILD", str(tmp_path / "grandchild"))
    assert run(timeout=2)["status"] == "timeout"
    grandchild = int((tmp_path / "grandchild").read_text())
    stat = Path(f"/proc/{grandchild}/stat")
    assert not stat.exists() or stat.read_text().split(") ")[1][0] == "Z"  # killed too

    missing = toolchain.Sanitizer(None, reason="not found")
    assert memcheck.run_memcheck(capture, candidate, tool=missing) == {
        "status": "skipped",
        "reason": "not found",
        "seconds": 0.0,
    }


def test_the_self_test_wants_the_overrun_and_nothing_else(tmp_path, stub, monkeypatch):
    monkeypatch.setenv("STUB_CHILD", json.dumps({"status": "ok", "inbounds": True}))
    report = LOG_004.replace("_gemm_kernel", "_ka_memcheck_overrun")
    monkeypatch.setenv("STUB_LOG", report)
    assert memcheck.selftest(stub)["ok"]
    monkeypatch.setenv("STUB_LOG", report.replace("in block (453", "_ka_memcheck_inbounds ("))
    assert "in-bounds" in memcheck.selftest(stub)["reason"]
    monkeypatch.setenv("STUB_LOG", "========= ERROR SUMMARY: 0 errors\n")
    assert "not reported" in memcheck.selftest(stub)["reason"]
    monkeypatch.setenv("STUB_CHILD", "")
    monkeypatch.setenv("STUB_LOG", LOG_NO_LAUNCH)
    check = memcheck.selftest(stub)
    assert not check["ok"] and "did not run" in check["reason"]
    assert memcheck.selftest(toolchain.Sanitizer(None, reason="none")) == {
        "ok": False,
        "reason": "none",
    }


# ------------------------------------------------------------------ the integration


def _recheck_passes(capture, snap, *, verdict, capture_sha256, timeout):
    result = {"status": "ok", "correct": True, "speedup": verdict["speedup"], "seeds": 3}
    return recheck.judge({**result, "cases": []}, verdict, recheck.SPEEDUP_TOLERANCE)


@pytest.mark.usefixtures("simulated")
def test_integration_refuses_a_kernel_with_memory_errors(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    attn, mlp = kernel(run, "attn", 2.0), kernel(run, "mlp", 1.5)
    checked: list[str] = []

    def fake_memcheck(capture, snap, *, capture_sha256, timeout):
        target = Path(snap).parent.parent.name
        checked.append(target)
        assert Path(capture) == run.capture_file(target) and capture_sha256
        if target == "attn":
            errors, report = memcheck.parse_log(LOG_004)
            reason = f"{errors} memory error(s) under compute-sanitizer memcheck, the first: x"
            out = {"status": memcheck.STATUS, "errors": errors, "report": report}
            return {**out, "reason": reason, "seconds": 41.5}
        return {"status": "ok", "cases": 2, "variants": [], "seconds": 7.4}

    orch.rechecker, orch.memchecker = _recheck_passes, fake_memcheck
    orch.worker = worker([])
    asyncio.run(orch.integrate())

    assert checked == ["attn", "mlp"]
    data = read_json(run.root / "integration.json")
    assert [a["item"] for a in data["accepted"]] == [mlp]
    checks = {r["target"]: r for r in data["recheck"]}
    bad = checks["attn"]
    assert bad["item"] == attn and bad["status"] == memcheck.STATUS and not bad["passed"]
    assert (
        bad["memcheck"]["seconds"] == 41.5
        and bad["memcheck"]["report"] == memcheck.parse_log(LOG_004)[1]
    )
    assert "it passed the evaluator and the re-check" in bad["reason"]
    assert checks["mlp"]["passed"] and checks["mlp"]["memcheck"]["seconds"] == 7.4
    assert "memcheck clean on 2 case(s) (7.4 s)" in recheck.describe(checks["mlp"])
    events = ledger.events(run)
    assert [(e["target"], e["status"]) for e in events if e["event"] == "memcheck"] == [
        ("attn", "memcheck"),
        ("mlp", "ok"),
    ]
    failed = [e for e in events if e["event"] == "recheck_failed"]
    assert [(e["target"], e["status"]) for e in failed] == [("attn", "memcheck")]
    report = write_report(run).read_text()
    assert "recheck `attn`" in report and "FAILED (memcheck)" in report
    assert "  at _gemm_kernel+0x3440 in 004_triton_bf16_v3_6902171c.py:42" in report

    # the engineer of the target sees the refusal and the sanitizer's report
    digest = improve.kernel_digest(run, Arm("attn", KERNEL, 1.0), 2, 6, Policy())
    assert "## Refused by the integration" in digest
    assert f"`history/{Path(attn).name}` (memcheck)" in digest
    assert "  at _gemm_kernel+0x3440 in 004_triton_bf16_v3_6902171c.py:42" in digest
    assert improve.refusals(run, "mlp") == []

    checked.clear()  # decided memchecks are reused with their re-checks
    asyncio.run(orch.integrate(reuse=True))
    assert checked == []
    again = read_json(run.root / "integration.json")
    assert [(r["target"], r["passed"]) for r in again["recheck"]] == [
        ("attn", False),
        ("mlp", True),
    ]


@pytest.mark.usefixtures("simulated")
def test_an_unavailable_or_broken_memcheck_keeps_the_kernel(tmp_path):
    orch = make(tmp_path)
    run = orch.run
    attn = kernel(run, "attn", 2.0)
    answers = [
        {"status": "skipped", "reason": "compute-sanitizer unavailable: not found", "seconds": 0},
        RuntimeError("the sanitizer broke"),
        {"status": "ok", "cases": 2, "seconds": 3.0},
    ]

    def fake_memcheck(capture, snap, *, capture_sha256, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    orch.rechecker, orch.memchecker = _recheck_passes, fake_memcheck
    orch.worker = worker([])
    asyncio.run(orch.integrate())
    data = read_json(run.root / "integration.json")
    assert [a["item"] for a in data["accepted"]] == [attn]
    (rec,) = data["recheck"]
    assert rec["passed"] and rec["memcheck"]["status"] == "skipped"
    assert "memcheck skipped: compute-sanitizer unavailable" in recheck.describe(rec)

    asyncio.run(orch.integrate(reuse=True))  # not decided: checked again (it broke)
    (rec,) = read_json(run.root / "integration.json")["recheck"]
    assert rec["passed"] and rec["memcheck"]["status"] == "error"
    assert "the sanitizer broke" in rec["memcheck"]["reason"]

    asyncio.run(orch.integrate(reuse=True))
    (rec,) = read_json(run.root / "integration.json")["recheck"]
    assert rec["memcheck"]["status"] == "ok" and answers == []

    sim = make(tmp_path / "sim")  # a simulated run without a fake: skipped
    kernel(sim.run, "attn", 2.0)
    sim.rechecker, sim.worker = _recheck_passes, worker([])
    asyncio.run(sim.integrate())
    (rec,) = read_json(sim.run.root / "integration.json")["recheck"]
    assert rec["memcheck"] == {"status": "skipped", "reason": "simulated run"}
    assert not [e for e in ledger.events(sim.run) if e["event"] == "memcheck"]


# ------------------------------------------------------------------ GPU

#: The bundled Triton RMSNorm with BLOCK_M rows per program for 4+ rows, rows not masked on
#: load: right for every row count, but a partial last tile reads past the input.
TILED = textwrap.dedent(
    """
    import torch
    import triton
    import triton.language as tl
    from torch import nn


    @triton.jit
    def _tiled_rmsnorm(x_ptr, w_ptr, y_ptr, M, n_cols, eps, BLOCK_M: tl.constexpr,
                       BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK)
        cmask = cols < n_cols
        x = tl.load(x_ptr + rows[:, None] * n_cols + cols[None, :], mask=cmask[None, :],
                    other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / n_cols
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0)
        y = (x * tl.rsqrt(var + eps)[:, None]).to(w.dtype) * w[None, :]
        tl.store(y_ptr + rows[:, None] * n_cols + cols[None, :], y,
                 mask=(rows < M)[:, None] & cmask[None, :])


    class Tiled(nn.Module):
        def __init__(self, reference):
            super().__init__()
            self.weight = reference.weight
            self.eps = float(getattr(reference, "variance_epsilon", 1e-6))

        def forward(self, hidden_states):
            shape = hidden_states.shape
            x = hidden_states.reshape(-1, shape[-1]).contiguous()
            y = torch.empty_like(x)
            M, n = x.shape
            block_m = 4 if M >= 4 else 1
            _tiled_rmsnorm[(triton.cdiv(M, block_m),)](
                x, self.weight, y, M, n, self.eps, BLOCK_M=block_m,
                BLOCK=triton.next_power_of_2(n))
            return y.view(shape)


    def build(reference):
        return Tiled(reference)
    """
)


def _usable() -> toolchain.Sanitizer:
    tool = toolchain.sanitizer()
    if not tool.path:
        pytest.skip(tool.reason or "no compute-sanitizer")
    return tool


@pytest.mark.gpu
def test_memcheck_self_test():
    check = memcheck.selftest(_usable())
    print(check)
    assert check["ok"], check
    assert "_ka_memcheck_overrun" in check["report"]


@pytest.mark.gpu
def test_memcheck_catches_an_overrun_on_a_partial_tile_the_evaluator_accepts(tmp_path):
    from kernel_agent.kernels.evaluate import run_evaluation
    from kernel_agent.selftest import make_rmsnorm_capture

    tool = _usable()
    capture = make_rmsnorm_capture(tmp_path / "rmsnorm.pt")  # 256 rows and 1 row
    honest = memcheck.run_memcheck(capture, EXAMPLES_DIR / "triton_rmsnorm.py", tool=tool)
    print(memcheck.describe(honest))
    assert honest["status"] == "ok", honest
    assert honest["variants"] == [{"case": 0, "shapes": [[1, 255, 2048]]}]

    tiled = tmp_path / "tiled.py"
    tiled.write_text(TILED)
    verdict = run_evaluation(capture, tiled)
    why = {k: verdict.get(k) for k in ("status", "stage", "failed_check", "error")}
    assert verdict["correct"], why  # every check of the evaluator passes
    captured = memcheck.run_memcheck(capture, tiled, tool=tool, variants=False)
    assert captured["status"] == "ok", captured  # 256 rows: whole tiles only
    result = memcheck.run_memcheck(capture, tiled, tool=tool)
    print(memcheck.describe(result), result.get("report"))
    assert result["status"] == memcheck.STATUS and result["errors"] > 0, result
    assert result["report"].startswith("Invalid __global__ read")
    assert "_tiled_rmsnorm" in result["report"] and result["seconds"] > 0
    assert os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") is None  # only in the child
