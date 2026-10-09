"""What the megakernel kit (issue #225) adds to the checks: racecheck and synccheck next to
memcheck (log parsers on real logs, the runs with a stub sanitizer, the refusal), the two-run
determinism check of native candidates and the ``hang`` status of a megakernel's watchdog."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from test_evaluator_exploits import make_capture, restored_globals

from kernel_agent import toolchain
from kernel_agent.kernels import evaluate as evaluate_mod
from kernel_agent.kernels import memcheck
from kernel_agent.kernels.evaluate import evaluate
from kernel_agent.native import project as proj
from kernel_agent.native.megakernel import runtime
from kernel_agent.native.megakernel import schedule as mks

#: Real logs of compute-sanitizer 2026.3 (CUDA 13.4) on an RTX 5070 Ti: a kernel reading
#: shared memory another thread writes without a barrier, and a __syncthreads() only 16
#: threads of a block reach.
FIXTURES = Path(__file__).parent / "fixtures" / "sanitizer"

MANIFEST = """
[project]
name = "{name}"

[build]
backend = "command"
command = ["true"]
outputs = ["none.so"]
load = "none"
"""

#: A native entry (pure torch: nothing to compile on the CPU) of an RMSNorm; ``{body}``
#: changes what a call returns.
ENTRY = """
import torch
from torch import nn

{head}

class Norm(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.eps = reference.variance_epsilon
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * h.to(x.dtype)
{body}
        return out


def build(reference):
    return Norm(reference)
"""


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv(proj.CACHE_ENV, str(tmp_path / "native-cache"))


def native(root: Path, name: str, body: str = "", head: str = "") -> Path:
    root.mkdir(parents=True)
    (root / proj.MANIFEST).write_text(MANIFEST.format(name=name))
    (root / "candidate.py").write_text(
        ENTRY.format(body=textwrap.indent(textwrap.dedent(body), " " * 8), head=head)
    )
    return proj.write_bundle(root, root.parent / "bundles")


# ------------------------------------------------------------------ racecheck and synccheck


def test_racecheck_and_synccheck_logs_parse():
    race = (FIXTURES / "racecheck_probe.log").read_text()
    errors, warnings, report = memcheck.parse_race_log(race)
    assert (errors, warnings) == (2, 0)
    lines = report.splitlines()
    assert lines[0] == (
        "Error: Race reported between Write access at ka_race(int *)+0x90 in probe.cu:13"
    )
    assert lines[1] == "and Read access at ka_race(int *)+0xa0 in probe.cu:14 [2048 hazards]"
    assert len(lines) == 2  # the first report only
    sync = (FIXTURES / "synccheck_probe.log").read_text()
    errors, report = memcheck.parse_log(sync)
    assert errors == 16
    assert report.splitlines() == [
        "Barrier error detected. Divergent thread(s) in warp.",
        "at ka_divergent(int *)+0xb0 in probe.cu:20",
        "by thread (0,0,0) in block (0,0,0)",
    ]
    clean = (
        "========= COMPUTE-SANITIZER\n"
        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
    )
    assert memcheck.parse_race_log(clean) == (0, 0, "")
    warned = race.replace("Error: Race", "Warning: Race").replace(
        "2 hazards displayed (2 errors, 0 warnings)", "2 hazards displayed (0 errors, 2 warnings)"
    )
    errors, warnings, report = memcheck.parse_race_log(warned)
    assert (errors, warnings) == (0, 2) and report.startswith("Warning: Race reported")


def test_which_candidates_get_racecheck_and_synccheck(tmp_path):
    bundle = native(tmp_path / "p", "plain")
    assert memcheck.extra_tools(bundle) == (memcheck.EXTRA_TOOLS, "a native project")
    assert memcheck.extra_tools(tmp_path / "p")[0] == memcheck.EXTRA_TOOLS
    triton = tmp_path / "t.py"
    triton.write_text("import triton\ndef build(reference):\n    return reference\n")
    assert memcheck.extra_tools(triton) == ((), "")
    cuda = tmp_path / "c.py"
    cuda.write_text('SRC = "__global__ void k() { __shared__ float s[32]; }"\n')
    tools, why = memcheck.extra_tools(cuda)
    assert tools == memcheck.EXTRA_TOOLS and why == "its source uses __shared__"


#: A stand-in for compute-sanitizer: writes the log of $STUB_LOG_<tool> (or $STUB_LOG) to
#: --log-file, records each tool's arguments and answers for the checked process.
STUB = """#!{python}
import json, os, sys
args = sys.argv[1:]
tool = args[args.index("--tool") + 1]
log = args[args.index("--log-file") + 1]
with open(os.environ["STUB_RECORD"], "a") as f:
    f.write(json.dumps({{"tool": tool, "args": args}}) + "\\n")
open(log, "w").write(os.environ.get("STUB_LOG_" + tool, os.environ.get("STUB_LOG", "")))
nonce = sys.stdin.readline().strip()
print("@@KA_MEMCHECK@@" + nonce + "@@" + os.environ["STUB_CHILD"])
"""

CLEAN = "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"


@pytest.fixture
def stub(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_AGENT_LOCK_HELD", "1")
    monkeypatch.setenv("STUB_RECORD", str(tmp_path / "record.jsonl"))
    monkeypatch.setenv("STUB_CHILD", json.dumps({"status": "ok", "cases": 2}))
    monkeypatch.setenv("STUB_LOG", CLEAN)
    path = tmp_path / "stub" / "compute-sanitizer"
    path.parent.mkdir()
    path.write_text(STUB.format(python=sys.executable))
    path.chmod(0o755)
    return toolchain.Sanitizer(str(path), "stub")


def _records(tmp_path):
    return [json.loads(line) for line in (tmp_path / "record.jsonl").read_text().splitlines()]


def test_a_native_candidates_memcheck_record_lists_all_three_tools(tmp_path, stub, monkeypatch):
    capture = tmp_path / "c.pt"
    torch.save({"module": torch.nn.Identity(), "cases": []}, capture)
    bundle = native(tmp_path / "p", "engine")
    result = memcheck.run_memcheck(capture, bundle, tool=stub)
    assert result["status"] == "ok", result
    assert list(result["tools"]) == ["memcheck", "racecheck", "synccheck"]
    assert all(rec["status"] == "ok" for rec in result["tools"].values())
    assert result["tools"]["racecheck"]["why"] == "a native project"
    runs = _records(tmp_path)
    assert [r["tool"] for r in runs] == ["memcheck", "racecheck", "synccheck"]
    assert "--no-variants" not in runs[0]["args"]  # the odd sizes are memcheck's
    assert all("--no-variants" in r["args"] for r in runs[1:])
    assert runs[1]["args"][:4] == ["--tool", "racecheck", "--racecheck-report", "analysis"]
    line = memcheck.describe(result)
    assert line.startswith("memcheck clean on 2 case(s); racecheck clean, synccheck clean")

    race = (FIXTURES / "racecheck_probe.log").read_text()
    monkeypatch.setenv("STUB_LOG_racecheck", race)
    result = memcheck.run_memcheck(capture, bundle, tool=stub)
    assert result["status"] == "racecheck" and memcheck.failed(result)
    assert result["errors"] == 2 and "ka_race(int *)+0x90" in result["reason"]
    assert result["tools"]["synccheck"]["status"] == "ok"
    refused = memcheck.refuse({"passed": True, "status": "ok"}, result)
    assert refused["status"] == "racecheck" and not refused["passed"]
    assert "shared-memory race" in refused["reason"] and "schedule" in refused["reason"]
    assert not memcheck.pending(refused)

    monkeypatch.delenv("STUB_LOG_racecheck")
    monkeypatch.setenv("STUB_LOG_synccheck", (FIXTURES / "synccheck_probe.log").read_text())
    result = memcheck.run_memcheck(capture, bundle, tool=stub)
    assert result["status"] == "synccheck" and result["errors"] == 16
    assert memcheck.describe(result).startswith("synccheck FAILED")

    monkeypatch.setenv("STUB_LOG_memcheck", "========= Invalid __global__ read\n")
    result = memcheck.run_memcheck(capture, bundle, tool=stub)  # refused: the others not run
    assert result["status"] == memcheck.STATUS
    assert result["tools"]["racecheck"] == {"status": "skipped", "reason": "memcheck memcheck"}


def test_a_plain_triton_candidate_runs_memcheck_only(tmp_path, stub):
    capture = tmp_path / "c.pt"
    torch.save({"module": torch.nn.Identity(), "cases": []}, capture)
    candidate = tmp_path / "k.py"
    candidate.write_text("def build(reference):\n    return reference\n")
    result = memcheck.run_memcheck(capture, candidate, tool=stub)
    assert result["status"] == "ok" and "tools" not in result
    assert [r["tool"] for r in _records(tmp_path)] == ["memcheck"]


# ------------------------------------------------------------------ determinism


FLIP = """
if self.calls % 2:  # every other call: the last bit of one element (within any tolerance)
    out = out.clone()
    out.view(-1)[0] = torch.nextafter(out.view(-1)[0], torch.tensor(float("inf")))
"""


def test_a_native_candidate_must_give_the_same_bits_twice(tmp_path):
    capture = make_capture("rmsnorm", tmp_path / "rms.pt", "cpu")
    steady = native(tmp_path / "steady", "steady")
    flaky = native(tmp_path / "flaky", "flaky", body=FLIP)
    declared = native(
        tmp_path / "declared",
        "declared",
        body=FLIP,
        head='ORDER_DEPENDENT_ATOMICS = "split-K partials summed with fp32 atomicAdd"',
    )
    with restored_globals():
        ok = evaluate(capture, steady, device="cpu")
        bad = evaluate(capture, flaky, device="cpu")
        allowed = evaluate(capture, declared, device="cpu")
    assert ok["status"] == "ok" and ok["determinism"] == {"checked": True, "cases": 2}, ok
    assert bad["status"] == "incorrect" and bad["stage"] == "determinism", bad
    assert bad["failed_check"] == {"case": 0, "check": "determinism"}
    assert bad["determinism"]["where"] == "call.output: 1 of 16384 elements differ"
    assert "ORDER_DEPENDENT_ATOMICS" in bad["error"]
    assert allowed["status"] == "ok", allowed
    assert allowed["determinism"] == {
        "checked": False,
        "declared": "split-K partials summed with fp32 atomicAdd",
    }


#: A native entry whose sources synchronise blocks through global memory (the stress check's
#: trigger: here only words of a comment, the entry being pure torch).
COUNTERS = "# its CUDA side would signal counters with red.release.gpu and poll with ld.acquire"
#: Wrong on one call (#150), as a counter reset racing the next launch would be.
ONCE = """
if self.calls == 150:
    out = out.clone()
    out.view(-1)[0] = torch.nextafter(out.view(-1)[0], torch.tensor(float("inf")))
"""


def test_native_candidates_with_counters_are_stressed_back_to_back(tmp_path, monkeypatch):
    """Issue #248: two runs agree, yet one call of hundreds is wrong; the stress check finds it."""
    monkeypatch.setenv(evaluate_mod.STRESS_ENV, "200")
    capture = make_capture("rmsnorm", tmp_path / "rms.pt", "cpu")
    racy = native(tmp_path / "racy", "racy", body=ONCE, head=COUNTERS)
    clean = native(tmp_path / "clean", "clean", head=COUNTERS)
    with restored_globals():
        bad = evaluate(capture, racy, device="cpu")
        ok = evaluate(capture, clean, device="cpu")
    assert bad["status"] == "incorrect" and bad["stage"] == "determinism", bad
    assert bad["failed_check"]["check"] == "stress"
    stress = bad["determinism"]["stress"]
    assert (stress["calls"], stress["wrong"]) == (200, 1) and stress["first"] is not None
    assert "back-to-back calls differ bit for bit" in bad["error"]
    assert ok["status"] == "ok", ok
    assert ok["determinism"]["stress"]["wrong"] == 0 and ok["determinism"]["stress"]["calls"] == 200
    monkeypatch.setenv(evaluate_mod.STRESS_ENV, "0")  # switched off: two runs only
    with restored_globals():
        assert "stress" not in evaluate(capture, clean, device="cpu")["determinism"]


def test_the_stress_inputs_are_exact_variants_of_the_captured_ones():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    ids = torch.tensor([1, 2])
    variants = [evaluate_mod._variant((x, ids), {"s": x}, k) for k in range(4)]
    assert torch.equal(variants[0][0][0], x) and variants[0][0][0] is not x
    assert torch.equal(variants[1][0][0], -x) and torch.equal(variants[2][0][0], x.roll(1, -1))
    assert torch.equal(variants[3][1]["s"], x * 0.5) and variants[3][1]["s"] is variants[3][0][0]
    assert all(torch.equal(v[0][1], ids) for v in variants)  # integer tensors stay
    assert not bool(evaluate_mod._differs([x], [x.clone()]))
    assert bool(evaluate_mod._differs([x], [variants[1][0][0]]))
    assert evaluate_mod._differs([x], [x[:1]]) is True


def test_single_file_candidates_are_not_run_twice(tmp_path):
    capture = make_capture("rmsnorm", tmp_path / "rms.pt", "cpu")
    plain = tmp_path / "plain.py"
    plain.write_text(ENTRY.format(body="", head=""))
    with restored_globals():
        result = evaluate(capture, plain, device="cpu")
    assert result["status"] == "ok" and "determinism" not in result


def test_bitwise_diff_sees_nan_payloads_and_shapes():
    a = torch.tensor([1.0, float("nan"), 3.0])
    assert evaluate_mod._bitwise_diff(a, a.clone(), "x") is None  # NaN == NaN bit for bit
    b = a.clone()
    b[2] = torch.nextafter(b[2], torch.tensor(9.0))
    assert evaluate_mod._bitwise_diff(a, b, "x") == "x: 1 of 3 elements differ"
    assert "x.k[1]" in evaluate_mod._bitwise_diff({"k": [a, a]}, {"k": [a, b]}, "x")
    assert "(3,)" in evaluate_mod._bitwise_diff(a, a[:2], "x")


# ------------------------------------------------------------------ the watchdog's hang


HANG = {"code": "hang", "instr": 7, "op": "layer3", "tile": 5, "queue": 2, "counter": 4,
        "value": 63, "target": 64, "watchdog_ms": 2000.0}  # fmt: skip


def test_a_megakernel_hang_is_recorded_as_status_hang(tmp_path):
    capture = make_capture("rmsnorm", tmp_path / "rms.pt", "cpu")
    hung = native(
        tmp_path / "hung",
        "hung",
        head="from kernel_agent.native.megakernel import runtime",
        body=f"raise runtime.MegakernelHang({HANG!r})",
    )
    with restored_globals():
        result = evaluate(capture, hung, device="cpu")
    assert result["status"] == "hang" and not result["correct"], result
    assert result["hang"]["instr"] == 7 and result["hang"]["counter"] == 4
    assert result["error"].startswith(
        "megakernel hang: instruction 7 (layer3[5]) on queue 2 waited longer than 2000.0 ms on "
        "counter 4 (63 of 64)"
    )
    steady = native(tmp_path / "steady", "steady2")
    with restored_globals():  # the next candidate's failures are its own
        assert evaluate(capture, steady, device="cpu")["status"] == "ok"
    assert runtime.hangs() == []


def test_the_runtime_decodes_the_watchdogs_status_words():
    sched = mks.build([mks.Op("a", 2, 3), mks.Op("b", 2, 3)], [mks.Edge("a", "b")], queues=2)
    rt = object.__new__(runtime.Runtime)  # its buffers without a GPU
    rt.schedule, rt.watchdog_ms = sched, 50.0
    rt.status = torch.zeros(runtime.STATUS_WORDS, dtype=torch.int32)
    assert rt.failure() is None
    rt.check()
    victim = next(i for i in sched.instrs if i.op == "b")
    words = {
        runtime.S_INSTR: victim.id,
        runtime.S_QUEUE: victim.queue,
        runtime.S_COUNTER: 0,
        runtime.S_VALUE: 2,
        runtime.S_TARGET: 3,
        runtime.S_OPCODE: 2,
        runtime.S_CODE: 1,
    }
    for k, v in words.items():
        rt.status[k] = v
    info = rt.failure()
    assert info["code"] == "hang" and (info["op"], info["tile"]) == ("b", victim.tile)
    with pytest.raises(runtime.MegakernelHang, match=r"waited longer than 50\.0 ms on counter 0"):
        rt.check()
    runtime.forget()
    runtime._LIVE.add(rt)
    assert runtime.hangs() == [info]
    rt.status[runtime.S_CODE] = 4
    assert "page pool holds 3" in runtime.describe(rt.failure())
    runtime.forget()


def test_the_watchdog_limit_comes_from_the_argument_the_environment_or_the_default(
    monkeypatch,
):
    monkeypatch.delenv(runtime.WATCHDOG_ENV, raising=False)
    assert runtime.watchdog_limit() == runtime.DEFAULT_WATCHDOG_MS
    monkeypatch.setenv(runtime.WATCHDOG_ENV, "600000")
    assert runtime.watchdog_limit() == 600000.0 and runtime.watchdog_limit(5) == 5.0
    assert memcheck.ENV[runtime.WATCHDOG_ENV] == "600000"  # sanitized runs are ~100x slower


# ------------------------------------------------------------------ the native digest


def test_the_native_digest_points_to_the_kit_for_grid_syncs_and_many_launches(tmp_path):
    from kernel_agent.native import engine
    from kernel_agent.workspace import RunDir, append_jsonl

    run = RunDir(tmp_path / "run")
    target = engine.TARGET_PREFIX + "stack"
    history = run.history_dir(target)
    history.mkdir(parents=True)
    root = tmp_path / "stack"
    native(root, "stack")
    (root / "csrc").mkdir()
    (root / "csrc" / "layer.cu").write_text("// one phase per layer\ncg::this_grid().sync();\n")
    best = proj.write_bundle(root, history).rename(history / "002_stack_0badc0de.py")
    append_jsonl(run.results_file(target), {"snapshot": best.name, "kernel_launches_candidate": 4})
    rows = [
        {"target": target, "snapshot": "001_stack_00000000.py", "correct": True, "speedup": 1.5},
        {"target": target, "snapshot": best.name, "correct": True, "speedup": 2.0},
        {"target": target, "snapshot": "003_stack_11111111.py", "correct": False, "speedup": 9.0},
        {"target": "attention", "snapshot": "x.py", "correct": True, "speedup": 3.0},
    ]
    lines = engine.megakernel_hint(run, rows)
    text = "\n".join(lines)
    assert lines[:2] == ["", "## Megakernel kit"]
    assert (
        f"`{target}`: its best kernel `{best.name}` synchronises its whole grid and launches 4 "
        "kernels per call." in text
    )
    assert "megakernel.md" in text and "examples/native_megakernel" in text
    (root / "csrc" / "layer.cu").write_text("// counters, one launch\n")
    calm = proj.write_bundle(root, history).rename(history / "004_stack_22222222.py")
    append_jsonl(run.results_file(target), {"snapshot": calm.name, "kernel_launches_candidate": 1})
    rows.append({"target": target, "snapshot": calm.name, "correct": True, "speedup": 2.5})
    assert engine.megakernel_hint(run, rows) == []
