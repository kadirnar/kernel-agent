"""The CPU compile matrix (#256, ``tests/arch_matrix.py``): every bundled example and
kernel-agent's own device code, compiled without a GPU for sm_75 / sm_80 / sm_86 / sm_89 /
sm_120, compiles exactly on the GPUs it declares (``ARCHS``, or ``ARCHS_COMPILES`` where a
limit is runtime-only); Triton's tensor-core forms, ``cp.async`` pipelining and shared
memory per arch. Verdicts are cached by content (a cold run compiles for minutes, a warm one
reads them back in a second); a missing compiler skips its rows with the reason."""

from __future__ import annotations

import ast
from pathlib import Path

import arch_matrix as am
import pytest

from kernel_agent import gpu_arch

GROUPS, LEFT_OUT = am.collect()


@pytest.fixture(scope="session")
def matrix() -> am.Matrix:
    return am.build(GROUPS)


def _skipped(matrix: am.Matrix, group: am.Group) -> set[str]:
    return {
        r.skipped
        for u in group.units
        for arch in am.MATRIX
        if u.applies(arch) and (r := matrix.results[(group.name, u.name, arch)]).skipped
    }


# ------------------------------------------------------------------ what the matrix covers


def test_every_bundled_example_is_in_the_matrix_or_says_why_not():
    names = {p.name for p in am.examples()}
    examples = {g.name for g in GROUPS if g.example}
    assert examples | set(LEFT_OUT) == names
    assert not examples & set(LEFT_OUT)
    for table in (am.TRITON, am.CUTE, am.TILELANG, am.OUT_OF_MATRIX):
        assert set(table) <= names, set(table) - names  # no entry outlives its example
    for group in GROUPS:
        assert group.units, (
            f"{group.name}: nothing to compile. A CUDA example needs a literal CUDA_SRC, an "
            "NVRTC one SRC; a Triton, CuTe DSL or TileLang one an entry in tests/arch_matrix.py "
            "(TRITON / CUTE / TILELANG); else list it in OUT_OF_MATRIX with the reason"
        )
        names_ = [u.name for u in group.units]
        assert len(set(names_)) == len(names_), group.name
    # the Hopper / datacenter Blackwell templates name no GPU of the matrix
    assert "no GPU of the matrix" in LEFT_OUT["cute_sm90_gemm_ws.py"]
    assert "no GPU of the matrix" in LEFT_OUT["cute_sm100_gemm_tcgen05.py"]


def _launched_kernels(path: Path) -> dict[str, list[str]]:
    """``@triton.jit`` functions a file launches -> their parameters (helpers that other
    ``@triton.jit`` functions call are compiled with them)."""
    tree = ast.parse(path.read_text())
    jit = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d).split("(")[0] in ("triton.jit", "jit") for d in node.decorator_list)
    }
    called = {
        call.func.id
        for fn in jit.values()
        for call in ast.walk(fn)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    return {name: [a.arg for a in fn.args.args] for name, fn in jit.items() if name not in called}


@pytest.mark.parametrize(
    ("path", "specs"),
    [(am.EXAMPLES / name, specs) for name, specs in am.TRITON.items()]
    + [(am.PACKAGE / rel, specs) for rel, specs in am.LIBRARY_TRITON.items()],
    ids=[*am.TRITON, *am.LIBRARY_TRITON],
)
def test_triton_specs_match_the_kernels(path, specs):
    kernels = _launched_kernels(path)
    assert set(kernels) == {s.kernel for s in specs}, (
        f"{path.name}: every launched @triton.jit kernel needs a TritonSpec in "
        f"tests/arch_matrix.py (as its launches specialise it); kernels {sorted(kernels)}"
    )
    for spec in specs:
        # the same check the worker makes: every parameter typed or constexpr, nothing else
        am.triton_signature(kernels[spec.kernel], am._args(spec.args), spec.consts)
        for where in [spec.where or "", *(w for w, _ in spec.mma)]:
            gpu_arch.supports(where, (12, 0))  # raises on a malformed declaration


def test_every_example_runs_only_where_it_compiles():
    for group in GROUPS:
        if not group.example:
            continue
        for arch, runs in am._expected(group.runs).items():
            assert group.expected[arch] or not runs, (
                f"{group.name}: ARCHS = {group.runs!r} includes {arch}, its {group.declared} "
                "does not"
            )


# ------------------------------------------------------------------ the matrix


@pytest.mark.parametrize("name", [g.name for g in GROUPS])
def test_compiles_exactly_where_declared(matrix, name):
    group = matrix.group(name)
    if skipped := _skipped(matrix, group):
        pytest.skip("; ".join(sorted(skipped)))
    bad = matrix.mismatches(group)
    assert not bad, "\n".join(bad)


def _triton_results(matrix: am.Matrix):
    """``(group, unit, arch, result)`` of every Triton specialisation that compiled."""
    rows = []
    for group in matrix.groups:
        for unit in group.units:
            for arch in am.MATRIX:
                result = matrix.results.get((group.name, unit.name, arch))
                if unit.spec is not None and result is not None and result.ok:
                    rows.append((group, unit, arch, result))
    if not rows:
        pytest.skip("triton is not installed")
    return rows


def test_triton_tensor_core_forms_per_arch(matrix):
    """bf16 / fp16 ``tl.dot`` has no MMA on sm_75 (FMA; ``backends.ARCH_RULES``) and
    m16n8k16 from sm_80; int8 is s8 m16n8k32 from sm_80; e4m3 is m16n8k32 on sm_89 and the
    block-scaled MMA on sm_12x."""
    problems, checked = [], set()
    for group, unit, arch, result in _triton_results(matrix):
        assert unit.spec is not None
        cap = am.CAPS[arch]
        form = next((f for where, f in unit.spec.mma if gpu_arch.supports(where, cap)), None)
        if form is None:
            continue
        checked.add((arch, form))
        if form == "" and result.mma:
            problems.append(
                f"{group.name} {unit.name} {arch}: mma.sync {result.mma}, none expected"
            )
        elif form and not any(form in m for m in result.mma):
            problems.append(f"{group.name} {unit.name} {arch}: {result.mma}, expected {form}")
    assert not problems, "\n".join(problems)
    forms = {f for _, f in checked}
    assert "" in forms and "m16n8k32.row.col.satfinite.s32.s8.s8.s32" in forms
    assert ("sm_89", "m16n8k32.row.col.f32.e4m3.e4m3.f32") in checked


def test_triton_k_loops_are_pipelined_from_sm80(matrix):
    """The W8A8 GEMMs' K loops load their tiles with ``cp.async`` (multi-stage) wherever
    they compile on sm_80+, with the JIT's 16-byte specialisation."""
    pipelined = [
        (group, unit, arch, result)
        for group, unit, arch, result in _triton_results(matrix)
        if unit.spec is not None and unit.spec.pipelined and am.CAPS[arch] >= (8, 0)
    ]
    assert {g.name for g, *_ in pipelined} >= {
        "triton_int8_w8a8_gemm.py",
        "triton_fp8_w8a8_gemm.py",
    }
    for group, unit, arch, result in pipelined:
        assert result.cp_async, f"{group.name} {unit.name} {arch}: no cp.async in its PTX"


def test_triton_shared_memory_fits_every_arch(matrix):
    for group, unit, arch, result in _triton_results(matrix):
        limit = gpu_arch.SMEM_PER_BLOCK_KB[am.CAPS[arch]] * 1024
        assert result.shared <= limit, f"{group.name} {unit.name} {arch}: {result.shared} B"


# ------------------------------------------------------------------ the machinery (no compile)


def _fake(expected: dict[str, bool], got: dict[str, bool], example: bool = True) -> am.Matrix:
    unit = am.Unit("kernel", "nvcc", {})
    group = am.Group("ex.py", "ARCHS = 'sm_80+'", expected, [unit], example, "sm_80+")
    results = {
        ("ex.py", "kernel", arch): am.Result(ok, "" if ok else "ptxas error: needs sm_80")
        for arch, ok in got.items()
    }
    return am.Matrix([group], {}, results)


def test_a_declaration_that_is_too_narrow_or_too_wide_says_how_to_fix_it():
    expected = am._expected("sm_80+")
    agree = _fake(expected, expected)
    assert agree.mismatches(agree.groups[0]) == []
    assert "| ex.py | ARCHS = 'sm_80+' | x | C | C | C | C |" in agree.table()
    wider = _fake(expected, dict.fromkeys(am.MATRIX, True))
    (msg,) = wider.mismatches(wider.groups[0])
    assert msg.startswith("ex.py compiles for sm_75 but ARCHS = 'sm_80+' excludes it")
    assert "widen ARCHS" in msg and "ARCHS_COMPILES" in msg
    assert "| C! | C | C | C | C |" in wider.table()
    narrower = _fake(expected, {**expected, "sm_86": False})
    (msg,) = narrower.mismatches(narrower.groups[0])
    assert "ARCHS = 'sm_80+' includes sm_86 but it does not compile there" in msg
    assert "ptxas error: needs sm_80" in msg and "narrow the declaration or fix the code" in msg
    assert narrower.failures()[-1] == "ex.py sm_86: kernel: ptxas error: needs sm_80"


def test_a_missing_compiler_is_a_reason_not_a_failure(monkeypatch, tmp_path):
    tools = am.Tools(None, {}, {})
    assert "no nvcc" in str(tools.missing("nvcc")) and "no nvcc" in str(tools.missing("tilelang"))
    monkeypatch.setattr(am.Tools, "installed", staticmethod(lambda module: False))
    assert tools.missing("cute") == "cutlass is not installed"
    assert tools.missing("triton") == "triton is not installed"
    unit = am.Unit("kernel", "cute", {"path": "x", "code": ""})
    group = am.Group("cute_x.py", "no ARCHS (every GPU)", am._expected(None), [unit], True)
    monkeypatch.setattr(am.Tools, "find", classmethod(lambda cls: tools))
    built = am.build([group], am.Cache(tmp_path))
    assert all(r.skipped == "cutlass is not installed" for r in built.results.values())
    assert built.verdict(group, "sm_80") is None and built.mismatches(group) == []
    assert _skipped(built, group) == {"cutlass is not installed"}


def test_verdicts_are_cached_by_content(tmp_path):
    tools = am.Tools("/usr/bin/nvcc", {}, {"torch": "2.14", "nvcc": "13.4"})
    unit = am.Unit("kernel", "nvcc", {"kind": "inline", "source": "a", "flags": []})
    key = am.key_of(unit, "sm_80", tools)
    assert key != am.key_of(unit, "sm_86", tools)
    other = am.Unit("kernel", "nvcc", {"kind": "inline", "source": "b", "flags": []})
    assert key != am.key_of(other, "sm_80", tools)
    newer = am.Tools("/usr/bin/nvcc", {}, {"torch": "2.14", "nvcc": "13.5"})
    assert key != am.key_of(unit, "sm_80", newer)
    cache = am.Cache(tmp_path)
    assert cache.get(key) is None
    with cache.lock(key):
        cache.put(key, am.Result(False, "error: x"))
        cache.put("t", am.Result(False, "timed out", cacheable=False))
    assert cache.get(key) == am.Result(False, "error: x") and cache.get("t") is None
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{key}.json"]  # no lock left
    assert am.Cache(None).get(key) is None


def test_example_compiles_reads_archs_compiles(tmp_path):
    path = tmp_path / "cuda_x.py"
    path.write_text('ARCHS = "sm_90+"\nARCHS_WHY = "griddepcontrol"\n')
    assert gpu_arch.example_compiles(path) == "sm_90+"
    path.write_text(path.read_text() + 'ARCHS_COMPILES = "sm_75+"\n')
    assert gpu_arch.example_compiles(path) == "sm_75+"
    assert gpu_arch.example_requirement(path) == ("sm_90+", "griddepcontrol")
    assert gpu_arch.example_compiles(tmp_path / "missing.py") is None
