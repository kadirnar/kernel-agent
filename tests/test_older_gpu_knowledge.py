"""#259: knowledge for Turing, Ampere and Ada. The family files are linked from the prompt's
section, the faked T4 prompt carries the fp16 advice, no skill claims that Triton's int8
`tl.dot` runs on FMA below sm_80 (it does not compile for sm_75), the sources are fetchable,
and the Triton facts the files state still hold for the installed Triton. CPU only."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kernel_agent import gpu_arch, precisions, skills, toolchain
from kernel_agent.agent import prompts, web

#: phrases that put a dot on CUDA cores
FMA_WORDS = r"\b(to|on|is|are|becomes?) FMA\b|falls? back|no tensor cores|CUDA cores"
FAMILY_FILES = {"pre_ampere": "turing.md", "ampere": "ampere.md", "ada": "ada.md"}
SKILL_DIR = gpu_arch.KNOWLEDGE.parent


def _statements(text: str) -> list[str]:
    """Bullets, table rows and paragraphs of a markdown text, whitespace-normalised."""
    out: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        starts = stripped.startswith(("* ", "- ", "|", "#")) or re.match(r"\d+\. ", stripped)
        if not stripped or starts:
            if current:
                out.append(" ".join(current))
            current = []
        if stripped.startswith(("|", "#")):
            out.append(stripped)
        elif stripped:
            current.append(stripped)
    if current:
        out.append(" ".join(current))
    return out


def _int8_dot_claims(text: str) -> list[str]:
    """Statements that put Triton's `tl.dot` on an old GPU (sm_75 / Turing / below sm_80) on
    FMA or without tensor cores, for int8 or without naming the dtype it holds for."""
    bad = []
    for s in _statements(text):
        if "tl.dot" not in s or not re.search(r"sm_75|Turing|below sm_80", s):
            continue
        int8 = re.search(r"\bint8\b|INT8|\bs8\b|IMMA", s)
        fma = re.search(FMA_WORDS, s)
        if not fma:
            continue
        if int8 and not re.search(r"(does not|doesn't|fails? to) compile", s):
            bad.append(s)  # int8 next to "FMA" without saying that it does not compile
        dtype = r"(fp16|bf16|fp32)[^.]{0,40}tl\.dot|tl\.dot`? on (fp16|bf16|fp32)"
        if not int8 and not re.search(dtype, s):
            bad.append(s)  # "tl.dot is FMA below sm_80" for every dtype, int8 included
    return bad


def test_no_skill_says_triton_int8_dot_runs_on_fma_below_sm80():
    found = {
        str(path.relative_to(skills.SKILLS_DIR)): claims
        for path in sorted(skills.SKILLS_DIR.rglob("*.md"))
        if (claims := _int8_dot_claims(path.read_text()))
    }
    assert found == {}
    # the check catches the wording the audit found (gpus.md and int8-w8a8 before #259)
    old = (
        "* Triton's `tl.dot` falls back to FMA (CUDA cores) below sm_80: use Triton for\n"
        "  memory-bound glue, cuBLAS or CUDA C++ fp16 `mma.sync` for GEMMs.\n"
        "* INT8 tensor cores on sm_75: `int8_w8a8` runs there through cuBLASLt or CUDA C++\n"
        "  (Triton's `tl.dot` has no tensor cores below sm_80).\n"
        "| Turing sm_75 | `mma.sync` m8n8k16 s8; Triton `tl.dot` has no tensor cores below "
        "sm_80 |\n"
    )
    assert len(_int8_dot_claims(old)) == 3


@pytest.mark.parametrize(("key", "name"), FAMILY_FILES.items())
def test_family_sections_link_their_files(key, name):
    section = gpu_arch.knowledge_section(key)
    assert f"]({name})" in section, key
    text = (SKILL_DIR / name).read_text()
    assert "](measure-first.md)" in text and "## Not measured on this family yet" in text
    assert "RTX 5070 Ti (sm_120, measured)" in text  # the measured row of the rate table
    assert f"]({name})" in skills.get("gpu-architectures").path.read_text()


def _t4_prompt() -> str:
    gpu = toolchain.GPUInfo("Tesla T4", (7, 5), 15.0, 40, 4.0, 64.0, 65.0)
    backends = {"cuda": True, "triton": True, "cute": False, "nvrtc": True, "tilelang": True}
    tc = toolchain.Toolchain(gpu, "2.14", "13.0", None, "13.0", backends, [], {}, None)
    return prompts.planner_prompt(
        {"repo_id": "org/model", "modality": "llm"},
        {"median_ms": 1.0},
        "# Profile",
        ["cuda", "triton"],
        3,
        "py",
        tc.summary(),
        quality="near-lossless",
        precisions=precisions.allowed("near-lossless", None, (7, 5)),
    )


def test_the_t4_prompt_has_the_fp16_advice_and_the_turing_file():
    text = _t4_prompt()
    assert "# This GPU: Tesla T4 (sm_75, Volta / Turing)" in text
    assert "Run 16-bit work in fp16 here" in text
    assert "int8 `tl.dot` does not compile for sm_75" in text
    links = re.findall(r"\]\((/[^)]+\.md)\)", text)  # absolute: a session reads them as given
    assert str(SKILL_DIR / "turing.md") in links and all(Path(p).is_file() for p in links)
    assert "ampere.md" not in text and "ada.md" not in text  # only this family's file


def test_prompt_links_stay_relative_in_the_skill_and_unknown_ones_untouched():
    assert gpu_arch._absolute_links("[a](turing.md) [b](nothing.md)") == (
        f"[a]({SKILL_DIR / 'turing.md'}) [b](nothing.md)"
    )


def test_new_files_cite_fetchable_sources():
    files = [SKILL_DIR / n for n in (*FAMILY_FILES.values(), "measure-first.md")]
    for path in files:
        for url in re.findall(r"https://[^\s)>,]+", path.read_text()):
            assert web.allowed(url, web.domains()), (path.name, url)


def _compile(fn, signature: dict, cc: int) -> str:
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    attrs = {(i,): [["tt.divisibility", 16]] for i in range(3)}  # 16-byte aligned pointers
    source = ASTSource(fn=fn, signature=signature, constexprs={"N": 32}, attrs=attrs)
    compiled = triton.compile(source, target=GPUTarget("cuda", cc, 32), options={"num_warps": 4})
    return compiled.asm["ptx"]


def test_the_triton_facts_of_the_turing_file_hold():
    """turing.md / sm75-sm89.md, Triton 3.8: fp16 `tl.dot` is FMA on sm_75 and HMMA on sm_80;
    int8 does not compile for sm_75; e4m3 is refused below sm_89. A Triton upgrade that
    changes one of them fails here: update the files with the new behaviour."""
    triton = pytest.importorskip("triton")
    import triton.language as tl

    @triton.jit
    def dot(a, b, c, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(a + r[:, None] * N + r[None, :])
        y = tl.load(b + r[:, None] * N + r[None, :])
        tl.store(c + r[:, None] * N + r[None, :], tl.dot(x, y))

    fp16 = {"a": "*fp16", "b": "*fp16", "c": "*fp32", "N": "constexpr"}
    assert "mma.sync" not in _compile(dot, fp16, 75)
    assert "mma.sync.aligned.m16n8k16" in _compile(dot, fp16, 80)
    with pytest.raises(Exception, match="PassManager::run failed"):
        _compile(dot, {**fp16, "a": "*i8", "b": "*i8", "c": "*i32"}, 75)
    with pytest.raises(Exception, match="fp8e4nv not supported"):
        _compile(dot, {**fp16, "a": "*fp8e4nv", "b": "*fp8e4nv"}, 86)
