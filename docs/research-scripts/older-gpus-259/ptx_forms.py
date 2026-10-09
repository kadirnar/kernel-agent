"""Which PTX instruction forms compile for which arch (issue #259). CPU only (NVRTC, no GPU).

Each form is compiled alone in a tiny kernel with NVRTC (``cuda.core``) to a cubin for
sm_75, sm_80, sm_86, sm_89 and sm_120; "OK" means ptxas accepted it for that arch, otherwise
the first error line. Run:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src python docs/research-scripts/older-gpus-259/ptx_forms.py \
        > docs/research-scripts/older-gpus-259/ptx_forms.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from cuda.bindings import nvrtc  # noqa: E402
from cuda.core import Program, ProgramOptions  # noqa: E402

from kernel_agent.toolchain import cuda_include_dirs  # noqa: E402

ARCHS = ("sm_75", "sm_80", "sm_86", "sm_89", "sm_120")
F4 = '"+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)'
H2 = '"+r"(h0), "+r"(h1)'
I2 = '"+r"(i0), "+r"(i1)'
I4 = '"+r"(i0), "+r"(i1), "+r"(i2), "+r"(i3)'


def mma(ptx: str, out: str, n_out: int, n_a: int, n_b: int) -> str:
    regs, k = [], 0
    for n in (n_out, n_a, n_b):
        regs.append("{" + ",".join(f"%{k + j}" for j in range(n)) + "}")
        k += n
    regs.append("{" + ",".join(f"%{j}" for j in range(n_out)) + "}")
    ins = ", ".join(['"r"(a0)', '"r"(a1)', '"r"(a2)', '"r"(a3)'][:n_a])
    ins += ", " + ", ".join(['"r"(b0)', '"r"(b1)'][:n_b])
    return f'asm volatile("{ptx} {",".join(regs)};" : {out} : {ins});'


#: name -> (minimum arch per the PTX ISA, body)
FORMS = {
    # --- fp16 / bf16 / tf32 HMMA
    "mma m8n8k4 f32.f16 (Volta form)": (
        70,
        'asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 '
        '{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};" : '
        '"+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(e0), "+f"(e1), "+f"(e2), "+f"(e3) '
        ': "r"(a0), "r"(a1), "r"(b0), "r"(b1));',
    ),
    "mma m16n8k8 f32.f16.f16.f32 (Turing HMMA)": (
        75,
        mma("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32", F4, 4, 2, 1),
    ),
    "mma m16n8k8 f16.f16.f16.f16 (Turing HMMA, fp16 acc)": (
        75,
        mma("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16", H2, 2, 2, 1),
    ),
    "mma m16n8k16 f32.f16.f16.f32": (
        80,
        mma("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", F4, 4, 4, 2),
    ),
    "mma m16n8k16 f16.f16.f16.f16 (fp16 acc)": (
        80,
        mma("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", H2, 2, 4, 2),
    ),
    "mma m16n8k16 f32.bf16.bf16.f32": (
        80,
        mma("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", F4, 4, 4, 2),
    ),
    "mma m16n8k8 f32.bf16.bf16.f32": (
        80,
        mma("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32", F4, 4, 2, 1),
    ),
    "mma m16n8k8 f32.tf32.tf32.f32": (
        80,
        mma("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", F4, 4, 4, 2),
    ),
    # --- integer IMMA
    "mma m8n8k16 s32.s8.s8.s32 (Turing IMMA)": (
        75,
        mma("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32", I2, 2, 1, 1),
    ),
    "mma m8n8k32 s32.s4.s4.s32 (Turing IMMA int4)": (
        75,
        mma("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32", I2, 2, 1, 1),
    ),
    "mma m16n8k16 s32.s8.s8.s32": (
        80,
        mma("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32", I4, 4, 2, 1),
    ),
    "mma m16n8k32 s32.s8.s8.s32": (
        80,
        mma("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", I4, 4, 4, 2),
    ),
    "mma m16n8k64 s32.s4.s4.s32": (
        80,
        mma("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32", I4, 4, 4, 2),
    ),
    # --- FP8 QMMA
    "mma m16n8k32 f32.e4m3.e4m3.f32": (
        89,
        mma("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", F4, 4, 4, 2),
    ),
    "mma m16n8k32 f32.e5m2.e5m2.f32": (
        89,
        mma("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32", F4, 4, 4, 2),
    ),
    "mma m16n8k32 f16.e4m3.e4m3.f16 (fp16 acc)": (
        89,
        mma("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", H2, 2, 4, 2),
    ),
    # --- data movement
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16": (
        75,
        'asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : '
        '"=r"(i0), "=r"(i1), "=r"(i2), "=r"(i3) : "r"(saddr));',
    ),
    "ldmatrix .trans": (
        75,
        'asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : '
        '"=r"(i0), "=r"(i1) : "r"(saddr));',
    ),
    "movmatrix.sync.aligned.m8n8.trans.b16": (
        75,
        'asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(i0) : "r"(a0));',
    ),
    "cp.async.cg.shared.global 16 B + commit/wait_group": (
        80,
        'asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(saddr), "l"(g));'
        'asm volatile("cp.async.commit_group;");'
        'asm volatile("cp.async.wait_group 0;");',
    ),
    "cp.async.ca.shared.global 4 B": (
        80,
        'asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(saddr), "l"(g));',
    ),
    "mbarrier.init.shared.b64": (
        80,
        'asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(saddr));',
    ),
    "cp.async.bulk.shared::cluster.global (TMA bulk copy)": (
        90,
        'asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes '
        '[%0], [%1], 16, [%2];" :: "r"(saddr), "l"(g), "r"(saddr));',
    ),
    "griddepcontrol.wait (PDL)": (90, 'asm volatile("griddepcontrol.wait;" ::: "memory");'),
    # --- conversions and misc
    "cvt.rn.f16x2.e4m3x2 (hardware e4m3 -> fp16)": (
        89,
        'unsigned short c = (unsigned short)a0; '
        'asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(i0) : "h"(c));',
    ),
    "cvt.rn.satfinite.e4m3x2.f32 (hardware fp32 -> e4m3)": (
        89,
        'unsigned short c; asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(c) '
        ': "f"(d0), "f"(d1)); i0 = c;',
    ),
    "cvt.rn.bf16x2.f32": (
        80,
        'asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(i0) : "f"(d0), "f"(d1));',
    ),
    "fma.rn.bf16x2": (
        80,
        'asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(i0) : "r"(a0), "r"(a1), "r"(a2));',
    ),
    "redux.sync.add.s32": (
        80,
        'asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(i0) : "r"(a0));',
    ),
}

#: CUDA C++ API forms (headers), name -> (minimum arch per the CUDA Programming Guide, body)
API = {
    "wmma 16x16x16 half -> float": (
        70,
        "wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa; "
        "wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb; "
        "wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc; wmma::fill_fragment(fc, 0.f); "
        "wmma::load_matrix_sync(fa, (const half*)g, 16); "
        "wmma::load_matrix_sync(fb, (const half*)g, 16); wmma::mma_sync(fc, fa, fb, fc); "
        "wmma::store_matrix_sync((float*)g, fc, 16, wmma::mem_row_major);",
    ),
    "wmma 16x16x16 signed char -> int": (
        72,
        "wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa; "
        "wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb; "
        "wmma::fragment<wmma::accumulator, 16, 16, 16, int> fc; wmma::fill_fragment(fc, 0); "
        "wmma::load_matrix_sync(fa, (const signed char*)g, 16); "
        "wmma::load_matrix_sync(fb, (const signed char*)g, 16); wmma::mma_sync(fc, fa, fb, fc); "
        "wmma::store_matrix_sync((int*)g, fc, 16, wmma::mem_row_major);",
    ),
    "wmma 16x16x16 __nv_bfloat16 -> float": (
        80,
        "wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa; "
        "wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb; "
        "wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc; wmma::fill_fragment(fc, 0.f); "
        "wmma::load_matrix_sync(fa, (const __nv_bfloat16*)g, 16); "
        "wmma::load_matrix_sync(fb, (const __nv_bfloat16*)g, 16); wmma::mma_sync(fc, fa, fb, fc); "
        "wmma::store_matrix_sync((float*)g, fc, 16, wmma::mem_row_major);",
    ),
    "__hfma on __nv_bfloat16 (cuda_bf16.h)": (
        0,  # the header emulates it through fp32 below sm_80
        "__nv_bfloat16 x = __float2bfloat16(d0); x = __hfma(x, x, x); d0 = __bfloat162float(x);",
    ),
    "__nv_cvt_fp8x2_to_halfraw2 e4m3 (cuda_fp8.h)": (
        0,  # software conversion below sm_89 (see ptx_jit_run.py for its result)
        "__half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)a0, __NV_E4M3); "
        "i0 = *(unsigned int*)&h;",
    ),
}

PRELUDE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void k(float* out, const float* g, unsigned int seed) {
  __shared__ __align__(128) unsigned char smem[1024];
  unsigned int saddr = (unsigned int)__cvta_generic_to_shared(smem);
  unsigned int a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u;
  unsigned int b0 = a0 * 11u, b1 = a0 * 13u;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f, e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
  unsigned int h0 = 0u, h1 = 0u;
  int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
  smem[threadIdx.x] = (unsigned char)a0;
  __syncthreads();
"""
EPILOGUE = r"""
  out[threadIdx.x] = d0 + d1 + d2 + d3 + e0 + e1 + e2 + e3 + (float)(h0 + h1)
                     + (float)(i0 + i1 + i2 + i3) + (float)smem[threadIdx.x + 1];
}
"""


def compile_for(body: str, arch: str) -> str:
    src = PRELUDE + "  " + body + "\n" + EPILOGUE
    opts = ProgramOptions(arch=arch, std="c++17", include_path=cuda_include_dirs())
    try:
        Program(src, code_type="c++", options=opts).compile("cubin")
        return "OK"
    except Exception as exc:
        text = str(exc)
        found = re.findall(r"(?:error|fatal)[^\n]*", text)
        return "FAIL " + (found[0] if found else text.replace("\n", " "))[:160]


def main() -> None:
    err, major, minor = nvrtc.nvrtcVersion()
    print(f"# NVRTC {major}.{minor} (cuda.core); cubin per arch; 'OK' = accepted by ptxas")
    for arch in ("sm_70", "sm_72", "sm_75"):  # the oldest target this CUDA accepts
        try:
            options = ProgramOptions(arch=arch, std="c++17")
            Program('extern "C" __global__ void k(float* x) { x[0] = 1.f; }', code_type="c++", options=options).compile("cubin")
            print(f"target {arch}: accepted")
        except Exception as exc:
            print(f"target {arch}: rejected ({' '.join(str(exc).split())[-60:]})")
    print(f"{'form':58s} {'min':>5s}  " + "  ".join(f"{a:7s}" for a in ARCHS))
    mismatches = []
    for table in (FORMS, API):
        for name, (minimum, body) in table.items():
            results = {arch: compile_for(body, arch) for arch in ARCHS}
            cells = "  ".join(f"{'OK' if r == 'OK' else 'fail':7s}" for r in results.values())
            print(f"{name:58s} {'sm_' + str(minimum) if minimum else '-':>5s}  {cells}")
            for arch, r in results.items():
                cc = int(arch.split("_")[1])
                if minimum and (r == "OK") != (cc >= minimum):
                    mismatches.append(f"{name} {arch}: {r}")
                if r != "OK":
                    print(f"    {arch}: {r}")
    print("mismatches with the documented minimum arch:", mismatches or "none")


if __name__ == "__main__":
    main()
