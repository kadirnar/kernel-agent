// Tiny kernels whose SASS the opcode census (kernel_agent/kernels/sass.py) is checked on.
// make_fixtures.py compiles this file on the CPU for each architecture (nvcc -cubin,
// KA_SM = the compute capability) and writes the `cuobjdump -sass` text next to it: the
// opcode names in sass.py's tables are the ones these cubins contain. The kernels need not
// compute anything useful; each one issues the instructions its name says.

#include <cuda.h>  // CUtensorMap
#include <cuda_fp16.h>

// Global loads and stores of two widths, shared memory, a barrier, a shuffle, fp32 math and
// atomics on global and shared memory.
extern "C" __global__ void ka_ldst(const float4* x, const float* s, float4* y, float* t) {
  __shared__ float buf[256];
  __shared__ float total;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 v = x[i];
  buf[threadIdx.x] = s[i];
  __syncthreads();
  float w = fmaf(buf[(threadIdx.x + 1) & 255], v.x, v.y * v.z) + v.w;
  w += __shfl_xor_sync(0xffffffffu, w, 1);
  atomicAdd(&total, w);
  __syncthreads();
  y[i] = make_float4(w, v.y, v.z, total);
  atomicAdd(t, w);
  t[i + 1] = atomicMax(reinterpret_cast<int*>(t) + 2, i);  // an atomic that returns a value
}

// Packed fp16 math.
extern "C" __global__ void ka_half2(const __half2* a, const __half2* b, __half2* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __half2 x = a[i], y = b[i];
  c[i] = __hfma2(__hadd2(x, y), __hmul2(x, y), y);
}

// Register spills: 48 values live across a loop under a 32-register cap.
extern "C" __global__ void __maxnreg__(32) ka_spill(float* x, int n) {
  float v[48];
#pragma unroll
  for (int k = 0; k < 48; ++k) v[k] = x[k * 256 + threadIdx.x];
#pragma unroll 1
  for (int it = 0; it < n; ++it) {
#pragma unroll
    for (int k = 0; k < 48; ++k) v[k] = v[k] * v[(k + 1) % 48] + 1.0f;
  }
#pragma unroll
  for (int k = 0; k < 48; ++k) x[k * 256 + threadIdx.x] = v[k];
}

// Local memory: a per-thread array indexed at run time (what spills look like: LDL / STL).
extern "C" __global__ void ka_local(float* out, const int* idx) {
  float a[64];
#pragma unroll 1
  for (int k = 0; k < 64; ++k) a[k] = out[k * blockDim.x + threadIdx.x];
  out[threadIdx.x] = a[idx[threadIdx.x] & 63];
}

#if KA_SM < 80
// Turing's tensor-core forms, the only mma.sync shapes below sm_80: fp16 m16n8k8 (HMMA) and
// s8 m8n8k16 (IMMA); no bf16, no m16n8k16 / m16n8k32, no cp.async.
extern "C" __global__ void ka_mma_turing(const unsigned* g, float* out, int* iout) {
  unsigned a0 = g[threadIdx.x], a1 = a0 * 3u, b0 = a0 * 11u;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
      "{%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(b0));
  int i0 = 0, i1 = 0;
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
               : "+r"(i0), "+r"(i1)
               : "r"(a0), "r"(b0));
  out[threadIdx.x] = d0 + d1 + d2 + d3;
  iout[threadIdx.x] = i0 + i1;
}
#endif

#if KA_SM >= 80
// ldmatrix (LDSM) from shared memory filled by cp.async (LDGSTS); bf16 mma.sync (HMMA).
extern "C" __global__ void ka_mma_bf16(const unsigned* g, float* out) {
  __shared__ __align__(128) unsigned smem[1024];
  unsigned s = (unsigned)__cvta_generic_to_shared(smem + threadIdx.x * 4);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g + threadIdx.x * 4));
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  unsigned a0, a1, a2, a3, b0, b1;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(s));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(b0), "=r"(b1)
               : "r"(s + 512));
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  out[threadIdx.x] = d0 + d1 + d2 + d3;
}

// INT8 mma.sync (IMMA), s8 x s8 -> s32.
extern "C" __global__ void ka_mma_s8(const unsigned* g, int* out) {
  unsigned a0 = g[threadIdx.x], a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u,
           b1 = a0 * 13u;
  int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  out[threadIdx.x] = d0 + d1 + d2 + d3;
}
#endif

#if KA_SM >= 89
// FP8 e4m3 mma.sync with fp32 accumulation (QMMA on sm_89 / sm_12x; emulated elsewhere).
extern "C" __global__ void ka_mma_e4m3(const unsigned* g, float* out) {
  unsigned a0 = g[threadIdx.x], a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u,
           b1 = a0 * 13u;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  out[threadIdx.x] = d0 + d1 + d2 + d3;
}
#endif

#if KA_SM >= 90
// TMA: a tensor-map tile load (cp.async.bulk.tensor), a bulk copy (cp.async.bulk) and the
// mbarrier they complete on.
extern "C" __global__ void ka_tma(const __grid_constant__ CUtensorMap map, const float* g,
                                  float* out) {
  __shared__ __align__(128) float tile[64 * 64];
  __shared__ __align__(8) unsigned long long bar;
  unsigned b = (unsigned)__cvta_generic_to_shared(&bar);
  unsigned t = (unsigned)__cvta_generic_to_shared(tile);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(32768));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
        "[%1, {%2, %3}], [%4];" ::"r"(t),
        "l"(&map), "r"(0), "r"(0), "r"(b)
        : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 256, "
        "[%2];" ::"r"(t + 16384),
        "l"(g), "r"(b)
        : "memory");
  }
  asm volatile(
      "{\n.reg .pred p;\nWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra WAIT;\n}" ::"r"(b)
      : "memory");
  out[threadIdx.x] = tile[threadIdx.x];
  if (threadIdx.x == 0) {  // the tile back through the tensor map (a TMA store)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(&map),
        "r"(0), "r"(0), "r"(t)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group 0;" ::: "memory");
  }
}

// stmatrix (sm_90+): registers to shared memory in mma fragment layout.
extern "C" __global__ void ka_stmatrix(const unsigned* g, unsigned* out) {
  __shared__ __align__(128) unsigned smem[1024];
  unsigned s = (unsigned)__cvta_generic_to_shared(smem + threadIdx.x * 4);
  unsigned r0 = g[threadIdx.x], r1 = r0 * 3u, r2 = r0 * 5u, r3 = r0 * 7u;
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" ::"r"(s),
               "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
  __syncthreads();
  out[threadIdx.x] = smem[(threadIdx.x * 7) & 1023];
}
#endif

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// Hopper warpgroup MMA from shared-memory descriptors: bf16 and e4m3 (wgmma, sm_90a).
extern "C" __global__ void ka_wgmma(unsigned long long da, unsigned long long db, float* out) {
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, 1, 1, 1, 0, "
      "0;"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "l"(da), "l"(db));
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 1, 1, 1;"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "l"(da), "l"(db));
  int n0 = 0, n1 = 0, n2 = 0, n3 = 0;
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 {%0,%1,%2,%3}, %4, %5, 1;"
               : "+r"(n0), "+r"(n1), "+r"(n2), "+r"(n3)
               : "l"(da), "l"(db));
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  out[threadIdx.x] = d0 + d1 + d2 + d3 + (float)(n0 + n1 + n2 + n3);
}
#endif

#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
// Datacenter Blackwell tensor-memory MMA (tcgen05, sm_100a): allocate TMEM, one MMA of each
// kind from shared-memory descriptors (the block-scaled ones with their scale factors in
// TMEM), commit, load the accumulator back.
extern "C" __global__ void ka_tcgen05(unsigned long long da, unsigned long long db,
                                      unsigned idesc, float* out) {
  __shared__ unsigned taddr_smem;
  __shared__ __align__(8) unsigned long long bar;
  unsigned dst = (unsigned)__cvta_generic_to_shared(&taddr_smem);
  unsigned b = (unsigned)__cvta_generic_to_shared(&bar);
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" ::"r"(dst));
  }
  __syncthreads();
  unsigned tmem = taddr_smem;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b));
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(tmem),
        "l"(da), "l"(db), "r"(idesc), "r"(1));
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}" ::"r"(tmem),
        "l"(da), "l"(db), "r"(idesc), "r"(1));
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p;\n}" ::"r"(tmem),
        "l"(da), "l"(db), "r"(idesc), "r"(1));
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%5], p;\n}"
        ::"r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(1), "r"(tmem + 16));
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, "
        "[%5], [%5], p;\n}" ::"r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(1), "r"(tmem + 16));
    // The A operand in tensor memory (a second tmem[] operand, before the descriptor), plain
    // and block-scaled with separate A / B scale addresses (one tmem[] scale operand), and
    // MXFP4 (kind::mxf4): the census tells block-scaled MMAs by the operand after idesc[].
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%5], %2, %3, p;\n}" ::"r"(tmem),
        "l"(da), "l"(db), "r"(idesc), "r"(1), "r"(tmem + 8));
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], [%5], %2, %3, [%6], [%7], "
        "p;\n}" ::"r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(1), "r"(tmem + 8),
        "r"(tmem + 16), "r"(tmem + 20));
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n}"
        ::"r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(1), "r"(tmem + 16), "r"(tmem + 20));
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(b));
  }
  asm volatile(
      "{\n.reg .pred p;\nWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra WAIT;\n}" ::"r"(b)
      : "memory");
  unsigned r0;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r0) : "r"(tmem));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(tmem + 32), "r"(r0));
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  out[threadIdx.x] = __uint_as_float(r0);
  __syncthreads();
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" ::"r"(tmem));
  }
}
#endif

#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
// GeForce Blackwell block-scaled mma.sync (sm_120a): MXFP8 (e4m3, ue8m0 scales) and
// NVFP4-style e2m1 with ue8m0 scales.
extern "C" __global__ void ka_mma_block_scaled(const unsigned* g, float* out) {
  unsigned a0 = g[threadIdx.x], a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u,
           b1 = a0 * 13u, sfa = 127u, sfb = 127u;
  unsigned short z = 0;
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
      "e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
      "{%13}, {%14,%15};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z),
        "r"(sfb), "h"(z), "h"(z));
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1."
      "e2m1.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
      "{%13}, {%14,%15};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z),
        "r"(sfb), "h"(z), "h"(z));
  out[threadIdx.x] = d0 + d1 + d2 + d3;
}
#endif
