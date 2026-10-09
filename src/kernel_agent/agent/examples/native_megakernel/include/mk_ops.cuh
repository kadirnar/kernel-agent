// Opcodes of the example megakernel (device code only: no torch headers, so the CPU tests can
// compile it to PTX for every architecture). Generic and op level, reusable by any stage:
//
//   RMSNORM        y = gamma * bf16(x * rsqrt(mean(x^2) + eps))       (one block, one vector)
//   GEMV           rows [row0, row0 + rows) of y = W h, bf16 weights from the page pool;
//                  h = x[k0:k0 + klen], or with a gamma the RMSNorm of x fused in (the norm
//                  over all K columns); epilogue: bf16 output (+ a residual) or fp32 partials
//                  for a split-K reduce
//   GEMV_FP8       the same with e4m3 weights and an fp32 scale per output row
//   RESIDUAL       y = a + b over a range
//   SPLITK_REDUCE  y[r] = bf16(sum_s part[s][r]) (+ a residual)
//   ARGMAX         out = argmax(x[0:n]) (first maximum; NaN counts as the largest, as torch);
//                  with A_ADVANCE also the token advance of a decode step: the step state's
//                  token := the argmax, position += 1, history[position] := the argmax
//   GLU            y = bf16(act(a)) * b over a range (act: silu, gelu tanh, gelu erf; a
//                  gated MLP's activation)
//
// and, in DecodeOps (mk_decode.cuh, included at the end), the decode-step opcodes: EMBED,
// ROPE_KV, ATTN_DECODE, ATTN_COMBINE (split-KV attention over a KV cache, milestones 6 and 7
// of the megakernel ladder).
//
// A whole-row GEMV tile (K of 1024 or 2048, up to 16 rows) splits K across the threads
// (gemv_split); the other shapes (split-K slices, wider K, more rows) stage h in shared memory
// (gemv). Rounding follows the PyTorch reference: RMSNorm normalises in fp32, rounds to bf16,
// then multiplies by the bf16 weight (rounded again); a projection rounds its fp32 sum to
// bf16; a residual add rounds bf16 + bf16 once. Activations written inside the megakernel are
// read after the counters' acquire (plain or __ldcg loads, never __ldg); weights come from the
// pool; the barriers are the consumers' (ka_mk::sync), not __syncthreads.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "ka_mk.cuh"

namespace mk {

typedef __nv_bfloat16 bf16;

// Opcode numbers and argument slots: mirrored in kernel_agent/native/megakernel/opcodes.py.
enum Opcode : int { NOP = 0, RMSNORM = 1, GEMV = 2, GEMV_FP8 = 3, RESIDUAL = 4, SPLITK_REDUCE = 5,
                    ARGMAX = 6, GLU = 7 };

enum GemvArg : int {
  G_X = 0, G_X_OFF, G_GAMMA, G_GAMMA_OFF, G_EPS, G_OUT, G_OUT_OFF, G_RES, G_RES_OFF, G_ROW0,
  G_ROWS, G_K, G_K0, G_KLEN, G_PART, G_PART_OFF, G_SCALE, G_SCALE_OFF
};
enum NormArg : int { N_X = 0, N_X_OFF, N_GAMMA, N_GAMMA_OFF, N_EPS, N_OUT, N_OUT_OFF, N_N };
enum ResidualArg : int { R_A = 0, R_A_OFF, R_B, R_B_OFF, R_OUT, R_OUT_OFF, R_I0, R_N };
enum ReduceArg : int {
  S_PART = 0, S_PART_OFF, S_SPLITS, S_STRIDE, S_ROW0, S_ROWS, S_OUT, S_OUT_OFF, S_RES, S_RES_OFF
};
enum ArgmaxArg : int {
  A_X = 0, A_X_OFF, A_N, A_OUT, A_OUT_OFF, A_ADVANCE, A_STEP, A_STEP_OFF, A_HIST, A_HIST_LEN
};
// The step state of a decode step (int32 words on the device): the token it embeds and the
// position it appends at; ARGMAX with A_ADVANCE writes the next ones. A_ADVANCE 0 (a word the
// schedule leaves zero) is a plain argmax.
enum StepWord : int { STEP_TOKEN = 0, STEP_POS = 1 };
enum GluArg : int { U_A = 0, U_A_OFF, U_B, U_B_OFF, U_OUT, U_OUT_OFF, U_I0, U_N, U_ACT };
enum GluAct : int { GLU_SILU = 0, GLU_GELU_TANH = 1, GLU_GELU = 2 };

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPageBytes = 8192;
// GEMV columns per tile (h staged in shared memory; with a fused norm, all K columns of
// x sit in registers: K <= kMaxSlice); rows per GEMV tile: at most 32 * kWarps
constexpr int kMaxSlice = 4096;
constexpr int kScratchBytes = kMaxSlice * 2 + 128;
typedef ka_mk::Ctx<kPageBytes> Ctx;

// ---------------------------------------------------------------- helpers

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sum over the block; `red` holds kWarps floats of scratch. Every thread gets the sum.
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  ka_mk::sync<kThreads>();
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kWarps; ++k) s += red[k];
  ka_mk::sync<kThreads>();
  return s;
}

__device__ __forceinline__ float sumsq8(uint4 v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(p[j]);
    s = fmaf(f.x, f.x, fmaf(f.y, f.y, s));
  }
  return s;
}

// gamma * bf16(x * inv), element-wise over 8 bf16 (the reference's two roundings)
__device__ __forceinline__ uint4 norm8(uint4 x, uint4 g, float inv) {
  uint4 out;
  const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&x);
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&g);
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(xp[j]);
    const float2 n = __bfloat1622float2(__floats2bfloat162_rn(f.x * inv, f.y * inv));
    const float2 w = __bfloat1622float2(gp[j]);
    op[j] = __floats2bfloat162_rn(w.x * n.x, w.y * n.y);
  }
  return out;
}

__device__ __forceinline__ float dot8(uint4 w, uint4 h, float acc) {
  const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w);
  const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&h);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 a = __bfloat1622float2(wp[j]), b = __bfloat1622float2(hp[j]);
    acc = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
  }
  return acc;
}

// 16 e4m3 weights (one 16-byte vector) times h[16] (two bf16 vectors)
__device__ __forceinline__ float dot16_fp8(uint4 w, uint4 h0, uint4 h1, float acc) {
  const __nv_fp8x2_storage_t* wp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const __half2 hw = __half2(__nv_cvt_fp8x2_to_halfraw2(wp[j], __NV_E4M3));
    const float2 a = __half22float2(hw);
    const uint4& hv = j < 4 ? h0 : h1;
    const float2 b =
        __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&hv)[j & 3]);
    acc = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
  }
  return acc;
}

// ---------------------------------------------------------------- opcodes

__device__ __forceinline__ void rmsnorm(const Ctx& c) {
  const bf16* x = c.ptr<const bf16>(c.arg(N_X)) + c.arg(N_X_OFF);
  const bf16* g = c.ptr<const bf16>(c.arg(N_GAMMA)) + c.arg(N_GAMMA_OFF);
  bf16* y = c.ptr<bf16>(c.arg(N_OUT)) + c.arg(N_OUT_OFF);
  const int n = c.arg(N_N);
  float* red = reinterpret_cast<float*>(c.scratch + kMaxSlice * 2);
  float ss = 0.f;
  for (int k = threadIdx.x * 8; k < n; k += kThreads * 8)
    ss += sumsq8(__ldcg(reinterpret_cast<const uint4*>(x + k)));
  const float inv = rsqrtf(block_sum(ss, red) / n + c.arg_f(N_EPS));
  for (int k = threadIdx.x * 8; k < n; k += kThreads * 8)
    *reinterpret_cast<uint4*>(y + k) = norm8(__ldcg(reinterpret_cast<const uint4*>(x + k)),
                                             __ldg(reinterpret_cast<const uint4*>(g + k)), inv);
}

// N words of the pool at shared-memory address a (ld.shared, not a generic load)
template <int N>
__device__ __forceinline__ void ld_words_smem(uint32_t a, uint32_t (&v)[N]) {
  if constexpr (N % 4 == 0) {
#pragma unroll
    for (int k = 0; k < N / 4; ++k)
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(v[4 * k]), "=r"(v[4 * k + 1]), "=r"(v[4 * k + 2]), "=r"(v[4 * k + 3])
                   : "r"(a + 16 * k));
  } else if constexpr (N == 2) {
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v[0]), "=r"(v[1]) : "r"(a));
  } else {
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v[0]) : "r"(a));
  }
}

// N words of global memory: `coherent` for data other blocks wrote in this launch (a plain,
// L1-cacheable load: after the acquire it sees their writes, and the 8 warps' reads of the
// same lines hit L1), else __ldg for read-only data
template <int N>
__device__ __forceinline__ void ld_words(const void* p, uint32_t (&v)[N], bool coherent) {
  if constexpr (N % 4 == 0) {
#pragma unroll
    for (int k = 0; k < N / 4; ++k) {
      const uint4* q = reinterpret_cast<const uint4*>(p) + k;
      const uint4 u = coherent ? *q : __ldg(q);
      v[4 * k] = u.x, v[4 * k + 1] = u.y, v[4 * k + 2] = u.z, v[4 * k + 3] = u.w;
    }
  } else if constexpr (N == 2) {
    const uint2* q = reinterpret_cast<const uint2*>(p);
    const uint2 u = coherent ? *q : __ldg(q);
    v[0] = u.x, v[1] = u.y;
  } else {
    const unsigned* q = reinterpret_cast<const unsigned*>(p);
    v[0] = coherent ? *q : __ldg(q);
  }
}

__device__ __forceinline__ float2 bf2(uint32_t w) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w));
}

// v[0..R) of every lane → lane l holds the warp's sum of v[l / (32 / R)]
template <int R>
__device__ __forceinline__ float transpose_reduce(float (&v)[R], int lane) {
#pragma unroll
  for (int s = 0; (R >> s) > 1; ++s) {
    const int o = 16 >> s, n = R >> s;
    const bool upper = lane & o;
#pragma unroll
    for (int j = 0; j < n / 2; ++j) {
      const float send = upper ? v[j] : v[j + n / 2];
      const float keep = upper ? v[j + n / 2] : v[j];
      v[j] = keep + __shfl_xor_sync(0xffffffffu, send, o);
    }
  }
  float s = v[0];
#pragma unroll
  for (int o = 16 / R; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

// GEMV tile over whole rows, K = 256 * CPL columns and at most R rows: the 256 threads split
// K, not the rows. Thread (w, l) owns columns [(w * 32 + l) * CPL, +CPL) of every row: it
// loads and normalises only its CPL columns of x (the norm's sum of squares: one block
// reduction), then multiplies them into all R rows, every row's weight words loaded from the
// pool first, the products after (back-to-back shared loads, independent FMA chains). Each
// warp folds its R partial sums across its lanes (a transpose reduction: R - 1 shuffles), one
// shared-memory exchange and one barrier sum the 8 warps, and thread r finishes row r (scale,
// residual or fp32 partial). Measured on an RTX 5070 Ti (sm_120), a 16-row tile of a
// [1024, 1024] layer: ~0.5 us from its weights landing to h ready, ~0.3 us of products.
template <int CPL, int R, bool kFp8>
__device__ __forceinline__ void gemv_split(const Ctx& c) {
  constexpr int N = CPL / 2;  // bf16x2 words of a column chunk of x (and of a bf16 weight row)
  const bf16* x = c.ptr<const bf16>(c.arg(G_X)) + c.arg(G_X_OFF);
  constexpr int K = CPL * 256;
  const int row0 = c.arg(G_ROW0), rows = c.arg(G_ROWS), gamma = c.arg(G_GAMMA);
  const int part = c.arg(G_PART), res = c.arg(G_RES);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int own = (warp * 32 + lane) * CPL;  // this thread's first column
  const bool norm = gamma >= 0;
  // 1. one batch of loads: the own chunk of x and of gamma; thread r: row r's residual / scale
  uint32_t mine[N], gw[N];
  ld_words<N>(x + own, mine, true);
  if (norm) ld_words<N>(c.ptr<const bf16>(gamma) + c.arg(G_GAMMA_OFF) + own, gw, false);
  const int t = threadIdx.x;
  float rv = 0.f, sv = 1.f;
  if (t < rows && res >= 0 && part < 0)
    rv = __bfloat162float(__ldcg(c.ptr<const bf16>(res) + c.arg(G_RES_OFF) + row0 + t));
  if (kFp8 && t < rows)
    sv = __ldg(c.ptr<const float>(c.arg(G_SCALE)) + c.arg(G_SCALE_OFF) + row0 + t);
  // 2. h: the own columns, normalised over all K (the sum of squares: one block reduction)
  float h[CPL];
  float inv = 1.f;
  if (norm) {
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      const float2 f = bf2(mine[k]);
      ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
    }
    ss = warp_sum(ss);
    float* partial = reinterpret_cast<float*>(c.scratch + 4096);  // kWarps floats
    if (lane == 0) partial[warp] = ss;
    ka_mk::sync<kThreads>();
    ss = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) ss += partial[w];
    inv = rsqrtf(ss / K + c.arg_f(G_EPS));
  }
#pragma unroll
  for (int k = 0; k < N; ++k) {
    float2 f = bf2(mine[k]);
    if (norm) {
      const float2 n = __bfloat1622float2(__floats2bfloat162_rn(f.x * inv, f.y * inv));
      const float2 g = bf2(gw[k]);
      f = __bfloat1622float2(__floats2bfloat162_rn(g.x * n.x, g.y * n.y));
    }
    h[2 * k] = f.x, h[2 * k + 1] = f.y;
  }
  c.mark(0);
  // 3. the own columns of every row: weights from the pool. Rows that never straddle a page
  //    (kPageBytes a multiple of the row): one shared address per page, compile-time offsets
  constexpr int kEsize = kFp8 ? 1 : 2;
  constexpr int kRowBytes = K * kEsize;
  constexpr bool kWhole = kPageBytes % kRowBytes == 0;
  constexpr int kTilePages = kWhole ? (R * kRowBytes + kPageBytes - 1) / kPageBytes : 1;
  uint32_t base[kTilePages];
#pragma unroll
  for (int k = 0; k < kTilePages; ++k)
    base[k] = k * kPageBytes < rows * kRowBytes ? ka_mk::smem_addr(c.weights(k * kPageBytes)) : 0u;
  // every row's words first (back-to-back shared loads), then the products
  constexpr int M = kFp8 ? CPL / 4 : N;  // words of one row's chunk
  uint32_t ww[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (r < rows) {
      const uint32_t wp =
          kWhole ? base[r * kRowBytes / kPageBytes] + (r * kRowBytes) % kPageBytes + own * kEsize
                 : ka_mk::smem_addr(c.weights(r * kRowBytes + own * kEsize));
      ld_words_smem<M>(wp, ww[r]);
    } else {
#pragma unroll
      for (int k = 0; k < M; ++k) ww[r][k] = 0u;
    }
  }
  float v[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float acc = 0.f;
    if constexpr (kFp8) {
#pragma unroll
      for (int k = 0; k < M; ++k) {
        const __nv_fp8x2_storage_t* q = reinterpret_cast<const __nv_fp8x2_storage_t*>(&ww[r][k]);
        const float2 a = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(q[0], __NV_E4M3)));
        const float2 b = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(q[1], __NV_E4M3)));
        acc = fmaf(a.x, h[4 * k], fmaf(a.y, h[4 * k + 1], acc));
        acc = fmaf(b.x, h[4 * k + 2], fmaf(b.y, h[4 * k + 3], acc));
      }
    } else {
#pragma unroll
      for (int k = 0; k < M; ++k) {
        const float2 a = bf2(ww[r][k]);
        acc = fmaf(a.x, h[2 * k], fmaf(a.y, h[2 * k + 1], acc));
      }
    }
    v[r] = acc;
  }
  c.mark(1);  // the products done (trace mark 1; mark 0: h ready)
  // 4. rows' sums: within the warp (transpose reduction), then across the 8 warps
  const float s = transpose_reduce<R>(v, lane);
  float* red = reinterpret_cast<float*>(c.scratch);  // kWarps * R floats
  if (lane % (32 / R) == 0) red[warp * R + lane / (32 / R)] = s;
  ka_mk::sync<kThreads>();
  if (t < rows) {
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += red[w * R + t];
    const int row = row0 + t;
    if (kFp8) acc *= sv;
    if (part >= 0) {
      c.ptr<float>(part)[c.arg(G_PART_OFF) + row] = acc;
    } else {
      bf16 y = __float2bfloat16(acc);
      if (res >= 0) y = __float2bfloat16(rv + __bfloat162float(y));
      c.ptr<bf16>(c.arg(G_OUT))[c.arg(G_OUT_OFF) + row] = y;
    }
  }
}

// The split-K-free whole-row tiles take gemv_split (K of 1024 or 2048 columns, up to 16 rows),
// the others the shared-memory path of gemv().
template <bool kFp8>
__device__ __forceinline__ bool gemv_whole(const Ctx& c) {
  const int K = c.arg(G_K), rows = c.arg(G_ROWS);
  if (c.arg(G_K0) != 0 || c.arg(G_KLEN) != K || rows > 16) return false;
  if (K == 1024) {
    gemv_split<4, 16, kFp8>(c);
  } else if (K == 2048) {
    gemv_split<8, 16, kFp8>(c);
  } else {
    return false;
  }
  return true;
}

// GEMV tile. Every global load it needs is issued at once, right after the wait (x, the
// norm's gamma, the residual of the rows each warp finishes): one L2 round trip on the
// critical path; the weights are already in the pool.
template <bool kFp8>
__device__ __forceinline__ void gemv(const Ctx& c) {
  constexpr int kVec = kMaxSlice / (8 * kThreads);  // x vectors per thread
  const bf16* x = c.ptr<const bf16>(c.arg(G_X)) + c.arg(G_X_OFF);
  const int K = c.arg(G_K), k0 = c.arg(G_K0), klen = c.arg(G_KLEN);
  const int row0 = c.arg(G_ROW0), rows = c.arg(G_ROWS), gamma = c.arg(G_GAMMA);
  const int part = c.arg(G_PART), res = c.arg(G_RES);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  bf16* h = reinterpret_cast<bf16*>(c.scratch);
  float* red = reinterpret_cast<float*>(c.scratch + kMaxSlice * 2);
  // 1. the loads: with a norm all K columns of x (its sum of squares) and the slice's gamma,
  //    else the slice; lane m of a warp loads the residual of the warp's m-th row
  const bool norm = gamma >= 0;
  const int lo = norm ? k0 : 0;  // the slice among the loaded columns
  const int span = norm ? K : klen;
  const bf16* xs = norm ? x : x + k0;
  const bf16* g = norm ? c.ptr<const bf16>(gamma) + c.arg(G_GAMMA_OFF) : nullptr;
  uint4 xv[kVec], gv[kVec];
#pragma unroll
  for (int v = 0; v < kVec; ++v) {
    const int k = (threadIdx.x + v * kThreads) * 8;
    xv[v] = k < span ? __ldcg(reinterpret_cast<const uint4*>(xs + k)) : make_uint4(0, 0, 0, 0);
    gv[v] = norm && k >= lo && k < lo + klen ? __ldg(reinterpret_cast<const uint4*>(g + k))
                                             : make_uint4(0, 0, 0, 0);
  }
  float rv = 0.f;
  if (res >= 0 && part < 0 && warp + kWarps * lane < rows)
    rv = __bfloat162float(
        __ldcg(c.ptr<const bf16>(res) + c.arg(G_RES_OFF) + row0 + warp + kWarps * lane));
  // 2. the norm over all K columns, 3. h: the slice (normalised) in shared memory
  float inv = 1.f;
  if (norm) {
    float ss = 0.f;
#pragma unroll
    for (int v = 0; v < kVec; ++v) ss += sumsq8(xv[v]);
    inv = rsqrtf(block_sum(ss, red) / K + c.arg_f(G_EPS));
  }
#pragma unroll
  for (int v = 0; v < kVec; ++v) {
    const int k = (threadIdx.x + v * kThreads) * 8;
    if (k >= lo && k < lo + klen)
      *reinterpret_cast<uint4*>(h + k - lo) = norm ? norm8(xv[v], gv[v], inv) : xv[v];
  }
  ka_mk::sync<kThreads>();
  // 4. one warp per row, two rows and two 16-byte columns of each at a time (independent
  //    loads and FMA chains: the loop is latency-bound, not bandwidth-bound)
  const int row_bytes = kFp8 ? klen : klen * 2;
  const float* scale = kFp8 ? c.ptr<const float>(c.arg(G_SCALE)) + c.arg(G_SCALE_OFF) : nullptr;
  for (int j = warp, m = 0; j < rows; j += 2 * kWarps, m += 2) {
    const bool two = j + kWarps < rows;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};  // row j: [0], [1]; row j + kWarps: [2], [3]
    for (int b = lane * 16; b < row_bytes; b += 2 * 32 * 16) {
      const int b2 = b + 32 * 16;
      const bool next = b2 < row_bytes;
      const uint4 z = make_uint4(0, 0, 0, 0);
      const uint4 w00 = *reinterpret_cast<const uint4*>(c.weights(j * row_bytes + b));
      const uint4 w01 = next ? *reinterpret_cast<const uint4*>(c.weights(j * row_bytes + b2)) : z;
      const uint4 w10 =
          two ? *reinterpret_cast<const uint4*>(c.weights((j + kWarps) * row_bytes + b)) : z;
      const uint4 w11 = two && next
                            ? *reinterpret_cast<const uint4*>(c.weights((j + kWarps) * row_bytes + b2))
                            : z;
      if (kFp8) {
        const uint4* h0 = reinterpret_cast<const uint4*>(h + b);
        const uint4* h1 = reinterpret_cast<const uint4*>(h + (next ? b2 : b));
        acc[0] = dot16_fp8(w00, h0[0], h0[1], acc[0]);
        acc[1] = dot16_fp8(w01, h1[0], h1[1], acc[1]);
        acc[2] = dot16_fp8(w10, h0[0], h0[1], acc[2]);
        acc[3] = dot16_fp8(w11, h1[0], h1[1], acc[3]);
      } else {
        const uint4 h0 = *reinterpret_cast<const uint4*>(h + b / 2);
        const uint4 h1 = next ? *reinterpret_cast<const uint4*>(h + b2 / 2) : z;
        acc[0] = dot8(w00, h0, acc[0]);
        acc[1] = dot8(w01, h1, acc[1]);
        acc[2] = dot8(w10, h0, acc[2]);
        acc[3] = dot8(w11, h1, acc[3]);
      }
    }
    float s0 = acc[0] + acc[1], s1 = acc[2] + acc[3];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {  // both rows' sums in one pass
      s0 += __shfl_xor_sync(0xffffffffu, s0, o);
      s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    const float r0 = __shfl_sync(0xffffffffu, rv, m);  // the rows' residuals (lane m, m + 1)
    const float r1 = __shfl_sync(0xffffffffu, rv, m + 1 < 32 ? m + 1 : m);
    if (lane < (two ? 2 : 1)) {  // lane 0 writes row j, lane 1 row j + kWarps
      const int row = row0 + j + lane * kWarps;
      float acc_row = lane ? s1 : s0;
      if (kFp8) acc_row *= __ldg(scale + row);
      if (part >= 0) {
        c.ptr<float>(part)[c.arg(G_PART_OFF) + row] = acc_row;
      } else {
        bf16 y = __float2bfloat16(acc_row);
        if (res >= 0) y = __float2bfloat16((lane ? r1 : r0) + __bfloat162float(y));
        c.ptr<bf16>(c.arg(G_OUT))[c.arg(G_OUT_OFF) + row] = y;
      }
    }
  }
}

__device__ __forceinline__ void residual(const Ctx& c) {
  const bf16* a = c.ptr<const bf16>(c.arg(R_A)) + c.arg(R_A_OFF);
  const bf16* b = c.ptr<const bf16>(c.arg(R_B)) + c.arg(R_B_OFF);
  bf16* y = c.ptr<bf16>(c.arg(R_OUT)) + c.arg(R_OUT_OFF);
  const int i0 = c.arg(R_I0), n = c.arg(R_N);
  for (int i = i0 + threadIdx.x; i < i0 + n; i += kThreads)
    y[i] = __float2bfloat16(__bfloat162float(__ldcg(a + i)) + __bfloat162float(__ldcg(b + i)));
}

// The activation of a GLU in fp32, as torch computes it for bf16 (opmath float)
__device__ __forceinline__ float glu_act(float v, int act) {
  if (act == GLU_SILU) return v / (1.f + expf(-v));
  if (act == GLU_GELU_TANH) {
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.f + tanhf(inner));
  }
  return 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
}

// y = bf16(act(a)) * b: torch's two roundings (the activation's bf16 output, then the product)
__device__ __forceinline__ void glu(const Ctx& c) {
  const bf16* a = c.ptr<const bf16>(c.arg(U_A)) + c.arg(U_A_OFF);
  const bf16* b = c.ptr<const bf16>(c.arg(U_B)) + c.arg(U_B_OFF);
  bf16* y = c.ptr<bf16>(c.arg(U_OUT)) + c.arg(U_OUT_OFF);
  const int i0 = c.arg(U_I0), n = c.arg(U_N), act = c.arg(U_ACT);
  for (int i = i0 + threadIdx.x; i < i0 + n; i += kThreads) {
    const float g = __bfloat162float(__float2bfloat16(glu_act(__bfloat162float(__ldcg(a + i)),
                                                              act)));
    y[i] = __float2bfloat16(g * __bfloat162float(__ldcg(b + i)));
  }
}

__device__ __forceinline__ void splitk_reduce(const Ctx& c) {
  const float* part = c.ptr<const float>(c.arg(S_PART)) + c.arg(S_PART_OFF);
  const int splits = c.arg(S_SPLITS), stride = c.arg(S_STRIDE);
  const int row0 = c.arg(S_ROW0), rows = c.arg(S_ROWS), res = c.arg(S_RES);
  bf16* y = c.ptr<bf16>(c.arg(S_OUT)) + c.arg(S_OUT_OFF);
  for (int r = row0 + threadIdx.x; r < row0 + rows; r += kThreads) {
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) acc += __ldcg(part + static_cast<size_t>(s) * stride + r);
    bf16 out = __float2bfloat16(acc);
    if (res >= 0) {
      const float rv = __bfloat162float(__ldcg(c.ptr<const bf16>(res) + c.arg(S_RES_OFF) + r));
      out = __float2bfloat16(rv + __bfloat162float(out));
    }
    y[r] = out;
  }
}

// a beats b: NaN first, then the larger value, then the smaller index (torch.argmax)
__device__ __forceinline__ bool beats(float a, int ia, float b, int ib) {
  const bool na = a != a, nb = b != b;
  if (na != nb) return na;
  if (!na && a != b) return a > b;
  return ia < ib;
}

// The visiting order does not matter: beats() is a total order (NaN, value, index).
__device__ __forceinline__ void argmax(const Ctx& c) {
  const bf16* x = c.ptr<const bf16>(c.arg(A_X)) + c.arg(A_X_OFF);
  const int n = c.arg(A_N);
  float best = -INFINITY;
  int at = 0x7fffffff;
  // 16-byte loads where x is aligned (a vocabulary of logits: 4 loads per thread at 8192)
  const int vec = reinterpret_cast<uintptr_t>(x) % 16 == 0 ? n / 8 : 0;
  for (int v = threadIdx.x; v < vec; v += kThreads) {
    const uint4 u = __ldcg(reinterpret_cast<const uint4*>(x) + v);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = bf2(w[j]);
      if (beats(f.x, 8 * v + 2 * j, best, at)) best = f.x, at = 8 * v + 2 * j;
      if (beats(f.y, 8 * v + 2 * j + 1, best, at)) best = f.y, at = 8 * v + 2 * j + 1;
    }
  }
  for (int i = vec * 8 + threadIdx.x; i < n; i += kThreads) {
    const float v = __bfloat162float(__ldcg(x + i));
    if (beats(v, i, best, at)) best = v, at = i;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float v = __shfl_xor_sync(0xffffffffu, best, o);
    const int i = __shfl_xor_sync(0xffffffffu, at, o);
    if (beats(v, i, best, at)) best = v, at = i;
  }
  float* vals = reinterpret_cast<float*>(c.scratch + kMaxSlice * 2);
  int* idx = reinterpret_cast<int*>(c.scratch);
  if ((threadIdx.x & 31) == 0) vals[threadIdx.x >> 5] = best, idx[threadIdx.x >> 5] = at;
  ka_mk::sync<kThreads>();
  if (threadIdx.x == 0) {
    for (int k = 1; k < kWarps; ++k)
      if (beats(vals[k], idx[k], best, at)) best = vals[k], at = idx[k];
    const int token = at < n ? at : 0;
    c.ptr<long long>(c.arg(A_OUT))[c.arg(A_OUT_OFF)] = token;
    if (c.arg(A_ADVANCE)) {
      // the next step's inputs, on the device: every reader of the state in this launch (the
      // embedding, the RoPE / KV append, the attention) precedes this instruction through
      // the schedule's edges, and the next launch starts after this one
      int* st = c.ptr<int>(c.arg(A_STEP)) + c.arg(A_STEP_OFF);
      const int pos = __ldcg(st + STEP_POS) + 1;
      st[STEP_TOKEN] = token;
      st[STEP_POS] = pos;
      const int hist = c.arg(A_HIST);
      if (hist >= 0 && pos >= 0 && pos < c.arg(A_HIST_LEN)) c.ptr<int>(hist)[pos] = token;
    }
  }
}

struct Ops {
  static constexpr int kThreads = mk::kThreads;
  static constexpr int kPageBytes = mk::kPageBytes;
  static constexpr int kScratchBytes = mk::kScratchBytes;

  __device__ static bool run(const Ctx& c) {
    switch (c.op()) {
      case NOP: return true;
      case RMSNORM: rmsnorm(c); return true;
      case GEMV:
        if (!gemv_whole<false>(c)) gemv<false>(c);
        return true;
      case GEMV_FP8:
        if (!gemv_whole<true>(c)) gemv<true>(c);
        return true;
      case RESIDUAL: residual(c); return true;
      case SPLITK_REDUCE: splitk_reduce(c); return true;
      case ARGMAX: argmax(c); return true;
      case GLU: glu(c); return true;
      default: return false;
    }
  }
};

}  // namespace mk

// The decode-step opcodes and DecodeOps (the generic opcodes and those): a separate
// instantiation of the interpreter, so the chain's kernel<Ops> keeps its own registers.
#include "mk_decode.cuh"
