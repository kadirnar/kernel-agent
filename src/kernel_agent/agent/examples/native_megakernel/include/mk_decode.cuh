// Decode-step opcodes of the example megakernel (issue #225, milestones 6 and 7) and their
// interpreter DecodeOps: included at the end of mk_ops.cuh (device code only). Op level and
// model-agnostic; bf16 or fp16 activations and KV caches (a dtype argument: 0 bf16, 1 fp16):
//
//   EMBED         out = table[token]: the row of the token the step state holds (a device
//                 value, not a host one)
//   ROPE_KV       one KV head: its q heads and its k head rotated at the step's position
//                 (rotate-half RoPE from fp32 cos / sin tables, rounded like an eager model:
//                 q * cos + rotate_half(q) * sin in the activation dtype), q into a buffer,
//                 k and v appended to the caches at the position
//   ATTN_DECODE   one query token's attention over one chunk of a KV head's cache for a
//                 block of up to kMaxQBlock of its q heads (GQA: they share every K / V load):
//                 the chunk's max, sum and unnormalised output, fp32 (split-KV)
//   ATTN_COMBINE  the chunks of a KV head's q heads reduced (flash-decoding):
//                 o = sum_c exp(m_c - M) o_c / sum_c exp(m_c - M) l_c, rounded to the dtype
//
// (the gated MLP's activation is the generic GLU opcode of mk_ops.cuh)
//
// The KV length is a device value: the int32 at a tensor's offset plus an addend (the step
// state's position + 1), clamped to the cache capacity. A schedule has a fixed number of
// splits per KV head; the opcode derives the chunk from the length (kv_chunk: `chunk` keys,
// or with chunk 0 the length spread over the splits, rounded up to kChunkGranule keys), so
// every split works at every length. A chunk past the length writes nothing; the combine
// derives the same chunks and reads only the non-empty ones. Python mirror:
// kernel_agent.native.megakernel.decode (kv_split).
//
// Caches: [kv heads][capacity][head dim] per layer (one tensor-table entry each); partial
// buffers: fp32 rows prow0 + q_head * splits + split of head-dim outputs, and of (max, sum)
// pairs; the max is in the log2 domain (scores x scale x log2(e): the softmax uses exp2).
// Every reduction runs in a fixed order (shuffle butterflies, warps in turn): the same bits
// on every call.
#pragma once

#include <cuda_fp16.h>

namespace mk {

// Opcode numbers (after enum Opcode's) and argument slots: mirrored in
// kernel_agent/native/megakernel/opcodes.py.
enum DecodeOpcode : int { EMBED = 8, ROPE_KV = 9, ATTN_DECODE = 10, ATTN_COMBINE = 11 };

enum EmbedArg : int {
  EM_TABLE = 0, EM_VOCAB, EM_DIM, EM_ESIZE, EM_TOKEN, EM_TOKEN_OFF, EM_OUT, EM_OUT_OFF
};
enum RopeArg : int {
  RK_QKV = 0, RK_QKV_OFF, RK_QHEADS, RK_KVHEADS, RK_KV_HEAD, RK_GROUP, RK_DIM, RK_QOUT,
  RK_QOUT_OFF, RK_K, RK_V, RK_CAP, RK_POS, RK_POS_OFF, RK_COS, RK_SIN, RK_DTYPE
};
enum AttnArg : int {
  AT_Q = 0, AT_Q_OFF, AT_K, AT_V, AT_CAP, AT_KV_HEAD, AT_Q0, AT_QN, AT_SPLIT, AT_SPLITS, AT_CHUNK,
  AT_LENGTH, AT_LENGTH_OFF, AT_LENGTH_ADD, AT_SCALE, AT_PO, AT_PML, AT_PROW0, AT_DIM, AT_DTYPE
};
enum CombineArg : int {
  CB_PO = 0, CB_PML, CB_PROW0, CB_SPLITS, CB_CHUNK, CB_LENGTH, CB_LENGTH_OFF, CB_LENGTH_ADD,
  CB_CAP, CB_Q0, CB_QN, CB_OUT, CB_OUT_OFF, CB_DIM, CB_DTYPE
};

constexpr float kLog2e = 1.4426950408889634f;
// q heads per attention tile: each thread keeps 8 columns of each one's q and output in
// registers (2 x 32 floats at 4)
constexpr int kMaxQBlock = 4;
// key passes per batch of an attention tile: each thread has 2 x kKvPasses 16-byte K / V
// loads in flight (16 KB per SM at 256 threads), issued before any arithmetic. 4 spilled
// ~100 bytes on sm_80-sm_120: the 288-thread interpreter gets 168 registers per thread
constexpr int kKvPasses = 2;
// keys per chunk of a balanced split are a multiple of this
constexpr int kChunkGranule = 16;

// ---------------------------------------------------------------- numbers of a dtype

template <typename T>
struct Num;
template <>
struct Num<bf16> {
  static __device__ __forceinline__ float2 f2(uint32_t w) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w));
  }
  static __device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ bf16 to(float v) { return __float2bfloat16(v); }
};
template <>
struct Num<__half> {
  static __device__ __forceinline__ float2 f2(uint32_t w) {
    return __half22float2(*reinterpret_cast<const __half2*>(&w));
  }
  static __device__ __forceinline__ float f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half to(float v) { return __float2half(v); }
};

// v rounded to T and back (an eager op's result in the activation dtype)
template <typename T>
__device__ __forceinline__ float rnd(float v) {
  return Num<T>::f(Num<T>::to(v));
}

template <typename T>
__device__ __forceinline__ void unpack8(uint4 v, float (&f)[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 p = Num<T>::f2(w[j]);
    f[2 * j] = p.x, f[2 * j + 1] = p.y;
  }
}

// arr[g] for a runtime g without local memory (the arrays are register-resident)
template <int N>
__device__ __forceinline__ float pick(const float (&arr)[N], int g) {
  float r = arr[0];
#pragma unroll
  for (int k = 1; k < N; ++k)
    if (g == k) r = arr[k];
  return r;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// ---------------------------------------------------------------- the KV length and its chunks

// The valid KV length: the int32 at tensor[off] (none when tensor < 0) plus `add`, in [0, cap].
// Every thread reads it (one L2 transaction per warp): data an earlier launch (or this one,
// before the counters) wrote, so __ldcg, never __ldg.
__device__ __forceinline__ int kv_length(const Ctx& c, int tensor, int off, int add, int cap) {
  long long len = add;
  if (tensor >= 0) len += __ldcg(c.ptr<const int>(tensor) + off);
  return static_cast<int>(len < 0 ? 0 : (len > cap ? cap : len));
}

// Keys per chunk at length `len`: `chunk` when > 0, else the length spread over the splits,
// rounded up to kChunkGranule (at least one granule).
__device__ __forceinline__ int kv_chunk(int len, int splits, int chunk) {
  if (chunk > 0) return chunk;
  const int per = (len + splits - 1) / splits;
  return per <= kChunkGranule ? kChunkGranule
                              : (per + kChunkGranule - 1) / kChunkGranule * kChunkGranule;
}

// ---------------------------------------------------------------- EMBED

__device__ __forceinline__ void embed(const Ctx& c) {
  const int vocab = c.arg(EM_VOCAB), esize = c.arg(EM_ESIZE);
  const int bytes = c.arg(EM_DIM) * esize;
  int tok = __ldcg(c.ptr<const int>(c.arg(EM_TOKEN)) + c.arg(EM_TOKEN_OFF));
  tok = tok < 0 ? 0 : (tok >= vocab ? vocab - 1 : tok);  // a bad token reads a valid row
  const unsigned char* src = c.ptr<const unsigned char>(c.arg(EM_TABLE)) +
                             static_cast<size_t>(tok) * bytes;
  unsigned char* dst = c.ptr<unsigned char>(c.arg(EM_OUT)) +
                       static_cast<size_t>(c.arg(EM_OUT_OFF)) * esize;
  const uintptr_t both = reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst);
  if (bytes % 16 == 0 && both % 16 == 0) {
    for (int k = threadIdx.x; k < bytes / 16; k += kThreads)
      reinterpret_cast<uint4*>(dst)[k] = __ldg(reinterpret_cast<const uint4*>(src) + k);
  } else {
    for (int k = threadIdx.x; k < bytes; k += kThreads) dst[k] = __ldg(src + k);
  }
}

// ---------------------------------------------------------------- ROPE_KV

// One KV head h: (q heads h * group .. + group) and k head h of the projection's output
// [q heads | kv heads (k) | kv heads (v)] x dim rotated at the step's position (the cos / sin
// tables' row, rounded to T as an eager model rounds them), q into qout, k and v into the
// caches' row at the position. A position outside [0, cap) appends nothing (the cache is
// full: the host stops before) and rotates q at the nearest row.
template <typename T>
__device__ __forceinline__ void rope_kv(const Ctx& c) {
  const int dim = c.arg(RK_DIM), half = dim / 2, group = c.arg(RK_GROUP), h = c.arg(RK_KV_HEAD);
  const int hq = c.arg(RK_QHEADS), hkv = c.arg(RK_KVHEADS), cap = c.arg(RK_CAP);
  const T* qkv = c.ptr<const T>(c.arg(RK_QKV)) + c.arg(RK_QKV_OFF);
  T* qo = c.ptr<T>(c.arg(RK_QOUT)) + c.arg(RK_QOUT_OFF);
  const int pos = __ldcg(c.ptr<const int>(c.arg(RK_POS)) + c.arg(RK_POS_OFF));
  const bool store = pos >= 0 && pos < cap;
  const int row = store ? pos : (pos < 0 ? 0 : cap - 1);
  const int cs = c.arg(RK_COS), sn = c.arg(RK_SIN);
  const size_t at = (static_cast<size_t>(h) * cap + row) * dim;
  T* kc = c.ptr<T>(c.arg(RK_K)) + at;
  T* vc = c.ptr<T>(c.arg(RK_V)) + at;
  for (int i = threadIdx.x; i < (group + 1) * half; i += kThreads) {
    const int hh = i / half, k = i % half;
    const T* src = qkv + static_cast<size_t>(hh < group ? h * group + hh : hq + h) * dim;
    const float x1 = Num<T>::f(src[k]), x2 = Num<T>::f(src[k + half]);  // written this launch
    float y1 = x1, y2 = x2;
    if (cs >= 0) {
      const float co = rnd<T>(__ldg(c.ptr<const float>(cs) + static_cast<size_t>(row) * half + k));
      const float si = rnd<T>(__ldg(c.ptr<const float>(sn) + static_cast<size_t>(row) * half + k));
      // q * cos + rotate_half(q) * sin, every product and the sum rounded to T
      y1 = rnd<T>(x1 * co) - rnd<T>(x2 * si);
      y2 = rnd<T>(x2 * co) + rnd<T>(x1 * si);
    }
    if (hh < group) {
      T* dst = qo + static_cast<size_t>(h * group + hh) * dim;
      dst[k] = Num<T>::to(y1);
      dst[k + half] = Num<T>::to(y2);
    } else if (store) {
      kc[k] = Num<T>::to(y1);
      kc[k + half] = Num<T>::to(y2);
    }
  }
  if (store)
    for (int i = threadIdx.x; i < dim; i += kThreads)
      vc[i] = qkv[static_cast<size_t>(hq + hkv + h) * dim + i];
}

// ---------------------------------------------------------------- ATTN_DECODE

// One chunk of one KV head's cache for q heads [q0, q0 + qn) (qn <= kMaxQBlock), head dim D.
// The chunk's keys are contiguous rows of the cache: thread t owns 8 columns (one 16-byte
// load) of key t / (D / 8) of each pass, so a pass reads 256 x 16 contiguous bytes of K and
// of V, kKvPasses passes per batch, every load of a batch issued before its arithmetic. Each
// key's score is summed over its D / 8 threads (a shuffle butterfly), and every thread keeps
// an online softmax (running max, sum, output) over its own keys for its 8 columns. At the
// end the keys' threads of a warp combine (butterfly), then the 8 warps in turn through
// shared memory: the chunk's max M, sum L and unnormalised output O per q head, fp32.
template <int D, typename T>
__device__ __forceinline__ void attn_decode(const Ctx& c) {
  constexpr int LPK = D / 8;           // threads per key
  constexpr int KPP = kThreads / LPK;  // keys per pass of the block
  constexpr int G = kMaxQBlock;
  const int len =
      kv_length(c, c.arg(AT_LENGTH), c.arg(AT_LENGTH_OFF), c.arg(AT_LENGTH_ADD), c.arg(AT_CAP));
  const int splits = c.arg(AT_SPLITS), split = c.arg(AT_SPLIT);
  const int chunk = kv_chunk(len, splits, c.arg(AT_CHUNK));
  const int k0 = split * chunk;
  const int k1 = len < k0 + chunk ? len : k0 + chunk;
  if (k0 >= k1) return;  // uniform: an empty chunk writes nothing, the combine skips it
  const int qn = c.arg(AT_QN), q0 = c.arg(AT_Q0);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int kp = t / LPK, d0 = (t % LPK) * 8;
  const T* q = c.ptr<const T>(c.arg(AT_Q)) + c.arg(AT_Q_OFF) + static_cast<size_t>(q0) * D;
  const size_t head = static_cast<size_t>(c.arg(AT_KV_HEAD)) * c.arg(AT_CAP) * D;
  const T* kc = c.ptr<const T>(c.arg(AT_K)) + head + d0;
  const T* vc = c.ptr<const T>(c.arg(AT_V)) + head + d0;
  const float scale = c.arg_f(AT_SCALE) * kLog2e;  // scores in the log2 domain: exp2 below

  // q (written in this launch: a plain load after the acquire), pre-scaled
  float qf[G][8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (g < qn) unpack8<T>(*reinterpret_cast<const uint4*>(q + g * D + d0), f);
#pragma unroll
    for (int e = 0; e < 8; ++e) qf[g][e] = f[e] * scale;
  }
  float m[G], l[G], acc[G][8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m[g] = -INFINITY, l[g] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[g][e] = 0.f;
  }

  for (int base = k0; base < k1; base += KPP * kKvPasses) {
    // 1. the batch's K and V (the current position's row was appended in this launch, the
    //    others by earlier launches: plain loads after the acquire, never __ldg)
    uint4 kv[kKvPasses], vv[kKvPasses];
#pragma unroll
    for (int p = 0; p < kKvPasses; ++p) {
      const int j = base + p * KPP + kp;
      const uint4 z = make_uint4(0, 0, 0, 0);
      kv[p] = j < k1 ? *reinterpret_cast<const uint4*>(kc + static_cast<size_t>(j) * D) : z;
      vv[p] = j < k1 ? *reinterpret_cast<const uint4*>(vc + static_cast<size_t>(j) * D) : z;
    }
    // 2. scores: the own 8 columns, then summed over the key's LPK threads
    float s[kKvPasses][G];
#pragma unroll
    for (int p = 0; p < kKvPasses; ++p) {
      float kf[8];
      unpack8<T>(kv[p], kf);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) a = fmaf(qf[g][e], kf[e], a);
        s[p][g] = a;
      }
    }
#pragma unroll
    for (int o = 1; o < LPK; o <<= 1)
#pragma unroll
      for (int p = 0; p < kKvPasses; ++p)
#pragma unroll
        for (int g = 0; g < G; ++g) s[p][g] += __shfl_xor_sync(0xffffffffu, s[p][g], o);
#pragma unroll
    for (int p = 0; p < kKvPasses; ++p)
      if (base + p * KPP + kp >= k1)
#pragma unroll
        for (int g = 0; g < G; ++g) s[p][g] = -INFINITY;
    // 3. online softmax over the thread's keys: the new max, the old terms rescaled, the
    //    batch's terms added (a thread with no valid key yet keeps m = -inf, l = 0)
#pragma unroll
    for (int g = 0; g < G; ++g) {
      if (g >= qn) continue;
      float mb = m[g];
#pragma unroll
      for (int p = 0; p < kKvPasses; ++p) mb = fmaxf(mb, s[p][g]);
      const float alpha = mb == -INFINITY ? 1.f : exp2f(m[g] - mb);
      l[g] *= alpha;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[g][e] *= alpha;
      m[g] = mb;
    }
#pragma unroll
    for (int p = 0; p < kKvPasses; ++p) {
      float vf[8];
      unpack8<T>(vv[p], vf);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        if (g >= qn) continue;
        const float pr = m[g] == -INFINITY ? 0.f : exp2f(s[p][g] - m[g]);
        l[g] += pr;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[g][e] = fmaf(pr, vf[e], acc[g][e]);
      }
    }
  }

  c.mark(0);  // trace mark 0: the chunk's keys done (mark 1: the reductions)
  // 4. the keys of a warp: threads with the same columns (lane bits >= log2(LPK)), butterfly
  //    (every head: a head past qn stays m = -inf, l = 0; no branch around the shuffles,
  //    which ptxas would otherwise route through its slow divergent-shuffle path)
#pragma unroll
  for (int o = LPK; o < 32; o <<= 1) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float mo = __shfl_xor_sync(0xffffffffu, m[g], o);
      const float lo = __shfl_xor_sync(0xffffffffu, l[g], o);
      const float mn = fmaxf(m[g], mo);
      const float a = m[g] == -INFINITY ? 0.f : exp2f(m[g] - mn);
      const float b = mo == -INFINITY ? 0.f : exp2f(mo - mn);
      l[g] = l[g] * a + lo * b;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[g][e] = acc[g][e] * a + __shfl_xor_sync(0xffffffffu, acc[g][e], o) * b;
      m[g] = mn;
    }
  }
  // 5. the warps, in turn: their (max, sum) first, then their outputs scaled to the tile's
  //    max through as many shared-memory slots as fit, summed in warp order
  float* wml = reinterpret_cast<float*>(c.scratch);  // [warp][G][max, sum]
  float* slot = wml + kWarps * G * 2;                 // [slots][qn * D]
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < G; ++g) wml[(warp * G + g) * 2] = m[g], wml[(warp * G + g) * 2 + 1] = l[g];
  ka_mk::sync<kThreads>();
  float tm[G], tl[G], fw[G];  // the tile's max and sum per head, this warp's factor
#pragma unroll
  for (int g = 0; g < G; ++g) {
    tm[g] = -INFINITY, tl[g] = 0.f, fw[g] = 0.f;
    if (g >= qn) continue;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) tm[g] = fmaxf(tm[g], wml[(w * G + g) * 2]);
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const float mw = wml[(w * G + g) * 2];
      if (mw != -INFINITY) tl[g] += exp2f(mw - tm[g]) * wml[(w * G + g) * 2 + 1];
    }
    fw[g] = m[g] == -INFINITY ? 0.f : exp2f(m[g] - tm[g]);
  }
  const int width = qn * D;  // floats of one warp's output
  constexpr int kSlotBytes = kScratchBytes - kWarps * G * 2 * 4;
  const int slots = kSlotBytes / (width * 4) < kWarps ? kSlotBytes / (width * 4) : kWarps;
  constexpr int kPer = G * D / kThreads > 0 ? G * D / kThreads : 1;  // elements per thread
  float tot[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) tot[k] = 0.f;
  for (int w0 = 0; w0 < kWarps; w0 += slots) {
    if (warp >= w0 && warp < w0 + slots && lane < LPK) {
      float* dst = slot + (warp - w0) * width + d0;
#pragma unroll
      for (int g = 0; g < G; ++g) {
        if (g >= qn) continue;
#pragma unroll
        for (int e = 0; e < 8; ++e) dst[g * D + e] = acc[g][e] * fw[g];
      }
    }
    ka_mk::sync<kThreads>();
    const int w1 = w0 + slots < kWarps ? w0 + slots : kWarps;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const int e = t + k * kThreads;
      if (e < width)
        for (int w = w0; w < w1; ++w) tot[k] += slot[(w - w0) * width + e];
    }
    ka_mk::sync<kThreads>();
  }
  c.mark(1);
  // 6. the chunk's partials
  float* po = c.ptr<float>(c.arg(AT_PO));
  float* pml = c.ptr<float>(c.arg(AT_PML));
  const size_t prow0 = static_cast<size_t>(c.arg(AT_PROW0));
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int e = t + k * kThreads;
    if (e >= width) continue;
    const int g = e / D, d = e % D;
    const size_t row = prow0 + static_cast<size_t>(q0 + g) * splits + split;
    po[row * D + d] = tot[k];
    if (d == 0) pml[row * 2] = pick(tm, g), pml[row * 2 + 1] = pick(tl, g);
  }
}

__device__ __forceinline__ bool attn_decode_any(const Ctx& c) {
  const int dim = c.arg(AT_DIM), dtype = c.arg(AT_DTYPE), qn = c.arg(AT_QN);
  if (qn < 1 || qn > kMaxQBlock || c.arg(AT_SPLITS) < 1) return false;
  if (dtype == 0) {
    if (dim == 64) return attn_decode<64, bf16>(c), true;
    if (dim == 128) return attn_decode<128, bf16>(c), true;
    if (dim == 256) return attn_decode<256, bf16>(c), true;
  } else if (dtype == 1) {
    if (dim == 64) return attn_decode<64, __half>(c), true;
    if (dim == 128) return attn_decode<128, __half>(c), true;
    if (dim == 256) return attn_decode<256, __half>(c), true;
  }
  return false;  // the interpreter reports the instruction (status bad_opcode)
}

// ---------------------------------------------------------------- ATTN_COMBINE

// q heads [q0, q0 + qn): each one's non-empty chunks reduced. A warp per head finds the
// chunks' weights exp2(m_c - M) / L (shared memory, [qn][chunks]), then every output element
// sums its chunks' outputs with them.
template <typename T>
__device__ __forceinline__ void attn_combine(const Ctx& c) {
  const int len =
      kv_length(c, c.arg(CB_LENGTH), c.arg(CB_LENGTH_OFF), c.arg(CB_LENGTH_ADD), c.arg(CB_CAP));
  const int splits = c.arg(CB_SPLITS);
  const int chunk = kv_chunk(len, splits, c.arg(CB_CHUNK));
  const int full = (len + chunk - 1) / chunk;
  const int nc = full < splits ? full : splits;
  const int dim = c.arg(CB_DIM), q0 = c.arg(CB_Q0), qn = c.arg(CB_QN);
  const size_t prow0 = static_cast<size_t>(c.arg(CB_PROW0));
  const float* po = c.ptr<const float>(c.arg(CB_PO));
  const float* pml = c.ptr<const float>(c.arg(CB_PML));
  T* out = c.ptr<T>(c.arg(CB_OUT)) + c.arg(CB_OUT_OFF);
  float* wt = reinterpret_cast<float*>(c.scratch);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int h = warp; h < qn; h += kWarps) {
    const size_t row0 = prow0 + static_cast<size_t>(q0 + h) * splits;
    float mx = -INFINITY;
    for (int k = lane; k < nc; k += 32) mx = fmaxf(mx, __ldcg(pml + (row0 + k) * 2));
    mx = warp_max(mx);
    float sum = 0.f;
    for (int k = lane; k < nc; k += 32)
      sum += exp2f(__ldcg(pml + (row0 + k) * 2) - mx) * __ldcg(pml + (row0 + k) * 2 + 1);
    sum = warp_sum(sum);
    for (int k = lane; k < nc; k += 32)
      wt[h * nc + k] = exp2f(__ldcg(pml + (row0 + k) * 2) - mx) / sum;
  }
  ka_mk::sync<kThreads>();
  for (int e = threadIdx.x; e < qn * dim; e += kThreads) {
    const int h = e / dim, d = e % dim;
    const float* src = po + (prow0 + static_cast<size_t>(q0 + h) * splits) * dim + d;
    float o = 0.f;
#pragma unroll 4
    for (int k = 0; k < nc; ++k)
      o = fmaf(wt[h * nc + k], __ldcg(src + static_cast<size_t>(k) * dim), o);
    out[static_cast<size_t>(q0 + h) * dim + d] = Num<T>::to(o);
  }
}

// The decode opcodes' dispatch: false for a dtype or head dim they do not have.
__device__ __forceinline__ bool run_decode(const Ctx& c) {
  switch (c.op()) {
    case EMBED: embed(c); return true;
    case ROPE_KV:
      if (c.arg(RK_DTYPE) == 0) return rope_kv<bf16>(c), true;
      if (c.arg(RK_DTYPE) == 1) return rope_kv<__half>(c), true;
      return false;
    case ATTN_DECODE: return attn_decode_any(c);
    case ATTN_COMBINE:
      if (c.arg(CB_DTYPE) == 0) return attn_combine<bf16>(c), true;
      if (c.arg(CB_DTYPE) == 1) return attn_combine<__half>(c), true;
      return false;
    default: return false;
  }
}

// The decode step's interpreter: the generic opcodes of Ops and the decode ones. Its own
// instantiation (ka_mk::kernel<DecodeOps>): the attention's registers (q and output columns
// of 4 heads, 2 x kKvPasses loads in flight) do not change the chain's kernel<Ops>, which
// stays at its own register count without spills.
struct DecodeOps {
  static constexpr int kThreads = Ops::kThreads;
  static constexpr int kPageBytes = Ops::kPageBytes;
  static constexpr int kScratchBytes = Ops::kScratchBytes;

  __device__ static bool run(const Ctx& c) { return c.op() < EMBED ? Ops::run(c) : run_decode(c); }
};

}  // namespace mk
