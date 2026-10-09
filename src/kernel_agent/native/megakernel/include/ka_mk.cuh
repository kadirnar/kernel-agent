// ka_mk.cuh: the megakernel runtime of kernel-agent's native engines (issue #225).
//
// One block per SM interprets a queue of instructions that Python scheduled ahead of time
// (kernel_agent/native/megakernel/schedule.py) and reuses across calls. Instructions wait
// for their inputs on global-memory counters, not on grid barriers: a producer tile adds 1
// to its chunk's counter after writing its output (red.release.gpu), a consumer polls the
// counter until it reaches its target (ld.acquire.gpu, or relaxed polls and an acquire
// fence). Every block is resident (cooperative launch attribute), so a waiting block never
// waits for one that has not started. No cooperative-groups grid sync anywhere.
//
// Weights stream through a shared-memory page pool, filled by a producer warp of its own
// (warp specialisation): it walks the queue ahead of the consumer threads and copies each
// instruction's weights into free pages, page by page, with at most `inflight` pages in
// flight (cp.async.bulk + mbarrier on sm_90+ including sm_120; cp.async + mbarrier on
// sm_80-sm_89; plain copies before). So the DRAM stream continues while the consumers wait on
// counters or compute, and the consumers' own loads (the activations on the critical path)
// do not queue behind a burst of weight copies. The instruction rows are staged in a small
// shared-memory ring a few rows ahead, the first tensor-table entries in shared memory.
//
// A project defines its opcodes in an `Ops` struct:
//
//   struct MyOps {
//     static constexpr int kThreads = 256;        // consumer threads (the producer warp extra)
//     static constexpr int kPageBytes = 8192;     // page size (multiple of 128)
//     static constexpr int kScratchBytes = 8192;  // per-block shared scratch for the opcodes
//     __device__ static bool run(const ka_mk::Ctx<kPageBytes>& ctx);  // false: bad opcode
//   };
//
// `run` executes one instruction with the consumer threads: ctx.op(), ctx.arg(k),
// ctx.ptr<T>(tensor) (the runtime's tensor table), ctx.weights(byte) (the instruction's
// weights in the pool), ctx.scratch, ctx.mark(k). It synchronises with
// `ka_mk::sync<kThreads>()` (a named barrier of the consumers), never __syncthreads(): the
// producer warp does not take part. The interpreter waits on the instruction's counters
// before `run` and signals its counter after it; `run` must not return early on some threads
// only. Data other blocks wrote in this launch (activations) is read after the acquire with
// plain or __ldcg loads; never with ld.global.nc (__ldg), whose cache can be stale.
//
// Host side (one-time setup, then one launch per call; capture it in a CUDA graph):
//
//   int pages = ka_mk::pool_pages<MyOps>();             // pages that fit this GPU's smem
//   int queues = ka_mk::queues<MyOps>(pages, stream);   // resident blocks: schedule for this
//   ka_mk::Params p = {program, counters, tensors, n_tensors, status, trace, timeout_ns,
//                      pages, inflight};
//   C10_CUDA_CHECK(ka_mk::launch<MyOps>(p, queues, stream));
//
// Watchdog: a wait longer than p.timeout_ns (globaltimer) sets the abort flag; every block
// stops, the first one writes the instruction id, counter, value and target into p.status
// (pinned host memory, readable after a hang without a CUDA call) and the kernel returns.
// The last block to finish zeroes the counters, so the next launch starts clean without a
// memset (also after an abort). A runtime's buffers (counters, activations, status) serve one
// launch at a time: launch its calls in stream order, never two at once (issue #248). KA_MK_TRACE (compile flag): per-instruction globaltimer
// stamps into p.trace (weights issued, wait start, counters met, weights landed, end; SM;
// two opcode marks).
//
// Needs sm_70+ (acquire / release, nanosleep); bf16 opcodes need sm_80+.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "ka_launch.cuh"

#ifndef KA_MK_TRACE
#define KA_MK_TRACE 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define KA_MK_BULK 1  // cp.async.bulk + mbarrier transaction counts
#else
#define KA_MK_BULK 0
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define KA_MK_MBARRIER 1  // mbarriers (sm_80: completed by cp.async.mbarrier.arrive)
#else
#define KA_MK_MBARRIER 0
#endif

namespace ka_mk {

// ---------------------------------------------------------------- ABI (schedule.py)

constexpr int kMagic = 0x4B414D4B;  // "KAMK"
constexpr int kVersion = 1;
constexpr int kWords = 32;  // int32 per instruction: opcode + 31 arguments
enum : int {
  H_MAGIC = 0, H_VERSION, H_QUEUES, H_INSTRS, H_COUNTERS, H_WAITS, H_WORDS,
  H_QUEUE_OFF, H_INSTR_OFF, H_WAIT_OFF, H_TOTAL
};
// An instruction's words: its first wait inline (W0_COUNTER -1: none), the others in the
// wait list; its signal; its prefetch; then the opcode's arguments.
enum : int {
  OP = 0, ID, W0_COUNTER, W0_TARGET, WAIT_FIRST, WAIT_COUNT, SIGNAL, SIGNAL_N, PF_TENSOR,
  PF_OFFSET, PF_BYTES, PF_HINT, ARG0
};
enum : int { kEvictNormal = 0, kEvictFirst = 1, kEvictLast = 2 };  // PF_HINT

// status words (pinned host memory) and codes (runtime.py)
enum : int {
  S_CODE = 0, S_INSTR, S_QUEUE, S_OPCODE, S_COUNTER, S_VALUE, S_TARGET, S_SM, S_WORDS = 16
};
enum : int { kOk = 0, kHang = 1, kBadProgram = 2, kBadOpcode = 3, kPoolTooSmall = 4 };

// words after the schedule's counters in the counter buffer
enum : int { C_DONE = 0, C_ABORT = 1, kInternal = 2 };

constexpr int kProducerThreads = 32;  // the producer warp, after the consumer threads
constexpr int kMaxPages = 32;         // full / empty barriers reserved in shared memory
constexpr int kChunkRows = 4;         // instruction rows staged per chunk (512 bytes)
constexpr int kChunks = 3;            // chunks in the row ring: the current one and two ahead
// trace columns: weights issued, wait start, counters met, weights landed, end (ns), the SM,
// and two marks an opcode sets inside its run (Ctx::mark: where its time goes)
constexpr int kTraceCols = 8;
// the first kTableCache entries of the tensor table, copied to shared memory at the start:
// an opcode's pointer lookup is no global load (a dependent L2 round trip before its data)
constexpr int kTableCache = 96;
// shared memory before the scratch: full and empty barriers (one per page), the consumers'
// broadcast word and abort flag, the row ring, the tensor-table cache
constexpr int kFullOffset = 0;
constexpr int kEmptyOffset = kMaxPages * 8;
constexpr int kFlagOffset = 2 * kMaxPages * 8;
constexpr int kRowsOffset = kFlagOffset + 128;
constexpr int kTableOffset = kRowsOffset + kChunks * kChunkRows * kWords * 4;
constexpr int kHeadBytes = kTableOffset + kTableCache * 8;

struct Params {
  const int* program;                 // the serialised schedule
  int* counters;                      // counters + kInternal words, zero before the first launch
  const unsigned long long* tensors;  // tensor table: device pointers
  int n_tensors;                      // its length
  int* status;                        // S_WORDS int32 of pinned host memory, or nullptr
  long long* trace;                   // [instructions, kTraceCols], or nullptr
  long long timeout_ns;               // watchdog: the longest wait on one counter
  int n_pages;                        // pages in the pool (sized by pool_pages)
  int inflight;                       // pages the producer keeps in flight (0: n_pages)
};

// ---------------------------------------------------------------- memory-model primitives

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ int ld_relaxed(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void red_release(int* p, int n) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(n) : "memory");
}

__device__ __forceinline__ int atom_add_acq_rel(int* p, int n) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
               : "=r"(old) : "l"(p), "r"(n) : "memory");
  return old;
}

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int smid() {
  int id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// The consumer threads' barrier (named barrier 1; the producer warp keeps going). Opcodes
// use it instead of __syncthreads().
template <int Threads>
__device__ __forceinline__ void sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(Threads) : "memory");
}

// Spin (one thread) until *counter >= target: true; false when another block aborted or the
// wait exceeded timeout_ns. A counter already met costs one acquire read. Otherwise the polls
// are relaxed reads and the one that sees the target is followed by fence.acq_rel.gpu: a
// PTX acquire pattern, which orders the block's later reads after the producers' released
// writes without another trip to L2. Tight spins first (a dependency is usually a
// microsecond away), then nanosleep back-off up to 256 ns.
__device__ __forceinline__ bool wait(const int* counter, int target, long long timeout_ns,
                                     const int* abort) {
  if (ld_acquire(counter) >= target) return true;
  const unsigned long long t0 = now_ns();
  unsigned sleep_ns = 0;
  for (unsigned spin = 1; ld_relaxed(counter) < target; ++spin) {
    if (sleep_ns) __nanosleep(sleep_ns);
    if ((spin & 63u) == 0) {
      sleep_ns = sleep_ns ? (sleep_ns < 256u ? 2u * sleep_ns : 256u) : 32u;
      if (ld_relaxed(abort)) return false;
      if (static_cast<long long>(now_ns() - t0) > timeout_ns) return false;
    }
  }
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
  return true;
}

// One thread, after the block's writes are done (a consumer barrier before it): make them
// visible to the waiters of `counter` and add n.
__device__ __forceinline__ void signal(int* counter, int n) { red_release(counter, n); }

// ---------------------------------------------------------------- page barriers

// Each page has a "full" barrier (its weights landed) and an "empty" one (the consumers are
// done with it). Producer and consumers both walk the pages in ring order and track the
// parity of each page's next phase in a bitmask, so neither needs to tell the other which.
namespace detail {

__device__ __forceinline__ void bar_init(uint64_t* b, int count) {
#if KA_MK_MBARRIER
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(b)), "r"(count)
               : "memory");
#else
  *reinterpret_cast<volatile int*>(b) = 0;  // a phase counter instead (sm_70 / sm_75)
  (void)count;
#endif
}

// Wait until phase `parity` of barrier b completed (every calling thread).
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t parity) {
#if KA_MK_BULK
  uint32_t done = 0;
  while (!done)
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(smem_addr(b)), "r"(parity)
        : "memory");
#elif KA_MK_MBARRIER
  uint32_t done = 0;
  while (!done)
    asm volatile(
        "{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(smem_addr(b)), "r"(parity)
        : "memory");
#else
  // phase counter: phase k completed when the counter passed k (parity: k's low bit)
  volatile int* c = reinterpret_cast<volatile int*>(b);
  while ((*c & 1) == static_cast<int>(parity)) {
  }
  __threadfence_block();
#endif
}

// One arrival on barrier b (count 1: completes the phase).
__device__ __forceinline__ void bar_arrive(uint64_t* b) {
#if KA_MK_MBARRIER
  asm volatile("{ .reg .b64 st; mbarrier.arrive.shared::cta.b64 st, [%0]; }" ::"r"(smem_addr(b))
               : "memory");
#else
  __threadfence_block();
  atomicAdd(reinterpret_cast<int*>(b), 1);
#endif
}

#if KA_MK_MBARRIER
// The L2 cache policy of a prefetch's PF_HINT: evict_first for weights streamed once per call
// (they do not push the program, the activations and the next call's data out of L2),
// evict_last for weights that should stay L2-resident across calls.
__device__ __forceinline__ uint64_t l2_policy(int hint) {
  uint64_t policy;
  if (hint == kEvictFirst)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  else if (hint == kEvictLast)
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  else
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}
#endif

// The producer warp: copy `n` bytes from `src` into page `dst`, completing barrier `full`.
// sm_90+: one bulk copy by lane 0 (transaction count n); sm_80-sm_89: 16-byte cp.async by
// the 32 lanes, each lane's completion an arrival (count 32); before: plain copies.
__device__ __forceinline__ void fill(unsigned char* dst, const unsigned char* src, int n,
                                     uint64_t* full, int hint, int lane) {
#if KA_MK_BULK
  if (lane == 0) {
    const uint64_t policy = l2_policy(hint);
    const uint32_t bar = smem_addr(full);
    asm volatile(
        "{ .reg .b64 st; mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1; }" ::"r"(bar),
        "r"(n)
        : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;" ::"r"(smem_addr(dst)),
        "l"(src), "r"(n), "r"(bar), "l"(policy)
        : "memory");
  }
#elif KA_MK_MBARRIER
  const uint64_t policy = l2_policy(hint);
  for (int off = lane * 16; off < n; off += kProducerThreads * 16)
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(
                     smem_addr(dst + off)),
                 "l"(src + off), "l"(policy)
                 : "memory");
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_addr(full))
               : "memory");
#else
  (void)hint;
  for (int off = lane * 16; off < n; off += kProducerThreads * 16)
    *reinterpret_cast<uint4*>(dst + off) = __ldg(reinterpret_cast<const uint4*>(src + off));
  __syncwarp();
  if (lane == 0) bar_arrive(full);
#endif
}

// The consumers' arrivals that free a page: every consumer thread arrives once its own reads
// of the page are done (CUTLASS's consumer_release pattern; one arrival per warp after a
// barrier is as correct but racecheck reports it as a WAR hazard against the next fill), or
// thread 0 alone where a phase counter stands in for the mbarrier (before sm_80).
__device__ __forceinline__ constexpr int empty_count(int consumers) {
#if KA_MK_MBARRIER
  return consumers;
#else
  (void)consumers;
  return 1;
#endif
}

__device__ __forceinline__ bool empty_arrives(int t) {
#if KA_MK_MBARRIER
  return t >= 0;
#else
  return t == 0;
#endif
}

__device__ __forceinline__ constexpr int full_count() {
#if KA_MK_BULK
  return 1;  // the producer's arrive.expect_tx; the bytes complete the phase
#elif KA_MK_MBARRIER
  return kProducerThreads;  // one cp.async.mbarrier.arrive.noinc per producer lane
#else
  return 1;
#endif
}

}  // namespace detail

// The instruction being run: its words (staged in shared memory), the tensor table, its
// weights in the pool.
template <int PageBytes>
struct Ctx {
  const int* w;
  const unsigned long long* tensors;  // the shared-memory copy of the table's first entries
  const unsigned long long* table;    // the whole table (global memory)
  unsigned char* pool;  // page 0
  int first_page;
  int n_pages;
  unsigned char* scratch;
  int queue;
  int instr;
  long long* trace;  // KA_MK_TRACE builds with a trace buffer, else nullptr

  __device__ __forceinline__ int op() const { return w[OP]; }
  __device__ __forceinline__ int arg(int k) const { return w[ARG0 + k]; }
  __device__ __forceinline__ float arg_f(int k) const { return __int_as_float(arg(k)); }
  template <typename T>
  __device__ __forceinline__ T* ptr(int tensor) const {
    return reinterpret_cast<T*>(tensor < kTableCache ? tensors[tensor] : table[tensor]);
  }
  // Byte `byte` of the instruction's weights (pages are not contiguous where the ring wraps:
  // never let one access straddle a PageBytes boundary).
  __device__ __forceinline__ const unsigned char* weights(int byte) const {
    int page = first_page + byte / PageBytes;  // < 2 * n_pages: one subtraction wraps it
    if (page >= n_pages) page -= n_pages;
    return pool + page * PageBytes + byte % PageBytes;
  }
  // Trace mark k (0 or 1) of this instruction: the globaltimer now (thread 0; KA_MK_TRACE).
  __device__ __forceinline__ void mark(int k) const {
#if KA_MK_TRACE
    if (trace != nullptr && threadIdx.x == 0)
      trace[static_cast<size_t>(instr) * kTraceCols + 6 + k] = static_cast<long long>(now_ns());
#else
    (void)k;
#endif
  }
};

namespace detail {

// The row ring: chunk c of the queue's rows (kChunkRows rows, 32 uint4) lives in slot
// c % kChunks. Thread t < 32 moves uint4 t of a chunk.
__device__ __forceinline__ uint4 fetch_chunk(const int* code, int begin, int end, int chunk,
                                             int t) {
  const int row = begin + chunk * kChunkRows + t / 8;
  if (t >= kChunkRows * 8 || row >= end) return make_uint4(0, 0, 0, 0);
  return __ldg(reinterpret_cast<const uint4*>(code + static_cast<size_t>(row) * kWords) + t % 8);
}

__device__ __forceinline__ void put_chunk(int* rows, int chunk, int t, uint4 v) {
  if (t < kChunkRows * 8)
    reinterpret_cast<uint4*>(rows + (chunk % kChunks) * kChunkRows * kWords)[t] = v;
}

__device__ __forceinline__ const int* row_of(const int* rows, int rel) {
  return rows + ((rel / kChunkRows) % kChunks) * kChunkRows * kWords + (rel % kChunkRows) * kWords;
}

// The first block to abort reports why (pinned host memory, written through to the host).
__device__ __forceinline__ void report(const Params& p, int* abort, int code, int instr, int queue,
                                       int opcode, int counter, int value, int target) {
  if (atomicCAS(abort, 0, 1) != 0 || p.status == nullptr) return;
  volatile int* s = p.status;
  s[S_INSTR] = instr;
  s[S_QUEUE] = queue;
  s[S_OPCODE] = opcode;
  s[S_COUNTER] = counter;
  s[S_VALUE] = value;
  s[S_TARGET] = target;
  s[S_SM] = smid();
  __threadfence_system();
  s[S_CODE] = code;
  __threadfence_system();
}

// Thread 0: wait for every counter of instruction w (the inline one, then the list).
__device__ __forceinline__ bool wait_all(const Params& p, const int* w, const int* waits,
                                         int* internal, int i, int q) {
  const int n_list = w[WAIT_COUNT];
  for (int k = -1; k < n_list; ++k) {
    const int c = k < 0 ? w[W0_COUNTER] : __ldg(waits + 2 * (w[WAIT_FIRST] + k));
    if (c < 0) continue;
    const int target = k < 0 ? w[W0_TARGET] : __ldg(waits + 2 * (w[WAIT_FIRST] + k) + 1);
    if (!wait(p.counters + c, target, p.timeout_ns, internal + C_ABORT)) {
      report(p, internal + C_ABORT, kHang, i, q, w[OP], c, ld_relaxed(p.counters + c), target);
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ int pages_of(int tensor, int bytes, int page_bytes) {
  return (tensor >= 0 && bytes > 0) ? (bytes + page_bytes - 1) / page_bytes : 0;
}

}  // namespace detail

// A trace stamp: `value` is evaluated only in KA_MK_TRACE builds with a trace buffer.
#if KA_MK_TRACE
#define KA_MK_STAMP(p, instr, col, value)                                             \
  do {                                                                                \
    if ((p).trace != nullptr)                                                         \
      (p).trace[static_cast<size_t>(instr) * ::ka_mk::kTraceCols + (col)] = (value); \
  } while (0)
#else
#define KA_MK_STAMP(p, instr, col, value) \
  do {                                    \
  } while (0)
#endif

// ---------------------------------------------------------------- the producer warp

// The consumers' stop flag, read and written atomically (a polled flag, not a race).
__device__ __forceinline__ bool stopped(int* flag) { return atomicAdd(flag, 0) != 0; }

// Walk the queue's instructions in order and copy each one's weights into the pool's pages in
// ring order: a page is reused once the consumers released it (its empty barrier), and at
// most `inflight` pages are in flight at once (the oldest one's full barrier).
template <int PB>
__device__ __forceinline__ void produce(const Params& p, const int* code, int begin, int end,
                                        unsigned char* pool, uint64_t* full, uint64_t* empty,
                                        const unsigned long long* tensors, int* abort,
                                        int lane) {
  const int n_pages = p.n_pages;
  const int cap = p.inflight > 0 && p.inflight < n_pages ? p.inflight : n_pages;
  uint32_t used = 0;          // pages filled at least once
  uint32_t empty_phase = 0;   // per page: parity of the empty phase to wait for next
  uint32_t fills = 0;         // per page: parity of its number of fills so far (its latest
                              // fill completes the full phase of parity fills ^ 1)
  int page = 0, oldest = 0, flying = 0;
  for (int j = begin; j < end && !stopped(abort); ++j) {
    const int* row = code + static_cast<size_t>(j) * kWords;
    const int tensor = __ldg(row + PF_TENSOR), bytes = __ldg(row + PF_BYTES);
    if (detail::pages_of(tensor, bytes, PB) > n_pages) break;  // the consumers report it
    if (tensor < 0 || bytes <= 0) continue;
    const unsigned long long base = tensor < kTableCache ? tensors[tensor] : p.tensors[tensor];
    const unsigned char* src = reinterpret_cast<const unsigned char*>(base) +
                               static_cast<size_t>(__ldg(row + PF_OFFSET)) * 16;
    const int hint = __ldg(row + PF_HINT);
    for (int done = 0; done < bytes && !stopped(abort); done += PB) {
      if (flying == cap) {  // the oldest page in flight lands first
        detail::bar_wait(full + oldest, ((fills >> oldest) & 1u) ^ 1u);
        oldest = oldest + 1 == n_pages ? 0 : oldest + 1;
        --flying;
      }
      if (used & (1u << page)) {  // its previous instruction's consumers are done with it
        const uint32_t parity = (empty_phase >> page) & 1u;
        while (!stopped(abort)) {
          uint32_t ok = 1;
#if KA_MK_BULK
          asm volatile(
              "{ .reg .pred q; mbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2; "
              "selp.u32 %0, 1, 0, q; }"
              : "=r"(ok)
              : "r"(smem_addr(empty + page)), "r"(parity)
              : "memory");
#elif KA_MK_MBARRIER
          asm volatile(
              "{ .reg .pred q; mbarrier.test_wait.parity.shared::cta.b64 q, [%1], %2; "
              "selp.u32 %0, 1, 0, q; }"
              : "=r"(ok)
              : "r"(smem_addr(empty + page)), "r"(parity)
              : "memory");
#else
          ok = (*reinterpret_cast<volatile int*>(empty + page) & 1) != static_cast<int>(parity);
#endif
          if (ok) break;
        }
        if (stopped(abort)) break;
        empty_phase ^= 1u << page;
#if KA_MK_BULK
        // the consumers' generic reads of the page before the async proxy's writes
        if (lane == 0) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
      }
      if (done == 0 && lane == 0) KA_MK_STAMP(p, j, 0, static_cast<long long>(now_ns()));
      const int n = bytes - done < PB ? bytes - done : PB;
      detail::fill(pool + page * PB, src + done, n, full + page, hint, lane);
      fills ^= 1u << page;
      used |= 1u << page;
      page = page + 1 == n_pages ? 0 : page + 1;
      ++flying;
    }
  }
  // every fill in flight lands before the block's shared memory goes away (after an abort the
  // consumers may not have waited for them)
  while (flying > 0) {
    detail::bar_wait(full + oldest, ((fills >> oldest) & 1u) ^ 1u);
    oldest = oldest + 1 == n_pages ? 0 : oldest + 1;
    --flying;
  }
}

// ---------------------------------------------------------------- the interpreter

template <class Ops>
__device__ __forceinline__ void interpret(const Params& p) {
  constexpr int PB = Ops::kPageBytes;
  constexpr int T = Ops::kThreads;
  static_assert(PB % 128 == 0, "kPageBytes must be a multiple of 128");
  static_assert(T % 32 == 0 && T >= kChunks * kChunkRows * 8, "96+ consumer threads, whole warps");
  extern __shared__ __align__(128) unsigned char ka_mk_smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(ka_mk_smem + kFullOffset);
  uint64_t* empty = reinterpret_cast<uint64_t*>(ka_mk_smem + kEmptyOffset);
  volatile int* flag = reinterpret_cast<volatile int*>(ka_mk_smem + kFlagOffset);
  int* aborted_here = const_cast<int*>(flag) + 1;  // the consumers stopped: the producer too
  int* rows = reinterpret_cast<int*>(ka_mk_smem + kRowsOffset);
  unsigned long long* tensors = reinterpret_cast<unsigned long long*>(ka_mk_smem + kTableOffset);
  unsigned char* scratch = ka_mk_smem + kHeadBytes;
  unsigned char* pool = scratch + (Ops::kScratchBytes + 127) / 128 * 128;

  const int* prog = p.program;
  if (__ldg(prog + H_MAGIC) != kMagic || __ldg(prog + H_VERSION) != kVersion ||
      __ldg(prog + H_WORDS) != kWords || __ldg(prog + H_INSTR_OFF) % kWords != 0 ||
      p.n_pages < 1 || p.n_pages > kMaxPages) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && p.status != nullptr) {
      p.status[S_CODE] = kBadProgram;
      __threadfence_system();
    }
    return;  // uniform: no block runs anything, the counters stay zero
  }
  const int n_counters = __ldg(prog + H_COUNTERS);
  int* internal = p.counters + n_counters;
  const int* table = prog + __ldg(prog + H_QUEUE_OFF);
  const int* code = prog + __ldg(prog + H_INSTR_OFF);
  const int* waits = prog + __ldg(prog + H_WAIT_OFF);
  const int q = blockIdx.x;
  const bool mine = q < __ldg(prog + H_QUEUES);
  const int begin = mine ? __ldg(table + q) : 0;
  const int end = mine ? __ldg(table + q + 1) : 0;
  const int t = threadIdx.x;

  // the first kChunks chunks of rows into the ring, the next one into registers; the
  // tensor-table cache; the page barriers
  if (t < kChunks * kChunkRows * 8)
    detail::put_chunk(rows, t / 32, t % 32, detail::fetch_chunk(code, begin, end, t / 32, t % 32));
  uint4 staged = detail::fetch_chunk(code, begin, end, kChunks, t);
  for (int k = t; k < kTableCache && k < p.n_tensors; k += T + kProducerThreads)
    tensors[k] = __ldg(p.tensors + k);
  if (t == 0) {
    for (int k = 0; k < p.n_pages; ++k) {
      detail::bar_init(full + k, detail::full_count());
      detail::bar_init(empty + k, detail::empty_count(T));
    }
    *aborted_here = 0;
#if KA_MK_BULK
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
  }
  __syncthreads();

  if (t >= T) {  // the producer warp
    produce<PB>(p, code, begin, end, pool, full, empty, tensors, aborted_here, t - T);
  } else {
    // the consumers
    uint32_t phase = 0;  // per page: parity of the full phase to wait for next
    uint32_t freed = 0;  // per page: parity of the empty phase this thread arrives on next
    uint32_t owed = 0;   // per page: an arrival of this thread whose phase it has not observed
    int tail = 0;        // the ring page of the next instruction's first weights
    bool aborted = false;
    for (int i = begin; i < end; ++i) {
      const int rel = i - begin;
      if (rel % kChunkRows == 0 && rel > 0) {
        // chunk c starts: chunk c - 1's slot (read before the last consumer barrier) takes
        // chunk c + 2 from the registers, which load chunk c + 3
        const int c = rel / kChunkRows;
        detail::put_chunk(rows, c + kChunks - 1, t, staged);
        staged = detail::fetch_chunk(code, begin, end, c + kChunks, t);
      }
      const int* w = detail::row_of(rows, rel);
      const int need = detail::pages_of(w[PF_TENSOR], w[PF_BYTES], PB);
      // the words used after the run, read now: once the run's last barrier is passed, warp 0
      // may already refill this row's slot of the ring for a later chunk
      const int op = w[OP], sig = w[SIGNAL], sig_n = w[SIGNAL_N];
      if (need > p.n_pages) {
        if (t == 0)
          detail::report(p, internal + C_ABORT, kPoolTooSmall, i, q, w[OP], -1, w[PF_BYTES],
                         p.n_pages * PB);
        aborted = true;
        break;
      }
      // 1. the instruction's counters (thread 0 waits; the barrier hands its acquire to all)
      if (t == 0) {
        KA_MK_STAMP(p, i, 1, static_cast<long long>(now_ns()));
        KA_MK_STAMP(p, i, 5, smid());
      }
      if (w[W0_COUNTER] >= 0 || w[WAIT_COUNT] > 0) {
        if (t == 0) *flag = detail::wait_all(p, w, waits, internal, i, q);
        sync<T>();
        if (!*flag) {
          aborted = true;
          break;
        }
      }
      if (t == 0) KA_MK_STAMP(p, i, 2, static_cast<long long>(now_ns()));
      // 2. its weights landed (every thread waits: the copies become visible to it)
      for (int k = 0; k < need; ++k) {
        const int page = tail + k < p.n_pages ? tail + k : tail + k - p.n_pages;
        detail::bar_wait(full + page, (phase >> page) & 1u);
        phase ^= 1u << page;
      }
      if (t == 0) KA_MK_STAMP(p, i, 3, static_cast<long long>(now_ns()));
      // 3. run it, 4. release its pages, signal its counter
      const Ctx<PB> ctx{w, tensors, p.tensors, pool, tail, p.n_pages, scratch, q, i, p.trace};
      const bool known = Ops::run(ctx);
      sync<T>();
      if (t == 0) {
        // the signal first: the waiting blocks are on the critical path, the producer is not
        if (!known) {
          detail::report(p, internal + C_ABORT, kBadOpcode, i, q, op, -1, 0, 0);
        } else if (sig >= 0) {
          signal(p.counters + sig, sig_n);
        }
        KA_MK_STAMP(p, i, 4, static_cast<long long>(now_ns()));
      }
      // its pages are free: every consumer thread is done with them. Before a thread arrives
      // on a page's empty barrier again it observes the phase its previous arrival there
      // belonged to: long complete by then (the producer waited for it before the refill this
      // instruction read), so the wait is one test. Without it compute-sanitizer synccheck
      // reports "Missing wait" at these arrivals and stops the warp (measured on an A10,
      // sm_86, compute-sanitizer 2025.2.1: the example from 6 layers on with its 11-page pool;
      // 151552 errors at 28 layers), and the integration refuses a correct megakernel.
      if (detail::empty_arrives(t))
        for (int k = 0; k < need; ++k) {
          const int page = tail + k < p.n_pages ? tail + k : tail + k - p.n_pages;
          if (owed & (1u << page)) detail::bar_wait(empty + page, ((freed >> page) & 1u) ^ 1u);
          detail::bar_arrive(empty + page);
          freed ^= 1u << page;
          owed |= 1u << page;
        }
      tail = (tail + need) % p.n_pages;
      if (!known) {
        aborted = true;
        break;
      }
    }
    if (aborted && t == 0) atomicExch(aborted_here, 1);  // the producer stops and drains
  }

  // the last block to finish zeroes the counters for the next launch
  __syncthreads();
  if (t == 0) *flag = atom_add_acq_rel(internal + C_DONE, 1) == static_cast<int>(gridDim.x) - 1;
  __syncthreads();
  if (*flag) {
    for (int c = t; c < n_counters; c += T + kProducerThreads) p.counters[c] = 0;
    if (t == 0) {
      internal[C_ABORT] = 0;
      internal[C_DONE] = 0;
    }
  }
}

template <class Ops>
__global__ void __launch_bounds__(Ops::kThreads + kProducerThreads, 1) kernel(Params p) {
  interpret<Ops>(p);
}

// ---------------------------------------------------------------- host side

template <class Ops>
constexpr size_t smem_bytes(int n_pages) {
  return kHeadBytes + (Ops::kScratchBytes + 127) / 128 * 128 +
         static_cast<size_t>(n_pages) * Ops::kPageBytes;
}

// Pages of Ops::kPageBytes that fit the current device's opt-in shared memory per block
// (up to `max_pages`; 0 when not even one does).
template <class Ops>
inline int pool_pages(int max_pages = kMaxPages) {
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  cudaFuncAttributes attr = {};
  if (cudaFuncGetAttributes(&attr, kernel<Ops>) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  const long long room = static_cast<long long>(optin) -
                         static_cast<long long>(attr.sharedSizeBytes) -
                         static_cast<long long>(smem_bytes<Ops>(0));
  const long long pages = room > 0 ? room / Ops::kPageBytes : 0;
  const int cap = max_pages < kMaxPages ? max_pages : kMaxPages;
  return static_cast<int>(pages < cap ? pages : cap);
}

// Blocks of the interpreter that can be resident on the SMs of `stream` with `n_pages` pages
// each (ka_coresident_blocks: green-context aware): schedule for this many queues.
template <class Ops>
inline int queues(int n_pages, cudaStream_t stream) {
  const size_t smem = smem_bytes<Ops>(n_pages);
  if (cudaFuncSetAttribute(kernel<Ops>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return ka_coresident_blocks(kernel<Ops>, Ops::kThreads + kProducerThreads, smem, stream);
}

// One launch of the interpreter: `n_queues` blocks (the schedule's queues), cooperative so
// that every block is resident. cudaErrorCooperativeLaunchTooLarge when they do not fit.
template <class Ops>
inline cudaError_t launch(const Params& p, int n_queues, cudaStream_t stream, bool pdl = false) {
  const size_t smem = smem_bytes<Ops>(p.n_pages);
  cudaError_t err = cudaFuncSetAttribute(kernel<Ops>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem));
  if (err != cudaSuccess) return err;
  KaLaunch opt;
  opt.cooperative = true;
  opt.pdl = pdl;
  return ka_launch(kernel<Ops>, dim3(n_queues), dim3(Ops::kThreads + kProducerThreads), smem,
                   stream, opt, p);
}

}  // namespace ka_mk
