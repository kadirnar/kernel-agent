"""Direct cuBLASLt FP8 matmul from C++ (load_inline): heuristic probe per scale mode,
algo timing, and a cached call path (descriptors + chosen algo per shape and mode).

Y[M, N] (row-major bf16) = X[M, K] @ W[N, K]^T. In cuBLASLt's column-major terms:
D (N x M, ld N) = op(A) x B with A = W (K x N col-major, ld K, transposed: "TN", the
layout FP8 needs) and B = X (K x M col-major, ld K). So cuBLASLt's A scale is the
weight's and its B scale the activations'.
"""

from __future__ import annotations

import os

from kernel_agent import toolchain

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <map>
#include <tuple>
#include <vector>

static cublasLtHandle_t lt() {
  static cublasLtHandle_t h = nullptr;
  if (!h) TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS);
  return h;
}

static const size_t WS = 32u << 20;
static void* workspace() {
  static at::Tensor ws;
  if (!ws.defined()) ws = at::empty({(int64_t)WS}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  return ws.data_ptr();
}

struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  int best = 0;
  int status = 0;
};

using Key = std::tuple<int64_t, int64_t, int64_t, int, int, int>;
static std::map<Key, Plan> plans;

static Plan& plan(int64_t M, int64_t N, int64_t K, int modeA, int modeB, int batch) {
  Key key{M, N, K, modeA, modeB, batch};
  auto it = plans.find(key);
  if (it != plans.end()) return it->second;
  Plan& p = plans[key];
  cublasStatus_t s;
  s = cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (s) { p.status = 1000 + s; return p; }
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtMatmulMatrixScale_t ma = (cublasLtMatmulMatrixScale_t)modeA, mb = (cublasLtMatmulMatrixScale_t)modeB;
  s = cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &ma, sizeof(ma));
  if (s) { p.status = 2000 + s; return p; }
  s = cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mb, sizeof(mb));
  if (s) { p.status = 3000 + s; return p; }
  {  // non-null, 256-byte aligned scale pointers for the heuristic (real ones set per call)
    const void* dummy = (const char*)workspace() + (WS - (1u << 20));
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummy, sizeof(dummy));
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummy, sizeof(dummy));
  }
  cublasLtMatrixLayoutCreate(&p.la, CUDA_R_8F_E4M3, K / batch, N, K);
  cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_8F_E4M3, K / batch, M, K);
  cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, N, M, N);
  if (batch > 1) {  // split-K by batching over K halves (see memory note): A/B offsets K/batch
    int32_t b = batch; int64_t ka = K / batch, kb = K / batch, kc = (int64_t)M * N;
    cublasLtMatrixLayoutSetAttribute(p.la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b));
    cublasLtMatrixLayoutSetAttribute(p.lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b));
    cublasLtMatrixLayoutSetAttribute(p.lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b));
    cublasLtMatrixLayoutSetAttribute(p.la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ka, sizeof(ka));
    cublasLtMatrixLayoutSetAttribute(p.lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &kb, sizeof(kb));
    cublasLtMatrixLayoutSetAttribute(p.lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &kc, sizeof(kc));
  }
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = WS;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p.algos.resize(16);
  int got = 0;
  s = cublasLtMatmulAlgoGetHeuristic(lt(), p.desc, p.la, p.lb, p.lc, p.lc, pref, 16, p.algos.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  if (s) { p.status = 4000 + s; p.algos.clear(); return p; }
  p.algos.resize(got);
  return p;
}

static cublasStatus_t launch(Plan& p, int algo, const at::Tensor& w, const at::Tensor& x,
                             const at::Tensor& sw, const at::Tensor& sx, at::Tensor& y) {
  const void* pa = sw.data_ptr();
  const void* pb = sx.data_ptr();
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb));
  float alpha = 1.f, beta = 0.f;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  return cublasLtMatmul(lt(), p.desc, &alpha, w.data_ptr(), p.la, x.data_ptr(), p.lb, &beta,
                        y.data_ptr(), p.lc, y.data_ptr(), p.lc, &p.algos[algo].algo,
                        workspace(), WS, stream);
}

// Heuristic support: (status, number of algos) for a shape and scale modes.
std::vector<int64_t> probe(int64_t M, int64_t N, int64_t K, int64_t modeA, int64_t modeB, int64_t batch) {
  Plan& p = plan(M, N, K, (int)modeA, (int)modeB, (int)batch);
  return {p.status, (int64_t)p.algos.size()};
}

// Time every heuristic algo (min over `rounds` interleaved rounds of `iters` calls); keep the best.
std::vector<double> tune(at::Tensor w, at::Tensor x, at::Tensor sw, at::Tensor sx, at::Tensor y,
                         int64_t modeA, int64_t modeB, int64_t batch, int64_t iters, int64_t rounds) {
  int64_t M = x.size(0), N = w.size(0), K = w.size(1);
  Plan& p = plan(M, N, K, (int)modeA, (int)modeB, (int)batch);
  TORCH_CHECK(p.status == 0 && !p.algos.empty(), "no algo: status ", p.status);
  int n = (int)p.algos.size();
  std::vector<double> best(n, 1e30);
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  for (int r = 0; r < rounds; ++r) {
    for (int i = 0; i < n; ++i) {
      if (launch(p, i, w, x, sw, sx, y) != CUBLAS_STATUS_SUCCESS) { best[i] = -1; continue; }
      cudaEventRecord(a, stream);
      for (int t = 0; t < iters; ++t) launch(p, i, w, x, sw, sx, y);
      cudaEventRecord(b, stream);
      cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b);
      best[i] = std::min(best[i], 1000.0 * ms / iters);
    }
  }
  int arg = 0;
  for (int i = 1; i < n; ++i) if (best[i] > 0 && (best[arg] < 0 || best[i] < best[arg])) arg = i;
  p.best = arg;
  cudaEventDestroy(a); cudaEventDestroy(b);
  return best;
}

// The cached call: plan lookup + scale pointers + cublasLtMatmul (no allocation).
void run(at::Tensor w, at::Tensor x, at::Tensor sw, at::Tensor sx, at::Tensor y,
         int64_t modeA, int64_t modeB, int64_t batch) {
  Plan& p = plan(x.size(0), w.size(0), w.size(1), (int)modeA, (int)modeB, (int)batch);
  TORCH_CHECK(launch(p, p.best, w, x, sw, sx, y) == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed");
}

int64_t version() { return (int64_t)cublasLtGetVersion(); }

// at::_scaled_mm called from C++ (no Python dispatch): its own host cost.
at::Tensor aten_scaled_mm(at::Tensor x, at::Tensor wt, at::Tensor sx, at::Tensor sw) {
  return at::_scaled_mm(x, wt, sx, sw, std::nullopt, std::nullopt, at::kBFloat16, false);
}

// n cached cuBLASLt calls behind one binding (a layer's GEMMs from one C++ launcher).
void run_n(at::Tensor w, at::Tensor x, at::Tensor sw, at::Tensor sx, at::Tensor y, int64_t n) {
  Plan& p = plan(x.size(0), w.size(0), w.size(1), 0, 0, 1);
  for (int64_t i = 0; i < n; ++i) launch(p, p.best, w, x, sw, sx, y);
}

// An empty op through the same binding path: the floor of a pybind11 call.
void noop(at::Tensor w, at::Tensor x, at::Tensor sw, at::Tensor sx, at::Tensor y,
          int64_t modeA, int64_t modeB, int64_t batch) {}
"""


def load():
    tc = toolchain.setup()
    from torch.utils.cpp_extension import load_inline

    lib = os.path.join(tc.cuda_home, "lib")
    return load_inline(
        name="ka132_lt",
        cpp_sources=SRC,
        functions=["probe", "tune", "run", "version", "noop", "aten_scaled_mm", "run_n"],
        with_cuda=True,
        extra_ldflags=[f"-L{lib}", "-lcublasLt"],
        extra_cflags=["-O3"],
        verbose=False,
    )
