"""Example candidate (CUDA C++ via load_inline): a chain of dependent decode GEMVs in one
CUDA graph, with programmatic dependent launch (PDL) between the layers.

Reference: ``reference.layers``, an ``nn.ModuleList`` of square bias-free bf16 ``nn.Linear``
applied in sequence (``kernel_agent.selftest.GemvChain``: a stand-in for any stack of short
memory-bound kernels, e.g. a decoder's per-token projections). Any hidden size that is a
multiple of 256 up to 4096 and any number of layers; one-row (decode) calls run:

* one kernel per layer, one warp per output row. The row's weights are loaded into
  registers (``ld.global.nc.L1::no_allocate``) *before* ``ka_pdl_wait()``; with PDL the
  next layer's grid launches while this one drains and streams its own weights meanwhile,
  and reads x (the previous layer's output) only after the wait (the PDL idiom of
  ``ka_launch.cuh``, ``kernel_agent.concurrency.include_dir()``);
* every layer captured once, in ``build()``, into a CUDA graph (``ka_launch`` with
  ``opt.pdl``: the captured launches keep programmatic edges); a call copies x into the
  graph's input buffer, replays the graph and returns a copy of its output.

Other row counts take the reference math (``F.linear`` per layer: a fallback).

The kernel is the one measured in docs/PARALLEL.md §4.6 (RTX 5070 Ti, 28 layers of
[1024, 1024] bf16, 59 MB: streamed from DRAM): eager launches 231.6 us, one CUDA graph
101.2 us, one graph with PDL edges 77.7 us; the DRAM floor (one launch over all rows) is
76.7 us. A graph kernel boundary costs ~0.9 us, which PDL hides. ``pdl`` is a ``build()``
keyword, so ``sweep_candidate`` can compare both.
"""

import hashlib

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.cpp_extension import load_inline

from kernel_agent import concurrency

#: GPUs this example runs on (``kernel_agent.gpu_arch.supports``: ``doctor --smoke`` skips
#: it elsewhere and says why).
ARCHS = "sm_90+"
ARCHS_WHY = "programmatic dependent launch (griddepcontrol)"

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include "ka_launch.cuh"

typedef __nv_bfloat16 bf16;

// 128-bit load that bypasses L1: every weight byte is read once
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// y = W x for one layer, W [n, n] row-major, n = 256 * KI. One warp per output row: lane l
// holds the row's elements [(i * 32 + l) * 8, +8) for i < KI. x is written by the previous
// layer: plain loads (not .nc), only after ka_pdl_wait().
template <int KI>
__global__ void __launch_bounds__(256) gemv_layer(const bf16* __restrict__ w, const bf16* x,
                                                  bf16* y, int n) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  uint4 wr[KI];
  if (row < n) {
    const bf16* p = w + (size_t)row * n;
#pragma unroll
    for (int i = 0; i < KI; ++i) wr[i] = ld_stream(p + (size_t)(i * 32 + lane) * 8);
  }
  ka_pdl_launch_dependents();  // the next layer may launch and start loading its weights
  ka_pdl_wait();               // the previous layer finished: x is complete and visible
  if (row >= n) return;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < KI; ++i) {
    const uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)(i * 32 + lane) * 8);
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wr[i]);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 a = __bfloat1622float2(wp[j]), b = __bfloat1622float2(xp[j]);
      acc = fmaf(a.x, b.x, acc);
      acc = fmaf(a.y, b.y, acc);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) y[row] = __float2bfloat16(acc);
}

#define KA_LAYER(K)                                                                 \
  case K:                                                                           \
    err = ka_launch(gemv_layer<K>, grid, block, 0, stream, opt, w, x, y, n);        \
    break;

// Every layer in order on the current stream (captured into a graph by the caller): layer l
// reads buf[l % 2] and writes buf[(l + 1) % 2]; the result is buf[layers % 2]. A layer
// writes only after its ka_pdl_wait(), when the layer before it (which read that half of
// buf) has completed.
void chain(std::vector<torch::Tensor> weights, torch::Tensor buf, bool pdl) {
  TORCH_CHECK(buf.dim() == 2 && buf.size(0) == 2 && buf.is_contiguous() &&
              buf.scalar_type() == at::kBFloat16, "chain: buf must be a contiguous bf16 [2, n]");
  const int n = (int)buf.size(1);
  TORCH_CHECK(n % 256 == 0 && n >= 256 && n <= 4096, "chain: n must be 256 * k, k <= 16");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bf16* b = reinterpret_cast<bf16*>(buf.data_ptr());
  KaLaunch opt;
  opt.pdl = pdl;
  const dim3 grid((n + 7) / 8), block(256);
  for (size_t l = 0; l < weights.size(); ++l) {
    TORCH_CHECK(weights[l].size(0) == n && weights[l].size(1) == n, "chain: W must be [n, n]");
    const bf16* w = reinterpret_cast<const bf16*>(weights[l].data_ptr());
    const bf16* x = b + (l & 1) * (size_t)n;
    bf16* y = b + ((l + 1) & 1) * (size_t)n;
    cudaError_t err = cudaSuccess;
    switch (n / 256) {
      KA_LAYER(1) KA_LAYER(2) KA_LAYER(3) KA_LAYER(4) KA_LAYER(5) KA_LAYER(6) KA_LAYER(7)
      KA_LAYER(8) KA_LAYER(9) KA_LAYER(10) KA_LAYER(11) KA_LAYER(12) KA_LAYER(13)
      KA_LAYER(14) KA_LAYER(15) KA_LAYER(16)
      default: TORCH_CHECK(false, "chain: unsupported n");
    }
    C10_CUDA_CHECK(err);
  }
}
"""
CPP_SRC = "void chain(std::vector<torch::Tensor> weights, torch::Tensor buf, bool pdl);"

_ext = None


def _load():
    global _ext
    if _ext is None:
        header = (concurrency.include_dir() / concurrency.HEADER).read_text()
        tag = hashlib.sha1((CUDA_SRC + header).encode()).hexdigest()[:8]
        _ext = load_inline(
            name=f"ka_pdl_gemv_chain_{tag}",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["chain"],
            extra_cuda_cflags=["-O3"],
            extra_include_paths=[str(concurrency.include_dir())],
        )
    return _ext


class GraphedChain(nn.Module):
    """The layers of ``reference`` as one CUDA graph of GEMV kernels (PDL edges with
    ``pdl``), captured once for one-row calls."""

    def __init__(self, reference: nn.Module, pdl: bool = True) -> None:
        super().__init__()
        first = reference.layers[0].weight
        self.n = first.shape[0]
        with torch.inference_mode(False):  # buffers updated in place, in and out of inference
            # the weights as they are now (kept alive: the graph holds their addresses)
            self._weights = [layer.weight.detach() for layer in reference.layers]
            self._buf = torch.zeros(2, self.n, device=first.device, dtype=torch.bfloat16)
        ext = _load()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(first.device):
            ext.chain(self._weights, self._buf, pdl)  # warm-up: loads the kernels
            torch.cuda.synchronize()
            with torch.cuda.graph(self._graph):
                ext.chain(self._weights, self._buf, pdl)
        self._x = self._buf[0]  # the graph's input ...
        self._y = self._buf[len(self._weights) % 2]  # ... and output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() != self.n or x.dtype != torch.bfloat16 or x.device != self._buf.device:
            for w in self._weights:  # other row counts: the reference math (a fallback)
                x = F.linear(x, w)
            return x
        self._x.copy_(x.reshape(-1))
        self._graph.replay()
        return self._y.clone().view(x.shape)  # a fresh tensor: the buffer is reused


def build(reference: nn.Module, pdl: bool = True) -> nn.Module:
    layers = getattr(reference, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        return reference
    n = getattr(layers[0], "in_features", 0)
    ok = (
        torch.cuda.is_available()
        and n % 256 == 0
        and 256 <= n <= 4096
        and all(
            isinstance(layer, nn.Linear)
            and layer.bias is None
            and layer.in_features == layer.out_features == n
            and layer.weight.dtype == torch.bfloat16
            and layer.weight.is_cuda
            and layer.weight.is_contiguous()
            and layer.weight.data_ptr() % 16 == 0  # 128-bit weight loads
            and layer.weight.device == layers[0].weight.device
            for layer in layers
        )
    )
    return GraphedChain(reference, pdl) if ok else reference
