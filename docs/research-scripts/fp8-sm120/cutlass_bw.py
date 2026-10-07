"""CUTLASS 4.1 (tilelang's 3rdparty copy) SM120 blockwise-scaled FP8 GEMM, as in CUTLASS
example 87a/87b: e4m3 x e4m3 -> bf16, fp32 scales per 1x128 (A = activations) and
128x128 (B = weight) blocks, TMA + warp-specialised mainloop (sm_120a)."""

from __future__ import annotations

import os

from kernel_agent import toolchain

CUTLASS = "/home/kadir/kadir_projects/kernel-agent/.venv/lib/python3.12/site-packages/tilelang/3rdparty/cutlass"

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

template <class TileShape, class Schedule>
struct BW {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignA = 16, AlignB = 16, AlignC = 8;
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using ClusterShape = Shape<_1, _1, _1>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementC, LayoutC, AlignC, ElementC, LayoutC, AlignC,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignB, float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static at::Tensor& workspace(size_t bytes) {
    static at::Tensor ws;
    if (!ws.defined() || (size_t)ws.numel() < bytes)
      ws = at::empty({(int64_t)std::max<size_t>(bytes, 1)}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
    return ws;
  }

  // x [M, K] e4m3, w [N, K] e4m3, sx [K/128, M] fp32 (M-major), sw [K/128, N/128] fp32, y [M, N] bf16
  static void run(at::Tensor x, at::Tensor w, at::Tensor sx, at::Tensor sw, at::Tensor y) {
    int M = x.size(0), N = w.size(0), K = x.size(1);
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    auto sa = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto sb = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto sc = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
    auto lsfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto lsfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {(ElementA*)x.data_ptr(), sa, (ElementB*)w.data_ptr(), sb,
         sx.data_ptr<float>(), lsfa, sw.data_ptr<float>(), lsfb},
        {{1.f, 0.f}, (ElementC*)y.data_ptr(), sc, (ElementC*)y.data_ptr(), sc}};
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
    size_t bytes = Gemm::get_workspace_size(args);
    auto& ws = workspace(bytes);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "init failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run failed");
  }
};

// Dense (tensorwise: scalar alpha) FP8 GEMM on the same SM120 TMA warp-specialised mainloop.
template <class TileShape, class Schedule>
struct Dense {
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  using ClusterShape = Shape<_1, _1, _1>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementC, LayoutC, 8, ElementC, LayoutC, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 16, ElementB, LayoutB, 16, float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static void run(at::Tensor x, at::Tensor w, at::Tensor y) {
    int M = x.size(0), N = w.size(0), K = x.size(1);
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    auto sa = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto sb = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto sc = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {(ElementA*)x.data_ptr(), sa, (ElementB*)w.data_ptr(), sb},
        {{1.f, 0.f}, (ElementC*)y.data_ptr(), sc, (ElementC*)y.data_ptr(), sc}};
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
    size_t bytes = Gemm::get_workspace_size(args);
    static at::Tensor ws;
    if (!ws.defined() || (size_t)ws.numel() < bytes)
      ws = at::empty({(int64_t)std::max<size_t>(bytes, 1)}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "init failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run failed");
  }
};
using DCoop128 = Dense<Shape<_128, _128, _128>, cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
using DPing64 = Dense<Shape<_64, _128, _128>, cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
using DPing64n64 = Dense<Shape<_64, _64, _128>, cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
void dense_coop128(at::Tensor x, at::Tensor w, at::Tensor y) { DCoop128::run(x, w, y); }
void dense_ping64(at::Tensor x, at::Tensor w, at::Tensor y) { DPing64::run(x, w, y); }
void dense_ping64n64(at::Tensor x, at::Tensor w, at::Tensor y) { DPing64n64::run(x, w, y); }

// MXFP8 (e4m3 + ue8m0 per 32 along K, both operands) on the block-scaled MMA
// (mma.sync kind::mxf8f6f4.block_scale, SASS QMMA.SF): CUTLASS example-79-style.
template <class TileShape, class Schedule>
struct MX {
  using ElementPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  using ClusterShape = Shape<_1, _1, _1>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementC, LayoutC, 8, ElementC, LayoutC, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementPair, cutlass::layout::RowMajor, 16, ElementPair, cutlass::layout::ColumnMajor, 16,
      float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Cfg = typename Main::Sm1xxBlkScaledConfig;

  static std::vector<int64_t> sf_sizes(int M, int N, int K) {
    auto la = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto lb = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    return {(int64_t)size(filter_zeros(la)), (int64_t)size(filter_zeros(lb))};
  }

  static void run(at::Tensor x, at::Tensor w, at::Tensor sfa, at::Tensor sfb, at::Tensor y) {
    int M = x.size(0), N = w.size(0), K = x.size(1);
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    auto sa = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto sb = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto sc = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
    auto la = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto lb = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    using EA = typename Main::ElementA;
    using EB = typename Main::ElementB;
    using ESF = typename Main::ElementSF;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {(EA*)x.data_ptr(), sa, (EB*)w.data_ptr(), sb, (ESF*)sfa.data_ptr(), la, (ESF*)sfb.data_ptr(), lb},
        {{1.f, 0.f}, (ElementC*)y.data_ptr(), sc, (ElementC*)y.data_ptr(), sc}};
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
    size_t bytes = Gemm::get_workspace_size(args);
    static at::Tensor ws;
    if (!ws.defined() || (size_t)ws.numel() < bytes)
      ws = at::empty({(int64_t)std::max<size_t>(bytes, 1)}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto st = gemm.initialize(args, ws.data_ptr(), stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess, "init failed: ", cutlassGetStatusString(st), " cuda: ",
                cudaGetErrorString(cudaGetLastError()), " smem ", (int)sizeof(typename Kernel::SharedStorage));
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run failed");
  }
};
using MXCoop = MX<Shape<_128, _128, _128>, cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
#define MX_ONLY_ONE 1
using MXPing = MX<Shape<_128, _128, _128>, cutlass::gemm::collective::KernelScheduleAuto>;
std::vector<int64_t> mx_sf_sizes(int64_t M, int64_t N, int64_t K) { return MXCoop::sf_sizes(M, N, K); }
void mx_coop128(at::Tensor x, at::Tensor w, at::Tensor a, at::Tensor b, at::Tensor y) { MXCoop::run(x, w, a, b, y); }
void mx_ping64(at::Tensor x, at::Tensor w, at::Tensor a, at::Tensor b, at::Tensor y) { MXPing::run(x, w, a, b, y); }

using Coop128 = BW<Shape<_128, _128, _128>, cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120>;
using Ping64 = BW<Shape<_64, _128, _128>, cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>;

void coop128(at::Tensor x, at::Tensor w, at::Tensor sx, at::Tensor sw, at::Tensor y) { Coop128::run(x, w, sx, sw, y); }
void ping64(at::Tensor x, at::Tensor w, at::Tensor sx, at::Tensor sw, at::Tensor y) { Ping64::run(x, w, sx, sw, y); }
"""


def load():
    tc = toolchain.setup()
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="ka132_cutlass_bw",
        cpp_sources="void coop128(at::Tensor x, at::Tensor w, at::Tensor sx, at::Tensor sw, at::Tensor y);\n"
        "void ping64(at::Tensor x, at::Tensor w, at::Tensor sx, at::Tensor sw, at::Tensor y);\n"
        "void dense_coop128(at::Tensor x, at::Tensor w, at::Tensor y);\n"
        "void dense_ping64(at::Tensor x, at::Tensor w, at::Tensor y);\n"
        "void dense_ping64n64(at::Tensor x, at::Tensor w, at::Tensor y);\n"
        "std::vector<int64_t> mx_sf_sizes(int64_t M, int64_t N, int64_t K);\n"
        "void mx_coop128(at::Tensor x, at::Tensor w, at::Tensor a, at::Tensor b, at::Tensor y);\n"
        "void mx_ping64(at::Tensor x, at::Tensor w, at::Tensor a, at::Tensor b, at::Tensor y);",
        cuda_sources=SRC,
        functions=["coop128", "ping64", "dense_coop128", "dense_ping64", "dense_ping64n64", "mx_sf_sizes", "mx_coop128", "mx_ping64"],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120a,code=sm_120a",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-O3",
            f"-I{CUTLASS}/include",
            f"-I{CUTLASS}/tools/util/include",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=0",
        ],
        verbose=bool(os.environ.get("VERBOSE")),
    )


if __name__ == "__main__":
    import time

    t = time.time()
    load()
    print(f"built in {time.time() - t:.0f} s")
