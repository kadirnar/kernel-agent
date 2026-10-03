# TileLang backend (`import tilelang, tilelang.language as T`)

Verified example: `examples/tilelang_rmsnorm.py`; a verified GEMM:

```python
@tilelang.jit(out_idx=[-1])  # last arg is allocated & returned
def matmul(M, N, K, bM=128, bN=128, bK=32, dtype="float16", acc="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
            As = T.alloc_shared((bM, bK), dtype)
            Bs = T.alloc_shared((bK, bN), dtype)
            Cl = T.alloc_fragment((bM, bN), acc)
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=3):  # cp.async / TMA pipelining
                T.copy(A[by * bM, k * bK], As)
                T.copy(B[k * bK, bx * bN], Bs)
                T.gemm(As, Bs, Cl)  # tensor cores
            T.copy(Cl, C[by * bM, bx * bN])

    return main


kernel = matmul(1024, 1024, 1024)  # compiles (nvcc) on first call; cache per shape
c = kernel(a, b)  # torch tensors in, torch tensor out
```

Key facts
* Buffers: `T.alloc_shared`, `T.alloc_fragment` (registers, tiled across
  threads), `T.alloc_local`. Copies: `T.copy(src_region, dst)` handles
  vectorisation, async copies and TMA where available.
* Parallel loops: `for i, j in T.Parallel(a, b)` (elementwise on fragments),
  `T.serial`, `T.unroll`, `T.Pipelined(n, num_stages=s)`.
* Reductions: `T.reduce_sum(src, dst, dim=1)`, `T.reduce_max`, `T.reduce_abssum`.
* Math: `T.exp`, `T.exp2`, `T.rsqrt`, `T.max`, `T.if_then_else`; casts with
  `T.Cast(dtype, x)`. Dtypes are strings: "float16", "bfloat16", "float32".
* GEMM variants: `T.gemm(A, B, C, transpose_B=True)` (e.g. Q @ K^T).
* FlashAttention in TileLang = `T.gemm` for QK^T, online softmax with
  `T.reduce_max`/`T.exp2` on fragments, `T.gemm` for PV; see upstream
  `examples/flash_attention` in the tile-ai/tilelang repo for full versions
  (including decode / split-KV variants).
* Shapes are compile-time constants; compile per distinct shape and cache the
  kernel in a dict (decode shapes repeat, so this is cheap). Symbolic dims are
  possible with `T.dynamic("m")` / `T.symbolic("m")`.
* `tilelang.jit(..., pass_configs={...})` and autotuning via
  `tilelang.autotune(configs=[...])`.
* Uses nvcc: the toolchain adds `-allow-unsupported-compiler` and the CCCL
  compatibility override automatically when needed.
* Per-call host overhead ≈ 30 us (measured); fuse more work per kernel.
