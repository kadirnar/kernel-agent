# Triton backend

Verified example: `examples/triton_rmsnorm.py`.

```python
import triton, triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptr + rm[:, None] * stride_am + (k * BLOCK_K + rk)[None, :] * stride_ak,
            mask=(rm[:, None] < M) & ((k * BLOCK_K + rk)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k * BLOCK_K + rk)[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=((k * BLOCK_K + rk)[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)  # tensor cores; fp32 accumulate
    # fused epilogue goes here (bias, activation, residual, cast)
    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )
```

Notes
* Launch: `kernel[grid](...)` where `grid` is a tuple or `lambda meta: (...)`.
* Block sizes must be powers of two (`triton.next_power_of_2`). `tl.dot` needs
  every dimension ≥ 16.
* Reductions: `tl.sum`, `tl.max`, `tl.argmax`; math: `tl.exp`, `tl.exp2`,
  `tl.rsqrt`, `tl.sigmoid`, `tl.math.*`; `tl.where`; `tl.cast`/`.to()`.
* Always compute in fp32 (`.to(tl.float32)`) and cast on store, matching the
  reference's cast points exactly (HF RMSNorm casts the normalised value to the
  input dtype *before* multiplying by the weight).
* Autotune only on shapes that vary; `key=[...]`. Autotuning runs on first
  call (cost paid during warm-up). Avoid autotune when shapes change every call.
* Launch overhead is ~30-40 us of Python per call. For decode-time micro-ops,
  fuse more per kernel; for elementwise chains do everything in one kernel.
* Persistent kernels: `grid = (num_SMs,)` and loop over tiles inside.
* Debug: `TRITON_INTERPRET=1` runs kernels on CPU (slow) for logic bugs.
* Matmul with tiny M (decode GEMV): use a split-K or row-per-program design with
  `tl.sum(a[None, :] * w, axis=1)` rather than `tl.dot`, and vector loads of the
  weight (memory bound — goal is to stream weights at full bandwidth).
