// Register-only mma.sync throughput on sm_120a (issue #135): which FP8 instruction runs at
// full rate? Each warp runs CH independent accumulator chains for ITERS iterations.
// Build: nvcc -O3 -arch=sm_120a mma_rate.cu -o mma_rate
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int ITERS = 4096;
constexpr int CH = 8;

template <int V>
__global__ void kern(float *out, uint32_t seed) {
  uint32_t a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u, b0 = a0 * 11u, b1 = a0 * 13u;
  // keep values tiny / finite: mask exponent bits of e4m3 so no NaN (0x7f / 0xff)
  a0 &= 0x3f3f3f3fu; a1 &= 0x3f3f3f3fu; a2 &= 0x3f3f3f3fu; a3 &= 0x3f3f3f3fu; b0 &= 0x3f3f3f3fu; b1 &= 0x3f3f3f3fu;
  float d[CH][4] = {};
  uint32_t h[CH][2] = {};
  uint32_t sfa = 127, sfb = 127;
  uint16_t z = 0;
#pragma unroll 1
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      if constexpr (V == 0) {  // sm_89-style e4m3, fp32 accumulate
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (V == 1) {  // sm_120 kind::f8f6f4 e4m3, fp32 accumulate
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (V == 2) {  // sm_120 block-scaled MXFP8 (ue8m0 per 32), fp32 accumulate
        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
                     : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
      } else if constexpr (V == 3) {  // sm_89-style e4m3, fp16 accumulate
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
                     : "+r"(h[c][0]), "+r"(h[c][1])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (V == 4) {  // bf16 m16n8k16, fp32 accumulate (k16: half the FLOPs per instr)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (V == 5) {  // sm_120 kind::f8f6f4 e4m3, fp16 accumulate
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
                     : "+r"(h[c][0]), "+r"(h[c][1])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < CH; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3] + __half2float(__ushort_as_half((unsigned short)(h[c][0] & 0xffff))) +
                                    (float)h[c][1];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int V>
double run(const char *name, int k_per_instr, int sms) {
  int blocks = sms * 4, threads = 256;
  float *out;
  cudaMalloc(&out, sizeof(float) * blocks * threads);
  kern<V><<<blocks, threads>>>(out, 1);  // warm
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best = 1e30f;
  for (int r = 0; r < 5; ++r) {
    cudaEventRecord(e0);
    kern<V><<<blocks, threads>>>(out, r + 2);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    if (ms < best) best = ms;
  }
  double warps = blocks * (threads / 32.0);
  double flops = warps * ITERS * CH * 2.0 * 16 * 8 * k_per_instr;
  double tf = flops / (best * 1e-3) / 1e12;
  printf("%-58s %8.1f TFLOP/s  (%.3f ms)  err=%s\n", name, tf, best, cudaGetErrorString(cudaGetLastError()));
  cudaFree(out);
  return tf;
}

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("%s sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
  int sms = p.multiProcessorCount;
  for (int pass = 0; pass < 2; ++pass) {
    run<4>("bf16 m16n8k16 f32-acc", 16, sms);
    run<0>("e4m3 m16n8k32 f32-acc (sm_89 form)", 32, sms);
    run<1>("e4m3 kind::f8f6f4 m16n8k32 f32-acc", 32, sms);
    run<2>("e4m3 kind::mxf8f6f4.block_scale m16n8k32 f32-acc", 32, sms);
    run<3>("e4m3 m16n8k32 f16-acc (sm_89 form)", 32, sms);
    run<5>("e4m3 kind::f8f6f4 m16n8k32 f16-acc", 32, sms);
  }
  return 0;
}
