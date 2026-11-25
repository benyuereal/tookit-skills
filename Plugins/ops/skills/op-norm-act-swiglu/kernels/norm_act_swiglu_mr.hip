// 三合一融合 · 多行/CTA 版（手写 HIP，gfx938）
//
//   xf   = x + residual
//   h    = w * xf / sqrt(mean(xf^2) + eps)      (归约沿 H)
//   g,u  = h[:, :I], h[:, I:]                   (I = H/2)
//   out  = silu(g) * u                          只写 I 宽
//
// ## 与单行版的差别（这就是优化本身）
//   单行版：grid=M，每 CTA 搬 16KB 却要做 2 次 __syncthreads + 6 轮 shuffle。
//           归约的固定延迟摊在 16KB 上藏不住，实测只到地板 66%。
//   多行版：grid=M/BLOCK_ROWS，同样的 barrier 摊到 BLOCK_ROWS 份数据上。
//           **行间归约互相独立**，所以一组 barrier 能同时服务所有行。
//
// ## 配置约束
//   H == THREADS*8*VEC 且 THREADS*8 == I 且 VEC == 2
//   M=8192 H=4096 I=2048 ⇒ THREADS=256, VEC=2
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using bf16 = hip_bfloat16;

#ifndef WARPS
#define WARPS 4
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef BLOCK_ROWS
#define BLOCK_ROWS 2
#endif
#define VEC 2

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__global__ void __launch_bounds__(THREADS) norm_act_swiglu_mr_k(
    const bf16* __restrict__ x, const bf16* __restrict__ r,
    const bf16* __restrict__ w, bf16* __restrict__ out,
    int M, int H, float eps) {
  // 静态 LDS（会进 kernel metadata，可被 inspect_so 回读）
  __shared__ float warp_part[BLOCK_ROWS][WARPS];
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int I = H >> 1;
  const int off = threadIdx.x * 8;

  // 寄存器数组按 BLOCK_ROWS 展开。**索引必须是编译期常量**
  // （运行期索引会让整块溢出到私有内存，实测慢 3.3 倍且不报错）。
  bf16 xv[BLOCK_ROWS][VEC * 8], rv[BLOCK_ROWS][VEC * 8];
  float xf[BLOCK_ROWS][VEC * 8];
  float acc[BLOCK_ROWS];

  #pragma unroll
  for (int br = 0; br < BLOCK_ROWS; ++br) {
    const int row = blockIdx.x * BLOCK_ROWS + br;
    acc[br] = 0.f;
    if (row >= M) continue;
    const size_t base = (size_t)row * H;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const int o = (threadIdx.x + j * THREADS) * 8;
      *(uint4*)&xv[br][j * 8] = *(const uint4*)(x + base + o);
      *(uint4*)&rv[br][j * 8] = *(const uint4*)(r + base + o);
    }
    #pragma unroll
    for (int i = 0; i < VEC * 8; ++i) {
      xf[br][i] = static_cast<float>(xv[br][i]) + static_cast<float>(rv[br][i]);
      acc[br] += xf[br][i] * xf[br][i];
    }
  }

  // 归约：所有行的 shuffle 串在一起做，然后**一组 barrier 服务所有行**
  #pragma unroll
  for (int br = 0; br < BLOCK_ROWS; ++br) {
    #pragma unroll
    for (int o = 32; o > 0; o >>= 1) acc[br] += __shfl_down(acc[br], o);
  }
  if (lane == 0) {
    #pragma unroll
    for (int br = 0; br < BLOCK_ROWS; ++br) warp_part[br][warp] = acc[br];
  }
  __syncthreads();
  if (warp == 0 && lane < BLOCK_ROWS) {
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < WARPS; ++i) s += warp_part[lane][i];
    warp_part[lane][0] = rsqrtf(s / (float)H + eps);
  }
  __syncthreads();

  #pragma unroll
  for (int br = 0; br < BLOCK_ROWS; ++br) {
    const int row = blockIdx.x * BLOCK_ROWS + br;
    if (row >= M) continue;
    const float inv = warp_part[br][0];
    uint4 ov;
    bf16* op = (bf16*)&ov;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float g = xf[br][k] * inv * static_cast<float>(w[off + k]);
      const float u = xf[br][8 + k] * inv * static_cast<float>(w[off + I + k]);
      op[k] = bf16(silu(g) * u);
    }
    *(uint4*)(out + (size_t)row * I + off) = ov;
  }
}

extern "C" void norm_act_swiglu_mr_launch(const void* x, const void* r,
                                          const void* w, void* out,
                                          int M, int H, float eps) {
  const int grid = (M + BLOCK_ROWS - 1) / BLOCK_ROWS;
  norm_act_swiglu_mr_k<<<grid, THREADS>>>((const bf16*)x, (const bf16*)r,
                                          (const bf16*)w, (bf16*)out, M, H, eps);
}
