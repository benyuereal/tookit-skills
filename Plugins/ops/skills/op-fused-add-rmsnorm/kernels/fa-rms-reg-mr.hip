// 融合残差 RMSNorm · 多行/CTA 版（gfx938）
//
//   xf = x + residual
//   y  = w * xf / sqrt(mean(xf^2, -1) + eps)
//
// ## 与单行版的差别（这就是 gen2 要验证的东西）
//
// 单行版 grid=M，每 CTA 搬 8KB（H=4096 时），却要做 **2 次 __syncthreads**。
// 归约的固定延迟摊在 8KB 上藏不住 —— 实测只到地板的 83%。
//
// 多行版 grid=M/BLOCK_ROWS，**同样的一组 barrier 服务 BLOCK_ROWS 行**
// （行间归约互相独立）。代价是寄存器压力上升。
//
// 这个结论在 `op-norm-act-swiglu` 上已经验证过（1/2/4 行 = 148/140/153 µs，
// 2 行最优，4 行因 vgpr=84 退化）。本卡是在验证它是否**跨算子成立**——
// 这是判断一条机制能不能进 hw_kb 的标准。
//
// ## 配置约束
//   H == THREADS*8*VEC
//   M=8192 H=4096 ⇒ THREADS=256, VEC=2
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using bf16 = hip_bfloat16;

#ifndef WARPS
#define WARPS 4
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef VEC
#define VEC 2
#endif
#ifndef BLOCK_ROWS
#define BLOCK_ROWS 2
#endif
// 是否额外写一份 residual_out。两张 rmsnorm 卡的语义不同，必须可关。
#ifndef WRITE_RO
#define WRITE_RO 0
#endif

__global__ void __launch_bounds__(THREADS) fused_rms_mr_k(
    const bf16* __restrict__ x, const bf16* __restrict__ r,
    const bf16* __restrict__ w, bf16* __restrict__ y, bf16* __restrict__ ro,
    int M, int H, float eps) {
  // 静态 LDS（可被 inspect_so 回读）；BLOCK_ROWS × WARPS 的归约表
  __shared__ float warp_part[BLOCK_ROWS][WARPS];
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;

  // 寄存器数组按 BLOCK_ROWS 展开。
  // **索引必须是编译期常量** —— 运行期索引会让整块溢出到私有内存，
  // 实测 private 0→528B，慢 3.3 倍且不报错（见 hw_kb/private_spill_trap）。
  bf16 xv[BLOCK_ROWS][VEC * 8], rv[BLOCK_ROWS][VEC * 8];
  float xf[BLOCK_ROWS][VEC * 8];
  float acc[BLOCK_ROWS];

  // ---- ① 读入 + 平方和（所有行一起做，访存可以互相重叠）
  #pragma unroll
  for (int br = 0; br < BLOCK_ROWS; ++br) {
    const int row = blockIdx.x * BLOCK_ROWS + br;
    acc[br] = 0.f;
    if (row >= M) continue;
    const size_t base = (size_t)row * H;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const int off = (threadIdx.x + j * THREADS) * 8;
      *(uint4*)&xv[br][j * 8] = *(const uint4*)(x + base + off);
      *(uint4*)&rv[br][j * 8] = *(const uint4*)(r + base + off);
    }
    #pragma unroll
    for (int i = 0; i < VEC * 8; ++i) {
      xf[br][i] = static_cast<float>(xv[br][i]) + static_cast<float>(rv[br][i]);
      acc[br] += xf[br][i] * xf[br][i];
    }
  }

  // ---- ② 归约：所有行的 shuffle 串起来，然后**一组 barrier 服务所有行**
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

  // ---- ③ 写回
  #pragma unroll
  for (int br = 0; br < BLOCK_ROWS; ++br) {
    const int row = blockIdx.x * BLOCK_ROWS + br;
    if (row >= M) continue;
    const float inv = warp_part[br][0];
    const size_t base = (size_t)row * H;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const int off = (threadIdx.x + j * THREADS) * 8;
      uint4 rv_out, yv_out;
      bf16* rp = (bf16*)&rv_out;
      bf16* yp = (bf16*)&yv_out;
      uint4 wv = *(const uint4*)(w + off);
      const bf16* wp = (const bf16*)&wv;
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float v = xf[br][j * 8 + k];
        rp[k] = bf16(v);
        yp[k] = bf16(v * inv * static_cast<float>(wp[k]));
      }
#if WRITE_RO
      *(uint4*)(ro + base + off) = rv_out;
#endif
      *(uint4*)(y + base + off) = yv_out;
    }
  }
}

extern "C" void fused_rms_mr_launch(const void* x, const void* r, const void* w,
                                    void* y, void* ro, int M, int H, float eps) {
  const int grid = (M + BLOCK_ROWS - 1) / BLOCK_ROWS;
  fused_rms_mr_k<<<grid, THREADS>>>((const bf16*)x, (const bf16*)r,
                                    (const bf16*)w, (bf16*)y, (bf16*)ro,
                                    M, H, eps);
}
