// fused-add RMSNorm —— 寄存器驻留版（供 runs/forged 的动态变体使用）
// y = w * (x+r) * rsqrt(mean((x+r)^2)+eps),  residual_out = x+r
//
// 与 scaffold 默认骨架的差别（这就是「优化」本身，不是换个写法）：
//   默认骨架：整行读进动态 LDS → 归约 → 从 LDS 读回写   ⇒ 实测 212 GB/s
//   本实现  ：(x+r) 留在寄存器（16 float/线程）→ 归约 → 直接写 ⇒ 实测 1646 GB/s
// 机理：H=4096 时每行 16 KB 的 LDS 往返，在 64 CU / 8 MB L2 上把访存打成了
// 4 次往返而不是 2 读 2 写；寄存器驻留把 LDS 用量降到 WARPS*4 字节。
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using bf16 = hip_bfloat16;

#ifndef WARPS
#define WARPS 4
#endif
#ifndef THREADS
#define THREADS 256
#endif
// VEC 必须是**编译期常量**：改成运行期变量（int vec = ept/8）会让
// xv[]/rv[]/xf[] 无法驻留寄存器，整块溢出到私有内存（实测 private 0→528 B，
// 0.164 ms → 0.538 ms，慢 3.3x）。数组索引必须能被编译器证明是常量。
#ifndef VEC
#define VEC 2
#endif

__global__ void __launch_bounds__(THREADS) fused_rms_k(
    const bf16* __restrict__ x, const bf16* __restrict__ r,
    const bf16* __restrict__ w, bf16* __restrict__ y, bf16* __restrict__ ro,
    int M, int H, float eps) {
  __shared__ float warp_part[WARPS];
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int row = blockIdx.x;
  // 每线程元素数由 VEC 决定，**且必须与 H 精确相等**：
  // VEC*8*THREADS == H。不等就会越界读或漏读，且不会报错——只会算错。
  // 调用方 run() 负责按 H 选 VEC，这里不再做运行期兜底（兜底会掩盖错误）。

  const size_t base = (size_t)row * H;
  const bf16* xr = x + base;
  const bf16* rr = r + base;
  bf16* orow = ro + base;
  bf16* yrow = y + base;

  bf16 xv[64], rv[64];
  float xf[64];
  float acc = 0.f;
  for (int j = 0; j < VEC; ++j) {
    const int off = (threadIdx.x + j * THREADS) * 8;
    *(uint4*)&xv[j * 8] = *(const uint4*)(xr + off);
    *(uint4*)&rv[j * 8] = *(const uint4*)(rr + off);
  }
  for (int i = 0; i < VEC * 8; ++i) {
    xf[i] = static_cast<float>(xv[i]) + static_cast<float>(rv[i]);
    acc += xf[i] * xf[i];
  }
  #pragma unroll
  for (int o = 32; o > 0; o >>= 1) acc += __shfl_down(acc, o);
  if (lane == 0) warp_part[warp] = acc;
  __syncthreads();
  if (warp == 0 && lane == 0) {
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < WARPS; ++i) s += warp_part[i];
    warp_part[0] = rsqrtf(s / (float)H + eps);
  }
  __syncthreads();
  const float inv = warp_part[0];

  for (int j = 0; j < VEC; ++j) {
    const int off = (threadIdx.x + j * THREADS) * 8;
    uint4 rv_out, yv_out;
    bf16* rp = (bf16*)&rv_out;
    bf16* yp = (bf16*)&yv_out;
    uint4 wv = *(const uint4*)(w + off);
    const bf16* wp = (const bf16*)&wv;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float v = xf[j * 8 + k];
      rp[k] = bf16(v);
      yp[k] = bf16(v * inv * static_cast<float>(wp[k]));
    }
    *(uint4*)(orow + off) = rv_out;
    *(uint4*)(yrow + off) = yv_out;
  }
}

// 供 ctypes 调用的入口。dims 用 int（M,H），stream 传 NULL 走默认流。
extern "C" void fused_rms_launch(const void* x, const void* r, const void* w,
                                 void* y, void* ro, int M, int H, float eps) {
  fused_rms_k<<<M, THREADS>>>((const bf16*)x, (const bf16*)r, (const bf16*)w,
                              (bf16*)y, (bf16*)ro, M, H, eps);
}
