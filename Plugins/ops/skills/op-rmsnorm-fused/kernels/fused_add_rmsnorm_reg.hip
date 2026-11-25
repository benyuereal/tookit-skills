// 由 kernel_scaffold.py 生成 —— 融合残差 RMSNorm（库形态，可直接 ctypes 加载）
//
// y = w * (x+r) * rsqrt(mean((x+r)^2) + eps),  residual_out = x+r
//
// 已做对的（**不要改坏，这两条是本实现比厂商快的全部原因**）：
//   1. (x+r) 留寄存器 —— 不落 LDS，省掉整行往返
//   2. 16 字节向量读写（uint4 = 8 个 bf16）
// 已知可改：
//   1. VEC（每线程向量数）与 THREADS —— 用 --params 扫，必须重新编译
//   2. 归约：跨 warp 现在过静态 LDS 表，可换 ds_bpermute 省一次 __syncthreads
//   3. 多行/CTA：当前 1 行 1 CTA，M 小时 grid 填不满 64 CU
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using bf16 = hip_bfloat16;

#ifndef WARPS
#define WARPS 1
#endif
#ifndef THREADS
#define THREADS 64
#endif
#ifndef VEC
#define VEC 8
#endif

__global__ void __launch_bounds__(THREADS) fused_rms_k(
    const bf16* __restrict__ x, const bf16* __restrict__ r,
    const bf16* __restrict__ w, bf16* __restrict__ y, bf16* __restrict__ ro,
    int M, int H, float eps) {
  __shared__ float warp_part[WARPS];
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int row = blockIdx.x;

  const size_t base = (size_t)row * H;
  const bf16* xr = x + base;
  const bf16* rr = r + base;
  bf16* orow = ro + base;
  bf16* yrow = y + base;

  bf16 xv[VEC * 8], rv[VEC * 8];
  float xf[VEC * 8];
  float acc = 0.f;
  #pragma unroll
  for (int j = 0; j < VEC; ++j) {
    const int off = (threadIdx.x + j * THREADS) * 8;
    *(uint4*)&xv[j * 8] = *(const uint4*)(xr + off);
    *(uint4*)&rv[j * 8] = *(const uint4*)(rr + off);
  }
  #pragma unroll
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
      const float v = xf[j * 8 + k];
      rp[k] = bf16(v);
      yp[k] = bf16(v * inv * static_cast<float>(wp[k]));
    }
    *(uint4*)(orow + off) = rv_out;
    *(uint4*)(yrow + off) = yv_out;
  }
}

// 库入口：ctypes 用。stream 走 NULL（默认流），与 torch 默认流一致。
extern "C" void fused_rms_launch(const void* x, const void* r, const void* w,
                                 void* y, void* ro, int M, int H, float eps) {
  fused_rms_k<<<M, THREADS>>>((const bf16*)x, (const bf16*)r, (const bf16*)w,
                              (bf16*)y, (bf16*)ro, M, H, eps);
}
