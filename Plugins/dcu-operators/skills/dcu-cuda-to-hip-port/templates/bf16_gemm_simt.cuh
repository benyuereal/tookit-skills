#pragma once
// 通用 SIMT bf16 GEMM (DCU 移植, 2026-08 数值验证 PASS): out[t,n] = sum_k x[t,k] * w[n,k]
// x: [t, k] bf16 (token-major), w: [n, k] bf16, out: [t, n] 行主序
// 2D grid: blockIdx.x = n 维 (每 CTA RowsPerCta 行), blockIdx.y = t 维 (每 CTA ColsPerTile 列)
// 每 warp 管 1 个输出行 (n 维), K 维用 32-lane 收缩, 形状通用 (n/k/t 运行时参数)
// 自包含: 不依赖 mma.cuh, 纯 SIMT (gfx936 64-lane wavefront, 32-lane 子组模型)
// 验证: n=8216 k=5120 t=51 max_rel=3.88e-3 0/419016 错; n=96 k=5120 t=51 max_rel=3.75e-3 0 错
// 用途: 27B 的 bf16 输入投影 (attn_input/gdn_input/gdn_gating) 的 medium-t 路径 (t>small_t 上限)
#include "ops/common/math.cuh"   // div_up
#include "ops/common/warp.cuh"   // warp_reduce_sum
#include <hip/hip_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {

template <int ColsPerTile, int RowsPerCta = 8, int Slab = 1024>
__global__ void bf16_gemm_simt_kernel(const __hip_bfloat16* __restrict__ x,
                                      const __hip_bfloat16* __restrict__ w,
                                      __hip_bfloat16* __restrict__ out,
                                      std::int32_t n, std::int32_t k, std::int32_t t) {
    constexpr int kPhases = Slab / 256; // 4
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    if (row >= n) { return; }
    const int col0  = static_cast<int>(blockIdx.y) * ColsPerTile;
    const int ncols = min(ColsPerTile, t - col0);
    if (ncols <= 0) { return; }

    // ⚠️ __shared__ 数组必须编译期大小 (DTK 不支持 VLA 做 __shared__)
    __shared__ __hip_bfloat16 s_w[RowsPerCta][2][Slab];

    const __hip_bfloat16* wrow = w + static_cast<std::int64_t>(row) * k;

    float acc[ColsPerTile];
#pragma unroll
    for (int i = 0; i < ColsPerTile; ++i) { acc[i] = 0.0f; }

    const int nslabs = div_up(k, Slab);

    // 加载 slab s 到 s_w[warp][buf]: 每 lane 32 个值 (4 phase x 8), 覆盖 1024
    // ⚠️ 必须覆盖整个 Slab (1024), 只加载前 256 会读到未初始化垃圾 → 全 inf
    auto load_slab = [&](int s, int buf) {
        const int base = s * Slab;
#pragma unroll
        for (int p = 0; p < kPhases; ++p) {
            const int koff = p * 256 + lane * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int gk = base + koff + j;
                s_w[warp][buf][koff + j] = (gk < k) ? wrow[gk] : __float2bfloat16_rn(0.0f);
            }
        }
    };

    load_slab(0, 0);
    __syncthreads();

    for (int s = 0; s < nslabs; ++s) {
        const int buf = s & 1;
        if (s + 1 < nslabs) { load_slab(s + 1, 1 - buf); }
        __syncthreads();
        const int slab_base = s * Slab;
#pragma unroll
        for (int p = 0; p < kPhases; ++p) {
            const int koff = p * 256 + lane * 8;
            float wv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { wv[j] = __bfloat162float(s_w[warp][buf][koff + j]); }
#pragma unroll
            for (int tt = 0; tt < ColsPerTile; ++tt) {
                if (tt < ncols) {
                    const __hip_bfloat16* xrow = x + static_cast<std::int64_t>(col0 + tt) * k;
                    const int gk = slab_base + koff;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float xv = (gk + j < k) ? __bfloat162float(xrow[gk + j]) : 0.0f;
                        acc[tt] = fmaf(wv[j], xv, acc[tt]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int tt = 0; tt < ColsPerTile; ++tt) {
        if (tt >= ncols) { continue; }
        float a = acc[tt];
        a = warp_reduce_sum(a);  // 32-lane 子组 reduce (gfx936 上正确, 见 SKILL §4.5)
        if (lane == 0) {
            out[static_cast<std::int64_t>(col0 + tt) * n + row] = __float2bfloat16_rn(a);
        }
    }
}

} // namespace ninfer::ops::detail

// host 端 launch 示例:
//   const dim3 grid(div_up(n, RowsPerCta), div_up(t, ColsPerTile));
//   bf16_gemm_simt_kernel<8><<<grid, RowsPerCta * 32>>>(dx, dw, do_, n, k, t);
//   hipGetLastError(); hipDeviceSynchronize();
