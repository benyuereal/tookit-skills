// hip_gemm_tiled.cpp —— 通用 tiled GEMM（float），无 Matrix Core intrinsic 依赖。
// 用途：验证 dcu_build_kernel.sh 的「运行时探测 gfx → 编译」流程在任意 DCU 架构上跑通。
//       不写死 offload-arch；不依赖 gfx928 专属 __builtin_amdgcn_mmac_*。
// 正确性：与 CPU 参考比对，打印 METRIC 行（呼应 profile 纪律）。
//
// 编译（用 dcu_build_kernel.sh，自动探测目标 gfx）：
//   bash dcu_build_kernel.sh hip_gemm_tiled.cpp -o gemm
// 运行：
//   ./gemm        # 输出 METRIC max_diff=... / errors=... / status=PASS|FAIL

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define BM 16
#define BN 16
#define BK 16

__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;   // block(BN, BM): tx∈[0,BN) ty∈[0,BM)
    int row = by * BM + ty;
    int col = bx * BN + tx;
    float acc = 0.0f;
    for (int t = 0; t < (K + BK - 1) / BK; ++t) {
        if (row < M && (t * BK + tx) < K) As[ty][tx] = A[row * K + t * BK + tx];
        else As[ty][tx] = 0.0f;
        if ((t * BK + ty) < K && col < N) Bs[ty][tx] = B[(t * BK + ty) * N + col];
        else Bs[ty][tx] = 0.0f;
        __syncthreads();
        for (int k = 0; k < BK; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
    return;
}

int main() {
    const int M = 256, N = 256, K = 256;
    const size_t szA = (size_t)M * K * sizeof(float);
    const size_t szB = (size_t)K * N * sizeof(float);
    const size_t szC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(szA), *hB = (float*)malloc(szB);
    float *hC = (float*)malloc(szC), *hCref = (float*)malloc(szC);
    if (!hA || !hB || !hC || !hCref) { fprintf(stderr, "host alloc fail\n"); return 2; }

    srand(1234);
    for (int i = 0; i < M * K; ++i) hA[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand() % 100) / 10.0f;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += hA[r * K + k] * hB[k * N + c];
            hCref[r * N + c] = s;
        }

    float *dA, *dB, *dC;
    if (hipMalloc(&dA, szA) != hipSuccess ||
        hipMalloc(&dB, szB) != hipSuccess ||
        hipMalloc(&dC, szC) != hipSuccess) {
        fprintf(stderr, "device alloc fail\n"); return 2;
    }
    hipMemcpy(dA, hA, szA, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, szB, hipMemcpyHostToDevice);

    dim3 block(BN, BM);                       // tx=BN, ty=BM
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    hipLaunchKernelGGL(gemm_tiled, grid, block, 0, 0, dA, dB, dC, M, N, K);
    hipMemcpy(hC, dC, szC, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    float maxdiff = 0.0f; int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        float d = fabsf(hC[i] - hCref[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-2f) ++errors;
    }
    printf("METRIC max_diff=%.6f\n", maxdiff);
    printf("METRIC errors=%d\n", errors);
    printf("METRIC status=%s\n", errors == 0 ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hCref);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return errors == 0 ? 0 : 1;
}
