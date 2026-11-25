// 朴素 tiled fp8 GEMM：C[M,N] = A[M,K] @ B[K,N]
//
// **不是优化实现，是打通链路 + 正确性基准。**
// fp8 用 hip_fp8.h 的 __hip_fp8_e4m3 解析，**不能把 uint8 直接转 float**
// —— 位模式非线性，0x38 在 e4m3 里是 1.0，转 float 却是 56.0。
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#define TILE 16

__device__ __forceinline__ float fp8_to_f32(unsigned char b) {
  __hip_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}

__global__ void gemm_fp8_k(const unsigned char* A, const unsigned char* B,
                           float* C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int k0 = t * TILE + threadIdx.x;
    int k1 = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && k0 < K) ? fp8_to_f32(A[row * K + k0]) : 0.f;
    Bs[threadIdx.y][threadIdx.x] =
        (k1 < K && col < N) ? fp8_to_f32(B[k1 * N + col]) : 0.f;
    __syncthreads();
    for (int i = 0; i < TILE; ++i) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

extern "C" void gemm_fp8_launch(const void* A, const void* B, void* C,
                                int M, int N, int K) {
  dim3 blk(TILE, TILE);
  dim3 grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_fp8_k<<<grd, blk>>>((const unsigned char*)A, (const unsigned char*)B,
                           (float*)C, M, N, K);
}
