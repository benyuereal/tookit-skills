// hip_shim.h — CUDA -> DTK/HIP compatibility shims for NInfer-style ports.
// DTK (gfx936) provides mask-less warp shuffles (__shfl/__shfl_xor/__shfl_down/
// __shfl_up) and no __shfl_*_sync. On AMD the mask is ignored (whole warp
// participates), so we map the CUDA masked forms onto the mask-less forms.
//
// Usage: include this AFTER <hip/hip_runtime.h>.
//
// CRITICAL: do NOT wrap the shims in #if defined(__HIP_DEVICE_COMPILE__) —
// hipcc's host pass also parses kernel bodies, and the guard makes the host
// pass fail with "use of undeclared identifier '__shfl_sync'". __device__
// declarations are visible in both passes; bodies only compile in the device
// pass, so no guard is needed.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>

// ---- warp shuffle: CUDA masked -> DTK mask-less ----
__device__ __forceinline__ int __shfl_sync(unsigned mask, int var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ unsigned __shfl_sync(unsigned mask, unsigned var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ float __shfl_sync(unsigned mask, float var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ double __shfl_sync(unsigned mask, double var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ long long __shfl_sync(unsigned mask, long long var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ long __shfl_sync(unsigned mask, long var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ unsigned long __shfl_sync(unsigned mask, unsigned long var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ short __shfl_sync(unsigned mask, short var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}
__device__ __forceinline__ unsigned short __shfl_sync(unsigned mask, unsigned short var, int src, int width = 32) {
    (void)mask; return __shfl(var, src, width);
}

__device__ __forceinline__ int __shfl_xor_sync(unsigned mask, int var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ unsigned __shfl_xor_sync(unsigned mask, unsigned var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ float __shfl_xor_sync(unsigned mask, float var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ double __shfl_xor_sync(unsigned mask, double var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ long long __shfl_xor_sync(unsigned mask, long long var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ long __shfl_xor_sync(unsigned mask, long var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ short __shfl_xor_sync(unsigned mask, short var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}
__device__ __forceinline__ unsigned short __shfl_xor_sync(unsigned mask, unsigned short var, int lane, int width = 32) {
    (void)mask; return __shfl_xor(var, lane, width);
}

__device__ __forceinline__ int __shfl_down_sync(unsigned mask, int var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ unsigned __shfl_down_sync(unsigned mask, unsigned var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ float __shfl_down_sync(unsigned mask, float var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ double __shfl_down_sync(unsigned mask, double var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ long long __shfl_down_sync(unsigned mask, long long var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ long __shfl_down_sync(unsigned mask, long var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ unsigned long __shfl_down_sync(unsigned mask, unsigned long var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ short __shfl_down_sync(unsigned mask, short var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}
__device__ __forceinline__ unsigned short __shfl_down_sync(unsigned mask, unsigned short var, int delta, int width = 32) {
    (void)mask; return __shfl_down(var, delta, width);
}

__device__ __forceinline__ int __shfl_up_sync(unsigned mask, int var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ unsigned __shfl_up_sync(unsigned mask, unsigned var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ float __shfl_up_sync(unsigned mask, float var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ double __shfl_up_sync(unsigned mask, double var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ long long __shfl_up_sync(unsigned mask, long long var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ long __shfl_up_sync(unsigned mask, long var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ unsigned long __shfl_up_sync(unsigned mask, unsigned long var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ short __shfl_up_sync(unsigned mask, short var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
__device__ __forceinline__ unsigned short __shfl_up_sync(unsigned mask, unsigned short var, int delta, int width = 32) {
    (void)mask; return __shfl_up(var, delta, width);
}
