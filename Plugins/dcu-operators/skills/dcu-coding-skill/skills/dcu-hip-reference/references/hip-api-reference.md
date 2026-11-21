# HIP API Reference (Core)

本文件提供 HIP 核心 API 的速查清单，详细语义以官方文档为准。

## Memory

- `hipMalloc(void** ptr, size_t size)`
- `hipFree(void* ptr)`
- `hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)`
- `hipMemcpyAsync(...)`

## Device

- `hipGetDeviceCount(int* count)`
- `hipSetDevice(int deviceId)`
- `hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId)`

## Stream

- `hipStreamCreate(hipStream_t* stream)`
- `hipStreamSynchronize(hipStream_t stream)`
- `hipStreamDestroy(hipStream_t stream)`
