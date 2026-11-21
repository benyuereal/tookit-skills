# HIP Runtime Guide

## 常见流程

1. 设备选择：`hipGetDeviceCount` / `hipSetDevice`
2. 内存申请：`hipMalloc`
3. Host↔Device 拷贝：`hipMemcpy` 或 `hipMemcpyAsync`
4. Kernel 启动
5. 同步与回收：`hipDeviceSynchronize` / `hipFree`

## 常见错误

- `hipErrorInvalidValue`: 参数空指针或长度不合法
- `hipErrorOutOfMemory`: 显存不足
- `hipErrorInvalidDevice`: 设备号越界
