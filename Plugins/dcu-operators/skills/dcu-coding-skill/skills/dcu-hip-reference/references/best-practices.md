# HIP Best Practices

- 优先使用异步 API（`hipMemcpyAsync` + stream）提高并发度
- 对频繁调用的 API 保持错误码检查：`hipGetLastError` / `hipGetErrorString`
- 大规模 kernel 调优前先确认内存访问模式与对齐
- 在迁移场景中，先保证功能等价，再做性能优化
