# debug-cuda-crash — Debugging CUDA/HIP Kernel Crashes

用 **kernel API 预调用日志** 调试 SGLang / vLLM 中的 CUDA/HIP kernel crash（非法内存访问、设备侧断言、
NaN/Inf、OOM 等）。通过预调用记录、同步复现、张量校验与架构感知工具定位根因。

## 核心内容

- **Kernel API 日志**：基础（函数名）/ 详细（带输入元数据）/ 完整（张量统计）/ crash-safe（执行前保存输入）
- 复现 LLM CUDA crash + 多进程调试
- 过滤 Level-10 dumps
- 常见 CUDA 错误与排查表（Illegal Memory Access / Device-Side Assert / NaN-Inf / OOM）
- 与 compute-sanitizer 结合、kernel 级 `printf()` 调试
- 环境变量参考、最佳实践、陷阱（无日志 / 输出过多 / CUDA Graph capture 期间跳过统计）

## 衔接

- 若在线程栈指向 vLLM/SGLang 内部，可再用 `dcu-vmfault-coredump`（VM fault 原始取证）交叉验证。

详见 `SKILL.md`。
