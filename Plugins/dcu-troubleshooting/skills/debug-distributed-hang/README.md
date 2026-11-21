# debug-distributed-hang — Debugging Distributed Hangs in SGLang

调试 SGLang 分布式（TP / PP / DP / EP）中的**卡死/挂起**：区分真正的集体死锁与慢 rank 长尾、崩溃或
资源耗尽。通过逐 rank 日志、张量哈希 diff、二分定位根因。

## 核心内容

- **确认并定位挂起**：watchdog / py-spy、NCCL debug 日志、CUDA coredump、识别 collective
- 逐 rank 日志：记录模式、该记什么、对大张量哈希、避免隐式同步
- **diff 找分叉点**：基础 diff / 事件计数 / 找首个分叉
- **二分根因**：识别输入 → 跨 rank diff 输入 → 递归
- 常见根因与修复：浮点非确定性、随机数分歧、条件代码路径、PP send/recv 不匹配
- 验证修复

## 衔接

- 确认是集体死锁后再深入；若是单卡数值问题转 `dcu-vllm-precision` / `dcu-train-precision`。

详见 `SKILL.md`。
