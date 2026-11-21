# dcu-framework-profilers（六大框架 Profiler 实战）

用户向 README。

## 是什么
框架级（性能分析五层体系的第二层）Profiler 实战手册：在环境对齐（`dcu-env`）之后、Kernel 级（`dcu-hipprof`）之前，先用框架 Profiler 拿算子总耗时排序，确定瓶颈是计算 / 搬运 / 通信。来源：《DCU性能分析及常用优化技术手册》Ch3。

## 覆盖框架
| 框架 | Profiler 工具 | 关键关注 |
|---|---|---|
| PyTorch | `torch.profiler` | Kernel 耗时、显存、CPU-GPU 重叠、通信占比 |
| TensorFlow | TF Profiler | Step 时间、输入流水线、XLA、Op 统计 |
| ONNXRuntime | ORT Session Profiler | 节点级耗时、EP 选择效率、内存模式 |
| MIGraphX | DCU Profiler + `MIGRAPHX_TRACE_*` | 算子融合、HIP Kernel、编译优化 |
| vLLM | PyTorch Profiler + VLLM profile | PagedAttention、调度延迟、KV Cache |
| Megatron / LlamaFactory | PyTorch Profiler | 通信-计算重叠、TP/PP/DP、LoRA 效率 |

> DCU 上 `torch.cuda.*` 自动映射 HIP，示例以 NVIDIA 命名书写，ROCm 映射可直接复用。

## 每框架给什么
启用方式 + 关注指标 + 核心代码示例 + 常见瓶颈信号。

## 下一步路由
- 框架定位到热点算子 → Kernel 级深挖：`dcu-hipprof`（Trace/PMC）或 `dcu-perf-profiling`（rocprof/MIopen）。
- 分布式通信瓶颈 → `dcu-hipprof` / `dcu-rccl-test`。
- 优化方案要验证闭环 → `dcu-op-optimization`（强制正确性 + 性能量化对比）。
- 真实跨平台对标案例 → `dcu-perf-cases`。
- SGLang 实战（部署参数对性能影响极大，坑在启动参数而非 profiler）→ 见 `dcu-vllm-deploy` 4.2 + `dcu-perf-cases` 案例 7。

## 相关 skill
dcu-perf-analysis（总纲）/ dcu-perf-profiling / dcu-hipprof / dcu-op-optimization / dcu-perf-cases

## 溯源
主手册 Ch3：内部文档 file_id=<internal-file-id>
