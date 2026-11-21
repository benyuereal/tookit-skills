# DCU/HCU LLM 推理优化历史规则

本参考提炼自 dcu-adapt 的多个 worktree：BW10 vLLM 0.23、BW1000
SGLang 0.5.14/0.5.15 DSpark、K100AI FP4/W8A8、BW1100 DeepSeek-V4-Pro
和 BW1000 GLM-5.2。它记录可复用的判断规则，不固定某次实验的绝对性能。

## 目录

1. 实验契约
2. 正确性门禁
3. 性能门禁
4. Profiling 与归因
5. 平台与集成风险
6. 已反复验证的模式
7. 轮次归档

## 1. 实验契约

每轮只验证一个主 hypothesis，并在运行前固定：

- 模型、checkpoint 哈希、精度和量化格式。
- 框架 commit、补丁集合、共享库哈希和启用开关。
- GPU 型号、架构、数量、DTK/驱动、容器或 Notebook 生命周期。
- TP/PP/DP/EP、图模式、attention/MoE 后端和显存参数。
- 输入/输出长度、并发、temperature、ignore_eos、数据集和随机种子。
- control、candidate、接受阈值、失败阈值、回滚入口和证据目录。

不要把不同 worktree、不同启动或不同宿主窗口的绝对数字直接相减。

## 2. 正确性门禁

按以下顺序 fail closed：

1. 编译和加载：目标架构正确，符号存在，实际路径命中新产物。
2. Operator oracle：覆盖生产 shape、dtype、stride、边界值和多 rank。
3. 数值语义：检查 NaN/Inf、max/mean error；量化、归约和采样路径优先要求
   bit-exact。仅 cosine=1.0 不足以证明语义一致。
4. Graph/eager 等价：捕获前后逐位比较，覆盖 warmup、稳态和 fallback。
5. E2E token：固定 prompt 的 output IDs/hash、精确输入/输出 token 数一致。
6. 并发隔离：至少覆盖 c=1 和目标并发，防止 KV/cache/workspace 交叉污染。
7. 质量与稳定性：必要时跑 GSM8K/MMLU；持续请求、health、VRAM 和错误日志通过。

“输出看起来合理”和“服务返回 200”都不是充分正确性证据。IDs 一致也不能证明
候选路径已激活；必须同时提供路径命中、map、日志或 profiler 证据。

## 3. 性能门禁

- 先做 operator gate，再把节省量乘以稳态调用次数，投影 E2E 上限。
- 投影低于噪声或预注册门槛时停止。历史常用门槛为至少 0.5 ms 或 1%，具体由
  工作负载方差决定。
- 端到端使用同实例、同窗口、交错 control/candidate；至少三次方向一致。
- 运行前检查同租户 GPU 进程、利用率、显存、温度、频率、VMFault/RAS 和服务残留。
- 报告 median、离散度、原始样本和 TPOT 公式。长输出 TPOT 使用
  `(total_time_ms - ttft_ms) / (output_tokens - 1)`。
- 候选只有在正确性、E2E、稳定性和回滚门全部通过后才能进入 accepted 栈。

不要连续叠加未验证候选。Rejected 结果与 accepted 结果同样归档。

## 4. Profiling 与归因

- 先判断 dispatch/launch-bound、compute-bound、memory-bound、collective-bound，或
  speculation acceptance-bound。
- 图模式可能隐藏 Python/source mapping；必要时抓 graph-off mapping trace 和
  graph-on formal trace，再用墙钟和独立 harness 交叉验证。
- profiler 会膨胀时间；kernel 合计可以超过墙钟。必须解释 overlap 和 profiler
  inflation，不要强行让表格相加等于 HTTP 时间。
- 建立互斥预算组：attention、dense GEMV、MoE、collective、elementwise/reduction、
  copy、scheduler/glue、draft/verify。每组记录当前预算、可回收上限和闭合证据。
- 优化后重新 profile。瓶颈会从 dispatch 转移到 GEMV、MoE、attention 或 RCCL。
- 当观测与先验冲突时，以路径命中和 trace 为准；历史上“注意力必然是瓶颈”、
  “MoE 未融合”等假设均曾被实测否决。

## 5. 平台与集成风险

- gfx928、gfx936、gfx938 以及厂商 MACA 产物不可混用。
- 独立 kernel 快不等于 serving 快。Python/ctypes、分配、clone、layout 转换、同步和
  TP8 rank 偏斜可吃掉 5 倍以上 operator 收益。
- 每层少量 host 工作会被 collective 放大：最慢 rank 决定 AllReduce 完成时间。
- CUDA/HIP Graph、RCCL、自定义 all-reduce 和外部 AITER/CK 资产必须按目标平台
  单独验证，不能从 NVIDIA/ROCm 主线能力推断 DCU 可用。
- VMFault 可能来自候选，也可能来自宿主或同租户污染。使用同窗口 clone/control、
  精确 PID/PGID 恢复和干净重启区分因果。
- BW1000 类临时环境每次恢复都重新探测版本、文件哈希、设备、模型和补丁命中。

## 6. 已反复验证的模式

优先考虑：

- 消除真实死链和重复计算。
- 图化稳定、静态、已证明逐位等价的完整 stage block。
- 减少高频 dispatch 或把生产 shape 的多步链融合到同一 kernel。
- 对 skinny GEMV/MoE 依据真实 shape、权重字节和 HBM roofline 选择 geometry。
- 提升并发或投机解码，但必须验证 acceptance length/rate 和 draft/verify 正确性。
- 指令级替换只有在目标指令和生产路径命中得到证明时才接受。

常见无效方向：

- 在 dispatch-bound 阶段只加速单个小 kernel。
- standalone 快但需要每层 Python 包装、分配或 layout 变换的替换。
- 低于 E2E 噪声门槛的微优化。
- 未验证 graph/collective 兼容性的跨平台移植。
- 只看一次绝对基线，忽略同租户 HBM/LLC/RCCL 干扰。
- acceptance 接近零的投机解码；它通常比非投机路径更慢。

## 7. 轮次归档

每轮至少保留：

```text
round_N/
├── PLAN.md
├── REPORT.md
├── manifest.json
├── results/
│   ├── correctness.json
│   └── benchmark.json
├── artifacts/
└── rollback/
```

Manifest 记录输入、命令、commit、环境、产物哈希和证据文件。Report 明确写
`ACCEPTED`、`REJECTED`、`INCONCLUSIVE` 或 `DIAGNOSTIC ONLY`，并说明下一轮唯一
主 hypothesis。不要把远端二进制、模型、trace、凭据或大日志提交进 Skill 仓库。
