# vLLM 并行策略详解

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖张量并行（TP）、专家并行（EP）、流水线并行（PP）、数据并行（DP）与上下文并行（CP）的设计原理与实现。

---

## 目录

1. [概述](#1-概述)
2. [并行体系总览](#2-并行体系总览)
3. [Tensor Parallelism（张量并行）](#3-tensor-parallelism张量并行)
4. [Expert Parallelism（专家并行）](#4-expert-parallelism专家并行)
5. [Pipeline Parallelism（流水线并行）](#5-pipeline-parallelism流水线并行)
6. [Data Parallelism（数据并行）](#6-data-parallelism数据并行)
7. [Context Parallelism（上下文并行）](#7-context-parallelism上下文并行)
8. [并行通信后端](#8-并行通信后端)
9. [EPLB（专家并行负载均衡）](#9-eplb专家并行负载均衡)
10. [实际案例：GLM-5.1 启动参数分析](#10-实际案例glm-51-启动参数分析)
11. [并行策略选择指南](#11-并行策略选择指南)
12. [配置参数参考](#12-配置参数参考)
13. [常见问题](#13-常见问题)

---

## 1. 概述

vLLM 支持 5 种并行策略，可组合使用以支持从单卡到数百卡的大规模推理：

| 并行策略 | 缩写 | 切分维度 | 主要用途 |
|---------|------|---------|---------|
| **张量并行** | TP | 模型权重（按层内维度切分） | 单卡放不下模型权重 |
| **专家并行** | EP | MoE 的 Experts | MoE 模型横向扩展 |
| **流水线并行** | PP | 模型层（按层切分） | 超大模型跨节点 |
| **数据并行** | DP | 请求（按 batch 切分） | 提升吞吐 |
| **上下文并行** | CP | 序列长度（KV Cache 切分） | 超长序列（128K+） |

### 关键文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/distributed/parallel_state.py` | 1811 | 并行组初始化与通信协调 |
| `vllm/config/parallel.py` | 719 | ParallelConfig 配置定义 |
| `vllm/model_executor/layers/linear.py` | 1571 | TP 切分的线性层实现 |
| `vllm/model_executor/layers/fused_moe/layer.py` | 2304 | MoE 层（含 EP 支持） |
| `vllm/model_executor/layers/fused_moe/config.py` | 1173 | MoE 并行配置 |
| `vllm/distributed/device_communicators/all2all.py` | 606 | EP all2all 通信实现 |
| `vllm/distributed/eplb/eplb_state.py` | 1204 | 专家并行负载均衡 |
| `vllm/v1/worker/worker_base.py` | 375 | Worker 基类（PP 相关） |
| `vllm/v1/engine/coordinator.py` | 394 | DP 协调器 |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | 594 | TP 切分的 Embedding 层 |

---

## 2. 并行体系总览

### 2.1 并行拓扑

vLLM 的并行布局为 **5 维拓扑**：

```
布局顺序: ExternalDP × DP × PP × PCP × TP
```

```
例：world_size=16, DP=2, PP=2, TP=4

all_ranks = torch.arange(16).reshape(
    -1,   # ExternalDP (自动)
    2,    # DP
    2,    # PP
    1,    # PCP (默认1)
    4,    # TP
)

结果:
  ExternalDP_0:
    DP_0: [PP_0: [TP_0,TP_1,TP_2,TP_3], PP_1: [TP_0,TP_1,TP_2,TP_3]]
    DP_1: [PP_0: [TP_0,TP_1,TP_2,TP_3], PP_1: [TP_0,TP_1,TP_2,TP_3]]
```

### 2.2 并行组初始化

`initialize_model_parallel()` 按以下顺序创建各组：

```
initialize_model_parallel(TP=4, PP=2, DP=2, PCP=1, DCP=1)
  │
  ├── 1. _TP  ← view(-1, TP).unbind()  每组 4 个 GPU
  ├── 2. _DCP ← reshape(-1, DCP).unbind()
  ├── 3. _PCP ← transpose(PCP_dim).reshape(-1, PCP).unbind()
  ├── 4. _PP  ← transpose(PP_dim).reshape(-1, PP).unbind()
  ├── 5. _DP  ← transpose(DP_dim).reshape(-1, DP).unbind()
  └── 6. _EP  ← transpose(DP_dim, PP_dim).reshape(-1, DP×PCP×TP).unbind()
                 (仅 MoE 模型)
```

### 2.3 GroupCoordinator（组协调器）

每个并行组由 `GroupCoordinator` 管理：

```python
class GroupCoordinator:
    rank: int               # 组内 rank
    world_size: int         # 组大小
    ranks: list[int]        # 全局 rank 列表
    device_group: ProcessGroup  # NCCL 通信组
    cpu_group: ProcessGroup     # Gloo 通信组（CPU）
    device_communicator: DeviceCommunicatorBase | None  # 自定义通信器

    # 通信原语
    all_reduce(tensor)
    all_gather(tensor, dim)
    broadcast(tensor, src)
    send(tensor, dst) / recv(size, dtype, src)
    dispatch(hidden_states, router_logits)  # EP all2all
    combine(hidden_states)                  # EP all2all
```

### 2.4 world_size 计算公式

```python
# 引擎内部
world_size = PP × TP × PCP

# 包含 DP
world_size_across_dp = world_size × DP

# 总 GPU 数
total_gpus = ExternalDP × DP × PP × PCP × TP
```

---

## 3. Tensor Parallelism（张量并行）

### 3.1 原理

将模型的权重矩阵按列或行切分到多个 GPU，每次前向/反向通过 **all-reduce** 同步。

### 3.2 切分方式

#### ColumnParallelLinear（列并行）

```
原始: Y = X · A   (A: [input_size, output_size])

TP=4 切分:
  A = [A_0, A_1, A_2, A_3]   每块 [input_size, output_size/4]
  Y_i = X · A_i              每个 GPU 计算一部分

  输出: Y = [Y_0, Y_1, Y_2, Y_3]  (分布在 4 个 GPU)
```

实现（`linear.py:485`）：

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, ...):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = output_size // self.tp_size
        # 权重: [input_size, output_size_per_partition]

    def forward(self, input_):
        output = self.quant_method.apply(self, input_)  # X · A_i
        if self.gather_output:
            output = tensor_model_parallel_all_gather(output, dim=-1)
        return output, bias
```

**用于**：Q/K/V 投影、FFN 的第一层（gate_proj, up_proj）

#### QKVParallelLinear

继承 `ColumnParallelLinear`，将 Q/K/V 三个权重矩阵拼接在一起切分：

```python
class QKVParallelLinear(ColumnParallelLinear):
    # head_size × (num_heads + num_kv_heads × 2)
    # 按 head 维度切分：Q_heads_0, K_heads_0, V_heads_0 | Q_heads_1, ...
```

#### MergedColumnParallelLinear

继承 `ColumnParallelLinear`，将 gate_proj 和 up_proj 拼接在一起。

#### RowParallelLinear（行并行）

```
原始: Y = X · A   (A: [input_size, output_size])

TP=4 切分:
  A = [A_0; A_1; A_2; A_3]   每块 [input_size/4, output_size]
  Y_i = X_i · A_i             每个 GPU 用部分输入计算

  Y = all_reduce(Y_0 + Y_1 + Y_2 + Y_3)
```

实现（`linear.py:1377`）：

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, ...):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = input_size // self.tp_size

    def forward(self, input_):
        output = self.quant_method.apply(self, input_)  # X_i · A_i
        output = tensor_model_parallel_all_reduce(output)  # 求和
        return output, bias
```

**用于**：O 投影（attention output）、FFN 的第二层（down_proj）

#### VocabParallelEmbedding

```python
# Embedding 权重: [vocab_size, hidden_size]
# TP 切分: 按 vocab_size 维度切分
# 输入 token_id 映射到对应 partition
# 输出通过 all_reduce 或不做（取决于配置）
```

### 3.3 TP 通信模式

```
Attention:
  QKV Proj (ColumnParallel) → 各 GPU 计算部分 head 的 Attention
    → O Proj (RowParallel) → all_reduce

FFN:
  gate/up Proj (ColumnParallel) → 各 GPU 计算部分神经元
    → down Proj (RowParallel) → all_reduce

Embedding:
  VocabParallelEmbedding → (可选 all_reduce)
```

### 3.4 TP 关键特性

- **通信量**：每层 2 次 all-reduce（RowParallelLinear 输出时）
- **显存节省**：每 GPU 权重 = 原始权重 / TP
- **最佳 GPU 数**：TP=2, 4, 8（需 head 数可整除）
- **限制**：TP 组内 GPU 需要高带宽互联（NVLink/NVSwitch）

---

## 4. Expert Parallelism（专家并行）

### 4.1 原理

将 MoE 模型的 **Experts 分布到多个 GPU**，每个 GPU 持有部分完整的 Expert。Token 通过 **all2all** 通信路由到对应的 Expert 所在 GPU。

### 4.2 与 TP 的关系

启用 EP 后，MoE 层内部 **不再使用 TP**：

```
未启用 EP (TP=8):
  每个 GPU: 1/8 的所有 Expert 权重 (TP 切分)

启用 EP (TP=8):
  每个 GPU: 1/8 数量的完整 Expert 权重 (EP 切分)
  ep_size = tp_size, tp_size(在MoE层) = 1
```

这意味着 `FusedMoEParallelConfig.make()` 中：

```python
if use_ep:
    ep_size = tp_size
    ep_rank = tp_rank
    return FusedMoEParallelConfig(
        tp_size=1, tp_rank=0,   # MoE 内部不再有 TP
        ep_size=ep_size, ep_rank=ep_rank, ...
    )
```

### 4.3 Expert 分布策略

#### Linear（线性分布）

```
8 experts, ep_size=4:
  GPU 0: experts [0, 1]
  GPU 1: experts [2, 3]
  GPU 2: experts [4, 5]
  GPU 3: experts [6, 7]
```

#### Round-Robin（轮询分布）

```
8 experts, ep_size=4:
  GPU 0: experts [0, 4]
  GPU 1: experts [1, 5]
  GPU 2: experts [2, 6]
  GPU 3: experts [3, 7]
```

适合有分组专家（expert_group）的模型，有利于负载均衡。

### 4.4 All2All 通信

MoE 前向需要两次 all2all：

```
Forward:
  1. dispatch: tokens → expert 所在 GPU (all2all)
  2. expert compute: 各 GPU 独立计算自己的 experts
  3. combine: 计算结果 → 原始 GPU (all2all)
```

vLLM 支持多种 all2all 后端：

| 后端 | 场景 | 特点 |
|------|------|------|
| `allgather_reducescatter` | 默认 | 基于 allgather + reducescatter 实现 |
| `naive` | 通用 | 基于 broadcast 的简单实现 |
| `pplx` | NVIDIA | pplx 优化 kernel，可处理重复 token |
| `deepep_high_throughput` | NVIDIA | DeepEP 高吞吐 kernel |
| `deepep_low_latency` | NVIDIA | DeepEP 低延迟 kernel |
| `mori` | — | mori 优化 kernel |
| `flashinfer_all2allv` | NVIDIA (MNNVL) | FlashInfer all2allv，需要 NVSwitch |

### 4.5 EP 数据流

```
输入: hidden_states [num_tokens, hidden_size] + router_logits [num_tokens, num_experts]

1. Router: top-k 选择 experts
2. dispatch(hidden_states, router_logits):
   - all2all 发送 token 到对应 expert 的 GPU
   - 输入: [local_tokens, hidden]
   - 输出: [routed_tokens, hidden] (分布在各 expert GPU)
3. Expert Compute: 各 GPU 独立执行自己的 experts
4. combine( expert_output):
   - all2all 发送结果回原始 GPU
   - 输入: [routed_tokens, hidden]
   - 输出: [local_tokens, hidden]
5. (可选) shared_experts: 所有 GPU 都有的共享 experts
```

### 4.6 Sequence Parallel MoE

当 `TP > 1` 且 `DP > 1` 时，可启用 **序列并行 MoE**（`use_sequence_parallel_moe`）：

- Attention 层的 all-reduce 会复制 token 到所有 TP rank
- 但 EP dispatch 前先做 **reduce_scatter** 去除重复 token
- 避免重复计算和通信

---

## 5. Pipeline Parallelism（流水线并行）

### 5.1 原理

将模型按层切分为多个 **stage**，每个 stage 在独立 GPU 上运行。通过 **send/recv** 在 stage 间传递中间结果。

### 5.2 层分配

```
例：60 层模型, PP=3

Stage 0 (GPU 0-3): Layers 0-19   (前 20 层)
Stage 1 (GPU 4-7): Layers 20-39  (中间 20 层)
Stage 2 (GPU 8-11): Layers 40-59 (最后 20 层)
```

### 5.3 PP 通信模式

```
Stage 0               Stage 1               Stage 2
  │                      │                      │
  ├─ Layers 0-19 ──send─→├─ Layers 20-39 ─send─→├─ Layers 40-59
  │                      │                      │
  │                 recv←─┤                 recv←─┤
  ▼                      ▼                      ▼
```

每个 stage 通过 `intermediate_tensors` 传递 `hidden_states`。

### 5.4 调度方式

- **同步 PP**（`VLLM_USE_PP_SYNC=1`）：每个 stage 完成后 `torch.cuda.synchronize()`
- **异步 PP**（默认）：通过 NCCL send/recv 实现流水线重叠

### 5.5 PP 关键特性

- **通信量**：每层边界一次 send/recv（hidden_states 大小）
- **显存节省**：每 GPU 权重 = 原始权重 / PP
- **缺点**：GPU 利用率有气泡（idle time），需配合 micro-batch 减少
- **使用场景**：TP=8 仍放不下模型时（如 405B+ 模型）

---

## 6. Data Parallelism（数据并行）

### 6.1 原理

多个引擎副本处理不同的请求，通过协调器分配请求。

### 6.2 DP 模式

#### 内部 LB 模式（默认）

```
                 ┌─────────────┐
   Request ───→  │ DPCoordinator│  (负载均衡)
                 └──┬───┬───┬──┘
                    │   │   │
            ┌───────▼───▼───▼───────┐
            │   DPAsyncMPClient     │
            └──┬──────────────┬─────┘
               │              │
        ┌──────▼──────┐ ┌─────▼──────┐
        │ Engine 0    │ │ Engine 1   │
        │ (TP=4)      │ │ (TP=4)     │
        │ Requests    │ │ Requests   │
        │ [A,B]       │ │ [C,D]      │
        └─────────────┘ └────────────┘
```

- `DPAsyncMPClient`：根据各引擎队列长度选择目标引擎
- `DPLBAsyncMPClient`：带负载均衡的内部 LB

#### 外部 LB 模式

```
                 ┌─────────────┐
   Request ───→  │ External LB │  (K8s/HAProxy)
                 └──┬───┬───┬──┘
                    │   │   │
            ┌───────▼───▼───▼───────┐
            │  vLLM Instance 0      │
            │  (DP rank 0)          │
            └───────────────────────┘
```

- 每个 DP rank 是独立的 vLLM 实例（独立 API Server）
- 外部 LB 分发请求到不同实例

### 6.3 DP 同步

#### 同步点

1. **模型前向**：DP 组内所有 rank 必须同时调用 `execute_model`
2. **KV Cache 大小**：取所有 rank 中最小的可用显存
3. **完成状态**：`has_unfinished_dp()` 聚合所有 rank 的完成状态

```python
@staticmethod
def has_unfinished_dp(dp_group, has_unfinished):
    # all_reduce(MAX) → 任一 rank 有未完成请求，则继续
    tensor = torch.tensor([has_unfinished], dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
    return bool(tensor.item())
```

### 6.4 DP Wave 机制

请求分 **wave** 发送到所有 DP 引擎：

- 所有引擎完成当前 wave 后才开始下一个 wave
- 确保每个 wave 的 batch size 最大化

### 6.5 DP Padding

在 `get_cudagraph_and_dp_padding()` 中：
- 各 DP rank 的 token 数可能不同
- Padding 使所有 rank 的 token 数一致，避免死锁
- `num_tokens_across_dp = num_tokens_after_padding × dp_size`

### 6.6 DP 关键特性

- **通信开销**：仅同步少量元数据（Gloo all_reduce），无模型梯度通信
- **吞吐提升**：近似线性（DP 倍）
- **限制**：需 DP 组内所有 rank 同时调用 generate

---

## 7. Context Parallelism（上下文并行）

### 7.1 原理

将超长序列的 KV Cache 切分到多个 GPU，每个 GPU 只存储部分 token 的 KV Cache。

### 7.2 两种模式

#### Prefill Context Parallelism（PCP）

- 配置：`prefill_context_parallel_size`
- 加速长 prompt 的 prefill 阶段
- 多个 GPU 并行处理同一 prompt 的不同部分

#### Decode Context Parallelism（DCP）

- 配置：`decode_context_parallel_size`
- 加速长序列的 decode 阶段
- KV Cache 按 interleave 方式分布在 GPU 间

### 7.3 DCP KV Cache 分布

```
DCP=4, block_size=16, interleave_size=4:

Token 0-3   → GPU 0
Token 4-7   → GPU 1
Token 8-11  → GPU 2
Token 12-15 → GPU 3
Token 16-19 → GPU 0
...
```

每个 GPU 只存储 `1/dcp_size` 的 KV Cache，`block_size` 相应放大：

```python
if dcp_world_size * pcp_world_size > 1:
    block_size *= dcp_world_size * pcp_world_size
```

### 7.4 CP 限制

- DCP size ≤ TP size（复用 TP 组的 GPU）
- DCP + PCP 目前不兼容 Hybrid KV Cache Manager
- 需要模型支持 CP 注意力模式

---

## 8. 并行通信后端

### 8.1 通信器层次

```
GroupCoordinator
  ├── device_group (ProcessGroup)     ← PyTorch NCCL/Gloo
  ├── cpu_group (ProcessGroup)        ← Gloo (CPU 通信)
  └── device_communicator             ← 自定义通信器 (可选)
       ├── CudaCommunicator           ← CUDA IPC + NCCL
       ├── CustomAllreduce            ← 自定义 all-reduce kernel
       ├── QuickAllreduce             ← 快速 all-reduce
       ├── PyncclCommunicator         ← Python NCCL 封装
       ├── RayCommunicator            ← Ray 通信
       ├── CpuCommunicator            ← CPU 通信
       └── XpuCommunicator            ← Intel XPU 通信
```

### 8.2 Custom All-Reduce

当 `disable_custom_all_reduce=False`（默认），且 GPU 在同一节点时：

- 使用 `CustomAllreduce` 替代 NCCL all-reduce
- 通过 CUDA IPC + shared memory 实现
- 比 NCCL 减少约 30% 延迟

**自动禁用场景**：
- 多节点（`nnodes > 1`）
- 手动指定 `--disable-custom-all-reduce`
- 非 CUDA 平台

### 8.3 Message Queue Broadcaster

TP 组使用 **消息队列广播器** 进行 SchedulerOutput 分发：

- 通过 `shm_broadcast` 在 TP rank 间广播输入数据
- 避免每个 rank 独立准备输入
- `ShmObjectStorage` 管理共享内存中的 Python 对象

### 8.4 通信操作速查

| 操作 | 用途 | 实现 |
|------|------|------|
| `all_reduce` | RowParallel 输出求和、DP 状态同步 | NCCL / CustomAllreduce |
| `all_gather` | ColumnParallel 输出收集 | NCCL |
| `reduce_scatter` | Sequence Parallel MoE | NCCL |
| `broadcast` | SchedulerOutput 分发 | SHM / NCCL |
| `send/recv` | PP 中间结果传输 | NCCL P2P |
| `dispatch/combine` | EP all2all token 路由 | all2all 后端 |

---

## 9. EPLB（专家并行负载均衡）

### 9.1 原理

MoE 模型中不同 Expert 的负载可能不均衡，EPLB 通过 **动态调整 Expert 分布** 来平衡负载。

### 9.2 工作流程

```
每 window_size 步:
  1. 收集各 GPU 的 Expert 负载统计
  2. 计算最优 Expert 分配方案
  3. 通过 all2all 重新分布 Expert 权重
  4. 更新 expert_map
```

### 9.3 配置

```python
EPLBConfig:
    window_size: int = 1000         # 负载统计窗口
    step_interval: int = 3000       # 重分布间隔
    num_redundant_experts: int = 0  # 冗余 Expert 数（增加灵活性）
    log_balancedness: bool = False  # 记录负载均衡度
    use_async: bool = False         # 异步重分布
```

### 9.4 启用条件

```python
# 必须同时满足:
enable_expert_parallel=True
enable_eplb=True
TP × DP > 1
```

---

## 10. 实际案例：GLM-5.1 启动参数分析

```bash
vllm serve /models/models/GLM-5.1-INT4-w4a8 \
    -tp 8 \
    --enable-expert-parallel \
    --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --gpu-memory-utilization 0.92 \
    --max-model-len 50000 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --block-size 64 \
    --enable-chunked-prefill \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --dtype bfloat16 \
    --quantization slimquant_w4a8
```

### 10.1 并行分析

| 参数 | 值 | 含义 |
|------|---|------|
| `-tp 8` | TP=8 | 8 卡张量并行，模型权重按 head/FFN 维度切分到 8 张 GPU |
| `--enable-expert-parallel` | EP=ON | MoE 层的 8 个 Expert 分片到 8 张 GPU，每 GPU 持有完整 Expert |
| `PP` | 默认 1 | 未指定，不使用流水线并行 |
| `DP` | 默认 1 | 未指定，不使用数据并行 |
| `CP` | 默认 1 | 未指定，不使用上下文并行 |

### 10.2 并行组合效果

```
8 GPU 拓扑: ExternalDP(1) × DP(1) × PP(1) × PCP(1) × TP(8)

Dense 层 (Attention, FFN 等):
  TP=8 切分
  - Q/K/V/O Proj: 按 head 维度切分
  - gate/up/down Proj: 按 hidden 维度切分
  - 每次 RowParallel 后 all-reduce

MoE 层 (FusedMoE):
  EP=8 替代 TP=8
  - 每 GPU 持有 global_num_experts/8 个完整 Expert
  - Token 通过 all2all 路由到对应 Expert 的 GPU
  - Expert 计算完全并行，无 TP 通信
```

### 10.3 为什么同时用 TP 和 EP

GLM-5.1 是 MoE 模型，同时使用 TP 和 EP 的原因：

1. **TP 处理 Dense 层**：Attention 和普通 FFN 层通过 TP 切分
2. **EP 处理 MoE 层**：Expert 权重通常远大于 Dense 层，EP 避免了 MoE 层的 all-reduce 开销
3. **互补**：Dense 层数量少但每层计算量大（TP 合适），MoE 层 Expert 多且独立（EP 合适）

### 10.4 其他参数含义

| 参数 | 含义 |
|------|------|
| `--disable-custom-all-reduce` | 禁用自定义 all-reduce，使用 NCCL（多节点必需，这里可能是兼容性考虑） |
| `--quantization slimquant_w4a8` | INT4 权重 + FP8 激活量化，大幅减少显存 |
| `--max-model-len 50000` | 支持 50K 上下文 |
| `--block-size 64` | KV Cache block 64 tokens（配合 FlashAttention） |
| `--enable-chunked-prefill` | 长 prompt 分块处理，避免阻塞 decode |
| `--max-num-batched-tokens 8192` | 每步最多处理 8192 tokens |

---

## 11. 并行策略选择指南

### 11.1 决策流程

```
模型是否能放入单卡?
  ├── 是 → TP=1, PP=1
  │   ├── 需要更高吞吐? → DP=2+
  │   └── MoE 模型? → EP (配合 TP 的 GPU 数)
  │
  └── 否 → 需要多少卡?
       ├── 2-4 卡 → TP=2 或 4
       ├── 4-8 卡 → TP=4 或 8
       │   └── MoE 模型? → TP + EP
       └── 8+ 卡 → TP × PP
            └── MoE 模型? → TP × EP 或 TP × PP × EP
```

### 11.2 场景推荐

| 场景 | 推荐策略 | 示例命令 |
|------|---------|---------|
| 7B Dense，单卡 80GB | TP=1 | `vllm serve model` |
| 70B Dense（~140GB） | TP=2 或 TP=4 | `vllm serve model -tp 4` |
| 405B Dense（~810GB） | TP=8 + PP=2 | `vllm serve model -tp 8 -pp 2` |
| MoE（如 GLM-5.1, DeepSeek-V3） | TP + EP | `vllm serve model -tp 8 --enable-expert-parallel` |
| 高吞吐在线服务 | DP=2+ | `vllm serve model --data-parallel-size 2` |
| 超长序列（128K+） | +CP | `vllm serve model --prefill-context-parallel-size 2` |
| 多节点 | TP + PP + DP | `vllm serve model -tp 8 -pp 2 --data-parallel-size 2` |

### 11.3 约束条件

| 策略 | 约束 |
|------|------|
| TP | `num_heads % TP == 0`，TP ≤ 单节点 GPU 数 |
| PP | 层数可被 PP 整除（否则自动 padding），需 send/recv 支持 |
| EP | 仅 MoE 模型，`num_experts % ep_size == 0` |
| DP | DP 组内需同步调用 generate |
| CP | PCP 仅 prefill，DCP ≤ TP，不兼容 Hybrid KV Cache |
| TP+EP | MoE 内部 ep_size = tp_size（EP 替代 TP） |

---

## 12. 配置参数参考

### 12.1 ParallelConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tensor_parallel_size` | 1 | 张量并行度 |
| `pipeline_parallel_size` | 1 | 流水线并行度 |
| `data_parallel_size` | 1 | 数据并行度 |
| `prefill_context_parallel_size` | 1 | Prefill 上下文并行度 |
| `decode_context_parallel_size` | 1 | Decode 上下文并行度 |
| `enable_expert_parallel` | False | 启用专家并行 |
| `enable_eplb` | False | 启用专家负载均衡 |
| `expert_placement_strategy` | "linear" | Expert 分布策略 (linear/round_robin) |
| `all2all_backend` | "allgather_reducescatter" | EP all2all 后端 |
| `disable_custom_all_reduce` | False | 禁用自定义 all-reduce |
| `distributed_executor_backend` | None (自动) | 执行器后端 (mp/ray/uni) |
| `nnodes` | 1 | 节点数 |
| `data_parallel_backend` | "mp" | DP 后端 (mp/ray) |
| `data_parallel_external_lb` | False | 外部 LB 模式 |
| `data_parallel_hybrid_lb` | False | 混合 LB 模式 |

### 12.2 EPLBConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 1000 | 负载统计窗口（步数） |
| `step_interval` | 3000 | 重分布间隔（步数） |
| `num_redundant_experts` | 0 | 冗余 Expert 数 |
| `log_balancedness` | False | 记录负载均衡度 |
| `use_async` | False | 异步重分布 |

### 12.3 相关环境变量

```bash
VLLM_DP_SIZE=1                  # DP 大小
VLLM_DP_RANK=0                  # DP rank
VLLM_DP_MASTER_IP=127.0.0.1     # DP master IP
VLLM_DP_MASTER_PORT=29500       # DP master 端口
VLLM_USE_PP_SYNC=0              # PP 同步模式
VLLM_ALL2ALL_BACKEND=allgather_reducescatter  # (已废弃，用 --all2all-backend)
```

---

## 13. 常见问题

### 13.1 TP 相关

| 问题 | 原因 | 解决 |
|------|------|------|
| TP 初始化失败 | head 数不能被 TP 整除 | 调整 TP 为 head 数的因数 |
| TP 性能不达预期 | 通信瓶颈 | 确保 GPU 间有 NVLink，减少跨节点 TP |
| all-reduce 报错 | 自定义 all-reduce 不兼容 | 设置 `--disable-custom-all-reduce` |

### 13.2 EP 相关

| 问题 | 原因 | 解决 |
|------|------|------|
| EP 报 "not initialized" | 非 MoE 模型 | 只有 MoE 模型才创建 EP 组 |
| EP 负载不均 | Expert 分布不均衡 | 启用 `--enable-eplb` |
| all2all 性能差 | 后端不合适 | 尝试 `--all2all-backend deepep_low_latency` |

### 13.3 DP 相关

| 问题 | 原因 | 解决 |
|------|------|------|
| DP 死锁 | rank 间调用不同步 | 确保所有 DP rank 同时调用 generate |
| DP 吞吐无提升 | batch size 已达上限 | 增大 `max_num_seqs` 和 `max_num_batched_tokens` |

### 13.4 PP 相关

| 问题 | 原因 | 解决 |
|------|------|------|
| PP 气泡大 | micro-batch 不够 | 增大并发请求数 |
| PP 通信慢 | 跨节点 send/recv | 确保节点间有高速网络（RDMA） |

### 13.5 一般问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 多节点初始化失败 | 网络配置问题 | 检查 `master_addr`、`master_port`、`nnodes` |
| GPU 显存分布不均 | 并行策略导致 | 正常现象，EP/TP/PP 各有不同的显存使用模式 |
| `--disable-custom-all-reduce` 何时需要 | 多节点或兼容性问题 | 多节点自动禁用，单节点一般不需要 |

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-07-15  
> **核心文件**：`vllm/distributed/parallel_state.py`, `vllm/config/parallel.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/fused_moe/config.py`