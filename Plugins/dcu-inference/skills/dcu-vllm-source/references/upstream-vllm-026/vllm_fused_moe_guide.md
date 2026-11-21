# vLLM FusedMoE 模块详解

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖 MoE 层的设计、路由算法、专家并行（EP）通信、量化 kernel 与 EPLB 负载均衡。

---

## 目录

1. [概述](#1-概述)
2. [模块架构](#2-模块架构)
3. [FusedMoE 层（layer.py）](#3-fusedmoe-层layerpy)
4. [配置数据结构（config.py）](#4-配置数据结构configpy)
5. [MoE 路由子系统（router/）](#5-moe-路由子系统router)
6. [DeepSeek-V3 分组路由算法](#6-deepseek-v3-分组路由算法)
7. [模块化 Kernel 抽象（modular_kernel.py）](#7-模块化-kernel-抽象modular_kernelpy)
8. [核心 Fused MoE Kernel（fused_moe.py）](#8-核心-fused-moe-kernelfused_moepy)
9. [专家并行（EP）All2All 通信](#9-专家并行ep-all2all-通信)
10. [量化 Kernel 矩阵](#10-量化-kernel-矩阵)
11. [EPLB 专家负载均衡](#11-eplb-专家负载均衡)
12. [共享专家（Shared Experts）](#12-共享专家shared-experts)
13. [权重加载与切分](#13-权重加载与切分)
14. [配置参数参考](#14-配置参数参考)
15. [实际案例：GLM-5.1 MoE 推理](#15-实际案例glm-51-moe-推理)
16. [常见问题](#16-常见问题)

---

## 1. 概述

MoE（Mixture of Experts）是现代大模型（DeepSeek-V3、GLM-5.1、Qwen3-MoE 等）的核心结构。每个 token 只激活部分 Expert，以更少的计算量换取更大的参数容量。vLLM 的 FusedMoE 模块是整个代码库中**最大、最复杂的子模块**（约 50 个文件，2.4 万行），负责：

1. **路由（Routing）**：根据 router logits 选择 top-k 个 Expert
2. **专家计算**：融合的 GEMM + 激活 + GEMM
3. **专家并行通信**：通过 all2all 在 EP 组间路由 token
4. **量化**：支持 fp8/int8/int4/nvfp4/mxfp4 等多种精度
5. **负载均衡**：EPLB 动态重分布 Expert

### 关键文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/model_executor/layers/fused_moe/layer.py` | 2304 | FusedMoE 主类（nn.Module 封装） |
| `vllm/model_executor/layers/fused_moe/config.py` | 1173 | FusedMoEConfig / FusedMoEParallelConfig / FusedMoEQuantConfig |
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | 2670 | 核心 Triton kernel + fused_experts_impl |
| `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` | 1120 | Batched 格式 kernel（EP 用） |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | 1429 | 模块化 kernel 抽象基类 |
| `vllm/model_executor/layers/fused_moe/router/` | ~1500 | 路由算法（5 种 Router） |
| `vllm/model_executor/layers/fused_moe/all2all_utils.py` | 209 | EP all2all handler 工厂 |
| `vllm/model_executor/layers/fused_moe/*_prepare_finalize.py` | ~1600 | 各 all2all 后端实现 |
| `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` | 243 | Token-Expert 块对齐 |
| `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` | 171 | TopK 权重应用与归约 |
| `vllm/model_executor/layers/fused_moe/utils.py` | 538 | 量化、激活、工具函数 |

---

## 2. 模块架构

### 2.1 整体数据流

```
hidden_states [M, H]  +  router_logits [M, E]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    FusedMoE.forward()                     │
│                                                          │
│  [Router] select_experts(hidden, logits)                 │
│     → topk_weights [M, K], topk_ids [M, K]              │
│     → (EPLB: logical→physical 映射 + 负载记录)            │
│                                                          │
│  [Quantize-Dispatch]  prepare(a1, topk_weights, ids)     │
│     → 量化输入 + (EP: all2all dispatch 发送 token)        │
│                                                          │
│  [Permute-Experts-Unpermute]  fused_experts.apply()      │
│     → moe_align_block_size (token 按 expert 排序)         │
│     → GEMM1: X @ W1^T  (gate-up projection)              │
│     → Activation: silu(gate) * up                        │
│     → GEMM2: A2 @ W2^T (down projection)                 │
│     → (EP: unpermute)                                    │
│                                                          │
│  [Combine]  finalize(output, fused_output, weights, ids) │
│     → 应用 topk_weights + 跨 K 归约                       │
│     → (EP: all2all combine 发送结果回原 GPU)              │
│                                                          │
│  (可选) shared_experts: 所有 token 都经过的共享专家        │
└──────────────────────────────────────────────────────────┘
       │
       ▼
output [M, H]
```

### 2.2 组件分层

```
FusedMoE (nn.Module)
├── FusedMoEParallelConfig   — 并行配置 (TP/DP/PCP/EP)
├── FusedMoEConfig           — 运行时配置 (维度/dtype/激活)
├── FusedMoERouter           — 路由器
│   ├── FusedTopKRouter          (默认 softmax/sigmoid topk)
│   ├── FusedTopKBiasRouter      (带 bias 的 topk)
│   ├── GroupedTopKRouter        (DeepSeek 分组 topk)
│   ├── CustomRoutingRouter      (自定义路由, 如 Llama4)
│   └── RoutingSimulatorRouter   (测试用模拟路由)
├── FusedMoEMethodBase       — 计算方法
│   ├── UnquantizedFusedMoEMethod  (无量化)
│   ├── FusedMoEModularMethod       (模块化: prepare+gemm+finalize)
│   └── 各量化方法 (fp8/gptq/awq/cutlass/deepep...)
├── _expert_map              — 全局→本地 Expert 映射 (EP)
├── eplb_state               — EPLB 负载均衡状态
└── shared_experts           — 共享专家 (可选, 模型子类提供)
```

---

## 3. FusedMoE 层（layer.py）

### 3.1 核心类

```python
class FusedMoE(CustomOp):
    def __init__(
        num_experts: int,                      # 逻辑 Expert 数
        hidden_size: int,
        intermediate_size: int,
        top_k: int,                            # 每 token 激活的 Expert 数
        num_expert_group: int | None = None,   # 分组数 (DeepSeek)
        topk_group: int | None = None,         # 每组选 topk_group 个组
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        pcp_size: int | None = None,
        scoring_func: str = "softmax",         # softmax / sigmoid
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,  # DeepSeek bias
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        expert_placement_strategy: "linear" | "round_robin" = "linear",
        ...
    )
```

### 3.2 初始化流程（`__init__`）

1. **并行配置**：构建 `FusedMoEParallelConfig.make(tp_size, pcp_size, dp_size, vllm_parallel_config)`，决定 TP/DP/PCP/EP 的组合
2. **Expert 数量**：`global_num_experts = num_experts + num_redundant_experts`，`logical_num_experts = num_experts`
3. **hidden_size 对齐**：`maybe_roundup_hidden_size`（DeepEP 后端要求特定对齐）
4. **Expert 映射**（EP 时）：`determine_expert_map(ep_size, ep_rank, global_num_experts, ...)` 计算 `local_num_experts`、`_expert_map`、`expert_mask`
5. **EPLB 设置**：`eplb_state = EplbLayerState()`，`expert_placement_strategy`
6. **创建路由器**：`create_fused_moe_router(...)` → 根据参数选择具体 Router 类型
7. **构建 FusedMoEConfig**：运行时配置
8. **解析量化方法**：`quant_config.get_quant_method(self, prefix)` 或 `UnquantizedFusedMoEMethod`
9. **创建权重**：`quant_method.create_weights(...)`
10. **路由捕获**（可选）：若 `enable_return_routed_experts`，设置 capture 回调

### 3.3 关键属性

```python
# 并行属性 (委托给 moe_parallel_config)
tp_size, dp_size, pcp_size, ep_size
tp_rank, dp_rank, pcp_rank, ep_rank
use_ep                       # 是否启用 EP
use_pplx_kernels             # all2all == "pplx"
use_deepep_ht_kernels        # all2all == "deepep_high_throughput"
use_deepep_ll_kernels        # all2all == "deepep_low_latency"
use_mori_kernels             # all2all == "mori"
use_dp_chunking              # 是否分块处理

# Expert 映射
expert_map                   # 全局→本地索引 (EP), -1 表示非本地
local_num_experts            # 本 rank 持有的 Expert 数
```

### 3.4 Forward 流程

```python
def forward_native(hidden_states, router_logits, ...):
    # 1. hidden_size padding (若被 roundup)
    # 2. 注册 reduce_output (TP/EP > 1 时 all-reduce)
    # 3. 调用 torch.ops.vllm.moe_forward / moe_forward_shared 自定义 op
    #    → forward_impl
    # 4. reduce + slice 回原始 hidden_size

def forward_impl(hidden_states, router_logits, ...):
    # 1. 确定 has_separate_shared_experts
    # 2. (可选) shared_experts_stream 设置 (独立 CUDA stream 并行)
    # 3. (可选) 内部 gate 计算 router_logits (overlapped 模式)
    # 4. 若 use_dp_chunking: forward_impl_chunked
    # 5. 若 do_naive_dispatch_combine: get_ep_group().dispatch(...)
    # 6. (可选) PCP all-gather
    # 7. 矩阵计算:
    #    - 若 is_monolithic: quant_method.apply_monolithic(layer, x, router_logits)
    #    - 否则: topk = router.select_experts(x, router_logits)
    #            quant_method.apply(layer, x, topk_weights, topk_ids)
    # 8. (可选) shared_experts 在独立 stream 执行
    # 9. combine_output: get_ep_group().combine(...) / pcp reduce_scatter
```

### 3.5 模块化 Kernel 切换

权重加载后，`maybe_init_modular_kernel()` 被调用：

```python
def maybe_init_modular_kernel(self):
    prepare_finalize = self.quant_method.maybe_make_prepare_finalize(routing_tables)
    if prepare_finalize is not None:
        # 使用 all2all 时，将 quant_method 包装为 FusedMoEModularMethod
        self.quant_method = FusedMoEModularMethod.make(
            self, self.quant_method, prepare_finalize, self.shared_experts)
```

- **非 EP 部署**：保持原 quant_method（monolithic 或直接 apply）
- **EP 部署**：包装为 `FusedMoEModularMethod`（prepare + gemm + finalize）

---

## 4. 配置数据结构（config.py）

### 4.1 FusedMoEParallelConfig

```python
@dataclass
class FusedMoEParallelConfig:
    tp_size, pcp_size, dp_size, ep_size: int
    tp_rank, pcp_rank, dp_rank, ep_rank: int
    use_ep: bool
    all2all_backend: str          # naive/allgather_reducescatter/pplx/deepep_*/mori
    enable_eplb: bool

    @property
    def use_all2all_kernels(self): return self.dp_size > 1 and self.use_ep
    @property
    def use_pplx_kernels(self): ...
    @property
    def use_deepep_ht_kernels(self): ...
    @property
    def use_batched_activation_format(self):
        return self.use_deepep_ll_kernels or self.use_pplx_kernels
```

**`make()` 工厂关键逻辑**：

```python
use_ep = (dp_size * pcp_size * tp_size > 1) and enable_expert_parallel

if not use_ep:
    # 非 EP: TP 切分权重, ep_size=1
    flatten_tp_size = dp_size * pcp_size * tp_size
    return FusedMoEParallelConfig(tp_size=flatten_tp_size, ep_size=1, ...)
else:
    # EP: 每设备持有完整 Expert, MoE 内部无 TP
    ep_size = flatten_tp_size   # = tp_size (DP+EP 时)
    return FusedMoEParallelConfig(tp_size=1, ep_size=ep_size, ...)
```

**`flatten_tp_across_dp_and_pcp`**：将 TP 维度展平到 DP×PCP×TP，使权重在所有设备上分片：
```python
flatten_tp_size = dp_size * pcp_size * tp_size
flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
```

### 4.2 FusedMoEConfig

运行时配置，传给 kernel：

```python
@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int          # top_k
    hidden_dim: int
    intermediate_size_per_partition: int
    num_local_experts: int
    activation: str                 # silu / gelu / swigluoai / swiglustep
    device: torch.device
    routing_method: RoutingMethodType
    moe_parallel_config: FusedMoEParallelConfig
    in_dtype: torch.dtype
    router_logits_dtype: torch.dtype
    max_num_tokens: int             # VLLM_MOE_DP_CHUNK_SIZE
    has_bias: bool
    is_act_and_mul: bool
    is_lora_enabled: bool
```

### 4.3 FusedMoEQuantConfig

量化描述，包含 4 个 `FusedMoEQuantDesc`：`_a1`（激活1）、`_a2`（激活2）、`_w1`（权重1）、`_w2`（权重2）。

```python
@dataclass
class FusedMoEQuantDesc:
    dtype: torch.dtype | str | None    # None=未量化, "nvfp4"/"mxfp4"/"int4"/...
    shape: GroupShape | None           # (-1,-1) per-tensor, (1,-1) per-row, (128,128) block
    scale: torch.Tensor | None
    alpha_or_gscale: ...               # nvfp4 alphas / W4A8 通道 scales
    zp: ...                            # int4/int8 zero points
    bias: ...                          # GPT-OSS bias
```

**关键属性**：`quant_dtype`、`is_quantized`、`is_per_act_token`、`is_block_quantized`、`block_shape`、`use_fp8_w8a8`、`use_int8_w8a8`、`use_int4_w4a16`、`use_nvfp4_w4a16`、`use_mxfp4_w4a16` 等。

**工厂函数**：`fp8_w8a8_moe_quant_config`、`int8_w8a8_moe_quant_config`、`gptq_marlin_moe_quant_config`、`nvfp4_moe_quant_config`、`mxfp4_w4a16_moe_quant_config` 等。

### 4.4 RoutingMethodType

```python
class RoutingMethodType(IntEnum):
    Default = 0           # Softmax → TopK
    Renormalize = 1       # TopK → Softmax/Sigmoid
    DeepSeekV3 = 2        # Sigmoid → Bias → 组内Top2 → 组Top4 → Expert Top8
    Llama4 = 3            # Top1 → Sigmoid
    RenormalizeNaive = 4  # Softmax/Sigmoid → TopK → Renormalize
    TopK = 5              # TopK (无 softmax)
    Custom = 6
    Simulated = 7
    Unspecified = 8
```

此 enum 与 FlashInfer TRT-LLM fused MoE runner 保持同步，用于 kernel 选择。

---

## 5. MoE 路由子系统（router/）

### 5.1 路由器抽象

```python
class FusedMoERouter(ABC):
    @property
    @abstractmethod
    def routing_method_type(self) -> RoutingMethodType: ...

    @abstractmethod
    def select_experts(hidden_states, router_logits) -> (topk_weights, topk_ids): ...
```

### 5.2 BaseRouter 模板方法

`BaseRouter` 实现模板方法模式，统一处理 EPLB：

```python
class BaseRouter(FusedMoERouter):
    def select_experts(hidden_states, router_logits):
        # 1. 验证 EPLB 状态
        # 2. 获取 indices_type (从 quant_method 回调)
        # 3. topk_weights, topk_ids = self._compute_routing(...)  # 子类实现
        # 4. topk_ids = self._apply_eplb_mapping(topk_ids)  # EPLB: logical→physical
        # 5. topk_ids = self._convert_indices_dtype(topk_ids, indices_type)
        return topk_weights, topk_ids
```

### 5.3 路由器类型

| 路由器 | 触发条件 | 算法 | routing_method_type |
|--------|---------|------|---------------------|
| **FusedTopKRouter** | 默认 | softmax/sigmoid + topk | Renormalize / RenormalizeNaive |
| **FusedTopKBiasRouter** | `e_score_correction_bias` 非 None | topk + bias | Renormalize / RenormalizeNaive |
| **GroupedTopKRouter** | `use_grouped_topk=True` | DeepSeek 分组 topk | DeepSeekV3 (sigmoid) / Unspecified |
| **CustomRoutingRouter** | `custom_routing_function` 非 None | 自定义函数 | Llama4 / Custom |
| **RoutingSimulatorRouter** | `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` | 模拟（测试用） | Simulated |

### 5.4 工厂 `create_fused_moe_router`

选择优先级（严格顺序）：

```python
def create_fused_moe_router(...):
    if envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY != "":
        return RoutingSimulatorRouter(...)
    if use_grouped_topk:
        return GroupedTopKRouter(...)
    if custom_routing_function is not None:
        return CustomRoutingRouter(...)
    if e_score_correction_bias is not None:
        return FusedTopKBiasRouter(...)
    return FusedTopKRouter(...)   # 默认
```

### 5.5 FusedTopKRouter（默认）

```python
def fused_topk(hidden_states, gating_output, topk, renormalize, ...):
    # 调用 C++ kernel: ops.topk_softmax 或 ops.topk_sigmoid
    # 在单个融合 kernel 中完成: scoring + topk 选择 + (可选) renormalize
    # ROCm: rocm_aiter_ops.topk_softmax/sigmoid
    return topk_weights, topk_ids, token_expert_indices
```

### 5.6 关键设计不变量

所有路由器共享的设计原则：

1. **Bias 只影响选择，不影响权重**：`e_score_correction_bias` 加到用于**选择** Expert 的分数上，而路由权重取自**原始无 bias** 的分数
2. **`routed_scaling_factor`** 在选择后乘到 `topk_weights`（分组路径在 renormalize 之后）
3. **`indices_type_getter` 是回调**（非值），因为决定 index dtype 的模块化 kernel 在路由器构造之后才创建
4. **EPLB 映射集中在 BaseRouter**（logical→physical + 负载记录），对所有路由算法透明
5. **Batch invariance**：`vllm_is_batch_invariant()` 控制 `torch.topk` 的 `sorted` 参数

---

## 6. DeepSeek-V3 分组路由算法

这是最重要的路由算法，DeepSeek-V2/V3、GLM-5.1 等模型使用。`GroupedTopKRouter._compute_routing` 调用 `grouped_topk()`：

### 6.1 算法流程

```
输入: router_logits [M, E], e_score_correction_bias [E]
      num_expert_group, topk_group, top_k, scoring_func="sigmoid"

1. 评分: scores = sigmoid(logits)                         # [M, E]

2. (有 bias 时) 保存原始分数: original_scores = scores
   scores = scores + e_score_correction_bias              # bias 仅用于选择

3. 组分数计算:
   组内取 top-2 求和:
   group_scores = scores.view(M, num_expert_group, -1)
                       .topk(2, dim=-1)[0].sum(dim=-1)    # [M, num_expert_group]

4. 选 topk_group 个组:
   group_idx = torch.topk(group_scores, k=topk_group)[1]  # [M, topk_group]

5. 构建 mask:
   group_mask = zeros(M, num_expert_group)
   group_mask.scatter_(1, group_idx, 1)
   score_mask = group_mask.unsqueeze(-1).expand(...).reshape(M, E)

6. 屏蔽非选中组的 Expert:
   tmp_scores = scores.masked_fill(~score_mask.bool(), -inf)

7. 最终选择:
   topk_ids = torch.topk(tmp_scores, k=top_k)[1]          # [M, top_k]
   topk_weights = original_scores.gather(1, topk_ids)     # 取无 bias 的原始分数

8. Renormalize: topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

9. Scaling: topk_weights *= routed_scaling_factor
```

### 6.2 DeepSeek-V3 典型参数

```
num_experts = 256 (routed)
num_expert_group = 8     # 256/8 = 32 experts per group
topk_group = 4           # 选 4 个组
top_k = 8                # 最终选 8 个 Expert
scoring_func = "sigmoid"
routed_scaling_factor = 2.5
```

含义：每 token 先选 4 个组（每组 32 Expert），再从这 4×32=128 个候选中选 8 个 Expert。

### 6.3 融合 kernel 路径

当满足条件时（`VLLM_USE_FUSED_MOE_GROUPED_TOPK` + CUDA + 组数≤32 + topk≤32 + 有 bias），调用融合 C++ kernel：

```python
def fused_grouped_topk(...):
    return ops.grouped_topk(...)   # 单个 kernel 完成全部算法
```

### 6.4 融合 Gate 路径

当 `VLLM_ENABLE_MOE_FUSED_GATE` + bias + 分组 topk 时，`use_fused_gate=True`，调用单个融合 gate kernel：

```python
op.moe_fused_gate(router_logits, e_score_correction_bias,
                  num_expert_group, topk_group, top_k,
                  routed_scaling_factor=..., n_share_experts_fusion=...)
```

---

## 7. 模块化 Kernel 抽象（modular_kernel.py）

将 MoE 拆解为可组合的组件：`[Router] → [Quantize-Dispatch] → [Permute-Experts-Unpermute] → [Combine]`

### 7.1 核心抽象类

```python
class FusedMoEActivationFormat(Enum):
    Standard = "standard"           # (num_tokens, hidden)
    BatchedExperts = "batched_experts"  # (num_experts, max_tokens, hidden)

class FusedMoEPrepareAndFinalize(ABC):
    """[Quantize-Dispatch] + [Combine] — 量化+分发 与 合并+归约"""
    def prepare(a1, topk_weights, topk_ids, num_experts, expert_map, ...) -> PrepareResultType
    def finalize(output, fused_expert_output, topk_weights, topk_ids, ...)
    def activation_format() -> FusedMoEActivationFormat
    def output_is_reduced() -> bool      # combine 是否已 all-reduce
    def supports_async() -> bool

class FusedMoEPermuteExpertsUnpermute(ABC):
    """[Permute-Experts-Unpermute] — 核心专家计算"""
    def apply(output, hidden_states, w1, w2, topk_weights, topk_ids, activation, ...)
    def workspace_shapes(M, N, K, topk, ...)
    def supports_chunking() -> bool
    def supports_expert_map() -> bool
    def finalize_weight_and_reduce_impl() -> TopKWeightAndReduce

class FusedMoEModularKernel(torch.nn.Module):
    """组合 PrepareAndFinalize + PermuteExpertsUnpermute"""
    def forward(a1, w1, w2, topk_weights, topk_ids, ...):
        # 1. prepare → 量化 + dispatch
        # 2. 分配 workspace
        # 3. (分块) fused_experts.apply
        # 4. finalize → combine + weight + reduce
```

### 7.2 TopKWeightAndReduce 实现

| 实现 | 用途 |
|------|------|
| `TopKWeightAndReduceNoOP` | No-op：kernel 内部已做权重+归约（Triton/DeepGemm/CUTLASS） |
| `TopKWeightAndReduceContiguous` | 对 `(M, topk, K)` 输出：应用 weights + `ops.moe_sum` |
| `TopKWeightAndReduceNaiveBatched` | 对 `(E, batch, K)` 输出：遍历 Expert 掩码累加 |
| `TopKWeightAndReduceDelegate` | 占位，由 PrepareAndFinalize 决定实际实现 |

### 7.3 调用链

```
FusedMoEModularMethod.apply(layer, x, topk_weights, topk_ids)
  → fused_experts(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight, ...)
    → prepare_finalize.prepare(x, topk_weights, topk_ids, ...)     # 量化+dispatch
    → (chunk loop) fused_experts.apply(workspace, q_x, w1, w2, ...) # GEMM1+act+GEMM2
    → prepare_finalize.finalize(output, fused_output, ...)          # combine+reduce
```

---

## 8. 核心 Fused MoE Kernel（fused_moe.py）

### 8.1 fused_experts_impl 算法

非 EP 部署的主路径（monolithic）：

```
输入: hidden_states [M, H], w1 [E, 2N, H], w2 [E, H, N], topk_weights, topk_ids

1. (可选) Marlin W16A16 快速路径
2. (可选) LMSLIM int8/fp8/w4a8 快速路径
3. 配置选择: try_get_optimal_moe_config (JSON 调优配置 / 默认)
4. Workspace 分配:
   intermediate_cache1 = [M, top_k, N]   (GEMM1 输出)
   intermediate_cache2 = [M*top_k, N]    (激活输出)
   intermediate_cache3 = [M, top_k, H]   (GEMM2 输出)

5. (分块循环, CHUNK_SIZE = VLLM_FUSED_MOE_CHUNK_SIZE):
   a. 量化输入: moe_kernel_quantize_input(hidden_states)
   b. 块对齐: moe_align_block_size(topk_ids) → sorted_token_ids, expert_ids
      (naive 模式: tokens*topk*4 <= num_experts 时跳过排序)
   c. GEMM1: dispatch_fused_moe_kernel(q_x, w1, cache1, ...)
      = X @ W1^T  (gate-up projection, 每 Expert 独立)
   d. 激活: apply_moe_activation("silu", cache2, cache1)
      = silu(gate) * up
   e. 量化中间结果: moe_kernel_quantize_input(cache2)
   f. GEMM2: dispatch_fused_moe_kernel(q_cache2, w2, cache3, ...)
      = A2 @ W2^T  (down projection, top_k=1)
   g. 归约: ops.moe_sum(cache3, output)  # 跨 top_k 维求和

6. 返回 output [M, H]
```

### 8.2 Triton GEMM Kernel

`fused_moe_kernel`（主 kernel，支持 fp8/int8/未量化）：

```python
@triton.jit
def fused_moe_kernel(...):
    # Grid: (cdiv(EM, BLOCK_M) * cdiv(N, BLOCK_N))
    # GROUP_SIZE_M 分组 program ID 提升 L2 cache 复用
    # 通过 sorted_token_ids 映射 block 行到实际 token 索引
    # 通过 expert_ids 选择 Expert 权重矩阵
    # expert_ids == -1: 写零 (EP 时非本地 Expert)
    # K 维循环 + fp32 累加器
    # 反量化 (b_scale * a_scale) 后加 bias
    # MUL_ROUTED_WEIGHT: 在 fp32 中应用路由权重
```

`fused_moe_kernel_gptq_awq`（int4_w4a16 / int8_w8a16）：

```python
@triton.jit
def fused_moe_kernel_gptq_awq(...):
    # 处理 int4 打包权重 (4-bit packing, bit shifting)
    # 支持 zero points (AWQ vs GPTQ)
    # 按 group 应用 b_scale 和 b_zp
```

### 8.3 Kernel 分发

```python
def dispatch_fused_moe_kernel(...):
    if use_int8_w8a16 or use_int4_w4a16 with block_shape[1] > 0:
        if should_moe_wna16_use_cuda(...):
            return invoke_fused_moe_wna16_cuda_kernel(...)   # ops.moe_wna16_gemm
        else:
            return invoke_fused_moe_wna16_triton_kernel(...) # fused_moe_kernel_gptq_awq
    else:
        return invoke_fused_moe_triton_kernel(...)           # fused_moe_kernel
```

### 8.4 配置选择

```python
def try_get_optimal_moe_config(E, N, K, ...):
    # 1. 运行时 override (get_config())
    # 2. 调优 JSON (configs/ 目录, 按架构/CU数命名)
    # 3. 默认配置 (get_default_config)

def get_default_config(...):
    # batch-invariant: {M:64, N:64, K:32, GROUP_SIZE_M:8}
    # FP8 block-wise: {M:64, N:block_shape[0], K:block_shape[1], GROUP_SIZE_M:32}
    # M <= E: {M:16, N:32, K:64, GROUP_SIZE_M:1}
```

### 8.5 moe_align_block_size

```python
def moe_align_block_size(topk_ids, block_size, num_experts, expert_map, ...):
    """按 Expert 排序 token，并将每个 Expert 的 token 数填充到 block_size 的倍数"""
    # 返回:
    #   sorted_token_ids: 按 Expert 排序的 token 索引 (含 padding)
    #   expert_ids: 每个 M-block 的 Expert 索引 (-1=非本地)
    #   num_tokens_post_padded: padding 后的总 token 数
    # 委托给 C++ kernel: ops.moe_align_block_size
```

### 8.6 TritonExperts（模块化版本）

```python
class TritonExperts(FusedMoEPermuteExpertsUnpermute):
    activation_format = Standard
    supports_chunking = True
    supports_expert_map = True
    finalize_weight_and_reduce_impl = TopKWeightAndReduceNoOP  # kernel 内部已做

    def apply(...):
        # 同 fused_experts_impl 但无分块 (分块由 FusedMoEModularKernel 处理)
        # moe_align_block_size + invoke_fused_moe_triton_kernel × 2 + activation + moe_sum
```

---

## 9. 专家并行（EP）All2All 通信

### 9.1 通信流程

```
[Dispatcher GPU]                    [Expert GPU]
   hidden_states                       │
   topk_ids ───────dispatch──────────→ │ (all2all 发送 token 到 Expert 所在 GPU)
       │                               ▼
       │                          expert_compute (GEMM1 + act + GEMM2)
       │                               │
       │ ←──────────combine───────────── (all2all 发送结果回原 GPU)
       ▼
   output (加权归约)
```

### 9.2 All2All 后端

| 后端 | 实现类 | activation_format | 特点 |
|------|--------|-------------------|------|
| `allgather_reducescatter` | (naive) | Standard | 默认，基于 allgather+reducescatter |
| `naive` | (naive) | Standard | 基于 broadcast 的简单实现 |
| `pplx` | `PplxPrepareAndFinalize` | BatchedExperts | PPLX 优化 kernel，支持异步，可处理重复 token |
| `deepep_high_throughput` | `DeepEPHTPrepareAndFinalize` | Standard | DeepEP 高吞吐，hidden_size 需对齐 512 字节 |
| `deepep_low_latency` | `DeepEPLLPrepareAndFinalize` | BatchedExperts | DeepEP 低延迟，仅支持特定 hidden_size，支持 FP8 dispatch |
| `mori` | `MoriPrepareAndFinalize` | BatchedExperts | Mori kernel，支持 FP8 dispatch |
| `flashinfer_all2allv` | (flashinfer) | — | FlashInfer all2allv，需 MNNVL/NVSwitch |

### 9.3 后端选择

```python
def maybe_make_prepare_finalize(moe, quant_config, routing_tables):
    if not moe.moe_parallel_config.use_all2all_kernels:
        return None   # 非 EP，无 all2all

    all2all_manager = get_ep_group().device_communicator.all2all_manager

    if moe.use_pplx_kernels:
        return PplxPrepareAndFinalize(handle, ...)
    elif moe.use_deepep_ht_kernels:
        return DeepEPHTPrepareAndFinalize(handle, ...)
    elif moe.use_deepep_ll_kernels:
        return DeepEPLLPrepareAndFinalize(handle, ..., use_fp8_dispatch=...)
    elif moe.use_mori_kernels:
        return MoriPrepareAndFinalize(handle, ..., use_fp8_dispatch=...)
```

### 9.4 PPLX 后端详解

```python
class PplxPrepareAndFinalize(FusedMoEPrepareAndFinalize):
    activation_format = BatchedExperts     # (E, max_tokens, H)
    topk_indices_dtype = uint32
    output_is_reduced = True               # combine 内部已 all-reduce
    supports_async = True                  # 支持 DBO 异步

    def prepare_async(a1, topk_weights, topk_ids, ...):
        # 1. 量化输入
        # 2. self.a2a.dispatch(out_expert_x, dp_x=a1q, indices=topk_ids, do_send=True, do_recv=False)
        # 返回 (recv_hook, receiver)

    def finalize_async(output, fused_expert_output, topk_weights, topk_ids, ...):
        # self.a2a.combine(out_tokens=output, indices=topk_ids, weights=topk_weights,
        #                  expert_y=fused_expert_output, do_send=True, do_recv=False)
        # 返回 recv_hook
```

### 9.5 DeepEP 后端特点

**High Throughput (HT)**：
- `activation_format = Standard`
- hidden_size 需对齐到 512 字节（`xfer_atom_size = 32 × 16`）
- `topk_indices_dtype = int64`
- `output_is_reduced = True`
- 支持 num_dispatchers ∈ {2,4,8,16,24,32,64,128,144,160}

**Low Latency (LL)**：
- `activation_format = BatchedExperts`
- 仅支持特定 hidden_size：`[2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]`
- 支持 FP8 dispatch（`DEEPEP_QUANT_BLOCK_SHAPE = [128, 128]`）减少数据传输
- 支持路由表（round_robin 放置时）

### 9.6 Sequence Parallel MoE

当 `TP > 1` 且 `DP > 1` 且使用 DeepEP/naive/mori 后端时，启用序列并行 MoE：

```python
@property
def use_sequence_parallel_moe(self):
    return (all2all_backend in ["allgather_reducescatter", "naive",
            "deepep_high_throughput", "deepep_low_latency", "mori"]
            and enable_expert_parallel
            and tensor_parallel_size > 1
            and data_parallel_size > 1)
```

- Attention 层的 all-reduce 会复制 token 到所有 TP rank
- EP dispatch 前先 reduce_scatter 去除重复 token，避免重复计算和通信

---

## 10. 量化 Kernel 矩阵

| 文件 | Kernel 类 | 精度 | activation_format | 选用场景 |
|------|----------|------|-------------------|---------|
| `fused_moe.py` | `TritonExperts`, `TritonWNA16Experts` | fp16/bf16/fp8/int8/int4/OCP MX | Standard | 默认 fallback，LoRA，batch-invariant |
| `fused_batched_moe.py` | `BatchedTritonExperts` | fp16/bf16/fp8 | BatchedExperts | DP+EP batched 格式 |
| `fused_marlin_moe.py` | `MarlinExperts`, `BatchedMarlinExperts` | fp8_w8a16/int4_w4a16/mxfp4/nvfp4 | Standard/Batched | SM75+，Marlin 布局 |
| `fuse_moe_w16a16_marlin.py` | `fused_experts_impl_w16a16_marlin` | w16a16 (Marlin packed) | Standard | 权重为 Marlin W16A16 打包 |
| `fused_marlin_moe_w4a16.py` | `fused_marlin_moe_int4` | int4_w4a16 (GPTQ/AWQ) | Standard | Marlin int4 GPTQ/AWQ |
| `deep_gemm_moe.py` | `DeepGemmExperts` | fp8_w8a8 (block 128)/int8 | Standard | `VLLM_USE_DEEP_GEMM=1`，N>512 |
| `batched_deep_gemm_moe.py` | `BatchedDeepGemmExperts` | fp8_w8a8 (block 128) | BatchedExperts | DeepGemm + batched |
| `triton_deep_gemm_moe.py` | `TritonOrDeepGemmExperts` | fp8_w8a8 | Standard | DeepGemm + 小 M 时 fallback Triton |
| `cutlass_moe.py` | `CutlassExpertsFp8/Fp4/W4A8Fp8` | FP8/NVFP4/W4A8 | Standard/Batched | CUTLASS kernel |
| `trtllm_moe.py` / `flashinfer_trtllm_moe.py` | — | FP8/NVFP4 | Standard | TRT-LLM fused MoE |
| `flashinfer_cutlass_moe.py` | — | NVFP4 | — | FlashInfer CUTLASS (throughput 模式) |
| `flashinfer_cutedsl_moe.py` | — | NVFP4 | — | FlashInfer CuteDSL |
| `rocm_aiter_fused_moe.py` | — | fp16/bf16/fp8 | Standard | AMD ROCm AITER |
| `cpu_fused_moe.py` | — | fp16/bf16/fp32 | Standard | CPU 后端 |
| `oracle/fp8.py` | `select_fp8_moe_backend` | fp8 | — | FP8 后端选择逻辑 |
| `oracle/nvfp4.py` | `select_nvfp4_moe_backend` | nvfp4 | — | NVFP4 后端选择逻辑 |
| `oracle/unquantized.py` | `select_unquantized_moe_backend` | 未量化 | — | 未量化后端选择逻辑 |

### 后端选择（oracle）

`oracle/` 目录实现后端选择逻辑，根据设备能力、quant scheme、parallel config 选择最优 kernel：

```python
def select_fp8_moe_backend(moe_config, weight_key, activation_key, ...):
    # 检查设备 capability (SM 版本)
    # 检查 quant scheme 支持
    # 返回 FusedMoEPermuteExpertsUnpermute 子类
```

---

## 11. EPLB 专家负载均衡

### 11.1 原理

MoE 模型中不同 Expert 的负载（被路由的 token 数）可能不均。EPLB 通过**动态重分布 Expert**（包括冗余副本）来平衡负载。

### 11.2 核心数据结构

```python
@dataclass
class EplbLayerState:
    expert_load_view: torch.Tensor | None        # 每个 physical Expert 的负载统计
    logical_to_physical_map: torch.Tensor | None # logical→physical 映射
    logical_replica_count: torch.Tensor | None   # 每个 logical Expert 的副本数
```

### 11.3 工作流程

```
每 step_interval 步:
  1. 收集各层的 expert_load_view (路由时通过 scatter_add_ 累积)
  2. 全局 all-reduce 汇总各 rank 的负载统计
  3. 计算最优 logical_to_physical_map (default policy)
     - 热门 Expert 增加副本 (分配到多个 physical 位置)
     - 冷门 Expert 减少副本
  4. 通过 all2all 在 EP rank 间重分布 Expert 权重
     - get_expert_weights() 取出本 rank 的 Expert 权重
     - rearrange_expert_weights_inplace() 重排
  5. 更新每层状态: set_eplb_state() + update_expert_map()
  6. 路由时: _apply_eplb_mapping() 将 logical ID 映射到 physical ID
     - 用 (position % replica_count) 伪随机选择副本
     - scatter_add_ 记录负载
```

### 11.4 路由集成

`BaseRouter._apply_eplb_mapping`：

```python
def _apply_eplb_mapping(self, topk_ids):
    if not self.enable_eplb:
        return topk_ids
    # eplb_map_to_physical_and_record:
    #   1. logical_to_physical_map[topk_ids] → physical_ids
    #   2. position % replica_count 选择副本
    #   3. scatter_add_ 记录负载到 expert_load_view
    return physical_ids
```

### 11.5 启用条件

```python
enable_expert_parallel = True
enable_eplb = True
TP × DP > 1
global_num_experts % ep_size == 0   # 均匀分布
```

---

## 12. 共享专家（Shared Experts）

DeepSeek-V2/V3 等模型除了 routed experts，还有 **shared experts**（所有 token 都经过）。

### 12.1 集成方式

`FusedMoE.shared_experts` 属性默认返回 `None`，模型子类（如 DeepSeekV2）重写返回共享专家模块。

### 12.2 SharedFusedMoE（重叠执行）

```python
class SharedFusedMoE(FusedMoE):
    def __init__(shared_experts, gate=None, use_overlapped=True, ...):
        # use_overlapped: 共享专家与 all2all dispatch 通信重叠执行
        # 禁用条件: EPLB + 非 allgather_reducescatter 后端, 或 flashinfer + DP, 或 marlin
```

### 12.3 独立 CUDA Stream

```python
def _maybe_setup_shared_experts_stream(hidden_states, ...):
    # 当 CUDA + 分离共享专家 + 非分块 + stream 可用 + token 数 ≤ 阈值:
    #   1. clone hidden_states
    #   2. 切换到 shared_experts_stream
    #   3. 共享专家在独立 stream 执行, 与路由/routed-expert matmul 重叠
    #   4. current_stream().wait_stream(shared_experts_stream)
```

### 12.4 Forward 输出

- `shared_experts is None`：返回单个 `output`
- `shared_experts is not None`：返回 `(shared_output, fused_output)` 元组

`must_reduce_shared_expert_outputs()`：当 `FusedMoEModularMethod.fused_experts.output_is_reduced()` 为 True 时（combine 已 all-reduce），共享专家输出需提前 reduce。

---

## 13. 权重加载与切分

### 13.1 权重布局

```python
# w13_weight (gate-up projection): [num_experts, 2*intermediate_size, hidden_size]
#   (gated 时 2N, 非 gated 时 N)
# w2_weight (down projection): [num_experts, hidden_size, intermediate_size]
#
# TP 切分: 按 intermediate_size 维度切 (N → N/tp_size)
# EP 切分: 按 num_experts 维度切 (E → local_num_experts)
# use_nn_moe: 权重转置 (.T) 用于 nn.Linear 风格布局
```

### 13.2 Expert 映射

```python
def determine_expert_map(ep_size, ep_rank, global_num_experts, strategy):
    base = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base + 1 if ep_rank < remainder else base

    expert_map = torch.full((global_num_experts,), -1, dtype=int32)
    if strategy == "linear":
        start = ep_rank * base + min(ep_rank, remainder)
        expert_map[start:start+local_num_experts] = arange(local_num_experts)
    elif strategy == "round_robin":
        local = arange(ep_rank, global_num_experts, ep_size)
        expert_map[local] = arange(local_num_experts)
    # expert_map[global_id] = local_id, -1 = 非本地
    return local_num_experts, expert_map, expert_mask
```

### 13.3 权重加载器

`FusedMoE.weight_loader`（标记 `supports_moe_loading = True`）：
- 处理 mxfp4、GGUF、BitsAndBytes、GPTQ/compressed-tensors 转置
- 支持 per-tensor/channel/group/block scales、g_idx、ModelOpt
- 通过 `_map_global_expert_id_to_local_expert_id` 跳过非本地 Expert
- EPLB 时用 `EplbState.build_initial_global_physical_to_logical_map` 构建映射

### 13.4 加载后处理

`process_weights_after_loading`（在各 quant method 中）：
- Marlin W16A16：预打包权重
- ROCm：权重 padding
- XPU：构建 IPEX 融合
- CPU：构建 SGLFusedMOE / CPUFusedMOE
- CUDA：`_setup_kernel` → 转换为运行时格式

---

## 14. 配置参数参考

### 14.1 FusedMoE 构造参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_experts` | — | 逻辑 Expert 数 |
| `top_k` | — | 每 token 激活的 Expert 数 |
| `num_expert_group` | None | 分组数（DeepSeek 分组路由） |
| `topk_group` | None | 选中的组数 |
| `scoring_func` | "softmax" | softmax / sigmoid |
| `routed_scaling_factor` | 1.0 | 路由缩放因子 |
| `e_score_correction_bias` | None | DeepSeek 评分修正 bias |
| `enable_eplb` | False | 启用 EPLB |
| `num_redundant_experts` | 0 | 冗余 Expert 数（EPLB） |
| `expert_placement_strategy` | "linear" | linear / round_robin |
| `is_sequence_parallel` | False | 序列并行 MoE |
| `n_shared_experts` | None | 共享专家数（ROCm AITER 融合） |

### 14.2 并行相关（ParallelConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_expert_parallel` | False | 启用 EP |
| `enable_eplb` | False | 启用 EPLB |
| `expert_placement_strategy` | "linear" | Expert 放置策略 |
| `all2all_backend` | "allgather_reducescatter" | EP all2all 后端 |
| `eplb_config.window_size` | 1000 | 负载统计窗口 |
| `eplb_config.step_interval` | 3000 | 重分布间隔 |
| `eplb_config.num_redundant_experts` | 0 | 冗余 Expert 数 |

### 14.3 环境变量

```bash
VLLM_FUSED_MOE_CHUNK_SIZE=32768          # fused_experts 分块大小
VLLM_MOE_DP_CHUNK_SIZE=32768             # DP 分块大小
VLLM_USE_FUSED_MOE_GROUPED_TOPK=0        # 分组 topk 融合 kernel
VLLM_ENABLE_MOE_FUSED_GATE=0             # 融合 gate kernel
VLLM_USE_DEEP_GEMM=0                     # DeepGemm FP8
VLLM_DISABLE_SHARED_EXPERTS_STREAM=0     # 禁用共享专家独立 stream
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=...  # 共享专家 stream token 阈值
VLLM_USE_MOE_W16A16_TRITON=0             # W16A16 用 Triton 而非 Marlin
VLLM_MOE_ROUTING_SIMULATION_STRATEGY=""  # 路由模拟（测试）
VLLM_MOE_ROUTER_CAPTURE=0                # 捕获 router_logits（分析）
```

---

## 15. 实际案例：GLM-5.1 MoE 推理

```bash
vllm serve /models/GLM-5.1-INT4-w4a8 \
    -tp 8 --enable-expert-parallel \
    --quantization slimquant_w4a8 ...
```

### 15.1 MoE 层执行分析

```
GLM-5.1 是 MoE 模型, 每层包含:
- Dense 部分 (Attention, norm): TP=8 切分
- MoE 部分 (FusedMoE): EP=8 替代 TP

MoE 层配置 (推测, DeepSeek 风格):
  num_experts = 256 (routed)
  num_expert_group = 8
  topk_group = 4
  top_k = 8
  scoring_func = sigmoid
  e_score_correction_bias = [256]  (从 checkpoint 加载)
  routed_scaling_factor = 2.5
  shared_experts = 共享专家模块
```

### 15.2 并行配置解析

```python
# tp_size=8, enable_expert_parallel=True
FusedMoEParallelConfig.make(tp_size=8, pcp_size=1, dp_size=1, ...):
    use_ep = (1*1*8 > 1) and True = True
    # EP 模式: 每设备持有完整 Expert, MoE 内部无 TP
    ep_size = 8
    ep_rank = tp_rank
    tp_size = 1   # MoE 内部
    → 每 GPU: 256/8 = 32 个本地 Expert
```

### 15.3 路由流程

```python
# 1. router_logits = gate(hidden_states)           # [M, 256]
# 2. router = GroupedTopKRouter(top_k=8, num_expert_group=8, topk_group=4,
#                               scoring_func="sigmoid", e_score_correction_bias=..., ...)
# 3. router.select_experts(hidden, router_logits):
#    a. scores = sigmoid(logits)                    # [M, 256]
#    b. scores += e_score_correction_bias
#    c. group_scores = scores.view(M,8,32).topk(2).sum(-1)  # [M, 8]
#    d. group_idx = topk(group_scores, k=4)         # 选 4 组
#    e. mask 非选中组
#    f. topk_ids = topk(masked_scores, k=8)         # [M, 8]
#    g. topk_weights = original_scores.gather(topk_ids)
#    h. renormalize + routed_scaling_factor
#    i. (EPLB off: 无 physical 映射)
```

### 15.4 量化路径（slimquant_w4a8）

```
W4A8: 权重 INT4, 激活 INT8
→ FusedMoEQuantConfig: use_int4_w4a8 = True
→ Kernel 选择: oracle 逻辑根据设备/形状选择
  - 可能用 lmslim fused_experts_impl_w4a8 快速路径
  - 或 Triton WNA16 kernel
```

---

## 16. 常见问题

### 16.1 路由相关

| 问题 | 原因 | 解决 |
|------|------|------|
| 路由结果不一致 | `torch.topk` 非确定性 | 启用 batch-invariant 模式 |
| DeepSeek 路由精度异常 | bias 应用错误 | 确认 bias 只用于选择，权重取原始分数 |
| 分组路由 fallback | `num_experts % num_expert_group != 0` | 检查模型 config 的分组参数 |

### 16.2 EP 相关

| 问题 | 原因 | 解决 |
|------|------|------|
| EP 未启用 | `enable_expert_parallel=False` | 加 `--enable-expert-parallel` |
| all2all 性能差 | 后端不匹配 | 尝试 `--all2all-backend deepep_low_latency` |
| DeepEP LL 报错 | hidden_size 不支持 | 检查是否在 `[2048,2560,3072,4096,5120,6144,7168,8192]` |
| Expert 负载不均 | 路由倾斜 | 启用 `--enable-eplb` |
| all2all 重复计算 | TP+DP+EP 序列并行未启用 | 检查 `use_sequence_parallel_moe` 条件 |

### 16.3 量化相关

| 问题 | 原因 | 解决 |
|------|------|------|
| FP8 精度损失 | scale 配置不当 | 检查 `per_act_token_quant` / `block_shape` |
| Marlin 不支持 | 设备 SM < 75 或形状不符 | fallback 到 Triton |
| NVFP4 需要 Blackwell | 设备不支持 | 用 mxfp4 或 fp8 替代 |

### 16.4 性能调优

| 优化 | 方法 |
|------|------|
| 减少通信 | DeepEP LL 的 FP8 dispatch |
| 重叠计算 | shared_experts_stream |
| 分块处理 | `VLLM_MOE_DP_CHUNK_SIZE` |
| 融合 kernel | `VLLM_ENABLE_MOE_FUSED_GATE` / `VLLM_USE_FUSED_MOE_GROUPED_TOPK` |
| 负载均衡 | `--enable-eplb` + 调整 `window_size` |

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-08-01  
> **核心文件**：`vllm/model_executor/layers/fused_moe/layer.py`, `config.py`, `fused_moe.py`, `modular_kernel.py`, `router/`, `all2all_utils.py`, `*_prepare_finalize.py`
