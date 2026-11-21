# vLLM GPUModelRunner 模块详解

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖 GPUModelRunner 的设计原理、核心流程、InputBatch 机制与各子系统。

---

## 目录

1. [概述](#1-概述)
2. [模块架构](#2-模块架构)
3. [初始化流程](#3-初始化流程)
4. [RequestState（请求状态管理）](#4-requeststate请求状态管理)
5. [InputBatch（输入批处理）](#5-inputbatch输入批处理)
6. [execute_model（模型执行核心流程）](#6-execute_model模型执行核心流程)
7. [prepare_inputs（输入准备详解）](#7-prepare_inputs输入准备详解)
8. [采样系统](#8-采样系统)
9. [CUDA Graph 管理](#9-cuda-graph-管理)
10. [投机解码集成](#10-投机解码集成)
11. [多模态集成](#11-多模态集成)
12. [结构化输出](#12-结构化输出)
13. [异步调度与输出](#13-异步调度与输出)
14. [KV Cache 与 Attention 后端集成](#14-kv-cache-与-attention-后端集成)
15. [LoRA 集成](#15-lora-集成)
16. [配置参数参考](#16-配置参数参考)
17. [调优建议](#17-调优建议)

---

## 1. 概述

GPUModelRunner 是 vLLM V1 架构中**将调度决策转化为 GPU 执行**的核心组件。它运行在 GPU Worker 进程中，负责：

1. **请求状态管理**：维护每个请求的 token_ids、computed_tokens、采样状态
2. **输入准备**：将 SchedulerOutput 转换为模型可消费的 InputBatch
3. **模型前向传播**：通过 CUDA Graph 或 Eager 模式执行模型
4. **采样**：logits → penalties → temperature → top-k/top-p → Gumbel 采样
5. **后处理**：更新请求状态、管理投机解码 draft tokens

### 关键文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/worker/gpu/model_runner.py` | 980 | GPUModelRunner 主类，核心流程编排 |
| `vllm/v1/worker/gpu/input_batch.py` | 514 | InputBatch 数据结构 + Triton kernel 实现 |
| `vllm/v1/worker/gpu/states.py` | 104 | RequestState 请求状态管理 |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | 285 | CUDA Graph 捕获与管理 |
| `vllm/v1/worker/gpu/attn_utils.py` | 196 | Attention 后端初始化与 Metadata 构建 |
| `vllm/v1/worker/gpu/block_table.py` | 227 | GPU 端 BlockTables（Triton kernel） |
| `vllm/v1/worker/gpu/sample/sampler.py` | 137 | 采样器主逻辑 |
| `vllm/v1/worker/gpu/sample/penalties.py` | 241 | 惩罚项处理 |
| `vllm/v1/worker/gpu/sample/logit_bias.py` | 287 | Logit Bias 处理 |
| `vllm/v1/worker/gpu/sample/gumbel.py` | 150 | Gumbel 采样 |
| `vllm/v1/worker/gpu/sample/logprob.py` | 138 | Logprobs 计算 |
| `vllm/v1/worker/gpu/sample/min_p.py` | 58 | Min-P 采样 |
| `vllm/v1/worker/gpu/async_utils.py` | 98 | 异步 GPU→CPU 拷贝 |
| `vllm/v1/worker/gpu/structured_outputs.py` | 100 | 结构化输出 grammar bitmask |
| `vllm/v1/worker/gpu/spec_decode/eagle.py` | — | EAGLE 投机解码 |
| `vllm/v1/worker/gpu/spec_decode/rejection_sample.py` | — | 拒绝采样验证 |

---

## 2. 模块架构

### 2.1 总体数据流

```
SchedulerOutput (CPU)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    GPUModelRunner                         │
│                                                          │
│  execute_model(scheduler_output)                         │
│    │                                                     │
│    ├── 1. finish_requests()    清理完成的请求              │
│    ├── 2. free_states()        释放编码器缓存              │
│    ├── 3. add_requests()       注册新请求                  │
│    ├── 4. update_requests()    更新已有请求的 Block        │
│    ├── 5. block_tables.apply_staged_writes()              │
│    │                                                     │
│    ├── 6. prepare_inputs() ──→ InputBatch                │
│    │      ├── token_ids, positions, seq_lens              │
│    │      ├── block_tables (gather)                       │
│    │      ├── slot_mappings (compute)                     │
│    │      └── attn_metadata (build)                       │
│    │                                                     │
│    ├── 7. 多模态编码 (如有)                                │
│    │                                                     │
│    ├── 8. Model Forward                                  │
│    │      ├── CUDA Graph replay (use_cudagraph=True)      │
│    │      └── Eager forward (use_cudagraph=False)         │
│    │                                                     │
│    └── 9. 返回 None（异步模式下延迟采样）                   │
│                                                          │
│  sample_tokens(grammar_output)                            │
│    │                                                     │
│    ├── 10. sample()                                      │
│    │      ├── compute_logits                              │
│    │      ├── grammar bitmask (如有)                       │
│    │      ├── logit_bias → penalties → temperature        │
│    │      ├── min_p → top_k/top_p → gumbel               │
│    │      └── rejection_sample (投机解码)                  │
│    │                                                     │
│    ├── 11. prompt_logprobs 计算                           │
│    ├── 12. postprocess() 更新状态                         │
│    └── 13. propose_draft() 投机解码 (如有)                 │
│                                                          │
│  返回: AsyncOutput / ModelRunnerOutput                    │
└──────────────────────────────────────────────────────────┘
```

### 2.2 子模块关系

```
GPUModelRunner
├── RequestState          — 请求 token/状态 管理
├── InputBuffers          — GPU 端输入缓冲区
├── BlockTables           — GPU 端 Block Table
├── CudaGraphManager      — CUDA Graph 捕获/重放
├── Sampler               — 采样器
│   ├── SamplingStates    — 采样参数状态
│   ├── PenaltiesState    — 惩罚项状态
│   └── LogitBiasState    — Logit Bias 状态
├── PromptLogprobsWorker  — Prompt logprobs 计算
├── StructuredOutputsWorker — 结构化输出 grammar
├── EncoderRunner         — 多模态编码器
├── MRopeState            — M-RoPE 位置编码
├── LoraState             — LoRA 状态管理
├── KVConnector           — KV 传输连接器
└── Speculator            — 投机解码 (EAGLE/MTP)
```

---

## 3. 初始化流程

### 3.1 GPUModelRunner.__init__()

```python
class GPUModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
```

**初始化步骤**：

1. **配置解包**：从 VllmConfig 中提取所有子配置（model、cache、parallel、scheduler、speculative 等）
2. **KV Cache 数据类型**：如果 `cache_dtype != "auto"`，设置量化 KV Cache dtype（如 fp8）
3. **维度计算**：vocab_size、max_model_len、max_num_tokens（从 scheduler_config）、max_num_reqs
4. **多模态初始化**（可选）：
   - `EncoderRunner`：管理多模态编码器（图像/音频/视频）
   - `MRopeState`：管理 M-RoPE 位置编码（Qwen2-VL 等模型）
5. **异步调度**（可选）：
   - 创建独立的 `output_copy_stream` 和 `input_prep_event`、`structured_outputs_event`
6. **投机解码**（可选）：
   - `init_speculator()` 初始化 EAGLE/MTP speculator
   - 设置 `num_speculative_steps`
7. **核心组件创建**：
   - `RequestState(max_num_reqs, max_model_len, ...)`：请求状态管理
   - `InputBuffers(max_num_reqs, max_num_tokens, device)`：GPU 端缓冲区
   - `Sampler(max_num_reqs, vocab_size, device)`：采样器
   - `PromptLogprobsWorker`：Prompt logprobs 计算
   - `CudaGraphManager(vllm_config, uses_mrope, device)`：CUDA Graph 管理
   - `StructuredOutputsWorker`：结构化输出 grammar bitmask
   - `LoraState`：LoRA adapter 状态
8. **UVA Buffer Pools**：`tmp_idx_mapping`、`tmp_cu_num_logits`、`tmp_query_start_loc`

### 3.2 load_model()

```
load_model()
  ├── get_model_loader() → 加载模型权重
  ├── load_lora_model() (如有 LoRA)
  ├── speculator.load_model() (如有投机解码)
  └── prepare_communication_buffer_for_model()
```

使用 `DeviceMemoryProfiler` 记录模型加载时间和显存消耗。

### 3.3 initialize_kv_cache()

```
initialize_kv_cache(kv_cache_config)
  ├── 创建 BlockTables (GPU 端块表，Triton 实现)
  ├── init_attn_backend() → 初始化注意力后端
  │     ├── 为每个 KV Cache Group 选择 AttentionBackend
  │     └── 创建 AttentionMetadataBuilder
  ├── init_kv_cache() → 分配并 reshape KV Cache 张量
  │     ├── _allocate_kv_cache() → 分配 int8 原始张量
  │     ├── _reshape_kv_cache() → reshape + permute 到后端期望的 stride order
  │     └── bind_kv_cache() → 绑定到 forward_context
  └── get_kv_connector() → 初始化 KV 传输连接器 (P/D 分离场景)
```

### 3.4 profile_run()（显存分析）

```python
def profile_run(self):
    # 1. Dummy forward (跳过 attention 节省显存)
    hidden_states, sample_hidden_states = self._dummy_run(max_num_tokens, skip_attn=True)
    # 2. Dummy 采样
    self._dummy_sampler_run(sample_hidden_states)
    # 3. 投机解码也做 dummy run
    if self.do_spec_decode:
        self.speculator.run_model(...)
    torch.cuda.synchronize()
```

### 3.5 capture_model()（CUDA Graph 捕获）

```
capture_model()
  ├── gc.collect() + torch.cuda.empty_cache()
  ├── cudagraph_manager.capture()
  │     └── 对每个 capture_size (从大到小) 捕获 CUDA Graph
  └── speculator.capture_model() (如有)
```

通常在 5~20 秒内完成。

---

## 4. RequestState（请求状态管理）

位于 `vllm/v1/worker/gpu/states.py`。

### 4.1 核心数据结构

```python
class RequestState:
    # 索引管理
    req_id_to_index: dict[str, int]          # req_id → 内部索引 (0 ~ max_num_reqs-1)
    index_to_req_id: dict[int, str]          # 内部索引 → req_id
    free_indices: list[int]                  # 空闲索引池 (类似 OS PID 管理)

    # 请求元数据 (numpy/CPU)
    prompt_len: np.ndarray                   # [max_num_reqs] 原始 prompt 长度
    prefill_token_ids: StagedWriteTensor     # [max_num_reqs, max_model_len] UVA 存储
    prefill_len: UvaBackedTensor             # [max_num_reqs] prefill 阶段的 token 总数
    num_computed_prefill_tokens: np.ndarray  # [max_num_reqs] 已计算的 prefill token 数

    # 运行时状态 (GPU)
    num_computed_tokens: StagedWriteTensor   # [max_num_reqs] 每个请求已计算的总 token 数
    last_sampled_tokens: torch.Tensor        # [max_num_reqs, 1] 上一次采样的 token ID
    draft_tokens: torch.Tensor               # [max_num_reqs, num_spec_steps] draft tokens
    next_prefill_tokens: torch.Tensor        # [max_num_reqs] 下一个要 prefill 的 token
```

### 4.2 索引管理

使用 **空闲索引池** 机制：

```python
# 分配索引
req_idx = self.free_indices.pop()   # 从空闲池取出
self.req_id_to_index[req_id] = req_idx

# 释放索引
self.free_indices.append(req_idx)   # 归还到空闲池
self.req_id_to_index.pop(req_id)
```

### 4.3 StagedWriteTensor（分阶段写入）

`prefill_token_ids` 和 `num_computed_tokens` 使用 `StagedWriteTensor`：

- `stage_write(req_idx, offset, data)`：在 CPU 端累积多个写入请求
- `apply_write()`：一次性批量提交所有写入到 GPU/UVA

**设计动机**：避免每次 `add_request` 触发独立的 GPU 拷贝，大幅减少 kernel launch 开销。

### 4.4 UvaBackedTensor（UVA 支持）

对于极大张量（如 `prefill_token_ids` 可达 `128 × 32768 × 4 bytes ≈ 16 MB` 或更大），使用 **UVA（Unified Virtual Addressing）** 代替纯 GPU 内存：

- 数据存储在 CPU pinned memory
- GPU kernel 通过 PCIe/NVLink 直接访问
- 避免显式的 CPU→GPU 拷贝，节省 GPU 显存

### 4.5 生命周期

```
add_request(req_id, prompt_len, prefill_token_ids, num_computed_tokens)
    │
    ├── 分配索引、记录 prompt 元数据
    ├── stage_write prefill_token_ids (下轮 prefill 需要的 token)
    ├── stage_write num_computed_tokens (前缀缓存命中的 token 数)
    │
    ▼
apply_staged_writes()  ← 批量提交到 GPU/UVA
    │
    ▼
[模型执行过程中持续更新 num_computed_tokens、last_sampled_tokens]
    │
    ▼
remove_request(req_id)
    └── 归还索引到 free_indices
```

---

## 5. InputBatch（输入批处理）

位于 `vllm/v1/worker/gpu/input_batch.py`。

### 5.1 InputBuffers（GPU 预分配缓冲区）

```python
class InputBuffers:
    input_ids: torch.Tensor       # [max_num_tokens] int32
    positions: torch.Tensor        # [max_num_tokens] int64
    query_start_loc: torch.Tensor  # [max_num_reqs + 1] int32
    seq_lens: torch.Tensor         # [max_num_reqs] int32
```

每个 step 复用，避免反复分配/释放 GPU 内存。

### 5.2 InputBatch 字段详解

```python
@dataclass
class InputBatch:
    # === 请求标识 ===
    req_ids: list[str]                         # batch_idx → req_id
    num_reqs: int                              # 请求数量

    # === 索引映射 ===
    idx_mapping: torch.Tensor                  # [num_reqs] batch_idx → req_state_idx (GPU)
    idx_mapping_np: np.ndarray                 # [num_reqs] batch_idx → req_state_idx (CPU)
    expanded_idx_mapping: torch.Tensor         # [total_num_logits] 投机解码时展开

    # === Token 分配 ===
    num_scheduled_tokens: np.ndarray           # [num_reqs] 每个请求的 token 数
    num_tokens: int                            # 总 token 数
    num_tokens_after_padding: int              # DP/CUDA Graph padding 后
    num_draft_tokens: int                      # draft token 总数

    # === 序列信息 (Ragged Batch) ===
    query_start_loc: torch.Tensor              # [num_reqs + 1] 各请求 token 起始偏移
    query_start_loc_np: np.ndarray
    seq_lens: torch.Tensor                     # [num_reqs] 各请求序列总长度

    # === 模型输入 ===
    input_ids: torch.Tensor                    # [num_tokens_after_padding] int32
    positions: torch.Tensor                    # [num_tokens_after_padding] int64
    mrope_positions: torch.Tensor | None       # [3, num_tokens_after_padding] M-RoPE
    inputs_embeds: torch.Tensor | None         # [num_tokens_after_padding, hidden_size]

    # === Attention 元数据 ===
    attn_metadata: dict[str, Any]              # layer_name → AttentionMetadata
    slot_mappings: dict[str, torch.Tensor]     # layer_name → slot_mapping tensor

    # === 采样索引 ===
    logits_indices: torch.Tensor               # [total_num_logits] 需要提取 logits 的位置
    cu_num_logits: torch.Tensor                # [num_reqs + 1] 每个请求的 logits 数量前缀和
    cu_num_logits_np: np.ndarray
```

### 5.3 Ragged Batch 布局

vLLM 使用 **ragged batch**（非均匀批次），不同请求的 token 拼接在一起：

```
请求 A (3 tokens, decode):  [A_last_sampled, A_draft0, A_draft1]
请求 B (1 token,  decode):  [B_last_sampled]
请求 C (2 tokens, prefill): [C_prefill_10, C_prefill_11]

input_ids:       [A_prev, A_d0, A_d1, B_prev, C_10, C_11]
positions:       [5,      6,    7,    3,      10,   11 ]
seq_lens:        [8,                    4,      12    ]
query_start_loc: [0,       3,           4,      6     ]
```

### 5.4 Triton Kernel 实现

InputBatch 的关键操作全部使用 Triton kernel：

| Kernel | 文件 | 功能 |
|--------|------|------|
| `_prepare_prefill_inputs_kernel` | input_batch.py | 从 prefill_token_ids 拷贝 token 到 input_ids，预取下一个 token |
| `_prepare_pos_seq_lens_kernel` | input_batch.py | 计算每个请求的 positions 和 seq_lens |
| `_combine_sampled_and_draft_tokens_kernel` | input_batch.py | 合并 last_sampled_token + draft_tokens 到 input_ids |
| `_get_num_sampled_and_rejected_kernel` | input_batch.py | 计算采样数量和拒绝数量（chunked prefill 时两者均为 0） |
| `_post_update_kernel` | input_batch.py | 更新 num_computed_tokens、last_sampled_tokens、output_bin_counts |
| `_expand_idx_mapping_kernel` | input_batch.py | 展开 idx_mapping（投机解码） |
| `_gather_block_tables_kernel` | block_table.py | 从所有 KV Cache Group 收集 Block Table |
| `_compute_slot_mappings_kernel` | block_table.py | 计算每个 token 的 KV cache slot |

---

## 6. execute_model（模型执行核心流程）

位于 `vllm/v1/worker/gpu/model_runner.py:802`。

### 6.1 完整流程

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    intermediate_tensors: Any | None = None,  # PP 场景，当前未使用
    dummy_run: bool = False,                   # profile / warmup
    skip_attn_for_dummy_run: bool = False,
) -> ModelRunnerOutput | None:
```

#### 第一阶段：状态更新（非 dummy_run）

```
1. finish_requests(scheduler_output)
   ├── 获取 finished_req_ids + preempted_req_ids (抢占也要清理)
   ├── req_states.remove_request(req_id)       归还索引
   ├── encoder_runner.remove_request(req_id)   清理编码器状态
   ├── prompt_logprobs_worker.remove_request()
   └── lora_state.remove_request()

2. free_states(scheduler_output)
   └── encoder_runner.free_encoder_cache(mm_hash)  释放编码器缓存

3. add_requests(scheduler_output)
   ├── 遍历 scheduled_new_reqs
   │   ├── req_states.add_request(req_id, prompt_len, prefill_token_ids, num_computed)
   │   ├── encoder_runner.add_request()             (如有)
   │   ├── mrope_states.init_prefill_mrope_positions() (如有)
   │   ├── block_tables.append_block_ids(req_index, block_ids, overwrite=True)
   │   ├── sampler.add_request(req_index, prompt_len, sampling_params)
   │   ├── prompt_logprobs_worker.add_request()
   │   └── lora_state.add_request()
   ├── req_states.apply_staged_writes()        批量提交 RequestState
   └── sampler.apply_staged_writes()           批量提交采样状态

4. update_requests(scheduler_output)
   └── 遍历 scheduled_cached_reqs
       └── block_tables.append_block_ids(req_index, new_block_ids, overwrite=False)  追加新 Block

5. block_tables.apply_staged_writes()          批量提交 Block Table 更新
```

#### 第二阶段：CUDA Graph 决策

```
6. cudagraph_manager.get_cudagraph_size(total_tokens, ...)
   → 返回匹配的 cudagraph_size 或 None (eager 模式)

7. get_cudagraph_and_dp_padding(total_tokens, cudagraph_size, dp_size, dp_rank)
   → use_cudagraph: bool
   → num_tokens_after_padding: int    (填充到 cudagraph_size，或 DP padding)
   → num_tokens_across_dp: int        (所有 DP rank 的全局 token 数)
```

#### 第三阶段：输入准备

```
8. prepare_inputs(scheduler_output, num_tokens_after_padding) → InputBatch

9. LoRA 激活 (如有)
   └── lora_state.make_lora_inputs() → _set_active_loras()

10. 多模态编码 (如有)
    ├── encoder_runner.prepare_mm_inputs()
    ├── encoder_runner.execute_mm_encoder(model, mm_hashes, mm_kwargs)
    ├── encoder_runner.gather_mm_embeddings()
    └── encoder_runner.get_inputs_embeds() → input_batch.inputs_embeds
```

#### 第四阶段：模型前向

```
11. if use_cudagraph:
        # CUDA Graph 重放 — input_ids/positions/block_tables 已在 buffer 中
        kv_connector.pre_forward(scheduler_output)
        hidden_states = cudagraph_manager.run(num_tokens_after_padding)
    else:
        # Eager Forward
        positions = input_batch.positions
        if uses_mrope:
            positions = input_batch.mrope_positions  # 3D positions
        with set_forward_context(
            attn_metadata, vllm_config,
            num_tokens=num_tokens_after_padding,
            cudagraph_runtime_mode=NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=input_batch.slot_mappings,
        ):
            kv_connector.pre_forward(scheduler_output)
            hidden_states = self.model(
                input_ids=input_batch.input_ids,
                positions=positions,
                inputs_embeds=input_batch.inputs_embeds,
            )

12. kv_connector.post_forward(scheduler_output)

13. self.execute_model_state = (hidden_states, input_batch, kv_connector_output)
    return None   ← 异步模式下延迟采样
```

### 6.2 forward_context

`set_forward_context` 设置全局上下文，Attention 层从中获取 `attn_metadata`、`slot_mapping` 等信息。

```python
with set_forward_context(
    attn_metadata,            # {layer_name: AttentionMetadata}
    vllm_config,              # 全局配置
    num_tokens=num_tokens,
    cudagraph_runtime_mode=...,  # NONE / PIECEWISE / FULL
    num_tokens_across_dp=...,    # DP 全局 token 数
    slot_mapping=...,          # {layer_name: slot_mapping tensor}
):
    output = model(input_ids, positions, inputs_embeds)
```

### 6.3 设计要点

**为什么 execute_model 返回 None？**

支持 **异步调度**（async_scheduling）：模型前向和采样可以解耦。

- **同步模式**：execute_model → 立即 sample_tokens
- **异步模式**：execute_model 返回 None，调度器在后续 step 中调用 sample_tokens，允许调度超前于采样

状态通过 `self.execute_model_state` 传递。

---

## 7. prepare_inputs（输入准备详解）

位于 `vllm/v1/worker/gpu/model_runner.py:498`。

### 7.1 请求排序

```python
# Decode first, then prefill
req_ids = sorted(
    num_scheduled_tokens.keys(),
    key=lambda k: num_scheduled_tokens[k],
)
```

按 token 数升序排列，**decode 请求（1 token）在前，prefill 请求（多 token）在后**。

有利于 CUDA Graph 优化：decode-only batch 匹配专门的 `FULL_DECODE_ONLY` graph。

### 7.2 idx_mapping 构建

```
req_ids (排序后):         ["req_C", "req_A", "req_B"]
req_state:               {"req_A": 0, "req_B": 1, "req_C": 2}
idx_mapping_np = [2, 0, 1]   # batch_idx → req_state_idx
```

### 7.3 投机解码 Token 展开

**无投机解码**（常见情况）：
```
所有请求 1 logit:
  cu_num_logits = [0, 1, 2, 3]
  expanded_idx_mapping = idx_mapping
```

**有投机解码**：
```
请求 A: 3 draft tokens → 4 logits (draft_0, draft_1, draft_2, final)
请求 B: 0 draft tokens → 1 logits (final only)
请求 C: 2 draft tokens → 3 logits

cu_num_logits = [0, 4, 5, 8]
expanded_idx_mapping = [0, 0, 0, 0, 1, 2, 2, 2]
```

### 7.4 Block Table 收集

```python
block_tables = self.block_tables.gather_block_tables(idx_mapping)
```

Triton kernel 将指定请求的 Block Table 从各 KV Cache Group 收集到一个连续的批次。

### 7.5 Prefill Token 准备

```python
prepare_prefill_inputs(
    input_ids,                        # 输出: 写入 prefill token
    next_prefill_tokens,              # 输出: 预取下一个 prefill token
    idx_mapping,                      # batch_idx → req_state_idx
    query_start_loc,                  # 各请求在 input_ids 中的起始位置
    prefill_token_ids,                # 源: 所有 prefill token
    prefill_len,                      # 每个请求的 prefill 总长度
    num_computed_tokens,              # 已计算的 token 数
)
```

Triton kernel 逻辑：
1. 如果 `num_computed >= prefill_len`：不是 prefill 请求，跳过
2. 否则：拷贝 `prefill_token_ids[num_computed : num_computed + query_len]` 到 `input_ids`
3. 如果还有剩余 prefill token，预取下一个到 `next_prefill_tokens`

### 7.6 Positions 和 SeqLens 计算

```python
prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
```

- **positions[i]** = `num_computed_tokens + offset_in_query`
- **seq_lens[i]** = `num_computed_tokens + query_len`

额外线程块用于将未使用的 `seq_lens` 填充为 0（Full CUDA Graph 需要）。

### 7.7 Decode Token 合并

```python
logits_indices = combine_sampled_and_draft_tokens(
    input_ids, idx_mapping,
    last_sampled_tokens,    # 上次 step 的采样结果
    query_start_loc, seq_lens,
    prefill_len, draft_tokens,
    cu_num_logits, total_num_logits,
)
```

对于 decode 请求（`seq_len > prefill_len`）：
1. 将 `last_sampled_tokens[req]` 写入 `input_ids` 的最后一个 logit 位置
2. 将 `draft_tokens[req]` 写入对应位置
3. 返回 `logits_indices`：每个 logit 在 `input_ids` 中的索引

### 7.8 Slot Mapping 计算

```python
slot_mappings = self.block_tables.compute_slot_mappings(
    idx_mapping, query_start_loc, positions[:num_tokens]
)
# 分散到各层
slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, kv_cache_config)
```

每个 KV Cache Group 一个 slot_mapping，同一 Group 内的层共享。

### 7.9 Attention Metadata 构建

```python
attn_metadata = build_attn_metadata(
    attn_metadata_builders, num_reqs, num_tokens,
    query_start_loc_gpu, query_start_loc_cpu,
    seq_lens, max_seq_len,
    block_tables, slot_mappings, kv_cache_config,
)
```

为每个 KV Cache Group 创建 `CommonAttentionMetadata`，然后由各后端的 `AttentionMetadataBuilder.build()` 生成后端特定的 Metadata（flash_attn / flashinfer / triton / ...）。

---

## 8. 采样系统

位于 `vllm/v1/worker/gpu/sample/`。

### 8.1 完整采样流水线

```
hidden_states
    │
    ▼
compute_logits(hidden_states)              ← model.lm_head
    │
    ▼
[Grammar Bitmask]                           ← 结构化输出约束 (Triton kernel)
    │  logits[disallowed] = -inf
    ▼
logits.clone() → FP32                       ← 精度转换
    │
    ▼
logit_bias_state.apply_logit_bias()        ← allowed_token_ids, min_tokens
    │
    ▼
penalties_state.apply_penalties()          ← presence/freq/rep penalty
    │
    ▼
apply_temperature()                        ← logits /= temperature
    │
    ▼
apply_min_p()                              ← filter below max_prob * min_p (可选)
    │
    ▼
apply_top_k_top_p()                        ← top-k + top-p filter (可选)
    │
    ▼
gumbel_sample()                            ← Gumbel-Max 采样
    │
    ▼
[rejection_sample()]                       ← 投机解码验证 (可选)
    │
    ▼
compute_topk_logprobs()                    ← logprobs 计算 (可选)
    │
    ▼
SamplerOutput(sampled_token_ids, logprobs_tensors, num_nans)
```

### 8.2 Sampler 类

```python
class Sampler:
    def __init__(self, max_num_reqs, vocab_size, device, logprobs_mode):
        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(max_num_reqs, vocab_size, device)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)

    def add_request(req_idx, prompt_len, sampling_params):
        # 注册采样参数到三个子状态

    def __call__(self, logits, idx_mapping, idx_mapping_np, pos) -> SamplerOutput:
        # 1. 可选 NaN 检测
        # 2. self.sample() → (sampled_tokens, processed_logits)
        # 3. 可选 logprobs 计算
        #    - raw_logprobs: 用原始 logits
        #    - processed_logprobs: 用采样后的 logits (复用)
        # 4. 返回 SamplerOutput
```

### 8.3 SamplingStates

管理每个请求的独立采样参数，存储在 GPU tensor 中通过 `idx_mapping` 索引：

| 参数 | 默认值 | GPU Tensor |
|------|--------|-----------|
| `temperature` | 1.0 | `temperature.gpu` [max_num_reqs] float32 |
| `top_k` | 0 | `top_k.gpu` [max_num_reqs] int32 |
| `top_p` | 1.0 | `top_p.gpu` [max_num_reqs] float32 |
| `min_p` | 0.0 | `min_p.gpu` [max_num_reqs] float32 |
| `seed` | — | `seeds.gpu` [max_num_reqs] int64 |
| `max_num_logprobs` | 0 | int32 tensor |

条件执行优化：
```python
do_top_k = self.sampling_states.do_top_k(idx_mapping_np)  # 只有非零才执行
if do_top_k:
    top_k = self.sampling_states.top_k.gpu[idx_mapping]
```

### 8.4 PenaltiesState

```python
class PenaltiesState:
    output_bin_counts: torch.Tensor  # [max_num_reqs, vocab_size] token 出现次数
    presence_penalty: float           # 出现惩罚
    frequency_penalty: float          # 频率惩罚
    repetition_penalty: float         # 重复惩罚
```

**公式**：
```
logits[token] -= presence_penalty                    (如果 token 出现过)
logits[token] -= frequency_penalty * count(token)    (频率惩罚)
logits[token] /= repetition_penalty                  (重复惩罚，对非 prompt token)
```

`output_bin_counts` 由 `post_update_kernel` 每个 step 更新。

### 8.5 LogitBiasState

- `allowed_token_ids`：将非允许 token 设为 -inf
- `min_tokens`：在生成前 N 个 token 时禁止 EOS
- `logit_bias`：自定义偏置

### 8.6 Gumbel 采样

```python
def gumbel_sample(logits, idx_mapping, temperature, seeds, pos, apply_temperature):
    # 1. 生成 Gumbel 噪声: G = -log(-log(U)), U ~ Uniform(0,1)
    # 2. token = argmax(logits/temperature + G)
```

使用请求级别的种子 + position 确保可复现。

### 8.7 Prompt Logprobs

`PromptLogprobsWorker` 计算 prompt 部分的 logprobs：

- 在所有 prefill token 计算完成后触发
- 分块计算避免 OOM（每块最多 max_num_batched_tokens 个 logits）
- 返回 `{req_id: LogprobsTensors | None}`

---

## 9. CUDA Graph 管理

位于 `vllm/v1/worker/gpu/cudagraph_utils.py`。

### 9.1 CUDAGraphMode

| 模式 | 适用场景 | 说明 |
|------|---------|------|
| `NONE` | 开发调试 | 不使用 CUDA Graph |
| `PIECEWISE` | 通用 | 分段 CUDA Graph（部分 op 使用） |
| `FULL` | — | 全图捕获（prefill + decode 混合） |
| `FULL_DECODE_ONLY` | 推荐 | 仅 decode 全图捕获 |
| `FULL_AND_PIECEWISE` | V1 默认 | FULL + PIECEWISE 组合 |

### 9.2 Capture Sizes

```python
# 默认 capture sizes
cudagraph_capture_sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, ...]
```

**大小映射表**（向上取整）：
```python
cudagraph_sizes = {
    1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8,
    9: 16, 10: 16, ..., 16: 16, 17: 24, ...
}
```

### 9.3 捕获流程

```python
def capture_graph(self, num_tokens, model, input_buffers, ...):
    # 1. 准备 dummy 输入 (均匀分布 token)
    attn_metadata, slot_mappings = prepare_inputs_to_capture(...)

    # 2. Warmup — 触发 CUDA JIT 编译
    with set_forward_context(attn_metadata, ..., cudagraph_runtime_mode=NONE):
        hidden_states = model(input_ids, positions, inputs_embeds)

    # 3. CUDA Graph 录制
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool):
        hidden_states = model(input_ids, positions, inputs_embeds)
        self.hidden_states[:num_tokens] = hidden_states

    self.graphs[num_tokens] = graph
```

大图先捕获（`sorted(sizes, reverse=True)`），避免显存碎片。

### 9.4 运行时决策

```python
def get_cudagraph_size(num_tokens, num_tokens_per_request, cudagraph_sizes, mode):
    size = cudagraph_sizes.get(num_tokens_after_padding)
    if size is None:
        return None  # 没有匹配的 graph → eager

    is_mixed = any(x > 1 for x in num_tokens_per_request)
    if is_mixed and mode.mixed_mode() != FULL:
        return None  # 混合 prefill+decode → eager (除非 FULL 模式)

    return size
```

### 9.5 Graph Replay

```python
def run(self, num_tokens: int) -> torch.Tensor:
    self.graphs[num_tokens].replay()
    return self.hidden_states[:num_tokens]
```

输入数据在 `prepare_inputs` 阶段已写入 CUDA Graph 的输入缓冲区，replay 时直接使用。

### 9.6 DP Padding

Data Parallel 场景下，需要 padding 使所有 DP rank 的 token 数一致：
```python
def get_cudagraph_and_dp_padding(total_tokens, cudagraph_size, dp_size, dp_rank):
    num_tokens_after_padding = max(total_tokens, ...)  # DP padding + CUDA Graph padding
    num_tokens_across_dp = num_tokens_after_padding * dp_size
```

---

## 10. 投机解码集成

位于 `vllm/v1/worker/gpu/spec_decode/`。

### 10.1 完整流程

```
execute_model()
    ├── model forward → hidden_states
    │
    ▼
sample_tokens(grammar_output)
    │
    ├── sample() → sampled_token_ids + processed_logits
    │     └── [内部] gumbel_sample() → tokens for each logit position
    │
    ├── rejection_sample()
    │     ├── 比较 sampled_tokens 和 input_ids (包含 draft tokens)
    │     ├── 接受所有匹配的 token (包括 draft 验证)
    │     ├── 拒绝第一个不匹配的 draft 及之后的所有 draft
    │     └── 返回 (valid_tokens, num_sampled)
    │
    ├── postprocess()
    │     ├── 更新 last_sampled_tokens (最后一个被接受的 token)
    │     ├── 更新 output_bin_counts (penalty 使用)
    │     └── 更新 num_computed_tokens (减去被拒绝的 draft tokens)
    │
    └── propose_draft()
          ├── speculator.propose(hidden_states, ...) → draft_tokens
          └── 存储到 req_states.draft_tokens
```

### 10.2 拒绝采样算法

```
input_ids (decode 部分):    [last_sampled, draft_0, draft_1, ..., draft_k]
sampled_tokens:             [sample_0,      sample_1, sample_2, ..., sample_k]

比较 (i 从 1 到 k):
  如果 sample_i == draft_{i-1}: 接受 draft_{i-1}
  否则: 拒绝 draft_{i-1} 及之后的所有 draft

num_sampled = 1 + 接受的 draft 数量
num_rejected = (k+1) - num_sampled
```

### 10.3 Speculator 类型

| 方法 | 类 | 特点 |
|------|---|------|
| `eagle` | EagleSpeculator | 基于 hidden states 的小型 draft model |
| `eagle3` | Eagle3Speculator | EAGLE-3 版本 |
| `mtp` | MTPSpeculator | 模型原生 Multi-Token Prediction (Qwen3, DeepSeek) |
| `ngram` | NgramSpeculator | N-gram 匹配，无需额外模型 |

---

## 11. 多模态集成

### 11.1 EncoderRunner

```
get_mm_embeddings(scheduled_encoder_inputs, input_batch)
    │
    ├── 1. prepare_mm_inputs()
    │     ├── 收集需要编码的多模态输入
    │     ├── 按 hash 去重 (相同输入复用编码结果)
    │     └── 返回 (mm_hashes: 需要编码的项, mm_kwargs: 编码参数)
    │
    ├── 2. execute_mm_encoder(model, mm_hashes, mm_kwargs)
    │     └── 执行视觉/音频编码器前向
    │
    └── 3. gather_mm_embeddings(req_ids, num_tokens, ...)
          ├── 按请求收集编码结果
          ├── 返回 (mm_embeds: 嵌入列表, is_mm_embed: 每个 token 是否为多模态)
```

### 11.2 嵌入合并

```python
inputs_embeds = self.encoder_runner.get_inputs_embeds(
    model, input_batch.input_ids, mm_embeds, is_mm_embed
)
# input_ids → token embeddings
# is_mm_embed == True 的位置替换为多模态 embeddings
input_batch.inputs_embeds = inputs_embeds[:num_tokens_after_padding]
```

### 11.3 M-RoPE（Multi-Resolution RoPE）

```python
class MRopeState:
    mrope_positions: torch.Tensor  # [3, max_num_tokens]
```

用于 Qwen2-VL 等多分辨率模型，为不同的 RoPE 维度使用独立的位置编码（t, h, w 三维）。

---

## 12. 结构化输出

位于 `vllm/v1/worker/gpu/structured_outputs.py`。

### 12.1 Grammar Bitmask 应用

```python
class StructuredOutputsWorker:
    def apply_grammar_bitmask(self, logits, input_batch, grammar_req_ids, grammar_bitmask):
        # 1. 为每个需要 grammar 约束的 logit 构建映射
        # 2. Triton kernel 并行应用 bitmask:
        #    - unpack bitmask (每 32 个 vocab token 一个 int32)
        #    - logits[disallowed_vocab_ids] = -inf
```

### 12.2 Bitmask 格式

- `grammar_bitmask`: `[num_masks, ceil(vocab_size / 32)]` int32
- bit=1 表示 token 被允许
- bit=0 表示 token 被禁止（logit 设为 -inf）

---

## 13. 异步调度与输出

### 13.1 异步调度

当 `async_scheduling=True`：

```
Step N:
  execute_model() → 保存 (hidden_states, input_batch, kv_output) 到 self.execute_model_state
  return None

Step N+1:
  sample_tokens()     ← 采样上一个 step 的结果
  execute_model()     ← 执行当前 step 的模型前向
  return AsyncOutput  ← 返回上一个 step 的采样结果
```

**优势**：CPU 调度与 GPU 执行重叠，提升吞吐。
**代价**：增加一帧延迟。

### 13.2 AsyncOutput（异步 D2H 拷贝）

```python
class AsyncOutput:
    def __init__(self, model_runner_output, sampler_output, num_sampled_tokens,
                 copy_stream, copy_event):
        # 在独立的 CUDA stream 上执行 GPU→CPU 拷贝
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(default_stream)    # 等待主 stream 完成
            sampled_token_ids = async_copy_to_np(...)  # 异步 D2H
            logprobs_tensors = logprobs.to_cpu_nonblocking()
            copy_event.record(copy_stream)             # 记录完成

    def get_output(self) -> ModelRunnerOutput:
        copy_event.synchronize()   # 等待拷贝完成
        # 转换 np → Python list
        return model_runner_output
```

**流水线效果**：
```
主 Stream:       [exec N]  [sample N]  [exec N+1]  [sample N+1] ...
拷贝 Stream:               [D2H N]                  [D2H N+1]    ...
```

---

## 14. KV Cache 与 Attention 后端集成

### 14.1 BlockTables（GPU 端）

```python
class BlockTables:
    block_tables: list[StagedWriteTensor]    # 分阶段写入 [max_num_reqs, max_blocks]
    input_block_tables: list[torch.Tensor]   # 前向传播使用的 Block Table
    slot_mappings: torch.Tensor              # [num_groups, max_num_tokens] int64
    num_blocks: UvaBackedTensor              # [num_groups, max_num_reqs] 每请求 Block 数

    def append_block_ids(req_index, new_block_ids, overwrite):  # 分阶段写入 Block ID
    def apply_staged_writes():                                    # 批量提交
    def gather_block_tables(idx_mapping):                         # Triton kernel 收集
    def compute_slot_mappings(idx_mapping, query_start_loc, pos): # Triton kernel 计算
```

### 14.2 Attention 后端初始化

```python
def init_attn_backend(kv_cache_config, vllm_config, device):
    for each KV Cache Group:
        # 1. 从模型层获取 attention backend
        attn_backend = attn_layers[layer_name].get_attn_backend()

        # 2. 创建 AttentionMetadataBuilder
        builder = attn_backend.get_builder_cls()(kv_cache_spec, layer_names, ...)

    return attn_backends_dict, attn_metadata_builders_list
```

### 14.3 KV Cache 张量初始化

```python
def init_kv_cache(runner_kv_caches, forward_context, kv_cache_config, attn_backends, device):
    # 1. 分配原始 int8 张量 (page_size * num_blocks bytes)
    raw_tensors = _allocate_kv_cache(kv_cache_config, device)

    # 2. Reshape 到后端期望的格式
    #    int8 → view(dtype) → view(kv_cache_shape) → permute(stride_order)
    kv_caches = _reshape_kv_cache(kv_cache_config, raw_tensors, attn_backends)

    # 3. 绑定到 forward_context
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)
```

### 14.4 CommonAttentionMetadata

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor       # [num_reqs + 1] ragged batch 起始偏移
    seq_lens: torch.Tensor              # [num_reqs] 各请求序列长度
    max_seq_len: int                    # 最大序列长度
    num_reqs: int                       # 请求数
    num_actual_tokens: int              # 实际 token 数
    max_query_len: int                  # 最大 query 长度
    block_table_tensor: torch.Tensor    # [num_reqs, max_blocks] GPU Block Table
    slot_mapping: torch.Tensor          # [num_tokens] token → KV slot 映射
    causal: bool = True
```

各 Attention 后端的 `build()` 方法在此基础上生成后端特定的 Metadata。

---

## 15. LoRA 集成

```python
class LoraState:
    def add_request(req_id, req_index, lora_request)
    def remove_request(req_id)
    def make_lora_inputs(req_ids, idx_mapping_np, num_scheduled_tokens)
        → (lora_request_ids, prompt_mapping)
```

在 `execute_model` 中：
```python
if self.lora_config:
    lora_inputs = self.lora_state.make_lora_inputs(...)
    self._set_active_loras(*lora_inputs)  # 设置当前 batch 的 LoRA adapter
```

继承自 `LoRAModelRunnerMixin`，通过 `_set_active_loras` 激活每个请求对应的 LoRA adapter。

---

## 16. 配置参数参考

### 16.1 核心维度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_num_batched_tokens` | 2048 | 每步最大 token 数，决定 InputBuffers 大小 |
| `max_num_seqs` | 128 | 最大并发请求数，决定 RequestState / BlockTable 行数 |
| `max_model_len` | — | 最大模型长度，决定 prefill_token_ids 列数 |

### 16.2 CUDA Graph

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cudagraph_mode` | FULL_AND_PIECEWISE (V1) | CUDA Graph 模式 |
| `cudagraph_capture_sizes` | [1,2,4,8,...,64,...] | 捕获的 batch size 列表 |

### 16.3 投机解码

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_speculative_tokens` | None | 投机 token 数 |
| `method` | None | eagle / eagle3 / mtp / ngram |

### 16.4 异步调度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `async_scheduling` | None (自动) | 启用异步调度（模型前向与采样解耦） |

### 16.5 可观测

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kv_cache_metrics` | False | 启用 KV Cache 驱逐指标 |
| `kv_cache_metrics_sample` | 0.01 | 指标采样率 |

---

## 17. 调优建议

### 17.1 CUDA Graph 调优

- **FULL_DECODE_ONLY** 是最推荐的模式：decode 受益最大
- 增大 `cudagraph_capture_sizes` 覆盖更多 batch size 减少 eager fallback
- 代价：启动时间变长 + 显存增加（每个 size 一个 graph）
- 可用 `--cudagraph-mode NONE` 临时关闭以排查问题

### 17.2 输入准备优化

- `StagedWriteTensor` 批量写入避免过多小 kernel launch
- `UvaBackedTensor` 对大张量使用 UVA（如 prefill_token_ids 可达数 GB）
- Triton kernel 直接利用 GPU 并行性处理 ragged batch 操作

### 17.3 采样优化

- 条件性执行 `do_top_k()` / `do_top_p()` / `do_min_p()` 避免无用计算
- `processed_logprobs` 模式复用采样后的 logits 计算 logprobs
- NaN 检测默认关闭（`VLLM_COMPUTE_NANS_IN_LOGITS=0`）

### 17.4 异步调度

- 适合吞吐敏感场景（隐藏 D2H 拷贝延迟）
- 不适合延迟敏感场景（额外一帧延迟）
- DP 场景下自动调整 token 数使所有 rank 一致

### 17.5 显存优化

- 预分配 `InputBuffers` 避免 per-step 分配
- `prefill_token_ids` 使用 UVA 节省 GPU 显存（最多可节省 `max_num_reqs × max_model_len × 4 bytes`）
- 降低 `max_num_reqs` 可线性减少 `RequestState` 和 `Sampler` 的显存占用

### 17.6 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| CUDA Graph 捕获失败 | 显存不足或 batch size 超限 | 减少 `cudagraph_capture_sizes`，切换到 PIECEWISE |
| 采样结果异常 | NaN in logits | 启用 `VLLM_COMPUTE_NANS_IN_LOGITS=1` 诊断 |
| 大量 OOM | prefill_token_ids 或 KV Cache 过大 | 降低 `max_num_reqs` 或 `max_model_len` |
| 投机解码无加速 | draft 命中率太低 | 调整 `num_speculative_tokens`，或换用更适合的方法 |
| Eager fallback 频繁 | CUDA Graph 覆盖不足 | 增加 `cudagraph_capture_sizes` |
| 多模态请求延迟 | 编码器成为瓶颈 | 启用编码器缓存，减少重复编码 |

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-07-14  
> **核心文件**：`vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/input_batch.py`, `vllm/v1/worker/gpu/states.py`, `vllm/v1/worker/gpu/cudagraph_utils.py`, `vllm/v1/worker/gpu/attn_utils.py`, `vllm/v1/worker/gpu/sample/sampler.py`