# vLLM Model Runner V2 模块详解

> 基于 upstream main（2026-09 快照，V1 架构），全面覆盖实验性 Model Runner V2（`vllm/v1/worker/gpu/`）的设计动机、模块化架构、ModelState 抽象、CUDA Graph 管理与 V1 的对比。
>
> **版本说明**：本文档追随 upstream main。本地 v0.26.1rc0 代码（`<local_vllm_checkout>`）缺少 main 新增的 `adaptive_verification.py`、`ubatch_utils.py`、`ec_connector.py`、`dflash2/`、`multi_module_mtp/` 等模块（见 [15. 相对 v0.26.1 的增量](#15-相对-v0261-的增量)）。

---

## 目录

1. [概述](#1-概述)
2. [设计动机：V1 的问题](#2-设计动机v1-的问题)
3. [模块架构](#3-模块架构)
4. [V2 GPUModelRunner 核心](#4-v2-gpumodelrunner-核心)
5. [ModelState 抽象](#5-modelstate-抽象)
6. [PoolingRunner 抽象](#6-poolingrunner-抽象)
7. [CUDA Graph 管理（V2）](#7-cuda-graph-管理v2)
8. [InputBatch 与 RequestState（V2）](#8-inputbatch-与-requeststatev2)
9. [Attention 后端初始化（V2）](#9-attention-后端初始化v2)
10. [V1 vs V2 对比](#10-v1-vs-v2-对比)
11. [启用 V2](#11-启用-v2)
12. [配置参数与环境变量](#12-配置参数与环境变量)
13. [main 新特性（v0.26.1 之后）](#13-main-新特性v0261-之后)
14. [常见问题](#14-常见问题)
15. [相对 v0.26.1 的增量](#15-相对-v0261-的增量)

---

## 1. 概述

Model Runner V2 是 vLLM 的**实验性新模型执行器**，位于 `vllm/v1/worker/gpu/` 目录（README 标注 "[Experimental] Model Runner V2"，由 Woosuk Kwon 主导开发）。它是对 V1 runner（`vllm/v1/worker/gpu_model_runner.py`，v0.26.1 为 7916 行 / main 为 7629 行）的**模块化重构**，核心目标：

> 将模型特定行为从 runner 中剥离，使 runner 保持**最小化、模型无关、稳定**。

### 关键文件

| 文件 | 行数（main） | 职责 |
|------|------|------|
| `vllm/v1/worker/gpu/model_runner.py` | 2270 | V2 GPUModelRunner（核心，模型无关） |
| `vllm/v1/worker/gpu/model_states/interface.py` | 189 | ModelState 抽象基类 |
| `vllm/v1/worker/gpu/model_states/default.py` | 192 | DefaultModelState（标准生成模型） |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | 342 | Mamba 混合模型状态 |
| `vllm/v1/worker/gpu/model_states/encoder_decoder.py` | 192 | 编码器-解码器状态 |
| `vllm/v1/worker/gpu/model_states/encoder_only.py` | 230 | 编码器-only 状态（BERT） |
| `vllm/v1/worker/gpu/model_states/mm_pruning.py` | 145 | 多模态剪枝状态 |
| `vllm/v1/worker/gpu/pool/pooling_runner.py` | 191 | PoolingRunner（嵌入/重排模型） |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | 645 | V2 CUDA Graph 管理 |
| `vllm/v1/worker/gpu/input_batch.py` | 716 | V2 InputBatch + InputBuffers |
| `vllm/v1/worker/gpu/states.py` | 133 | V2 RequestState |
| `vllm/v1/worker/gpu/attn_utils.py` | 616 | V2 Attention 后端初始化 |
| `vllm/v1/worker/gpu/block_table.py` | 317 | V2 BlockTables |
| `vllm/v1/worker/gpu/pcp_manager.py` | 680 | Prefill CP 管理 |
| `vllm/v1/worker/gpu/pp_utils.py` | 199 | PP 处理 |
| `vllm/v1/worker/gpu/warmup.py` | 376 | 预热 |
| `vllm/v1/worker/gpu/eplb_utils.py` | 168 | EPLB 专家负载均衡控制器 |
| `vllm/v1/worker/gpu/cp_utils.py` | 61 | DCP（decode context parallel）seq_lens 准备 |
| `vllm/v1/worker/gpu/buffer_utils.py` | 316 | StagedWriteTensor / UVA buffer 工具 |
| `vllm/v1/worker/gpu/ec_connector.py` | 111 | EC（encoder cache）transfer 连接器 |
| `vllm/v1/worker/gpu/ubatch_utils.py` | 500 | 微批（DBO）执行辅助 |
| `vllm/v1/worker/gpu/shutdown.py` | 20 | 优雅关闭 |
| `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` | 507 | 自适应验证（DSpark draft 预算动态分配） |
| `vllm/v1/worker/gpu/sample/` | — | 采样器（sampler/gumbel/penalties/logit_bias/batch_shard） |
| `vllm/v1/worker/gpu/spec_decode/` | — | 投机解码 speculator（eagle/mtp/dflash/dflash2/dspark/multi_module_mtp…） |

---

## 2. 设计动机：V1 的问题

### 2.1 V1 runner 的痛点

V1 的 `gpu_model_runner.py`（7916 行）是一个**巨型单体**，存在以下问题：

1. **模型特定逻辑混杂**：Mamba、编码器-解码器、多模态、pooling 等模型特定行为都堆在 runner 里，用大量 `if/else` 分支
2. **难以维护**：修改一个模型类型可能影响其他模型
3. **难以扩展**：新增模型类型需要修改 runner 核心
4. **CUDA Graph 复杂**：不同模型的 CG 支持级别不同，逻辑分散

### 2.2 V2 的设计原则

V2 runner 文件头注释明确：

```
This model runner is shared by all models: text and multimodal,
generative and embedding, public and private. As a result, this file
must only contain code that is common to every model. Model-specific
behavior belongs in the appropriate model-specific files.

* Be paranoid about changing this file. It should remain stable.
* Be even more paranoid about adding new lines. It should remain minimal.

Even for shared features (for example, different parallelism modes),
keep the complexity out of this path. The less common the feature,
the more it should be hidden. Prefer utility functions defined
elsewhere and call them from here, instead of embedding
feature-specific logic directly.
```

**核心思想**：
- runner 只保留**所有模型共有**的代码
- 模型特定行为通过 **ModelState** 抽象外置
- 用**组合**（composition）替代**继承/分支**

---

## 3. 模块架构

### 3.1 整体结构

```
┌─────────────────────────────────────────────────────────┐
│  V2 GPUModelRunner (model_runner.py, 2270 行)            │
│  模型无关，保持最小化                                      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ModelState (model_states/)                      │   │
│  │  - DefaultModelState        (标准生成)            │   │
│  │  - MambaHybridModelState    (Mamba 混合)         │   │
│  │  - EncoderDecoderModelState (Whisper 等)         │   │
│  │  - EncoderOnlyModelState    (BERT 等)            │   │
│  │  - MmPruningModelState      (多模态剪枝)         │   │
│  │  职责: 输入准备、attention metadata、多模态嵌入     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  PoolingRunner (pool/)                           │   │
│  │  职责: 嵌入/重排模型的 pooling                     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ModelCudaGraphManager (cudagraph_utils.py)      │   │
│  │  职责: CUDA Graph 捕获/重放（BatchExecutionDesc）  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  RequestState / InputBatch / BlockTables         │   │
│  │  职责: 请求状态、输入批处理、块表                   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Sampler / RejectionSampler / Speculator         │   │
│  │  职责: 采样、拒绝采样、投机解码                     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  PPHandler / PCPManager / EPLBController         │   │
│  │  职责: 流水线并行、prefill CP、专家负载均衡         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 与 V1 的目录对比

```
V1: vllm/v1/worker/gpu_model_runner.py  (7916 行单体)

V2: vllm/v1/worker/gpu/
    ├── model_runner.py          (2270 行核心)
    ├── model_states/            (1333 行，模型特定)
    ├── pool/                    (191 行，pooling)
    ├── sample/                  (采样)
    ├── spec_decode/             (投机解码，含 adaptive_verification.py)
    ├── mm/                      (多模态)
    ├── metrics/                 (指标)
    ├── cudagraph_utils.py       (645 行)
    ├── input_batch.py           (716 行)
    ├── attn_utils.py            (616 行)
    ├── pcp_manager.py           (680 行)
    ├── pp_utils.py              (199 行)
    ├── block_table.py           (317 行)
    ├── buffer_utils.py          (316 行)
    ├── eplb_utils.py            (168 行)
    ├── cp_utils.py              (61 行，DCP)
    ├── ec_connector.py          (111 行)
    ├── ubatch_utils.py          (500 行，微批 DBO)
    ├── states.py                (133 行，RequestState)
    ├── warmup.py                (376 行)
    └── ...
```

V2 总代码量比 V1 多，但**核心 runner 从 7916 行降到 2270 行**（main；v0.26.1 为 1702 行），模型特定逻辑被外置到独立模块。

---

## 4. V2 GPUModelRunner 核心

### 4.1 初始化

```python
class GPUModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config, device):
        # 配置解包
        self.model_config / cache_config / compilation_config / ...

        # 并行
        self.use_pp = parallel_config.pipeline_parallel_size > 1
        self.is_first_pp_rank / is_last_pp_rank
        self.dp_size / dp_rank
        self.dcp_size / use_dcp  # decode context parallel
        self.check_ep_fault      # EP all2all 故障检测

        # 多模态
        self.encoder_cache = EncoderCache() if supports_mm and is_first_pp_rank

        # 投机解码
        self.speculator = init_speculator(...) if speculative_config
        self.use_aux_hidden_state_outputs = method in ("eagle3", "dflash", "dspark")
        self.draft_tokens_handler = DraftTokensHandler(device)

        # Pooling
        self.is_pooling_model = model_config.runner_type == "pooling"
        self.pooling_runner = None  # 最后 PP rank 初始化

        # 请求状态
        self.req_states = RequestState(...)
        self.input_buffers = InputBuffers(...)
        self.pp_handler = PPHandler(...) if use_pp

        # 延迟初始化（load_model 后）
        self.sampler / rejection_sampler / prompt_logprobs_worker
        self.structured_outputs_worker / cudagraph_manager

        # LoRA
        self.lora_state = LoraState(...)
        self.lora_capture_cases = get_lora_capture_cases(...)

        # KV Connector
        self.kv_connector = NO_OP_KV_CONNECTOR

        # EPLB
        self.eplb = EPLBController(parallel_config, device)
```

### 4.2 核心方法

| 方法 | 行号（main） | 职责 |
|------|------|------|
| `load_model` | 388 | 加载模型 + LoRA + speculator + model_state |
| `initialize_kv_cache` | 563 | 初始化 KV cache + attention 后端 + adaptive verification / ubatch / fast prefill |
| `profile_run` | 912 | 显存分析 |
| `capture_model` | 962 | CUDA Graph 捕获（`profile_only` 参数，adaptive verification 成本曲线 profiling） |
| `gather_batch_req_state` | 1177 | 从 RequestState 池收集本步 batch 状态（main 新增独立方法） |
| `execute_model` | 1590 | 模型前向（核心） |
| `sample_tokens` | 1937 | 采样 |

### 4.3 execute_model 流程

```python
def execute_model(self, scheduler_output, intermediate_tensors=None, ...):
    # 1. 状态更新
    self._finish_requests(scheduler_output)
    self._free_requests(scheduler_output)
    self._add_new_requests(scheduler_output)
    self._update_states(scheduler_output)
    self.block_tables.apply_staged_writes()

    # 2. 准备输入（通过 model_state）
    input_batch = self._prepare_inputs(scheduler_output)
    # model_state.prepare_inputs(input_batch, req_states)
    # model_state.prepare_attn(input_batch, cudagraph_mode, block_tables, ...)

    # 3. 多模态嵌入（通过 model_state）
    if self.model_state.supports_mm_inputs:
        mm_embeds = self.model_state.get_mm_embeddings(...)

    # 4. 模型前向（CUDA Graph 或 eager）
    desc = self.cudagraph_manager.dispatch(...)
    if desc:
        hidden_states = self.cudagraph_manager.run_fullgraph(desc)
    else:
        with set_forward_context(attn_metadata, ...):
            hidden_states = self.model(input_ids, positions, inputs_embeds)

    # 5. 投机解码 draft
    if self.speculator:
        draft_tokens = self.speculator.propose(...)
        self.req_states.draft_tokens[idx_mapping] = draft_tokens

    # 6. 保存状态供 sample_tokens
    self.execute_model_state = ExecuteModelState(hidden_states, input_batch, ...)
    return None
```

### 4.4 sample_tokens 流程

```python
def sample_tokens(self, grammar_output=None):
    hidden_states, input_batch, ... = self.execute_model_state
    self.execute_model_state = None

    # 1. 提取 logits
    logits = self.model.compute_logits(sample_hidden_states)

    # 2. 结构化输出 grammar bitmask
    if grammar_output:
        self.structured_outputs_worker.apply_grammar_bitmask(...)

    # 3. 采样（投机解码用 rejection_sampler）
    if self.speculator:
        sampler_output = self.rejection_sampler(logits, input_batch, draft_logits)
    else:
        sampler_output = self.sampler(logits, ...)

    # 4. prompt logprobs
    prompt_logprobs = self.prompt_logprobs_worker.compute_prompt_logprobs(...)

    # 5. 后处理（更新状态）
    self._postprocess(input_batch, sampler_output, ...)

    # 6. 返回
    return ModelRunnerOutput(...)
```

---

## 5. ModelState 抽象

### 5.1 设计思想

ModelState 是 V2 的核心抽象，将**模型特定行为**从 runner 剥离：

```python
class ModelState(ABC):
    def __init__(self, vllm_config, model, encoder_cache, device):
        self.model = model
        self.supports_mm_inputs = encoder_cache is not None
        if encoder_cache:
            self.encoder_runner = EncoderRunner(...)

    # 请求生命周期
    def add_request(self, req_index, new_req_data): ...
    def remove_request(self, req_id): ...
    def apply_staged_writes(self): ...

    # 输入准备（抽象方法）
    @abstractmethod
    def prepare_inputs(self, input_batch, req_states) -> dict: ...
    @abstractmethod
    def prepare_dummy_inputs(self, num_reqs, num_tokens) -> dict: ...
    @abstractmethod
    def prepare_attn(self, input_batch, cudagraph_mode, block_tables,
                     slot_mappings, attn_groups, kv_cache_config,
                     for_capture=False) -> dict: ...

    # 多模态
    @abstractmethod
    def get_mm_embeddings(self, scheduled_encoder_inputs, input_batch,
                          req_states) -> torch.Tensor | None: ...
    def gather_mm_embeddings(self, input_batch, draft_lookahead=0): ...

    # 状态钩子
    def preprocess_state(self, input_batch, block_tables, kv_cache_config,
                         num_computed_tokens): ...
    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens): ...

    # CUDA Graph 支持级别
    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
        return AttentionCGSupport.ALWAYS, None

    # 自定义采样器
    def custom_sampler(self, sampler) -> tuple | None:
        return None

    num_new_sampled_tokens_per_step: int = 1
```

### 5.2 init_model_state 工厂

```python
def init_model_state(vllm_config, model, encoder_cache, device):
    # 1. 模型自定义 ModelState（若定义了 get_model_state_cls）
    if hasattr(model, "get_model_state_cls"):
        return model.get_model_state_cls()(vllm_config, model, encoder_cache, device)

    # 2. 编码器-解码器（Whisper, CohereASR, NemotronParse）
    if any(isinstance(m, CrossAttention) for m in model.modules()):
        return EncoderDecoderModelState(...)

    # 3. 编码器-only（BERT/RoBERTa）
    if any(isinstance(m, EncoderOnlyAttention) for m in model.modules()):
        return EncoderOnlyModelState(...)

    # 4. Mamba 混合模型
    if vllm_config.model_config.is_hybrid:
        return MambaHybridModelState(...)

    # 5. 默认（标准生成模型）
    return DefaultModelState(...)
```

### 5.3 各 ModelState 实现

| ModelState | 适用模型 | 特殊行为 |
|-----------|---------|---------|
| `DefaultModelState` | 标准生成模型 | 基础输入准备 + attention metadata |
| `MambaHybridModelState` | Mamba/SSM 混合 | `preprocess_state` 预拷贝 Mamba 状态跨 block 边界；`postprocess_state` 更新状态；per-group attention |
| `EncoderDecoderModelState` | Whisper 等 | 编码器输入处理 + cross-attention |
| `EncoderOnlyModelState` | BERT 等 | 非因果 self-attention，无 KV cache |
| `MmPruningModelState` | 多模态剪枝 | 多模态 token 剪枝 |

### 5.4 MambaHybridModelState 示例

```python
class MambaHybridModelState(DefaultModelState):
    def preprocess_state(self, input_batch, block_tables, kv_cache_config,
                         num_computed_tokens):
        # Mamba "align" 前缀缓存: 预拷贝状态跨 block 边界
        self._ensure_align_ctx(...)

    def prepare_attn(self, ...):
        # Mamba 层 + attention 层的混合 metadata
        # per-group block table / slot mapping

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens):
        # 更新 Mamba 状态（scatter num_accepted）
        _scatter_num_accepted_kernel(...)
```

---

## 6. PoolingRunner 抽象

### 6.1 设计

PoolingRunner 处理**嵌入/重排模型**（非生成模型）的 pooling 操作：

```python
class PoolingRunner:
    def __init__(self, model, vllm_config):
        self.model = model
        # 检测支持的 pooling 任务

    @staticmethod
    def get_supported_tasks(model) -> list[PoolingTask]:
        # embedding / classification / reranking / ...

    def add_request(self, req_index, new_req_data): ...
    def remove_request(self, req_index): ...

    def _get_pooling_metadata(self, ...) -> PoolingMetadata: ...

    def pool(self, hidden_states, pooling_metadata) -> torch.Tensor:
        # 执行 pooling（mean/last/cls/...）
        return self.model.pooler(hidden_states, pooling_metadata)

    def dummy_pooler_run(self, hidden_states): ...
```

### 6.2 与生成模型的区别

| 维度 | 生成模型 | Pooling 模型 |
|------|---------|-------------|
| 输出 | token 序列 | 向量/分数 |
| 采样 | 有（Sampler） | 无 |
| Pooling | 无 | 有（PoolingRunner） |
| KV cache | 有 | 通常无（encoder-only） |
| 投机解码 | 支持 | 不支持 |

---

## 7. CUDA Graph 管理（V2）

### 7.1 BatchExecutionDescriptor

V2 用 `BatchExecutionDescriptor` 作为 CUDA Graph 的 key（比 V1 的 `BatchDescriptor` 更丰富）：

```python
@dataclass
class BatchExecutionDescriptor:
    num_tokens: int           # token 数
    num_reqs: int             # 请求数
    num_loras: int            # LoRA 数
    cudagraph_mode: CUDAGraphMode  # FULL / PIECEWISE / NONE
    # ... 其他维度
```

### 7.2 CudaGraphManager

```python
class CudaGraphManager:
    def __init__(self, ...):
        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self._candidates: dict[tuple[int,int], list[BatchExecutionDescriptor]] = {}
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}

    def dispatch(self, num_tokens, num_reqs, num_loras, ...) -> BatchExecutionDescriptor | None:
        # 查找匹配的 CG descriptor
        # 考虑 attention 后端的 CG 支持级别
        ...

    def run_fullgraph(self, desc):
        # 重放 FULL CG
        self.graphs[desc].replay()
```

### 7.3 ModelCudaGraphManager

```python
class ModelCudaGraphManager(CudaGraphManager):
    # 模型特定的 CG 管理
    # 考虑 model_state.get_additional_cg_support()
    # 不同模型有不同的 CG 支持级别
```

### 7.4 AttentionCGSupport

```python
class AttentionCGSupport(Enum):
    ALWAYS = ...      # 总是支持 CG
    UNIFORM_BATCH = ...  # 仅 uniform batch 支持
    NEVER = ...       # 不支持 CG
```

不同 attention 后端/模型有不同的 CG 支持级别，V2 通过 `get_additional_cg_support` 统一查询。

---

## 8. InputBatch 与 RequestState（V2）

### 8.0 关键机制：固定 buffer + 每步重建 batch（"固定车位"）

这是理解 V2 数据流的核心，也是常被问到的"固定停车位 / 增量更新 batch"问题：

**vLLM 没有"请求占固定车位、batch 增量更新"的机制。** 每个 step 的 `InputBatch` 都是**新建**的（`execute_model` → `prepare_inputs` 从 `RequestState` 池重新收集），batch 内请求顺序每步都可以重排。真正"固定"的是两层：

1. **`InputBuffers`：预分配的固定 GPU buffer**（`input_batch.py`）

```python
class InputBuffers:
    def __init__(self, max_num_reqs, max_num_tokens, device):
        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.dcp_local_seq_lens = ...  # DCP: per-request 本地 seq_lens
```

   大小按 `max_num_reqs`/`max_num_tokens` 一次性分配，每步**原地覆写**前缀。这是 CUDA Graph 要求输入地址固定的原因 —— graph 重放时读的是同一块内存。

2. **`RequestState` 池：请求占固定 slot，但只改映射**

   每个请求在 `RequestState` 池里占一个 `req_idx`（`free_indices` 分配，请求结束归还）。每步 batch 重排时**不动状态数据**，只重建 `idx_mapping`（`batch_idx -> req_state_idx`）：

```
step N:   batch_idx [0,1,2]  ->  req_idx [5, 2, 9]     (idx_mapping 重建)
step N+1: batch_idx [0,1,2]  ->  req_idx [2, 9, 12]   (req 5 完成被移除，新 req 12 加入)
                RequestState 池（固定 slot，数据原地更新）
                slot 2: num_computed_tokens += 1, last_sampled_tokens 更新
                slot 9: ...
                slot 12: 新请求写入 prompt
```

   状态更新（`num_computed_tokens`、`last_sampled_tokens`、`draft_tokens`）通过 `StagedWriteTensor` 批量 stage + `apply_staged_writes()` 一次性拷贝，避免逐请求的 CPU→GPU 小拷贝。

**总结**：固定的是 **buffer 地址**（CUDA Graph 需要）和 **请求状态 slot**（避免状态搬迁）；不固定的是 **batch 组成与顺序**（每步按调度结果重建 `InputBatch` + `idx_mapping`）。

### 8.1 InputBatch

V2 的 InputBatch 与 V1 类似，但更模块化（main 新增字段加粗）：

```python
@dataclass
class InputBatch:
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int
    idx_mapping: torch.Tensor          # batch_idx -> req_state_idx
    idx_mapping_np: np.ndarray
    expanded_idx_mapping: torch.Tensor # 投机解码展开（含 draft token 行）
    expanded_local_pos: torch.Tensor
    num_scheduled_tokens: np.ndarray   # adaptive verification 下是上界
    num_tokens: int
    num_tokens_after_padding: int
    num_draft_tokens: int
    num_draft_tokens_per_req: np.ndarray | None
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    dcp_local_seq_lens: torch.Tensor | None   # DCP
    num_computed_tokens_np: np.ndarray
    prefill_len_np: np.ndarray
    num_computed_prefill_tokens_np: np.ndarray
    is_prefilling_np: np.ndarray
    has_prefill: bool
    input_ids: torch.Tensor
    positions: torch.Tensor
    is_padding: torch.Tensor           # VLLM_MOE_SKIP_PADDING 标记尾部 padding 行
    logits_indices: torch.Tensor
    cu_num_logits: torch.Tensor
    has_structured_output_reqs: bool
    prompt_lens: torch.Tensor | None   # R-SWA
    max_query_len: int | None = None   # main 新增: CG descriptor 承诺的 query 长度
    fast_prefill: FastPrefillBatchMetadata | None = None  # main 新增: KV-sharing 快速 prefill
```

main 新增 `set_dummy_context()`：给 dummy 请求伪造 `context_len` 个已计算 token，用于 adaptive verification 的 step 成本 profiling（decode 形状）。

### 8.2 RequestState

```python
class RequestState:
    def __init__(self, max_num_reqs, max_model_len, max_num_batched_tokens,
                 num_speculative_steps, vocab_size, device):
        # slot 管理
        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # 全量 token ids（UVA 内存，可能数 GB，省 GPU 显存）
        self.all_token_ids = StagedWriteTensor((max_num_reqs, max_model_len),
                                               dtype=torch.int32, uva_instead_of_gpu=True)
        # prompt_len: 用户 prompt 长度; prefill_len: 进入 runner 的长度
        # （抢占恢复后 prefill_len > prompt_len，prompt logprobs/frequency penalties 依赖此区分）
        self.prompt_len = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.total_len = StagedWriteTensor(max_num_reqs, dtype=torch.int32)  # 随生成增长

        # 已计算 token 数（GPU 值 + CPU 乐观镜像上界）
        self.num_computed_prefill_tokens = np.zeros(max_num_reqs, np.int32)
        self.num_computed_tokens = StagedWriteTensor(max_num_reqs, dtype=torch.int32)
        self.num_computed_tokens_np = np.zeros(max_num_reqs, np.int32)

        self.last_sampled_tokens = torch.zeros(max_num_reqs, 1, dtype=torch.int64)
        self.max_seq_len = np.zeros(max_num_reqs, np.int32)  # prompt_len + max_tokens
        self.draft_tokens = torch.zeros(max_num_reqs, num_speculative_steps, dtype=torch.int64)
        self.next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32)

    def add_request(self, req_id, prompt_len, all_token_ids,
                    num_computed_tokens, max_tokens) -> None: ...
    def apply_staged_writes(self) -> None:  # 批量 stage -> 一次性拷贝
    def remove_request(self, req_id) -> int | None:  # 归还 slot
```

main 新增 `num_prefill_lookahead` 参数（prefill 前瞻 token 数）。

### 8.3 Triton Kernel

V2 的 InputBatch 准备使用 Triton kernel（与 V1 类似）：
- `prepare_prefill_inputs`
- `prepare_pos_seq_lens`
- `combine_sampled_and_draft_tokens`
- `post_update`
- `post_update_num_computed_tokens`
- `expand_idx_mapping`

---

## 9. Attention 后端初始化（V2）

### 9.1 init_attn_backend

```python
def init_attn_backend(kv_cache_config, vllm_config, device):
    # 为每个 KV cache group 初始化 attention 后端
    # 返回 attn_backends + attn_metadata_builders
    ...
```

### 9.2 AttentionGroup

V2 引入 `AttentionGroup` 概念，将 attention 层按后端分组：

```python
class AttentionGroup:
    backend: AttentionBackend
    layer_names: list[str]
    metadata_builder: AttentionMetadataBuilder
```

### 9.3 init_kv_cache

```python
def init_kv_cache(runner_kv_caches, forward_context, kv_cache_config,
                  attn_backends, device):
    # 分配 KV cache tensor
    # reshape 到后端期望的格式
    # 绑定到 forward_context
    ...
```

---

## 10. V1 vs V2 对比

### 10.1 架构对比

| 维度 | V1 | V2 |
|------|----|----|
| 核心 runner | 7916 行单体（main 7629） | 2270 行（模型无关） |
| 模型特定逻辑 | 混在 runner 里 | 外置到 ModelState |
| 多模态 | runner 内 if/else | ModelState.get_mm_embeddings |
| Mamba | runner 内分支 | MambaHybridModelState |
| Pooling | runner 内分支 | PoolingRunner |
| 编码器-解码器 | runner 内分支 | EncoderDecoderModelState |
| CUDA Graph key | BatchDescriptor | BatchExecutionDescriptor |
| CG 支持级别 | 分散 | AttentionCGSupport 统一 |
| 投机解码 | runner 内 | spec_decode/ 子模块 |
| 采样 | runner 内 | sample/ 子模块 |

### 10.2 代码量对比

```
V1: gpu_model_runner.py = 7916 行（单体；main 7629 行）

V2: gpu/ 目录（main，模块化）
    - model_runner.py = 2270 行（核心）
    - model_states/ = 1333 行（模型特定）
    - cudagraph_utils.py = 645 行
    - input_batch.py = 716 行
    - attn_utils.py = 616 行
    - pcp_manager.py = 680 行
    - ubatch_utils.py = 500 行
    - spec_decode/adaptive_verification.py = 507 行
    - sample/ = ~1500 行
    - spec_decode/ = ~2500 行
    - 其他 = ~2000 行
```

V2 总代码量更多，但**核心 runner 大幅精简**，模型特定逻辑被清晰外置。

### 10.3 设计优势

| 优势 | 说明 |
|------|------|
| 可维护性 | 修改 Mamba 不影响标准模型 |
| 可扩展性 | 新增模型类型只需新增 ModelState |
| 稳定性 | 核心 runner 保持最小化，变更少 |
| 可测试性 | 各模块可独立测试 |
| 清晰性 | 职责分离，代码意图明确 |

### 10.4 设计代价

| 代价 | 说明 |
|------|------|
| 间接层 | runner → ModelState → 具体实现，多一层调用 |
| 学习成本 | 需要理解 ModelState 抽象 |
| 实验性 | 仍在开发中，可能有 bug |
| 性能 | 额外的抽象层可能有微小开销 |

---

## 11. 启用 V2

### 11.1 环境变量

```bash
# 显式启用/禁用 V2 model runner
export VLLM_USE_V2_MODEL_RUNNER=1   # 强制 V2
export VLLM_USE_V2_MODEL_RUNNER=0   # 强制 V1
# 不设置（默认 None）→ 自动检测（见 11.2）
```

### 11.2 自动检测逻辑（`VllmConfig.use_v2_model_runner`）

不设环境变量时按以下规则自动选择（v0.26.1 起，main 延续）：

```python
@property
def use_v2_model_runner(self) -> bool:
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2                      # 1. 显式设置优先

    # 2. 只有 V2 实现的功能 → 强制 V2（不支持时 _validate_v2_model_runner 报错，
    #    而不是静默回退 V1）
    if self.parallel_config.prefill_context_parallel_size > 1:  # PCP
        return True
    if self.speculative_config and self.speculative_config.method == "dspark":
        return True
    if self._dflash_needs_multi_kv_group():  # 混合 sliding/full DFlash draft
        return True
    if self.model_config.is_diffusion:
        return True

    # 3. 非默认 V2 模型 → V1
    if not self._is_default_v2_model_runner_model():
        return False

    # 4. 依赖/特性检查
    if not HAS_TRITON:
        return False                        # V2 需要 Triton
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        return False                        # 日志提示不支持的特性，回退 V1

    return True                             # 5. 默认 V2
```

**注意**：新版本的默认方向已转向 V2 —— 默认 V2 模型列表内的模型，只要 Triton 可用且无不支持特性，**默认就走 V2**；V1 成为需要显式 `VLLM_USE_V2_MODEL_RUNNER=0` 的选项。

### 11.3 Worker 选择逻辑

```python
# gpu_worker.py
if self.use_v2_model_runner:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    self.model_runner = GPUModelRunnerV2(vllm_config, device)
else:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
    self.model_runner = GPUModelRunnerV1(vllm_config, device)
```

### 11.4 适用场景

| 场景 | 推荐 |
|------|------|
| 默认 V2 模型列表内的模型 | V2（默认） |
| PCP / DSpark / 多 KV group DFlash / diffusion | 只能 V2 |
| 旧模型/特性未覆盖 | V1（自动回退或显式 `=0`） |
| Mamba 混合 / 多模态 | V2（更好的支持） |

---

## 12. 配置参数与环境变量

### 12.1 环境变量

```bash
VLLM_USE_V2_MODEL_RUNNER=1|0|未设置   # 未设置时自动检测（见 11.2）
VLLM_MOE_SKIP_PADDING=1               # main 新增: 标记 CG padding 行，MoE kernel 跳过无效行
```

### 12.2 VllmConfig

```python
# vllm/config/vllm.py
@property
def use_v2_model_runner(self) -> bool:
    # 显式 env > 强制 V2 功能（PCP/dspark/多KV group DFlash/diffusion）
    # > 默认 V2 模型列表 + Triton + 无不支持特性
    ...
```

### 12.3 相关配置

V2 使用与 V1 相同的 `CompilationConfig`、`SchedulerConfig`、`CacheConfig` 等。main 新增的联动配置：

- `cache_config.kv_sharing_fast_prefill` → 启用 `FastPrefillHelper`（KV-sharing 层快速 prefill）
- speculator 的 `enable_adaptive_verification` → 启用 `AdaptiveVerificationManager`，并把 CG 模式解析为 `FULL_DECODE_ONLY` 或 `FULL_AND_PIECEWISE`（varlen decode 路由）
- `parallel_config` 的 DCP（decode context parallel）→ `cp_utils.maybe_prepare_dcp_local_seq_lens` 准备 per-request 本地 seq_lens

---

## 13. main 新特性（v0.26.1 之后）

以下特性在 upstream main 的 V2 runner 中新增（`model_runner.py` 1702 → 2270 行的主要来源），本地 v0.26.1 代码尚无。

### 13.1 Adaptive Verification（自适应验证）

`spec_decode/adaptive_verification.py`（507 行），面向 DSpark 投机解码：

- **问题**：固定 draft 预算下，各请求的 draft token 接受率差异大，统一预算造成算力浪费
- **机制**：
  1. `capture_model` 阶段用 `batches_to_profile()` + `set_dummy_context()`（伪造 decode 形状）profiling 各 batch 形状的 step 成本，拟合**成本曲线**（`set_initial_cost_curves`）
  2. 每步 `_prepare_inputs` 中 `compact_batch()` 在 **GPU 端**重新分配 draft 预算：`_assign_draft_token_budget` 按每个 (request, step) slot 的**存活概率**（该请求逐位置置信度的连乘）做全局 top-k 准入 —— 存活概率沿请求单调递减，所以全局 top-k 天然保证每请求连续准入
  3. `num_scheduled_tokens` 变为**上界**，实际 `num_tokens` 可小于其和（`InputBatch` 注释明确）
- **CG 联动**：`resolve_adaptive_cudagraph_mode` 为 varlen decode 选独立路由（`FULL_DECODE_ONLY` 或 `FULL_AND_PIECEWISE`）

### 13.2 微批执行（UBatch / DBO）

`ubatch_utils.py`（500 行）：`maybe_build_ubatch_runner` 构建 `UBatchRunner`，把一个大 batch 切成 microbatch 双缓冲流水（DBO, Dual-Batch Overlap），attention 与 GEMM 重叠。复用 V1 的 `vllm/v1/worker/ubatching.py` 切片逻辑，V2 侧负责 forward context / slot mapping 的 per-ubatch 构建。

### 13.3 Fast Prefill（KV-sharing 快速 prefill）

`cache_config.kv_sharing_fast_prefill` 开启时，`initialize_kv_cache` 创建 `FastPrefillHelper`；每步 `InputBatch.fast_prefill`（`FastPrefillBatchMetadata`）武装 KV-sharing 层的快速 prefill 路径 —— 共享 KV 的层不必重算全部 prefill。dummy（CG 捕获）batch 不带该字段，KV-sharing 层全量执行。

### 13.4 DCP（Decode Context Parallel）

`cp_utils.py`：`maybe_prepare_dcp_local_seq_lens` 为 DCP 组准备 per-request **本地** seq_lens（`InputBuffers.dcp_local_seq_lens` / `InputBatch.dcp_local_seq_lens`），decode 阶段把长上下文切到多卡并行。

### 13.5 其他

| 特性 | 位置 | 说明 |
|------|------|------|
| Watermarking | `vllm/v1/watermarking/` + `GPUWatermarkSampler` | 输出水印，采样器级集成 |
| Routed experts capturer | `init_routed_experts_capturer` | 捕获 MoE 路由专家选择（供 EPLB/分析） |
| EC Connector | `ec_connector.py` | encoder cache 跨实例 transfer（`ECConnectorOutput`） |
| EPLB 增强 | `eplb_utils.py` | `maybe_register_speculator`（speculator 也参与专家负载均衡）、`step_eplb_after` 装饰器 |
| JIT warmup | `JitWarmupRegistry` | 捕获前 JIT 预热 |
| GC 冻结 | `freeze_gc_for_cudagraph_capture` | CG 捕获期间冻结 GC 防干扰 |
| `CUDAGraphStat` | `vllm/compilation/cuda_graph.py` | CG 命中/重放统计 |
| spec_decode 新 speculator | `dflash2/`、`multi_module_mtp/`、`extract_hidden_states.py` | DFlash v2、多模块 MTP |
| `shutdown.py` | — | 优雅关闭钩子 |

---

## 14. 常见问题

### 14.1 启用问题

| 问题 | 原因 | 解决 |
|------|------|------|
| V2 不支持某模型 | 模型未实现 ModelState | 用 V1 或贡献 ModelState |
| V2 性能不如 V1 | 实验性，可能有 bug | 用 V1 或报告 issue |
| V2 启动失败 | 依赖未满足 | 检查 VLLM_USE_V2_MODEL_RUNNER 文档 |

### 14.2 功能差异

| 功能 | V1 | V2 |
|------|----|----|
| 标准生成 | ✅ | ✅ |
| 多模态 | ✅ | ✅（更好） |
| Mamba | ✅ | ✅（更好） |
| Pooling | ✅ | ✅（PoolingRunner） |
| 编码器-解码器 | ✅ | ✅（EncoderDecoderModelState） |
| 投机解码 | ✅ | ✅（spec_decode/） |
| LoRA | ✅ | ✅ |
| 量化 | ✅ | ✅ |
| 分布式 | ✅ | ✅ |

### 14.3 调试

```bash
# 启用 V2 + 详细日志
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=DEBUG

# 查看 ModelState 选择
# 日志会显示 init_model_state 选择的 ModelState 类型
```

---

## 15. 相对 v0.26.1 的增量

本地 v0.26.1rc0 代码（`<local_vllm_checkout>`）与 upstream main 的 V2 差异清单，升级代码时对照：

| 类别 | v0.26.1 | main |
|------|---------|------|
| `model_runner.py` | 1702 行 | 2270 行 |
| `input_batch.py` | 635 行 | 716 行（+`max_query_len`/`fast_prefill`/`is_padding`/`has_prefill`/`set_dummy_context`） |
| `states.py` | 129 行 | 133 行（+`num_prefill_lookahead`） |
| `spec_decode/` | eagle/mtp/dflash/dspark/gemma4/autoregressive | +`adaptive_verification.py`、`dflash2/`、`multi_module_mtp/`、`extract_hidden_states.py` |
| `sample/` | sampler/gumbel/penalties/logit_bias… | +`batch_shard.py` |
| 新文件 | — | `ubatch_utils.py`(500)、`ec_connector.py`(<device-D>)、`shutdown.py`(20) |
| 新机制 | — | adaptive verification、微批 DBO、fast prefill、watermarking、routed experts capturer、JIT warmup、GC 冻结 |
| 启用逻辑 | 默认 V1，`VLLM_USE_V2_MODEL_RUNNER=1` 启用 | 自动检测，默认 V2 模型列表内默认走 V2 |

---

> **文档版本**：基于 upstream main（2026-09 快照，V1 架构 + 实验性 V2 runner）  
> **生成日期**：2026-08-29（2026-09-11 增量更新至 main）  
> **核心文件**：`vllm/v1/worker/gpu/model_runner.py`, `model_states/`, `pool/`, `cudagraph_utils.py`, `vllm/v1/worker/gpu_worker.py`
