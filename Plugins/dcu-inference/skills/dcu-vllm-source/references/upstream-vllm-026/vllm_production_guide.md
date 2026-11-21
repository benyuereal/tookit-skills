# vLLM 生产级推理框架指导手册

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖架构、数据流、参数、最佳实践与配置。

---

## 目录

1. [整体架构](#1-整体架构)
2. [核心模块详解](#2-核心模块详解)
3. [数据流详解](#3-数据流详解)
4. [配置系统](#4-配置系统)
5. [并行策略](#5-并行策略)
6. [Attention 后端与 Kernel](#6-attention-后端与-kernel)
7. [推理优化技术](#7-推理优化技术)
8. [API 服务与部署](#8-api-服务与部署)
9. [生产最佳实践](#9-生产最佳实践)
10. [环境变量参考](#10-环境变量参考)
11. [架构图](#11-架构图)

---

## 1. 整体架构

### 1.1 架构概览

vLLM V1 采用 **Engine Frontend + Engine Core** 的分离架构：

```
┌─────────────────────────────────────────────────────────┐
│                    API Server (FastAPI)                    │
│   /v1/chat/completions  /v1/completions  /v1/embeddings  │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                 AsyncLLM / LLMEngine                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ InputProcessor│  │OutputProcessor│  │EngineCoreClient│  │
│  └──────────────┘  └──────────────┘  └───────┬───────┘  │
└──────────────────────────────────────────────────────────┘
                          │ ZMQ / In-Process
┌─────────────────────────▼───────────────────────────────┐
│                     EngineCore                            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Scheduler  │  │  Executor    │  │  KV Cache Mgr │  │
│  └──────────────┘  └──────┬───────┘  └───────────────┘  │
│                           │                               │
│  ┌────────────────────────▼────────────────────────────┐ │
│  │              GPUModelRunner (V1)                      │ │
│  │  InputBatch → Model Forward → Sampling               │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 1.2 模块层次

```
vllm/
├── config/          # 配置系统（20+ 配置文件）
├── engine/          # 引擎入口（兼容层，重定向到 V1）
├── entrypoints/     # API 服务入口（OpenAI 兼容、gRPC、离线推理）
├── v1/              # V1 核心引擎
│   ├── engine/      # 引擎前端（LLMEngine, AsyncLLM, EngineCoreClient）
│   ├── core/        # 引擎核心（Scheduler, KV Cache, Block Pool）
│   ├── executor/    # 执行器抽象层（多进程/Ray/单进程）
│   ├── worker/      # Worker 实现（GPU/CPU/XPU/TPU）
│   │   └── gpu/     # GPU 专用模块
│   ├── attention/   # V1 注意力后端注册与选择
│   ├── sample/      # 采样器
│   └── spec_decode/ # 投机解码
├── attention/       # 注意力层抽象
├── model_executor/  # 模型执行
│   ├── models/      # 210+ 模型架构
│   ├── layers/      # 模型层（Linear, Attention, MoE, RoPE）
│   └── model_loader/# 模型加载
├── distributed/     # 分布式通信（TP/PP/DP/EP）
├── multimodal/      # 多模态处理（图像/音频/视频）
├── lora/            # LoRA 适配器管理
├── compilation/     # 编译优化（CUDA Graph, torch.compile）
├── device_allocator/# 设备内存分配
├── transformers_utils/ # HuggingFace 集成
└── csrc/            # C++/CUDA 自定义 Kernel
```

---

## 2. 核心模块详解

### 2.1 请求生命周期

```
用户请求
    │
    ▼
InputProcessor.process_inputs()
    ├── Tokenization（tokenizer encode）
    ├── 多模态数据处理（图像/音频/视频特征提取）
    ├── 生成 EngineCoreRequest（token_ids, sampling_params, mm_features）
    └── assign_request_id（内部唯一ID）
    │
    ▼
OutputProcessor.add_request()
    ├── 创建 RequestState
    ├── 绑定 detokenizer
    └── 创建 RequestOutputCollector 队列
    │
    ▼
EngineCoreClient.add_request()
    ├── 序列化 EngineCoreRequest
    └── 通过 ZMQ 发送到 EngineCore 进程
    │
    ▼
EngineCore.preprocess_add_request()
    ├── 多模态特征缓存
    ├── 结构化输出语法编译
    └── 创建 Request 对象 → Scheduler.add_request()
    │
    ▼
EngineCore.step() 循环:
    ├── Scheduler.schedule()
    │   ├── 从 waiting 队列取请求
    │   ├── 从 running 列表取已有请求
    │   ├── 检查 prefix cache 命中
    │   ├── KV Cache 分配（allocate_slots）
    │   ├── Preemption（KV Cache 不足时抢占）
    │   └── 构建 SchedulerOutput（batch, block_tables）
    │
    ├── Executor.execute_model(scheduler_output)
    │   └── GPUModelRunner.execute_model()
    │       ├── InputBatch 准备（token_ids, positions, attn_metadata）
    │       ├── 多模态嵌入合并
    │       ├── Model Forward（CUDA Graph 或 eager）
    │       └── 返回 hidden_states
    │
    ├── Executor.sample_tokens(grammar_output)
    │   ├── 提取 logits（仅需采样的位置）
    │   ├── 应用 logit bias / penalties / temperature
    │   ├── Top-K / Top-P 过滤
    │   ├── Gumbel 采样
    │   └── 返回 sampled_token_ids
    │
    └── Scheduler.update_from_output()
        ├── 更新 request 状态
        ├── 检查 stop 条件（EOS, stop_strings, max_tokens）
        └── 生成 EngineCoreOutputs
    │
    ▼
OutputProcessor.process_outputs()
    ├── Detokenizer.update() → 文本解码 + stop string 检测
    ├── LogprobsProcessor → logprobs 计算
    ├── 构建 RequestOutput
    └── 推入 RequestOutputCollector 队列
    │
    ▼
AsyncLLM.generate() yield → API Server → HTTP Response
```

### 2.2 Scheduler（调度器）

调度器是 vLLM 的核心决策模块，负责决定每个 step 中哪些请求可以执行。

**核心算法**（`schedule_default`）：

1. **Running 请求处理**：遍历所有已运行的请求，为每个请求分配 token 预算，继续生成
2. **Waiting 请求准入**：从等待队列取出请求，检查 token 预算、最大并发数限制
3. **Chunked Prefill**：长 prompt 拆分处理，每步最多处理 `long_prefill_token_threshold` 个 token
4. **Prefix Caching**：检查请求的 prompt 是否已有缓存的 KV Cache
5. **Preemption**：KV Cache 不足时，抢占最低优先级请求（PRIORITY）或最后一个请求（FCFS）

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_num_batched_tokens` | 2048 | 每步最大处理 token 数 |
| `max_num_seqs` | 128 | 最大并发序列数 |
| `enable_chunked_prefill` | True | 启用分块预填充 |
| `long_prefill_token_threshold` | 0（auto：max_model_len * 4%） | 长 prompt 阈值 |
| `policy` | fcfs | 调度策略（fcfs/priority） |
| `async_scheduling` | None（自动选择） | 异步调度（允许调度超前于输出） |

**SchedulerOutput 结构**：

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]       # 首次调度的请求（完整数据）
    scheduled_cached_reqs: list[CachedRequestData]  # 已缓存的请求（增量数据）
    num_scheduled_tokens: dict[str, int]            # 每个请求的调度 token 数
    total_num_scheduled_tokens: int                 # 总调度 token 数
    scheduled_spec_decode_tokens: dict              # 投机解码 token
    finished_req_ids: set[str]                      # 完成的请求 ID
    preempted_req_ids: set[str]                     # 被抢占的请求 ID
    num_common_prefix_blocks: int                   # 公共前缀 block 数
```

### 2.3 KV Cache 管理

KV Cache 是推理显存的主要消耗，vLLM 采用 **PagedAttention** 机制管理。

**PagedAttention 原理**：
- KV Cache 被划分为固定大小的 **Block**（默认 16 tokens）
- 物理 Block 不连续，通过 **Block Table** 映射逻辑位置到物理位置
- 类似操作系统的虚拟内存分页

**KV Cache 类型**：
- **Local KV Cache**：GPU 显存中的 KV Cache
- **External KV Cache**（KV Transfer）：通过 RDMA/NVLink 在 GPU 间传输 KV Cache
- **CPU Offload**：将不活跃的 KV Cache 卸载到 CPU 内存

```python
class KVCacheManager:
    def allocate_slots(request, num_tokens)     # 分配 KV Cache slot
    def free(request)                           # 释放 KV Cache
    def get_computed_blocks(request)            # 查询 prefix cache 命中
    def cache_blocks(request, token_indices)    # 缓存新计算的 block
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 16（默认）/64（flash_attn+flash_mla） | Block 大小（token 数） |
| `gpu_memory_utilization` | 0.9 | GPU 显存利用率 |
| `enable_prefix_caching` | True | 启用前缀缓存 |
| `cpu_offload_gb` | 0 | CPU 卸载大小（GB） |
| `swap_space` | 4 | 交换空间（GB） |

### 2.4 GPUModelRunner（GPU 模型执行器）

**文件路径**：`vllm/v1/worker/gpu/model_runner.py`（V1 架构的核心实现）

**核心流程**：

```python
class GPUModelRunner:
    def execute_model(scheduler_output) -> ModelRunnerOutput | None:
        # 1. 状态管理
        self._finish_requests()
        self._free_requests()
        self._add_new_requests()
        self._update_states()
        self.input_batch.block_table.commit_staged_writes()

        # 2. 准备输入
        input_batch = self._prepare_inputs(scheduler_output)
        # 包含：token_ids, positions, seq_lens, slot_mapping,
        #       query_start_loc, block_tables, attn_metadata

        # 3. 多模态编码
        if multimodal_inputs:
            multimodal_embeddings = self.mm_runner.run(...)
            input_batch.merge_multimodal_embeddings(multimodal_embeddings)

        # 4. 模型前向传播
        with set_forward_context(attn_metadata, ...):
            output = self.model(
                input_ids=input_batch.token_ids,
                positions=input_batch.positions,
            )

        # 5. 采样（异步模式下延迟到 sample_tokens）
        return None

    def sample_tokens(grammar_output) -> SamplerOutput:
        # 提取 logits → 应用 bias/penalties → top-k/top-p → Gumbel 采样
        ...
```

**InputBatch 结构**（GPU 端批量管理）：
- `token_ids`：当前步输入 token
- `positions`：token 的位置编码
- `seq_lens`：每个请求的序列长度
- `slot_mapping`：token 到 KV Cache slot 的映射
- `query_start_loc`：用于 ragged batch attention
- `block_tables`：Block Table 映射
- `attn_metadata`：Attention 后端所需的元数据

### 2.5 Model Executor 层

**模型层目录**：`vllm/model_executor/layers/`

核心层：
- **Linear 层**：`ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`（支持 TP 分片）
- **Attention 层**：`Attention`（vLLM 自定义注意力层，后端可插拔）
- **MLA 层**：`MLA`（Multi-head Latent Attention，DeepSeek 等模型使用）
- **MoE 层**：`fused_moe/` 目录，48+ 文件实现各种 MoE 融合 Kernel
- **RoPE 层**：`rotary_embedding/` 目录，18+ 文件实现各种位置编码
- **Normalization 层**：`RMSNorm`, `LayerNorm`（支持量化和融合）
- **LoRA 层**：支持 LoRA adapter 的并行线性层

---

## 3. 数据流详解

### 3.1 请求处理数据流

```
┌─────────────────────┐
│ HTTP Request        │  POST /v1/chat/completions
│ {model, messages,   │
│  temperature, ...}  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Chat Template       │  Jinja2 渲染 → prompt string
│ + Tokenization      │  tokenizer.encode() → token_ids
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ EngineCoreRequest   │  msgspec 序列化
│ (request_id,        │  ZMQ 传输到 EngineCore 进程
│  prompt_token_ids,  │
│  sampling_params,   │
│  mm_features, ...)  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Request (Scheduler) │  调度器内部状态
│ (request_id,        │  waiting → running
│  prompt_token_ids,  │  num_computed_tokens
│  sampling_params,   │  status: WAITING/RUNNING/PREEMPTED/FINISHED
│  status, ...)       │
└──────────┬──────────┘
           ▼ (每 step)
┌─────────────────────┐
│ SchedulerOutput     │  批量调度结果
│ (scheduled_new_reqs,│  首次请求 vs 缓存请求
│  scheduled_cached,  │  每请求 token 分配
│  num_scheduled,     │
│  block_tables, ...) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ InputBatch (GPU)    │  GPU 端 Tensor 批量
│ token_ids: [batch]  │
│ positions: [batch]  │
│ slot_mapping: [...] │
│ attn_metadata: {...}│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Model Forward       │  CUDA Graph 或 eager
│ Attention + FFN     │  → hidden_states
│ + MoE (可选)        │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Logits Extraction   │  仅提取采样位置的 logits
│ + Grammar Bitmask   │  结构化输出约束
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Sampler (GPU)       │  bias → penalties → temp
│                     │  → top-k/top-p → Gumbel
│                     │  → sampled_token_ids
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ EngineCoreOutput    │  msgspec 序列化
│ (request_id,        │  ZMQ 传回 Engine Frontend
│  new_token_ids,     │
│  finish_reason,     │
│  new_logprobs, ...) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Detokenizer         │  token_id → text
│ + Stop Check        │  stop_string 检测
│ + Logprobs Calc     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ RequestOutput       │  SSE stream 或 JSON
│ (request_id,        │  HTTP Response
│  outputs: [{text,   │
│    token_ids,       │
│    logprobs, ...}], │
│  usage, ...)        │
└─────────────────────┘
```

### 3.2 KV Cache 数据流

```
请求到达
    │
    ▼
prefix cache 查询
    ├── 命中: num_computed_tokens = cached_tokens
    │   block_table 复用已有物理 Block
    └── 未命中: num_computed_tokens = 0
        分配新的物理 Block
    │
    ▼
每个 step 计算新 token 后:
    ├── KV Cache 写入（GPU kernel: reshape_and_cache）
    ├── Block Table 更新
    └── 标记新 block 为 computed
    │
    ▼
请求完成 / 被抢占:
    ├── 释放物理 Block（加入 free list）
    └── 如果 block 是 prefix 的一部分: 保留（prefix caching）
```

### 3.3 分布式通信数据流

```
SchedulerOutput (CPU)
    │
    ▼ broadcast (SHM/NCCL) → 所有 Worker
    │
┌───▼───────────────────────┐
│ Worker 0 (TP Rank 0)      │
│ Worker 1 (TP Rank 1)      │  每个 TP rank 处理模型的一部分
│ ...                       │  TP 内 all_reduce 同步激活值
│ Worker N (TP Rank N)      │
└───┬───────────────────────┘
    │
    ▼ (PP 场景)
    │
    intermediate_tensors (hidden_states)
    │ 通过 P2P 传输到下一个 PP stage
    ▼
    next PP stage 的 Worker 0 (TP Rank 0)
    ...
    │
    ▼
ModelRunnerOutput (GPU Tensor)
    │
    ▼ gather → Executor
    │ → EngineCoreClient → OutputProcessor
```

---

## 4. 配置系统

### 4.1 配置层次

vLLM 从上到下有四个配置层次：

1. **VllmConfig**（顶层容器）：聚合所有子配置
2. **子配置类**：`ModelConfig`, `CacheConfig`, `ParallelConfig`, `SchedulerConfig` 等
3. **OptimizationLevel**：O0-O3 预设，自动配置编译和 CUDA Graph
4. **环境变量**：232 个 `VLLM_*` 环境变量，运行时覆盖

### 4.2 配置类详解

#### ModelConfig（模型配置）

```python
@dataclass
class ModelConfig:
    model: str = "Qwen/Qwen3-0.6B"      # HF 模型名或路径
    tokenizer: str | None = None         # tokenizer（默认与 model 相同）
    dtype: str = "auto"                  # auto/float16/bfloat16/float32
    seed: int = 0                        # 随机种子
    max_model_len: int | None = None     # 最大模型长度（auto 从 HF config 推导）
    trust_remote_code: bool = False      # 信任远程代码
    tokenizer_mode: str = "auto"         # auto/slow/mistral
    runner: str = "auto"                 # auto/generate/pooling/draft
    quantization: str | None = None      # 量化方法
    hf_config_overrides: dict | None     # HF config 覆盖
    mm_processor_kwargs: dict | None     # 多模态处理器参数
```

#### ParallelConfig（并行配置）

```python
@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1              # 张量并行度
    pipeline_parallel_size: int = 1            # 流水线并行度
    data_parallel_size: int = 1                # 数据并行度
    prefill_context_parallel_size: int = 1     # Prefill 上下文并行
    decode_context_parallel_size: int = 1      # Decode 上下文并行
    enable_expert_parallel: bool = False       # 专家并行
    enable_eplb: bool = False                  # 专家负载均衡
    distributed_executor_backend: str | None   # ray/mp/uni/external_launcher
    all2all_backend: str = "allgather_reducescatter"
    # world_size = PP × TP × PCP（prefill_context_parallel_size）
```

#### SchedulerConfig（调度器配置）

```python
@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 2048           # 每步最大 token 数
    max_num_seqs: int = 128                      # 最大并发序列数
    enable_chunked_prefill: bool = True          # 分块预填充
    long_prefill_token_threshold: int = 0       # 长 prompt 阈值（0=auto：max_model_len*4%）
    max_num_partial_prefills: int = 1            # 最大部分预填充数
    policy: str = "fcfs"                         # fcfs/priority
    async_scheduling: bool | None = None         # 异步调度
    stream_interval: int = 1                     # 流式输出间隔
```

#### CacheConfig（KV Cache 配置）

```python
@dataclass
class CacheConfig:
    block_size: BlockSize = 16          # Block 大小（flash_attn+flash_mla 时默认为 64）
    gpu_memory_utilization: float = 0.9    # GPU 显存利用率
    swap_space: int = 4                    # CPU 交换空间（GB）
    cache_dtype: str = "auto"              # KV Cache 数据类型
    enable_prefix_caching: bool = True     # 前缀缓存
    prefix_caching_hash_algo: str = "sha256"
    cpu_offload_gb: float = 0              # CPU 卸载
    kv_cache_memory_bytes: int | None      # 精确指定 KV Cache 大小
    enable_kv_cache_events: bool = False   # KV Cache 事件
```

#### AttentionConfig（注意力配置）

```python
@dataclass
class AttentionConfig:
    backend: AttentionBackendEnum | None = None   # 注意力后端（auto 选择）
    flash_attn_version: int | None = None         # FlashAttention 版本
    use_prefill_decode_attention: bool = False    # 分离式 prefill/decode
    flash_attn_max_num_splits_for_cuda_graph: int = 32
    use_cudnn_prefill: bool = False               # cuDNN prefill
    use_trtllm_ragged_deepseek_prefill: bool = True
```

#### LoadConfig（加载配置）

```python
@dataclass
class LoadConfig:
    load_format: str = "auto"                     # auto/safetensors/pt/npz
    download_dir: str | None = None               # 下载目录
    safetensors_load_strategy: str = "lazy"       # lazy/eager
    ignore_patterns: list[str] = ["original/**/*"]
    use_tqdm_on_load: bool = True
    model_loader_extra_config: dict = ...     # 模型加载器额外配置
```

#### LoRAConfig（LoRA 配置）

```python
@dataclass
class LoRAConfig:
    max_lora_rank: int = 16
    max_loras: int = 1
    max_cpu_loras: int | None = None  # 默认等于 max_loras
    fully_sharded_loras: bool = False
    lora_dtype: str = "auto"
    enable_tower_connector_lora: bool = False
```

#### SpeculativeConfig（投机解码配置）

```python
@dataclass
class SpeculativeConfig:
    num_speculative_tokens: int | None = None     # 投机 token 数
    method: str | None = None                     # ngram/medusa/eagle/eagle3/mtp/suffix
    model: str | None = None                      # draft model 路径
    draft_tensor_parallel_size: int | None = None
    # ngram 特有参数
    prompt_lookup_min: int = 0
    prompt_lookup_max: int = 0
```

#### CompilationConfig（编译配置）

```python
@dataclass
class CompilationConfig:
    mode: CompilationMode = None      # None/VLLM_COMPILE 等（V1 默认 VLLM_COMPILE）
    cudagraph_mode: CUDAGraphMode = None  # None/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE（V1 默认 FULL_AND_PIECEWISE）
    cudagraph_capture_sizes: list    # CUDA Graph 捕获的 batch size
    use_inductor: bool = True        # 使用 Inductor 编译器（已废弃，由 mode 控制）
    cudagraph_specialize_lora: bool = True
```

#### ObservabilityConfig（可观测性配置）

```python
@dataclass
class ObservabilityConfig:
    show_hidden_metrics_for_version: str | None = None
    otlp_traces_endpoint: str | None = None
    collect_detailed_traces: bool | None = None
    kv_cache_metrics: bool = False
    kv_cache_metrics_sample: float = 0.01
    cudagraph_metrics: bool = False
    enable_mfu_metrics: bool = False
```

#### OptimizationLevel（优化预设）

| Level | Compilation Mode | CUDA Graph | 适用场景 |
|-------|-----------------|------------|---------|
| O0 | NONE | NONE | 开发调试 |
| O1 | VLLM_COMPILE | PIECEWISE | 快速优化（编译+分段图） |
| O2 | VLLM_COMPILE | FULL_AND_PIECEWISE | 生产默认（全图+分段图） |
| O3 | VLLM_COMPILE | FULL_AND_PIECEWISE | 最大优化（同 O2） |

---

## 5. 并行策略

### 5.1 张量并行（Tensor Parallelism, TP）

**原理**：将模型的权重矩阵按列/行切分到多个 GPU，每次计算后通过 all-reduce 同步。

**适用模型**：单卡放不下模型权重时使用。

**切分方式**：
- **Attention**：按 head 维度切分 Q/K/V，每个 GPU 计算部分 head
- **FFN/MLP**：ColumnParallelLinear（列切分） + RowParallelLinear（行切分）
- **MoE**：按 expert 维度切分

**通信**：
- `all_reduce`：RowParallelLinear 后聚合
- `all_gather`：ColumnParallelLinear 前收集
- `reduce_scatter`：MoE 场景下使用

### 5.2 流水线并行（Pipeline Parallelism, PP）

**原理**：将模型按层切分为多个 stage，各 stage 流水线执行。

**调度方式**：
- 通过 `send/recv` 在 stage 间传递 `IntermediateTensors`
- 使用 `interleaved` 调度减少流水线气泡
- 层分配：`get_pp_indices` 均匀分配

### 5.3 数据并行（Data Parallelism, DP）

**原理**：多个引擎副本处理不同请求，通过 Coordinator 协调。

**两种模式**：
1. **外部负载均衡**：`DPAsyncMPClient`，由外部 LB 分发请求
2. **内部负载均衡**：`DPLBAsyncMPClient`，内部根据队列长度选择引擎

**Wave 机制**：
- 请求分 wave 发送到所有 DP 引擎
- 所有引擎完成当前 wave 后才开始下一个 wave
- 确保 batch size 最大化

### 5.4 上下文并行（Context Parallelism, CP）

**原理**：将长序列的 KV Cache 切分到多个 GPU。

- **Prefill CP**：`prefill_context_parallel_size`，加速长 prompt 处理
- **Decode CP**：`decode_context_parallel_size`，加速长序列生成
- **关键实现**：`vllm/v1/worker/cp_utils.py`

### 5.5 专家并行（Expert Parallelism, EP）

**原理**：将 MoE 模型的 experts 分布到多个 GPU。

```python
ParallelConfig:
    enable_expert_parallel=True
    enable_eplb=True                          # 专家负载均衡
    # EPLBConfig:
    #   window_size=1000                       # 负载均衡窗口
    all2all_backend="allgather_reducescatter" # all-to-all 通信后端
```

### 5.6 并行策略选择指南

| 场景 | 推荐策略 |
|------|---------|
| 7B 模型，单卡 80GB | TP=1, PP=1 |
| 70B 模型（FP16，~140GB） | TP=2 或 TP=4 |
| 405B 模型（FP16） | TP=8 + PP=2 |
| MoE 模型（如 DeepSeek-V3） | TP + EP |
| 高吞吐场景 | DP=2+ |
| 超长序列（128K+） | CP |

---

## 6. Attention 后端与 Kernel

### 6.1 注意力后端列表

vLLM 支持 22 种注意力后端，通过 `AttentionBackendEnum` 注册：

| 后端 | 平台 | 适用场景 |
|------|------|---------|
| **FLASH_ATTN** | NVIDIA | 默认后端，FlashAttention 2/3 |
| **FLASHINFER** | NVIDIA | FlashInfer 后端，优化 decode |
| **TRITON_ATTN** | 通用 | Triton 实现，CPU/NVIDIA/ROCm |
| **FLASHINFER_MLA** | NVIDIA | DeepSeek MLA 专用 |
| **TRITON_MLA** | 通用 | MLA Triton 实现 |
| **CUTLASS_MLA** | NVIDIA | MLA CUTLASS 实现 |
| **FLASHMLA** | NVIDIA | FlashMLA 优化实现 |
| **FLASHMLA_SPARSE** | NVIDIA | FlashMLA 稀疏版本 |
| **FLASH_ATTN_MLA** | NVIDIA | FlashAttention MLA 适配 |
| **FLEX_ATTENTION** | 通用 | PyTorch FlexAttention |
| **TREE_ATTN** | 通用 | Tree Attention |
| **TURBOQUANT** | NVIDIA | 量化注意力后端 |
| **ROCM_ATTN** | AMD | ROCm 注意力 |
| **ROCM_AITER_FA** | AMD | AITER Flash Attention |
| **ROCM_AITER_MLA** | AMD | AITER MLA |
| **ROCM_AITER_MLA_SPARSE** | AMD | AITER MLA 稀疏 |
| **ROCM_AITER_UNIFIED_ATTN** | AMD | AITER 统一注意力 |
| **CPU_ATTN** | CPU | CPU 后端 |
| **IPEX** | Intel | Intel XPU 后端 |
| **NO_ATTENTION** | 通用 | 无注意力（池化模型） |

### 6.2 后端选择逻辑

```python
def get_attn_backend(attn_config, model_config, ...):
    # 1. 用户指定后端 → 直接使用
    # 2. MLA 模型 → MLA 专用后端
    # 3. 平台默认后端（NVIDIA → FLASH_ATTN, ROCm → ROCM_ATTN）
    # 4. 回退到 TRITON_ATTN
```

### 6.3 PagedAttention 实现

```python
# 核心操作
torch.ops.vllm.flash_attn_varlen(
    query, key, value,
    seqlen_q, seqlen_kv,        # 变长序列
    max_seqlen_q, max_seqlen_kv,
    softmax_scale,
    causal=True,
    block_table=block_tables,   # PagedAttention Block Table
    ...
)
```

### 6.4 MLA（Multi-head Latent Attention）

DeepSeek-V2/V3 引入的 MLA 将 KV 压缩到低秩潜空间：

```
传统 Attention: Q, K, V ∈ R^(num_heads × head_dim)
MLA: Q ∈ R^(num_heads × head_dim), KV_latent ∈ R^(kv_lora_rank)
     使用时通过解压缩矩阵恢复 K, V
```

vLLM 为 MLA 提供了 10+ 个后端，关键优化点：
- KV_latent 直接存入 KV Cache（节省显存）
- 解压缩与 Attention 融合执行

### 6.5 Custom CUDA Kernels

`csrc/` 目录下的核心 Kernel：

| 模块 | Kernel | 用途 |
|------|--------|------|
| **attention/** | paged_attention_v1/v2 | PagedAttention |
| **activation_kernels.cu** | silu_and_mul, gelu_and_mul | 融合激活函数 |
| **layernorm_kernels.cu** | rms_norm, fused_add_rms_norm | RMSNorm + 残差融合 |
| **cache_kernels.cu** | reshape_and_cache, copy_blocks | KV Cache 管理 |
| **sampler.cu** | top_k_top_p_sampling | GPU 采样 |
| **quantization/** | awq, gptq_marlin, fp8 | 量化推理 |
| **moe/** | moe_align_block_size, fused_topk | MoE 融合操作 |
| **custom_all_reduce.cu** | custom_all_reduce | TP 通信 |
| **pos_encoding_kernels.cu** | rotary_embedding | RoPE |

---

## 7. 推理优化技术

### 7.1 Prefix Caching（前缀缓存）

**自动启用**（`enable_prefix_caching=True`）：
- 相同前缀的请求共享 KV Cache
- 使用 SHA256 哈希标识 cache prefix
- 系统 prompt / few-shot examples 自动复用
- 首次请求后，后续请求跳过前缀的 prefill

**最佳实践**：
```python
# 相同 system prompt → 自动共享
messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # 这个会被缓存
    {"role": "user", "content": "..."}                               # 从这开始计算
]
```

### 7.2 Chunked Prefill（分块预填充）

**默认启用**（`enable_chunked_prefill=True`）：
- 将长 prompt 拆分成多个 chunk 处理
- 避免长 prompt 阻塞 decode 请求
- 提升 TTFT（Time To First Token）

**工作原理**：
```
Step 1: Prefill chunk 0 (2048 tokens) + Decode 已有请求
Step 2: Prefill chunk 1 (2048 tokens) + Decode
Step 3: Prefill chunk 2 (剩余) + Decode
```

### 7.3 CUDA Graph

**原理**：将 GPU 操作录制成图，消除 kernel launch overhead。

**模式**：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| NONE | 不使用 | 开发/调试 |
| PIECEWISE | 分段捕获（推荐） | 生产环境 |
| FULL | 全图捕获 | batch size 固定 |
| FULL_DECODE_ONLY | 仅 decode 全图 | decode-only 场景 |

**Graph 捕获的 batch sizes**：
```python
cudagraph_capture_sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, ...]
# 匹配到最近的已捕获 size 时使用 CUDA Graph
# 否则 fallback 到 eager
```

### 7.4 torch.compile

**VLLM_COMPILE 模式**：使用自定义编译后端（`vllm.compilation`）

关键组件：
- `backends.py`：自定义 Inductor backend
- `fx_passes.py`：FX 图优化 passes
- `cuda_graph.py`：CUDA Graph 与 compile 集成
- `piecewise_backend.py`：分段编译以减少重编译

### 7.5 投机解码（Speculative Decoding）

**原理**：用小模型或简单方法快速生成 draft tokens，然后 target model 一次性验证。

| 方法 | Draft 来源 | 适用模型 |
|------|-----------|---------|
| **Eagle/Eagle3** | 基于 target hidden states 的 draft model | Llama, Qwen 等 |
| **Medusa** | 额外的预测头 | 任意（需训练） |
| **MTP** | 模型的 Multi-Token Prediction 头 | Qwen3, DeepSeek |
| **N-gram** | prompt/context 中的 n-gram 匹配 | 任意（无需额外模型） |
| **Suffix** | 后缀树匹配 | 任意（无需额外模型） |

**配置示例**：
```python
speculative_config = SpeculativeConfig(
    method="eagle3",
    model="path/to/eagle3/model",
    num_speculative_tokens=5,
)
```

### 7.6 结构化输出（Structured Output）

**支持格式**：
- JSON Schema（`guided_json`）
- Regex（`guided_regex`）
- Grammar / EBNF（`guided_grammar`）
- Choice（`guided_choice`）
- Structural Tag

**后端**：
- **xgrammar**（默认，最快）
- **guidance**
- **outlines**
- **lm-format-enforcer**

**配置**：
```python
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json_schema={"type": "object", "properties": {...}},
        # 或
        regex=r"```python\n[\s\S]*?\n```",
        # 或
        choice=["Positive", "Negative", "Neutral"],
    )
)
```

### 7.7 量化

| 方法 | 说明 | 精度 |
|------|------|------|
| AWQ | Activation-aware weight quantization | INT4 |
| GPTQ | Post-training quantization | INT4/INT8 |
| GPTQ-Marlin | GPTQ 优化 kernel | INT4 |
| FP8 W8A8 | Weight & Activation FP8 | FP8 |
| bitsandbytes | 在线量化 | INT4/INT8 |
| GGUF | llama.cpp 格式 | 多种 |
| FP4 | 4-bit floating point | FP4 |
| MXFP4 | Micro-exponent FP4 | FP4 |

### 7.8 LoRA 适配器服务

```python
# 加载 LoRA
await engine.add_lora(LoRARequest(
    lora_name="my_adapter",
    lora_int_id=1,
    lora_path="/path/to/adapter",
))

# 使用 LoRA
output = await engine.generate(
    prompt="...",
    sampling_params=sampling_params,
    lora_request=LoRARequest("my_adapter", 1, "/path/to/adapter"),
)
```

**LRU 缓存**：自动管理 adapter 加载/卸载，`max_loras` 控制最大并发 adapter 数。

### 7.9 Automatic Prefix Caching (APC)

不同于 Prefix Caching，APC 自动检测并缓存相同模式的前缀，无需显式配置。

### 7.10 CPU Offloading

```python
CacheConfig:
    cpu_offload_gb=16     # 将 16GB KV Cache 卸载到 CPU
    swap_space=8          # 交换空间
```

---

## 8. API 服务与部署

### 8.1 启动服务

```bash
# OpenAI 兼容 API
vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --port 8000

# gRPC 服务（更高吞吐）
vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --grpc-port 8001
```

### 8.2 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI chat completions |
| `/v1/completions` | POST | OpenAI text completions |
| `/v1/embeddings` | POST | 文本嵌入 |
| `/v1/models` | GET | 模型列表 |
| `/v1/tokenize` | POST | Tokenize |
| `/v1/detokenize` | POST | Detokenize |
| `/v1/audio/transcriptions` | POST | 语音转文字 |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/messages` | POST | Anthropic Messages API |
| `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus 指标 |
| `/load` | GET | 模型信息 |
| `/version` | GET | vLLM 版本 |

### 8.3 离线推理

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-32B",
    tensor_parallel_size=4,
    max_model_len=32768,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
)

outputs = llm.generate(["Hello, how are you?"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### 8.4 SamplingParams 参数

```python
@dataclass
class SamplingParams:
    n: int = 1                              # 每个 prompt 生成数量
    temperature: float = 1.0                # 温度
    top_p: float = 1.0                      # Nucleus 采样
    top_k: int = 0                          # Top-K 采样（0=禁用）
    min_p: float = 0.0                      # Min-P 采样
    presence_penalty: float = 0.0           # 存在惩罚
    frequency_penalty: float = 0.0          # 频率惩罚
    repetition_penalty: float = 1.0         # 重复惩罚
    max_tokens: int | None = 16              # 最大生成 token 数
    min_tokens: int = 0                     # 最小生成 token 数
    stop: str | list[str] = None            # 停止字符串
    stop_token_ids: list[int] = None        # 停止 token ID
    ignore_eos: bool = False                # 忽略 EOS
    logprobs: int | None = None             # logprobs 数量
    prompt_logprobs: int | None = None      # prompt logprobs
    seed: int | None = None                 # 随机种子
    output_kind: RequestOutputKind = CUMULATIVE  # CUMULATIVE/DELTA/FINAL_ONLY
    structured_outputs: StructuredOutputsParams | None  # 结构化输出
    logit_bias: dict[int, float] = None     # logit bias
    allowed_token_ids: list[int] = None     # 允许的 token
    skip_special_tokens: bool = True        # 跳过特殊 token
    include_stop_str_in_output: bool = False
    truncate_prompt_tokens: int | None = None
```

### 8.5 Docker 部署

```dockerfile
FROM vllm/vllm-openai:latest

# 或
docker run --gpus all \
    -v /path/to/models:/models \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model /models/Qwen3-32B \
    --tensor-parallel-size 4
```

### 8.6 Kubernetes 部署建议

```yaml
# 关键资源限制
resources:
  limits:
    nvidia.com/gpu: 4
    memory: 256Gi
  requests:
    nvidia.com/gpu: 4
    memory: 200Gi

# 环境变量
env:
  - name: VLLM_LOGGING_LEVEL
    value: "INFO"
  - name: VLLM_LOG_STATS_INTERVAL
    value: "10"
  - name: NCCL_DEBUG
    value: "WARN"
```

---

## 9. 生产最佳实践

### 9.1 模型选择与加载

1. **优先使用 `auto` dtype**：自动选择 BF16/FP16
2. **量化模型**：显存有限时使用 AWQ/GPTQ/FP8
3. **max_model_len**：设置为实际需要的最大长度，不要超过模型训练长度
4. **trust_remote_code**：仅对可信任的模型启用

### 9.2 显存优化

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `gpu_memory_utilization` | 0.85-0.90 | 留 10-15% 给 CUDA context |
| `enable_prefix_caching` | True | 显著减少重复计算 |
| `block_size` | 16 (flash_attn) | 默认值通常最优 |
| `cpu_offload_gb` | 4-16 | 长尾请求场景 |
| `swap_space` | 4 | 基本够用 |

**显存计算**：
```
总显存 = 模型权重 + KV Cache + CUDA Context + 激活值
KV Cache = num_layers × 2 × num_heads × head_dim × max_model_len × dtype_size
```

### 9.3 延迟优化

| 优化 | 效果 | 代价 |
|------|------|------|
| CUDA Graph（PIECEWISE） | TTFT ↓20-30% | 微小显存增加 |
| Chunked Prefill | TTFT ↓50%+ | 略微降低吞吐 |
| FlashInfer backend | Decode ↓10-15% | 需要安装 FlashInfer |
| FP8 KV Cache | 显存占用减半 | 精度轻微损失 |

### 9.4 吞吐优化

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_num_seqs` | 128-256 | 显存允许下尽量大 |
| `max_num_batched_tokens` | 4096-8192 | GPU 型号决定 |
| `data_parallel_size` | 2+ | 扩展吞吐 |
| `enable_chunked_prefill` | True | 避免长 prompt 阻塞 |

### 9.5 采样策略

```python
# 创意写作 → 高温度
SamplingParams(temperature=0.8, top_p=0.95, top_k=50)

# 代码生成 → 低温度
SamplingParams(temperature=0.2, top_p=0.95)

# 事实问答 → 贪婪
SamplingParams(temperature=0, top_k=1)  # greedy

# 精确 JSON → 结构化输出
SamplingParams(
    temperature=0,
    structured_outputs=StructuredOutputsParams(json_schema=...)
)
```

### 9.6 健康监控

```bash
# 健康检查
curl http://localhost:8000/health

# Prometheus 指标
curl http://localhost:8000/metrics

# 关键指标
# - vllm:time_to_first_tokens_seconds: TTFT 分布
# - vllm:time_per_output_token_seconds: TPOT（每 token 延迟）
# - vllm:num_requests_waiting: 等待队列长度
# - vllm:num_requests_running: 运行请求数
# - vllm:gpu_cache_usage_perc: KV Cache 使用率
# - vllm:request_success_total: 成功请求数
```

### 9.7 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| OOM | KV Cache 过大 | 降低 `max_model_len` 或 `max_num_seqs` |
| 高 TTFT | 长 prompt 阻塞 | 启用 chunked prefill |
| 吞吐低 | batch size 太小 | 增大 `max_num_batched_tokens` |
| 精度异常 | 量化失真 | 切换到更高精度量化或 BF16 |
| 网络超时 | 生成时间长 | 调整 `VLLM_ENGINE_ITERATION_TIMEOUT_S` |

### 9.8 安全建议

1. **API Key 认证**：设置 `VLLM_API_KEY` 环境变量
2. **输入验证**：限制 `max_tokens` 和 `max_model_len`
3. **网络隔离**：生产环境使用内部网络
4. **trust_remote_code**：生产环境设为 `False`
5. **日志脱敏**：避免记录完整 prompt/output

---

## 10. 环境变量参考

### 关键环境变量

```bash
# === 核心 ===
VLLM_TARGET_DEVICE=cuda                       # 目标设备: cuda/rocm/cpu
VLLM_USE_V2_MODEL_RUNNER=1                    # 使用 V2 Model Runner（默认）

# === 注意力 ===
VLLM_USE_FLASHINFER_SAMPLER=0                 # 使用 FlashInfer 采样器
VLLM_USE_TRITON_FLASH_ATTN=0                  # 使用 Triton Flash Attention
VLLM_USE_FLASH_MLA=0                          # 使用 FlashMLA
VLLM_USE_CUDAGRAPH_SIZES="1,2,4,8"           # CUDA Graph capture sizes

# === MoE ===
VLLM_FUSED_MOE_CHUNK_SIZE=65536               # Fused MoE chunk 大小
VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1   # MoE activation chunking
VLLM_USE_DEEP_GEMM=0                          # 使用 DeepGemm（DeepSeek）

# === 分布式 ===
VLLM_PP_LAYER_PARTITION=""                    # 自定义 PP 层分配
VLLM_WORKER_MULTIPROC_METHOD=spawn            # 多进程启动方法
VLLM_DP_RANK=""                               # DP rank

# === 日志 ===
VLLM_LOGGING_LEVEL=INFO                       # 日志级别
VLLM_LOG_STATS_INTERVAL=10                    # 统计日志间隔（秒）

# === 多模态 ===
VLLM_IMAGE_FETCH_TIMEOUT=5                    # 图片获取超时
VLLM_VIDEO_FETCH_TIMEOUT=30                   # 视频获取超时
VLLM_AUDIO_FETCH_TIMEOUT=10                   # 音频获取超时
VLLM_MEDIA_LOADING_THREAD_COUNT=8             # 媒体加载线程数

# === LoRA ===
VLLM_ALLOW_RUNTIME_LORA_UPDATING=0           # 允许运行时更新 LoRA

# === 结构化输出 ===
VLLM_XGRAMMAR_CACHE_MB=0                      # xgrammar 缓存大小

# === 引擎 ===
VLLM_ENGINE_ITERATION_TIMEOUT_S=60            # 引擎迭代超时
VLLM_ENGINE_READY_TIMEOUT_S=600               # 引擎就绪超时

# === API ===
VLLM_API_KEY=""                               # API 认证密钥
VLLM_SERVER_DEV_MODE=0                        # 开发模式

# === 缓存 ===
VLLM_CACHE_ROOT=~/.cache/vllm                 # 缓存根目录
VLLM_CONFIG_ROOT=~/.config/vllm               # 配置根目录

# === 模型重定向 ===
VLLM_MODEL_REDIRECT_PATH=""                   # 模型路径重定向文件

# === ROCm 专用 ===
VLLM_ROCM_USE_AITER=0                         # 使用 AITER
VLLM_ROCM_USE_AITER_UNIFIED_ATTN=0            # 使用 AITER 统一注意力
VLLM_ROCM_CUSTOM_PAGED_ATTN=0                 # 自定义 paged attention
```

### 完整的 VLLM_ 环境变量

vLLM 有 **232 个** `VLLM_*` 环境变量。查看完整列表：
```bash
python -c "from vllm.envs import *; [print(k) for k in dir() if k.startswith('VLLM_')]"
```

---

## 11. 架构图

### 11.1 系统架构全景图

```
                             ┌──────────────────────┐
                             │    Client / Users     │
                             └──────────┬───────────┘
                                        │ HTTP / gRPC / MCP
                                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          API Server (FastAPI)                               │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ Chat Completions │  │ Completions  │  │ Embeddings   │  │ Tokenization │  │
│  │ /v1/chat/...     │  │ /v1/...      │  │ /v1/...      │  │ /v1/tokenize │  │
│  └─────────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │
│                                                                             │
│  Middleware: CORS, Auth (API Key), X-Request-Id, Scaling, Response Logging │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                          Engine Frontend                                    │
│                                                                             │
│  ┌────────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │    AsyncLLM / LLMEngine        │  │    InputProcessor                │  │
│  │  ┌────────────────────────┐    │  │  - Tokenization                 │  │
│  │  │ OutputProcessor        │    │  │  - Multimodal preprocessing     │  │
│  │  │ - Detokenizer          │    │  │  - EngineCoreRequest 构造        │  │
│  │  │ - LogprobsProcessor    │    │  │  - Request ID 分配               │  │
│  │  │ - RequestOutputCollector│   │  └──────────────────────────────────┘  │
│  │  └────────────────────────┘    │                                        │
│  └────────────────────────────────┘                                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      EngineCoreClient                                 │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐   │  │
│  │  │ Inproc   │  │SyncMP    │  │AsyncMP   │  │ DPAsyncMP /       │   │  │
│  │  │ (LLM)    │  │ (ZMQ)    │  │ (ZMQ)    │  │ DPLBAsyncMP (DP)  │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └───────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │ ZMQ / In-Process
┌───────────────────────────────────▼───────────────────────────────────────┐
│                           EngineCore                                        │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐  │
│  │       Scheduler          │    │         KV Cache Manager              │  │
│  │  - schedule_default()    │    │  - allocate_slots()                  │  │
│  │  - schedule_split_pd()   │    │  - free() / cache_blocks()           │  │
│  │  - Chunked Prefill       │    │  - get_computed_blocks() (prefix)    │  │
│  │  - Prefix Cache 查询     │    │  - Block Table 管理                  │  │
│  │  - Preemption 决策       │    │  - CPU Offload                       │  │
│  │  - request_queue (FCFS/  │    │  - KV Transfer/Events                │  │
│  │    Priority)             │    │                                       │  │
│  └───────────┬──────────────┘    └──────────────────┬───────────────────┘  │
│              │                                      │                      │
│              ▼                                      ▼                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          Executor                                     │  │
│  │  ┌────────────────┐ ┌────────────────┐ ┌─────────────────────────┐  │  │
│  │  │MultiprocExecutor│ │RayDistributedEx│ │    UniProcExecutor      │  │  │
│  │  │ (default)      │ │ (Ray cluster) │ │    (single process)     │  │  │
│  │  └───────┬────────┘ └───────┬────────┘ └───────────┬─────────────┘  │  │
│  └──────────┼──────────────────┼───────────────────────┼──────────────┘  │
└─────────────┼──────────────────┼───────────────────────┼────────────────┘
              │                  │                       │
              ▼                  ▼                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          Worker Layer                                       │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐ │
│  │   GPUWorker (CUDA)   │  │  XPUWorker (Intel)   │  │ CPUWorker        │ │
│  │  ┌────────────────┐  │  │                      │  │                  │ │
│  │  │ GPUModelRunner │  │  │                      │  │                  │ │
│  │  │ - InputBatch   │  │  │                      │  │                  │ │
│  │  │ - Model Forward│  │  │                      │  │                  │ │
│  │  │ - Sampler      │  │  │                      │  │                  │ │
│  │  │ - CudagraphMgr │  │  │                      │  │                  │ │
│  │  └────────────────┘  │  │                      │  │                  │ │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

### 11.2 模型执行流水线

```
SchedulerOutput
      │
      ▼
┌─────────────────┐
│ Request State    │  finish / free / add / update
│ Management       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ InputBatch       │  token_ids, positions, seq_lens,
│ Preparation      │  slot_mapping, block_tables, attn_metadata
└────────┬────────┘
         ▼
┌─────────────────┐
│ Multimodal       │  图像/音频/视频编码（如有）
│ Encoding         │
└────────┬────────┘
         ▼
┌─────────────────┐
│ CUDA Graph /     │
│ Eager Dispatch   │  选择 CUDA Graph 或 Eager 执行
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────┐
│              Model Forward                   │
│                                              │
│  Embedding                                   │
│      │                                       │
│      ▼                                       │
│  ┌──────────────────────┐                    │
│  │  Transformer Layer 0  │                    │
│  │  ├── Attention        │                    │
│  │  │   ├── Q/K/V Proj   │                    │
│  │  │   ├── RoPE         │                    │
│  │  │   ├── PagedAttention│                   │
│  │  │   └── O Proj       │                    │
│  │  ├── Residual + Norm  │                    │
│  │  ├── FFN / MoE        │                    │
│  │  └── Residual + Norm  │                    │
│  └──────────────────────┘                    │
│      │                                       │
│      ▼                                       │
│  ... (repeat for N layers)                   │
│      │                                       │
│      ▼                                       │
│  ┌──────────────────────┐                    │
│  │  Final LayerNorm      │                    │
│  └──────────────────────┘                    │
│      │                                       │
│      ▼                                       │
│  ┌──────────────────────┐                    │
│  │  LM Head (logits)     │                    │
│  └──────────────────────┘                    │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│              Sampling (GPU)                  │
│                                              │
│  Logits → Bias → Penalties → Temperature →  │
│  Min-P → Top-K/Top-P → Gumbel → Token IDs   │
└────────────────────┬────────────────────────┘
                     ▼
              ModelRunnerOutput
```

### 11.3 分布式拓扑

```
┌────────────────────────────────────────────────────────────────┐
│                     TP（张量并行）                               │
│                                                                 │
│   GPU 0          GPU 1          GPU 2          GPU 3           │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐         │
│  │ Q[0:8] │    │ Q[8:16]│    │ Q[16:24]│   │ Q[24:32]│         │
│  │ K[0:8] │    │ K[8:16]│    │ K[16:24]│   │ K[24:32]│         │
│  │ V[0:8] │    │ V[8:16]│    │ V[16:24]│   │ V[24:32]│         │
│  └───┬────┘    └───┬────┘    └───┬────┘    └───┬────┘         │
│      │ all_reduce   │            │            │                │
│      └──────────────┴────────────┴────────────┘                │
│                      │                                         │
│              同步后的激活值                                      │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                     PP（流水线并行）                             │
│                                                                 │
│  Stage 0 (GPU 0-3)    Stage 1 (GPU 4-7)    Stage 2 (GPU 8-11)  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Layers 0-19  │───▶│ Layers 20-39 │───▶│ Layers 40-59 │       │
│  │ TP=4         │    │ TP=4         │    │ TP=4         │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│       P2P send/recv      P2P send/recv                          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                     DP（数据并行）                               │
│                                                                 │
│              ┌─────────────┐                                    │
│              │ DPCoordinator│  ← 负载均衡协调                     │
│              └──┬───┬───┬──┘                                    │
│                 │   │   │                                       │
│      ┌──────────▼───▼───▼──────────┐                            │
│      │         Load Balancer       │                            │
│      └──┬──────────────┬───────────┘                            │
│         │              │                                        │
│         ▼              ▼                                        │
│  ┌────────────┐ ┌────────────┐                                  │
│  │ Engine 0   │ │ Engine 1   │                                  │
│  │ (TP=4)     │ │ (TP=4)     │                                  │
│  │ Requests   │ │ Requests   │                                  │
│  │ [A,B,C]    │ │ [D,E,F]    │                                  │
│  └────────────┘ └────────────┘                                  │
└────────────────────────────────────────────────────────────────┘
```

### 11.4 请求状态机

```
                    ┌─────────┐
                    │ WAITING │  ← 新请求进入等待队列
                    └────┬────┘
                         │ schedule() 分配 KV Cache
                         ▼
                    ┌─────────┐
              ┌─────│ RUNNING │─────┐
              │     └────┬────┘     │
              │          │          │
              │          ▼          │
              │  ┌───────────┐      │
              │  │ DECODING  │      │
              │  └─────┬─────┘      │
              │        │            │
              │        ▼            │
              │  finish_reason      │ preempt (KV Cache 不足)
              │  = STOP/LENGTH      │
              │        │            │
              ▼        │            ▼
        ┌─────────┐    │      ┌───────────┐
        │FINISHED │    │      │ PREEMPTED  │
        │(STOP)   │    │      │ (重新排队) │
        └─────────┘    │      └─────┬──────┘
                       │            │
                       ▼            ▼
                  ┌─────────────────────┐
                  │   FINISHED (LENGTH) │
                  │   或 FREE (ABORT)   │
                  └─────────────────────┘
```

---

## 附录

### A. 模型支持列表（部分重点架构）

| 模型家族 | 文件 | 特性 |
|---------|------|------|
| **Qwen 3** | `qwen3.py`, `qwen3_moe.py` | MoE, MTP, MLA, 多模态 |
| **Llama 4** | `llama4.py` | FP8, RoPE, Eagle |
| **DeepSeek-V3** | `deepseek_v3.py` | MLA, MoE, MTP |
| **GLM 4** | `glm4.py`, `glm4_moe.py` | MoE, MTP, 多模态 |
| **Gemma 3** | `gemma3.py`, `gemma3n.py` | 多模态 |
| **Mistral Large 3** | `mistral_large_3.py` | Eagle |
| **Phi 4** | `phi4mm.py` | 多模态 |
| **Nemotron** | `nemotron_h.py`, `nemotron_nas.py` | NAS, 多模态 |
| **Command R** | `commandr.py` | Cohere |
| **Mamba 2** | `mamba2.py` | SSM（非 Transformer） |
| **Whisper** | `whisper.py` | ASR |
| **BERT** | `bert.py` | Embedding/Reranking |

### B. 常用量化方案对比

| 方案 | 精度 | 显存节省 | 速度 | 适用 GPU |
|------|------|---------|------|---------|
| BF16/FP16 | 无损 | 0% | 基准 | 全部 |
| FP8 W8A8 | 微损 | 50% | ↑20-40% | H100/H200/B200 |
| GPTQ INT4 | 小损 | 75% | ↑30-50% | 全部 |
| AWQ INT4 | 小损 | 75% | ↑30-50% | 全部 |
| GPTQ-Marlin INT4 | 小损 | 75% | ↑50-70% | 全部 |
| GGUF Q4_K_M | 中损 | 75% | 取决于实现 | CPU/GPU |

### C. 启动参数快速参考

```bash
vllm serve <model> [OPTIONS]

# === 模型 ===
--model <path>                # 模型路径或 HuggingFace 名称
--tokenizer <path>            # Tokenizer 路径
--dtype <auto|float16|bfloat16|float32>
--max-model-len <int>         # 最大模型长度
--quantization <method>       # 量化方法

# === 并行 ===
--tensor-parallel-size <int>  # 张量并行度（默认 1）
--pipeline-parallel-size <int># 流水线并行度（默认 1）
--data-parallel-size <int>    # 数据并行度（默认 1）
--enable-expert-parallel      # 启用专家并行

# === 调度 ===
--max-num-batched-tokens <int># 每步最大 token 数（默认 2048）
--max-num-seqs <int>          # 最大并发序列（默认 128）
--enable-chunked-prefill      # 启用分块预填充
--scheduler-policy <policy>   # fcfs 或 priority

# === KV Cache ===
--gpu-memory-utilization <float> # GPU 显存利用率（默认 0.9）
--cpu-offload-gb <float>      # CPU 卸载（GB）
--block-size <int>            # Block 大小
--enable-prefix-caching       # 前缀缓存

# === Attention ===
--attention-backend <backend> # 注意力后端

# === 投机解码 ===
--speculative-model <path>    # Draft model 路径
--num-speculative-tokens <int># 投机 token 数

# === 编译 ===
--compilation-mode <mode>     # NONE/DYNAMO_TRACE_ONCE/VLLM_COMPILE
--cudagraph-mode <mode>       # NONE/PIECEWISE/FULL

# === LoRA ===
--enable-lora                 # 启用 LoRA
--max-lora-rank <int>         # 最大 LoRA rank
--max-loras <int>             # 最大并发 LoRA adapter

# === 服务 ===
--host <host>                 # 服务地址（默认 0.0.0.0）
--port <port>                 # 服务端口（默认 8000）
--uvicorn-log-level <level>   # Uvicorn 日志级别
--api-key <key>               # API 认证密钥

# === 可观测 ===
--otlp-traces-endpoint <url>  # OTLP traces 端点
--enable-kv-cache-metrics     # 启用 KV Cache 指标
```

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-06-19  
> **适用场景**：生产环境部署、性能调优、架构理解
