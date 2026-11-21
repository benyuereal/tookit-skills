# vLLM V1 调度器详解

> **适用版本**: vLLM v0.15.x+ (V1 Engine)
> **核心文件**: `vllm/v1/core/sched/scheduler.py` (约2740行)
> **最后更新**: 2026-06-29

---

## 目录

1. [调度器概述](#1-调度器概述)
2. [调度器接口 (SchedulerInterface)](#2-调度器接口-schedulerinterface)
3. [请求生命周期与状态机](#3-请求生命周期与状态机)
4. [请求队列 (RequestQueue)](#4-请求队列-requestqueue)
5. [调度器数据结构](#5-调度器数据结构)
6. [核心调度器实现 (Scheduler)](#6-核心调度器实现-scheduler)
   - [6.1 初始化](#61-初始化)
   - [6.2 schedule() 调度分发器](#62-schedule-调度分发器)
   - [6.3 schedule_default() 默认调度算法](#63-schedule_default-默认调度算法)
   - [6.4 schedule_split_pd() PD分离调度算法](#64-schedule_split_pd-pd分离调度算法)
   - [6.5 抢占机制](#65-抢占机制-preempt_request)
   - [6.6 update_from_output() 输出处理](#66-update_from_output-输出处理)
   - [6.7 前缀缓存交互](#67-前缀缓存交互)
   - [6.8 编码器输入调度](#68-编码器输入调度)
   - [6.9 其他关键方法](#69-其他关键方法)
7. [异步调度器 (AsyncScheduler)](#7-异步调度器-asyncscheduler)
8. [停止条件检查 (check_stop)](#8-停止条件检查-check_stop)
9. [完整调度流程图](#9-完整调度流程图)
10. [配置参数参考 (SchedulerConfig)](#10-配置参数参考-schedulerconfig)
11. [最佳实践与调优建议](#11-最佳实践与调优建议)

---

## 1. 调度器概述

### 1.1 在 EngineCore 中的角色

调度器（Scheduler）是 vLLM V1 引擎的核心组件，位于 `EngineCore` 的忙碌循环中，负责在每个前向传播步骤之前决定**哪些请求**、**各处理多少 token**。调度器和 ModelRunner 构成引擎的核心循环：

```
+-------------------+          +-------------------+          +-------------------+
|                   |  request |                   | schedule |                   |
|   Frontend        | -------> |   EngineCore      | -------> |   Scheduler       |
|   (API Server)    |          |   (Busy Loop)     |          |                   |
|                   | <------- |                   | <------- |                   |
+-------------------+  output  +-------------------+  output  +-------------------+
                                                              |         ^
                                                              | schedule| output
                                                              v         |
                                                           +-------------------+
                                                           |   ModelRunner     |
                                                           |   (GPU Worker)    |
                                                           +-------------------+
```

**忙碌循环伪代码**（每次前向传播执行一次）：

```python
while True:
    scheduler_output = scheduler.schedule()       # 步骤1: 调度决策
    model_output = model_runner.execute(scheduler_output)  # 步骤2: GPU 执行
    engine_outputs = scheduler.update_from_output(  # 步骤3: 处理输出
        scheduler_output, model_output
    )
    send_to_frontend(engine_outputs)              # 步骤4: 返回给前端
```

### 1.2 调度器职责

1. **Token 预算管理**: 每个 step 最多处理 `max_num_batched_tokens` 个 token
2. **序列预算管理**: 最多同时运行 `max_num_seqs` 个请求
3. **KV Cache 分配**: 通过 `KVCacheManager` 管理显存中的 KV 缓存块
4. **前缀缓存利用**: 检测和复用共享前缀，减少重复计算
5. **抢占与恢复**: 当显存不足时，抢占低优先级请求，释放 KV Cache
6. **分块预填充 (Chunked Prefill)**: 将长 prompt 拆分到多个 step 中处理
7. **投机解码协调**: 管理 draft token 的调度、验证和接受/拒绝
8. **多模态支持**: 调度视觉编码器等编码器输入
9. **结构化输出**: 与 FSM/语法管理器协作，约束输出格式
10. **PD 分离**: 支持 Prefill/Decode 分离部署模式

### 1.3 文件组织结构

```
vllm/v1/core/sched/
    interface.py          # SchedulerInterface 抽象基类
    scheduler.py          # Scheduler 核心实现 (约2740行)
    async_scheduler.py    # AsyncScheduler (异步调度)
    output.py             # SchedulerOutput, NewRequestData, CachedRequestData
    request_queue.py      # FCFSRequestQueue, PriorityRequestQueue
    utils.py              # check_stop(), remove_all()
```

---

## 2. 调度器接口 (SchedulerInterface)

`SchedulerInterface` 定义在 `vllm/v1/core/sched/interface.py`，是调度器的抽象合约。

### 2.1 初始化签名

```python
class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: VllmConfig,           # 完整 vLLM 配置
        kv_cache_config: KVCacheConfig,    # KV 缓存配置
        structured_output_manager: StructuredOutputManager,
        block_size: int,                   # KV 缓存块大小 (通常16)
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,# 是否在输出中包含已完成集合
        log_stats: bool = False,           # 是否记录统计信息
    ) -> None: ...
```

### 2.2 核心方法一览

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `schedule()` | `SchedulerOutput` | 主调度循环，每次前向传播前调用 |
| `update_from_output(scheduler_output, model_runner_output)` | `dict[int, EngineCoreOutputs]` | 处理模型输出，更新请求状态 |
| `update_draft_token_ids(draft_token_ids)` | `None` | 更新投机解码的 draft token |
| `update_draft_token_ids_in_output(draft_token_ids, scheduler_output)` | `None` | 在 scheduler output 中更新 draft token |
| `add_request(request)` | `None` | 添加新请求到等待队列 |
| `finish_requests(request_ids, finished_status)` | `None` | 中止/结束请求 |
| `get_num_unfinished_requests()` | `int` | 未完成请求数 |
| `has_unfinished_requests()` | `bool` | 是否有未完成请求 |
| `has_finished_requests()` | `bool` | 是否有已完成请求待清理 |
| `reset_prefix_cache(reset_running_requests, reset_connector)` | `bool` | 重置前缀缓存（如模型热更新时） |
| `get_request_counts()` | `tuple[int, int]` | 返回 `(num_running, num_waiting)` |
| `make_stats()` | `Optional[SchedulerStats]` | 生成统计信息 |
| `shutdown()` | `None` | 关闭调度器 |
| `get_kv_connector()` | `Optional[KVConnectorBase_V1]` | 获取 KV 连接器（PD 分离） |
| `get_grammar_bitmask(scheduler_output)` | `GrammarOutput \| None` | 获取语法约束位掩码 |

### 2.3 schedule() 的调度语义

`schedule()` 方法的注释精确描述了调度语义：

> 调度决策在迭代级别做出。每个调度步骤对应模型的一次前向传播。调度器本质上是产生一个 `{req_id: num_tokens}` 字典，指定此步骤中每个请求要处理多少 token。
>
> - 对于新请求，`num_tokens` 可以大到 prompt token 数量
> - 对于自回归生成中的请求，`num_tokens` 可以是 1
> - 对于分块预填充、前缀缓存、投机解码等，`num_tokens` 介于两者之间

---

## 3. 请求生命周期与状态机

### 3.1 Request 对象

`Request` 类定义在 `vllm/v1/request.py`，代表一个推理请求的完整状态。

#### 关键字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `request_id` | `str` | 唯一请求标识 |
| `client_index` | `int` | 客户端索引（用于路由输出） |
| `priority` | `int` | 优先级（越小越优先） |
| `sampling_params` | `SamplingParams \| None` | 采样参数 |
| `pooling_params` | `PoolingParams \| None` | 池化参数 |
| `eos_token_id` | `int \| None` | EOS token ID |
| `lora_request` | `LoRARequest \| None` | LoRA 适配器请求 |
| `structured_output_request` | `StructuredOutputRequest \| None` | 结构化输出请求 |
| `arrival_time` | `float` | 到达时间戳 |
| `status` | `RequestStatus` | 当前状态 |
| `events` | `list[EngineCoreEvent]` | 事件日志 |
| `stop_reason` | `int \| str \| None` | 停止原因 |
| `max_tokens` | `int` | 最大输出 token 数 |
| `prompt_token_ids` | `list[int] \| None` | Prompt token ID 列表 |
| `prompt_embeds` | `torch.Tensor \| None` | Prompt 嵌入 |
| `num_prompt_tokens` | `int` | Prompt token 数量 |
| `_output_token_ids` | `list[int]` | 已生成的输出 token |
| `_all_token_ids` | `list[int]` | 全部 token (prompt + output) |
| `num_output_placeholders` | `int` | 异步调度中的输出占位符数量 |
| `spec_token_ids` | `list[int]` | 投机解码 draft token |
| `num_computed_tokens` | `int` | 已计算 token 数（含前缀缓存命中） |
| `num_cached_tokens` | `int` | 前缀缓存命中数 |
| `num_preemptions` | `int` | 被抢占次数 |
| `num_external_computed_tokens` | `int` | 远程计算的 token 数（PD 分离） |
| `block_hashes` | `list[BlockHash]` | 块哈希（用于前缀缓存） |
| `mm_features` | `list[MultiModalFeatureSpec]` | 多模态特征规格 |
| `resumable` | `bool` | 是否可恢复（流式请求） |
| `streaming_queue` | `deque[StreamingUpdate \| None] \| None` | 流式更新队列 |

#### 关键属性

```python
@property
def num_tokens(self) -> int:
    """prompt token + output token 总数"""
    return len(self._all_token_ids)

@property
def num_tokens_with_spec(self) -> int:
    """含投机 token 的总数"""
    return len(self._all_token_ids) + len(self.spec_token_ids)

@property
def num_output_tokens(self) -> int:
    """已生成的输出 token 数"""
    return len(self._output_token_ids)
```

#### 优先级比较（用于 PriorityRequestQueue）

```python
def __lt__(self, other: "Request") -> bool:
    """比较规则: priority → arrival_time → request_id → id(object)"""
    if self.priority != other.priority:
        return self.priority < other.priority
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time
    if self.request_id != other.request_id:
        return self.request_id < other.request_id
    return id(self) < id(other)
```

### 3.2 RequestStatus 枚举

定义在 `vllm/v1/request.py:285`。

```python
class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()                # 等待调度
    WAITING_FOR_FSM = enum.auto()        # 等待 FSM 语法编译
    WAITING_FOR_REMOTE_KVS = enum.auto() # 等待远程 KV 传输 (PD分离)
    WAITING_FOR_STREAMING_REQ = enum.auto() # 等待下一个流式输入块
    RUNNING = enum.auto()                # 正在执行
    PREEMPTED = enum.auto()              # 已被抢占
    # --- 以下为终态 (status > PREEMPTED) ---
    FINISHED_STOPPED = enum.auto()       # 正常停止 (EOS/stop token)
    FINISHED_LENGTH_CAPPED = enum.auto() # 达到长度上限
    FINISHED_ABORTED = enum.auto()       # 被中止
    FINISHED_IGNORED = enum.auto()       # prompt超长被忽略
    FINISHED_ERROR = enum.auto()         # 错误终止
```

判断完成状态的方法：

```python
@staticmethod
def is_finished(status: "RequestStatus") -> bool:
    return status > RequestStatus.PREEMPTED
```

### 3.3 请求状态机

```
                               +---------------------+
                               |  add_request()      |
                               |  (新请求到达)         |
                               +----------+----------+
                                          |
                                          v
                               +---------------------+
                         +---->|       WAITING       |
                         |     +----------+----------+
                         |                |
                         |   structured   | 需要结构化输出
                         |   output?       v
                         |     +---------------------+
                         |     |   WAITING_FOR_FSM   |--+
                         |     +---------------------+  | FSM 编译完成
                         |                |              v
                         |   remote KV?   |    +--------+
                         |     v          |    |
                         |  +---------------------+
                         |  | WAITING_FOR_REMOTE  |--+
                         |  |      _KVS           |  | KV 传输完成
                         |  +---------------------+  |
                         |                |           v
                         |   streaming?   |    +------+
                         |     v          |    |
                         |  +---------------------+
                         |  | WAITING_FOR_        |--+
                         |  | STREAMING_REQ       |  | 下一个 chunk 到达
                         |  +---------------------+  |
                         |                |           v
                         |                |    +------+
                         |   schedule()   |    |
                         |   KV cache 可分配 |
                         |                v
                         |     +---------------------+
                         |     |       RUNNING       |
                         |     +----------+----------+
                         |                |
                         |   KV cache     |  显存不足
                         |   分配失败      v
                         |     +---------------------+
                         +-----|     PREEMPTED       |
                               +---------------------+
                                        |
                                        | 下次调度恢复
                                        v
                               +--------+
                               | WAITING |
                               +----+----+
                                    |
                              生成完成/停止条件触发
                                    |
                                    v
                    +---------------+---------------+
                    |               |               |
                    v               v               v
          FINISHED_STOPPED  FINISHED_LENGTH   FINISHED_ABORTED
          (EOS/stop token)     _CAPPED        (客户端断开)
                             (达到max_tokens)

          FINISHED_IGNORED  FINISHED_ERROR
          (prompt超长)      (异常错误)
```

**状态转换规则**：

| 当前状态 | 触发条件 | 下一状态 |
|----------|----------|----------|
| (新建) | `add_request()` | `WAITING` |
| `WAITING` | 有 `structured_output_request` | `WAITING_FOR_FSM` |
| `WAITING_FOR_FSM` | FSM 语法编译完成 | `WAITING` |
| `WAITING` | 远程 KV 需要加载 | `WAITING_FOR_REMOTE_KVS` |
| `WAITING_FOR_REMOTE_KVS` | KV 传输完成 | `WAITING` 或 `PREEMPTED` |
| `WAITING` | 流式请求等待下一个 chunk | `WAITING_FOR_STREAMING_REQ` |
| `WAITING` | `schedule()` 成功分配 KV cache | `RUNNING` |
| `RUNNING` | KV cache 分配失败，被抢占 | `PREEMPTED` |
| `PREEMPTED` | 下次 `schedule()` 成功 | `RUNNING` |
| `RUNNING` | `check_stop()` 返回 True | 对应 `FINISHED_*` |
| 任意非终态 | `finish_requests()` | `FINISHED_ABORTED` |

---

## 4. 请求队列 (RequestQueue)

定义在 `vllm/v1/core/sched/request_queue.py`。

### 4.1 架构

```
                    RequestQueue (ABC)
                    /                \
                   /                  \
       FCFSRequestQueue       PriorityRequestQueue
       (deque[Request])          (heapq)
```

### 4.2 FCFSRequestQueue

基于 Python `deque`，严格先进先出：

```python
class FCFSRequestQueue(deque[Request], RequestQueue):
    def add_request(self, request: Request) -> None:
        self.append(request)          # 添加到队列尾部

    def pop_request(self) -> Request:
        return self.popleft()         # 从队列头部取出

    def prepend_request(self, request: Request) -> None:
        self.appendleft(request)      # 添加回队列头部（用于抢占恢复）
```

**适用场景**: 所有请求权重相同的场景，保证请求按到达顺序处理。

### 4.3 PriorityRequestQueue

基于 Python `heapq`，按优先级排序：

```python
class PriorityRequestQueue(RequestQueue):
    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, request)  # 按 Request.__lt__ 排序

    def pop_request(self) -> Request:
        return heapq.heappop(self._heap)
```

排序规则由 `Request.__lt__` 定义：**priority 小 → arrival_time 早 → request_id 字典序小**。

**适用场景**: 需要区分请求优先级的场景，如在线服务中 VIP 用户优先。

### 4.4 工厂方法

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"

def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
```

### 4.5 抢占时的队列操作

抢占时，请求被放回队列**头部**：

```python
# FCFS: 被抢占的请求优先于其他等待请求
self.waiting.prepend_request(request)  # → appendleft()

# Priority: prepend_request 等价于 add_request (按优先级排序)
self.waiting.prepend_request(request)  # → add_request() (heappush)
```

---

## 5. 调度器数据结构

定义在 `vllm/v1/core/sched/output.py`。

### 5.1 NewRequestData

首次被调度的请求的完整数据：

```python
@dataclass
class NewRequestData:
    req_id: str                                    # 请求 ID
    prompt_token_ids: list[int] | None             # Prompt token 列表
    mm_features: list[MultiModalFeatureSpec]       # 多模态特征
    sampling_params: SamplingParams | None         # 采样参数
    pooling_params: PoolingParams | None           # 池化参数
    block_ids: tuple[list[int], ...]               # 分配的 KV cache block ID
    num_computed_tokens: int                       # 已计算 token 数（含缓存命中）
    lora_request: LoRARequest | None               # LoRA 请求
    prompt_embeds: torch.Tensor | None = None      # Prompt 嵌入
    prefill_token_ids: list[int] | None = None     # 仅 v2 model runner 使用
```

**作用**: Worker 收到 `NewRequestData` 后，将其缓存到本地，后续 step 只需发送增量数据。

### 5.2 CachedRequestData

已调度过的请求的增量数据（减少通信开销）：

```python
@dataclass
class CachedRequestData:
    req_ids: list[str]                             # 请求 ID 列表
    resumed_req_ids: set[str]                      # 从抢占中恢复的请求 ID
    new_token_ids: list[list[int]]                 # 新 token (仅 PP 使用)
    all_token_ids: dict[str, list[int]]            # 上一步未调度的请求的全量 token
    new_block_ids: list[tuple[list[int], ...] | None]  # 新分配的 block ID
    num_computed_tokens: list[int]                 # 已计算 token 数
    num_output_tokens: list[int]                   # 输出 token 数
```

**关键区别**：
- `resumed_req_ids` 中的请求，`new_block_ids` **替换**已有 block IDs（因为被抢占后重新分配）
- 不在 `resumed_req_ids` 中的请求，`new_block_ids` **追加**到已有 block IDs

### 5.3 SchedulerOutput

调度器的输出，包含一步调度所需的所有信息：

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]       # 首次调度的请求
    scheduled_cached_reqs: CachedRequestData       # 已缓存请求的增量
    num_scheduled_tokens: dict[str, int]           # req_id → 本步调度 token 数
    total_num_scheduled_tokens: int                # 总调度 token 数
    scheduled_spec_decode_tokens: dict[str, list[int]]  # req_id → draft token
    scheduled_encoder_inputs: dict[str, list[int]]      # req_id → 编码器输入索引
    num_common_prefix_blocks: list[int]            # 各 KV cache group 的公共前缀块数
    finished_req_ids: set[str]                     # 上一步到这一步之间完成的请求
    free_encoder_mm_hashes: list[str]              # 要释放的编码器缓存 hash
    preempted_req_ids: set[str] | None             # 本步被抢占的请求
    has_structured_output_requests: bool           # 是否有结构化输出请求
    pending_structured_output_tokens: bool         # 是否有待处理的 SO token
    num_invalid_spec_tokens: dict[str, int] | None # 无效投机 token 数
    kv_connector_metadata: KVConnectorMetadata | None    # KV 连接器元数据
    ec_connector_metadata: ECConnectorMetadata | None    # EC 连接器元数据
```

### 5.4 GrammarOutput

```python
@dataclass
class GrammarOutput:
    structured_output_request_ids: list[str]         # SO 请求 ID 列表
    grammar_bitmask: npt.NDArray[np.int32]           # 位掩码矩阵
```

位掩码是一个二维数组，每一行对应一个 SO 请求，列表示词表中每个 token 是否被语法允许。

---

## 6. 核心调度器实现 (Scheduler)

`Scheduler` 类定义在 `vllm/v1/core/sched/scheduler.py:64`，继承自 `SchedulerInterface`，约 2740 行。

### 6.1 初始化

```python
def __init__(self, vllm_config, kv_cache_config, structured_output_manager,
             block_size, mm_registry, include_finished_set, log_stats):
```

#### 关键内部状态

| 属性 | 类型 | 说明 |
|------|------|------|
| `self.requests` | `dict[str, Request]` | 所有已知请求 (req_id → Request) |
| `self.waiting` | `RequestQueue` | 等待队列 |
| `self.running` | `list[Request]` | 运行队列（保持插入顺序） |
| `self.finished_req_ids` | `set[str]` | 本步完成的请求 ID（用于通知 worker 释放缓存） |
| `self.preempted_req_ids` | `set[str]` | 本步被抢占的请求 ID |
| `self.prev_step_scheduled_req_ids` | `set[str]` | 上一步调度的请求 ID |
| `self.kv_cache_manager` | `KVCacheManager` | KV 缓存管理器 |
| `self.encoder_cache_manager` | `EncoderCacheManager` | 编码器缓存管理器 |

#### 预算约束

| 属性 | 来源 | 说明 |
|------|------|------|
| `self.max_num_scheduled_tokens` | `scheduler_config.max_num_batched_tokens` | 每步最大 token 数 |
| `self.max_num_running_reqs` | `scheduler_config.max_num_seqs` | 最大并发请求数 |
| `self.max_model_len` | `model_config.max_model_len` | 模型最大序列长度 |
| `self.block_size` | 构造参数 | KV cache 块大小（通常 16） |

#### 投机解码配置

```python
speculative_config = vllm_config.speculative_config
self.use_eagle = False
self.num_spec_tokens = self.num_lookahead_tokens = 0
if speculative_config:
    self.num_spec_tokens = speculative_config.num_speculative_tokens
    if speculative_config.use_eagle():
        self.use_eagle = True
        self.num_lookahead_tokens = self.num_spec_tokens
    if speculative_config.uses_draft_model():
        self.num_lookahead_tokens = self.num_spec_tokens
```

#### KV Connector 初始化（PD 分离）

```python
self.connector = None
if self.vllm_config.kv_transfer_config is not None:
    self.connector = KVConnectorFactory.create_connector(
        config=self.vllm_config,
        role=KVConnectorRole.SCHEDULER,
        ...
    )
```

### 6.2 schedule() 调度分发器

```python
def schedule(self) -> SchedulerOutput:
    if envs.VLLM_USE_PD_SPLIT:
        if self.connector is not None:
            return self.schedule_default()   # 有 connector 时用默认
        if self.use_mla:
            if self.full_cuda_graph and self.num_spec_tokens > 0:
                return self.schedule_split_pd()  # MLA + CUDA Graph + 投机解码
            else:
                return self.schedule_default()
        else:
            return self.schedule_split_pd()  # 非 MLA 的 PD 分离
    else:
        return self.schedule_default()       # 非 PD 分离模式
```

**决策逻辑**：

```
                         VLLM_USE_PD_SPLIT?
                        /                \
                      No                  Yes
                       |                   |
              schedule_default()    self.connector is not None?
                                       /              \
                                     Yes               No
                                      |                 |
                              schedule_default()   self.use_mla?
                                                     /         \
                                                   Yes          No
                                                    |            |
                                          full_cuda_graph    schedule_split_pd()
                                          AND num_spec>0?
                                           /         \
                                         Yes          No
                                          |            |
                                  schedule_split_pd() schedule_default()
```

### 6.3 schedule_default() 默认调度算法

`schedule_default()` 是核心调度算法，分为 4 个阶段。

#### 阶段 0: 初始化

```python
def schedule_default(self) -> SchedulerOutput:
    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens       # token 预算
    encoder_compute_budget = self.max_num_encoder_input_tokens
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}
```

#### 阶段 1: 调度 RUNNING 请求

遍历 `self.running` 列表中的请求，为每个请求计算还需要处理多少新 token。

```
开始
  |
  v
req_index = 0
  |
  v
req_index < len(running) AND token_budget > 0? --No--> 进入阶段2
  |
 Yes
  |
  v
request = self.running[req_index]
  |
  v
(PP模式) num_output_placeholders > 0? --Yes--> req_index++, continue
  |
 No
  |
  v
(异步调度) 已达到 max_tokens? --Yes--> req_index++, continue
  |
 No
  |
  v
num_new_tokens = num_tokens_with_spec + num_output_placeholders
                 - num_computed_tokens
  |
  v
long_prefill_token_threshold 截断
  |
  v
min(num_new_tokens, token_budget, max_model_len-1-num_computed_tokens)
  |
  v
调度编码器输入 (_try_schedule_encoder_inputs)
  |
  v
Mamba 块对齐? (_mamba_block_aligned_split)
  |
  v
num_new_tokens == 0? --Yes--> req_index++, continue
  |
 No
  |
  v
+--------------------------------------------------+
|  KV Cache 分配循环:                               |
|  while True:                                      |
|    new_blocks = kv_cache_manager.allocate_slots() |
|    if new_blocks is not None: break               |
|                                                   |
|    # 分配失败, 尝试抢占                            |
|    if PRIORITY 策略:                               |
|      preempted_req = max(running, key=priority)   |
|    else (FCFS):                                   |
|      preempted_req = running.pop()                |
|                                                   |
|    _preempt_request(preempted_req)                |
|    if preempted_req == request: break             |
+--------------------------------------------------+
  |
  v
new_blocks is None? --Yes--> break (无法调度此请求)
  |
 No
  |
  v
记录调度: scheduled_running_reqs.append(request)
更新 token_budget, num_scheduled_tokens
处理 spec_token_ids
处理 encoder_inputs
req_index++
  |
  v
回到循环开始
```

**关键细节**：

- `num_new_tokens` 的计算考虑了投机 token 和输出占位符
- `long_prefill_token_threshold` 用于限制长 prompt 的单步 token 数
- 如果 `num_new_tokens == 0`，使用 `continue` 而非 `break`，允许后续请求被调度
- KV Cache 分配失败时，FCFS 策略弹出 `running` 列表末尾（最少 computed tokens 的请求），PRIORITY 策略选择最低优先级

**FCFS 抢占选择 vs PRIORITY 抢占选择**：

```python
# PRIORITY: 抢占优先级最低的（priority 值最大）
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time),
    )
    self.running.remove(preempted_req)

# FCFS: 抢占 running 列表末尾（即最先到达的，
# 但由于 running 列表按调度顺序排列，末尾的请求
# 通常 computed_tokens 最少）
else:
    preempted_req = self.running.pop()
```

如果被抢占的请求已经在本步被调度（在 `scheduled_running_reqs` 中），则需要回滚：
- 恢复 `token_budget`
- 移除 `num_scheduled_tokens` 中的条目
- 恢复 `encoder_compute_budget`

#### 阶段 2: 调度 WAITING 请求

```
开始
  |
  v
preempted_reqs 为空? --No--> 跳过阶段2
  |
 Yes
  |
  v
self.waiting 非空 AND token_budget > 0? --No--> 进入阶段3
  |
 Yes
  |
  v
len(running) == max_num_running_reqs? --Yes--> break
  |
 No
  |
  v
request = self.waiting.peek_request()
  |
  v
状态检查:
  - WAITING_FOR_REMOTE_KVS? → 检查是否 ready，否则跳过
  - WAITING_FOR_FSM? → 检查 FSM 是否编译完成，否则跳过
  - WAITING_FOR_STREAMING_REQ? → 跳过
  |
  v
LoRA 约束检查 (max_loras)
  |
  v
前缀缓存查询:
  if num_computed_tokens == 0:
      new_computed_blocks, num_new_local_computed_tokens =
          kv_cache_manager.get_computed_blocks(request)

      if connector:
          ext_tokens = connector.get_num_new_matched_tokens(...)
          num_external_computed_tokens = ext_tokens
          if ext_tokens is None: 跳过

      num_computed_tokens = local + external
  |
  v
load_kv_async? → num_new_tokens = 0 (等待远程KV)
  |
  v
num_new_tokens = request.num_tokens - num_computed_tokens
long_prefill_token_threshold 截断
chunked_prefill 未启用且 num_new_tokens > token_budget? → break
  |
  v
调度编码器输入
Mamba 块对齐
  |
  v
KV Cache 分配 (allocate_slots)
  |
  v
new_blocks is None? --Yes--> break
  |
  v
self.waiting.pop_request()
  |
  v
load_kv_async? --Yes--> 放入 skipped，状态改为 WAITING_FOR_REMOTE_KVS
  |
 No
  |
  v
self.running.append(request)
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
更新 token_budget
处理 encoder_inputs
  |
  v
回到循环开始
```

**关键细节**：

- 前缀缓存查询：`get_computed_blocks()` 返回本地 KV cache 中已计算好的 block 和 token 数
- 外部缓存：如果配置了 KVConnector，还会查询远程 KV cache
- `num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens`
- 新请求的 `num_cached_tokens` 仅在首次设置（`< 0` 时设置）
- 被跳过的请求通过 `skipped_waiting_requests` 收集，最后放回等待队列头部

#### 阶段 3: 编码器输入

参见 [6.8 编码器输入调度](#68-编码器输入调度)。

#### 阶段 4: 后处理

```python
# 1. 检查约束
total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
assert len(self.running) <= self.max_num_running_reqs

# 2. 计算公共前缀块
num_common_prefix_blocks = kv_cache_manager.get_num_common_prefix_blocks(...)

# 3. 构建 NewRequestData 列表
new_reqs_data = [NewRequestData.from_request(req, block_ids)
                 for req in scheduled_new_reqs]

# 4. 构建 CachedRequestData
cached_reqs_data = self._make_cached_request_data(...)

# 5. 构建 SchedulerOutput
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,
    scheduled_cached_reqs=cached_reqs_data,
    ...
)

# 6. 构建 connector metadata
if self.connector:
    scheduler_output.kv_connector_metadata = self.connector.build_connector_meta(...)

# 7. 调用 _update_after_schedule
self._update_after_schedule(scheduler_output)

return scheduler_output
```

### 6.4 schedule_split_pd() PD分离调度算法

`schedule_split_pd()` 与 `schedule_default()` 的核心区别是**调度顺序相反**：先调度 WAITING 请求（prefill），再调度 RUNNING 请求（decode）。

```
schedule_default():          schedule_split_pd():
  RUNNING 优先                 WAITING 优先
  1. 调度 RUNNING 请求         1. 调度 WAITING 请求
  2. 调度 WAITING 请求          2. 调度 RUNNING 请求
  3. 编码器输入                3. 编码器输入
  4. 后处理                   4. 后处理
```

**为什么这样做**：在 PD 分离部署中，prefill 节点和 decode 节点是分开的。在 prefill 节点上，应优先处理新请求（prefill），再处理正在进行的请求（decode）。这确保 prefill 获得足够的 token 预算。

**另一个区别**：`schedule_split_pd` 的 RUNNING 阶段仅在**没有新的 prefill 请求**时才执行：

```python
# 仅在没有新请求和恢复请求时才调度 RUNNING 请求
if not scheduled_new_reqs and not scheduled_resumed_reqs:
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        ...
```

### 6.5 抢占机制 (_preempt_request)

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    assert request.status == RequestStatus.RUNNING

    self.kv_cache_manager.free(request)        # 释放 KV cache 块
    self.encoder_cache_manager.free(request)   # 释放编码器缓存
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0            # 重置计算进度
    request.spec_token_ids.clear()             # 清除投机 token
    request.num_preemptions += 1               # 抢占计数 +1

    self.waiting.prepend_request(request)      # 放回等待队列头部
```

**抢占策略对比**：

| 策略 | 选择逻辑 | 抢占目标 | 优点 |
|------|----------|----------|------|
| FCFS | `self.running.pop()` | 最近调度的请求（computed tokens 最少） | 最小化浪费的计算 |
| PRIORITY | `max(running, key=lambda r: (r.priority, r.arrival_time))` | 优先级最低的请求 | 保护高优先级请求 |

**抢占回滚**：如果被抢占的请求已在本步被调度，需要：
1. 从 `scheduled_running_reqs` 中移除
2. 恢复 `token_budget`
3. 移除 `num_scheduled_tokens` 条目
4. 恢复 `encoder_compute_budget`
5. `req_index -= 1`（因为 `running` 列表少了一个元素）

### 6.6 update_from_output() 输出处理

`update_from_output()` 处理 ModelRunner 的输出，更新每个请求的状态并构建返回给前端的 `EngineCoreOutputs`。

```
update_from_output(scheduler_output, model_runner_output)
  |
  v
提取 model_runner_output:
  sampled_token_ids, logprobs, prompt_logprobs_dict,
  pooler_outputs, num_nans_in_logits, kv_connector_output
  |
  v
处理 KV load 失败 (invalid_block_ids)
  _handle_invalid_blocks() → failed_kv_load_req_ids
  |
  v
遍历 num_scheduled_tokens 中的每个 req_id:
  |
  +-- 获取 generated_token_ids
  |
  +-- 投机解码处理:
  |     if scheduled_spec_token_ids and generated_token_ids:
  |       num_draft = len(scheduled_spec_token_ids)
  |       num_accepted = len(generated_token_ids) - 1
  |       num_rejected = num_draft - num_accepted
  |       request.num_computed_tokens -= num_rejected  # 回退被拒绝的 token
  |
  +-- _update_request_with_output(request, new_token_ids):
  |     for each output_token in new_token_ids:
  |       request.append_output_token_ids(output_token)
  |       stopped = check_stop(request, max_model_len)
  |       if stopped: break
  |     return new_token_ids, stopped
  |
  +-- 检查停止:
  |     if stopped:
  |       finish_reason = request.get_finished_reason()
  |       _handle_stopped_request(request)
  |       _free_request(request) → KV cache 释放
  |
  +-- 提取 logprobs
  |
  +-- 更新结构化输出 FSM 状态
  |
  +-- 构建 EngineCoreOutput
  |
  v
清理运行队列和等待队列中已停止的请求
  |
  v
处理 KV connector output (finished_recving, finished_sending)
  |
  v
收集和发布 KV cache 事件
  |
  v
构建并返回 dict[client_index, EngineCoreOutputs]
```

**投机解码的 token 计数调整**：

```python
# 如果 draft 中有 token 被拒绝，需要回退 num_computed_tokens
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected

# 异步调度下同样调整 num_output_placeholders
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

**KV Load 失败处理** (`_handle_invalid_blocks`)：

```python
# 两种策略:
# 1. "recompute": 重新计算受影响的 token
# 2. "fail": 直接失败请求 (FINISHED_ERROR)
```

### 6.7 前缀缓存交互

#### get_computed_blocks()

在调度 WAITING 请求时，查询前缀缓存：

```python
if request.num_computed_tokens == 0:
    new_computed_blocks, num_new_local_computed_tokens = (
        self.kv_cache_manager.get_computed_blocks(request)
    )
```

`get_computed_blocks()` 内部：
1. 计算请求的 block hashes
2. 在 KV cache 中查找匹配的 block
3. 返回已缓存的 block 列表和对应的 token 数

#### allocate_slots()

为请求分配新的 KV cache 槽位：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,                    # 新 token 数
    num_new_computed_tokens=...,       # 本地已计算的 token 数
    new_computed_blocks=...,           # 前缀缓存命中的 block
    num_lookahead_tokens=...,          # 投机解码预留
    num_external_computed_tokens=...,  # 外部已计算的 token 数
    delay_cache_blocks=...,            # 是否延迟缓存
    num_encoder_tokens=...,            # 编码器 token 数
)
```

#### free() 和 cache_blocks()

```python
# 释放请求占用的所有 KV cache 块
self.kv_cache_manager.free(request)

# 缓存请求的指定范围 token 对应的 KV 块
self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
```

#### 重置前缀缓存

```python
def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
    if reset_running_requests:
        # 抢占所有 running 请求
        while self.running:
            request = self.running.pop()
            self._preempt_request(request, timestamp)
            request.num_output_placeholders = 0
            request.discard_latest_async_tokens = True
        self.prev_step_scheduled_req_ids.clear()

    # 清空 KV cache
    reset_successful = self.kv_cache_manager.reset_prefix_cache()

    if reset_connector:
        self.connector.reset_cache()
```

### 6.8 编码器输入调度

`_try_schedule_encoder_inputs()` 决定在当前 step 中需要调度哪些多模态编码器输入。

**调度条件**：
1. 编码器输入的输出 token 范围与当前 step 的计算范围重叠
2. 编码器输入尚未计算并缓存在本地
3. 编码器输入不在远程缓存中
4. 有足够的编码器 token 预算
5. 编码器缓存有空间存储

**编码器输入与文本 token 的交互**：

```
文本序列: [T0] [T1] [IMG_START] [IMG_EMBED_0] ... [IMG_EMBED_N] [IMG_END] [T2] ...
                            |<--- encoder input i --->|
                            start_pos              start_pos + num_encoder_tokens

计算范围: [num_computed_tokens ....................... num_computed_tokens + num_new_tokens)
```

当计算范围与 `[start_pos, start_pos + num_encoder_tokens)` 重叠时，编码器输入需要被调度。

**编码器预算管理**：

```python
# 如果编码器缓存或预算不足，调整 num_new_tokens
if not self.encoder_cache_manager.can_allocate(...):
    if num_computed_tokens < start_pos:
        # 只调度到编码器输入之前的文本 token
        num_new_tokens = start_pos - num_computed_tokens
    else:
        # 前缀缓存导致 computed_tokens > start_pos，无法调度
        num_new_tokens = 0
    break
```

**禁止分块多模态输入** (`disable_chunked_mm_input`)：

```python
if self.scheduler_config.disable_chunked_mm_input:
    # 如果本步只能覆盖部分多模态输入，回退到该输入之前
    if (num_computed_tokens < start_pos
        and (num_computed_tokens + num_new_tokens) < (start_pos + num_encoder_tokens)):
        num_new_tokens = start_pos - num_computed_tokens
        break
```

**外部编码器缓存 (ECConnector)**：

```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(item_identifier):
    # 编码器输入在远程缓存中，标记为外部加载
    external_load_encoder_input.append(i)
    num_embeds_to_schedule += num_encoder_embeds
    continue
```

### 6.9 其他关键方法

#### _update_after_schedule()

在 `schedule()` 末尾调用，推进 `num_computed_tokens`：

```python
def _update_after_schedule(self, scheduler_output):
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests[req_id]
        request.num_computed_tokens += num_scheduled_token  # 预推进

        if request.has_encoder_inputs:
            self._free_encoder_inputs(request)  # 释放不再需要的编码器输入

    self.finished_req_ids = set()  # 清空完成的请求 ID
```

**为什么预推进**：这样在下一个 step，chunked prefill 的请求可以立即被再次调度。

#### _make_cached_request_data()

构建 `CachedRequestData`，处理 running 和 resumed 请求：

```python
def _make_cached_request_data(self, running_reqs, resumed_reqs,
                               num_scheduled_tokens, spec_decode_tokens,
                               req_to_new_blocks):
    # 关键逻辑:
    # 1. PP 模式下发送 new_token_ids (因为 worker 间无直接通信)
    # 2. resumed_reqs 的 new_block_ids 替换已有 block IDs
    # 3. 未在上一步调度的请求发送完整 all_token_ids
```

#### add_request()

```python
def add_request(self, request: Request):
    existing = self.requests.get(request.request_id)
    if existing is not None:
        # 流式请求的后续 chunk
        update = StreamingUpdate.from_request(request)
        if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
            existing.streaming_queue.append(update)  # 入队
        elif update is not None:
            self._update_request_as_session(existing, update)  # 立即处理
    else:
        if request.resumable:
            request.streaming_queue = deque()
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
```

#### finish_requests()

```python
def finish_requests(self, request_ids, finished_status):
    # 分两遍处理:
    # 第一遍: 收集要从队列中移除的请求
    # 第二遍: 设置状态并释放资源

    # 从 running 队列移除
    if running_requests_to_remove:
        self.running = remove_all(self.running, running_requests_to_remove)
    # 从 waiting 队列移除
    if waiting_requests_to_remove:
        self.waiting.remove_requests(waiting_requests_to_remove)

    # 释放请求
    for request in valid_requests:
        request.status = finished_status
        self._free_request(request)
```

#### _update_request_as_session()

用于流式请求：当上一个 chunk 完成后，用新的 chunk 更新 session：

```python
def _update_request_as_session(self, session, update):
    # 保留已计算的输出 token 作为新的 prompt 前缀
    kept_output_tokens = session._all_token_ids[
        session.num_prompt_tokens : num_computed_tokens
    ]
    # 扩展 prompt
    session.prompt_token_ids.extend(kept_output_tokens)
    session.prompt_token_ids.extend(update.prompt_token_ids or ())
    session._all_token_ids.extend(update.prompt_token_ids or ())
    session.num_prompt_tokens = len(session.prompt_token_ids)
```

#### _handle_stopped_request()

处理已停止的请求，支持流式继续：

```python
def _handle_stopped_request(self, request) -> bool:
    if not request.resumable:
        return True  # 非流式: 真正完成

    if request.streaming_queue:
        update = request.streaming_queue.popleft()
        if update is None:
            return True  # 流式结束哨兵
        self._update_request_as_session(request, update)  # 继续下一个 chunk
    else:
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        self.num_waiting_for_streaming_input += 1

    self.waiting.add_request(request)
    return False  # 未真正完成，等待继续
```

#### _free_request()

```python
def _free_request(self, request):
    delay_free_blocks, kv_xfer_params = self._connector_finished(request)
    self.encoder_cache_manager.free(request)
    self.finished_req_ids.add(request.request_id)

    if not delay_free_blocks:
        self._free_blocks(request)  # 释放 KV cache 并删除请求

    return kv_xfer_params  # 传递给前端的 KV transfer 参数
```

---

## 7. 异步调度器 (AsyncScheduler)

`AsyncScheduler` 定义在 `vllm/v1/core/sched/async_scheduler.py`，继承自 `Scheduler`。

### 7.1 核心思想

异步调度通过**输出占位符 (output placeholders)** 让调度器可以提前调度未来步骤的计算，填补 GPU 空闲间隙，提高吞吐量。

```
同步调度:
  Step N:  调度 → 执行 → 等待输出 → 调度 Step N+1 → ...
                               ↑
                          GPU 空闲间隙

异步调度:
  Step N:  调度(含占位符) → 执行 → 输出到达时已预分配好 KV cache
  Step N+1: 调度(含占位符) → 执行 → ...
  (调度和输出处理可重叠)
```

### 7.2 _update_after_schedule() 重写

```python
def _update_after_schedule(self, scheduler_output):
    super()._update_after_schedule(scheduler_output)

    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))

        if (request.num_computed_tokens
            == request.num_tokens
            + request.num_output_placeholders
            + cur_num_spec_tokens):
            # 请求将在本步生成一个新 token + num_spec_tokens 个投机 token
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            request.spec_token_ids = [-1] * self.num_spec_tokens  # 占位符
```

**条件解读**：当 `num_computed_tokens == num_tokens + num_output_placeholders + cur_num_spec_tokens` 时，意味着所有 prompt token 和已有的输出占位符都已被计算，下一步将产生新的输出 token。

### 7.3 _update_request_with_output() 重写

```python
def _update_request_with_output(self, request, new_token_ids):
    if request.discard_latest_async_tokens:
        # 强制抢占（如 reset_prefix_cache）时丢弃异步 token
        request.discard_latest_async_tokens = False
        return [], False

    status_before_update = request.status
    new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)

    # 减少输出占位符计数
    request.num_output_placeholders -= len(new_token_ids)

    # 对 RUNNING 请求（非 preempted）缓存 KV blocks
    if status_before_update == RequestStatus.RUNNING:
        self.kv_cache_manager.cache_blocks(
            request, request.num_computed_tokens - request.num_output_placeholders
        )

    return new_token_ids, stopped
```

**关键差异**：

| 操作 | 同步调度 | 异步调度 |
|------|----------|----------|
| 输出占位符 | 不使用 | 提前增加 `num_output_placeholders` |
| 投机 token | 实际 token ID | 占位符 `[-1] * num_spec_tokens` |
| KV cache 缓存时机 | `_update_after_schedule` | `_update_request_with_output` 中 |
| 强制抢占 | N/A | 丢弃最新的异步 token |

### 7.4 结构化输出标记

```python
# 设置 SchedulerOutput 上的标记
scheduler_output.has_structured_output_requests = has_structured_output_requests
scheduler_output.pending_structured_output_tokens = pending_structured_output_tokens
```

这些标记用于 Worker 端判断是否需要进行语法位掩码计算。

---

## 8. 停止条件检查 (check_stop)

定义在 `vllm/v1/core/sched/utils.py`。

```python
def check_stop(request: Request, max_model_len: int) -> bool:
    # 1. min_tokens 检查
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    # 2. EOS token 检查
    last_token_id = request.output_token_ids[-1]
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    # 3. stop_token_ids 检查
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    # 4. 长度限制检查
    if (request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    return False
```

**检查顺序**：

1. **min_tokens**: 如果输出 token 数不足 `min_tokens`，即使生成 EOS 也不停止
2. **EOS token**: 生成了 EOS token → `FINISHED_STOPPED`
3. **stop_token_ids**: 生成了自定义停止 token → `FINISHED_STOPPED`（记录 `stop_reason`）
4. **长度上限**: 达到 `max_model_len` 或 `max_tokens` → `FINISHED_LENGTH_CAPPED`

**注意**：`check_stop` 是逐 token 调用的（在 `_update_request_with_output` 中每个新 token 都调用一次），一旦停止就截断后续 token。

---

## 9. 完整调度流程图

### 9.1 schedule() 总体流程

```
+------------------------------------------------------------------+
|                        schedule() 入口                            |
+----------------------------------+-------------------------------+
                                   |
                                   v
                    +-----------------------------+
                    | VLLM_USE_PD_SPLIT?          |
                    +----+-------------------+----+
                         |                   |
                        No                  Yes
                         |                   |
                         v                   v
                +-----------+   +---------------------------+
                | schedule_ |   | connector is not None?    |
                | default() |   +----+-----------------+----+
                +-----------+        |                 |
                                     Yes               No
                                      |                 |
                                      v                 v
                              +-----------+   +-------------------+
                              | schedule_ |   | use_mla?          |
                              | default() |   +----+----------+---+
                              +-----------+        |          |
                                                   Yes        No
                                                    |          |
                                                    v          v
                                          +--------------+ +--------------+
                                          | full_cuda_   | | schedule_    |
                                          | graph AND    | | split_pd()   |
                                          | num_spec>0?  | +--------------+
                                          +---+------+---+
                                              |      |
                                             Yes    No
                                              |      |
                                              v      v
                                     +--------+ +-----------+
                                     |schedule| | schedule_ |
                                     |_split_ | | default() |
                                     |pd()    | +-----------+
                                     +--------+
```

### 9.2 schedule_default() 详细流程

```
+====================================================================+
|                     schedule_default()                             |
+====================================================================+
|                                                                    |
|  阶段 0: 初始化                                                     |
|    token_budget = max_num_scheduled_tokens                         |
|    encoder_compute_budget = max_num_encoder_input_tokens           |
|    scheduled_new_reqs = [], scheduled_running_reqs = []            |
|    preempted_reqs = []                                             |
+--------------------------------------------------------------------+
|                                                                    |
|  阶段 1: 调度 RUNNING 请求                                          |
|                                                                    |
|    for req in self.running:                                        |
|        num_new = num_tokens_with_spec - num_computed_tokens        |
|        cap with long_prefill_token_threshold                       |
|        cap with token_budget                                       |
|        cap with max_model_len                                      |
|                                                                    |
|        try allocate_slots(req, num_new):                           |
|            success → schedule req                                  |
|            failure → preempt lowest priority req                   |
|                      if still fails → break                        |
|                                                                    |
|        if scheduled:                                               |
|            token_budget -= num_new                                 |
|            handle spec tokens, encoder inputs                      |
+--------------------------------------------------------------------+
|                                                                    |
|  阶段 2: 调度 WAITING 请求                                          |
|                                                                    |
|    while waiting and token_budget > 0:                             |
|        req = waiting.peek()                                        |
|        skip if WAITING_FOR_REMOTE_KVS / FSM / STREAMING            |
|                                                                    |
|        # 前缀缓存查询                                               |
|        new_blocks, num_local = get_computed_blocks(req)            |
|        if connector:                                               |
|            num_external = connector.get_num_new_matched_tokens()   |
|        num_computed = num_local + num_external                     |
|                                                                    |
|        num_new = req.num_tokens - num_computed                     |
|        cap with long_prefill_token_threshold                       |
|                                                                    |
|        try allocate_slots(req, num_new):                           |
|            success → pop from waiting, add to running              |
|            failure → break                                         |
|                                                                    |
|        if scheduled:                                               |
|            req.status = RUNNING                                    |
|            req.num_computed_tokens = num_computed                  |
|            token_budget -= num_new                                 |
|            handle encoder inputs                                   |
|                                                                    |
|    put back skipped requests to waiting queue                      |
+--------------------------------------------------------------------+
|                                                                    |
|  阶段 3: 编码器输入（内嵌在阶段1和阶段2中）                            |
|                                                                    |
|    _try_schedule_encoder_inputs(req, num_computed, num_new,        |
|                                  encoder_budget)                   |
|    → 返回需要调度的 encoder input 索引列表                           |
+--------------------------------------------------------------------+
|                                                                    |
|  阶段 4: 后处理                                                    |
|    - 构建 NewRequestData 列表                                       |
|    - 构建 CachedRequestData                                        |
|    - 计算 common_prefix_blocks                                     |
|    - 构建 connector metadata                                       |
|    - _update_after_schedule() → 推进 num_computed_tokens           |
|    - 返回 SchedulerOutput                                          |
+====================================================================+
```

### 9.3 update_from_output() 详细流程

```
+====================================================================+
|                   update_from_output()                              |
+====================================================================+
|                                                                    |
|  输入: scheduler_output, model_runner_output                       |
|                                                                    |
|  1. 提取 model_runner_output:                                      |
|     sampled_token_ids, logprobs, pooler_outputs, etc.              |
|                                                                    |
|  2. _handle_invalid_blocks():                                      |
|     处理 KV load 失败的请求                                         |
|     → failed_kv_load_req_ids (跳过这些请求)                         |
+--------------------------------------------------------------------+
|                                                                    |
|  3. 遍历 num_scheduled_tokens 中的每个 req_id:                      |
|                                                                    |
|     [跳过已完成的请求]                                               |
|                                                                    |
|     generated_token_ids = sampled_token_ids[req_index]              |
|                                                                    |
|     [投机解码处理]                                                   |
|     if scheduled_spec_token_ids:                                   |
|         num_accepted = len(generated) - 1                          |
|         num_rejected = len(draft) - num_accepted                   |
|         request.num_computed_tokens -= num_rejected                |
|                                                                    |
|     [更新请求]                                                      |
|     new_token_ids, stopped =                                       |
|         _update_request_with_output(request, generated_token_ids)  |
|         ├── append_output_token_ids(token)                         |
|         └── check_stop(request, max_model_len)                     |
|                                                                    |
|     [处理停止]                                                      |
|     if stopped:                                                    |
|         finish_reason = get_finished_reason()                      |
|         _handle_stopped_request(request)                           |
|         _free_request(request)                                     |
|                                                                    |
|     [提取 logprobs]                                                |
|                                                                    |
|     [更新结构化输出 FSM]                                            |
|                                                                    |
|     [构建 EngineCoreOutput]                                        |
+--------------------------------------------------------------------+
|                                                                    |
|  4. 清理队列:                                                       |
|     running = remove_all(running, stopped_running_reqs)            |
|     waiting.remove_requests(stopped_preempted_reqs)                |
|                                                                    |
|  5. 处理 KV connector output:                                      |
|     finished_recving → 标记为可调度                                |
|     finished_sending → 释放 blocks                                 |
|                                                                    |
|  6. 收集 KV cache 事件并发布                                       |
|                                                                    |
|  7. 构建 EngineCoreOutputs (按 client_index 分组)                   |
|                                                                    |
|  8. 生成统计信息 (SchedulerStats)                                   |
|                                                                    |
|  返回: dict[client_index, EngineCoreOutputs]                        |
+====================================================================+
```

### 9.4 完整引擎循环

```
+------------------------------------------------------------------+
|                     EngineCore 忙碌循环                            |
+==================================================================+
|                                                                  |
|   +--------------------+                                         |
|   | scheduler.schedule()|  ← 每步调用一次                          |
|   +--------+-----------+                                         |
|            |                                                     |
|            v                                                     |
|   +--------------------+                                         |
|   | 检查是否有未完成请求  |                                         |
|   +--------+-----------+                                         |
|            |                                                     |
|       No   |   Yes                                               |
|        +---+---+                                                 |
|        |       |                                                 |
|        v       v                                                 |
|   等待新请求  +-------------------+                               |
|        |     | 分发到调度方法:     |                               |
|        |     | schedule_default() |                               |
|        |     | schedule_split_pd()|                               |
|        |     +--------+----------+                               |
|        |              |                                          |
|        |              v                                          |
|        |     +--------------------+                              |
|        |     | ModelRunner        |                              |
|        |     | .execute(output)   |  ← GPU 前向传播               |
|        |     +--------+-----------+                              |
|        |              |                                          |
|        |              v                                          |
|        |     +--------------------+                              |
|        |     | scheduler          |                              |
|        +---->| .update_from_output|  ← 处理输出、更新状态          |
|              +--------+-----------+                              |
|                       |                                          |
|                       v                                          |
|              +--------------------+                              |
|              | 返回 EngineCore    |                              |
|              | Outputs 给前端      |                              |
|              +--------------------+                              |
|                       |                                          |
|                       v                                          |
|              (回到循环开始)                                        |
+==================================================================+
```

---

## 10. 配置参数参考 (SchedulerConfig)

`SchedulerConfig` 定义在 `vllm/config/scheduler.py`。

### 10.1 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `runner_type` | `Literal["generate", "pooling", "draft"]` | `"generate"` | 运行器类型 |
| `max_num_batched_tokens` | `int` | `2048` | **每步最大 token 数**，核心吞吐量参数 |
| `max_num_seqs` | `int` | `128` | **最大并发序列数** |
| `max_num_partial_prefills` | `int` | `1` | 分块预填充时最大并发部分 prefill 数 |
| `max_long_partial_prefills` | `int` | `1` | 超过 long_prefill_token_threshold 的最大并发数 |
| `long_prefill_token_threshold` | `int` | `0` | 长 prompt 阈值（超过此值视为长 prompt） |
| `enable_chunked_prefill` | `bool` | `True` | 是否启用分块预填充 |
| `is_multimodal_model` | `bool` | `False` | 是否多模态模型 |
| `max_num_encoder_input_tokens` | `int` | 自动计算 | 编码器计算预算 |
| `encoder_cache_size` | `int` | 自动计算 | 编码器缓存大小 |
| `policy` | `Literal["fcfs", "priority"]` | `"fcfs"` | 调度策略 |
| `disable_chunked_mm_input` | `bool` | `False` | 禁止分块多模态输入 |
| `scheduler_cls` | `str \| type` | `None` | 自定义调度器类 |
| `async_scheduling` | `bool` | `None` | 是否启用异步调度 |
| `stream_interval` | `int` | `1` | 流式输出间隔（token 数） |

### 10.2 自动推导规则

```python
def __post_init__(self, max_model_len, is_encoder_decoder):
    # 编码器-解码器模型禁用分块预填充
    if is_encoder_decoder:
        self.enable_chunked_prefill = False
        self.long_prefill_token_threshold = 0

    # 编码器预算默认等于 token 预算
    self.max_num_encoder_input_tokens = self.max_num_batched_tokens
    self.encoder_cache_size = self.max_num_batched_tokens

    # 多部分预填充的默认 long_prefill_token_threshold
    if self.max_num_partial_prefills > 1 and self.long_prefill_token_threshold == 0:
        self.long_prefill_token_threshold = int(max_model_len * 0.04)
```

### 10.3 参数校验

```python
def verify_max_model_len(self, max_model_len):
    # 1. max_num_batched_tokens < max_model_len 且未启用 chunked prefill → 报错
    # 2. max_num_batched_tokens < max_num_seqs → 报错
    # 3. max_num_batched_tokens > max_num_seqs * max_model_len → 警告
    # 4. max_num_partial_prefills > 1 但 enable_chunked_prefill=False → 报错
    # 5. long_prefill_token_threshold > max_model_len → 报错
    # 6. max_long_partial_prefills > max_num_partial_prefills → 报错
```

---

## 11. 最佳实践与调优建议

### 11.1 Token 预算调优

`max_num_batched_tokens` 是最重要的性能参数：

- **值太小**: GPU 利用率不足，吞吐量低
- **值太大**: 延迟增加，可能导致 OOM
- **推荐起点**: 2048~8192，根据 GPU 显存和模型大小调整

### 11.2 序列预算调优

`max_num_seqs` 控制并发度：

- 对于在线服务（低延迟优先）：设置较大值（如 256），让更多请求能同时处理
- 对于离线批处理（高吞吐优先）：适度设置，避免过多请求争抢 token 预算
- 注意：`max_num_batched_tokens >= max_num_seqs`（每个序列至少 1 个 token）

### 11.3 分块预填充调优

`enable_chunked_prefill` 默认启用，影响长 prompt 的处理方式：

- **启用时**: 长 prompt 分多个 step 处理，与其他请求交错，改善延迟公平性
- **禁用时**: 长 prompt 一次性处理，可能阻塞其他请求
- `long_prefill_token_threshold`: 设为 `max_model_len * 0.04` 是合理的起点
- `max_num_partial_prefills`: 控制同时处理的长 prompt 数，建议 1~4

### 11.4 调度策略选择

- **FCFS**: 适合大多数场景，简单公平
- **PRIORITY**: 适合需要区分请求优先级的在线服务（如 VIP 用户、紧急查询）

### 11.5 异步调度

- 默认情况下会根据配置自动选择
- 异步调度可以提高 GPU 利用率，减少空闲间隙
- 代价是稍微增加调度复杂度

### 11.6 投机解码

- `num_speculative_tokens`: 建议 1~5，更大的值不一定带来更好的加速
- 投机 token 也消耗 token 预算和 KV cache
- 在 `update_from_output` 中正确处理接受/拒绝，确保 token 计数一致

### 11.7 KV Cache 管理

- `block_size`: 通常 16，影响缓存粒度和内存碎片
- 前缀缓存通过 `cache_config.enable_prefix_caching` 控制
- 模型热更新时需要调用 `reset_prefix_cache(reset_running_requests=True)`

### 11.8 常见问题排查

| 问题 | 可能原因 | 排查方向 |
|------|----------|----------|
| 请求长时间停留在 WAITING | token 预算不足 / KV cache 满 | 检查 `max_num_batched_tokens` 和 KV cache 使用率 |
| 频繁抢占 | KV cache 不够 | 增加 GPU 显存配额或减少 `max_num_seqs` |
| 吞吐量低 | GPU 利用率不足 | 增大 `max_num_batched_tokens`，开启异步调度 |
| 延迟抖动大 | 长 prompt 阻塞 | 启用 chunked prefill，调整 `long_prefill_token_threshold` |
| OOM | KV cache 超分 | 减少 `max_num_batched_tokens` 或 `max_num_seqs` |
| 投机解码加速不明显 | 接受率低 | 减少 `num_speculative_tokens` |

---

## 附录: 术语对照表

| 英文 | 中文 | 说明 |
|------|------|------|
| Schedule | 调度 | 决定每个 step 处理哪些请求及多少 token |
| Chunked Prefill | 分块预填充 | 将长 prompt 拆分到多个 step 处理 |
| Prefix Caching | 前缀缓存 | 缓存和复用共享前缀的 KV cache |
| Preemption | 抢占 | 释放低优先级请求的 KV cache 给新请求 |
| Speculative Decoding | 投机解码 | 用 draft model 预测 token，主模型验证 |
| Token Budget | Token 预算 | 每步最多处理的 token 数 |
| Sequence Budget | 序列预算 | 最大并发请求数 |
| KV Cache | KV 缓存 | 存储注意力机制的 Key-Value 中间结果 |
| FSM | 有限状态机 | 用于结构化输出的语法约束 |
| PD Split | Prefill/Decode 分离 | 将 prefill 和 decode 阶段分配到不同节点 |
| KV Connector | KV 连接器 | 管理跨节点的 KV 传输（PD 分离） |
| Output Placeholder | 输出占位符 | 异步调度中提前预留的输出 token 位置 |
| Async Scheduling | 异步调度 | 提前调度未来步骤以减少 GPU 空闲 |
| Grammar Bitmask | 语法位掩码 | 指示词表中哪些 token 被结构化输出语法允许 |
| Encoder Cache | 编码器缓存 | 存储多模态编码器（如视觉编码器）的输出 |
| EC Connector | EC 连接器 | 管理跨节点的编码器缓存传输 |
