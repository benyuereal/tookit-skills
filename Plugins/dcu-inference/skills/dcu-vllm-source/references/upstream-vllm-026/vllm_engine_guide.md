# vLLM Engine + Executor 模块详解（请求处理管线）

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖请求从 API 入口到 Worker 执行、再流式返回的完整管线：AsyncLLM、EngineCore、ZMQ 通信、Executor 进程架构。

---

## 目录

1. [概述](#1-概述)
2. [进程架构总览](#2-进程架构总览)
3. [请求全生命周期](#3-请求全生命周期)
4. [AsyncLLM（前端入口）](#4-asyncllm前端入口)
5. [InputProcessor（输入处理）](#5-inputprocessor输入处理)
6. [EngineCore（引擎核心）](#6-enginecore引擎核心)
7. [ZMQ 通信协议（EngineCoreClient）](#7-zmq-通信协议enginecoreclient)
8. [Executor 层（多进程/Ray 执行器）](#8-executor-层多进程ray-执行器)
9. [OutputProcessor 与 Detokenizer（输出处理）](#9-outputprocessor-与-detokenizer输出处理)
10. [Abort 与错误处理](#10-abort-与错误处理)
11. [DP 协调器](#11-dp-协调器)
12. [LLMEngine（离线同步路径）](#12-llmengine离线同步路径)
13. [关键数据结构](#13-关键数据结构)
14. [配置参数](#14-配置参数)
15. [常见问题与排查](#15-常见问题与排查)

---

## 1. 概述

Engine + Executor 是 vLLM 的**请求处理管线**，连接 API 前端与 GPU Worker：

- **Engine 层**（`vllm/v1/engine/`）：请求入口、输入/输出处理、调度循环、进程间通信
- **Executor 层**（`vllm/v1/executor/`）：将调度决策分发到多个 Worker 进程（多进程/Ray）

### 关键文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/engine/async_llm.py` | ~1500 | AsyncLLM：异步请求入口（add_request/abort/流式输出） |
| `vllm/v1/engine/core.py` | ~1500 | EngineCore：调度主循环（step()） |
| `vllm/v1/engine/core_client.py` | ~700 | EngineCoreClient：ZMQ 通信客户端 |
| `vllm/v1/engine/coordinator.py` | ~400 | DP 协调器（多 DP rank 请求分发） |
| `vllm/v1/engine/input_processor.py` | ~400 | 输入处理（tokenize、校验、多模态） |
| `vllm/v1/engine/output_processor.py` | ~600 | 输出处理（detokenize、流式、logprobs） |
| `vllm/v1/engine/detokenizer.py` | ~200 | 增量 detokenizer |
| `vllm/v1/engine/llm_engine.py` | ~300 | LLMEngine（离线同步路径） |
| `vllm/v1/executor/abstract.py` | ~200 | Executor 抽象基类 |
| `vllm/v1/executor/multiproc_executor.py` | ~700 | 多进程执行器（ZMQ） |
| `vllm/v1/executor/uniproc_executor.py` | ~100 | 单进程执行器 |
| `vllm/v1/executor/ray_executor.py` | ~500 | Ray 执行器 |

---

## 2. 进程架构总览

### 2.1 多进程架构（默认，`distributed_executor_backend=mp`）

```
┌─────────────────────────────────────────────────────────────┐
│  API Server 进程 (uvicorn)                                   │
│  ├── OpenAI 兼容 API (/v1/chat/completions 等)               │
│  ├── AsyncLLM (async_llm.py)                                │
│  │   ├── InputProcessor     (输入处理)                       │
│  │   ├── OutputProcessor    (输出处理 + detokenize)          │
│  │   └── EngineCoreClient   (ZMQ 客户端)                     │
│  └── 请求队列 / 流式响应 (SSE)                                │
└──────────────────────────┬──────────────────────────────────┘
                           │ ZMQ (PUSH/PULL)
┌──────────────────────────▼──────────────────────────────────┐
│  EngineCore 进程 (独立进程, 独立事件循环)                      │
│  ├── EngineCore (core.py)                                   │
│  │   ├── Scheduler        (调度器)                           │
│  │   ├── KVCacheManager   (KV Cache 管理)                    │
│  │   └── step() 主循环                                        │
│  └── MultiprocExecutor (executor)                           │
│       │ ZMQ (PUSH/PULL, 每 worker 一对)                      │
└───────┬──────────┬──────────┬──────────┬────────────────────┘
        │          │          │          │
   ┌────▼───┐ ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
   │Worker 0│ │Worker 1│ │Worker 2│ │Worker N│   (每 GPU 一个进程)
   │GPU 0   │ │GPU 1   │ │GPU 2   │ │GPU N   │
   │GPUModelRunner       │
   └────────┘ └────────┘ └────────┘ └────────┘
```

### 2.2 进程职责划分

| 进程 | 组件 | 职责 |
|------|------|------|
| API Server | AsyncLLM、Input/OutputProcessor | 请求解析、tokenize、detokenize、流式输出 |
| EngineCore | Scheduler、KVCacheManager、Executor | 调度决策、KV 管理、分发执行 |
| Worker × N | GPUModelRunner | GPU 前向、采样 |

**为什么分离**：
- 调度循环（CPU 密集）与 API 服务（IO 密集）分离，互不阻塞
- EngineCore 独占调度，避免 GIL 竞争
- Worker 进程专注 GPU 计算

### 2.3 单进程模式（`uni`）

TP=1 且无 DP 时可用 `UniProcExecutor`：EngineCore 与 Worker 同进程，无 ZMQ 开销。

---

## 3. 请求全生命周期

```
1. 用户请求
   POST /v1/chat/completions
        │
2. API Server (api_server.py)
   ├── 解析 OpenAI 格式请求
   ├── 构造 SamplingParams
   └── async_llm.generate(request_id, prompt, params)
        │
3. AsyncLLM.add_request()
   ├── InputProcessor.process_input()
   │   ├── tokenize (prompt → token_ids)
   │   ├── 多模态处理 (图像/音频 → mm_features)
   │   ├── 校验 (长度、格式)
   │   └── 构造 Request 对象
   ├── 注册输出队列 (output_processor.add_request)
   └── engine_core.add_request_async(request)
        │
4. ZMQ 传输
   EngineCoreClient → ZMQ PUSH → EngineCore 进程
        │
5. EngineCore.add_request()
   └── scheduler.add_request(request)  (进入 waiting 队列)
        │
6. EngineCore.step() 主循环 (每步)
   ├── scheduler.schedule() → SchedulerOutput
   │   ├── 前缀缓存查询
   │   ├── KV Cache 分配 (allocate_slots)
   │   └── 抢占决策
   ├── executor.execute_model(scheduler_output)
   │   └── ZMQ → Workers → GPUModelRunner.execute_model()
   │       └── 模型前向 + 采样 → ModelRunnerOutput
   ├── scheduler.update_from_output(model_runner_output)
   │   ├── 更新请求状态
   │   ├── 检测完成 (stop token / max_tokens)
   │   └── 释放 KV Cache
   └── 产生 EngineCoreOutputs (新 token / 完成信号)
        │
7. ZMQ 返回
   EngineCore → ZMQ PULL → EngineCoreClient
        │
8. AsyncLLM 输出处理
   ├── OutputProcessor.process_model_output()
   │   ├── Detokenizer 增量解码 (token_ids → 文本)
   │   ├── logprobs 处理
   │   └── 检测完成
   └── 流式 yield (SSE chunk) 给 API Server
        │
9. 请求完成
   ├── 输出 finish_reason (stop/length/error)
   └── 清理 (释放请求状态)
```

---

## 4. AsyncLLM（前端入口）

### 4.1 类结构

```python
class AsyncLLM:
    def __init__(self, vllm_config, ...):
        self.input_processor = InputProcessor(vllm_config)
        self.output_processor = OutputProcessor(vllm_config, ...)
        self.engine_core = EngineCoreClient.make_client(...)  # ZMQ 客户端
        self.log_stats = ...

    async def add_request(self, request_id, prompt, sampling_params, ...)
    async def generate(self, prompt, sampling_params, request_id, ...) -> AsyncGenerator
    async def abort(self, request_id)
    async def encode(self, ...)  # pooling 模型
```

### 4.2 add_request 流程

```python
async def add_request(self, request_id, prompt, sampling_params, arrival_time, ...):
    # 1. 输入处理
    engine_input, token_type_ids = self.input_processor.process_input(
        request_id, prompt, sampling_params, arrival_time, ...)

    # 2. 注册输出处理
    self.output_processor.add_request(request, prompt, ...)

    # 3. 发送到 EngineCore (异步, 不阻塞)
    await self.engine_core.add_request_async(request)
```

### 4.3 generate（流式生成）

```python
async def generate(self, prompt, sampling_params, request_id, ...) -> AsyncGenerator:
    # 注册请求
    await self.add_request(...)

    # 从输出队列流式读取
    async for output in self.output_processor.request_output_iterator(request_id):
        if output.finished:
            yield output
            break
        yield output  # SSE chunk
```

### 4.4 输出队列机制

每个请求有一个 `asyncio.Queue`，`OutputProcessor` 将处理后的输出放入队列，`generate()` 迭代器从队列读取。请求完成时放入 `EngineCoreOutput(finished=True)` 哨兵。

---

## 5. InputProcessor（输入处理）

### 5.1 职责

```python
class InputProcessor:
    def process_input(self, request_id, prompt, params, arrival_time, ...)
        -> tuple[EngineCoreRequest, list[int]]:
        # 1. prompt 规范化 (str / list[int] / 多模态 dict)
        # 2. tokenize (str → token_ids)
        # 3. 多模态处理 (图像/音频 → mm_features + placeholder tokens)
        # 4. 校验:
        #    - 长度 ≤ max_model_len
        #    - 多模态输入数量限制
        #    - 结构化输出 (grammar) 编译
        # 5. 构造 EngineCoreRequest
```

### 5.2 多模态输入处理

```
用户输入: {"content": [{"type": "image_url", ...}, {"type": "text", "text": "..."}]}
    │
    ▼
1. 提取多模态项 (图像/音频)
2. 编码器预处理 (resize、normalize)
3. 计算 placeholder token 数 (每个图像 → N 个特殊 token)
4. 构造 mm_features (数据 + 位置)
5. prompt = [image_placeholder×N] + text_tokens
```

### 5.3 结构化输出

```python
# grammar 编译 (JSON schema → FSM)
if sampling_params.structural_tag:
    grammar = compile_grammar(schema)
    request.structured_output_request = StructuredOutputRequest(grammar)
```

---

## 6. EngineCore（引擎核心）

### 6.1 类结构

```python
class EngineCore:
    def __init__(self, vllm_config, ...):
        self.scheduler = Scheduler(vllm_config, kv_cache_config, ...)
        self.model_executor = executor_class(vllm_config)  # MultiprocExecutor 等
        self.log_stats = ...

    def add_request(self, request: EngineCoreRequest)
    def abort_requests(self, request_ids)
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]
    def run_busy_loop(self)  # 主循环
```

### 6.2 step() 主循环

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    # 1. 调度
    scheduler_output = self.scheduler.schedule()

    # 2. 执行 (如果调度了 token)
    if scheduler_output.total_num_scheduled_tokens > 0:
        model_executor_output = self.model_executor.execute_model(
            scheduler_output, non_block=True)
    else:
        model_executor_output = self.model_executor.sample_tokens(
            scheduler_output, non_block=True)

    # 3. 等待执行完成 (异步)
    future = model_executor_output.future
    future.result()  # 阻塞直到 Worker 完成
    model_runner_output = future.output

    # 4. 更新调度器状态
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_runner_output)

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

### 6.3 run_busy_loop（事件循环）

```python
def run_busy_loop(self):
    # 独立事件循环 (独立于 API Server)
    while True:
        # 1. 处理输入消息 (add_request / abort)
        self._process_input_queue()

        # 2. 执行一步
        outputs, has_work = self.step()

        # 3. 发送输出
        if outputs:
            self._send_outputs(outputs)

        # 4. 无工作时休眠 (等待新请求)
        if not has_work:
            self._wait_for_input()  # ZMQ poll with timeout
```

### 6.4 异步调度（async_scheduling）

```python
# 启用时, step() 不等待执行完成
# 调度下一步与上一步的 GPU 执行重叠
if self.scheduler_config.async_scheduling:
    # 不阻塞, 立即调度下一步
    # Worker 端用 CUDA stream 重叠执行
```

---

## 7. ZMQ 通信协议（EngineCoreClient）

### 7.1 通信拓扑

```
AsyncLLM (API Server 进程)
    │
    ├── Input ZMQ: PUSH → EngineCore 的 PULL
    │   (add_request / abort 消息)
    │
    └── Output ZMQ: EngineCore 的 PUSH → PULL
        (EngineCoreOutputs 消息)
```

### 7.2 EngineCoreClient

```python
class EngineCoreClient(ABC):
    @classmethod
    def make_client(cls, ...):
        # 根据配置选择实现:
        # - InprocClient: 同进程 (uni 模式)
        # - MPClient: 多进程 (ZMQ)
        # - DPClient: 多 DP rank (ZMQ + 协调)

    async def add_request_async(self, request)
    async def abort_requests_async(self, request_ids)
    def get_output_async(self) -> EngineCoreOutputs
```

### 7.3 消息格式

```python
# 输入消息 (API Server → EngineCore)
InputMessage = {
    "type": "add_request" | "abort",
    "request": EngineCoreRequest,  # 序列化
}

# 输出消息 (EngineCore → API Server)
EngineCoreOutputs = {
    "outputs": dict[client_index, list[EngineCoreOutput]],
    "finished_requests": set[request_id],
}
```

### 7.4 序列化

- 使用 `msgspec`（高性能结构化序列化）
- 张量数据（多模态）用共享内存传递（避免序列化开销）
- `EngineCoreRequest` 包含：token_ids、mm_features、sampling_params、lora_request

### 7.5 背压控制

```python
# 输入队列有上限, 防止 API Server 发送过快
MAX_INPUT_QUEUE_SIZE = 1000
# 队列满时 add_request 阻塞 (背压)
```

---

## 8. Executor 层（多进程/Ray 执行器）

### 8.1 Executor 抽象

```python
class Executor(ABC):
    def __init__(self, vllm_config):
        self._init_executors()  # 创建 worker 进程

    def collective_rpc(self, method, args, kwargs, ...) -> list:
        # 向所有 worker 广播方法调用
        # 返回每个 worker 的结果列表

    def execute_model(self, scheduler_output, non_block=False)
    def sample_tokens(self, grammar_output, non_block=False)
    def add_lora(self, lora)
    def check_health(self)
```

### 8.2 MultiprocExecutor（多进程，默认）

```python
class MultiprocExecutor(Executor):
    def _init_executors(self):
        # 为每个 GPU 启动一个 worker 进程
        for rank in range(world_size):
            proc = mp.Process(target=worker_main, args=(rank, ...))
            proc.start()
            # 每 worker 一对 ZMQ socket:
            #   input:  PUSH → worker PULL
            #   output: worker PUSH → PULL

    def collective_rpc(self, method, args, kwargs, non_block=False):
        # 1. 序列化 (method, args, kwargs)
        # 2. 广播到所有 worker (ZMQ PUB/SUB 或 每 worker PUSH)
        # 3. 收集结果 (ZMQ PULL)
        # 4. 反序列化返回
```

### 8.3 Worker 进程

```python
def worker_main(rank, vllm_config, ...):
    # 1. 初始化分布式环境 (NCCL)
    init_distributed_environment(...)
    initialize_model_parallel(...)

    # 2. 创建 Worker
    worker = Worker(vllm_config, rank, device)
    worker.init_device()
    worker.load_model()
    worker.determine_available_memory()
    worker.initialize_from_config(kv_cache_config)

    # 3. 事件循环
    while True:
        # 接收 RPC 消息
        method, args, kwargs = zmq_recv()
        # 执行方法
        result = getattr(worker, method)(*args, **kwargs)
        # 返回结果
        zmq_send(result)
```

### 8.4 UniProcExecutor（单进程）

```python
class UniProcExecutor(Executor):
    # TP=1 且无 DP 时使用
    # Worker 与 EngineCore 同进程, 无 ZMQ 开销
    def collective_rpc(self, method, args, kwargs):
        return [getattr(self.worker, method)(*args, **kwargs)]
```

### 8.5 RayExecutor（Ray 分布式）

```python
class RayDistributedExecutor(Executor):
    # 多节点部署时使用
    # 用 Ray placement group 管理 worker
    def _init_executors(self):
        # 创建 Ray placement group (每 GPU 一个 bundle)
        # 启动 Ray actor (worker)
    def collective_rpc(self, method, args, kwargs):
        # ray.remote 广播到所有 actor
```

### 8.6 执行器选择

```python
def get_padded_worker_and_dummy_model(vllm_config):
    backend = vllm_config.parallel_config.distributed_executor_backend
    if backend == "mp":
        return MultiprocExecutor
    elif backend == "ray":
        return RayDistributedExecutor
    elif backend == "uni":
        return UniProcExecutor
```

---

## 9. OutputProcessor 与 Detokenizer（输出处理）

### 9.1 OutputProcessor

```python
class OutputProcessor:
    def __init__(self, vllm_config, ...):
        self.detokenizer = IncrementalDetokenizer(tokenizer)
        self.request_states: dict[request_id, RequestState]

    def add_request(self, request, prompt, ...)
    def process_model_output(self, engine_core_output) -> list[EngineOutput]
    def request_output_iterator(self, request_id) -> AsyncGenerator
```

### 9.2 增量 Detokenizer

```python
class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.decode_only_mask = ...  # 避免重复解码

    def decode(self, new_token_ids, ...) -> str:
        # 增量解码: 只解码新增 token
        # 处理 BPE 合并 (最后一个 token 可能不完整)
        # 返回新增文本 (不含已输出部分)
```

**增量解码的关键**：
- BPE tokenizer 的最后一个 token 可能与下一个 token 合并
- 因此每次保留最后一个 token 不输出，等下一个 token 到来再确认
- 避免输出 "Hel" → "Hello" 的重复

### 9.3 流式输出

```python
def process_model_output(self, engine_core_output):
    for output in engine_core_output.outputs:
        # 1. 增量 detokenize
        new_text = self.detokenizer.decode(output.new_token_ids)

        # 2. 构造 EngineOutput
        engine_output = EngineOutput(
            request_id=output.request_id,
            new_token_ids=output.new_token_ids,
            new_text=new_text,
            logprobs=output.logprobs,
            finished=output.finished,
            finish_reason=output.finish_reason,
        )

        # 3. 放入请求的输出队列
        self.request_states[request_id].output_queue.put_nowait(engine_output)
```

### 9.4 logprobs 处理

```python
# logprobs 从 GPU 张量转换为 Python 结构
# 处理 top-k logprobs
# 转换为 OpenAI 格式 (dict[token, logprob])
```

---

## 10. Abort 与错误处理

### 10.1 Abort 流程

```
用户: DELETE /v1/completions/{request_id} 或客户端断开
    │
API Server:
    async_llm.abort(request_id)
    │
    ├── output_processor.abort_request(request_id)  (标记完成)
    └── engine_core.abort_requests_async([request_id])
         │
         ZMQ → EngineCore:
         scheduler.finish_requests([request_id], FINISHED_ABORTED)
         │
         ├── 从 running/waiting 队列移除
         ├── 释放 KV Cache
         └── 产生完成信号
```

### 10.2 客户端断开检测

```python
# API Server 检测 SSE 连接断开
async def stream_response():
    try:
        async for chunk in generate():
            yield chunk
    except asyncio.CancelledError:
        # 客户端断开, abort 请求
        await async_llm.abort(request_id)
```

### 10.3 错误处理

```python
# 模型执行错误 (OOM、NCCL 错误等)
try:
    model_runner_output = future.result()
except Exception as e:
    # 标记所有请求为 FINISHED_ERROR
    scheduler.finish_requests(all_req_ids, FINISHED_ERROR)
    # 返回错误给客户端
    raise
```

---

## 11. DP 协调器

### 11.1 DP 请求分发

```python
class DPCoordinator:
    # 多 DP rank 时, 协调请求分发
    # 每个 DP rank 有独立的 EngineCore

    def add_request(self, request):
        # 选择目标 DP rank (负载均衡)
        target_rank = self._select_rank()
        # 转发到目标 rank 的 EngineCore
        self.clients[target_rank].add_request(request)
```

### 11.2 负载均衡策略

```python
# 基于各 rank 的队列长度选择
def _select_rank(self):
    # 最少请求的 rank
    return min(self.ranks, key=lambda r: self.queue_len[r])
```

### 11.3 DP 同步

```python
# 所有 DP rank 必须同步执行 (NCCL 集合通信)
# 某 rank 无请求时, 用 dummy batch 填充
def coordinate_batch_across_dp(...):
    # all_reduce 各 rank 的 token 数
    # 统一 batch 大小 (padding)
```

---

## 12. LLMEngine（离线同步路径）

### 12.1 用途

`LLMEngine` 是同步（非异步）引擎，用于离线批量推理（`LLM` 类）：

```python
class LLMEngine:
    def __init__(self, vllm_config, ...)
    def add_request(self, request_id, prompt, params)  # 同步
    def step(self) -> list[EngineCoreOutput]  # 同步执行一步
    def abort_request(self, request_id)
```

### 12.2 与 AsyncLLM 的区别

| 维度 | AsyncLLM | LLMEngine |
|------|----------|-----------|
| 用途 | 在线服务 (API) | 离线批量推理 |
| 接口 | async/await | 同步 |
| 输出 | 流式 (AsyncGenerator) | 批量 (list) |
| 进程 | 多进程 (ZMQ) | 可单进程 |

### 12.3 LLM 类（用户接口）

```python
class LLM:
    def __init__(self, model, ...)
        self.llm_engine = LLMEngine.from_engine_args(...)

    def generate(self, prompts, sampling_params) -> list[RequestOutput]:
        # 批量生成 (同步)
        for prompt in prompts:
            self.llm_engine.add_request(...)
        while self.llm_engine.has_unfinished_requests():
            self.llm_engine.step()
        return outputs
```

---

## 13. 关键数据结构

### 13.1 EngineCoreRequest

```python
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int]
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    priority: int
```

### 13.2 EngineCoreOutput

```python
@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    finish_reason: FinishReason | None
    stop_reason: int | str | None
    logprobs: LogprobsTensors | None
    prompt_logprobs_tensors: LogprobsTensors | None
    pooling_output: torch.Tensor | None
    num_cached_tokens: int
    finished: bool
```

### 13.3 EngineCoreOutputs

```python
@dataclass
class EngineCoreOutputs:
    outputs: list[EngineCoreOutput]
    finished_requests: set[str]
    scheduler_stats: SchedulerStats | None
```

### 13.4 EngineOutput（API 层）

```python
@dataclass
class EngineOutput:
    request_id: str
    index: int
    token_id: int
    logprobs: LogprobsTensors | None
    text: str  # 增量文本
    finish_reason: FinishReason | None
    finished: bool
```

---

## 14. 配置参数

### 14.1 相关配置

| 配置 | 默认 | 说明 |
|------|------|------|
| `distributed_executor_backend` | auto | mp / ray / uni |
| `async_scheduling` | None | 异步调度 |
| `data_parallel_size` | 1 | DP rank 数 |
| `data_parallel_backend` | mp | DP 后端 (mp/ray) |
| `max_num_seqs` | 128 | 最大并发请求 |
| `max_num_batched_tokens` | 2048 | 每步最大 token |

### 14.2 环境变量

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=1    # 启用多进程 (默认)
VLLM_LOGGING_LEVEL=INFO             # 日志级别
VLLM_SERVER_DEV_MODE=0              # 开发模式
```

---

## 15. 常见问题与排查

### 15.1 请求卡住

| 现象 | 原因 | 排查 |
|------|------|------|
| 请求无响应 | EngineCore 进程崩溃 | 检查 EngineCore 日志 |
| 输出停滞 | Worker 执行慢 (GPU 瓶颈) | 检查 GPU 利用率 |
| 流式中断 | ZMQ 连接断开 | 检查网络/进程 |

### 15.2 内存问题

| 现象 | 原因 | 解决 |
|------|------|------|
| API Server OOM | 多模态输入缓存 | 限制输入大小 |
| EngineCore OOM | 请求队列过大 | 限制 max_num_seqs |
| Worker OOM | KV Cache 不足 | 降低 gpu_memory_utilization |

### 15.3 性能问题

| 现象 | 原因 | 解决 |
|------|------|------|
| 首 token 延迟高 | 调度 + prefill | 启用 chunked prefill |
| 吞吐低 | 进程间通信开销 | 用 uni 模式 (TP=1) |
| DP 不均衡 | 请求分发不均 | 检查 DP 协调器 |

### 15.4 调试技巧

```bash
# 查看 EngineCore 日志
VLLM_LOGGING_LEVEL=DEBUG vllm serve ...

# 查看请求状态
curl localhost:8000/v1/loads  # 负载信息

# 监控指标
curl localhost:8000/metrics  # Prometheus 指标
# vllm:num_requests_running
# vllm:num_requests_waiting
# vllm:time_to_first_token_seconds
```

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-09-10  
> **核心文件**：`vllm/v1/engine/async_llm.py`, `core.py`, `core_client.py`, `input_processor.py`, `output_processor.py`, `vllm/v1/executor/multiproc_executor.py`
