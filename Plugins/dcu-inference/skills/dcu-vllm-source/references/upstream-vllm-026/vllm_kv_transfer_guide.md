# vLLM KV Transfer / P-D 分离模块详解

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖 Prefill/Decode 分离部署的 KV Cache 传输机制：KVConnector 接口、传输流程、请求状态机、与调度器/Worker 的集成。

---

## 目录

1. [概述](#1-概述)
2. [P-D 分离架构](#2-p-d-分离架构)
3. [KVConnector 接口](#3-kvconnector-接口)
4. [Connector 实现清单](#4-connector-实现清单)
5. [KV 传输完整数据流](#5-kv-传输完整数据流)
6. [请求状态机](#6-请求状态机)
7. [与调度器的集成](#7-与调度器的集成)
8. [与 Worker 的集成](#8-与-worker-的集成)
9. [KVConnectorMetadata 流程](#9-kvconnectormetadata-流程)
10. [配置参数](#10-配置参数)
11. [实际部署案例](#11-实际部署案例)
12. [常见问题与排查](#12-常见问题与排查)

---

## 1. 概述

### 1.1 什么是 P-D 分离

Prefill/Decode 分离（Disaggregated Prefill）将 LLM 推理的两个阶段部署到**不同的 GPU 实例**：

- **Prefill 实例**：处理 prompt 的 prefill（计算密集，KV Cache 生成）
- **Decode 实例**：处理 token 生成（带宽密集，KV Cache 读取）

**为什么分离**：
- Prefill 和 Decode 的资源特性不同（计算 vs 带宽），混部互相干扰
- 分离后可独立扩缩容（prefill 瓶颈时加 prefill 实例）
- 避免 prefill 长 prompt 阻塞 decode 的 TTFT

**代价**：prefill 完成后需将 KV Cache 传输到 decode 实例（网络开销）。

### 1.2 关键文件

| 文件 | 职责 |
|------|------|
| `vllm/distributed/kv_transfer/README.md` | 设计概览 |
| `vllm/distributed/kv_transfer/kv_transfer_state.py` | 全局 connector 状态 |
| `vllm/distributed/kv_transfer/kv_connector/base.py` | KVConnector 抽象基类 |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | connector 工厂 |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | V1 connector 基类（KVConnectorBase_V1） |
| `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` | LMCache connector |
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/` | P2P connector（NIXL 等） |
| `vllm/config/kv_transfer.py` | KVTransferConfig |
| `vllm/v1/core/sched/scheduler.py` | 调度器集成 |
| `vllm/v1/worker/gpu/kv_connector.py` | Worker 端 connector |

---

## 2. P-D 分离架构

### 2.1 部署拓扑

```
                    ┌─────────────────┐
                    │   Frontend /    │
                    │   Coordinator   │
                    │  (请求路由)      │
                    └───┬─────────┬───┘
                        │         │
            ┌───────────▼──┐  ┌───▼───────────┐
            │ Prefill 实例  │  │ Decode 实例    │
            │ (GPU 组 A)    │  │ (GPU 组 B)     │
            │               │  │               │
            │ 1. prefill    │  │ 3. decode     │
            │ 2. 生成 KV    │  │ 4. 逐 token   │
            │    Cache      │  │    生成        │
            └───────┬───────┘  └───────▲───────┘
                    │                  │
                    │  KV Cache 传输    │
                    └──────────────────┘
                    (NIXL / LMCache / P2P)
```

### 2.2 两种部署模式

| 模式 | 说明 |
|------|------|
| **KV 传输模式** | prefill 实例计算 KV，传输给 decode 实例（本模块重点） |
| **共享存储模式** | prefill 写 KV 到共享存储（LMCache/Redis），decode 读取 |

### 2.3 角色划分

每个实例通过 `kv_transfer_config.kv_role` 声明角色：

```python
kv_role: "kv_both" | "kv_producer" | "kv_consumer"
# kv_producer: prefill 实例 (生成 KV)
# kv_consumer: decode 实例 (消费 KV)
# kv_both:     单实例 (不分离, 本地前缀缓存)
```

---

## 3. KVConnector 接口

### 3.1 KVConnectorBase_V1（V1 基类）

```python
class KVConnectorBase_V1(ABC):
    def __init__(self, vllm_config, role, kv_cache_config, dp_rank)

    # === 调度器侧 (SCHEDULER role) ===

    def get_num_new_matched_tokens(self, request, num_computed_tokens)
        -> tuple[int, bool]:
        """查询外部缓存命中的 token 数
        返回: (外部命中 token 数, 是否异步加载)
        """

    def update_state_after_alloc(self, request, block_ids,
                                  num_external_tokens):
        """KV 分配后更新 connector 状态
        (记录哪些 block 需要加载/保存)
        """

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        """构建本步的传输元数据
        (哪些请求要 send/recv, 哪些 block)
        """

    def request_finished(self, request, block_ids) -> tuple[bool, dict]:
        """请求完成时调用
        返回: (是否延迟释放 block, kv_transfer_params)
        """

    # === Worker 侧 (WORKER role) ===

    def start_load_kv(self, forward_context, **kwargs):
        """开始加载 KV (decode 实例, prefill 前)"""

    def wait_for_save(self):
        """等待 KV 保存完成 (prefill 实例)"""

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        """保存某层的 KV (prefill 实例, 每层调用)"""

    def get_finished(self, finished_req_ids)
        -> tuple[list[str], list[str]]:
        """查询完成的传输
        返回: (完成的 recv 请求, 完成的 send 请求)
        """

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """传输统计"""

    def shutdown(self):
        """清理"""
```

### 3.2 双角色设计

同一个 connector 类在两个进程角色中实例化：

| 角色 | 位置 | 职责 |
|------|------|------|
| `SCHEDULER` | EngineCore 进程 | 决策传输（哪些请求/block）、构建 metadata |
| `WORKER` | Worker 进程 | 执行传输（实际 send/recv KV 数据） |

```python
# 工厂根据角色创建
def create_connector(vllm_config, role, kv_cache_config, dp_rank):
    connector_cls = KVConnectorFactory.get_connector_class(vllm_config)
    return connector_cls(vllm_config, role, kv_cache_config, dp_rank)
```

### 3.3 关键抽象方法说明

| 方法 | 调用时机 | 作用 |
|------|---------|------|
| `get_num_new_matched_tokens` | 调度 waiting 请求时 | 查询外部 KV 命中（decode 实例查 prefill 是否已完成） |
| `update_state_after_alloc` | KV 分配后 | 记录传输计划 |
| `build_connector_meta` | 每步调度后 | 生成本步传输指令 |
| `request_finished` | 请求完成 | 触发 KV 保存/传输 |
| `start_load_kv` | decode 前向开始前 | 启动 KV 加载 |
| `save_kv_layer` | prefill 每层后向 | 逐层保存 KV |
| `get_finished` | 每步 | 查询传输完成状态 |

---

## 4. Connector 实现清单

### 4.1 可用 Connector

| Connector | 文件 | 传输机制 | 适用场景 |
|-----------|------|---------|---------|
| **LMCacheConnectorV1** | `lmcache_connector.py` | 共享存储（CPU/Redis/S3） | 跨实例共享 KV、离线缓存 |
| **P2NConnector** | `p2p/p2p_connector.py` | P2P 网络（NIXL） | prefill→decode 直接传输 |
| **NixlConnector** | `p2p/nixl_connector.py` | NIXL 库（RDMA） | 高性能 RDMA 传输 |
| **MultiConnector** | `multi_connector.py` | 组合多个 connector | 混合场景 |
| **OffloadingConnector** | `offloading_connector.py` | CPU offload | KV 卸载到 CPU |
| **ExampleConnector** | `example_connector.py` | 示例实现 | 开发参考 |

### 4.2 LMCacheConnectorV1

最通用的 connector，支持多种后端：

```python
class LMCacheConnectorV1(KVConnectorBase_V1):
    # 后端:
    # - local: CPU 内存
    # - redis: Redis 集群
    # - s3: 对象存储
    # - vllm: vLLM 原生 P2P

    # 特点:
    # - 支持 KV 缓存复用 (跨请求/跨实例)
    # - 支持异步加载
    # - 支持部分传输 (按 block)
```

### 4.3 NixlConnector（P2P RDMA）

高性能 P2P 传输，基于 NVIDIA NIXL 库：

```python
class NixlConnector(KVConnectorBase_V1):
    # 特点:
    # - RDMA 直接内存传输 (绕过 CPU)
    # - prefill 实例注册 KV 内存区域
    # - decode 实例通过 NIXL 拉取
    # - 支持零拷贝
```

### 4.4 工厂注册

```python
# factory.py
class KVConnectorFactory:
    @staticmethod
    def get_connector_class(vllm_config) -> type[KVConnectorBase_V1]:
        name = vllm_config.kv_transfer_config.kv_connector
        # 从注册表查找
        return _CONNECTOR_REGISTRY[name]

# 注册自定义 connector
@KVConnectorFactory.register_connector("my_connector")
class MyConnector(KVConnectorBase_V1):
    ...
```

---

## 5. KV 传输完整数据流

### 5.1 完整流程

```
=== Prefill 实例 (kv_producer) ===

1. 请求到达, 调度 prefill
   scheduler.schedule() → 分配 KV blocks

2. prefill 前向
   Worker.execute_model()
   ├── 模型前向 (计算 KV)
   └── connector.save_kv_layer(layer_name, kv_layer, ...)
       (每层 attention 后向时逐层保存 KV)

3. prefill 完成
   scheduler: request_finished(request, block_ids)
   └── connector 标记 KV 可传输
       (注册到 NIXL / 写入 LMCache)

4. 传输
   decode 实例请求 KV
   └── NIXL RDMA 传输 / LMCache 读取
       (KV blocks: prefill GPU → decode GPU)

=== Decode 实例 (kv_consumer) ===

1. 请求到达 (带 kv_transfer_params: prefill 实例地址)
   scheduler.add_request(request)

2. 调度时查询外部 KV
   connector.get_num_new_matched_tokens(request, 0)
   → 返回 (num_external_tokens, load_async)
   (prefill 已完成的 token 数)

3. 分配 KV + 更新状态
   connector.update_state_after_alloc(request, block_ids, num_external)
   (记录哪些 block 要加载)

4. decode 前向开始前
   connector.start_load_kv(forward_context)
   (启动 KV 加载, 可与计算重叠)

5. 等待加载完成
   connector.get_finished(finished_req_ids)
   → (finished_recving, finished_sending)
   (加载完成后请求才能 decode)

6. decode 逐 token 生成
   (使用加载的 KV + 新生成的 KV)
```

### 5.2 传输数据

```
传输内容: KV Cache blocks
  - 每个 block: [block_size, num_kv_heads, head_size] × 2 (K+V)
  - 传输单位: block (与 PagedAttention 对齐)
  - 传输量: num_prefill_tokens / block_size × block_size_bytes

传输元数据 (KVConnectorMetadata):
  - 要 send 的请求 + block_ids
  - 要 recv 的请求 + block_ids
  - 目标/源实例地址
```

### 5.3 与计算的重叠

```
理想情况: KV 传输与 decode 计算重叠
  ┌──────────────────────────────────┐
  │ KV 传输 (网络)                    │
  └──────────────────────────────────┘
  ┌──────────────────────────────────┐
  │ decode 计算 (GPU)                 │
  └──────────────────────────────────┘
  (传输完成前, 请求处于 WAITING_FOR_REMOTE_KVS)
```

---

## 6. 请求状态机

### 6.1 P-D 分离下的请求状态

```
                    ┌─────────────┐
                    │  WAITING    │ (新请求)
                    └──────┬──────┘
                           │ 调度 + KV 分配
                           │ + 外部 KV 查询
                    ┌──────▼──────────────┐
                    │ WAITING_FOR_REMOTE_  │ (decode 实例,
                    │ KVS                  │  等待 KV 加载)
                    └──────┬──────────────┘
                           │ KV 加载完成
                           │ (get_finished)
                    ┌──────▼──────┐
                    │  RUNNING    │ (decode 生成)
                    └──────┬──────┘
                           │ 完成/abort
                    ┌──────▼──────┐
                    │  FINISHED   │
                    └─────────────┘
```

### 6.2 关键状态转换

| 转换 | 触发 | 说明 |
|------|------|------|
| WAITING → WAITING_FOR_REMOTE_KVS | `load_async=True` | decode 实例异步加载 KV |
| WAITING_FOR_REMOTE_KVS → WAITING | `get_num_new_matched_tokens` 返回 ready | KV 加载完成 |
| WAITING → RUNNING | 正常调度 | prefill 实例直接 prefill |
| RUNNING → FINISHED | 生成完成 | prefill 完成后 KV 可传输 |

### 6.3 调度器中的状态处理

```python
# scheduler.py
if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    is_ready = self._update_waiting_for_remote_kv(request)
    if is_ready:
        # KV 加载完成, 回到 WAITING
        request.status = RequestStatus.WAITING
    else:
        # 仍在加载, 跳过本步
        continue
```

---

## 7. 与调度器的集成

### 7.1 调度器中的 connector 调用点

```python
class Scheduler:
    def __init__(self, ...):
        # 创建 SCHEDULER 角色 connector
        self.connector = KVConnectorFactory.create_connector(
            vllm_config, role=KVConnectorRole.SCHEDULER,
            kv_cache_config, dp_rank)

    def schedule(self):
        for request in waiting_queue:
            # 1. 查询外部 KV 命中
            if self.connector:
                ext_tokens, load_async = \
                    self.connector.get_num_new_matched_tokens(
                        request, num_local_computed)
                request.num_external_computed_tokens = ext_tokens

            # 2. 分配 KV (含外部 token)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens,
                num_external_computed_tokens=ext_tokens,
                delay_cache_blocks=load_async)

            # 3. 更新 connector 状态
            if self.connector:
                self.connector.update_state_after_alloc(
                    request, block_ids, ext_tokens)

            # 4. 异步加载 → WAITING_FOR_REMOTE_KVS
            if load_async:
                request.status = WAITING_FOR_REMOTE_KVS

        # 5. 构建传输元数据
        if self.connector:
            scheduler_output.kv_connector_metadata = \
                self.connector.build_connector_meta(scheduler_output)

    def update_from_output(self, scheduler_output, model_runner_output):
        # 6. 查询传输完成
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)
```

### 7.2 外部 token 的 KV 分配

```python
# allocate_slots 考虑外部 token
num_computed = num_local_computed + num_external_computed
# 外部 token 的 KV 从远端加载, 本地只需分配 block 占位
# delay_cache_blocks: 异步加载时延迟缓存 (避免覆盖加载中数据)
```

---

## 8. 与 Worker 的集成

### 8.1 Worker 端 connector

```python
# gpu/kv_connector.py
class KVConnector:
    def __init__(self, vllm_config, kv_caches):
        # 创建 WORKER 角色 connector
        self.connector = KVConnectorFactory.create_connector(
            vllm_config, role=KVConnectorRole.WORKER, ...)

    def pre_forward(self, scheduler_output):
        # 前向前: 启动 KV 加载
        if scheduler_output.kv_connector_metadata:
            self.connector.start_load_kv(forward_context)

    def post_forward(self, scheduler_output):
        # 前向后: 查询传输完成
        finished = self.connector.get_finished(finished_req_ids)
        return KVConnectorOutput(
            finished_recving=finished[0],
            finished_sending=finished[1],
            stats=self.connector.get_kv_connector_stats())
```

### 8.2 逐层 KV 保存

```python
# prefill 实例, 每层 attention 后向时
# attention layer 调用 connector.save_kv_layer
def save_kv_layer(self, layer_name, kv_layer, attn_metadata):
    # 保存该层的 KV 到传输缓冲
    # (NIXL: 注册内存; LMCache: 写入存储)
```

### 8.3 KVConnectorOutput

```python
@dataclass
class KVConnectorOutput:
    finished_recving: list[str]  # 完成加载的请求
    finished_sending: list[str]  # 完成发送的请求
    invalid_block_ids: set[int]  # 加载失败的 block
    kv_connector_stats: KVConnectorStats | None
```

---

## 9. KVConnectorMetadata 流程

### 9.1 Metadata 结构

```python
@dataclass
class KVConnectorMetadata:
    # 本步要传输的请求
    recving_reqs: dict[req_id, RecvingReqMeta]  # 要加载的
    sending_reqs: dict[req_id, SendingReqMeta]  # 要保存的

@dataclass
class RecvingReqMeta:
    req_id: str
    block_ids: list[int]      # 要加载的 block
    num_external_tokens: int  # 外部 token 数
    remote_addr: str          # 源实例地址

@dataclass
class SendingReqMeta:
    req_id: str
    block_ids: list[int]      # 要保存的 block
    num_tokens: int
```

### 9.2 流程

```
Scheduler (SCHEDULER role)
    build_connector_meta(scheduler_output)
    → KVConnectorMetadata (本步传输计划)
    → 放入 scheduler_output.kv_connector_metadata
        │
        │ (随 SchedulerOutput 传到 Worker)
        ▼
Worker (WORKER role)
    pre_forward: start_load_kv(metadata)  (启动加载)
    post_forward: get_finished()          (查询完成)
    → KVConnectorOutput
        │
        │ (随 ModelRunnerOutput 传回 Scheduler)
        ▼
Scheduler
    _update_from_kv_xfer_finished(output)
    (更新请求状态: WAITING_FOR_REMOTE_KVS → WAITING)
```

---

## 10. 配置参数

### 10.1 KVTransferConfig

```python
@dataclass
class KVTransferConfig:
    kv_connector: str | None = None      # connector 名称
    kv_role: str | None = None           # kv_producer/kv_consumer/kv_both
    kv_rank: int | None = None           # 实例 rank
    kv_parallel_size: int = 1            # P-D 并行度
    kv_ip: str = "127.0.0.1"             # 实例 IP
    kv_port: int = 0                     # 实例端口
    kv_buffer_size: float = 1e9          # 传输缓冲 (bytes)
    kv_connector_extra_config: dict = {} # connector 特定配置
    kv_load_failure_policy: str = "recompute"  # 加载失败策略
```

### 10.2 启动参数

```bash
# Prefill 实例
vllm serve model \
    --kv-transfer-config '{"kv_connector": "NixlConnector",
                           "kv_role": "kv_producer",
                           "kv_port": 5643}'

# Decode 实例
vllm serve model \
    --kv-transfer-config '{"kv_connector": "NixlConnector",
                           "kv_role": "kv_consumer",
                           "kv_port": 5644}'
```

### 10.3 加载失败策略

| 策略 | 说明 |
|------|------|
| `recompute` | 加载失败时重新计算（默认，安全） |
| `fail` | 加载失败时请求报错 |

---

## 11. 实际部署案例

### 11.1 典型 P-D 分离部署

```
Frontend (请求路由)
    │
    ├── Prefill 池 (2 实例, 各 8 GPU)
    │   └── kv_role=kv_producer
    │
    └── Decode 池 (4 实例, 各 8 GPU)
        └── kv_role=kv_consumer

请求流程:
1. Frontend 将请求路由到 prefill 实例
2. Prefill 实例计算 KV, 传输到 decode 实例
3. Frontend 将请求路由到 decode 实例
4. Decode 实例加载 KV, 逐 token 生成
```

### 11.2 扩缩容策略

| 瓶颈 | 扩缩 |
|------|------|
| TTFT 高 (prefill 慢) | 加 prefill 实例 |
| TPOT 高 (decode 慢) | 加 decode 实例 |
| 传输慢 | 加网络带宽 / 用 RDMA |

### 11.3 与 EP/TP 的组合

```
Prefill 实例: TP=8 (prefill 计算密集, 大 batch)
Decode 实例:  TP=8 + EP (decode 带宽密集, MoE 用 EP)
```

---

## 12. 常见问题与排查

### 12.1 传输问题

| 现象 | 原因 | 解决 |
|------|------|------|
| KV 加载超时 | 网络慢/带宽不足 | 用 RDMA (NIXL) |
| 加载失败 | block 不匹配 | 检查 `kv_load_failure_policy` |
| 传输量过大 | prefill 太长 | 限制 max_model_len |

### 12.2 状态问题

| 现象 | 原因 | 解决 |
|------|------|------|
| 请求卡在 WAITING_FOR_REMOTE_KVS | KV 未传输完成 | 检查 prefill 实例状态 |
| 请求重复计算 | 外部 KV 查询失败 | 检查 connector 配置 |
| KV 被覆盖 | delay_cache_blocks 未设 | 检查异步加载配置 |

### 12.3 配置问题

| 现象 | 原因 | 解决 |
|------|------|------|
| connector 未创建 | kv_connector 未配置 | 检查 kv_transfer_config |
| 角色错误 | kv_role 配置错 | producer=prefill, consumer=decode |
| 端口冲突 | kv_port 重复 | 每实例唯一端口 |

### 12.4 调试技巧

```bash
# 查看 connector 日志
VLLM_LOGGING_LEVEL=DEBUG vllm serve ...

# 监控传输指标
curl localhost:8000/metrics
# vllm:kv_transfer_time_seconds
# vllm:kv_transfer_bytes
# vllm:num_requests_waiting_for_remote_kvs

# 检查传输状态
# WAITING_FOR_REMOTE_KVS 请求数应短暂 (传输快时)
```

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-09-10  
> **核心文件**：`vllm/distributed/kv_transfer/kv_connector/base.py`, `v1/base.py`, `factory.py`, `vllm/config/kv_transfer.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu/kv_connector.py`
