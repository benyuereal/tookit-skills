# vLLM KV Cache 模块详解

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖 KV Cache 的设计原理、核心数据结构、管理策略与实现细节。

---

## 目录

1. [概述](#1-概述)
2. [核心设计：PagedAttention](#2-核心设计pagedattention)
3. [模块架构](#3-模块架构)
4. [核心数据结构](#4-核心数据结构)
5. [KV Cache 配置与初始化](#5-kv-cache-配置与初始化)
6. [KV Cache 管理器 (KVCacheManager)](#6-kv-cache-管理器-kvcachemanager)
7. [KV Cache 协调器 (KVCacheCoordinator)](#7-kv-cache-协调器-kvcachecoordinator)
8. [单类型 KV Cache 管理器](#8-单类型-kv-cache-管理器)
9. [BlockPool（块池）](#9-blockpool块池)
10. [BlockTable（块表）](#10-blocktable块表)
11. [Prefix Caching（前缀缓存）](#11-prefix-caching前缀缓存)
12. [Hybrid KV Cache（混合 KV Cache）](#12-hybrid-kv-cache混合-kv-cache)
13. [调度器与 KV Cache 的交互](#13-调度器与-kv-cache-的交互)
14. [KV Cache 事件与指标](#14-kv-cache-事件与指标)
15. [配置参数参考](#15-配置参数参考)
16. [调优建议](#16-调优建议)

---

## 1. 概述

KV Cache 是 LLM 推理中最大的显存消耗者。在自回归生成过程中，每个 token 的 Key 和 Value 需要被缓存以避免重复计算。vLLM 通过 **PagedAttention** 机制实现了高效的 KV Cache 管理，核心思想源自操作系统中的虚拟内存分页。

### 关键文件

| 文件 | 职责 |
|------|------|
| `vllm/v1/kv_cache_interface.py` | KV Cache 接口定义（KVCacheSpec, KVCacheConfig 等数据结构） |
| `vllm/v1/core/kv_cache_manager.py` | KV Cache 管理器入口（调度器通过它操作 KV Cache） |
| `vllm/v1/core/kv_cache_coordinator.py` | KV Cache 协调器（协调多个 KV Cache 组） |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 单类型 KV Cache 管理器（按注意力类型分派逻辑） |
| `vllm/v1/core/block_pool.py` | 块池（BlockPool），管理 Block 的分配、释放与缓存 |
| `vllm/v1/core/kv_cache_utils.py` | KV Cache 工具函数（哈希计算、配置生成等） |
| `vllm/v1/core/kv_cache_metrics.py` | KV Cache 指标采集 |
| `vllm/v1/worker/block_table.py` | 通用 BlockTable（CPU 侧） |
| `vllm/v1/worker/gpu/block_table.py` | GPU BlockTables（Triton 实现） |
| `vllm/config/cache.py` | CacheConfig 配置类 |

---

## 2. 核心设计：PagedAttention

### 2.1 设计动机

传统的 KV Cache 为每个请求预分配一块连续显存（max_model_len × kv_size），存在严重的显存浪费：
- 实际生成的序列长度通常远小于 max_model_len
- 不同请求之间的共享前缀（如 system prompt）无法复用

### 2.2 PagedAttention 原理

PagedAttention 将 KV Cache 划分为固定大小的 **Block**（默认 16 或 64 个 token），通过 **Block Table** 映射逻辑位置到物理 Block：

```
逻辑视角（每个请求）:       物理视角（GPU 显存）:
┌─────────────────┐        ┌───┬───┬───┬───┬───┬───┬───┐
│ Block 0: tokens │        │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │
│   0-15          │──┐     ├───┼───┼───┼───┼───┼───┼───┤
├─────────────────┤  │     │   │   │ A │ B │   │ C │   │
│ Block 1: tokens │  │     └───┴───┴─▲─┴─▲─┴───┴─▲─┴───┘
│   16-31         │──┤              │   │       │
├─────────────────┤  │     Request A:  2,  3
│ Block 2: tokens │  │     Request B:  3,  4,  5
│   32-47         │──┘
└─────────────────┘        Block Table 映射 逻辑块 → 物理块
```

### 2.3 Block Table 工作流

```
请求 token_ids → Block Table → 物理 Block ID → GPU 显存地址
     [0..15]  →   [0] → 7    → block_7
     [16..31] →   [1] → 3    → block_3
     [32..47] →   [2] → 12   → block_12
```

Attention Kernel 通过 `block_tables` 参数获取物理 Block 地址，实现非连续显存访问。

---

## 3. 模块架构

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────┐
│                    Scheduler                         │
│  - schedule() 决策                                   │
│  - 调用 KVCacheManager 分配/释放                      │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                 KVCacheManager                       │
│  - allocate_slots() / free() / get_computed_blocks() │
│  - 统一接口，隐藏内部实现                              │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              KVCacheCoordinator                      │
│  - UnitaryKVCacheCoordinator  (单一注意力类型)        │
│  - HybridKVCacheCoordinator   (混合注意力类型)        │
│  - KVCacheCoordinatorNoPrefixCache (无前缀缓存)       │
│  - 协调多个 SingleTypeKVCacheManager                 │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│          SingleTypeKVCacheManager (× N)              │
│  - FullAttentionManager      (全注意力)              │
│  - SlidingWindowManager      (滑动窗口)              │
│  - ChunkedLocalAttentionManager (分块局部注意力)      │
│  - MambaManager              (Mamba/SSM)             │
│  - CrossAttentionManager     (交叉注意力)            │
│  - SinkFullAttentionManager  (Sink注意力)            │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                   BlockPool                          │
│  - FreeKVCacheBlockQueue (LRU 空闲链表)              │
│  - BlockHashToBlockMap   (前缀缓存哈希表)             │
│  - get_new_blocks() / free_blocks() / touch()        │
└─────────────────────────────────────────────────────┘
```

### 3.2 双端架构

KV Cache 管理分为**调度端（CPU）**和**执行端（GPU）**：

| 端 | 位置 | 职责 |
|----|------|------|
| **调度端** | EngineCore (CPU) | 逻辑管理：Block 分配/释放、前缀缓存查找、引用计数 |
| **执行端** | GPU Worker | 物理执行：BlockTable 张量、slot_mapping 计算、attention kernel |

两端通过 `block_ids` 列表传递信息：调度端决定哪些物理 Block 被使用，执行端将 `block_ids` 写入 GPU 端的 BlockTable 张量。

---

## 4. 核心数据结构

### 4.1 KVCacheSpec（KV Cache 规格）

定义每层 KV Cache 的格式。位于 `vllm/v1/kv_cache_interface.py`。

```python
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int                    # Block 大小（tokens）
    page_size_bytes: int               # 每页（block）的字节数
    def max_memory_usage_bytes(vllm_config) -> int  # 最大显存用量
```

**子类型**：

| 类型 | 用途 | 关键字段 |
|------|------|---------|
| `FullAttentionSpec` | 标准全注意力 | `num_kv_heads`, `head_size`, `dtype`, `sliding_window` |
| `MLAAttentionSpec` | DeepSeek MLA | 继承 FullAttentionSpec，特殊 page_size 计算 |
| `SlidingWindowSpec` | 滑动窗口注意力 | `sliding_window` |
| `ChunkedLocalAttentionSpec` | 分块局部注意力 | `attention_chunk_size` |
| `MambaSpec` | Mamba/SSM 模型 | `shapes`, `dtypes`, `mamba_type`, `mamba_cache_mode` |
| `CrossAttentionSpec` | 交叉注意力（编码器-解码器） | 继承 AttentionSpec |
| `SinkFullAttentionSpec` | Sink 注意力 | `sink_len` |
| `UniformTypeKVCacheSpecs` | 统一类型的多层规格 | `kv_cache_specs: dict` |

**page_size_bytes 计算**（以 FullAttentionSpec 为例）：

```
real_page_size_bytes = block_size × num_kv_heads × (head_size + head_size_v) × dtype_size
```

KV 的 K 和 V 各占一半空间，所以：
```
page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size  (head_size_v == head_size 时)
```

### 4.2 KVCacheBlock（KV Cache 块）

位于 `vllm/v1/core/kv_cache_utils.py`。

```python
@dataclass
class KVCacheBlock:
    block_id: int                              # 物理 Block ID (0 ~ num_gpu_blocks-1)
    ref_cnt: int = 0                           # 引用计数
    _block_hash: BlockHashWithGroupId | None   # 前缀缓存哈希
    prev_free_block: KVCacheBlock | None       # 空闲链表前驱
    next_free_block: KVCacheBlock | None       # 空闲链表后继
    is_null: bool = False                      # 是否为 null 占位块
```

**生命周期**：
```
[FREE] → get_new_blocks() → [ALLOCATED] (ref_cnt=1)
                                  │
                                  ├→ touch() → ref_cnt++  (被其他请求共享)
                                  ├→ free() → ref_cnt-- → [FREE] (返回空闲链表)
                                  └→ cache_blocks() → block_hash 被设置 → [CACHED]
```

### 4.3 FreeKVCacheBlockQueue（空闲块队列）

基于双向链表的 LRU 空闲队列，位于 `vllm/v1/core/kv_cache_utils.py`。

```python
class FreeKVCacheBlockQueue:
    fake_free_list_head   # 哨兵头节点
    fake_free_list_tail   # 哨兵尾节点
    num_free_blocks       # 空闲块计数

    def popleft() -> KVCacheBlock          # 从头部取一个空闲块
    def popleft_n(n) -> list[KVCacheBlock] # 取 n 个
    def remove(block)                      # O(1) 从中间移除
    def append(block)                      # 追加到尾部
    def append_n(blocks)                   # 批量追加
```

**LRU 排序策略**：
1. 最近最少使用的 Block 在队列头部（优先被分配）
2. 如果两个 Block 最后访问时间相同，包含更多哈希 token 的 Block 更靠前

### 4.4 KVCacheBlocks（分配结果）

位于 `vllm/v1/core/kv_cache_manager.py`。

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[i][j] = 第 i 个 kv_cache_group 的第 j 个 Block
```

这是 `KVCacheManager` 对外暴露的统一接口，隐藏内部 `SingleTypeKVCacheManager` 的细节。

### 4.5 KVCacheConfig（KV Cache 配置）

位于 `vllm/v1/kv_cache_interface.py`。

```python
@dataclass
class KVCacheConfig:
    num_blocks: int                     # Block 总数
    kv_cache_tensors: list[KVCacheTensor]  # 张量初始化描述
    kv_cache_groups: list[KVCacheGroupSpec] # KV Cache 组
```

### 4.6 KVCacheGroupSpec（KV Cache 组规格）

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]       # 该组包含的层名称
    kv_cache_spec: KVCacheSpec   # 该组的 KV Cache 规格
```

### 4.7 BlockHash 相关类型

```python
BlockHash = NewType("BlockHash", bytes)                         # 单个 Block 的哈希
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)   # 哈希 + 组 ID
BlockHashList = list[BlockHash] | BlockHashListWithBlockSize    # Block 哈希列表
```

---

## 5. KV Cache 配置与初始化

### 5.1 初始化流程

```
get_kv_cache_configs()
    │
    ├── 1. 合并所有 worker 的 kv_cache_specs
    │
    ├── 2. get_kv_cache_groups()
    │       ├── 统一类型 → _get_kv_cache_groups_uniform_spec()
    │       ├── 混合类型 (uniform page size) → _get_kv_cache_groups_uniform_page_size()
    │       └── 混合类型 (不同 page size) → unify_kv_cache_spec_page_size()
    │
    ├── 3. 自动推导 max_model_len（如果 original_max_model_len == -1）
    │
    ├── 4. 检查显存是否足够
    │
    ├── 5. get_kv_cache_config_from_groups() → 计算 num_blocks
    │
    └── 6. 取所有 worker 的最小 num_blocks 并缩减 tensor size
```

### 5.2 num_blocks 计算

```python
num_blocks = available_memory // page_size // num_layers
```

对于混合模型（hybrid），使用 group_size（每组中的层数）代替 num_layers：
```python
num_blocks = available_memory // page_size // group_size
```

### 5.3 KV Cache 张量初始化

```
Worker.initialize_kv_cache(kv_cache_config)
    │
    ├── 创建 BlockTables（GPU 端块表）
    │
    ├── 初始化 Attention 后端
    │
    ├── init_kv_cache() → 为每层分配 GPU Tensor
    │       kv_cache_tensors 描述如何共享内存：
    │       - 同一 KVCacheTensor 的层共享同一个 Tensor
    │       - 不同 KVCacheTensor 的层使用不同 Tensor
    │       size = page_size × num_blocks
    │
    └── 初始化 KV Connector（用于 KV 传输场景）
```

### 5.4 KV Cache Groups 分组策略

#### 场景 1：统一类型（最常见）

所有层使用相同注意力类型（如全注意力），所有层放入一个组，共享同一 Block Table。

```
Model: 32 layers, all FullAttention
Groups: [(layer_0, layer_1, ..., layer_31)]
```

#### 场景 2：混合类型（Hybrid）

模型包含多种注意力类型（如 Gemma3：5 滑动窗口 + 1 全注意力），按模式分组。

```
Model: 5 sw + 1 full (pattern repeated)
Groups: 
  Group 0 (full):  [layer_5, layer_11, layer_17, ...]
  Group 1 (sw):    [layer_0, layer_6, layer_12, ...]
  Group 2 (sw):    [layer_1, layer_7, layer_13, ...]
  ...
```

每组使用独立的 Block Table，但共享物理内存池。Group size 为所有类型中最少的层数。

### 5.5 Block Size 选择

| 场景 | block_size | 说明 |
|------|-----------|------|
| 默认（非 flash_attn） | 16 | 通用默认值 |
| FlashAttention + FlashMLA | 64 | 更大的 block 减少 Block Table 开销 |
| 混合模型（不同 page size） | 自动对齐 | 通过增大较小层的 block_size 统一 page size |

---

## 6. KV Cache 管理器 (KVCacheManager)

位于 `vllm/v1/core/kv_cache_manager.py`，是调度器与 KV Cache 之间的核心接口。

### 6.1 初始化

```python
class KVCacheManager:
    def __init__(
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,       # 启用前缀缓存
        use_eagle: bool = False,           # EAGLE 投机解码
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,           # Decode Context Parallelism
        pcp_world_size: int = 1,           # Prefill Context Parallelism
    )
```

### 6.2 核心方法

#### get_computed_blocks(request) → (KVCacheBlocks, int)

查询前缀缓存命中情况。

```
请求的 block_hashes → find_longest_cache_hit() → 命中的 Block 列表 + 命中 token 数
```

- 如果 `enable_caching=False` 或 `request.skip_reading_prefix_cache=True`，直接返回空
- `max_cache_hit_length = request.num_tokens - 1`（最后 1 个 token 必须重新计算以获得 logits）

#### allocate_slots(request, num_new_tokens, ...) → KVCacheBlocks | None

为请求分配 KV Cache slot。这是最核心的方法。

**参数**：
| 参数 | 说明 |
|------|------|
| `num_new_tokens` | 需要新计算的 token 数 |
| `num_new_computed_tokens` | 前缀缓存新命中的 token 数 |
| `new_computed_blocks` | 前缀缓存命中的 Block |
| `num_lookahead_tokens` | 投机解码的 lookahead token |
| `num_external_computed_tokens` | 外部 KV 连接器缓存的 token 数 |
| `delay_cache_blocks` | 是否延迟缓存（P/D 分离场景） |
| `num_encoder_tokens` | 编码器 token 数（交叉注意力） |

**Block 布局**：
```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
           |           < to be allocated >                            |
----------------------------------------------------------------------
           | < to be cached >                                        |
----------------------------------------------------------------------
|    prefix-cached tokens (vLLM)     | ext cached by connector       |
----------------------------------------------------------------------
```

**三步分配流程**：

1. **释放不必要的 Block**：调用 `remove_skipped_blocks()` 释放滑动窗口外的 token
2. **处理前缀 token**：`allocate_new_computed_blocks()` 处理前缀缓存命中 + 外部 token
3. **分配新 Block**：`allocate_new_blocks()` 为待计算 token 分配新 Block

返回 `None` 表示分配失败（空闲 Block 不足），触发抢占（preemption）。

#### free(request)

释放请求的所有 Block。**按逆序释放**，确保尾部 Block 优先被回收（LRU 语义）。

#### get_num_common_prefix_blocks(running_request_id) → list[int]

计算所有运行中请求的公共前缀 Block 数（用于 Cascade Attention 优化）。

一个 Block 是公共前缀当且仅当：`block.ref_cnt == len(req_to_blocks)`（所有请求都引用它）。

#### reset_prefix_cache() → bool

重置前缀缓存（RLHF 权重更新后使用）。

#### cache_blocks(request, num_computed_tokens)

缓存已完成计算的 Block（标记 block_hash，加入哈希表）。

---

## 7. KV Cache 协调器 (KVCacheCoordinator)

位于 `vllm/v1/core/kv_cache_coordinator.py`。

### 7.1 三种实现

通过工厂函数 `get_kv_cache_coordinator()` 选择：

```python
def get_kv_cache_coordinator(...) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(...)  # 无前缀缓存
    if len(kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(...)         # 单一注意力类型
    return HybridKVCacheCoordinator(...)              # 混合注意力类型
```

### 7.2 UnitaryKVCacheCoordinator

适用于只有一种 KV Cache 类型的模型（最常见）。

**find_longest_cache_hit 算法**：
```
从左到右扫描 block_hashes：
  for each block_hash in block_hashes:
      if cached_block := block_pool.get_cached_block(block_hash):
          hit_blocks.append(cached_block)
      else:
          break  # 遇到第一个未缓存的 block 就停止
```

- 如果使用 EAGLE，丢弃最后一个命中的 Block（需要重新计算以获得 hidden states）
- 对齐到 `alignment_tokens` 的倍数

### 7.3 HybridKVCacheCoordinator

适用于混合注意力类型的模型（如 Gemma3 的 full + sliding_window）。

**find_longest_cache_hit 算法**（迭代固定点算法）：

```
1. 全注意力（FullAttention）先扫描，获得初始 hit_length
2. 其他注意力类型（如 SlidingWindow）基于 hit_length 查找
3. 如果其他类型缩短了 hit_length，全注意力截断到新长度
4. 收敛后返回最终结果
```

对于简单的混合模型（1 full + 1 other），只需一轮迭代即可。

### 7.4 KVCacheCoordinatorNoPrefixCache

禁用前缀缓存时使用，`find_longest_cache_hit` 始终返回空。

---

## 8. 单类型 KV Cache 管理器

位于 `vllm/v1/core/single_type_kv_cache_manager.py`。

### 8.1 管理器类型映射

```python
spec_manager_map = {
    FullAttentionSpec:          FullAttentionManager,
    TQFullAttentionSpec:       FullAttentionManager,
    MLAAttentionSpec:          FullAttentionManager,
    SlidingWindowSpec:         SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec:                 MambaManager,
    CrossAttentionSpec:        CrossAttentionManager,
    SinkFullAttentionSpec:     SinkFullAttentionManager,
}
```

### 8.2 FullAttentionManager

最常用的管理器，处理标准全注意力层。

**find_longest_cache_hit**：从左到右顺序扫描，遇到第一个未缓存 Block 就停止。

**get_num_common_prefix_blocks**：遍历 Block 列表，`ref_cnt == len(req_to_blocks)` 的连续 Block 都是公共前缀。

**get_num_skipped_tokens**：始终返回 0（全注意力不跳过任何 token）。

### 8.3 SlidingWindowManager

处理滑动窗口注意力层。

**get_num_skipped_tokens 算法**：
```
例：sliding_window=4, num_computed_tokens=7
Tokens:  [ 0  1  2  3  4  5  6  7 ]
                               ^ next token
                     |-----------| sliding window for next token
          |--skipped---|
get_num_skipped_tokens(7) = 7 - 4 + 1 = 4
```

被跳过的 Block 会被替换为 `null_block`（占位，不消耗实际显存）。

**find_longest_cache_hit**：从右向左扫描，需要连续的 `sliding_window_contiguous_blocks` 个 Block 命中才算前缀缓存命中。

```
例：block_size=4, sliding_window=8
sliding_window_contiguous_blocks = ceil((8-1)/4) = 2
需要找到连续的 2 个缓存命中 Block
结果: [NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]
```

### 8.4 ChunkedLocalAttentionManager

处理分块局部注意力（如 LLaMA 4）。

**get_num_skipped_tokens**：
```
例：chunk_size=8, num_computed_tokens=13
跳过 = (13 // 8) * 8 = 8 个 token
```

**find_longest_cache_hit**：窗口外的 Block 直接标记为 null_block（已计算），窗口内正常进行前缀缓存查找。

### 8.5 MambaManager

处理 Mamba/SSM 层。

三种缓存模式：
- **"none"**：无前缀缓存，只保留最后 1 个 token 的状态
- **"all"**：缓存所有 `i * block_size` 位置的 Mamba 状态
- **"align"**：只缓存每个调度步最后一个 token + `i * block_size` 位置的状态

### 8.6 CrossAttentionManager

处理编码器-解码器模型的交叉注意力（如 Whisper）。

- 不支持前缀缓存（每个请求的编码器状态不同）
- 一次性静态分配 Block（基于编码器输入长度）

### 8.7 SinkFullAttentionManager

处理 Sink 注意力（保留开头的 "sink token"）。

- 初始化时从空闲队列中取出 `sink_len // block_size` 个 Block 作为 sink blocks
- 这些 sink blocks 在整个服务生命周期中不会被释放

---

## 9. BlockPool（块池）

位于 `vllm/v1/core/block_pool.py`，是 KV Cache 物理 Block 的管理核心。

### 9.1 核心属性

```python
class BlockPool:
    blocks: list[KVCacheBlock]                      # 所有 Block
    free_block_queue: FreeKVCacheBlockQueue          # 空闲块 LRU 队列
    cached_block_hash_to_block: BlockHashToBlockMap  # 前缀缓存哈希 → Block
    null_block: KVCacheBlock                         # 占位 null block (block_id=0)
    kv_event_queue: list[KVCacheEvent]               # KV Cache 事件队列
```

### 9.2 核心方法

#### get_new_blocks(num_blocks) → list[KVCacheBlock]

从空闲队列取出 Block：
1. 从 `free_block_queue` 头部 pop n 个 Block
2. 对每个 Block：
   - 如果开启了前缀缓存，调用 `_maybe_evict_cached_block()`（清除其旧哈希）
   - 设置 `ref_cnt = 1`

#### touch(blocks)

增加 Block 的引用计数（前缀缓存命中时）：
- `ref_cnt == 0` 的 Block 从空闲队列中移除（它不再是 eviction candidate）
- `ref_cnt += 1`

#### free_blocks(ordered_blocks)

释放 Block（请求完成或被抢占时）：
- 每个 Block 的 `ref_cnt -= 1`
- `ref_cnt == 0` 且非 null 的 Block 追加回空闲队列尾部

#### cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id)

将已满的 Block 加入前缀缓存：
1. 计算每个 Block 的哈希值（由 Request 对象的 `block_hashes` 提供）
2. 设置 `block.block_hash`
3. 插入 `cached_block_hash_to_block` 哈希表

#### get_cached_block(block_hash, kv_cache_group_ids) → list[KVCacheBlock] | None

根据 Block 哈希查找缓存。

#### evict_blocks(block_ids)

按 Block ID 驱逐缓存（KV 连接器报告外部驱逐时使用）。

#### reset_prefix_cache()

清空前缀缓存哈希表，重置所有 Block 的哈希。

### 9.3 BlockHashToBlockMap

前缀缓存的核心哈希表：

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

- **单 Block**：大多数情况下，一个哈希只对应一个 Block
- **多 Block**：当有重复 Block 时（目前不进行去重），升级为 dict

### 9.4 驱逐策略

当 `get_new_blocks()` 从空闲队列取出 Block 时，如果该 Block 之前被缓存过：

1. 从 `cached_block_hash_to_block` 中移除
2. 重置 `block.block_hash = None`
3. 触发 `BlockRemoved` 事件（如果开启了 KV 事件）

---

## 10. BlockTable（块表）

### 10.1 通用 BlockTable（CPU 端）

位于 `vllm/v1/worker/block_table.py`。

```python
class BlockTable:
    block_table: CpuGpuBuffer      # [max_num_reqs, max_num_blocks_per_req] int32
    num_blocks_per_row: np.ndarray # [max_num_reqs] int32
    slot_mapping: CpuGpuBuffer     # [max_num_batched_tokens] int64
```

**核心方法**：

- `append_row(block_ids, row_idx)`：追加 Block ID 到指定请求的行
- `add_row(block_ids, row_idx)`：设置（覆盖）指定请求的 Block ID
- `compute_slot_mapping(req_indices, positions)`：计算 slot_mapping
  ```
  block_number = block_table[req_idx][position // block_size]
  block_offset = position % block_size
  slot_mapping = block_number * block_size + block_offset
  ```
- `commit_block_table(num_reqs)`：将 CPU 端数据同步到 GPU

**Hybrid Block 支持**：当 `kernel_block_size ≠ block_size` 时，进行 Block ID 映射：
```
例：block_size=32, kernel_block_size=16
kv_manager_block_id 0 → kernel block ids [0, 1]
kv_manager_block_id 1 → kernel block ids [2, 3]
```

### 10.2 MultiGroupBlockTable

```python
class MultiGroupBlockTable:
    block_tables: list[BlockTable]  # 每个 KV Cache 组一个
```

支持多个 KV Cache 组的 BlockTable，每个组有独立的 block_size 和 Block ID 空间。

### 10.3 GPU BlockTables（执行端）

位于 `vllm/v1/worker/gpu/block_table.py`，使用 Triton kernel 实现高性能操作。

```python
class BlockTables:
    block_tables: list[StagedWriteTensor]  # 支持分阶段写入
    input_block_tables: list[torch.Tensor] # 模型前向传播使用的 Block Table
    slot_mappings: torch.Tensor            # [num_kv_cache_groups, max_num_batched_tokens]
```

**核心方法**：

- `append_block_ids(req_index, new_block_ids, overwrite)`：分阶段写入 Block ID
- `apply_staged_writes()`：批量提交所有分阶段写入
- `gather_block_tables(idx_mapping)`：Triton kernel 从所有组收集 Block Table
- `compute_slot_mappings(idx_mapping, query_start_loc, positions)`：Triton kernel 计算 slot mappings

**Triton Kernel 实现**：
- `_gather_block_tables_kernel`：并行从每组、每个请求收集 Block Table
- `_compute_slot_mappings_kernel`：并行计算每组、每个请求的 slot_mapping
  ```
  slot_id = block_number * block_size + (position % block_size)
  ```

---

## 11. Prefix Caching（前缀缓存）

### 11.1 工作原理

```
请求 A: "Hello, how are you today?"  → block_hashes: [H0, H1]
请求 B: "Hello, how are you doing?"  → block_hashes: [H0, H1, H2]
                                        ↑    ↑
                                    命中！直接复用 Block
```

1. 每个请求的 prompt token_ids 被分块计算哈希
2. 哈希链通过 `parent_block_hash → current_block_hash` 连接
3. 查找时从左到右匹配哈希链，遇到第一个不匹配就停止
4. 命中的 Block 通过 `touch()` 增加引用计数，防止被驱逐

### 11.2 Block 哈希计算

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH  # 随机种子
    return hash_function((parent_block_hash, curr_block_token_ids, extra_keys))
```

**额外哈希键（extra_keys）**：
- **多模态请求**：`(mm_hash, start_offset)` 表示 Block 中包含的多模态输入
- **LoRA 请求**：LoRA adapter 名称
- **Cache Salt**：请求指定的缓存盐值
- **Prompt Embeddings**：embedding 数据的字节表示

### 11.3 哈希算法

| 算法 | 说明 |
|------|------|
| `sha256` | 默认，使用 Pickle 序列化 + SHA-256 |
| `sha256_cbor` | 使用 CBOR 序列化 + SHA-256（跨语言兼容） |
| `xxhash` | 使用 Pickle + xxHash（更快，非加密） |
| `xxhash_cbor` | 使用 CBOR + xxHash（更快，可复现） |

### 11.4 EAGLE 投机解码的兼容性

使用 EAGLE 时，`find_longest_cache_hit` 会丢弃最后一个命中的 Block：
- 原因：EAGLE 的 draft head 需要 target model 最后一个 token 的 hidden states
- 因此最后一个 Block 必须重新计算

### 11.5 前缀缓存的限制

- 不支持部分 Block 缓存（必须整 Block 对齐）
- 混合模型不同 block_size 时需要哈希粒度对齐
- 使用 `PYTHONHASHSEED` 确保跨进程可复现的哈希

---

## 12. Hybrid KV Cache（混合 KV Cache）

### 12.1 场景

某些模型同时包含多种注意力层：

| 模型 | 注意力类型 |
|------|-----------|
| Gemma 3 | Full Attention + Sliding Window |
| LLaMA 4 | Full Attention + Chunked Local Attention |
| GPT-OSS-20B + Eagle | Full Attention + Sliding Window |

### 12.2 设计假设

1. **物理内存 per block 必须相同**（避免内存碎片）
2. **block_size 可以不同**，但必须是哈希 block_size 的倍数
3. **物理内存 per token per layer 相同**（可放宽）
4. **每组层数相同**（可放宽，但需要 padding）
5. **组内注意力类型相同**（一个例外：`disable_hybrid_kv_cache_manager` 时）

### 12.3 统一 Page Size

当不同注意力类型的 page_size_bytes 不同时：

```
layer A: page_size = 512 bytes, block_size = 16
layer B: page_size = 256 bytes, block_size = 16

→ 将 layer B 的 block_size 增大到 32 → page_size = 512 bytes
```

### 12.4 Padding 策略

```
例：8 full + 9 sw
→ group_size = 8
→ padded: 8 full + 16 sw (浪费 ≤ 12.5% KV Cache 显存)
→ groups: (full.0, sw.0, sw.8), (full.1, sw.1, sw.9), ..., (padding, sw.7, sw.15)
```

### 12.5 禁用混合管理器

设置 `disable_hybrid_kv_cache_manager=True` 时：
- 将 SlidingWindowSpec 和 ChunkedLocalAttentionSpec 转换为 FullAttentionSpec
- 所有层使用全注意力管理（浪费滑动窗口外的显存，但计算仍优化）

---

## 13. 调度器与 KV Cache 的交互

调度器 (`vllm/v1/core/sched/scheduler.py`) 是 KV Cache 的主要调用者。

### 13.1 调度流程中的 KV Cache 操作

```
schedule()
  │
  ├── 1. 处理 WAITING 请求
  │     ├── get_computed_blocks(request)     ← 查询前缀缓存
  │     ├── allocate_slots(request, ...)      ← 分配 KV Cache slot
  │     └── 如果分配失败 → Preemption          ← 抢占已有请求
  │
  ├── 2. 处理 RUNNING 请求
  │     ├── allocate_slots(request, ...)      ← 为下一个 token 分配
  │     └── 如果分配失败 → Preemption
  │
  └── 3. 构建 SchedulerOutput
        ├── block_table_ids (每个请求的 Block ID 列表)
        ├── num_common_prefix_blocks          ← Cascade Attention
        └── preempted_req_ids
```

### 13.2 Preemption（抢占）

当 `allocate_slots()` 返回 `None`（空闲 Block 不足）时：

**FCFS 策略**：抢占最后一个 running 请求
**Priority 策略**：抢占最低优先级的请求

抢占操作：
1. 从 `running` 列表中移除被抢占的请求
2. 调用 `kv_cache_manager.free(request)` 释放其 Block
3. 被抢占的请求回到 `waiting` 队列，下次调度时重新计算（利用前缀缓存加速）

### 13.3 Block Table ID 传递

```python
# 调度器输出
scheduler_output.block_table_ids = [
    [0, 5, 12],    # Request A: 3 个 block
    [3, 7, 9, 15], # Request B: 4 个 block
]

# Worker 端
for req_idx, block_ids in enumerate(scheduler_output.block_table_ids):
    block_tables.append_row(block_ids, req_idx)
```

### 13.4 Cascade Attention 优化

`get_num_common_prefix_blocks()` 找到所有请求共享的公共前缀 Block，用于 Cascade Attention 优化（公共前缀的 KV 只需计算一次）。

---

## 14. KV Cache 事件与指标

### 14.1 KV Cache Events

位于 `vllm/distributed/kv_events.py`。

| 事件 | 触发时机 |
|------|---------|
| `BlockStored` | Block 被缓存（加入哈希表） |
| `BlockRemoved` | Block 被驱逐（从哈希表移除） |
| `AllBlocksCleared` | 前缀缓存被重置 |

### 14.2 KVCacheMetricsCollector

位于 `vllm/v1/core/kv_cache_metrics.py`。

**采样指标**（`sample_rate` 控制采样率，默认 1%）：
- `lifetime_seconds`：Block 从创建到驱逐的存活时间
- `idle_seconds`：Block 的最后访问到驱逐的空闲时间
- `reuse_gaps_seconds`：Block 多次访问之间的间隔

**配置**：
```python
ObservabilityConfig:
    kv_cache_metrics: bool = False       # 启用指标采集
    kv_cache_metrics_sample: float = 0.01 # 采样率
```

### 14.3 PrefixCacheStats

```python
class PrefixCacheStats:
    num_tokens: int      # 请求的总 token 数
    num_hits: int        # 前缀缓存命中的 token 数
    preempted: bool      # 是否来自被抢占后重新调度的请求
    reset: bool          # 前缀缓存是否被重置
```

---

## 15. 配置参数参考

### 15.1 CacheConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 16 / 64 (flash_attn+flash_mla) | Block 大小（tokens） |
| `gpu_memory_utilization` | 0.9 | GPU 显存利用率 |
| `swap_space` | 4 | CPU 交换空间（GiB） |
| `cache_dtype` | "auto" | KV Cache 数据类型 |
| `enable_prefix_caching` | True | 启用前缀缓存 |
| `prefix_caching_hash_algo` | "sha256" | 前缀缓存哈希算法 |
| `cpu_offload_gb` | 0 | CPU 卸载大小（GiB） |
| `kv_cache_memory_bytes` | None | 精确指定 KV Cache 大小（覆盖 gpu_memory_utilization） |
| `num_gpu_blocks_override` | None | 覆盖 num_gpu_blocks（测试用） |
| `mamba_block_size` | None | Mamba 专用 Block 大小 |
| `mamba_cache_mode` | "none" | Mamba 缓存模式（none/all/align） |
| `kv_offloading_size` | None | KV Cache 卸载缓冲区大小（GiB） |
| `kv_offloading_backend` | "native" | 卸载后端（native/lmcache） |

### 15.2 相关调度器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_chunked_prefill` | True | 分块预填充（影响 KV Cache 分配模式） |
| `disable_hybrid_kv_cache_manager` | False | 禁用混合 KV Cache 管理器 |
| `max_num_seqs` | 128 | 最大并发序列数（决定 Block Table 行数） |

### 15.3 相关环境变量

```bash
VLLM_USE_FLASH_ATTN_PA=1       # 影响默认 block_size
VLLM_USE_FLASH_MLA=1           # 影响默认 block_size
VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0  # 使用整数 Block 哈希（兼容性）
PYTHONHASHSEED=<seed>          # Block 哈希可复现性
```

---

## 16. 调优建议

### 16.1 显存计算

```
KV Cache 显存 ≈ num_layers × 2 × num_kv_heads × head_size × max_model_len × dtype_size

例：Qwen3-32B (64 layers, 8 kv_heads, 128 head_size, 32768 max_len, BF16)
= 64 × 2 × 8 × 128 × 32768 × 2
= 64 × 2 × 8 × 128 × 32768 × 2 bytes
≈ 8.59 GB (per GPU, TP=1)
```

### 16.2 Block Size 选择

| block_size | 优点 | 缺点 |
|-----------|------|------|
| 16（小） | 更少的内存碎片，更好的 prefix cache 命中 | 更大的 Block Table，更多的 kernel launch |
| 64（大） | 更小的 Block Table，更少的 kernel launch | 更多的内存碎片（内部碎片） |

**建议**：
- FlashAttention 场景使用默认 64
- 其他场景使用默认 16
- 一般不需要手动调整

### 16.3 前缀缓存优化

1. **保持 `enable_prefix_caching=True`**（默认），显著减少重复计算
2. **相同的 system prompt** 会被自动共享
3. **设置 `PYTHONHASHSEED`** 确保跨进程可复现的哈希
4. **多模态/LoRA 请求** 自动携带额外哈希键

### 16.4 混合模型优化

- 默认启用混合 KV Cache 管理器（节省滑动窗口外的显存）
- 如果遇到兼容性问题，可设置 `--disable-hybrid-kv-cache-manager`
- 注意 padding 造成的显存浪费（通常 ≤ 12.5%）

### 16.5 抢占调优

- 增大 `gpu_memory_utilization` 可减少抢占频率
- 增大 `max_num_seqs` 前确认 KV Cache 容量充足
- 使用 `num_gpu_blocks_override` 测试不同 Block 数量下的抢占行为

### 16.6 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| OOM | KV Cache 超限 | 降低 `max_model_len`、`max_num_seqs` 或 `gpu_memory_utilization` |
| 频繁抢占 | 并发请求太多 | 增大 KV Cache 或减少 `max_num_seqs` |
| 前缀缓存命中率低 | Block 对齐问题 | 确保 prompt 前缀是 block_size 的整数倍 |
| 混合模型显存浪费 | padding 开销 | 接受 ≤12.5% 或禁用混合管理器 |

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-07-14  
> **核心文件**：`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/kv_cache_coordinator.py`, `vllm/v1/core/single_type_kv_cache_manager.py`, `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/kv_cache_interface.py`, `vllm/v1/worker/gpu/block_table.py`