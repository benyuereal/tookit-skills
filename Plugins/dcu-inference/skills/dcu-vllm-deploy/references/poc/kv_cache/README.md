# KV cache 管理器最小化 POC —— BlockPool + 前缀缓存 (APC)

一个 ~200 行的纯 Python 实现，把 vLLM V1 `KVCacheManager` 的核心机制浓缩出来，
让你**跑起来、看见** block 分配 / 前缀缓存复用 / 引用计数是怎么工作的。

只依赖 Python 标准库，CPU 直接跑，不需要 GPU/DCU：

```bash
python3 mini_kv_cache.py
```

## 它演示了什么

| 机制 | 在 trace 里怎么看 |
|------|------------------|
| **BlockPool（free list）** | `[alloc] A block_table=[0,1,2] 池子占用 3/12`：固定 12 块，free list 管理空闲块 |
| **按 block_size 向上取整** | 11 token 的请求占 `ceil(11/4)=3` 块 |
| **前缀缓存 APC** | `[APC] B 命中前缀缓存，跳过 8 token prefill (复用物理块 [0, 1])`：相同 system prompt 前缀直接复用 |
| **TTFT 降低** | `B: TTFT=3 (无 APC 需 11，省 8 token)`：命中前缀的请求 prefill 工作量大幅减少 |
| **引用计数** | `A 完成 → 物理块 0 仍被其他请求引用，暂不释放`：前缀块被 3 请求引用，前 2 个完成不释放 |
| **最后一个引用才释放** | `C 完成 → 释放物理块 0 (引用计数归零，回到 free list)` |

## 怎么读 trace

```
[APC] A 未命中，全部 11 token 需 prefill
[APC] B 命中前缀缓存，跳过 8 token prefill (复用物理块 [0, 1])
[alloc] B block_table=[0, 1, 3] 池子占用 4/12
[prefill] B 实际算 3 token，TTFT=3
```

- `命中前缀缓存，跳过 N token prefill`：前 N 个 token 的 KV 已在缓存里，不用重算
- `复用物理块 [0, 1]`：B 的 block table 前两项直接指向 A 用过的物理块
- `TTFT=3`：B 只需 prefill 未命中的 3 个 token（时间单位 = prefill token 数）

## 引用计数演示

3 个请求共享相同 8 token（2 个满块）的 system prompt 前缀：

```
前缀物理块 [0, 1] 当前引用计数: [3, 3]
A 完成 → 物理块 0/1 仍被引用，暂不释放
B 完成 → 物理块 0/1 仍被引用，暂不释放
C 完成 → 释放物理块 0/1 (引用计数归零，回到 free list)
```

前 2 个请求完成时前缀块不释放（还有别的请求在用），第 3 个完成才真正释放。
这就是 APC 能跨请求复用前缀 KV 的关键。

## 和真实 vLLM 的差异（简化点）

| 简化 | 真实 vLLM |
|------|----------|
| block 哈希 = 前缀 token 元组 | 真实用 `(parent_hash, token_ids)` 链式哈希（`kv_cache_utils.get_block_hash`），保证前缀语义 |
| 只缓存写满的块（部分块不缓存） | 同，真实也只缓存 full block |
| 引用计数手动维护 | 真实 `KVCacheBlock.ref_count` + `FreeKVCacheBlockQueue`，满块进 free queue 可被 LRU 驱逐 |
| 命中即复用，无驱逐 | 真实池满时按 LRU 驱逐缓存块（`evict_blocks`） |
| 没有抢占 / swap | 真实 KV 满时可 recompute 或 swap 到 CPU |
| 时间单位 = prefill token 数 | 真实 TTFT 受 GPU 算力、batch 大小等影响 |

对应真实源码：`vllm/v1/core/kv_cache_manager.py`（`KVCacheManager.get_computed_blocks` /
`allocate_slots` / `cache_blocks` / `free`）、`vllm/v1/core/block_pool.py`
（`BlockPool` / `BlockHashToBlockMap`）。

想扩展的话，可以加：LRU 驱逐、抢占 recompute、swap 到 CPU、多 KV cache group。
