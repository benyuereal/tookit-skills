# PagedAttention 最小化 POC —— KV cache 分页是怎么工作的

一个 ~200 行的纯 Python 实现，把 PagedAttention 的核心机制浓缩出来，
让你**跑起来、看见** KV cache 分页 / block table / 分页 attention 是怎么工作的。

只依赖 Python 标准库，CPU 直接跑，不需要 GPU/DCU：

```bash
python3 mini_paged_attention.py
```

## 它演示了什么

| 机制 | 在 trace 里怎么看 |
|------|------------------|
| **物理 block 池** | `[alloc] A 逻辑块 0 -> 物理块 0`：显存切成 16 个固定 block（每块 4 token 的 K/V 槽位），所有请求共享 |
| **Block Table** | `A block_table = [0, 1]`、`B block_table = [2, 3]`：逻辑块 → 物理块的映射，物理块任意散布 |
| **分页写入** | `[write] A token 4 -> 物理块 1 slot 0`：token 按 `i//4` 查 block table 得物理块、`i%4` 得块内槽位 |
| **分页 attention** | `[attention] A: 从物理块 [0, 1] gather 6 个 token 的 K/V`：每个请求只 gather 自己 block table 指向的物理块 |
| **KV 交错、各取各的** | A 的 KV 在块 0/1，B 的 KV 在块 2/3，交错存放但 attention 互不污染 |
| **对比不分页** | 最后打印：不分页预留 max_len 浪费 50%，分页按实际 token 数只占一半 |

## 怎么读 trace

```
[alloc] A 逻辑块 0 -> 物理块 0  (free list 剩 15 块)
[write] A token 4 -> 物理块 1 slot 0
[attention] A: 从物理块 [0, 1] gather 6 个 token 的 K/V
    token 5: score=  9.00  weight= 0.777
    output = [5.714, 0.000]
```

- `逻辑块 i -> 物理块 pb`：block table 把请求的第 i 个逻辑块映射到物理块 pb
- `token i -> 物理块 pb slot s`：`pb = block_table[i//4]`，`s = i%4`
- `从物理块 [...] gather`：attention 时按 block table 把分散的物理块里的 K/V 收集起来算

## 核心收益

不分页时每个请求要预留 `max_len` 的连续空间，实际只用一小部分 → 显存浪费。
分页后 KV 按 block 分配、可任意散布，利用率从 ~20-40% 提到 98%+，
从而能塞下更大的 batch。这就是 PagedAttention 的核心价值。

## 和真实 vLLM 的差异（简化点）

| 简化 | 真实 vLLM |
|------|----------|
| K/V 用 2 维小向量，纯 Python 列表 | 真实是 GPU 上的高维张量，attention 用 CUDA kernel |
| attention 是纯 Python 循环 gather + 点积 | 真实用 PagedAttention CUDA kernel，按 block 分块算，支持 flash attention |
| block 池 16 块、block_size=4 | 真实 block_size 通常 16，池子大小由显存决定 |
| 没有 prefix cache | 真实有 APC，相同前缀复用 block（见 `poc/kv_cache`） |
| 没有 swap / 抢占 | 真实 KV 满时可 swap 到 CPU 或 recompute |

想扩展的话，可以加：prefix cache 命中复用、block 的 LRU 驱逐、swap 到 CPU。
