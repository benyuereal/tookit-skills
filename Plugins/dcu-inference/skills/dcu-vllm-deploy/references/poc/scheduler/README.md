# 调度器最小化 POC —— 动态批处理是怎么实现的

一个 ~200 行的纯 Python 调度器，把 vLLM V1 调度器的核心决策浓缩出来，
让你**跑起来、看见**动态批处理 / Chunked Prefill / KV 抢占是怎么工作的。

只依赖 Python 标准库，CPU 直接跑，不需要 GPU/DCU：

```bash
python3 mini_scheduler.py
```

对应文档：[`docs/vllm_scheduler_guide.md`](../../docs/vllm_scheduler_guide.md)

## 它演示了什么

| 机制 | 在 trace 里怎么看 |
|------|------------------|
| **动态批处理** | 每个 step 的 `batch` 里混着多个请求，decode 请求每步只算 1 token（`[Dx1]`），新请求 prefill 进来（`[PxN]`），一起算 |
| **Chunked Prefill** | 长 prompt 不一步算完：A 的 40 token prompt 被切成 `Px32` + `Px8` 两步，受 token 预算限制 |
| **Token Budget** | `tokens=32/32` 每步最多 32 token（prefill+decode 合计），控制单步延迟 |
| **KV 容量 + 抢占** | `kv=16/16` 满时，`⚡抢占 D` 把请求踢回 waiting、KV 释放、重算；C 因 KV 满等到 step 12 才进 batch |

## 怎么读 trace

```
step   0 | batch= 1 | tokens= 32/32 | kv=8/16  | A[Px32]
step   1 | batch= 2 | tokens= 16/32 | kv=12/16 | A[Px8] B[Px8]
step   2 | batch= 2 | tokens=  2/32 | kv=14/16 | A[Dx1] B[Dx1]
...
step   6 | batch= 2 | tokens=  2/32 | kv=16/16 | A[Dx1] B[Dx1]  ⚡抢占 D
```

- `A[Px32]`：A 这一步做 prefill，算 32 个 token
- `A[Dx1]`：A 这一步做 decode，算 1 个 token（生成 1 个）
- `tokens=32/32`：本步用了 32 token，预算 32
- `kv=16/16`：KV cache 用了 16 块，总共 16 块（满了）
- `⚡抢占 D`：本步把 D 抢占回 waiting

## 汇总

```
req    prompt  gen  TTFT  finish       state
A          40   10     2      11    finished
B           8    6     2       7    finished
C          20    8    13      20    finished
D           6    5     9      13    finished
```

- **TTFT**：首 token 产出的 step（prefill 算完那一刻）
- **finish**：生成满 `max_tokens` 的 step
- C 的 TTFT=13 比 A/B 晚很多，因为 KV 满时它进不了 batch，得等 KV 释放

## 和真实 vLLM 的差异（简化点）

| 简化 | 真实 vLLM |
|------|----------|
| 抢占用 recompute（computed 清零重算） | 可选 recompute 或 swap 到 CPU |
| 调度顺序：先 running 后 waiting，FIFO | 有优先级、FCFS/priority 策略、preempt 选最晚的 |
| KV 块 = ceil(computed/block_size) | 同，但还有 prefix cache 命中复用 |
| 没有 prefix cache | 有 APC，相同前缀复用 block |
| 单步 decode 固定 1 token | 投机解码时一步可验证多个 token |

想扩展的话，可以加：prefix cache 命中、优先级调度、swap 抢占、投机解码的 token 预算。
