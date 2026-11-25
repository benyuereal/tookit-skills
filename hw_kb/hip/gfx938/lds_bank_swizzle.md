# lds_bank_swizzle

> **状态：❌ refuted**（在测过的场景下无效）
> **实测**：PMC 显示 `SQ_LDS_BANK_CONFLICT/SQ_INSTS_LDS = 9.6`，XOR swizzle 解决后
> **只快 0.36%** @ `event_isolated`
> 复现：`/workspace/mqa_opt/ab.py --rows 48 --ctx 32768`
> **发现于**：`mqa_opt`，2026-11-25

## 问题

LDS 的 bank 冲突是经典优化点：16 个 lane 读同一 bank 会串行化。

实测场景（paged MQA logits，bf16 16×128 tile）：**9.6 次冲突/指令**，
分析性根因清楚 —— **256B 行跨距是 0 mod 32 banks**，所有 lane 打到 bank 0。

## 机制

LDS 有 32 个 bank（每 bank 4B 宽）。地址 `a` 落在 `bank = (a / 4) % 32`。
行跨距 256B = 64 dword = 0 mod 32 → 所有行的同一列落在同一个 bank。

XOR swizzle：把 16B chunk 的列索引按行号异或，让 8-lane phase 打到 8 个不同 bank。

## 做法

```c
// 写入与读取都要 swizzle（这是最容易漏的一半）
const int swz = (chunk_col ^ (row & 7));
```

## 实测

| 变体 | 耗时 | 相对 |
|---|---|---|
| baseline（无 swizzle） | 0.2372 ms | 1.000 |
| **XOR swizzle** | **0.2364 ms** | **1.004** |
| 主收益（backpressure 路径） | 0.1873 ms | 1.266 |

**9.6 次冲突/指令是真问题，修了也确实解决了，但只快 0.36%。**

## 什么时候不适用

**这是这条机制真正有价值的部分。**

我们推断出的边界：
- 当 kernel 已经**接近带宽上限**时，LDS 冲突不是瓶颈
- 本 case 的 baseline 已达 1234 GB/s（71% 峰值），冲突的代价被访存延迟掩盖
- 只有当 **LDS 吞吐本身是瓶颈**时（即 kernel 受限于指令发射而非 HBM）才值得改

## 反例/陷阱

**这是本仓最重要的一条反例：**

> **PMC 计数器非零 ≠ 它是瓶颈。**

计数器告诉你「这里有问题」，不告诉你「修了有收益」。
`SQ_LDS_BANK_CONFLICT = 9.6` 是一个真实的、有明确机理的问题，
修掉它也确实解决了 —— 但收益只有 0.36%。

**任何基于 PMC 的归因，必须配一次「修掉它」的实测。** 否则那只是相关性。

## 关联

- `Plugins/ops/skills/op-profiling/references/counters.md` —— D4 记录了
  `SQ_LDS_BANK_CONFLICT` 在本机有时采不到真实值
