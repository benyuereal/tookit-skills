---
name: op-gemm-bf16
description: "BF16 稠密矩阵乘 out=A@B（A[M,K], B[K,N] → [M,N]）。用途：判断给定形状该用哪个库、上限多少。触发场景：自研 GEMM 前先确认厂商库已经做到哪、还剩多少空间。不适用：想自己写一个比 hipBLASLt 快的 GEMM——那不可能。闭环：本卡只给知识与选型事实，做优化必须走 op-optimize 的 loop.py。"
---

# op-gemm-bf16

## 闭环（必读）

本卡只提供该算子的形状、字节数与既有事实，**不承担实验流程**。
做优化请回到入口 skill `op-optimize`，实验只经 `loop.py run`。

## 这张卡是选型题，不是优化题

目标不是写一个比 hipBLASLt 快的 GEMM（不可能），是回答：
**给定形状该用哪个库、上限多少。**

因此 comparison 的正确形态是「库 A vs 库 B」，不是「自研 vs 厂商」。

## 现状

- 工坊完整（contract / shapes / _common.py / candidates / roofline / bench）
- 有 2 个可跑变体：`ref-fp32`（正确性锚点）、`torch-mm`（bf16 hipBLASLt）
- **缺竞品候选**：`lightop.gemm`（M≥24 时结果错，见 libraries-bounds.md）、
  hipBLASLt 直调（未接）、deepgemm（已 clone 未构建）

首条实测（2026-11-27，event_isolated）：
```
ref-fp32   7.1317 ms   <- 正确性锚点，不是竞品
torch-mm   0.9382 ms   ratio=0.1316
```
**这不是优化成果** —— ref 是 fp32 参考，慢是必然的。
