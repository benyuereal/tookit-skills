---
name: op-fusion-rmsnorm-rope
description: "RMSNorm + RoPE 融合。输入 q/k [B,S,H,D] 与 w/cos_sin_cache，输出 fused q_out/k_out。用途：省掉 norm 输出落盘再读回来加 rope 的那趟中间态往返。触发场景：attention 前的 norm+rope 连续两步。不适用：D 不是 2 的倍数。闭环：本卡只给知识与形状事实，做优化必须走 op-optimize 的 loop.py。"
---

# op-fusion-rmsnorm-rope

## 闭环（必读）

本卡只提供该算子的形状、字节数与既有事实，**不承担实验流程**。
做优化请回到入口 skill `op-optimize`，实验只经 `loop.py run`。

## 融合省的是什么

未融合 3 读 3 写（norm 输出落盘、再读回来加 rope）；
融合后 2 读 2 写，**省掉中间态那一趟往返**。

## 现状

- 工坊完整，reference 验证过（rope 是纯旋转，L2 范数比 = 1.000000、cos²+sin² ≡ 1）
- 有 `torch-unfused` 对照组（数学等价，但真的分两趟写中间态）
- **缺竞品候选**：需先核对 lightop 是否有 rope 融合算子

实测（prefill-8k，3 轮，back_to_back）：
```
torch-unfused  3.5065ms  vs  ref-heavy  差 8.4%
```
两个都是 torch 实现，差距只来自多一趟显存往返。
**真正的融合倍数要手写 HIP 才能给，那个还没写。**
