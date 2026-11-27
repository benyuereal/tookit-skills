---
name: op-mqa-logits
description: "DSA indexer 打分 logits[i,j]=Σ_h relu(q·k)·w，H=32/D=128 固定。用途：sparse attention 选块前的打分。触发场景：prefill 或 decode 里 DSA 的 top-k 选块阶段。不适用：不是注意力本身。闭环：本卡只给知识与形状事实，做优化必须走 op-optimize 的 loop.py。"
---

# op-mqa-logits

## 闭环（必读）

本卡只提供该算子的形状、字节数与既有事实，**不承担实验流程**。
做优化请回到入口 skill `op-optimize`，实验只经 `loop.py run`。

## 生产占比（实测 trace）

来自仓外的 GLM-5.3 算子差异图（`/workspace/optimize-glm53/OPERATOR-DIFF-V2-V3.md`，不在本仓）：

| | BF16 版 | INT8 版 | 变化 |
|---|---|---|---|
| prefill | 29.96 ms/次，占 **35.0%** | 4.54 ms/次，占 4.9% | **6.6×** |
| decode | 1.77 ms/次，占 **32.5%** | 0.27 ms/次，占 6.7% | **6.6×** |

## 一个必须先验证的疑点

bf16 → int8 只省一半带宽，**理论上限 2.0×**。实测 6.6× 差 3.3 倍，无法用带宽解释。
**在验证 INT8 版是不是同语义之前，6.6× 不能当目标。**

见 `roofline.json` 的 `quant_ceiling`。

## 现状

工坊完整，只有 reference，无竞品候选。
