---
name: op-fusion-add-rmsnorm-quant
description: "残差相加 + RMSNorm + 动态 int8 量化三合一。输入 x/residual [M,H] 与 w[H]，输出 y(int8)/y_scale/res_out。用途：省掉 norm 输出先写出去再读回来量化的中间态。触发场景：MLP 前段残差+归一化后紧接量化。不适用：per-channel 量化。闭环：本卡只给知识与形状事实，做优化必须走 op-optimize 的 loop.py。"
---

# op-fusion-add-rmsnorm-quant

## 闭环（必读）

本卡只提供该算子的形状、字节数与既有事实，**不承担实验流程**。
做优化请回到入口 skill `op-optimize`，实验只经 `loop.py run`。

## 关键语义不变量（都写进 contract.yaml）

- **quant_scope**：per-row 对称 int8，不是 per-channel
- **clamp_before_cast**：先 clamp 到 [-127,127] 再 round 再 cast
  —— 反过来在 bf16 下会先溢出成 inf
- **residual_semantics**：res_out 写回 (x+r)，不是 r
- **QMAX=127**，不用 -128，留对称

## 现状

- 工坊完整，reference 验证过（int8 范围合规、scale 全正、
  反量化误差 0.00393 ≈ int8 理论值）
- **只有 reference，无竞品候选**

## 与 op-fused-add-rmsnorm 的关系

本卡是它的超集（多一趟量化）。那张卡的寄存器驻留方案在这里同样适用，
且量化可以在寄存器里做，**不需要额外读一遍** —— 这是本卡最值得做的方向。
