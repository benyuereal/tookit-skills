---
name: op-fused-add-rmsnorm
description: "残差 + RMSNorm 融合（fused_add_rmsnorm）。decode/prefill 逐层必经的 memory-bound 算子，字节数完全确定，是整个引擎里最适合校准带宽判据的载体。现成实现：lightop.RMSNormAdd（同族 RMSNormAddDropout / RMSNormAddDropoutBias）。触发场景：norm 层占用可观的 step 时间、要判断 norm 是否已到带宽地板、残差是否原地写、需要引入最简单的 memory-bound 参照物。不适用：LayerNorm 的均值中心化语义需改判据；MoE 的 SwiGLU 走 op-mlp-swiglu；QK-norm 走 op-qk-norm。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
kind: card
---

# 残差 + RMSNorm 融合

## 一、这段算什么

```
xf = x + r                       # 先融残差
rms = sqrt(mean(xf², -1) + eps)  # eps 在 sqrt 内（与 lightop.RMSNormAdd 一致）
y = w * (xf / rms)
```

## 二、为什么它是 P0 第一（校准件）

1. **字节数完全确定**：融合后读 `x` + 读 `r` + 读 `w` + 写 `y` = `3*M*H*2B`；
   不融合则多一次往返。分母没有争议。
2. **无复用结构可调**：没有 tile 选型、没有 split-K。地板算出来就是天花板。
3. **结论可证伪**：如果它到不了地板，说明**判据本身有问题**（带宽口径、计时口径），
   而不是这个算子有问题。这正是校准件的用途。
4. **现成实现明确**：`lightop.RMSNormAdd`，已在本机核实可 import。

## 三、变体矩阵

见 [`variants.yaml`](variants.yaml)。关键分叉：

| 分叉 | 选项 | 影响 |
|---|---|---|
| 残差是否融合 | 融合 / 分开 | 融合省一次 HBM 往返——这是唯一真收益 |
| 残差写回 | 原地 / 新张量 | **原地写会让输入被改，多变体对比必须各喂副本** |
| eps 位置 | sqrt 内 / 外 | 小值输入下有可见差异 |

## 四、形状档位

见 [`shapes.yaml`](shapes.yaml)：`tiny` / `decode-m1` / `decode-m32` / `prefill-8k`。

`tiny` 只用于验证门是否通，**绝不用于性能结论**。

## 五、判据

- **带宽达成率**：`3*M*H*2B / median_ms` 对比实测流式峰值 **1724 GB/s**（不是标称 1100）。
- **精度**：对 fp32 参考 `max_abs ≤ contract 容差`。
- **残差语义**：输出必须等于 `rmsnorm(x + r)`，不是 `rmsnorm(x) + r`。
- **两边各喂副本**：原地写的变体会污染对比。

## 六、已知的坑

| 坑 | 机制 | 判据 |
|---|---|---|
| `eps` 在 sqrt 内外 | `sqrt(mean+eps)` vs `sqrt(mean)+eps` | 与参考实现对比小值输入 |
| 残差原地写 | 后跑的变体吃到被改的输入 | 每个变体喂独立副本 |
| 用标称 1100 做分母 | 已有空间被算成 87%，误判到顶 | 一律用实测 1724 |
| hidden 非 2 的幂 | 向量化宽度受限 | 用 hidden=4096 之外的形状测 |

## 七、复现

```bash
# 正确性门（tiny 档，无需后端）
python3 scripts/opcheck.py correctness \
    --contract Plugins/ops/skills/op-fused-add-rmsnorm/contract.yaml \
    --shape tiny --variants lightop-rmsnormadd

# 完整复现（含 A/B 与 trace 预算）
./reproduce.sh all
```

## 验收

1. 精度：对 fp32 参考 `max_abs ≤ contract 容差`。
2. 残差语义正确：等于 `rmsnorm(x + r)`。
3. 带宽达成率用**实测峰值 1724 GB/s** 做分母，并写明口径。
4. `mutates_inputs` 声明与实际一致。
5. tiny 档通过全部门（证明 adapter 与门本身是通的）。
6. `reproduce.sh check` 能一条命令跑出同样结果。

## 不适用场景

LayerNorm（需先做均值中心化，字节数不同，判据要改）；
MoE 的 SwiGLU 激活（走 `op-mlp-swiglu`）；
QK-norm 的 per-head 归一（走 `op-qk-norm`）；
compute-bound 的算子（本卡判据只在 memory-bound 下成立）。
