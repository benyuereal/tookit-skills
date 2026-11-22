---
name: op-rmsnorm-fused
description: "融合 RMSNorm（含残差与可选乘子）。最典型的 bandwidth-bound 算子——HBM 只读写一遍就是它的全部上限，最适合用来证伪「还有优化空间」。触发场景：norm 层占用了可观的 step 时间、想判断 norm 是否已到带宽地板、残差是否原地写、要引入一个最简单的 memory-bound 参照物。不适用：LayerNorm 的均值中心化语义走同一张卡但需改判据；MoE 的 SwiGLU 激活走 op-silu-gelu；注意力前的 QK-norm 融合走 op-custom-fusion-kernel 思路。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
kind: card
---

# 融合 RMSNorm

## 一、这段算什么

```
rms  = sqrt(mean(x², dim=-1) + eps)          # 沿 hidden 维
y    = weight * (x / rms)                     # 逐元素
（融合残差时）y = weight * ((x + residual) / rms(x + residual))
```

## 二、为什么这张卡值得先做

它是**最容易证伪的算子**：

- 字节数完全确定：读 `x`（+ `residual`），写 `y`。没有中间态（融合的意义就在这）。
- 因此带宽地板可以精确算出来，达成率的分母没有争议。
- 如果测出「已达地板」，就**直接关掉这个优化方向**，省下后面的所有尝试。
- 如果测出「离地板很远」，那问题一定在实现（访存模式、向量化），不在算法。

**这是整个引擎里最适合用来校准判据的算子。**

## 三、变体矩阵

见 [`variants.yaml`](variants.yaml)。关键分叉：

| 分叉 | 选项 | 影响 |
|---|---|---|
| 残差是否融合 | 融合 / 分开 | 融合省一次 HBM 往返，是唯一真收益 |
| 残差写回 | 原地 / 新张量 | **原地写会让输入被改，多变体对比时必须各喂副本** |
| 归一化维 | hidden / 其它 | RMSNorm 固定沿最后一维 |

## 四、形状档位

见 [`shapes.yaml`](shapes.yaml)。decode 档 M 小、hidden 大；prefill 档 M 大。

## 五、判据

- **带宽达成率**：`bytes / median_ms` 对比实测流式峰值（1724 GB/s，不是标称 1100）。
- **精度**：`max_abs` 对 fp32 参考 ≤ 契约容差；`eps` 的位置影响数值。
- **残差语义**：若声明融合残差，输出必须等于 `rmsnorm(x + residual)`，不是 `rmsnorm(x) + residual`。

## 六、已知的坑

| 坑 | 机制 | 判据 |
|---|---|---|
| `eps` 加在 sqrt 内外 | `sqrt(mean+eps)` vs `sqrt(mean)+eps`，数值不同 | 与参考实现对比小值输入 |
| 残差原地写 | 多变体共用输入时，后跑的吃到被改的输入 | 每个变体喂独立副本 |
| 用标称 1100 做分母 | 会把已有空间算成 87%，误判到顶 | 一律用实测 1724 |
| hidden 非 2 的幂 | 向量化宽度受限，达不到理论带宽 | 用 hidden=4096 之外的形状测 |

## 七、复现

```bash
python3 scripts/opcheck.py correctness \
    --contract Plugins/ops/skills/op-rmsnorm-fused/contract.yaml \
    --shape tiny --variants ref-heavy
./reproduce.sh all
```

## 验收

1. 精度：`max_abs` 对 fp32 参考 ≤ 契约容差。
2. 残差语义正确（若声明融合）。
3. 带宽达成率用**实测峰值 1724 GB/s** 做分母，并写明口径。
4. 声明 `mutates_inputs` 与实际一致（原地写残差的变体必须标 true）。
5. tiny 档通过全部门（证明 adapter 与门本身是通的）。

## 不适用场景

LayerNorm（需先做均值中心化，字节数不同，判据要改）；
MoE 的 SwiGLU 激活（走 `op-silu-gelu`）；
需要跨层融合的 QK-norm（那涉及 kernel 结构改造，走融合 kernel 思路）；
compute-bound 的算子（本卡的判据只在 memory-bound 下成立）。
