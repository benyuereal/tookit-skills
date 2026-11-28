---
name: op-moe-topk-gating
description: "MoE 第 2 段：gate logits → top-k 选择 → 归一化权重（topk gating）。覆盖 tie-break 语义、topk 与 softmax 的顺序、fp32 累加口径、以及「为什么同一输入两次跑出不同专家」。触发场景：MoE 路由结果不稳定、专家选择与参考实现不一致、topk 的边界语义要定、想优化这段的延迟。不适用：专家内 GEMM 走 op-moe-w8a8；token 重排与 padding 走 op-moe-align-permute；整引擎并行策略不进本卡。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
kind: card
---

# MoE top-k gating（MoE 7 段中的第 2 段）

## 闭环（必读）

本卡只提供该算子的形状、字节数与既有事实，**不承担实验流程**。
做优化请回到入口 skill `op-optimize`，实验只经：

    python3 -I <skill根>/Plugins/ops/engine/loop.py -C <卡目录> run <tag>

单次手测的数字不能当结论提交（判 `invalid:no_loop_evidence`）。

## 一、这段算什么

```
logits = x @ W_gate^T                  # [M, E]   E = 专家数
probs  = softmax(logits, dim=-1)       # 可选：有的实现用 sigmoid
topk_v, topk_i = topk(probs, K)        # [M, K]   选 K 个专家
w      = topk_v / sum(topk_v)          # 归一化；有的实现在 softmax 前归一
```

## 二、为什么要单独一张卡

这一段**没有 GEMM 可优化**（M×E 的矩阵乘在 E=128 时不构成瓶颈），
但它决定了后面 5 段的全部行为：

- **选错专家** → 后面的 grouped GEMM 算的是错的 token 集合，且**不会报错**
- **tie-break 不稳定** → 同一输入两次跑出不同专家 → determinism 判不了
- **归一化顺序错** → 权重和不为 1，输出整体偏大或偏小

这是 MoE 里唯一「错了也静默」的一段。

## 三、变体矩阵

见 [`variants.yaml`](variants.yaml)。三个关键分叉：

| 分叉 | 选项 | 影响 |
|---|---|---|
| 归一化时机 | softmax 前 / 后 | 权重值不同，下游 GEMM 的 scale 不同 |
| tie-break | 取小索引 / 取大索引 / 不定义 | 不定义则不可复现 |
| 累加精度 | fp32 / bf16 | bf16 在 E=128 时会有可见偏差 |

## 四、形状档位

见 [`shapes.yaml`](shapes.yaml)：`decode-m1` / `decode-m32` / `prefill-8k` / `tiny`。

`tiny` 只用于验证门是否通，**绝不用于性能结论**。

## 五、判据

- **确定性**：同输入同 seed 连跑 5 次，`topk_i` 必须逐位一致。
- **参照一致**：与 fp32 参考实现的 `topk_i` 完全一致（这是离散量，容差无意义）。
- **权重归一**：`sum(w)` 与 1 的偏差 ≤ 1e-6。
- **精度**：`probs` 对 fp32 参考的 `max_abs ≤ contract 容差`。

## 六、已知的坑

| 坑 | 机制 | 判据 |
|---|---|---|
| tie-break 未定义 | 两个专家分数相等时，不同实现选不同索引 | 构造等分输入，看选哪个；不稳定的实现直接判不通过 |
| softmax 与 topk 顺序颠倒 | 先 topk 再 softmax 会在 K 个子集上归一，与全分布归一不同 | 对比 `sum(w)` |
| bf16 累加 | E=128 时误差累积明显 | 与 fp32 参考比 `max_abs` |
| 专家数不是 2 的幂 | 某些 kernel 的 padding 假设 E 是 2 的幂 | 用 E=100 这类形状测 |

## 七、复现

```bash
# 正确性（tiny 档，adapter 自检）
python3 scripts/opcheck.py correctness \
    --contract Plugins/ops/skills/op-moe-topk-gating/contract.yaml \
    --shape tiny --variants ref-heavy

# 完整复现（含 A/B 与 trace 预算）
./reproduce.sh all
```

## 八、这一段的上下游

```
第 1 段 gate/softmax ← 本卡 → 第 3 段 align/permute → 第 4 段 grouped GEMM
```

本卡的输出 `(topk_i, w)` 是下一段的输入。改这一段等于改所有下游的输入集合。

## 验收

1. `topk_i` 在同 seed 连跑 5 次逐位一致（determinism）。
2. 与 fp32 参考的 `topk_i` 完全一致（离散量，不做容差）。
3. `sum(w)` 与 1 偏差 ≤ 1e-6。
4. `probs` 对 fp32 参考 `max_abs ≤ contract 容差`。
5. tie-break 行为在等分输入下可复现（不是随机）。

## 不适用场景

专家内的 GEMM（走 `op-moe-w8a8`）；
token 重排、padding、哨兵语义（走 `op-moe-align-permute`）；
整模型的并行策略与显存水位（引擎层问题，不进算子卡）；
单个稠密 GEMM 的 tile 选型（走 `op-gemm-fp8`）。
