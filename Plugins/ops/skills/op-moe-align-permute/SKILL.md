---
name: op-moe-align-permute
description: "MoE 第 3 段：把 token 按专家重排并补齐到对齐块（align block size + permute）。覆盖 padding 哨兵语义、对齐倍数、以及「-1 只能垫在队尾」这条会静默算错的纪律。触发场景：MoE 输出比参考实现偏、grouped GEMM 的行数不对、对齐后出现无意义 token、要判断某个 padding 位置能不能用。不适用：专家选择走 op-moe-topk-gating；专家内 GEMM 走 op-moe-w8a8；稀疏注意力的 topk/-1 语义走 op-sparse-mla。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
kind: card
---

# MoE align-block-size + permute（第 3 段）

## 一、这段算什么

```
给定 topk_idx [M, K]：
  counts[e]      = 每个专家收到的 token 数
  offsets[e]     = 前缀和，并按 align 倍数向上取整
  dst_slot[i]    = 第 i 个 token 在第 e 个专家块里的目标位置
  permuted_rows  = gather(x, src_by_slot)     # 空槽位填 padding 值
```

## 二、为什么这一段必须单独建卡

**padding 的哨兵值错一位，结果静默算错，不报错。**

仓库已有教训：`-1` 只能垫在**队尾**（`valid_NI` 按块首计数截断循环）；
前置 `-1` 会让整个输出恒为 0。这个 bug 不会抛异常，只会让精度莫名其妙地偏。

## 三、变体矩阵

见 [`variants.yaml`](variants.yaml)。关键分叉：

| 分叉 | 选项 | 影响 |
|---|---|---|
| padding 位置 | 队尾 / 队首 / 交错 | **队首会让输出恒为 0**（实测） |
| 哨兵值 | -1 / 0 / 专用常量 | 必须与下游 GEMM 的 mask 语义一致 |
| 对齐倍数 | 128 / 64 / 16 | 与 kernel 的 tile 高度绑定 |

## 四、形状档位

见 [`shapes.yaml`](shapes.yaml)。

## 五、判据

- **置换正确**：`unpermute(permute(x))` 与输入逐位一致（可逆性）。
- **计数一致**：每个专家的 `counts[e]` 与 `topk_idx` 统计一致。
- **对齐正确**：每个专家块的起始偏移是 align 倍数的整数倍。
- **哨兵语义**：padding 位置的值必须与 `contract.yaml` 声明的哨兵一致，
  且**只在队尾**。

## 六、已知的坑

| 坑 | 机制 | 判据 |
|---|---|---|
| `-1` 前置 | `valid_NI` 按块首计数截断循环 → 全前置输出恒为 0 | 构造全 padding 块，看输出是否为 0 |
| 对齐倍数与 tile 高度不匹配 | 下游 GEMM 的 tile 读越界 | 检查 offsets 是否为 tile 高度的倍数 |
| counts 的 dtype | int32 vs int64 混用导致偏移溢出 | 大 E 大 M 时检查 |

## 七、复现

```bash
python3 scripts/opcheck.py correctness \
    --contract Plugins/ops/skills/op-moe-align-permute/contract.yaml \
    --shape tiny --variants ref-heavy
./reproduce.sh all
```

## 验收

1. `unpermute(permute(x))` 与输入逐位一致。
2. 各专家 `counts[e]` 与 `topk_idx` 的统计一致。
3. 每个专家块起始偏移是 align 倍数的整数倍。
4. padding 只在队尾，且值等于声明的哨兵。
5. 全 padding 块（某专家 0 个 token）不导致下游输出异常。

## 不适用场景

专家选择与 gating（走 `op-moe-topk-gating`）；
专家内 GEMM（走 `op-moe-w8a8`）；
稀疏注意力的 topk/-1 语义（走 `op-sparse-mla`——那个 `-1` 是无效索引，语义不同）；
分布式 dispatch/combine（走 `op-moe-combine-dispatch`）。
