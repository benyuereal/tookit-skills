# 算子图谱索引（op-atlas）

> **图谱是索引层，不是知识副本。** 卡片只回答四个问题：它算什么 / 谁结构上赢 /
> 上限在哪 / 上次结论。技术描述一律指向别处（`libraries.md` 的 §号、
> `pipeline-laws.md` 的 L 号、kb 台账），卡片 schema 用
> `additionalProperties: false` 锁死，写不进去。
>
> 校验：`python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas`

## 怎么用

1. 按算子家族定位到卡 → 读 `impls` 看谁在这个档位结构上赢
2. 读 `ceiling` 看上限在哪（`kind: missing` 是合法状态，表示天花板未知）
3. 读 `last_verdict` 看上次结论（`ratio` 必须带 `vs`，裸倍数不许写）
4. 要优化它 → 进 `op-optimize` 闭环；要查库的脾气 → 读 `libraries.md` 对应 §

## 算子一览

| 算子 | 家族 | 状态 | 首选实现 | 上限 | 上次结论 |
|---|---|---|---|---|---|
| [`mqa-logits`](cards/mqa-logits.yaml) | attention | measured | lightop grouped | bandwidth | 1.29x vs G=3（epilogue bpermute） |
| [`gemm-fp8`](cards/gemm-fp8.yaml) | gemm | draft | tilelang-hygon | compute | 无实测 |
| [`moe-grouped-gemm`](cards/moe-grouped-gemm.yaml) | moe | draft | deepgemm-das | mixed | 无实测 |

## 家族

| 家族 | 判据 | 文件 |
|---|---|---|
| gemm | 纯矩阵乘，复用结构由 M 档决定 | [families/gemm.md](families/gemm.md) |
| attention | Q/K/V 参与，瓶颈通常在 KV 搬运而非 GEMM | [families/attention.md](families/attention.md) |
| moe | token 按专家重排后分组计算 | [families/moe.md](families/moe.md) |
| elementwise | 逐元素 / norm / 激活，memory-bound | [families/elementwise.md](families/elementwise.md) |
| comm | 集合通信与计算 overlap | [families/comm.md](families/comm.md) |
| sampling | topk / 采样，形状随数据变化 | [families/sampling.md](families/sampling.md) |

## 加一个新算子

1. 在 `cards/` 加一个 yaml，按 schema 填（字段白名单见 `validate.py`）
2. 在上表加一行
3. 跑校验器，E1~E8 全过才提交

**别做的事**：不要在卡片里写技术描述、绝对耗时、无分母百分比。
校验器会拦（E6 字段白名单、E7 结论红线）。
