---
name: op-stack-design
description: "算子优化的选型层：先认库、再排流水线、最后才动 kernel。按「M 档 × 复用结构」把工作负载匹配到十一类算子库（hbird 手写形态 / hytlass / LightOp / AITER / Moe(moe_c) / DeepGEMM / flash-attn·FlashMLA / TileLang / Triton / BoltOPs / TileKernels），每个库给结构定律、可调面、已知还留在地上的税；再用十条流水线定律（驻留优先于 tile、barrier 宽度、预取深度不是延迟杠杆、寄存器是生命周期等）在写代码前排掉结构性不可行的设计。触发场景：拿到一个算子不知道用哪个库、要不要手写、某个库在这台机器上到底强在哪、想把单算子突破换成整库选型 + 流水线设计、接手一个手写 kernel 库要判断还有没有空间。不适用：已经定了算子要单个调优走 op-optimize（本 skill 只到「选谁 + 怎么排」）；预测与地板算式走 op-roofline；量法与判定走 op-bench / op-gate。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
kind: tool
---

# 算子选型与流水线（先选对库，再谈快）

## 一、为什么要有这一层

单个算子的局部变形有上限。一轮做过 43 条判死只落地 1 个已知收益，因为大多数「候选」不是实现问题，
而是**选错了库**，或**流水线排布结构上就不成立**。这一层的产出通常是一句话结论 + 一条被判死的路线，
成本几乎为零。

| 域 | 回答 | 入口 |
|---|---|---|
| **本 skill（选型 + 排布）** | 用哪个库、流水线怎么排、哪些设计编译前就该否掉 | `references/libraries.md`、`references/pipeline-laws.md` |
| `op-roofline` | 这个 shape 的地板与绑定约束 | `derive.py` |
| `op-optimize` | 定了实现之后，一轮轮找下一个改动 | `loop.py` + `candidates.yaml` |

## 二、四步

### 1. 口径（一行）

`算子 | shape 档位 | dtype | 卡与 gfx | DTK | 生产布局`

**生产布局**最常被漏：微基准 bit-identical 且更快、上生产掉分，九成是输入布局 ≠ 生产 kernel 的输出布局。

### 2. 定 M 档与复用结构

两个问题，答案决定 `references/libraries.md` §一 里只有一行对得上：

1. **M 落哪档**：≤8（权重主导·GEMV）/ 16–128（权重主导·ladder）/ ≥256（算力主导）/ 分组 MoE。
2. **复用结构**：收缩维是否正交（双 GEMM 共用一份输入）、token 是否已物理重排、有无数据相关 gather。

答不出说明档位没定清，此时挑库挑错的后续实验都是学费。

### 3. 查画像：谁已经赢了、赢多少、为什么

读该库的三行：**结构定律 / 可调面 / 已知还留在地上的税**。产出一张表：
`候选实现 | 相对谁 | 倍数或机制`。

- 「已知还留在地上的税」不是未知空间（例：某库 stock 对大 M 用无软件流水线，是已知的 1.1–2×），别当新发现立项。
- 「结构性吃亏」的库不必试——那不是调参能救的（例：通用模板在 M 64–256 只有参考实现的 0.29–0.41×）。

### 4. 排流水线，先证可行再写代码

按 `references/pipeline-laws.md` 自查 L0 → L4，不满足就把这条设计否掉，记进 `candidates.yaml:refuted`
（`why` 写机制，用 L 编号）。L5–L10 用于事后解释为什么没到预测。

产出通常是这种句子：「按 L1，这个设计多要 16 KB LDS，驻留从 3 打到 2，价格 27.5%，而它自己的收益上限
10.5% ⇒ 否」。比跑一次实验便宜两个数量级。

## 三、和现有工具的接法

| 要做 | 用 |
|---|---|
| 算地板 / 出绑定约束 | `op-roofline/derive.py`（唯一天花板权威） |
| 判断设计是否溢出 | `op-roofline/tools/hipbudget.py`（只作相对比较） |
| 隔离一个变量（驻留 / launch / 分配） | 改**不改变语义**的量，先证 bit-identical 再计时 |
| 定了实现后的候选队列 | `op-optimize/loop.py`；本 skill 的结论作为候选的 `direction` 与 `expect` |
| 量法与判定 | `op-bench` / `op-gate` |

本 skill 不做实验、不写 kernel、不维护第二套计时或判定工具。

## 四、深引用

- 十一个库的画像与匹配表 → [`references/libraries.md`](references/libraries.md)
- 十条流水线定律（每条带反例）→ [`references/pipeline-laws.md`](references/pipeline-laws.md)
- 源码地图 / 构建 / 适配点 → `dcu-operators/dcu-{lightop,aiter,deepgemm,flashattn,tilelang,triton}-source`、`dcu-lightop-fusion`、`dcu-lightop-moe`、`dcu-deepgemm-arch`
- 算子级四问演算 → `op-roofline/references/pipeline.md`
