# quant

量化与反量化。

## 本族判据

1. **标度粒度决定 kernel 结构**：per-tensor / per-token / per-block 三种粒度
   对应完全不同的实现，**不能跨粒度比较达成率**。这是本族最贵的口径错。
2. **fp8 的 scale 广播会打破驻留假设**：`pipeline.md` Q5 的三条反例里，
   FP8 + persistent 变慢的根因就是「scale 广播」。
   所以 persistent 相关结论必须按 dtype 分岔，不能跨精度复用。
3. **满量程语义**：ROCm 上是 fnuz，gfx950 除外。量化代码不能照搬 NVIDIA e4m3。
4. **prepack 是一次性成本**：weight pack/shuffle 的收益与推理时每 token 的
   读写次数成反比；权重格式固定的部署才摊得掉。

## fp8_scale_broadcast（不建卡，判据写在这里）

| 关注点 | 内容 |
|---|---|
| 三种粒度 | per-tensor（一个标度）/ per-token（每行一个）/ per-block（每 128 元素一个） |
| 为什么不能混 | 三种粒度对应不同的 epilogue 与不同的 LDS 占用，**达成率的分母都不同** |
| 最快踩到的坑 | 按 per-tensor 的口径去读 per-block 的实现，会得出「快了很多」的假结论 |
| 该看哪里 | `libraries.md §3`（LightOp 量化块）、`§5`（moe_c 的精度模式矩阵） |

## 实现横评

| 实现 | 层级 | 适合什么 | 不适合什么 | 证据 |
|---|---|---|---|---|
| `lightop` 量化融合 | kernel | memory-bound 链上的量化 | compute-bound 主体 | `libraries.md §3` |
| `moe_c` 精度矩阵 | kernel | 按 (精度 × gfx) 查表 | 表没命中的组合 | `libraries.md §5` |
| `TileKernels` 量化族 | kernel | 模型层量化融合（SwiGLU+quant） | 单条无链算子 | `libraries.md §11` |

## 本族不建卡的条目

| 条目 | 为什么不建卡 | 归处 |
|---|---|---|
| `fp8_scale_broadcast` | 它是**判据**不是算子：三种粒度对应三套 epilogue，写成卡片会让人以为可以跨粒度比 | 本文件上方 |
