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

## fp8 慢于同形状 bf16 ⇒ 先怀疑 dequant 回退，不要先怀疑硬件上限

**这是 2026-11-23 一次真实误判换来的判据，优先级高。**

现象：测到 fp8 只有 49.8 TF，而同机 bf16 有 274 TF，fp8 反而慢 5.5 倍。
当时的结论是「本机 fp8 硬件不行 / fp8 峰值就是 49.8」——**错**。

正确的排查顺序：

| 步 | 查什么 | 本例 |
|---|---|---|
| 1 | **同库同形状**对照（fp8 vs bf16，同一个库的同一族 kernel） | aiter fp8 47.7 vs aiter bf16 24.1 ⇒ **fp8 快 1.98×**，有原生路径 |
| 2 | 若是同库比值 ≈1.0 ⇒ **dequant 回退**：fp8 输入被反量化后仍走 bf16 MMA | 本例不是 |
| 3 | 若是同库比值 <1.0（fp8 更慢）⇒ 确实回退，去看 scale/epilogue 路径 | — |
| 4 | 只有排除回退后，才谈「实现质量」与「硬件上限」 | 本例：aiter 这一族整体低效（bf16 也只到 nominal 的 8.6%） |

**为什么"先怀疑硬件"是错的**：fp8 与 bf16 走同一套 MFMA 单元，双速率是硬件设计。
如果同库 fp8 反而更慢，那只可能是软件路径（回退/scale 广播/布局转换）出问题。
**把软件问题记成硬件上限，会直接导致「已到顶」的错误结论，从而放弃优化。**

**跨库比较不能用来判精度特性。** 本机实测：aiter 与 hipBLASLt 的 bf16 差 8.8×。
拿 aiter 的 fp8 去比 hipBLASLt 的 bf16，比出来的是**库的差距**，不是精度的差距。

口径关联：
- gfx938 是 **2:1 双速率**（fp8 nominal 560 TF / bf16 nominal 280 TF），
  实测同库 fp8/bf16 = 1.98× 印证。见 `op-roofline/constants.yaml:compute.nominal_peak_tf`。
- 分母必须按精度分开：拿 bf16 的分母算 fp8 达成率会错 2 倍。

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
