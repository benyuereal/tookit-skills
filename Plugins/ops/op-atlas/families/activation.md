# activation

激活与门控（SwiGLU / GeGLU / GELU）。多与 GEMM epilogue 融合。

## 本族判据

1. 融合进 epilogue 能省一次写回，但会打断 GEMM 的流水线——先量分拆版的和
2. 近似式：tanh 近似 vs erf 精确，数值不同且不报错
3. gate/up 双路的顺序：`silu(gate)*up` 与 `silu(up)*gate` 结果不同
4. 激活是逐元素，本身不是瓶颈；瓶颈永远在读写次数

## 实现横评

| 实现 | 层级 | 适合什么 | 不适合什么 | 证据 |
|---|---|---|---|---|
| `lightop` 融合族 | kernel | 需要融合残差/门控 | 单条无链算子（融合没有可省的往返） | `libraries.md §3` |
| `torch` 原生 | kernel | 基线对照 | 追求极限 | 本仓实测 |
| `TileLang` | kernel | 形状需要表达力 | 逼近手写极限（先付布局税） | `libraries.md §8` |

## 本族不建卡的条目

| 条目 | 为什么不建卡 | 归处 |
|---|---|---|
| `logits_softmax_ce`（交叉熵） | **训练侧算子**。本仓库定位是推理优化引擎，它永远不会有 `last_verdict: measured` | 若确要覆盖训练，应另建 atlas，不往这里塞 |
