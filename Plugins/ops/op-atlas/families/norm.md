# norm

逐元素归一化（RMSNorm / LayerNorm / QK-norm）。
memory-bound，字节数确定，是校准带宽判据的最佳载体。

## 本族判据

1. 归一化沿哪一维：换维度就是另一个算子，不能跨判据
2. eps 在 sqrt 内还是外：`sqrt(mean+eps)` 与 `sqrt(mean)+eps` 在小值下有可见差异
3. 融合残差是否原地写：原地写会污染多变体对比，必须各喂副本
4. 字节数：不融合 2*M*H*2B，融合 3*M*H*2B —— 分母没有争议，适合校准判据

## 实现横评

| 实现 | 层级 | 适合什么 | 不适合什么 | 证据 |
|---|---|---|---|---|
| `lightop` 融合族 | kernel | 需要融合残差/门控 | 单条无链算子（融合没有可省的往返） | `libraries.md §3` |
| `torch` 原生 | kernel | 基线对照 | 追求极限 | 本仓实测 |
| `TileLang` | kernel | 形状需要表达力 | 逼近手写极限（先付布局税） | `libraries.md §8` |
