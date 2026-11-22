# elementwise

逐元素 / norm / 激活 / 量化融合。**memory-bound**：收益来源单一——省 HBM 往返。

## 本族特有的判据

1. **先算字节数，再谈优化**。这一族没有复用结构可调，地板是确定的：
   `bytes / 实测峰值带宽`。算不出地板说明还没想清楚读写几次。
2. **融合只对 memory-bound 链成立**。把中间态留在寄存器/smem、HBM 只读写一遍，
   是这一族唯一的结构性收益。对 compute-bound 主体做融合是净负。
3. **融合的依赖链要单独算**：融合把本可并行的 kernel 串成一条链并多占 smem。
   先量分拆版的和，再决定融不融。
4. **破坏融合的常见原因**：(a) 中间态精度是 fp32 但下游要 bf16；
   (b) 归约维与向量化维冲突；(c) 需要跨 CTA 的全局归约。

## 本族攻击面

- `layout-pack`：每一步是否多付一次 pack/transpose
- `fusion-boundary`：融合是否把可并行部分串成依赖链
- `occupancy`：小 M 时 CTA 数不足导致带宽上不去

## 已知的结构性陷阱

| 陷阱 | 机制 |
|---|---|
| 对 compute-bound 主体做融合 | 省下的往返抵不过多占的 smem 与串行化 |
| 用标称带宽做分母 | 会把已有空间算成已到顶 |
| 原地写残差类输入 | 多变体对比时后跑的吃到被改的输入 |

## 库选择

优先 `libraries.md §3`（LightOp 融合族）；单条无链算子不必融合。

## 实现横评

| 实现 | 层级 | 适合什么 | 不适合什么 | 证据 |
|---|---|---|---|---|
| `lightop` 融合族 | kernel | memory-bound 链（norm / 激活 / rope / 量化 / 采样），希望一个 kernel 吃掉整条链 | compute-bound 的 GEMM 主体（融合净负） | `libraries.md §3` |
| `TileKernels` | kernel | 模型结构层算子（mHC / Engram / GDN）+ 可训练层 | 单条 memory-bound 链（LightOp 更成熟）；融合链本身可能更慢 | `libraries.md §11` |
| `torch` 原生 | kernel | 基线对照；快速验证语义 | 追求极限（无融合、多次往返） | 本仓实测 |
| `BoltOPs` | kernel | decode 形状的成品集合，自带 autotune CLI | 大 M prefill（Triton 路线固有税） | `libraries.md §10` |

**读这张表的方式**：这一族没有"选谁更快"的问题，只有"要不要融合"。
先确认是不是 memory-bound 链（字节数确定、无计算主体），是才融；
否则直接用 torch 原生反而省事。
