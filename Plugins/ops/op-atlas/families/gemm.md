# gemm

纯矩阵乘。**复用结构由 M 档决定**，杠杆随档位完全不同。

## 本族特有的判据

1. **先答 M 档，再谈优化**：
   - M≤8（decode GEMV）：权重主导，每步必读整份权重 ⇒ 唯一杠杆是字节流与 CTA 数
   - M=16–128：一档一个 tile 高度 + plan 层按 shape 选 tile/nw/split-K
   - M≥256（prefill）：复用已足够，杠杆是 MMAC 发射率与每 barrier 工作量
   **档位没定清就挑库，必然选错。**
2. **tile 形状不是第一约束**：驻留（CTA/CU）比 tile 更早成为约束——
   同一份 body，LDS 从 16KB 涨到 32KB 就把 3 CTA/CU 打到 2，墙钟 +27.5%。
3. **split-K 的 s 必须按 shape 单独搜**，不同服务权重的最优点各不相同。
4. **量化档位很宽**：W8A8/W4A8/W4A16/WFP4A8 各自对应不同的 tile 与 epilogue，
   不能拿一个档的结论套另一个档。

## 本族攻击面（跨族通用的见 surfaces.yaml）

- `occupancy`：CTA/CU 是否被 LDS 或寄存器打到更低档
- `split-k`：切片数是否按 shape 搜过
- `grid-tail`：grid 是否为 CU 的整数倍
- `table-miss`：shape 是否落在库的调优表里

## 已知的结构性陷阱

| 陷阱 | 机制 | 出处 |
|---|---|---|
| 用单一 tile 吃所有 M | 小 M 付 padding、大 M 付 IO | libraries.md §1 |
| 无软件流水线的 stock 配置 | 大 M 上留 1.1–2× | libraries.md §1 |
| 稠密 GEMM 循环调 per-expert | grouped 形态是 graph 化稠密路径的 2.9x | libraries.md §一 |

## 库选择

| 档位 | 首选 |
|---|---|
| M≤128 | `libraries.md §1`（hbird 形态，一档一 tile + plan 层） |
| M≥256 | `libraries.md §6`（DeepGEMM）/ hipBLASLt / §8（TileLang） |
| 非标准布局 / 融合 epilogue | `libraries.md §2`（hytlass） |
