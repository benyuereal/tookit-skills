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

## 实现横评

| 实现 | 层级 | 适合什么 | 不适合什么 | 证据 |
|---|---|---|---|---|
| `hbird 形态` | kernel | 单 shape、单 dtype，愿意维护 body + plan | 要做**一类** GEMM（先看 hytlass）；团队没有 plan 层纪律 | `libraries.md §1` |
| `hytlass` | kernel | 一类 GEMM：非标准布局、融合 epilogue、split-K/stream-K | 只要一个现成 GEMM；M≤8 decode | `libraries.md §2` |
| `DeepGEMM` | kernel | grouped / MoE GEMM，需按 layout 或 arch 精细控制 | 普通稠密大 GEMM（hipBLASLt 够用） | `libraries.md §6` |
| `hipBLASLt` | kernel | 常规稠密 GEMM | 精细 layout 控制；gfx938 的 FP8 | `libraries.md §6` |
| `TileLang` | kernel | 形状不定、要 tile 级表达力 | GEMM 主体（布局税） | `libraries.md §8` |

## 手工流水线的完整实例（外部参考）

同机同 gfx938 的 `/workspace/hbird-gemm` 是一份「压到硬件极限」的完整记录，
**负结果比正结果值钱**，以下是它已经试过的：

| 项 | 结论 | 数字 |
|---|---|---|
| Phase 0-10 阶段表 | 每步改什么、涨多少 | 见 `docs/DESIGN.md` |
| A-read 流水线（3 buffer + 2 步 lead） | 正收益，且比双缓冲更省寄存器（24 vs 32 dwords） | 2.1711 → 2.1070 ms |
| 4 缓冲 | 无额外收益 | 2.1085 ms |
| `buffer_load ... slc` 缓存提示 | **死路**：默认 descriptor 最优 | gate_up 0.868×（慢 13%），bit-exact |
| 源码级 4-stage prefetch 深度 | **死路**：深度由 scheduler 定，源码控制不了 | 关 pass 后 4.5263 → 2.6548 ms，但 vmcnt 模式一样 |
| `#pragma unroll 1` 用在 epilogue | **死路**：累加器索引变动态 → 128 个寄存器全溢出 | 388 条 `buffer_store` |
| 非转置 A 的 LDS 读（fp8） | **编译不过**：MLS dispatcher 只有 `_trans` atom | — |
| 每 CU 只放 1 CTA | 慢 27.5% | 2716.2 vs 2130.0 µs |

**适用条件**：tile 256×256×64 fp8 / 512 threads / 2 CTA per CU / 768-VGPR /
M=1..128 decode 档 / sclk 1350 MHz。换 tile、dtype、M 档数字会变；
但「slc 负收益」「源码控制不了 waitcnt 深度」来自机制，大概率可迁移。

ISA 层原语见 `op-hip-compile/references/isa-census.md §四点五/四点六`。
