---
name: dcu-hytlass-source
description: "hytlass（HYGON HCU Templates for Linear Algebra Subroutines）源码结构与用法索引——CUTLASS 3.5.0 的 HCU 移植，C++ 模板库，两套编程模型（CUTLASS 2.x 的 threadblock/warp + mmac/ds_read_matrix 原语；CUTLASS 3.x CuTe，仓内称 HuTe），19 个示例覆盖 basic/split-K/stream-K/batch/group/fused/conv/稀疏/带 softmax 与 amax 的 epilogue。触发场景：想用 C++ 模板搭自己的 GEMM 而不从零写 HIP、要非标准 layout 或融合 epilogue 的 GEMM、要把手写 kernel 的设计固化成可复用模板、要判断某 CUTLASS 风格 kernel 在 HCU 上怎么落、查 hytlass 的构建方式与示例选型。不适用：只是调一个现成 GEMM 的 shape（走 dcu-deepgemm-source 的调优表或 hipBLASLt）；M≤8 的 decode GEMV（手写 ladder 更快）；已定实现要一轮轮找下一个改动（走 perf/dcu-op-optimization）。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
---

# hytlass 源码与用法索引（dcu-hytlass-source）

> 定位：hytlass 是 **CUTLASS 3.5.0 的 HCU 移植**（固定 commit），C++ 模板库，BSD-3-Clause，
> 官方开源版在 HYGON-AI 组织的 `hytlass` 仓（`release` 分支）。
> 它补的是算子选型里一直空着的那一段：**既不是「用现成库」，也不是「从零写 HIP」，而是「用 C++ 模板搭一类 GEMM」**。
> 与兄弟 skill 的分工：`dcu-kernel-dev` 讲手写 HIP；`dcu-deepgemm-source` 讲现成 GEMM 的路径与调优表；本 skill 只讲 hytlass 的结构与边界。

## §0 定位：它是什么、不是什么

- **是什么**：CUTLASS 3.5.0（commit `7d49e6c7e2f8896c47f586706e67e1fb215529dc`）的 HCU 移植。tile 形状、stage 数、KernelSchedule、epilogue 全是**编译期模板参数**，运行时只剩 launch。
- **兼容两条模型**：
  1. **CUTLASS 2.x 风格**——threadblock / warp 分层，用 BW（gfx936）的 `mmac` 指令与 `ds_read_matrix` 原语；适合把已有手写 kernel 的设计搬进模板。
  2. **CUTLASS 3.x CuTe 模型**（仓内称 **HuTe**）——用 layout / tensor / atom 描述数据流，tile 与 pipeline 在 CuTe 层组合；适合新写、需要复杂 layout 的 kernel。
- **支持数据类型**：TF32 / FP16 / BF16 / I8 / U8。
- **不是什么**：不是 autotune 框架（没有 shape→plan 表）、不是运行时调度器（没有后端回退）、不是「调一个参数就变快」的黑盒。

## §1 示例清单（19 个，选型起点）

示例本身就是 schedule 样板，按「要解决什么」挑最近的一个改，比从空文件写快得多：

| 示例 | 对应问题 |
|---|---|
| `00_hytlass_basic_gemm` | 稠密 GEMM 的最小正确结构（先跑通它再谈优化） |
| `01_hytlass_serial_splitk_gemm` | K 很长、M×N 网格填不满 CU：串行 split-K |
| `02_hytlass_parallel_splitk_gemm` | 同上但要更多并行度：并行 split-K（多一份 reduction 成本） |
| `03_hytlass_streamk_gemm` | 网格与 CU 数不整除、尾波浪费明显：stream-K 把 K 切成更细的工作单元 |
| `04_hytlass_batch_gemm` | 多组同形状小矩阵（attention 的 QKV、多头） |
| `05_hytlass_group_gemm` | MoE 专家分组，每组 M 不同 |
| `06_hute_streamk` / `07_hute_batch_gemm` / `15_hute_group_gemm` | 同样三类，走 CuTe/HuTe 模型 |
| `08_hytlass_fused_gemm` | GEMM + epilogue 融合（bias / 激活 / 量化一类） |
| `09`–`13` 卷积族 | implicit GEMM、wgrad split-K、group conv、depthwise SIMT、融合 fprop |
| `14_gather_scatter_fusion` | gather/scatter 与计算的融合形态 |
| `16_hytlass_gemm_softmax` | GEMM + softmax（注意力类算子的骨架） |
| `17_ell_block_sparse_gemm` | ELL block 稀疏矩阵乘 |
| `18_gemm_with_abs_max` | epilogue 里带 amax 归约（量化前的统计量） |

> 挑法：先按「token/矩阵怎么组织」定 batch / group / 普通，再按「网格够不够」定要不要 split-K / stream-K，最后按「epilogue 要不要融合」定 fused。同一条需求的两套实现（2.x 与 HuTe）都在，对照着看是理解两种模型最快的方式。
> **两个特别值得先看的**：`16_gemm_softmax` 是 attention 类算子的骨架；`18_gemm_with_abs_max` 是「epilogue 里顺手算量化统计量」的标准做法 —— 这两条正是融合 epilogue 最容易做错的地方。

## §2 构建方式

- **依赖门槛**：**DTK 25.10+**、CMake 3.19+、**C++17**。
- **形态**：header-only 模板库 + CMake 示例工程；示例按 CMake target 编，编出来是独立可执行文件（自带计时/校验），不是 Python 模块。
- **接入自己的项目**：把 `hytlass` 的头目录加入 include path，按示例的 CMake 写法选 arch 与模板实例；每个 (dtype × tile × schedule) 是一个独立实例，**实例数会直接反映到编译时间与二进制体积**。
- **验证顺序**：先跑 `basic_gemm` 确认工具链与 arch 正确，再换成同 dtype 的目标 schedule，最后才动 tile。

## §3 何时选它 / 何时别选

- **选它**：要搭**一类** GEMM——非标准 layout、融合 epilogue、split-K / stream-K 调度、batch/group 形态，而现成库的入口对不上。
- **选它**：想把已有的手写 kernel 固化成可复用模板（换 tile / 换 dtype / 换 schedule 不用重写指令序列）。
- **选它**：要移植上游 CUTLASS 风格的 kernel 到 HCU，希望接口与设计思路保持相近。
- **选它**：**只是想知道某个 GEMM 结构在 HCU 上可不可行**——用示例做可行性实验比写完整 kernel 便宜。
- **别选**：已有现成入口覆盖的形状（普通 fp8/einsum 走 hipBLASLt，MoE 分组走 DeepGEMM，M≤8 decode 走手写 thin-M 桶）——模板的接入与编译成本花得不值。
- **别选**：只需要调 shape 级参数（tile/num_stages 表）而不改结构——那是在调优表里加条目，不是换模板库。
- **别选**：团队不熟 C++ 模板 / CuTe——类型报错的调试成本可能高于手写 HIP 的收益。
- **结构性吃亏**：它把「排流水线」的成本前移成一次设计成本，运行时**没有** shape 适配层；形状分布很散的工作负载（每档 shape 都要一个实例）在这种模型下是负担而不是优势。

## §4 与手写 HIP 的分界（重要）

- **示例是「跑得通」，不是「调优过」**。示例里的 tile / stage / 调度选择是 best practice 的示范，**不保证**是该形状的最优点；把示例参数当默认值上线，通常会输给针对该 shape 搜过的手写 kernel。
- **分界线**：要**一类** GEMM（多 shape、多 dtype、要复用结构）→ hytlass；要**一个** GEMM 的极限（单 shape、愿意维护 plan 层与编译 flag 集）→ 手写 ladder。
- **两者可叠加**：用手写 kernel 搜出的最优点（tile 高度、stage 数、split-K 切片）可以作为模板实例的参数，再用模板去覆盖其余 shape；但**不要**把「模板编出来的实例」直接当成「手写 kernel 的替代品」做基准对比，先确认两边 epilogue 与 padding 策略一致。
- **测量纪律不变**：改模板参数就是改编译产物，必须重编后再测；编译 flag 是测量的一部分（见 `ops/op-hip-compile`）。

## §5 已知边界

- **DTK 25.10+**、CMake 3.19+、C++17，低于此版本的工具链不支持。
- **上游是固定 commit**（CUTLASS 3.5.0 的某一个快照），不是 rolling 上游：升级要整块 re-port，不能只 diff 单个文件；上游 CUTLASS 的新特性不会自动进来。
- **没有 autotune / 没有 plan 层**：shape→实例的映射要自己写（这正是手写路线里 `op-stack-design` 说的「plan 层」缺口）。
- **arch 覆盖**：以仓内示例支持的 gfx 为准，换 arch 先看示例能不能编过、再谈性能。
- **不含 attention / 融合链**：它是线性代数模板库，attention 的 KV gather 与 softmax 不在它的表达范围。

## §6 相关 skill

- `dcu-kernel-dev`：手写 HIP 与 TileLang 两条路径的方法学（hytlass 是第三条：模板路径）。
- `dcu-deepgemm-source` / `dcu-deepgemm-arch`：现成 GEMM 的路径与调优表（先确认现成入口真的不覆盖，再上模板）。
- `ops/op-stack-design`：选型层，hytlass 在「匹配表」里对应「要做一类新 GEMM」那一行。
- `ops/op-hip-compile`：编译 flag 与 `.so` 检查（模板实例的编译产物同样适用）。
- `perf/dcu-op-optimization`：定了实现之后的一轮轮优化闭环。
