---
name: op-profiling
description: "算子性能剖析：PMC 计数器采回来以后怎么读、怎么判瓶颈——本 skill 不提供解析脚本，只给一份可直接照抄的参考实现与判据：稳态行怎么选、噪声 kernel 怎么排、TA_* 怎么跨 SE 求和、哪些计数器不存在，以及三级证据纪律（贴顶类可当结论 / 比值只用于区分变体 / 描述性事实不能当结论）。触发场景：拿 PMC 读数判断「卡在算力还是内存还是指令发射」、外包或自己写一个 PMC CSV 解析、决定哪几组计数器能拼成一条证据、跨变体找区分项。不适用：PMC/trace 怎么采、hipprof 工具本身怎么跑走 dcu-hipprof；先算上界/地板（不测、按机器模型推导）、能不能宣布到顶走 op-roofline；bench 报告口径与统计走 op-bench；契约容差与 floor 写法走 op-contract。"
version: 3.0.0
author: DCU Toolkit Contributors
license: MIT
---

# 性能剖析：把计数器读成判据

## 一、职责与边界（只做一件事：把一次 PMC 采集读成能下判断的证据）

| 做 | 不做 |
|---|---|
| 给出「怎么读 PMC CSV」的**参考实现 + 判据**（§四 三级证据、§五 口径陷阱、§六 可直接照抄的代码） | 不维护解析脚本：解析代码由 agent/人按 §六 现写，40~60 行标准库即可 |
| 说明每一行数字的口径与来源，缺的写 `unknown(<原因>)` | 不内置任何硬件常数；不在代码里下结论（没有 `bound`/`facts`/`ratios` 判定字段） |

**唯一产出**：一份「采集 → 解析 → 判读」的方法与可复制代码。数字只出原值，结论由写报告的人按 §四 得出。

## 二、什么时候用

按要回答的问题选工具，不要一律上 PMC（有 overhead，且首次 launch 数据不可用）：

| 想问 | 用什么 | 为什么不是 PMC |
|---|---|---|
| 这个改动快了多少 | `op-bench`（`scripts/opbench.py`） | PMC 墙钟只到 ±1%，且有 launch 虚高 |
| 卡在算力还是内存 | **本 skill**：先按 §六 拿原值，再按 §四 自行算 MBU/MFU | 计时测不出「上限在哪」 |
| 片上开销在哪 | **本 skill**：原值里取 VALU/LDS，自行算 `非MFMA VALU/MFMA`、`LDS/MFMA` | 比值比绝对条数稳，能跨形状比 |
| 多个变体并排比 | 用 §六 同一份参考实现、同一口径各算一遍再并排 | 口径不同就没有可比性 |
| 某段指令占多少 ⇒ 删掉省多少 | **都不该这么问** | 见 §五 陷阱 6 |

## 三、采集与解析

```bash
HIP_VISIBLE_DEVICES=<idle> hipprof --pmc --pmc-type 3 -o /tmp/pmcout/x -- python3 bench.py
```

解析**没有现成可执行文件**（刻意的：长期维护一个 CSV 解析器不划算）。复制 §六 的参考实现到临时脚本，改几行 `COUNTERS` 即可；多项目共用同一套规则，别另起一套。

## 四、输出怎么读（判据，必须由人执行）

机械输出只给数字。把它读成证据时，**三类证据地位不同，不能混用**：

| 类别 | 含义 | 能不能当结论 |
|---|---|---|
| 贴顶类（`bound` 语义） | `NEAR-BANDWIDTH-CEILING` / `NEAR-COMPUTE-CEILING` / `MMA-ISSUE-SATURATED` | **能**，但仅当**口径齐全且分母可信**：四个输入（FLOPs、DRAM 字节、实测峰值、CU 数）一个不缺、峰值按同 SoC 实测并带来源与日期、天花板取自 `op-roofline/derive.py`。缺一项或分母未锁 ⇒ 结论**扣下**（`withheld_bound` 语义），不许写 |
| 比值类（`ratios` 语义） | `NONMFMA-VALU/MFMA`、`LDS/MFMA` | **能区分变体**，不能说明绝对上限；必须同 FLOPs / 每个 MMA 的 FLOPs / CU 口径 |
| 描述类（`facts` 语义） | `CTA/CU=1`、`issue_util<25%`、`L2 命中率≥80%`、`VALU/LDS 开销重` | **不能**，描述性事实，多个内核会同时命中 |

**为什么必须分三级**：实测 `flash_mla` 与 tilelang `bm32` **驻留 CTA 数相同（都是 2，2026-11-13 用 LDS 探针复核）、`issue_util` 都低于 10%**，两者却差 **2.18×**；把 `CTA/CU` 这类描述写成「占用率受限 ⇒ 有优化空间」，会得出「快 2.18× 的那个也有问题」的荒谬结论。真正的区分项是 `非MFMA VALU/MFMA`（0.73 vs **3.11**）与 `LDS/MFMA`（0.53 vs **1.67**）。

判读顺序（ledger §二 24 / KB §11.5）：

1. 算力：`mfma_issue_pct ≥ 75%` 才算被算力卡住（我们 18%，fm 38%，都不是）。
2. 内存：DRAM 字节 + `TA 数据停顿` + 池子敏感性**三个一起看**。本项目三项都指向「不缺内存」：字节与 fm 相等、TA 数据等待只有 fm 的 0.09×、池缩 64× 只快 1.05×。
3. 片上开销：只看两个比值（`非MFMA VALU/MFMA`、`LDS/MFMA`）。
4. **别把 bank 冲突当优化目标**：厂商调优 matmul 自己有 12.6% 冲突占用率照样到 88% 峰值，我们 bm32 只有 9.8%；真区分项是 LDS 指令条数与非 MFMA 向量指令条数。

### 瓶颈类型 → 允许的动作（人类判据表，不是脚本）

这张表**不是可选项、也不是解析器的输出**：它规定「拿到一条证据之后，下一步允许做什么」。动作名是**攻击面**（对应 `op-optimize` 候选的 `direction`），不是一条命令。输入必须是上文的三级证据——**描述类事实不能单独作为输入**。

| 瓶颈类型（判据） | 允许的动作 | 不允许 |
|---|---|---|
| 算力饱和（`mfma_issue_pct ≥ 75%`，四个分母齐全且峰值取同 SoC 实测） | `reduce_flops`：换更宽的 MMA / 减少 FLOPs / 换算法 | 继续扫 tile、`num_stages`（救不了分母）；分母不全时写「到顶」 |
| 内存饱和（MBU 高 **且** DRAM 字节与对手相当 **且** TA 数据停顿、池子敏感性同向） | `reduce_bytes` / 提复用（换布局、加大 reuse、按有效量折算） | 换分母（标称 vs 实测峰值）；拿「L1 命中率」当证据（该计数器不存在）；只凭 MBU 高就下结论 |
| 片上搬运饱和（比值类：`非MFMA VALU/MFMA`、`LDS/MFMA` 显著高于对手） | `reduce_instructions` / `restructure`（换布局或换算法，减少每条 MMA 配的搬运指令） | 当成内存饱和去减 DRAM 字节；把 `SQ_LDS_BANK_CONFLICT` 当优化目标（厂商 matmul 12.6% 冲突照样到 88% 峰值） |
| 并发受限（`CTA/CU=1` 只是描述类事实；须叠加寄存器/LDS 预算才成立，见 `op-roofline` R3/R6） | `increase_concurrency`：先算预算，再改分片 / 压 VGPR / 压 LDS | 直接把「占用率受限」写成结论；不探测常数就外推收益 |
| 描述性事实命中「占用率受限」（`CTA/CU`、`issue_util`、L2 驻留） | **只能 `gather_evidence`**：补比值类或贴顶类证据 | 任何 `reduce_*` / `restructure` / `increase_concurrency` 动作；直接立项或写进报告结论 |
| **算力与带宽都是个位数 %**（如都 <10%） | 既不是算力也不是带宽：**每 tile 工作量太小 / CTA 太多**（小 M 算子常见，如 M≤8 的打分算子） | 继续调 tile 形状或换 MMA 宽度；真正的杠杆是 **persistent CTA**（减少 CTA 数与每 CTA 的 prologue） |
| 证据不足（分母缺、口径不统一、首次 launch 或噪声行未排除） | **只能 `gather_evidence`** | 任何优化动作；写「到顶 / 物理上限 / 没有空间」 |

两条与三级证据对齐的纪律：① 同一行里的「判据」必须**同时**满足才允许该动作，单条描述类事实永远不够；② 证据不足时表里唯一合法动作是 `gather_evidence`——这就是「口径不全只允许补证据」的执行版。

## 五、口径陷阱（全部实测坐实）

每条都曾真实读错一次；§六 的参考实现把选行/排除噪声/跨 SE 求和写进了代码，但引用数字时你仍要知道口径。

| # | 陷阱 | 错误做法的结果 | 正确口径 |
|---|---|---|---|
| 1 | **PMC 下首次 launch 的 GRBM 虚高 ~40%** | bm32 取首行 1.83M vs 稳态 1.29M，墙钟虚高 | 取**单次 launch 中 GRBM 最小**的行（`--row steady`，默认） |
| 2 | **自动选主 kernel 必须排除 torch/rocprim** | `randperm` 的 `normal_and_transform` 单次 GRBM 1.29M，比主 kernel 1.14M 还大 ⇒ 表里出现 VGPR=48/LDS=0 的假数据 | 排除 `_ZN2at`/`_ZN7rocprim` 等前缀，再按**单次** GRBM 取最大 |
| 3 | **每卡 CU 数不是常数** | 硬编码 69，本机实为 69/70/71 ⇒ 同一 CSV 的 cycle/MMA 差 ±6% | 当场探测（dmesg SE 位图）或 `--cu` 显式传；探测不到写 `unknown`，不猜 |
| 4 | **`TA_*` 按 CU 编号跨 8 个 SE 求和** | `max(TA_TA_BUSY)/GRBM` = **352%**（>100% 的荒谬值） | `Σ TA_TA_BUSY[n] / (CU × GRBM)` ⇒ 33%/27% |
| 5 | **CU 编号只有 0..8 存在**，9..15 恒 0 | 逐编号均值/中位数、横向比大小全无意义 | 只求和（每个活跃 CU 恰计一次，不受屏蔽影响） |
| 6 | **「某段指令占 7.5%」≠「删掉省 7.5%」** | `reord` 变体删掉那条 16 标量 LDS 拷贝，反而**慢 1.1%** —— 它实际是 prefetch | 用消融实测（`op-sparse-mla/bench/bench.py --mode workset\|config`），不用占比推算 |
| 7 | **`SQ_INSTS_VALU` 含 MFMA** | 直接当纯向量指令数 ⇒ 片上开销被高估 | `非MFMA = SQ_INSTS_VALU − FLOPs/8192`。别用 MACs/4096，差 1.23×；解析代码只标注「含 MFMA」，不替你拆 |

以及两条硬事实：

- **L1 不存在命中率计数器**：只有 `TA_TA_BUSY`（忙）和 `TCP_TCP_TA_DATA_STALL_CYCLES`（数据缺失停顿）的周期占比；报告里的「L1 命中率 x%」都是编的。
- **`TA_FLAT_READ_WAVEFRONTS` 只统计 flat（通用寻址）访问**：常规 global load 走 buffer 不计数（GEMM 核实测 61 次 launch 合计仅 4096）；量访存 volume 用 `SQ_INSTS_VMEM_RD/WR`。

## 六、参考实现：怎么读一份 PMC CSV

完整参考实现见 **[`references/pmc_reference.py.md`](references/pmc_reference.py.md)**（标准库 ~60 行，可直接复制到临时脚本）。它只负责**选对行、排掉噪声、按实例求和、打印原值**，不解释、不判定；按本次要看的计数器改 `COUNTERS` 即可。**为什么必须这么做**（每条对应一次真实读错，数字见 §五）：

- **选行与噪声**：稳态取单次 launch 中 GRBM 最小的行（首行虚高 ~40%）；自动选主 kernel 先排 `_ZN2at`/`_ZN7rocprim` 前缀，否则 `randperm` 内部 kernel 会夺冠（单次 GRBM 比主 kernel 还大）。
- **跨实例求和**：`TA_*` 的 `[n]` 是按 CU 编号跨 8 个 SE 分开的实例，只能求和；`max(TA_TA_BUSY)/GRBM` 会得 352% 的荒谬值，均值/逐编号比大小也无意义。L1 没有命中率计数器，只有 busy/data-stall 的周期占比。
- **计数器语义**：`SQ_INSTS_VALU` 含 MFMA（要 `非MFMA` 自己减 `FLOPs/8192`）；`TA_FLAT_READ_WAVEFRONTS` 只统计 flat 寻址，量 volume 用 `SQ_INSTS_VMEM_RD/WR`。

### 纪律

- **只输出原值，不下结论**：MBU/MFU、`非MFMA VALU/MFMA` 要不要算、拿什么当分母，由写报告的人自己定；**分母必须和数字写在同一行**。
- **硬件常数一律当场探测或显式传入**：CU 数、寄存器预算、LDS 容量、每 MMA 的 FLOPs、MMA 宽度、MFMA 发射率、L2 miss 搬运字节，都不许写死默认值（gfx938 实测：768-VGPR 预算 `196608`、LDS 64 KB/CU、bf16 MMA `16×16×16` / fp8 `16×16×32`；其他 gfx 未验证）；探测不到写 `unknown(<原因>)`。反例：`VF_LIMIT=65536` 是「非 768-VGPR 模式」的预算，本项目 kernel 全部走 768-VGPR，它把 `CTA/CU` 从 2 算成 1，进而得出一条 verified 的错误结论。反验：hbird-gemm「172 VGPR × 512 线程实测 2 CTA/CU」——65536 会得 **0 个 CTA**。**任何常数推出物理不可能的结果（0 个 CTA、>100% 占用率）时，先判常数错，不判数据错。**

## 七、与其它 skill 的关系

计时口径的唯一来源是 `op-bench`（PMC 墙钟 ≠ bench 耗时）；`op-roofline` 负责「应该多快 + 绑定约束 + 唯一权威天花板 `derive.py`」，本 skill 负责「实际采到了什么」，对不上就是模型错；人算出的贴顶证据喂给 `op-gate` 的两级天花板门禁，比值作为 `op-optimize` 候选的 `expect`。本 skill 不提供可执行解析器，**同一套规则只有 §六 一份**。

## 八、红线

- 绝对 µs/ms/GB/s 只进 `~/.config/dcu-toolkit/kb/op-opt/`；本 skill 只放**相对比值**与**口径**。
- 峰值带宽必须按 SoC 显式给定（本机实测 1724 GB/s，不是型号标称 1100 —— 换分母能把「到顶了」变成「还有 1.8×」）；缺峰值时 MBU 不成立，不许拿标称值凑。
- 口径不全（缺 FLOPs / DRAM 字节 / 实测峰值 / CU）或分母不可信时，报告里**不得**出现「到顶」「物理上限」「没有空间」；此时只允许补证据。
- **天花板只有一个权威点：`op-roofline/derive.py`**。§六 的参考实现不自算 `ceiling_us`，也不接受「临时给个字节数估算」——这正是 D1（同一份数据 38.9µs vs 77.9µs 两个天花板并存）的成因；**一个数只能有一个分母、分母与数字写在同一行**，引用达成率必须同时写明分母来源与测量日期。同一个数两处算必然漂，87%/55%（标称 vs 实测峰值）也是这么来的。

## 知识卡索引

本 skill 踩过的坑已沉淀为 `references/cards.md`（一卡一段；六段固定：知识点 / 错误写法 / 正确写法 / 识别方法 / 证据要求 / 参考）；卡片只放物理常数与相对倍数，绝对性能数字在 `~/.config/dcu-toolkit/kb/op-opt/`。

| id | 一句话结论 | 卡片 |
|---|---|---|
| `OP-PROFILING-001` | 描述性事实不能当结论：多个 kernel 共享同一几何会同时命中 | [OP-PROFILING-001](references/cards.md#OP-PROFILING-001) |
| `OP-PROFILING-002` | 口径不全（缺分母/证据）时禁止写「到顶」 | [OP-PROFILING-002](references/cards.md#OP-PROFILING-002) |
| `OP-PROFILING-003` | 计时口径：median + MAD，单次/均值不得当结论，frame 计时会虚高 | [OP-PROFILING-003](references/cards.md#OP-PROFILING-003) |
