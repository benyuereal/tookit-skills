---
name: ops
description: "算子卡域：以算子为对象组织（一算子一目录）。最小卡 = SKILL.md + contract.yaml（机器可读契约）+ variants.yaml（实现变体矩阵）+ shapes.yaml（形状档位）+ _common.py（adapter：输入构造/参考/变体表）；bench/ 只在需要消融时写，正确性与不变量走包根 scripts/opcheck.py，ledger.md 与 candidates.yaml 在进优化闭环后才有。回答「这个算子在 DCU 上是什么、有哪几种实现、哪个形状档位用哪个、判据是什么、上次结论是什么」。与 dcu-operators（源码/框架索引）、dcu-perf（剖析与闭环方法）正交。触发场景：点名某个算子（GEMM/MoE/稀疏注意力/RMSNorm/topk/allreduce）、要给算子建卡、要查某算子的历史结论、要判断某个 kernel 改动算不算通过。"
version: 1.6.0
author: DCU Toolkit Contributors
license: MIT
---

# 算子卡域（ops）

## 一、定位：三个域正交

| 域 | 回答的问题 | 典型入口 |
|---|---|---|
| `dcu-operators` | 这个**框架/源码**怎么读、怎么写（aiter / lightop / triton / tilelang / deepgemm …） | `dcu-tilelang-source`、`dcu-lightop-moe` |
| `dcu-perf` | 怎么**发现热点**、怎么定优化目标、闭环怎么走 | `dcu-perf-profiling`、`dcu-op-optimization` |
| **`ops`（本域）** | 这个**算子**是什么、有哪些实现、哪个档位用哪个、判据是什么、上次结论是什么 | `op-sparse-mla`、`op-gemm-fp8` |

三者正交、互不替代：**找方法**去 `dcu-perf`，**找源码**去 `dcu-operators`，**认算子 / 查结论 / 跑基准 / 判通过**来本域。

### 第一站：算子图谱

不知道该从哪下手时，先看 [`op-atlas/INDEX.md`](op-atlas/INDEX.md)——一页表列出
每个算子的家族、状态、首选实现、上限与上次结论。图谱是**索引**：技术细节一律
指向 `op-stack-design/references/libraries.md` 的 §号与 `pipeline-laws.md` 的 L 号，
卡片里没有字段能承载技术描述（schema 白名单锁死，校验为
`python3 -I op-atlas/validate.py --atlas op-atlas`）。

> 引擎分层与决策路径（L0 认算子 → L1 选型 → L2 闭环 → L3 证据）见
> `../../docs/ENGINE.md`（开发态解析到包根 `docs/`；安装态解析到
> `<skills根>/_dcu-toolkit-shared/docs/ENGINE.md`，由 install.sh 同步）。

> 命名约定：本域 skill 一律 `op-` 前缀、不带 `dcu-` 前缀（`dcu-` 保留给源码/框架索引类 skill）。`op-` 前缀的作用是让路由表一眼可辨，并保证 `Plugins/ops/skills/` 下全是算子卡或算子卡方法论。

## 二、目录约定：最小五件套 + 按需四件

```
Plugins/ops/skills/op-<算子名>/
├── SKILL.md          ≤240 行：是什么 / 变体矩阵 / 档位表 / 判据 / 坑（带 source+verified_on）
│                     上限由 test_doc_links.py 卡（现状最长 op-contract 235 行）；超了先删，删不动再拆 references/
├── contract.yaml     * 机器可读契约：形状、dtype、布局约束、容差、不变量、门槛、地板
├── variants.yaml     * 实现变体矩阵：来源 → 适用档位 → 状态 → 约束
├── shapes.yaml       * 形状档位表，一档一个顶层键（decode-m1 / decode-m32 / prefill-8k / prefill-320k）
└── _common.py        adapter：本算子的输入构造 / fp32 参考 / 变体注册表 / 输入隔离
```

按需再加，**没有就不建，别放空占位**：

- `bench/`：只在需要**消融**（workset / config / split 这类只改一个变量的实验）时写。
  常规计时与正确性走包根 `scripts/opbench.py` + `scripts/opcheck.py`，不要每卡复制一套。
- `ledger.md` + `candidates.yaml`：进入优化闭环（`op-optimize/loop.py`）后才需要。
  绝对数字进 `~/.config/dcu-toolkit/kb/`，卡里只留相对倍数与结论。
- `references/`：深引用（源码位置表、知识卡 `cards.md`）。
- `dev/`：手写实现的 `.hip` 源码；`.so` 等编译产物不入库、也不随包分发。

> **不建 `tests/`**：门的流程只有一份（包根 `scripts/opcheck.py`：correctness / invariants / padding），
> 判据读本卡 `contract.yaml` 与 `variants.yaml`。卡内只放 adapter（`_common.py`）写本算子的数学。

**为什么这么分**（不是拍脑袋，是踩出来的）：

| 件 | 解决的实际问题 |
|---|---|
| `contract.yaml` | 没有契约时，「优化通过」= 模型自己觉得快了。有了契约，`scripts/opverify.py` 判，模型只负责解释 |
| `variants.yaml` | DCU 上一个算子普遍有 5 种实现（aiter AOT ASM / lightop / triton / tilelang / 手写）。只写「推荐用 X」是错的，要写「M≤16 用 A、M≥64 用 B、长上下文转 C、D 在 gfx936 不可用」 |
| `shapes.yaml` | 同一算子在 4096 和 65536 上可以差 6 倍，结论相反。结论必须挂在 `(算子, 档位, gfx, dtype)` 四元组上，否则互相污染。**一张表而不是一档一文件**：7 个档位共用一份 schema，一档一文件时改字段要改 7 处、漂了看不出来 |
| `_common.py`（adapter） | 不变量（确定性/batch 独立/index 置换）与「输入是否被原地改写」这类结构错误，参考对比抓不到；流程共用 `scripts/opcheck.py`，卡内只提供本算子的数学 |
| `ledger.md` | 防重复劳动：下次进来先看台账，`at_floor` 的直接回「已贴地板」 |

## 三、状态机（`data/ops/registry.json`）

```
draft ──► measured ──► tuned ──► at_floor ──► deprecated
             │            │          │
             └────────────┴──────────┘
                   （回退需写理由）
```

| 状态 | 含义 | 进入条件（脚本可判） |
|---|---|---|
| `draft` | 只有卡，没有实测 | — |
| `measured` | 有该档位的基线数字 | `scripts/opbench.py` 产出 METRIC 行 + report json（卡内 `bench/` 只在需要消融时才有） |
| `tuned` | 有改动且通过 A/B 显著性 | `opverify.py --run` 返回 PASS 且加速比过噪声门 |
| `at_floor` | 已贴物理地板，**不许重开** | 达成率 ≥ 85% 峰值（带宽/算力），且地板依据写进 `contract.yaml` |

> caveat **「贴地板」有两个阈值，别混**：本表的 **85%** 是**卡的生命周期状态**（机器值 `contract.yaml:at_floor_threshold`，判定状态机走到哪一档）；`op-gate` 的 `OP-GATE-001` 二级门 **实测/地板 ≤ 1.5×（≈67%）** 是**「物理到顶」收工声明**的必要条件之一，还要叠加一级门（无合法减量）才成立。两者用途不同、**不得互相替代**：85% 到了不等于可以发「物理到顶」，≤1.5× 也不等于卡可以标 `at_floor`。
| `deprecated` | 实现被替代 | 写 `superseded_by` |

**硬规则**：`at_floor` 必须带 `floor.basis` + `evidence`（可回溯路径），且必须在 `ledger.md` 留一行。没有依据的 `at_floor` 视为无效——这正是我们反复重开同一个优化（Marlin GEMM 被误诊三次）的原因。

## 四、起一个新算子（五步，每步一条判据）

已有的算子直接查 §五 的索引；**还没实现的算子从这五步走**。

| # | 做什么 | 判据 / 常见错 |
|---|---|---|
| 1 | **写公式**：输出是什么、输入是什么、哪一维是收缩维 | 写不出来就别开始 |
| 2 | **算两个地板**：FLOPs 与 bytes → 绑定 = max(算力, 内存)。**按 kernel 实际实现的复用算，不按理论值** | **写不出模型 = 还没理解它为什么慢，不许写 kernel**（`op-roofline`） |
| 3 | **普查**：谁已经实现了（torch / 厂商库如 lightop / 开源）、参考用什么、容差定多少、真实形状从哪来（模型 config / 生产 trace） | **没做这步，后面所有「快了 X%」都没有对手** |
| 4 | **建卡**：`contract.yaml`（**先写**）→ `shapes.yaml` → `variants.yaml`（含「不可用」的）→ `_common.py` → `SKILL.md`；登记 `data/ops/registry.json` 为 `draft` | 计时用 `scripts/opbench.py`，**禁止手搓 time.time()**；不建卡内 `tests/`，门走 `scripts/opcheck.py` |
| 5 | **先测基线再写自己的**：拿现成实现测准，再决定要不要写 | 基线要同进程现测，别引用历史数字 |
| **5b** | **自研实现超过基线 1.3× 时，先把它做到接近基线，再谈优化** | 否则你测的是自己的实现水平，不是那个假设。本项目实测：按「折 query 省 K 流量」写的第一版比 lightop 慢 6.7×，而慢的原因是 GEMM 形状太小 + 按 h 归约，与访存无关 —— 差 6.7× 时谈带宽优化没有意义 |

### 算例：MQA logits（DSA indexer 打分）—— 第 2 步的坑

`logits[i,j] = Σ_h max(0, Q[i,h,:]·K[j,:]) · W[i,h]`，H=32、D=128、fp8。

**算术强度必须按「kernel 实际实现的复用」算**：真实 decode 内核的 grid 是「KV 块 × batch」，
query 之间**没有复用** ⇒ K 被整份读 `ceil(S/q_per_ktile)` 遍 ⇒ `AI = 2·q_per_ktile·H/bytes`。

| q_per_ktile | AI | 绑定 |
|---:|---|---|
| 1（现成 decode 内核） | 64 FLOP/B | **访存受限** |
| next_n（把投机 token 并进 N 维） | ×next_n | 向算力移动 |
| BM（prefill 的正常 tile） | 8192 FLOP/B | 算力受限 |

机器平衡点 ≈ 313 FLOP/B ⇒ **decode 档的杠杆是「把跨 query 的复用并进 GEMM 的 N 维」，不是调 tile**。
（这就是模型初版把方向算反、后来实测更正的那条，见 `op-roofline/models/mqa_logits.py`。）

## 五、本域 skill 索引

### 5.1 唯一权威表（同一个问题只信一处）

同一件事在多个 skill 里都出现过时，**以「唯一权威」列为准**；其余位置只作导读或历史格式，不重复定义。
两处说法不一致，按本表指向的那份执行。

| 问题 | 唯一权威 | 其他地方的定位 |
|---|---|---|
| 什么叫「通过」/ 四档决策 | `ops/op-gate` | `dcu-op-optimization` 只作方法学导读，引用不重复定义 |
| 怎么量（口径、MAD、显著性） | `ops/op-bench` + `scripts/opbench.py` | — |
| 天花板是多少 | `ops/op-roofline` 的 `derive.py` | 任何别处写的达成率都必须引用它 |
| 这个算子是什么 / 上次结论 | `ops/op-<算子>` 卡 + `data/ops/registry.json` | — |
| 候选队列 / 中断续跑 | `ops/op-optimize` 的 `loop.py` | `dcu-op-optimization` 的 `candidates.jsonl` 是历史格式，**已被 `loop.py` 取代** |
| PMC 怎么读 | `ops/op-profiling` + `scripts/pmc.py` | — |
| 实测流式带宽 / 访问模式上限 | `scripts/probe_stream.py` / `scripts/probe_pattern.py` | 达成率分母的唯一来源：前者给 read/copy/write 流式峰值，后者给「行内连续 X 字节」的 pattern ceiling |
| 选哪个库 / 流水线怎么排 | `ops/op-stack-design` | `dcu-operators/*` 只讲源码地图 |
| 多轮可恢复的编排 | `dcu-core/dcu-orchestrator` | `humanize-rlcr` 是其中一种 harness 实现 |

> 新工作一律写 `candidates.yaml`，不要再用 `candidates.jsonl`；约定与判定看 `op-gate`，不要在本域之外重写。

### 5.2 本域 skill 列表

状态列就是 `data/ops/registry.json` 的 `status`（由 `Plugins/ops/tests/test_doc_links.py` 校验一致）：
**先看这一列决定进哪张卡，不用逐个翻 registry**。方法学 skill 没有状态机，记 `—`。

| skill | 状态 | 定位 |
|---|---|---|
| `op-contract` | — | 契约怎么写：字段规范、容差分级、不变量清单、地板依据 |
| `op-bench` | — | 怎么量：口径三问、六条铁律、MAD 显著性、地板判定、跨卡 ratio |
| `op-profiling` | — | 实测卡在哪：PMC 硬件计数器怎么读→三级证据 + 口径陷阱表 + 可执行解析器 `scripts/pmc.py`（唯一权威实现）；**和 op-bench 是两套口径，别混** |
| `op-gate` | — | 怎么判：四道门（执行身份→正确性→不变量→A-B）、四档决策、回退与留痕 |
| `op-optimize` | — | 怎么找下一个改动：候选队列→单实验→机器判定→落盘记账；**单模型、可中断、不重复劳动**（配 `loop.py`） |
| `op-stack-design` | — | 选哪个库、流水线怎么排：十一库画像（结构定律/可调面/已知遗留税/何时选它）+ 十条流水线定律，**写代码前排掉结构性不可行的设计**；单算子突破换成整库选型 |
| `op-roofline` | — | 改动从哪来：物理常数→推导规则→算子模型，**先出预测与绑定约束再写代码**（配 `derive.py` / `hipbudget.py`）；写不出预测说明还没理解为什么慢 |
| `op-source-patch` | — | **源码级改动**：定位→取源→编译→改(patch)→回归→留痕六步。把「卡片说哪个实现好」和「源码怎么改掉编出来验回去」接上；本地已有源码不重拉；只允许 patch 不直接改树 |
| `op-hip-compile` | — | 手写 HIP 的编译调优：为什么要 aicc 不要 hipcc、怎样**逐字镜像** tilelang 的 `-mllvm` 集（含 FAST_MATH 条件项，漏一条慢 1.16×/CTA 而数值不变）、从 .so 设备 ELF 读 VGPR 算占用率（配 `derive_flags.py` / `inspect_so.py`） |
| `op-sparse-mla` | `tuned` | 稀疏 MLA 前向（DSA）：q/kv/indices 契约、tilelang / flash_mla / 生产三变体、-1 哨兵语义 |
| `op-moe-w8a8` | `draft` | FP8 MoE（routed GEMM1/GEMM2 + silu-mul + moe_sum）：M 分档、融合收益判据 |
| `op-gemm-fp8` | `measured` | FP8 GEMM：hipBLASLt 在 gfx938 无方案，aiter opus 实测 49.8 TF（比 bf16 慢 5.3×） |
| `op-moe-topk-gating` | `draft` | **MoE 第 2 段**：gate→topk→归一化。索引是离散量必须逐位一致；tie-break 需固定 |
| `op-moe-align-permute` | `draft` | **MoE 第 3 段**：align_block_size + permute。`-1` 只能垫队尾，前置会让输出恒为 0 |
| `op-fused-add-rmsnorm` | `draft` | 残差 + RMSNorm 融合：**校准件**，字节数完全确定；用 lightop.RMSNormAdd。2026-11-25 降 draft（缺可复跑证据） |
| `op-rmsnorm-fused` | `measured` | 融合 RMSNorm：等语义（2读2写）242 µs/1108 GB/s；比 lightop 227 µs 慢 6.7%，融合相对分离 1.14x |

## 六、数据分层（红线，必须遵守）

| 层 | 放什么 | 是否入 public 仓 |
|---|---|---|
| 算子卡（本目录） | 方法、模板、schema、**相对倍数**、物理地板常数（峰值带宽/算力） | 是 |
| `data/ops/registry.json` | 状态机 + 地板判定 + 档位清单 | 是（只放状态，不放绝对延迟） |
| `~/.config/dcu-toolkit/kb/op-opt/` | 真机**绝对数字** + 环境指纹 + 证据路径 | 否 |

理由见 `data/kb/README.md` 条目纪律第 3 条：性能数字随硬件/DTK/版本变化，写进 skill 会变成过期真理。**算子卡里只允许出现两种数字：物理常数（峰值带宽、理论 TFLOPS）与相对倍数（x.xx 倍、±x%）。绝对 µs/ms 一律进 `kb/`。**

## 七、与兄弟域的关系

- 方法学闭环（profile→定位→改→验证→沉淀）→ `dcu-perf/dcu-op-optimization`；本域提供它缺的「算子对象」与「契约」。
- 多步编排（gate / 审批 / 回滚 / 证据）→ `dcu-core/dcu-orchestrator` 的 `op-optimize` 工作流；本域的 `contract.yaml` 就是该工作流 `op-validate` 节点的验收依据。
- 框架/源码细节 → `dcu-operators/*`；本域 `references/` 只放指针，不复制源码索引。
- 故障（VM fault / hang / 精度）→ `dcu-troubleshooting/*`。

## 八、使用须知

- 单步任务直接点名 `op-<算子>`，不必经 `dcu-skill-router`。
- 本域不存源码树、不存绝对性能数字、不存内网地址。
- 新增/修改算子卡时，`contract.yaml` 与 `SKILL.md` 必须同时改；`opverify.py` 只认契约，不认文档。
