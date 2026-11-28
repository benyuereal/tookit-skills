# 代码地图（CODE_MAP）

> 用途：**按「我要干什么」找文件，不按文件名猜。**
>
> 本文件由 `scripts/` 与 `Plugins/ops/engine/` 的实际文件生成，
> 并由 `Plugins/ops/tests/test_code_map.py` 守住：新增 .py 而不更新地图，测试会红。
> 地图会腐烂，除非有人守。

## 一、分层约定（先看这一节，再看表）

| 位置 | 是什么 | 规矩 |
|---|---|---|
| `Plugins/ops/engine/` | **库** | 无 CLI（`loop.py` 例外：它本身是驱动器）。不做 IO 之外的副作用 |
| `scripts/` | **CLI / 一次性工具** | 薄，负责 argparse + 打印 + 退出码；实现尽量沉到 engine/ |
| `Plugins/ops/skills/op-*/` | **卡**（数据 + 该卡的 bench 适配器） | 卡里只放这一张卡独有的东西：contract / shapes / candidates / bench/ |
| `Plugins/ops/tests/` | **契约测试** | 每条测试对应一条纪律，不是覆盖率 |

**同名不是笔误，是分层**：`scripts/floor.py` 是 CLI，`engine/floor.py` 是实现；
`scripts/explore.py` 是 CLI，`engine/propose.py` 是实现。找实现去 engine/，找命令去 scripts/。

## 二、命名约定

| 前缀 | 含义 | 例子 |
|---|---|---|
| `load_*` | 只读数据，不做判定 | `card.load_candidates` |
| `resolve_*` | 从多个来源合成**一个**值；合不出来不抛异常，把缺什么写进 `missing` | `floor.resolve` |
| `build_*` | 构造（落盘前） | `op_bootstrap.build_card` |
| `check_*` | 返回问题列表，不抛异常 | `budget_problems` |
| `cmd_*` | `loop.py` 的一个子命令 | `cmd_run` |
| `_engine("x")` | 延迟加载兄弟模块，避免包依赖 | 各 CLI 的入口 |

## 三、我要做什么 → 看哪个文件

| 我要… | 从这里进 | 说明 |
|---|---|---|
| 来一个全新算子（只有题目和数学式） | `scripts/op_bootstrap.py` | 算分母、建 9 个文件、自动登记；建完就能 `loop run` |
| 跑一次实验并记账 | `Plugins/ops/engine/loop.py -C <卡> run <候选>` | 六道门 + 精度门 + 落账 + 写回 atlas |
| 让引擎自己多跑几轮 | `scripts/advance.py` | plan → gate → run → judge → record → replan |
| 卡上没有候选了，出几道题 | `scripts/explore.py` | 从 roofline/带宽/占用率出**拟做**；不是待做 |
| 知道「还能优化多少」 | `scripts/floor.py` | 内存/算力地板 + remaining + gate |
| 判一个实现对不对 | `scripts/opcheck.py` | 正确性 / padding / 边界 |
| 判一次 run 算不算数 | `scripts/opverify.py` | 按 contract.yaml 裁决 |
| 看硬件还能榨多少 | `scripts/pmc.py` / `probe_pattern.py` / `isa_probe.py` | PMC / 访存上限 / 汇编 |
| 查「这个操作以前谁做过」 | `scripts/atlas_query.py` / `scripts/recall.py` | 图谱查询 / 跨卡检索 |
| 算分：深度/覆盖/质量 | `scripts/scorecard.py` / `scripts/depth.py` | 三维可机检 |

## 四、全量清单（按角色分组，共 86 个文件）

### ① 驱动一次优化（入口）

| 文件 | 职责 |
|---|---|
| `scripts/op_bootstrap.py` | 从「一个算子题目」到「一张能跑的卡」：算分母、建 9 个文件、登记进仓库 |
| `Plugins/ops/engine/loop.py` | 队列 → 单实验 → 机器判定 → 落盘记账。**六道门都在这里**，也是唯一会跑 bench 的地方 |
| `scripts/advance.py` | 多轮编排：plan → gate → run → judge → record → replan，一轮没人值守也能走完 |
| `scripts/explore.py` | 没有候选时从通用信号出题（**拟做**，不是待做）。CLI，实现在 engine/propose.py |
| `scripts/floor.py` | 「还能优化多少」出一个数。CLI，实现在 engine/floor.py |

### ② 库（engine/，无 CLI）

| 文件 | 职责 |
|---|---|
| `Plugins/ops/engine/card.py` | 读算子卡的**唯一入口**：两种 candidates 顶层格式、overlay 合并、五段缺省 |
| `Plugins/ops/engine/floor.py` | 地板唯一实现：derive 模型 → roofline.json → atlas 卡，取不到不抛异常、把缺什么写进 missing |
| `Plugins/ops/engine/signals.py` | 采信号：derive 几何量 / runs 达成 / 已实现变体 / 方向预算，集中取一次 |
| `Plugins/ops/engine/propose.py` | 信号 → 候选（七条规则）+ 人读渲染。**对外文本是契约**，改话术要当改接口 |
| `Plugins/ops/engine/ledger.py` | 记账：候选计划、判死/停滞流水、overlay、进度文件、judge 判定搬用 |
| `Plugins/ops/engine/cli.py` | explore / floor 的薄 CLI 层：只解析参数、打印、定退出码 |
| `Plugins/ops/engine/judge.py` | 从 description+shapes 生成 judge、GPU 校准并**冻结容差**（fail-closed） |
| `Plugins/ops/engine/ablate.py` | 消融流水线：champion 逐项删方法 → 引擎测 → 自动生成 ablate spec（归因仍由 scripts/ablate.py 算） |
| `Plugins/ops/engine/guard.py` | 命令级守卫：测量只能走 loop.py run；只读查看放行（D1 的代码级强制） |
| `Plugins/ops/engine/nextstep.py` | 唯一的策略实现：算出「下一步做什么」（run/implement/explore/fix/stop） |
| `Plugins/ops/engine/orchestrator.py` | **循环拥有者**：iterate 控制流 + CLI（可 fake 单测） |
| `Plugins/ops/engine/driver.py` | 兼容 shim：转发到 orchestrator（旧命令/测试仍可用） |
| `Plugins/ops/engine/contract.py` | 迭代契约：IterationRecord / Backend 接缝 / 判定与噪声策略 |
| `Plugins/ops/engine/prompt.py` | 给 agent 的结构化 prompt（反馈块 + 落卡硬约束） |
| `Plugins/ops/engine/backends.py` | 三类 backend：Fake / WorkspaceGit / ClaudeAgent / EngineEvaluator 的组合门面 |
| `Plugins/ops/engine/iterate.py` | 工作流反转的 CLI：agent 只交实现，测量/判定/记账由引擎做 |
| `scripts/coordinate.py` | 进程级协调者：在不依赖模型自觉性时验证引擎能否闭环（非产品主路） |

### ③ 判「对不对 / 算不算数」

| 文件 | 职责 |
|---|---|
| `scripts/opcheck.py` | 算子卡的共用检查模板：正确性 / padding / 边界，别再给每张卡抄一份测试 |
| `scripts/opverify.py` | 按 contract.yaml 判定一次 run 是否通过（done_when 的外部裁决点） |
| `scripts/check_semantics.py` | 比较两个实现前，先断言它们干的活是一样的 |
| `scripts/check_means.py` | 手工候选必须先写清算理，再开跑 |
| `scripts/check_expect.py` | 候选的 expect 必须是数值（散文要显式带 expect_confidence: low） |
| `scripts/check_card_semantics.py` | 卡片与源文件语义一致性检查（D8 的机器判据） |
| `scripts/check_schema_fields.py` | 加字段前先问一句：这个字段 schema 认不认 |
| `scripts/check_transform.py` | C1 验收：每条 surface 必须能落到「具体改法」 |
| `scripts/check_diagnosis.py` | 诊断记录必须带「假设 → 判定」四字段 |
| `scripts/check_curriculum.py` | surfaces.yaml 与 curriculum.yaml 的一致性 |
| `scripts/check_s_direction.py` | 找出「S 项满分但整体没变快」的卡（方向门） |
| `scripts/anticheat.py` | 盲测前的最后一道闸：实现里不许夹带题目特定信息 |

### ④ 量得准（测量原语）

| 文件 | 职责 |
|---|---|
| `scripts/opbench.py` | 算子微基准的通用计时与统计（标准库优先，torch 可选） |
| `scripts/measure_mode.py` | 测量模式的单一真源。**它决定一个结论能不能被 claim** |
| `scripts/bench_preflight.py` | 跑任何 bench 之前的前置检查。**默认开，不是可选项** |
| `scripts/efficiency.py` | MFU / MBU 利用率计算器（分母带来源，放置条件必填） |
| `scripts/results_schema.py` | 给 perf 实测结果强制加上「运行放置条件」 |
| `scripts/geomean.py` | 跨 shape 的几何平均加速比，作为主指标 |

### ⑤ 看硬件（PMC / 带宽 / ISA / 环境）

| 文件 | 职责 |
|---|---|
| `scripts/pmc.py` | PMC CSV 的可执行解析器（唯一权威实现） |
| `scripts/pmc_probe.py` | 逐个数一遍：这个 PMC 字段在本机到底采不采得到 |
| `scripts/probe_pattern.py` | 访存 run-length 扫描：拿到「这个访问模式自己的带宽上限」 |
| `scripts/probe_stream.py` | 本机流式带宽实测（只读 / copy / 纯写三种口径分开命名） |
| `scripts/bw_calibrate.py` | 标定「访存模式 → 可达带宽」的阶梯，产出 bw-ladder.yaml |
| `scripts/isa_probe.py` | 让 agent 自己能看汇编，而不是靠人肉查一次 |
| `scripts/env_probe.py` | 跑任何 kernel/bench 之前，先确认环境真的能用 |
| `scripts/gpu_guard.py` | 这块卡现在能不能用来做测量 |
| `scripts/lang_select.py` | 这个算子、这个场景，该用什么语言实现 |

### ⑥ 造实现（forge / 骨架 / 设计空间）

| 文件 | 职责 |
|---|---|
| `scripts/kernel_scaffold.py` | 从访问模式模板生成**可编译的** kernel 骨架 |
| `scripts/variant_forge.py` | 从现有 Triton 模板生成**参数化变体** |
| `scripts/forge_step.py` | 「诊断 → 造变体 → 编译 → 测量」里造+编译的那一环 |
| `scripts/design_space.py` | 枚举 tile 配置并预测哪个好 |
| `scripts/fmla_patch.py` | 给 FlashMLA-das 打 d=512 支持补丁 |
| `scripts/minimal_repro.py` | kernel 崩了/算错了，用递减实验定位到根因 |
| `scripts/ablate.py` | 消融归因（纯读：不跑 bench / 不写 runs/ / 不改卡片） |

### ⑦ 评分与治理（深度 / 覆盖 / 质量）

| 文件 | 职责 |
|---|---|
| `scripts/scorecard.py` | 三维（深度/覆盖/质量）可机检评分卡 |
| `scripts/depth.py` | 把「优化有多深」变成一个可以算出来的数（DS） |
| `scripts/self_iteration.py` | 「自我迭代能力」评分器（六步链路是否接通） |
| `scripts/self_evolve.py` | 把这一轮的赢家封存成下一代的基线，再打一轮 |
| `scripts/calibrate.py` | EV 事前校准：expected_effect 到底兑不兑现 |
| `scripts/aggregate.py` | 按家族聚合，让「哪张卡优化得最好」变成一个可排序的问题 |
| `scripts/rank_impls.py` | 跨实现排序的唯一口径 |
| `scripts/skill_eval.py` | 把 `<skill>/evals/evals.json` 里的用例真的跑一遍 |
| `scripts/run_evals.py` | evals 静态检查器：结构 + 全库 id 唯一性 |
| `scripts/evalrate.py` | 「可靠评估/小时」作为一等指标 |
| `scripts/review_apply.py` | 把人的审阅结果落到卡片 progress —— **人只有降级权，没有提权** |

### ⑧ 登记 / 检索 / 索引

| 文件 | 职责 |
|---|---|
| `scripts/card_sync.py` | 一张卡要登记的 5 个位置，一个入口全管 |
| `scripts/gen_gaps.py` | 生成 op-atlas/gaps.md：已知未覆盖 + 证据缺口 |
| `scripts/gen_tool_registry.py` | 由 tool_probe 的**实跑结果**生成工具注册表 |
| `scripts/atlas_query.py` | 图谱的两个交付物：在哪（where）与用哪个（select） |
| `scripts/recall.py` | 跨卡检索 memory/ledger.jsonl：按 (bound,signal)/card 过滤 |
| `scripts/state.py` | 状态看板：扫 progress.stage，输出阶段分布 |
| `scripts/ingest.py` | 按 ingest/schema.yaml 从外部语料抽吸纳记录 |
| `scripts/ingest_check.py` | 吸纳记录的守卫 |
| `scripts/inbox.py` | 把一次 agent 产出追加成一条**待审记录**（append-only） |
| `scripts/opmanifest.py` | 执行身份冻结（T10 / P0-1，原 T2，缺陷 D9） |

### ⑨ 基建 / 工具链

| 文件 | 职责 |
|---|---|
| `scripts/tool_probe.py` | 逐个真跑一遍，把工具可用性变成机器判据 |
| `scripts/opgate_checks.py` | ops 门禁的检查函数（从 hooks/ops_gate.py 拆出来） |
| `scripts/par_check.py` | 并行 agent 的写范围冲突检测 |
| `scripts/json_schema_mini.py` | 零依赖 JSON Schema 子集校验器 |
| `scripts/roofline_golden.py` | 模板的**绝对正确性**校验 |
| `scripts/send_report.py` | 把进展汇报发到邮箱 |
| `scripts/hygon-upstream-sync.py` | HYGON-AI GitHub 开源上游同步工具（dcu skills 用） |
| `scripts/gen_code_map.py` | 生成 docs/CODE_MAP.md 本身（按角色分组；由测试守护，不许腐烂） |
| `scripts/template_render.py` | scripts 侧的模板渲染器（{{key}} 占位；与 engine/instrument 同语义） |

---

## 五、这张地图怎么维护

1. 新增 `.py` → 在 `GROUPS` 里加一行（生成器在 `scripts/gen_code_map.py`）；
2. 跑 `python3 -I scripts/gen_code_map.py` 重生成；
3. `pytest Plugins/ops/tests/test_code_map.py` 必须绿。

测试守两件事：**仓库里每个 .py 都在地图里**、**地图里每个路径都真实存在**。
只守前者会留下死链，只守后者会漏新文件。
