# ENGINE —— 调度引擎的边界规则

> 本文回答一件事：**哪些代码属于引擎，哪些不是，谁可以依赖谁。**
> 判定标准只有一条：**引擎不含算子知识。** 一旦某段代码需要知道
> 「这是 rmsnorm」或「sparse-mla 的 skv 怎么扫」，它就不属于引擎。

---

> **2026-11-25 更新**：补了三处这段时间新增的东西 ——
> §七 场景路由（语言选型的调用链）、§八 冷启动（新算子的入口）、
> §九 本仓已踩过的「写了没接上」清单。原有的边界规则一条没改。

---

## 一、依赖规则（硬约束，违反即架构退化）

```
engine/  ──读──▶  data/       （纯 YAML 知识，无代码）
   │       ──调──▶  skill/bench/  （只通过 CLI 契约，不 import）
   │       ──写──▶  memory/      （只由 engine 写）
   │
data/    ✗ 不许 import 任何代码
skill/   ✗ 不许 import engine（skill 不知道谁在调度它）
memory/  ✗ 不许是代码（数据 + 只读检索脚本）
```

**四条禁令，每条都有血例：**

| 禁令 | 违反后会怎样 | 本仓已发生的实例 |
|---|---|---|
| `data/` 不许含代码 | 知识变成逻辑，改一个数要改代码 | `surfaces.yaml` 的 `class` 与 `bound-to-class.yaml` 的 `first_try` 互相矛盾，谁也不是真源 |
| `skill/` 不许 import engine | 装一个 skill 要连带引擎 | `op-optimize` 现在带一个 1942 行的 `loop.py` |
| `memory/` 只由 engine 写 | 两套记账，数字对不上 | `ledger.md`（人写散文）vs `runs/loop/*.jsonl`（机器写），DS 的 R 项数的是散文 |
| engine 不含算子知识 | 每加一个算子就改引擎 | `loop.py` 硬编码 sparse-mla 的 `--s/--topk/--heads/--skv`，rmsnorm 直接跑不了 |

**当前最危险的一处**：`loop.py` 用 `parents[4]` 硬编码去找 `scripts/measure_mode.py`。
跨了四个目录层级，改一次目录结构就断。这是「两者不在同一依赖层」的症状。

---

## 二、模块表

| 模块 | 职责 | 位置 | 依赖 | 禁止 |
|---|---|---|---|---|
| **loop** | 单卡实验循环：队列→跑→判→记 | `engine/` | data, skill/bench(CLI), memory | 含算子知识 |
| **advance** | 多轮编排（**S3 才做**） | `engine/` | loop, plan, memory | 直接调 bench |
| **plan** | EV 排序 + 前提检查（**S3 才做**） | `engine/` | data, memory | 读 skill 实现 |
| **lang_select** | 语言候选排序 | `engine/` | data, memory | 执行编译 |
| **measure_mode** | 测量模式单一真源 | `engine/` | 无 | 依赖任何 skill |
| **judge** | 唯一显著性判定 | `engine/` | measure_mode | 被复制第二份 |
| **recall** | 记忆检索 | `engine/` | memory | 写 memory |
| **bench_contract** | `ab.py` 契约校验 | `engine/` | 无 | 碰实现 |
| `surfaces.yaml` / `lang-candidates.yaml` / `dominant-to-surface.yaml` | 知识 | `data/ops/` | — | 含可执行逻辑 |
| `bench/ab.py` **实现** | 算子特有的测量适配 | `skill/` | — | import engine |
| `_common.py` | 输入构造 / fp32 参考 | `skill/` | — | import engine |
| `candidates.yaml` | 单卡候选队列 | `skill/` | — | 跨卡 |
| `runs/loop/*.jsonl` | 单卡实验流水（**权威**） | `skill/` | — | 被手改 |
| `memory/ledger.jsonl` | 跨卡记忆索引 | `memory/` | — | 人手写 |

### 为什么 `runs/loop/` 留 skill 而 `memory/` 独立

- `runs/loop/` 是**单卡的流水账**，跟着卡走，卸载 skill 就该一起走
- `memory/` 是**跨卡检索**用的，散在 16 个 skill 里就搜不到——这正是今天
  `trigger_pattern` 检索做不了的直接原因

**两者字段基本相同，但只有 `runs/loop/` 是权威**。`memory/` 是索引，由 `loop.py`
在写流水时顺手 append 一条。**一个动作写两处，一处权威。**

---

## 三、三处过早抽象的判定与替代

### 1. `bench` 实现抽取 —— ❌ 过早

**现状**：7 份 `ab.py`，**接口已统一**（全部 `--shape/--variants/--base/--rounds/--json`），
但**实现分三档**：5 份 96 行、`op-sparse-mla` 162 行（skv 扫描）、
`op-rmsnorm-fused` 198 行（pool 策略）。

**为什么不该抽**：差异全部来自**算子特有**逻辑。现在抽公共库，等于把算子知识塞进公共层。

**替代**：`engine/bench_contract.py` 只校验契约——
跑 `ab.py --help` 确认有那 5 个参数，跑一次确认 JSON 里有
`measure_mode` / `shape` / `runs[].variants[]`。**不碰实现。**
等 ≥3 份出现**完全相同**的逻辑再抽。

**判据**：`grep -c` 三份实现里的相同函数体，≥80% 相同时才抽。

### 2. `memory` 独立服务 —— ⚠️ 部分过早

**不做**：数据库、索引服务、向量检索。

**做**：一个 `memory/ledger.jsonl` + 一个 60 行 `recall.py`（按 `(bound, signal)` 过滤）。

**判据**：当 `ledger.jsonl` 超过 ~5000 行、`recall` 单次超 1 秒时，才考虑索引。

### 3. `lang_select` 接进 EV —— ❌ 类型错误

**问题**：`plan.py` 的 EV 四个因子（`remaining`、`1−利用率`、`已证伪次数`、`前提通过`）
**全部针对攻击面**。语言没有对应的量——「Triton 的 remaining」不存在。

**替代**：**两阶段**
```
1. plan.py → top3 攻击面（现有逻辑不动）
2. 对每条攻击面，lang_select → 可用语言
3. 最终动作 = (surface, language) 二元组
```
唯一交叉：某语言在该卡上已被实测且更慢 → 该组合的 EV 打折，
复用现有 `(1 − 0.5×已证伪次数)`，把 `(surface, language)` 做成 `trigger_pattern` 的复合键。

---

## 四、迁移顺序（先立规矩，后搬文件）

| 步 | 做什么 | 风险 |
|---|---|---|
| **M1** | 建 `docs/ENGINE.md`（本文）+ `engine/` 目录 + 模块表 | 无 |
| **M2** | `loop.py` 加 shim：`skills/op-optimize/loop.py` 只做转发 + 弃用告警 | 低 |
| **M3** | 扩 `test_loop_uniqueness.py`：shim 里不许有 >20 行实现 | 低 |
| **M4** | 迁 5 处 `importlib` 测试路径 | 低 |
| **M5** | 改跨插件 1 处（`dcu-orchestrator/tasks/op-optimize.yaml`） | 低 |
| **M6** | 真正把 1942 行搬到 `engine/loop.py`，shim 转正 | 中 |
| **M7** | 40 处文档路径批量改 | 低但量大 |

**必须先做 M2+M3 再 M6**：先建立「它是引擎」的机器判据，再搬文件。
反过来会在搬的过程中产生大量临时引用。

---

## 五、迁移实操教训（2026-11-24 实测，踩过两次）

### 1. `Path(__file__)` 的常量必须一次性全查，不要按报错逐个修

迁 `loop.py` 到 `engine/` 时，有两类路径常量断了，**我是按报错逐个修**的：

| 断的常量 | 报错现象 |
|---|---|
| `bound-to-class.yaml` | 所有卡报「bound='mixed' 在 bound-to-class.yaml 里没有映射」 |
| `SURFACES_PATH` / `SCENARIOS_PATH` | `enumerate` 报「找不到 references/surfaces.yaml」 |

**正确做法**：搬迁前先
```bash
grep -n 'Path(__file__)' <被迁文件> | grep -v '^.*resolve().parents'
```
把**所有**依赖 `__file__` 的常量列出来，一次改完。

### 2. shim 无法完美模拟模块身份

`skills/op-optimize/loop.py` 作为 shim 时，尝试过三种方案让被 import 的行为与 engine 一致：

| 方案 | 失败原因 |
|---|---|
| `runpy.run_path(..., "__main__")` | 无条件执行 `main()`，pytest 下 argparse 无参数 → `sys.exit(2)` |
| `globals().update(engine.__dict__)` | **只拷名字**，测试改 `loop.VERIFY_TIMEOUT_S` 改的是 shim 的字典，engine 读不到 |
| `exec(engine_src, globals())` | `__file__` 仍是 shim 的 → `REPO_ROOT` 和 `references` 路径全错 |

**结论**：shim 只能保证「当脚本跑」时行为一致。**任何测试如果依赖模块级可变全局，必须直连 engine 路径**，不许经过 shim。
这条已写进 shim 的 docstring，7 个测试已改。

### 3. 测试应当镜像真实布局

`test_ops_gate` 的模拟布局里原本拷的是 `skills/op-optimize/loop.py`。
迁移后那个文件成了 shim，它会去找 `Plugins/ops/engine/loop.py`——
所以模拟布局里**也要有 engine**，否则 shim `sys.exit`。

**规律**：凡是「复制到临时目录再跑」的测试，迁目录后都要检查它复制的那份是否还自足。

---

## 七、场景路由：语言选型怎么被调用（2026-11-25 新增）

### 为什么加这一节

`scripts/lang_select.py` 存在了很久，判据和数据都对：

```
rmsnorm prefill-8k：lightop(HIP) 197.0µs / tilelang 221.0 / triton 242.0
```

**但 `grep lang_select` 全仓只有文档引用，零调用方。**
外部证据（opbench 三臂对照）：裸 CC / 本 skill / SkillHub **三臂都没写过一行 HIP**，
而本仓已有实测更快的现成实现。

**这是「写了没接上」的第一个实例，也是最贵的一个** ——
路线选择这个影响最大的决定，被交给了 agent 的习惯。

### 现在的调用链

```
roofline.json（数字）
   ↓
lang_select.route_scenario（4 个数 → 语言优先级）
   ↓
advance.run_advance 的 build_plan **之前**
   ↓
候选的 action_class 必须与路线一致
```

### 路由规则住在 data/ 不是代码里

`data/ops/lang-candidates.yaml` 的 `routing.rules`，6 条，按顺序第一条命中即停
（顺序即优先级，不靠权重打分）。

**依赖规则**：`scripts/lang_select.py` 读 yaml，规则不硬编码在代码里。
加了测试 `test_rules_are_loaded_from_yaml_not_hardcoded` 防止两边分叉。

### 两条设计决定

1. **不约定「起手语言」**。规则按场景命中，`hip` 和 `triton` 都可能是第一。
2. **实测压过先验**。有 bench 数据时按实测排，路由只在无实测时生效。

---

## 八、冷启动：新算子的入口（2026-11-25 新增）

### 之前的断链

```
roofline.json 需要 roofline 分析
  → 需要 model（models/*.py）
    → 需要有人先懂这个算子
⇒ 新算子卡在循环的第一环
```

实测：`derive.py` 只认 sparse-mla（参数硬编码）；
`grep "建卡" scripts/*.py` = 空；卡片 schema 20+ 字段且 `additionalProperties: false`。

### 现在的链路（六步，第四步不能省）

```
op_bootstrap → 补 reference → 跑一次基线 → floor → lang_select → advance
                                    ↑
                            没有 t_actual 时路由「无规则命中」
```

**这不是缺陷，是物理事实** —— 不知道该多快就不知道该往哪走。

### 三条纪律

1. **bytes 是分母，算错方向就反**。所以 `op_bootstrap` 的反向用例全部
   围绕「什么情况下 bytes 会算错」：维度名缺失、family/pattern 不匹配、
   复合算子硬套单一 pattern。
2. **复合算子必须拒**。含 `@` 的 math 不许用 elementwise pattern ——
   GEMM 由 FLOPs 约束、elementwise 由字节约束，两种分母不能混。
3. **reference 故意留 NotImplementedError**。参考实现是正确性的唯一锚，
   猜一个能让整个门禁「通过」一个算错的实现（两边都错）。

---

## 九、本仓已踩过的「写了没接上」清单

每一条都有机器判据守着（否则会再犯）。

| # | 工具 | 当时的状态 | 现在的守卫 |
|---|---|---|---|
| 1 | `lang_select.py` | 判据正确，零调用方 | `advance.py` 在 plan 前调用；测试断言调用顺序 |
| 2 | `acc_probe` 机制 | `loop.py` 有代码，文件不存在 → 静默跳过 | 文件已建；`decide()` 在 `maxdiff=None` 时改看 acc |
| 3 | HIP kernel | 实测更快，藏在 `runs/`（gitignore） | 挪到 `kernels/` 并登记 `variants.yaml` |
| 4 | `ablate.py` | 能算，但分不清消融与计时探针 | `derived_from` / `numerically_valid` 强制声明 |
| 5 | `surfaces.yaml` 前提检查 | 11/17 条恒失败（`grep traffic` 查的键不存在） | 17/17 通过；占位符无落地判 `unrunnable` 而非假通过 |
| 6 | `ingest_check` | 只查记录内字段，顶层 `absorption` 漏检 | 补数组校验；修后抓出 14 处键名不一致 |

**规律**：工具「存在」和工具「生效」之间隔着一个调用点。
写完之后必须回答：**谁调用它？如果没人调用，它怎么被触发？**

---

## 十、当前模块表补充（2026-11-25）

| 模块 | 职责 | 位置 | 被谁调用 |
|---|---|---|---|
| `op_bootstrap` | 算子描述 → 能跑的卡 | `scripts/` | 人 / agent（冷启动第一步） |
| `lang_select` | 场景 → 语言优先级 | `scripts/` | `advance.py`（plan 之前） |
| `floor` | 地板计算（分母唯一来源） | `scripts/` | `advance` / `lang_select` / 人 |
| `forge_step` | 生成 → 编译 → 体检 → 登记变体 | `scripts/` | `advance --forge` |
| `kernel_scaffold` | 生成 HIP 骨架（库形态） | `scripts/` | `forge_step` / `variant_forge` |
| `ablate` | 消融归因（3×MAD） | `scripts/` | 人（找收益来源时） |
| `ingest*` | 外部语料吸纳 | `scripts/` | 人 |
| `check_schema_fields` | 加字段前先问 schema 认不认 | `scripts/` | 人（改卡前必跑） |

**新增依赖边（需守）**：
- `advance` → `lang_select`（读 roofline，不读 skill 实现）
- `forge_step` → `kernel_scaffold`（按路径加载，不靠 sys.path）
- `op_bootstrap` → `floor` 的算法（复算 bytes，不重新实现地板）
