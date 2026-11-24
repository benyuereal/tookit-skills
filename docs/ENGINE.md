# ENGINE —— 调度引擎的边界规则

> 本文回答一件事：**哪些代码属于引擎，哪些不是，谁可以依赖谁。**
> 判定标准只有一条：**引擎不含算子知识。** 一旦某段代码需要知道
> 「这是 rmsnorm」或「sparse-mla 的 skv 怎么扫」，它就不属于引擎。

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
