# 怎么用这套东西做算子优化

> 写给**用的人**（不是改代码的人）。改代码看 `updates/handoff/HANDOFF-2026-11-25.md`。
> 本文回答三件事：从哪进、怎么驱动、什么时候该找别的东西。

---

## 零、一句话记住这个

> **做算子优化，只需要记住一个名字：`op-optimize`。**
> 其它 `op-*` 都是它内部会调的零件。

如果你的 agent 不知道从哪进，把你的需求这样说就行：

```
用 op-optimize 优化 <算子名>
```

它会自己往下走。其它入口（`lang_select` / `advance` / `op_bootstrap`）
**不需要你记**——它们在流程里会被自动调用，或者由 Claude 按需调。

例外只有两种：
- 你要**自己跑脚本** → §二 用法 C
- 你要**测这套东西** → `docs/OPBENCH-USAGE.md`

---

## 一、从哪进 —— 唯一入口是 `Plugins/ops/skills/op-optimize`

仓库有两个看起来都像「算子优化入口」的东西，**别搞混**：

| | 是什么 | 什么时候用 |
|---|---|---|
| `Plugins/ops/skills/op-optimize` | **执行层**：真的跑实验、判显著、落账 | 你要**真的优化一个算子** |
| `Plugins/dcu-core/skills/dcu-orchestrator/workflows/op-optimize.yaml` | **编排层**：六节点工作流的定义（profile→roofline→constrain→optimize→validate→distill） | 你要**规划一次完整的优化项目** |

**关系**：orchestrator 的 `op-optimize` 节点里，实际执行的是 `Plugins/ops/skills/op-optimize` 那套。
前者管「先做什么后做什么」，后者管「这一步具体怎么跑」。

**别做的事**：不要绕过 orchestrator 直接改 `workflows/*.yaml` 来找新路径；
也不要绕过 `op-optimize` 直接调 `bench/ab.py`——那样六道门全被跳过。

---

## 二、怎么驱动 —— 三种用法，按 token 成本排

### 用法 A · 一句话（最省心，最费 token）

```
优化 op-norm-act-swiglu，目标 beat lightop
```

Claude 会读 `op-optimize` 的 SKILL.md，自己走完六步。**问题是没有退出条件**——
它可能一直跑下去，也可能在一轮之后就停。

### 用法 B · 带退出条件（推荐）

```
优化 op-norm-act-swiglu，最多跑 5 轮，
卡住（连续 2 轮 refuted 或 remaining < 10%）就停下来告诉我卡在哪
```

比 A 好，因为它给了**明确的停手条件**。这几个词是 `advance.py` 真认的：

| 说这个 | 触发什么 |
|---|---|
| 「最多 N 轮」 | `--max-rounds N` |
| 「连续 2 轮没进展就停」 | `MAX_CONSECUTIVE_REFUTED=2`（默认行为） |
| 「到地板就别折腾了」 | `remaining < 0.10` 时 `at_floor` 退出 |
| 「没有更好的变体就自己造」 | `--forge` |

### 用法 C · 我自己跑，Claude 做分析（最省 token）

```bash
# 1. 先看这个算子现在什么状况
python3 -I Plugins/ops/engine/loop.py -C Plugins/ops/skills/op-<卡> status

# 2. 跑多轮（--dry-run 先看计划对不对）
python3 -I scripts/advance.py --card Plugins/ops/skills/op-<卡> --max-rounds 3 --dry-run
python3 -I scripts/advance.py --card Plugins/ops/skills/op-<卡> --max-rounds 3 --forge

# 3. 让 Claude 读结果做归因
#    「读 <卡>/runs/loop/*.jsonl，告诉我这轮收益来自哪个改动」
```

**推荐用法 C**：`advance.py` 的输出已经是结构化 JSONL，
Claude 读它比重新跑一遍便宜得多。

---

## 三、一个算子从零到有结论 —— 六步

```bash
# ① 建卡（新算子才需要；已有卡跳过）
python3 -I scripts/op_bootstrap.py --name <算子> --family <族> \
  --shape "M=8192,H=4096" --inputs "x:bf16[M,H]" --outputs "y:bf16[M,H]" \
  --pattern <访问模式>

# ② 补 reference —— **不能跳过**
#    参考实现是正确性的唯一锚。bootstrap 故意留 NotImplementedError。

# ③ 跑一次基线，拿到实测耗时
#    没有这个数，后面的所有判据都算不出来

# ④ 出 roofline，看语言路由给什么建议
python3 -I scripts/lang_select.py --card <算子> --shape s0

# ⑤ 跑完整循环
python3 -I scripts/advance.py --card Plugins/ops/skills/op-<算子> --max-rounds 3 --forge

# ⑥ 登记（一个入口管 5 个位置）
python3 -I scripts/card_sync.py --card <算子>
```

**第 ③ 步不能省**。实测：没有实测耗时（`t_actual`）时，语言路由输出
「无规则命中 —— 先补齐 roofline 再选语言」。这不是缺陷，是物理事实：
不知道该多快，就不知道该往哪走。

---

## 四、什么时候该找别的东西

我们这套只管「**优化已经能跑的算子**」。下面这些情况要出门左转：

| 你要做的事 | 该去哪 | 为什么不是我们 |
|---|---|---|
| 用 TileLang 写算子（不是 HIP） | `Plugins/dcu-operators/skills/dcu-kernel-dev` §三 | 我们只有 HIP 路径，TileLang 的安装/写法/DCU 坑它更全 |
| CUDA 代码迁到 HIP | `dcu-cuda-hip-port` | 移植是另一件事 |
| 从头设计一个新 kernel 的算法 | `dcu-kernel-dev` §十（结构化 playbook） | 我们是「在已有实现上迭代」，不是「从零设计」 |
| 端到端推理性能优化 | `Plugins/dcu-perf/skills/dcu-inference-optimization` | 我们只管单个算子 |
| 找外部项目能不能借鉴 | `dcu-external-opt-scout` | 开源筛查是它的活 |
| 看 PMC / trace | `dcu-hipprof` / `dcu-llm-benchmark` | 我们只消费它们的数，不采集 |

### 但有一条线不能越

**`dcu-kernel-dev` 不是我们的替代品，也不是我们的上游。** 它讲「该不该用 HIP、
TileLang 怎么写」；我们讲「已有实现之后怎么迭代到极限」。两边都用「算子开发」
这个词，但**判据和流程完全不同**。

实际用法是：**它的方法论可以引用，它的结论不能直接抄进我们的卡**。
因为它的数字没有 `event_isolated` 口径、没有 3×MAD 判定，直接搬进来会污染我们的门禁。

---

## 五、三个最容易踩的坑（用的时候会碰到）

### 坑 1 · 拿不同口径的数字比

```
lightop 197.0 µs（event_isolated）
我们 180.5 µs（back_to_back）
→ 看起来快 8.4%，实际口径不同，不可比
```

**规矩**：任何「快了多少」必须同一次 base 测量、同一个 `measure_mode`。
`advance.py` 已经强制了，但你手工跑 `bench/ab.py` 时要自己注意。

### 坑 2 · 在噪声上做决策

base 的 MAD 超过 5% 时，任何判定都不可信。`judge()` 会自动判 `INCONCLUSIVE`
并**把记录隔离到 `runs/loop/noise/`**，不进权威流水。

**你该做的**：看到 `INCONCLUSIVE` 且原因是「基线噪声」，就重跑，不要调参数试图救。

### 坑 3 · 新卡登记漏位置

一张卡要在 5 处出现（卡片 / registry / INDEX / `ops/SKILL.md` / gaps）。
漏一处测试就红。

**用 `card_sync.py`**，别手工补：

```bash
python3 -I scripts/card_sync.py --card <算子>     # 修
python3 -I scripts/card_sync.py --check           # 只查（CI 用）
```

---

## 六、怎么判断「这套东西是好的」

**唯一验收：盲测一个没见过的算子。**

合格线三条同时成立：
1. 出现 `significant=true` 且 `measure_mode=event_isolated` 的记录
2. `acc_ok=true`（精度有数）
3. 卡片 `ratio < 0.95`（真比基线快）

**只满足 1、2 不满足 3** = 流程通了但没赢。
**满足 1、3 不满足 2** = 快了但精度没验（不许声称）。

---

## 六点五、如果是要交给评测方测

看 [`OPBENCH-USAGE.md`](OPBENCH-USAGE.md)。里面写了三件事：

1. **给完整仓库，不只给一个 skill 目录** —— 只给单个 skill 等于把手脚绑住
2. **明确说「可以用 scripts/」** —— 否则 `lang_select` 是死代码（我们实测过）
3. **接受「第一次跑基线」的开销** —— 没有 `t_actual` 时语言路由 fail-closed

还有一份「三条硬指标 + 三种假成功」的判定表。

## 七、速查表

| 我想… | 命令 |
|---|---|
| 看某张卡现在什么状态 | `loop.py -C <卡> status` |
| 看有哪些方向可选 | `loop.py -C <卡> enumerate` |
| 看下一步该做什么 | `loop.py -C <卡> resume` |
| 跑多轮 | `scripts/advance.py --card <卡> --max-rounds N --forge` |
| 只出计划不跑 | 上面加 `--dry-run` |
| 看语言该怎么选 | `scripts/lang_select.py --card <卡>` |
| 建新卡 | `scripts/op_bootstrap.py --name ...` |
| 登记/同步 5 个位置 | `scripts/card_sync.py --card <卡>` |
| 查技术债 | `docs/DEBT.md`（D1–D11） |
| 查架构边界 | `docs/ENGINE.md` |
| 查历史决策 | `updates/engineering-iteration/roadmap-history/` |
