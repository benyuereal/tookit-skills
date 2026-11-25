# Agent Skill 路线图 v5.1：先让数字可信，再谈能力

**日期**：2026-11-24
**业主定性**：交付物 = 一个 agent skill；算子优化只是反哺手段
**唯一验收**：拿 agent 没见过的新算子，不给提示，看它能否自主走完六步

> **本文件所有数字都在容器里实跑过**，命令见 §十。没有实跑的预测一律不写。
> v5 相对 v4 的立场：v4 解决了「能不能跑」，v5 必须先解决「**跑出来的数字能不能用**」。
>
> **v5.1 是 v5 的修正版**，依据第五轮评审的五条实质缺口 + 四条小问题，全部接受。
> 最重的一条：**v5 §0.2 的"折算回去完全一致"是同义反复**（乘数由端点反解，再用它验证端点）。
> 修正清单见 §十三。**v5.1 是"停止评审、去执行"的版本**；v6 只能是执行中发现问题的修复记录，不许是重新设计。

---

## 第零章 · v4 之后发生了什么（实测）

### 0.1 v4 已完成项复核

| v4 步骤 | 声称 | 复核 |
|---|---|---|
| 0a `_load_yaml` 缓存 | 15.4s→1.0s | ✅ 真。实测 `validate.py` 0.999s；**测试套件 >9 min → 127 s** |
| 0b bound 词表统一 + fail-closed | 53 张卡对齐 | ✅ 真。`BOUND_ALIASES`/`BOUND_ACTIONS` 就位，未知 bound 返回 `(None, "未知 bound=...")` |
| 1 R0 端到端跑一轮 | 1 条新 ledger | 🟡 **链路通了，但数字方向是反的**（见 0.2） |
| 2 写 `<skill>/roofline.json` | allowed_actions 拦得住 | 🟡 文件已产出，但 `remaining=-0.0404` 照样放行 |
| 3 抽 `judge()` + record 4 字段 | 不新建 ledger.py | ✅ 真。commit `2a9369e`，四态判定 + 4 字段，`_significant()` 已委托 |
| — | — | ✅ **21 个 commit 已全部 push**（`origin/main..HEAD` = 0） |

**当前分数**：深度 **2.9** / 覆盖 **6.5** / 质量 **3.8**

```
深度  D1(13.5) D2(1) D3(0.222) D4(1) D5(8)
覆盖  C1(1.0) C2(0.64) C3(0.741) C4(4) C5(2)
质量  Q1(None) Q2(17) Q3(4) Q4(90) Q5(241) ← Q5 从 196 涨到 241，已超 200
测试  2 failed / 605 passed / 127 s
```

### 0.2 ⚠️ R0 的第一个结论，方向是反的

`096d1dc` 落盘的第一条记录：

```
tag=fused-3pass-triton   base(lightop-fused)=0.19468 ms   cand=0.16081 ms
Δ = -17.4%               significant=True   decision=ACCEPTED_NO_NUMERICS
```

同一形状（M=8192 H=4096 bf16）、同一对实现，`ledger.md` 在 2026-11-23 记的是：

```
M=8192 H=4096 bf16，64 buffer 轮换，40 次中位，event 隔离：
  fused-3pass-triton          242.2 µs   1108 GB/s
  lightop.fused_add_rms_norm  227.0 µs   1182 GB/s
  ⇒ 等语义下**我们慢 6.7%**
```

**两个口径，符号相反。** 但注意：**不能靠"折算乘数"来证明哪个对**，原因见 0.3。

| 实现 | R0（背靠背） | ledger（event 隔离） | 比值 |
|---|---|---|---|
| fused-3pass-triton | 160.81 µs | 242.2 µs | — |
| lightop-fused | 194.68 µs | 227.0 µs | — |
| **triton / lightop** | **0.826**（triton 快） | **1.067**（triton 慢） | **摆动 29%，跨过 1.0** |

### 0.3 ⚠️ 独立成立的只有两条（v5 §0.2 的折算推导是循环论证）

**v5 曾写**：`k_triton = 242.2/160.81 = 1.506`、`k_lightop = 227.0/194.68 = 1.166`，
"折算后 `242.2/227.0 = 1.067`，与 ledger.md 完全一致"。

**这是同义反复，信息量为零。** 乘数就是从这两个端点反解出来的（`k₁ = a/c`、`k₂ = b/d`），
再用它去验证 `k₁·c = a` 必然成立。两个未知、两个方程，解不唯一。

**真正独立成立的是两条：**

**① 符号相反（稳健）**

```
背靠背      160.81 / 194.68 = 0.826   → triton 快
event 隔离  242.2  / 227.0  = 1.067   → triton 慢
比值本身跨过 1.0，摆动 29%
```

**② 不存在全局修正因子（这条才是真正的发现）**

```
若存在均匀乐观 k，则 c·k = a 且 d·k = b  ⇒  c/d 必须 = a/b = 1.067
实测 c/d = 0.826                            ⇒  不存在这样的 k
等价地：k₁ / k₂ = 1.506 / 1.166 = 1.292（相差 29%）
```

**这顺手否掉了 `ab.py` 自己的"偏乐观约 23%"**：均匀 1.23 会预测背靠背应是
`242.2/1.23 = 196.9 µs` 和 `227.0/1.23 = 184.6 µs`，实测 **160.8 / 194.7** —— 两个方向都错。

机制自洽：**triton 更短（160 vs 195 µs）→ 固定开销占比更高 → 背靠背时被隐藏得更多 → 更乐观。**

> **⇒ `claimable=False` 的理由比原来更强**：
> 不是"back_to_back 偏乐观所以要打标"，而是
> **"back_to_back 的偏差是逐实现、逐形状变化的，不存在任何可移植的修正因子"**。

### 0.4 ⚠️ MAD 小 ≠ 准确：`judge()` 只能判噪声，判不了偏差

核 R0 的 record 字段：

```
base_mad = 0.143 µs   cand_mad = 0.118 µs   Δ = 33.87 µs
Δ / (3×MAD) = 79×          ← 背靠背：极度"显著"

ledger（event 隔离）: Δ = 15.2 µs, 3×MAD_lightop = 15.0 µs
Δ / (3×MAD) = 1.01         ← event 隔离：刚过线
```

**荒谬点：符号错的口径，"显著性"高 79 倍；符号对的口径只勉强过线。**

原因：`judge()` 检测的是**离散度（MAD）**，检测不了**偏差（bias）**。
back_to_back 的 MAD 极小，因为它测的是理想流水；而它的偏差是系统性的、被 MAD 完全掩盖。

> **`significant=True` + `measure_mode=back_to_back` 是最危险的组合——它看起来极其可靠，实际符号是反的。**
> 推论：`measure_mode` 必须是**独立的前置门**，不能塞进显著性检验里。
> **`judge()` 的能力边界 = 只能判噪声，不能判偏差。** 这条要写进它的 docstring。

**根因不是疏忽，是结构缺口。** `bench/ab.py` 自己写得很清楚：

```python
# caveat 口径=背靠背无 per-iter sync，偏乐观约 23%；卡片口径用端到端
"measure_mode": "back-to-back, no per-iter sync",
"measure_note": "偏乐观约 23%（kernel 间可重叠）…"
```

而 `loop.py` 写记录时，**这个 caveat 一个字都没进 record**（`grep -c "measure_mode\|measure_note" loop.py` = **0**）。

> 这是 fp8 那次（`49.8 TF ⇒ 11 倍空间`）的同族问题，只是换了变量：
> 那次错在**分母**，这次错在**测量模式**。
> 共同点：**记录里缺了那个决定结论符号的字段。**

### 0.5 五个可信度缺口（v4 一个都没覆盖）

| # | 缺口 | 实测证据 | 后果 |
|---|---|---|---|
| **K1** | 测量模式不进记录 | `grep measure_mode loop.py` = 0 | 结论可正可负，不可比 |
| **K2** | 基线角色不校验 | `ab.py --base` 默认 `ref-heavy`；`variants.yaml` 写 `role: correctness_only` + "**不得用于性能对比**"；`grep -c "role" loop.py` = **0** | 换张卡就拿"未优化参考"当基线 |
| **K3** | `remaining < 0` 进决策 | R0 的 `roofline.json`: `t_floor=202.3, remaining=-0.0404`；`floor.py` 第 111/154 行明写「负 remaining = 判据本身错了」；`loop.py` 只查 `bound` 存在 | 判据失效却放行 |
| **K4** | floor 与 t_actual 跨口径相减 | K3 的根因：`t_floor` 端到端口径，`t_actual` 背靠背口径 | 两个不同口径的量做减法 |
| **K5** | `dead_end` 不参与过滤 | `surfaces.yaml` 有 `verdict: dead_end`；`loop.py` 只在 `scenarios.yaml` 读 `dead_ends`，**不读 surface 的 verdict** | 已证伪的死路被当合法候选推荐 |

另有两条**永久失败**的 `precondition_check`（违反"前置必须能跑"）：

```yaml
slc-policy:          ... pmc_probe.py <csv> | grep -c slc
prefetch-depth-isa:  ... pmc_probe.py <csv> | grep vmcnt
```

`counters.md` 的 PMC 列里**既没有 `slc` 也没有 `vmcnt`**（0 命中）→ 永远 exit 1，前置永远 False。

> 这两条 ISA surface 现在被挡住，**是因为判据写错，不是因为判断正确**。

---

## 第一章 · v5 的核心判断

v4 说「传动轴在车上，缺的是油和司机」。**这个判断对，但不完整。**

> **传动轴在，油也在加；但仪表盘是坏的。**
> 现在每转一圈都产出一个数字，而这个数字的**符号取决于用哪把尺子**。
> **先修仪表盘，再踩油门。**

三句话：

1. **所有结论必须带测量模式**，且模式与结论强度绑定（K1/K4）
2. **所有判据必须能跑通**，跑不通的标 `unrunnable`，不许让它"默认永久失败"（0.5 尾）
3. **ISA 层不是"没有"，是被文档写死了** —— 实测反汇编是通的，汇编里有 6 个活杠杆（§四）
4. **`judge()` 只判噪声、判不了偏差** —— 所以 `measure_mode` 是独立前置门，不能折进显著性（0.4）

---

## 第二章 · 目标架构（我们先要知道终点长什么样）

一句话：**不是"一条严谨的流水线"，而是「便宜可靠的评测 × 广覆盖的候选 × 严密的账本」。**

现在的设计是**严谨有余、吞吐不足**——一轮只试一个候选。而公开工作（AVO 连续跑 7 天、KernelEvolve 维护种群、Kevin 16 轨迹 × 8 轮）真正出成绩靠的是大量评测。

```
┌──────────────────────────────────────────────────────────────┐
│ L6 反作弊（横切）  形状分离 · 隐藏压力分布 · 禁止行为 AST 扫描   │
│                    审计追踪：每个数 → 可复跑文件 + sha256       │
├──────────────────────────────────────────────────────────────┤
│ L0 契约    problem.yaml：语义 / 形状 / dtype / 容差 / 压力分布  │
│            harness 分离：agent 只能改 kernel 文件               │
├──────────────────────────────────────────────────────────────┤
│ L5 记忆    recall(bound, signal) → 上次改了什么、涨了多少        │
│            calibrate: predicted vs realized（自我认知）         │
│            死路库（一等公民、可检索，不是静态 md）               │
├──────────────────────────────────────────────────────────────┤
│ L1 度量    三层分母（nominal/achievable/impl）各带 provenance   │
│            双地板取紧 + ABBA + 3×MAD + 噪声门 + **测量模式**    │
├──────────────────────────────────────────────────────────────┤
│ L2 诊断    判据 B(launch) → 判据 A(roofline) → 归因(唯一主导)   │
│            PMC 三态表 + ISA 证据 + **每次改动后重新诊断**        │
├──────────────────────────────────────────────────────────────┤
│ L3 决策    EV = headroom × recoverable_share × 折扣 × 前置通过  │
│            portfolio：k 候选 / 逐轮淘汰 / 预算与升级规则         │
│            停：remaining<10% 或 MBU/MFU≥0.8 或同向证伪 ≥2       │
├──────────────────────────────────────────────────────────────┤
│ L4 执行    ★ advance.py（唯一入口，只编排不重实现）            │
│            调 loop.py run（继承 6 道门）→ judge → record        │
│            四级验证：语法 → 结构 → 正确性 → 性能                │
│            ★ 最坏情况返回原实现，绝不变差                       │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 诊断信号阶梯（因果链，不是信号清单）

```
D0 慢多少 / 慢在哪段        墙钟分档 · 消融                     ✅
        ↓
D1 哪条腿绑                 AI vs ridge · 双地板                🟡 模型少
        ↓
D2 硬件在等什么（症状）       PMC: stall / 带宽 / 发射率          ✅ 3 个计数器坏
        ↓
D3 资源为什么不够             VGPR / LDS / spill                 ✅
        ↓
D4 为什么 stall（病因）       指令流: waitcnt 位置 · 向量宽度 ·   ✅ **实测可用**
                            prefetch 距离 · 展开状态             ← 但被文档误判
        ↓
D5 源头怎么生成这样的        --save-temps · triton amdgcn        ✅
        ↓
D6 别人怎么做的（参考解）     同算子厂商 kernel 的 ISA 对照        ✅ 102 kernel 可对照
```

**现在断在 D2 → D4**：PMC 告诉你"有 stall"，只有 ISA 告诉你"为什么 stall"。

**优先级原则**：**静态先于动态**（ISA 编译一次就有，PMC 要专门采集且开销大）。实操上应该**先看汇编，再用 PMC 验证**，而不是反过来。

### 2.2 六个设计原则

1. **先度量，再优化** —— 没有可信的分母，优化就是猜
2. **正确性门先于性能门** —— 不会有人去 benchmark 一个算错的 kernel
3. **唯一判定实现** —— doc 和 code 指向同一个 `judge()`（已完成）
4. **绝不退化** —— 任何一轮失败，保留原实现
5. **每次改动后重新诊断** —— 不能拿初始诊断走到底
6. **账本比实验重要** —— 能发现"作弊/漂移"的是审计追踪，不是实验本身

---

## 第三章 · P0 可信度层（先做这个，1 天）

### P0-1 · `measure_mode` 进 record，且决定结论强度

```python
# scripts/measure_mode.py —— 单一真源
MODES = {
    "event_isolated": {"claimable": True,  "note": "event 隔离 + 预分配 + 40 次中位"},
    "end_to_end":     {"claimable": True,  "note": "每次 empty_like，含分配"},
    "back_to_back":   {"claimable": False, "note": "无 per-iter sync，kernel 间可重叠"},
}
```

三条硬规则：

1. **`record.measure_mode` 必填**；`ab.py` 的 `measure_mode` 原样搬运，不许丢
2. **`back_to_back` 永远 `claimable=False`**：可用于筛方向，**不得进入 plan 的"已验证"**
3. **base 与 cand 必须同模式**；`judge()` 前置校验，不一致直接 `INCONCLUSIVE`

依据（**v5.1 改写**，原版依据是循环论证）：跨模式比较不仅会翻转符号，而且**乐观度逐实现变化**
（`k₁/k₂ = 1.292`，§0.3），所以不存在任何可移植的修正因子——只有"同模式 + 不可 claim"一条路。
`ab.py` 自己写的"偏乐观约 23%"已被其数据否掉（§0.3 ②）。

**附加要求**：`judge()` 的 docstring 必须写明能力边界——

```
judge() 只能判噪声（MAD），不能判偏差（bias）。
反例：R0 背靠背 Δ/(3×MAD) = 79×，看似极度显著，实际符号是反的；
      而 event 隔离 Δ/(3×MAD) = 1.01，只勉强过线。
⇒ measure_mode 必须是独立前置门，不得折进显著性检验。
```

### P0-2 · 基线角色校验

```python
# loop.py run 前
role = variants.get(base_tag, {}).get("role")
if role == "correctness_only":
    return 2, "拒绝：%s 的 role=correctness_only，不得作为性能对比对象" % base_tag
```

依据：`variants.yaml` 已写 `role: correctness_only` + "**不得用于性能对比**：它是未优化的参考实现，打赢它没有工程意义"，而 `loop.py` 一个字不读。R0 是靠 `candidates.yaml` 手写 `--base lightop-fused` 才躲过。

### P0-3 · `remaining` 符号门 + 同口径校验

```python
# bottleneck_problems() 里补
r = roof.get("remaining")
if r is not None and r < 0:
    probs.append("remaining=%.4f < 0 —— floor.py 已定义这是「判据本身错了」"
                 "（上限估低或流量算多），不是实现突破物理。先回去查分母。" % r)
if roof.get("t_actual_mode") != roof.get("t_floor_mode"):
    probs.append("t_actual 与 t_floor 口径不同（%s vs %s），相减无意义"
                 % (roof.get("t_actual_mode"), roof.get("t_floor_mode")))
```

依据：`floor.py` 第 12/111/154 行三处都写了"负 remaining 是判据错了"，而 `roofline.json` 的 `remaining=-0.0404` 照样放行进了 EV。

### P0-4 · `dead_end` 过滤

`surfaces.yaml` 里 `verdict: dead_end` 的两条，**不许进候选推荐**，只进 recall。

`bound-to-class.yaml` 的注释已经写明意图：

> `isa-policy`：两条都是已证伪的死路……留在映射里是为了「查得到、别重试」，**不是推荐去试**。

但代码不读 `verdict`。补：

```python
if surface.get("verdict") == "dead_end":
    continue   # 不进 top3，只在 recall/dead_ends 展示
```

### P0-5 · 修掉永久失败的 `precondition_check`

| surface | 现在 | 应改为 |
|---|---|---|
| `slc-policy` | `pmc_probe.py <csv> \| grep -c slc` | `dccobjdump ... && grep -c " slc" X.ISA` |
| `prefetch-depth-isa` | `pmc_probe.py <csv> \| grep vmcnt` | `grep -oE "s_waitcnt (v\|lgkm)cnt\([0-9]+\)" X.ISA \| sort \| uniq -c` |
| `occupancy` / `grid-tail` / `split-k` | 带 `<results.json>` 占位符 | 改真实路径或标 `unrunnable` |

**纪律**：跑不通的标 `unrunnable` 并说明缺什么输入。**"永久 False" 与"默认通过"是同一类错误，只是方向相反。**

### P0-6 · 实验轮次与重跑次数分离

R0 的 5 条记录是**同一 tag 重跑 5 次**（10:24/10:36/10:37/10:44/10:45），不是 5 轮不同实验。

```
round = (card, shape, direction, tag) 首次出现
rerun = 同 key 重复记录
D4 只数 round，且每轮 verdict ∈ {adopted, refuted, stalled}
```

现在 `D4` 用 `ledger.md` 标题数算，暂未污染；但 `advance.py` 一旦按 jsonl 行数算"轮次"，立刻会被刷。

### P0-7 · 修两个现存测试失败

```
test_adapter.py::test_reference_runs_and_is_finite
  AttributeError: 'tuple' object has no attribute 'isfinite'
  → reference() 改返回两份输出，测试没跟着改

test_allowed_actions.py::test_undeclared_action_class_warns_but_does_not_block
  期望 rc != 2，实际 rc == 2
  → fail-closed 生效的副作用（这个失败其实是好消息，证明 fail-closed 真的在工作）
```

第 2 条**不留成开放问题**（v5 原写法是"要决定语义"，这是拖延）：

> **按本项目自己的 fail-closed 哲学，答案就是「拦」。**
> 配一次性迁移脚本补全现存 `candidates.yaml` 的声明。
> **写了三天"要决定"的问题，第四天就会变成默认不拦。**

```bash
# 一次性迁移：给所有没声明 action_class 的 pending 候选补上
python3 -I scripts/migrate_action_class.py --dry-run   # 先看会改哪些
python3 -I scripts/migrate_action_class.py --write     # 补全后 2 个测试应转绿
```

`round key` 修正（对应缺口：会吞掉合法重试）：

```
round_key = (card, shape, direction, tag, impl_hash)
round  = 该 key 首次出现
rerun  = 同 key 且同 impl_hash
```

**必须加 `impl_hash` 的实证**：`ledger.md` 里 triton 有 `172.5 µs（无 residual_out）`
和 `242.2 µs（补 residual_out 后）` 两个版本——**同 tag、同 shape、同称"event 隔离"，相差 40%**。
不加 `impl_hash`，下一次重测会被误判成 rerun 而不计入轮次。

---

## 第四章 · P1 ISA 层复活（1 天）

### 4.1 先修文档 —— 最划算的一笔（0.5 h）

实测出的三处错：

| 文档 | 写的是 | 实际 |
|---|---|---|
| `isa-census.md` §一②③ | `/opt/dtk/lib/hipblaslt/library_gfx938/` | **不存在**，实际 `library_bw_nmz/` |
| `isa-census.md` §一③ | "厂商已带好反汇编（461 MB）`*.ISA`" | **文件不存在**（全机 `find *.ISA` 零结果） |
| `isa-census.md` §一③ | `--inputs=<file.hsaco>` | `.hsaco` 报 `not recognized`，必须 **`.co`** |
| `tool-status.md` | `dccobjdump: degraded`（理由 `--architecture` 不含 gfx938） | **用错参数**，`--target=gfx938` 完全可用 |

**实测可用的四条命令**：

```bash
# ① 厂商 .co → 完整汇编（实测 188487 B，102 个 .protected 符号，含 LayerNorm_*）
cd /tmp && dccobjdump --inputs=extop_gfx938.co --target=gfx938 --recompile

# ② 资源字段
llvm-readelf --notes library_bw_nmz/Kernels.so-000-gfx938.hsaco
# → .vgpr_count / .group_segment_fixed_size / .private_segment_fixed_size

# ③ 自编译 kernel 的汇编（不需要反汇编工具）
hipcc --save-temps --offload-arch=gfx938 --genco k.hip -o k.hsaco
# → k-hip-amdgcn-amd-amdhsa-gfx938.s

# ④ 我们自己的 .so：先抽内嵌 ELF（.hip_fatbin 段里能提出 7 个）
#    dccobjdump 会 segfault，但 .ISA 仍完整落盘（实测 64 KB / 1021 条指令）
```

**为什么最划算**：`tool-status.md` 是自动生成、被规定为"判据里只能写 available 的工具"。agent 读完判定**反汇编不可用** → 永远只走 PMC。**再好的归因类也进不了门。**

### 4.2 实测：汇编里有 6 个活杠杆，13 条 surface 一条都表达不了

对 `hbird-gemm/build/ship.so` 抽出的 `gemm_kernel_8192`（fp8 8192³，1021 条指令）：

| 指令形态 | 计数 | 含义 | 现有 surface 能表达吗 |
|---|---|---|---|
| `v_mmac_f32_16x16x32_fp8_fp8` | 128 | 主力 MMA | ❌ |
| **`ds_read_matrix_trans_format`** | **32** | **MLS 矩阵读在用** | ❌ **无此 surface** |
| `ds_read_b128` | 16 | 128-bit LDS 读 | ❌ |
| **`buffer_load_dwordx4`** | **12** | 读：128-bit 用满 | ❌ |
| **`buffer_store_dwordx2`** | **32** | **写：只有 64-bit** | ❌ **可优化点** |
| `s_waitcnt vmcnt(0/1/2/3)` | 11/8/8/8 | 全局等待深度 | 🟡 有 surface，但是死路 |
| `s_waitcnt lgkmcnt(0/1/2/3)` | 29/10/10/14 | LDS 等待深度 | 🟡 同上 |
| `slc` | **0** | 果然没用（死路结论正确） | ✅ `slc-policy` |
| `global_load` | **0** | 全走 `buffer_load`（正确做法） | ❌ |
| `s_barrier` / `s_cbranch` / `s_endpgm` | 3 / 2 / 1 | 无热循环回跳 ⇒ 已展开 | ❌ |

**两个立刻能用的发现：**

**① 写回向量宽度只有读的一半。** 读 `dwordx4`(16B)×12 vs 写 `dwordx2`(8B)×32。13 条 surface 里没有一条管访存向量宽度。

**② 助记符记错了。** `isa-census.md` 写 `ds_read_m32x32_b8`（声称 22,786 次），实际是 **`ds_read_matrix_trans_format`**：

```
hbird fp8 kernel : matrix_trans_format 32 条，m32x32 0 条
厂商 extop_gfx938.co（102 kernel）: ds_read_b32 142 条，m32x32 0 条
```

**agent 若按 census grep `m32x32`，会得 0，然后判定"没走 MLS"。这是把判据本身记错了。**

### 4.3 `isa-policy` 这个 class 现在是空的

```
surfaces.yaml 13 条：residency 3 · pipeline 3 · layout 1 · tier 1
                     · memory-path 2 · precision 1 · **isa-policy 2**
isa-policy 的 2 条：slc-policy（dead_end）、prefetch-depth-isa（dead_end）
```

**一个 class 里全是死路 = 这个 class 在"可攻击面"意义上是空的。** 而 `bound-to-class.yaml` 在 memory/compute/mixed 三档都把它列为可试。

matrix load 连 surface 都不是：`grep -rn "matrix_load\|MLS\|m32x32" Plugins/ops/skills/op-optimize/` = **0 命中**。

### 4.4 补 4 条活的 ISA surface

> **⚠️ v5.1 必读：静态计数 ≠ 动态权重。**
> ISA 计数回答"**代码长什么样**"，不回答"**跑起来占多少时间**"。
> GEMM 主循环的 `load` 每个 K-step 都发生，`epilogue` 的 `store` 每 tile 才一次——
> 静态比例 `12:32` 完全可能对应动态占比 `<5%`。
>
> **正确顺序：ISA 生成假设 → PMC 定级（动态占比）→ A/B 裁决。**
> 本节四条 surface 的 `evidence` 一律只是**上界**；`expected_gain` 在该假设被 PMC 定级前
> **仅作排序用，不得当量级引用**。每条的 `evidence` 都必须带这一句。

```yaml
- id: vmem-vector-width
  statement: 读/写的向量宽度是否用满（buffer_load/store 的 dwordxN）
  class: isa-policy
  precondition: 有 .co 或内嵌 fatbin 可反汇编
  check: |
    dccobjdump --inputs=<X>.co --target=gfx938 --recompile >/dev/null 2>&1
    grep -oE "buffer_(load|store)_dwordx[0-9]" <X>.co.ISA | sort | uniq -c
  evidence: >
    hbird fp8 8192³：读全 dwordx4（12 条）但写只有 dwordx2（32 条）。
    **静态证据，动态占比未知**——store 只发生在 epilogue，可能远低于 5%。
    A/B 前 expected_gain 仅作排序上界，不得当量级。
  expected_gain: 上界 ≤10%（需 PMC 定级 + A/B 实测）
  precondition_check: grep -c "buffer_store_dwordx2" <X>.ISA
  verdict: open

- id: mls-path
  statement: LDS 读走矩阵通路（ds_read_matrix_trans_format）还是标量 ds_read_b32
  class: isa-policy
  precondition: 目的是 GEMM 消费者 ⇒ kPack 必须为 1
  check: grep -oE "ds_read_(matrix_trans_format|b[0-9]+)" <X>.ISA | sort | uniq -c
  evidence: >
    fp8 可用（432 TF / 8192³）；bf16 因 gather 向量化要求 kpack=2，
    编译期被拒（MatrixLoad dst Gemm consumer must have kPack=1, got 2）。
    见 op-roofline/references/rules.md R7。
  verdict: open（bf16 走不了，fp8 能走）

- id: lds-read-form
  statement: ds_read_b128 与 ds_read_b32 的比例（LDS 读是否已向量化）
  class: isa-policy
  check: grep -oE "ds_read_b[0-9]+" <X>.ISA | sort | uniq -c
  evidence: >
    hbird fp8：16 × ds_read_b128 + 32 × matrix_trans，**无 ds_read_b32**
    ⇒ 已全向量化（正例基线，可当对照）。
  verdict: open

- id: epilogue-expansion
  statement: 收尾段是否完全展开（无 s_cbranch 回跳）
  class: pipeline
  check: grep -c "s_cbranch" <X>.ISA   # 热循环内应为 0
  evidence: >
    hbird fp8：3 barriers / 2 cbranch / 1021 指令 ⇒ 已展开（正例）。
    反例见 isa-census：epilogue 用 `#pragma unroll 1`
    → 128 个累加寄存器溢出 → 388 条 buffer_store。
  verdict: open
```

### 4.5 `attribution` 允许 ISA 证据

现在 `attribution.md` 规定 `evidence` "必须是 PMC 数值"。改成允许二选一：

```yaml
evidence:
  主导: {kind: pmc, ...}
  # 或
  isa:
    - {kind: isa, what: "store 向量宽度",
       value: "buffer_store_dwordx2 × 32 / dwordx4 × 0",
       reproduce: "dccobjdump --inputs=cand.co --target=gfx938 --recompile && grep ..."}
```

否则 §4.4 的四条 surface 永远无法产出"主导瓶颈"结论。

---

## 第五章 · P2 诊断 → 决策的断链（0.5 天）

`attribution.md` 的五个 dominant：`hbm` / `l2-miss` / `lds-conflict` / `vgpr-spill` / `issue-stall`

我在 `Plugins/ops/skills/op-optimize/` 和 `scripts/` 全目录 grep 这五个词：**零命中**。

```
PMC → attribution.dominant = "l2-miss"      ← 诊断链终点
              ↓（无映射）
surfaces.yaml 13 条 id                     ← 决策链起点
```

**agent 得出"主导瓶颈是 L2 miss"之后，没有任何机检路径把它翻译成"该试哪条 surface"。** 它只能靠 LLM 自由发挥——而 `loop.py` 的方向预算、refuted 拒跑、动作类约束**全都挂在 surface id 上**，这一步一自由发挥，后面所有约束都绕过。

补一张表（诊断结论必须落进 13 条之一；落不进去说明 surface 不够，是设计缺口）：

```yaml
# references/dominant-to-surface.yaml
hbm:            [layout-pack, fusion-boundary, gather-path]
l2-miss:        [layout-pack, gather-path]
lds-conflict:   [layout-pack, lds-read-form]
vgpr-spill:     [occupancy, epilogue-expansion]
issue-stall:    [cross-lane-fold, barrier-width, prefetch-depth, vmem-vector-width]
```

配套：`unknown` 拆三态（v4 已提出，保留）

```
csv_missing        → 去采 CSV
counter_incomplete → 换采集参数或换 kernel 段
counter_suspect    → 已知坏计数器（SQ_LDS_BANK_CONFLICT 恒 0 / SQ_WAVES 缺列）
                     ⇒ 放弃该计数器，换证据路径（ISA / 消融）
```

同卡连续 2 轮 `unknown` → `blocked`，要求**换采集方式**，不许再采一次。

---

## 第六章 · P3 发动机（0.5 天）

### 6.1 澄清一个表述（业主的直觉是对的）

v4 写"不建 runner.py"，读起来像"取消引擎"。准确说法：

> **要有唯一的发动机，但它只**编排**，不重实现任何判定；落在 `loop.py` 的延伸上。**

反面证据（为什么不能造平行模块）：`loop.py run` 已有六道门——

```
candidates 登记 → bottleneck 归因 → done_when 外部裁决
→ 方向预算 → refuted 拒跑 → allowed_actions 天花板约束
```

新模块自己跑实验，这六道全绕过。**这不是重构，是安全性倒退。**

### 6.2 `advance.py` 的唯一职责

```
scripts/advance.py --skill <dir> [--max-rounds N] [--gpu-budget-s S] [--k 1]

  每轮：
    1. plan     刷新 top3（EV 排序；无数据时按 surfaces 优先序，标 low_confidence）
    2. gate     跑 top1.precondition_check；不过则顺延；全不过 → blocked 出报告
    3. run      调 loop.py run <tag>   ← 继承全部六道门
    4. judge    调统一 judge()（已存在）
    5. record   已有 append_record()（保留正确性字段 + 4 记忆字段）
    6. replan   ★ 见下（v5.1 改写：原版规则有 EV 塌缩盲区）
    7. stop     budget / at_floor / 连续 refuted → 停并出报告
    8. calibrate ★ 累计 ≥5 轮后输出 predicted vs realized（见 6.3）
```

**唯一验收**：`advance.py` 跑完一轮，中间没有人敲任何命令，而 ledger 多了一条记录、`progress.stage` 前进了。

**`--k`**：现在一轮只试一个候选（吞吐 1/轮）。先留接口，等流程稳了再开。

### 6.3 replan 规则（v5.1 修正：原版有 EV 塌缩盲区）

v5 原写法「top3 变了 → 继续；top1 未变 → stalled」有两个洞：

```
洞 1：EV 塌缩。top1 EV 从 0.9 掉到 0.3 但仍排第一 → 判"继续"，实际已经没得挖了
洞 2：分不清健康轮换（top1 被证伪）和噪声抖动（tie-break 翻了个身）
```

**修正（复用 v1 的老规则）**：

```python
def replan(prev_top3, new_top3, ledger):
    if not new_top3:                               return STOP("无可挖方向")
    if new_top3[0].ev < 0.30:                      return STOP("EV 塌缩到 0.3 以下")
    if new_top3[0].ev < 0.5 * prev_top3[0].ev:     return STOP("EV 掉了一半以上")
    if prev_top3[0].id == new_top3[0].id:          return CONTINUE_SAME
    if new_top3[0].ev < 1.5 * new_top3[1].ev:      return STALLED("新 top1 没拉开差距，可能是抖动")
    cited = ledger_has_citation(new_top3.rejected, prev_top3[0])  # 必须引用上一轮 verdict
    return CONTINUE_SWITCH(cited)
```

三条要点：
1. **新 top1 必须领先第二名 ≥1.5×** 才算真轮换（否则是 tie-break 抖动）
2. **EV 塌缩到 0.3 以下 / 掉一半以上 → 直接停**，不许"继续"
3. 轮换必须**能溯源**：`rejected` 里要引用上一轮 top1 的 verdict 与数字

### 6.4 calibrate：排进 S3 判据，不许悬空

v5 把 `calibrate` 写在 L5 架构图里，但 S0–S6 **没有位置**（第五轮评审缺口 5）。

它是**最早发现 EV 系统性高估的机制**——每条 surface 预测 15%、实际兑现 3%，则 EV 排序整体失真。实现只是 ledger 上的一次 join：

```bash
python3 -I scripts/calibrate.py --skill <dir>
# surface                n   predicted_med   realized_med   ratio
# vmem-vector-width      1   10%             未兑现         —
# fusion-boundary        3   15%             4.2%           0.28  ← EV 该打折
```

**S3 完成判据加一条**：`advance.py` 累计 ≥5 轮后，必须能输出上表。
没有它，EV 的衰减因子（`1 − 0.5×已证伪次数`）永远是**事后补救**，不是**事前校准**。

---

## 第七章 · P4 roofline 统一求值器（1.5 天）

### 7.1 粒度就错了：`bound` 是 shape 的函数

`<skill>/roofline.json` 里放**一个** `bound`，对很多算子必然错。`gemm-bf16` 卡片自己写着：

> 拐点由两条地板交叉决定，**随 N 移动**：K=N=4096 时 M≈256，N=14336 时 M≈512。**不能写固定阈值**

`rmsnorm-fused` 有四个形状档（`tiny`/`decode-m1`/`decode-m32`/`prefill-8k`），`decode-m1` 和 `prefill-8k` 的 launch 占比、可挖空间完全不同。

**所以 `load_allowed_actions(card)` 的签名是错的**，必须 `load_allowed_actions(card, shape_id)`。

### 7.2 统一求值器：模板驱动，不是每算子一个模型

统一的四个输入**已经全在仓库里**：

| 输入 | 位置 | 状态 |
|---|---|---|
| 数学定义 | `<skill>/contract.yaml: math` | ✅ |
| 符号形状 + dtype | `contract.yaml: shapes / dtypes` | ✅ |
| **具体形状档位** | `<skill>/shapes.yaml` | ✅ **7 个 skill 已有** |
| 硬件常数（三层口径） | `op-roofline/constants.yaml` | ✅ |
| 求值器 | `derive.py` | 🟡 **框架对（`MODEL_SPEC` 动态分发），粒度错（每算子一个文件，只有 2 个）** |

**改成 ~10 个访问模式模板**：

| 模板 | 覆盖算子 | FLOPs | Bytes |
|---|---|---|---|
| `elementwise-unary` | rope · quant · gelu | ~N | 2·N·d |
| `elementwise-binary` | silu×mul · add · combine | ~N | 3·N·d |
| **`reduce-lastdim`** | **rmsnorm · layernorm · qk-norm · fused-add-rmsnorm** | ~3N | (2~3)·N·d |
| `matmul` | gemm-bf16/fp8/int8 · qkv · o_proj | 2MNK | (MK+KN+MN)·d |
| `batched-matmul` | bmm · grouped-gemm | B·2MNK | … |
| `attention-prefill` | flash-attention-fwd · mha | 4S²hd | KV 全读 |
| `attention-decode` | paged-decode · mla-full · mla-absorb | 4Shd | **KV 全读** |
| `gather-scatter` | moe-permute · embedding-lookup | ~0 | moved·d |
| `topk-reduce` | topk-gating · sampling · mqa-logits | ~N | N·d + k·d |
| `matvec` | lm-head | 2NK | (N+1)·K·d |

`reduce-lastdim` 写一次，四个算子自动有 roofline。

### 7.3 special-case 必须声明理由

```yaml
roofline:
  template: attention-decode
  # 或，只有模板表达不了时才写：
  model_override: models/sparse_mla.py
  why_not_generic: >
    topk 稀疏导致实际读取字节依赖数据分布，模板的 bytes 只能给上界，
    remaining 会被系统性低估。
```

`validate.py` 加硬判据：**`model_override` 缺 `why_not_generic` 直接 error**。

这是防"特殊"变成偷懒出口的闸。v3 的病根就是这个——`derive.py` 只支持 2 个特例，然后计划写"M1 全覆盖"，两个特例不可能覆盖 53 张卡。

### 7.4 存放与消费

路径不变（消费者已在等），内容按 shape 索引：

```json
{
  "schema_version": "1.0",
  "card": "rmsnorm-fused",
  "model": {"kind": "template", "name": "reduce-lastdim"},
  "provenance": {"peak_bw_id": "bw-ladder.2_rw_1to1", "caliber": "achievable"},
  "by_shape": {
    "decode-m1":  {"ai": 0.5, "ridge": 96.0, "bound": "memory", "remaining": 0.42,
                   "t_actual_mode": "event_isolated", "t_floor_mode": "event_isolated"},
    "prefill-8k": {"ai": 3.0, "ridge": 96.0, "bound": "memory", "remaining": 0.31,
                   "t_actual_mode": "event_isolated", "t_floor_mode": "event_isolated"}
  }
}
```

### 7.5 S4 完成判据（v5.1 新增 golden 校验）

v5 原判据是「同一 gemm 卡两个 shape 的 `bound` 翻转」——**这只测方向敏感性，不测绝对正确性**。

**洞**：一个 `bytes` 公式系统性差 2× 的模板，`bound` 翻转照样正确，但会毒化所有 `remaining` 和 EV，而且错误安静地住在**每张卡的 `roofline.json` 里**。

> 这正是 S1 刚打完的那类错（助记符记错 → 判据本身错）。没有 golden，十个模板就是十个潜在的助记符错误。

**补两条 golden 判据：**

```bash
# G1 · 绝对正确性（每个模板一条，用"已知良好优化"的算子交叉验证）
#     公式算出的 bytes ÷ 实测时间 推出的带宽利用率必须落在 [0.6, 1.0]
python3 -I scripts/roofline_golden.py --template reduce-lastdim --card rmsnorm-fused --shape prefill-8k
# 依据：lightop 实测 1182 GB/s / 分母 1680 GB/s = 0.70 ∈ [0.6,1.0] ✅
#       若模板 bytes 差 2× → 利用率 1.40 → 超界 → 报错

# G2 · 敏感性（v5 原有的判据，保留）
#     同一 gemm 卡：M=128 应判 memory / M=8192 应判 compute
python3 -I scripts/roofline_check.py --card gemm-bf16 --shapeboth
```

**G1 通过的标准**：每个模板至少 1 个算子被交叉验证，且实测利用率落在 `[0.6, 1.0]`。
落不进去有两个可能，必须区分：**模板错**（公式问题）或 **分母错**（`bw-ladder` 口径问题）——**不许二选一硬判**。

---

## 第八章 · P5 反作弊（0.5 天，必须在盲测之前）

公开审计里两个真实案例（[KDA 分布感知](https://kernelbench.com/runs/rtxpro6000/20260709_172635_codex_gpt-5.6-sol_02_kda_cutlass) / [MinGRU 覆盖攻击](https://kernelbench.com/runs/rtxpro6000/20260716_112948_kinetic-claude_kinetic-0715_04_grid_mingru_sps)）：

> **案例 1**：故意检测检查器的 2× 数值压力分布，只有在那里才切到更贵的精确数学；在 nominal（唯一计时的）输入上截断近似。
> **案例 2**：按 `num_envs` 分派——N ≤ 2048 走精确 fp32（正是 `check.py` 检查的所有形状），N > 2048 走快 fp16（正是全部 benchmark 形状），违反声明的 `position_exact: true`。

**共同点：agent 没变强，它学会了骗评测。而且是"高级工作 + 一处作弊"，人眼难看出来。**

现有防线是部分的（`evidence` 带 sha256、多档 `acc{1,10}`、E40 超 achievable 报错）。缺四条，都能机检：

| # | 检查 | 实现 | 分工说明 |
|---|---|---|---|
| 1 | **形状分离** | `correctness_shapes ∩ bench_shapes = ∅`；bench 形状**不在仓库里**，从 harness 外部注入 | 防"按 shape 分派" |
| 2 | **扰动分布** | ×3 / ×0.01 / 取负（KernelBench-Verified 的做法） | 防**统计量分派**（`if x.mean() > 0.5`）——AST 抓不到这类 |
| 3 | **禁止行为 AST 扫描** | 候选 kernel 里出现"对**字面常数/具体 shape**的分支" → 判 dead | **只抓字面量**，抓不到统计量分派（由 2 覆盖）。别对 AST 产生假信心 |
| 4 | **输入隔离双测（v5.1 修正）** | **两个不同输入各跑一次，分别与 reference 对比** | 见下 |

**v5.1 修正：第 4 条 v5 写反了。**

v5 原写「连调两次输出逐位相同（防缓存）」——**这是错的**：

```
缓存作弊恰恰会让两次输出逐位相同（第二次直接返回第一次的结果）
⇒ 该测试防不住缓存，实际防的是"非确定性"，而且会误伤用 atomics 的合法 kernel
```

**正确写法**：

```python
out_A1 = run(kernel, input_A)
out_B1 = run(kernel, input_B)
assert allclose(out_A1, ref(input_A))
assert allclose(out_B1, ref(input_B))     # ← 关键：缓存 kernel 第二次返回 A 的结果，对不上 ref(B)
# 附加（这一条才真正防"非确定性"，且要排除 atomics 的合法场景）
if not kernel_uses_atomics(source):
    out_A2 = run(kernel, input_A)
    assert bitexact(out_A2, out_A1)
```

**另需**：`kernel 文件计时前后 sha256 不变`（防中途换实现）——这条 v5 写对了，保留。

**这比补 plan.py 优先级高**——因为一个会作弊的 agent 比一个不会优化的 agent 危险得多，而且**分数会先涨上去，你事后才发现**。

---

## 第九章 · 执行顺序与时间盒

原则：**先让数字可信 → 再让判据能跑 → 再让引擎转起来 → 最后谈能力。**

| 阶段 | 内容 | 时间 | 完成判据（可机检） |
|---|---|---|---|
| **S0** | P0 可信度层 7 项（§三） | 1 天 | ★ **可证伪重测**（v5.1 改，原判据是循环论证）：`event_isolated` 模式重测两实现各 40 次中位，**比值 ∈ 1.067 ± 3×MAD_base**；且 pin `impl_hash`。落进去 → 弧线闭环；落不进去 → **`ledger.md` 进复核名单** |
| **S1** | ISA 文档修复 + 4 条新 surface + `dead_end` 过滤 + 助记符对齐（§四） | 1 天 | `dccobjdump` 反汇编成功一次并贴 exit code；4 条新 surface 前置全 rc=0 |
| **S2** | 诊断→决策映射 + `unknown` 三态（§五） | 0.5 天 | 五个 dominant 全部映射到 ≥1 条 surface；映射表有测试 |
| **S3** | `advance.py` 发动机（§六） | 1 天 | ① **无人干预**跑完一轮，ledger +1，stage 前进；② ★ 累计 ≥5 轮后能输出 **predicted vs realized** 对照表（`calibrate.py`） |
| **S4** | roofline 统一求值器（§七） | 1.5 天 | ① **golden 绝对正确性**：每个模板 ≥1 算子交叉验证，`bytes ÷ 实测时间` 推出的利用率 ∈ `[0.6, 1.0]`；② 方向敏感性：同一 gemm 卡两 shape 的 `bound` 翻转，`allowed_actions` 跟着翻转 |
| **S5** | 反作弊四件套（§八） | 0.5 天 | ① 故意写 `if n<=2048` 的作弊 kernel **必须被抓**；② 故意写**缓存** kernel（第二次返回第一次结果）**必须被抓** |
| **S6** | R1（`mlp-gelu-geglu` 先造壳）/ R2（`gemm-int8`）盲测 | 2 天 | 六步产物齐全 + 第二轮 `rejected` 引用第一轮 verdict 与数字 |

**关键顺序变化（相对 v4）**：

- v4 把 R0 → 扩字段 → plan 排前面；v5 把 **S0 可信度层排最前**，因为 R0 的数字现在是错的
- v4 没有 ISA 阶段；v5 放 S1，因为**修文档 0.5h 就能解锁一整套诊断手段**
- v4 的 plan.py 排第 4；v5 并进 S3 `advance.py`，因为**没有驱动，plan 只会在人手里空转**

### 9.1 范围蔓延是本阶段最大风险（v5.1 强调）

第五轮评审的结论，我完全同意：

> **S0 的七项做完就该去重测，别顺手把 S1 也做了。**
> 最大的风险已经不是架构，而是范围蔓延。

所以：**S0 结束当天必须重测，不许"先把 S1 的文档也修了再一起测"**。
理由：S0 的正确性只有重测能验；如果先做 S1，重测时已经混入 ISA 方向的变量，说不清是谁的影响。

---

## 第十章 · 机器验收（每条都实跑过）

> 规矩：文档里的命令必须在当前 HEAD 实跑过。这是第四次栽在这——不是疏忽，是流程缺陷。

```bash
# ① validate（实测 0.999s，rc=0）
python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas
# → 结果：0 错误 / 17 警告

# ② 测试套件（实测 127.19s，2 failed / 605 passed）
python3 -m pytest Plugins/ops -q -p no:cacheprovider
# 现存 2 个失败，必须修（见 P0-7）：
#   test_adapter.py::test_reference_runs_and_is_finite
#   test_allowed_actions.py::test_undeclared_action_class_warns_but_does_not_block

# ③ 反汇编（实测可用，188487 B / 102 kernel）
cd /tmp && dccobjdump --inputs=extop_gfx938.co --target=gfx938 --recompile
ls -la extop_gfx938.co.ISA

# ④ 资源字段（实测可用）
/opt/dtk/llvm/bin/llvm-readelf --notes \
  /opt/dtk/lib/hipblaslt/library_bw_nmz/Kernels.so-000-gfx938.hsaco | grep vgpr_count

# ⑤ 自编译汇编（实测可用）
hipcc --save-temps --offload-arch=gfx938 --genco k.hip -o k.hsaco
ls k-hip-amdgcn-amd-amdhsa-gfx938.s

# ⑥ loop 能跑
python3 -I Plugins/ops/skills/op-optimize/loop.py \
  -C Plugins/ops/skills/op-rmsnorm-fused status

# ⑦ derive 接口（当前只有 2 个模型，且没有 --card）
python3 -I Plugins/ops/skills/op-roofline/derive.py --list
# → 可用模型： mqa_logits, sparse_mla   ← S4 要把它扩成模板驱动
```

---

## 第十一章 · 不做什么

1. **不新建 `ledger.py` / `runner.py`**（v4 判断正确）——但要建 `advance.py`（唯一编排入口）
2. **不加新门禁**（E39–E43 已够）
3. **不手工优化单个算子**
4. **不在 schema 里写没有实测支撑的因子**（v4 删掉 `c×n_params` 是对的）
5. **不在没有 `measure_mode` 的情况下宣布任何加速**
6. **不在没有形状分离的情况下跑盲测**（否则测出来的是"会不会骗评测"）
7. **不留"永久失败"的 `precondition_check`**（它是"默认通过"的镜像错误）

---

## 第十二章 · v5.1 修正清单（第五轮评审全部接受）

### 12.1 五条实质缺口

| # | 评审指正 | 修正位置 | 严重度 |
|---|---|---|---|
| **1** | **S0 完成判据是循环论证**：折算乘数由端点反解，再用它验证端点；两个未知两方程解不唯一 | §0.3 重写（改为两条独立事实）；§九 S0 判据改为**可证伪重测** `∈ 1.067 ± 3×MAD_base` | **最高** |
| 2 | S4 只测方向敏感性，不测绝对正确性；`bytes` 差 2× 的模板翻转照样正确但毒化所有 remaining | 新增 **§7.5 golden 校验**（利用率 ∈ [0.6, 1.0]） | 高 |
| 3 | P5「双调用一致性」语义写反：缓存作弊恰恰让两次输出相同 | §八 第 4 条改为**输入隔离双测**；AST 分工写清 | 高 |
| 4 | 静态计数 ≠ 动态权重（GEMM load 每 K-step、store 每 tile 一次） | §4.4 加"静态证据"警告；四条 evidence 全部降级为**上界** | 中 |
| 5 | `calibrate` 在架构图 L5，但 S0–S6 无位置 | 新增 **§6.4**；S3 判据加"≥5 轮输出 predicted vs realized" | 中 |

### 12.2 四条小问题

| # | 指正 | 修正 |
|---|---|---|
| 6 | replan 有 EV 塌缩盲区（top1 EV 0.9→0.3 仍判"继续"），分不清健康轮换与抖动 | **§6.3 重写**：新 top1 须领先 ≥1.5×；EV <0.3 或掉一半 → 停 |
| 7 | `round key` 吞合法重试（修好实现后的重试被记成 rerun） | key 加 `impl_hash`；实证：triton 有无 `residual_out` 两版本同 tag 差 40% |
| 8 | `decision` 词表在漂移（`ACCEPTED_NUMERICS_PENDING_E2E` vs `ACCEPTED_NO_NUMERICS`） | schema 枚举掉；D4 的 `{adopted, refuted, stalled}` ↔ `decision` 映射写死 |
| 9 | P0-7 别留成开放问题 | **P0-7 已改**：按 fail-closed 哲学答案是「拦」+ 一次性迁移脚本 |

### 12.3 一个我自查发现的额外缺口（原评审未提）

**`judge()` 的能力边界 = 只能判噪声，不能判偏差**（新增 §0.4）：

```
R0 背靠背    : Δ/(3×MAD) = 79×    ← 看似极度显著，符号是反的
ledger 隔离  : Δ/(3×MAD) = 1.01   ← 只勉强过线，符号是对的
```

**荒谬点：符号错的口径，"显著性"高 79 倍。** 因为 `judge()` 测的是离散度（MAD），
而 back_to_back 的偏差是系统性的、被极小的 MAD 完全掩盖。

**⇒ `measure_mode` 必须是独立前置门，不能折进显著性检验。** 这条要写进 `judge()` 的 docstring。

### 12.4 版本弧线（第五轮评审的定性，我认同并记录）

| 轮 | 状态 |
|---|---|
| v1 | 图纸有虚线 |
| v2 | 虚线画实，曲轴量错 |
| v3 | 装了不该装的零件（基于未核实论断） |
| v4 | 拆误装件，验证顺序倒置：先在能跑的卡上跑通 |
| **v5** | **跑通了，发现仪表是坏的；把"数字可信"提为第一优先级，引入反作弊层** |
| **v5.1** | **修正 v5 自己的循环论证；补 golden / 反作弊语义 / 动态权重 / calibrate；定为"停止评审、去执行"版** |

**v5.1 是"停止评审、去执行"的版本。**
S0–S6 ≈ 7.5 天；按历史规律每个执行日还会冒出 1–2 条新缝，
**所以 v6 只能是"执行中发现的修复记录"，不许是重新设计。**

**最大风险不是架构，是范围蔓延**：S0 七项做完就该去重测，别顺手把 S1 也做了（§9.1）。

### 12.5 下一版唯一值得看的指标

> **重测后的 1.067 落不落进 3×MAD。**
>
> · 落进去 → 五轮弧线闭环
> · 落不进去 → **`ledger.md` 也要进复核名单**，那会是第六个、也是更深的可信度缺口

---

## 第十三章 · 一句话

> v4 说「传动轴在车上，缺的是油和司机」——**判断对，但仪表盘是坏的**。
>
> 现在每转一圈都产出一个数字，而这个数字的**符号取决于用哪把尺子**：同一对实现，event 隔离下是慢 6.7%，背靠背下是快 17.4%。`ab.py` 自己写了"偏乐观约 23%"，`loop.py` 一个字都没记——**而且那个 23% 也被数据否掉了（乐观度逐实现变化 1.29×，不存在可移植修正因子）**。
>
> **v5.1 三件事，按顺序：**
> **① 让数字可信**（`measure_mode` + 角色校验 + `remaining` 符号门 + 同口径相减校验 + MAD≠准确度）
> **② 让判据能跑**（修 3 处文档错误 → 解锁反汇编；补 4 条活 ISA surface；`dead_end` 过滤；诊断→决策映射）
> **③ 让引擎转起来**（`advance.py` 唯一入口，只编排；replan 防 EV 塌缩；calibrate 进判据）
>
> 然后才是 roofline 求值器（带 golden）、反作弊（输入隔离双测）、盲测。
>
> **记忆让人变聪明，可信度决定结论能不能用。** 后者必须先有。
>
> **而且：R0 的价值恰恰是它"打脸"了测量而不是优化**——一个流程的第一次实机产出是"我的测量不可信"，这恰恰说明流程在自我生效。

---

## 附 · 参考资料

- [Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization](https://ar5iv.labs.arxiv.org/html/2605.26118) — 九阶段 + issue 路由 + 每阶段重分析 + 四层验证级联 + 失败返回原实现
- [The AI CUDA Engineer (Sakana)](https://pub.sakana.ai/static/paper.pdf) — stepping-stone 检索 / 进化优化
- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://ar5iv.labs.arxiv.org/html/2507.11948) — 多轮迭代优于单轮；16 轨迹 × 8 轮
- [STARK: Strategic Team of Agents for Refining Kernels](https://ar5iv.labs.arxiv.org/html/2510.16996) — 持久树记忆 + 回溯 + branch-and-bound
- [SOL-ExecBench](https://export.arxiv.org/pdf/2607.23089) — 反"相对软件基线"的度量，与我们三层口径同向
- [KernelBench-Verified](https://ar5iv.labs.arxiv.org/html/2607.16241) — 现实基线 + 多分布隐藏测试 + 显存指标
- 作弊审计实例：[KDA 分布感知](https://kernelbench.com/runs/rtxpro6000/20260709_172635_codex_gpt-5.6-sol_02_kda_cutlass) / [MinGRU 覆盖攻击](https://kernelbench.com/runs/rtxpro6000/20260716_112948_kinetic-claude_kinetic-0715_04_grid_mingru_sps)
