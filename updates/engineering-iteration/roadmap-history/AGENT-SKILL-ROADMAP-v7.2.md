# Agent Skill 路线图 v7.2：产出已经超过落账速度

**日期**：2026-11-24（v7.1 → v7.2）
**业主定性**：深度目标 = agent 拥有自主迭代研发算子的能力
**v7.2 的一句话**：**能力已经到 7 分，指标还停在 2.9，因为"能改"做到了而"能落账"没做。**

---

## 第零章 · 现状（17:55 实测）

### 0.1 这 3 小时做出来的东西是真的

```
e18d589  诊断→造变体→编译→测量 接通，advance 可自己写实现
86a1f3f  forge 一次枚举 K 组参数，赢家由 judge 判而非默认值
c5c2654  自研 HIP 实现超过厂商 —— 1454 vs 1327 GB/s
```

**测试 921 passed / 0 failed。validate 0 错误。**

### 0.2 而且它刚跑出一组数据，比已落账的任何记录都强

`86a1f3f` 的 sweep（同批测，base lightop **0.2019 ms / MAD 0.0005**）：

| 配置 | 耗时 | vs 厂商 | 3×MAD=0.0015 显著？ |
|---|---|---|---|
| **t64-VEC8** | **0.1805 ms** | **−10.6%** | ✅ |
| t256-VEC2 | 0.1810 ms | −10.4% | ✅ |
| t128-VEC4 | 0.1827 ms | −9.5% | ✅ |
| t512-VEC1 | 0.1835 ms | −9.1% | ✅ |

**四组全部显著超过厂商 9.1~10.6%。**

### 0.3 但账本上是空的

```
$ grep -c "rms-v2-reg\|auto-fusedadd" runs/loop/*.jsonl
0

卡片 last_verdict.comparison:
  cand_us: 213.0   base_us: 197.4   ratio: 1.079     ← 我们还慢 7.9%

memory/ledger.jsonl
0 bytes
```

**真实：快 10.6%。账本：慢 7.9%。方向反了，而且已经反了三次巡检。**

### 0.4 指标分：深度三轮不动，质量被代码债拖住

| 维度 | 13:56 | 17:03 | 17:47 | 趋势 |
|---|---|---|---|---|
| 深度 | 2.9 | 2.9 | **2.9** | 3 轮不动 |
| 覆盖 | 6.5 | 6.5 | 6.5 | 3 轮不动 |
| 质量 | 6.7 | 6.6 | **6.6** | ⬇ |

```
深度  D1(13.5) D2(1) D3(0.222) D4(1) D5(8)
质量  Q1(0) Q2(17) Q3(4) Q4(110) Q5(257)
```

**Q4（`except Exception`）单调上升：89 → 100 → 101 → 108 → 110**（6 小时 +21）

---

## 第一章 · 核心诊断：产出速度 > 落账速度

### 1.1 数字

```
今天产出：11 个 commit、9 个工具、921 个测试、4 组显著超厂商的数据
今天落账：1 条有效记录（rmsnorm，且还是旧的 213.0/197.4）
        memory/ledger.jsonl = 0 字节
```

**每多做一个工具，就有更多的结果找不到地方落。**

### 1.2 为什么这不是小问题

三维评分卡是**证据驱动**的。证据不落账，会发生三件事：

1. **分数不动** → 深度 2.9 三轮不变（如实，但不是因为没能力）
2. **下一轮 plan 会重复劳动** → `recall` 查不到，agent 会重跑已经跑过的实验
3. **归因无法做** → `ablate` 需要知道"冠军由哪几项组成"，而冠军没有记录

> **落账不是文书工作，是迭代链的输入。没有它，loop 的每一轮都是第一轮。**

---

## 第二章 · v7.2 三件事（按 ROI 排，两天做完）

### A · 落账流水线（半天）⭐ 最高优先

**不要只补这一次，要建一个机制。**

现在的问题：`advance --forge` 生成 → 编译 → 测量 → 判定，但**测定结果只留在 commit message 和临时 json 里**。

**改法**：让 `advance --forge` 在测定后**直接写 record**，不等 `loop.py run` 的判定路径：

```python
# scripts/advance.py，forge 分支结尾
for cand in sweep_results:                     # 4 组 (tag, ms, mad)
    append_record(card, {
        "ts": ..., "card": card.name, "tag": cand.tag,
        "base_ms": batch_base_ms, "base_mad": batch_base_mad,
        "cand_ms": cand.ms, "mad": cand.mad,
        "measure_mode": "event_isolated",      # 必填，来自 ab 输出
        "caliber": {...},
        "decision": judge(...),                # 复用唯一判定
        "significant": ...,
        "direction": "register-resident",      # 来自 transform
        "expected_effect": "-10%",             # 来自 candidates means
        "trigger_pattern": {"bound": "memory", "signal": "lds-roundtrip",
                            "direction": "register-resident"},
    })
```

**同时更新卡片**（用最好的一组）：

```yaml
last_verdict.comparison:
  baseline: lightop-fused
  baseline_role: competitor
  cand: t64-VEC8
  cand_us: 180.5        # 0.1805 ms
  base_us: 201.9        # 0.2019 ms
  ratio: 0.894
  significant: true
  measure_mode: event_isolated
```

**验收（四条全过）**：
```bash
# 1) ledger 有记录
grep -c "t64-VEC8\|auto-fusedadd" Plugins/ops/skills/op-rmsnorm-fused/runs/loop/*.jsonl   # ≥1

# 2) 卡片方向正确
python3 -I -c "import yaml;d=yaml.safe_load(open('Plugins/ops/op-atlas/cards/rmsnorm-fused.yaml'));c=d['last_verdict']['comparison'];assert c['ratio']<0.95, c;print('ratio',c['ratio'])"

# 3) memory 不再空
wc -l memory/ledger.jsonl        # ≥1

# 4) validate 不破
python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas   # 0 错误
```

### B · 修精度门（半天）—— A 的必要条件

**不修 B，A 落账的记录也不能 claim。**

实测四个变体全 FAIL，**含厂商**：

```
FAIL lightop-fused        max_abs=3.125e-02 (限 2.0e-03)   ← 厂商也 FAIL
FAIL fused-3pass-triton   max_abs=1.562e-02
FAIL fused-3pass-tilelang max_abs=1.562e-02
FAIL rms-v2-reg           max_abs=1.562e-02
```

**根因（算术）**：`max_abs = 2.0e-3` **比 bf16 的 1 个 ULP 还小**。

| 数值区间 | bf16 ULP | 2.0e-3 容得下 |
|---|---|---|
| [0.5, 1) | 0.0039 | **0.51 ULP** |
| [1, 2) | 0.0078 | **0.26 ULP** |
| [2, 4) | **0.0156** | **0.13 ULP** |

实测 `1.5625e-2 = 2⁻⁶`，**正好是 [2,4) 区间的 1 个 ULP**——完美正确的舍入，不是错误。

**修法**：

```yaml
tolerance:
  vs_fp32_ref:
    mode: ulp_bound
    max_ulp: 2
    note: |
      bf16 输出最小可分辨步长随量级变化（[2,4) 区间 ULP=2^-6）。
      绝对容差 2.0e-3 在 bf16 上不可能满足——它小于 1 ULP。
      改用 ULP 判据：max_abs / ULP(max_abs 处的值) ≤ max_ulp。
      2026-11-24 实测四个变体（含厂商）均为 1 ULP，全部应 PASS。
```

**同步改四张卡**（容差是同一来源抄的）：`rmsnorm-fused` / `fused-add-rmsnorm` / `fusion-add-rmsnorm-quant` / `fusion-rmsnorm-rope`

**验收**：
```bash
python3 -I scripts/opcheck.py correctness \
  --contract Plugins/ops/skills/op-rmsnorm-fused/contract.yaml --shape prefill-8k \
  --variants lightop-fused,fused-3pass-triton,fused-3pass-tilelang,rms-v2-reg
# → 4 个 PASS，每个报 max_ulp ≤ 2（不再报 max_abs 超限）
```

> **这是全项目第三例"判据本身错"**（前两例：ISA 助记符、lightop 227.0）。
> 共同点：**把"实现有问题"和"判据有问题"混为一谈，而默认怀疑实现。**

### C · 第 2 张卡（1 天）—— 唯一能推动深度分的事

**为什么必须是第 2 张卡**：深度 D1 是 53 张卡的均值，一张卡满分只值 **+0.42**。要涨到 5+，必须有第 2、3 张。

**选哪张**：`op-fused-add-rmsnorm`（已核实齐备度）

```
bench/          ✅
_common.py      ✅
contract.yaml   ✅
shapes.yaml     ✅ (tiny / decode-m1 / decode-m32 / prefill-8k)
variants.yaml   ✅
tests/          ✅
candidates.yaml ❌  ← 缺
roofline.json   ❌  ← 缺
```

**为什么是它**：它就是 rmsnorm 去掉残差融合。`rms-v2-reg` 的核心结构（**寄存器驻留 + 16B 向量访存**）可直接迁移——**这是"经验能不能复用"的第一次真测试，也是 L5 记忆层该干的活。**

**做法**：
1. 写 `candidates.yaml`，其中 `means` 写清"复用 rms-v2-reg 的寄存器驻留结构"，`expected_effect: -8%~-12%`
2. 写 `roofline.json`（`reduce-lastdim` 模板，`bound: memory`）
3. `advance --forge` 生成变体 → 编译 → 体检 → 测
4. **落账**（走 A 的流水线）

**验收**：
```bash
# 1) 显著加速
grep -h "fusedadd" Plugins/ops/skills/op-fused-add-rmsnorm/runs/loop/*.jsonl | \
  python3 -I -c "import sys,json;rs=[json.loads(l) for l in sys.stdin]; \
  assert any(r['significant'] and r['measure_mode']=='event_isolated' for r in rs); print('OK')"

# 2) 卡片 ratio < 0.95
python3 -I -c "import yaml;d=yaml.safe_load(open('Plugins/ops/op-atlas/cards/fused-add-rmsnorm.yaml'));print(d['last_verdict']['comparison']['ratio'])"

# 3) 反作弊过
python3 -I scripts/anticheat.py --scan <生成的 kernel> --shape-sep
```

---

## 第三章 · 一个必须警惕的反直觉发现

### 3.1 sweep 四组极差只有 1.66%

```
t64-VEC8   0.1805
t256-VEC2  0.1810
t128-VEC4  0.1827
t512-VEC1  0.1835
```

**参数在 VEC>=1 范围内几乎不影响结果。** 真决定因素是**结构**（寄存器驻留 + 16B 向量），不是 tile 参数。

### 3.2 推论一：`design_space.py` 枚举 486 个配置的价值可能很小

如果参数不重要，那"枚举 486 个配置并按预测排序"**大部分是在搜索一个平坦的空间**。

**建议改法**：`design_space` 的排序依据从"预测时间"改为"**结构差异**"：

```yaml
# 不再按预测 ms 排序，按结构分组
groups:
  - structure: register-resident   # (x+r) 留寄存器
  - structure: lds-roundtrip       # 走 LDS 中转
  - structure: vector-width-16B    # 16 字节访存
# 同结构内只取 1-2 个代表（因为参数不重要）
```

**判据**：枚举结果从 486 → **≤20 个代表性配置**，且每组标注结构差异。

### 3.3 推论二：它自己的归因也被这次 sweep 推翻了

> 上一版 PARAM_BASIS 写「256/VEC2 快 24%，因为 512 线程占更多调度槽」。
> 那 24% 来自**一次孤立单跑**（base 未同批测量）。同批重测后只差 1.7%
> —— **差距不存在，归因也错了**。

**这是本轮最有价值的一条**：它用同批测量推翻了自己的归因。**这正是 v7 说的 L3「因果」的雏形。**

**但也说明**：ablate.py（367 行，在写）必须**同批测**，否则会重犯这个错。

---

## 第四章 · 质量债：Q4 单调上升

```
89 → 100 → 101 → 108 → 110     6 小时 +21
```

**921 个测试是好事，但 `except Exception` +21 意味着新增代码里有 21 处吞异常——而吞异常的代码，测试往往测不出来。**

**建议**（最小成本）：
1. 新增代码的 `except Exception` 必须带 `# noqa: BLE001` + 一句原因（现状已有部分这么写）
2. 设一个护栏：**`Q4 > 110` 时 CI/巡检报警**（不是阻断，是可见）
3. 不要求现在清债，只要求**停止增长**

**判据**：下一轮巡检 `Q4 ≤ 110`（不涨）。

---

## 第五章 · 评分预估

### 5.1 各阶段

| 做完 | D1 | D2 | D3 | D4 | D5 | **深度** |
|---|---|---|---|---|---|---|
| 现在 | 13.5 | 1 | 0.222 | 1 | 8 | **2.9** |
| A+B（落账+精度门）| 13.6 | 1 | 0.222 | 1 | 8 | **2.9** |
| + 补多形状（S 0→1.0）| 13.9 | 1 | 0.222 | 1 | 8 | **3.0** |
| + C（第 2 张卡 DS 8→65）| 15.0 | 2 | 0.26 | 2 | 10 | **≈4.0** |
| + 第 3 张卡 | 16.1 | 3 | 0.30 | 3 | 10 | **≈5.1** |

**算例（A+B+C 完成后）**：
```
D = 0.30×(15.0/45) + 0.25×(2/5) + 0.20×(0.26/0.85) + 0.15×(2/8) + 0.10×(10/10)
  = 0.100 + 0.100 + 0.061 + 0.0375 + 0.100 = 0.398
⇒ 深度 ≈ 4.0
```

### 5.2 诚实的期望管理

```
深度 4.0 = 2 张卡有真加速（当前 1 张）
深度 5.1 = 3 张卡
深度 6.6 = 5 张卡
到不了 9：53 张卡里 26 张 draft，均值上限 8.7，且需要 3-4 周
```

---

## 第六章 · 不做什么

1. **不加第 12 个工具**（现有 9 个 + ablate/check_means 在写，够了）
2. **不扩 design_space 的配置数**（sweep 已证明参数不重要，486 已经是浪费）
3. **不借 SkillHub 的判据**（`min(ms)` 选冠军、`noise_threshold=2%` 固定阈值）
4. **不在 B（精度门）完成前宣布"超厂商"**
5. **不做新的评分维度**（三维不动）
6. **不让 Q4 继续涨**（≥110 就报警）
7. **`check_means` 不要一刀切强制**（它自己发现的：11 个 pending 候选 0 个写了 means，
   一刀切会导致 100% 停摆。按它提的 warn → 逐卡转 block）

---

## 第七章 · 一句话

> **能力已经到 7 分，指标还停在 2.9——因为"能改"做到了，"能证明"和"能落账"没做。**
>
> **它现在有 4 组显著超厂商 10% 的数据躺在 commit message 里，而卡片上写着"我们慢 7.9%"。**
>
> **v7.2 三件事：A 落账流水线（半天）→ B 修精度门（半天，A 的必要条件）→ C 第 2 张卡（1 天，唯一能推动深度分的）。**
>
> **顺序不能反：B 不修，A 落账的记录也不能 claim；A 不做，C 的成绩又找不到地方落。**
>
> **而我最看重的一条是它自己做到的**：用同批测量推翻了自己的"快 24%"归因（实为 1.7%）。
> **能推翻自己的归因，才是研发；不能，就只是搜索。**
