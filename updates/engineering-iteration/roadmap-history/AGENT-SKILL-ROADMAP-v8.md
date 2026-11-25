# Agent Skill 路线图 v8（业主修订版）

**日期**：2026-11-25
**v8 原文**：`AGENT-SKILL-ROADMAP-v8.orig.md`（其他 agent 整理，370 行）
**评审**：`V8-REVIEW.md`
**业主定性（本轮）**：
> 「我觉得我们现在缺的是自主迭代的能力。补充其他算子可能不算是最主要的。」

**v8 修订版的一句话**：
> **我们不是缺第 4、5 张卡，是缺「把第 1 张卡推到第 4 代」的机制。**

---

## 第零章 · 为什么修订（用数据说）

### 0.1 现有 25 条实验记录的全貌

```
ACCEPTED_NO_NUMERICS            11
INCONCLUSIVE                     6
REJECTED_NO_GAIN                 4
ACCEPTED_NUMERICS_PENDING_E2E    3
ACCEPTED_NUMERICS                1
```

按时间排：

| 时间 | 卡 | 判定 | Δ |
|---|---|---|---|
| 11-24 17:30-17:32 | rmsnorm-fused | 7 × ACCEPTED | −9~−11% |
| 11-24 20:06-20:10 | rmsnorm-fused | 3 × REJECTED | +3~+21% |
| 11-25 00:30-00:48 | rmsnorm-fused | 1 ACCEPTED + 6 INCONCLUSIVE | — |
| 11-25 03:37 | fused-add-rmsnorm | ACCEPTED | **−45.1%** |
| 11-25 11:11 | norm-act-swiglu | ACCEPTED | **−50.9%** |

### 0.2 三个致命观察

**① 只有 rmsnorm-fused 跑多轮，且第二轮之后全线走低**

11-24 17:30 那次 sweep 出了 7 条 ACCEPTED（−9~−11%），
之后 3 小时的所有尝试：3 条 REJECTED（+3~+21%，即变慢）、6 条 INCONCLUSIVE。

**这不是「优化到极限」，是「不知道下一步该改什么」。**

**② fused-add-rmsnorm 和 norm-act-swiglu 都是「一次成功、立即归档」**

-45.1% 和 -50.9% 都是**第一次尝试就拿到的大幅收益**，之后再没跑过第二轮。

**一次就成功，说明方案显而易见；也只有一次，说明我们不会在好方案上继续推。**

**③ 25 条记录里，0 条是「以上一轮赢家为基线」**

全部是「候选 vs 固定基线」。`advance.py` 的 `replan()` 做的是
**在候选队列里换向**（`CONTINUE_SWITCH`），
**从来没有「把赢家设为新基线再打一代」**。

### 0.3 这就是「自主迭代能力」的定义

| | 我们现在 | ForgeKernel |
|---|---|---|
| gen0 相对 PyTorch | 我们 −45% | 它 +0.3% |
| gen1 相对 gen0 | **不存在** | +9.9% |
| gen2 相对 gen1 | **不存在** | +4.2% |
| 累计 | −45%（一次性） | +29%（滚四代） |

**注意第一行**：我们 gen0 就比它强得多（−45% vs +0.3%）。
**但它的累计成绩来自「滚四代」，我们的来自「赢一次」。**

如果有一天遇到一个 gen0 只能做 −3% 的算子，
**我们没有任何机制把它推到 −20%。那才是能力的真考验。**

---

## 第一章 · v8 修订版的核心判断

### 1.1 和原文的分歧

| 原文主张 | 修订版 | 依据 |
|---|---|---|
| E1 自演化排第一，但目标卡选了 `fused-add-rmsnorm` | ✅ 对，但要**先补 M 项**（半天）再 promotion | 它 remaining=+20.3%，有空间；但 M=0.5 意味着没机制证据，promotion 后无法解释收益 |
| E4「十张满卡」是唯一推分的事 | 🟡 **推分对，但排第二** | 分数是结果；没有自演化，第 10 张卡同样是「一次成功就归档」 |
| E3 hw_kb 要新写 7 条 | ✅ 已写 5 条，剩 2 条 | `hw_kb/hip/gfx938/` 已有 |
| D3 补 19 张 comparison 是「第五天一天」 | ❌ **实际要 9-10 天** | 每张要真跑 A/B，不是填字段 |

### 1.2 修订后的四件事

```
F1 · 自演化闭环（3 天）    ← 原文 E1，本轮最高优先
F2 · 机制沉淀（1 天）      ← 原文 E3，但"沉淀什么"由 F1 决定
F2.5 · 两张卡补满（1 天）  ← 原文 E4 的前两张，作为 F1 的输入与验证
F3 · 课程化（1 天）        ← 原文 E2
F4 · 广度扩张（之后）      ← 原文 E4 剩余 + D3
```

**为什么 F1 排第一**：它是唯一一件「不做就永远追不上」的事。
F2.5 和 F3 都是它的输入/输出。

---

## 第二章 · F1 · 自演化闭环（3 天）

### 2.1 现状诊断

`advance.py` 的 `replan()`（第 695 行）：

```python
def replan(prev_top, new_top, ledger=None):
    if not new_top:                    return STOP
    if new_top[0].ev < 0.30:           return STOP      # EV 塌缩
    if prev_top and new_top[0].ev < 0.5 * prev_top[0].ev:  return STOP
    if prev_top and prev_top[0].id == new_top[0].id:       return CONTINUE_SAME
    if len(new_top) >= 2 and new_top[0].ev < 1.5 * new_top[1].ev:  return STALLED
    return CONTINUE_SWITCH
```

**它只做一件事：决定下一轮试哪个候选。**
它不知道「上一轮赢了什么」，也就不会把赢家变成新基线。

### 2.2 要加的东西

#### ① 封存（seal）

某个变体判 `ACCEPTED_*` 且 `ratio < 0.98` 时，把它封存成**下一代的参照**：

```
<卡>/reference/
  sealed/
    gen1/
      kernel.hip          ← 源码副本
      build.sh            ← 编译命令（含所有 -D）
      MANIFEST.json       ← sha256 + 编译参数 + 实测值 + 口径
```

**MANIFEST 必须含**（这是它和「随便复制的文件」的区别）：

```json
{"gen": 1,
 "sha256": "...",
 "compiled_with": {"BLOCK_ROWS": 2, "THREADS": 256},
 "measured": {"ms": 0.1496, "mode": "event_isolated", "rounds": 5,
              "acc_ok": true, "max_ulp": 1.0},
 "verdict_record": "runs/loop/20261125.jsonl#42",
 "beat": {"base": "two-step", "base_ms": 0.3050, "ratio": 0.494}}
```

#### ② 新目标（target）

```json
// <卡>/gen2/target.json
{"gen": 2,
 "reference": "reference/sealed/gen1",
 "base_ms": 0.1496,
 "target_ms": 0.1425,        // = base_ms / step，step 默认 1.05
 "step": 1.05,
 "noise_floor": 0.0211,      // = gen1 的五轮 spread（实测），不是固定值
 "stop_if": "增益 < noise_floor 或 连续 2 轮无 ACCEPTED"}
```

**`noise_floor` 必须是实测的 spread**，不是常数。这是我们和 ForgeKernel
做法一致、但比我们现有 `MAX_CONSECUTIVE_REFUTED=2` 更精确的地方。

#### ③ 停止条件（写进代码，不靠人判）

```python
def should_stop(gen_records, noise_floor):
    if not gen_records:
        return True, "这一代没有确认的最优"
    best = min(r["ratio"] for r in gen_records)
    if (1.0 - best) < noise_floor:
        return True, "增益 %.3f < 噪声地板 %.3f" % (1 - best, noise_floor)
    return False, ""
```

#### ④ 分数怎么算（这是最容易做错的地方）

**promotion 之后，`comparison.baseline` 必须改，但 `floor` 不能改。**

```
comparison.base_us  ← 改成上一代的最优（0.1496 ms）
comparison.cand_us  ← 这一代的最优
comparison.ratio    ← 这一代相对上一代

floor.t_actual_us   ← 这一代的最优（绝对值）
floor.remaining     ← 仍相对**硬件地板**（不变）
```

**为什么 floor 不能变**：地板是硬件属性，不因为你换了基线就变。
如果 promotion 后把 floor 也改了，就会出现「remaining 一直是 20%」的假象。

#### ⑤ 与现有 `ab.py` 的关系

**不改 `ab.py`**。它已经支持 `--base <变体名>`。
promotion 只是让 `--base` 指向一个**注册过的封存变体**。

### 2.3 验收（四条，全是机检）

```bash
# a) 封存有效：MANIFEST 含 sha256 + 编译参数 + 口径
python3 -I scripts/self_evolve.py --card fused-add-rmsnorm --verify-seal gen1

# b) 新目标是量化的，且比基线快
python3 -I -c "import json;d=json.load(open('.../gen2/target.json'));
               assert 0 < d['target_ms'] < d['base_ms']"

# c) 至少跑出 1 代，且这一代有 ACCEPTED 记录
test -f <卡>/gen2/record.json && grep -q '"decision": "ACCEPTED' <卡>/gen2/record.json

# d) 链条能停（不许无限跑）
python3 -I scripts/self_evolve.py --card <卡> --dry-run | grep -q "stop_if"
```

### 2.4 预期与期望管理

**如果 gen1 像 ForgeKernel 那样再涨 4-10%**：
`fused-add-rmsnorm` 的 `ratio` 从 0.549 再降 → DS 的 U 项变好，但**D1 只涨 0.1 左右**。

**如果 gen1 涨不动（很可能）**：
这是**更有价值的结果** —— 说明这个算子真的到极限了，或者我们的候选枚举不够深。
两种情况都比「不知道」好。

**F1 的价值不在分数，在证明「我们能滚起来」。**

---

## 第三章 · F2 · 机制沉淀（1 天）

### 3.1 现状

`hw_kb/hip/gfx938/` 已有 5 条（**已完成，不用重写**）：

| 条目 | 状态 |
|---|---|
| `reg_resident_vs_lds` | ✅ 6.9x 差距 |
| `private_spill_trap` | ✅ 慢 3.3x 且不报错 |
| `vec16_uint4` | ✅ 上 1000 GB/s 的必要条件 |
| `warp_reduce_shfl` | ✅ 屏障 8→2 |
| `lds_bank_swizzle` | ❌ **refuted**：解决了真问题但只快 0.36% |

### 3.2 本轮要补的（由 F1 的结果决定）

**如果 F1 的 gen1 涨了**：把**涨的那个改动**写成第 6 条。
这是最有价值的 —— 它是「连续两代都有效的机制」，比单次收益更可信。

**如果 F1 的 gen1 没涨**：把**试过但没用的那个**写成第 6 条，标 `❌ refuted`。
反例比正例更值钱（`lds_bank_swizzle` 已经证明了这一点）。

### 3.3 明确不做的

- ❌ 不从 ForgeKernel 抄 200 条（我们没有那些 GPU 的实测）
- ❌ 不写没有实测的条目（`⬜ unverified` 只能占位，不能进索引）
- ❌ 不为了凑数把「寄存器驻留」拆成三条

---

## 第四章 · F2.5 · 两张卡补满（1 天）

### 4.1 为什么需要这一步

F1 的 promotion 需要「有空间的卡」当输入。现状：

| 卡 | remaining | 适合 promotion？ |
|---|---|---|
| `fused-add-rmsnorm` | **+20.3%** | ✅ 但缺 M 项（无机制证据） |
| `norm-act-swiglu` | **+21.1%** | ✅ 但缺第 3 形状（S=0.5） |
| `rmsnorm-fused` | **−0.15%** | ❌ 已到地板 |

### 4.2 做什么（各半天）

**`fused-add-rmsnorm`**：采 PMC，把 M 从 0.5 → 1.0，DS 92 → 96。
- 具体：用 `hipprof` 采 `fa-rms-reg` 的 LDS / VGPR / stall 计数
- **难点**：我们的 PMC 采不到 `SQ_LDS_BANK_CONFLICT`（D4 债）
- **兜底**：用 ISA 反汇编代替（`dccobjdump` 数 `ds_read` / `global_load` 指令数）

**`norm-act-swiglu`**：补第 3 个形状（如 M=4096），S 0.5 → 1.0，DS 89 → 96。

### 4.3 验收

```bash
python3 -I scripts/depth.py --card fused-add-rmsnorm   # DS >= 96
python3 -I scripts/depth.py --card norm-act-swiglu     # DS >= 96
```

---

## 第五章 · F3 · 课程化（1 天）

### 5.1 做什么

`sources.yaml` 的 17 条 surface 是**平铺的**。加 `curriculum.yaml` 分层：

```yaml
M0:   # 架构层：定了就冻结，改起来最贵
  - occupancy          # tile / warp / 驻留预算
  - layout-pack        # 数据布局（决定后续所有访存形态）
  - fusion-boundary    # kernel 切分
  - split-k
M1:   # 访存层：在 M0 结构内选
  - vmem-vector-width
  - mls-path
  - lds-read-form
M2:   # 调度层
  - prefetch-depth
  - barrier-width
  - cross-lane-fold
M3:   # 尾部
  - precision-drop
  - grid-tail
```

### 5.2 验收

```bash
python3 -I scripts/check_curriculum.py
# → 每条 surface 有且只有一个层；M0 的每条都有 gate
```

### 5.3 为什么排第四

**它不推分，且没有 F1 就没法验证有没有用。**
课程的真正价值是「让 promotion 每代都有事可做」——
先有 promotion，课程才有意义。

---

## 第六章 · F4 · 广度扩张（之后，不排期）

### 6.1 原文的 E4 剩余 + D3

| 项 | 成本 | 推分 |
|---|---|---|
| `rmsnorm-fused` 补多形状 | 半天 | D2 |
| `gemm-bf16` / `gemm-fp8` 补对比 + 轮次 | 2 天 | D2/D3 |
| D3 补 19 张 comparison | **9-10 天** | 最后一公里 |

### 6.2 什么时候做

**当 F1 在一个新算子上验证过之后。**

如果 F1 的自演化机制只在 `fused-add-rmsnorm` 上跑通，
那它可能只是「这张卡恰好有空间」。**换一张没做过的卡再跑一遍**，才算机制成立。

---

## 第七章 · 精算（沿用原文，已验证正确）

```
D = 10 × Σ(w × ratio)
D2/D3/D4/D5 全满 = 0.70 → D1 只需 0.20 → DS 均值 30
```

**这个结论的重要性**：不需要 10 张 90 分的卡，**10 张 30 分的就够**。

| 检查点 | 现在 | 要 9 分 |
|---|---|---|
| D1 DS 均值 | 16.5 | **30** |
| D2 ≥70 卡 | 3 | **5** |
| D3 comparison | 0.296 | **1.00** |
| D4 ≥3 轮 | 1 | **8** |
| D5 PMC | 8 | **10** |

**注意 D4**：现在只有 1 张卡有 ≥3 轮记录。**F1 直接推 D4。**

---

## 第八章 · 执行顺序（本轮）

```
第 1 天上午   F2.5-a  fused-add-rmsnorm 补 PMC（DS 92→96）
第 1 天下午   F2.5-b  norm-act-swiglu 补第 3 形状（DS 89→96）
第 2 天       F1-a    封存机制 + MANIFEST + 停止条件
第 3 天       F1-b    在 fused-add-rmsnorm 上跑 gen2，验证闭环
第 4 天       F1-c    在 norm-act-swiglu 上再跑一遍（验证不是偶然）
第 5 天       F2      把 F1 的结果沉淀成 hw_kb 第 6 条
第 6 天       F3      课程化 curriculum.yaml
```

**每天的自检**：
```bash
python3 -I scripts/depth.py --json | grep -E "D1|D4"   # D4 有没有涨
```

---

## 第九章 · 纪律（继承原文，加一条）

1. **没有新增卡片数据，这一天算没做事**（原文，仍然有效）
2. **连续两代都没涨，必须停下来写清楚为什么**（新增）
   —— 「试了没用」是有价值的结果，「一直在试但不说结果」不是
3. **promotion 后 floor 不能改**（新增）
   —— 地板是硬件属性，不因为换了基线就变
4. **不加第 12 个工具**（原文）
5. **不为分数降级回退**（原文）

---

## 第十章 · 一句话

> **v5–v7 解决了「单个算子能优化多准」；ForgeKernel 证明成绩来自「能把一个算子推几代」。**
>
> **我们现在 25 条记录里，0 条是「以上一轮赢家为基线」。**
> **这就是自主迭代能力的全部差距。**
>
> **v8 本轮：先补两张卡（1 天）→ 建 promotion 闭环（2 天）→ 跑通两代（2 天）→ 沉淀机制 + 课程化（2 天）。**
>
> **不再用「做了几张卡」衡量进度，用「有几条链滚过两代以上」。**
