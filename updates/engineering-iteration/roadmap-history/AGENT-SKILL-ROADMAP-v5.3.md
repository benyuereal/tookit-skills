# Agent Skill 路线图 v5.3：补上"改"的能力 —— 从自主诊断到自主研发

**日期**：2026-11-24
**业主定性**：深度目标 = **让 agent 拥有自主迭代研发算子的能力**
**v5.3 的位置**：v5.1（数字可信）→ v5.2（执行记录：S1–S5 已落地，质量 3.7→6.7）→ **v5.3（从"诊断"到"研发"）**

---

## 第零章 · 现状（14:00 实测）

### 0.1 已完成的（v5.1 的 S0–S5 全部绿灯）

```
S0 可信度层   ✅  measure_mode / 角色校验 / remaining 符号门 / dead_end / 轮次分离
S1 ISA 复活   ✅  surfaces 13→17（4 条活 surface）+ isa_probe + 三处文档修正
S2 诊断映射   ✅  dominant-to-surface.yaml
S3 编排       🟡  calibrate.py + advance.py（只 dry-run 过）
S4 roofline   ✅  11 个访问模式模板 + golden 校验（selftest 4/4）
S5 反作弊     ✅  四件套（selftest 11/11，缓存 kernel 被抓）
测试           ✅  736 passed / 0 failed / 141s
```

**分数**：深度 **2.9** / 覆盖 **6.5** / 质量 **6.7**

### 0.2 八层能力核对（针对"自主研发"这个目标）

| 层 | 你要的能力 | 现状 | 证据 |
|---|---|---|---|
| 1 | 资源状态（带宽/算力） | ✅ 有 | constants.yaml 三层口径 + bw-ladder 四档 |
| 2 | roofline 模型 | ✅ 有 | templates.py **11 模板** |
| 3 | **流水线布局 / tile 分析** | 🟡 **几乎空白** | `models/` 只有 2 个；`sparse_mla` 是唯一带 tile 参数的（`bm/bi/thr/split`）；11 模板是**流量级** |
| 4 | 诊断瓶颈 | ✅ 有 | PMC 三态 + isa_probe（实测输出"假设 + 限制 + 纪律"） |
| 5 | 决定方向 | ✅ 有 | 17 surfaces + EV + dominant→surface |
| 6 | 验证效果 | ✅ 强 | golden + anticheat + judge + measure_mode |
| 7 | **改代码** | ❌ **无** | `surfaces.yaml` 的 `transform` 字段 **0 命中** |
| 8 | **迭代收敛** | ❌ **无实证** | advance.py 只 dry-run；**0 个算子变快** |

### 0.3 三个具体缺口（实测）

**缺口 1 · surface 只到"改哪里"，没到"怎么改"**

```
17 条 surface 字段统计：
  precondition_check 17 · expected_gain 17 · check 17
  transform 0                                  ← 没有
```

`occupancy` 告诉你"CTA/CU 太低、收益上界 ≤15%"，但**不告诉你**：降 VGPR 是改 tile？改 unroll？改 mma 形状？改成多少？

**缺口 2 · 候选只能引用"已有实现"，不能"改现有实现"**

```yaml
candidates schema: {id, direction, action_class, expect, done_when, variant}
                                                              ↑
                                              指向一个已手写好的实现
```

它无法表达「把当前实现的 BLOCK_M 从 128 改成 64 再测」。**所以现在的循环是"选型"，不是"研发"。**

**缺口 3 · 没有 design space，也没有"改一个参数会怎样"的模型**

`derive.py` 能算**给定配置**的地板，`sparse_mla` 甚至接受 tile 参数——**但没有任何东西枚举这些组合**，也不能回答"BM 128→64 会怎样"。

> 结论：现在是 **"会看病、会开药方、不会动手术"**。

---

## 第一章 · 目标（单一可证伪验收）

> **给 agent 一个它没见过、且仓库里没有厂商实现的算子**（例如某个融合算子），
> **只给 shape 和精度要求，不给任何方向提示**，让它自己：
> **① 建 roofline 模型 → ② 枚举 design space → ③ 生成 kernel → ④ 编译并 isa_probe 读回 → ⑤ 改 2–3 轮 → ⑥ 达到算出的地板 ≥85%**
> **全程无人干预。**

**这一条过了，"自主迭代研发算子"才成立。过不了，前面所有基础设施都只算"测量更准"。**

分层定义（明确边界，避免无限扩张）：

| 层 | 内容 | v5.3 是否覆盖 |
|---|---|---|
| **L1 参数/布局** | tile 形状、warp 数、pipeline 深度、split-K、向量宽度、cache hint | ✅ **全部覆盖** |
| **L2 结构变换** | fusion boundary、layout pack、precision path、persistent | 🟡 半覆盖（transform 字段描述，代码生成后置） |
| **L3 算法创新** | 重写数学（FlashAttention 级） | ❌ 不在范围 |

---

## 第二章 · 六个补强包

### C1 · `surface.transform` 字段（半天，最高杠杆）

给 17 条 surface 补"具体怎么改"。三种 kind：

```yaml
- id: occupancy
  transform:
    - kind: param
      what: "{BLOCK_M, num_warps, num_stages}"
      candidates: ["BLOCK_M 128→64", "BLOCK_M 128→96", "num_warps 8→4", "num_stages 3→2"]
      legality: "LDS = BLOCK_M*BLOCK_K*2B ≤ 64KB；VGPR ≤ 255；grid % CU 任意"
      predicted_dir: "降 VGPR ⇒ CTA/CU 从 1 升到 2 ⇒ 延迟隐藏改善"
    - kind: code
      what: "累加器 fp32 分片累加（128 个累加寄存器拆成 2×64）"
      legality: "需 bit-exact 或 rel_l2 ≤ contract 容差"
    - kind: isa
      what: "写回从 dwordx2 提到 dwordx4"
      legality: "N % 8 == 0；地址 16B 对齐"
```

**验收（可机检）**：
```bash
python3 -I scripts/check_transform.py                 # 17 条全覆盖
python3 -I scripts/check_transform.py --legality      # 每条 legality 可被脚本解析（非散文）
```
判据：**17/17 有 transform；且 ≥10 条是 `kind: param`（可参数化，不是散文）。**

---

### C2 · `candidates.patch` 字段 + 变体生成（1 天）

让候选能表达"基于现有实现改参数"：

```yaml
- id: bm64
  direction: occupancy
  action_class: raise-concurrency
  parent: fused-3pass-triton        # 相对于哪个实现改
  patch: {BLOCK_M: 64, num_warps: 4}
  expect: "-8%"
  expect_basis: "golden: BLOCK_M 128→64 ⇒ VGPR 212→~120 ⇒ CTA/CU 1→2"
  done_when: ...
```

配套 `variant_forge.py`：读 `parent` + `patch` → 生成新变体（triton 用 `constexpr` 覆盖；HIP 用宏）。

**验收**：
```bash
python3 -I scripts/variant_forge.py --card op-rmsnorm-fused --cand bm64 --dry-run
# → 输出生成的变体源码 + 参数替换 diff
python3 -I scripts/variant_forge.py --card op-rmsnorm-fused --cand bm64 --build
# → 编译成功，isa_probe 读出 VGPR 与 parent 不同
```
判据：**同一 parent 生成 2 个不同 patch，`isa_probe` 读出的 VGPR/LDS 必须互不相同**（否则参数没接进代码）。

---

### C3 · `design_space.py`（1 天，与 C4 并行）

```
design_space.py --card X --shape Y [--json]
  → 枚举 (BM, BN, BK, warps, stages, split_k, vec_width)
  → 合法性过滤：LDS ≤ 64KB / VGPR ≤ 255 / K % BK == 0 / BK % mma_k == 0
                / grid ≥ CU / N % vec_width == 0
  → 每个合法配置用 roofline 模型算：预测地板 / bound / 预测 occupancy
  → 按"预测 remaining"排序输出
```

**验收**：
```bash
python3 -I scripts/design_space.py --card gemm-bf16 --shape prefill-8k --json | \
  python3 -I -c "import json,sys; d=json.load(sys.stdin); assert len(d['configs'])>=20"
```
判据：**≥20 个合法配置；每个带预测地板与 bound；非法配置要给出被拒理由**。

---

### C4 · tile 级模型 ×3（2 天）

现在只有 `sparse_mla` 一个 tile 模型。补三个族：

| 模型 | 输入 | 输出 |
|---|---|---|
| `models/matmul.py` | M,N,K + BM,BN,BK,warps,stages | 地板 / 预测 VGPR / LDS / occupancy / 每 K-step 的 MMA 数 |
| `models/attention_decode.py` | S,H,D,topk + CTA 配置 | 地板 / wave 数 / L2 复用率 |
| `models/reduce_lastdim.py` | M,H + BM,warps,vec | 地板 / 归约 shuffle 深度 / 行内冲突 |

**验收（关键，防"模板错但翻转正确"）**：
```bash
python3 -I scripts/model_golden.py --model matmul --against impl-ranking
```
判据：**对已有实测数据做回测，预测 vs 实测误差 < 30%**；且**同一模型改一个参数，预测值必须单调/可解释地变化**（不许是常数）。

---

### C5 · `kernel_scaffold.py` 三条硬判据（1.5 天，已在建）

**不满足这三条就是摆设**：

1. 生成的骨架**能编译**（triton 或 hipcc 至少一条）
2. 骨架参数**能被 `isa_probe` 读回**（VGPR/LDS/指令数）
3. **参数化**：BM/BN/BK/warps/stages 是变量，不是硬编码

**验收**：
```bash
python3 -I scripts/kernel_scaffold.py --card X --lang triton --params BM=128,BN=128,BK=64 --out /tmp/a.py
python3 -I scripts/kernel_scaffold.py --card X --lang triton --params BM=64,BN=128,BK=64  --out /tmp/b.py
python3 -I scripts/isa_probe.py /tmp/a.py --build --json   # 与 b 的 VGPR/LDS 必须不同
```
判据：**4 个参数组合 → 4 份能编译的骨架 → isa_probe 读出 4 组不同的资源画像**。

---

### C6 · 真跑一轮（1 天，依赖 C2 + C5）

`advance.py` 至今只 dry-run。**必须真跑**：真实候选 → 真实编译 → 真实 bench → 真出一条 ledger。

**验收（v5.1 §6.2 原判据 + 溯源）**：
```bash
python3 -I scripts/advance.py --card <有壳的卡> --max-rounds 3
python3 -I scripts/ledger.py --rounds | python3 -I -c "
import json,sys; r=json.load(sys.stdin)
assert len(r)>=2
assert r[1]['rejected'] cites r[0]'s verdict"
```
判据：**无人干预 ≥2 轮；第 2 轮 top3 ≠ 第 1 轮；且 `rejected` 引用第 1 轮的 verdict 与数字。**

---

## 第三章 · 依赖与时序

```
C1 transform ──┐
               ├─→ C2 patch ──┐
C5 scaffold ───┘              ├─→ C6 真跑 ─→ 目标验收
C3 design_space ─→ C4 tile 模型 ┘
```

| 周 | 内容 | 累计 |
|---|---|---|
| **D1** | C1（半天）+ C3 起 | 最高杠杆先做 |
| **D2** | C3 完成 + C4 第一个模型（matmul） | design space 能用 |
| **D3** | C2 patch + variant_forge | "改一个参数"能跑 |
| **D4** | C4 剩余两模型 + 回测 <30% | tile 分析成立 |
| **D5** | C5 scaffold 三条判据 | 能生成 kernel |
| **D6** | C6 真跑两轮 + 修 | 首个自主迭代实证 |
| **D7** | 目标验收（没见过的算子 ≥85% 地板） | **能力成立** |

**关键路径**：C1 → C2 → C6。C3↔C4 与 C5 可并行。

---

## 第四章 · 评分预估（诚实，带算式）

### 4.1 深度分公式

```
D = 10 × Σ(w_i × ratio_i)
  D1 DS 均分 / 45          w=0.30
  D2 ≥70 卡数 / 5          w=0.25
  D3 comparison 覆盖 / 0.85 w=0.20
  D4 ≥3 轮卡数 / 8         w=0.15
  D5 PMC/归因证据卡数 / 10  w=0.10
```

### 4.2 三档场景

**场景 A · 只做补强（C1–C5，不加一张新证据）**

| 项 | 现值 | 补强后 | 说明 |
|---|---|---|---|
| D1 | 13.5 | 13.5 | 补强不产生 measured 数据 |
| D2 | 1 | 1 | 同上 |
| D3 | 0.222 | 0.222 | 同上 |
| D4 | 1 | 1 | 同上 |
| D5 | 8 | 11 | ISA 证据让 3 张卡能写归因 |

```
D = 0.3×0.30 + 0.25×0.20 + 0.2×0.261 + 0.15×0.125 + 0.1×1.0 = 0.311
⇒ **深度 3.1（几乎不动）**
```

> **这是最重要的一句：补强本身不涨分。** 分数是证据驱动的，补强只是让证据"能涨"的前提。
> 谁把补强当成绩，就会重演"基础设施完备、实验为零"。

**场景 B · 补强 + C6 真跑 3 张有壳的卡 ×3 轮**

假设 3 张卡 DS 8→65，5 张卡 DS 8→40：

| 项 | 现值 | 补强后 | 算式 |
|---|---|---|---|
| D1 | 13.5 | **20.2** | (715.5 + 3×57 + 5×32) / 53 |
| D2 | 1 | **4** | 4/5 = 0.8 |
| D3 | 0.222 | **0.407** | 11/27 |
| D4 | 1 | **4** | 4/8 = 0.5 |
| D5 | 8 | **12** | 12/10 → 1.0 |

```
D = 0.3×0.449 + 0.25×0.8 + 0.2×0.479 + 0.15×0.5 + 0.1×1.0
  = 0.135 + 0.200 + 0.096 + 0.075 + 0.100 = 0.606
⇒ **深度 6.1**
```

**场景 C · 场景 B + 覆盖与质量债一起清**

| 维度 | 现在 | 预估 |
|---|---|---|
| 深度 | 2.9 | **6.1 – 6.7** |
| 覆盖 | 6.5 | **7.8 – 8.2**（C2 0.64→0.80 / C3 0.741→0.93 / C4 4→6 / C5 2→4）|
| 质量 | 6.7 | **8.0 – 8.5**（Q2 17→6 / Q4 101→70 / Q3 4→6 / Q5 257→200）|

### 4.3 为什么深度到不了 9

```
D1（DS 均分）是结构性瓶颈：
  53 张卡里 26 张 draft（DS 0–15），均值被死死压低。
  即使 10 张卡做到 DS 70，均值也只有 13.5 + 10×62/53 ≈ 25.2 → ratio 0.56
  D 上限 ≈ 0.3×0.56 + 0.25×1 + 0.2×1 + 0.15×1 + 0.1×1 = 0.868 → **8.7**
  但要 10 张卡都做到 DS 70，需要 10×3 轮真实实验 —— 那是 3–4 周的工作量
```

**所以深度 6 是"能力成立"的分数，8+ 是"覆盖也做满"的分数，9 需要把 26 张 draft 卡也做完。**

---

## 第五章 · 每包的验收命令（全实跑，贴 rc）

```bash
# C1
python3 -I scripts/check_transform.py              # 17/17 有 transform
python3 -I scripts/check_transform.py --legality   # legality 可解析

# C2
python3 -I scripts/variant_forge.py --card X --cand bm64 --build
python3 -I scripts/isa_probe.py <生成的变体> --json | jq '.vgpr,.lds'   # 与 parent 不同

# C3
python3 -I scripts/design_space.py --card gemm-bf16 --shape prefill-8k --json | jq '.configs|length'   # ≥20

# C4
python3 -I scripts/model_golden.py --model matmul --against impl-ranking   # 误差 <30%

# C5
python3 -I scripts/kernel_scaffold.py --card X --lang triton --params BM=128,BN=128,BK=64 --out /tmp/a.py
python3 -I scripts/isa_probe.py /tmp/a.py --build --json

# C6
python3 -I scripts/advance.py --card <有壳的卡> --max-rounds 3
```

---

## 第六章 · 不做什么

1. **不再加门禁**（E39–E43 + S5 四件套已够）
2. **不做 L3 算法创新**（重写数学超出范围）
3. **不新建评分维度**（三维卡够用，加维度会稀释）
4. **不为了分数刷字段**（补强 ≠ 刷 DS）
5. **不在没有 C6 实证的情况下宣布"能力成立"**

---

## 第七章 · 一句话

> **v5.1 让数字可信，v5.2 让判据能跑，v5.3 要补上最后一道门：从"知道该改哪"到"知道怎么改、改成多少、改完能验证"。**
>
> 三个洞：**surface 没有 transform（怎么改）· candidates 没有 patch（改谁）· 没有 design space（改到什么值）**。
>
> **补强本身不涨分（场景 A：3.1）**——它只是让深度**能**涨。真正涨分要 C6 真跑：**深度 6.1**。
>
> **验收只有一条：给它一个没见过的算子，看它能不能自己把地板吃满 85%。**
> 在那之前，"深度 2.9"是准确的——**深度不是"流程多深"，是"能不能自己做出一个加速"**。
