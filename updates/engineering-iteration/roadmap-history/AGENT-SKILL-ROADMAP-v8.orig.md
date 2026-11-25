# Agent Skill 路线图 v8：从「精度」转向「广度 + 自演化」

**日期**：2026-11-25（v7.2 → v8）
**业主定性**：深度目标 = agent 拥有自主迭代研发算子的能力
**v8 的一句话**：**v5–v7 解决的是"单个算子能多准"，ForgeKernel 证明成绩来自"多少算子能爬上去"。v8 转向广度。**

---

## 第零章 · 现状（16:12 实测）

### 0.1 三项分数

| 维度 | 现在 | 目标 |
|---|---|---|
| **深度** | **4.3**（D1 16.5 / D2 3 / D3 0.296 / D4 1 / D5 8）| 9.0 |
| **覆盖** | **7.0**（C1 0.889 / C2 0.64 / C3 0.815 / C4 4 / C5 4）| 9.0 |
| **质量** | **3.8**（Q1=1 个测试失败 / Q4 119 / Q5 263）| 9.0 |

### 0.2 五张卡的分布（这是问题的全部）

| DS | 卡 | B | U | S | R | M | 缺口 |
|---|---|---|---|---|---|---|---|
| **92** | `fused-add-rmsnorm` | 1.0 | 0.98 | 1.0 | 1.0 | **0.5** | 缺 PMC |
| **89** | `norm-act-swiglu` | 1.0 | 0.97 | **0.5** | 1.0 | 1.0 | 缺第 3 个形状 |
| **74** | `rmsnorm-fused` | 1.0 | 0.78 | **0.0** | 1.0 | 1.0 | 缺多形状 |
| 68 | `gemm-bf16` | **0.3** | 1.0 | 1.0 | **0.0** | 1.0 | 缺对比、缺轮次 |
| 50 | `gemm-fp8` | **0.0** | 1.0 | 0.5 | **0.0** | 1.0 | 缺对比、缺轮次 |

**其余 48 张卡都在 40 分以下，其中 42 张在 0–19。**

### 0.3 这 3 小时做了什么（诚实记录）

```
75fadb3  docs: ForgeKernel 可吸收清单 + 八维度评分
ba60062  docs: ForgeKernel 深度对比 —— 4 样可借鉴，2 样是结构性差距
949917a  feat(严谨入口): CLAUDE.md + 插件描述 + 契约测试
```

**三份都是文档。分数一分没动。**（但内容质量不错，见 §四）

---

## 第一章 · ForgeKernel 带来的核心认知

已把 `git@github.com:OpenBMB/ForgeKernel.git` 拉到 `/workspace/ForgeKernel`（17 MB）。

### 1.1 事实对照

| | ForgeKernel | 我们 |
|---|---|---|
| 代码 | **134,943 行** py + 47,078 行 doc | ~5,700 行 + 1,368 测试 |
| 硬件知识 | **hw_kb 200 条**（每条 GPU probe 认证 + SASS 分析）| 3 份 reference |
| 已完成算子 | **27 个** | **3 / 53 张**（5.7%）|
| 成绩 | 胜 **18/22** 专家库，geomean **+39.4%**，最高 **+170.6%** | 3 张卡胜厂商，最高 2.09x |
| 榜单 | **#1** SOL-ExecBench / CANN-Bench / Ascend-950DT | opbench 内部 |
| 平台 | NVIDIA H100/B200（sm90/sm100）| DCU/ROCm gfx938 |

### 1.2 差距不在代码量，在两种假设

| | 假设 | 工程化的对象 | 产物 |
|---|---|---|---|
| **ForgeKernel** | **agent 能力弱** | **路径**：把优化切成"低能力 agent 也能爬"的课程 | 27 个算子 |
| **我们** | **agent 能力强** | **诊断**：roofline / 口径 / 反作弊做得更准 | 3 个 92 分样板 |

**它的 `meta_milestone.md` 把这一点说穿了**：

> milestone 不是"把算子代码写成散文"，是把"构建高性能算子"分解成一条**平滑课程**：
> 一串按依赖排序的、每个都局部简单、让**低能力 agent 能一步一步爬上去**的决策。

**我们的 17 条 surface + 11 个模板是「知识」，不是「课程」**——它告诉你该往哪治，
不告诉你**先做哪个、后做哪个、做到什么算过关**。

### 1.3 它的三个机制，我们一个都没有

#### 机制 1 · 自演化：每次获胜的 kernel 升格成下一次的参考

```
gen0  vs FlashAttention-4          +0.3%
gen1  vs ForgeKernel gen0(sealed)  +9.9%   ← 累计 +10%
gen2  vs ForgeKernel gen1(sealed)  +4.2%   ← 累计 +15%
from_agent vs /goal agent(sealed)  +5.0%   ← 累计 +29%
```

**停止条件写在代码里**：增益落在"这一代 own same-job 确认比值的散度"内（且不低于
`NOISE_FLOOR_MIN`），或这一代没有确认的最优。

> **我们的卡是一次性的**：做到 92 分就归档，**从来没有"拿自己的最优当新参考再打一轮"**。

#### 机制 2 · 架构优先课程：M0 定结构并冻结，M1..Mk 只许调优

M0 把**最容易搞错、改起来最贵**的决策全部定下（kernel 切分、tile 形状、warp 角色、
寄存器/shared 预算、什么常驻、发射顺序），**并冻结**。M1→Mk 只在这个结构内调优，
每关由**正确性 + 性能 + 机制**三条线验收。证明只有新结构能突破时，arch agent **回头改 M0**。

> 我们的 `transform` 是**平铺列表**，没有依赖顺序、没有"先定什么后调什么"。

#### 机制 3 · hw_kb：200 条带 probe 证书的硬件机制

```
每条格式：Certification source + Status(✅ GPU-certified / ⬜ unverified)
          + PROBE PASS max_diff=... + 认证范围 + 未认证项 + Mechanism + Sources
算子无关 → 所有算子复用
按 language × arch 分树（cutedsl/sm100 与 cuda/sm100 是两棵树）
```

> 我们 `isa-census.md` 是**一份文件**。它是 **200 份，每份带证书**。

---

## 第二章 · v8 的转向

### 2.1 一句话

> **v5.1 让数字可信 → v6 让 agent 能改 → v7 想让 agent 证明因果 → v8 承认：
> 我们缺的不是因果，是「把这个流程重复 10 遍」的机制。**

### 2.2 为什么转向

| 证据 | 说明 |
|---|---|
| 3 张卡 89–92 分，其余 50 张 ≤68 | **峰值已证，广度未证** |
| ForgeKernel 27 个算子 / +39.4% | **成绩来自广度**，不是单个算子多快 |
| 它靠 reference promotion 从 +0.3% 滚到 +29% | **自演化是复利，我们是单利** |
| 我们 3 小时产出了 3 份文档，分数 0 变化 | **文档不产生卡片数据** |

### 2.3 精算：9 分需要什么

```
D = 10 × Σ(w × ratio)
  D1 DS 均分/45 × 0.30   D2 ≥70卡/5 × 0.25   D3 comparison/0.85 × 0.20
  D4 ≥3轮/8 × 0.15       D5 PMC/10 × 0.10
```

**D2/D3/D4/D5 全拉满 = 0.70 → D1 只需 ratio 0.667 → DS 均值只要 30（不是 45）**

| 检查点 | 现在 | 要 9 分 | 差距 |
|---|---|---|---|
| D1 DS 均值 | 16.5 | **30** | +13.5 |
| D2 ≥70 卡 | 3 | **5** | +2 |
| **D3 comparison** | **0.296（8/27）** | **1.00** | **+19** ⚠️ 最硬 |
| D4 ≥3 轮 | 1 | **8** | +7 |
| D5 PMC | 8 | **10** | +2 |

**「10 张卡做满」能同时喂饱 D1/D2/D4/D5。只有 D3 要额外补覆盖率。**

---

## 第三章 · v8 的四个包

### E1 · 自演化（半天）⭐ **最高 ROI**

**这是 v8 的第一件事，也是最便宜的一件事。**

现状：我们有 3 张 89–92 分的卡，**从没试过把它们当基线再打一轮**。

```bash
# 目标接口
python3 -I scripts/self_evolve.py --card fused-add-rmsnorm --step 1.05 --max-gen 2
```

**做什么**：
1. 把当前最优 kernel 复制成 `reference/sealed/`（只读 + sha256）
2. 生成新任务：同一个 shape、新的目标是 `当前 × (1/1.05)`
3. `advance` 按新目标再跑一轮
4. **停止条件写在代码里**：
   - 增益 ≤ 噪声底（本代 own same-job 确认比值的散度，且不低于 `NOISE_FLOOR_MIN`）
   - 或这一代没有确认的最优
5. 每一代的结果落 `gen_k` 记录，链路可追

**验收（四条全过）**：
```bash
# 1) 封存有效
ls <card>/reference/sealed/*.so && grep -c sha256 <card>/reference/sealed/MANIFEST
# 2) 新目标是量化的
python3 -I -c "import json;d=json.load(open('<card>/gen1/target.json'));assert 0<d['target_ms']<d['base_ms']"
# 3) 至少跑出 1 代
ls <card>/gen1/record.json
# 4) 链条能停（不许无限跑）
grep -c "noise_floor\|NOISE_FLOOR_MIN" scripts/self_evolve.py
```

**预期收益**：如果 gen1 像 ForgeKernel 那样再涨 4–10%，
`fused-add-rmsnorm` 的 ratio 会从 0.549 再降 → DS 的 U 项和 comparison 都更好看，
**但 D1 只涨 0.1 左右**。**E1 的价值不在分数，在证明"我们能滚起来"。**

---

### E2 · 课程化：把 17 条 surface 排成 M0/M1..Mk（1 天）

**现状**：`surfaces.yaml` 17 条是**平铺的**，每条有 `transform`（怎么改）、
`precondition_check`（能不能做）、`expected_gain`（上界）。**但没有依赖顺序。**

**改法**：加一个 `curriculum.yaml`，把 surface 分成三层：

```yaml
# M0 架构层：定了就冻结（改起来最贵）
M0:
  - occupancy          # tile / warp / 驻留预算
  - layout-pack        # 数据布局（决定后续所有访存形态）
  - fusion-boundary    # kernel 切分
  - split-k            # 并行切分
gates: [正确性, 达到预测地板的 X%, 机制行（ISA/PMC 证据一行）]

# M1..Mk 调优层：只在 M0 结构内调
M1: [vmem-vector-width, mls-path, lds-read-form]
M2: [prefetch-depth, barrier-width, cross-lane-fold, epilogue-expansion]
M3: [precision-drop, table-miss, grid-tail]
```

**验收**：
```bash
python3 -I scripts/check_curriculum.py
# → 每个 surface 有且只有一个层；M0 的每条都有 gate；M1+ 不许改 M0 的决策
```

**为什么值**：现在 agent 拿到 17 条不知道该先做哪个。ForgeKernel 的 27 个算子
证明**顺序比选项更重要**。

---

### E3 · hw_kb：把已有实测沉淀成带证书的条目（持续）

**不新建知识**——我们已经有（散在 references 里）：

| 已有实测 | 该沉淀成 |
|---|---|
| 寄存器驻留 vs LDS 中转（三张卡的根因）| `hw_kb/hip/gfx938/reg_resident_vs_lds.md` |
| 16B 向量访存（fused-add-rmsnorm v5 的 5-7%）| `hw_kb/hip/gfx938/vec16_uint4.md` |
| `VEC` 运行期化 → private 溢出 → **慢 3.3×** | `hw_kb/hip/gfx938/private_spill_trap.md` |
| 写回只有 dwordx2（ISA 实测）| `hw_kb/hip/gfx938/store_width.md` |
| `ds_read_matrix_trans_format` = MLS 的真名（ISA）| `hw_kb/hip/gfx938/mls_mnemonic.md` |
| `s_waitcnt` 深度由 scheduler 决定（死路）| `hw_kb/hip/gfx938/waitcnt_scheduler.md` |
| `VEC/THREADS` 必须编译期固化 | `hw_kb/hip/gfx938/compile_time_const.md` |

**每条格式（照 ForgeKernel）**：
```markdown
# reg_resident_vs_lds
> **Certification**: ✅ GPU-certified  **Probe**: A/B 同批，event_isolated，5 轮
> 认证范围：M=8192 H=4096 bf16
> ⬜ 未认证：其他 H / fp8

## Mechanism（算子无关，只讲机制）
## Sources（实测读数 + reproduce 命令）
```

**验收**：`hw_kb/hip/gfx938/` 下 **≥7 条**，每条带 `✅/⬜` 状态 + probe 实测 + reproduce。

---

### E4 · 十张满卡（2 天）⭐ **唯一能真正推分的事**

**配方（已验证 3 次）**：
```
竞品对比（event_isolated，ratio<0.95，significant）
+ utilization kind 的 ratio
+ ≥2 形状 geomean
+ ledger.md ≥3 轮且每轮有决策词
+ PMC 或 ISA 归因证据
```

**候选（按现有分数从高到低，先补缺口最小的）**：

| 优先级 | 卡 | 现有 DS | 缺什么 | 成本 |
|---|---|---|---|---|
| 1 | `fused-add-rmsnorm` | 92 | 只缺 PMC（M 0.5→1.0）| **最低** |
| 2 | `norm-act-swiglu` | 89 | 只缺第 3 形状（S 0.5→1.0）| 低 |
| 3 | `rmsnorm-fused` | 74 | 缺多形状（S 0→1.0）| 低 |
| 4 | `gemm-bf16` | 68 | 缺对比 + 轮次 | 中 |
| 5 | `gemm-fp8` | 50 | 缺对比 + 轮次 + 形状 | 中 |
| 6–10 | `mha-prefill` / `mla-absorb` / `moe-w8a8` / `fusion-rmsnorm-rope` / `fusion-add-rmsnorm-quant` | ≤40 | 全套 | 中 |

**验收（每张都按这四条）**：
```bash
# a) DS ≥ 90
python3 -I scripts/depth.py --json | ... assert DS>=90
# b) 无 0 项
assert all(v>0 for v in (B,U,S,R,M))
# c) validate 0 错误
python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas
# d) pytest 0 failed
python3 -m pytest Plugins/ops -q
```

**同时要做的（D3 的 19 张）**：给其余 measured 卡补竞品对比。
**这一项不能用"更快"解决，只能一张一张补。**

---

## 第四章 · 和已写的两份 ForgeKernel 文档的关系

`docs/FORGEKERNEL-ABSORB.md`（230 行）和 `docs/FORGEKERNEL-VS-US.md`（249 行）
**已经做得很好**，识别出：

| 它的结论 | 我的判断 |
|---|---|
| A1 hw_kb 最该吸收 | ✅ **同意** = v8 的 E3 |
| A2 reference 独立成文件 | ✅ 同意，是 E1 的前置 |
| A3 ablate 双向 | ✅ 已有 `ablate.py`，补"单独加"方向 |
| A4 promote 机制（自演化）| ✅ **同意 = E1，v8 排第一** |
| B1 architecture-first | ✅ 同意 = E2 |
| B2 结构化 milestone 课程 | ✅ 同意 = E2 |
| B3 teacher/student 双 agent | 🟡 成本高，v8 不做（见 §六）|
| 我们更严的 3 样（measure_mode / 三层分母 / anticheat）| ✅ **同意，别照搬它的** |
| 明确不借的 | ✅ 同意 |

**v8 和这两份文档的唯一区别**：**它们回答"能借什么"，v8 回答"做完能涨几分、按什么顺序"。**

---

## 第五章 · 评分预估

### 5.1 逐包

| 做完 | D1 | D2 | D3 | D4 | D5 | 深度 | 覆盖 | 质量 |
|---|---|---|---|---|---|---|---|---|
| 现在 | 16.5 | 3 | 0.296 | 1 | 8 | **4.3** | 7.0 | 3.8 |
| E1 自演化 | 16.7 | 3 | 0.296 | 1 | 8 | **4.3** | 7.0 | 3.8 |
| E2 课程化 | 16.7 | 3 | 0.296 | 1 | 8 | **4.3** | 7.0 | 3.8 |
| E3 hw_kb | 16.7 | 3 | 0.296 | 1 | **10** | **4.5** | 7.0 | 3.8 |
| **E4 十张满卡** | **30** | **5** | **~0.6** | **8** | **10** | **≈7.4** | 7.0 | 3.8 |
| + D3 补齐 19 张 | 30 | 5 | **0.85** | 8 | 10 | **≈9.0** | 7.0 | 3.8 |

### 5.2 诚实的期望管理

```
E1/E2/E3 是机制，分数几乎不动（4.3 → 4.5）
E4 是唯一推分的（4.5 → 7.4）
D3 补齐是 9 分的最后一公里（7.4 → 9.0）
```

**这就是为什么 v8 把 E4 排第三而不是第五：机制再好，不做满卡就不涨分。**

---

## 第六章 · 不做什么

1. **不借 ForgeKernel 的代码**（134,943 行，我们 1/24 的代码量已做到 2.09x）
2. **不做 teacher/student 双 agent**（B3）—— 成本高，且我们的 reference 已经可信
3. **不加第 12 个工具**（现有 9 个 + ablate/check_means/scaffold/forge_step 够了）
4. **不再写对比文档**（已写 479 行，够了）
5. **不拆 engine/loop.py**（1,955 行，0 次故障）
6. **不为分数降级回退**（D3 降 draft 导致的覆盖 −0.3 已确认接受）

---

## 第七章 · 执行顺序（不许跳）

```
第 1 天上午  E1 自演化（半天，fused-add-rmsnorm 打 gen1）
第 1 天下午  E3 hw_kb 7 条（用已有实测，不新跑）
第 2 天      E2 课程化 curriculum.yaml + check
第 3-4 天    E4 十张满卡（从缺口最小的 92 分开补）
第 5 天      D3 补齐 19 张 comparison
第 6 天      跑 scorecard --append，看是否到 9.0
```

**每天的硬约束**：**如果没有新增卡片数据，这一天算没做事。**
（写文档、加工具、改口径都不算——这是 v8 唯一的纪律。）

---

## 第八章 · 一句话

> **v5–v7 我们把「诊断」做到了 ForgeKernel 都没有的精度（三层分母、measure_mode 可 claim、反作弊反向测试）；但它用 27 个算子证明：成绩来自「多少算子能爬上去」，不是「单个算子多准」。**
>
> **v8 四件事：E1 自演化（半天，把 92 分的卡封存成新参考再打一轮）→ E2 课程化（把 17 条 surface 排成 M0 冻结 + M1..Mk）→ E3 hw_kb（把已有 7 条实测沉淀成带证书的条目）→ E4 十张满卡（唯一推分的事）。**
>
> **精算过：十张满卡 → 深度 7.4；再把 19 张 comparison 补齐 → 9.0。**
>
> **最便宜的那件事只要半天：把 92 分那张卡封存，再打一代。我们做过的所有优化都是一次性的——这可能是全部差距里最便宜的那个。**
