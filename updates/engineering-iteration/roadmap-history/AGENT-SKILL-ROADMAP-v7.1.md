# Agent Skill 路线图 v7.1：第一个真加速出来了，但三关没过

**日期**：2026-11-24（v7 → v7.1）
**业主定性**：深度目标 = agent 拥有自主迭代研发算子的能力
**v7.1 相对 v7**：v7 写于 16:13，当时还没有真加速。16:18 出了第一个，本版重写。

---

## 第零章 · 16:18 发生了什么：`c5c2654` 自研 HIP 超过厂商

### 0.1 事实

```
rms-v2-reg（自研 HIP，93 行）vs lightop.fused_add_rms_norm（厂商）

  base  202.24 µs   1327 GB/s
  cand  184.64 µs   1454 GB/s
  Δ     −8.70%
  |Δ| = 17.6 µs  >  3×MAD = 1.92 µs   → 显著
  measure_mode = event_isolated       → 可 claim 口径
```

**这是整个 arc 的第一个真实加速。** 之前的 11 条记录全是 `REJECTED_NO_GAIN` / `INCONCLUSIVE`。

### 0.2 优化思路是站得住的

```
rms-v2-reg：把 (x+r) 留在寄存器（16 float/线程），
            LDS 只放 WARPS*4 字节的归约表
vs lightop：走 LDS 中转
```

**而且它低于模型预测的 floor**（`t_floor=202.3 µs`）：

> 本实现 184.6 µs 低于 floor.py 的 202.3 µs —— **不是突破物理，是归约折扣 0.79 偏保守**。
> 实测该折扣应为 159.8/184.6 = 0.866。

**这是第一条可校准记录**：模型被判据打脸，而判据该改。这正是"账本比实验重要"的正面案例。

### 0.3 还附带一个真教训

> `VEC/THREADS` 必须编译期固化。运行期传 `VEC` 会让 `xv[]/rv[]` 退化成私有内存
> （private 0→528B，0.165→0.538ms，**慢 3.3×**）。这是 `inspect_so` 读出来的。

**ISA 工具第一次直接产生了一个优化决策。** S1 那 1 天修文档的投入开始回本。

---

## 第一章 · 但三关没过

### 关 1 · 没落账 ⭐ 最要命

```
$ grep -c "rms-v2-reg" Plugins/ops/skills/op-rmsnorm-fused/runs/loop/*.jsonl
0
```

**结果只在 `results/forged-vs-lightop.json` 里，没进 ledger。**

卡片也还是旧数据：

```yaml
last_verdict.comparison:
  baseline: lightop-fused
  cand: fused-3pass-tilelang
  cand_us: 213.0        ← 旧的
  base_us: 197.4        ← 旧的
  ratio: 1.079          ← 我们还慢 7.9%
```

**所以：真实成绩 184.6 vs 202.2（快 8.7%）→ 卡上记的是 213.0 vs 197.4（慢 7.9%）。**
**方向都是反的。**

> 这正是 v7 §0.3 写的那条："11 条记录全在一张卡上，`memory/ledger.jsonl` 0 字节"。
> **数字不落账，就等于没发生。**

### 关 2 · 精度门是错的（详见第二章）

```
$ opcheck correctness --variants lightop-fused,fused-3pass-triton,fused-3pass-tilelang,rms-v2-reg

FAIL lightop-fused          max_abs=3.125e-02 (限 2.0e-03)  ← 厂商也 FAIL
FAIL fused-3pass-triton     max_abs=1.562e-02 (限 2.0e-03)
FAIL fused-3pass-tilelang   max_abs=1.562e-02 (限 2.0e-03)
FAIL rms-v2-reg             max_abs=1.562e-02 (限 2.0e-03)  ← 新的也 FAIL
```

**四个全 FAIL，包括厂商。** 而 `cosine=1.0000000`、`mean_abs=3.5e-8`。

### 关 3 · 归因还没做

知道"改完变快了"（L2 相关），不知道"是寄存器驻留让它快的"（L3 因果）。
**v7 原本要做的 `ablate.py` 还没建**，`grep -c "ablat" scripts/*.py` 仍为 0。

---

## 第二章 · 精度门为什么是错的：bf16 的 ULP 比容差还大

### 2.1 算术

`contract.yaml` 写的容差：

```yaml
tolerance:
  vs_fp32_ref:
    max_abs: 2.0e-3
    note: "bf16 输入 + fp32 累加；eps 位置不同会带来可见差异"
```

**bf16 的 ULP（最小可分辨步长）**：

| 数值区间 | bf16 ULP | `2.0e-3` 容得下几 ULP |
|---|---|---|
| [0.5, 1) | 0.0039 | **0.51** ← 不到 1 ULP |
| [1, 2) | 0.0078 | **0.26** |
| [2, 4) | **0.0156** | **0.13** |

**而实测 `max_abs = 1.5625e-2 = 2⁻⁶`，正好是 [2,4) 区间的 1 个 ULP。**

> **结论：这个容差在 bf16 输出上数学上不可能满足——它比 1 个 ULP 还小。**
>
> 实测算出的 `1.5625e-2` **是完美正确的 bf16 舍入，不是错误**。

### 2.2 这是全项目第三例「判据本身错」

| # | 判据错在哪 | 表现为 | 修正 |
|---|---|---|---|
| 1 | `isa-census` 助记符写成 `ds_read_m32x32_b8` | grep 得 0 → 误判"没走 MLS" | 改为 `ds_read_matrix_trans_format` |
| 2 | `ledger.md` 的 lightop 227.0 µs | 结论"慢 6.7%"实为"慢 20%" | 重测，227.0 是异常值 |
| **3** | **`contract.tolerance.max_abs = 2.0e-3`** | **4 个变体全 FAIL，含厂商** | **见下** |

**共同点：把"实现有问题"和"判据有问题"混为一谈，而默认怀疑实现。**

### 2.3 修法（三个选项，选 2 + 3）

| # | 方案 | 评价 |
|---|---|---|
| 1 | 放宽 `max_abs` 到 2 ULP（如 3.2e-2） | 简单，但"2 ULP"仍依赖数据分布，不通用 |
| **2** | **按数据范围算容差**：`max_abs_tol = k × ULP(max_abs_value)`，k=2 | ✅ 推荐。自解释、可复算 |
| **3** | **加一条 `ulp_check`**：`max_abs / ULP(max_abs_value) ≤ 2` | ✅ 推荐。直接报"偏了几 ULP"，比绝对值有信息量 |

**新判据**：

```yaml
tolerance:
  vs_fp32_ref:
    mode: ulp_bound
    max_ulp: 2
    note: |
      bf16 输出最小可分辨步长随数值量级变化（[2,4) 区间 ULP=2^-6）。
      绝对容差 2.0e-3 在 bf16 上不可能满足——它小于 1 ULP。
      改用 ULP 判据：max_abs / ULP(max_abs 处的值) ≤ max_ulp。
      2026-11-24 实测四个变体（含厂商）均为 1 ULP，全部应 PASS。
```

**这个修改会同时修好四张卡**（`rmsnorm-fused` / `fused-add-rmsnorm` / `fusion-add-rmsnorm-quant` / `fusion-rmsnorm-rope`）——它们的容差是同一来源抄的。

---

## 第三章 · 为什么深度分还是 2.9（结构阻尼）

### 3.1 你的直觉对：这个成绩"理应"更高

但实际上即使完美落账，深度也只动 0.0–0.4。原因是**结构性的**：

```
rmsnorm-fused 当前 DS = 74
  B(comparison)  1.0 × 25 = 25   ✅ 已有
  U(utilization) 0.78 × 25 = 19.5  ← 新成绩 1454/1680 = 0.866，可涨到 21.7
  S(shapes)      0.0 × 20 = 0    ❌ 只测了 prefill-8k 一个形状
  R(rounds)      1.0 × 15 = 15   ✅ 已有
  M(mechanism)   1.0 × 15 = 15   ✅ 已有
                 合计 = 74
```

**最好情况**（U 涨满 + 补齐多形状 S）：DS 74 → **96**。

```
D1 = (715.5 + 22) / 53 = 13.92   ← 均值只从 13.5 涨到 13.9
D = 0.3×(13.92/45) + 0.25×0.2 + 0.2×0.261 + 0.15×0.125 + 0.1×0.8
  = 0.093 + 0.050 + 0.052 + 0.019 + 0.080 = 0.294
⇒ 深度 2.9（四舍五入后不变）
```

### 3.2 根因：D1 是 53 张卡的均值

```
26 张 draft 卡（DS 0–15）  +  22 张 measured 卡低于 40  +  1 张 74
```

**一张卡做到满分，只给均值贡献 +0.42。**

> **这不是度量错了，是度量"测的是覆盖面"而不是"测的是最高水平"。**

### 3.3 建议：加一个「单卡记录」维度，而不是改 D1

**不改现有三维**（改了历史不可比），**加一个并列展示的里程碑指标**：

```yaml
# 不进总分，只展示。因为它衡量的是"天花板"，不是"平均水位"
best_card:
  card: rmsnorm-fused
  ds: 74                    # 当前
  ds_after_fix: 96          # 落账 + 补 S 后
  beat_vendor: true
  vendor_gap_pct: -8.70
  measure_mode: event_isolated
  reproducible: results/forged-vs-lightop.json
```

**理由**：`beat_vendor` 这类里程碑，用"均值"表达不出来；用"最佳单卡 + 是否超厂商 + 口径"能精确表达。

**这也是 v7.1 唯一建议动的评分相关内容**——不动三维，加一个展示项。

---

## 第四章 · 最小化借鉴 SkillHub：只借三件

对比过 `das/SkillHub`（44 skill，22 MB）的算子优化模块。**结论：它的管控比我们松，只有三件值得借。**

### 4.1 先明确两边强项（免得借错）

| 管控层 | SkillHub | 我们 |
|---|---|---|
| **L1 流程闸门**（exit code 串联 9 步）| ✅ **更密** | 🟡 6 道门，缺文档闸门 |
| **L2 动作拦截**（动手前拦）| ❌ **0 个 hook** | ✅ **PreToolUse hook，fail-closed** |
| **L3 证据可信** | ❌ 基本没有 | ✅ sha256 + measure_mode + anticheat + 三层分母 |

**它的管控停在"脚本里"；agent 只要绕过脚本就自由。我们的门禁在"动作层"。**

### 4.2 借这 3 件（其余全部不借）

#### 借 1 · `analysis.md` 前置闸门（半天）⭐ 唯一真缺的

**它怎么做的**：
```python
# close-iter 前强制检查
if analysis_check.returncode != 0:
    print({"status": "analysis_invalid",
           "guidance": "Write a non-placeholder ## Kernel Computation Logic section ..."})
    sys.exit(3)
```

**为什么该借**：我们现在的顺序是"先跑再看"——只要有 `expect` 就能跑实验。
**先写清推理再跑，才是研发；先跑再看是搜索。**

**怎么落地（最小版）**：
```yaml
# candidates.yaml 每个 pending 候选多一个必填字段
means: |
  通过把 (x+r) 留在寄存器、LDS 只放 WARPS*4 字节归约表，
  消除 LDS 中转的额外往返。
predicted_effect: -6%~-10%
expected_bottleneck: memory        # 断言"我预期卡在哪"
```

`loop.py run` 前置校验：**三个字段缺一个 → 拒跑（rc=2）**。

**判据**：故意删掉 `means` 字段 → `run` 必须 rc=2。

#### 借 2 · `ablate.py` 的消融思路（1 天）

**它怎么做的（209 行）**：
```python
"""attribution(m) = ms_ablated(m) - ms_champion
   Positive means the method helped."""
```

**为什么该借**：我们只能证明"改完变快了"（L3 缺），`attribution.share` 靠人估。

**只借思路，不借代码**——我们的 `variant_forge.py`（838 行）已经能生成变体，只需要加一个 `--ablate` 参数：从冠军版本生成"去掉第 m 项"的变体。

**判据**：N 项消融的 `attribution_share` 之和 ∈ [0.8, 1.2]（能解释全部收益）。

#### 借 3 · `branch_explore.py` 的批量择优（1 天）

**它怎么做的（428 行）**：K 支并行编译 + 择优 + frontier 复用。

**为什么该借**：`design_space.py` 能枚举 **486 个配置**，但 `advance.py` **一次只跑一个**。
**枚举了不算，跑了才算。**

**判据**：K=4 时整轮墙钟 ≤ 4 × 单次 bench，frontier 里 3 条可复用。

### 4.3 明确**不借**（会退化）

| 不借 | 原因 |
|---|---|
| ❌ `orchestrate.py`（881 行流程引擎）| 与 `advance.py` 造成两套编排 |
| ❌ `state.json`（614 行状态机）| 与 `progress` + `memory/` 造成第三处记账 |
| ❌ `roofline.py` 的分母（`peak_bw = 2000.0` nominal）| **那是倒退**。我们用 achievable 1680 |
| ❌ `noise_threshold_pct = 2%` 固定阈值 | 无 MAD、无自适应。我们用 `judge()` 四态 |
| ❌ 它的 `_validate_outputs`（只有 `torch.allclose`）| 无形状分离、无扰动、无缓存检测 |
| ❌ 整套 skill 直接拼进来 | 它是独立体系，硬拼会打架 |

---

## 第五章 · v7.1 的四个包（顺序即优先级）

### D0 · 修精度门 + 落账（半天）⭐ 排第一

这两件不做，`c5c2654` 的成绩等于没发生。

```bash
# ① 精度门改 ULP 判据
#    contract.yaml: mode: ulp_bound, max_ulp: 2
#    同步修 rmsnorm-fused / fused-add-rmsnorm / fusion-add-rmsnorm-quant / fusion-rmsnorm-rope 四张卡

# ② 落账三处
#    a) 写 ledger: <card>/runs/loop/*.jsonl 新增 rms-v2-reg 记录
#    b) 更新卡片 last_verdict.comparison（cand_us 184.64 / base_us 202.24 / ratio 0.913）
#    c) 写 memory/ledger.jsonl（第一次写入，打破 0 字节）
```

**判据（四条全过）**：
```bash
# 1) 精度门改完，四个变体全 PASS
python3 -I scripts/opcheck.py correctness --contract <卡>/contract.yaml --shape prefill-8k \
  --variants lightop-fused,fused-3pass-triton,fused-3pass-tilelang,rms-v2-reg
# → 4 个 PASS，max_ulp ≤ 2

# 2) ledger 有记录
grep -c "rms-v2-reg" <卡>/runs/loop/*.jsonl     # ≥1

# 3) 卡片方向正确
python3 -I -c "import yaml; d=yaml.safe_load(open('<卡>.yaml')); \
  c=d['last_verdict']['comparison']; assert c['ratio']<1.0, c"
# → ratio 0.913 < 1.0（我们更快）

# 4) memory 不再空
wc -l memory/ledger.jsonl                        # ≥1
```

### D1 · `analysis.md` 前置闸门（半天）

见 §4.2 借 1。

### D2 · `ablate.py`（1 天）

**对 `rms-v2-reg` 做消融**，这是天赐的实验对象——它的设计有三项独立改动：

```
① 寄存器驻留 (x+r)      ② LDS 只放 WARPS*4 字节归约表    ③ 16 字节向量读写
```

**拆掉任一项，测差多少 → 得到真实 attribution。**

判据：三项 `attribution_share` 之和 ∈ [0.8, 1.2]。

### D3 · `branch_explore.py`（1 天）

见 §4.2 借 3。

---

## 第六章 · 评分预估（修正版）

### 6.1 现在 vs 落账后

| 项 | 现在 | D0 落账后 | 说明 |
|---|---|---|---|
| rmsnorm-fused DS | 74 | **76** | U 0.78 → 0.866 |
| D1（均值）| 13.5 | **13.5**（13.54 四舍五入）| +1/53 |
| D2（≥70 卡）| 1 | 1 | 不变 |
| D3（comparison 覆盖）| 0.222 | 0.222 | 已在分子里 |
| D4（≥3轮）| 1 | 1 | 不变 |
| **深度** | **2.9** | **2.9** | **不动** |

### 6.2 补齐多形状后

| 项 | 值 |
|---|---|
| rmsnorm-fused DS | 74 → **96**（S 0→1.0）|
| D1 | 13.5 → **13.9** |
| **深度** | **2.9 → 3.0** |

### 6.3 D2 做完后（ablate 让 M 更实，但不加数量）

```
深度 3.0（ablate 不增加有证据的卡数，只加证据质量）
```

### 6.4 什么时候深度才真涨

```
需要"多张卡都有真加速"。一张卡的满分 = +0.42 均值。
要 D 从 2.9 → 5.1：需要 3 张卡 DS 8→65
要 D 从 2.9 → 6.6：需要 5 张卡 DS 8→65
```

**现在有 1 张（rmsnorm-fused）。还差 2 张。**

**下一张的候选**（按成本排）：
| 卡 | 有 skill 壳 | 有 bottleneck | 成本 |
|---|---|---|---|
| `fused-add-rmsnorm` | ✅ | ❌ | 低（同族，可直接复用 rms-v2-reg 思路）|
| `fusion-rmsnorm-rope` | ✅ | ✅ | 中 |
| `fusion-add-rmsnorm-quant` | ✅ | ✅ | 中 |

**建议：下一张打 `fused-add-rmsnorm`**——它就是 rmsnorm 去掉残差融合，`rms-v2-reg` 的寄存器驻留思路可直接迁移。**这是"经验复用"的第一次真实测试，也是 L5 记忆层该干的活。**

---

## 第七章 · 不做什么

1. **不加第 10 个工具。** 现有 9 个 + 新增 3 个（ablate/branch/analysis gate）足够。
2. **不拆 `engine/loop.py`**（1955 行，0 次故障；今天唯一故障是 shim 缺 `__main__` guard，搬迁引入的）。
3. **不动三维评分卡。** 加一个 `best_card` 展示项，不改 D1 的算法（改了历史不可比）。
4. **不借 SkillHub 的编排/状态/分母/阈值**（见 §4.3）。
5. **不在 `attribution.share` 还是估计值时宣布"归因完成"**。
6. **不宣布 `c5c2654` 完成**，直到 D0 的四条判据全过。

---

## 第八章 · 一句话

> **第一个真加速出来了：184.6 vs 202.2 µs，−8.7%，event_isolated 可 claim 口径。**
> **而且它是被 ISA 工具（inspect_so 看到 private 0→528B）直接指点出来的**——S1 那 1 天修文档的投入开始回本。
>
> **但三关没过：没落账（卡上还写着"我们慢 7.9%"）、精度门误判（bf16 的 1 ULP 比容差还大，四个变体含厂商全 FAIL）、归因没做。**
>
> **精度门那一条是全项目第三例"判据本身错"**——把"实现有问题"和"判据有问题"混为一谈，而默认怀疑实现。
>
> **v7.1 四件事：D0 修门+落账（半天）→ D1 analysis 前置闸门（半天，唯一真该借 SkillHub 的）→ D2 消融（1 天）→ D3 批量择优（1 天）。**
>
> **而深度分不会因为这一个大跳——一张卡满分只值 +0.42 均值。下一张打 `fused-add-rmsnorm`，那才是"经验能不能复用"的真测试。**
