# Agent Skill 路线图 v7：从「相关」到「因果」

**日期**：2026-11-24
**业主定性**：深度目标 = agent 拥有自主迭代研发算子的能力
**版本弧线**：v5.1 数字可信 → v6 能改 → **v7 能证明改对了**

> 本文件所有数字都在容器里实跑过。没有实跑的预测一律标注为预测。

---

## 第零章 · v6 之后发生了什么（15:34 实测）

### 0.1 v6 的六个包，五个完成，一个只做了一半

| 包 | 内容 | 状态 | 证据 |
|---|---|---|---|
| **C1** | `surface.transform` 字段 | ✅ | 17 条 surface **全部**有 transform；`check_transform.py` 604 行 |
| **C2** | `candidates.patch` + `variant_forge` | ✅ | `variant_forge.py` **838 行**（只做 Triton，承认边界）|
| **C3** | `design_space.py` | ✅ | 229 行，枚举 **486 个配置**并按预测排序 |
| **C4** | tile 级模型 ×3 | ✅ | `models/` 现有 5 个：`matmul` / `attention_decode` / `reduce_lastdim` / `sparse_mla` / `mqa_logits`；回测误差 **≤0.005**（11 点）|
| **C5** | `kernel_scaffold.py` 三条硬判据 | 🟡 | 303 行，能生成可编译骨架；**但 HIP 骨架里 BLOCK_M 是硬编码**，改参数要重写 → "4 个组合 → 4 组不同资源画像"**做不到** |
| **C6** | 真跑一轮 | 🟡 | `advance.py` 666 行跑通 2 轮；**但跑的是 rmsnorm（有 lightop 基线），测的是选型不是研发** |

**九个工具全部就位**：

```
check_transform 604 · variant_forge 838 · design_space 229
roofline_golden 212 · anticheat 336 · calibrate 516
advance 666 · isa_probe 365 · kernel_scaffold 303
                              合计 ≈ 3,870 行
```

### 0.2 C6 真跑的结果：两轮，两个失败

```
round 1  triton     Δ +24.2%   REJECTED_NO_GAIN
round 2  tilelang   Δ  −2.6%   INCONCLUSIVE
```

口径全部 `event_isolated` ✅，`falsified` 按规则置位 ✅，无人干预 ✅。

**这是 v6 最有价值的产出——它诚实地跑出了两个"没变快"。**

但它也暴露了问题的本质：**两轮都是"拿现成实现去比"，不是"改一版新的"**。

### 0.3 11 条记录全在一张卡上

```
Plugins/ops/skills/op-rmsnorm-fused/runs/loop/20261124.jsonl   11 条
memory/ledger.jsonl                                            0 字节
```

`memory/` 是 v5.2 建的跨卡记忆索引，**至今一条没写进去**。`recall` 做出来了，没有东西可检索。

### 0.4 分数：深度一动不动

| 维度 | v5.1 末 | v6 末（现在） | Δ |
|---|---|---|---|
| **深度** | 2.9 | **2.9** | **±0** |
| **覆盖** | 6.5 | **6.5** | **±0** |
| 质量 | 3.7 | **6.7** | +3.0 |

```
深度  D1(13.5) D2(1) D3(0.222) D4(1) D5(8)
覆盖  C1(1.0) C2(0.64) C3(0.741) C4(4) C5(2)
```

**建了 3,870 行工具，深度 +0.0。**

---

## 第一章 · 为什么建了 9 个工具，深度一分没涨

### 1.1 表层原因：深度分的分子是「有证据的卡」

DS 的五个分量（B/U/S/R/M）**全部来自卡片上的实测数据**。工具不产生数据，只让"产生数据"变容易。

所以"建工具 → 分数不动"是**设计使然，不是失败**。v5.3 就写明了：

> 场景 A（只做补强）：深度 **3.1**（几乎不动）
> 场景 B（补强 + 真跑）：深度 **6.1**

**现在正好落在场景 A。**

### 1.2 深层原因：我们只能证明「相关」，不能证明「因果」 ⭐

这是 v7 要解决的真问题。证据分三层：

```
L1 变化（change）
   代码变了。               ← 我们有：git diff、variant_forge
   证明力：无

L2 相关（correlation）
   改完变快了。             ← 我们有：judge + measure_mode + ABBA
   证明力：弱。可能是别的变量变了、可能是噪声、可能是热漂移

L3 因果（causation）
   是这个改动让它变快的。    ← 我们完全没有
   证明力：强。这是"研发"唯一可接受的证据
```

**现有工具全部停在 L2：**

| 工具 | 到哪一层 |
|---|---|
| `variant_forge` | L1（生成变化）|
| `design_space` | L1（枚举变化）|
| `advance` | L2（跑 + 判显著）|
| `calibrate` | L2 的反省（预测 vs 实现，还是相关）|
| `roofline_golden` | 分母正确性（不是归因）|
| `anticheat` | 防伪（不是证明）|

**唯一能到 L3 的方法是消融（ablation）**：

```
attribution(m) = ms(去掉方法 m 的版本) − ms(冠军版本)

正数 ⇒ 这个方法真的有用（去掉它变慢了）
零/负 ⇒ 这个方法没用，或者它只是搭便车
```

**`grep -c "ablat" scripts/*.py` = 0。我们没有。**

### 1.3 后果：`attribution.md` 要求的 `share` 字段，我们写不出来

`references/attribution.md` 规定：

```yaml
attribution:
  dominant: l2-miss       # 五类之一
  share: 0.62             # 该瓶颈能解释的时间占比
  evidence:               # 必须是数值
    l2_hit_rate: 0.41
```

**`share: 0.62` 从哪来？** 现在的答案是：**人凭经验估**。而同一份文档第 38 行写着：

> `share` 是**估计值**，允许粗（0.5 / 0.6 / 0.7 这种粒度）。

**估计值 = 不是证据。** 而 card 上的 `bottleneck.share` 会进 DS 的 M 项、进 EV 的排序、进 `plan` 的第一名。**一个靠估的数字，正在驱动整个决策链。**

---

## 第二章 · 从 SkillHub 学到的三件事（外部对标）

刚把 `das/SkillHub` 拉下来对比过（44 个 skill，22 MB，Operator Development 类 10 个）。它和我们不是一类东西：

| | SkillHub | 我们 |
|---|---|---|
| 定位 | **工具箱**（单一职责，可独立安装）| **闭环系统**（1 个引擎 + 1 张图谱）|
| 规模 | `kernel-triton-optimize` = 28 脚本 / **11,104 行** | 9 工具 / 3,870 行 + 引擎 1,955 行 |
| 实证 | 多轮真跑过（否则写不出 attribution 公式）| 跑过 1 张卡 11 条记录 |

**但有三件它做得比我们深，必须学：**

### 2.1 它有 `ablate.py`（209 行）—— 到 L3 的唯一路径

```python
"""Single-method ablation for attribution.

For each method applied in the champion kernel, this script expects the agent
to have generated an ablated kernel (champion minus that one method) under
  iterv{i}/ablations/{method_id}/kernel.<ext>

  attribution(m) = ms_ablated(m) - ms_champion
  Positive means the method helped (removing it slowed things down).
  Zero/negative means the method did not help or actually hurt.
"""
```

**这就是我们缺的那一块。** 209 行，思路极简：**把冠军版本逐项拆掉，各测一遍。**

### 2.2 它有 `branch_explore.py`（428 行）—— 吞吐 ×K

```python
"""Branch-and-Select: compile and benchmark K candidate kernels in parallel.

All K branches share the same method combination (from methods.json) but
differ in hyperparameters (tile size, num_stages, num_warps, etc.).

  1. Compiles all K kernels
  2. Benchmarks each with validation
  3. Selects champion = fastest valid kernel
  4. Copies champion to iterv{i}/kernel.<ext>
  5. Returns non-champions as frontier candidates in state
"""
```

我们有 `design_space.py` 能枚举 **486 个配置**，但 `advance.py` **一次只跑一个**。**枚举了不算，跑了才算。**

### 2.3 它强制「原生 autotune 公平性」预步骤

> 如果原生 kernel 使用 `@triton.autotune`，先对每个目标 shape 跑**原生 autotune 空间**，捕获获胜配置，**然后才**在优化包装器里 bypass autotune。没有这项证据，`setup` 直接失败。

**这是"baseline 必须是竞品"纪律的最后一个洞。** 我们目前只覆盖了"用哪个实现"，没覆盖"**同一个实现的原生 autotune 有没有跑满**"。如果 triton kernel 自己开了 autotune 而我们固定配置去比，赢了是假的。

`grep -rn "disable.autotune\|autotune.*winner" scripts/*.py engine/*.py` = **0 命中**。

### 2.4 但我们有三件它完全没有的东西 —— 不能丢

| 能力 | 我们 | SkillHub |
|---|---|---|
| **反作弊四件套** | ✅ 11 项自测，缓存 kernel 能抓 | ❌ 0 个检测 |
| **三层口径** | ✅ nominal / achievable / impl，1680 vs 2000 分明 | ❌ **只有 nominal**（`peak_bw_gbs = 2000.0`）|
| **算子图谱** | ✅ 53 张卡 / 12 族 | ❌ 无图谱，"通用"= 每次从零推导 |

> 第二项尤其要紧：**用 2000 当分母，所有带宽达成率都被系统性低估 19%**。这正是我们 fp8 那次栽过的坑。

---

## 第三章 · v7 的四个补强包

### D1 · `ablate.py` —— 归因从散文变成数字（1 天）

**输入**：一个冠军实现 + 它用到的 N 个变换（来自 `candidates.yaml` 的 patch 链或 `methods` 列表）
**输出**：每个变换的 `attribution = ms(去掉它) − ms(冠军)`

```python
# scripts/ablate.py
"""
逐项消融：把冠军版本拆掉一项，看慢多少。

  attribution(m) = ms(champion − m) − ms(champion)
    正数 ⇒ 方法 m 真的有用
    零/负 ⇒ 搭便车或无效

与 judge() 的关系：judge 判"这次改动显著吗"，ablate 判"这个改动贡献了多少"。
两者都需要，且都用 measure_mode 口径门。
"""
```

**配套改动**：
1. `variant_forge.py` 加 `--ablate <method>`：从冠军版本生成"去掉第 m 项"的变体
2. `candidates.yaml` 的 patch 链要能表达"冠军由哪几项组成"（现在 `patch` 是平铺 dict，需要有序列表）
3. `record` 增加 `ablation: {method: delta_ms, attribution_share: float}`

**为什么这是 D1**：它把 `attribution.share` 从估计变成实测，而 `share` 正在驱动 EV 排序。

**验收**：
```bash
# 对已有 C6 冠军（fused-3pass-tilelang）做消融
python3 -I scripts/ablate.py \
  --card Plugins/ops/skills/op-rmsnorm-fused \
  --champion fused-3pass-tilelang \
  --methods tile_pack,vec4,num_warps4 \
  --json | python3 -I -c "
import json,sys
d=json.load(sys.stdin)
assert len(d['ablations'])==3, '消融数不对'
s=sum(a['attribution_share'] for a in d['ablations'])
assert 0.8 <= s <= 1.2, '归因份额不闭合（sum=%.2f，应≈1）' % s
print('OK 归因份额闭合：', [(a['method'], round(a['attribution_share'],2)) for a in d['ablations']])
"
```

**判据**：**N 项消融的 `attribution_share` 之和落在 0.8~1.2**。
这条比"跑通了"强得多——它要求归因**能解释全部收益**，不能有神秘消失的部分。

---

### D2 · `branch_explore.py` —— 一次跑 K 个（1 天）

**输入**：`design_space.py` 排好序的前 K 个配置
**输出**：冠军 + frontier（非冠军保留，下轮可复用）

```python
# scripts/branch_explore.py
"""
Branch-and-Select：编译并测 K 个候选，留最快的，其余进 frontier。

与 advance.py 的关系：advance 是"多轮"，branch 是"一轮里的多路"。
       advance(round) = plan → gate → branch_explore(K) → judge → record
"""
```

**关键设计**：不做真并行（GPU 只有一张），做**批量串行 + 一次决策**：

```
for k in 1..K:  build(variant_k); bench(variant_k)
champion = argmin(median_ms among valid)
frontier = 其余 valid 的（下轮可直接用，不重编）
```

**验收**：
```bash
python3 -I scripts/branch_explore.py \
  --card <卡> --topk 4 --from-design-space --json | python3 -I -c "
import json,sys
d=json.load(sys.stdin)
assert len(d['branches'])==4
assert d['champion']['median_ms'] == min(b['median_ms'] for b in d['branches'] if b['valid'])
assert len(d['frontier'])==3, '冠军之外的应进 frontier'
print('OK 4 支选 1，frontier 3 条')
"
```

**判据**：**K=4 时，整轮墙钟 ≤ 4 × 单次 bench（不许有指数爆炸）**，且 frontier 里有 3 条可复用。

---

### D3 · autotune 公平性预步骤（半天）

**问题**：如果被优化的 kernel 自带 `@triton.autotune`，而我们固定配置去比，赢了是假的。

**实现**：
```python
# scripts/check_baseline_fair.py
"""
baseline 公平性三查：
  ① 源文件里有没有 @triton.autotune / @triton.heuristics
  ② 有的话，自动 tune 空间的 winner 是否被捕获并烘焙
  ③ 没有捕获 ⇒ 拒绝继续（fail-closed）
"""
```

**验收**：
```bash
python3 -I scripts/check_baseline_fair.py --card <卡> --variant fused-3pass-triton
# 三种输出，都要跑到：
#   native_autotune_absent       → OK，直接用
#   native_autotune_baked        → OK，winner 已烘焙，打印 source/hash/config
#   native_autotune_bypassed     → 拒绝（rc=2），要求先捕获 winner
```

**判据**：三种状态都能触发，且 `bypassed` 必须 rc=2。

---

### D4 · 一个真加速（2 天，依赖 D1+D2）

**这是 v7 唯一的真验收，也是唯一能推动深度分的。**

v6 的 C6 跑的是 rmsnorm（有 lightop 基线）→ 测的是**选型**。
v7 要跑的是**改一版新的，并证明它变快了**。

**前置：先解决 rmsnorm 的下一步，还是换卡？**

建议**先在 rmsnorm 上做**，理由：

1. 已有 11 条记录、3 轮历史、`bottleneck` 写清了（"2读2写交织 + 行内归约"）
2. C6 已经跑出 `tilelang Δ−2.6%`（接近但没过线）——**这是最接近成功的一步**
3. 换卡要重建造壳（`mlp-gelu-geglu` 连 skill 目录都没有，且语义有歧义）

**具体目标**：把 `fused-3pass-tilelang` 从 `−2.6%（INCONCLUSIVE）` 推到 `显著为正`。

已知可用杠杆（来自 `bottleneck.root_cause`）：
- `2读2写交织` → `vmem-vector-width`（实测写只有 dwordx2）
- `行内归约` → `reduce-lastdim` 模型的三档锚点
- `H 的奇偶性不影响带宽`（轮1 已证）→ 排除向量化宽度

**验收（三条全过才算）**：
```bash
# ① 显著变快，且不是噪声
python3 -I scripts/advance.py --card Plugins/ops/skills/op-rmsnorm-fused \
  --max-rounds 3 --k 4
# → 至少 1 条 record 满足：significant=True AND decision 以 ACCEPTED 开头
#   AND measure_mode=event_isolated

# ② 归因闭合（D1）
python3 -I scripts/ablate.py --card <卡> --champion <新冠军> --methods <全部> --json | \
  python3 -I -c "import json,sys; d=json.load(sys.stdin); s=sum(a['attribution_share'] for a in d['ablations']); assert 0.8<=s<=1.2"

# ③ 反作弊通过（S5 已有）
python3 -I scripts/anticheat.py --scan <新冠军的源码> --shape-sep --correctness <shapes>
```

**判据**：**三条全过，缺一条不算完成。**

---

## 第四章 · 时序

```
D1  ablate.py              ← 最高杠杆：让 share 从估计变实测
D2  branch_explore.py      ← 吞吐 ×K
D3  check_baseline_fair    ← 半天，补 baseline 最后一个洞
D4  真加速（依赖 D1+D2）    ← 唯一推动深度分的一步
```

| 天 | 内容 | 可验证产物 |
|---|---|---|
| 1 | D1 + D3 | 消融闭合（sum≈1）；三种 autotune 状态都能触发 |
| 2 | D2 | K=4 跑通，frontier 3 条 |
| 3–4 | D4 真加速 | ≥1 条 `ACCEPTED_*` + 归因闭合 + 反作弊通过 |

**关键：D1/D2/D3 是工具，D4 是验收。前三天不要动 D4 的代码，第四天不要加新工具。**

---

## 第五章 · 评分预估（带算式）

### 5.1 现状

```
D1=13.5  D2=1  D3=0.222  D4=1  D5=8
权重：0.30 / 0.25 / 0.20 / 0.15 / 0.10
```

### 5.2 三档场景

**场景 A · D1–D3 做完，D4 没跑成**

| 项 | 值 | 说明 |
|---|---|---|
| D1 | 13.5 | 工具不产生 DS |
| D2 | 1 | 同上 |
| D3 | 0.222 | 同上 |
| D4 | 1 | 同上 |
| D5 | 8 | ablate 让 D5 更实，但数量不变 |

```
D = 0.3×0.300 + 0.25×0.200 + 0.2×0.261 + 0.15×0.125 + 0.1×0.800
  = 0.090 + 0.050 + 0.052 + 0.019 + 0.080 = 0.291
⇒ 深度 2.9（不动）
```

**场景 B · D4 在 3 张卡上跑成**

假设 3 张卡 DS 8→65：

| 项 | 值 | 算式 |
|---|---|---|
| D1 | 16.7 | (715.5 + 3×57) / 53 |
| D2 | 3 | 3/5 |
| D3 | 0.30 | 11/27（多 1 张有 comparison）|
| D4 | 4 | 4/8 |
| D5 | 11 | 11/10 → 1.0 |

```
D = 0.3×0.371 + 0.25×0.600 + 0.2×0.353 + 0.15×0.500 + 0.1×1.0
  = 0.111 + 0.150 + 0.071 + 0.075 + 0.100 = 0.507
⇒ 深度 5.1
```

**场景 C · 5 张卡跑成**

```
D = 0.3×0.420 + 0.25×1.000 + 0.2×0.353 + 0.15×0.750 + 0.1×1.0
  = 0.126 + 0.250 + 0.071 + 0.113 + 0.100 = 0.660
⇒ 深度 6.6
```

### 5.3 为什么到不了 9

```
53 张卡里 26 张 draft（DS 0–15），均值被死压。
即使 10 张卡做到 DS 70：D1 = (715.5 + 10×62)/53 = 25.2 → ratio 0.56
D 上限 = 0.3×0.56 + 0.25×1 + 0.2×1 + 0.15×1 + 0.1×1 = 0.868 → 8.7
但要 10 张卡 ×3 轮真实实验 —— 3–4 周
```

**深度 6.6 是"能力成立"的分数；8+ 要"覆盖也做满"；9 要把 26 张 draft 卡也做完。**

---

## 第六章 · 不做什么（v7 比 v6 更容易犯的错）

1. **不加第 10 个工具**。现在有 9 个，全部自测通过，全部没在真加速里用过。**先把这 9 个用一次。**
2. **不拆 `engine/loop.py`**。1955 行 / 59 函数 / 0 次故障。今天的唯一故障是 shim 缺 `__main__` guard（搬迁引入的），不是文件太大。
3. **不做 L3 算法创新**（重写数学）。
4. **不用 rmsnorm 之外的卡做"能力验收"**——但如果 D4 在 rmsnorm 上失败，才考虑换卡，且换卡要先造壳、单独验收。
5. **不在 `attribution.share` 还是估计值的情况下宣布"归因完成"**。
6. **不把 SkillHub 的 skill 直接拼进来**——它的 `kernel-triton-optimize` 有自己完整的 orchestrate/state/methods，硬拼会造成第三处记账。**只吸收 `ablate.py`（209 行）和 `branch_explore.py`（428 行）的逻辑。**

---

## 第七章 · 一句话

> **v5.1 让数字可信，v6 让 agent 能改，v7 要让 agent 能"证明改对了"。**
>
> 建了 3,870 行工具，深度 +0.0。原因不是工具没用，是**所有工具都停在 L2「相关」（改完变快了），没有一个到 L3「因果」（是这个改动让它变快的）**。
>
> 而 `attribution.share` 现在靠人估（"0.5 / 0.6 / 0.7 这种粒度"），**却正在驱动 EV 排序和 plan 的第一名**。
>
> **v7 的四件事：`ablate`（到 L3）→ `branch_explore`（吞吐 ×K）→ autotune 公平性（补 baseline 最后一洞）→ 一个真加速（唯一能推动深度分的事）。**
>
> **验收只有一条：让某个算子显著变快，且能证明是哪个改动让它快的。**
