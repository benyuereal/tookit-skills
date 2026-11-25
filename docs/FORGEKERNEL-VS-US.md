# ForgeKernel vs 我们的自主迭代 —— 深度对比与可借鉴之处

> 对比时间：2026-11-25
> 对象：`/workspace/ForgeKernel`（arXiv:2611.04998，OpenBMB，NVIDIA Hopper/Blackwell）
> 我们的：`/workspace/toolkit_skills/Plugins/ops`（海光 DCU gfx938）
>
> **先说结论**：它比我们成熟至少一代。值得借鉴的有 4 样，其中 **2 样是结构性差距**，
> 不是「加个功能」能补上的。但也有 3 样**我们做得比它更严**，不要因为对方更强就全盘照搬。

---

## 一、根本定位差异（这决定了后面所有差异）

| | ForgeKernel | 我们 |
|---|---|---|
| 平台 | NVIDIA Hopper/Blackwell（CUDA + CuTe DSL） | 海光 DCU gfx938（HIP + Triton + TileLang） |
| 工具链 | Nsight Compute (`ncu`) 一等公民 | PMC 采集不完整（`SQ_LDS_BANK_CONFLICT` 采不到） |
| 语言 | CuTe DSL 为主 | HIP 为主 |
| 代码量 | 38k 行 Python | 23k 行 |
| 成熟度 | 有论文、27 算子、公开排行榜第一 | 3 张卡做到超厂商，3 张卡有完整壳 |

**这个差异很重要**：它很多设计依赖 `ncu` 能精确归因到 stall reason。
我们的 PMC 采不到那些字段（D4 已登记），所以**照搬它的归因流程会落空**。

---

## 二、它比我们强的地方（4 样，按重要性）

### ★★★ 1. 自进化：每一代接受的 kernel 成为下一代 reference

**它怎么做的**（`scripts/self_evolve.py`）：

```
第 1 代：reference = PyTorch 九步实现
  → student 写一个融合 kernel → 被 judge 判通过 → **promote 成第 2 代的 reference**
第 2 代：reference = 第 1 代的最佳实现
  → student 必须**再超过它** → 通过就连任第 3 代
...
直到增益低于噪声地板才停
```

论文数据：每一代还能再快 **+9.9%**，包括已经超过专家库的 kernel
（GDN prefill 最终 +100% over FlashInfer，GQA +29% over FlashAttention-4）。

**我们现在是什么**：`advance.py` 的 replan 有 `CONTINUE_SWITCH` / `CONTINUE_SAME`，
但 **reference 永远是固定的厂商实现或 `ref-heavy`**。每轮都在和同一个基线比。

**这个差距的后果**：我们第一轮做到 2.03x 之后，第二轮如果还是跟原始基线比，
很容易「随便改改也比基线快」→ 判 ACCEPTED → 实际上毫无进展。
**它的机制天然杜绝了这件事**。

**它的实现细节**（`scripts/self_evolve.py`，关键参数）：

```python
NOISE_FLOOR_MIN = 0.015       # 增益低于 1.5% 算「摸到地板」，链条停
def noise_floor(best):        # 噪声地板 = 本代 confirmation ratio 的 spread
    return max(NOISE_FLOOR_MIN, (max - min) / min)
def next_generation(...):     # tasks/<op>-g<k>/：把 best 拷进 reference/，target 设 step 倍
    cfg["target"]["ratio"] = float(step)     # 默认 --step 1.05
```

两个设计值得注意：
1. **噪声地板是本代实测的 spread，不是固定值** —— 这正是我们「同批才谈百分比」
   那条纪律的做法，它做进了代码。
2. **每一代是新 task 目录**（`<op>-g2/`），不是原地改 —— 这样「每一代可复跑、
   可提交、可在别的机器重现」。

**借鉴成本**：中等。需要在 `advance.py` 里加「ACCEPTED 后把该变体写进卡的
`baseline_ref`」+ 新卡重新生成 reference。难点是**正确性**：
新 reference 必须自己先过 judge，否则会拿一个错的当基准。
**我们比它多一个难点**：新 reference 还得重新算 `roofline.json`
（因为分母变了），而它不需要（它不区分口径层）。

---

### ★★★ 2. Architecture-first：M0 把最容易错的决策先定死

**它怎么做的**（`meta_milestone.md`）：

里程碑不是「把优化过程写成散文」，而是**把优化路径结构化**：

```
M0 · 架构决策（kernel 切分 / tile 形状 / warp 角色 / 寄存器与 smem 预算 / 编排）
     ↕ 一旦 gate 通过就**冻结**
M1..Mk · 只在这个结构内调参
     ↕ 每级有 gate：正确性 + 性能 + 机制三线
  ↑ 如果测量表明只有换结构才能突破，arch agent **回去改 M0**，而不是让后续里程碑绕过它
```

关键设计：**「最容易搞错、且后来最难改的决策」全部压到 M0**。
而且它有 `⊗ atomic co-commit` 标记——某些改动必须同时落地，单独落一个必错。

**我们现在是什么**：`candidates.yaml` 是一维队列，按 EV 排序。
**没有「决策层级」的概念** —— 「换寄存器驻留结构」和「调 num_warps」
在同一个队列里竞争。

**这个差距的后果**：实测过。`op-norm-act-swiglu` 上我们先试了 BLOCK_ROWS=1/2/4，
花了三轮才发现「2 行最优」；而如果先定架构（1 行/CTA 的归约延迟藏不住 → 必须多行），
一轮就能到。

**借鉴成本**：高。需要给 `candidates.yaml` 加 `tier: architecture | tuning` 字段，
并且 `advance.py` 要先跑完所有 architecture 级候选才进 tuning。
但**这是结构性改造**，不是加个字段就行——`replan` 的 EV 逻辑要重写。

---

### ★★ 3. Lever pricing：同时测「单独加」和「单独减」

**它怎么做的**（`skills/lever-pricing/SKILL.md`）：

```json
{"kernel_file": "kernel.py", "rounds": 3,
 "levers": [{"name": "ring_per_stage", "edits": [...], "in_stack": true}],
 "variants": [{"name": "ring+lane", "levers": ["ring_per_stage", "lane_owned"],
               "question": "is lane_owned a dependency on the ring?"}]}
```

一句话概括它的价值：

> 「这是唯一能看出**某个 lever 单独看没用、但被其他 lever 依赖**（加了约等于 0，
> 减了显著变负），或者**只有减掉其他 lever 后才可见**的读法。」

**我们现在是什么**：`ablate.py` 只做「冠军减去某项」。
**只有「减」这一个方向**。

**这个差距的后果**：`mqa_opt` 那次实测就撞上了——`abl_noepi`/`abl_noshfl` 是
从基准派生的探测项，我们算出全负 share 却没法判断「是真的有害还是缺了配套项」。

**借鉴成本**：低。`ablate.py` 加一个 `--direction both`：
除了「冠军 − lever」，再测「基准 + lever」。两边的 delta 分别是正/负的才算真 lever。

---

### ★★ 4. 独立 judge：和 reference 是两个独立陈述

**它怎么做的**（`src/forgekernel/judge.py`）：

> 一个 teacher 回合读任务，往 `<run>/check/` 写三个文件：
> `definition.json`（签名）、`judge.py`（**语义的清晰陈述，从不为性能优化**）、
> `groups.json`（输入分组，每个形状至少三组）。
>
> 「如果 task 的 reference 是 `/goal` 会话写的，judge 只从 description 写出来：
> **两者是同一个算子的独立陈述**，生成的 reference 只有在 judge 通过时才被接受。」

**我们现在是什么**：`_common.py` 里同时有 `reference()` 和 `get_runner()`。
**同一个文件、同一个人写**。

**这个差距的后果**：reference 写错时，实现也跟着错，两边都错 → 门禁「通过」。
我们已经踩过：`rmsnorm-fused` 的 reference 一开始只返回一份输出，
导致「我们少干 1/3 的活」——
**参考实现自己就不对，门禁看不出来**。

**借鉴成本**：低-中。可以让 reference 从独立的 `.py` 文件加载
（而不是同一个 `_common.py`），并且要求「reference 和 adapter 不同文件」。

---

## 三、我们做得比它严的地方（3 样，别照搬）

### 1. 口径分层：我们有 `measure_mode`，它没有

我们的记录里每条都带 `measure_mode: event_isolated | back_to_back | unknown`，
且 `back_to_back` **判不可 claim**（实测无法全局修正：k₁/k₂=1.292）。

ForgeKernel 的计时是「同一 job 内 confirmation ratio」，它假设同 job 可比。
**在 NVIDIA 上这个假设基本成立**（时钟稳定），但在我们的共享 DCU 上不成立
（实测同一份代码 MAD 从 4.9% 漂到 7.2%）。

**不要借**：它的计时口径。

### 2. 分母三层：`nominal` / `achievable` / `impl_measured`

我们强制每个峰值常数要有 `reproduce` 指向真实脚本，且区分三层口径。
它的 `hw_kb/` 存硬件机制，但没有对「分母是哪一层」做结构化约束。

**不要借**：它的性能数字直接引用方式。

### 3. 反应式反作弊：`anticheat.py`

我们有三层：形状分离 / 扰动测试（×3/×0.01/取负）/ 输入隔离双测。
它没有对应机制（论文里也没提）。

**不过说实话**：它的 student 看不到 reference 源码（run 会 seal），
这本身就是一种防作弊。我们的机制是**事后检测**，它是**事前隔离**。
**两者可以叠加，但不能互相替代**。

---

## 四、明确不借的（会退化）

| 不借 | 原因 |
|---|---|
| ❌ 整体架构（38k 行） | 它是 NVIDIA 专用（CuTe DSL + ncu + sm90/sm100）。我们的 PMC 采不到它依赖的字段 |
| ❌ 「teacher/student」双 agent 模型 | 它的 teacher 是另一个 LLM 调用。我们单 agent 跑，加这层会让 token 成本翻倍 |
| ❌ `forge new` / `forge run` 的 CLI 形态 | 我们已有 `advance.py`，再加一套编排就是第三处记账 |
| ❌ milestone.md 的完整模板体系 | 它的 `meta_milestone.md` + `docs/milestones-template.md` 是一套很重的规范。我们 `candidates.yaml` 够用，只该借「分层」这一个概念 |

---

## 五、建议的执行顺序（按性价比）

| # | 做什么 | 成本 | 收益 | 风险 |
|---|---|---|---|---|
| **1** | **reference 独立成文件**（借第 4 条） | 半天 | 防「两边都错」 | 低 —— 只改加载方式 |
| **2** | **ablate 双向**（借第 3 条） | 半天 | 能分辨「真有害」与「缺配套」 | 低 —— 加个参数 |
| **3** | **promote 机制**（借第 1 条） | 2-3 天 | 杜绝「跟原始基线比显得一直在进步」 | 中 —— 新 reference 必须先过 judge |
| **4** | **候选分层** architecture/tuning（借第 2 条） | 1 周 | 少走「先调参后补结构」的弯路 | 高 —— 要重写 replan 的 EV |

**我的建议**：先做 1 和 2（一天内），做完再评估 3。
第 4 条成本高且我们还没撞到足够的痛点——等第 4、5 张卡做完，看是否真需要。

---

## 六、一句话总结

**它解决的核心问题我们也有，但它解决得更彻底**：

| 问题 | 它的解法 | 我们的解法 |
|---|---|---|
| 「一直在优化」怎么证明真的在进步？ | 每代 promote，必须超过上一代 | 跟固定基线比（会被「随便改改也比基线快」骗过） |
| 「先调参还是先改结构」？ | M0 冻结架构，后续只调 | 一维队列按 EV 排（容易被短期收益带偏） |
| 「这个 lever 有用吗」？ | 加/减双向测 | 只测减 |
| 「reference 写错了怎么办」？ | 独立 judge 交叉验证 | 同文件写，两边可能一起错 |

**前两条是结构性差距，后两条是小改进。**
但第 1 条（promote）其实是**我们已经在做但做错了方向**的事——
`advance.py` 的 replan 本来就有「换向」的概念，只是没把「换向后的赢家」变成新基线。

---

## 七、一个容易被忽略的差异：它把「噪声地板」做进了代码

`self_evolve.py` 里：

```python
NOISE_FLOOR_MIN = 0.015
def noise_floor(best):
    """本代 confirmation ratio 的 spread，never below NOISE_FLOOR_MIN。"""
    return max(NOISE_FLOOR_MIN, spread)
```

**它用「本代实测的 spread」当停止判据，而不是拍一个阈值。**

我们有同样的纪律（`judge()` 的 3×MAD、`measure_mode`），但**用的是固定 5% 熔断
+ 3×MAD**，不是「本代 spread」。在我们的共享卡上，这个差别可能很重要——
D4 那条债说的就是「同一份代码在四张卡上 MAD 从 4.9% 漂到 7.2%」。

**这条可以单独借**，成本极低：`advance.py` 的 `MAX_CONSECUTIVE_REFUTED` 之外，
再加一个「连续两代增益 < 本代 spread 就停」。
