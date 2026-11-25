# 从 ForgeKernel 能吸收什么（具体清单）

> 配套：`FORGEKERNEL-VS-US.md`（差异分析）。本文只讲**能动手做的**。
> 原则：**借机制，不借代码；借方法，不借数字。**

---

## 一、能直接吸收的（4 项，全部有明确落点）

### A1 · hw_kb：硬件知识库（▶ 最该吸收）

**它是什么**：`hw_kb/<language>/<arch>/` 下 200 个 markdown 文件，
每个文件讲**一条硬件接口的用法**，**算子无关**。

格式（每篇都长这样）：

```markdown
# branchfree_masked_elementwise_pass

> **Certification source: the sm100 tree** -- **Status: ✅ GPU-certified**
> `PROBE PASS max_diff=3.8657465348084646e-07`
> (六条 contract 同时 PROBE PASS，readings in Sources)

## Purpose
...
## Mechanism
...
## Sources   ← 实测读数
```

三条关键设计：

1. **每条必须带认证状态**（`✅ GPU-certified` / `⬜ unverified`）+ **探针实测值**。
   没有实测的不能标 certified。
2. **算子无关**。它讲「怎么用 `tcgen05.ld` 读回累加器」，不讲「rmsnorm 怎么写」。
   所以能被所有算子复用。
3. **按语言 × 架构分树**。`cutedsl/sm100` 和 `cuda/sm100` 是两个独立树，
   一个语言的认证不迁移到另一个。

**我们现在的对应物**：`Plugins/ops/skills/op-hip-compile/references/` 有零散的知识
（gfx938 的 11 条编译选项、指令面普查、LDS 读形态），但：

- 没有统一的「每条带认证状态 + 探针值」格式
- 没有 `hw_kb/` 这样**跨算子复用**的独立目录
- 知识散在 skill 的 references 里，换个算子就找不到

**落点**：

```
hw_kb/
├── README.md           规则：认证状态 + 探针值必填
├── hip/
│   ├── gfx938/
│   │   ├── lds_bank_swizzle.md          ← 从 mqa_opt 的实测提炼
│   │   ├── reg_resident_vs_lds.md       ← 从 rmsnorm 三张卡提炼
│   │   ├── vec16_uint4.md               ← 16B 向量访存
│   │   ├── private_spill_trap.md        ← VEC 运行期化导致溢出（慢 3.3x）
│   │   └── warp_reduce_shfl.md          ← warp shuffle + 静态 LDS 表
│   └── gfx942/（若有）
├── triton/
│   └── gfx938/
└── tilelang/
    └── gfx938/
```

**成本**：低。我们已有 5-8 条实测过的机制（见上面列表），只需按格式重写。

**收益**：高。这是**跨算子复用**的唯一载体。现在每张新卡都要重新踩一遍
「寄存器驻留 vs LDS」的坑，有了 hw_kb 就能直接查。

---

### A2 · reference 独立成文件

**它怎么做的**：`judge.py` 从 description 独立写出，和 reference 是两个独立陈述。

**我们的落点**：

```
Plugins/ops/skills/op-<卡>/
  _common.py          ← 只剩 build_inputs / get_runner / compare
  reference.py        ← 独立文件，只放 reference()，**不许 import _common**
```

**判据**（可机检）：`reference.py` 不能 import `_common`，反之亦然。
两边都算一遍，结果必须一致 —— 这是「交叉验证」。

**成本**：半天。**收益**：防「两边都错」（已踩过 rmsnorm 的少写一份输出）。

---

### A3 · ablate 双向

**它怎么做的**（`skills/lever-pricing/SKILL.md`）：

> 「唯一能看出**某个 lever 单独看没用、但被其他 lever 依赖**
> （加了约等于 0，减了显著变负）的读法。」

**我们的落点**：`scripts/ablate.py` 加 `--direction both`：

```
方向 A（现有）：冠军 − lever      → delta_A > 0 表示该项有正贡献
方向 B（新增）：基准 + lever      → delta_B < 0 表示该项单独也有效
```

**四种组合的解读**：

| delta_A（减） | delta_B（加） | 结论 |
|---|---|---|
| 显著变慢 | 显著变快 | **真 lever**，独立有效 |
| 不显著 | 显著变快 | 真 lever，但被其他项掩盖 |
| 显著变慢 | 不显著 | **依赖项** —— 单独没用，缺了不行 |
| 不显著 | 不显著 | 噪声，删掉 |

**成本**：半天。**收益**：`mqa_opt` 那次就是「依赖项」被误判成「有害」。

---

### A4 · promote 机制（自进化）

**它怎么做的**：`scripts/self_evolve.py`

```python
NOISE_FLOOR_MIN = 0.015
def noise_floor(best):
    """本代 confirmation ratio 的 spread，never below NOISE_FLOOR_MIN。"""
    return max(NOISE_FLOOR_MIN, spread)
def next_generation(task, best, dest, step):
    """tasks/<op>-g<k>/：把 best 拷进 reference/，target 设 step 倍。"""
    cfg["target"]["ratio"] = float(step)
```

**我们的落点**：`advance.py` 加 `--promote`：

```
某变体判 ACCEPTED 且 ratio < 1.0
  → 写进卡片的 baseline_ref（新基线）
  → 重算 roofline.json（分母变了）
  → 下一轮跟它比，而不是跟原始基线
```

**关键约束**（它做了我们没做的）：

1. **新基线必须先过 judge** —— 拿一个错的当基准是最坏的。
2. **噪声地板=本代 spread**，不是固定值。
3. **每代是新目录**，不是原地改 —— 可复跑、可提交。

**成本**：2-3 天。**收益**：杜绝「跟原始基线比显得一直在进步」。
**风险**：中 —— 分母变了要重算 roofline，而它不需要（不区分口径层）。

---

## 二、只能借思路、不能借实现的（3 项）

### B1 · Architecture-first（M0 冻结）

**思路**：把「最容易搞错、后来最难改」的决策先定死，后续只调参。

**为什么不能照搬**：它的 M0 是一份完整的 `milestone.md`，配套
`docs/milestones-template.md` + `meta_milestone.md` 两套规范。太重。

**我们能做的最小版本**：`candidates.yaml` 加一个 `tier` 字段：

```yaml
- id: reg-resident
  tier: architecture        # architecture | tuning
  ...
- id: num-warps-sweep
  tier: tuning
```

`advance.py`：**所有 architecture 级候选跑完之前，不碰 tuning 级**。

**成本**：改 replan 的 EV 逻辑，1 周。**先不做** —— 等第 4、5 张卡看是否真需要。

### B2 · 结构化 milestone 课程

**思路**：把优化路径拆成「局部容易的决策序列」，按依赖排序。

**为什么不能照搬**：它假设有一个「低能力 student agent」需要被引导。
我们是单 agent，层级越少越好。

### B3 · teacher/student 双 agent

**不借**：token 成本翻倍，且我们的 judge 已经很严。

---

## 三、明确不借的（会退化）

| 不借 | 原因 |
|---|---|
| ❌ 整体架构 38k 行 | NVIDIA 专用（CuTe DSL + ncu + sm90/sm100）；我们的 PMC 采不到它依赖的字段 |
| ❌ `forge new` / `forge run` CLI | 会造成第三处编排记账 |
| ❌ 它的计时口径（同 job confirmation ratio） | 我们的共享卡上不成立（MAD 4.9%~7.2% 漂移） |
| ❌ 它的性能数字直接引用 | 没有 `measure_mode` 分层，没有分母三层口径 |

---

## 四、执行顺序（我建议）

| # | 做什么 | 成本 | 收益 | 何时做 |
|---|---|---|---|---|
| 1 | **hw_kb 建起来**（先放 5 条我们已有的机制） | 半天 | **最高** —— 跨算子复用的载体 | 现在 |
| 2 | reference 独立成文件 | 半天 | 防两边都错 | 现在 |
| 3 | ablate 双向 | 半天 | 分辨真 lever vs 依赖项 | 下一步 |
| 4 | promote 机制 | 2-3 天 | 杜绝假进步 | 第 4 张卡之前 |
| 5 | 候选分层 | 1 周 | 少走弯路 | 看情况 |

**为什么 hw_kb 排第一**：它是唯一能**真正跨算子复用**的东西。
我们现在每张新卡都要重新踩一遍「寄存器驻留 vs LDS」的坑 ——
`rmsnorm-fused`、`fused-add-rmsnorm`、`norm-act-swiglu` 三张卡，
每张都重新发现了同一条机制。**这是纯浪费。**

---

## 五、我们已经有的、可以立刻进 hw_kb 的 5 条

这五条都是**实测过**的（有数字、有复现路径），只是没整理成通用格式：

| 机制 | 实测证据 | 来源 |
|---|---|---|
| `reg_resident_vs_lds` | 寄存器驻留 1454 GB/s vs 整行 LDS 212 GB/s（6.8x） | rmsnorm-fused |
| `vec16_uint4` | 16B 向量读写是把带宽打上去的必要条件 | 三张卡都用 |
| `private_spill_trap` | VEC 运行期化 → private 0→528B → 慢 3.3x **且不报错** | rmsnorm-fused |
| `lds_bank_swizzle` | XOR swizzle 解决 9.6 冲突/指令，**但只快 0.36%**（反例） | mqa_opt |
| `warp_reduce_shfl` | shuffle + 静态 LDS 表（**动态 LDS 不可读**） | 三张卡都用 |

**特别值得写进去的是第三条**：`private_spill_trap`。
它是个「错了也不报错、只是慢 3.3 倍」的陷阱，最容易被误判成「这个方向不行」。
