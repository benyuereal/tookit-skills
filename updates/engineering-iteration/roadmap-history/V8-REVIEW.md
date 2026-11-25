# v8 路线图评审（2026-11-25）

> 评审对象：`AGENT-SKILL-ROADMAP-v8.md`（370 行，其他 agent 整理）
> 结论：**方向对、精算对、三处需要修正、一处排错了。**

---

## 一、精算验证：完全正确

v8 的核心推导：

```
D = 10 × Σ(w × ratio)
若 D2/D3/D4/D5 全拉满 (各 1.0) = 0.70
D1 需要 0.20 → D1 = DS均值/45 × 0.30 → DS 均值 = 30
```

**我逐项验算，结果 0.90 = 9.0 分，完全正确。**

**这个结论的重要性被低估了**：它意味着不需要 10 张 90 分的卡，
**10 张 30 分的卡就够**。所以 E4 的目标不是「把 3 张做到 100」，
是「把 10 张做到 30 以上」，**成本差一个数量级**。

---

## 二、三处需要修正

### 修正 1 · E3（hw_kb）已经做了一半

v8 说「沉淀 7 条」，现状：

```
hw_kb/hip/gfx938/
  reg_resident_vs_lds.md      ✅ 已写
  private_spill_trap.md       ✅ 已写
  vec16_uint4.md              ✅ 已写
  warp_reduce_shfl.md         ✅ 已写
  lds_bank_swizzle.md         ✅ 已写（反例，refuted）
  README.md                   ✅ 索引
```

**5 条已完成。** 剩余 2 条（`store_width` / `mls_mnemonic`）需要 ISA 证据，
成本比 v8 估的低。

### 修正 2 · E1 的目标卡选错了

v8 说「第 1 天上午 E1 自演化（fused-add-rmsnorm 打 gen1）」——**这张对**。
但候选表里 `rmsnorm-fused` 排第 3 是错的：

| 卡 | remaining | 结论 |
|---|---|---|
| `fused-add-rmsnorm` | **+20.3%** | ✅ 有空间，适合 promotion |
| `norm-act-swiglu` | **+21.1%** | ✅ 有空间，适合 promotion |
| `rmsnorm-fused` | **−0.15%** | ❌ **已到地板**，promotion 会立刻停 |

**证据**：`rmsnorm-fused` 的 `floor.verdict = at_floor`（我已修正过一次）。

**修正**：E1 的两张卡应是 `fused-add-rmsnorm` 和 `norm-act-swiglu`。
`rmsnorm-fused` 那张该做的是「补多形状」（D2/S 项），不是 promotion。

### 修正 3 · D3 的 19 张「补 comparison」不是填字段

v8 说「给其余 measured 卡补竞品对比」——**这需要真跑 A/B，不是改 YAML**。

D3 的定义（`scripts/scorecard.py:107`）：

```python
def _comparison_ok(card):
    """反作弊：comparison 必须有 baseline_role / cand_us / base_us，空块不算。"""
```

`cand_us` / `base_us` 必须是**实测数**。19 张卡 × 每张至少一次 A/B（含竞品安装、基线复现、
多形状）≈ **每张半天**。

**修正**：D3 补齐的实际成本是 **9-10 天**，不是 v8 暗示的「第五天一天做完」。

---

## 三、一处战略问题（比修正更重要）

v8 第七章的执行顺序：

```
第 1 天上午  E1 自演化
第 1 天下午  E3 hw_kb 7 条
第 2 天      E2 课程化
第 3-4 天    E4 十张满卡
第 5 天      D3 补齐 19 张
```

**问题：E4 排在第三，但它才是唯一推分的。**

v8 自己也说了「机制再好，不做满卡就不涨分」，但执行顺序仍然是「先机制后满卡」。

**我的修正**：

```
第 1 天上午  E4 的第 1、2 张（fused-add-rmsnorm 补 PMC → 96；norm-act-swiglu 补第 3 形状 → 96）
第 1 天下午  E1 自演化（用刚做到 96 的两张打 gen1，看能不能再涨）
第 2 天      E4 的第 3-5 张（从缺口最小往上补）
第 3 天      E3 hw_kb 剩余 2 条
第 4 天      E2 课程化
第 5 天起    D3 补齐（需要 9-10 天，要单独排期）
```

**理由**：E1 的 promotion 需要有「有空间的卡」当输入。
先把 `fused-add-rmsnorm` 补到 96（只差 PMC，半天），再 promotion 才有意义。

---

## 四、v8 最该被记住的一句话

> **「如果没有新增卡片数据，这一天算没做事。写文档、加工具、改口径都不算。」**

这句话是对的，而且是对**我**说的。过去 3 小时我产出 3 份文档 + 5 条 hw_kb，**分数没动**。

但它有一个副作用要注意：**hw_kb 本身不是「卡片数据」，却是跨算子复用的唯一载体**。
所以不能机械执行——**E3 的 5 条已经做完，那半天是值得的**，
只是不该再花第二天。

---

## 五、我的修订版优先级

| 顺序 | 做什么 | 成本 | 直接推分？ |
|---|---|---|---|
| 1 | `fused-add-rmsnorm` 补 PMC（M 0.5→1.0） | 半天 | ✅ DS 92→96 |
| 2 | `norm-act-swiglu` 补第 3 形状（S 0.5→1.0） | 半天 | ✅ DS 89→96 |
| 3 | E1 用这两张打 gen1 | 1 天 | 🟡 验证机制 |
| 4 | E4 剩下 3 张（rmsnorm-fused 补形状 / gemm-bf16 / gemm-fp8） | 2 天 | ✅ D2 → 5 |
| 5 | E3 hw_kb 剩余 2 条 | 半天 | ❌ |
| 6 | E2 课程化 | 1 天 | ❌ |
| 7 | D3 补齐 19 张 | **9-10 天** | ✅ 最后一公里 |

**前 4 项做完（约 4 天）：深度 ≈ 7.4。**
**加 D3 做完（约 2 周）：深度 ≈ 9.0。**

---

## 六、给子 agent 的分工建议

如果按 v8 开多个 agent，**写范围必须切开**（见 `docs/PARALLEL.md` 的规则）：

| agent | 写范围 | 任务 |
|---|---|---|
| A | `skills/op-fused-add-rmsnorm/**` | 采 PMC，补 M 项 |
| B | `skills/op-norm-act-swiglu/**` | 补第 3 个形状的实测 |
| C | `scripts/self_evolve.py`（新建） | E1 自演化机制 |
| D | `curriculum.yaml`（新建）+ `check_curriculum.py` | E2 课程化 |
| E | `hw_kb/hip/gfx938/` 剩余 2 条 | E3 |

**A 和 B 可以并行**（不同卡，无共享文件）。
**C 依赖 A/B 的结果**（需要 96 分的卡当输入）—— 不能并行。
**D、E 完全独立**，可并行。

**GPU 互斥**：A、B 都要跑 bench，必须 `flock /tmp/opbench-gpu.lock`。
