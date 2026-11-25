# 给评测方的使用建议：怎么测才能测出这套东西的真实能力

> 读者：跑三臂/多臂对照实验的评测方（opbench 等）。
> 目的：让「测出来的分数」反映能力，而不是反映**接口没对上**。

---

## 一、先说一个实测教训（你们已经踩过）

2026-11-25 那次 `norm-fused-add-rmsnorm-l1` 三臂对照：

| 臂 | 写了几行 HIP | 最终比值 |
|---|---|---|
| A 裸 CC | 0 | 1.298× |
| **C toolkit-skills** | **0** | 1.297× |
| D SkillHub | 0 | 1.303× |

**三臂都没写过一行 HIP。** 而当时 `op-rmsnorm-fused/runs/forged/` 下
已经有 4 个实测更快的 HIP kernel —— 只因为它们藏在 `runs/`（gitignore 目录），
agent 看不见。

**这不是能力问题，是可见性问题。** 我们这边已经修了（挪到 `kernels/` 并登记
`variants.yaml`），但评测设计上也值得注意：**如果测试目录只给 SKILL.md，
agent 不会知道仓库里有什么。**

---

## 二、三条建议（按重要性）

### 建议 1 · 给完整仓库，不只给一个 skill

**现状推测**：如果只把 `Plugins/ops/skills/op-<目标>/` 复制进测试目录，
agent 就看不到：
- `scripts/op_bootstrap.py`（新算子入口）
- `scripts/lang_select.py`（语言路由）
- `scripts/advance.py`（多轮编排）
- 其他卡的 `kernels/*.hip`（可参考的实现）

**建议**：把整个 `toolkit_skills/` 挂进去（或至少 `Plugins/ops/` + `scripts/`），
让 agent 能 `ls` 到、能 `grep` 到。

**为什么**：我们这套的设计前提就是「工具链完整」——`op-optimize` 会调
`lang_select`、`advance` 会调 `forge_step`。**只给一个 skill 目录等于把手脚绑住。**

### 建议 2 · 明确告诉 agent「可以用 scripts/」

如果评测的 prompt 只写「优化这个算子」，agent 很可能：
- 直接读 `SKILL.md`
- 手写 Triton
- 停在 1.3×

**建议在三臂的 prompt 里加一句**：

> 仓库根目录有 `scripts/`，里面是这套 skill 的工具链。
> 先跑 `python3 -I scripts/lang_select.py --card <卡名>` 看它建议用什么语言，
> 再决定怎么做。

这一句能直接把 `lang_select` 接进流程——**否则它是死代码**（我们自己实测过：
工具存在但零调用方）。

### 建议 3 · 接受「第一次跑基线」这次开销

新算子的完整流程是六步，第 3 步（跑一次基线拿 `t_actual`）**不能省**：

```
op_bootstrap → 补 reference → 跑一次基线 → floor → lang_select → advance
```

**原因**：语言路由需要 `utilization`，而它需要实测耗时。没有 `t_actual` 时
路由输出「无规则命中 —— 先补齐 roofline 再选语言」。

这是设计，不是缺陷。**如果评测按 wall-clock 计分，请把这一次基线跑算进预算**，
否则会被误判成「这个 skill 起手慢」。

---

## 三、怎么判断测的是「能力」而不是「运气」

### 三条硬指标（都必须有）

| 指标 | 在哪看 | 合格线 |
|---|---|---|
| 显著性 | `<卡>/runs/loop/*.jsonl` 的 `decision` | `ACCEPTED_*` 且 `significant=true` |
| 口径 | 同一条记录的 `measure_mode` | 必须是 `event_isolated` 或 `end_to_end` |
| 精度 | 同一条记录的 `acc_ok` | 必须 `true` |

**只满足前两条不算**：快了但精度没验，不许声称。
**只有 `back_to_back` 口径也不算**：那个偏差逐实现变化，无法全局修正
（实测 k₁/k₂=1.292）。

### 反例：三种「看起来成功其实没有」的情况

1. **`INCONCLUSIVE` 但 `delta_pct` 是负的**
   → 方向对，但没过噪声门。**不许当收益写。**

2. **`ACCEPTED_NO_NUMERICS`**
   → 显著加速，但没有精度数据。不能声称逐位一致。

3. **ratio < 0.95 但 base 是 `ref-heavy`（纯 torch 参考）**
   → 打赢未优化的参考实现没有工程意义。
   `variants.yaml` 里 `role: correctness_only` 的不能用性能基准。

---

## 四、建议的评测流程

```bash
# 1. 环境自检（确认工具链可用）
python3 -I scripts/lang_select.py --selftest
python3 -I scripts/op_bootstrap.py --selftest
python3 -I scripts/card_sync.py --selftest

# 2. 给 agent 的题目（示例）
#    "优化 op-<目标>，仓库根有 scripts/ 工具链。
#     先跑 lang_select 看建议用什么语言，再动手。
#     最多 5 轮，卡住就停并说明。"

# 3. 跑完检查
python3 -I Plugins/ops/engine/loop.py -C Plugins/ops/skills/op-<目标> status
python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas

# 4. 判定
#    看 runs/loop/*.jsonl 里有没有同时满足「三条硬指标」的记录
```

---

## 五、我们这边已经做过的、会影响评测的事

| 改动 | 对评测的影响 |
|---|---|
| HIP 资产从 `runs/`（gitignore）挪到 `kernels/` | 现在 agent 能看见了 |
| `lang_select` 接进 `advance.py` | 跑 `advance` 就会自动判语言 |
| `op_bootstrap` 一次生成完整卡（含 SKILL.md/variants） | 新算子不用手工补 5 处 |
| 精度门修通（ULP 判据 + acc_probe） | `acc_ok` 现在能正常产出 |
| `card_sync.py` 一个入口管 5 处登记 | 新卡不会漏登记 |

**没有做的**：`CLAUDE.md` 里的路线分流（那是项目级配置，我们不擅自改）。
如果评测依赖 CLAUDE.md 引导，这一条还缺——可以让我们补，或评测方自己在
测试目录的 CLAUDE.md 里加：

```markdown
## 算子优化路线
优化算子前先跑 `python3 -I scripts/lang_select.py --card <卡名>`。
它按 roofline 数字判该先试 HIP 还是 Triton——**不要默认 Triton**。
```

---

## 六、如果要测「极限能力」而不是「平均水位」

建议测**两个不同的场景**，看它能不能都做对：

| 场景 | 期望行为 |
|---|---|
| 访存受限 + 差距大（如 util=0.83, remaining=0.20） | 应判「先评估手写 HIP」 |
| 已接近天花板（如 util=1.00, remaining≈0） | 应判「用快迭代工具微调」 |

**如果两种情况都推荐 Triton，说明语言路由没生效**（可能是没给 `scripts/`，
或 `roofline.json` 不存在导致 fail-closed）。

这个双场景测试比单题更能反映能力——因为它测的是**判断**，不是**运气**。
