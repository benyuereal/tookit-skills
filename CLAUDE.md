# toolkit_skills —— 算子优化引擎

## 第一件事：做算子优化，从这里进

**唯一入口：`Plugins/ops/skills/op-optimize`**

```
op-optimize（定口径 → 复现基线 → 定位瓶颈 → 列候选 → 单实验 → 记账）
   ├─ 该用什么语言？        → python3 -I scripts/lang_select.py --card <卡名>
   ├─ 天花板/地板在哪？      → op-roofline / python3 -I scripts/floor.py
   ├─ PMC 怎么归因？        → op-profiling
   ├─ 手写 HIP 怎么编译调优？ → op-hip-compile
   ├─ 新算子从零建卡        → python3 -I scripts/op_bootstrap.py
   └─ 多轮自动推进          → python3 -I scripts/advance.py --card <卡> --forge
```

**不要用 `dcu-orchestrator` 的 `op-optimize` 工作流代替它。**
那个是「项目级编排」（profile→roofline→constrain→optimize→validate→distill），
管的是「先做什么后做什么」；本仓库的 `op-optimize` 是**执行层**，
管「这一步具体怎么跑」。两者不是同一条链，跑起来会各算各的账。

---

## 三条硬规矩（跳过就等于绕过门禁）

1. **实验只经 `Plugins/ops/engine/loop.py run`**，不直接调 `bench/ab.py`
   —— 那六道门（登记/归因/完成判定/方向预算/已证伪/天花板）会被全绕过。

2. **任何「快了多少」必须同批 base 测量 + `measure_mode=event_isolated`**
   —— 孤立单跑的数字会混进机器漂移。实测过一次「快 24%」，
   同批复测只有 1.7%，归因也是错的。

3. **新卡登记用 `scripts/card_sync.py`**，别手工补 5 个位置
   —— 卡片 / registry.json / INDEX.md / ops/SKILL.md / gaps.md。

---

## 新算子从零开始的六步（第 3 步不能省）

```bash
# ① 建卡
python3 -I scripts/op_bootstrap.py --name <算子> --family <族> \
  --shape "M=8192,H=4096" --inputs "x:bf16[M,H]" --outputs "y:bf16[M,H]" \
  --pattern <访问模式>

# ② 补 reference —— 不能跳过。参考实现是正确性的唯一锚，
#    bootstrap 故意留 NotImplementedError（猜一个会让门禁「通过」算错的实现）。

# ③ 跑一次基线，拿到实测耗时 t_actual
#    没有它，语言路由会输出「无规则命中」——不知道该多快就不知道该往哪走。

# ④ 出 roofline，看语言建议
python3 -I scripts/lang_select.py --card <算子> --shape s0

# ⑤ 跑循环
python3 -I scripts/advance.py --card Plugins/ops/skills/op-<算子> --max-rounds 3 --forge

# ⑥ 登记
python3 -I scripts/card_sync.py --card <算子>
```

---

## 怎么判断有没有做出东西

三条同时成立才算：

| 指标 | 在哪看 | 合格线 |
|---|---|---|
| 显著性 | `<卡>/runs/loop/*.jsonl` 的 `decision` | `ACCEPTED_*` 且 `significant=true` |
| 口径 | 同条记录的 `measure_mode` | `event_isolated` 或 `end_to_end` |
| 精度 | 同条记录的 `acc_ok` | `true` |
| 相对基线 | 卡片 `last_verdict.comparison.ratio` | `< 0.95` |

**「快了但精度没验」不许声称**（`ACCEPTED_NO_NUMERICS`）。
**只有 `back_to_back` 口径也不算**（偏差逐实现变化，无全局修正因子）。

---

## 目录导航

| 我要… | 去哪 |
|---|---|
| 用这套东西做优化 | **本文件 + `Plugins/ops/skills/op-optimize/SKILL.md`** |
| 完整用法（三种驱动方式、常见坑） | `docs/HOW-TO-USE.md` |
| 架构边界（哪些代码属于引擎） | `docs/ENGINE.md` |
| 技术债（D1–D11，每条都有机器判据） | `docs/DEBT.md` |
| 历史决策为什么那么定 | `updates/engineering-iteration/roadmap-history/` |
| 交给评测方测 | `docs/OPBENCH-USAGE.md` |
| 其他 DCU 域（环境/推理/故障排查） | `AGENTS.md` |

---

## 开发这个仓库时

```bash
python3 -m pytest Plugins/ops -q                    # 全量（约 3 分钟）
python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas   # 卡片校验
python3 -I scripts/card_sync.py --check             # 5 个登记位置一致性
```

**改 schema 前先跑**：
```bash
python3 -I scripts/check_schema_fields.py <文件> --field <新字段名>
```
同一个错犯过三次（`comparison_note` / `downgrade_reason` / `status='draft'`）。
