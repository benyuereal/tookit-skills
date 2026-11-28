<!--
report.md — loop.py 报告渲染模板（`write_report` 落盘 / `cmd_report` 打印）。

约定：`<!-- name -->` 单独一行开一个命名片段，片段内容是标记行之后、下一个标记行
之前；非最后一段会去掉紧邻下一个标记的那一个换行（片段之间的分隔换行），最后一段
（wr_main）原样读到文件末尾 —— 所以本文件**故意不以换行结尾**，`wr_main` 的
`{{next_section}}` 就是最后一个字节。需要片段以换行结尾时，在下一个标记前留一空行。
占位符语义与 `instrument.render` 相同（`{{key}}`，缺失渲染成空串）。
-->
<!-- wr_claimable -->
可 claim
<!-- wr_unclaimable -->
**不可 claim**
<!-- wr_significant -->
显著
<!-- wr_insignificant -->
在噪声内，不建议据此下结论
<!-- wr_floor -->

## 3. 有效流量与达成率

| 项 | 值 |
|---|---|
| 地板 | {{floor_us}} µs（来源 `{{floor_source}}`） |
| 实测 | {{actual_us}} µs |
| 达成率 | {{eff_pct}}% |
| 判定 | `{{gate}}` |

分母依据：{{basis}}

<!-- wr_acc_item -->
- qscale={{qscale}}：max_abs={{max_abs}}，max_ulp={{max_ulp}}，cosine={{cosine}}，nonfinite={{nonfinite}}
<!-- wr_acc_footer -->
- 容差：{{tolerance}}
- **acc_ok={{acc_ok}}**

<!-- wr_acc_missing -->
- ⚠ 没有精度探针结果（本卡缺 `bench/acc_probe.py`）—— 结论只能标「未验证精度」

<!-- wr_next -->

## 7. 下一步

引擎建议：`{{next_action}}` —— {{next_why}}

```bash
{{next_cmd}}
```

<!-- cr_empty -->
还没有记录（先跑 loop.py run <tag>）
<!-- cr_table -->
| SKV | tag | base | cand | Δ | vs fm | maxdiff | 判定 |
|---|---|---|---|---|---|---|---|
<!-- cr_row -->
| {{skv}} | {{tag}} | {{base}} | {{cand}} | {{delta}} | {{fm}} | {{maxdiff}} | {{tail}} |
<!-- cr_held -->

### 未达主张门槛（不计入交付结论）{{n}} 条
这些实验可以跑，但不能升级成「快了 x%%」。补齐后重跑 report。

<!-- cr_held_item -->
- {{tag}} @skv{{skv}}：{{reasons}}
<!-- cr_help -->

补齐方法：
  caliber_incomplete → bench/ab.py 的 JSON 顶层加 caliber{what,alloc,thermal,device}
  not_significant    → 加 --rounds（默认 3，建议 ≥5），让 MAD 收敛
  accuracy_missing   → 去掉 --no-acc 并确认 bench/acc_probe.py 能跑
<!-- wr_main -->
# {{card_name}} — 实测报告（由 loop.py 生成）

> 本文件由 `loop.py run` 用**实测数据**生成；判定与数字都可回溯到
> `{{card_name}}/runs/loop/*.jsonl`。

## 1. 落到的实现

- 变体名：`{{tag}}`
- 适配器：`{{card_name}}/bench/ab.py`（`_common.get_runner` 分派）
- 口径：`{{measure_mode}}`（{{claim_note}}）

## 2. 实测耗时

| 项 | 值 |
|---|---|
| 基线（base） | {{base_ms}} ms（MAD {{base_mad}}） |
| 本实现（cand） | {{cand_ms}} ms（MAD {{mad}}） |
| Δ | {{delta}}（{{sig_note}}） |
{{floor_section}}
## 4. 正确性

{{acc_block}}
## 5. 引擎判定

- `decision` = **{{decision}}**
- 理由：{{why}}

## 6. 复跑

```bash
{{cmdline}}
```
{{next_section}}