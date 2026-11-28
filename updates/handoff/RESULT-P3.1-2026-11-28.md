# RESULT · P3.1（fake backend + 控制流单测）· 2026-11-28

> 执行者：work 会话。对应 `HANDOFF-2026-11-28-P3-driver-plan.md` §二 P3.1。
> **全程不碰 GPU、不调 claude**（计划硬要求）。

## 一、改了什么

| 文件 | 内容 | commit |
|---|---|---|
| `Plugins/ops/engine/driver.py` | `iterate()` 控制流：render → run_agent → commit → eval → verdict → record；`FakeBackend`（确定性、可配 flat / fail_every / green_from / times）；`render_student_prompt()`；CLI `--demo` / `--render-demo` | 见文末 |
| `Plugins/ops/tests/test_driver_fake.py` | 6 个单测：完整 3 轮、relaxed-revert（性能回退保留 / 正确性破坏回滚）、pre-green、plateau、impl_fails、结构化反馈 | 同上 |

## 二、验收命令的实际输出

### C1-1 单测

```bash
python3 -m pytest Plugins/ops/tests/test_driver_fake.py -q
```

```
......                                                                   [100%]
6 passed in 0.03s
```

覆盖的语义：
- `test_three_rounds_produce_complete_records`：3 条记录，`workspace_commit` / `claim` 非空
- `test_relaxed_revert_keeps_perf_regression_but_rolls_back_correctness`：
  变慢但正确 → `NEUTRAL` 且 `reset_calls == []`；正确性破坏 → `REVERT` 且回滚到上一轮 head
- `test_pre_green_is_neutral_until_first_correctness_pass`：前两轮 `NEUTRAL`，第三轮才 `KEEP`
- `test_plateau_stops_at_noise_floor`：每轮 0.5% 改善 < 1.5% → ≤4 轮停，`stop_reason=plateau`
- `test_impl_fails_stops_after_consecutive_correctness_breaks`：连续 3 次破坏 → `impl_fails`
- `test_prompt_contains_structured_feedback_and_refuted_directions`：prompt 含
  `verdict / vs_base / refuted_directions / claim` 与死路清单

### C1-2 demo

```bash
python3 -I Plugins/ops/engine/driver.py --demo 3 --out /tmp/demo.jsonl
python3 -c "import json;rs=[json.loads(l) for l in open('/tmp/demo.jsonl')];print(len(rs));assert all(r['workspace_commit'] and r['claim'] for r in rs)"
```

```
r1  KEEP    cand=0.160 base=0.200 claim=PASS
r2  KEEP    cand=0.128 base=0.200 claim=PASS
r3  KEEP    cand=0.102 base=0.200 claim=PASS  stop=rounds_done
→ /tmp/demo.jsonl（3 条）
rc=0
3
fields ok
```

## 三、没做到的 / 有疑问的

1. **verdict 只按"同 job 比值 vs 噪声底"**，没有 ForgeKernel 的 `REJECTED_*` 语义映射
   （例如 claim FAIL 但 correctness 通过时目前记 NEUTRAL，不额外 REVERT）。计划里
   P3.2 的规则是"claim FAIL 且失败项是 accuracy → REVERT"，本实现尚未接 claim 文本解析
   （fake 的 claim 是字符串），P3.2 接真 claim 时要补。
2. **`changed_files` 由 backend.eval 提供**（fake 里为空）；P3.2 的 `GitBackend` 要从
   `git diff --name-only` 填。
3. **`refuted_directions` 目前取 eval 返回值**；真实来源应是 `candidates.yaml` 的
   `refuted` + 方向预算，P3.2 再接。
4. **没有 P3.2 的真 backend**（Claude / GPU / git）。按计划，C1 过了才允许接真 agent。
5. **plateau 的噪声底固定 1.5%**，没有按"同代 ratios spread"动态算（P3.3 的活）。
