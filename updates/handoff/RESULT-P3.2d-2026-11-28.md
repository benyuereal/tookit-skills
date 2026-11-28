# RESULT · P3.2d（correctness 语义 + NO_WORK）· 2026-11-28

> 回应 `UPDATE-MULTIROUND-2026-11-28-2220.md` 的两条 bug。提交 `3885a0c`。

## 一、Bug 1：correctness 把「没收益」当「算错了」

**现象**（iter=2）：`acc max_ulp=0.50 / cosine=0.9999986`，却被 `REJECTED_NO_GAIN`
映射成 `correctness=False` → `REVERT`，把 agent 的改动回滚了。

**修法**：判定 → correctness 只走一个映射：

```python
def correctness_of_decision(decision):
    d = str(decision or "")
    if not d: return False
    return d not in ("REJECTED_ACCURACY", "ERROR", "IMPL_BROKEN")
```

`REJECTED_NO_GAIN` / `INCONCLUSIVE` / `CALIBRATION_ONLY` / `ACCEPTED_*` 全部视为
"实现是对的"，交 `decide_verdict` 按 relaxed-revert 判 KEEP/NEUTRAL，**不回滚**。
`decision` 仍原样记在记录里，不丢信息。

单元测试（`test_no_gain_is_not_a_correctness_failure`）覆盖 7 种 decision。

## 二、Bug 2：队列空仍派 agent（NO_WORK）

**现象**：iter=2 的 agent 自述"本轮没有新 kernel 可写"，却仍被派了 20 分钟。

**修法**：
- 每轮**调度 agent 之前**调用 `backend.next_candidate()`：
  - 返回下一个 `status` 非 done/implemented 且 `variant` **未在 `_common.py` 登记**的候选；
  - 没有 → 记 `IterationRecord(verdict="NO_WORK", stop_reason="queue_empty")`… 直接停，
    **不启动 claude 子进程**，也不计入 `noop` / `impl_fails`。
- 有 → 用它作为本轮 prompt 的目标与 `loop.py run` 的 tag（**候选轮转**）。

单元测试（`test_queue_empty_is_no_work_without_spawning_agent`）断言
`be.agent_calls == []` —— 队列空时一个 agent 子进程都没起。

## 三、验收状态（对下一次真实运行）

| 项 | 状态 |
|---|---|
| (a) 没有 correctness=False 而 acc ok 的记录 | ✅ 映射已修，单测覆盖 |
| (b) NO_WORK 后没有 claude 子进程 | ✅ 代码级（调度前短路）+ 单测 |
| (c) 记录 ≥3 条 | ⏳ 下一次运行验收（已准备 3 个未实现候选） |
| (d) ≥1 KEEP 且 changed_files 含代码文件 | ⏳ 下一次运行验收 |

本提交前正在跑的 3 轮（`fused-mr8`）用的是旧代码，只作对照：
`r1 KEEP`（-36.8%，`_common.py`）、`r2 REVERT`（the bug in §一）。

## 四、下一次运行配置

- 卡：`op-fused-add-rmsnorm`，工作区 `.artifacts/driver/op-fused_p32/workspace`
- 候选：预置 3 个 **未实现** 变体（`fa-rms-r1/r2/r3`），每轮 `next_candidate` 取一个
- 预算：1200s/轮，`--demo 3`
- 通过条件：记录 ≥3、至少 1 条 KEEP（changed_files 含 `_common.py`）、无
  correctness=False 且 acc ok 的记录

## 五、没做的

- `NO_WORK` 之后自动调 `explore.py` 出题（现在只在记录里建议，没自动跑）。
- `next_candidate` 只看"variant 是否登记"，不看候选 `status=implementing` 的中间态。
