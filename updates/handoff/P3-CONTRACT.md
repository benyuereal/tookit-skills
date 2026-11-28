# P3-CONTRACT · driver 契约（P3.0）

> 作者：work 会话。对应 `HANDOFF-2026-11-28-P3-driver-plan.md` §二 P3.0。
> 本文件只定义**接缝与语义**；实现在 `Plugins/ops/engine/driver.py`。

## 1. 两个类型

### `IterationRecord`（一轮的产物）

| 字段 | 含义 | 谁填 |
|---|---|---|
| `iter` / `round_id` | 轮次 / 唯一 id（resume 用） | driver |
| `change` | agent 自述「改了什么」，**不含数字** | agent 返回值 |
| `changed_files` | 本轮改动的文件（来自 git diff） | backend.eval |
| `workspace_commit` | 本轮 commit hash；**没有它就不能 revert** | backend.commit |
| `correctness` | `True` / `False` / `None`（pre-green 未评估） | backend.eval |
| `verdict` | `KEEP \| REVERT \| NEUTRAL \| REFUTED` | driver |
| `base_ms` / `cand_ms` / `vs_base` | 同 job back-to-back 的耗时与比值 | backend.eval |
| `utilization` / `floor_us` | 达成率与地板（取不到就 None） | backend.eval |
| `acc` | `max_ulp` / `cosine` / `evidence_kind` | backend.eval |
| `claim` | `PASS` 或 `FAIL:<reason>` | backend.eval（调 claim） |
| `refuted_directions` | 死路清单，喂回下一轮 | backend.eval |
| `noise_floor` | 本代噪声底（默认 1.5%） | driver |
| `stop_reason` | `plateau` / `impl_fails` / `rounds_done` | driver |

### `Backend`（副作用接缝）

```python
run_agent(prompt, role="student") -> str   # 只返回"改了什么"，不得解析数字
commit(message) -> str                     # 返回 commit hash
reset_to(commit) -> None                   # 只在正确性破坏时调用
eval(tag) -> dict                          # 跑测量 + claim，返回记录所需字段
```

**不变量**：
1. **控制流不直接碰 GPU / 网络 / git** —— 全部经 Backend，因此可以 fake 单测。
2. **agent 不产数字** —— `run_agent` 的返回值只进 `change`，任何数字字段都必须来自 `eval`。
3. **relaxed-revert** —— 性能回退**保留**（当脚手架）；只有 `correctness is False` 才 `reset_to`。
4. **pre-green** —— 第一次 `correctness is True` 之前，verdict 一律 `NEUTRAL`（不评估性能）。
5. **停止条件** —— 连续 `MAX_PLATEAU=3` 轮非 KEEP → `plateau`；连续 `MAX_IMPL_FAILS=3` 轮正确性破坏 → `impl_fails`。

## 2. 实现边界

| 阶段 | 内容 | 是否碰 GPU |
|---|---|---|
| **P3.0** | 本文件 + `driver.py` 的类型与 Protocol + `--selfcheck` | 否 |
| **P3.1** | `iterate()` 控制流 + `FakeBackend` + `tests/test_driver_fake.py` + `--demo` | 否 |
| P3.2 | `ClaudeBackend` / `GpuBackend`（复用 `loop.py run`）/ `GitBackend` + `loop.py iterate` | 是 |

## 3. 验收对应

- **C0（P3.0）**：`driver.py --selfcheck` rc=0；`grep -c "class Backend\|class IterationRecord"` = 2。
- **C1（P3.1）**：`pytest tests/test_driver_fake.py -q` 全绿；
  `driver.py --demo 3 --out /tmp/demo.jsonl` 产出 3 条，且 `workspace_commit` / `claim` 非空。
- **C1 之前的门槛**：**P3.1 不过，不许接真 agent**（计划原文）。
