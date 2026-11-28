# RESULT · P3.0（契约先行）· 2026-11-28

> 执行者：work 会话。对应 `HANDOFF-2026-11-28-P3-driver-plan.md` §二 P3.0。

## 一、改了什么

| 文件 | 内容 | commit |
|---|---|---|
| `updates/handoff/P3-CONTRACT.md` | 新增：`IterationRecord` 字段表、`Backend` 四方法、五条不变量（不产数字 / relaxed-revert / pre-green / 停止条件 / 控制流与副作用分离） | 见文末 |
| `Plugins/ops/engine/driver.py` | 新增：`@dataclass IterationRecord`（17 字段）+ `class Backend(Protocol)`（run_agent/commit/reset_to/eval）+ `--selfcheck`。**只有类型与 docstring，控制流在 P3.1 同一文件内**（计划允许） | 同上 |

## 二、验收命令的实际输出

### C0

```bash
python3 -I Plugins/ops/engine/driver.py --selfcheck ; echo "rc=$?"
grep -c "class Backend\|class IterationRecord" Plugins/ops/engine/driver.py
ls updates/handoff/P3-CONTRACT.md
```

```
✓ selfcheck: IterationRecord(17 字段) + Backend(4 方法) 自洽
rc=0
2
updates/handoff/P3-CONTRACT.md
```

## 三、没做到的 / 有疑问的

1. `Backend` 用 `typing.Protocol`（结构化子类型），**不是** ABC。好处是 fake 不必显式继承；
   坏处是 `isinstance` 不生效。`--selfcheck` 只能查方法存在性，查不到签名一致。P3.2 接真
   backend 时建议再加一个 `runtime_checkable` 的签名自检（或用 mypy）。
2. `IterationRecord.vs_base` 定义为"同 job back-to-back 比值"；P3.2 的 `GpuBackend` 必须
   从 `loop.py run` 的记录里取 `cand_ms/base_ms`，**不能用跨 job 平均**。
3. `stop_reason` 是后补字段（P3.3 用）；P3.0 的契约表里已列，但语义在 P3.3 才实现。
