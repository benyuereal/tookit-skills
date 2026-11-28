# RESULT · P3.3（动态噪声底 + plateau）· 2026-11-28

## 一、改了什么

| 文件 | 内容 |
|---|---|
| `engine/driver.py` | `noise_floor_from(ratios, floor=0.015)`：同代 cand/base 比值的 spread 作噪声底，**不低于 1.5%**；`iterate()` 维护本代 ratios，每轮判定用动态 noise，并写进 `IterationRecord.noise_floor` |
| `tests/test_driver_fake.py` | 新增/调整 4 条：3 轮 flat 必须 plateau、spread 抬升地板、宽 spread 把增益变 NEUTRAL |

## 二、验收（C3）

```bash
python3 -I Plugins/ops/engine/driver.py --demo 5 --fake-flat --out /tmp/flat.jsonl
python3 -c "import json;rs=[json.loads(l) for l in open('/tmp/flat.jsonl')];print(rs[-1]['verdict'], rs[-1]['stop_reason'])"
```

```
r1  NEUTRAL cand=0.199 base=0.200 claim=PASS
r2  NEUTRAL cand=0.199 base=0.200 claim=PASS
r3  NEUTRAL cand=0.199 base=0.200 claim=PASS  stop=plateau
last NEUTRAL plateau rounds 3
```

单测：`pytest test_driver_fake.py -q → 10 passed`。

## 三、语义与限制

- **"同代"定义**：本代已经产生的 `cand/base` 比值；第一轮只有地板（1.5%）。
- **spread 用单次测量**：每轮只有一条记录，不是 ForgeKernel 的"同 job 重复 N 次"。
  因此 spread 反映的是**轮间波动**，不是单轮的测量噪声。要更严格需要每轮重复测量
  （P3.4/后续加 `--repeat`）。
- **plateau 计数**：连续 `MAX_PLATEAU=3` 轮非 KEEP 即停；`noop` 与 `impl_fails`
  各自有独立停止条件。
