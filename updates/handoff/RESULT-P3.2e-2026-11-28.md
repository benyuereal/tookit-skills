# RESULT · P3.2e（多轮验收 + 结构化输出 S1/S4）· 2026-11-28

## 一、多轮验收（修后代码，3 个未实现候选）

运行：`op-fused-add-rmsnorm`，`fused-r1/r2/r3`，1200s/轮，`--demo 3`。
记录：`.artifacts/driver/records/iterate-20261128_223634.jsonl`（逐轮追加）。

| iter | verdict | claim | cand | base | correctness | 代码改动 |
|---|---|---|---|---|---|---|
| 1 | **KEEP** | PASS | 0.1445 | 0.2566 | True | `_common.py` + `kernels/fa-rms-r1.hip` |
| 2 | NEUTRAL | PASS | 0.1446 | 0.1424 | True | `_common.py` + `kernels/fa-rms-r2.hip` |
| 3 | NEUTRAL | PASS | 0.1606 | 0.1422 | True | `_common.py` + `kernels/fa-rms-r3.hip` |

`stop_reason=rounds_done`；3 条记录，逐轮落盘。

### 验收对照

| 项 | 结果 |
|---|---|
| (a) 无 correctness=False 而 acc ok | ✅ 三轮全 True |
| (b) NO_WORK 后不 spawn agent | ✅ 单测 `test_queue_empty_is_no_work_without_spawning_agent`（本机队列未耗尽，未触发） |
| (c) 记录 ≥3 | ✅ 3 |
| (d) ≥1 KEEP 且 changed_files 含代码文件 | ✅ r1 KEEP，含 `_common.py` |

第 2 轮正是旧 bug 的场景（`REJECTED_NO_GAIN`）：现在是 `correctness=True` + NEUTRAL，
不再被误判 REVERT。agent 效率也从 47 turns/463s 降到 23 turns/166s（system prompt + 预算）。

## 二、结构化输出 S1 + S4（AUDIT-STRUCTURE）

- **S1**：`claim --json` → `{tag, found, verdict, exit_code, checks:[{key,name,passed,why}], reasons:[key]}`
  - `backends._claim_for` 与 `iterate.run_claim` 只读 JSON；
    消费者侧文案解析点 grep = **0**
  - `loop.py` 的人类可读输出保留原文案（那是对外文本，不是解析）
- **S4**：`_implemented_variants` 用 `ast` 读 `VARIANTS` / `HIP_VARIANTS` 字面量（不再正则）
- 141 条相关测试全绿（loader/nextstep/loop/claim_gate/ablate/driver）

## 三、其他已落地（同步提交）

- `_loader.py`：10 处按路径加载合并为唯一实现；失败模块不再留在 `sys.modules`
- `card.load_candidates`：按 (path, mtime, size) 缓存
- 渲染模板化：`write_report`/`cmd_report`（17 golden）、`tool_probe`/`atlas_query`、
  `variant_forge`/`bw_calibrate`、`op_bootstrap`/`gen_tool_registry`、`calibrate`/`self_iteration`
- 全局「实现者契约」system prompt + `instrument/claude_agent.sh`
- 命令模板（playbook）：`instrument.command` + `commands.yaml`

## 四、仍未做

- S2 `run --json`、S3 `status --json`、S5 `docs/ENGINE.md` 契约表
- `except Exception` 的"静默降级"清单（只加 lint，不改行为）
- 消融流水线接线（`--task-kind ablate`、spec 落盘、调用 `scripts/ablate.py`）
