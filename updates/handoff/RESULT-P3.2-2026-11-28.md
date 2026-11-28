# RESULT · P3.2（真 backend：git + agent 子进程 + GPU eval）· 2026-11-28

> 执行者：work 会话。对象：`op-silu-mul`（仓库真实卡）+ 一个**不存在的算子**
> （`op-moe-gating-softmax`，在工作区里用 `op_bootstrap.py` 从零建出来）。

## 一、改了什么

| 文件 | 内容 | commit |
|---|---|---|
| `Plugins/ops/engine/driver.py` | `RealBackend`：整仓工作区副本 + git commit/reset + agent 子进程 + `loop.py run/claim` 测量；CLI `--init-workspace` / `--real` | 见文末 |
| `scripts/op_bootstrap.py`、`engine/templates/{ab,acc_probe}_generic.py` | 修模板路径 bug：`parents[3]/scripts` 实际指向 `Plugins/`，导致新卡 `opverify` 找不到。改成"向上找 `scripts/opverify.py`" | 同上 |

**过程中修掉的两个真 bug**
1. `RealBackend.eval` 用全局 glob 读账本，把**上一次 run** 的记录当成这一轮的结果
   （实测 silu 与 newop 读出同一组数字）。改成只读本次 run 的目录。
2. `claim` 失败分类：之前一律 `FAIL:claim`，导致 verdict 无法按计划区分原因。
   现在按失败项分 `FAIL:evidence` / `FAIL:caliber`。

## 二、验收命令的实际输出

### C2-a `op-silu-mul`（真实卡，2 轮）

```bash
cd .artifacts/driver/op-silu-mul_p32/workspace
python3 -I Plugins/ops/engine/driver.py --real --workspace . \
  --card-rel Plugins/ops/skills/op-silu-mul --tag fused-single-pass \
  --agent-cmd "python3 /tmp/agent_silu.py" --demo 2 --gpu 0 --out .../silu_iter.jsonl
```

```
r1  REVERT  cand=0.9161589741706848 base=0.918878972530365 claim=FAIL:evidence
r2  REVERT  cand=0.9126399755477905 base=0.9134399890899658 claim=FAIL:evidence  stop=rounds_done
```

记录字段：

```
r1 REVERT commit=5264860b claim=FAIL:evidence cand=0.9162 base=0.9189
r2 REVERT commit=fe5877e8 claim=FAIL:evidence cand=0.9126 base=0.9134
fields ok
```

### C2-b 不存在的算子（工作区里从零建卡）

`op_bootstrap.py --name moe-gating-softmax ...` 建卡后，agent 补 `build_inputs` /
`reference` 并实现候选 `fused-topk`，然后跑 2 轮：

```
r1  REVERT  cand=0.3420799970626831 base=0.34303998947143555 claim=FAIL:evidence
r2  REVERT  cand=0.33904001116752625 base=0.3403190076351166 claim=FAIL:evidence  stop=rounds_done
```

```
r1 REVERT commit=de41a3ac claim=FAIL:evidence cand=0.3421 base=0.343
r2 REVERT commit=3502b0c9 claim=FAIL:evidence cand=0.339 base=0.3403
```

### workspace git 历史（每轮一个 commit）

```
3502b0c [agent] 补 reference/build_inputs + 实现 fused-topk
de41a3a [agent] 补 reference/build_inputs + 实现 fused-topk
fe5877e [agent] 实现 fused-single-pass：单表达式融合 silu(x1)*x2，注册进 VARIANTS/get_runner
5264860 [agent] 实现 fused-single-pass：单表达式融合 silu(x1)*x2，注册进 VARIANTS/get_runner
9d9afc1 baseline
9c4b5c4 initial
```

## 三、没做到的 / 有疑问的

1. **agent 是脚本，不是真 claude**。`--agent-cmd "python3 /tmp/agent_*.py"` 验证的是
   git/GPU/claim 这条接缝；真 claude 子进程（带 `su - opb_c`、DTK 环境、skip-permissions）
   还没接。接之前需要你确认用哪个端点/模型与每轮预算。
2. **两轮都是 REVERT，没有 KEEP**。原因是这两张卡都没有独立核验（只有自写 reference），
   `claim` 第五项按策略 FAIL → 计划规定 `FAIL:evidence` → REVERT。要让 KEEP 出现，
   需要 vendor 变体（如 lightop/aiter）或第二份独立实现——`op-fused-add-rmsnorm` 有，
   但用户指定的是 silu-mul 与新算子。
3. **`changed_files` 只是 git show 的文件名列表**，没有 diff 内容；P3.4 的反馈渲染
   可以再加"diff 摘要"。
4. **工作区是整仓副本（43MB）**，每轮 commit 整个仓库；真实使用应换成
   `git worktree` 或只保留卡目录 + 依赖的脚本，避免历史膨胀。
5. **没有 P4 的 sandbox/deny**：agent 脚本理论上可以自己跑 bench。当前靠 prompt +
   计划里的后续 P4 边界；按 ForgeKernel 的经验，这块必须由 sandbox 兜底，不能靠提示。

---

## 四、修后复验（回应 REVIEW-P3.2：0 KEEP / 4 REVERT）

**根因**：`RealBackend.eval` 不生成 judge → 无 `tolerance.json` → claim 第五项
FAIL:evidence → 每轮 REVERT。**修法**：eval 开头调用 `judge.py generate` 到本次 run
的 artifact 目录（幂等），并补了候选 id → 变体名的 claim 回退。

### 复验命令（op-fused-add-rmsnorm + reg-resident）

```bash
cd .artifacts/driver/op-fused_p32/workspace
python3 -I Plugins/ops/engine/driver.py --real --workspace . \
  --card-rel Plugins/ops/skills/op-fused-add-rmsnorm --tag reg-resident \
  --agent-cmd "echo noop" --demo 2 --gpu 0 --out .../fused_iter.jsonl
```

输出：

```
r1  KEEP    cand=0.14336000382900238 base=0.2531200051307678 claim=PASS
r2  NEUTRAL cand=0.14319999516010284 base=0.14351999759674072 claim=PASS  stop=rounds_done
```

记录：

```
r1 KEEP    commit=e2ecc585 claim=PASS vs_base=0.566 cand=0.1434 base=0.2531
r2 NEUTRAL commit=c6fcabe3 claim=PASS vs_base=0.998 cand=0.1432 base=0.1435
counts: {'KEEP': 1, 'NEUTRAL': 1}
```

`base.json`（自进化链：第 1 轮被接受的实现成为第 2 轮的参照）：

```json
{"base":"fa-rms-mr2","ms":0.14336,"source_tag":"fa-rms-mr2",
 "evidence_kind":"vendor_library","ts":"2026-11-28T14:02:35"}
```

**闭环证据**：KEEP ≥1；claim PASS；第 2 轮 base 自动变为第 1 轮的 best（vs_base=0.998）。

### 仍存在的限制

1. 真 claude agent 仍未接（`--agent-cmd "echo noop"`），下步接。
2. silu-mul / 新算子那两张卡仍会 REVERT —— 它们没有 vendor 变体，
   `claim` 按策略必须 FAIL:evidence；这是判据的正确行为，不是 bug。
