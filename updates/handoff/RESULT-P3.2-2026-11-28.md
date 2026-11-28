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

---

## 五、复审修复（REVIEW-P3.2b：KEEP 是假阳性）

**根因**：agent `change="noop"`、`changed_files=[]` 却判 KEEP —— 把卡里既有的
`fa-rms-mr2` 与一个慢基线比了一次，还把它晋升成新基线。

**修法**
1. `INVALID_NOOP` 守卫：`changed_files` 里没有任何 `Plugins/ops/skills/**` 的改动
   → 判 `INVALID_NOOP`，不 KEEP、不 REVERT、**不晋升**，连续 1 次即停（`stop_reason=noop`）。
2. `changed_files` 过滤：`memory/`、`results/`、`.artifacts/`、`Plugins/ops/op-atlas/`
   都是引擎产物，不计为 agent 改动。
3. `loop.py` 增加 `OPS_NO_PROMOTE=1` 守卫；driver 在 noop 轮自动设置。

**复验（echo noop，op-fused）**

```
r1  INVALID_NOOP cand=0.14368 base=0.252 claim=PASS  stop=noop
→ 1 条记录；base.json 不存在（符合"Noop 不晋升"）
```

单测新增 2 条（noop / 引擎产物过滤），共 8 条全绿。

---

## 六、真 claude agent 的一轮（回应 REVIEW-P3.2b 的 (c)）

**做法**：`su - opb_c -c 'bash /tmp/agent_claude.sh'`（DTK + tsoc 环境 +
`claude -p --dangerously-skip-permissions`），在 `op-fused-add-rmsnorm` 上实现
候选 `fused-mr4`（编译 `kernels/mr4.so`）。

**结果**

```
r1  KEEP  cand=0.1441590040922165 base=0.24991999566555023 claim=PASS  stop=rounds_done
记录：change="[agent-timeout after 1200s]"
      changed_files=["Plugins/ops/skills/op-fused-add-rmsnorm/kernels/build.sh"]
```

- agent 在超时前**确实完成了实现**：`mr4.so` 已编译（22.8KB），并写下 `kernels/build.sh`；
  引擎随后用新变体测量 → `-42.3%` → claim PASS → KEEP。
- **D1 检查**：agent 会话 38 次工具调用，命中的 3 条 "bench" 全是 `ls`/`cat`
  （列目录、读文件），**没有任何 `python ... ab.py` / `acc_probe` / `cuda.Event` 执行**。
- 该轮 `change` 是 `[agent-timeout ...]`，因为 claude 在 1200s 上限被切、没来得及返回
  自述；但文件改动与测量都在账本里，判定不依赖自述。

### 新增的两个健壮性修复

1. `run_agent` 捕获 `TimeoutExpired`，返回 `[agent-timeout ...]` 而不是炸掉 driver；
2. 工作区权限：仓库根 751（opb_c 可穿行；`engine/` 仍 750，实测读不到），
   git 全部加 `-c safe.directory=*`（工作区 chown 给 agent 后 root 跑 git 会报
   dubious ownership）。

### 仍存在的限制

- **agent 慢**：1200s 只够它编译 + 写 build.sh，没进入"回报改了什么"。建议正式跑
  用 1800–2400s/轮，或把 prompt 收窄到"只做这一件事"。
- 真 claude 的会话里没有 bench 执行，但这仍是**靠 prompt 约束**；P4 的 sandbox/deny
  落地前，不能声称"物理上不可能自己测"。

---

## 七、复审 P3.2c 修复

### ① 记录落库（阻断项）

`--out` 不传时默认写到本次 run 的 artifact 账本目录：
`<artifact>/runs/loop/iterate-<ts>.jsonl`（仓库内、gitignored）。

实测（noop 轮，不传 --out）：

```
r1 INVALID_NOOP ... stop=noop
→ .artifacts/driver/op-fused-add-rmsnorm_p32/runs/loop/iterate-20261128_170447.jsonl
```

真 claude 那一轮的 KEEP 记录也从 `/tmp` 挪进了
`.artifacts/driver/records/iterate-20261128_real-mr4.jsonl`（内容未改）。
删除了修复前产生假阳性的 `fused_iter.jsonl`（已在 §五 记录）。

验收 R1/R2/R3：
```
R1  find .artifacts -name 'iterate-*.jsonl' | wc -l        → 2
R2  KEEP 且 changed_files 含 Plugins/ops/skills            → 1
R3  change=noop 且 verdict=KEEP                            → 0
```

### ② 这一轮 KEEP 的含金量（写进口径，不夸大）

1. **变体是预声明的**：`fa-rms-mr4` 本来就在
   `_common.HIP_VARIANTS = {"fa-rms-mr2": (...), "fa-rms-mr4": ("mr4.so", ...)}`。
   agent 做的是**把已声明的变体编译出来**（`mr4.so` + `build.sh`），
   **不是提出/实现一个新优化**。这是实打实的工作，但证据强度低于"新变体"。
2. **agent 超时**：1200s 在回报"改了什么"之前被切；正式跑建议 1800–2400s/轮。
3. **基线偏慢**：本轮 `base=0.2499`，而这张卡 lightop 历史实测是 `0.2030`（差 23%）。
   `vs_base=0.5768` 是**同一 job 内 back-to-back** 的比值，但 base 选的是本卡
   `defaults.bench_args` 里的 lightop；正式报告要同时给出 lightop 的独立复测值。

### ③ 测试

`test_claim_gate`（8 条）与 `test_loop`（7 条）的失败根因是**夹具记录没有
`evidence_kind`**（P1 之后 claim 第五项按证据种类判）。修法是给夹具补
`evidence_kind="vendor_library"`，**没有放宽判据**。

```
pytest test_claim_gate.py test_loop.py -q  → 96 passed
```
全量仍在跑；目标回到 13 failed（剩余为 loop_uniqueness 等既有问题）。

### ③ 测试（续）—— 从 18 红降到目标 13

- `test_claim_gate` ×8 / `test_loop` ×7：夹具补 `evidence_kind=vendor_library`（不放宽判据）。
- `test_loop_uniqueness` ×3：
  1. 扫描把 `.artifacts/driver/.../workspace`（整仓副本）当成第二份实现
     → 扫描器排除 `.artifacts/` 与 `runs/`；
  2. `driver.py` 自己定义了 `MAX_IMPL_FAILS`（预算常量重复）
     → 改为从 `engine/loop.py` 引用，不再复制。
- `test_driver_fake` 随之改用常量 3。

```
pytest test_loop_uniqueness.py test_driver_fake.py test_claim_gate.py test_loop.py -q
→ 112 passed
```
全量目标：13 failed（剩余为 card_index / doc_links / evidence / ingest /
ops_gate / refute 等既有问题，非本次改动引入）。
