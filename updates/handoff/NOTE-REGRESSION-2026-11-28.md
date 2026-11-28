# NOTE-REGRESSION · 2026-11-28 15:00 · 测试从 14 → 33（lead 已修 2，剩 18 是你的）

巡检时全仓 `pytest Plugins/ops -q` = **33 failed**。基线（P0 之前）是 13。

## 我修掉的 2 条（我的机制，我的锅）

`test_code_map` ×2：新模块没登记进代码地图。
已修：`scripts/gen_code_map.py` 补 5 条（judge/nextstep/driver/iterate/coordinate），
并把 `engine/` 的分层守卫白名单显式化（judge/driver/iterate 是合法的可运行工具）。
现在 `test_code_map.py` = 4 passed。

## 剩下 18 条是你的，分三类

### ① `test_loop_uniqueness` ×3 —— **`.artifacts/` 里放整仓副本，污染了 repo 级 glob**

```
AssertionError: Lists differ:
  ['.artifacts/driver/op-fused_p32/workspace/Plugins/ops/engine/driver.py',
   '.artifacts/driver/op-fused_p32/workspace/Plugins/ops/engine/loop.py', ... 6 个]
!= ['Plugins/ops/engine/loop.py']
```

`test_loop_uniqueness` 会全仓 glob `def cmd_run(`，期望只有一份。但
`.artifacts/driver/*/workspace/` 是**整仓副本**，里面当然也有 `loop.py`。

**这暴露的是设计问题，不只是测试问题**：把整仓副本放在仓库内 →
① 每轮 commit 整个仓库；② 任何 repo-wide 检查都会扫到副本；③ `.gitignore` 一旦漏了就进历史。

**修**（二选一，推荐 A）：
- **A. 工作区移到仓库外**（`/tmp/forge-ws/<op>_<id>/`），或 `git worktree`（共享对象库，不复制）；
- **B. 若必须留在仓内**：`.gitignore` 加 `.artifacts/`，且所有 repo-wide 检查排除 `.artifacts/`。

### ② `test_claim_gate` ×8 —— 新增的 `evidence_unverified` 打破既有夹具

```
['acc...','caliber_incomplete','evidence_unverified','not_significant']
!= ['acc...','caliber_incomplete','not_significant']
```

`claim_gate` 现在多了一条 evidence 判据，但**既有测试夹具没有 judge/tolerance**，
于是全部多出一个 reason。

**修**：给这些夹具补最小 evidence（或统一一个 `make_evidence()` 夹具），
**不要为了过测试把 evidence 判据删掉** —— 那是我们最核心的一条。

### ③ `test_loop` ×7 —— 需要你自己看

我没细查（不在我这次的巡检深度内）。但按前两类的模式，大概率是
`loop.py` 的行为被 P1/P2/P3 的改动影响（base 来源、账本路径、claim 接入）。

## 一句话

**测试变红 20 条，其中 18 条指向同一个根因：改动没有跑全仓回归。**
按计划铁律第 5 条（不许让测试变红），请先修到 13，再继续 P3.3。
