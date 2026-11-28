# REVIEW-P3.2 · 2026-11-28 · lead 验收结论

验收对象：commit `3a82e97` + `RESULT-P3.2-2026-11-28.md`
验收方式：lead 独立读代码 + 读记录 + 复跑命令。

## 判定：**机件通过，但闭环没闭 —— KEEP 一次都没出现**

`grep '"verdict"' .artifacts/driver/*.jsonl | sort | uniq -c` →

```
4  "verdict": "REVERT"
0  "verdict": "KEEP"
```

**循环在转，但永远不"接受"任何结果。** 这不是调参问题，是一条接线漏了。

---

## 一、根因（一条链，三行代码）

```
① loop.py::load_tolerance_json(card)
     读 <card>/judge/tolerance.json  或  $JUDGE_TOLERANCE
② 但 RealBackend.eval 从不生成 judge
     → 没有 tolerance.json
③ → claim 第五项判 FAIL:evidence
④ → _verdict() 里 "claim.startswith(('FAIL:accuracy','FAIL:evidence'))" → REVERT
```

所以每轮必然 REVERT。**修法：`RealBackend.eval` 在跑 run 之前先确保 judge 存在**：

```python
def _ensure_judge(self, card_rel: Path) -> None:
    jd = card_rel / "judge"
    if (jd / "tolerance.json").is_file():
        return                                    # 幂等：已有就不重生成
    subprocess.run([sys.executable, "-I",
                    str(self.ws / "Plugins/ops/engine/judge.py"), "generate",
                    "--card", str(card_rel), "--out", str(jd)],
                   cwd=str(self.ws), env=self.env, check=False)
```

一行调用，放在 `eval()` 开头。

## 二、第二个问题：验证用的卡选错了

你验证用的两张卡：

| 卡 | variants 的 role | 能否出现 KEEP |
|---|---|---|
| `op-silu-mul` | `['correctness_only']` | **永远不能** —— 没有独立核验，claim 注定 FAIL:evidence |
| `moe-gating-softmax`（工作区新建） | 同上 | **永远不能** |
| **`op-fused-add-rmsnorm`** | `['correctness_only','competitor','competitor','candidate','candidate']` | **能** —— lightop/aiter 是 vendor，claim 可 PASS |

我在 P1 验收时已经用它验证过 `claim PASS`：

```
base.json = {"base":"fa-rms-mr2","ms":0.1416,"evidence_kind":"vendor_library",…}
```

**所以：请在 `op-fused-add-rmsnorm` + `--tag reg-resident` 上跑 2 轮**，
应该能看到 `delta ≈ -41%` + `claim PASS` → **KEEP**。那才是"闭环闭了"的证据。

## 三、做对了的部分（复跑确认）

| 项 | 状态 |
|---|---|
| `IterationRecord` 17 字段，含 `vs_base`（同 job 比值）/ `claim` / `workspace_commit` / `changed_files` | ✅ 实测记录里都有值 |
| `_verdict()` relaxed-revert 语义 | ✅ 逻辑正确（性能回退 NEUTRAL，正确性/证据破坏才 REVERT） |
| 每轮 git commit（`[agent] …`） | ✅ 工作区 git log 看得到 6 个 commit |
| plateau 计数器（`MAX_PLATEAU=3`） | ✅ 代码在 |
| 你修的两个真 bug | ✅ 都实锤：①全局 glob 读账本会把上一轮记录当本轮；②claim 失败分类 `FAIL:evidence`/`FAIL:caliber` |

## 四、还没做（按优先级）

1. **接线：judge → tolerance**（上面第一节，1 行）← 阻塞 KEEP
2. **换卡验证 KEEP**（op-fused-add-rmsnorm + reg-resident）
3. **agent 从脚本换成真 claude**：`--agent-cmd` 目前指向 `/tmp/agent_*.py`。
   接真模型前需确认：端点/模型、每轮 wall-clock 上限、失败重试次数。
4. **`changed_files` 只有文件名没有 diff 摘要** → P3.4 反馈渲染再加。
5. **workspace 是整仓副本（43MB）/ 每轮 commit 全仓** → 建议 `git worktree` 或只保留卡目录。
6. **无 P4 边界**：agent 脚本理论上可以自己跑 bench。这必须靠 sandbox 兜底，不能靠 prompt。

## 五、下一步（照这个顺序，别跳）

```
P3.2-fix  接线 judge→tolerance，并在 op-fused-add-rmsnorm 上跑出 1 条 KEEP
P3.3      噪声底动态化（同代 spread，不低于 1.5%）+ plateau 真停
P3.4      反馈渲染（结构化记录 + diff 摘要 + 死路清单）
P3.5      重试/心跳
P3.6      真 claude 接进来（要业主确认端点与预算）
```

**验收命令（我下次会重跑这些）**：

```bash
cd /workspace/tookit-skills
# K1：闭环出现 KEEP
grep -h '"verdict"' .artifacts/driver/*.jsonl | grep -c KEEP      # 期望 ≥1
# K2：那一条的 claim 是 PASS
grep -h '"verdict": "KEEP"' .artifacts/driver/*.jsonl | grep -c '"claim": "PASS"'   # 期望 ≥1
# K3：judge 被自动生成（不是手工设环境变量）
ls .artifacts/driver/*/workspace/Plugins/ops/skills/op-fused-add-rmsnorm/judge/tolerance.json
```
