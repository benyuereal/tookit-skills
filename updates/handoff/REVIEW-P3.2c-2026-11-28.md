# REVIEW-P3.2c · 2026-11-28 15:50 · lead 复审（第二次复审后）

验收对象：commit `a06e5d2 feat(driver): 真 claude agent 一轮 —— KEEP 来自 agent 改动，D1 成立`

## 判定：**机制这次真的通了，但还不能作为「闭环证据」入账 —— 差三口气**

### 一、修对了的三件（我逐条验过）

| 项 | 证据 |
|---|---|
| **`INVALID_NOOP` 守卫生效** | `/tmp/noop_iter.jsonl` → `verdict=INVALID_NOOP, claim=PASS, stop=noop`，且不晋升 |
| **`changed_files` 过滤生效** | noop 轮 `changed_files=[]`（引擎产物已排除） |
| **真 claude agent 跑通一轮** | `/tmp/real_iter.jsonl` → `KEEP, claim=PASS, base=0.2499 cand=0.1442 vs_base=0.5768`，`changed_files=['.../kernels/build.sh']` |

### 二、D1 我亲自查了 —— **成立**

driver 拉起的真 claude agent（`/home/opb_c/.claude/projects/-workspace-tookit-skills--artifacts-driver-*/`）：

```
会话 69d3c7ce：46 次 Bash，含 bench 字样：只读 0 / 真执行 0
会话 21759bcb：57 次 Bash，含 bench 字样：只读 1 / 真执行 0（那条是 cat/sed 读 ab.py）
```

**agent 全程没有执行过任何 benchmark。** 现场物证：`kernels/mr4.so`（22,872B，15:17）、
`kernels/build.sh`（15:15）。**D1（交付里不许有 agent 自测的数字）成立。**

## 三、三口气（第一条必须补，后两条要写进报告的口径）

### ① 记录必须落库（**阻断项**）

现在证据在 `/tmp/real_iter.jsonl` —— **临时目录**。
`/tmp` 一清、机器一换，这条 KEEP 就没了，任何人都无法复验。

**要求**：`--out` 默认指向 `<artifact>/runs/loop/iterate-<ts>.jsonl`，
且该路径在仓库内（`.artifacts/` 已 gitignore）。

### ② 这一轮的 KEEP 含金量有限（**必须在报告里写明**）

1. **变体是预声明的**：`fused-mr4` / `fa-rms-mr4` 本来就在卡里
   `HIP_VARIANTS = {"fa-rms-mr2": (...), "fa-rms-mr4": ("mr4.so", ...)}`。
   agent 做的是**把已声明的变体编译出来**，不是提出/实现一个新优化。
   —— 这是**实打实的工作**（编译 + 写 build.sh），但不是我 review 时说的
   "实现一个新变体"。证据强度不同，报告里不能混。
2. **agent 超时**：`change="[agent-timeout after 1200s]"`。它连"改了什么"都没自述完，
   1200s 只够编译。正式跑建议 1800–2400s/轮，或把 prompt 收窄。
3. **基线偏慢**：`base=0.2499`，而这张卡 lightop 的历史实测是 **0.2030**（差 23%）。
   -42.3% 里有多少来自"基线飘低"，需要**同一轮内 back-to-back 重测**确认。
   （我们 `IterationRecord` 已经有 `vs_base` 是同一 job 内测的 —— 但 base 选的是谁要说清。）

### ③ 测试还是 31 failed（**他 18 条一条没修**）

```
test_claim_gate ×8 / test_loop ×7 / test_loop_uniqueness ×3   ← 都是你的
（我修的 test_code_map ×2 已绿）
```

计划铁律第 5 条：不许让测试变红。**这条不修，后面所有"闭环 N 轮"的数字都不能进报告。**

## 四、下一步（按顺序）

```
1. 记录落库：--out 默认进 <artifact>/runs/loop/
2. 修 18 条测试 → 回到 13
3. 一轮"真·新变体"：让 agent 实现一个**不预先存在**的候选（例如新的 tile 切法），
   而不是编译已声明的 mr4
4. 基线与候选同轮 back-to-back，报告里写清 base 是谁
```

**验收（我下次会跑）**

```bash
# R1 记录在仓库里
find .artifacts -name 'iterate-*.jsonl' | grep -c .          # ≥1
# R2 KEEP 且 changed_files 含 skills
grep -rh '"verdict": "KEEP"' .artifacts --include='*.jsonl' | grep -c 'Plugins/ops/skills'
# R3 无 noop 却 KEEP
grep -rh '"change": "noop"' .artifacts --include='*.jsonl' | grep -c KEEP   # 期望 0
# R4 测试回到基线
python3 -m pytest Plugins/ops -q 2>&1 | tail -1                # 期望 13 failed
```
