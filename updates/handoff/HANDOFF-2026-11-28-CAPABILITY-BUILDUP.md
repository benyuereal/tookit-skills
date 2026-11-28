# HANDOFF · 2026-11-28 · 自主迭代能力补强计划（A→F）

> 基线评估见 §0。执行者：work 会话；验收者：lead。
> 上一份：`HANDOFF-2026-11-28-P3-driver-plan.md`（P3 已完成到 P3.2c）。

---

## §0 现状基线（2026-11-28 19:30，lead 实测）

| 维度 | 现在 | ForgeKernel | 差距性质 |
|---|---|---|---|
| 自主规划 | 6/10 `nextstep.py` 给决策边，无"换方向/改架构"规划 | 有 meta 层 | 中 |
| 自主执行 | 6/10 `driver.py` 612 行 + `iterate.py` 703 行，**真跑通 1 轮** | driver 11,994 行 | 大 |
| 自主判定 | **8/10** judge 冻结容差 + claim 五项 + 证据分级 | 无显式分级 | **我们领先** |
| 自主停止 | 6/10 动态噪声底 + plateau 有代码，**真机未触发** | 有且论文验证 | 中 |
| 自主进化 | **3/10** 只有单代基线晋升 | `self_evolve.py` 多代链 | 大 |
| 不依赖自觉 | **2/10** 无 sandbox / 无 deny，靠 prompt | sandbox 真边界 | **最大** |
| 规模/覆盖 | 3/10 engine 6,917 行 / KB≈9 / 21 卡（半数十 DS<40） | 32,288 行 / KB 200 / 27 算子 | 大 |

**实测战果（2026-11-28 20:00 复核，措辞已更正）：**

| 项 | 实测 |
|---|---|
| 落库的 iterate 记录 | **1 条**（`iterate-20261128_real-mr4.jsonl`，`iter=1`，`stop_reason=rounds_done`） |
| 真 agent 的 iterate 调用 | **3 次，全部 `--max-rounds 1`** |
| `--max-rounds` 默认值 | **没有默认值**（`required=True`），每次显式给 |
| 多轮验证 | **只在 fake backend 上验过**（`--demo 3`、`--fake-flat 5` 触发 plateau） |
| `--max-rounds 2` | 只有一次，且是 `--dry-run` |
| 含 `_common.py` 改动的 KEEP | **0** |
| 沙箱 | **无** |

> **关键更正**：不是"只能跑一轮"，是**"从来没有让真 agent 连续跑过多轮"**。
> 机制上支持（fake 跑过 3 轮、plateau 在 r3 正常触发），但真 agent 一次都没试过。
> 所以 A 段的任务不是"修复多轮"，而是**第一次真正跑多轮**。

**结论：形状对了；真 agent 的样本是「3 次调用 = 3 轮，全部单轮」。四块关键能力（边界/跨代/架构层/规模）为零。当前估算 3.5/9。**

---

## §1 补强顺序（按"决定可信度"排序，不是按难度）

```
A. 把样本从 1 变成 8          ← 没有样本，后面全是空谈
B. 边界（sandbox/deny/hook）  ← 没有边界，所有数字建立在"agent 很乖"上
C. 跨代自进化（≥3 代）        ← 才叫"进化"，不是"跑几轮"
D. 架构层 M0                  ← 深度来源：先定结构，再在结构内榨
E. 规模与知识（KB 认证流水线、多算子覆盖）
F. 对外可比（三臂重跑 + 自主度指标）
```

**A 和 B 必须做；C 决定"进化"这个词能不能用；D 决定"深度"；E/F 是补规模与可信度。**

---

## §2 A —— 把样本从 1 变成 8（最高优先）

### 任务

1. `iterate --max-rounds 8` 在**一张老卡 + 一张新卡**各跑一轮完整 8 轮（不是两次 4 轮）。
2. 其中**至少 1 轮必须是"真·新变体"**：agent 在 `_common.py` 里登记一个**不预先存在**的变体
   （不是编译已声明的 `fa-rms-mr4`）。
3. 每轮记录落 `<artifact>/runs/loop/iterate-<ts>.jsonl`（已实现）。

### 前置（**业主拍板**）

- 每轮 agent 预算从 **1200s → 1800–2400s**。现在 1200s 只够编译，agent 必然 timeout
  （实测 `change="[agent-timeout after 1200s]"`）。

### 验收 A1–A4

```bash
cd /workspace/tookit-skills
# A1 轮次 ≥8（单次运行连续产生）
cat $(find .artifacts -name 'iterate-*.jsonl') | grep -c '"verdict"'          # ≥8
# A2 至少 1 条"真·新变体" KEEP
grep -h '"verdict": "KEEP"' .artifacts/**/iterate-*.jsonl | grep -c '_common.py'   # ≥1
# A3 无 timeout 占位的成功轮
grep -h '"verdict": "KEEP"' .artifacts/**/iterate-*.jsonl | grep -c 'agent-timeout'  # 期望 0
# A4 每轮都有 claim 结论（无静默跳过）
python3 -c "import glob,json;rs=[json.loads(l) for f in glob.glob('.artifacts/**/iterate-*.jsonl',recursive=True) for l in open(f) if l.strip()];assert all(r.get('claim') for r in rs);print(len(rs))"
```

**通过条件**：A1–A4 全过，且日志里能看到 `stop_reason`（`rounds_done` / `plateau` / `impl_fails` 之一）。

---

## §3 B —— 边界：把"靠自律"换成"靠物理"（第二优先）

> ForgeKernel 的分层：**Bash sandbox（真边界）> permissions.deny > PreToolUse hook（绊线）**。
> 它自己注释：*"This is a tripwire, not a boundary."*

### 我们的目的与它不同

它防"信息泄漏"（学生上网搜实现）；**我们防"不可审计的旁路数字"**。
所以规则也不同：

| 层 | 我们的规则 | 为什么 |
|---|---|---|
| **B1 sandbox** | agent 用户**不能写** `.artifacts/**/runs/loop/`、不能读 `judge/tolerance.json` | 账本与容差只能引擎写；否则"审计"变自证 |
| **B2 permissions.deny** | 禁直接执行 `<卡>/bench/ab.py`、`acc_probe.py`；禁写 `tolerance.json` | 测量入口唯一 |
| **B3 PreToolUse hook** | 命中以上时**返回 exit 2 + 可读理由**（"测量请走 loop.py run"） | 绊线：给理由，不是给安全 |

### 验收 B1–B3

```bash
# B1 以 agent 身份写账本必须失败
su - opb_c -c 'echo x >> /workspace/tookit-skills/.artifacts/driver/records/t.jsonl' ; echo "rc=$?"   # 期望 rc≠0
# B2 以 agent 身份直接跑 bench 必须失败
su - opb_c -c 'cd /workspace/tookit-skills && python3 Plugins/ops/skills/op-fused-add-rmsnorm/bench/ab.py --help' ; echo "rc=$?"  # 期望 rc≠0
# B3 hook 阻断时有可读理由（stderr 含 "loop.py run"）
```

**通过条件**：B1/B2 都是**物理失败**（不是靠 prompt），B3 有可读理由。
**注意**：B3 单独不算数 —— 没有 B1 的 hook 等于没有边界。

---

## §4 C —— 跨代自进化（≥3 代）

### 任务

照 `self_evolve.py` 的形状：

```
gen k:  reference = gen k-1 的 best（第一代用卡里现有 reference）
        iterate(N 轮) → best_k
        若 (best_k vs reference) > 噪声底：
            写 tasks/<op>-g<k+1>/：task.toml + shapes + best_k 作为 reference
            下一代目标 = --step × 当前
        否则 stop（plateau）
```

**关键约束**：晋升前必须 `claim` PASS（已实现，别绕）。

### 验收 C1–C3

```bash
# C1 至少 3 代目录
ls -d tasks/*-g2 tasks/*-g3 2>/dev/null | wc -l      # ≥2（g2/g3）
# C2 每代有记录且 base 指向上代 best
# C3 停止原因写清楚（plateau 或 max-gen）
grep -h 'stop_reason' .artifacts/**/iterate-*.jsonl | tail -3
```

**通过条件**：≥3 代，每代增益可追溯，链条可在别的机器上复跑（`tasks/` 自带 reference）。

---

## §5 D —— 架构层 M0（"深度"的来源）

### 任务

1. 第一轮**之前**加一步「定架构」：切分方式 / tile 形状 / warp 角色 / 寄存器与 LDS 预算，
   产出 `M0.json`（或 `artifact/M0.md`）。
2. **M0 gate 通过后冻结**：后续轮的 `changed_files` 不许碰架构相关文件
   （`_common.py` 里的 tile 常量、`kernels/*.hip` 的主体结构）。
3. 测量证明"只有换架构才能突破"时，**回 M0 重定**，而不是让后续轮绕着走。

### 验收 D1–D3

```bash
# D1 M0 产物存在
ls .artifacts/**/M0.json 2>/dev/null | head -1
# D2 M0 之后的轮次没有改架构文件
# D3 有"回 M0"的记录（至少在一次 plateau 之后）
```

**通过条件**：D1 有；D2 在 ≥3 轮的运行里成立；D3 至少出现一次。

---

## §6 E —— 规模与知识

| 项 | 目标 | 验收 |
|---|---|---|
| KB 认证流水线 | probe → 入账 → `kb_lint` 全过；每条带 `evidence:` 指针 | 未认证条目 = 0 或列在 `uncertified.md` |
| KB 数量 | ≥30 条（现在 ≈9） | `ls hw_kb/**/*.md \| wc -l` |
| 算子覆盖 | 8 轮跑在 ≥3 个算子上（老卡 / 新卡 / 全新算子） | 每张卡都有 iterate 记录 |

---

## §7 F —— 对外可比 + "自主度"指标

### 三臂重跑（前置：B 边界已落地）

只有引擎真的成为唯一测量入口，三臂对照才在测"引擎价值"，否则还是"自愿采纳率"。

### 指标先定（别事后解释）

| 指标 | 定义 | 目标 |
|---|---|---|
| **无人介入轮次** | 连续 `iterate` 跑完不需人操作的轮数 | ≥8 |
| **真变体率** | `_common.py` 有改动的 KEEP / 总 KEEP | ≥0.5 |
| **claim PASS 率** | PASS / 总记录 | ≥0.8 |
| **KEEP 率** | KEEP / 总轮次 | ≥0.3 |
| **代际增益** | 每代 best vs 上代 | > 噪声底，≥3 代 |
| **vs 裸 CC** | 同任务 C vs A | ≥20% |

---

## §8 分工与节奏

| 谁 | 做什么 |
|---|---|
| **work 会话** | A（8 轮）、B（sandbox/deny/hook）、C（跨代）、D（M0）、E（KB 流水线） |
| **lead** | 每阶段验收（重跑命令，不看结论）、F 的三臂重跑与指标定义 |
| **业主** | ① 拍板每轮预算 1800–2400s ② 决定是否照 ForgeKernel 立 `tasks/` 新目录结构 |

**节奏**：A → B 串行（A 用的还是旧沙箱），C/D 可并行，E 持续，F 最后。

**一次只开一个阶段，每个阶段完成 → `updates/handoff/RESULT-<阶段>-<日期>.md` → lead 重跑验收 → 通过才开下一个。**

---

## §9 不做（防止走回头路）

1. **不照抄 12k 行 driver** —— 我们 1.3k 行覆盖了 80% 的价值（因为不做 teacher/student 双层蒸馏）。
2. **不先做双层 agent**（teacher/student）—— 先单层跑满 8 轮，再谈分层。
3. **不在 B 之前跑 C/D** —— 没有边界的样本不可信。
4. **不为过测试放宽判据** —— `evidence_unverified` 这类判据不许删（已强调三次）。
5. **不追 KB 200 条** —— 数不是目标，"每条能认证"才是。
