# HANDOFF · 2026-11-28 · 第二份：从「审计层」到「工作流反转」的路线

> 接 `HANDOFF-2026-11-28-object-layer-plan.md`（那份是 P0–P2 的执行计划，**本份不重复**）。
> 本份是**设计路线**：P3 之后怎么走，以及为什么这么走。
>
> 读者：work 会话（执行）+ 业主（拍板）。lead 只写规格与验收，不改 `engine/**`。

---

## 0. 一句话结论（今晚读 ForgeKernel 修正的认知）

> **它不"强制 agent 走引擎"，它把"测量"从 agent 手里拿走了。**

agent 的任务是**写实现**；测、判、记、喂回，全在 driver 里。
于是"要不要用引擎"根本不是个选择题 —— 工作流里没有第二条路。

我们的 C 臂之所以 4 小时 0 账本，不是它不听话，是**它的任务要一份报告，而报告里的数字只能靠它自己测**。
自己测比学我们的引擎便宜，理性选择就是绕开。

---

## 1. 三段证据（不是推测）

| # | 证据 | 结论 |
|---|---|---|
| ① | 我们三轮实测：C 臂拿到正确的 `next` 指令、绝对路径、自检结论后，仍然先花 10–20 分钟读源码，不执行 | **提示词彻底无效**。改 CLAUDE.md / hook 文案 / 命令自足化，都只把"开始调 next"从 0 变成有，没把"执行"变成有 |
| ② | ForgeKernel `agent_policy.py` 注释：早期只写 prompt，审计 transcript 发现学生在偷搜参考实现并改写进 `kernel.py` | 他们也踩过同一条路，**结论也是"prompt text 不管用"** |
| ③ | ForgeKernel 的三层：Bash sandbox（真边界）> `permissions.deny` > PreToolUse hook（自述「tripwire, not a boundary」） | **hook 是最软的一层**，只负责"把规则说出来"。指望 hook 强制流程是方向错误 |

---

## 2. 设计原则（五条，按重要性）

1. **测量权收归引擎。** agent 不产数字，只产实现。数字由引擎产生、由 `claim` 审计。
2. **交付物里不含数字。** 只要报告要求"给出耗时/达成率"，agent 就必须自己测，一切强制手段都会失效。
3. **反馈结构化。** 下一轮喂给 agent 的不是自然语言总结，是上一轮记录的字段（判定/耗时/达成率/失败机制）。
4. **边界靠 sandbox，不靠 hook。** hook 只做绊线：给一句可读理由，补 sandbox 表达不了的两点。
5. **fail-closed。** 判据不齐就不放行（宁可标黄，不可标绿）。这条已经在 P0 的 `claim` 里落地。

---

## 3. 阶段计划（P3 之后）

> P0–P2 见第一份 handoff，正在执行。以下按依赖排序，**一次只开一个 P**。

### P3 —— 工作流反转（核心，决定"9 分"能不能测出来）

**目标**：agent 写实现 → 引擎测 → 引擎判 → 结构化喂回 → 下一轮。全程 agent 不碰测量。

**3.1 问题定义格式（替代"题面 markdown"）**

新任务格式 `tasks/<op>/task.toml`，照 ForgeKernel 的"用户只写问题"：

```toml
[task]
name = "silu-mul"
language = "triton|cuda|hip|cutedsl"
arch = "gfx938"

[target]
ratio  = 1.20          # 相对 reference 的目标倍数
timing = "device"      # device | e2e

[shapes]
file = "shapes.json"   # 只给形状，不给实现

[reference]
kind  = "vendor"       # vendor | source | none
impl  = "lightop.silu_mul"
```

**run 不能改问题**：`loop.py` 的任何子命令都不接受覆盖 `target` / `shapes` / `tolerance` 的参数（P0 的 A6 已经在验这一条）。

**3.2 driver：`loop.py iterate`**

```
loop.py -C <卡> iterate --max-rounds N --agent-cmd "<命令>"
```

每轮：
1. 渲染 student prompt（含上一轮结构化记录，见 3.3）
2. 调 agent（子进程）→ 拿回"改了什么"（**不含数字**）
3. 引擎跑 `run` → 落 `.artifacts/.../runs/loop/` → `claim` 审计
4. 判定：KEEP / REVERT / NEUTRAL（照 ForgeKernel：**性能回退保留、正确性破坏回滚**）
5. 写 `IterationRecord`

**3.3 反馈通道（这条是"深度"的来源）**

下一轮 prompt 必须包含**结构化**的上一轮结果，不是总结：

```json
{"round": 3, "verdict": "KEEP", "cand_ms": 0.142, "base_ms": 0.203,
 "vs_ref": 1.43, "utilization": 0.82, "gate": "headroom",
 "acc": {"max_ulp": 0.5, "cosine": 0.999998, "evidence_kind": "vendor_library"},
 "claim": "PASS", "refuted_directions": ["fusion-boundary"]}
```

**验收（机器可检）**

```bash
# D1 agent 全程没有自己测：它的会话里没有任何 bench 类命令
grep -c 'bench\|ab\.py\|cuda.Event' <agent 会话 jsonl>   # 期望 0

# D2 账本里有 N 条记录，且工程由引擎产生
python3 -c "import glob,json;rs=[json.loads(l) for f in glob.glob('.artifacts/**/runs/loop/*.jsonl',recursive=True) for l in open(f)];print(len(rs));assert all(r.get('impl') and r.get('command') for r in rs)"

# D3 每轮 claim 都有结论（PASS 或带理由的 FAIL），无静默跳过
for t in <tags>; do python3 -I Plugins/ops/engine/loop.py -C <卡> claim --tag $t; done

# D4 报告由引擎生成（--report），不是 agent 手写
head -3 <artifact>/report.md   # 期望含 "由 loop.py 生成"
```

**通过条件**：D1–D4 全过，且在 `op-silu-mul` 上跑满 ≥3 轮、至少 1 轮 KEEP。

---

### P4 —— 边界与绊线（照 ForgeKernel 三层，但目的不同）

> 我们的目的**不是**"防止 agent 上网查实现"（那是他们的信息隔离），
> 而是**"防止出现无法审计的旁路数字"** 和 **"防不可复现"**。

| 层 | 内容 | 为什么 |
|---|---|---|
| ① Bash sandbox | agent 的 shell 不能写 `.artifacts/` / `runs/loop/`（只能引擎写账本） | 账本一旦可被 agent 写，"审计"就是自证 |
| ② `permissions.deny` | 禁 agent 直接调 `<卡>/bench/ab.py`、禁写 `tolerance.json` | 测量入口唯一；容差必须在 run 期间冻结（P0 的 A3/A4 已在验） |
| ③ PreToolUse hook | 只给可读理由（"测量请走 `loop.py run`"），**不作为边界** | 学他们：hook 是 tripwire |

**验收**

```bash
# E1 sandbox 真的挡住写账本：agent 身份写 .artifacts/**/runs/loop/ 应失败
# E2 容差被改后 claim 必须 FAIL（sha256 不符）
# E3 hook 返回 exit 2 且 stderr 含可读理由
```

**注意**：③ 单独不构成安全；E1 才是关键。别把希望寄托在 hook 上。

---

### P5 —— 自进化链（照 ForgeKernel `self_evolve.py`）

**目标**：每一代被接受的实现成为下一代的 reference，直到增益落到噪声底。

```
gen k: base = 上一代 best  →  iterate  →  得 best_k
       若 best_k 相对 base 的增益 > 噪声底（≥1.5%）→ 写 tasks/<op>-g<k+1>/
       否则停（plataeu）
```

**关键约束（我们已经定过）**：晋升前必须 `claim` PASS；不过就不晋升，日志写明原因。

**验收**

```bash
# F1 两代之间 base 变化，且第二代 record 的 base 字段 == 第一代 best
# F2 增益落在噪声底（<1.5%）时停止，日志写 "plateau"
# F3 未过 claim 的结果不出现在 base.json 里
```

---

### P6 —— 知识库与"降低经验依赖"

**目标**：`hw_kb` 从 9 条扩到能支撑"未知算子"的程度，且每条都带**可复现证据**。

要求（比现在的 hw_kb 严）：
- 一条一文件，作用域写到架构（`hw_kb/hip/gfx938/<mechanism>.md`）
- 每条必须有 `evidence:` 指针（哪个 experiment/record 验过它），没有就算 not-certified
- 只讲机制，不讲算子（照 ForgeKernel：「Operator-agnostic; mechanism only」）

**验收**：`kb_lint` 全过；未认证条目数 = 0 或显式列在 `uncertified.md`。

---

## 4. 与 ForgeKernel 的对照（抄形状，不抄规模）

| 维度 | ForgeKernel | 我们（目标） | 我们现在的差距 |
|---|---|---|---|
| 问题定义 | `tasks/<op>/task.toml` | **P3.1** | 只有题面 markdown |
| 参考实现 | 恰好一个，读源码/黑盒 | **P0 judge** | contract.yaml 里散着 |
| 判据生成 | teacher 写 judge + GPU 校准冻结 | **P0** | 手写、无校准动作 |
| 测量权 | driver 提交 GPU 作业 | **P3.2** | agent 自己测 ← 根因 |
| 反馈 | 下一轮 prompt 带 IterationRecord | **P3.3** | 无 |
| 账本 | IterationRecord 20+ 字段（同 job 比值/host tax/basis） | **已做**（record/claim） | 字段少，但五项审计齐 |
| 门禁 | 每轮 gate + precision_gate | **已做**（claim）+ P3 每轮 | 缺"每轮" |
| 信息隔离 | sandbox + deny + hook | **P4**（目的不同） | 无 |
| 自进化 | 每代 best 晋升为 reference | **P5** | 无 |
| 知识库 | hw_kb 200 条、架构作用域、只讲机制 | **P6** | 9 条 |
| 规模 | 32k 行 harness | 不追 | ~4.3k 行 |

**不追的部分**：12k 行 driver 的规模、27 个算子的覆盖、论文级实验。我们抄**形状**。

---

## 5. 不做什么（避免走回头路）

1. **不再**用提示词/hook 文案去"劝"agent 走引擎 —— 今晚已证伪，两轮全部无效。
2. **不把 hook 当边界** —— ForgeKernel 自己写着它是 tripwire。
3. **不在 judge 校准前做 driver** —— 会把错误判据自动化成每小时几十次。
4. **不做通用系统** —— 每个 P 都要求一条竖切片跑通（一个真实算子）。
5. **不让 agent 写账本** —— 审计必须由引擎单方面产生。

---

## 6. 怎么判断"真的到 9 分"（指标先定，别事后解释）

| 指标 | 定义 | 目标 |
|---|---|---|
| **自主轮次** | `iterate` 连续跑完、无需人介入的轮数 | ≥ 8 |
| **claim PASS 率** | 账本中 claim PASS / 总记录 | ≥ 0.8（其余必须有 FAIL 理由） |
| **vs 裸 CC 的 Δ** | 同一任务，C 臂 vs A 臂（三臂对照） | 明显领先（≥ 20%） |
| **新算子可用性** | 新算子从 `task.toml` 到第一个 claim PASS 的步数 | 0 次人工介入 |
| **老算子可回归** | 已有卡在迭代后账本连续性与 claim 一致性 | 100% 一致 |

**注意**：第 3 项要在**工作流反转之后**重做三臂对照，否则测的还是"自愿采纳率"。

---

## 7. 分工与节奏

- **work 会话**：P0 → P1 → P2（进行中），之后 P3.2 / P3.3 的 driver 实现、P4 的三层、P5 / P6
- **lead**：P3.1 的问题定义格式（本文件已给草案）、每个 P 的验收、与 opbench 的接缝（`.artifacts/` 路径、C 臂契约、三臂重跑时机）
- **业主拍板**：P3 是否立项（要不要照 ForgeKernel 做工作流反转，而不是继续修补现有 loop）

**节奏**：一次只开一个 P；每个 P 完成 → `updates/handoff/RESULT-<P>-<date>.md` → lead 重跑验收命令 → 通过才开下一个。

---

## 8. 我现在就能做的（不碰 `engine/**`）

1. 把 P3.1 的 `task.toml` 格式写成**可解析的最小实现**（一个 `tasks/` 示例 + schema 校验），放 `Plugins/ops/tasks/`；
2. 起草 P4 的 sandbox/deny 配置（`settings.json` 片段 + hook 的 `decide()` 规格）；
3. 等 work 会话的 `RESULT-P0-2026-11-28.md` 出现，重跑 A1–A6 验收。
