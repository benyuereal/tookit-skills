# HANDOFF · 2026-11-28 · P3 执行计划：把「人肉循环」换成 driver

> 执行者：work 会话。验收者：lead（不改 `engine/**`，只重跑验收命令）。
> 前置：P0/P1/P2 已通过（见 `STATUS.md`）。
> 本文件只讲 **P3：自主迭代 / 自主规划 / 自主执行**。P4（边界）不在本份。

---

## 一、先诊断：我们到底差在哪（不是"差一个 driver"这么笼统）

ForgeKernel `driver.py` 的 docstring 一句话是它的全部职责：

> The driver chains the primitives (scaffold / submit / trajectory / ncu / resume) into the
> inner student loop and the outer meta loop, **driving sub-agents and GPU jobs through a
> single injected `backend` seam. All control flow here is deterministic given the backend,
> so it is unit-tested with a fake.**

拆成可核对的能力，我们逐条对：

| # | driver 拥有的能力 | 我们现在 | 后果 |
|---|---|---|---|
| **G1** | **"一轮"的定义**：agent 交代码 → 引擎测 → 记录 → 决定下一轮 | `loop.py run` 只是"一次实验"，**没有"轮"这个概念** | 无法自动推进 |
| **G2** | **agent 调用接缝** `be.run_agent(prompt, role=)`，可注入、可 fake | `coordinate.py::AgentImplementer` 很原始（su - opb_c + copytree），且没接进主流程 | 循环自己发起不了下一轮 |
| **G3** | **工作区版本语义**：每轮 commit；**relaxed-revert**（性能回退保留、只有正确性破坏才回滚） | 无版本管理，改动直接落盘 | 改坏回不去；也说不出"这轮改了什么" |
| **G4** | **停止条件**：噪声底（同代 ratios spread ≥1.5%）+ plateau | `nextstep` 只有 floor 判断，**无 plateau** | 要么无限跑，要么过早停 |
| **G5** | **probe 子循环**：直到拿到一条记录才继续（失败会重试/修补） | `run` 失败直接 rc≠0，**没人重试** | 一次失败就断链 |
| **G6** | **重试/超时/心跳**（截断 turn 重试、API outage 重试、log 心跳） | 无 | agent 截断或挂掉，循环静默死 |
| **G7** | **pre-green 语义**：第一次 correctness PASS 之前，子步骤记 NEUTRAL 不评估 | 无 | 半成品被当成品测 |
| **G8** | **resume / prefix-reuse**：plateau 后算恢复点，reset 工作区到那里 | 无 | 不能从"上一代最优"继续 |
| **G9** | **确定性可测**：控制流与副作用分离，fake backend 单测 | `loop.py` 里直接 subprocess 跑 GPU | driver 逻辑没法单测 |

**结论：我们缺的不是一个文件，是 G1–G9 九种能力。** 其中 **G2/G9 必须先做**（不先分离副作用，后面全是调试地狱）。

---

## 二、P3 拆成 6 步（每步独立验收，不许并行铺开）

### P3.0 契约先行（半天）

**产出**：`updates/handoff/P3-CONTRACT.md` + `Plugins/ops/engine/driver.py` 里的
两个 Protocol/dataclass（**只有类型和 docstring，不写实现**）。

**1) 一轮的产物 `IterationRecord`（最小字段集，先少后加）**

```python
@dataclass
class IterationRecord:
    iter: int
    round_id: str                  # 唯一 id，便于 resume
    change: str                    # agent 自述"改了什么"（一句话，不含数字）
    changed_files: list[str]       # 本轮改动的文件（从 git diff 取）
    workspace_commit: str          # 本轮提交的 commit hash（有了才能 revert）
    correctness: bool | None       # None = pre-green 未评估
    verdict: str                   # KEEP | REVERT | NEUTRAL | REFUTED
    base_ms: float | None
    cand_ms: float | None
    vs_base: float | None          # **同 job back-to-back 比值**（不是跨 job 平均）
    utilization: float | None
    floor_us: float | None
    acc: dict                      # max_ulp/cosine/evidence_kind
    claim: str                     # PASS | FAIL:<reason>
    refuted_directions: list[str]  # 死路清单，喂回下一轮
    noise_floor: float             # 本代噪声底（决定"算不算提升"）
```

**2) backend 接口（G2 + G9 的核心）**

```python
class Backend(Protocol):
    def run_agent(self, prompt: str, role: str = "student") -> str: ...
    def commit(self, message: str) -> str: ...          # 返回 commit hash
    def reset_to(self, commit: str) -> None: ...
    def eval(self, tag: str) -> dict: ...               # 跑测量，返回 {cand_ms, base_ms, acc, ...}
```

**验收 C0**：

```bash
python3 -I Plugins/ops/engine/driver.py --selfcheck   # 只检查类型/契约自洽，不跑 GPU
grep -c "class Backend\|class IterationRecord" Plugins/ops/engine/driver.py   # 期望 2
ls updates/handoff/P3-CONTRACT.md
```

---

### P3.1 fake backend + 控制流单测（1 天）**← 最关键的一步**

**目标**：把"调 agent / 跑 GPU / git"全部抽到 `Backend` 后面，然后**用一个 fake backend
把完整控制流跑通并单测**，全程不碰 GPU。

**产出**：`Plugins/ops/engine/driver.py` 的 `iterate()` + `tests/test_driver_fake.py`

**验收 C1**：

```bash
# 1) fake backend 能跑完整 3 轮，不需要 GPU、不需要 claude
python3 -m pytest Plugins/ops/tests/test_driver_fake.py -q          # 全绿
# 2) fake 跑完必须有 3 条 IterationRecord，且字段不空
python3 -I Plugins/ops/engine/driver.py --demo 3 --out /tmp/demo.jsonl
python3 -c "import json;rs=[json.loads(l) for l in open('/tmp/demo.jsonl')];print(len(rs));assert all(r['workspace_commit'] and r['claim'] for r in rs)"
```

**这一条不过，不许进 P3.2。** 没有 fake 的单测，接上真 claude 之后每个 bug 都要花 20 分钟复现。

---

### P3.2 真 backend：一轮跑起来（2 天）

**产出**：
- `ClaudeBackend`（子进程调 agent）+ `GpuBackend`（复用 `loop.py run`）+ `GitBackend`
- `loop.py iterate --max-rounds N --agent-cmd "<cmd>"`（agent 必填，**不给默认**）

**一轮的定义（照 driver.py）**：

```
render_prompt(record 列表) → agent（只交实现，不交数字）
  → commit（拿到 workspace_commit）
  → eval（跑 loop.py run，落 record）
  → 判定 verdict（KEEP / REVERT / NEUTRAL）
  → 若 correctness 破坏 → reset_to(上一轮 commit)
  → append IterationRecord
```

**relaxed-revert 硬语义**：**性能回退保留**（当脚手架），**只有正确性破坏才回滚**。

**验收 C2**（用 `op-silu-mul`，仓库真实卡）：

```bash
# 2 轮真跑（agent-cmd 由 work 会话提供，建议 claude -p）
python3 -I Plugins/ops/engine/loop.py -C Plugins/ops/skills/op-silu-mul \
    iterate --max-rounds 2 --agent-cmd "<cmd>"
# 断言：账本 ≥2 条，每条有 workspace_commit / claim / verdict
python3 -c "
import glob,json
rs=[json.loads(l) for f in glob.glob('.artifacts/**/runs/loop/*.jsonl',recursive=True) for l in open(f)]
it=[r for r in rs if r.get('verdict')]
print('迭代记录:',len(it)); assert len(it)>=2
print('字段完整:', all(r.get('workspace_commit') and r.get('claim') for r in it))"
# 断言：agent 的会话里没有 bench 命令（它没自己测）—— 这条是 P3 的灵魂
```

---

### P3.3 停止条件：噪声底 + plateau（半天）

**产出**：`noise_floor(ratios)`（同代 spread，**不低于 1.5%**）+ plateau 判定

规则（照 ForgeKernel）：
- 一轮的增益 > 噪声底 → KEEP，继续
- 连续 `MAX_PLATEAU=3` 轮增益 ≤ 噪声底 → `plateau`，停止（或转 P3.6 外层）
- correctness 连续破坏 `MAX_IMPL_FAILS` 次 → 停止并记录

**验收 C3**：

```bash
# fake 场景：3 轮无提升 → 必须停，且 reason=plateau
python3 -I Plugins/ops/engine/driver.py --demo 5 --fake-flat --out /tmp/flat.jsonl
python3 -c "import json;rs=[json.loads(l) for l in open('/tmp/flat.jsonl')];print(rs[-1]['verdict'], rs[-1].get('stop_reason'))"
# 期望：stop_reason=plateau，且停在 ≤4 轮
```

---

### P3.4 反馈渲染：下一轮的输入（半天）

**产出**：`render_student_prompt(records, card)`，输出里**必须**包含：

1. 上一轮结构化记录（不是自然语言总结）：`verdict / cand_ms / vs_base / utilization / claim`
2. **死路清单** `refuted_directions`（否则 agent 会反复试已证伪的方向）
3. 上一轮 `changed_files` + `change`（它自己写的，用来避免重复）

**验收 C4**：

```bash
python3 -I Plugins/ops/engine/driver.py --render-demo --out /tmp/prompt.txt
grep -c "verdict\|refuted_directions\|vs_base" /tmp/prompt.txt   # 期望 ≥3
python3 -c "t=open('/tmp/prompt.txt').read();assert 'last_error' not in t or True;print(len(t))"
```

---

### P3.5 重试与心跳（半天）

**产出**：agent turn 失败重试（bounded）、API 超时重试、每轮写一行心跳日志

**验收 C5**：

```bash
# fake backend 注入"第 2 轮 agent 返回空"→ 必须重试并继续
python3 -m pytest Plugins/ops/tests/test_driver_fake.py -q -k retry
# 心跳日志每轮一行
grep -c "iter" <run log>
```

---

### P3.6 外层（**后置，本轮不做**）

plateau 后：meta 诊断 → 换方向 / 改架构（M0）→ 计算 prefix-reuse 恢复点 → resume。
这是 ForgeKernel 的 `outer loop`，等 P3.1–P3.5 稳了再说。

---

## 三、铁律

1. **P3.1 不过不许进 P3.2** —— 没 fake 单测就接真 agent，等于放弃可调试性。
2. **agent 只交实现，不交数字。** 交付物里出现 agent 自测的耗时 = 本轮失败。
3. **不许把控制流写进 `loop.py`** —— `loop.py` 保持"跑一次实验"的职责；
   循环逻辑全部在 `engine/driver.py`。这是为了 G9（可测）。
4. **`--agent-cmd` 必填，不给默认值** —— 测试用 fake，生产显式给。
5. **动过的文件**：`engine/driver.py`（新建）、`engine/loop.py` 的 `iterate` 子命令、
   `engine/nextstep.py`（如需）、`tests/test_driver_fake.py`（新建）、`updates/handoff/P3-CONTRACT.md`。
   `Plugins/ops/skills/**`、`toolkit-opbench/**` 一律不动。

---

## 四、汇报格式

每完成一步写 `updates/handoff/RESULT-P3.<n>-2026-11-28.md`，三段（同 P0 格式）：
改了什么（含 commit）、验收命令**逐字输出含 rc**、没做到的。

lead 会重跑 C0–C5 全部命令；**对不上就打回**。

---

## 五、我（lead）这边同时做什么

- 起草 P4（sandbox + deny + hook）的规格，等 P3 过；
- 准备 opbench 侧的接缝（`.artifacts` 路径、C 臂契约、三臂重跑时机）；
- 每份 RESULT 到达即复跑验收。
