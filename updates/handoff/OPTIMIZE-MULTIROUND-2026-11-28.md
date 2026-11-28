# 多轮迭代优化建议 · 2026-11-28 21:50

> 业主定调：**多轮迭代优先级最高**。本文件是 lead 的监控结论与优化建议，
> 每轮跑完更新一次。执行者：work 会话；验收：lead。

---

## 一、现状：机制**首次跑通 2 轮**，但两轮都空转

仓库里现在有 **3 条** iterate 记录：

| 时间 | iter | verdict | claim | changed_files | 说明 |
|---|---|---|---|---|---|
| 17:04 | 1 | **KEEP** | PASS | `kernels/build.sh` | 旧 run，agent 编译了预声明变体 |
| 21:41 | 1 | **INVALID_NOOP** | PASS | `ledger.md` | agent 自述"候选已在位，本轮无代码增量" |
| 21:41 | 2 | **INVALID_NOOP** | PASS | `engine/loop.py` | agent 超时 1200s |

**好消息**：`iter=2` 出现了 —— **多轮控制流第一次真的跑到了第 2 轮**（之前每次都死在 1 轮）。

**坏消息**：两轮都是 `INVALID_NOOP`，`stop_reason=agent_timeout`，循环没有产出。

**而且注意这两个数字**：

```
iter=1: base=0.2523  cand=0.1603  vs_base=0.635   ← cand 比 base 快 36%
iter=2: base=0.2522  cand=0.1608  vs_base=0.638
```

**cand 一直很快，但判定一直是 NOOP** —— 因为那 36% 是**上一轮就有的实现**带来的，不是这一轮。
**归因是对的**（这正是 AGENT_CODE_RE / INVALID_NOOP 该做的事），**但它说明循环在空转**。

---

## 二、根因链（一条，四个环节）

```
① run 用 --tag fused-new
      ↓
② candidates.yaml 里没有 fused-new（只有 reg-resident）
      ↓
③ prompt 的「本轮要实现的候选」整段缺失 → agent 收到空任务
      ↓
④ agent 花 1200s 发现「候选其实早已实现」→ 写 ledger.md / 改 engine 文件 → 超时
      ↓
⑤ AGENT_CODE_RE 只认代码文件 → INVALID_NOOP → MAX_NOOP=2 → 停
```

**这不是 agent 的问题，是任务定义的问题。** agent 甚至很诚实地说：
> 「实现完成 —— 候选 `fused-new` 已在位，本轮无代码增量。」

它没有装样子，是我们给了它一个**没有新工作可做**的任务。

---

## 三、优化建议（按优先级）

### P0-1 派任务前校验：候选必须存在且未实现

**现在**：`--tag` 随便给，找不到就当没有候选，agent 空转。
**要改**：`driver.py` 启动时 preflight，三条全过才开跑：

```python
def preflight_candidate(card: Path, tag: str) -> list[str]:
    """返回问题列表；非空则拒绝开跑。"""
    cands = load_candidates(card).get("candidates") or []
    hit = [c for c in cands if str(c.get("id")) == tag or str(c.get("variant")) == tag]
    if not hit:
        return ["--tag %r 不在 candidates.yaml 里（有：%s）"
                % (tag, ", ".join(str(c.get("id")) for c in cands))]
    if str(hit[0].get("status") or "") == "done":
        return ["候选 %r 已实现（status=done），派给 agent 只会空转" % tag]
    return []
```

**验收**：
```bash
# 用不存在的 tag 启动 → 立刻退出，不开跑
python3 -I Plugins/ops/engine/driver.py --real --card-rel ... --tag no-such-tag --demo 3
# 期望：rc≠0 + 打印 "不在 candidates.yaml 里"
```

---

### P0-2 队列空时不要派 agent，改切 explore

**现在**：没有可做候选时仍然调 agent（1200s 空转）。
**要改**：`iterate()` 进入下一轮前检查「还有没有未实现的候选」；
没有就 **不调 agent**，直接记一条 `verdict=NO_WORK` 并转 `explore.py`（出题）。

```python
if not has_pending_candidate(card):
    rec.verdict = "NO_WORK"
    rec.stop_reason = "queue_empty"
    rec.change = "无可派候选；已建议 explore（出题后再实现）"
    break
```

**验收**：候选耗尽的卡上跑 `--demo 3` → 只产生 1 条记录、`stop_reason=queue_empty`、
**不进 agent 子进程**（看进程列表）。

---

### P0-3 prompt 必须带时间预算与停止条件

**现在** prompt 只说"你只负责写实现"，agent 会自行决定研究多久。
**要改**：加一段硬约束（照 ForgeKernel 的 student prompt 风格）：

```
## 本轮预算（硬约束）
你有 20 分钟。做完这三件事立刻停，不要继续：
  ① 在 <卡>/_common.py 登记/修改变体（或改 kernels/*.hip）
  ② 跑一次最小验证（编译成功即可，**不要测性能**）
  ③ 用一句话写「改了什么」，然后结束
禁止：读整个仓库、翻历史账本、比对其它变体、改 engine/ 下任何文件。
```

**验收**：新一轮的 agent 会话里，`tool_use` 总数明显下降（目标 ≤ 25），
且**没有** `Plugins/ops/engine/**` 的写入。

---

### P1-4 agent 能改测量引擎（**完整性风险**）

记录里 iter=2 的 `changed_files` 是 **`Plugins/ops/engine/loop.py`**。
而 driver 的测量走的是**工作区里的** `Plugins/ops/engine/loop.py`：

```python
[sys.executable, "-I", str(self.ws / "Plugins/ops/engine/loop.py"), ...]
```

**也就是说：agent 可以改掉用来测它的尺子。**

现在只是靠 `AGENT_CODE_RE` 把这次改动过滤成 noop（结果对），但**没有阻止它改**。

**要改**：
1. 每轮 commit 后检查 `changed_files` 是否含 `Plugins/ops/engine/` 或 `scripts/`：
   有 → **回滚 + 记 `TAMPERED` + 停**，并在报告里写明；
2. 更好：workspace 里 `engine/`、`scripts/`、`tests/` 设只读（chown root + chmod a-w），
   agent 用户写不进去。

**验收**：
```bash
# 构造 agent 改 loop.py 的 fake 场景 → verdict 必须是 TAMPERED，不是 INVALID_NOOP
python3 -m pytest Plugins/ops/tests/test_driver_fake.py -q -k tamper
```

---

### P1-5 记录每轮落盘，不要最后一次性写

现在 2 条记录在同一个文件 `iterate-20261128_214123.jsonl`，是跑完才写的。
**中途崩了就全丢**（今天已经发生过两次 run 被 kill）。

**要改**：每轮 `_append_record()` 追加一行（JSONL 天然支持）。

**验收**：跑 `--demo 3` 且第 2 轮人为 kill → 文件里仍有 1 条。

---

### P1-6 超时要有诊断：agent 死前在干什么

现在 `changed_files=['engine/loop.py']` + `change="[agent-timeout after 1200s] partial="` ——
看不出它在干什么。**要改**：超时时把最后 N 条工具调用拼进 `change`（截 300 字）：

```
[agent-timeout after 1200s] last: Bash(sed -n '60,200p' tests/test_adapter.py) → ...
```

**验收**：超时记录里能看出它卡在"读文件"还是"写代码"。

---

### P2-7 候选状态机（防止重复派已实现的候选）

给 `candidates.yaml` 的候选加 `status` 流转：
`pending → implementing → implemented`，`iterate` 只派 `pending`。
现在靠 agent 自己说"已在位"，太晚。

### P2-8 `--repeat` 噪声底还没有真跑验证

R1 修复后伪造测试通过，但**没有一次真跑用到 `repeat_spread`**。
第一次拿到 KEEP 时必须核对 `noise_floor` 字段不是固定 1.5%。

### P2-9 一键验收脚本

```bash
python3 -I scripts/check_multiround.py --dir .artifacts
# 断言：轮次≥3 / 无 noop 却 KEEP / 无 TAMPERED / verdict 分布打印
```

---

## 四、下一轮实验的建议配置（等 P0 修完再跑）

```bash
# 用一张真的有 pending 候选的卡；先确认 preflight 通过
python3 -I Plugins/ops/engine/driver.py --real \
    --workspace .artifacts/driver/<run>/workspace \
    --card-rel Plugins/ops/skills/<有 pending 候选的卡> \
    --tag <该候选 id> \
    --agent-cmd "su - opb_c -c 'bash /tmp/agent_claude.sh'" \
    --demo 3 --gpu 0 --agent-timeout 1200
```

**通过条件（三条同时）**：
1. 产生 **≥3 条**记录（不是 1 条就停）
2. 其中 **≥1 条 `verdict=KEEP` 且 `changed_files` 含代码文件**
3. **0 条** `TAMPERED`，**0 条** `NO_WORK` 之后还继续调 agent

---

## 五、当前阻塞（一句话）

> **不是循环跑不起来（它已经跑到 iter=2 了），是"没有值得跑的任务"。**
> P0-1/P0-2 修完，agent 才有活干；P0-3 修完，它才会在 20 分钟内干完。
