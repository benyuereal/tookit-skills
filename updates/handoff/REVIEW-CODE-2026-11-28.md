# REVIEW-CODE · 2026-11-28 20:45 · lead 主动代码审查（driver.py 控制流）

> 触发：业主要求"不要等到出问题才介入"。
> 方式：逐行读 `driver.py` 的 `iterate()` / `_verdict()` / `render_student_prompt()` /
> `noise_floor_from()` / `RealBackend.eval()`，不依赖任何运行结果。

## 结论：**4 个真问题（1 个严重），2 个已在上一版修好**

---

## 🔴 R1（严重）噪声底用「跨轮比值」，语义错了 —— 一次大改进会压死后面所有 KEEP

```python
# driver.py: iterate() 里
ratios.append(1.0 + delta)            # ← 跨轮、跨不同实现的 cand/base
noise = noise_floor_from(ratios)

# driver.py:137
def noise_floor_from(vals, floor=NOISE_FLOOR):
    if len(vals) < 2: return floor
    return max(floor, max(vals) - min(vals))     # ← 值域跨度，不是测量噪声
```

**推演**（真实数值量级）：

```
r1: cand=0.144 base=0.250 → delta=-0.42 → ratios=[0.58]
r2: 只有 1 个值 → noise = 1.5%
    假设 r2 真改进 10% → delta=-0.10 → KEEP（正确）→ ratios=[0.58, 0.90]
r3: noise = max(0.015, 0.90-0.58) = 0.32   ← 噪声底变成 32%
    即使 r3 真改进 20%（delta=-0.20）→ -0.20 < -0.32 不成立 → NEUTRAL（错杀）
```

**为什么这是错的**：ForgeKernel 的噪声底来自**同一个 job 里重复测量同一个 kernel**
的比值 spread（测量噪声）。我们这里把**不同实现之间的真实差异**当成了噪声。

**修法（二选一，推荐 A）**：
- **A. 每轮重复测量**：`loop.py run --repeat 3`，噪声底 = 同一条记录的 3 次比值 spread。
  这是唯一语义正确的做法。
- **B. 退而求其次**：只在"同一 base 的相邻两轮"之间算，且用**相对偏差**
  `abs(a-b)/a`，不要用绝对差；并给噪声底设上限（如 ≤5%），否则一旦有代际差异就爆。

**验收**：连续 3 轮，r1 大改进、r2 小改进、r3 小改进 —— r2/r3 都必须能被判定（KEEP 或 NEUTRAL），
**不允许因为 r1 的改进把噪声底抬到 r2/r3 的增益之上**。

---

## 🔴 R2（真 bug）`changed_files` 两处来源不一致

```python
files = list(getattr(backend, "changed_files", []) or [])   # ← noop 判断用这个
...
rec = IterationRecord(
    ...
    changed_files=list(ev.get("changed_files") or []),       # ← 记录里写的是 eval 的
```

**后果**：`_verdict` 用 backend 的列表判 noop（对），但**记录落盘的是 eval 的列表**。
两者不一致时，账本会显示 `changed_files=[]` 而实际有改动 —— 我们 20:27 那条
`INVALID_NOOP` 记录就是这个形态，无法从记录区分"真没改"和"改了解析不到"。

**修法**：`rec.changed_files = files`（backend 是唯一真源）。`ev` 里的只作对照，
不一致时打一条警告。

**验收**：构造一个"backend 说有改动、eval 说没有"的 fake 场景 → 记录里必须是 backend 的列表。

---

## 🟠 R3 `MAX_NOOP = 1` 太激进：一次超时就整轮停

```python
MAX_NOOP = 1
...
if noops >= MAX_NOOP: rec.stop_reason = "noop"; break
```

agent 超时（`TimeoutExpired` → 返回字符串）→ `changed_files=[]` → `INVALID_NOOP` → **一次就停**。
今天两次 3 轮实验都是这样在第 1 轮死的。

**修法**：
- `MAX_NOOP` 默认 **2**，且可配（`--max-noop`）。
- **区分"noop"和"timeout"**：`run_agent` 返回 `[agent-timeout ...]` 时应记
  `stop_reason="agent_timeout"`，不占 noop 配额（超时是基础设施问题，不是 agent 没干活）。

**验收**：fake 注入"第 1 轮 timeout、第 2 轮正常改文件" → 必须跑到第 2 轮，不被 noop 提前终止。

---

## 🟠 R4 记录落进「可丢弃的工作区副本」

实际路径：

```
.artifacts/driver/newop_p32/workspace/.artifacts/driver/op-moe-gating-softmax_p32/runs/loop/iterate-*.jsonl
                 ^^^^^^^^^^^^^^^^^^^^^ 工作区（可删）      ^^^^ 嵌套的 .artifacts
```

工作区是整仓副本，重跑时会被重建 → **账本跟着工作区一起消失**。
`--out` 默认值目前落在工作区内部。

**修法**：`--out` 默认指向**仓库级** `.artifacts/driver/records/iterate-<run_id>-<ts>.jsonl`
（就是 `iterate-20261128_real-mr4.jsonl` 现在待的那个目录），不要放进 workspace。

**验收**：`rm -rf .artifacts/driver/<run>/workspace` 之后，迭代记录仍然存在。

---

## ✅ 上一版已修好的（我确认）

| 项 | 现状 |
|---|---|
| prompt 强制落卡 | 已有："**每完成一段就落卡**，不要等最后一次性写回。引擎只认工作区里的改动" |
| `/tmp` 定向 | 已有："`/tmp` 只允许临时验证；临时文件请放 `<工作区>/.scratch/`（引擎看得见）" |
| scratch 诊断 | `RealBackend.scratch_files()` + noop 时把 `scratch=...` 拼进 `change` |
| 超时保留输出 | `TimeoutExpired` 抓 `e.stdout[-400:]` 拼进返回串 |
| judge 幂等生成 | `eval()` 开头 `self._ensure_judge()` |

---

## 仍缺（P4 范围，不在本次审查结论里）

- **D1 无强制**：prompt 说"不能自己测"，但没有 sandbox。agent 现在跑 bench 也能跑。
- 工作区权限：仓库根 751、`engine/` 750，agent 读不到 engine 源码
  （今天老卡 agent 花了 30 分钟在查 `ls -la .git/index`、`stat` 属主，很可能就是这个）。

---

## 优先级

```
R1（噪声底语义） > R4（记录落盘位置） > R3（noop 太激进） > R2（changed_files 不一致）
```

R1 不修，多轮跑出来的判定全是错的；R4 不修，跑完的证据会丢。
R3/R4 都是十几行的改动，R1 需要 `--repeat`（半天）。
