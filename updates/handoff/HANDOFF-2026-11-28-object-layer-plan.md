<!-- lead 状态横幅：每次验收后更新，放在最顶部确保被看到 -->
> ## ✅ 状态（lead 维护，先看这里）
>
> **P0 / P1 / P2 全部通过**（2026-11-28 13:1x，lead 独立复跑 A1–A7 / B1–B3 / C1–C3）
> 汇总与遗留问题：[`STATUS.md`](STATUS.md)
>
> 下一步：**P3 工作流反转**，见
> [`HANDOFF-2026-11-28-workflow-inversion-roadmap.md`](HANDOFF-2026-11-28-workflow-inversion-roadmap.md)。
> 开工前请回 `STATUS.md` 第三节的三个问题。

---

# HANDOFF · 2026-11-28 · 对象层执行计划（lead 定稿 → work 会话执行）

> 按 `updates/README.md` 的约定：这是**交接快照**，旧的保留。
> 上一份：`handoff/HANDOFF-2026-11-25.md`。
>
> **角色**：work 会话执行；lead 只写契约与验收，**不改 `engine/**` 任何文件**。
> 返回报告写到同目录：`handoff/RESULT-P0-2026-11-28.md`（P1/P2 同理）。

---

## 一、总目标（本轮）

把**对象层**做出来：从 `description.md + shapes.json`（**不给 reference 实现**）
生成 judge → GPU 校准并冻结容差 → 落 `.artifacts/` → `claim` 五项 PASS。

这是 driver 之前的唯一卡点：**judge 不正，driver 只会把错误放大**。
（两个会话独立得出同一结论：先对象，后循环。）

---

## 二、铁律（违反即打回）

1. **竖切片优先**：一次只做一件事，做完写报告，不要并行铺开。
2. **验收必须用仓库真实卡**（`op-silu-mul` 或 `op-fused-add-rmsnorm`），
   不许拿 `/tmp` 下的临时卡当最终验收。
3. **只能动**：`engine/loop.py`、`engine/templates/`、`engine/judge.py`（新建）。
   `Plugins/ops/skills/**`、`toolkit-opbench/**`、`harness/arms/c.py` **一律不动**。
4. 每次提交用**路径受限提交**，commit message 末尾一行 `To the other session: <一句话>`。
5. **不许为了让测试变绿而改判据。** 判据不过就写「不过」并说明为什么。

---

## 三、P0 —— judge 对象层（最高优先级）

### 任务

1. 新建 `Plugins/ops/engine/judge.py`。
2. `loop.py` 改成**消费** `tolerance.json`，不再从 `contract.yaml` 直接读容差。
3. `claim` 的第五项从「有没有精度证据」改成**按证据种类判**。

### 接口契约（lead 定，不得改；要改先写文件问）

**输出目录**（相对 run 根）：

```
.artifacts/<arm>/<op>_<id>/judge/
    judge.py            # 可执行，CLI 见下
    tolerance.json      # 冻结的容差
    CALIBRATION.md      # 人读的校准记录
```

**`judge.py` 的 CLI（唯一被允许的调用方式；loop 不得 import 它的内部函数）**：

```bash
python3 judge.py --shape <shape-name> --json <out.json>
# 输出: {"shape": "...", "max_ulp": 0.5, "cosine": 0.999998, "nonfinite": 0,
#        "ref_absmax": 15.63, "evidence_kind": "independent_reference"}
```

**`tolerance.json` 最小字段**：

```json
{"max_ulp": 2.0, "cosine": 0.99999, "rtol": 0.02, "atol": 0.01,
 "evidence_kind": "independent_reference|vendor_library|self-consistent-only",
 "calibrated_at": "2026-11-28T13:00:00", "calibration_cmd": "<完整命令>",
 "sha256": "<tolerance.json 除 sha256 字段外规范化后的 sha256>"}
```

**`claim` 第五项新判据**：

| evidence_kind | 判定 |
|---|---|
| `independent_reference` | PASS |
| `vendor_library` | PASS |
| `self-consistent-only` | **FAIL**，记录标 `evidence: self-consistent-only`，理由：「无独立核验，不能声称数值正确」 |
| 缺失 | FAIL |

**失败语义（fail-closed）**：生成不出 judge 或校准不出容差 → **rc=2，且不写
`tolerance.json`**。宁可让 run 跑不了，也不许拿未校准的容差进账本。

### 验收（逐条跑，输出逐字贴进 RESULT，不许转述）

```bash
cd /workspace/tookit-skills

# A1 生成 judge（仓库真实卡，不给 reference）
rm -rf /tmp/a0 && mkdir -p /tmp/a0/judge
python3 -I Plugins/ops/engine/judge.py generate \
    --card Plugins/ops/skills/op-silu-mul --out /tmp/a0/judge/ ; echo "rc=$?"
ls /tmp/a0/judge/        # 期望 judge.py tolerance.json CALIBRATION.md 三件齐全

# A2 judge 可执行且给出证据
python3 -I /tmp/a0/judge/judge.py --shape s0 --json /tmp/a0/judge/out.json ; echo "rc=$?"
python3 -c "import json;d=json.load(open('/tmp/a0/judge/out.json'));print(d['evidence_kind'], d['max_ulp'], d['cosine'])"

# A3 容差冻结证据齐全
python3 -c "import json;d=json.load(open('/tmp/a0/judge/tolerance.json'));assert d['sha256'] and d['calibration_cmd'] and d['evidence_kind'];print('OK')"

# A4 容差在 run 期间不可被改：run 前后 sha256 相同
sha256sum /tmp/a0/judge/tolerance.json

# A5 claim 第五项按证据种类判（self-consistent-only 必须 FAIL）
#    造一条 evidence_kind=self-consistent-only 的记录，claim 应 FAIL 并打印理由

# A6 run 不接受覆盖任务定义的参数
python3 -I Plugins/ops/engine/loop.py -C Plugins/ops/skills/op-silu-mul \
    run baseline-ref --tolerance 0.1 2>&1 | tail -2   # 期望：报错 rc≠0
```

**P0 通过条件**：A1–A6 全部符合期望，且 A5 的 FAIL 理由能看出「无独立核验」。

---

## 四、P1 —— 基线晋升 × claim 联动

`base.json` 写入前必须跑 `claim` 的**同一套检查函数**；不过就不晋升，并在日志写明原因。
（否则一个没过审计的结果被升成基线，后面每代都建在它上面，整条链就脏了。）

验收：

```bash
# B1 一条 claim 会 FAIL 的记录（self-consistent-only）→ 不产生/不更新 base.json
# B2 一条 claim PASS 的记录 → base.json 更新，内容含 source_tag / evidence_kind
# B3 日志里有「不晋升原因」
```

---

## 五、P2 —— `.artifacts/` 迁移（与三臂隔离对齐）

`runs/loop/` 从卡目录迁到 `.artifacts/<arm>/<op>_<id>/runs/loop/`；卡目录不再写运行时数据。

**为什么不用软链**：opbench 的 `_loop_evidence` 用递归 glob
`glob(cwd/**/runs/loop/*.jsonl, recursive=True)`，`.artifacts/...` 能被直接找到。

验收：

```bash
# C1 有账本落在 .artifacts 下
find .artifacts -path '*/runs/loop/*.jsonl' | head -3
echo "count=$(find .artifacts -path '*/runs/loop/*.jsonl' | wc -l)"

# C2 卡目录干净（运行时不再写卡）
(cd Plugins/ops/skills/op-silu-mul && git status --porcelain .) ; echo "（应为空）"

# C3 递归 glob 能找到（模拟 opbench 侧）
python3 -c "import glob;print(glob.glob('/workspace/tookit-skills/.artifacts/**/runs/loop/*.jsonl', recursive=True)[:3])"
```

---

## 六、汇报格式（每完成一项写一个新文件）

路径：`updates/handoff/RESULT-P0-2026-11-28.md`（P1/P2 同理）。三段：

1. **改了什么**：文件 + 一句话（含 commit hash）
2. **验收命令的实际输出**：逐字贴，含 `rc=`，**不许转述**
3. **没做到的 / 有疑问的**：直说。做不到比假装做到有价值。

> lead 会按上面每条命令**自己重跑一遍**核对；对不上就打回，不看结论只看输出。

---

## 七、lead 接下来做什么

1. 不改 `engine/**`；
2. 每收到一份 `RESULT-P*.md`，重跑该 P 项全部验收命令，给出「通过 / 打回 + 具体哪条不符」；
3. 同时起草 **P3（hook 硬门）** 规格 —— `PreToolUse exit 2` 拦「绕过引擎自己测」，
   那是 opbench 侧，等 P0–P2 过了再排。

---

## 八、现在就能开始

从 **P0** 开始。第一件事：把 `judge.py` 的 CLI 骨架和 `tolerance.json` 写入逻辑写出来
（生成质量可以先粗糙，但**接口和 fail-closed 语义必须先对**）。
第一版做完立刻写 `updates/handoff/RESULT-P0-2026-11-28.md`。
