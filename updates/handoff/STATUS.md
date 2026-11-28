# STATUS · 2026-11-28 · lead 验收汇总（这是当前唯一入口）

> 维护者：lead（验收方）。每次验收后更新本文件 + 计划文件顶部的状态横幅。
> 上一版：`REVIEW-P0-2026-11-28.md`（当时 P0 打回，已过期，保留作记录）。

---

## 一、判定表（lead 独立复跑，不是看你的结论）

| 项 | 判定 | 我复跑到的证据 |
|---|---|---|
| **P0 judge 对象层** | ✅ **通过（v2）** | A1–A7 全过，见下 |
| **P1 基线晋升 × claim** | ✅ **通过** | B1 不晋升 / B2 晋升 / B3 有原因 |
| **P2 `.artifacts/` 迁移** | ✅ **基本通过** | 35 条账本在 `.artifacts`；`base.json` 已迁 |

### P0 关键复验（A7 是我加的，也是上一轮打回的那条）

```
A1 judge 生成      : 3 件齐全（judge.py / tolerance.json / CALIBRATION.md）
A2 judge CLI       : {"max_ulp":0.0,"cosine":1.0,"evidence_kind":"..."}
A3 冻结证据        : 缺字段 = 无；含 sha256 / calibration_cmd / calibrated_at
A4 sha256 自校验   : match = True（我按"除 sha256 外规范化"独立重算）
A5 claim 判 FAIL   : ✗ 有精度证据 —— self-consistent-only 无独立核验 ✔ 正确
A6 拒绝覆盖任务定义 : rc=2（真实退出码，不是管道）
A7 两条路径各一例   : op-silu-mul → self-consistent-only ✔
                     op-fused-add-rmsnorm → vendor_library ✔   A7 = PASS
```

> 上一轮打回的根因（`opverify.parse_yaml` 读不了 `variants.yaml`）已修：
> 现在 `_read_yaml()` 优先 PyYAML。**修得对，而且两条路径都验过了。**

### P1 复验

```
B1 无 judge 容差 → run reg-resident → 「不晋升 fa-rms-mr2：claim 未通过（evidence_unverified）」
   base.json 不存在 ✔
B2 有 vendor_library 容差 → 「基线晋升：下一轮以 fa-rms-mr2 为参照（…；claim PASS）」
   base.json = {"base":"fa-rms-mr2","ms":0.1416,"evidence_kind":"vendor_library",…} ✔
B3 不晋升原因出现在日志 ✔
```

### P2 复验

```
.artifacts/arm-C/op-fused-add-rmsnorm_attempt-1/runs/loop/      （沙箱形态）
.artifacts/local/op-fused-add-rmsnorm_tookit-skills/runs/loop/  （本地形态）
账本：.artifacts 下 35 条；卡目录下 4 条（历史遗留）
```

---

## 二、遗留问题（不阻塞，但要记账）

| # | 问题 | 建议 |
|---|---|---|
| 1 | 卡目录还有 **4 条历史账本**（迁移前产生） | 迁进 `.artifacts` 或标 `legacy`，否则 `_loop_evidence` 会重复计数 |
| 2 | `judge` 容差只对**最小形状**校准（op-fused 实际跑 prefill-8k，校准用 tiny） | `tolerance.json` 加 `shapes: [...]` 记录覆盖范围；形状敏感的量要分档 |
| 3 | `claim` 加严后，历史记录 `evidence_kind` 为空 → `held` 变多 | 这是预期的 fail-closed；但报告口径要统一（`held` ≠ FAIL，是"缺证据待补"） |
| 4 | `.artifacts/local/<随机串>_tookit-skills/` 目录名带随机后缀（如 `tmpqtjyiedz`） | 固定成 `<op>_<card>` 或加 `--run-id`，否则每次跑都新建一个目录 |

---

## 三、下一步：P3（工作流反转）

**执行计划：** [`HANDOFF-2026-11-28-P3-driver-plan.md`](HANDOFF-2026-11-28-P3-driver-plan.md)

核心引擎相对 ForgeKernel `driver.py` 的九处缺口（详见该文件第一节）：

| # | 缺什么 | 先做哪步 |
|---|---|---|
| G1 | "一轮"的定义（agent 交代码 → 测 → 记 → 决定下一轮） | P3.2 |
| G2 | agent 调用接缝（可注入、可 fake） | **P3.1** |
| G3 | 工作区版本语义 + relaxed-revert | P3.2 |
| G4 | 噪声底 + plateau 停止条件 | P3.3 |
| G5 | probe 子循环（失败重试直到拿到记录） | P3.5 |
| G6 | 重试 / 超时 / 心跳 | P3.5 |
| G7 | pre-green 语义（未过正确性前记 NEUTRAL） | P3.2 |
| G8 | resume / prefix-reuse | P3.6（后置） |
| G9 | 控制流与副作用分离，fake backend 单测 | **P3.1** |

**G2/G9 必须先做**：不先把副作用抽到 `Backend` 后面，接上真 agent 之后每个 bug 都要复现 20 分钟。

---



见 [`HANDOFF-2026-11-28-workflow-inversion-roadmap.md`](HANDOFF-2026-11-28-workflow-inversion-roadmap.md) 的 P3 段。

**开工前需要你回的三个问题**（我不替你拍，但给了建议）：

1. **`iterate` 的 agent 调用方式**：子进程 `claude -p` 还是外部传入（`--agent-cmd`）？
   建议 **`--agent-cmd` 必填**，默认不给 —— 测试时可以用假实现，避免把测试绑死在某个 CLI。
2. **一轮的边界**：agent 交完就结束（不管它跑多久），还是给 wall-clock/轮次双限？
   建议**双限**（照 opbench 的 budget 机制），否则一个卡住的 agent 会吃掉整晚。
3. **反馈里要不要带"死路清单"**：上一轮 `refuted_directions` 要不要显式喂回去？
   建议**要**，否则 agent 会反复试同一个被证伪的方向（这是我们 `refuted` 机制的价值所在）。

---

## 四、我现在做什么

- 收到你 P3 的 `RESULT-P3` 后，重跑 P3 的 D1–D4 验收（**D1 是 key：agent 会话里不得有任何 bench 命令**）；
- 同时准备 P4（边界与绊线）的规格。

**不用回我 P0/P1/P2 了 —— 都过了。** 回上面三个问题即可。
