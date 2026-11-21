---
name: op-optimize
description: "算子优化的搜索循环：队列 → 单个实验 → 机器判定 → 落盘记账。把「猜一个改动、手搓代码、跑一次、自己看着数字说快了」换成可复现、可中断、单模型也能跑完的闭环。配套 loop.py：拒绝重跑已证伪项、判定由脚本算不由模型说、状态落盘不占上下文。触发场景：开始优化一个算子 kernel、手上有很多候选改动要排序、写 candidates.yaml 候选清单（direction/priority）怕被 loop.py 拒跑、优化做了一半被中断要接着干、要把一轮优化收口成交付件。不适用：单次量法的口径走 op-bench；四闸四档判定走 op-gate。"
version: 1.0.1
author: DCU Toolkit Contributors
license: MIT
kind: tool
---

# 算子优化循环（单模型可执行版）

## 一、定位

| skill | 管什么 |
|---|---|
| `op-contract` | 约定：什么叫对、容差多少 |
| `op-bench` | 怎么量：口径三问、六条判据 |
| `op-gate` | 怎么判：四闸四档（执行身份 → 正确性 → 不变量 → A-B） |
| **`op-optimize`（本 skill）** | **怎么找下一个改动、怎么保证不重复劳动、怎么在上下文会丢的前提下继续** |

前三个是「一枪打准」，本 skill 是「连续开枪时不跑偏」。

**不知道该用哪个工具**（计时 / PMC / 天花板预测 / 消融 / 门禁 / 台账）→ 先看
[`references/tooling.md`](references/tooling.md) 的两列映射表 + 已弃用清单，再动手。为同一件事写第二套工具会重踩老坑。

## 二、为什么做成循环，而不是一次调优

这个项目里最贵的成本不是「想不到优化」，而是**相信自己的一次错误测量**：同一 session 出过 3 个互相矛盾的结论；微基准更快、上生产掉分；把「某段指令值 7.5%」读成「删掉能省 7.5%」，实测反而慢 1.1%；把 `SQ_LDS_BANK_CONFLICT` 当主因，标定后发现比厂商调优 matmul 还低。共同点是**模型对自己的读数过度自信**，所以循环的关键是**让「快了」这个判断不经过模型的嘴**。

## 三、循环（六步，一步一产物）

```
① 定口径 ──► ② 复现基线 ──► ③ 定位瓶颈 ──► ④ 列候选 ──► ⑤ 单实验 ──► ⑥ 记账
     ▲                                                              │
     └────────────────── 收敛？否 ─────────────────────────────────┘
```

| 步 | 产物 | 判据 |
|---|---|---|
| ① 定口径 | 一行口径字符串（形状/dtype/topk 有效数/卡） | 答不出 `op-bench` 的口径三问就先补口径 |
| ② 复现基线 | 基线数字 **现测**（不引用历史） | 与历史差 >5% 先查环境（频率/空闲卡）再继续 |
| ③ 定位瓶颈 | 一句话：「卡在 X，证据是 Y」 | 证据要能**区分**：换了它结果会变，不换不变 |
| ④ 列候选 | `candidates.yaml` 的 pending 队列，带 `direction` + priority | 每个候选写 `direction`（挂到哪条攻击面）和 `expect`（预期幅度）；缺 `direction` 脚本拒跑 |
| ⑤ 单实验 | `runs/loop/<日期>.jsonl` 一条记录 | **一次只跑一个假设**，判定由 `loop.py` 算 |
| ⑥ 记账 | ledger 一行 + candidates 状态更新 | 证伪要写**机制**，不写「试过，慢」 |

**收敛条件**（满足任一即停）：所有候选都判过且没有 ACCEPTED；剩下候选都需要「重构级」改动（先交付结论，再开新轮）；连续 3 个候选落在同一机制上 —— 说明瓶颈描述错了，回 ③。

## 四、单模型纪律（本 skill 的存在理由）

单模型（无子 agent 并行、上下文会丢）能跑完一轮优化，靠的是把下面几件事外置：

| 外置什么 | 落到哪 | 为什么 |
|---|---|---|
| **状态** | `<卡>/runs/loop/*.jsonl` + `candidates.yaml` | 上下文丢了 `loop.py status` 一跑就能接着干 |
| **判定** | `loop.py` 的 `decide()`，模型只读结论 | 「快了 12%」由脚本算，不由模型说 |
| **记忆** | `candidates.yaml:refuted`（已判死**唯一**落点） | 由脚本**强制**拒绝重跑，不靠模型记不记得；`ledger.md` §三 只是证据叙述，**不做判定** |
| **进度** | `priority` + `expect` | 中断后知道下一个该跑谁、预期多少 |
| **完成判定** | `done_when` + `loop.py verify --tag <id>` | 「做完了」由命令返回 0 裁决（§七） |

配套的三条行为判据：

1. **一次一个实验，跑完立刻记账。** 不要攒着「等全部跑完一起看」——中途上下文一压缩，攒的东西全丢。
2. **每 3 个实验回写一次 ledger。** ledger 是给「下一个模型」看的，不是给自己看的。
3. **想不出下一个候选时，回 ③ 补一个区分性实验**，不是把已有候选重跑一遍等奇迹。

## 四·五、开跑前先写作战宪章

任何一轮优化开始前，按 [`references/charter.md`](references/charter.md) 填六项（口径 / 预算 / 停止条件 / 升级契约 / 记录纪律 / 开场检查清单）。宪章把「问人」提前到开场一次，之后不再打断；没有宪章的循环会无限试错——本项目一轮做过 43 条判死只落地 1 个已知收益。

## 五、候选从哪来

按可信度排序：实测证据 > 结构推理 > 外部经验；另有本项目实测的反直觉清单、隔离单变量手法、判死写法。
全部细节见 [`references/candidates.md`](references/candidates.md)。

## 六、变体怎么生成（不手改生产文件）

用**文本变换**从生产源码生成变体（锚点替换，不落盘到手改副本），不要 fork 一份手改。每条替换锚点先 `assert src.count(anchor) == N`（同一个 kernel 常有 2~4 个 body，只改到一个会静默算错）；改完打印 diff 摘要（「变体没编进去」和「变体没效果」在计时上长得一样）；锚点用 `%` 不要用 `%%`；失败的替换报错退出，不跳过（跳过会产出「看起来跑了但没改」的变体）。

**先验假设**：任何新变体跑之前，先问「它改了哪几行、为什么这几行是瓶颈」。答不出就先补这一步。

## 七、完成判定：`done_when` 由**外部命令**裁决

单模型最容易骗自己的一句话是「已经到顶了 / 这个候选做完了」——它没有参照物。
所以完成判定被外置成候选自己的一条 shell 命令（`done_when`；执行口径见
[`references/tooling.md`](references/tooling.md)）：

```yaml
candidates:
  - id: hip_glm_contract
    direction: bypass-tilelang-codegen
    status: pending
    done_when: 'python3 -I scripts/opcheck.py correctness --contract "$OPS_CARD/contract.yaml" --shape decode-glm53 --variants hip-glm'
```

`python3 loop.py -C <卡> verify --tag <id>` 跑它，`rc=0` 才 adopted：`run` 对「adopted 但没写
`done_when`」直接 rc=2；`verify` 在 rc≠0 或超时（记 124）时不改 YAML、只回显输出尾部 20 行；
事件带 `done_when_rc` 写进 `runs/loop/*.jsonl`。判的只有退出码（组合命令用 `&&`，日志里出现
"pass" 不算数）；写不出别人能复跑的命令，说明这条候选还没到「完成」那一步——参考卡片的
`maxn32` 就因此从 `adopted` 降回 `pending`。

## 八、判定口径（三层，别混）

```bash
python3 loop.py -C <卡> run <tag> [--skv ...] [--force]
```

1. **性能（硬门，MAD）**：`3 × max(MAD_base, MAD_cand)` 之内一律 INCONCLUSIVE（按噪声内处理，不写成收益）。base 每轮重测（漂移同时作用在两边）。
2. **精度主口径（硬门）**：生产 logit 尺度下对 **fp32 稠密参考**的 `max_abs ≤ contract 容差`，且 `nonfinite == 0`。
3. **精度压力档（只报警）**：更高 qscale 下判 **相对退化倍数** `cand_rel / base_rel`。**绝对容差不能用在压力档上** —— base 自己在 qscale=10 就有 1.55e-2，会把正确改动一起否掉（本 skill 第一次跑就栽在这，`maxn32` 因此被判死，而它在主口径下与 base 精度相同）。

判定输出四档：

| 判定 | 含义 | 下一步 |
|---|---|---|
| `ACCEPTED_BITEXACT` | 显著更快且逐位一致 | 直接落生产并复测 |
| `ACCEPTED_NUMERICS_PENDING_E2E` | 显著更快但非逐位 | **要**补端到端质量门（GPQA/HumanEval 一档），微基准通过不算数 |
| `INCONCLUSIVE` / `REJECTED_NO_GAIN` | 噪声内 / 更慢 | 按 §九 记一笔（实现坏 / 方向死），别重跑 |
| `REJECTED_ACCURACY` | 精度或 nonfinite 不过 | 同上，并且要记下是哪个 qscale |

> **压力档报警 ≠ 否决。** `maxn32` 在 qscale=10 退化 2.0×，主口径完全一致 —— 这类只有端到端能裁决，不要用微基准替它做决定。

## 九、失败分两类，别混

一次实验失败，归因只有两种，**命令不同、后果不同**：**实现写坏了**（`impl-broken`，方向预算不动，连挂 3 次判死该实现）与 **攻击面不成立**（`direction-refuted`，方向预算 +1，2 次封方向）。把实现的 bug 记成方向证伪，等于用一个实现错误处死一整条路。

完整判据、上限、`candidates.yaml` schema 与 `why` 写法见
[`references/failure-taxonomy.md`](references/failure-taxonomy.md)。

## 十、交付件格式

一轮优化收口时，产物是这四件：① **结论**（相对基线/对手的倍数，带口径字符串）；② **前提**（形状 / 卡 / DTK / 变体前置条件）；③ **已证伪假设**（带机制的清单，即 `candidates.yaml:refuted`）；④ **复现脚本**（一条命令重跑出结论，路径写进 skill 树而不是 `/tmp`）。

## 十一、一条命令看懂现状

```bash
python3 loop.py -C <卡> status|queue|report   # 现状 / 方向预算 + 待做 + 已判死附机制（违规 rc=2） / 交付表
python3 loop.py -C <卡> impl-broken <tag> --why "..."       # 实现写坏了
python3 loop.py -C <卡> direction-refuted <tag> --why "..." # 攻击面不成立
python3 loop.py -C <卡> verify --tag <id>                   # 跑 done_when；rc=0 才 adopted（§七）
```

参考实现：`op-sparse-mla`（32 条判死 + 3 条待做 + 3 个方向，`candidates.yaml` 可直接抄 schema）。
