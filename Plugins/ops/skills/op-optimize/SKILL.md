---
name: op-optimize
description: "算子优化的搜索循环：队列 → 单个实验 → 机器判定 → 落盘记账。把「猜一个改动、手搓代码、跑一次、自己看着数字说快了」换成可复现、可中断、单模型也能跑完的闭环。配套 loop.py：拒绝重跑已证伪项、判定由脚本算不由模型说、状态落盘不占上下文。触发场景：开始优化一个算子 kernel、手上有很多候选改动要排序、写 candidates.yaml 候选清单（direction/priority）怕被 loop.py 拒跑、优化做了一半被中断要接着干、要把一轮优化收口成交付件。不适用：单次量法的口径走 op-bench；四闸四档判定走 op-gate。触发场景补充：**新算子第一次做优化**（先 scripts/op_bootstrap.py 建卡，再补 reference、跑一次基线，四步缺一不可）；**不确定该用 HIP 还是 Triton**（跑 scripts/lang_select.py，它按 roofline 数字判，不要默认 Triton）；**多轮自动推进**（scripts/advance.py --forge，队列空了自己造变体）；**新卡登记漏位置**（scripts/card_sync.py 一个入口管 5 处）。完整用法见 docs/HOW-TO-USE.md。"
version: 1.0.1
author: DCU Toolkit Contributors
license: MIT
kind: tool
---

# 算子优化循环（单模型可执行版）

## 不适用场景

- 只想量一个数、不做迭代 → 走 `op-bench`
- 只做四闸四档判定、不改代码 → 走 `op-gate`
- 只是查某算子的实现/形状/档位 → 查对应的 `op-<算子名>` 卡，不启动本循环
- 编译期问题（aicc/编译选项/寄存器）→ 走 `op-hip-compile`

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

### 5.1 实现路线必须先分流（不要默认从 Triton 起步）

> 依据：opbench 三臂对照（2026-11-25）——裸 CC / 本 skill / SkillHub **三臂都没写过一行 HIP**，
> 而本仓 `op-rmsnorm-fused/runs/forged/` 下已有 4 个实测更快的现成 HIP kernel。
> 缺的不是能力，是**没有任何一条规则说「什么时候该考虑 HIP」**。

按这个顺序排除，**不要跳步**：

| 步 | 判据 | 动作 |
|---|---|---|
| 1 | 现成库实现已达 **>90%** 地板带宽/算力 | 用它。别写 kernel |
| 2 | 未达 90%，且 roofline 显示 **>1.2×** 空间 | **先评估手写 HIP**，再考虑 Triton |
| 3 | Triton 已试且未达标 | 走 `op-hip-compile` 调编译选项 |
| 4 | 被 launch 地板支配（`launch_us / t_actual > 0.30`） | 别优化 kernel，做融合/减 launch |

**为什么 2 排在 Triton 前面**：Triton 的优势是快速迭代，劣势是寄存器/向量化控制不如手写。
访存受限的 elementwise/归约类算子，手写 HIP 通常能多拿 15~25% 带宽
（本仓实测：rmsnorm 类 2 个算子上都是 40%+ 差距）。

**必须排除的反例**（写了也白写）：
- 纯 GEMM / 纯 attention：库实现已接近硬件上限
- 被 launch 地板支配的小算子：瓶颈不在 kernel 内部

**机器判据**（写在 `candidates.yaml` 里即可被 `check_means.py` 看到）：

```yaml
- id: hand-hip-vec
  direction: occupancy
  action_class: improve-locality     # 落在允许集合里才不会被 loop.py 拒跑
  means: 手写 HIP：中间结果留寄存器 + 16B 向量访存，不落 LDS
  predicted_effect: -20%~-40%
  expected_bottleneck: memory-bound
```

判据：当 `utilization < 0.90` 且 `remaining > 0.20` 时，**候选列表里必须有至少一条
`action_class: improve-locality` 且 `means` 提到寄存器/向量化的条目**，否则说明路线分流漏了。

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

完整判据、上限、`candidates.yaml` schema 与 `why` 写法见 [`references/failure-taxonomy.md`](references/failure-taxonomy.md)。

### 九·五、坏了先归位：6 条故障定位面

§九 分「怎么记账」；[`references/failure-surfaces.yaml`](references/failure-surfaces.yaml) 分「是哪一类故障」——**先定位，再记账**（归因错了账也是错的）。出现「崩了 / 算错了 / 改了没效果」时按序过一遍（代价从低到高），每条带可跑 `minimal_repro` + 能区分的判据 + D 编号：`device-bind`（纯 torch 正常但自研 kernel 崩 ⇒ `HIP_VISIBLE_DEVICES` 重映射，**最贵 5 小时**）、`uninit-struct`（结构体缺 `{}`）、`build-stale`（产物没重编 / 改错文件）、`verifier-lies`（校验器说通过但没生效）、`wrong-variable`（没控制变量）、`minimal-repro`（大 kernel 先减到 5 行）。

## 十、交付件格式

一轮优化收口时，产物是这四件：① **结论**（相对基线/对手的倍数，带口径字符串）；② **前提**（形状 / 卡 / DTK / 变体前置条件）；③ **已证伪假设**（带机制的清单，即 `candidates.yaml:refuted`）；④ **复现脚本**（一条命令重跑出结论，路径写进 skill 树而不是 `/tmp`）。

## 十一、一条命令看懂现状

```bash
python3 loop.py -C <卡> resume                 # * 先跑这条：下一个该跑谁（上下文丢了也能接着干）
python3 loop.py -C <卡> enumerate              # 方向菜单（OPEN/SEALED），direction 只能从这挑
python3 loop.py -C <卡> status|queue|report    # 现状 / 方向预算 + 待做 + 已判死附机制（违规 rc=2） / 交付表
python3 loop.py -C <卡> impl-broken <tag> --why "..."       # 实现写坏了
python3 loop.py -C <卡> direction-refuted <tag> --why "..." # 攻击面不成立
python3 loop.py -C <卡> verify --tag <id>                   # 跑 done_when；rc=0 才 adopted（§七）
```

### 主张门：什么时候「快了 x%」才算数

`claim_gate` 把「跑过」和「能主张」分开——**约束面是断言面，不是动作面**：

| 门 | 判据 | 缺了会怎样 |
|---|---|---|
| 口径 | `caliber{what,alloc,state,shape,device}` 齐全 | `caliber_incomplete` |
| 显著性 | `\|Δ\| > 3×max(MAD)` | `not_significant` |
| 正确性 | 精度探针跑过且不是 `--no-acc` | `accuracy_missing` |

`run` **永不拒绝**（`--no-acc` 也能跑，用于快速筛方向），
但没过门的记录进 `held`，不能升级成交付结论：
`report` 默认不显示它们，`verify`/`adopt` 对 held 候选 rc=2。
补齐后重跑即可，`report --all` 会给出逐条补齐方法。

> 方向从哪来：不要凭空想。`references/surfaces.yaml` 预枚举了 11 条攻击面，每条挂一条 `pipeline-laws.md` 的定律。`enumerate` 会标出 `SEALED`（已证伪 2 次的方向），开跑前就能看见，不必跑一次才知道被封。

参考实现：`op-sparse-mla`（43 条判死 + 3 条待做 + 3 个方向，`candidates.yaml` 可直接抄 schema）。

### 完整链路的示范：`cases/`

[`cases/001-swiglu-fused-vs-split/`](cases/001-swiglu-fused-vs-split/) 是一次
**从场景到结论**的完整走通，建议新任务先读它：

| 环节 | 该 case 怎么做的 |
|---|---|
| 场景判定 | 按 `scenarios.yaml` 的 detect 判为 bandwidth-bound |
| 取攻击面 | `enumerate --scenario bandwidth-bound` → 第 1 优先 fusion-boundary |
| 定位 | `lightop.FuseBiasSwiGLU` + signature + 一行 smoke_test |
| 实测 | 融合 1.5696 ms vs 分拆 3.5242 ms = 2.25x，3×MAD 显著 |
| 分母 | 按访存比选口径：读 2 写 1 ⇒ copy 峰值（不是只读峰值） |
| 结论 | 达成率未到顶 ⇒ **没到顶**，下一个攻击面已列出 |

里面有一条特别值得读的**踩坑记录**：用 25MB buffer 测带宽得到 4571 GB/s，
是小 buffer 命中 L2 的假象。所有带宽数字必须用 ≥1GB buffer 实测。

### 知识层到底值不值钱：`cases/knowledge-audit.md`

10 个 case 的事后审计，回答「卡/场景表/判据有没有真的改变做法」：

- 知识层参与了 **7/11（64%）** 的关键决定，但**纯粹靠它定案的只有 4 条**
- 它的主要价值是**防止方向性错误**（别直接改 kernel、别指望同引擎重叠），
  **不是**告诉你优化点在哪
- 4 条判据是**被教训出来的**（等 FLOPs、记账怀疑、SQ 扫描、launch 地板）
  ⇒ 本仓的增量主要来自实战反哺，不是先验设计

**它没有回答、也回答不了的是：没有知识层会不会一样好。** 那需要 A/B，
协议写在文末（需要独立执行者——我知道预期答案，无法假装不知道）。
