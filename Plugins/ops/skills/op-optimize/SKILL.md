---
name: op-optimize
description: "算子优化的搜索循环：队列 → 单个实验 → 机器判定 → 落盘记账。把「猜一个改动、手搓代码、跑一次、自己看着数字说快了」换成可复现、可中断、单模型也能跑完的闭环。配套 loop.py：拒绝重跑已证伪项、判定由脚本算不由模型说、状态落盘不占上下文。触发场景：开始优化一个算子 kernel、手上有很多候选改动要排序、写 candidates.yaml 候选清单（direction/priority）怕被 loop.py 拒跑、优化做了一半被中断要接着干、要把一轮优化收口成交付件。不适用：单次量法的口径走 op-bench；四闸四档判定走 op-gate。"
version: 1.0.1
author: DCU Toolkit Contributors
license: MIT
---

# 算子优化循环（单模型可执行版）

## 一、定位

| skill | 管什么 |
|---|---|
| `op-contract` | 约定：什么叫对、容差多少 |
| `op-bench` | 怎么量：口径三问、六条铁律 |
| `op-gate` | 怎么判：四闸四档（执行身份 → 正确性 → 不变量 → A-B） |
| **`op-optimize`（本 skill）** | **怎么找下一个改动、怎么保证不重复劳动、怎么在上下文会丢的前提下继续** |

前三个是「一枪打准」，本 skill 是「连续开枪时不跑偏」。

**不知道该用哪个工具**（计时 / PMC / 天花板预测 / 消融 / 门禁 / 台账）→ 先看
[`references/tooling.md`](references/tooling.md) 的两列映射表 + 已弃用清单，再动手。为同一件事写第二套工具会重踩老坑。

## 二、为什么必须做成循环，而不是一次调优

这个项目里最贵的成本不是「想不到优化」，而是**相信自己的一次错误测量**：同一 session 出过 3 个互相矛盾的结论（形状测错）；微基准 bit-identical 且更快、上生产掉分（输入布局 ≠ 生产 kernel 输出布局）；把「某段指令值 7.5%」读成「删掉能省 7.5%」，实测删掉反而慢 1.1%（那段是 prefetch）；把 `SQ_LDS_BANK_CONFLICT` 当主因，标定后发现自己比厂商调优 matmul 还低。

**共同点：模型对自己的读数过度自信。** 所以循环的关键不是「多试几个想法」，是**让「快了」这个判断不经过模型的嘴**。

## 三、循环（六步，一步一产物）

```
① 定口径 ──► ② 复现基线 ──► ③ 定位瓶颈 ──► ④ 列候选 ──► ⑤ 单实验 ──► ⑥ 记账
     ▲                                                              │
     └────────────────── 收敛？否 ─────────────────────────────────┘
```

| 步 | 产物 | 判据 |
|---|---|---|
| ① 定口径 | 一行口径字符串（形状/dtype/topk 有效数/卡） | 答不出 `op-bench` 的口径三问就别开始 |
| ② 复现基线 | 基线数字 **现测**（不引用历史） | 与历史差 >5% 先查环境（频率/空闲卡）再继续 |
| ③ 定位瓶颈 | 一句话：「卡在 X，证据是 Y」 | 证据必须是**区分性**的：换了它结果会变，不换不变 |
| ④ 列候选 | `candidates.yaml` 的 pending 队列，带 `direction` + priority | 每个候选要写 `direction`（挂到哪条攻击面）和 `expect`（预期幅度）；缺 `direction` 脚本直接拒跑 |
| ⑤ 单实验 | `runs/loop/<日期>.jsonl` 一条记录 | **一次只跑一个假设**，判定由 `loop.py` 算 |
| ⑥ 记账 | ledger 一行 + candidates 状态更新 | 证伪要写**机制**，不写「试过，慢」 |

**收敛条件**（满足任一即停，别硬凑）：
- 所有候选都判过且没有 ACCEPTED；
- 剩下候选都需要「重构级」改动（先把结论交付，再开新轮）；
- 连续 3 个候选落在同一机制上 —— 说明瓶颈描述错了，回 ③。

## 四、单模型纪律（本 skill 的存在理由）

单模型（无子 agent 并行、上下文会丢）能跑完一轮优化，靠的是把下面四件事外置：

| 外置什么 | 落到哪 | 为什么 |
|---|---|---|
| **状态** | `<卡>/runs/loop/*.jsonl` + `candidates.yaml` | 上下文丢了 `loop.py status` 一跑就能接着干 |
| **判定** | `loop.py` 的 `decide()`，模型只读结论 | 模型不能自己说「快了 12%」 |
| **记忆** | `candidates.yaml:refuted`（已判死**唯一**落点） | 由脚本**强制**拒绝重跑，不靠模型记不记得；`ledger.md` §三 只是证据叙述，**不做判定** |
| **进度** | `priority` + `expect` | 中断后知道下一个该跑谁、预期多少 |
| **完成判定** | `done_when` + `loop.py verify --tag <id>` | 模型说「做完了」不算数，命令返回 0 才算（§七） |

配套的三条行为约束：

1. **一次一个实验，跑完立刻记账。** 不要攒着「等全部跑完一起看」——中途上下文一压缩，攒的东西全丢。
2. **每 3 个实验回写一次 ledger。** ledger 是给「下一个模型」看的，不是给自己看的。
3. **不许并行等奇迹。** 想不出下一个候选时，正确动作是回 ③ 补一个区分性实验，不是把已有候选重跑一遍。

## 四·五、开跑前先写作战宪章

任何一轮优化开始前，按 [`references/charter.md`](references/charter.md) 填六项：**口径 / 预算 / 停止条件 / 升级契约 / 记录纪律 / 开场检查清单**。

没有宪章的循环会无限试错——本项目一轮做过 43 条判死只落地 1 个已知收益。
**宪章把「问人」提前到开场一次，之后不再打断。**

## 五、候选从哪来（按可信度排序）

1. **实测证据**：PMC / mem-only / 消融已经指出某个单元或某段代码 —— 最高优先级。
2. **结构推理**：从寄存器预算、LDS 容量、带宽模型算出来的可行性边界。算出来「不可能」的要立刻记进证伪清单（例：`Q(128 VGPR) + acc_o(128 VGPR) = 256` ⇒ bm64/thr256 必然溢出，与写法无关）。
3. **外部经验**：别人实现里的写法。**最低优先级** —— 本机 ISA 可能根本没有那条指令（实测 gfx938 没有 `ds_read_b64_tr_b16`，而在 gfx90a/gfx950 上有）。

**反直觉清单**（本项目实测，可直接迁移）：
- 「把小 tile 合成大 tile 更高效」→ 错。4 路切分是**寄存器生命周期分段**；合并后生命周期全程重叠，直接顶爆 VGPR 上限（spill 30，慢 2.12×）。
- 「减少指令数就能提速」→ 错。删掉最大的一块 VALU（每块每线程 32 条 mul）只换回 1.1%。
- 「某段指令占比 N% ⇒ 删掉省 N%」→ 错。它可能已经被重叠吃掉了（那一处删掉反而慢 1.1%）。
- 「把寄存器数组搬到 shared 换 occupancy」→ 通常错。LDS 流量正是另一侧要砍的东西，两条方向互相打架。
- 「工具算出来的 occupancy 是可信的」→ 先反验常数。`VF_LIMIT=65536` 让 2 CTA/CU 显示成 1，据此下的「无 CTA 可切」结论整条是错的。**判据：常数若推出物理不可能（0 个 CTA、占用率 >100%），先判常数错，不判数据错。**
- 「状态标了 adopted 就等于可用」→ 错。代码可能在后续清理里被删掉了（清理实验脚手架时把某个已 adopted 变体的实现一起带走，状态却一直停在 adopted，下一次以为它能用）。**判据：状态必须能被机器验证——文件在不在、跑不跑得起来；验证不了就降级成 `recorded`。**
- 「历史数字可以跨会话比」→ 错。同一 kernel 两次会话差 10%（flash_mla 0.6454 vs 0.7193）。**只比同一张表内的比值，基线每次现测。**
- 「低占用的收益可以按比例外推」→ 错。同一 kernel 1→2 CTA/CU 是 68%，换一个 kernel 只有 27.5%；外推只能当上界。

**隔离一个变量的手法**（比「多试几次」有用得多）：改一个**不改变语义**的量，让输出 bit-identical，只留一个变量在动。例：动态 LDS 申请量 28672→40960（kernel 只读前 28672）强制 1 CTA/CU，就单独量出了 occupancy 的敏感度。同类：改 launch 配置不改 kernel 逻辑、改分配不改算法。

**判死的写法**：只写「试过，慢」的条目等于没写——下一个模型还会再试一遍。必须写**机制**（为什么慢、被什么挡住、在什么条件下会反转），机制能迁移到别的算子，结论不能。

## 六、变体怎么生成（不要手改生产文件）

用**文本变换**从生产源码生成变体（锚点替换，不落盘到手改副本），不要 fork 一份手改：

| 规则 | 理由 |
|---|---|
| 每个替换锚点先 `assert src.count(anchor) == N` | 同一个 kernel 常有 2~4 个 body，**只改到一个 body 会静默算错**（本项目 4 次假失败全栽在这） |
| 改完必须打印 diff 摘要 | 「变体没编进去」和「变体没效果」在计时上长得一模一样 |
| 锚点用 `%` 不要用 `%%` | f-string 里写错一次，变体语法错误，浪费一轮 |
| 失败的替换要**报错退出**，不要跳过 | 跳过会产出「看起来跑了但没改」的变体 |

**先验假设**：任何新变体跑之前，先问「它改了哪几行、为什么这几行是瓶颈」。答不出就别跑。

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

`python3 loop.py -C <卡> verify --tag <id>` 跑它。**rc=0 才允许 adopted** —— 这是 `run` /
`verify` / `adopt` 三条路径上的硬校验，不是提示词：

| 规则 | 落点（机器强制，不靠模型记） |
|---|---|
| `status: adopted` 必须带 `done_when` | `run` 前置校验：adopted 但没写命令 ⇒ rc=2，拒绝再跑 |
| **只有命令返回 0 才允许 adopted** | `verify`：rc≠0 或超时（记 124）⇒ **不改 YAML**，只回显输出尾部 20 行 |
| 事件可审计 | `runs/loop/*.jsonl` 记录带 `done_when_rc`；`verify` 写 `VERIFIED` / `VERIFY_FAILED` |
| 没有「不跑命令就采纳」的旁路 | `adopt` 是 `verify` 的旧名（等价）；没写 `done_when` 时它 rc=2 |

**判的只有退出码**（组合命令用 `&&`，日志里出现 "pass" 不算数）；写不出别人能复跑的命令 ⇒
这条候选还没到「完成」那一步——参考卡片的 `maxn32` 就因此从 `adopted` 降回 `pending`。

## 八、判定口径（三层，别混）

```bash
python3 loop.py -C <卡> run <tag> [--skv ...] [--force]
```

1. **性能（硬门，MAD）**：`3 × max(MAD_base, MAD_cand)` 之内一律 INCONCLUSIVE，**不许写成收益**。
   base 必须每轮重测（漂移同时作用在两边）。
2. **精度主口径（硬门）**：生产 logit 尺度下对 **fp32 稠密参考**的 `max_abs ≤ contract 容差`，且 `nonfinite == 0`。
3. **精度压力档（只报警）**：更高 qscale 下判 **相对退化倍数** `cand_rel / base_rel`。
   **绝对容差不能用在压力档上** —— base 自己在 qscale=10 就有 1.55e-2，会把正确改动一起否掉（本 skill 第一次跑就栽在这，`maxn32` 因此被判死，而它在主口径下与 base 精度相同）。

判定输出四档：

| 判定 | 含义 | 下一步 |
|---|---|---|
| `ACCEPTED_BITEXACT` | 显著更快且逐位一致 | 直接落生产并复测 |
| `ACCEPTED_NUMERICS_PENDING_E2E` | 显著更快但非逐位 | **必须**补端到端质量门（GPQA/HumanEval 一档），微基准通过不算数 |
| `INCONCLUSIVE` / `REJECTED_NO_GAIN` | 噪声内 / 更慢 | 按 §九 记一笔（实现坏 / 方向死），别重跑 |
| `REJECTED_ACCURACY` | 精度或 nonfinite 不过 | 同上，并且要记下是哪个 qscale |

> **压力档报警 ≠ 否决。** `maxn32` 在 qscale=10 退化 2.0×，主口径完全一致 —— 这类只有端到端能裁决，不要用微基准替它做决定。

## 九、失败分两类，别混（混了就等于白丢一条路）

一次实验失败，归因只有两种，**命令不同、后果不同**：

| 什么时候用 | 命令 | 方向预算 | 上限 |
|---|---|---|---|
| **实现写坏了**：编译不过 / 布局推断拒绝 / 锚点没落上 / 算错 / 崩。这次没跑对，说明不了这条攻击面不行 | `impl-broken <tag> --why "..."` | **不动** | 连挂 3 次 ⇒ 该实现判死（进 `refuted`，`kind: impl`） |
| **攻击面不成立**：跑通了，但结论是这条路不行，且 `why` 写清了会连带杀掉哪一族 | `direction-refuted <tag> --why "..."` | +1 | 2 次 ⇒ 封方向，`run` 拒绝再受理该方向的任何候选 |
| 结构上直接否掉一条路、手上没有具体实现 | `direction-refuted --direction <id> --why "..."` | +1 | 同上 |

**为什么必须拆。** 把实现的 bug 记成方向证伪，等于用一个实现错误处死一整条路。
本项目 27 条实现级判死里没有一条证伪了攻击面本身（每条都是自己的机制挂掉：溢出 / LDS 超限 /
布局拒绝 / 就是慢），所以 `directions` 里三个方向的 `refutes` 全是 0 —— **不是没试过，是没试死过方向**。

`loop.py` 强制三件事（都不是「希望模型记住」）：

1. `pending` 候选必须声明 `direction`；没声明 / 拼错 / 挂在已封方向上 ⇒ **`run` 直接拒绝**（rc=2），
   因为「2 次封方向」对它是空转，规则等于没有；
2. 已判死的 tag `run` 拒跑；真要重跑必须 `--force`，并在记录里写清为什么这次不一样；
3. `queue` 有任何预算违规就 rc=2 —— P0-4 的 hook 直接拿它当门禁用。

schema（三段；已判死只有 `refuted` 一个落点）：

```yaml
directions:
  - id: raise-heads-per-wave
    statement: "抬 heads/wave 到 16，稀释「每 wave 重读整块 KV」"
    status: open          # open | exhausted（到 2 次证伪由脚本改）
    refutes: 0

candidates:
  - id: bm64_vgpr_squeeze
    direction: raise-heads-per-wave   # pending 必填，否则 run 拒跑
    status: pending                   # pending | adopted | verified（终结态由 verify 裁决）
    priority: 1
    expect: "判据三条同时成立：scratch==0 且 heads/wave==16 且 DRAM≤500 MB"
    done_when: 'python3 -I scripts/opcheck.py correctness --contract "$OPS_CARD/contract.yaml" --shape decode-glm53 --variants hip-glm'
                                      # adopted 必填：完成判定的外部裁决命令（verify 跑它，rc=0 才 adopted）
    fails:                            # impl-broken 追加；连挂 3 次判死
      - "2026-10-28 编译不过：layout 冲突"

refuted:                              # 别重开；why 必须写机制
  - id: merge
    kind: impl            # impl（实现判死）| direction（方向证伪）| claim（历史诊断类）
    direction: fuse-kv-splits
    why: "慢 2.12×，VGPR 256 / spill 30。4 路切分是寄存器生命周期分段：4 个小 buffer 顺序复用同一批 VGPR，合成 1 个大 buffer 后生命周期全程重叠，顶爆 256 上限。"
    on: 2026-10-27
```

> **计数的权威点在 yaml，不在 `runs/`**：`refutes` / `fails` 存在 `candidates.yaml`，`runs/loop/*.jsonl` 只是事件流水——`**/runs/` 在 `.gitignore` 里，计数若从流水推导，删一次 `runs/` 就等于把方向预算清零、把死马全部复活。

**`why` 必须写机制，不能只写结论。**「merge 慢 2.12×」只挡住 merge 这一个变体；「合并 buffer ⇒ 生命周期重叠 ⇒ 溢出」能挡住后面所有同类想法（`kv4`、`g2s` 都是同一个死法，提前识破能省三轮）。老命令 `refute` 已拆掉（只剩 rc=2 迁移提示），它原本把「实现写坏」和「方向不成立」记成同一件事，正是本节要消掉的歧义。

## 十、交付件格式

一轮优化收口时，产物**必须**是这四件（缺一件就当没做完）：

1. **结论**：相对基线/对手的倍数，带口径字符串；
2. **前提**：什么形状、什么卡、什么 DTK、哪些变体前置条件（如 `-1` 只能垫队尾）；
3. **已证伪假设**：带机制的清单（即 `candidates.yaml:refuted`）；
4. **复现脚本**：能一条命令重跑出结论的脚本，路径写进 skill 树而不是 `/tmp`。

## 十一、一条命令看懂现状

```bash
python3 loop.py -C <卡> status|queue|report   # 现状 / 方向预算 + 待做 + 已判死附机制（违规 rc=2） / 交付表
python3 loop.py -C <卡> impl-broken <tag> --why "..."       # 实现写坏了
python3 loop.py -C <卡> direction-refuted <tag> --why "..." # 攻击面不成立
python3 loop.py -C <卡> verify --tag <id>                   # 跑 done_when；rc=0 才 adopted（§七）
```


参考实现：`op-sparse-mla`（32 条判死 + 3 条待做 + 3 个方向，`candidates.yaml` 可直接抄 schema）。
