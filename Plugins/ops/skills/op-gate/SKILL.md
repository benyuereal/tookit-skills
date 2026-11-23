---
name: op-gate
description: "算子改动的判定门：五道门（执行身份 / 正确性 / 不变量 / 鲁棒性 / 端到端 A-B）+ 四档决策（ACCEPTED / REJECTED / INCONCLUSIVE / DIAGNOSTIC-ONLY）+ 回退与留痕。与 scripts/opverify.py 输出对齐，是 dcu-orchestrator 的 op-validate 节点依据。触发场景：改完 kernel 要判「能不能进生产」、A/B 结果有噪声拿不定、微基准快但端到端没收益该算哪种结论、精度对比过了还要不要跑不变量、要把结论写进台账或回退补丁。不适用：候选排序与中断后续跑走 op-optimize；PMC 瓶颈定位走 op-profiling。"
version: 1.0.1
author: DCU Toolkit Contributors
license: MIT
kind: tool
---

# 算子改动的判定门（四闸四档）

## 一、四道门，顺序不可换

```text
gate0 执行身份 → gate1 正确性 → gate2 不变量 → gate3 A-B 性能
gate0/1/2 任一 fail ⇒ REJECTED（后面不用跑；参考对比过了也不行）；gate3 噪声内 ⇒ INCONCLUSIVE；显著 ⇒ ACCEPTED
```

### gate0 执行身份（这些数字是这份代码跑出来的吗）

最便宜也最常被跳过的一道。**不测性能，只验「数字配代码」**：

- 源码 / 编译产物 / 进程实际加载的产物三者 sha256 与采集时一致；源码不比产物新（否则是改完没重建，测的是旧产物）
- 环境指纹（gfx / dtk / torch / python / 频率档 `sclk_mhz` / 卡）非空且不是占位符

只有契约声明了 `manifest.required_roles` 才出这道闸；一旦声明，缺 manifest 即 `REJECTED`，「这次先跳过」不是选项。写法见 `op-contract/SKILL.md §六`，工具是 `scripts/opmanifest.py`。

> 为什么排第一：gate1/2/3 全过也可能毫无意义——219.9µs 若是缓存里另一个 `.so` 跑出来的就白测；踩过的坑是微基准 bit-identical、生产掉分。2026-10-28 上机实测：sparse-mla `decode-glm53` 真 run 的 `identity` 闸 `pass`（`loaded` = `~/.tilelang/cache/…/executable.so`，`dtk=6.3.26113`，`sclk_mhz=1150`），同 run 的 correctness / invariants / perf 三闸 FAIL（骨架 run 无真数据）——**身份过了不等于结论成立**。

### gate1 正确性（对 fp32 参考）
- 参考实现用 fp32/long-double 累加，**不用另一个 GPU 实现当参考**（否则两个实现一起错就抓不到）。
- 大 shape 必须**按 query/行分块**算参考（一次性构造 `[S, SKV]` 分数矩阵会 OOM，S=2048/SKV=65536 需要 8TB）。
- 容差来自 `contract.yaml` 的 `tolerance.vs_fp32_ref`，不在代码里另写一份；输出报告 `{max_abs, mean_abs, cosine, n_compared}`，缺项按失败处理。

### gate2 不变量（结构正确性）
与参考对比抓不到的 bug，全靠这一闸：

| 不变量 | 怎么测 | 已知案例 |
|---|---|---|
| `determinism` | 同输入连跑 5 次，逐位比较 | M≥6 decode 非确定（state-copy 索引错 block） |
| `batch_independence` | 同一样本单独跑 vs 混在大 batch 里跑，比输出 | ASM MoE 内核串读 batch 同伴 |
| `padding_invariance` | 打乱/移动 `-1` 填充位，比有效位输出 | topk `-1` 污染 |
| `index_permutation_invariance` | 打乱 gather 索引顺序，比输出 | 稀疏 attention indices |
| `split_invariance` | 不同 SplitKV / 不同 tile 配置，必须 bit-exact | fused mqa 换 SplitKV 必须逐位一致 |

**判定口径**：不变量一律要求 **bit-exact**（同设备同实现），不是「误差在容差内」。做不到就在契约里写明理由。

> 判据提醒：MoE / 稀疏算子的无损性优先看**自洽性**（同 token 输出不依赖 batch 同伴），不要依赖自己写的 torch 参考——参考写错会同时放过 bug。

### 调用任何被测实现之前：输入与输出都要隔离

这两条不属于任何一道门，却能让四道门**全部给出假 PASS**，所以放在最前面：

- **输出先毒化**：wrapper 常用 `torch.empty` 建输出，kernel 没写到的位置会保留上一次的旧值——漏写看起来像「对了」。测试前把同尺寸显存填 NaN，跑完出现 NaN 就说明那块没写。
- **输入给副本**：**实现可能原地改写自己的输入**。实测 `flash_mla.flash_mla_sparse_fwd` 会把 `indices` 规范化后写回（有效索引前移、`-1` 后移，幂等），多个变体共用一份输入时，先跑的那个会改掉后跑那个的输入——我们因此把一个「前置 `-1` 输出恒为 0」的失败看成「两种布局都一致」。
- 顺序：基准与被测实现各拿独立副本；判完顺手断言输入未被改写（改写本身是该实现的事实，记进 `variants.yaml` 的 `mutates_inputs`）。

### gate3 端到端 A-B
- **同进程交替执行**（interleaved），不是「先跑完基线再跑优化版」——后者会把频率/温度漂移算成收益；至少 3 轮，每轮取中位数，用 `opbench.significant(k=3)` 判。
- 大 shape 的真实收益常常小于微基准：要补一个端到端口径（step 级 / 服务级），否则只能算 DIAGNOSTIC-ONLY；精度不能回退，端到端要带质量门（如固定评测集分数、HumanEval/GPQA 抽样）。

### gate2b 鲁棒性：未见过的 shape + stress 数值

**只在提交的形状上通过 ≠ 可靠。** 这是
CompoundingKernel（arXiv:2611.19230）的裁决里我们唯一缺的一类：
它的 L2/079 通过了所有 shape，但 FP16 stress 输入溢出，
他们宁可从 1.62× 退到 0.48× 也不交。

两类检查（`scripts/opcheck.py robustness`）：

| 类 | 内容 | 为什么 |
|---|---|---|
| A 未见过的 shape | 与 `shapes.yaml` 声明**全部不同**的极端形状（非 2 的幂、跨 tile 边界、topk=1 退化） | 只在提交形状上过 = 考试题和练习题一样 |
| B stress 数值 | 同一 shape 灌 `huge / tiny / mixed_sign / denormal / zeros` | 溢出、下溢、对消误差只在极端值下暴露 |

**判定口径（三条，缺一会让门失效）**：
- 只有「参考正常、被测异常」才算**抓到**；
- **参考自己也溢出 → SKIP**，不算抓到（那什么都没证明）；
- **变体跑不起来 → ERROR**，不算抓到（否则缺后端会被记成「检查有效」——
  我们第一版就是这样抓到 12 例假阳性，真信号被淹掉）。

> **门本身要被验证**：`--selftest-flaw` 注入一个「平时正确、极端处错」的变体。
> 如果连它都抓不到，说明门没生效。CI 里跑这一条并要求 `--require-catch`。

真实收获示例：`fused-add-rmsnorm` 的 A 类检查探出
`lightop.RMSNormAdd` 要求归一化维度 ≥ 64（H=4/3/5 直接 `cols >= 64 INTERNAL ASSERT FAILED`），
已记进该卡 `hard_preconditions`。

## 二、四档决策

| 决策 | 条件 | 允许的动作 |
|---|---|---|
| `ACCEPTED` | 四闸全过（契约未声明 `manifest` 时 gate0 不出现） | 合入 / 进 registry `tuned`；写台账 |
| `REJECTED` | 任一闸 fail | 回退改动；记录失败原因（这是最有价值的沉淀） |
| `INCONCLUSIVE` | 正确性过、性能在噪声内 | 保留分支不合并；补跑到能判定，或标 `DIAGNOSTIC-ONLY` |
| `DIAGNOSTIC-ONLY` | 微基准快但端到端无收益 / 只在非生产布局下快 | 只作诊断证据，不宣称收益 |

这套标签与 `dcu-perf/dcu-op-optimization` 的「四档决策」、`humanize-rlcr` 的决策标签同源，跨 skill 通用。

## 三、融合类改动的额外门

融合默认**净负**，要同时证明两件事：①收益不被搬移吃回去（省下的时间不能变成新的全局同步/原子操作/额外读写）；②微基准与 step 级都有真收益。已证伪的样本（防重开，机制细节见 `references/cards.md#OP-GATE-004`）：

| 方案 | 结果 | 根因 |
|---|---|---|
| MoE GEMM2 epilogue 融 `moe_sum`（atomicAdd） | 慢 2.5–3× | 全局 RMW + 冲突 |
| MoE GEMM1 epilogue 融 silu-mul（g1f） | 结构不可行 | gate 列与 up 列跨 block，纯 epilogue 融不到 |
| `moe_sum` triton 替代 | 慢 8× | 无 epilogue 融合收益，多一次读写 |
| MoE epilogue 融合（g2f） | 慢 12× | 同上 |

## 四、回退与留痕
- **改动可回退**：补丁单独文件、可 `--reverse`；生产副本改动先备份 `<file>.bak-<reason>-<date>`。回退后写一行台账：`回退原因 / 观测到的现象 / 复现命令`——不回退记录等于白跑。
- 非确定性问题的判据固定为 **`M=1` 干净、`M>1` 脏**（先排除 per-query 无关的 kernel 差异，再怀疑 block table / slot_mapping）。

## 五、和工具怎么接
```bash
python3 scripts/opverify.py --contract <card>/contract.yaml --run runs/<id> --json
python3 scripts/opcheck.py correctness --contract <card>/contract.yaml --shape <shape> --variants a,b --json runs/<id>/correctness.json
python3 scripts/opcheck.py invariants  --contract <card>/contract.yaml --shape <shape> --variant <v>
```

- 输出 `{"verdict", "gates", "reasons", "missing"}`：`missing` 非空 → **一律 `REJECTED`**（缺项按失败处理）；`gates.identity` 只在契约声明 `manifest.required_roles` 时出现，`skipped` 只能来自显式 `--allow-missing manifest` 且 `reasons` 必有留痕。
- `dcu-orchestrator` 的 `op-validate` 直接消费这份 JSON，`op-distill` 据此写 `kb/op-opt/` 台账；判定不依赖框架特性（Codex/Cursor/Gemini 无 hooks 同样能跑）——跨框架一致性的落点。
- correctness / invariants 的采集流程由共享脚本提供、判据读卡片 YAML，**不要再给每张卡抄测试脚本**。执行身份用 `scripts/opmanifest.py record` 采集（写法见 `op-contract/SKILL.md §六`）。

### 闸门什么时候自动跑（唯一一个 hook）

`Plugins/ops/hooks/ops_gate.py` 是**唯一**的 PreToolUse hook（接线 `.claude/settings.json`），只挂四个动作，**不挂就不查**。`.claude/settings.json` 的 matcher 是 `Bash|Write|Edit`，每个动作进 hook 后先过一遍毫秒级的路径/命令字符串预筛（`_worth_checking`），不值得查就立刻放行 —— 普通 Bash 与普通写文件不会触发任何文件读取或全仓扫描：

| 动作 | 触发 | 查什么 |
|---|---|---|
| `write_source` | 写 kernel 源码（manifest 声明的 source / `.hip .cu .cuh` / `perf/*.py`） | A 契约；（身份链已断时）B |
| `write_verified` | 写 `SKILL.md` / `references/*.md` 且**新增或改写**了 verified 标记 | F 提案（普通文档编辑放行） |
| `run_bench` | 卡片 `bench/` 下脚本 / `loop.py … run` / `scripts/opbench.py` | A + B |
| `git_commit` | `git commit` | A + B + C 方向预算（`loop.py queue`）+ D 交付件字段 |

（`write_artifact` 是 `write_source` 之外的一个分支：目标是 manifest 里 role=build/loaded 的产物时直接拦，查 E。）

- **门禁不重新实现判定**：A 调 `opverify.check_contract`，B 调 `opmanifest.validate_manifest`，C 是 `loop.py queue` 的退出码；检查函数放在 `scripts/opgate_checks.py`，hook 只管「什么时候问、问到了怎么拦」。
- **失败关闭**：门禁自身抛异常、脚本缺失、manifest 读不出、`result.json` 不是 JSON ⇒ 一律 `exit 2` 并说清坏在哪。**门禁自身出错绝不 `sys.exit(0)`**——那条路等于门禁一坏就全放行。
- **编辑「已声明的源码」放行身份闸**：改源码本身就是制造漂移，拿身份闸拦自己自相矛盾；漂移由下一次 `run_bench` / `git_commit` 拦。
- **认领靠内容不靠路径**：`manifest.json` / `result.json` 归哪张卡看文件里的 `op` 字段，不看 run 目录名（目录名是人起的，`2026-10-28-p04` 里没有算子名）——实测认错会让门禁把邻居的 manifest 当自己那份，于是**全程放行**，输出还写着「✓ 身份通过」。
- **没有环境变量开关**：要停就改 `.claude/settings.json`——显式、看得见、进得了 diff；`OPS_GATE=0` 之类的后门会把「门禁坏了」和「门禁不存在」变成同一个状态。留痕写在 `runs/ops-gate.jsonl`（best-effort，不参与任何判定）。

### 独立的一道门（F）：写 verified 资产先有提案
> **bootstrap 说明**：本节首次写入时本机制尚不存在，当时无提案（那份自批的提案已删除——模型不能替用户声明批准）。**此后修改本节，同目录要有 `<目标文件名>.proposal.json`。**


`verified` 是**资产**，不是一句形容词。2026-11-13 出过一次事故：一条写错的硬件常数
（`VF_LIMIT=65536`，768-VGPR 模式该是 `196608`）让整条 `status: verified` 的结论方向反了。
所以把「落盘」这一步做进 hook —— 参照 `ProposeSkills`：只产出 review card，**用户保存才落盘**，
且带 `evidence[]`（能回到现场的路径）。

- **触发面很窄**：只认 `SKILL.md` 与 `references/*.md`，且这次写**新增或改写**了
  `status: verified` / 含「已验证 / verified」的行。
- **普通文档编辑一律放行**（实测 3 条反例见下）：改标题、改错别字、改不含标记的普通段落、
  整篇重写但原样保留标记行 —— 判定粒度是**标记行**，不是整个文件。
- **要什么**：同目录 `<目标文件名>.proposal.json`

  ```json
  {"target": "cards.md", "evidence": ["<可回溯路径或命令>"], "approved_by": "<人名或标记>"}
  ```

  `target` 等于目标文件名；`evidence` 是非空字符串数组（路径或可复跑命令）；
  `approved_by` 非空。
- **缺什么拦什么**：缺文件 / 缺字段 / `target` 不符 / `evidence` 为空 / `approved_by` 为空 / 提案不是
  JSON ⇒ `exit 2`，消息里给出提案路径与格式。
- **失败关闭**：读不出这次要写什么（例如 `Write` 缺 `content`）⇒ `exit 2`，不猜。
- **删标记不拦**：删掉一条 verified 断言不会凭空造出结论（代价是证据被删，属于另一类问题）。

**怎么申请豁免**：没有环境变量开关，也没有「跳过本次」参数。

| 层级 | 怎么做 |
|---|---|
| 单文件 | 给那个文件写一份带 `evidence` 的 `.proposal.json`。**它就是审批记录，应当进 diff 一起 review** |
| 整条门 | 改 `.claude/settings.json` 的 `PreToolUse`：删掉这个 matcher，或把 `command` 指向你自己的脚本。显式、看得见、进得了 diff |
| 不要怎么做 | `OPS_GATE=0` / `--no-gate` 之类的后门。它会把「门禁坏了」和「门禁不存在」变成同一个状态（本文件 §五 的同一条理由） |

反例（这三类编辑不拦，`Plugins/ops/tests/test_ops_gate.py:TestVerifiedProposal` 里有对应用例）：

1. 把 `# 知识卡` 改成 `# 知识卡（2026-11-13 修订）`；
2. 改正文错别字或补一句解释（不含任何标记词）；
3. 标记行原样保留、只重写它旁边的普通段落。

## 六、常见误判

| 误判 | 表现 | 纠正 |
|---|---|---|
| 拿噪声当收益 | 「快了 3%」但 MAD 是 5% | 跑 `significant()`，过不了就 INCONCLUSIVE |
| 只测微基准 | 微基准 +40%，端到端 0 | 补 step 级；否则 DIAGNOSTIC-ONLY |
| 参考对比过就放心 | 换 batch 大小结果变了 | gate2 必跑 |
| 融合一定更快 | 微基准局部更快 | 默认净负，先证两条 |
| 用不同布局/频率的数比较 | 结论反转 | 同卡同布局同进程交替 |
| 多变体共用一个输入张量 | 先跑的变体改写了输入，后跑的看到被改过的数据，结论完全反转（平局/失败被看成一致） | 每个变体喂独立副本（`_common.run_isolated`），并记录 `mutates_inputs` |
| 用 `torch.empty` 接输出就判对 | kernel 漏写的位置留着上次的旧值，看起来「数值合理」 | 先把同尺寸显存毒化成 NaN，出现 NaN 即未写 |
| 没确认卡上有没有别人的作业 | 两个实现同时慢 ~17%，**比值还被压平**（0.79 → 1.00，看起来「没有差距」）；偶发一次输出对不上基准 | 跑前先看 `hy-smi`，钉一块 HCU% 为 0 的卡（`HIP_VISIBLE_DEVICES=<n>`）；跨实现只认同进程同卡比值。基准实现自己也可能抖，先确认被测实现逐位可重复再怀疑它 |
| 用「MB 级张量」当缓存常驻 | 4.6MB 表当初「进 L2」，得出「带宽不是瓶颈」的错误结论 | 缓存尺度按设备实际容量取（KB 级），并做 3~4 档工作集扫描；只有耗时随工作集出现台阶才算命中缓存 |

## 七、证据要求（通用）

卡片「### 证据要求」只写**本卡特有**的要点，通用四条如下（缺一条即按 `missing` 处理 → `REJECTED`）：

- **E1 口径**：卡号、是否锁频/时钟档位、DTK、dtype、访问模式、日期齐全；每个分母带 `source + verified_on + status`。缺分母来源本身就是不合格。
- **E2 对比**：A/B 同卡同进程交替、≥3 轮、median + MAD，`|Δ| > 3×MAD` 才算显著；「先跑基线再跑候选」与跨卡绝对数相减直接判不合格。
- **E3 端到端**：微基准之外要有 step 级/服务级口径，并带质量门（固定评测集/抽样）不回退；只有微基准的收益最多 `DIAGNOSTIC-ONLY`。
- **E4 可复现**：采集命令 + 产物路径（`bench.json` / `verdict.json` / `candidates.yaml:refuted`）+ 结论与产物字段一一对应；机制没落盘不算证据。

## 知识卡索引

本 skill 踩过的坑已沉淀为 `references/cards.md`（六段固定：知识点 / 错误写法 / 正确写法 / 识别方法 / 证据要求 / 参考）；卡片只放物理常数与相对倍数，绝对性能数字在 `~/.config/dcu-toolkit/kb/op-opt/`。

| id | 一句话结论 | 卡片 |
|---|---|---|
| `OP-GATE-001` | 「当前结构到顶」≠「物理到顶」：两级天花板同时满足才准收工 | [OP-GATE-001](references/cards.md#OP-GATE-001) |
| `OP-GATE-002` | 微基准输入布局必须与生产 kernel 输出布局一致 | [OP-GATE-002](references/cards.md#OP-GATE-002) |
| `OP-GATE-003` | 百分位阈值假象：P99 越界可能只是慢步占比跨过割线 | [OP-GATE-003](references/cards.md#OP-GATE-003) |
| `OP-GATE-004` | 融合 epilogue 前先算全局 RMW 冲突 | [OP-GATE-004](references/cards.md#OP-GATE-004) |
| `OP-CALIBER-001` | 结论到顶前先锁口径（标称峰值 vs 实测峰值） | [OP-CALIBER-001](references/cards.md#OP-CALIBER-001) |
| `OP-CALIBER-002` | 跨卡/跨机只比相对比值，不比绝对耗时 | [OP-CALIBER-002](references/cards.md#OP-CALIBER-002) |
