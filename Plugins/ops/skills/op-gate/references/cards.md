# op-gate 知识卡（6 张）

门禁类：判「这个改动算不算通过」先查这里踩过的坑。每卡六段固定（知识点 / 错误写法 / 正确写法 / 识别方法 / 证据要求 / 参考），**只有「参考」段允许绝对耗时/带宽**，绝对数字归 `~/.config/dcu-toolkit/kb/op-opt/`。`### 证据要求` 只写本卡特有点，卡号/档位/交替轮次/median+MAD/端到端等通用要求见 `op-gate/SKILL.md §七`；id 与 SKILL.md「## 知识卡索引」一一对应，由 `Plugins/ops/tests/test_card_index.py` 校验。`applies_to: gfx938` = 本机实测，换设备要重测。

## OP-CALIBER-001 · 结论到顶前先锁口径（标称峰值 vs 实测峰值）
```yaml
id: OP-CALIBER-001
name: 结论到顶前先锁口径（标称峰值 vs 实测峰值）
category: caliber
applies_to: gfx938
status: verified
```
### 知识点
同一份实测数据换分母，结论**相反**：型号标称峰值分母约 87%（看似到顶），本机流式只读实测分母约 55%（还有约 1.8 倍）。「到顶」= 实测值 / 一个明确锁定的分母。dtype 与访问模式同样要锁：本机 fp8 GEMM 可达峰值约为 bf16 反推峰值的 2.9 倍；纯读流式峰值与 copy(R+W) 峰值差约 1.3 倍。
**已证伪（不要重开）**：「本机标称带宽就是天花板」——本机流式只读实测比型号标称高约 1.5 倍，标称只是型号名；「某算子 87% 峰值」当定论引用——按实测分母重述后约 55%，方向翻转。
**反转条件**：换测量模式（只读/只写/读写混合）、换卡、换时钟档位、换 dtype 都要重锁分母。判断「分母没锁」不需要知道正确值——报告里没写分母来源本身就是命中。
### 错误写法
反例：报告写「达峰 87%，已接近物理上限」但分母来源缺失（实际用型号标称，unverified）；`contract.yaml` 的 `floor.peak` 直接填标称值；fp8 kernel 拿 bf16 峰值口径算出「达峰 259%」——荒谬值本身就是口径混了的反证。
### 正确写法
分母 = 同 SoC、同访问模式、同 dtype 的实测可达峰值，带 `source + verified_on + status`，且同时出现在契约 `floor` 与报告口径行（两处一致）。报告写「达成率 55%（分母=本机流式只读实测峰值，卡 6，2026-10-27；来源 constants.yaml）」。两个来源冲突**并列记录**，不取平均、不挑好看的；口径没锁之前禁止出现「到顶」「物理上限」「没有空间」。
### 识别方法
1. 百分比没同时给分母来源与测量日期，或分母 `status` 是 `unverified` / `derived` ⇒ 命中。
2. 访问模式与分母不同模式、fp8 的 FLOPs 配 bf16 峰值、或同一数据两个候选分母结论相反 ⇒ 命中（先锁口径，不许二选一）。
### 证据要求
- 两个分母下同一份实测数据的达成率与结论翻转（「为什么必须先锁」的证据）；分母的 `constants.yaml` 条目、测量命令、卡号、档位、访问模式、日期；契约 floor 与报告口径行两处一致；引用标称值须并给实测值并标 `unverified`。
### 参考
- Plugins/ops/skills/op-roofline/constants.yaml（conflicts: peak-bw-definition、fp8_gemm_peak_tf）与 Plugins/ops/skills/op-roofline/SKILL.md（硬规则 2「一个数只能有一个分母」）
- Plugins/ops/skills/op-profiling/SKILL.md（§六 参考实现不内置峰值常数、不吐 MBU；分母必须在报告里显式锁）与 Plugins/ops/skills/op-gate/SKILL.md（§七 通用证据要求、§六 常见误判）

## OP-CALIBER-002 · 跨卡/跨机只比相对比值，不比绝对耗时（时钟档位可差 1.5 倍以上）
```yaml
id: OP-CALIBER-002
name: 跨卡/跨机只比相对比值，不比绝对耗时（时钟档位可差 1.5 倍以上）
category: caliber
applies_to: 机制通用（多卡/多机对比）；gfx938 上有实测
status: verified
```
### 知识点
不同卡/机的时钟档位可差 **1.5 倍以上**（实测过一边锁 600MHz、一边跑 1150MHz）。此时把两边绝对耗时相减或相除，得到的「收益」全是错的——那 1.5× 会被写成优化成果。绝对耗时只在**同机、同档位、同进程、同布局**内可比；跨卡只能比 ratio：每个变体在同一张卡同一进程**交替执行**，每轮**现测基线**。
**已证伪（不要重开）**：「A 卡基线 1.2、B 卡优化版 0.8 ⇒ 优化了 33%」——两侧档位不同，比值被频率差污染；「先跑完基线再跑优化版」——频率/温度漂移被算成收益。
**反转条件**：两卡已锁同一档位、同 DTK、同布局，且同形基线重复测量误差 <5% 时，绝对数只能作 sanity check；正式结论仍以同进程 ratio 为准。换卡、换驱动、换锁频策略都要重做。
### 错误写法
反例：`gain = ms_A_card - ms_B_card`（两侧档位不同，差值无意义）；同卡但分时段先跑完基线再跑候选（频率/温度漂移被算成收益）；别人报告里的绝对耗时直接进对比表。
### 正确写法
每轮 `base_ms = time(base_fn); cand_ms = time(cand_fn)`（同卡同档位），`ratio = median(cand_ms) / median(base_ms)`，显著 = `|Δ| > 3 × max(MAD_base, MAD_cand)`。报告写四元组 `(算子, 形状档位, gfx, dtype)` + 卡号/是否锁频/进程边界；跨卡数据单标「不可比」，只用来看方向；基线每轮重测，与历史差 >5% 先查环境（频率、空闲卡）。
### 识别方法
1. base 与 cand 的卡号/档位不一致，或 base 只测一次/引用历史值 ⇒ 命中。
2. 百分比来自跨机绝对数相减、别人报告的绝对耗时直接进表、或 ratio 跨进程/跨卡 ⇒ 命中。
### 证据要求
- 每轮 base/cand 值与 median、MAD、ratio、显著性判定；卡号、档位、进程边界、交替顺序与轮数；跨卡数据单列并标「不可比」；基线漂移检查（本轮 vs 历史相对差）。
### 参考
- Plugins/ops/skills/op-bench/SKILL.md（铁律 2：跨卡只比 ratio，不比绝对耗时）
- Plugins/ops/skills/op-optimize/SKILL.md（② 复现基线：基线必须现测）与 Plugins/ops/skills/op-profiling/SKILL.md（计时与 PMC 两套口径，别混）；scripts/opbench.py（`significant()`：同卡同进程的 ratio 判据）

## OP-GATE-001 · 「当前结构到顶」≠「物理到顶」：两级天花板同时满足才准收工
```yaml
id: OP-GATE-001
name: 「当前结构到顶」≠「物理到顶」：两级天花板同时满足才准收工
category: gate
applies_to: gfx938
status: verified
```
### 知识点
收工声明（「到顶」「没有空间」「物理不可达」）必须同时满足两级，缺一级只能降级成「当前结构到顶」：**一级门（无合法减量）**——算法/访存/指令三条路径的候选全部列出，逐条被实测证伪或被模型在编译前否掉；「更大的改动还没试」「想不出新办法」都不是一级门。**二级门（实测贴地板）**——`实测 / 理论地板 ≤ 1.5`；地板由模型算出且已与实测对账（残差 <30%），分母是实测可达峰值（见 `OP-CALIBER-001`）。两级全满足才是「物理到顶」；只满足二级 = 一级门没做，只满足一级 = 模型/实现还没到位。**别把本卡的 ≤1.5× 和卡状态 `at_floor` 的 ≥85% 混用**：前者是收工声明阈值，后者是生命周期标记（权威定义见 `Plugins/ops/SKILL.md` §三 状态机）。
**已证伪（不要重开）**：「候选跑完一轮 ⇒ 到顶」——27 条实现级判死里至少 6 条（merge / kv4 / g2s / bm64 / thr128 / bi64）是编译前就能算掉的，证明「没想法」≠「没方案」；「外部说还有 45% ⇒ 一定有 45%」——MoE Marlin GEMM 曾被判「还能省 45%」，复核发现已贴带宽下限（隔离实测接近峰值带宽、随激活 expert 数线性），那 45% 不存在。
**反转条件**：地板改用标称峰值 ⇒ 二级门作废重算；出现新的合法减量（换算法/指令/布局）⇒ 一级门重开；模型残差 >30% ⇒ 先修模型，不判门。
### 错误写法
反例：`for cand in queue: run(cand)` 后直接 `print("到顶了")`（队列本身不全，既没有二级门比值也没有一级门证伪清单）；用 `bound=[]; facts=["CTA1-PER-CU"]` 这类描述性事实宣布占用率受限 ⇒ 到顶。
### 正确写法
一级门列出合法减量清单，每条结局 = ACCEPTED / REJECTED(机制) / 编译前否掉(规则号)；二级门给 `实测 / 地板 = 1.x`（地板取 `constants.yaml:measured`，附残差）+ 反转条件（换形状/并发/峰值口径/硬件时哪一级先失效）。二级门没过时措辞固定为「当前结构到顶」，并跟「物理上还差多少、卡在哪一项」；被模型否掉的候选也进清单（带规则号），与实测证伪等价；证伪清单留档 `candidates.yaml:refuted`（`loop.py run` 靠它拒跑），**不删**，机制写清。
### 识别方法
1. 缺任一级门的证据、只有「试过了」「没想法」、或一级门清单与候选队列都空 ⇒ 命中。
2. 二级门比值 >1.5、地板 `status≠measured`、残差 ≥30%、`bound[]` 空、`missing_evidence[]` 非空 ⇒ 命中。
3. 说「物理不可达」却没指出「哪一项物理量不能再减」⇒ 命中。
### 证据要求
- 一级门：候选清单（含编译前被否掉的 + 规则号）与每条结局/机制；二级门：地板推导（模型路径 + 常数条目）、实测值、比值、模型残差；反转条件。
- 声称「物理不可达」须给出被压死的那一项（带宽/算力/LDS/寄存器/发射槽）与实测达成率。
### 参考
- Plugins/ops/skills/op-gate/SKILL.md（§一 四道门、§二 四档决策、§三 融合额外门）与 Plugins/ops/skills/op-roofline/derive.py（地板与绑定约束的算法来源）
- Plugins/ops/skills/op-roofline/constants.yaml（measured 峰值与 conflicts）与 Plugins/ops/skills/op-sparse-mla/candidates.yaml（directions / candidates / refuted 三段）
- Plugins/ops/skills/op-sparse-mla/ledger.md（§二 结构结论、§三 已证伪）

## OP-GATE-002 · 微基准输入布局必须与生产 kernel 输出布局一致
```yaml
id: OP-GATE-002
name: 微基准输入布局必须与生产 kernel 输出布局一致
category: gate
applies_to: gfx938
status: verified
```
### 知识点
微基准的「bit-identical」只说明**在该布局下**两个实现等价。布局一旦与生产 kernel 的输出布局不同，它就是假象——证明的是一个生产上不存在的输入分布。本项目踩过三类：**① 受控 -inf 布局假象**——生产 kernel 输出的 logits 有限区是 token 全长，优化版裁剪的是这个有效区；微基准却自造「尾部整段死区」布局，于是完整版与裁剪版逐位相同，生产分数却掉。**② 形状错配**——用小一个数量级的 M 代生产形状外推分块收益，同一 session 得到三个互相矛盾的结论；换到生产形状重测后前两个全部作废。形状是布局的一部分（块数、每块列宽、扇出都随它变）。**③ 并发错配**——per-query 微基准逐位一致，并不覆盖 M>1 的 block table / slot mapping 竞态。判据固定为 **M=1 干净、M>1 脏**。
**已证伪（不要重开）**：「微基准 bit-identical ⇒ 生产安全」——只证明该布局、该并发度下等价。
**反转条件**：微基准直接吃生产 kernel 的输出张量与几何（同 stride / 同 seq_lens / 同块表）并对几何做签名断言，该风险消除；覆盖到生产并发档后，per-query 结论才可升级。
### 错误写法
反例：`x = build_inputs(shape)` 自造布局、生产几何从未进断言；`assert torch.equal(full_kernel(x), cut_kernel(x))` 只在「尾部整段 -inf」布局成立；`assert torch.equal(per_query(a), per_query(b))` 就宣布「并发下也一致」（未测 M>1，越权结论）。
### 正确写法
取 `geo = prod_wrapper.geometry()`（stride / seq_lens / block_table / 有效区长度），先 `assert geo.signature() == PROD_SIG` 再比数值，各变体用独立副本；并发档单独跑 M=1 与 M=生产并发两档。几何签名含「有限区长度 / 尾部 -inf 宽度 / 有效索引位置」，任一项与生产不符立即失败；形状扫描取生产档位（prefill/decode 各自最小与最大），不拿中间档外推；只覆盖 per-query 时结论必须显式标注「不覆盖并发」。
### 识别方法
1. 输入/输出 tensor 由 harness 重建 shape/stride/seq_lens，或没有断言生产几何签名 ⇒ 命中候选。
2. -inf 填充宽度、有效索引位置与生产输出布局不一致、或覆盖矩阵里没有 M>1 ⇒ 命中（后者结论降级）。
3. 「逐位一致」但生产指标/分数有变化 ⇒ 优先怀疑布局，而不是怀疑测量。
### 证据要求
- 生产几何的来源（wrapper 里实际传入的 stride / seq_lens / 块表构造代码位置）；微基准几何与生产几何的逐项对比表与签名断言。
- bit 对比覆盖矩阵（形状 × 并发度 × 索引分布，含 -1 填充位置）；只测 per-query 时写明「不覆盖并发」及并发档判据（M=1 / M>1）。
### 参考
- Plugins/ops/skills/op-gate/SKILL.md（§一 gate2 不变量、split_invariance；§六 常见误判）与 Plugins/ops/skills/op-sparse-mla/contract.yaml（layout_constraints、`-1` 必须后缀的约束）
- Plugins/ops/skills/op-sparse-mla/bench/bench.py（`--mode workset`：消融前必须复现生产输入构造）与 Plugins/ops/skills/op-sparse-mla/_common.py（`run_isolated`：每个变体独立输入副本）

## OP-GATE-003 · 百分位阈值假象：P99 越界可能只是慢步占比跨过割线
```yaml
id: OP-GATE-003
name: 百分位阈值假象：P99 越界可能只是慢步占比跨过割线
category: gate
applies_to: gfx938
status: verified
```
### 知识点
P99 是「最慢 1% 的边界」：由慢步的**占比**与**幅度**共同决定，不随快步改善而下降。每轮存在固定数量慢步时，降低快步耗时会**抬高**慢步占比；占比一旦跨过 1%，P99 立刻从「快步那侧」跳到「慢步那侧」——数值大幅跳变，但慢步本身一点没变慢。本项目实证（c32 并发 decode，生产卡）：每轮恒定 48 个慢步（每个是一次大 chunk prefill 阻塞），优化把快步显著压低后，慢步占比从约 0.88% 升到约 1.10%，恰好越过 1% 割线 ⇒ P99 报出「回归」；同期 P50 / P90 一路改善，慢步计数与单步耗时都没变。
**控制实验（用来证伪）**：只改输出长度（512 → 1024），慢步数不变、总步数变约 2 倍，慢步占比回落到割线以下 ⇒ P99 掉到基线的约 1/6.6，而每步耗时分布完全没动。⇒ 原「P99 回归」是分位阈值假象，不是性能回归。
**已证伪（不要重开）**：「P99 涨了 ⇒ 性能回归」；「P99 掉了 ⇒ 优化生效」——两者都必须先看占比。
**反转条件**：慢步**数量本身**增加、或单个慢步**变慢**，则 P99 上升是真回归；分位阈值换成 P99.9 / P95 要重新算割线；把慢步（prefill 阻塞）从测量窗口剔除后，P99 才有常规含义。
### 错误写法
反例：`before: P99=X; after: P99=1.x*X; verdict="回归"`（慢步占比可能只是跨过 1% 割线）；用 `run(out=512).P99` 与 `run(out=1024).P99` 直接比（总步数不同，分位点不可比）。
### 正确写法
判据三件套（缺一不许下结论）：① 直方图——每步耗时按慢/快两档计数，给出慢步数 / 总步数占比；② 同口径——两轮的输出长度、并发、prefill 策略一致；③ 控制实验——只改总步数（输出长度），若 P99 回到基线而慢步不变 ⇒ 判假象。P99 必须与 P50/P90 并列；只有 P99 动、P50/P90 不动时先算慢步占比；慢步是周期性阻塞时同时报「慢步单次耗时」与「每轮慢步数」。交付口径优先报 ITL 分布 + 吞吐，P99 只作附注并注明割线与慢步占比。
### 识别方法
1. P99 跳变时 P50/P90 同时改善或持平、慢步占比在 1%（所用分位阈值）附近、或慢步单次耗时没变 ⇒ 命中候选。
2. 两轮总步数不同 ⇒ 直接不可比，命中；控制实验（只改输出长度）P99 回落而慢步不变 ⇒ 坐实假象。
### 证据要求
- 每步耗时直方图或慢/快步计数，以及慢步占比与分位阈值的关系；两轮的输出长度、并发度、prefill chunk 策略、采样窗口是否一致。
- 控制实验的命令与结果（慢步数不变、总步数变化后 P99 的变化倍数）；结论必须区分「慢步变慢」与「慢步占比变大」。
### 参考
- Plugins/ops/skills/op-gate/SKILL.md（§一 gate3 端到端 A-B、噪声门）与 scripts/opbench.py（median + MAD 与 `significant()` 的噪声判据）
- Plugins/ops/skills/op-bench/SKILL.md（计时口径与分位数报告的边界）与 Plugins/ops/skills/op-profiling/SKILL.md（计时与 PMC 是两套口径）

## OP-GATE-004 · 融合 epilogue 前先算全局 RMW 冲突：带 atomicAdd 的融合在冲突下净负
```yaml
id: OP-GATE-004
name: 融合 epilogue 前先算全局 RMW 冲突：带 atomicAdd 的融合在冲突下净负
category: gate
applies_to: gfx938
status: verified
```
### 知识点
「把后处理融进 GEMM epilogue」只省一次写回 + 一次读入，前提是**后处理逐元素、且每个输出元素只被一个 block 写**。跨行归约（如 MoE 的 `moe_sum`：把 topk 个专家输出按权重相加）放进 epilogue 只能靠 `atomicAdd` 落全局内存——这是**全局 read-modify-write**：N 个 block 争抢同一批输出地址，硬件串行化，代价通常远大于省下的那次读写。本项目实测（同进程交替 A/B，median + MAD 判据）：`GEMM2 + moe_sum`（atomicAdd）融合**慢 2.5~3×**（根因全局 RMW + 写冲突）；triton 单独替代 `moe_sum` **慢 8×**（bit-exact，但多一次读写、无融合收益）；`GEMM1 + silu-mul + fp8 量化`（g1f）**结构不可行**——decode 的 `BLOCK_N=16`，gate 列 `[0,512)` 与 up 列 `[512,1024)` 跨 block，纯 epilogue 融不到一起，要融必须拉宽 `BLOCK_N`，等于动 GEMM 主体。
**已证伪（不要重开）**：「融合一定更快」；「微基准局部更快 ⇒ 端到端也更快」。融合默认**净负**，推翻必须同时证明：收益不被搬移吃回去（省下的时间不能变成新的全局同步/原子操作/额外读写），且端到端（step 级）有真收益。
**反转条件**：归约能被 tiling 化成「每个输出元素只被一个 block 覆盖」（无跨 block 写竞争），且省下的全局读写确实大于新增同步成本——才值得重试；换形状（如每行只被一个 block 覆盖的小 M）要重新算冲突度，不能沿用旧结论。
### 错误写法
反例：`for blk in blocks: ... gemm ...; atomicAdd(out[row], partial)`（同一行被 topk 个 block 写，只按「多一条原子指令」做预算）；`cand_ms / base_ms = 0.97 -> "融合快 3%"`（未看端到端，未算冲突度）。
### 正确写法
融合决策三段（缺一段不许落地）：① 冲突度 = 每个输出元素被多少个 block 写（atomicAdd 时 >1 即需量化）；② 搬移账 = 省下的全局读写字节 vs 新增的原子/同步次数与串行化成本；③ 两级门 = 微基准（同进程交替、median+MAD）+ 端到端（step 级/服务级）。冲突度 >1 时先按「不融合」基线设计；要试就在隔离微基准里量，并保留可 `--reverse` 的独立补丁；epilogue 归约优先走 shared/寄存器内归约（块内独占输出）。只过微基准的融合结论固定为 `DIAGNOSTIC-ONLY`，禁止宣称收益。
### 识别方法
1. epilogue 里出现 `atomicAdd` / `atomicMax` 等写全局动作，或每个输出位置被 >1 个 block 写 ⇒ 命中候选（先算冲突度）。
2. 收益账只写「省一次读写」、没写新增原子/同步代价，或只有微基准没有端到端 ⇒ 命中（后者最多 `DIAGNOSTIC-ONLY`）。
3. 融合改了 GEMM 主体（拉宽 `BLOCK_N`、改列划分）⇒ 已超出「纯 epilogue」，按结构性改动重新判。
### 证据要求
- 冲突度计算：输出元素数、写它的 block 数、归约维与 block 划分的关系；端到端 step 级/服务级口径与质量门不回退。
- 机制说明（RMW 冲突 / 串行化 / 额外读写）写进 `candidates.yaml:refuted`（`loop.py impl-broken` 或 `direction-refuted` 落盘）。
### 参考
- Plugins/ops/skills/op-moe-w8a8/SKILL.md（§四 融合判据与已证伪表）与 Plugins/ops/skills/op-gate/SKILL.md（§三 融合类改动的额外门）
- Plugins/ops/skills/op-bench/SKILL.md（铁律 4：MAD 显著性；铁律 2：同卡同进程交替）与 Plugins/ops/skills/op-optimize/SKILL.md（证伪清单必须写机制）；scripts/opbench.py（`significant()` 噪声门）
