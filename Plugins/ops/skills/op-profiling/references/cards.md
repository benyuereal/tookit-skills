# op-profiling 知识卡（3 张）

剖析类知识卡：把 PMC 计数器读成判据时，先看这里有没有踩过的坑。每张卡六段固定：知识点 / 错误写法 / 正确写法 / 识别方法 / 证据要求 / 参考；**只有「参考」段允许出现绝对耗时/带宽**，绝对数字归 `~/.config/dcu-toolkit/kb/op-opt/`。卡 `id` 与 `op-profiling/SKILL.md` 的「## 知识卡索引」一一对应，由 `Plugins/ops/tests/test_card_index.py` 校验。`applies_to: gfx938` = 本机实测，换设备要重测。

## OP-PROFILING-001 · 描述性事实不能当结论：多个 kernel 共享同一几何会同时命中

```yaml
id: OP-PROFILING-001
name: 描述性事实不能当结论：多个 kernel 共享同一几何会同时命中
category: instruction
applies_to: gfx938
status: verified
```
### 知识点
PMC 读数分三级，**证据地位不同**：贴顶类（MBU/MFU 贴天花板、MFMA 发射饱和）可当结论，前提见 `OP-PROFILING-002`；比值类（`非MFMA VALU/MFMA`、`LDS/MFMA`）只能区分变体；描述类（`CTA/CU`、`issue_util`、`L2 命中率`、`VALU/LDS 开销重`）**只是描述**。`scripts/pmc.py`（§六 唯一实现）只吐机械原值，三级划分与判读由人按 §四 执行。**为什么不能混用**：同一几何的 kernel 会同时命中同一批描述性事实。本项目 `flash_mla` 与 tilelang `bm32` **驻留 CTA 数相同（都是 2；2026-11-13 用动态 LDS 探针复核，旧的「=1」是常数错）、`issue_util` 都低于 10%**，两者却差 **2.18×**；把这类描述写成「占用率受限 ⇒ 有优化空间」，会得出「快 2.18× 的那个也有问题」的荒谬结论。真正能区分的是比值：`非MFMA VALU/MFMA`（约 0.73 vs 约 3.11）与 `LDS/MFMA`（约 0.53 vs 约 1.67）。
**已证伪（不要重开）**：「`CTA/CU=1` ⇒ 占用率受限 ⇒ 有优化空间」；「bank 冲突是优化目标」——厂商调优 matmul 自己有约 13% 的冲突占用率照样跑到实测峰值的近九成，我们的实现还更低。
**反转条件**：描述类事实只在**同一实现内部纵向对比**（改版前后）时有指示意义；跨实现比较必须换比值类证据，并固定同一 FLOPs / MFMA 口径 / CU 口径。
### 错误写法
反例：观察到 `CTA/CU=1` 且 `issue_util=7%` 就写「占用率受限，还有优化空间」（两个 kernel 同时命中，无法区分）；「`SQ_LDS_BANK_CONFLICT` 高 ⇒ 优先降冲突」（与是否贴峰值不相关）。
### 正确写法
判读顺序：① 算力——MFMA 发射率 ≥ 75% 才算被算力卡住（本项目多数实现远低于此）；② 内存——DRAM 字节 + TA 数据停顿 + 池子敏感性三项一起看，三项都指向「不缺内存」才排除；③ 片上开销——只看 `(SQ_INSTS_VALU − MFMA)/MFMA` 与 `SQ_INSTS_LDS/MFMA` 两个比值；④ 描述类事实只作定位线索，不进结论句。跨变体比较固定同一 FLOPs/MFMA 口径与 CU，并用同一形状三方对照（本实现 / 对手 / 已知高效 matmul）。
### 识别方法
1. 结论句引用了描述类事实（`CTA/CU`、`issue_util`、`L2 命中率`），或把它与「有空间」直接绑定 ⇒ 命中。
2. 跨变体下结论却没有任何可区分比值 ⇒ 命中；两实现比较时 FLOPs / MFMA 口径 / CU 不同 ⇒ 比值不可比。
3. 结论只用描述性事实、没有任何一个可跨变体区分的量 ⇒ 命中。
### 证据要求
- 同一形状的三方对照表（本实现 / 对手实现 / 一个已知高效的 matmul）；两个比值的具体数值与口径（FLOPs、每个 MMA 的 FLOPs 数、CU 取值）。
- 描述类事实只作线索列出，并说明为什么它不能区分（列出同样命中的对照 kernel）；若声称「卡在算力/内存」，给出 MBU/MFU 的完整分母（DRAM 字节 / 实测峰值 / FLOPs / CU）、来源与日期。
### 参考
- Plugins/ops/skills/op-profiling/SKILL.md（§四 三级证据由人执行、§六 权威实现只产原值）与 scripts/pmc.py；Plugins/ops/skills/op-sparse-mla/ledger.md（§二 24：比值区分变体的实测记录）
- Plugins/ops/skills/op-roofline/SKILL.md（预测侧与实测侧对账的分工）

## OP-PROFILING-002 · 口径不全（缺分母/证据）时禁止写「到顶」

```yaml
id: OP-PROFILING-002
name: 口径不全（缺分母/证据）时禁止写「到顶」
category: gate
applies_to: gfx938
status: verified
```
### 知识点
「到顶」需要四个外部口径齐全：FLOPs（反算 MFMA 条数）、DRAM 字节（MBU 分子）、同 SoC 实测峰值（MBU 分母）、CU 数（归一化分母），再加上 `op-roofline/derive.py` 给出的唯一权威天花板。缺任何一个，对应的 MBU/MFU 就不成立、写进「缺什么」清单；此时**没有证据**说明是否到顶，任何「到顶」「物理上限」「没有空间」的结论都被禁止。**为什么**：「算不出贴顶」≠「没有到达上限」，也≠「已经到达上限」——它只是**不知道**；把 `unknown` 当 0、或用一个没锁定的分母把 `unknown` 凑成百分比，是最常见的错法。`scripts/pmc.py` 故意不内置这些常数、也不吐 MBU/MFU，就是为了让缺口显式暴露。
**已证伪（不要重开）**：「没算出贴顶 ⇒ 说明还有空间」；「缺实测峰值就按型号标称值填」——本机流式只读实测峰值比型号标称高约 1.5 倍，拿标称当分母会把「到顶」翻成「还有约 1.8 倍」。
**反转条件**：补齐四个口径并锁定天花板后贴顶判断才有意义；峰值必须按 SoC 显式传入并记录来源与日期。换卡、换访问模式（只读/只写/读写混合）、换 dtype 都要重新锁分母。
### 错误写法
反例：只拿到 VALU / TA / LDS 原值，没算 MBU/MFU、也没有天花板，却写「memory bound，已到物理上限」（缺分母与推导）；`peak = 型号标称值（status: unverified）` 后 `mbu = achieved / peak`（结论随分母翻转）。
### 正确写法
缺证据时输出 `mbu: unknown（缺实测峰值：必须按 SoC 显式给，带来源与日期）`，只描述比值类/描述类事实，不出现「到顶/物理上限/没有空间」；补齐后写 `bound: NEAR-BANDWIDTH-CEILING（分母=同 SoC 实测可达峰值，来源 constants.yaml，日期）`。口径不齐时报告里只允许「证据不足，无法判定」；分母逐个写明来源；峰值缺失就不算 MBU，不许用标称值凑。补齐口径重算后再下贴顶判断，并把命令与取值写进证据。
### 识别方法
1. 「缺什么」清单非空还写「到顶」，或 MBU / MFU / 相关比值算不出来（缺分母）⇒ 命中。
2. 峰值来源不显式（SoC、访问模式、日期、`status`）或用型号标称值当分母 ⇒ 命中。
3. 把 `unknown` 当作 0 或当作「未到达」⇒ 命中。
### 证据要求
- 缺什么：哪一项、为什么缺（原文）；补齐后的重算命令与四个口径取值（FLOPs / DRAM 字节 / 实测峰值 / CU）。
- 贴顶判定的完整推导链（从四个输入到结论式子的每一步）；分母来源条目（`constants.yaml` 路径）、测量日期、访问模式与 dtype。
### 参考
- Plugins/ops/skills/op-profiling/SKILL.md（§四 三级证据、§六 权威实现不内置常数、§七 红线）；Plugins/ops/skills/op-gate/SKILL.md（两级天花板门禁消费人算出的贴顶证据）
- Plugins/ops/skills/op-roofline/constants.yaml（峰值常数的 `source / verified_on / status`）；Plugins/ops/skills/op-gate/references/cards.md#OP-GATE-001（两级天花板）

## OP-PROFILING-003 · 计时口径：必须 median + MAD；单次/均值不得当结论；流式 frame 计时会系统性虚高

```yaml
id: OP-PROFILING-003
name: 计时口径：必须 median + MAD；单次/均值不得当结论；流式 frame 计时会系统性虚高
category: caliber
applies_to: gfx938
status: verified
```
### 知识点
三条互不回退的口径事实：**① 单次/均值不能当结论**——DCU 上同一 kernel 抖动可达 5%，必须多轮取**中位数**，用 `|Δ| > k × MAD`（k 默认 3）判显著；过不了门是 `INCONCLUSIVE`（「在噪声内」），不许写成「快了 x%」。**② 流式 frame 计时会系统性虚高**——按 SSE frame 间隔折算 per-token 会把服务端 flush/批量行为算进单 token，本项目实测虚高约 **2.15×**；真实 per-token 必须用 `usage.completion_tokens` 折算。**③ PMC 墙钟 ≠ bench 耗时**——PMC 有采集开销且首次 launch 数据不可用；计时精度与显著性门在 `op-bench`，PMC 只给结构性证据。`scripts/pmc.py` 输出的 `EndNs-BeginNs` 只是采集时的墙钟，不要拿它当 bench 值，也不要反向去改 PMC 口径。
**已证伪（不要重开）**：「跑两次看着差不多就写收益」；「SSE frame 的间隔就是 ITL」；「PMC 墙钟能替代 bench」。
**反转条件**：只有两轮口径完全一致（同样的 token 计数来源、计时方式、冷/热状态）时才能相比；换成 usage token 后必须重算，不能复用旧的 frame 数字。
### 错误写法
反例：`t = time() ... time()` 只取一次样本就写「快了 3%」（没有 MAD，无法区分噪声）；`itl = t(frame[i+1]) - t(frame[i]); report: "TPOT = mean(itl)"`（frame 间隔当 per-token，未用 usage 折算）。
### 正确写法
`bench_fn(fn, repeats=3)`（每轮取中位数，最后取三轮中位数）；`significant(base, cand, k=3)` 判 `|Δ| > 3×max(MAD_base, MAD_cand)`；报告写「ratio=0.96，Δ > 3×MAD；口径=hot / op_call / 含分配；计时=cuda_event」；`per_token = wall_time / usage.completion_tokens`。结论必须带**口径三问**答案：测什么（kernel / 算子调用 / 端到端 step）、含不含分配、冷还是热；只用热轮，冷轮只用来暴露冷启动问题；端到端报告 P99 时必须并列 P50/P90 与慢步占比（见 `OP-GATE-003`），不能只看分位数。
### 识别方法
1. 没给出 MAD 与轮数、只给单个样本或均值 ⇒ 命中。
2. per-token 时延来源是 frame 间隔而非 `usage.completion_tokens` ⇒ 命中。
3. 是否把 PMC 墙钟当 bench 耗时、或跨两套口径比较 ⇒ 命中。
4. 结论缺口径三问，或冷/热两轮混用（用冷轮数字下结论）⇒ 命中。
### 证据要求
- 每轮样本、median、MAD、轮数与显著性判定式；token 计数来源（usage 字段名）与折算公式。
- 计时方式（CUDA/HIP event、wall clock、服务端 frame）与冷/热状态；口径三问的答案；若引用 PMC，写明它只用于结构性证据。
### 参考
- Plugins/ops/skills/op-bench/SKILL.md（口径三问、铁律 4 与 6、参数表）；scripts/opbench.py（`summarize()` / `significant()` / `bench_fn()`）
- Plugins/ops/skills/op-profiling/SKILL.md（§二 什么时候用 PMC、§七 红线）；Plugins/ops/skills/op-gate/SKILL.md（gate3 端到端 A-B 与噪声门）；Plugins/ops/skills/op-gate/references/cards.md#OP-GATE-003（分位数陷阱）
