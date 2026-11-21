# op-roofline 知识卡（3 张）

预测类知识卡：算地板/建模型时，先看这里有没有踩过的坑。每张卡六段固定：知识点 / 错误写法 / 正确写法 / 识别方法 / 证据要求 / 参考；**只有「参考」段允许出现绝对耗时/带宽**，绝对数字归 `~/.config/dcu-toolkit/kb/op-opt/`。卡 `id` 与 `op-roofline/SKILL.md` 的「## 知识卡索引」一一对应，由 `Plugins/ops/tests/test_card_index.py` 校验。`applies_to: gfx938` = 本机实测，换 gfx/设备要重测。

## OP-ROOFLINE-001 · VGPR 预算必须按作用域取 max，不能把不同作用域的声明平铺求和

```yaml
id: OP-ROOFLINE-001
name: VGPR 预算必须按作用域取 max，不能把不同作用域的声明平铺求和
category: instruction
applies_to: gfx938
status: verified
```
### 知识点
寄存器预算看的是**同时活跃**的变量，不是声明段里出现过的所有变量。正确算法：按作用域分别求**活集合**，再取各作用域的 **max**；不同作用域（不同分支、前后循环体）的变量不会同时活着，平铺求和会把预算高估数倍。硬上限：活变量声明量 ≥ 每线程上限（gfx938 的 768-VGPR 模式为 256，其他 gfx 未验证）⇒ **保证溢出**，编译前即可否掉。临时量不进声明段：实测可多出约 148 VGPR，所以「未超限」**不保证**不溢出，必须编译后读 `vgpr_count` / `vgpr_spill_count` 复核。
**已证伪（不要重开）**：「把 4 个小 buffer 合成 1 个大 buffer 更高效」——4 路切分是**寄存器生命周期分段**（顺序复用同一批 VGPR），合并成 1 个大 buffer 后生命周期全程重叠，直接顶爆上限（实测 spill 30、慢 2.12×）。同类死法：`kv4`、`g2s`、`bm64`、`thr128`、`bi64`。
**反转条件**：只有证明这些 buffer 的生命周期**确实不重叠**（顺序复用同一批寄存器）时，合并才可能成立；否则按 `hipbudget.py` 的活集合分析走，不按声明求和。
### 错误写法
反例：`vgpr = vgpr_gemm1 + vgpr_silu + vgpr_gemm2 + vgpr_cvt`（各分支/各循环的声明相加，三个阶段并不同时活跃）；`if vgpr < 256: "不会溢出"`（只看声明量，临时量另算，实测可多出约 148）。
### 正确写法
按作用域求活集合并取 max：gemm1 = {acc_s, q, kv}、silu = {acc_s, tmp}、gemm2 = {acc_o, acc_s, kv}，`budget = max(A,B,C)`；`budget >= 256 ⇒ 保证溢出`（编译前否掉），`< 256 ⇒ 仅"未超限"`。编译后复核 `vgpr_count` / `spill_count`；每 CU 驻留数取 `min(预算/(vgpr×线程数), LDS 容量/LDS)`。结论只允许两种措辞：「保证溢出」或「未超限（不保证不溢出）」；「未超限」之后必须补编译产物读数，两者不一致时以编译产物为准，并回头修活集合分析。
### 识别方法
1. 预算来自「声明段求和」、或没有区分作用域（分支、前后阶段）⇒ 命中。
2. 忽略临时量（实测可多出约 148 VGPR）却据此断言不溢出 ⇒ 命中。
3. 结论只有「未超限」就结束，没有编译产物复核 ⇒ 命中；有「合并 buffer 更高效」候选时先按生命周期分段核对，再判。
### 证据要求
- 每个作用域的活变量清单与预算、`max` 的取法；每线程上限（硬件常数）的来源。
- 编译产物读数：`vgpr_count`、`vgpr_spill_count`、每 CU 驻留 CTA 数；相对倍数与口径（同卡同进程、median + MAD）。
### 参考
- Plugins/ops/skills/op-roofline/SKILL.md（规则 R1 寄存器活集合、R6 LDS 预算、硬规则 4）；Plugins/ops/skills/op-roofline/references/rules.md（规则细则、反例与校准点）
- Plugins/ops/skills/op-roofline/tools/hipbudget.py（R1 的可执行版，`--live`）；Plugins/ops/skills/op-optimize/SKILL.md（§五 反直觉清单：合并 buffer 的死法）；Plugins/ops/skills/op-sparse-mla/ledger.md（结构结论与已证伪记录）

## OP-ROOFLINE-002 · packed 布局逼出窄向量化：尾部非对齐让访存效率掉到峰值零头

```yaml
id: OP-ROOFLINE-002
name: packed 布局逼出窄向量化：尾部非对齐让访存效率掉到峰值零头
category: memory
applies_to: gfx938
status: verified
```
### 知识点
访存效率由**每线程连续字节数**决定（规则 R4）：16B 可达实测流式峰值的 94~97%，4B 面远低于此。数据被打包成 `[slots, 132]` 这类尾部非对齐布局时，每行可连续读取宽度被限制在 ≤4B，向量化宽度被硬压下来，**mem-only 变体的实测带宽只有峰值的一小部分**；拆成分离平面（各张量各自连续）后，mem-only 才能接近实测可达峰值。**这个达成率有两个分母，引用必须写明用哪个**：按型号标称算是约 87%，按本机实测流式峰值算是约 **55%**（本机实测比标称高约 1.5 倍，见 `OP-CALIBER-001`）。分母换了，「到顶」的结论就翻转。关键推论：**性能不可跨布局比较**——packed 与分离平面的达成分母不同，混用会得出相反的「到顶」结论。
**已证伪（不要重开）**：「该 kernel 已贴带宽上限」——最初按名义读量与 packed 实测算出的高利用率，在分离平面与 mem-only 隔离实验后被推翻；「dot/计算占大头」——mem-only 变体只比整 kernel 低约 26%，说明瓶颈在访存效率而不是算术。
**反转条件**：访问改成对齐的大块连续（16B 及以上）后 packed 惩罚消失，结论反转为「带宽不再是主因」；换布局（packed ↔ 分离平面）必须重测，不能沿用旧达成率。
### 错误写法
反例：把 packed 与分离平面各自的 `achieved/peak` 相除得出「分离平面快 x%」（分母/布局都不同，不可跨布局比较）；用 `torch.randn(shape)` 构造「看起来一样」的输入（步长/对齐与生产输出不同）。
### 正确写法
两步实验：① mem-only 变体（GEMM/softmax 换成从 shared 的廉价累加，gather 与 shared 读写不变）——它到峰值的比例 = 访存天花板，整 kernel 与它的差额 = 「gather 与计算不重叠」的损失；② 布局对照——packed vs 分离平面，各自只与自己同布局的基线比 ratio。报告同时给名义吞吐、有效吞吐（`effective_topk` 折算）、每线程连续字节数；跨布局结论只写「方向」与「相对倍数」，并注明分母来源。生产布局优先；微基准输入从生产路径 dump，或至少用生产输出形状/步长构造；布局事实写进 `contract.yaml:layout_constraints` 并注明「两种布局性能不可跨布局比较」。
### 识别方法
1. 每线程连续字节数 ≤4B ⇒ 命中（窄向量化）；存在 `[slots, N+pad]` 尾部非对齐布局 ⇒ 检查向量化宽度。
2. 跨布局比较达成率或绝对耗时 ⇒ 命中。
3. 没做 mem-only 隔离实验就断言「带宽到顶/不是带宽问题」⇒ 命中。
4. 输入张量是 harness 自造却声称与生产等价 ⇒ 命中。
### 证据要求
- 布局与 stride 的具体形式、每线程连续字节数的推导。
- mem-only 变体与整 kernel 的对照结果，两者到峰值的比例及分母来源；同一布局内部的 ratio，明确标注不可跨布局比较。
- 名义量 vs 有效量的折算（`effective_topk = mean(indices>=0) × topk`，适用于带 padding 的算子）。
### 参考
- Plugins/ops/skills/op-roofline/SKILL.md（规则 R4：访问粒度决定有效带宽）；Plugins/ops/skills/op-roofline/references/rules.md（R4 的校准点与反例）；Plugins/ops/skills/op-contract/SKILL.md（§三 layout_constraints 的写法）
- Plugins/ops/skills/op-sparse-mla/SKILL.md（§四 判据：mem-only 变体、packed 布局）；Plugins/ops/skills/op-bench/SKILL.md（铁律 1 输入布局、铁律 3 有效量折算）

## OP-ROOFLINE-003 · 带宽利用率按「实测可达峰值」算而非型号标称，且区分读/写/混合口径

```yaml
id: OP-ROOFLINE-003
name: 带宽利用率按「实测可达峰值」算而非型号标称，且区分读/写/混合口径
category: caliber
applies_to: gfx938
status: verified
```
### 知识点
达成率 = 实测值 / **分母**，分母有三个独立的坑：**① 标称 vs 实测**——型号标称只是型号名，本项目实测流式只读峰值比型号标称**高约 1.5 倍**；拿标称做分母会把「已经到顶」写成「还有约 1.8 倍空间」（见 `OP-CALIBER-001`）。**② 读 / 写 / 混合**——纯读流式、纯写、copy（读+写）三个口径的实测可达峰值不同，量级差约 **1.3 倍**；用一个口径的分母去判另一个口径的 kernel，结论直接反转。**③ dtype / 算力口径**——fp8 与 bf16 的可达算力峰值不是同一个数（本机 fp8 GEMM 可达峰值约为 bf16 反推峰值的 2.9 倍），拿 fp8 的 FLOPs 配 bf16 峰值会算出荒谬的达成率（>100%），这本身就是口径混了的反证。
**已证伪（不要重开）**：「型号标称值就是天花板」；「纯读峰值可以套到 copy 上」；「一个算子的达峰百分比可以跨分母复用」。
**反转条件**：换访问模式、换 dtype、换卡、换时钟档位、换锁频策略 ⇒ 分母全部重锁；只有本机实测 STREAM 且注明「实测」时才可用实测值做分母。
### 错误写法
反例：`floor: {metric: hbm_bandwidth, peak: <型号标称>, basis: "型号参数"}`（契约地板直接写标称值，循环论证/高估分母）；读写混合 kernel 用纯读峰值，算出「达峰 >100%」（分母口径错了）。
### 正确写法
`floor.peak` 取同 SoC、同访问模式、同 dtype 的实测可达峰值，带 `source` / `status: measured` / `verified_on`；报告口径行写「达成率 x%（分母=本机流式只读实测值，卡 n，日期；读写混合口径另算）」。两个来源冲突时**并列记录**，不取平均、不挑好看的那个；分母要同时写进契约 `floor` 与报告口径行，两处一致。结论只写「达成率 x%」与状态；绝对性能数字进 `~/.config/dcu-toolkit/kb/op-opt/`。
### 识别方法
1. `floor.peak` 的来源是标称还是实测？标称/未标 `status` ⇒ 命中。
2. kernel 的访问模式（只读 / 只写 / 混合）与分母口径是否一致？不一致 ⇒ 命中。
3. 达成率是否 >100% 或异常接近 100%？先怀疑分母口径混了。
4. 是否跨 dtype 套用同一个算力峰值、或报告没写分母来源与测量日期 ⇒ 命中。
### 证据要求
- 分母的测量命令、访问模式、卡号、时钟档位、日期，以及 `constants.yaml` 中的条目路径与 `status`。
- 同一份实测在两个分母（标称 / 实测）下的达成率对比，说明结论如何翻转；访问模式声明（纯读、纯写还是读写混合；混合时写量与读量比例）；冲突来源的并列记录（若存在）。
### 参考
- Plugins/ops/skills/op-roofline/SKILL.md（硬规则 2：一个数只能有一个分母；§八 红线）；Plugins/ops/skills/op-roofline/constants.yaml（peak-bw 条目与 conflicts 记录）
- Plugins/ops/skills/op-contract/SKILL.md（§五 地板怎么写才算数）；Plugins/ops/skills/op-profiling/SKILL.md（§七 峰值带宽必须按 SoC 显式传）；Plugins/ops/skills/op-bench/SKILL.md（铁律 5：地板按带宽/算力模型算）
