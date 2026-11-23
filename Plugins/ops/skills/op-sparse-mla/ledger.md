# op-sparse-mla 结论台账

> 规则：本文件只写**档位 / 变体 / 相对值 / 状态 / 口径**。绝对耗时、带宽、TFLOPS、寄存器数、
> 索引池大小与消融表**不进 skill 树**，统一落在
> `~/.config/dcu-toolkit/kb/op-opt/sparse-mla-20261027.md`（条目纪律见 `data/kb/README.md` 第 3 条）。
> 每行必须能回到一次可复跑的 run。

## 一、档位 × 变体（相对值）

| 档位 | 变体 | 相对值 | 状态 | 证据 | 日期 |
|---|---|---|---|---|---|
| S=32 ctx64k topk2048 | tilelang | **1.00（基准）** | measured | `bench/bench.py` 2026-10-27 | 2026-10-27 |
| S=32 ctx64k topk2048 | flash_mla | 2.40×（慢） | measured | 同上 | 2026-10-27 |
| S=128 ctx8k H16 topk2048 | flash_mla | 0.81×（快） | measured | 同上 | 2026-10-27 |
| S=128 ctx64k H16 topk2048 | flash_mla | 0.80×（快） | measured | 同上 | 2026-10-27 |
| S=128 ctx128k / ctx320k H16 topk2048（**真池**） | flash_mla | 1.005× / 1.151×（慢） | measured | `bench/bench_long.py` 2026-10-28，pool=full | 2026-10-28 |
| S=512 ctx64k H16 topk2048 | flash_mla | 0.65×（快） | measured | 同上 | 2026-10-27 |
| S=2048 ctx64k H16 topk2048 | flash_mla | 0.61×（快） | measured | 同上 | 2026-10-27 |
| S=128 ctx64k H64 topk2048 | flash_mla | 0.35×（快） | measured | 同上 | 2026-10-27 |
| S=512 ctx64k H64 topk2048 | flash_mla | 0.30×（快） | measured | 同上 | 2026-10-27 |
| S=128 ctx64k H16 topk4096 | flash_mla | 0.62×（快） | measured | 同上 | 2026-10-27 |
| S=1 ctx128k / ctx320k | flash_mla | 3.5× / 3.1×（慢） | measured | `bench/bench_long.py` 2026-10-27（q_start=ctx ⇒ 真池） | 2026-10-27 |
| S=32 ctx128k / ctx320k | flash_mla | 2.6× / 2.8×（慢） | measured | 同上 | 2026-10-27 |
| S=128 ctx128k / ctx320k | flash_mla | 0.95× / 1.09× | measured | 同上；预取后 1.00× / 1.16× | 2026-10-27 |
| S=4096 ctx128k / ctx320k | flash_mla | 0.93× / 1.41× | measured | 同上；预取后 0.98× / 1.50× | 2026-10-27 |
| S=128 / S=4096 ctx320k | tilelang + gather 预取 | 0.946× / 0.939×（快 5~6%） | measured | `bench/prefetch.py` 2026-10-27，逐位一致 | 2026-10-27 |
| S=32 ctx320k（split 内核，ns=4） | tilelang + gather 预取 | 0.956×（快 4.4%） | measured | 同上，逐位一致 | 2026-10-27 |
| S=128 ctx64k（decode-s128 档） | tilelang + gather 预取 | 0.970×（快 3.0%） | measured | 同上，逐位一致 | 2026-10-27 |
| V4.1 Flash 形状 S=32 ctx320k topk512 H=64 | flash_mla | **0.84×（快 1.19×）** | measured | `build_inputs(pool="full")` 2026-10-27，真随机池（原 `bench_real.py`，脚本已并入 `_common`） | 2026-10-27 |
| V4.1 Flash 形状 S=128 ctx320k topk512 H=64 | flash_mla | 0.80×（快 1.26×） | measured | 同上 | 2026-10-27 |
| V4.1 Flash 形状 S=4096 ctx320k topk512 H=64 | flash_mla | 0.70×（快 1.42×） | measured | 同上 | 2026-10-27 |
| GLM5.3 形状 S=32 ctx128k topk2048 H=64 d512 | flash_mla | **0.92×（快 1.08×）** | measured | 同上，576 零尾等价 | 2026-10-27 |
| GLM5.3 形状 S=128 ctx128k topk2048 H=64 d512 | flash_mla | 0.52×（快 1.93×） | measured | 同上 | 2026-10-27 |
| GLM5.3 形状 S=4096 ctx128k topk2048 H=64 d512 | flash_mla | 0.43×（快 2.32×） | measured | 同上 | 2026-10-27 |
| **生产口径** S=128 topk2048（2000 有效 + 48×`-1`）SKV=32k | flash_mla | **2.89×（快）** | measured | `batch=4` 版；同档 SKV=128k 是 1.74×——**fm 对 KV 池更敏感**（fm 随池变、我们几乎不动）。绝对数见 kb §20.1 | 2026-10-27 |
| **生产口径** S=32 topk2048（2000 有效 + 48×`-1`）SKV=32k | flash_mla | **0.93×（我们快 1.07×）** | measured | 同 `batch=4` 口径；绝对数见 kb §20.1 | 2026-10-27 |
| ~~V4.1 S=32 ctx320k H=64~~ | ~~flash_mla 1.04×（我们快）~~ | **已作废**：热工作集口径，见上方口径红线 | retracted | `/tmp/alt/hb.py`（旧） | 2026-10-27 |
| S=1 ctx320k（生产档） | flash_mla | 3.1×（慢） | measured | `bench/bench_long.py` 2026-10-27 | 2026-10-27 |
| S=1 ctx320k（生产档） | tilelang + gather 预取 | 0.996×（≈持平，S=1 已被 launch 平底盖住） | measured | `bench/prefetch.py` | 2026-10-27 |
| S=32 ctx128k H64 topk2048 **d512（tail_dim=0）** | tilelang | **1.00（基准，真 512 内核）** | measured | `bench/bench.py --shape decode-glm53` 2026-10-27 | 2026-10-27 |
| S=32 ctx128k H64 topk2048 d512 | flash_mla | 1.03×（慢 3%） | measured | 同上，同进程交替 | 2026-10-27 |
| S=32 ctx128k H64 topk2048 d512 | tilelang + gather 预取 | 0.970×（快 3.0%） | measured | 同上，逐位一致 | 2026-10-27 |
| S=4096 ctx128k H64 topk2048 **d512（tail_dim=0）** | tilelang | **1.00（基准，真 512 内核）** | measured | `bench/bench.py --shape prefill-glm53` 2026-10-27 | 2026-10-27 |
| S=4096 ctx128k H64 topk2048 d512 | flash_mla | 0.50×（快 2.0×） | measured | 同上，同进程交替 | 2026-10-27 |
| S=4096 ctx128k H64 topk2048 d512 | tilelang + gather 预取 | 0.967×（快 3.3%） | measured | 同上，逐位一致 | 2026-10-27 |
| 真 512 内核 vs「576 尾部补零」等价法 | tilelang | 0.94~0.96×（真 512 快 4~5%） | measured | `t3b_perf_dist.py` 2026-10-27，两种 indices 分布 × S=32/4096 四点一致，逐位一致 | 2026-10-27 |
| S=1 ctx320k topk512 H64（V4.1 Flash 生产 decode） | tilelang | 可运行（改前撞断言，比不了） | measured | 原 `tests/t4_s1_topk512.py`，脚本已随 3a96bc8 移除（门的流程统一进 `scripts/opcheck.py`）；本行只留结论 | 2026-10-27 |
| **V4.1 Flash 生产 decode**（H=64 / d_qk=576 / topk=512） | 全部 | — | **todo（已立项）** | 档位 `decode-v41`；改造路径见该档 `notes`。缺口：本卡没有任何实现覆盖此形状 | 2026-10-29 |
| H=16 测试档（tilelang 示例形状，**无对应模型**） | hip / scalar_ref | 已移除 | retired | `/models` 无一模型是 H=16（两个生产模型均 H=64），见 SKILL §三 生产形状核对；源码仍在 1.6.0 提交 `2c9cf34` | 2026-10-29 |
| prefill-8k / prefill-320k；生产 `prod-vllm` | 全部 | — | draft | 待测 / 需在生产安装树跑，暂未纳入 | — |

**读法**：比值 = 其它变体耗时 / tilelang 耗时（同卡同进程）。<1 表示比 tilelang 快。

> caveat **口径红线（比值可不可信的唯一判据）**：`build_inputs` 的默认索引窗口只从 `[0, t+q_start)`
> 取索引 ⇒ 小 S 时工作集只有几 MB、**全在 L2**，等于给内存受限的 flash_mla 免了内存费，
> **系统性低估 tilelang**。判据是**池大小不是参数名**：`build_inputs` 输出 `pool_extent`/`pool_mb`，
> 池 < 64 MB 打 `warning`。跨实现比站位一律真随机池（`pool="full"` 或 `q_start_s_index >= SKV`，
> 两者索引逐位相同）或用 `bench/bench_long.py`（写死 full、不提供开关）。
> **任何「我们领先」的结论先过这条口径检查**——小 S 的「领先」我们误判过两次（原始数字见 kb §20.1 / §20.2）。

## 二、结构结论（跨档位成立）

**完整叙述、全部绝对数字、PMC/消融表见 `~/.config/dcu-toolkit/kb/op-opt/sparse-mla-20261027.md` §20.1（原 §二，编号不变）。** 卡里只留不依赖数字的结论：

1. **tilelang 不随 S/H 摊销**；flash_mla 有跨 head KV 复用（H 16→64 工作量 4×、耗时仅 1.6×）—— 最大结构短板（原 §二 1/2）。
2. **SKV 不是限制因素**；S=32 的胜负来自 flash_mla 的小 S 调度地板，不是我们扩展性更好（原 §二 3/4）。
3. **两者精度等价**，没有理由按精度偏好任一实现（原 §二 5）。
4. **带宽受限不是计算受限**：把 KV 表喂进缓存，同一份 GEMM 能跑出峰值的一半以上；GEMM 只占两成时间（原 §二 6/7）。
5. **H≥32 的 3× 差距来自 `REPLICATE_H` 复读**，且复读的是 **L1/L2 流量、不是 DRAM**（三变体 L2 miss 字节一比一相等）⇒「省 DRAM」是空目标（原 §二 8/24 + §三）。
6. **gather 已到访存物理极限**（mem-only 达峰值 94~97%），整 kernel 只有 ~55% ⇒ 剩下的全是**「gather 与计算不重叠」**；根因是 `occupancy = 1 CTA/CU`（VGPR 顶满），不是带宽、不是配置（原 §二 11/12）。
7. **1 块深寄存器预取有效且逐位一致**（只搬 spilt0~2；把 spilt3/tail 一起搬进 shared 慢 1.7×）。**bm32 落地后预取反而慢 20%** —— bm16 时代的预取结论作废（原 §二 13 + §三）。
8. **真实池下我们不是带宽受限、flash_mla 才是**：池缩 64× 我们只快 ~1.05×。⇒ 对我们「提高带宽利用率」是伪目标，瓶颈在计算侧（原 §二 18/19）。
9. **【已修正】「小 S 我们领先」是热工作集假象**，真池下变成落后 —— 见上面口径红线（原 §二 17）。
10. **`tail_dim=0` 已支持**（`if D_tail > 0:` 编译期条件，不留零长 fragment）；真 512 内核比「576 补零」只快 4~5% 且**逐位一致** ⇒ 零尾可当性能代理，**不能用来判 1% 级差异**（原 §二 20/21）。
11. **站位是分段的，交叉点 S≈56~64**：小 S 我们领先来自 fm 的固定地板（**不是能力优势，别引用**）；M=32（GLM5.3 生产 decode）我们落后（原 §二 22 表）。
12. **`num_split` 必须整除 `topk/block_I`**，否则撞断言 —— V4.1 Flash 生产 decode 起不来的真因，**不是精度问题**（原 §二 23）。
13. **PMC 定性：慢的不是 GEMM，是围着 GEMM 转的搬运指令流**。MFMA 条数一比一、DRAM 字节一比一、没有任何单元打满 ⇒ 卡在依赖/延迟。唯一量化区分项 = 「每条 MFMA 摊到几条搬运指令 / 几条 LDS」；**bank 冲突不是区分项**（原 §二 24）。
14. **LDS 指令量由「每 wave 摊到几个 head」决定**，不由 `block_m`/`threads` 单独决定；抬 heads/wave 的唯一途径是 bm64，而它在 thr256 必然 spill（预算：非 acc_o 的 VGPR 要砍掉约一半）（原 §二 26）。
15. **PV 的标量 LDS 是「收缩轴」决定的，换操作数也躲不掉**；转置输出只是把标量从 gemm 挪到 copy，一分没省 ⇒ **PV 这条彻底关闭**（原 §二 29）。
16. **softmax 链定价上限 ≈11.7%**（running max 每 N 块更新，N=32 饱和；`nomax` 会 nan、`max1` 更慢更不准）—— **数值换速度、非逐位一致，需单独拍板**。bm32 口径的优先级：softmax 区 > PV 标量 LDS > gather（原 §二 28/30）。
17. **测量纪律：必须钉空闲卡，且只认同进程同卡比值**。别人正跑着的卡上两实现同时慢 ~17%，**比值会被压平到看不出差距**（我们因此把 0.79 读成 1.00）；跨进程跑同档还有 8~12% 漂移（原 §二 15）。

## 三、已证伪 / 别再重开

> **机器权威在 [`candidates.yaml:refuted`](candidates.yaml)**（`loop.py queue` 打印，`loop.py run` 拒跑靠它）。
> 本文件**不做判定、不参与门禁**；原始数值、日志/脚本路径与「我自己推翻我自己」的记录见
> `~/.config/dcu-toolkit/kb/op-opt/sparse-mla-20261027.md` §20.2（原 §三）。
> 判据只有一条：**只在叙述里存在的结论，`loop.py` 拦不住，等于没有** —— 先补进 yaml 再写叙述。

卡里只留**能拦住一类重开的机制级教训**：

- **参数面已扫穿**：`block_m` / `block_I` / `threads` / `gemm policy` / `num_stages` / `SplitKV` / 预取深度 / LDS 布局全部实测否决；`REPLICATE_H=2` 两侧都是墙（bm64 必溢出、bm16 必多读）。
- **每换一次基线，旧结论全部作废**：bm32 落地后 bm16 时代的预取收益变成 −20%。任何优化都必须在当前基线上重测。
- **降 VGPR / 删指令的消融会虚增自己的成本**（寄存器压力释放也算收益），且消融**不能横加**（`no_softmax` 里含其它项）。
- **指令数是症状不是病因**：砍掉最大一块 VALU 只换回 1.1% ⇒ 别再以「减少指令」为目标，除非先证明它落在关键路径上。
- **硬件没有转置 LDS 读**（gfx938 实测 `invalid instruction`）；**一个 LDS 布局不可能同时让 QK（收缩 d）和 PV（收缩 k）向量化** —— 数学不可能，净收益恒为 0。
- **`LayoutInference` 的冲突是结构强制的**：QK 要「8 个连续 d 同线程」、PV 要「k 相邻线程」，一个 fragment 不可能同时满足 ⇒ shared 桥接不是写法问题。
- **删模板里「看似冗余」的前导 barrier 会静默算错**（它保护的是寄存器槽被下一轮覆写的 WAR）；正解是 `batch=4` 合并，不是删。
- **`topk=512` 是 flash_mla 算错，不是我们**；该档与 2016 档的横向对比一律不可用（非生产形状，不必追）。

## 四、变体前置条件（不满足会静默算错，不是性能问题）

完整现象与依据见 `~/.config/dcu-toolkit/kb/op-opt/sparse-mla-20261027.md` §20.3（原 §四）。

| 变体 | 前置条件 | 违反后 |
|---|---|---|
| `tilelang` | **`-1` 只能垫队尾**（有效索引在前） | 静默算错（垫得越多越错，全前置输出恒 0）；`flash_mla` 两种布局都对 |
| `tilelang` | `tail_dim` 支持 0（已验）或 2 的幂；`dim` 必须是 2 的幂；`dim+tail_dim` 只接受 512 / 576 | 抛异常（不是算错） |
| `tilelang` | `sm_scale` 显式按宽度取：576 → `1/sqrt(640)`、512 → `1/sqrt(512)` | 差一个常数因子（kernel 默认是对的，错在调用方） |
| `tilelang` | `topk` 必须是 `block_I`(32) 的整数倍 | 抛断言；生产 topk≈2000 需补到 2048 + `-1` 垫尾 |
| `flash_mla` | topk=512 / 2016 两档结果错误 | 该档横向对比不可用（非生产形状） |
| `tilelang-prefetch` | 只认两种块索引写法，认不出就**整体跳过** | 认不出却仍搬 copy ⇒ **静默算错**（踩过：cos 掉到 0.836） |
| `flash_mla` | 会**原地改写 `indices`**（有效索引前移、`-1` 后移到队尾；`q`/`kv` 不动） | 多变体共用输入时，后者看到被改过的布局（曾因此误判「两种布局一致」） |

## 五、待办（下次进来先看这里）

完整清单与交付口径表见 `~/.config/dcu-toolkit/kb/op-opt/sparse-mla-20261027.md` §20.4（原 §五）。卡里只留三条**方向性**判断：

- **唯一有实测支撑的杠杆**：thr256 下把非 acc_o 的 VGPR 压到放得下 bm64（判据：`scratch==0` 且 heads/wave=16 且 DRAM 不涨）；其余参数面已否决。
- **待用户拍板**：`maxn N=32`（−11.7%，320k 首次反超 fm，但非逐位一致）——需 GPQA/HumanEval 端到端不掉分才收。
- **`prod-vllm` / `prefill-8k` / `prefill-320k` 仍未测**；H=16 的旧站位行未换真池，换完之前任何跨实现「领先」结论不可引用。
