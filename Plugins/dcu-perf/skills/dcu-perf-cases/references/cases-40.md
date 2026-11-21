# 性能优化案例 §40（W8A8 四杠杆查表调优）

> 2026-10-16 自 `perf/dcu-perf-cases/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），案例四段式原文照录、未改一字；编号与原文一致，索引见主 SKILL.md 案例索引表。

## 40. Qwen3.8-27B W8A8 K100-AI 四杠杆查表调优（SGLang 0.5.12，best_config 注入）

> 场景：Qwen3.8-27B 在 SGLang 0.5.12（K100-AI x 2，gfx928，TP2，DTK 2604）上做 INT8 w8a8 高性能推理，竞赛 SLA 判分（4K TTFT 均值<10s&TPOT<80ms；64K TTFT<30s&TPOT<100ms）。
> 与案例39 的区别：案例39 是 vLLM `fla/ops` 路径的 QK 共享去冗余 kernel 改动；本案例是 **SGLang 路径**的 **GEMM launch-config 查表注入**（不改数学，只调启动配置）。
> 模型：Qwen3.8-27B-INT8-w8a8，64 层混合 = 48 层 GDN（gated delta net 线性注意力）+ 16 层 full attention，hidden 5120，DENSE；EAGLE 投机 steps3/topk1/draft4，mem-fraction 0.80，mamba-full-memory-ratio 1.5，chunked-prefill 16384。

## 现象 / 目标

官方 baseline（同款权重同环境）在两个 SLA 场景**最大并发均为 0**（连 C1 都过不了）。目标：抬 SLA 最大并发 + 提升 TTFT/TPOT/吞吐。

## 根因（kernel/配置级）

stock lightop 的 INT8 GEMM 查表对 decode 小 M 用**无软件流水线的 num_stages**、对 prefill 大 M（>128）用 **num_stages=0**，把本可贴墙的 GEMM 留了 1.1–2x 在桌上；decode 瘦 M (M≤8，EAGLE verify 交接边界) 在 stock 表无最优条目，离墙 1.63–2.10x；GDN chunked 线性注意力的 Triton 启动配置（BK/BV/num_warps/num_stages）对 gfx928 120 CU 非最优，64K 单 kernel 留 2.15x。

## 优化动作（4 个查表/配置杠杆，全按 gfx928 实测 shape 调优）

1. **0003 decode INT8 GEMM 查表**：`w8a8_int8.py` 注入按 (M,K,N) 查表的 best_config（num_warps/num_stages/BLOCK/SPLIT_K），decode M≤128 用实测最优；env 门控 `SGLANG_INT8_TUNE`。
2. **0006 prefill INT8 GEMM 查表**：同文件，prefill M≥129 用 `num_stages=2`（stock 为 0）；env 门控 `SGLANG_INT8_PREFILL_TUNE`。
3. **0007 GDN prefill tuned config**：`wy_fast.py` `recompute_w_u` 内联改（BK64/BV64/w4/s3→BK32/BV32/w2/s2）+ aiter `AITER_TRITON_CONFIGS_PATH` JSON（chunk_fwd_o/delta_h）；env 门控 `SGLANG_GDN_TUNE`。
4. **0008 decode 瘦 M (1,8) 表扩展**：gate_up/qkvz 加 (1,8)→`BM16/BN32/BK256/st4/w4`（0003/0006 表行不变，自包含追加）；无 env 门控，patch 在即生效。

**共性纪律**：四个杠杆都是"查表传 best_config"式结构安全改动——不改算子数学、不加显存副本、无 int8 布局/失步副作用 → 精度 0pp；未列 (M,K,N) 回退 stock（fail-closed）；0003/0006/0007 各独立 env 门控可单独 A/B；0008 靠 patch 有无切换 A/B。这与 int8-lm_head 三连败（改数值路径/布局）是本质区别。

## 量化结果（同窗背靠背 A/B，canary 双指标标窗口质量；2026-10-08 重新扫描窗口）

| 指标 | 优化前（stock 全关） | 优化后（提交配置） | 变化 |
|---|---|---|---|
| 4K c1 TTFT p95 | 1.558s | 1.309s | −16.0% |
| 4K c1 TPOT p95 | ~20.4ms | 18.75ms | −8.1% |
| 4K c1 输出吞吐 | ~53.4 tok/s | 57.61 tok/s | +7.9% |
| 64K c1 TTFT p95 | 33.76s | 28.98s | −14.2% |

SLA 最大并发（宽松均值判分）：**4K = C16**（c16 TTFT 8.33s 均值 / TPOT 54.07ms 均值，余量 1.2x/1.48x；c32 TPOT 101ms 双爆）、**64K = C1**（TTFT 29.02s 贴 30s 线，余量 3.3%，c8 双爆）。4K 上限由 **TPOT** 决定（TTFT 到 c16 仍 <10s）；64K 是 TTFT-bound 平台墙（compute-bound prefill + GDN 状态池），无单代码杠杆能压进 30s。

## 可复用经验

1. **best_config 查表注入 = 零精度风险的结构安全优化**（启动配置而非数值路径/布局），DCU 上 GEMM/attention kernel 调优首选。
2. **stock 的 num_stages 默认是坑**：小 M decode 无流水线、大 M prefill 用 0；精确 shape 微基准后再定 num_stages。
3. **decode 瘦 M (1,8) 必须单独成桶**（EAGLE verify 边界），不能并进大 M 配置；M=1/2/4/5/8 全扫 116 configs。
4. **GDN/mamba 状态池 → 并发上限公式 = 池 ÷ (EAGLE draft+1)**（max_mamba_cache_size=175÷5=35），并发由状态池（非 KV）决定。
5. **canary 同窗双指标标窗口质量**：服务器有 ±25% session 方差，跨会话绝对数不可比，A/B 必须同窗背靠背。

## 验证方法

性能 A/B：同卡同插件状态，唯一变量是 env + patch；canary pre/post 两臂一致才采信中间数据。精度：BF16 vs INT8 同 harness 2pp 门槛（GSM8K/RWQA/MathVision PASS；LCB v6 −22.5pp 为 INT8 量化版 权重代码生成的已知代价，非四杠杆所致，已如实披露）。
