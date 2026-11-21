---
name: dcu-op-optimization
description: "DCU算子优化闭环：profile→定位→改kernel→验证正确性→验证性能（方法学总纲）。触发场景：第一次接触某个算子的优化、要弄清整套闭环和每步的验收标准。不适用：已经进到算子卡里按 candidates.yaml + loop.py 跑单算子优化循环走 ops 域 op-optimize；四闸四档判定走 op-gate；PMC 读数归因走 op-profiling。"
version: 1.2.3
author: DCU Engineer
license: MIT
---

# DCU 算子优化标准闭环（dcu-op-optimization）

本 skill 是**方法学**，定义一套可重复、可审计的算子优化流程。**不绑定具体框架**（vLLM / SGLang / 自研均可套用）；不存**无溯源、会被当作固定基线复用**的性能数据（带 `[案例历史数据]` 标注的历史数字除外——那些只作方法论示例，随硬件/DTK/版本变化）。性能数据会变、会过期，本 skill 存的是"如何重新生成"而非数字本身。

## 流程路由（多轮 / 不稳定远程环境走 humanize 闭环）

> 本 skill 的 5 步闭环是**单人可手工执行**的最小流程。若任务需要**多轮迭代**、远程 GPU 环境**不稳定**（ssh 中断、排队、非静止 host）、或需要**跨轮可恢复**（算到一半能回滚重来），改用更强的闭环 harness，不必在本 skill 里手搓循环：

- **开工前立预注册实验计划** → `humanize-gen-plan`（固定 workload / 门禁 / 证据契约 / 停止条件）。
- **多轮可恢复优化闭环**（baseline → profile → patch → 同窗 A/B → 接受/回滚）→ `humanize-rlcr`（每轮 `round_N/` 带 PLAN/REPORT/manifest/results/rollback，且用 `validate_round.py` 校验）。
- 本 skill 是"**如何做 X**"的方法学（量化/定位/改/验证）；humanize 是"**如何把多轮 X 组织成可恢复、可审计的循环**"的 harness。**方法学取本 skill，循环骨架取 humanize**，两者不冲突。

> 关系：`humanize-rlcr` 的决策标签（ACCEPTED / REJECTED / INCONCLUSIVE / DIAGNOSTIC-ONLY）与本 skill 下文「四档决策」对齐；`humanize-rlcr` 的 `round_N/` 证据契约与本 skill 文末「输出物约定」同源（DCU 落地上 `profile/` 放 hipprof/rocprof 产物）。

## 触发条件（何时走本 skill）

- 某算子耗时占比异常（profiling 显示某 kernel 占整体 >30%）
- 某 shape 下算子性能不达标（带宽/算力利用率明显低于理论值）
- 想针对特定模型/量化路径做 kernel 级榨取
- 需要交付"优化前后对比"给项目/客户（标前、验收场景）

## 标准 5 步闭环（每步不可跳过）

> 原则：**先量化现状，再动代码；正确性优先于性能；性能结论必须前后同条件对比。**

### 第 1 步 · 复现 & 量化现状（baseline）
- 用 `dcu-perf-profiling` 的方法拿到该算子的：单调用耗时(ms)、调用次数、占总时间比、achieved bandwidth / peak FLOPS 占比。
- 固定输入 shape、batch、精度（FP16/BF16/FP8）、卡型号、DTK 版本——**这些条件变更任何一项，对比即无效**。
- 产出：一张 baseline 记录（见文末模板）。

### 第 2 步 · 定位瓶颈类型
用 `dcu-perf-profiling` 的 rocprof 火焰图 + 指标，判定算子属于哪类瓶颈：
- **Memory-bound**：算术强度低、带宽打满（常见于 elementwise、norm、copy）
- **Compute-bound**：SM 占用高但指令密集（常见于 GEMM、attention 部分阶段）
- **Latency-bound / 占用率低**：kernel 太小、launch overhead 占比高 → 考虑融合（fusion）
- **同步/通信瓶颈**：kernel 间 barrier、all-reduce 阻塞
不同瓶颈对应不同优化手法（见第 3 步）。
> 需要**精确定位到具体模式、决定先修哪个**时，走 `dcu-perf-profiling` §D 细粒度瓶颈诊断（六维分析 + 14 信号→成因→修复模式 + 时间线形态 + 输入分布检查），按量级排序后回到本闭环第 3 步实施。

### 第 3 步 · 实施优化（HIP kernel 改动）
内核编写/移植规范见 `dcu-kernel-dev`：
- Memory-bound → 合并访存、向量化加载、避免 bank conflict、用 LDS 做 tile 暂存
- Compute-bound → 提升占用率(occupancy)、指令级并行、用 tensor core / MFMA
- Latency-bound → kernel fusion（把多个小 kernel 融合为一个，减少 launch + 读写往返）
- 通信瓶颈 → 用 custom all-reduce / 计算通信重叠
- DCU 特有注意：ROCm/HIP 与 CUDA 的 warp→wavefront(64)、bank/conflict 差异、DTK 的 MFMA 指令集
- 若优化目标是 **vLLM 内部算子**：先读 `dcu-vllm-source` 定位该算子源码位置与注册点，确认改的是插件层还是 csrc 层。

### 第 4 步 · 验证正确性（correctness）
- **数值对齐**：与参考实现（CPU torch / 上游 CUDA 版 / 未优化版）做 diff，明确 tolerance（FP16 通常 atol/rtol ~1e-2~1e-3，FP8 更宽，按业务定）。
- 至少覆盖：典型 shape + 边界 shape（seq_len=1、batch=1、非 2 的幂）。
- 若接入了推理框架：跑框架自带 accuracy 回归（如 vLLM 的 `tests/` 或 lm-eval 小模型）确认端到端精度不掉。
- 结论必须写清："在 tolerance X 内对齐 / 不达标"。

### 第 5 步 · 验证性能（performance，必须同条件重跑）
- 回到第 1 步完全相同的环境/shape/精度，**重跑 profiling**。
- 产出对比表：算子名 | shape | 旧 ms | 新 ms | 加速比 | 正确性结论 | DTK 版本 | 卡型号。
- 结论规范：**必须给量化加速比**，禁止"感觉快了""应该更快了"。若加速比 <5% 需说明是否值得合并。

## 微基准 roofline（profiler 失效时的量化手段）

> 适用：profiler（hipprof/rocprof/torch.profiler）在 DCU 上拿不到 kernel 级耗时（trace OOM、只有 cpu_op 无 GPU kernel 事件、Engine 内嵌只导出 warmup kernel）时，用独立 micro-benchmark 量化"某 kernel 还有没有 headroom"。来源：SGLang serving 在 BW100/DTK 上做 kernel 层优化的实测方法。

### 方法（5 步）
1. **先立 copy-BW roof**：timed `tensor.copy_` 拿 HBM 峰值（BW100 单卡实测 ≈1.27 TB/s）。所有 memory-bound 声明都以它为分母（% of roof）。
2. **GEMM bench 必须打穿 L2 驻留**：轮换 6 份权重副本（~1 GB）跨计时迭代。热缓存 bench 曾把 skinny GEMM 测出"94% peak"，冷轮换一打穿就掉到 42%——这个差决定 kernel 目标是否真实。
3. **GDN/混合状态 decode kernel**：bench 真实框架 kernel（如 SGLang fla 的 `fused_recurrent_gated_delta_rule_packed_decode`），用真实 per-TP-rank 形状（v_heads×head_v×head_k fp32，r+w 流量）。实测 B≥32 时 88-90% roof → 无 headroom，这是 linear-attn decode 路径的预期状态。
4. **Triton on DCU**：非 dot 的 broadcast GEMV 能打满 roof（~1.4 TB/s，2.7× rocBLAS）但仅 M=1；M≥8 需要超峰值算力。`tl.dot` tensor-core split-K 在真实 verify 工作点（M=64-256）只有 rocBLAS 的 0.29-0.41×——**别围绕 tl.dot 规划 DCU kernel**。
5. **微基准收益必须折算成端到端占比**（如 draft-GEMV 占 TPOT ≈3%）再决定是否值得做。

### 陷阱（每个都花过真实时间）
- **lightop.gemm 首次调用后返回陈旧结果**（fresh 输入 relerr 1.0-1.6，疑 shape-keyed 缓存），曾造出假"1.68×"GEMM 收益。任何候选 kernel 的正确性门禁：与 `F.linear` 参考实现比，**每次计时调用都用 fresh 随机输入**，计时也用 fresh 输入。
- **SGLang profiler trace 在 DCU build 上**：`with_stack` trace 膨胀到 GB 级，flush 可 OOM-kill server（SIGQUIT）；`with_stack:false` 时 trace 可能只有 python_function/cpu_op 类别、无 GPU kernel 事件。别在重试上死循环，转 micro-benchmark。hipprof `--hip-trace` 可用，但 Engine 内嵌运行可能只导出 warmup kernel。
- **上游特性可移植性必须对照已装 build 审计**：承诺测试前先 grep `server_args.py` 确认 flag 存在。本 DCU 0.5.12 build：`enable_fused_qk_norm_rope`/`enable_aiter_allreduce_fusion`/`DFLASH` 有；`DSPARK`/`mamba_radix_cache_strategy`/`enable_linear_replayssm_spec` 无。上游 cookbook 的 `--chunked-prefill-size 2048`（NVIDIA hybrid GDN 推荐）在 DCU 4K workload 实测 **-5%/-4.5%**——上游调优数字不迁移。
- **端到端复测覆盖微基准**：已融合栈内的微"收益"（如 LightOp 家族消融 ±2%）可能消失；只认 quickbench 式 A/B 端到端数字（与用户纪律一致：成功 = 端到端吞吐/TTFT/TPOT）。
- **GDN prefill GEMM `num_stages=0`（无软件流水线）是 stock 通用默认，对大 M 明显非优**：stock lightop 对 M>128 用 num_stages=0，prefill（compute-bound）大 M 段留 1.14–1.99× 在桌上；实测 `num_stages=2` 全胜（K100-AI 4K c1 TTFT −12.8%、64K c1 −7.8%，TPOT 完全不变→证明 prefill-only 正确）。小 M decode 则 stock 无流水线、留 1.1–2×。
- **decode 瘦 M (1,8) 是独立调优桶**：EAGLE verify 交接边界 gate_up/qkvz 的 M≤8 在 stock 表无最优条目，离墙 1.63–2.10×；M=1/2/4/5/8 全扫 116 configs 后 `BM16/BN32/BK256/st4/w4` 跨 M 稳定胜出（+5.8% 输出吞吐、TPOT p95 −6.0%）。**瘦 M 桶必须单独扫，不能并进大 M 配置**。
- **GDN/mamba 状态池 → `max_running_requests` 换算 = 池 ÷ (EAGLE draft+1)**：max_mamba_cache_size=175 ÷ 5（draft4+1）＝ 35。GDN 是线性注意力，并发上限由**状态池（非 KV）**决定；用 `--mamba-full-memory-ratio` 抬池子（1.5 从 stock 11→35）。

### 实操配方
- **bench 脚本传进容器**：bind-mount 宿主路径 = 容器路径（`docker inspect --format '{{range .Mounts}}...'` 查）。scp 后**必验落地**（`ls -la hostfile`）——一次静默 scp 失败浪费过整个排队 benchmark run（日志看着正常）。
- **ssh 起长 benchmark**：`ssh host 'cd D && setsid nohup bash -c "cmd > log 2>&1; echo DONE > flag" >/dev/null 2>&1 </dev/null & exit 0'`——ssh 可能仍返回 exit 124（超时），但 job 通常存活。重跑前用 log 文件 + `pgrep` 验证（一次重跑造成重复队列争抢）。
- **HF 模型下载（国内受限 DCU 宿主）**：`HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 hf download <repo> --local-dir <dir>`（新 CLI 语法 `hf download`，非 `huggingface-cli download`）。不带 `HF_HUB_DISABLE_XET=1` 时 xet CAS server 经 mirror 返回 401。详见 `china-network-model-download`。

## 输出物约定（留痕模板）

```
# 算子优化记录：<算子名> / v<版本> / <DTK版本> / <卡型号>
## 环境
- 卡型号 / gfx：<从物理机 rocminfo 查，勿信容器内>
- DTK 版本：
- 框架+版本：vLLM 0.18.1 / 自研
- 精度：BF16
## Baseline（优化前）
- shape：
- 单调用 ms：  旧=   新=
- 占总时间比：
- 瓶颈类型：memory / compute / latency / comm
## 改动
- 文件：
- 手法：fusion / 向量化 / LDS / MFMA ...
## 正确性
- 参考实现：
- tolerance：
- 结论：对齐 / 不达标
## 性能
- 加速比：x
- 备注：
```

### 候选台账（candidates.jsonl）——多候选分支搜索时必留

单个算子优化通常会**并行/顺延探索多个候选**（KDA 推荐的 `candidates.jsonl`）。每确认一个候选（含被拒的）追加一行 JSONL，记录**候选名 + 父候选链接 + 状态**，让后人能从任一候选回溯它的祖先与结局：

```jsonl
{ "id": "c1", "parent": null, "status": "rejected", "reason": "bank conflict 未解，见 round_2/REPORT.md" }
{ "id": "c2", "parent": "c1", "status": "accepted", "reason": "scale-latency 后 GEMV +3.1%，过精度门禁" }
{ "id": "c3", "parent": "c1", "status": "inconclusive", "reason": "结果落在基线噪声 ±2% 内，无法判定" }
```

- `status` 用下面四档；`reason` 必须写具体（引用 metric / 归档路径），禁止"没提升"这种无法复现的话。
- 被拒 / 不明朗的候选**必须留档**（KDA 晋升规则：被拒记录原因，不要静默丢弃），否则下次会重复试错。

### 四档决策标签（KDA 修正版）

> 本 skill 原先三档 keep/revise/reject 不够，会**把"数据不干净却当成功"误报**。改用 KDA / `humanize-rlcr` 的四档，禁用"感觉快了"式模糊结论：

| 标签 | 含义 | 何时用 | 上报口径 |
|---|---|---|---|
| `ACCEPTED` | 满足任务契约 + 有证据改善/保持目标指标 | 正确性门禁过 + 同窗 A/B 超噪声门禁 | 可作正式结果上报 |
| `REJECTED` | 证据证其**劣化**或正确性不达标 | 性能回落或 diff 超 tol | 记录原因，回退 |
| `INCONCLUSIVE` | 结果落在噪声门禁内，**无法判定** | 基准噪声 ±2-3% 内、或有污染（trace OOM / CUDA Graph 污染） | **不得上报为成功**，明确"不确定" |
| `DIAGNOSTIC-ONLY` | 本轮只定位/取证，未改 kernel | 先 profile 摸清瓶颈 | 只算诊断，不进入候选晋升 |

### 预注册同窗 A/B 噪声门禁（对照 `humanize-rlcr`）

- **同窗**：control 与 candidate 在**同一时间窗、间隔交替**跑，不能先跑完 control 再跑 candidate（host 状态会漂移）。
- **预注册门禁**：开工前先写死"加速比必须超过 X%（如 >2×MAD 或 >5%）才判 ACCEPTED"，不跑完再临时定阈值。
- 置信度用 MAD（见 `dcu-kernel-dev` §十一）：metric 跑 5+ 次，`|delta| > 2×MAD` 才算真改善；更好是锁时钟（`rocm-smi --setclock` / `hy-smi`）。

## 实战案例库（真实 DCU 深度优化案例，方法论实例化）

> 来源：`<local_case_doc>.md`（海光内部真实案例集）。**本库只沉淀流程/工具/手法/坑，不存"会被当作固定基线复用"的绝对性能数字**（数字会过期）；案例中的 tok/s、加速比等一律标注 `[案例历史数据，随硬件/DTK/版本变化]`，仅作方法论示例。
> 每个案例标注它对应本 skill 5 步闭环的哪一步。

### DCU 排错 / 优化工具箱（先记这组环境变量与工具）

| 工具 / 变量 | 用途 | 出自案例 |
|---|---|---|
| `AMD_SERIALIZE_KERNEL=3` | 让 worker 栈停在 kernel 启动处而非 hipMemcpy，定位"卡死在 kernel" | 案例一 |
| `HSA_ENABLE_COREDUMP=1` + `hipprof --input-core core_0` | VMFault 时落 core dump，反解崩溃 kernel 的汇编/参数 | 案例二 |
| `ROCBLAS_LAYER=3` + `ROCBLAS_TENSILE_LIBPATH=<path>` | rocblas GEMM 调优配置注入（见案例六流程） | 案例六 |
| `HIPBLASLT_LOG_MASK=32`（抓 size）→ `HIPBLASLT_TUNING_FILE`（生成 config）→ `HIPBLASLT_TUNING_OVERRIDE_FILE`（生效） | hipBLASLt GEMM size 统计与 tuning 全链（W8A8/int8 稠密层）；点对点命中，N 须覆盖输入波动区间 | 见 `dcu-inference-optimization` §5.3 |
| `DEEPGEMM_PRINT_OP_SIZE=1`（+`_FILTER`/`_STATS`） | DeepGEMM 算子 size 打印（见 `dcu-deepgemm-source` §2） | DeepGEMM |
| `rocminfo`（物理机） | 查真实 gfx / CU 数量（容器内不可靠；**hardcode 的 gridx 必须按真实 CU 算**） | 案例一 |
| 自定义 rocblas auto_select_tools（`tuning_args.txt`） | 按平台 cu_count / hardware_type 生层 GEMM tuning 配置 | 案例六 |

**关键方法论结论**：海光插件库（lightop / flashattention / deepgemm）中存在**大量 hardcode 的 CU 数 / gridx**（如 `constexpr int gridx = 80*3`），**新卡 CU 数变化即可能卡死或越界**。改这类库第一动作：确认库里 hardcode 的 CU 假设与当前卡（BW100=64CU / K100=120CU / gfx94x）是否一致。

### 案例一：lightop `moe_marlin_w4a16` 卡死（排错定位流程）

- **对应闭环步骤**：排错（非性能优化，是正确性/可用性 bug 定位）
- **现象**：DeepSeek-R1-AWQ 在 BW100（64CU）启动卡死，**无日志、无 dmesg、无通知**（最难排的一类——静默卡死）。
- **分析流程（可复用）**：
  1. 用 `AMD_SERIALIZE_KERNEL=3` 把栈从 hipMemcpy 推到 kernel 启动处 → 确认卡在 `marlin_mm_impl`（lightop so）。
  2. 调用链反推：`benchmark → lightop.moe_marlin_w4a16 → csrc/export.cpp:813 → csrc/marlinw4a16/moe_w4a16.cu:234 → MOE_W4A16__reduce<<<sms*3,256,20000>>>`。
  3. 源码分析：`marlin.h` 里协作组按"有 240 个 block"算（gridx=240）；**BW100 只有 64CU，slice 写锁者落在不存在的 blockIdx∈[192,240) → 同组 block 永久等不到负值 → 裸自旋死锁**。
- **根因**：历史 hardcode 默认 CUs=80（gfx936），忽略 BW100 的 64CU。
- **修复手法**：按 gfx 分支 hardcode gridx（`#ifdef __gfx936__ gridx=80*3; else(K100) gridx=120*3`），长期建议改传参/环境变量。
- **源码落点**（与 `dcu-lightop-source` 交叉）：`csrc/export.cpp:813`、`csrc/marlinw4a16/moe_w4a16.cu:234`、`marlin.h`。
- **衍生建议（文档原话）**：插件库 git 分支混乱，需规范主分支/发布分支——属工程纪律，非技术。

### 案例二：FlashAttention `paged_attention_kernel` 越界崩溃（OOM 误判定位）

- **对应闭环步骤**：排错定位（KV Cache 超长上下文越界写）
- **现象**：SGLang 部署 Qwen3.5-35B-A3B，32k 上下文 + 32 并发 kernel error（BW150/BW100 复现）；16 并发正常。
- **分析流程（可复用，教科书级）**：
  1. `HSA_ENABLE_COREDUMP=1` 落 core → `hipprof --input-core core_0` 反解 fault_info，定位崩溃 kernel + PC + 参数（grid/groups/segment）。
  2. 从 core 反推崩溃的 `paged_attention_kernel<half,uint8(fp8_e5m2 KV),256,64,256,32>` 及 grid（x256,y32,z182）。
  3. **从张量大小反推规模**：out 2MB → num_heads=128；估算 `out_tmp` 需 `32*128*182*256*2 ≈ 364MB`，但源码 `constexpr temp_out_size=110000000`（110MB）**写死太小** → 越界写。
  4. 修复：改动态分配或至少把 110M 提到 600M 重编 flashattention → 压测通过。
- **方法结论**：超长上下文/KV Cache 场景，**先估算中间 buffer 上限是否超 hardcode 上限**；core dump + hipprof 反解是 DCU 上定位 kernel 崩溃的主路径。

### 案例三：GLM-5.1 Sparse MLA + MQA Logits 算子开发（新算子交付）

- **对应闭环步骤**：第 3 步（实施优化/新算子）+ 第 4/5 步（正确性+性能）
- **任务**：K100 AI（gfx928）上 GLM-5.1 Sparse Attention 缺 MQA Logits（非分页/分页）+ Sparse MLA Forward 三个算子 → 模型无法启动。
- **实现方案**：全部用 **TileLang** 实现（非手写 HIP）；新增 vLLM 文件：`vllm/v1/attention/backends/mla/sparse_mla_fwd.py`、`flashmla_sparse.py`、`model_executor/layers/mqa_logits.py` / `paged_mqa_logits.py` / `sparse_attn_indexer.py`。
- **踩坑**：TileLang lower pass 分支内误插 `__syncthreads()` 致死锁 → 合入 `release/0.1.7.post3`。
- **性能方法结论** `[案例历史数据]`：中小规模 TileLang 持平/优于 FlashMLA；**LDS 仅 64KB 时沿 head_dim 分片避免 spill**；**Online Softmax 省中间显存**；**num_split=4 在 120CU 达最优 TFLOPS**（继续增因归约开销反降）——这是"compute-bound 调 occupancy"的具体实例。
- **端到端** `[案例历史数据]`：从 OOM 不可用 → 功能打通 ~2 tok/s → TileLang 优化 ~5 tok/s（2.5x）→ MTP 生产使能 ~16 tok/s（3.2x）。
- **与算子库 skill 交叉**：MQA Logits 在 `dcu-aiter-source`/`dcu-deepgemm-source` 也有实现，属同类算子多库实现（选型见各自 §5）。

### 案例四：Qwen3-32B 融合 QK-Norm+RoPE+FP8 KV-store（kernel fusion 实例）

- **对应闭环步骤**：第 3 步（Latency-bound → fusion）
- **背景**：BW1000 上 Qwen3-32B Dense 模型，单并发/低并发场景；KV-cache=fp8-e5m2 时多个分散算子可融合。
- **手法**：把 `qk-norm(2 kernels) + rope(1) + index_put×2 + float8_copy×2`（~45-56us）**融合为一个 Triton kernel**（`_fused_qknorm_rope_kvstore_kernel`），按 `T*(QH+2*KVH)` 一个 program 一个 head，HALF=D/2 避免 neox rope 的 lane rotation。
- **接入方式**：`sglang_plugin.py` monkey-patch `Qwen3Attention.forward`，由环境变量 `SGLANG_USE_FUSED_QK_ROPE_KV=1` 开关。
- **性能方法结论** `[案例历史数据]`：TPOT/TTFT 均下降、每 step 微秒级省时、吞吐提升（可量化收益）。**这是"latency-bound 小 kernel 融合"的标准范式**（见第 2 步瓶颈判定）。

### 案例五：MoE 模型 `fused_moe_kernel` 占比优化（profile → tuning）

- **对应闭环步骤**：第 1 步（profile 量化现状）+ 第 3 步（MoE tuning）
- **手法**：先 profile 确认 `fused_moe_kernel` 占 TTFT 窗口 ~60%（每 linear layer 两次调用，36 linear + 4 full）；该 kernel 用 Triton 默认配置 → 写工具针对平台（BW1100）生成 tuned 配置，放入 vLLM/SGLang layer 下重测，明显提升。
- **工具产出**：per-batch breakdown（top3 kernel 表）+ TTFT 窗口 Top GPU kernels 表——**这是第 1 步 baseline 记录的直接范例**。
- **方法结论**：MoE 模型 GEMM 占比极高时，平台专属 tuning 配置（非默认 Triton autotune）是低投入高回报项。
- **与算子库交叉**：`fused_moe_kernel` 实现见 `dcu-aiter-moe` / `dcu-lightop-moe` / `dcu-deepgemm-source`。

### 案例六：rocblas GEMM Tuning（Dense 模型通用加速）

- **对应闭环步骤**：第 3 步（compute-bound GEMM → 库级 tuning，非改 kernel）
- **适用**：Dense 模型中 `Cijk_MT64x16x128` / `Cijk_MT32x32x4` 等 rocblas GEMM 占 PPL 较多时。
- **标准流程（可复用，逐步）**：
  1. `export ROCBLAS_LAYER=3` 起服务跑压测（多 batch），`cat infer.log | grep "./rocblas-bench -f gemm" | sort -n | uniq -c > rocblas_size.log`。
  2. 放到 `/opt/dtk/lib/rocblas/rocblas_layer3_size`，改 `/opt/dtk/lib/rocblas/auto_select_tools/configs/tuning_args.txt`（`hardware_type=BW`、`cu_count=<真实CU>`、`source=csv`、策略 common）。
  3. `python3 Auto_tuning.py 2>&1 | tee tuning.log`（耗时长）。
  4. 起服务加 `export ROCBLAS_TENSILE_LIBPATH=<new config path>`。
- **性能方法结论** `[案例历史数据]`：TTFT +1~3%、TPOT 相应提升。属**库级 tuning，零代码改动**，Dense 模型优先尝试。
- **注意**：`cu_count` 必须填真实 CU（见工具箱结论）；`should_clean=1` 前先备份原 config。
- **更多口径**（2026-10-16 官方教程补全）：layer3 独立日志抓取（`ROCBLAS_LAYER=3` + `ROCBLAS_LOG_BENCH_PATH=bench.log`）、抓取前置坑（先关 `SGLANG_DISAGG_HOST_STAGING_BUFFER`；开 cuda graph 时压测不写 bench.log 属正常）、tuning_args.txt 完整参数表（含 mode/chunk_size/tune·bench 次数/enable_bestkernel_csv/enable_mk_n_threshold，hardware_type 含 YY）、DSv4-0731-FP8 长上下文单并发实测（吞吐 +78~84%）→ 全部见 `dcu-inference-optimization` §5.2。

### 案例七：NVFP4 块缩放 GEMM 的 Colfax 七轮迭代法（GEMM bank conflict 系统化排查）

- **对应闭环步骤**：第 1–5 步全覆盖（量化 → 定位 → 迭代 → 验证），是标准 5 步闭环的**完整实例**
- **适用**：GEMM / Attention 类算子出现"带宽/算力利用率低但代码看不出问题"的隐蔽瓶颈
- **背景**：Colfax Research 在 RTX Pro 6000（Blackwell SM120）上优化 NVFP4 块缩放 GEMM，七轮迭代从 v1（swizzling 修复 L2）追到 v7（cuBLAS 持平），过程中依次拆解了 bank conflict、epilogue bug、warp 专用化等子问题
- **核心方法论（七轮迭代闭环）**：
  - **v1 swizzling**：问题形状下 DRAM 带宽 64–86%（非扩展性问题）→ 线程块重排改善 L2 缓存命中率
  - **v2 L2**：量化确认问题在 L2，未扩散到 L1 或 register
  - **v3 epilogue**：producer_acquire 过早 → consumer 读到旧值；producer_tail 位置错误 → 部分 tile 未写入。两者修复后 epilogue 正确
  - **v4 MMA warp**：引入专用 MMA warp（计算）和 STORE warp（数据搬运），barrier_id 2/3 协调两类 warp 的流水线
  - **v5 bank conflict（SFA）**：Nsight 测 `L1 Wavefronts Shared Excessive=8.39M` → SFA SMEM 布局导致行 r 与 r+8 同 bank → 改 stride 从 `(32,4):(4,128)` 改为 `(4,128)` → 每行独立 bank → 冲突消除
  - **v6 12路 MMA**：cuBLAS 的 block scaling 因子在 MMA 前要正确 unpack → 修复后 12路 MMA 全部对准
  - **v7 自动调优**：参数搜索空间（CTA tile / warp 分工 / MMA 配置），最终收敛到 cuBLAS 基准水平
- **方法结论**：bank conflict 是**隐形瓶颈**，只有量化成 Nsight/`hipprof` 指标才能拆解；改动 stride 布局（不改数据语义）可消除冲突而不影响正确性
- **与 DCU 适配**：DCU LDS 同样 32 bank / 4B，hipprof 对应指标可查 LDS stall；warp 专用化思路对应 DCU named barrier（`__builtin_amdgcn_sched_barrier`），可移植；TMA 在 DCU 无对应物，忽略
- **溯源**：https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/

### 案例八：NVFP4 Blackwell GEMM 的八轮微优化阶梯 + L2 双缓存调度（跨架构方法论迁移）

- **对应闭环步骤**：第 1–5 步全覆盖（量化每步 → 定位 → 微优化 → 验证），是标准 5 步闭环的**第二份完整实例**（与案例七 Colfax 互补）
- **背景**：cudaforfun 在 GB300（Blackwell SM100）从零手写 NVFP4 块缩放 GEMM，**纯 CUDA + 内联 PTX、零框架依赖、内核 100% 由 Claude 生成**，最终比 cuBLASLt 快 4.7%。整篇是一张「把每个硬件细节量化成 PFLOP/s」的优化阶梯。
- **核心方法论（与 dcu-kernel-dev §十一 铁律同构）**：每一步改动都打印 `METRIC`（PFLOP/s）行，只认量化 delta 推进；正确性门禁不过不谈性能。
- **优化阶梯（GB300 实测相对提升，标注为 NVIDIA 平台数据，**非 DCU 基线**）**：
  - **优化1 直接读 `%laneid` 特殊寄存器**：用 `asm("mov.u32 %0, %laneid")` 代替 `threadIdx.x % 32 == 0` 选 warp 内单 lane。两者数学等价，但前者让编译器证明控制流为 warp-uniform、走 uniform 数据通路，少 135 条 R2UR（register→uniform-register）指令、SASS 短 848 行 → **+77.5%（3.27→5.81 PFLOP/s）**。→ **DCU 适配**：DCU 有等价特殊寄存器 `s_laneid`（`__builtin_amdgcn_lane_id` / `%laneid`），凡 warp-uniform 的判定（SMEM/TMEM 描述符、barrier 地址、循环状态、lane0 选择）应直接读特殊寄存器，避免从 per-thread 的 `threadIdx` 推 uniform 判定，让编译器省去等价 R2UR 指令。**DCU 上是否同样显著待真机验证**。
  - **优化2 K=256 tile + 手动展开固定 K + 跨窗口 MMA 描述符**：把 AB tile 从 128×192 扩到 128×256 打满 TMA 效率；固定 K=768 全手动展开，先发 cp 再发 mma 打破串行；跨 128B 窗口的 MMA 用绝对地址描述符模式（`FIXED_K96_SW128`）。→ **5.81→6.68（+15%）**。DCU 对应：tile 对齐 MFMA shape、固定 K 手动展开、LDS 环形缓冲的 barrier 顺序手排（dcu-kernel-dev §15.2、§13）。
  - **共享内存核算**：SMEM 占用 99.1%（230.4/232.4 KB）；减一个 A/B stage 损 12%、减一个 scale stage 损 1.2% → **量化 stage 数对性能的影响**是必要步骤。DCU 对应：LDS 占用率同样要量化（dcu-kernel-dev §15.5 ISA 自检清单）。
  - **优化3 双累加器缓冲重叠（延迟同步而非移除同步）**：2×256 累加列 + 36 scale 列 = 548 > TMEM 512，把两个输出 tile 重叠 36 列，MMA 只需等 consumer 排空重叠边缘而非整块 → **6.68→6.82（+2.2%）**。DCU 对应：无 TMEM 等价物，但「双缓冲输出 tile 用列重叠隐藏 drain 延迟」是 `num_stages=2` 双缓冲的进阶思路（dcu-kernel-dev §15.2）。
  - **优化4 256-bit 宽向量化存储 `st.global.v8.b32`**：整 8 个 b32 一次写出 → **6.82→7.33（+7.5%）**，达 cuBLAS 100.3%。DCU 对应：宽向量化 store（`global_store_dwordx4` 或更宽），对齐 16B/128B（dcu-kernel-dev §14.2、§15.5）。
  - **优化5 缓存提示 `L1::no_allocate.L2::evict_first`**：输出只写一次、不再读，禁 L1 分配避免与输入流水线抢 L1，L2 标记 evict-first 让可复用输入占更多 L2。→ **7.33→7.42（+1.2%）**。DCU 对应：ROCm/DCU 有 GL2 缓存策略（`buffer_store` 的 `glc`/`dlc` 位、L2 驱逐策略），写一次的输出用 no-allocate + evict-first 保护输入 L2 驻留，**具体 intrinsic 待验证**。
  - **优化6 编译器线程数提示 `__block_size__(224)`（reqntid）**：给编译器精确线程数，去掉收敛保护 BSSY/BSYNC，VGPR 62→52。→ **7.42→7.44（+0.3%）**。DCU 对应：`__launch_bounds__`（dcu-kernel-dev §14.1）+ 给编译器精确线程数避免收敛保护。
  - **优化7 跳过无效尾部**：K=8192 用 K=768 主循环 + 显式 tail group，算 K=8256 而非 padding 出来的 K=8448（不跑被零填充的 MMA）。→ **7.44→7.49（+0.7%）**。
  - **优化8 K64 折叠尾部**：尾部混合 K96 + K64 指令精确凑到 8192（10×768 + 4×96 + 2×64），而非 6×96=8256。→ **7.49→7.56（+0.9%）**。
- **硬件感知调度（L2 实为两块分区缓存，本文最大方法论贡献）**：
  - GB300 双 die，各管约半 L2；物理地址由 hash 定 home side，SM 由所在 die 定 near side，二者可 mismatch，跨 die 链往返慢 ~2.3x（148ns vs 344ns）。
  - 分割比例不固定（76/76、72/80、74/78…），**每卡不同**，有效 L2 大小也不同。
  - **优化9 8×8 pockets + L2 ownership**：把每个 M 行（A 复用维度）分配给一个 requester side，B/SFB 共享；route table 每 tile 一个条目仅几 KB，每 SM 查询算哪个 tile。→ **7.57→7.65（+~1%）**。
  - **DCU 适配（advanced/experimental）**：AMD CDNA 多 die（gfx94x）有**完全相同的「每 die 半 GL2 + Infinity Fabric 跨 die 延迟」特征**（见 `dcu-topology`）；kernel 内按 die/SA 分组 tile、使 A 复用留在 near side 是可行方向，但 DCU 没有 NVIDIA 现成地址 hash 公式，需自己 microbenchmark 定 home side（类似 Aroun/QuickRunCUDA 的做法）。**DCU 上收益与实现待真机验证**。
- **基准测试纪律（强化 dcu-kernel-dev §十一 / 第 5 步）**：
  - 轮换输入：每次启动用不同 A/B 副本，kernel 看不到自己之前的 L2 状态（避免 self-warm 虚高）。
  - 多轮交替：与 cuBLASLt 交替 5 轮、轮换先后，每轮连续 25 启动取均值。
  - 冷却：每定时 arm 至少 8s 冷却，轮询直到空闲时钟恢复初始。
  - 锁时钟更稳：DCU 对应 `rocm-smi --setclock` / `hy-smi`。
  - **关键警示**：文章同内核用 4 种测法给出对 cuBLAS 不同优势（1.039x ~ 1.078x）——**基准协议本身影响 apparent 加速比**。DCU 上必须同协议同条件对比，禁止跨协议比（与 dcu-kernel-dev §十一「瓶颈声明只能源于本次 profiler 报告」同构）。
- **溯源**：https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4 （微信转载《在NVFP4上超越cuBLAS: 从零手写+Claude极限优化Blackwell GEMM》）

### 案例九：量化 LLM 单并发 decode 的 SIMT GEMV 优化（ALU-bound 诊断 + 手法/反例全集）

- **对应闭环步骤**：第 1–5 步全覆盖（量化 decode 场景的完整实例，与案例七/八的 GEMM 场景互补）
- **背景**：闭源 C++/CUDA 推理引擎（NInfer）移植到 DCU（gfx936，DTK 26.04）后，对 Qwen3.8-27B（Q4/Q5 混合量化）单并发 decode 的系统优化。MFMA 被 DTK 堵死，全 SIMT 路线。NInfer 专属细节（4 层量化转换/build 坑）见 `dcu-ninfer-port`，本案例只沉淀**跨项目可迁移**的量化 decode 优化经验。
- **★第一铁律：量化 decode 的 GEMV（T=1）是 ALU-bound（dequant 计算），不是 memory-bound（带宽）**——选错优化方向白费功夫。
  - **诊断方法**（profile 驱动，禁止读代码猜）：① 关 CUDA Graph profile（`hipprof --hip-trace --output-type 0 <app> --no-cuda-graph`）；② 分解 GEMV（T=1）/GEMM（T>1）/其他——单并发 decode 几乎 100% 是 GEMV；③ 算 achieved bandwidth = 读写字节/耗时，对比峰值（BW100 D2D ~1269 GB/s read+write，有效 ~634）；④ achieved_bw 远低于峰值（<40%）+ 指令密集 → ALU-bound，接近峰值 → memory-bound。
  - **本次结论** `[案例历史数据]`：Q5 GEMV（down_proj 55MB）memory time ≈ 0.044ms，实测 0.177ms → compute 是 memory 的 4 倍 → ALU-bound；有效带宽只到峰值 ~30%（70% 余量没吃满）。
  - **优化方向**：减少 dequant 指令（scale 延迟/sign_extend/向量化），**不是**减少显存流量（那对 memory-bound 才有效）。
- **GEMM vs GEMV 的 regime 差异（关键教训）**：GEMV（T=1）ALU-bound、每权重读 1 次用 1 次、优化=减少 dequant 指令；GEMM（T>1）memory-bound、每权重读 1 次用 T 次、优化=减少显存流量/提升带宽利用。**同一手法（如 fp16 trick）在 GEMM 有效、GEMV 更慢——必须按 regime 分别验证，不能跨 regime 套用**。
- **已验证有效的手法（按收益排序，可迁移到任何 group-quantized GEMV）**：
  1. **★scale 延迟（per-group scale 只乘 1 次）**——收益最大、最通用。group 内 64 值共享 1 scale，naive 每值乘 scale；延迟到 group 级：`p=q0*x0+q1*x1+...; acc+=p*s`，每 (lane,group) 省 (group_size-1) 次乘法。ALU-bound 收益明显、memory-bound 被掩盖（0 收益）。实测 `[案例历史数据]`：swiglu GEMV +1.7%、Q5 GEMV +3.1%、Q5 GEMM 0%。注意：改变累加顺序 → ~1 ULP 精度差，需过精度门禁（见 `dcu-kernel-dev` §8.5）。
  2. **sign_extend 用算术移位**：4/5-bit 打包符号扩展用算术右移替代 bit_extract+mask，零风险零精度影响，顺手做。
  3. **ACC8 多累加器**：8 个独立累加器交错累加隐藏 FMA 依赖链（ILP）；注意寄存器压力可能降 occupancy。
  4. **向量化加载**：int8 用 `uint4`（16B/次）、bf16 激活用 `uint4`（8 个 bf16），打满 128-bit。
- **★反例与失败教训（比成功更值钱，其他项目遇到直接跳过）**：
  1. **fp16 mantissa trick：GEMM 有效、GEMV 更慢**（regime 依赖）。int→float 用 fp16 位操作替代 `v_cvt_f32_i32`：GEMM（memory-bound）有效；GEMV（ALU-bound）更慢 -6%（gfx936 fp16 ALU 比整数 ALU 慢）→ 已回退。
  2. **int8/W8G32：精度更高但性能更慢**（精度-性能权衡）。Q5（0.625 字节/值）→ int8（1 字节/值）精度更高但显存流量 1.6x。独立 benchmark int8 u32 向量化比 Q5 GEMV 快 1.78x，但**集成后 W8 整体比 Q5 慢 11%**（带宽惩罚主导）。**教训**：量化位宽不是越高越好，ALU-bound decode 上带宽惩罚可能主导；独立 benchmark 加速比 ≠ 集成后加速比。W8 kernel 本身已高度优化（uint4 向量化+2-stage 流水线+无 bank conflict），慢是格式本身的显存流量代价。
  3. **kStages=2 双缓冲：GEMV 上更慢**（occupancy 损失）。GEMM 有效；GEMV 更慢（2-stage 占 shared memory 降 occupancy，而 GEMV 已 100% occupancy）→ kStages=1 最优。**双缓冲不是万能的——occupancy 已 100% 时加 stage 只占资源不提速**。
  4. **CUDA Graph：仅 +1.8%**（launch 空隙非瓶颈）。kernel-bound 的 decode 100% 在 kernel 本身，别指望 CUDA Graph 救 decode 吞吐。
  5. **v_pk_fma_f32 手写 packed FMA：被编译器堵死**。DTK 无 intrinsic，clang 17 内联汇编不支持 64 位 `l`/`d` 约束；但 dump 汇编发现**编译器已自动生成 `v_pk_fma_f32`**（~4.9% 指令）→ 不用手写。**教训**：先 `dccobjdump --show-sass` 看编译器已生成什么，别假设没做就手写。
- **★诊断坑（profile 工具在 DCU 上的坑，会误导瓶颈判断）**：
  1. **hipprof kernel 计时被 CUDA Graph 污染**：graph 下 kernel 时间 > wall 时间（如 61.6ms > 48ms）→ 不可信，**profile 必须关 CUDA Graph**。
  2. **PMC 计数器某些环境拿不到**：`hipprof --pmc` 输出空 → 用 kernel 时间分解 + achieved bandwidth 估算替代，别死等 PMC。
  3. **profile 解析截断 kernel 名**：`[-45:]` 截断 → 真正 #1 kernel 名字被切掉误判 → 别截断，用完整名+调用次数+总时间排序。
  4. **GEMM vs GEMV 调用次数混淆**：GEMM 高调用次数（64-128）是 prefill（每层 1 次×层数），decode 几乎 100% 是 GEMV → 按 T 区分，别只看 kernel 名/调用次数。
- **量化格式选型（精度 vs 性能权衡）**：ALU-bound decode 上位宽越高 → 显存流量越大 → 越慢（int8 比 Q5 慢 11% `[案例历史数据]`）。单并发 decode 优先 Q4/Q5（流量小+dequant 够快）；多并发/prefill 可上 W8（精度优先，带宽被 T 复用摊薄）。
- **occupancy 到头就别再调**：本次 GEMV 49 寄存器、无 spill、4 blocks/SM = gfx936 上限（100% occupancy）→ occupancy/kStages/FMA packed/scale 延迟都已到头，Q5+SIMT 单并发 decode 已接近极限，进一步提升需换格式（int8 带宽惩罚）或换路线（MFMA，DTK 26.04 堵死）。

### 案例对我们的启示（沉淀）

1. **静默卡死类 bug 第一动作**：确认插件库里 hardcode 的 CU/gridx 假设，新卡 CU 数变化即爆（案例一）。`rocminfo` 查真实 CU 优先于一切。
2. **kernel 崩溃定位主路径**：`HSA_ENABLE_COREDUMP=1` + `hipprof --input-core` 反解，比盲改快（案例二）。
3. **融合范式**：latency-bound 的小 kernel（norm/rope/copy）融合一个 Triton kernel 是通用手法（案例四）。
4. **平台 tuning 优先于改 kernel**：MoE/GEMM 占比高时，先试 Triton 平台配置 / rocblas tuning（案例五/六），零代码改动拿收益。
5. **新算子用 TileLang 而非手写 HIP**：开发更快，但注意 TileLang 自身 lower bug（案例三）。
6. **bank conflict 是隐形瓶颈**：必须量化成 `hipprof`/`rocprof` 指标才能拆解，改 stride 布局（不改数据语义）可消除（案例七 Colfax 溯源）。
7. **七轮迭代法**：复杂 GEMM 优化需系统化迭代（v1 swizzling → v2 L2 量化 → v3 epilogue → v4 warp 专用 → v5 bank → v6 MMA → v7 autotune），每轮必须有 `METRIC` 证据才推进（案例七）。
8. **数字不进 skill**：以上加速比均为案例历史数据；本 skill 只存流程+手法，数字在各项目用文末模板留痕。
9. **跨架构 micro-opt 手法可迁移（案例八 cudaforfun）**：特殊寄存器做 warp-uniform 判定（laneid 技巧，单点 +77.5%）、宽向量化 store、缓存策略提示（保护输入 L2 驻留）、双缓冲重叠隐藏 drain、尾部精确化（不在 padding 上跑 MMA）、精确线程数提示去收敛保护——这些与 NV 架构无关的手法对 DCU kernel 同样适用，但**绝对数字都是 NVIDIA 平台数据，DCU 收益以真机 `METRIC` 行为准**。
10. **L2 分区调度（案例八）**：多 die GPU 的 L2 是各 die 半区、跨 die 链有 ~2.3x 延迟，按 die/SA 分组 tile 使可复用输入留在 near side 是真实收益方向；DCU 多 die（gfx94x）同理，但需自己 microbenchmark 定 home side，**DCU 待验证**。
11. **基准协议影响 apparent 加速比（案例八）**：同内核 4 种测法得出 1.039x~1.078x 不同优势，再次印证「同协议同条件对比」是不可动摇的铁律（与 dcu-kernel-dev §十一 同构）。
12. **量化 decode 先判 ALU-bound 再动手（案例九）**：量化 GEMV（T=1）瓶颈是 dequant 计算不是带宽，优化方向是减 dequant 指令（scale 延迟最通用）；同一手法必须按 GEMV/GEMM regime 分别验证（fp16 trick 在 GEMM 有效、GEMV 更慢）；量化位宽不是越高越好（int8 精度更高但 1.6x 显存流量，ALU-bound decode 上反而更慢）；profile 必须关 CUDA Graph（kernel 计时被污染）。
13. **CT 反投影（back-projection）类 kernel 的三条通用手法**（CT 反投影 kernel 实战）：① 用满 wavefront（Wave64，DCU wavefront=64 线程，tiling 按 64 对齐避免半空 wave）② 通道规约前置（把 channel 维 reduce 提前到主循环外/前级，减少主循环内规约开销）③ 跨圈融合（相邻扫描圈/相邻投影角的计算融合，减少 kernel launch 与中间张量往返）。适用几何投影/扫描重建类 kernel。
14. **给核函数加本地变量缓存助编译器分配寄存器**（某科学计算软件实战）：对反复读同一内存位置的循环，引入本地变量缓存（loop-invariant 提到循环外 / 累加器用寄存器变量），帮助编译器更好地分配寄存器、减少内存访问。属"保证结果正确前提下的简单优化"，改动小、风险低，适合先试。

## 与兄弟 skill 的关系
- `dcu-perf-profiling`：第 1、2、5 步的度量手段（本 skill 不重复定义指标）。
- `dcu-vllm-source`：当优化目标是 vLLM 内部算子时，定位源码与注册点。
- `dcu-aiter-source` / `dcu-aiter-moe` / `dcu-lightop-source` / `dcu-lightop-moe` / `dcu-deepgemm-source`：案例涉及的真实算子源码落点（MoE GEMM / MQA Logits / marlin）交叉引用——改/查某算子先去对应 skill 定位。
- `dcu-ontrack-experience`：案例一（lightop 卡死）、案例二（FlashAttention 越界）已属典型故障，已在 ontrack 速查索引指向本案例库。
- `dcu-vllm-deploy`：优化完要部署验证时走它。
- `dcu-ninfer-port`：案例九（量化 decode 优化）的 NInfer 项目专属细节（4 层量化格式转换路径、build/link 坑、转换脚本）落在此 skill；通用优化经验在本 skill 案例九。
- `dcu-kernel-dev`（规划中）：HIP kernel 开发/移植规范细节。

## Torch Compile / Inductor 在 DCU 上谨慎使用（参考 CUDA 平台经验，DCU 适配性未验证）

> **来源与标注**：以下结论主要来自 SGLang 核心贡献者 BBuf 在 CUDA 平台（H100/H200/B200/5090）对 Diffusion 模型的系统性评测（公众号技术文 + GitHub 实测表格 how-to-optim-algorithm-in-cuda/issues/21，35 个模型全量对比 Eager vs Torch Compile）。**这些是 CUDA 平台观察，DCU/ROCm 上是否复现未经我们实测，仅作"参考、可尝试"用，禁止当作 DCU 已验证纪律硬编码。**

### 已知事实（ROCm 通用，非 DCU 专属断言）
- ROCm 栈对 `torch.compile` / Inductor 的**后端支持成熟度显著弱于 CUDA**，编译更久、更易图 break，这是 ROCm 工程现实，不依赖上面那篇评测。
- 因此在 DCU 上做推理/Diffusion 优化，**默认优先 `eager` + 手写/自动 fuse kernel + (Breakable) HIP Graph**，而非 `torch.compile`，是低风险默认项。

### 参考 CUDA 平台经验（BBuf 观察，DCU 待验证）
Torch Compile 在 CUDA 上被指出有如下痛点，**是否同样出现在 DCU 上需你实测**：
1. 编译时间随模型复杂度爆炸（复杂 Diffusion 首编半小时以上，部分 3 分钟编不完）；
2. 跨 GPU 性能跷跷板（同模型 H200 提升、B200/5090 反而明显下降）；
3. 手写 fuse kernel 在 eager 下提升、开 Torch Compile 后 end2end 反降（图被 break）；
4. `default` / `max-autotune` 等 mode 在不同模型上谁快无规律；
5. perfetto 时间线透明性低，难定位 kernel↔Python 调用，二次修改/评测困难。

### DCU 上的推荐动作（参考级，待你验证）
- **替代路线**：用 **HIP Graph**（`hipGraph`，CUDA Graph 的 ROCm 对应物）做图捕获/重播，平替"用 Torch Compile 实现 PieceWise CUDA Graph"——SGLang 的 **Breakable CUDA Graph (BCG)** 已证明在开销大的模型上可超越 Torch Compile，且更透明易调。DCU 上对应即 **hipGraph 捕获 decode/固定 shape 路径**（见 `dcu-sglang-source` 的 `capture_decode_cuda_graph` 适配点）。
- **验证纪律**：若要在 DCU 上评估 Torch Compile，必须**多机型对比**（<device-A> BW150 / <device-B> BW100 / BW1000 / BW1100），单卡结论不可靠——正是文章强调的"跨设备跷跷板"在 DCU 上更要防。
- **可借鉴的方法论**（不限于是否弃用 Torch Compile）：作者用 **Agent（Nvidia Kernel Design Agents 的 Fable + 简单 prompt flow + 2 个 skill）2 周把 35 个模型在 Eager 下性能全部持平/超越 Torch Compile**——即 Torch Compile 的 fuse 优化可被 Agent 自动逆向复刻成手写 eager kernel。这套 **Agent 驱动 kernel 优化 flow** 已沉淀进 `dcu-kernel-dev` §九（DCU 适配参考）。

## 使用须知
- 经验仅供参考，不保证修复。动代码前先核对：**设备型号 + 驱动/DTK 版本 + 报错原文**。
- 优化涉及**重编译框架 / 改 csrc**：必须在**指定镜像起的专用容器**内做；容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见（DEVCOUNT=0）。
- **不存"会被当作固定基线复用"的性能数字**：本 skill 只沉淀方法，无溯源的历史数字一律标注 `[案例历史数据]` 并注明随硬件/DTK/版本变化；需要复用的数据用上面的模板在各自项目里留痕。
- 高危操作（刷 vbios / 改 BIOS / 重装驱动）谨慎，确认环境（产线/客户）与带外恢复通道，优先测试机验证。
