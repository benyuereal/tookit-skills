# 性能优化案例 §17-§24（vLLM 推理优化案例集，源自 vllm-book-poc-scheduler，已脱敏）

> 2026-10-16 自 `perf/dcu-perf-cases/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），案例四段式原文照录、未改一字；编号与原文一致，索引见主 SKILL.md 案例索引表。

## 17-37. vLLM 推理优化案例集（源自 vllm-book-poc-scheduler 文档库，已脱敏）

> 来源：`vllm-book-poc-scheduler` 实战案例库（vLLM 0.26.x / 海光 DCU 实测）。已脱敏（客户名→泛称）、剥离数据图。
> 原理层（vLLM 上游机制）见 `dcu-vllm-source/references/upstream-vllm-026/`；症状速查见 `dcu-vllm-deploy/references/perf_symptom_lookup.md`。
> 注：gemma-4-31B AWQ 两阶段优化 = 本库案例 13；Qwen3.5-2B 多模态 prefill 排队 = 本库案例 6（不重复收录）。

## 17. ASR 端到端性能优化（CPU 前端 vs GPU 正交轴）

> 场景：电话客服实时转写，70 路并发、单条 10s 音频，要求端到端 < 500ms
> 模型：Qwen3-ASR-1.7B（音频编码器 + 28 层 thinker LLM，mrope）
> 硬件：海光 DCU BW150（gfx936），单卡 tp=1，bf16 无损
> 软件：vLLM 0.18.1+das.dtk2604（海光定制版）

## 现象

vLLM 原生部署，70 并发 10s 音频端到端 **1012ms**，距 500ms 目标差一倍。

## 定位

对基线做延迟分解，定位到三块问题，分属 CPU 前端与 GPU 推理：

| 问题 | 位置 | 现象 | 占基线延迟 |
|------|------|------|-----------|
| 音频解码阻塞 event loop | CPU 前端 | `librosa.load` 在 async 方法里同步调用，卡死单线程 event loop，70 并发解码串行排队，每个阻塞 ~89ms | ~9% |
| multipart 解析 GIL 串行 | CPU 前端 | `python-multipart` 纯 Python、GIL-bound，70 请求抢同一把 GIL 串行执行，解析 ~302ms | ~30% |
| GEMM 权重布局次优 | GPU 推理 | decode 每步 112 个 skinny GEMM（M=70≪K/N），hipBLAS 在默认 (N,K) 布局下选到次优 kernel，带宽利用率仅 72% | ~44% |

**两个关键认知：**
- "上传 302ms" 不是网络问题。loopback 传输实测 0.7ms，302ms 几乎全是 multipart 解析的 CPU 时间。
- GPU decode 是硬地板。28 层每步读权重 ~3.4GB，gfx936 实测 HBM 带宽 1.35TB/s（仅 MI300X 的 1/4），带宽受限。CUDA graph 已把 launch 开销压到 ~0.05ms/step（占 1%），launch 不是瓶颈。

**决定性线索（vDCU 多实例实验）：** 把单卡切成多个 vDCU 实例、每个实例各起一个 vLLM 服务（前置 nginx 分流），70 并发端到端延迟显著下降。这说明瓶颈不在 GPU（切卡后单实例算力更弱，但延迟反而降了），而在 CPU 前端的单进程串行。

## 方案

1. **音频解码异步化**：`librosa.load` 包进 `asyncio.to_thread()`，丢到线程池执行（底层 soundfile C 库释放 GIL，线程池能真并行）。
2. **多 API server 进程并行（核心）**：vLLM 原生参数 `--api-server-count N`，启动 N 个 API server 进程，共享同一个 EngineCore（单卡单引擎），通过 ZMQ 通信，同一端口 SO_REUSEPORT（内核负载均衡）。N 个进程 = N 个独立 GIL + N 个 event loop → multipart 解析真正并行（绕过 GIL）。
3. **GPU 权重转置**：`VLLM_ROCM_TRANSPOSE_WEIGHT=1`，加载权重时把 (N,K) 转置成 (K,N) 连续存储，apply 路径改用 `torch.mm`/`addmm`。数学完全等价，仅 bf16 累加顺序不同（max diff 0.5，不影响 argmax）。

**多进程饱和点（c=70，10s 音频，p50）：**

| 进程数 | upload(CPU) | server | decode(GPU) | total | 边际收益 |
|--------|-------------|--------|-------------|-------|---------|
| 1 | 302ms | 61ms | 366ms | 764ms | — |
| 2 | 238ms | 57ms | 310ms | 595ms | -169ms |
| 3 | 135ms | 57ms | 314ms | 513ms | -82ms |
| **4** | **50ms** | **52ms** | **328ms** | **440ms** | **-73ms ← 甜点** |
| 6 | 37ms | 64ms | 331ms | 435ms | -5ms ← 饱和 |

CPU 前端工作量固定，加进程边际递减。4 进程是甜点：upload 从占总延迟 43% 降到 12%，瓶颈转移到 GPU。

## 效果

| 阶段 | 端到端延迟 | 本步收益 | 优化轴 | 手段 |
|------|-----------|---------|--------|------|
| 原始基线 | 1012ms | — | — | vLLM 默认部署 |
| 音频解码异步化 | 764ms | -248ms | CPU 前端 | `librosa.load` → `asyncio.to_thread` |
| GPU 权重转置 | ~710ms | -54ms | GPU 推理 | `VLLM_ROCM_TRANSPOSE_WEIGHT=1` |
| 多 API server 进程并行 | **419ms** | -291ms | CPU 前端 | `--api-server-count 4` |
| **合计** | **419ms** | **-593ms（-59%）** | | 无损，逐字一致 |

端到端 p50 = 419ms < 500ms ✅ 达标，max = 476ms（最慢请求也达标）。70 并发请求输出逐字一致（temperature=0 确定性）。

## 可复用经验

1. **分清 GIL-bound 与 release-GIL 的 CPU 工作**：`librosa.load`（C 库）释放 GIL → 线程池可并行；`python-multipart`（纯 Python）不释放 GIL → 必须多进程。
2. **多实例对比实验能快速锁定瓶颈域**：当不确定瓶颈在 CPU 还是 GPU 时，做一个"增加并行度但削弱单路算力"的对比实验，比纯打点分析更快。
3. **多进程并行有饱和点**：找到"再加一档边际收益趋零"的点停下，别靠堆进程掩盖未定位的瓶颈。
4. **CPU 前端和 GPU 推理是正交两条轴**：独立可叠加。优化完 CPU 后瓶颈自然转移到 GPU，再从 GPU 侧打。
5. **原生参数优先于源码改动**：`--api-server-count` 零源码改动、可维护、升级友好。

## 关联文档

- [vllm_engine_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_engine_guide.md) — 进程模型、API Server / EngineCore / Worker 三进程架构、ZMQ 通信
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — 请求全生命周期、TTFT/TPOT 指标
- [vllm_compilation_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_compilation_guide.md) — CUDA Graph、算子融合
- [vllm_parallel_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_parallel_guide.md) — 并行策略、通信后端

## 18. Goodput 调优（SLO 约束下最大有效吞吐）

> 场景：生产环境落地，单纯追求理论最大吞吐会导致用户体验急剧恶化（首字慢、生成卡顿）
> 目标：提升 **Goodput（有效吞吐量）** —— 在严格满足业务延迟 SLO（TTFT < 2000ms，ITL < 20ms）前提下，系统能提供的最大 Token 处理速率
> 工具：`vllm bench serve` + 自研 `goodput-search.sh`

## 现象

追求理论最大吞吐时，TTFT/TPOT 经常突破 SLO 红线，用户体验差。需要在"吞吐"和"延迟"之间找平衡点。

## 方案：两步法

### 第一步：用无限制并发 (Request Rate = inf) 找并发边界

在请求速率无限大的极限压力下，系统会因排队展现出最差的 TTFT，同时 TPOT 也会展现出该并发下的真实基线。依次观察并发提升时的指标：

| Concurrency (RR=inf) | TTFT | TPOT | 是否满足 SLO |
|---------------------|------|------|-------------|
| 8 | 749.44ms | 19.83ms | ✅ 符合（TPOT 逼近红线） |
| 9 | 811.77ms | 20.95ms | ❌ TPOT > 20ms |

为满足 TPOT < 20ms 的硬性 SLO，系统能承受的最大并发数 (Max Concurrency) 必须锁定为 **8**。并发升至 9 时，底层 GPU 的 Batch 计算耗时已超过单字 20ms 的容忍度。

### 第二步：在边界并发 (Concurrency = 8) 附近，调 Request Rate 找吞吐拐点

以并发上限 8 为边界，将请求速率从 1.0 逐步拉升至 inf，寻找最大有效吞吐量：

**最优性能点（Sweet Spot）：Concurrency = 8 且 Request Rate = 3.0**，输出吞吐量达到最高（375.91 tokens/s），同时 TTFT (654.72ms) 远低于 2000ms 红线，TPOT (19.87ms) 刚好卡在 20ms 内。

## 关键认知

**并发数 (Concurrency) 的影响** —— 决定系统的吞吐上限与单次处理效率：
- 总吞吐量：显著正相关。并发 7→9，输出吞吐从 ~348 提升到 ~394 tokens/s。
- 输出延迟 (TPOT & ITL)：轻微负面影响。并发 7→9，平均 ITL 从 ~44.8ms 升到 ~50.7ms。
- 首字延迟 (TTFT)：负面影响。并发越高，Prefill 阶段要处理更庞大的上下文，TTFT 基线被抬高。

**请求速率 (Request-rate) 的影响** —— 决定系统的排队与拥堵程度：
- 首字延迟 (TTFT)：强正相关。请求速率增加（1.0→inf），TTFT 线性上升（并发 9 时从 584ms 飙到 811ms）。请求速率超过并发处理上限时，多余请求在队列排队，排队时间直接计入 TTFT。
- 吞吐量与输出延迟 (Throughput, TPOT, ITL)：几乎无影响（水平直线）。一旦系统达到满载，继续提高请求发送速率既不能增加吞吐，也不会改变单字解码速度。

**热力图观察：**
- 纵向（固定并发，Request Rate 增大）：TTFT 颜色越来越深（偏红）；Throughput/TPOT/ITL 每列颜色几乎一致。
- 横向（吞吐量断层）：Output Throughput 颜色变化是纯横向的。不管 Request Rate 是 1.0 还是 inf，只要并发从 7 提到 9，吞吐就从 ~348 跃升到 ~394。
- 甜点区：Concurrency = 9 且 Request Rate = 1.0~2.0，吞吐最高 393.99，TTFT 保持较低 584.20ms。

## 工具：goodput-search.sh

最小依赖下，支持不同并发数、request-rate 组合寻找最优 goodput：

```bash
# 无限制并发找边界
CONCURRENCY=6,7,8,9,10 ./goodput-search.sh
# 边界找 goodput
CONCURRENCY=8,9 REQUEST_RATE=1,2,3,4,5,6,7 ./goodput-search.sh
# 指定模型/数据集
BASE_URL=http://127.0.0.1:8000 MODEL='<model_dir>/DeepSeek-V3-INT8/' ./goodput-search.sh
DATASET_NAME=random RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=128 ./goodput-search.sh
```

## 数据图表

## 关联文档

- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — TTFT/TPOT/ITL 指标、请求全生命周期
- [vllm_scheduler_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_scheduler_guide.md) — 调度器、token budget、抢占
- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — KV Cache 显存、并发上限

## 19. 显存占用 / KV Cache 拐点与并发上限

> 场景：测试项目中常遇到"满足 SLA 条件的最大并发数"测试项，如何选择对 DCU 最有利的阈值
> 观察：TTFT 指标与 KV Cache 显存占用密切相关 —— 显存占满后请求进入 pending，TTFT 急剧上升
> 平台：BW1000，Qwen3-32B，TP2，`--gpu-memory-utilization=0.95`

## 现象

TTFT 随并发数增加出现明显拐点，怀疑拐点处 KV Cache 显存占用达到峰值。

## 理论推导（Qwen3-32B）

**关键架构参数：** 32.8B 参数，64 层，GQA 64(Q)/8(KV) 头，head_dim=128，FP16。

**显存分配：**
- 总显存：2 卡 × 64GB = 128GB
- GPU 利用率 95%：128 × 0.95 = 121.6GB
- 模型参数：32.8B × 2 bytes = 65.6GB
- 剩余可用于 KV Cache + 激活值：121.6 - 65.6 = **56GB**（激活值占用极小，忽略）

**每 Token KV Cache 占用：**
```
2 × num_layers × num_kv_heads × head_dim × dtype_size
= 2 × 64 × 8 × 128 × 2 bytes = 262,144 bytes ≈ 256 KB/token
```

**并发数计算（4096 in / 4096 out）：**
- 每请求 KV（Prefill）= 4096 × 256KB ≈ 1.0GB
- 每请求 KV（Decode）= (4096+4096) × 256KB ≈ 2.0GB
- Prefill 阶段最大并发 = 56 / 1.0 = **56**
- Decode 阶段最大并发 = 56 / 2.0 = **28**

## 实测验证

| 并发 | 现象 |
|------|------|
| 56 (PC off) / 60 (PC on) | TTFT 出现明显拐点 |
| 32 | TPOT 出现明显拐点 |
| 28 | 生成吞吐量最大值 |

**现象确认：** 56 并发（PC off）时 prefill 阶段 GPU KV Cache 占用率接近 100%；32 并发时 decode 阶段接近 100%。当 KV Cache 占用率达 100% 时，请求 pending，继续增大并发带来性能明显劣化（即柱状图中的拐点）。

**日志解读：** 模型参数占用 30.58GB（每卡），KV Cache 预留 27.88GB（每卡），可支持 228352 tokens（每卡）。
- Prefill 最大并发 = 228352 / 4096 = 55.76
- Decode 最大并发 = 228352 / (4096+4096) = 27.87
- 每 token KV = 27.88GB × 2 / 228352 = 256KB（与理论一致）

**理论 vs 实测：** Prefill 推算 56 ≈ 实测 56；Decode 推算 28 vs 实测 32（decode 后期部分请求结束释放 KV 导致）；吞吐最大值 28 与 Decode 拐点一致。

## 源码分析：KV Cache 显存计算

vLLM 采用 PagedAttention，将 KV Cache 划分为固定大小的 Block。关键类 `FullAttentionSpec`：
```
pages_size_bytes = coef × block_size × num_kv_heads × head_size × dtype_size
                 = 2 × 64 × 4 × 128 × 2 = 131,072 bytes = 128KB   (coef=2 表示 K 和 V)
num_blocks = available_memory // pages_size_bytes // num_layers
           = 29,930,553,344 // 131,072 // 64 = 3568
num_tokens = num_blocks × block_size = 3568 × 64 = 228,352
```

## 结论

1. TTFT/TPOT 的性能会在 KV Cache 显存占用饱和时出现明显劣化（拐点），尤其开启 Prefix Caching 时，首字延迟几乎劣化到和 PC OFF 一致的水平。
2. "最大生成吞吐量"的并发数基本与 Decode 阶段 KV Cache 占用饱和的并发数一致。
3. 可通过 vLLM 服务日志或理论计算推导出"拐点"对应的并发数，与实测基本一致。
4. 参考建议：通过日志/理论计算推导拐点并发数，进行少量实验即可挑选出 DCU 最优并发数及对应 TTFT/TPOT 阈值。

## 数据图表

## 关联文档

- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — PagedAttention、Block 管理、前缀缓存、LRU 驱逐
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — TTFT/TPOT 指标、并发上限
- [vllm_scheduler_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_scheduler_guide.md) — 调度、抢占、pending

## 20. 权重量化 W8A8 / AWQ / W4A8

> 背景：受制裁影响无法使用最强 HBM，DCU 带宽成为算力瓶颈。量化通过精度类型转换减少显存压力，以可控算力开销换更高显存利用率
> 收益：显存占用降至 1/2（W8A8）或 1/4（W4A8/AWQ），让单机无法承载的大模型得以单设备运行

## 现象

大模型显存占用高、推理成本高，单机装不下或并发上不去。

## 方案：DCU 主推三种量化

### 1. W8A8 (Channel-wise INT8) —— 显存 1/2

模型（如 DeepSeek-R1-INT8、Qwen3-235B-A22B-int8）由 llmcompressor 离线产出，自带 quantization_config，vLLM 自动识别，无需手动指定 `-q`。

```bash
export ALLREDUCE_STREAM_WITH_COMPUTE=1   # AllReduce 通信与计算流式重叠
export W8A8_SUPPORT_METHODS=3            # 1=Triton, 2=CUTLASS, 3=rocBLAS
export USE_FUSED_RMS_QUANT=1             # 融合 RMS+int8 量化算子
export USE_FUSED_SILU_MUL_QUANT=1        # 融合 SiLU+Mul+int8 量化算子
export VLLM_USE_TOPK_RENORM=1            # TopK 重归一化，走 lightop 算子融合
```

DeepSeek-R1 INT8 双机 16 卡（推荐 pp2+tp8）：
```bash
vllm serve <model_dir>/DeepSeek-R1-INT8/ \
--trust-remote-code --dtype bfloat16 \
--max-model-len 40960 -tp 8 -pp 2 \
--max-seq-len-to-capture 16384 \
--distributed-executor-backend ray \
--gpu-memory-utilization 0.9 --max-num-seqs 256 \
--block-size 64 --enable-chunked-prefill --enable-prefix-caching \
--compilation-config '{"full_cuda_graph": true}' \
--speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3}' \
--max-num-batched-tokens 16384
```

### 2. AWQ (W4A16) —— 显存 1/4

Weight-Only INT4，单机 8 卡即可承载 671B MoE 模型。MoE 模型用 `-q moe_wna16`，非 MoE 用 `-q awq`。

```bash
# MoE 模型环境变量（dense 模型不能加！）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PCIE_USE_CUSTOM_ALLREDUCE=1
export HIP_USE_GRAPH_QUEUE_POOL=1        # 减少 Graph 捕获/重放队列创建、同步开销
export VLLM_USE_GLOBAL_CACHE13=1         # MoE global memory cache，减少反复显存分配释放
export W4A16_MOE_CUDA=1                  # 仅限 DeepSeek-R1/V3-AWQ，其它 MoE 设 0
export VLLM_FUSED_MOE_CHUNK_SIZE=8192    # 限制单次 fused MoE 内核处理 token 数上界
```

DeepSeek-R1-AWQ + MTP（推荐最优性能）：
```bash
VLLM_USE_V1=1 vllm serve /path/DeepSeek-R1-awq \
--trust-remote-code --dtype float16 \
-q moe_wna16 -tp 8 \
--max-model-len 32768 --max-seq-len-to-capture 8192 \
--gpu-memory-utilization 0.92 --max-num-seqs 256 \
--speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3}' \
--block-size 64 --max-num-batched-tokens 8192 \
--compilation-config '{"full_cuda_graph": true}'
```

### 3. W4A8 (slimquant_w4a8) —— 显存 1/4

混合量化：精度敏感的 Attention/Linear 层保留 INT8，参数密集的 MoE 层（`.mlp.experts.`）量化为 INT4 并实施 **Top-95% 软截断**（计算 Scale 时取全通道 95% 分位数作截断阈值，主动硬截断 5% 极端异常点，换剩余 95% 数值在 -7~+7 区间更高量化分辨率）。

```bash
vllm serve /module3/DeepSeek-R1-0528-W4A8-V2 --port 20011 \
--trust-remote-code --dtype bfloat16 \
-q slimquant_w4a8_marlin \
--kv-cache-dtype fp8_e5m2 \
--max-model-len 49152 --max-num-batched-tokens 8192 \
-tp 8 --gpu-memory-utilization 0.93 --max-num-seqs 512 \
--disable-log-requests --block-size 64 --enforce-eager \
--no-enable-prefix-caching --enable-chunked-prefill \
--speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3}'
```

## 量化原理对比

| 方案 | 量化类型 | 启动方式 | 显存 | 核心原理 |
|------|---------|---------|------|---------|
| W8A8 (Channel-wise) | INT8 | 自动识别或 `-q compressed-tensors` | 1/2 | 权重 per-channel scale 转 int8；激活在线按 token 动态量化 int8；scaled int8 GEMM；scale 恢复 fp16 |
| AWQ (W4A16) | INT4 | MoE: `-q moe_wna16`；非 MoE: `-q awq` | 1/4 | 权重 4bit 打包 + group scale/zero；对 1% 显著权重做 per-channel scale 缩放补偿；kernel 内解包 (32 位拆 8 个 int4)；(q-zero)×scale 成 fp16；与 fp16 激活 MMA |
| W4A8 | INT4+INT8 | `-q slimquant_w4a8_marlin` | 1/4 | MoE experts: W4A8-Marlin（解包 int4 + 通道 scale 内核内缩放乘加）；Linear/Attention: W8A8 |

**Channel-wise vs Block-wise INT8：**
- Channel-wise：scale 是 1D 向量 (N,1)，GEMM epilogue 一次 1D 广播乘法，几乎零额外开销；可走 rocBLAS/hipBLASLt 厂商优化路径；寄存器压力小。**性能好**。
- Block-wise：scale 是 2D 矩阵 (N/128, K/128)，每个 128×128 tile 算完要从显存随机读 scale 做 2D 逐元素乘法，L2 cache 频繁未命中；必须走 Triton 自定义 kernel；寄存器用量激增 30-50%。**精度高**（Outlier 影响限制在 128×128 小方块内）。
- 简言之：**channel-wise 性能好，block-wise 精度好**。

## 精度与性能（Qwen3-32B，GSM8K）

| 模型 | GSM8K |
|------|-------|
| Qwen3-32B-fp16 | 96.63 |
| Qwen3-32B-GPTQ-Int8 | 96.36 |
| Qwen3-32B-W8A8 | 96.66 |
| Qwen3-32B-GPTQ-Int4 | 96.25 |
| Qwen3-32B-AWQ | 96.66 |

**结论：** 量化可极大减少显存开销，跑更大模型、支持更长上下文，尤其长文本、高并发场景有优势。W8A8 和 AWQ 性能不错，相比 FP16 有提升；GPTQ 劣化严重，不可用。相同卡数下，随并发增大 W8A8 性能逐渐超过 AWQ（大并发 W8A8 优势更大）；AWQ 显存需求只有 W8A8 一半，不追求极限性能时 AWQ 吞吐能效更高（需要 DCU 更少）。

## 数据图表

## 关联文档

- [vllm_quantization_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_quantization_guide.md) — 量化：FP8/GPTQ/AWQ/slimquant_w4a8、kernel 选择
- [vllm_fused_moe_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_fused_moe_guide.md) — MoE：路由、FusedMoE kernel、EP all2all、EPLB
- [vllm_parallel_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_parallel_guide.md) — TP/PP/EP 并行策略

## 21. KV Cache FP8 量化（e5m2 软件模拟）

> 背景：自回归解码中 KV cache 显存随上下文长度和并发数近线性增长，很多场景瓶颈不是算力而是 KV cache 显存
> 方案：vLLM FP8 KV cache 量化，把 K/V 张量以 FP8 存储，降低 KV cache 内存占用
> 收益：同样显存下能放更多 token / 更高并发，主要收益是吞吐提升、最大上下文/并发上限提升

## 现象

高并发下 KV cache 显存不够，请求 pending，吞吐上不去。

## 原理

- **写入缓存**：把 Key/Value 张量按缩放因子量化为 FP8，再写入 KV cache。
- **读取缓存**：做注意力计算前，从缓存取出 FP8 的 K/V，反量化回 FP16/BF16。
- **注意力计算**：softmax 输出 × V 等关键乘法在反量化后的高精度张量上进行。

**拿算力换显存**：主要收益是"允许分配约两倍 KV cache 空间，从而提升吞吐/并发或上下文长度"，但延迟未必降低（还没把"反量化 + attention"做成融合/量化 attention 的硬件加速路径）。

**e5m2 vs e4m3：**
- E5M2：5 位指数 + 2 位尾数（动态范围更大，精度更低）
- E4M3：4 位指数 + 3 位尾数（精度更高，动态范围更小 ~±240，需 FP32 缩放因子配合）
- 想要更稳数值范围（不易溢出/饱和）→ E5M2；想要更好精度且能接受缩放因子/校准复杂度 → E4M3
- 当前 vLLM FP8 KV cache 缩放因子只支持 per-tensor（标量），更细粒度（per-channel）还在开发

**DCU 硬件差异：**
- bw1100 具备 fp8 算力，支持 fp8 e4m3，性能更优
- K100_AI、gfx936 (bw10/bw100/bw150/bw1000) 没有 fp8 算力，但最新版本已有 fp8 e5m2 软件模拟实现

## 效果（BW150，fp8 e5m2 软件模拟）

- 大并发下，bw150 上 kvcache fp8 e5m2 量化，对 **tpot、decode 吞吐、生成吞吐均有加成**，**ttft 有劣化**（符合预期）；并发越大，几个参数大部分有优化趋势。
- 长上下文高并发场景有劣化趋势，建议避免 kvcache 用满的场景。
- 精度影响不大，但仍需关注客户场景终端精度。
- 无 pending 最大支持并发无劣化，长输入输出下提升较明显，提升比例在 **0-71%** 之间，仍有优化空间。

**为什么 tpot 在不开量化时增长不明显，而开启量化时增长明显？** 不开量化 kvcache 在高并发时显存不够，请求 waiting，同时 decode 并发数不再增大，因而变化不大；而开启量化支持更大并发，decode 属于 memory bound，tpot 随并发增加。

**triton 后端问题：** 2 个版本的 kvcache 量化效果很差（ttft/tpot/生成吞吐都有问题）。使用默认 triton 后端效果更差：triton 后端的 fp8 e5m2 显著劣化；因此可明确 fp8 问题出在 flash-attention 包的优化上。

## 数据图表

## 关联文档

- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — KV Cache、PagedAttention、Block 管理
- [vllm_quantization_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_quantization_guide.md) — 量化：FP8、kernel 选择
- [vllm_attention_backend_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_attention_backend_guide.md) — Attention 后端、flash-attention

## 22. APC 前缀缓存性能影响

> 功能：APC 自动缓存现有查询的 KV 缓存，新查询与现有查询共享相同前缀时直接复用 KV 缓存，跳过共享部分的计算
> 目标：验证开关 APC 对 TTFT、生成吞吐的影响，指导测试场景的标准化方案
> 平台：K100-AI ×8，DS-7B / DS-70B

## 现象

相同前缀的长文本 Prompt 的 Prefill 操作持续占用大量 GPU 计算资源，KV cache 利用率不够。

## 方案

对比测试 APC 开/关，覆盖在线/离线、固定/随机字符、不同输入输出长度、不同 batch：

| 测试维度 | 配置 |
|---------|------|
| 离线测试 | 输入输出 128-128 / 128-1024 / 1024-128 / 1024-1024，DS-7B，bs 1/8/16/32/64/128 |
| 在线测试 | 同上，DS-7B / DS-70B |
| 字符类型 | 固定字符 / 随机字符 |
| APC | 默认关闭 vs `--enable-prefix-caching` |

## 效果

- **官方 APC 测试脚本**：打开 APC 后，前缀匹配情况下预处理时间大幅降低（4 条 prompt 从 2.95s 降到 1.08s）。
- **标准性能测试**：固定字符/随机字符输入下，离线与在线测试的首字延迟与生成吞吐均有一定提升（提升比例不太稳定，取决于缓存命中率）。
- **应用场景**：常规场景下均有一定性能提升，尤其适用于**长输入短输出**场景。

## 结论与建议

1. **vllm 框架研究中**需考虑关闭 APC，防止引入随机性能影响导致研究结论存在不确定性。
2. **项目测试、性能测试、客户生产环境**中均建议打开 APC，无论离/在线场景或长/短文本场景（APC 是 vllm 官方支持的优化方案之一，高版本 vllm 可能默认开启）。
3. 项目中允许的情况下，可考虑固定输入内容并在正式测试前用固定输入充分预热 vllm 每个服务实例（提高缓存命中率）。
4. DCU 优势测试用例可考虑针对此功能设计极限性能测试（测试流程与工具充分利用 APC，例如预热固定问题增加缓存命中率）。

## 数据图表

## 关联文档

- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — 前缀缓存、Block 管理、LRU 驱逐
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — TTFT 指标、请求全生命周期

## 23. max_model_len / mnbt 对 TTFT 的影响

> 背景：vLLM 项目上有部分数据显示 max_model_len 对 TTFT 影响很大
> 参数：
> - `max_model_len`：单条请求输入+输出长度，默认读配置（ds7b 为 131072），找不到则默认 2048
> - `max_num_batched_tokens` (mnbt)：单并发最大处理的 token 数，默认 10240，与 `max_num_seqs`（最大并发数）共同约束调度器的并发能力
> 平台：BW150 ×8，DeepSeek-R1-Distill-70B

## 现象

请求排队等待 → TTFT 显著增加。当超出并发数造成请求排队，显著增加 TTFT。

**关键机制：** 未启用 chunked prefill 时，`mnbt = max(max_model_len, 2048)`。当 `prompt_length > mnbt`，prefill 在一个调度周期完成不了，请求会推迟到下个周期，从而显著增大 TTFT。

## 方案：调大 mnbt 的权衡

**Decode 是 IO 密集型**，吞吐量上限受限于显存带宽。mnbt 决定 vllm 单次调度允许处理的最大 tokens 数，当一个批次能处理更多 tokens，带宽利用率就越高，Decode 效率越高（TPOT 更好）。

但调大 mnbt，vllm 在一轮里能塞进更多 Token，虽然 Decode 效率提高，但这一轮计算本身耗时变长，waiting 队列里的请求需等更长时间，**TTFT 变差**。

**TTFT 长尾现象（3072/1024/128 组）：**
- mnbt 较小 (10240)：TTFT(P99) 达 61 秒，长尾严重。mnbt 小，前面积压太多任务，最后面的请求被挤压到很靠后的轮次。
- mnbt 较大 (131072)：每轮处理 Token 更多，每轮计算时间更长，但每轮能清掉的请求更多，末尾请求不会被搁置太久，优化了长尾。TTFT(P99) 55 秒，比小 mnbt 好。

**吞吐数据：**
- 1 并发：无论 mnbt 是 10240 还是 131072，对一条输入 3072 输出 1024 的请求，Token Budget 都能满足第一轮完成 Prefill+Decode，mnbt 大小对性能没影响。
- 64 并发：大 mnbt 让系统每轮处理更多 Token，Decode 效率提升，TPOT 变好，Decode 耗时减少，虽然 Prefill 时间也增加，但吞吐总体提升。
- 128 并发：系统进入过载区。Decode 阶段变快（Mean TPOT 从 103ms 降到 89ms），但 Prefill 排队太久，大 mnbt 导致 TTFT 暴涨（Mean 从 32s 增到 43s，Median 从 32s 跳到 37s）。Prefill 时间劣化占据上风，拉低全过程平均吞吐。128 并发 Prefill 劣化严重的原因：Decode 阶段很多请求被 running 踢出，进入 waiting 等待重新 Prefill。

## Chunked Prefill

vLLM 0.9.2 及以上版本默认开启（`--no-enable-chunked-prefill` 关闭）。允许将大的 prefill 请求分割成多个小块，与 decode 请求混合批处理，平衡计算密集型（prefill）和内存密集型（decode）操作。
- 降低首字延迟：长 prefill 不会完全阻塞系统
- 更好的吞吐量：更灵活的批处理策略

## 结论

- 调大 mnbt 在中等并发（64）下提升吞吐，但在过载并发（128）下因 Prefill 排队导致 TTFT 暴涨、吞吐反降。
- 短输入下调大 `max_num_seqs` 能提升服务质量。
- 长输入场景建议开启 chunked prefill，避免 prefill 阻塞。

## 数据图表

## 关联文档

- [vllm_scheduler_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_scheduler_guide.md) — 调度器、token budget、chunked prefill、抢占
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — TTFT/TPOT 指标
- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — KV Cache、并发上限

## 24. LMCache KV 卸载（CPU/NVMe offload）

> 功能：LMCache 把 KV cache 卸载到 CPU 内存 / NVMe，长文本预填充复用，绕过重复 prefill
> 目标：量化 TTFT、吞吐量变化，验证海光 DCU 兼容性和性能优化空间
> 痛点：TTFT 延迟高（长文本预填充计算开销大）

## 现象

长文本场景 TTFT 高，重复前缀反复 prefill 浪费算力。

## 方案：LMCache 三级缓存

- **L1 GPU**：KV cache 在 GPU 显存
- **L2 CPU offload**：卸载到 CPU 内存，`max_local_cpu_size` 建议设为 MemAvailable 的 70-80%
- **L3 持久化 (NVMe)**：长期保存复用价值高的缓存（如系统提示词），绕过预填充，显著降低 TTFT

```yaml
# CPU offload 配置
max_local_cpu_size: 160   # 单位 GB，建议 MemAvailable 的 70-80%
# NVMe offload 配置
max_local_cpu_size: 20    # CPU 缓存层，必须设置，建议 MemAvailable 的 10-20%
```

## 效果（性能影响分析）

**LMCache 效果显著：**
- **CPU offload 性能最佳**：在输出吞吐量和 TTFT 方面表现最优。
- **批量处理优势明显**：16k-1k 上下文，batch≥8 时启用缓存带来显著性能提升。
- **吞吐量对比**：batch=32 时，CPU offload 吞吐量（515 tok/s）是无 LMCache 缓存（254 tok/s）的 **2 倍**。
- **NVMe offload** 性能介于两者之间，约为 CPU offload 的 85-90%。
- **延迟对比**：CPU offload 极大降低延迟，batch=128 时 TTFT 仅为无缓存模式的 **29%**。随着批次增大，延迟差距更显著；无缓存模式在大批次时延迟呈指数级增长。

**推荐方案：**
- 注重推理性能 → 首选 CPU offload
- 注重性价比 → 可用 NVMe offload

## 注意（DCU 适配测试记录）

另一组 DCU 适配测试记录显示：从两组对比实验看，**目前并未看到 lmcache 对 vllm 模型推理有任何提升效果**，即使在官方建议使用场景下。怀疑是 DCU 适配的 vllm 并不支持 KV 缓存卸载到 CPU 上的操作。需确保设备 KV cache 能到 100% 再触发重计算和卸载。

> 结论：LMCache 在支持 KV offload 的版本上效果显著（CPU offload 吞吐 2 倍、TTFT 降至 29%），但 DCU 定制版需先确认是否支持 KV 卸载到 CPU。

## 数据图表

## 关联文档

- [vllm_kv_transfer_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kv_transfer_guide.md) — KVConnector、KV 传输、P-D 分离
- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — KV Cache、前缀缓存
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — TTFT 指标
