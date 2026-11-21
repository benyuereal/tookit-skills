# HiCache DSV4 单机多轮变长场景部署细节（2026-10 实测）

本文件是 `dcu-hicache` SKILL.md 的 session 级细节补充：DeepSeek-V4-Flash-0731（DSV4）在 8×DCU BW1100、SGLang 0.5.15、DP=8、DSPARK 投机解码下的 HiCache 三层（L1 GPU / L2 host / L3 NVMe）部署与多轮 Agent 变长前缀压测。

> 与 SKILL.md 现有 Qwen3.8-27B（Mamba 混合模型，SGLang 0.5.12，<device-A> 4 卡）场景不同：本场景是 **纯 KV 模型（DSV4）**，驱逐由 KV 主导，且带 **DSPARK 投机解码** + **DP=8**，配置与坑点有显著差异。两者不混用。

## 0. 场景与环境

- 模型：`DeepSeek-V4-Flash-FP8-w8a8`（DP=8，DSPARK 投机解码）
- 硬件：8× DCU BW1100（144GB/卡），宿主内存 1.5T，NVMe /data 3.5T
- 软件：SGLang 0.5.15.post2.dev564（DTK 26.04），Python 3.10
- 引擎路径：`/data/lst/dsv4-flash-0731_hicache`
- 目标：多轮对话 Agent、长输入短输出、多用户高并发下，GPU L1 池满载产生大量前缀驱逐时，对比 baseline（仅 L1）/ L2（host 中转）/ L3+minL2（host+NVMe 磁盘）对 prefill pending 排队的缓解效果。

## 1. 资源配置计算公式（DSV4，核心常量）

> **KV_BYTES_PER_TOKEN ≈ 10.7 KB/token（dsv4, fp8）**。推导：`KV_BYTES_PER_TOKEN = mem_fraction × 可用显存 / 默认池tokens = 0.8 × ~86GB / 6.4M ≈ 10.7 KB/token`。

① **L1（GPU 池）**：
```
L1_tokens = min( mem_fraction × GPU可用显存 / KV_BYTES_PER_TOKEN ,  max_total_tokens )
L1_GB     = L1_tokens × KV_BYTES_PER_TOKEN / 1e9
```
- 两个旋钮：`--mem-fraction-static`（预算上限，DSPARK 下**硬下限 0.406**，低于启动报错）、`--max-total-tokens`（实际截断，本实验用它压小池制造满载）。
- 本实验：`min(0.8 × 86GB / 10.7KB, 500000)` = 500K tokens/卡 ≈ 5.4GB/卡（8 卡 ~43GB）。

② **L2（host 池）**：
```
L2_tokens = L1_tokens × hicache_ratio        # 默认按比例
L2_GB     = L2_tokens × KV_BYTES_PER_TOKEN / 1e9
# 或直接指定（覆盖 ratio）：--hicache-size <GB>（注：DSV4 不支持，见 §4）
```
- 经验公式：`ratio ≈ 工作集W / L1 − 1`；有 L3 兜底时 ratio 可小（0.3~1），L2 只当热层。

③ **L3（NVMe）**：
```
L3_GB ≈ 总工作集KV − L1_GB − L2_GB
      = (总会话数 × 会话上下文 × KV_BYTES) − L1 − L2
```
- 本实验：总 KV ≈ 82GB，L1 仅驻留 43GB，L2=31GB → 落盘 275GB / 17.7 万文件（含 indexer/元数据，故 > 纯 KV 估算）。

**工作集估算**：并发工作集 `W_并发 = 每卡并发会话数 × 会话上下文`；累计 `W_总 = 总会话数 × 会话上下文`。本实验 users=64：每卡 8 会话 × 67K ≈ 536K tokens ≈ L1 池 500K → 满载驱逐。

## 2. 启动配置与关键参数

### 2.1 DSV4 用 SGLang 安装版（0.5.15），不是 dev fork
- DSV4-Flash-0731 **必须用 `/usr/local` 安装版 0.5.15**（dev fork 0.5.12 **无 DSPARK 支持**）。
- 驱逐 double-free 崩溃修复已打到安装版 `hiradix_cache.py`（patch 见 §4.2）。

### 2.2 HiCache 参数（DSV4 与 Qwen3.8 共性 + DSV4 差异）
- `--enable-hierarchical-cache`：启用 L1/L2/L3。
- `--hicache-ratio N`：L2 host 池 = 设备池 × N。**DSV4 不支持 `--hicache-size`，只能 ratio**（见 §4）。
- `--hicache-io-backend direct`：**DCU 必须用 direct**（kernel 后端 JIT kernel 在 DCU 上写野指针）。
- `--hicache-mem-layout layer_first`：MLA/DSV4 池支持 layer_first（扁平 host）。
- `--hicache-write-policy write_through`：官方推荐（host 始终有副本、驱逐无同步写开销）。write_back 省每 token D2H 拷贝，但驱逐有写开销且曾有 double-free 竞态（已 patch，见 §4.2）。
- `--hicache-storage-backend file` + `--hicache-storage-prefetch-policy timeout`：L3 磁盘层（NVMe offload）。timeout=预取进 GPU 且不无限阻塞（本场景用它，命中率 40%）。
- `--enable-cache-report`：必须，否则响应无 `cached_tokens_details` 命中明细。
- **L3 磁盘目录用环境变量**（不是命令行参数）：`export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/data/.../l3` + `mkdir -p`。

### 2.3 DSV4 满载驱逐的压池配置（核心手法）
```
--max-total-tokens 500000      # 压小 L1 池制造满载（默认 6.4M/卡，realistic 负载仅 1~2% 填不满、不驱逐）
--mem-fraction-static 0.8      # 不动（DSPARK 下 0.406 硬下限，低于启动报错）
--max-running-requests 64
```
把 L1 池从默认 6.4M/卡压到 500K/卡，users=64 时每卡 8 会话 × 67K ≈ 536K > 500K → 驱逐发生。这是让 HiCache "有意义的验证"的关键前置：**池不饱和就不驱逐，HiCache 无收益**。

### 2.4 完整启动脚本
三段**baseline / L2 / L3+minL2** 均基于原始 `server-dsv4-flash-0731-dp8.sh`（TP=8/DP=8，DSPARK，端口 10015），共同环境变量（DCU/HIP 拷贝路径调优 + SGLang 优化开关）：
```bash
export NCCL_SOCKET_IFNAME=p32p1
export GLOO_SOCKET_IFNAME=p32p1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_TORCH_PROFILER_DIR=./profile
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_USE_AITER_AG=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_RAGGED_VERIFY_MODE=compact
export SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2
export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
export SGLANG_DSPARK_FAST_KERNEL=1
export SGLANG_DSPARK_FAST_SAMPLING=1

sglang serve \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --tp-size 8 --dist-timeout 10000 --watchdog-timeout 3600 \
  --port 10015 --host 0.0.0.0 \
  --model-path /models/DeepSeek-V4-Flash-FP8-w8a8 \
  --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 64}' \
  --trust-remote-code --chunked-prefill-size 32768 \
  --disable-flashinfer-autotune --skip-server-warmup \
  --cuda-graph-max-bs-decode 256 --mem-fraction-static 0.8 \
  --max-total-tokens 500000 --speculative-algorithm DSPARK \
  --speculative-num-steps 1 --speculative-eagle-topk 1 \
  --max-running-requests 64 --context-length 81920 --enable-metrics \
  --dp 8 --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend none
```
**baseline** 到此为止（无 HiCache 参数）。
- **L2**：追加 `--enable-hierarchical-cache --hicache-ratio 8 --hicache-io-backend direct --hicache-mem-layout layer_first --hicache-write-policy write_through --enable-cache-report`。
- **L3+minL2**：同上但 `--hicache-ratio 1`，再追加 `--hicache-storage-backend file --hicache-storage-prefetch-policy timeout`；L2 参数由 `--hicache-ratio 1` → 31GB，打满后溢写 NVMe（真实落盘 275GB）。

> L2/L3 方案需额外 DCU/HIP 拷贝路径调优 env（HiCache 宿主池初始化需要）：`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000`、`SGLANG_KVALLOC_KERNEL=1`、`SGLANG_SET_CPU_AFFINITY=1`、`HIP_KERNEL_BATCH_CEILING=100`、`GPU_MAX_HW_QUEUES=3`、`HIP_GRAPH_ACCUMULATE_DISPATCH=1`、`HIP_H2D_DISABLE_COPY_BUFFER=0`、`HIP_D2H_DISABLE_COPY_BUFFER=0`、`HIP_H2D_DIRECT_COPY_THRESHOLD=32768`、`HIP_H2D_HSAAPI_COPY_THRESHOLD=32768`、`HIP_D2H_DIRECT_COPY_THRESHOLD=512`、`HIP_D2H_HSAAPI_COPY_THRESHOLD=512`、`HIP_GRAPH_USE_CMD_CACHE=0`、`HSA_KERNARG_POOL_SIZE=8388608`、`ROC_AQL_QUEUE_SIZE=131072`、`MC_ENABLE_DEST_DEVICE_AFFINITY=1`、`ALLREDUCE_STREAM_WITH_COMPUTE=1`。

### 2.5 压测命令（长输入短输出，多轮）
```
python3 codebuddy_perf_test_dsv4.py --seed 2026 --timeout 1800 \
  --output-dir result/real-baseline/l2/l3 loadtest \
  --users 1 16 64 --traffic-pattern burst --ramp-duration 15 \
  --turns 6 --min-prompt 2000 --max-prompt 11000 \
  --min-response 1 --max-response 16
```
- `--turns 6 --max-prompt 11000 --min-response 1 --max-response 16`：会话第 6 轮累积 ~67K tokens（贴近 81.9K 上限），多轮前缀可复用；短输出让 prefill 占主导。
- `--users 1 16 64`：users=64 是满载驱逐点（8 会话/卡 × 67K ≈ 池 500K）。

## 3. 性能对比（本场景，全量沉淀）

### 3.1 总览（users=1/16/64）
| users | 方案 | 成功率 | p95 | p50 | 总吞吐 | 输出吞吐 | 命中率 |
|---|---|---|---|---|---|---|---|
| 1 | baseline | 90.0% | 33.3s | 20.2s | 2,053 | 10.9 | ~50% |
| 1 | L2 | 90.0% | 33.8s | 21.6s | 2,034 | 10.8 | ~50% |
| 1 | L3+minL2 | 90.0% | 17.5s | 6.7s | 4,131 | 21.9 | ~60% |
| 16 | baseline | 90.6% | 141.1s | 48.5s | 9,780 | 51.7 | ~15% |
| 16 | L2 | 89.9% | 162.4s | 35.6s | 9,335 | 48.5 | ~19% |
| 16 | L3+minL2 | 93.7% | 134.9s | 21.2s | 15,393 | 90.3 | ~35% |
| 64 | baseline | 91.5% | 1308.8s | 174.0s | 6,421 | 34.5 | ~6% |
| 64 | L2 | 91.8% | 1231.4s | 175.7s | 7,273 | 40.8 | ~19% |
| 64 | L3+minL2 | 91.1% | 1170.7s | 137.5s | 7,347 | 39.8 | ~40% |

> 命中率定义：`prefill_cache / (prefill_cache + prefill_compute)`（总缓存命中率，含 L1+L2+L3 各层）。baseline 只有 L1，故 ≈ L1 命中率。

### 3.2 核心结论（users=64 满载驱逐）
| 指标 | baseline | L2 | L3+minL2 |
|---|---|---|---|
| p95 | 1308.8s | 1231.4s（↓5.9%） | 1170.7s（↓10.5%） |
| p50 | 174.0s | 175.7s | 137.5s（↓21%） |
| 总吞吐 | 6,421 | 7,273（+13.3%） | 7,347（+14.4%） |
| 缓存命中率 | ~6% | ~19% | ~40% |
| pending 峰值 | 581,215 | 427,463（↓26%） | 355,145（↓39%） |
| 峰值时刻吞吐 | 5.8 tok/s（停滞） | 2,621 tok/s | ~2,900 tok/s |

- **users=16（中等压力）是 L3 最大收益点**：总吞吐 +57%（9,780 → 15,393）、p50 ↓56%。
- **L3+minL2 的核心机制**：`--hicache-storage-prefetch-policy timeout` 主动预取，提前把磁盘前缀塞进 L1，请求到达即命中 → 这是 L3 命中率高达 ~40%（vs L2 ~19%）的关键。L2（无磁盘）溢出的 KV 直接丢弃 → 命中率上不去；L3（有磁盘）溢出保留 → 命中率更高。

### 3.3 资源占用实测
| 项 | baseline | L2 | L3+minL2 |
|---|---|---|---|
| L1 GPU 池 | 500K tokens/卡 ≈ 5.4GB/卡 | 同 | 同 |
| L1 满载 | 峰值 88% | 峰值 89% | 峰值 76% |
| L2 host 池 | — | ratio8→248GB | ratio1→31GB |
| L2 实际用量 | — | 17.4M tokens（54%） | 4.0M tokens（100% 满） |
| L3 磁盘 | — | — | 275GB / 177,206 文件 |
| L3 读回 | — | — | 7.07M tokens |
| 宿主内存可用 | ~851GB | ~618GB | ~824GB |

### 3.4 成功率说明（~90% 失败率，不影响对比）
失败全部为 **context 超限 400**：`--context-length 81920` 是 baseline 脚本保守设定，模型实际支持 1M 上下文（`max_position_embeddings=1048576`，YaRN×16）。max-prompt 11000×6 轮个别会话超 81.9K。三方案失败率一致，不影响相对对比。
**解决建议**：`--context-length` 提到 131072（模型支持 1M），会话完全在内 → 成功率 ~100%，且不影响池大小（池由 `--max-total-tokens` 单独控制）。

## 4. Bug 解决与 Patch（DCU 适配，DSV4 场景）

### 4.1 HiCache 宿主池初始化把 DCU HSA stream 卡死（最严重）
- **现象**：`--enable-hierarchical-cache` 启动时，worker 在 `memory_pool_host.py` 的指针张量构造处无限期卡死。native 栈：`hipMemcpyWithStream → hsa_signal_wait_scacquire(∞)`，GPU 0% 利用率，8 rank 全冻结。两次启动均复现（即使改逐元素赋值也卡）→ GPU stream 被前面的操作搞挂。
- **根因**：宿主池分配器 `HostTensorAllocator` 用 `alloc_mmap`（`mmap(MAP_SHARED|MAP_POPULATE)`）分配几百 GB host 内存再 `cudaHostRegister`。该路径在 DCU 上损坏 GPU DMA/HSA 状态。隔离验证：`torch.empty`+`cudaHostRegister` 单进程 405GB 正常；`alloc_mmap`+register 单进程 500s 无进展。
- **修复**（`patches/host_allocator_torch_empty_dcu_fix.patch`）：`HostTensorAllocator.allocate` 改为 `torch.empty(dims, dtype, device="cpu")`。修复后启动 ~5 分钟正常。
- **诊断线索**：启动只在 HiCache 宿主池初始化处卡死、GPU 0%、native 栈 `hsa_signal_wait_scacquire` → 大概率此 bug。

### 4.2 驱逐 double-free 崩溃
- **现象**：高并发（users=5/10）下 `_evict_backuped` 对已驱逐节点 `node.value is None` 再 `free(None)` → `AttributeError`。
- **修复**（`patches/hiradix_evict_backuped_None_value_fix.patch`）：`_evict_backuped` 开头加 `if node.value is None: 刷新叶子簿记, return 0`，幂等。

### 4.3 指针张量构造（保守加固）
- **修复**（`patches/memory_pool_host_dcu_ptr_tensor_fix.patch`）：`torch.tensor(list, device=gpu)` → 逐元素赋值 `_build_ptr_tensor`（最终非卡死根因，但保留更稳构造）。

### 4.4 其它工程坑
| 问题 | 现象 | 处理 |
|---|---|---|
| `--hicache-size` 不支持 DSV4 | 启动报 `DeepSeek V4 HiCache currently does not support --hicache-size` | 改用 `--hicache-ratio` 表达容量 |

### 4.5 Patch 应用方式
```bash
# 补丁位于 /data/lst/dsv4-flash-0731_hicache/patches/
# 目标文件 /usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/
cd /usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache
patch -p0 < /data/lst/dsv4-flash-0731_hicache/patches/host_allocator_torch_empty_dcu_fix.patch
patch -p0 < /data/lst/dsv4-flash-0731_hicache/patches/hiradix_evict_backuped_None_value_fix.patch
patch -p0 < /data/lst/dsv4-flash-0731_hicache/patches/memory_pool_host_dcu_ptr_tensor_fix.patch
```
> 补丁打在容器内 `dist-packages`，镜像重建后需重打（与 SKILL.md 现有 Qwen3.8 场景一致）。

## 5. 与 SKILL.md 现有场景的区别（易混点）

| 维度 | SKILL.md 现有（Qwen3.8-27B） | 本文件（DSV4-Flash-0731） |
|---|---|---|
| 模型 | Mamba 混合（KV+Mamba 状态） | 纯 KV（MLA） |
| SGLang | 0.5.12（dev） | 0.5.15（安装版，必须） |
| 卡数/DP | <device-A> 4 卡 TP=4 | 8× BW1100 DP=8/TP=8 + DSPARK |
| 压池手法 | —（池大，自然） | 主动 `--max-total-tokens` 压池制造驱逐 |
| 默认池 | 6.4M/卡 | 6.4M/卡（压到 500K 才满载） |
| DSPARK | 无 | `--speculative-algorithm DSPARK`（mem-fraction 0.406 硬下限） |
| 最严重 bug | Bug B（分页布局 D2H/H2D 回退） | §4.1 HSA stream 卡死（宿主池初始化） |
