# 性能优化案例 §25-§33（适配与精度类）

> 2026-10-16 自 `perf/dcu-perf-cases/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），案例四段式原文照录、未改一字；编号与原文一致，索引见主 SKILL.md 案例索引表。

## 25. Qwen3.5/3.6 MTP 适配与 APC 冲突

> 模型：Qwen3.5/3.6 系列（混合架构：Full Attention + Linear Attention）
> 功能：MTP（Multi-Token Prediction，多解码头）投机解码 + APC（Automatic Prefix Caching）
> 平台：BW1000 单机 4 卡

## 现象

Qwen3.5 在 vLLM 0.15.1 上开启 MTP + APC 时启动断言失败。

## 定位

**版本问题：** vLLM 0.15.1 发布于 2026 年 2 月初，Qwen3.5 模型发布于 2 月中旬。vLLM 0.15.1 设计时尚未对 Qwen3.5 这种混合架构（Full Attention + Linear Attention）的原生 MTP 功能进行适配和测试。

**缓存与解码的冲突：**
- Prefix Caching：开启时 vLLM 会为 Qwen3.5 的线性注意力层设置高效的缓存模式 `align`（`--mamba-cache-mode`）。
- Speculative Decoding (MTP)：Qwen3.5 实现多 Token 预测、加速推理的核心技术。
- 在 vLLM 0.15.1 中，推测解码的逻辑与为 Qwen3.5 模拟的 `align` 缓存管理模式存在内部冲突，导致服务启动时断言失败。

**三种缓存模式对比：**
- `none`：前缀缓存禁用时使用
- `all`：缓存所有在位置 i × block_size 处的 token 的 mamba 状态（支持该功能的模型默认行为）
- `align`：仅缓存每个调度步骤最后一个 token 的 mamba 状态，且该 token 必须位于 i × block_size 位置

注：参数名含 "mamba"，但 Qwen3.5 实际是纯 Transformer 架构，并未使用 Mamba 结构。

## 方案

- **临时方案**：基于 vLLM 0.15.1，关闭 APC（`--no-enable-prefix-caching`）开启 MTP。
- **源码方案**：vLLM 0.17.1 以上已修复 Qwen3.5 系列 MTP 与 APC 冲突。基于 0.15.1 源码修改，适配 DCU 底层同时启动 MTP + APC。修改涉及 `speculative.py`、`vllm.py`、`mamba2.py`、`nemotron_h.py`、`mamba_mixer2.py`、`extract_hidden_states.py`、`eagle.py`、`mamba_attn.py`、`mla/indexer.py`、`gpu/model_runner.py` 等。

## 效果

**MTP 优劣势汇总：**
- **MTP 优势场景**：短序列和中序列，开启 MTP 可提升吞吐量，尤其低并发场景可大幅提升（例如 4K-1K 单并发推理 tps 由 35 → 79，**提升 120%+**）。
- **MTP 劣势**：增加 TTFT 延迟，特别是高并发时（128、256 等）。
- **长上下文性能**：关闭 MTP 后，吞吐与 TTFT 性能普遍更优。

**配置推荐：**

| 场景 | 推荐配置 |
|------|---------|
| 短中上下文 (1K-1K、4K-1K) | 开启 MTP + 开启 APC |
| 长上下文 (16K-1K) | 关闭 MTP + 开启 APC |
| 注重 TTFT | 关闭 MTP + 开启 APC |
| 注重吞吐 | 短中上下文开 MTP+APC，长上下文关 MTP+APC |

**已知遗留（vllm018 镜像已解决）：**
- 同时开启 MTP 与 APC：极长上下文场景（16K-1K）256 并发请求 block 存在异常报错
- 开启 MTP：模型推理结果存在乱码现象

## 数据图表

## 关联文档

- [vllm_spec_decode_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_spec_decode_guide.md) — 投机解码：MTP/EAGLE/DFlash/DSpark
- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — 前缀缓存 (APC)、KV Cache
- [vllm_model_runner_v2_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_model_runner_v2_guide.md) — Model Runner、混合架构模型状态

## 26. language-model-only 与 KV 冗余

> 功能：`--language-model-only` 跳过视觉编码器和多模态配置，释放更多内存用于 KV 缓存
> 模型：Qwen3.5/3.6 系列（多模态模型，但纯文本场景可只跑语言模型）
> 平台：BW1000、K100AI

## 现象

多模态模型在纯文本场景下，视觉编码器占用显存但用不上，KV cache 空间受限。

## 方案

开启 `--language-model-only`，跳过视觉编码器和多模态配置。

```bash
# 开启 language-model-only
vllm serve <model_dir>/Qwen3.5-35B-A3B --tensor-parallel-size 4 \
--enable-prefix-caching --reasoning-parser qwen3 \
--gpu-memory-utilization 0.9 --port 8000 --language-model-only
```

## 效果

| 功能 | 输出吞吐 | TTFT | TPOT |
|------|---------|------|------|
| 开启 language-model-only | 小幅提升 | 小幅提升 | 持平 |

开启此功能后，可冗余出 **1.7% 的 KV cache 空间**，并小幅提升推理吞吐及 TTFT。

## 适用场景

- **推荐开启**：仅涉及纯文本（聊天、写作、代码编程、文本分析等）。
- **切勿开启**：需处理图像、视频等多模态输入，必须关闭，否则模型无法理解视觉信息。

## 数据图表

## 关联文档

- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — KV Cache 显存
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — 吞吐/TTFT 指标

## 27. EAGLE3 / N-gram 投机解码

> 背景：预测解码（投机解码）三大类：独立草稿模型 (draft-model)、无草稿 N-gram 匹配、Medusa/MTP（多解码头）
> 目标：验证除 MTP 外，EAGLE3 草稿模型和 N-gram 算法对吞吐的提升，储备 DCU 性能优化方案
> 平台：K100-AI ×8（EAGLE3）/ BW1000 ×8（N-gram）

## 现象

MTP 之外，还有哪些投机解码技术能在 DCU 上提升吞吐？

## 方案

- **EAGLE3 草稿模型**：独立草稿模型方式，小模型先并行生成 k 个 token，大模型一次性验证。
- **N-gram 匹配**：无草稿模型，纯算法级，无需额外模型，资源占用主要在 CPU。

## 效果

**EAGLE3 草稿模型：**
- 当前版本在多并发场景下会带来性能损耗，且无法支持更高规模（上下文 4k 及以上）文本。
- 单并发场景下吞吐有一定提升（**512-512 场景吞吐提升 75%**）。

**N-gram 算法：**
- 低并发、中低规模上下文场景下对吞吐有一定提升（**1024-1024 场景吞吐提升 35%**）。
- 并发及输入输出规模逐渐提高时存在明显"收益递减"问题。

## 结论

1. 在特定场景下应用 N-gram 技术可提升吞吐性能，目前版本不适合高并发、长文本场景。
2. EAGLE3 当前版本仅单并发有效，多并发反而损耗。
3. 后续 vllm 新版本 0.11.0 会有更多预测解码优化（如 ArcticInference），需时刻关注前沿技术并在 DCU 上应用，储备 DCU 性能优化方案。

## 数据图表

## 关联文档

- [vllm_spec_decode_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_spec_decode_guide.md) — 投机解码：MTP/EAGLE/DFlash/DSpark、拒绝采样
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — 吞吐/TPOT 指标

## 28. DS-R1 PP2+TP8 双机推理

> 模型：DeepSeek-R1-INT8（671B MoE，W8A8）
> 硬件：C86-4G + BW1000，中兴 ROCE 200G 交换机，双机 16 卡
> 镜像：vllm 0.9.2-ubuntu22.04-dtk26.04-0115-das1.8-py3.10-20260116

## 现象

长输入高并发（如 32K-1K 100 并发）时发生 OOM。

## 定位与方案

**长输入 OOM 问题：** 修改源码 `gpu_model_runner.py` 534 行，去掉 if 分支，直接执行 else 分支，跳过对不在批次中的请求的立即添加，避免集中内存分配，从而防止长输入导致的 OOM。

**PP2+TP8 部署：** 双机 16 卡，TP8（机内 8 卡张量并行）+ PP2（跨机 2 机流水线并行）。

```bash
# 关键环境变量
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=p14p2
export GLOO_SOCKET_IFNAME=p14p2
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
export NCCL_NET_GDR_READ=1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
# PP 并行
export VLLM_SCHED_ENABLE_MINIMAL_INJECTION=1
export VLLM_PP_LAYER_PARTITION=35,26
export VLLM_USE_PP_SYNC=1
# OOM 缓解
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_ENABLE_TBO=0
```

```bash
vllm serve <model_dir>/DeepSeek-R1-INT8/ --port 8000 \
--dtype bfloat16 -pp 2 -tp 8 \
--trust-remote-code --distributed-executor-backend ray \
--max-model-len 56000 --max-seq-len-to-capture 16384 \
--gpu-memory-utilization 0.9 --max-num-seqs 512 \
--disable-log-requests --block-size 64 \
--enable-chunked-prefill --max-num-batched-tokens 16384 \
--enable-prefix-caching --kv-cache-dtype fp8_e5m2 \
--compilation-config '{"full_cuda_graph": true}' \
--speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3}'
```

## 效果

**dtk2604 + pp2tp8 方案，推理吞吐量提高 1 倍以上，且 TTFT 随并发数增大大幅降低。**

对比 TP16 方案（双机 16 卡纯张量并行），PP2+TP8 在跨机通信上更优（PP 跨机只传激活值，TP 跨机每层 AllReduce 通信量大）。

## 数据图表

## 关联文档

- [vllm_parallel_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_parallel_guide.md) — TP/PP/EP 并行策略、通信后端
- [vllm_kv_transfer_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kv_transfer_guide.md) — P-D 分离、KV 传输
- [vllm_quantization_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_quantization_guide.md) — W8A8 量化

## 29. BW150 拓扑 / TP 加速

> 目标：对比 BW150 的 4 卡互联和 2 卡互联对模型性能的影响
> 平台：X7850H0，C86 7490 ×2，BW150 ×8
> 模型：Qwen3-32B、Qwen3-235B-AWQ

## 现象

不同卡间互联拓扑（4 卡互联 vs 2 卡互联）对 TP 加速比的影响不明。

## 方案

用 `all_reduce_perf` / `alltoall_perf` 微基准测试不同拓扑的通信带宽，再跑模型推理对比 TP2/TP4/TP8 加速情况。

```bash
# 2 卡 allreduce 最佳性能
NCCL_SIMPLE_CHANNELS=20 RCCL_P2P_XHCL_CHANNEL_NUM=16 RCCL_COLL_XHCL_CHANNEL_NUM=16 \
  ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 2
# 4 卡 allreduce 最佳性能（多 ring）
NCCL_RINGS="0 1 2 3|0 3 2 1|0 1 3 2|0 2 3 1|0 2 1 3|0 3 1 2" NCCL_ALGO=Ring \
  NCCL_SIMPLE_CHANNELS=20 RCCL_P2P_XHCL_CHANNEL_NUM=16 RCCL_COLL_XHCL_CHANNEL_NUM=16 \
  ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 4
# 8 卡 allreduce
./build/all_reduce_perf -b 2 -e 2G -f 2 -g 8
# all2all 同理
```

## 效果

- TP2、TP4、TP8 加速情况：TP 规模越大，加速比受通信开销影响越明显。
- 4 卡互联 vs 2 卡互联：4 卡互联拓扑通信带宽更高，TP4/TP8 加速比更优。
- `NCCL_RINGS` 多 ring 配置对 4 卡 allreduce 性能有显著提升。

## 结论

- 拓扑选择直接影响 TP 加速比，4 卡互联优于 2 卡互联。
- NCCL/RCCL 通信参数（channels、rings、algo）需针对拓扑调优。

## 数据图表

## 关联文档

- [vllm_parallel_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_parallel_guide.md) — TP 并行、通信后端、NCCL/RCCL

## 30. 推理带宽使用分析（UMC 采集）

> 目标：采集大模型推理全流程的带宽数据，深入分析内部数据流转机制，掌握推理全过程的带宽使用特征与流转规律
> 平台：BW1000，UMC 带宽采集
> 测试：`vllm bench serve`，"1024 1024" / "4096 1024" / "16384 1024"，预处理默认分块 10k

## 现象

需要量化推理各阶段（prefill/decode）的带宽使用，定位带宽瓶颈。

## 方案

用监控脚本采集 UMC 带宽（`rocm-smi` / UMC 计数器），结合 `vllm bench serve` 不同输入输出规模、不同 batch（bs1/bs2/bs4/bs8）的推理，对比带宽使用。

```python
# 监控脚本核心：正则解析 rocm-smi 输出的 HCU 带宽
REGEX_SHOWBW = re.compile(r'HCU\[(\d+)\].*?:\s*(\d+\.?\d*)', re.M)
# 周期性采集，记录时间戳 + 各 HCU 带宽
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
```

## 效果

- 多 bs 综合对比：不同 batch 下带宽使用差异明显。
- 补测不同 chunked-prefill 分块大小（10k vs 1k）：分块大小影响 prefill 阶段的带宽峰值。
- 注意：UMC 带宽采集存在异常，部分指令未能输出正常值，需结合其他指标交叉验证。

## 结论

- Decode 是 memory bound，带宽利用率是吞吐上限的关键。
- 调大 batch / mnbt 可提高带宽利用率（见 [max-model-len-mnbt](max-model-len-mnbt.md)）。
- 带宽分析需结合 UMC 计数器 + 推理指标交叉验证，单一指标可能有采集异常。

## 数据图表

## 关联文档

- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — 带宽/算力利用率、Roofline
- [vllm_scheduler_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_scheduler_guide.md) — batch、chunked prefill

## 31. MoE 路由选错专家乱码（lightop 修复）

> 场景：医疗大模型（某医疗大模型）在 DCU 上输出乱码/退化重复，BW1000 上正常
> 模型：某医疗MoE模型（BailingMoeV2，32 层，L1–L31 为 256 专家 MoE，top_k=8，grouped topk 路由）
> 硬件：海光 DCU BW100（gfx936）
> 软件：vLLM 0.15.1 / transformers 5.2.0（两者都乱码）

## 现象

| 平台 | 推理框架 | 输出 | 状态 |
|------|---------|------|------|
| DCU gfx936 (BW100) | vLLM 0.15.1 | `\x00\x00\x00...`（token 188 循环） | ❌ 乱码 |
| DCU gfx936 (BW100) | transformers 5.2.0 bf16 | `\nOkay\nOkay...` 重复退化 | ❌ 乱码 |
| DCU gfx936 (BW100) | transformers 5.2.0 fp32 | `\nOkay\nOkay...` 重复退化 | ❌ 乱码 |
| BW1000（其他机器） | vLLM | 正常医学回答 | ✅ 正常 |

bf16 和 fp32 都乱码 → 排除精度问题；transformers 原生（不用任何 vLLM 自定义 kernel）也乱码 → 问题不在 vLLM 特有算子。

## 定位

**三阶段排除法：**

1. **算子排查（全部排除）**：RMSNorm（cosine=1.0000）、FlashAttention（输出非 0 且量级合理）、路由算子 `ops.moe_fused_gate`（用 L1 真实权重对比，topk_ids 完全一致）、Expert FFN（transformers 纯 PyTorch 也乱码，排除）。
2. **权重完整性（排除）**：30 个 shard 逐字节对比 modelscope 远端一致、safe_open 全部可读、抽样 150 个 tensor 无 NaN/Inf/全 0、L4 权重分布与邻居层无显著差异。
3. **精确定位（命中）**：transformers 原生推理中 hook 所有 decoder layer 输出，发现残差从 L4 开始爆炸：

| 层 | std | max | 状态 |
|----|-----|-----|------|
| L0–L3 | 0.08 → 0.30 | 7.3 → 26.1 | 正常 |
| L4 | 2.19 | 598 | 骤升（~10× std） |
| L5 | 49.41 | 16937 | 爆炸（~170× std，饱和） |
| L6–L30 | ≈50 | ≈17200 | 饱和 |

对 L4 内部子算子 hook，爆炸点精确定位在 `L4.mlp`（MoE）：输入 std=0.173/max=2.6 → 输出 std=2.157/max=596.2。再拆 routed vs shared：shared 正常（std=0.046），**routed experts 把正常输入放大约 12×(std) / 220×(max)**。

**根因铁证（monkey-patch 抓 vLLM 实际 topk，而非重算）：**

| 对比项 | 结果 |
|--------|------|
| router_logits | cos=0.999998（gate 算得对） |
| topk_ids | **exact=0/32（一个专家都没选对）** |
| routed_sum | cos=0.125787（选错专家 → 加权和全错） |

```
vLLM topk_ids[0,:8]: [80, 112, 82, 106, 230, 236, 216, 206]
HF   topk_idx[0,:8]: [201, 195, 202, 17, 81, 120, 219, 16]
```

**因果链**：router_logits 正确 → `ops.moe_fused_gate`（vLLM C++/HIP fused kernel，DCU 默认路径）在 `n_group=8, topk_group=4, sigmoid + expert_bias` 参数下选错专家 → routed_sum 错 → MoE 层输出错 → 逐层 residual 累积放大（L1 cos=0.94 → L8 cos=0.01 → L31 爆炸到 1.6e4）→ logits 崩塌 → 单 token（188=`\x00`）垄断输出。

BW1000 正常的原因：走的是不同的算子后端，未触发有问题的 kernel。

## 方案

绕开有问题的 fused router kernel，改走正确的算子路径（`GroupedTopKRouter._compute_routing` 三分支）：

| 方案 | 环境变量 | 走的算子 | 说明 |
|------|---------|---------|------|
| ❌ 问题路径（默认） | `VLLM_ENABLE_MOE_FUSED_GATE=1` 且 `VLLM_USE_LIGHTOP=0` | `ops.moe_fused_gate`（vLLM C++/HIP） | 选错专家 |
| ✅ 方案 A（推荐） | `VLLM_ENABLE_MOE_FUSED_GATE=1` + `VLLM_USE_LIGHTOP=1` | `lightop.op.moe_fused_gate` | 高性能 fused，选专家正确 |
| ✅ 方案 B | `VLLM_ENABLE_MOE_FUSED_GATE=0` | 纯 Python `grouped_topk`（@torch.compile） | 与 HF 一致，正确但更慢 |

```bash
# 方案 A（推荐）
export VLLM_ENABLE_MOE_FUSED_GATE=1
export VLLM_USE_LIGHTOP=1
```

注意：`VLLM_ENABLE_MOE_FUSED_GATE` 不带值时为未设置（沿用默认 fused gate），需显式赋 0 才能禁用。

## 效果

- 方案 A 启动后请求正常返回医学回答，模型正确自述身份，输出连贯、无乱码、无退化重复。
- 修复验证（monkey-patch 31 个 MoE 层改用纯 Python grouped_topk）：topk_ids set match 4/4，L1 mlp_out cos 从 0.630 回升到 0.962，NULL(188) 消失。
- `test_lightop_vs_torch.py` 验证 lightop 与纯 Python 实现在同输入下 ids 100% 一致，方案 A 可安全替代。
- 残余 cos 差异（0.962 未到 1.0）来自 bf16 精度 + TP=4 all_reduce 累加顺序，不足以引发层间放大——说明根因是"选错专家"这种结构性错误，而非浮点漂移。

## 可复用经验

1. **重算 ≠ 实际值**：早期用纯 Python grouped_topk 重算 topk 与 HF 对比，得出"topk 一致"的错误结论，误导排查。必须 monkey-patch 抓框架实际运行时的 topk，才暴露 0/32 选错专家。
2. **register_forward_hook 的 in-place 污染**：`residual = hidden_states` 让 residual 与入参共享内存，后续 in-place 相加污染 args[0]，post-forward hook 抓到的是被污染的值。必须用 `register_forward_pre_hook` + 立即 clone。
3. **collective_rpc 的持久化陷阱**：vLLM v1 EngineCore 是子进程，collective_rpc 用 cloudpickle 序列化每次调用，模块级 dict 不跨调用持久化。抓到的张量必须存在 `worker.model_runner.model` 这个长生命周期对象上。
4. **逐层 cosine 衰减曲线是粗粒度定位利器**：embedding/L0 完全一致、L1 开始偏、L8 完全发散 → 矛头直接指向第一个 MoE 层。
5. **同一模型在不同硬件上表现不同，优先怀疑硬件相关的算子后端**（本例 BW100 走 vLLM fused kernel，BW1000 走 lightop）。
6. 排查脚本沉淀在 `scripts/`（算子级单测）与 `precision_compare/`（端到端精度对比框架），算子精度定位 agent 见 `op-locate-agent` 仓库。

## 关联文档

- [vllm_fused_moe_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_fused_moe_guide.md) — MoE 路由、FusedMoE、grouped topk
- [vllm_attention_backend_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_attention_backend_guide.md) — 算子后端选择
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — 精度问题排查方法论

## 32. MiniMax-M3 AWQ INT4 适配（456B MoE）

> 场景：456B 参数 MoE 多模态模型的 AWQ INT4 量化版在 DCU 上部署
> 模型：MiniMax-M3-AWQ-INT4（60 层 = 3 dense + 57 MoE，128 experts/层，top_k=4，compressed-tensors 4-bit，group_size=32，symmetric=false）
> 硬件：8× 海光 K100AI DCU（gfx928，64GB/卡），TP=8
> 软件：SGLang 海光定制版 + Triton 3.5.1 + PyTorch 2.9.0 + DTK 2604

## 现象

量化模型在 DCU 上无法正常运行，先后暴露 10 个问题：启动校验失败、权重加载失败、VMFault 崩溃、int overflow、输出错误（"sky is the color of water"）、重复、chat 复述 system prompt 等。

## 定位

按"逐层 hook → 分阶段诊断（gate→experts→combine→allreduce）→ 端到端数值验证"的方法逐个定位：

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | `minimax_m3_sparse` 不在 layer_types 白名单 | SGLang 校验白名单未包含 | 白名单添加该类型 |
| 2 | `CLIPVisionConfig.rope_theta` AttributeError | rope_theta 嵌套在子配置中未提取到顶层 | config 解析时提取到顶层 |
| 3 | AutoTokenizer 加载失败 | 自定义 tokenizer 类缺依赖 | 回退 PreTrainedTokenizerFast + 嵌入 chat_template |
| 4 | 推理时 Invalid address access / SIGABRT | Triton cache 残留不兼容 kernel binary | 清理 `/root/.triton/cache/tmp.*` 重新编译 |
| 5 | MoE zero_point overflow（int32 溢出） | 全零 zp 的 fallback `fill_(0x88888888)` 超出 int32 范围 | 按 uint8 视图 `fill_(0x88)` |
| 6 | Dense 层 0-2 误建为 MoE（128 个空 expert） | `moe_layer_freq=None` 时默认全 MoE，`mlp_layer_types` 字段未被使用 | 按 `mlp_layer_types[layer_id] != "dense"` 判断，dense 层不传 quant_config |
| 7 | shared_experts 权重 not found | checkpoint 中 shared_experts 为全精度，传 quant_config 导致参数名变成 `weight_packed` 不匹配 | `MiniMaxM3MLP(..., quant_config=None)` |
| 8 | 所有层 attention `weight_zero_point` 无法加载 | `_get_scheme_from_parts()` 创建 CompressedTensorsWNA16 时未传 `symmetric`，默认 True 导致 zp 不注册 | 显式传 `symmetric=weight_quant.symmetric`（False） |
| 9 | **输出错误（最终精度 root cause）** | `_ct_dequantize_hip` 中 ZP 解包 reshape 顺序错误：`[N//8, K_groups, 8]` 做 C-order reshape 时 nibble 维度和 K_groups 混合 | 先 `permute(0,2,1).contiguous()` 让 nibble 与 N 合并再 reshape |
| 10 | MoE combine 结果不可靠 | Triton sum reduce kernel 在 gfx928 上兼容性问题 | 改用 `torch.sum(intermediate_cache3, dim=1, out=...)` |

问题 9 的关键代码：

```python
# 错误：nibble 维度和 K_groups 混合
zp_unpacked = zp_unpacked.reshape(N // pack_factor * pack_factor, num_groups)
# 正确：先 permute 让 nibble 与 N 合并
zp_unpacked = zp_unpacked.permute(0, 2, 1).contiguous().reshape(-1, num_groups)
zp_unpacked = zp_unpacked[:N, :]
```

影响面：57 层 × 4 projection = **228 个 attention 矩阵的 dequant 全部有系统偏差**。

## 方案

6 个文件的 patch（unified patch 或文件直接替换）：

| 文件（相对 sglang/srt/） | 修改内容 |
|--------------------------|---------|
| `layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | ZP reshape permute 修复 + HIP dequant |
| `layers/quantization/compressed_tensors/compressed_tensors.py` | symmetric 参数传递 |
| `layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` | MoE zp overflow 修复 |
| `models/minimax_m3.py` | dense/MoE 层判断 + quant_config 控制 |
| `models/minimax_m3_vl.py` | 权重名映射 + fallback |
| `layers/moe/moe_runner/triton_utils/fused_moe.py` | HIP combine 改为 torch.sum |

最终启动命令：

```bash
sglang serve \
  --model-path <model_dir>/MiniMax-M3-AWQ-INT4 \
  --mem-fraction-static 0.55 --tp 8 --dtype bfloat16 \
  --quantization compressed-tensors \
  --attention-backend triton --mm-attention-backend triton_attn \
  --trust-remote-code --host 0.0.0.0 --port 8080 \
  --cuda-graph-max-bs 8 --chunked-prefill-size 4096 \
  --max-running-requests 64 --schedule-policy fcfs
```

## 效果

| 指标 | 数据 |
|------|------|
| 单用户 decode 吞吐 | ~19 tokens/s |
| CUDA Graph | 已启用（bs=1-8） |
| MoE kernel 端到端验证 | max_diff=0.024（BF16 精度，CPU ground truth 对比） |
| Attention dequant 验证 | 7 个不同层/projection，max_diff 全部 < 0.003 |
| 输出质量 | "The color of the sky is" → " blue." ✓；事实问答、chat 多轮、思维链推理全部通过 |

## 可复用经验

1. **默认参数是 bug 源**：`symmetric=True` 默认值导致 zp 静默丢失，权重加载"成功"但数值全错。
2. **reshape/permute 必验证**：多维 tensor 操作后检查元素映射正确性，C-order reshape 极易混淆维度语义。
3. **部分正确 ≠ 无 bug**："sky is the color of water" 说明 dequant 有系统偏差，不能接受"正常精度损失"的模糊结论——下结论前必须做逐元素验证（checkpoint ground truth 对比，而非仅统计量）。
4. **日志警告不能忽略**：`weight not found in params_dict` 直接指向权重加载失败，是量化适配的高频信号。
5. **inplace 诊断需 clone**：在 inplace 操作前保存快照，否则 hook 抓到的是被污染的值。
6. **Triton cache 是隐性变量**：换硬件/换版本后清理 `~/.triton/cache`，残留 kernel binary 会引发难复现的 VMFault。

## 关联文档

- [vllm_quantization_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_quantization_guide.md) — AWQ / WNA16 / compressed-tensors 量化原理
- [vllm_fused_moe_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_fused_moe_guide.md) — MoE 专家 FFN、combine、allreduce
- [vllm_attention_backend_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_attention_backend_guide.md) — triton attention 后端

## 33. DS-V3.2 压测空输出（tokenizer 覆盖）

> 场景：evalscope 对 DeepSeek-V3.2-FP8 做性能压测，收到空响应
> 模型：DeepSeek-V3.2-FP8（FP8 量化版）
> 硬件：BW1100，8 卡 TP=8
> 软件：SGLang 0.5.1rc0 + evalscope perf + transformers v5

## 现象

| 组件 | 表现 | 日志摘要 |
|------|------|---------|
| evalscope | 空输出 | `ERROR: Received empty response list!` |
| sglang | token 爆炸 & 乱码 | `ħå¹ħåº¦Templateŀ×¨Ġstanifiable...`（UTF-8 字节被当 Latin-1 解码的乱码） |
| sglang | token 过长 | `324912 > 131072`（序列长度超模型上限 2.5 倍） |

evalscope debug 日志显示：原始 token 序列 29996 个 → decode 成 182980 字符的乱码文本 → 重新 encode 后变成 30477 个 token，**编解码 token 数量差异巨大**。

## 定位

对比 evalscope 和 sglang 的日志 + 阅读 tokenizer 源码，发现两者编解码行为不一致：

1. **evalscope 和 sglang 的 tokenizer 不一致**：evalscope 走 Metaspace pre_tokenizer，sglang 走 ByteLevel，两者不匹配导致编解码 token 数量差异巨大。
2. **transformers v5 强行覆盖 tokenizer.json 配置（核心诱因）**：v5 对 `LlamaTokenizer` 做了重大修改，`__init__` 中重建 pre_tokenizer 和 decoder，把 tokenizer.json 里的 `Sequence (ByteLevel)` 覆盖成 `Metaspace`、decoder 从 `ByteLevel` 改成 `Sequence`，导致中文无法正确处理，跨 tokenizer 编码严重膨胀。
3. **为什么 Qwen3.5 没这个问题**：Qwen3.5 有独立的 `Qwen3_5Tokenizer` 类，不会被 v5 强制覆盖；DeepSeek-V3.2 没有独立 Tokenizer 类，用的是 `LlamaTokenizer`，被强制覆盖。

## 方案

| 方案 | 做法 | 适用场景 |
|------|------|---------|
| 1. fix v5（推荐，治本） | 修改 evalscope 源码：加载 tokenizer 后重新读 tokenizer.json，若 pre_tokenizer 类型与 json 不一致则还原 `backend.pre_tokenizer` / `backend.decoder` | 长期维护的生产环境 |
| 2. `--tokenize-prompt`（治标） | 跳过 decode 步骤，直接发送 token IDs，避免编解码不一致 | 紧急测试场景 |

fix v5 核心逻辑：

```python
def fix_v5_tokenizer_components(tokenizer, tokenizer_path):
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return tokenizer
    raw = RawTokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
    raw_pre = type(raw.pre_tokenizer).__name__
    loaded_pre = type(backend.pre_tokenizer).__name__
    if raw_pre and loaded_pre and raw_pre != loaded_pre:
        backend.pre_tokenizer = raw.pre_tokenizer   # 还原为 json 中的 ByteLevel
        backend.decoder = raw.decoder
    return tokenizer
```

## 效果

- 应用 fix v5 后 evalscope 与 sglang 的 token 数量一致，空输出消失，压测正常跑通。
- `--tokenize-prompt` 方案在紧急场景下立即生效，无需改代码。

## 可复用经验

1. **多组件系统必须统一 tokenizer 配置**：压测工具、推理服务、评测脚本各自加载 tokenizer，任何一处的版本行为差异都会造成"编解码不对称"，症状是 token 数膨胀/收缩 + 乱码 + 空响应。
2. **乱码形态能反推编码链路**：`åįģ` 这类"UTF-8 字节被按 Latin-1 解读"的乱码，说明文本在某个环节被 decode/encode 了两次且编码假设不一致。
3. **框架大版本升级要回归 tokenizer 行为**：transformers v5 这类"按类重建 tokenizer 组件"的改动，对没有独立 Tokenizer 类的模型（走 LlamaTokenizer 兜底的）影响最大。
4. **为自定义模型开发独立 Tokenizer 类**，避免依赖第三方库的默认实现。
5. 排查顺序：先对比两端日志的 token 数（29996 vs 30477）→ 再 decode 出文本看乱码形态 → 最后读 tokenizer 源码确认覆盖行为。

## 关联文档

- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — 请求生命周期、压测方法
- [vllm_quantization_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_quantization_guide.md) — FP8 量化版本部署
