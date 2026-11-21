# vLLM 性能优化快速查找手册

> **用法**：遇到性能问题或优化需求 → 按"症状/场景"定位到案例 → 看案例里的"方案 + 效果" → 点"关联文档"深入原理。
> 形成 **手册 → 案例 → 解决方案 → 原理文档** 的闭环。

---

## 一、按症状查找（"我遇到了 X 问题"）

| 症状 | 最可能原因 | 看这个案例 | 关键方案 |
|------|-----------|-----------|---------|
| **TTFT 高**（首字慢） | 长 prefill 阻塞 / 排队 / 前缀重复计算 / 长上下文 | [max-model-len-mnbt](dcu-perf-cases 案例23) · [apc-prefix-caching](dcu-perf-cases 案例22) · [lmcache-kv-offload](dcu-perf-cases 案例24) · [dsv4-flash-pd-disaggregation](dcu-perf-cases 案例37) | 开 chunked prefill · 开 APC 预热 · KV offload 复用前缀 · 长上下文 PD 分离 + CP8 |
| **TPOT 高**（出字慢） | decode 带宽瓶颈 / 算子次优 / 降频 | [gemma-spec-decode-aiter](dcu-perf-cases 案例13) · [asr-citic-perf](dcu-perf-cases 案例17) · [max-model-len-mnbt](dcu-perf-cases 案例23) | 调大 mnbt 提带宽利用率 · 算子调优 · 锁频 |
| **OOM**（显存不够） | KV cache 占满 / 长输入集中分配 | [mem-usage-study](dcu-perf-cases 案例19) · [weight-quantization](dcu-perf-cases 案例20) · [kvcache-quantization](dcu-perf-cases 案例21) | 量化降显存 · 推导拐点并发数 · 改 gpu_model_runner 跳过集中分配 |
| **吞吐上不去** | 并发不足 / 带宽利用率低 / 排队 | [goodput-tuning](dcu-perf-cases 案例18) · [max-model-len-mnbt](dcu-perf-cases 案例23) · [bandwidth-analysis](dcu-perf-cases 案例30) | 找 SLO 约束下甜点并发 · 调 mnbt · 带宽分析 |
| **请求 pending / 排队** | KV cache 用满触发抢占 | [mem-usage-study](dcu-perf-cases 案例19) · [kvcache-quantization](dcu-perf-cases 案例21) | KV 量化支持更大并发 · 推导拐点 |
| **乱码 / 精度异常** | 量化列序错 / dtype 不匹配 / 版本未适配 / MoE 路由错 / 视觉累加器精度 | [gemma-spec-decode-aiter](dcu-perf-cases 案例13) · [qwen35-mtp](dcu-perf-cases 案例25) · [antangelmed-moe-garbled](dcu-perf-cases 案例31) · [citic-flame-vision-precision](dcu-perf-cases 案例36) · [minimax-m3-awq-int4](dcu-perf-cases 案例32) | AWQ 列重排 · `--dtype float16` · `VLLM_USE_LIGHTOP=1` 绕开 fused gate · 视觉换 TRITON_ATTN · 量化 dequant 逐元素验证 |
| **空输出 / token 爆炸** | 跨组件 tokenizer 不一致 / 框架版本覆盖配置 | [deepseek-v32-empty-output](dcu-perf-cases 案例33) | fix v5 还原 tokenizer.json · `--tokenize-prompt` 直发 token |
| **崩溃 / 启动失败** | 算子不支持 / 编译 bug / 版本冲突 / KERNEL VMFault | [gemma-spec-decode-aiter](dcu-perf-cases 案例13) · [qwen35-mtp](dcu-perf-cases 案例25) · [dsr1-pp2tp8](dcu-perf-cases 案例28) · [vmfault-paged-attention](dcu-perf-cases 案例34) · [fa-compile-vmfault](dcu-perf-cases 案例35) | 查踩坑表 · 升级版本 · 改源码适配 · rocm-debug-agent 定位 VMFault · 自编译替换 FA |
| **多机性能差** | 跨机通信瓶颈 / 拓扑不佳 / 长上下文 TTFT | [dsr1-pp2tp8](dcu-perf-cases 案例28) · [bw150-topology](dcu-perf-cases 案例29) · [dsv4-flash-pd-disaggregation](dcu-perf-cases 案例37) | PP2+TP8 替代 TP16 · 调 NCCL/RCCL 参数 · 选优拓扑 · PD 分离 + CP8 |

---

## 二、按场景查找（"我要做 X 优化"）

| 场景/目标 | 案例 | 预期收益 |
|-----------|------|---------|
| **降 TTFT**（首字延迟） | [apc-prefix-caching](dcu-perf-cases 案例22) · [lmcache-kv-offload](dcu-perf-cases 案例24) · [max-model-len-mnbt](dcu-perf-cases 案例23) | APC 前缀复用 · KV offload TTFT 降至 29% · chunked prefill |
| **降 TPOT**（字间延迟） | [gemma-spec-decode-aiter](dcu-perf-cases 案例13) · [asr-citic-perf](dcu-perf-cases 案例17) | 投机解码 TPOT -61% · 权重转置 -11.7% |
| **提吞吐**（Goodput） | [goodput-tuning](dcu-perf-cases 案例18) · [max-model-len-mnbt](dcu-perf-cases 案例23) | 找 SLO 甜点 · 调 mnbt |
| **省显存 / 装更大模型** | [weight-quantization](dcu-perf-cases 案例20) · [kvcache-quantization](dcu-perf-cases 案例21) | W8A8 显存 1/2 · W4A8/AWQ 1/4 · KV FP8 并发 +71% |
| **投机解码加速** | [gemma-spec-decode-aiter](dcu-perf-cases 案例13) · [qwen35-mtp](dcu-perf-cases 案例25) · [spec-decode-eagle-ngram](dcu-perf-cases 案例27) | MTP 短中序列 +120% · EAGLE3 单并发 +75% · N-gram +35% |
| **多机部署** | [dsr1-pp2tp8](dcu-perf-cases 案例28) · [bw150-topology](dcu-perf-cases 案例29) | PP2+TP8 吞吐 +1 倍 · 拓扑调优 |
| **KV 复用 / 长文本** | [lmcache-kv-offload](dcu-perf-cases 案例24) · [apc-prefix-caching](dcu-perf-cases 案例22) | CPU offload 吞吐 2 倍 |
| **长上下文（16K+）部署** | [dsv4-flash-pd-disaggregation](dcu-perf-cases 案例37) | PD 分离 TPOT≤25 吞吐 +317%~1362% · CP8 prefill 5-8× |
| **多模态纯文本场景** | [qwen35-language-model-only](dcu-perf-cases 案例26) | 冗余 1.7% KV cache |
| **多模态视觉精度** | [citic-flame-vision-precision](dcu-perf-cases 案例36) | TRITON_ATTN 修复视觉识别，准确率 100% |
| **大 MoE 量化模型适配** | [minimax-m3-awq-int4](dcu-perf-cases 案例32) · [antangelmed-moe-garbled](dcu-perf-cases 案例31) | 456B AWQ INT4 上 DCU · MoE 路由乱码修复 |
| **带宽/性能剖析** | [bandwidth-analysis](dcu-perf-cases 案例30) · [mem-usage-study](dcu-perf-cases 案例19) · [zhejiang-gaoxin-qwen35-2b-perf](dcu-perf-cases 案例6) | UMC 带宽采集 · KV 拐点推导 · 端到端延迟分解 |

---

## 三、按指标查找（"我想优化 X 指标"）

| 指标 | 含义 | 优化方向 | 案例 |
|------|------|---------|------|
| **TTFT** | 首字延迟（Prefill 主导） | 降 prefill 计算 / 复用前缀 / 避免排队 | [apc](dcu-perf-cases 案例22) · [lmcache](dcu-perf-cases 案例24) · [mnbt](dcu-perf-cases 案例23) |
| **TPOT** | 字间延迟（Decode 主导） | 提带宽利用率 / 算子调优 / 投机解码 | [gemma](dcu-perf-cases 案例13) · [asr](dcu-perf-cases 案例17) · [mnbt](dcu-perf-cases 案例23) |
| **ITL** | 字间延迟（含抖动） | 同 TPOT + 减少抢占 | [goodput](dcu-perf-cases 案例18) · [mem](dcu-perf-cases 案例19) |
| **吞吐** | tokens/s | 提并发 / 提带宽利用率 / 量化 | [goodput](dcu-perf-cases 案例18) · [quant](dcu-perf-cases 案例20) |
| **Goodput** | SLO 约束下最大吞吐 | 找甜点并发 + 请求速率 | [goodput](dcu-perf-cases 案例18) |
| **显存** | 占用 / 并发上限 | 量化 / 推导拐点 | [quant](dcu-perf-cases 案例20) · [mem](dcu-perf-cases 案例19) |

---

## 四、按模型/硬件查找

| 模型/硬件 | 案例 |
|-----------|------|
| **Qwen3-ASR**（语音） | [asr-citic-perf](dcu-perf-cases 案例17) |
| **Qwen3.5/3.6**（混合架构） | [qwen35-mtp](dcu-perf-cases 案例25) · [qwen35-language-model-only](dcu-perf-cases 案例26) · [citic-flame-vision-precision](dcu-perf-cases 案例36) · [zhejiang-gaoxin-qwen35-2b-perf](dcu-perf-cases 案例6) · [vmfault-paged-attention](dcu-perf-cases 案例34) |
| **DeepSeek-R1**（671B MoE） | [dsr1-pp2tp8](dcu-perf-cases 案例28) · [weight-quantization](dcu-perf-cases 案例20) |
| **DeepSeek-V3.2**（FP8） | [deepseek-v32-empty-output](dcu-perf-cases 案例33) |
| **DeepSeek-V4-Flash**（FP8/INT8） | [dsv4-flash-pd-disaggregation](dcu-perf-cases 案例37) |
| **MiniMax-M3**（456B MoE） | [minimax-m3-awq-int4](dcu-perf-cases 案例32) |
| **某医疗MoE模型**（BailingMoeV2） | [antangelmed-moe-garbled](dcu-perf-cases 案例31) |
| **gemma-4-31B-AWQ** | [gemma-spec-decode-aiter](dcu-perf-cases 案例13) |
| **BW150**（gfx936） | [asr-citic-perf](dcu-perf-cases 案例17) · [bw150-topology](dcu-perf-cases 案例29) · [kvcache-quantization](dcu-perf-cases 案例21) |
| **BW100**（gfx936） | [antangelmed-moe-garbled](dcu-perf-cases 案例31) · [zhejiang-gaoxin-qwen35-2b-perf](dcu-perf-cases 案例6) |
| **BW1000** | [mem-usage-study](dcu-perf-cases 案例19) · [dsr1-pp2tp8](dcu-perf-cases 案例28) · [bandwidth-analysis](dcu-perf-cases 案例30) · [fa-compile-vmfault](dcu-perf-cases 案例35) · [vmfault-paged-attention](dcu-perf-cases 案例34) |
| **BW1100** | [deepseek-v32-empty-output](dcu-perf-cases 案例33) · [dsv4-flash-pd-disaggregation](dcu-perf-cases 案例37) |
| **BW10**（48 CUs） | [gemma-spec-decode-aiter](dcu-perf-cases 案例13) |
| **K100 / K100_AI** | [apc-prefix-caching](dcu-perf-cases 案例22) · [spec-decode-eagle-ngram](dcu-perf-cases 案例27) · [minimax-m3-awq-int4](dcu-perf-cases 案例32) · [citic-flame-vision-precision](dcu-perf-cases 案例36) |

---

## 五、高频优化手段速查

| 手段 | 开关/参数 | 效果 | 案例 |
|------|----------|------|------|
| **APC 前缀缓存** | `--enable-prefix-caching` | 前缀复用，降 TTFT | [apc](dcu-perf-cases 案例22) |
| **Chunked Prefill** | 默认开（0.9.2+） | 长 prefill 不阻塞 | [mnbt](dcu-perf-cases 案例23) |
| **KV FP8 量化** | `--kv-cache-dtype fp8_e5m2` | 并发 +71% | [kvcache-quant](dcu-perf-cases 案例21) |
| **权重 W8A8** | 自动 / `-q compressed-tensors` | 显存 1/2 | [quant](dcu-perf-cases 案例20) |
| **权重 W4A8** | `-q slimquant_w4a8_marlin` | 显存 1/4 | [quant](dcu-perf-cases 案例20) |
| **AWQ W4A16** | `-q awq` / `-q moe_wna16` | 显存 1/4 | [quant](dcu-perf-cases 案例20) |
| **MTP 投机解码** | `--speculative-config '{"method":"mtp",...}'` | 短中序列 +120% | [qwen35-mtp](dcu-perf-cases 案例25) |
| **aiter w4a16 GEMM** | `VLLM_AITER_W4A16_PATCH=1` | TPOT -25% | [gemma](dcu-perf-cases 案例13) |
| **flash fp8 KV** | `ATTN_FLASH_PREFILL=1` + `--kv-cache-dtype fp8_e5m2` | TPOT 再 -48% | [gemma](dcu-perf-cases 案例13) |
| **多 API server** | `--api-server-count N` | CPU 前端并行，端到端 -59% | [asr](dcu-perf-cases 案例17) |
| **权重转置** | `VLLM_ROCM_TRANSPOSE_WEIGHT=1` | decode -11.7% | [asr](dcu-perf-cases 案例17) |
| **锁频** | `rocm-smi --setsclk 6` | TPOT 回落 5-7ms | [gemma](dcu-perf-cases 案例13) |
| **language-model-only** | `--language-model-only` | 冗余 1.7% KV | [qwen35-lm](dcu-perf-cases 案例26) |
| **PP2+TP8** | `-pp 2 -tp 8` + NCCL 调优 | 双机吞吐 +1 倍 | [dsr1](dcu-perf-cases 案例28) |
| **Goodput 搜索** | `goodput-search.sh` | 找 SLO 甜点 | [goodput](dcu-perf-cases 案例18) |
| **lightop fused gate** | `VLLM_USE_LIGHTOP=1` + `VLLM_ENABLE_MOE_FUSED_GATE=1` | 绕开选错专家的 fused router，乱码修复 | [antangelmed](dcu-perf-cases 案例31) |
| **视觉 TRITON_ATTN** | `--mm-encoder-attn-backend TRITON_ATTN` | fp32 累加器，视觉精度修复 | [citic-flame](dcu-perf-cases 案例36) |
| **PIECEWISE cuda graph** | `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` | 规避大 batch paged attention VMFault | [vmfault](dcu-perf-cases 案例34) |
| **PD 分离** | `--disaggregation-mode prefill/decode` + router | 长上下文 TPOT≤25 吞吐 +317%~1362% | [dsv4-pd](dcu-perf-cases 案例37) |
| **NSA prefill CP8** | `--enable-nsa-prefill-context-parallel` | 16K+ prefill 吞吐 5-8×，TTFT 降至 12-19% | [dsv4-pd](dcu-perf-cases 案例37) |
| **fix v5 tokenizer** | evalscope `load_tokenizer_with_fix` | 跨组件 tokenizer 一致，空输出修复 | [dsv32](dcu-perf-cases 案例33) |

---

## 六、案例 → 原理文档 映射

每个案例末尾都有"关联文档"，指向 `docs/` 下的原理文档。核心映射：

| 案例主题 | 原理文档 |
|---------|---------|
| 进程模型 / API server / ZMQ | [vllm_engine_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_engine_guide.md) |
| 调度 / token budget / 抢占 / chunked prefill | [vllm_scheduler_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_scheduler_guide.md) |
| KV Cache / PagedAttention / 前缀缓存 | [vllm_kvcache_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_kvcache_guide.md) |
| 量化 / W8A8 / AWQ / W4A8 / FP8 | [vllm_quantization_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_quantization_guide.md) |
| 投机解码 / MTP / EAGLE / DFlash | [vllm_spec_decode_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_spec_decode_guide.md) |
| Attention 后端 / MLA / flash | [vllm_attention_backend_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_attention_backend_guide.md) |
| 编译 / CUDA Graph / 算子融合 | [vllm_compilation_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_compilation_guide.md) |
| 并行 / TP / PP / EP / 通信 | [vllm_parallel_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_parallel_guide.md) |
| MoE / 路由 / FusedMoE / EPLB | [vllm_fused_moe_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_fused_moe_guide.md) |
| KV 传输 / P-D 分离 / KVConnector | [vllm_kv_transfer_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_kv_transfer_guide.md) |
| Model Runner / V2 / 混合架构 | [vllm_model_runner_v2_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_model_runner_v2_guide.md) · [vllm_modelrunner_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_modelrunner_guide.md) |
| 总览 / TTFT / TPOT / 生命周期 | [vllm_production_guide](../../dcu-vllm-source/references/upstream-vllm-026/vllm_production_guide.md) |
