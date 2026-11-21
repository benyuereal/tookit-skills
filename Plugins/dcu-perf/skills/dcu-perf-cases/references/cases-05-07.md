# 性能优化案例 §5-§7（早期典型案例）

> 2026-10-16 自 `perf/dcu-perf-cases/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），案例四段式原文照录、未改一字；编号与原文一致，索引见主 SKILL.md 案例索引表。

## 5. Qwen3-32B SGLang 优化（GEMM 带宽瓶颈）

**硬件/框架**：BW1000 ×2，SGLang 0.5.10rc0，Qwen3-32B（Dense，32B）。
**现象（基线）**：单并发较低速；随 batch 增大吞吐先升后断崖下跌（超 KV Cache 饱和区），bs 到峰值区后 OutputPerBS 已明显下降。

**根因（profiler，input=2K/output=512）**
- GPU busy 95%，decode step 大；与 bs=1 的几乎相同，等效每 token 成本减半 → **decode 是纯权重带宽瓶颈**（非算力）。
- 算子占比（每层每步）：`gate_up Cijk_MT64x16x128` 占比最高（带宽远低于额定）、`down`/`qkv`/`o_proj`/`paged_attention`/`allreduce_1stage` 依次递减（占比）。
- 带宽实测远低于额定 → 小 GEMM 带宽未吃满是主矛盾。

**优化动作 / 结论**
- 断崖下跌在 KV Cache 饱和区，先算清 `mem-fraction-static` 与并发的 KV 上限，别盲目加 bs。
- decode 带宽瓶颈类：方向是提升权重带宽利用率（更大 batch、fp8 权重、Aiter GEMM），而非堆算力。
- 实测框架级 Profiler 用法见 `dcu-framework-profilers`（SGLang/vLLM 段）。

> 溯源：源自《DCU性能分析及常用优化技术手册》（内部文档 file_id=<internal-file-id>）SGLang 实战章节。

---

## 6. BW100 Qwen3.5-2B 多模态 vLLM（prefill 排队根因）

**硬件/框架**：BW100 ×2（DP=2, TP=1, FLASH_ATTN, mbt=30000），vLLM 0.18.1，Qwen3.5-2B 微调多模态。
**现象**：平均延迟与 P95 均超标（目标 ≤4s）；HBM 实测带宽低于额定。

**端到端拆解**：prefill(TTFT) 与 decode 约各占一半 —— 两段对半，只优化一段最多拿一半。

**decode 阶段（batch=32, 单卡 9.40ms）**
| 组件 | ms | 占比 | 性质 |
|---|---|---|---|
| GEMM | 4.08 | 43% | 带宽受限 |
| full_attention | 2.17 | 23% | 计算受限 |
| linear_attention | 1.10 | 12% | 计算受限 |
| 其他(norm/silu/conv/采样) | 2.05 | 22% | 计算/已融合 |

GEMM 带宽效率 68%（小 GEMM 77%，理论 8.99ms vs 实测 9.40ms，差 4%）→ **GEMM 优化空间已耗尽**。已验证无效方向：`VLLM_USE_NN=False`(负收益)、w8a8(精度排除)、hipBLASLt(77%近顶)、aiter GEMM(BW100 无配置)。调度日志 3011 step：纯 DECODE 96.6% 走 FULL cuda graph（已最优）。

**prefill 阶段（根因）**
| 并发 | TTFT | 放大 |
|---|---|---|
| 单请求 | 0.148s | 1× |
| 64 并发 | 1.818s(均)/2.7s(p95) | 12-18× |

根因：**请求陆续到达 + prefill 串行排队**（非 mbt 不够，mbt=30000 足够）。64 请求在第 1 批前 35 步内陆续 prefill 进 running，后到请求等前面消化。vision encoder 单图 ~5ms（仅占 TTFT 3%，非瓶颈）；prefix caching 文本命中但图像 59% 不命中，cache 省计算但解决不了排队。

**已验证无效方向汇总**：GEMM 转置(负)、prefill 优先调度(占比≤2.6%)、w8a8(精度排除)、vision backend(<1ms)、小 GEMM 带宽(已近顶)、DP=4(GEMM 效率反降)。唯一未确认：`linear_attn aiter kernel`。

**可迁移结论**：多模态/高并发下 P95 超标，**先拆 prefill/decode 占比**，若 prefill 放大 12-18× 基本是排队而非算力；cuda graph 已最优时别在 decode 内耗，去解请求到达节奏/调度。

---

## 7. SGLang mamba cache size 陷阱（多模态并发被限流）

**硬件/框架**：BW100 / BW150，SGLang（0.5.10rc0 / 0.5.12），Qwen3.5/3.6 多模态。
**现象**：设了 32 并发、cuda graph bs=256，但**实际只能跑 ~20 并发，KV Cache 没跑满**。

**根因**：多模态模型含 mamba 结构，SGLang 默认 `max-mamba-cache-size` 自动分配过小，**在压测阶段悄悄限制了并发**（即使显式设了 `--max-running-requests 32` / bench `--max-concurrency 32`）。这是「配置缺省值陷阱」，不是算力/带宽问题。

**验证启动脚本（可复用模板，BW 系多模态 SGLang 实战参数）**
```bash
option=" --chunked-prefill-size -1"
option+=" --numa-node 0 0 0 0 0 0 0 0"
option+=" --mamba-scheduler-strategy extra_buffer"
option+=" --cuda-graph-max-bs 256"
model_path=/models/Qwen3.6-27B
# BW1100 用 fp8_e4m3，BW1000/BW150/BW100 用 fp8_e5m2（e4m3 差异较大）
option+=" --kv-cache-dtype fp8_e5m2"
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_ENABLE_SPEC_V2=1
python3 -u -m sglang.launch_server ${option} \
  --model-path $model_path --host 0.0.0.0 --port 8008 \
  --served-model-name Qwen3.6-27B \
  --mm-attention-backend fa3 \
  --speculative-algorithm EAGLE --enable-piecewise-cuda-graph \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tp-size 2 --pp-size 1 --attention-backend fa3 \
  --page-size 64 --mem-fraction-static 0.9 \
  --keep-mm-feature-on-device --max-running-requests 32
```
压测：
```bash
python3 -m sglang.bench_serving --backend sglang --host 0.0.0.0 --port 8008 \
  --model /models/Qwen3.5-27B --dataset-name random-ids \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 \
  --num-prompts 64 --max-concurrency 32 --request-rate inf
```

**优化动作 / 结论**
- 多模态 SGLang 压测务必**显式设 `max-mamba-cache-size`**（或确认其自动分配足够），否则并发被静默限流，现象像「KV 没满但跑不动」。
- 完整启动参数（Aiter/Marlin/fp8/mamba/EAGLE 投机）见上模板；SGLang 源码适配与参数细节见 `dcu-sglang-source`，部署调优见 `dcu-vllm-deploy`。

> 溯源：源自 DCU SGLang 多模态（Qwen3.5/3.6，含 mamba）部署实测（<device-A> 设备），与 dcu-vllm-deploy §4.2 / dcu-sglang-source 同源；无独立 内部文档 文档。

---
