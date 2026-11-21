---
name: dcu-perf-cases
description: "DCU真实性能优化案例库：跨平台对标+根因定位+量化对比"
version: 1.1.0
author: DCU Engineer
license: MIT
---

# DCU 真实性能优化案例库

> 用法：本库是**案例沉淀**，不是方法论。分析方法论见 `dcu-perf-analysis` + `dcu-op-optimization`（强制量化对比、留痕、禁"感觉快了"）。每个案例按「现象 → 根因 → 优化动作 → 量化结果」四段式记录，方便横向对照同类问题。
>
> 扩展规范：新增案例成稿后写入 `references/` 对应主题的 cases-*.md（四段式不变），并在上方案例索引表登记一行（编号顺延）；跨平台对标须注明硬件/精度/框架版本，量化结果必须给出优化前后两组数。
>
> 溯源：案例原文 内部文档 file_id=<internal-file-id>（内部文档链接，已略）《LSTM-Unet-GNN模型性能对比分析(BW10/A10)》，作者<internal-author>。

---

## 0. 操作范围门（scope gate，执行前必读）

复现本库案例跑 benchmark 时，**默认只允许作用于"本会话自己启动"的资源**：

- **卡**：`hy-smi` 选空卡起服务/跑测试；卡被占时有界轮询或先问用户，**绝不杀他人的 vLLM/训练进程腾显存**。
- **端口**：起服务前 `ss -lntp` 查占用，只选空闲端口，不顶掉他人进程。
- **数据**：模型/数据集路径按各案例注明；不覆盖他人目录下的产物。
- 例外（须动他人资源才能完成任务）：逐次向用户报确切命令获批准后执行；长期成立的例外登记到下方白名单，其余情形仍按默认规则。

**例外白名单（经用户批准后逐条登记）**：暂无

## 1. LSTM 案例（小算子爆炸 / Persistent RNN 缺失）

**硬件**：海光 BW10 (32G) vs NVIDIA A10 (24G)
**现象**：LSTM 推理 BW10 吞吐 3438.92 vs A10 24125.73（A10 快 7-8 倍），耗时 14.89s vs 2.12s。

**根因（Kernel 级）**
- BW10：`hipExtModuleLaunchKernel` 调用 **880,881 次**，占 API 总耗时 90.8%；总 API 耗时 ~14.6s ≈ 总运行时间 → **CPU 侧 HIP API 调度成为绝对瓶颈**，GPU 常空等。
- A10：对应 `cudaLaunchKernel` 仅 **2,003 次**。
- 关键差异：NVIDIA 触发 cuDNN `RNN_blockPersist_fp_LSTM_HMMA`（持久化 RNN），把整条长度 1000 的序列在**单次 Kernel 启动**内算完 → 110 次×4 层 = 440 次调用。AMD 侧 MIOpen/rocBLAS 未触发时间维融合，按时间步展开 → 110×4×1000 = **440,000 次** GEMM + elemwise 调用（恰好 1000 倍差距）。
- 此外 BW10 的 `rnn_lstm_elemwise_fp16mix` 占 50%、`DropoutForward` 占 9.8%（基础算子库融合度低）。

**优化动作**：改写 HIP 后端实现类 NVIDIA 的 **Persistent RNN**（在 K100_AI 上测试，仓库 http://developer.sourcefind.cn/codes/zhengke/persistent_lstm_hip.git）。

**量化结果**：原始 LSTM.py 耗时 8-10s → 优化后 **3.08s**（进一步通用优化进行中）。

**可迁移结论**：BW10 上 LSTM 慢的主因不是算力，是**时间维融合缺失导致 kernel 启动次数爆炸**。遇到"API 调度占比≈总耗时"直接看 `hipExtModuleLaunchKernel` 次数。

> 补充（移动无线所实测，file_id=<internal-file-id>）：BW10 上 LSTM 进一步靠「替换 miopen package（MIOPEN_PACKAGE + LD_LIBRARY_PATH）」与「底层重写 persistent_lstm_hip（K100_AI 测 / bw10_adaptive_lstm_hip）」显著提升（bw10 优化后达 9086 vs 优化前 3370，37.6% of A10）；通用 LSTM 优化流程/组合测试见 `dcu-inference-optimization` §八、persistent kernel 写法见 `dcu-kernel-dev`。

---

## 2. UNet 案例（FP32 精度壁垒 + Launch Overhead）

**硬件**：BW10 vs A10，均 BS=16（A10 显存 24G 故 BS 从 32 降到 16 对齐）。
**现象**：DCU 与 A10 对比测试中，A10 吞吐约为 DCU 的 2.1 倍、耗时约为一半（量级对比，非固定基线）。

**根因**
- **精度壁垒（最高优先级）**：A10 默认 TF32（FP32 指数位保留+尾数位截断），自动吃 Tensor Core 红利；BW10 纯 FP32（`kme_implicitgemm_fp32_fwd_2d_nchw` 占 49.6%）。BW10 的 MIOpen 虽聪明地调了 Winograd（`winograd_MiOpen_FP32_3x3` 10.5%）但仍被 FP32 位宽拖累。
- **Launch Overhead 异常**：BW10 的 `hipExtModuleLaunchKernel` 耗时 **75.3s（占 API 时间 69.8%）**，而 `hipDeviceSynchronize` 仅 20.4% → CPU 下发算子成为拖慢因素（A10 反之，Synchronize 占主导说明 GPU-bound）。
- A10 承受约 11.5% `nchwToNhwcKernel` 布局转换代价，BW10 局部打包仅 ~2.1%。

**优化动作（软件层，不改模型）**
1. 开启 **AMP 混合精度** `torch.autocast(..., dtype=torch.float16)` → 强制 MIOpen 切 FP16 版本，峰值算力翻倍 + 访存减半（最关键一步）。
2. **channels_last** 内存排布 `to(memory_format=torch.channels_last)`。
3. **HIP Graph / CUDA Graph** 录制计算图回放，绕过 CPU 逐算子下发（解决 75s Launch Overhead）。

**量化结果**
| 配置 | 耗时(s) | 吞吐量(samples/s) |
|---|---|---|
| BW10 优化前 | 97.7 | 16.4 |
| BW10 优化后 | **33.9** | **47.2** |
| A10 优化前 | 46.7 | 34.2 |
| A10 优化后 | 20.8 | 76.9 |

**可迁移结论**：BW10 无 TF32 硬件红利时，**主动开 AMP** 是抹平差距最关键一步；CPU Launch Overhead 偏高用 Graph 回放解决。

---

## 3. GNN / GPT-2 案例（访存墙：Softmax/Elementwise 未融合）

**硬件**：BW10 vs A10。
**现象**：DCU 与对照卡对比：DCU 延迟高出约 45%，吞吐较低、实测算力较低、GPU 利用率较高（量级对比，非固定基线）。

**根因（GPU-bound，非调度）**
- 两侧 Host 端都是 `hip/cudaDeviceSynchronize` 占 ~94-96% → 典型 GPU-bound，瓶颈在算子执行。
- A10 侧：cutlass GEMM（Tensor Core）+ 大量 `s16816gemm_relu` **融合算子**（GEMM+ReLU 融合），Softmax 仅 11.4%。
- BW10 侧：**访存密集型算子成主瓶颈**——`softmax_warp_forward` 单算子占 **21.13%**（耗时第一），`elementwise_kernel_manual_unroll` 合计近 27%；真正 GEMM（Cijk_Alik 系列）被稀释。

**优化动作**
1. 引入 **FlashAttention**（或类似）：把 Q*K^T / Softmax / *V 融合为单算子，砍掉 21.13% 的 softmax 开销。
2. **算子融合（Graph Optimization / torch.compile / DTK 融合 pass）**：吃掉零碎 elementwise。
3. 替换基础算子库：elementwise 过高时可试海光优化 Triton 算子替代原生 PyTorch。

**可迁移结论**：DCU 跑 GPT-2 类慢的**核心不是 GEMM 算力，是访存墙**——Softmax/Elementwise 未融合导致频繁显存读写。优先 FlashAttention + 图优化。

---

## 4. 根因类型横向对照（速查）

| 根因类型 | 典型信号 | 首选解法 | 对应案例 |
|---|---|---|---|
| 小算子爆炸（融合缺失） | `hipExtModuleLaunchKernel` 次数≈总耗时，GPU 常空等 | Persistent/融合算子、Graph 回放 | LSTM |
| 精度壁垒 | 纯 FP32 硬扛，无 TF32/FP16 | 主动开 AMP（FP16） | UNet |
| 访存墙（Elementwise/Softmax 未融合） | Softmax/elementwise 占比异常高，GPU-bound | FlashAttention + 图融合 | GNN |
| CPU 稀疏算子链 / 拷贝事件爆炸 | GPU busy 占比低 + 拷贝占 GPU 活动大头 + 大量小 D2H/H2D 拷贝（单事件地板 ~4-5ms） | 算子结构融合（多表→单变量 + 单次 GPU gather），消除 CPU 往返 | TFRS 多热 embedding（案例 12） |

> 以上几类在 DCU 上出现频率极高，定位时先对号入座，再进 `dcu-hipprof` / `dcu-perf-profiling` 做 Kernel 级确认。

---

---

## 附录：并发压测通用方法论（去标识化，源自客户现场测试规范）

> 从客户现场并发压测规范提炼，**已剥离客户名/节点 IP/内部人名/具体基线数字**（原始项目记录不入库，见使用须知）。仅保留可复用的「方法」。

**测法（求极限并发）**
1. 固定输入档位（如 30k/60k/128k 输入 × 1k 输出），多轮（如 5 轮）测试，每轮记录每路 TTFT、TPOT，取最大/最小值。
2. **掐头去尾**：去掉各指标最大、最小的两个极端值，保留中间轮取均值（抗抖动）。
3. 按阈值（如 `max_ttft < 30s` 且 `min_tks > 15`）筛选能支持的最大并发。
4. 整机性能折算：单卡测 × 卡数 = 单台；TP=2 整机 ×4、TP=8 整机 ×1（按并行度折算）。

**判定口径**
- 性能 = 满足阈值条件下整机支持的**最大并发数**，不是单并发吞吐。
- 必须先和环境对齐（见 `dcu-env`）、再压测，否则并发数不可复现。
- 客户项目原始数据/基线表/日志**严禁写入 skill**，本附录只沉淀方法。

---

## 案例索引（§5 起正文已下沉 references/）

根因类型横向对照见上方 §4；各案例四段式正文按主题分组在 `references/`，编号与下沉前一致：

| 节 | 案例 | 正文 |
|---|---|---|
| §5 | Qwen3-32B SGLang 优化（GEMM 带宽瓶颈） | [references/cases-05-07.md](references/cases-05-07.md) |
| §6 | BW100 Qwen3.5-2B 多模态 vLLM（prefill 排队根因） | [references/cases-05-07.md](references/cases-05-07.md) |
| §7 | SGLang mamba cache size 陷阱（多模态并发被限流） | [references/cases-05-07.md](references/cases-05-07.md) |
| §8 | GLM-5.1 K100 AI INT8 部署（TileLang + Sparse MLA） | [references/cases-08-13.md](references/cases-08-13.md) |
| §9 | Qwen3-8B LoRA SFT 吞吐优化（Llama-Factory + Liger-Kernel，K100-AI×8） | [references/cases-08-13.md](references/cases-08-13.md) |
| §10 | K100 Qwen3.6-35B-A3B W8A8 单卡 Decode 优化（专用 INT8 算子 + 融合 + MTP3） | [references/cases-08-13.md](references/cases-08-13.md) |
| §11 | Qwen3.6-27B W8A8 K100AI 单卡 Decode 优化（R067 MTP4） | [references/cases-08-13.md](references/cases-08-13.md) |
| §12 | TFRS 推荐排序模型多热 Embedding 融合（TF Profiler + 事件数瓶颈） | [references/cases-08-13.md](references/cases-08-13.md) |
| §13 | BW10 gemma-4-31B AWQ 两阶段优化（aiter w4a16 GEMM + flash fp8 KV head512 + MTP） | [references/cases-08-13.md](references/cases-08-13.md) |
| §14 | 曙光 SHCA IB BW1000 双机 TP16 RDMA 调优（vLLM + Ray + MTP，GLM-5.2-int4） | [references/cases-14-16.md](references/cases-14-16.md) |
| §15 | Qwen3.8-27B W8A8 K100AI 三产品线（TP4/TP2/TP1）系统性长上下文调优 | [references/cases-14-16.md](references/cases-14-16.md) |
| §16 | Triton Server + ORT YOLOv7 BW100 单卡调优（调度层 × 算子层叠加） | [references/cases-14-16.md](references/cases-14-16.md) |
| §17-37 | vLLM 推理优化案例集（源自 vllm-book-poc-scheduler 文档库，已脱敏） | [references/cases-17-24.md](references/cases-17-24.md) |
| §17 | ASR 端到端性能优化（CPU 前端 vs GPU 正交轴） | [references/cases-17-24.md](references/cases-17-24.md) |
| §18 | Goodput 调优（SLO 约束下最大有效吞吐） | [references/cases-17-24.md](references/cases-17-24.md) |
| §19 | 显存占用 / KV Cache 拐点与并发上限 | [references/cases-17-24.md](references/cases-17-24.md) |
| §20 | 权重量化 W8A8 / AWQ / W4A8 | [references/cases-17-24.md](references/cases-17-24.md) |
| §21 | KV Cache FP8 量化（e5m2 软件模拟） | [references/cases-17-24.md](references/cases-17-24.md) |
| §22 | APC 前缀缓存性能影响 | [references/cases-17-24.md](references/cases-17-24.md) |
| §23 | max_model_len / mnbt 对 TTFT 的影响 | [references/cases-17-24.md](references/cases-17-24.md) |
| §24 | LMCache KV 卸载（CPU/NVMe offload） | [references/cases-17-24.md](references/cases-17-24.md) |
| §25 | Qwen3.5/3.6 MTP 适配与 APC 冲突 | [references/cases-25-33.md](references/cases-25-33.md) |
| §26 | language-model-only 与 KV 冗余 | [references/cases-25-33.md](references/cases-25-33.md) |
| §27 | EAGLE3 / N-gram 投机解码 | [references/cases-25-33.md](references/cases-25-33.md) |
| §28 | DS-R1 PP2+TP8 双机推理 | [references/cases-25-33.md](references/cases-25-33.md) |
| §29 | BW150 拓扑 / TP 加速 | [references/cases-25-33.md](references/cases-25-33.md) |
| §30 | 推理带宽使用分析（UMC 采集） | [references/cases-25-33.md](references/cases-25-33.md) |
| §31 | MoE 路由选错专家乱码（lightop 修复） | [references/cases-25-33.md](references/cases-25-33.md) |
| §32 | MiniMax-M3 AWQ INT4 适配（456B MoE） | [references/cases-25-33.md](references/cases-25-33.md) |
| §33 | DS-V3.2 压测空输出（tokenizer 覆盖） | [references/cases-25-33.md](references/cases-25-33.md) |
| §34 | 高并发 KERNEL VMFault（paged attention 越界） | [references/cases-34-39.md](references/cases-34-39.md) |
| §35 | 编译 FlashAttention 解决 FA VMFault | [references/cases-34-39.md](references/cases-34-39.md) |
| §36 | 视觉链路 SDPA 累加器精度（TRITON_ATTN） | [references/cases-34-39.md](references/cases-34-39.md) |
| §37 | DS-V4-Flash PD 分离 vs IFB 选型 | [references/cases-34-39.md](references/cases-34-39.md) |
| §38 | BW150/BW100 HBM 实测峰值与 QKNorm+RoPE naive kernel 带宽曲线（L2 thrashing 悬崖验证） | [references/cases-34-39.md](references/cases-34-39.md) |
| §39 | Qwen3.8-27B GDN prefill kernel 优化（vLLM fla/ops，QK 共享去冗余） | [references/cases-34-39.md](references/cases-34-39.md) |
| §40 | Qwen3.8-27B W8A8 K100-AI 四杠杆查表调优（SGLang 0.5.12，best_config 注入） | [references/cases-40.md](references/cases-40.md) |
| §41 | VideoJev-4B BW100 视觉 patch_embed conv3d→matmul（HF 推理路径，锁频 10x 假象） | [references/cases-41.md](references/cases-41.md) |

> 案例引用的外部文档链接在各 cases-*.md 的「关联文档」节内。

## 待追加案例（占位，后续扩展）

<!-- 新增案例：成稿写入 references/ 对应 cases-*.md（四段式），并在上方案例索引表登记；全新主题组新建 cases-<起止节号>.md -->

