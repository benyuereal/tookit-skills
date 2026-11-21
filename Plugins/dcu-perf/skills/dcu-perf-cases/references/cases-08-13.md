# 性能优化案例 §8-§13（部署/训练/量化调优）

> 2026-10-16 自 `perf/dcu-perf-cases/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），案例四段式原文照录、未改一字；编号与原文一致，索引见主 SKILL.md 案例索引表。

## 8. GLM-5.1 K100 AI INT8 部署（TileLang + Sparse MLA）

**硬件/框架**：海光 K100 AI（RDNA gfx928，64KB LDS / 120 CU / 支持 INT4·INT8 Tensor Core）×8（单机）/ ×16（双机）；vLLM，GLM-5.1（Sparse Attention + MLA）。

**背景**：GLM-5.1 推理链路核心是 Sparse Attention（Indexer 网络筛 top-k KV）+ MLA。Sparse MLA 每层三步：MQA Logits（非分页 Prefill / 分页 Decode）、Top-K 筛选、Sparse MLA Forward。其中三个算子在 K100 AI 无可用实现。

**根因（功能 + 性能双缺）**
- 功能缺：三个算子（MQA Logits 非分页 / Paged MQA Logits / Sparse MLA Forward）K100 AI 上无实现。
- 性能缺：PyTorch 组合算子方案在小 batch 下 launch overhead 超过计算本身；H=64 场景 FlashMLA 直接 OOM（其仅支持 head=128，专为 deepseek 设计）。

**优化动作（先 PyTorch 打通，再 TileLang 追性能）**
- 三个算子全用 TileLang 实现（Sparse MLA Forward / MQA Logits 非分页 / Paged MQA Logits），接入 vLLM `sparse_attn_indexer` 自定义算子，ROCm 路径切 TileLang。
- TileLang 设计决策（DCU 专属）：沿 **D 维度**分 tile（非 S）——K100 LDS 仅 64KB，MLA head_dim=128 整片加载会 spill 到 global memory；**Online Softmax** 避免全量 attention 矩阵存储；`num_split=4`（120 CU 下 4 达 92.78 TFLOPS，8 反而降约 12%，归约开销 > 并行收益）；`k_pack` 按 topk 分档（1024→2，2048→4）。
- 生产使能：INT4/INT8 量化（压带宽）+ CUDA Graph（`HIP_USE_GRAPH_QUEUE_POOL=1`）+ MTP 投机（`num_speculative_tokens=3`，实测 Accept Rate 60%–69%，每轮实际产出约 2.8–3.1 token）。

**量化结果（四段式，数值以源文档为准）**
- 端到端阶梯：基线 OOM → PyTorch 打通 → +TileLang（约 2.5x）→ +MTP（数量级提升，2.6x–3.2x）。
- 算子级：中小规模 TileLang 持平/优于 FlashMLA（规模较大时达 1.74x）；更大序列时 FlashMLA 反超约 20%（疑其对大序列 LDS bank conflict 处理更优）；特定头数场景 TileLang 正常运行（FlashMLA OOM）。
- MQA Logits 分页：中等 batch(B≥64) TileLang 与 lightop 持平/略优；小 batch(B=1) lightop 仍约 40% 优势（手写汇编 launch 路径极致优化）。
- 双机 vs 单机约 20% 折损，主要来自跨节点 AllReduce 通信开销。

**可迁移结论**
1. 国产硬件适配「效率优先于极限性能」：TileLang 开发周期从数周压到数天，性能达手写汇编 80%–108%；模型迭代快于硬件时，开发效率 > 5%–10% 极限差距。
2. Split-KV 是大规模并行有效策略，但 `num_split` 非越大越好——须实验找计算/归约平衡。
3. LDS 容量决定 tile 策略上限：64KB 须精算 Q/K tile；pipeline stage 与 tile 联动。
4. 全链路视角：kernel 优化仅贡献部分提升，量化 + Graph + MTP 贡献另大块；单点优化达不到生产可用。

> 部署 env（NCCL/RDMA/CUDA Graph/MTP 全套）、启动脚本（INT4/INT8 单机双机）、lower-pass bug 二分定位、paged_mqa_logits 完整代码 → 见 `dcu-kernel-dev`（§2/§5/§6）；TileLang 在 DCU 专属坑（warp_size=64 / backend=hip / k_pack / LDS）见同 skill §3·§4。
> 溯源：内部文档 file_id=<internal-file-id>（内部文档链接，已略），作者 王依涛。

---

## 9. Qwen3-8B LoRA SFT 吞吐优化（Llama-Factory + Liger-Kernel，K100-AI×8）

**硬件/框架**：海光 K100-AI ×8（单机 8 卡）；Llama-Factory v0.9.2；DTK 25.04.2（PyTorch 2.5.1，镜像 `image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk25.04.2-py3.10`）；模型 Qwen3-8B，LoRA SFT（lora_rank=8、batch=4、grad_accum=8、cutoff_len=2048、ds_z2、bf16）。

**背景**：在 DCU 上做 Qwen3-8B 的 LoRA SFT，对比「原生 transformers」「开启 Liger-Kernel」「Liger-Kernel + FA2」三者的训练吞吐。Liger-Kernel 通过 monkey-patch（`liger_kernel/transformers/monkey_patch.py`）把 HF 原始算子替换为 Triton 融合实现，对 Qwen3 共替换 4 处：① RoPE（`apply_rotary_pos_emb → liger_rotary_pos_emb`，融合 4 次矩阵乘+2 次加、不构造旋转矩阵、省显存）；② RMSNorm（`Qwen3RMSNorm → LigerRMSNorm`）；③ LM Head 前向+损失（`Qwen3ForCausalLM.forward → qwen3_lce_forward`，Fused Linear-Cross-Entropy，训练时不物化完整 logits）；④ SwiGLU MLP（`Qwen3MLP → LigerSwiGLUMLP`）。

**根因（为什么能提速）**
- 原生实现里 LM Head 先算出 `[B,L,V]`（V=词表巨大）的完整 logits 再算 CE，logits tensor 极占显存且访存密集；Liger 的 fused CE 把 lm_head 与 loss 融合，**训练时不物化完整 logits**，只在需要时算 → 显存与访存双降。
- RoPE / RMSNorm / SwiGLU 用 Triton 融合 kernel，减少中间 tensor 与 kernel 启动次数。
- `flash_attn: fa2` 进一步加速 attention 段。

**优化动作**
1. 装 Llama-Factory v0.9.2（`pip install .`）；装 Liger-Kernel：clone 后**改 `setup.py` 注释掉 `torchversion`**（DCU/特定环境避坑）再 `pip install .`。
2. 训练 yaml 开启 `enable_liger_kernel: True`；进一步试 `flash_attn: fa2`。

**量化结果（四段式，数值以源文档为准；下为作者实测，随硬件/驱动/模型/批次变化，仅供参考、非固定基线）**
| 配置 | 训练吞吐 (samples/s) | 相对原生提升 |
|---|---|---|
| 未使用 Liger-Kernel | 基准值（低） | 1.00× |
| 使用 Liger-Kernel | 中位值 | 提升（1.1× 量级） |
| Liger-Kernel + FA2 | 高值 | 提升更显著（1.3× 量级） |

> 注：源文档「训练效率 / 显存占用」两处对比图为 `![image]()` 空占位，未给出完整曲线与显存绝对数；上表为文档明确写出的吞吐三组数。显存绝对降幅、长序列下的增益幅度需在本机复测确认（Liger fused-CE 理论上主要降显存，吞吐增益来自 RoPE/Norm/SwiGLU 融合 + FA2）。

**可迁移结论**
1. DCU 上 Llama-Factory 做 LLM SFT，**开 `enable_liger_kernel: True` 是零代码改动的一行收益**（吞吐约 +15%）；叠加 `flash_attn: fa2` 可达约 +37%。
2. Liger-Kernel 的收益主要来自 Triton 融合算子（尤其 fused CE 省显存）——DCU 上需确认 Triton 后端可用（DTK 25.04.2 / torch 2.5.1 是否带可用 Triton）；FA2 在 DCU 上需 DTK 兼容层或 ROCm 版。
3. 安装 Liger-Kernel 的 `setup.py` 注释 `torchversion` 是 DCU 环境实测踩坑点，务必做。
4. 该优化属「训练吞吐/显存」侧，与既有推理侧案例（LSTM/UNet/GNN/GLM-5.1 部署）互补；排查训练瓶颈可联动 `dcu-perf-profiling` + `dcu-op-optimization`。

> 溯源：内部文档 file_id=<internal-file-id>（内部文档链接，已略）《Llama-factory大模型微调优化方案》，作者未署名（团队内共享优化总结，非客户交付物）。

---

## 10. K100 Qwen3.6-35B-A3B W8A8 单卡 Decode 优化（专用 INT8 算子 + 融合 + MTP3）

**硬件/框架**：海光 K100 AI（RDNA gfx928，64KB LDS / 120 CU / INT4·INT8 Tensor Core）×1 单卡；vLLM 0.18.1（镜像 `<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`，DTK 26.04）；模型 `Qwen3.6-35B-A3B-W8A8-DCU`（compressed-tensors W8A8：权重 INT8 per-channel、激活动态 INT8 per-token）。目标画像：**单卡 / TP=1 / 单并发 c1 / Agent 交互速度**。

**背景（为什么要做 + 测速口径纪律，这是本案例最该记住的一点）**
- 这是一次「从最原始版本到优化版本」的完整演进，核心价值在于证明：**即使完全关掉 MTP，单卡也已远快于原版**——不能把全部提速归因 MTP。
- 必须先区分两个「原版基线」，否则所有对比数字都被污染：
  - **6月旧报告原始口径**（含早期冷路径/测速口径差异）：单并发 c1、TP1、无 MTP，速度较低（低速口径）。
  - **8月重建的「热态可比基线」**：单卡 / MTP关 / 无自定义补丁 / 但启用原版 vLLM 自带 compile + CUDAGraph / 固定 128 输出 token / 热态 = 高基数值。
- 结论：对外答「6月报告原版速度」用低速口径值；对外答「统一测速条件下的优化基线」用高基数值。两者都不是固定基线，随硬件/镜像/批次变化。

**现象（演进路线，数值为该次记录值、非固定基线；只给优化前后对照，不贴原始日志）**
| 阶段 | 配置 | 单并发 c1 速度 | 备注 |
|---|---|---|---|
| 热态原版基线 | 原版 vLLM + 原版 W8A8，MTP关 | 高基数值 | 公平可比基线 |
| R180（纯模型优化，无MTP） | K100 AI 专用INT8 Linear/MoE/lm_head | 中位速 | 完全不依赖MTP |
| R184（纯模型 + QKVZ+BA融合） | 上 + 双指针融合 kernel | 中位速（再 + 少量） | 纯模型再提升 |
| R184 + 模型原生 MTP3 | 上 + qwen3_next_mtp, n_spec=3 | 高速（稳定/高接受率更高/峰值更高） | 高接受率负载更高 |

演化逻辑：**原版低速 → K100 AI 专用算子把纯模型做到中位速 → 在底座上叠加MTP3 → 稳定高速、高接受率更高**。MTP 贡献的是「最后一段」，不是全部。

**根因（kernel 级，原版为什么只有 ~20）**
- 原版 vLLM 的 INT8 小 GEMM（Linear / MoE / lm_head）走**通用 heuristic 选 Triton config**，没有针对 K100/gfx928 与 Qwen3.6 单并发 decode 的真实形状（M=1～4、K/N 固定）专门搜索。
- 两个主导大投影在 vendor heuristic 下严重偏慢（R174 实测，relative_l2=0 保证 bit-identical）：
  - `2048 → 12288`（带宽/proj 大投影）：vendor 默认配置 → 专用 config（**约 4×** 提速）
  - `2048 → 9216`：vendor 默认 → 专用 config（**约 2.9×**）
- 词表 248,320 的 lm_head（`248320 × 2048`）本身每 token 是大瓶颈，原版未做 K100 AI 专用 INT8。
- MoE（256专家 / 激活8 / E=512 / N=512）走通用 MoE tile，在 gfx928 上有浪费。
- Qwen3.6 大量使用 Gated DeltaNet：QKVZ 与 BA 两次独立投影各做一次量化+launch+写回，可融合。

**优化动作（可迁移的方法论，按作用排序）**
1. **先对齐测速口径**（最高优先级）：任何优化前后对比前，先固定输入/输出 token、单并发、热态、统一是否启用 compile/CUDAGraph，再比数。否则「旧报告低速口径 vs 新高基线口径」这类差异会污染所有结论。
2. **Shape-specific Triton config 搜索**：针对 Qwen3.6 decode 的真实 (M,K,N) 精确形状（含 M=1/2/3/4/5 各档）搜 Triton `bm/bn/bk/warps/waves`，并以 `relative_l2==0` 为门禁确认与原版 bit-identical。**只替换 decode 形状（M ≤ MAX_M，默认 1~4），其余 fallback 上游 heuristic**——保留安全网。
3. **运行时 monkey-patch 而非改 vLLM 源码**：通过 `PYTHONPATH` 注入 `sitecustomize.py`，hook 以下四处，对 decode 形状注入 K100 AI 专用路径、其余走上游：
   - `triton_scaled_mm` → 专用 INT8 Linear（`K100_SC_LINEAR_A8W8=1`）
   - `ParallelLMHead.process_weights_after_loading` → 专用 INT8 lm_head（`K100_LM_HEAD_W8A8=1`，M=1~5 搜 config）
   - `CompressedTensorsW8A8Int8MoEMethod.apply` → native INT8 MoE（`K100_NATIVE_INT8_MOE=1`）
   - `Qwen3_5GatedDeltaNet.forward` → QKVZ+BA 融合（`K100_FUSED_QKVZ_BA=1`）
4. **专用 INT8 MoE kernel（native HIP）**：`build_k100_channelwise_int8_moe.py` 编译 AITER 的 `moe_w8a8_block_wise_hip.h` 模板，**patch 其 final scale 实现为 true per-output-channel（group_size_n==1）**（上游把多 channel scale 塌缩成标量，仅在 128 channel 共享 scale 时正确）；gate_up 用 `run_moe_w8a8_gemm_block_wise<half,8,8,16,64,128,...>`、down 用 `<...,8,1,16,64,128,...>`，固定 template 参数吃满 K100 AI 形状。
5. **QKVZ+BA 双指针融合 W8A8 kernel**（R184）：把两次独立投影合并为一次 fused scaled_mm（N_QKVZ=12288 对齐 BLOCK_SIZE_N，branch 内只建一棵指针树），减少重复量化/launch/中间写回，~1~2% 收益且不动 KV cache。
6. **compile + CUDAGraph 针对单并发重配**：明确目标「K100 单卡 / c1 / Agent 交互」→ `TP=1`（免多卡通信）、`-cc.mode=3`、`--cudagraph-capture-sizes 1`（无 MTP）/ `1 4`（MTP 额外捕获 4-token 验证路径）、关 `combo_kernels`（`-cc.inductor_compile_config '{"combo_kernels": false}'`，淘汰对单并发无收益的组合 kernel）。
7. **最后才叠加模型原生 MTP3**：`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3,"quantization":"compressed-tensors"}'`，在已优化底座上叠加，并继续针对 `M=4` 验证路径优化 Linear/MoE/lm_head。
8. **一致性门禁纪律**：所有 kernel 替换必须经固定输出/SHA256/logprob 一致性验证（`relative_l2=0`、bit-identical），**不能只看裸 kernel 微基准**；首次请求因 Triton 编译变慢须单独标记剔除，但不得删后续稳定结果；同卡连续测试（禁用不同 GPU 一次性结果替代）。

**关键环境变量速查（运行时开关，决定是否注入专用路径）**
| 变量 | 作用 | 取值 |
|---|---|---|
| K100_SC_LINEAR_A8W8 | 专用 INT8 Linear 替换 | 1 开 |
| K100_SC_LINEAR_A8W8_MAX_M | Linear 仅对 M≤该值替换，其余 fallback | 4 |
| K100_LM_HEAD_W8A8 | 专用 INT8 lm_head | 1 开 |
| K100_LM_HEAD_W8A8_MAX_M | lm_head 专用路径 M 上限 | 4 |
| K100_NATIVE_INT8_MOE | native INT8 MoE | 1 开 |
| K100_NATIVE_INT8_MOE_MAX_M | MoE 专用路径 M 上限（decode） | 32 |
| K100_FUSED_QKVZ_BA | QKVZ+BA 融合（需 R184 patch） | 1 开 |
| K100_TRACE_LINEAR_SHAPES | 打印 (M,K,N) 调用分布（诊断用） | 1 开 / 0 关 |

**量化结果**：见上「现象」演进表（含 MTP3 后四档平均与峰值均更高）。

**可迁移结论**
1. **单并发 decode 是「固定小 shape 问题」，通用 heuristic 浪费极大**：针对真实 (M,K,N) 搜 config 即可白拿数倍提升，且零精度风险（门禁保证 bit-identical）。
2. **monkey-patch（sitecustomize + 环境变量开关）是 DCU/K100 AI 上注入专用 kernel 的最干净手段**：不改上游、保留 fallback 安全网、可按形状精确开关；这套「补丁按环境变量默认关、只对 decode 形状开」的模式建议作为 DCU 推理优化的标准接入范式。
3. **lm_head / MoE / 融合 kernel 这类「非 GEMM 主体」的零碎开销，在超大模型上累积可观**，优先搜它们往往比死磕主体 GEMM 性价比更高。
4. **MTP 是「加速器」不是「创造者」**：先把纯模型（无 MTP）做到接近理想，再叠 MTP，才说得清收益来源；把全部提速归因 MTP 会掩盖底座优化价值。
5. **测速口径必须先对齐**：旧报告冷路径/旧口径数字与统一热态基线可比性差，对外汇报务必区分「旧报告原始口径」与「热态可比基线」。
6. **复现/交付安全边界**（来自 CODEX 复现包，建议所有 DCU 优化交付照做）：只动指定 GPU、不动生产服务、每次只跑一个候选、一致性失败立即判失败、同卡连续测试、首编译请求单独标记剔除。

> 溯源：本地复现工程包 `<local_repro_pkg>_2026-08-07.gz`（含 `runtime_patch_r180_nomtp` / `patches/runtime_patch_r184_qkvz_ba_fused` 两个 sitecustomize、native_ext（k100_channelwise_int8_moe.hip + build 脚本）、MoE 调优配置 `config/E=256,N=512,device_name=K100_AI.json`、启动/验收/测速脚本）；配套交接文档 `<local_repro_doc>_副本.md`（复现目标、安全边界、验收判定）。专用 INT8 MoE kernel 写法 / AITER 模板 patch 见 `dcu-kernel-dev`；vLLM W8A8 部署见 `dcu-vllm-deploy`；MTP 投机与端到端压测见 `dcu-llm-benchmark`。

---

## 11. Qwen3.6-27B W8A8 K100AI 单卡 Decode 优化（R067 MTP4）

**硬件/框架**：海光 K100 AI（RDNA gfx928，64KB LDS / 120 CU / INT4·INT8 Tensor Core）×1 单卡；vLLM 0.18.1（镜像 `<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811`，DTK 26.04）；模型 `metax-tech/Qwen3.6-27B-W8A8`（compressed-tensors W8A8：权重 INT8 per-channel、激活动态 INT8 per-token；上游 lm_head 仍 BF16，本路径加载后额外量化为 INT8）。目标画像：**单卡 / TP=1 / 单并发 c1 / Agent 交互速度**。

**背景（为什么要做 + 测速口径纪律）**
- 这是公开 repo `DocPang/qwen36-27b-k100ai-w8a8-optimization` 的 **R067 冠军路径**。完整可复现，从"原版 ~3.2 tok/s → 缺省 MTP1/真实口径 ~19.2 → shape 感知 + MTP4 → 42.2978 tok/s"演进，核心价值是证明：**单卡 W8A8 decode 是小 shape 问题，用 shape 感知的 runtime patch 就能白拿数倍提升**。
- **原版基线有两种口径需要分开，否则对比数字被污染**：
  - stock W8A8（无任何优化）：约 **3.23 tok/s**（README 基线 magnitude）。
  - R026（公开 no-MTP 对照，热态可比）：约 **19.21 tok/s**（独立可复现路径）。
- 对外汇报要先说清口径：固定 512/单并发/热态/是否 compile+CUDAGraph 统一后，才做版本间对比。

**现象（演进路线，数值为该次记录值、非固定基线；只给优化前后对照，不贴原始日志）**
| 阶段 | 配置 | fixed-512 结果 | 备注 |
|---|---|---|---|
| stock W8A8 | 上游原版，无优化 | ~3.23 tok/s | 基线 magnitude |
| R026 | 上游 + 专用 INT8 Linear/lm_head/native GEMV，no-MTP | ~19.21 tok/s | no-MTP 对照 |
| R006 | + MTP3，旧公开路径 | 35.0339 tok/s median | 旧 MTP3 基准 |
| R064 | + MTP4（两段 selector 并入，R067 前身） | 42.1584 tok/s median | 已接受的 selector 基 |
| **R067** | + MTP4 + 1024-d/Top2K 两级 selector | **42.2978 tok/s median** | 当前冠军，六轮 42.4101/42.2421/42.2704/42.3055/42.2901/42.3168 |
| R068/R069 | + MTP5（R067 后探） | 40.9141 / ~40.59 tok/s | 被拒：verifier 成本涨，吞吐反降 |

**根因（kernel/shape 级，原版为什么慢）**
- 原版 vLLM 的 INT8 小 GEMM 走**通用 heuristic 选 Triton config**，没有针对 gfx928 与 Qwen3.6 单并发 decode 的真实 `(M,K,N)` 精确形状（M=1~5、K、N 固定）专门搜索 → 几个大投影在 vendor heuristic 下严重偏慢。
- 两种核心矛盾：
  - **词表 248,320 的 lm_head** 每 token 是大热点；上游 lm_head 仍 BF16，动态 INT8 量化后才是真提速点。
  - **两级 draft selector**：MTP draft 若每一步都完整跑 5120→248320 的 lm_head，成本爆炸；R067 用 1024-d 子空间 coarse Top2K 后再 full-K 打分，把 draft proposal 成本大降，但**final token 仍过 target verifier**（不是永久词表裁减）。
- MTP4 / MTP5 取舍：MTP 是投机解码，**接受率内容依赖**。R067 的 MTP4 验证路径 M=5 单独调优；MTP5 在当前成本模型重复循环但价更高，被拒。

**优化动作（可迁移的方法论，按作用排序）**
1. **先对齐测速口径**（最高优先级）：固定 512 / 单并发 / 热态 / 统一 compile+CUDAGraph 后，再比数。否则"3.23 vs 19.21"这类口径差异污染结论。
2. **真实 runtime shape 驱动 kernel 搜索**：从 Qwen3.6 runtime packed shape 出发（MLP gate/up `5120→34816`、MLP down `17408→5120`、GDN qkv+z `5120→16384`、GDN b+a `5120→96`、attention q+k+v `5120→14336`、GDN/attention out `6144→5120`、MTP fc `10240→5120`、lm_head `5120→248320`）搜 Triton config，以 `relative_l2==0` 门禁确认 bit-identical。**只替换 decode 形状（M ≤ MAX_M），其余 fallback 上游**。
3. **runtime monkey-patch（sitecustomize + 环境变量开关）**：`PYTHONPATH` 注入，hook 专用 INT8 lm_head（`K100_27B_LM_HEAD_W8A8=1`）+ 两级 selector（candidate_head.py）。default 关、只对 decode 形状开、其余上游 fallback——这是 DCU 单卡 decode 优化的标准接入范式。
4. **native INT8 GEMV**：`k100_int8_gemv_v6.hip` 本地 hipcc 编译（不发布预编译 .so，因 ABI/编译环境不同），对 `6144→5120` out 投影等专门化。源码公开，用 HCU buffer 资源 + `buffer_load_dwordx4` + 手控 `vmcnt` + 双 accumulator 链 + `sdot4`。
5. **应用已验证 DCU config 而非动 safetensors**：`apply_dcu_config.py` 校验上游/DCU 两个 SHA256 后覆盖 config.json，备份上游，不改权重文件。这是安全交付纪律——**修改 config 不算改模型**。
6. **首编译请求单独标记剔除**（一致性门禁）：Triton 首次编译变慢单次请求；后续稳定结果不得删来美化基线。
7. **MTP 最后叠加**：在纯模型优化底座上再开 MTP4（`method=qwen3_next_mtp`，`num_speculative_tokens=4`），并继续针对 M=5 验证路径调优——MTP 是"加速器"不是"创造者"，先纯模型接近理想再叠慢才说得清收益。

**关键环境变量速查（R067 运行时开关）**
| 变量 | 作用 | 取值 |
|---|---|---|
| K100_27B_LM_HEAD_W8A8 | 专用 INT8 lm_head | 1 开 |
| K100_27B_LM_HEAD_W8A8_MAX_M | lm_head 专用 M 上限（M=1 快路径 / M=5 精确） | 4 |
| K100_27B_LM_HEAD_W8A8_M5_CFG | M=5 verifier 精确路径 tile | 见 sitecustomize 头部注释 |

**量化结果（四段式，数值以 README/results 为准，非固定基线）**
- **R067 六轮 fixed-512**：六轮 tok/s 高度一致（median/mean 均约为 42.3 量级）；六轮 final SHA256 均一致。
- **R067 相对 R006 MTP3 约 +20.7%**；相对 stock 约 13×。
- **两级 selector 微基准**：K1024/N248320 老 config → R067 BN64/kpack2 新 config（约 1.3×）；748 个真实 draft hidden state 上 748/748 selector logits bitwise equal、Top2K set 相同（证明新 geometry 没改 proposal trajectory）。
- **数值边界说明**（本案例最关键的交付纪律）：加载时把上游 BF16 lm_head 量化为 INT8 是**显式数值变化**，不宣称与上游全局 bitwise identical；但 R067 的 selector 调度保持 R064 已接受的 proposal/acceptance trajectory，fixed benchmark 最终 SHA 一致。

**可迁移结论（跨项目通用）**
1. **单并发 decode 是「固定小 shape 问题」**：通用 heuristic 浪费极大。先 trace 真实 `(M,K,N)`，搜 shape-specific Triton config，白拿数倍提升，且用 `relative_l2==0` 门禁零精度风险。
2. **runtime monkey-patch（环境变量开关 + fallback 安全网 + 形状超精准）是 DCU K100 AI 注入专用 kernel 的最干净手段**：不改 vLLM 源码，可按形状精确开/关，其余自动上游——这是**标准接入范式**，参考 `dcu-vllm-runtime-patch`。
3. **lm_head / MoE / 融合 kernel 这类非 GEMM 主体的零碎开销在超大模型上累积可观**：优先后搜它们性价比更高——尤其词表巨大时 lm_head 每 token 是大热点。
4. **MTP 是「加速器」不是「创造者」**：先把纯模型（无 MTP）做到接近理想（R026 ~19.2 → 专用优化），再叠 MTP4 才说得清收益；本案例 R067 相对 R006 提升 20.7% 大部分来自底座优化而非 MTP 本身。
5. **测速口径必须先对齐**：stock W8A8（3.23）与 R026（19.21）是两种原版口径，对外汇报务必区分；固定 benchmark 用于版本间对比，网页随机 prompt 的 MTP 数字只能看应用行为。
6. **交付/复现安全边界**（CODEX 纪律）：固定镜像+模型+SHA 校验、只动指定 GPU、不动生产服务、每次只跑一个候选、一致性失败立即判失败、同卡连续测试、首编译请求单独标记剔除、build native 扩展在目标 K100AI 本地编译。

> 溯源：公开 repo `DocPang/qwen36-27b-k100ai-w8a8-optimization`（R067 主文件：README、patches/r067_mtp4/sitecustomize.py + candidate_head.py、docs/R067_RELEASE_NOTES.md + OPTIMIZATION_NOTES.md、results/RESULTS.md + benchmark_summary.json、configs 与 scripts/native）。补丁 complete 版已收进 `dcu-vllm-runtime-patch` references（r067_mtp4_qwen36_27b 系列）；方法论跨项目结论已并入 `dcu-vllm-runtime-patch` §11/§12。

---

## 12. TFRS 推荐排序模型多热 Embedding 融合（TF Profiler + 事件数瓶颈）

> ⚠ **该案例为 sketch/待完善**：源自 2026-08-19 用户截图，原文被截断，硬件/框架版本、item 1、量化最终绝对数均未转录完整（见下方各「待补」标注）。案例结构完整可作思路参考，但**勿当作完整基线案例复用**。

**硬件/框架**：DCU（型号待补）；TensorFlow（版本待补）+ TFRS（TPUEmbeddingForServing）；模型为官方 recommendation/ranking 排序模型，26 张多热 embedding 表。

**现象**：每步 0.325s，GPU 活动仅 ~140ms（原文写「29%」，与 140/325≈43% 对不上，按 43% 计并注明待核——**待核**）。

**根因（tf.profiler.experimental 150 步 trace + hipprof 量化）**
- 每步 0.325s 中 GPU 活动仅 ~140ms，其中 MemcpyDeviceToHost ~71ms/步、MemcpyHostToDevice ~18ms/步——拷贝占 GPU 活动大头；
- 每步 ~340 次 D2H 拷贝，tf_op 统计指向 embedding 链路的 CPU 算子（SparseFillEmptyRows 等）——TFRS TPUEmbeddingForServing 对 26 张多热表走 `safe_embedding_lookup_sparse → SparseFillEmptyRows → SparseSegmentMean(GradV2)`，数据 GPU→CPU 往返；
- Host 侧 SparseSegmentMean(GradV2) 链 CPU 执行 ~0.4s/步（各线程累计）；
- 事件时长地板 ~4-5ms/事件（DCU TF 栈每 kernel/拷贝固定开销）——**事件数量是决定性因素**，瓶颈在设备侧事件数量与 CPU 稀疏算子链，须从算子结构层面优化。

**优化动作（2 项生效，item 1 待补）**
1. **Item 0：融合多热 embedding（核心，~4.4×）**
   - 新增 `vendor_patch/fused_embedding.py`：`FusedEmbeddingLayer`——26 张表合并为单变量 `[sum(vocab)=4.08M, 128]`（2.0GB），一次 `tf.nn.embedding_lookup`（GPU gather）+ 按固定 multi_hot 偏移 `split + mean`；语义与官方 `combiner='mean'` 一致（每行非空、无除零），梯度单次 scatter 回写，无 CPU 往返；
   - 容器内修改 `official/recommendation/ranking/task.py`（备份 `vendor_patch/task.py.orig`）：`build_model` 中非 partial-tpu 分支改用 `FusedEmbeddingLayer`（`use_partial_tpu_embedding: true` 分支保持官方原样）；
   - 容器内修改 `official/recommendation/ranking/data/data_pipeline_multi_hot.py`（备份 `vendor_patch/data_pipeline_multi_hot.py.orig`）：合成分支把 26 张多热表 ids 按列合并为单张稠密 `int32` `[B, sum(multi_hot)=215]`（生成序列与官方逐表 uniform 完全一致），每步拷贝数 78 → 3，消除全部 CPU 稀疏算子。
2. **Item 1：（待补——原截图被截断，此项未显示）**

**量化结果（部分待补）**
- Item 0 单项：~4.4×；
- 两项合计：~7-8×（用户口述，最终每步耗时 / 吞吐绝对数待补）；
- 每步拷贝数：78 → 3。

**可迁移结论**
1. **「事件数瓶颈」与小算子爆炸 / launch overhead 类（LSTM 案例）同族，但事件来源不同**：LSTM 案例事件来自时间维融合缺失（kernel launch 爆炸），本案例事件来自 **CPU 稀疏算子链 + 大量 D2H/H2D 拷贝**。共同信号是「GPU busy 占比低 + 事件数决定总时长」；解法不同——LSTM 用 persistent kernel，本案例用算子结构融合。
2. **CPU 稀疏算子链（SparseFillEmptyRows / SparseSegmentMean）是事件爆炸源**：TFRS TPUEmbeddingForServing 的多热 embedding lookup 走 CPU 稀疏算子链，数据 GPU→CPU 往返。解法不是加速单算子，而是**重构算子结构**——多表合并单变量 + 单次 GPU gather + split/mean，消除 CPU 往返。
3. **数据侧必须同步改**：26 张 id 表合并为单张稠密 `[B,215]`（生成序列与官方逐表 uniform 完全一致）是拷贝数 78→3 的前提——只改模型侧不改数据管线，拷贝消不掉。
4. **语义一致性论证**：融合实现须证明与官方 `combiner='mean'` 语义一致（每行非空、无除零）+ 梯度单次 scatter 回写——这是「结构优化」类正确性论证的关键。

> 溯源：用户提供截图（2026-08-19），原文被截断，item 1 / 最终数字 / 硬件 / 框架版本待补。

## 13. BW10 gemma-4-31B AWQ 两阶段优化（aiter w4a16 GEMM + flash fp8 KV head512 + MTP）

**硬件/框架**：海光 BW10（gfx936，48 CU，32GB/卡）×4（TP4，生产用卡 0,4,2,3）；vLLM 0.23.0 DCU 定制版（0.23.0+das.48722a8.dtk2604）+ aiter DCU 定制版（0.1.3+das.dtk2604）+ torch 2.10.0；主模型 `gemma-4-31B-it-AWQ-4bit`（compressed-tensors AWQ uint4，group_size=32，asymmetric），draft 模型 `gemma-4-31B-it-assistant`（MTP，model_type=gemma4_assistant）。目标画像：**TP4 生产部署，精度无损**。

**背景（两阶段叠加，互不冲突，分别攻破两个瓶颈）**
- 瓶颈定位方法：vllm profiler 抓 decode step 算子耗时分布——`awq_gemm` 占比最高（第一瓶颈，阶段一解决）；attention 占比小，但 draft 侧 head_dim=512 的 full_attention 走 aiter 2D kernel 无 tuned config，单次调用慢（draft 侧性能洼地，阶段二解决）。**改动前先确认"改的算子确实是瓶颈"，避免无效优化。**
- 模型结构关键点：主模型 60 层，hidden=5376，GQA 32:16，head_dim=256；MTP draft 有一层 full_attention 的 global_head_dim=**512**（不是主模型的 256）、num_global_key_value_heads=4——这一层是阶段二优化重点。

**现象（演进路线，数值为该次记录值、非固定基线）**
| 阶段 | 配置 | TPOT | 相对 baseline | 吞吐 | MTP 接受率 |
|---|---|---|---|---|---|
| baseline | vllm 原生 triton w4a16 | 基准耗时长值 | 1.00x | 基准吞吐低值 | 基准精度 |
| 阶段一 | aiter w4a16 GEMM（tuned config） | 耗时段 | 1.34x | 吞吐中值 | 精度略高 |
| 阶段一+二 | 叠加 flash fp8 KV + head512 | 耗时低值 | 2.59x | 吞吐高值 | 精度基本不变 |

测试条件：TP4，batch 4，input 5120 / output 1024，MTP num_spec=3。精度无损：HumanEval pass@1 与基线持平，离线算子级 cos_sim = 1.000000。

**根因（kernel 级）**
- 阶段一：vllm 自带 triton w4a16 kernel 在 BW10 上性能不佳；aiter 有更快的 triton w4a16 kernel，但接入需解决权重重排、量化兼容、torch.compile 兼容。
- 阶段二：aiter 2D attention 对 head_dim=512 无 tuned config（单次调用慢）；flash-attention-cutlass 的 fp8 mixed kernel 原本不支持 head_dim=512（入口 TORCH_CHECK 限制 + gfx938 的 fp8 GEMM utils 不支持 512 会 static_assert）。

**优化动作（可迁移的方法论，按作用排序）**
1. **阶段一：aiter w4a16 GEMM 替换 vllm triton kernel**（patch 三段合一：vllm `triton_w4a16.py` + aiter `gemm_a16w4.py` + 10 个按 gemma4 各层 shape 调优的 config json）。关键技术点：
   - **GPTQ→AWQ 列重排（精度关键）**：vllm AWQ 权重是 GPTQ int32 打包 [K,N//8] 顺序 [0..7]，aiter 用 `AWQ_REVERSE_ORDER=[0,4,1,5,2,6,3,7]` 解包；送入前必须按 `AWQ_INV=[0,2,4,6,1,3,5,7]` 重排列。不做 → cos_sim=0.25（完全错）；做了 → cos_sim=1.000000。
   - **对称量化（uint4b8）兼容（精度关键）**：aiter dequant 算式 `(w - zeros) * scales` 必须有 zeros 张量，但对称量化无显式 zp（零点恒为 8）。解法：加载时造全 8 的 zp 张量（8 个 4bit zp pack 进 int32 = 0x88888888）；asymmetric 模型用真实 zp。
   - **torch.compile/cudagraph 兼容（性能关键）**：aiter triton driver 调用 inductor 编译不了。用 `torch.library.custom_op` + `register_fake` 包成注册算子，对 dynamo/inductor/cudagraph 全透明——opt-level 3 + cudagraph 49 张图全捕获成功，无 graph break。
   - **原始 qweight 释放（省大量显存/卡，解 TP1 OOM）**：repack 成 aiter 格式（aq/az）后原始 qweight+qzeros 不再需要，repack 完立即 `replace_parameter(layer, w_q_name, None)` 释放。TP4 每卡省内存；TP1 单卡受限显存装 31B 时这正是 OOM 与不 OOM 的区别。
   - **bf16 模型 + `--dtype float16`（兼容关键）**：aiter kernel 硬编码 fp16 输出，bf16 权重不处理会 `Half != BFloat16` 崩溃（尤其 compile/cudagraph 下）。零 kernel 改动，精度无损（gemma4 scale max=2.1875、权重 max=53760 远低于 fp16 上限 65504，fp16 尾数 10 位比 bf16 7 位更细）。
   - **BW10 专属 config 调优**：48 CUs；BLOCK_SIZE_M=16、BLOCK_SIZE_K=32、NG=1（**BK=64/NG=2 在 BW10 触发 VMFault 崩溃**）；小形状（N≤3584）BLOCK_SIZE_N=64+SPLITK=2，大形状（N=4096/5120/10752）BLOCK_SIZE_N=128+自适应 SPLITK（按 M 取 3~16）；num_warps=4、NUM_CUS=48。未调优的 aiter 默认 config 比 vllm 还慢 10-28%；调优后 kernel 级（M=4）快 1.6-2.33x。
   - **踩坑**：SPLITK 离线调优对 kernel 级提速 ~1.5x，但 decode 端到端 GEMM 时间波动 2-3x 会淹没收益——SPLITK 主要保证大形状不慢不崩，端到端 TPOT 收益来自 w4a16 kernel 本身。
2. **阶段二：flash fp8 KV + head_dim=512**（改三层：flash-attention-cutlass 源码编译 whl + vllm 3 文件 + aiter 1 文件）。关键技术点：
   - **fp8_e5m2 KV cache 放行（vllm 侧）**：上游 vllm 对 compressed-tensors 模型无条件禁用 fp8_e5m2 KV（attention.py 报错 "fp8_e5m2 kv-cache is not supported with fp8 checkpoints"），但 AWQ-4bit checkpoint 里并没有 fp8 KV scale（kv_cache_scheme=None），只是运行时想把 KV 存成 e5m2。改 3 文件：attention.py 放行 e5m2；rocm_aiter_unified_attn.py 读侧 view 用 e5m2 + 写侧走 triton（C++ reshape_and_cache_flash op 不支持 e5m2）；triton_reshape_and_cache_flash.py 写侧按字符串选 e5m2 dtype。
   - **e5m2 而非 e4m3 的选型**：fp8 KV 三路径（e4m3 triton / e5m2 triton / e5m2 flash）全测——e5m2 flash 最快且精度可接受；e4m3 reshape 与 e5m2 性能基本相同但精度 e5m2 有明显优势；HumanEval 验证 e5m2 路径精度无损（优于 fp16 基线）。
   - **Q-cast 解决 dtype 不匹配（aiter 侧）**：flash prefix decode kernel 要求 Q/K 同 dtype；fp8 KV 时 K 是 e5m2、Q 是 fp16，直接传报 `RuntimeError: For prefix decode, query and key must have the same dtype`。解法：Q cast 成 KV dtype 再传 + 附 unit descale（全 1，纯 bit-cast 数值无影响）。
   - **flash 路由条件（aiter 侧，unified_attention.py 加 3 个环境变量）**：`ATTN_FLASH_PREFILL=1`（prefill 走 flash）、`ATTN_FLASH_HEAD512=1`（draft head512 decode 走 flash）、MTP q_len≤8 decode 自动走 flash。**关键坑：`ATTN_FLASH_PREFILL=0` 会崩**——aiter triton 2D 在 fp8 KV prefill 有编译 bug（`'constexpr_type' object has no attribute 'is_ptr'`，e4m3/e5m2 都崩），phase2（ROCM_AITER_UNIFIED_ATTN + fp8 KV）的 prefill 必须走 flash，没有 triton 退路。
   - **head_dim=512 flash kernel 实现**：flash-attention-cutlass 改 19 文件（17 改 + 2 新建）。Layer A（10 文件）：fp8 mixed kernel 同时支持 e5m2/e4m3（e4m3 走软件 `__e4m32float` dequant，复用 e5m2 compact-LDS pipeline，cast 函数加 InputElement 模板参数 if-else 选）。Layer B（3 改 + 2 新建）：解除入口 TORCH_CHECK 限制；新建 2 个 target 文件实例化 `run_fp8_mha_fwd_prefix_prefill_<Float16/BFloat16, 512, 512>`（FP16_SWITCH 运行时分发需要 fp16+bf16 两个符号都在，漏 bf16 链接报 undefined symbol）。**只编译 gfx936 不编译 gfx938**（gfx938 fp8 GEMM utils 不支持 512 会 static_assert）。
3. **锁频（性能关键，bench/生产都要做）**：DCU 默认按负载动态调频，decode 负载低时频率掉下来，TPOT 偏高 5~7ms、吞吐掉 ~7%。必须把用的卡锁到 sclk level 6（760MHz）；某些容器下 rocm-smi 锁频会静默失败（权限不足），需 `rocm-smi --showclocks` 手动确认（正常 4 卡都 level 6，异常 level 0/300MHz 或 Perf 列 auto）。
4. **MTP 调参（num_speculative_tokens）**：基于实测 per-position 接受率（num_spec=3 时各位置接受率从高到低缓慢衰减）——衰减平缓，后续 draft token 仍有高接受率，提 num_spec 有正收益；num_spec=5 实测整体接受率下降（位置 3/4 衰减明显），batch 大时 step 变重。建议：生产先试 num_spec=4（零成本改一个数字），看实测再决定是否上 5。
5. **dspark 评估结论（不采用）**：并行起草（半自回归）替换 MTP 不值得——生产 vllm 0.23.0 无 dspark 代码（需升级未发布 dev 版丢失 DCU 定制）；无 gemma4 dspark checkpoint（需专门训练的 Markov head + confidence head）；dspark 需 non-causal attention，当前 ROCM_AITER_UNIFIED_ATTN 不支持（supports_non_causal=False）；MTP 接受率已 98%，瓶颈在 GEMM 非 attention，收益不匹配成本。

**生产启动配置要点（两阶段叠加，start_flash.sh）**
```bash
export HIP_VISIBLE_DEVICES=0,4,2,3          # TP4 用的 4 块卡（锁高性能模式）
export VLLM_AITER_W4A16_PATCH=1             # 阶段一：aiter w4a16 GEMM
export ATTN_FLASH_PREFILL=1                 # 阶段二：prefill 走 flash（必须=1，=0 会崩）
export ATTN_FLASH_HEAD512=1                 # 阶段二：draft head512 decode 走 flash
vllm serve "$MODEL_DIR" \
    --dtype float16 \                       # 对齐 aiter kernel fp16 输出
    --kv-cache-dtype fp8_e5m2 \             # 阶段二：fp8 KV
    --attention-backend ROCM_AITER_UNIFIED_ATTN \  # 阶段二：aiter unified（路由到 flash）
    --tensor-parallel-size 4 \
    --optimization-level 3 \                # torch.compile + cudagraph
    --enable-prefix-caching --enable-chunked-prefill \
    --async-scheduling \                    # 零气泡调度
    --performance-mode throughput \
    --speculative-config '{"method":"mtp","model":"$DRAFT_MODEL_DIR","num_speculative_tokens":3}'
```
阶段一→阶段二关键参数差异：`--attention-backend` TRITON_ATTN→ROCM_AITER_UNIFIED_ATTN；`--kv-cache-dtype` fp8→fp8_e5m2；`--dtype` 显式 float16；env 增加 ATTN_FLASH_PREFILL/HEAD512。

**量化结果**：见上「现象」表——TPOT 显著降低、吞吐大幅提升、精度无损（量级结论，非固定基线）。kernel 级（prefill q=512）：flash_fp8 与 triton 在同一量级（flash 明显快）。**已知回退：TTFT**——phase2 prefill 必须走 flash（triton 2D 在 fp8 KV 下崩），flash prefill kernel 级更快但端到端 TTFT 明显上升（Q-cast、路由开销），无 env 可绕，是该路径固有代价；生产以 TPOT 为主要指标，TTFT 回退可接受。

**可迁移结论**
1. **瓶颈定位先 profile 再动手**：vllm profiler 抓 decode step 算子耗时分布，确认"改的算子确实是瓶颈"。本案例 awq_gemm 占比最高是第一瓶颈，attention 占比小但 draft 侧 head512 是洼地——两个瓶颈分别用两个独立阶段攻破，互不冲突可叠加。
2. **`torch.library.custom_op` + `register_fake` 包装第三方 kernel 是对 torch.compile/cudagraph 全透明的标准手段**：aiter triton driver 调用 inductor 编译不了，包成注册算子后 49 张 cudagraph 全捕获、无 graph break。与案例 10/11 的 monkey-patch 范式互补：monkey-patch 适合不改源码注入，custom_op 适合把外部算子库正式接入 compile 链路。
3. **量化权重换 kernel 库，列序/打包格式是第一精度坑**：GPTQ 打包顺序与 aiter 解包顺序不一致，不重排 cos_sim=0.25；对称量化无显式 zp 要造全 8 zp 张量。换量化 kernel 前先用离线算子级 cos_sim 门禁验证（本案例 1.000000）。
4. **repack 后立即释放原始权重**：省大量显存/卡，TP1 装大模型时是 OOM 与不 OOM 的区别。
5. **fp8 KV 选 e5m2 不选 e4m3**：性能基本相同，但精度 e5m2 有明显优势；上游 vllm 对 compressed-tensors 模型无条件禁 e5m2 KV 是误伤（checkpoint 无 fp8 KV scale 时），需 patch 放行。
6. **MTP 调参看 per-position 接受率衰减**：每位置接受率衰减缓慢时提 num_spec 有正收益；衰减明显时（num_spec=5 的位置 3/4）batch 大 step 变重反噬。
7. **锁频是 bench/生产数字可信的前提**：不锁 sclk 时 decode 低负载降频，TPOT 虚高、吞吐下降（详见 `dcu-llm-benchmark` 前置纪律）。
8. **TTFT 回退要如实记录**：phase2 路径 TTFT 明显上升是"prefill 必须走 flash"的固有代价，无 env 可绕——优化交付要写清已知回退与取舍，不能只报 TPOT。

> 溯源：内部文档 file_id=<internal-file-id>（内部文档链接，已略）《【BW10】投机解码+aiter w4a16算子对gemma-4-31B-it-AWQ-4bit性能加速方案总结》；代码/patch 仓库 `http://<internal_ip_ci>:10068/<internal-user>/llm-optimize.git`（vllm/aiter-w4a16/ 阶段一 patch+config，vllm/flash-attn/ 阶段二 patch+whl，flash-attention-cutlass/ 源码改动）；flash 编译 vmfault 排查见 `dcu-ontrack-experience` B1 节。flash head_dim=512 的 19 文件源码改动细节以仓库 `flash-attention-cutlass/README.md` 为准，本案例不重复。
