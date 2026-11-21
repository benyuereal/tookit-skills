---
name: dcu-train-cookbook
description: "DCU/HCU 训练域 skill：训练/微调/RL 强化学习配方速查 + 训练框架版本锚点。覆盖 train-cookbook-das（Megatron/mAP-cv/Physical AI 微调配方）、verl-das/slime-das/AReaL-das（RL GRPO 矩阵）、Megatron-LM-das（hcu-megatron）、flash-train、DiffSynth-Studio-das、TurboPhysAI、hcu-train-simulator（训练性能估算）。"
version: 1.0.1
author: DCU Engineer
license: MIT
kind: knowledge
---

# DCU/HCU 训练域（dcu-train-cookbook）

> 定位：本地 dcu skills **训练域空白**（之前只有 `dcu-training-frameworks-source` 讲包结构，无训练/微调/RL 配方）。
> 本 skill 回答「在 DCU 上怎么训/微调/RL 一个模型」+「训练框架用什么版本、能训什么」。 
> 部署/推理（serving）配方转 `dcu-model-deploy-catalog`；训练速度/吞吐评估用 `model-compute-simulation`（通用）或 `hcu-train-simulator`（本域专用）。

> ⚠ 平台命名：官方实验室把平台叫 **HCU**，仓库名带 `-das`/`hcu-`；本地一律按 **DCU** 行文，仅保留仓库/源码原名。

## 更新机制（保证最新）

本文版本锚点来自 HYGON-AI GitHub 开源上游快照。刷新：

```bash
cd <agent-home>/skills/dcu
python3 ../_dcu-toolkit-shared/scripts/hygon-upstream-sync.py --check   # 快检版本锚点（~10 个 API 请求）
python3 ../_dcu-toolkit-shared/scripts/hygon-upstream-sync.py --full    # 全量 + 更新 cookbook 矩阵
```

触发：做训练/RL 任务前先 `--check` 确认框架版本没变；`--full` 建议每周跑一次。
快照/索引：`../_dcu-toolkit-shared/references/hygon-upstream/upstream-snapshot.json`、`repos-index.md`。

## 一、RL 强化学习（本地新域）

三套官方 HCU RL 框架，皆走「rollout 推理 + Actor 训练」范式。选型：规模大/多卡 → Megatron Actor；规模小/快速迭代 → FSDP/FSDP2 Actor；需要多模态 → vLLM rollout。

### verl-das（HCU 生态 RL 套件，覆盖最全）
- 基线：volcengine/verl + NVIDIA Megatron-LM；Apache-2.0。
- 算法：`grpo`、`fully_async_policy`、`one_step_off_policy`、`on_policy_distillation`、`multi_on_policy_distillation`。
- rolloute 后端：vLLM（`one_step_off_policy` 另支持 SGLang）。
- 训练后端：FSDP / FSDP2 / Megatron。
- 已验证模型矩阵（vLLM + 对应 Acton 后端）：
  | 模型 | 后端 |
  |---|---|
  | Qwen2.5-0.5B | FSDP |
  | Qwen2.5-VL-7B | FSDP2, Megatron |
  | Qwen3-8B | FSDP, Megatron |
  | Qwen3-235B | Megatron |
  | Qwen3-VL-8B | FSDP2, Megatron |
  | Qwen3.5-9B / 27B / 35B-A3B | Megatron(部分 FSDP2) |
  | DeepSeek-7B | FSDP |
  | Moonlight-16B-A3B | Megatron |
  | Seed-OSS-36B-Base | FSDP2 |

### slime-das（THUDM/slime v0.3.0 适配）
- 特性：Megatron-LM Actor + SGLang rollout + Ray；经 Megatron-Bridge 加载 HF 权重。
- 环境：Python 3.10+；需预装 DCU 版 PyTorch/SGLang/Megatron-LM/Megatron-Bridge（不随 requirements 装）。
- 配置：`export SLIME_ROOT=<repo> MEGATRON_BRIDGE_ROOT=... MEGATRON_LM_ROOT=... SGLANG_ROOT=...`，`hcu_example/common_env.sh` 设默认。
- 示例：`run_qwen3_4b.sh`（标准）、`run_qwen3_4b_fully_async.sh`（全异步 rollout）、`run_qwen3.5_4b.sh`（torch_dist ckpt）、GLM-4.6/GLM-5/DeepSeek-R1（4 层功能验证）、R1-Distill-Llama-70B（4 层）。

### AReaL-das（AReaL v1.0.4 适配）
- 特性：Ray 单/多节点 + SGLang rollout + FSDP/Megatron Actor；Python 3.11。
- 安装：建 venv → `pip install -r requirements-hcu.txt` → `pip install -e . --no-deps`（`--no-deps` 关键，防覆盖 DCU 软件栈）。
- 配置：`export AREAL_HOME=... SGLANG_ROOT=... MEGATRON_HOME=... MEGATRON_BRIDGE_HOME=...`（仅 glm_moe_dsa 模型需 hcu_megatron）。
- GRPO 示例：Qwen3-8B / Qwen3-VL-4B / Qwen3-30B-A3B（4 层，Megatron）/ Qwen3-1.7B / Qwen2.5-0.5B / GLM-5（4 层，Megatron）；全 SGLang rollout（多模态走 `run_qwen3_vl_4b_fsdp_sglang.sh`）。

## 二、训练/微调配方（train-cookbook-das）

按「模型/框架 × 卡型（BW1000/BW1100）组织，绿色对勾=已验证」：
- **Megatron 微调**（VLM）：Gemma 3-VL（BW1000 ✅）、Qwen 2/2.5-VL、Qwen 3-VL（BW1000/BW1100 ✅，Gemma 4-VL/GLM 4.5-VL 待验）。
- **mmcv**（CV/AD）：Bevfusion（BW1000/BW1100 ✅）。
- **Physical AI**：VLA（π0.5、GR00T-N1.7、OpenVLA、StarVLA）与 World Models（Fast-WAM）→ 均 🚧 开发中；配 TurboPhysAI 性能优化组件。
- 文档路径：`docs/model-framework/<megatron|mmcv|...>/<BW1000|BW1100>/<模型>.md`。

## 三、训练框架（Megatron-LM-das = hcu-megatron）

- 基于 NVIDIA Megatron-LM `core_r0.18.0`/`core_v0.18.2`（commit `571370c`）。BSD-3。
- 要求：`dtk >= 25.04`、`transformer-engine >= 2.4.0`、`torch >= 2.6.0`。
- 能力：LLM 预训练、LLM SFT、VLM SFT。
- 安装：`git clone -b <latest-branch> --recurse-submodules ...`；离线用 release 包 + `Megatron-LM@<version>` 子包。
- 依赖：Megatron-Bridge / Megatron-Energon / MindSpeed / VocabularyParallelism / Seq1F1B / Primus / NVIDIA TransformerEngine（第三方声明见 THIRD_PARTY_NOTICES.md）。
- ⚠ 与 `dcu-training-frameworks-source` 的差异：该 skill 记录的是海光自研 patch 层变体 `dcu_megatron`（内部 wheel 版本，无上游对应）；本 cookbook 按对齐 **core_v0.18.2** 的开源线组织，做迁移/对齐前先核对两者 commit 关系。

## 四、训练性能估算（hcu-train-simulator）

- CLI `hcu-train-sim`，输入 HF `config.json` + Megatron 并行策略 + 硬件参数 → 估算单次迭代显存/计算/通信/吞吐。
- 能力：按 PP stage 出显存（参数/梯度/优化器状态/激活，支持 TP/CP/PP/VPP/EP/ETP）；计算（dense/MoE 的 QKV/Attn/MLP/Router/LM Head/FA + RMSNorm/RoPE/SwiGLU 等非 GEMM）；通信（各域带宽+拓扑+掩盖比例）；端到端 TGS/MFU。
- 网格搜索：按 `search.candidates` 求最高 TGS Top10（滤超显存）；配置求解：给最小卡数 + 5GiB 余量方案；`run` 产出 `training_report_*.html`。
- 建模支持：Dense/MoE、MLA（`q_lora_rank/kv_lora_rank/qk_rope_head_dim/v_head_dim`）、MTP、GLM-5 DSA（`model_type: glm_moe_dsa`，`index_n_heads/index_head_dim/index_topk/indexer_types`）、Qwen3.5 VLM（`text_config+vision_config`，3:1 Gated DeltaNet 模式）。
- 局限：不精确建模数据搬移/Kernel launch/框架调度/fused kernels；端到端时间用于**方案对比与瓶颈分析**，不等价真实训练时钟。
- 与通用 `model-compute-simulation`（BBuf）互补：本工具按 DCU/HCU 硬件参数 + 官方并行建模更贴合，优先用本工具做 DCU 训练容量/并行规划。

## 五、算子/其他训练组件

- **flash-train**：训练算子库，C API 三级接入（框架绑定/便利 C API/分阶段 C API）+ 引擎选择 + Plan 复用；当前 GEMM(FP32, gfx938) 链路验证样例（性能未调优，慢于 torch.matmul），正式算子随版本发布。
- **DiffSynth-Studio-das**：MiniMax-H3 多卡训练适配 + 算子/通信优化 + 训练示例（视频生成训练）。
- **TurboPhysAI**：Physical AI 训练性能优化组件（算子/计算图/数据链路/训练过程优化，无需改模型源码，优化随 package 交付）。

## 六、使用须知

- 先 `../_dcu-toolkit-shared/references/hygon-upstream/repos-index.md` 看每个仓库的最新基线/特性，再进对应仓库 README（缓存于 `<agent-home>/cache/hygon-upstream/`）。
- RL 框架版本锚点：verl(无固定 tag)、slime v0.3.0、AReaL v1.0.4、Megatron core_v0.18.2；上线前用 `--check` 复核。
- 更换环境先核对 PyTorch/DTK/卡型，RL 框架对 DCU 软件栈版本敏感（`--no-deps` + 预装组件的原则）。
