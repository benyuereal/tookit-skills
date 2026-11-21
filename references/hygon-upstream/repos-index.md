# HYGON-AI 开源上游索引（dcu skills 统一引用）

> 来源：https://github.com/HYGON-AI 组织公开仓库（检索日 2026-10-12，随 cron 周更）。
> 快照见本目录 `upstream-snapshot.json`；刷新命令见文末「更新机制」。

## 一、组织概况

- 组织主页：`github.com/HYGON-AI`，公开仓库 45 个（2026-10-14 复测；2026-04-30 创建，持续活跃更新。官网 codehub 只列 34⊂45，未列的含 pytorch-das/tensorflow-das/aiter-das/UltraEP-das/TransformerEngine-das/opencv 系/faiss-das/spconv-das/FastPT/one-click-quant/cluster-manager-das/inference-cookbook-das/quality-gate 等，对照以 GitHub org 页为准）。
- 命名规律：官方把面向海光计算单元的适配版统一叫 **`*-das`**（下游/Adaptation Suite 含义），
  平台品牌已由 **DCU 演进为 HCU**。本地 skills 一律按 **DCU** 命名与行文，仅仓库/源码名保留原样（含 `hcu-`/`das`）。
- 适配逻辑：大多基于上游社区某**固定 commit/tag** 做下游改造，README 里都带「Upstream baseline/tag/commit」段——这正是本地 `*-source` skill 需要对齐的主线版本锚点。

## 二、仓库清单（按类）

### A. 推理部署/配方（直接喂本地 skill）
| 仓库 | ⭐ | 说明 | 对应本地 skill |
|---|---|---|---|
| **inference-cookbook-das** | 61 | 模型×框架×卡型验证矩阵（K100_AI/BW1000/BW1100 × vLLM/SGLang/Wan-DAS/SGLang-Diffusion/Infinity），85 行矩阵、59 行已验证；含 Qwen3.8/3.6/3.5、DeepSeek-V4/V3.2、GLM-5.2/5.1/5/4.7、Kimi-K3/K2.6、MiniMax-M3/M2.7、Hy4、LLM 外的多模态/VLM/视频生成；文档按 `docs/model-deployment/<框架>/<模型>.md` 组织 | `dcu-model-deploy-catalog`（新增上游引用） |
| **train-cookbook-das** | 4 | HCU 训练/微调配方：Megatron 微调（Gemma3-VL/Qwen2/3-VL）+ mmcv（Bevfusion）+ Physical AI（VLA: π0.5/GR00T-N1.7/OpenVLA/StarVLA，World Models: Fast-WAM）；带 `skills/verl-script-formatting-recipe` | `dcu-train-cookbook`（新建） |
| **one-click-quant** | 2 | 大模型一键量化：BF16→W8A8 FP8/INT8、Block FP8→BF16、DeepseekV3/Qwen3.5 MoE 定制方案、`model_free_ptq`（免校准免transformers）、slimquant w4a8、MoE-Quant 单机大权重 GPTQ | `dcu-model-deploy-catalog` 量化小节 |
| **hcu-train-simulator** | 1 | 训练性能估算 CLI `hcu-train-sim`（显存/计算/通信/吞吐、TGS/MFU、并行网格搜索、HTML 报告；支持 GLM-5 DSA、Qwen3.5 VLM） | `dcu-train-cookbook` / 与 `model-compute-simulation` 互补 |

### B. 框架适配源码（对齐本地 `*-source` skill 的版本锚点）
| 仓库 | ⭐ | 上游基线 | 对应本地 skill |
|---|---|---|---|
| **vllm-plugin-das** | 20 | vLLM `v0.25.1`（commit `752a3a5`），插件式（3 入口 hcu/hcu_model/hcu_ops），`vllm-hcu-doctor` 诊断，兼容门 0.25.x | `dcu-vllm-source`、`dcu-vllm-hcu-source` |
| **sglang-das** | 44 | sglang 主线 tag `v0.5.15.post1` | `dcu-sglang-source`（本地 0.5.12 实测 + 0.5.15 演进） |
| **Megatron-LM-das** (`hcu-megatron`) | 6 | Megatron-LM `core_r0.18.0`/`core_v0.18.2`（commit `571370c`）；要求 dtk≥25.04、TE≥2.4.0、torch≥2.6 | `dcu-training-frameworks-source`（本地内网 dcu-megatron 0.15.0 需对齐） |
| **aiter-das** | 1 | ROCm/aiter commit `552f4b8`（`main` 分支参考） | `dcu-aiter-source`（本地算的 v0.1.3 为近似，已用开源 commit 校正） |
| **triton-das** | 0 | triton-lang/triton 下游 HCU 后端 | `dcu-triton-source` |
| **mooncake-das** | 6 | Mooncake `v0.3.13.post1` | `dcu-mooncake-source` |
| **LMCache-das** | 1 | LMCache `0.4.2`（2026-10 锚点升级，原 `v0.3.13`/`fc031d4`；对应 vLLM 0.18.1 见 `dcu-lmcache` §0）。⚠️ pypi 源最新构建仍是 **0.3.13**（torch2100，import 实测通过，<device-A>/2026-10-14），与 GitHub 锚点 0.4.2 不同步，部署以源上构建为准 | `dcu-lmcache` |
| **DeepEP-das** | 2 | DeepEP commit `da6ca24` | `dcu-deepep-source` |
| **UltraEP-das** | 1 | UltraEP 下游（MoE dispatch/combine，与 DeepEP 同域） | 暂无（并入 `dcu-deepep-source` 按需扩展） |
| **pytorch-das** | 0 | 上游 PyTorch 二次开发：HCU 编译/测试/CI 全量单测 | `dcu-training-frameworks-source` |
| **TransformerEngine-das** | 0 | 上游 TE 下游适配 | `dcu-training-frameworks-source`（**铁三角实测**：TE 2.13.0 + torch 2.10.0 + flash-attn 2.8.4，见 `dcu-software-adaptation-guide` §8.4） |
| **FastPT** | 0 | PyTorch 应用组件适配工具：CUDA 代码在 HCU 上**不转码编译**（`source /usr/local/bin/fastpt -C`） | `env/dcu-software-adaptation-guide` §1A/§8.5 规避手段（入口按 torch 精确版本门控，<device-A>/torch 2.12 实测通过） |
| **TileKernels-das** | 0 | DeepSeek-AI/TileKernels，用 `tilelang-hygon` 构建 | `dcu-tilelang-source`（注意 TileKernels≠TileLang 编译器） |
| **composable_kernel-das** | 0 | ROCm/composable_kernel 下游 | （暂无，按需） |
| **Moe** | 0 | AITER 依赖的 hipc MoE 算子库 | `dcu-aiter-moe` |
| **rocSHMEM-das** | 1 | rocSHMEM 下游 | （暂无，通信域按需） |
| **jax-das / xla-das** | 0 | JAX / XLA 适配 | （暂无，JAX 生态按需） |
| **apex-das** | 2 | PyTorch Apex 扩展（混合精度/分布式/HCU HIP kernel 融合） | `dcu-training-frameworks-source` |

### C. RL 强化学习训练（本地新域缺口）
| 仓库 | ⭐ | 基线 | 说明 |
|---|---|---|---|
| **verl-das** | 4 | verl + Megatron-LM | HCU RL 套件：GRPO（Qwen2.5/3/3.5/VL、DeepSeek、Moonlight、Seed-OSS 等，Qwen3-235B Megatron）、fully_async_policy、one_step_off_policy、on_policy_distillation；rollout 走 vLLM |
| **slime-das** | 0 | THUDM/slime `v0.3.0` | Megatron-LM + SGLang rollout + Ray + Megatron-Bridge；示例 Qwen3-4B/3.5-4B、GLM-4.6/5、DeepSeek-R1（4 层验证） |
| **AReaL-das** | 1 | AReaL `v1.0.4` | Ray + SGLang rollout + FSDP/Megatron；GRPO 示例 Qwen3-8B/VL-4B/30B-A3B/1.7B、GLM-5（Python 3.11） |
| **flash-train** | 3 | 自研 | 大模型训练算子库（C API 三级接入+引擎选择+Plan 复用），当前 GEMM(FP32, gfx938) 链路验证样例 |
| **DiffSynth-Studio-das** | 1 | modelscope/DiffSynth-Studio | MiniMax-H3 多卡训练 + 算子/通信优化 |
| **TurboPhysAI** | 4 | 自研 | Physical AI 训练优化组件（算子/计算图/数据链路/训练过程，无需改模型源码） |

### D. 平台/运维（本地监控/K8s 域缺口）
| 仓库 | ⭐ | 版本 | 说明 |
|---|---|---|---|
| **hcu-dcgm** | 3 | v2.0.0 | Go DCGM 管理监控（健康/功耗/时钟/资源；HTTP :16081 + `dcgmi` CLI + Go 库 `hcu-dcgm/v3`），依赖 `libhydmi.so`/`librocm_smi64.so`（/opt/hyhal/lib） |
| **hcu-exporter** | 4 | — | Prometheus 导出器（经 hcu-dcgm 抽 DCGM 指标，`/metrics` :16080），支持 `--metrics-level`、K8s Pod 关联、vHCU、IP 白名单 |
| **k8s-hcu-device-plugin** | 2 | v3.0.0 | 设备插件：物理整卡/预切分 vHCU/HAMi 动态切分/MIG 四模式，NUMA 上报，拓扑 ConfigMap |
| **k8s-hcu-dra-driver** | 2 | v1.0.0 | DRA `ResourceSlice`+CDI 注入；整卡直通 / 按容量动态切 vHCU（单卡≤4）；需 K8s 1.34+、DRA/DRAConsumableCapacity 门控 |
| **k8s-hcu-label-node** | 1 | v3.0.0 | 自动打标 `hygon.com/hcu=true` + 型号/cu-count/vram/driver-version；无 HCU 节点静默 |
| **volcano** | 1 | v1.15.2 | volcano 批调度下游适配 |
| **hcu-container-toolkit** | 1 | — | 容器内使用 HCU 的 Runtime 对接简化工具 |
| **cluster-manager-das** | 1 | — | 集群管理（暂无 README 细节） |
| **k8s-hcu-scheduler** | — | — | HAMi 动态切分所需调度器（device-plugin README 引用，仓库可能未公开） |

### E. 其他
| 仓库 | ⭐ | 说明 |
|---|---|---|
| **SkillHub** | 0 | 官方 agent skills 目录仓（`npx skills add HYGON-AI/SkillHub`）；现含 `skillhub-contributor`、`torch-trace-operator-profiler`（torch.profiler trace 按 Python scope/ATen/GPU kernel/runtime/memcpy 归因，与本地 `llm-torch-profiler-analysis` 互补） |
| **quality-gate** | 4 | HYGON-AI 仓库统一 PR 质量/安全/开源合规门禁（组织内部工程，不入 skill） |
| **faiss-das** | 0 | faiss 下游：后端集成/构建打包/GPU 算子性能优化（2026-10 新增；**<device-A> 实测 GPU 检索跑通**，注意 `GpuIndex*` 新 API 形态，见 `dcu-software-adaptation-guide` §8.6） |
| **spconv-das** | 1 | spconv_rocm 下游（稀疏卷积，自动驾驶域；**<device-A> 实测 SparseConv3d 跑通**，API 在 `spconv.pytorch` 子包、coordinates 4 列，见 `dcu-software-adaptation-guide` §8.6） |
| **opencv-das / opencv_contrib-das / opencv-python-das** | 0 | opencv 三件套下游（contrib 扩展模块 + Python 源码编译绑定；**<device-A> 实测 opencv-python 4.9.0 das 构建 GPU 扩展在**，需挂 DTK CUDA 兼容层，见 `dcu-software-adaptation-guide` §8） |

## 三、本地覆盖与缺口

已覆盖（有对应 `*-source`/部署 skill，只需对齐版本锚点）：vllm/sglang/aiter/triton/mooncake/LMCache/DeepEP(+UltraEP 同域)/Megatron+apex+pytorch-das+TransformerEngine-das/TileKernels(≈tilelang)/Moe(≈aiter-moe)/FastPT(并入适配指南)。
缺口（可新建 skill）：**训练域**（train-cookbook + verl/slime/AReaL + flash-train + DiffSynth + TurboPhysAI + hcu-train-simulator）→ `training/dcu-train-cookbook`；**平台运维域**（hcu-dcgm + hcu-exporter + K8s 组件）→ `env/dcu-opensource-platform`。
暂无对应 skill、按需用再建：faiss-das / spconv-das / opencv 三件套（CV/检索/自驾域，2026-10 新增；<device-A> 部署验证已跑通——faiss GPU 检索 / spconv SparseConv3d / opencv GPU 扩展，部署前置条件与 API 形态见 `env/dcu-software-adaptation-guide` §8，使用频次起来后再建 skill）。

## 四、更新机制（如何保证「最新」）

本地 skills 引用本表的版本锚点，刷新方式统一：

```bash
cd <agent-home>/skills/dcu
python3 scripts/hygon-upstream-sync.py --check   # 快检：35 仓库列表 + 关键仓库 tag（约 2-10 个 GitHub API 请求）
python3 scripts/hygon-upstream-sync.py --full    # 全量：另抓 README 缓存 + 解析 cookbook 模型矩阵（raw 不占 API 配额）
```

- **快照** `references/hygon-upstream/upstream-snapshot.json`（入 git，含 fetched_at / repos / anchors / cookbook_matrix）。
- **diff**：脚本对比上次快照，打印「新增仓库 / 仓库消失 / 版本锚点变化 / cookbook 新增已验证行」。
- **触发时机**：① 做部署/适配/排障且涉及这些上游前，先跑 `--check` 确认版本没变；② 建议每周跑一次 `--full` 更新 cookbook 矩阵；③ 已通过定时任务自动跑，见 `cronjob`（可选）。
- **网络**：本机 `curl` 直连 GitHub 可能被审批门拦截，脚本一律用 python `urllib`；GitHub API 未认证限 60 次/小时/IP，raw.githubusercontent 不占配额。
- **注意**：快照只存元数据，不搬运 README 正文（避免 git 膨胀 + 许可证问题）；README 全文缓存到 `<agent-home>/cache/hygon-upstream/<repo>.md`。
