---
name: dcu-skill-router
description: DCU 系列 skills 路由与链路手册：按部署/推理/算子开发/性能调优/环境/训练/工程发布七域分组，索引全部 dcu-* skill（含包根平铺与 coding-skill 嵌套）+ 34 个通用工程 skill，路由表带域目录前缀；§四 七条实测高频任务链，§五 改名/合并 alias 对照。加载时机：跨 ≥2 域、多步不确定、或拿不准该用哪个 skill 时先查本表；不适用：单步明确任务直接点名叶子 skill、不必经此（2026-10 三端实测：agent 2799 次 skill 加载中 router 占 29 次≈1%，Claude Code 侧 44 次，DSH 本机 3/150≈2%——单步点名占绝对主导）；多步链的编排执行（审批/证据/重试）见 dcu-orchestrator
version: 2.2.2
author: DCU Engineer
license: MIT
---

# DCU 系列 skills 统一入口路由（dcu-skill-router）

> 定位：本 skill 是 **DCU 全部 skills 的总入口**，按「部署」「推理」「算子开发」「性能调优」「环境」「训练」「工程与发布」七域分组，列出每个 skill 的 name + 一句话定位 + 触发场景。加载时机：跨 ≥2 域、多步不确定、或拿不准用哪个 skill 时**先查本表**定位，再进具体 skill 取方法；单步明确任务直接点名叶子 skill（skill 目录每轮随上下文注入，直达省一次加载）。多步/模糊任务先查 §四 常见任务链路选链（2026-10 三端调用栈实测：单步点名占绝对主导，router 作入口仅 1-8%，本表价值在链路手册与旧名 alias）。
> 覆盖：95 个顶层 `dcu-*` skill + 5 个 `dcu-coding-skill/*` 嵌套（dcu-ci-testcase、dcu-triton-kernel、dcu-dispatcher、dcu-code-migrator、dcu-hip-reference）+ 34 个通用工程 skill（BBuf AI-Infra-Auto-Driven-SKILLS 与 skills-dcu-autoopt 整合，见各域注记；autoopt 包 3 个同名保留 BBuf 版未重复计入），合计 140 个 SKILL.md（dcu-core 9 / dcu-env 24 / dcu-perf 15 / dcu-inference 22 / dcu-operators 19 / dcu-troubleshooting 6 + 通用工程 34 + `ops` 算子卡域 6）。2026-10-21 同步 master：新增 `dcu-offline-container-deps`、`dcu-offline-wheel-install`、`dcu-hyqual-acceptance`、`dcu-external-kernel-validation`、`dcu-external-opt-scout`、`cross-engine-acceptance-gate`、`dcu-ontrack-experience` 7 个 skill + orchestrator 增量（research-baseline 工作流、baseline-run/env-reproduce/research-collect 任务契约、extract_candidate_facts/suggest_target 脚本、evals 套件）。

## 一、七域分组路由表

### 0. 跨域编排（多步任务链执行层，2026-10-17 新增）

| skill | 定位 | 触发场景 |
|---|---|---|
| `dcu-orchestrator` | 多任务薄编排层：5 条内置工作流（model-deploy / fault-triage / op-optimize / infer-optimize / research-baseline，三段式串行，gate 落在改状态/占卡步）+ 16 个 task 契约 + `../dcu-orchestrator/scripts/render_workflow.py`（依赖拓扑算就绪节点）/ `verify_evidence.py`（证据验收）/ `dcu_distill.py`（产物沉淀知识库） | 明确要按既定多步链走（部署验证/故障排查/算子优化/推理优化），需要门槛审批（proceed: ask）与 evidence 留痕 |

> 编排选型：编码类四意图（API 查询/代码迁移/CI 用例/Triton）走 §二 `dcu-dispatcher` 意图分发；跨域多步任务链（上表 4 条）走 `dcu-orchestrator`；单步任务直接点名七域 skill，不过编排器。多步链的路由决策见 §四。

### 1. 部署域（把模型/服务跑上 DCU）
| skill | 定位 | 触发场景 |
|---|---|---|
| `env/dcu-docker` | DCU 容器操作规范 | 起容器、挂 hyhal、改 daemon、拉镜像 |
| `env/dcu-ssh` | SSH 到 DCU 服务器远程执行 | 远程操作 DCU 节点 |
| `env/dcu-kubernetes` | DCU 在 K8s 上的部署 | K8s 集群、device-plugin 调度 |
| `inference/dcu-vllm-deploy` | vLLM 部署调优速查 | 起 vLLM 服务、TP/PP、容器约束 |
| `inference/dcu-model-deploy-verify` | 模型部署前真机验证 | 先把模型跑通再写文档 |
| `inference/dcu-model-portability` | 评估模型能否即插即用部署 | PyTorch 模型能否迁移到 DCU |
| `inference/dcu-model-deploy-catalog` | 96 个已验证模型部署配方速查（按模型查镜像/框架/启动命令/坑，LLM/VLM/Embedding/CV/图像视频/时序/科学计算） | 某具体模型在 DCU 上怎么部署（查现成配方） |
| `env/dcu-image-reverse-rebuild` | 逆向 DCU 镜像为 Dockerfile | 从镜像恢复可重建 Dockerfile |
| `env/dcu-image-to-native-env` | 镜像环境复刻为物理机原生 conda 环境（非 Dockerfile） | 镜像→物理机 anaconda 直接跑、拷包+DTK+激活脚本+模型实测 |
| `inference/dcu-libtorch` | C++ LibTorch 推理部署 | 提取 LibTorch + 算子验证 |
| `inference/dcu-pytorch-source` | 低版本 PyTorch 源码编译 | 自编译 2.1.0 wheel 等 |
| `env/dcu-dtk` | DTK 安装/版本/布局 | 装 DTK、查 /opt/dtk |
| `env/dcu-dtk-layout` | DTK 目录结构速查 | 找 DTK 工具/库路径 |
| `env/dcu-env-packages` | DCU 适配版软件包清单 | 查哪些包可用 |
| `env/dcu-adaptation-overview` | DCU 适配总览（版本兼容矩阵 DTK×torch×框架×算子包 + 软件能力矩阵 + 真机验证结果 + 升级 SOP） | 查"这环境装什么版本/这包能不能用/怎么升级" |
| `env/dcu-software-adaptation-guide` | DCU 软件适配通用参考（5 大适配形态决策 + 7 步 SOP + DTK 核心差异 + 适配点定位技巧 + 验证清单 + 19 包踩坑） | 拿到新软件包要适配到 DCU 时的总方法论 |
| `env/dcu-csv-passthrough` | 机密计算 + DCU 直通加密虚拟机（CSV/CSV2/CSV3·VFIO 直通·SM2/SM4） | 加密虚拟机直通 DCU、机密计算部署 |
| `operators/dcu-cuda-hip-port` | CUDA 代码移植到 DCU/HIP（DTK 环境，含 hip_shim.h 模板 + probe 配方） | CUDA→HIP 移植（DTK） |
| `operators/dcu-cuda-to-hip-port` | 独立 CUDA C++ 项目移植到海光 DCU/HIP（MFMA 可用性/link stub/e2e 精度调试） | 独立 CUDA 项目移植 DCU |

### 2. 推理域（DCU 上跑大模型推理）
| skill | 定位 | 触发场景 |
|---|---|---|
| `dcu-skill-router` | DCU 全部 skills 总入口路由 | 先查路由表再进具体 skill |
| `dcu-skill-on-device-verification` | 技能库真机验证 / 哪些技能能在设备上实跑 / 技能内容准确性核对（分层 A-E + 验证循环 + 双版本对照定性；包根平铺，跨域方法论） | ✅ |
| `inference/dcu-inference-infra` | 推理 infra 子域路由器 | 总入口，按场景分发 |
| `inference/dcu-inference-infra-skill-builder` | 推理 infra 子域 skill 构建器（版本拆分/新增维护/包根一致性自检） | 新增/维护 infra 子域 skill（**未随本包分发**；包内替代：按 `../_dcu-toolkit-shared/docs/STANDARDS.md` + `inference/dcu-inference-infra` 约定直接新增） |
| `perf/dcu-inference-optimization` | 通用推理优化方法论 | CNN/Transformer/LSTM 推理提速 |
| `inference/dcu-vllm-source` | vLLM 各版本源码适配索引（合并 0.15/0.18/0.21/0.23/0.26） | 按目标版本查结构 |
| `inference/dcu-vllm-hcu-source` | vllm-hcu 插件 patch 层源码（105 patch + HCUPlatform + 模块交换，0.21+ 时代独立包） | 查 0.21+ vLLM DCU 适配落点 / patch 机制 |
| `inference/dcu-ort-source` | ONNX Runtime DCU 适配（ROCm EP 链 DTK libgalaxyhip + MIGraphX EP，二进制 wheel） | ORT 包结构 / EP 库链接 / DTK 适配点 |
| `inference/dcu-mooncake-source` | Mooncake Transfer Engine DCU 适配（KV cache 跨节点 RDMA + EP buffer + HIP copy kernel） | PD 分离 KV 传输 / Mooncake 包结构 |
| `inference/dcu-fastsafetensors-source` | fastsafetensors DCU 适配（gpu_compat.h 运行时 dlopen CUDA/HIP + GDS/ThreeFS copier） | 权重并行加载 / GPU 兼容层 |
| `inference/dcu-training-frameworks-source` | DCU 训练框架适配合集（deepspeed 带 14 .hip 源码 + megatron-core + dcu-megatron patch 层 + apex + TE） | 训练框架包结构 / dcu-megatron patch 机制 |
| `inference/dcu-sglang-source` | SGLang 源码结构与海光适配点（含 DFlash2/MTP 投机解码适配、fa2/fa3 约束、compact cache、**HiCache 三层缓存 Bug A/B/C + L3 file 后端 + write policy 选型 + 命中观测**、**PD 分离部署模板与关键 DCU 环境变量**） | SGLang 推理、投机解码移植、HiCache 源码级适配、**PD 分离（disaggregation）部署参数与 1P1D/2P1D 落地** |
| `perf/dcu-llm-benchmark` | 大模型端到端压测方法论（含投机解码方案对比三铁律：OOD 陷阱/overlap 混淆/accept 口径 + PD 分离配比选型与单卡口径折算方法论） | vLLM/SGLang 压测规范、投机方案 A/B、PD 分离部署配比选型与跨厂家单卡对比 |
| `inference/dcu-vllm-custom-op` | vLLM 自定义算子开发流程 | 从零造算子 |
| `inference/dcu-vllm-op-replace` | vLLM 已有算子替换流程 | 替换/重定向算子到 vLLM |
| `inference/dcu-vllm-runtime-patch` | vLLM 运行时 monkey-patch 注入专用 kernel（env 开关+形状表+上游 fallback+bit-identical 门禁，不改源码） | 单卡/单并发 decode 优化、不改源码注入 kernel |
| `inference/dcu-cosyvoice-opt` | CosyVoice2 TTS 在 DCU 部署推理 | TTS 推理 |
| `inference/dcu-lmcache` | LMCache KV 缓存卸载优化 | KV 缓存优化 |
| `dcu-hicache` | SGLang 内置 HiCache 三层缓存（L1 GPU/L2 host/L3 磁盘）DCU 部署：Bug B/C 补丁 + seccomp/NUMA/SSH 坑 + 一键适配脚本（--rebuild 防环境污染）+ 验证方法 | HiCache 部署、三层缓存验证、启动崩排查 |
| `inference/dcu-migraphx-custom-op` | MIGraphX 自定义算子开发 | 写 MIGraphX 算子 |
| `inference/dcu-migraphx-driver` | MIGraphX 命令行工具解析 | migraphx-driver 扩展 |
| `perf/dcu-migraphx-perf` | MIGraphX 推理引擎性能优化最佳实践（测量/批处理/序列化/精度/融合/TensorCore/多框架/精度调试/环境配置） | MIGraphX 推理提速、读 perf 报告、FP16/INT8 量化、NHWC 布局 |
| `perf/dcu-ort-perf` | ONNX Runtime 推理引擎性能优化最佳实践（perf_test 测量/批处理/FP16/EP 选择/图优化/多线程/io_binding/内存复用/多实例/自定义算子/环境配置/**Triton Server 集成调优**） | ORT 推理提速、读 perf_test 报告、ROCm/MIGraphX EP 选择、io_binding 零拷贝、多实例提 QPS、**Triton+ORT 两层正交调优（config.pbtxt/七步 SOP/反模式）** |

### 3. 算子开发域（写/改 HIP/TileLang/汇编 kernel）
| skill | 定位 | 触发场景 |
|---|---|---|
| `operators/dcu-kernel-dev` | DCU 算子开发方法学（含 HIP 调优速查表/示例代码 + DCU 专属优化轴/工具速查，2026-08-13 已补强） | 手写 HIP + TileLang 两条路径 |
| `operators/dcu-triton-kernels` | Triton 原生 DSL 优化 PyTorch 算子的完整流程与十题实战索引（Benchmark 流程、五大优化模式、DCU 陷阱） | 写/优化 Triton kernel、跨 BW1000/C500/Ascend910B 迁移 |
| `operators/dcu-triton-source` | Triton DCU/HCU 编译器源码（独立 hcu backend：DTK 特化 libgalaxyhip/aillvm/gfx936-938 768-vgprs + hcutune_cli/opt_config_cli 调优） | Triton 编译问题 / DTK 工具链 / kernel 调优 CLI |
| `operators/dcu-deepep-source` | DeepEP DCU 适配（MoE 专家并行 dispatch/combine + low-latency 路由策略 + DTK 链接 + 多节点带宽测试方法与实测参考线） | MoE 通信瓶颈 / dispatch/combine / low-latency 路由 / DeepEP 跨节点带宽测试、topo.config 卡网映射踩坑 |
| `operators/dcu-tilelang-source` | TileLang DCU 适配（TVM 系 DSL 编译器 + contrib/hcu.py 海光 HIP 后处理 + TILELANG_OVERRIDE_DEVICE_SOURCE） | TileLang 编译 / codegen 替换 / DSV4 tilelang kernel |
| `operators/dcu-mhc-kernels-source` | MHC/线性注意力 kernel 包 DCU 适配（causal-conv1d HCU 接口 + fast-hadamard + mhc-fused-kernel，rc4 独有） | Mamba 因果卷积 / Hadamard / Megatron MHC 融合 kernel |
| `operators/dcu-hipdsa-source` | hip-dsa/hipdnn DCU 适配（cudnn.DSA 稀疏注意力 HIP 替代，14 个 .hip 源码在 wheel 内 + hipdnn PReLU） | DeepSeek DSA 稀疏注意力 / FlashMLA HIP 替代 |
| `operators/dcu-kernelwiki` | DCU kernel 优化知识库 | 查询某算子怎么写才对 |
| `perf/dcu-op-optimization` | 算子优化闭环方法论 | profile→定位→改 kernel→验证 |
| `operators/dcu-aiter-source` | AITER 算子库源码索引 | AITER 算子取/改 |
| `operators/dcu-aiter-moe` | AITER MoE 融合算子四后端 | MoE 算子深度专题 |
| `operators/dcu-deepgemm-source` | DeepGEMM 通用 GEMM 源码索引 | DeepGEMM 取/改 |
| `operators/dcu-hytlass-source` | hytlass（CUTLASS 3.5.0 的 HCU 移植）源码与用法：C++ 模板搭 GEMM、两套编程模型（2.x mmac / 3.x HuTe）、11 个示例、与手写 HIP 的分界 | 想自己搭 GEMM / 非标准 layout / 融合 epilogue / 移植 CUTLASS 风格 kernel |
| `operators/dcu-deepgemm-arch` | DeepGEMM 三路径架构与 MODE 调优机制 | 选 GEMM 路径 / layout 陷阱 / 加调优表 |
| `operators/dcu-flashattn-source` | flash-attn 海光 DCU 适配源码索引（三层适配结构 + FA3 不可移植结论 + 版本谱系） | flash-attn 怎么适配 / 开源最新版能否适配 / 能否提速 |
| `operators/dcu-lightop-source` | LightOp 海光融合算子库源码 | LightOp 取/改 |
| `operators/dcu-lightop-moe` | LightOp MoE 融合算子多量化路径 | LightOp MoE 专题 |
| `operators/dcu-lightop-fusion` | LightOp 算子融合方法论（模式/手法/checklist） | 写新融合 kernel / 理解融合实现 |
| `dcu-custom-fusion-kernel` | 自定义 DCU HIP 融合 kernel（env 门控 + 上游 fallback；如 QK-Norm+RoPE 融合，包根平铺属 dcu-core） | 现有算子不支持某模型配置时写融合 kernel |
| `perf/dcu-hipprof` | hipprof 性能分析总纲（Trace/PMC/RCCL/泄漏/CoreDump，子域已并入父） | 综合分析总入口/方法论 |
| `inference/add-jit-kernel` | 上游 SGLang 轻量 JIT CUDA kernel（绑定/测试/基准/CI 注册/服务链路验证） | 加 JIT kernel（需 checkout 含 python/sglang/jit_kernel） |
| `inference/add-sgl-kernel` | sgl-kernel 重量级 AOT CUDA/C++ 算子（注册/Python API/构建/测试/wheel 集成） | 加 AOT 算子（需 CUTLASS/wheel 集成） |
| `inference/sglang-diffusion-add-model` | 图像/视频扩散模型接入 SGLang 原生 pipeline/stage/model 结构 | 扩散模型移植（非自回归 LLM） |

### 4. 性能调优域（找瓶颈、量化对比）
| skill | 定位 | 触发场景 |
|---|---|---|
| `perf/dcu-perf-analysis` | DCU 性能分析全栈总纲 | 五层体系+工具选型 |
| `perf/dcu-perf-profiling` | DCU 性能剖析方法论（含细粒度瓶颈诊断：六维分析+14 信号→成因→修复模式+时间线形态，读 hipprof/rocprof 输出） | rocprof/rocTracer 命令模板 + 采到 profile 后怎么读/命中哪个模式/先修哪个 |
| `perf/dcu-perf-cases` | 真实性能优化案例库 | 跨平台对标+根因定位 |
| `perf/dcu-framework-profilers` | 六大框架 Profiler 实战 | PyTorch/ORT/MIGraphX/vLLM profiler |
| `perf/dcu-ort-profile` | ORT Profile 机制+MOE 显存修复 | ORT 推理剖析 |
| `dcu-kernel-layer-microbench` | kernel 层微基准（LLM 推理调优的 kernel 级细定位，包根平铺属 dcu-core，与 perf-profiling 互补） | kernel 级细定位 / kernel 调优 |
| `perf/dcu-gdn-prefill-kernel-opt` | GDN prefill kernel 优化（DCU 上 GDN prefill TTFT） | 优化 GDN prefill TTFT |
| `env/dcu-hy-smi` | hy-smi 查询监控 | 温度/功耗/显存/拓扑巡检 |
| `env/dcu-topology` | DCU 多卡多节点拓扑优化 | 拓扑规划 |
| `env/dcu-cpu-affinity` | CPU 绑核优化（NUMA） | numactl 绑定方案 |
| `troubleshooting/dcu-train-precision` | DCU 训练精度排查 | NaN/逐层比对/NV 对齐 |
| `troubleshooting/dcu-vllm-precision` | vLLM 推理精度异常定位（算子级：逐层 cos 对比→层内钻取→绕过验证→出报告） | 输出乱码·NULL·token 异常 / 与 HF 对不齐 / 定位具体算子 |
| `troubleshooting/dcu-vmfault-coredump` | HSA VMFault 崩溃取证：faulting kernel 名 + args ABI 解码 + gpucore 内存快照读越界值（补强 hipprof CoreDump 实战域） | `HSA_STATUS_ERROR_EXCEPTION 0x1016` / TRAPSTS=0x80000000 / 越界访存崩溃 |
| `perf/dcu-vllm-trace-analysis` | vLLM V1 torch.profiler trace 按类别（gemm/w8a16/MoE/attn/triton…）汇总 kernel 统计并产 xlsx | 已有 trace.json 想看各类 kernel 耗时占比；与 llm-torch-profiler-analysis 互补 |
| `env/dcu-rccl-test` | RCCL 多卡带宽测试 | 通信测试 |
| `env/dcu-hyqual-acceptance` | HyQual 官方压力验收（5 卡型镜像矩阵 + du-bootstrap/DuDiag + OOB） | 验收/标前/压测 |
| `env/dcu-roce-network` | 多机 RoCEv2 网络环境搭建（IP/MTU/静态 ARP 全量注入（关键）/QoS·PFC·ECN/RDMA 连通性验证）——RCCL 多机测试的物理网络前置层 | 多机 DCU 集群新组网、RCCL 带宽不达标先查网络层、RoCE 无损网络配置 |
| `env/dcu-rccl-source` | RCCL 源码级链路分析 | 源码 PCIe 混合链路 |
| `perf/generate-profile` | 采集可复现的 SGLang/vLLM 服务 profile（graph-off/on、prefill/decode、rank 级 trace） | 优化前取证、瓶颈证据采集 |
| `perf/llm-pipeline-analysis` | 服务 trace 延迟拆解到 prefill/decode/MoE/attn/通信/rank 长尾（互斥分组） | torch-profiler 全局 triage 后定位阶段 |
| `perf/llm-serving-capacity-planner` | 显存容量规划：模型几何/KV cache/图池/allocator/TP·PP·EP 拓扑 | fit 检查、OOM/VMFault 诊断、mem-fraction 调优 |
| `perf/humanize-gen-plan` | 优化需求转预注册实验计划（固定工作负载/门禁/证据契约/停止条件） | 远程 GPU 优化开工前立计划 |
| `perf/humanize-rlcr` | 可恢复的证据驱动优化闭环（baseline→profile→patch→A/B→接受/回滚） | 多轮优化迭代、不稳定远程环境 |
| `perf/llm-torch-profiler-analysis` | 统一 LLM torch.profiler 分析（SGLang/vLLM/TRT-LLM：kernel/overlap/fuse/空泡） | 已有 trace.json 分析或 live profiling |
| `perf/llm-serving-auto-benchmark` | 跨框架自动 benchmark 搜索（SGLang/vLLM/TRT-LLM 同负载找最优部署命令） | 同模型跨框架比吞吐/延迟、找最佳启动参数 |
| `perf/model-compute-simulation` | 算子级算力/FLOPs/MFU 估算与并行 what-if | 问某模型算力/显存/并行占比 |
| `perf/model-architecture-diagram` | 公开原始模型结构图检索（DeepSeek/GLM/Qwen/Kimi 等） | 要某模型架构图/结构图链接 |
| `perf/sglang-prod-incident-triage` | SGLang 线上问题 replay-first 调试流 | 服务异常/延迟回归/崩溃 dump 转 replay |
| `perf/vllm-pr-summary` | vLLM 官方 PR 抓取+结构化分析（Mermaid 图+风险） | 分析/总结 vLLM PR 号 |
| `perf/torch-profiler-layer-track` | 给 Torch Profiler trace 加 L0/L1 层导航标记与紧凑 GPU 轨道（Perfetto 大 trace 导航） | trace 层数多看不清哪层、CUDA Graph 几百条 stream 行 |
| `perf/sglang-model-day0-support` | SGLang 新模型 Day-0 接入：支持范围/架构缺口/PR DAG/验证矩阵/发布门禁 | 新模型接入 SGLang 前梳理要做什么支持 |
| `perf/sglang-humanize-review` | 用 SGLang 人类 review 语料（至 2026-07-27）做 PR 审查 | 审 SGLang PR/diff 要 maintainer 风格反馈 |
| `perf/model-pr-optimization-history` | 开源社区模型优化 PR 历史知识库（SGLang/vLLM/TRT-LLM/TokenSpeed 按模型族，v0.1.5 同步） | 改模型专属 serving 路径前查社区 PR 证据 |
| `perf/dcu-vllm-wiki` | DCU/HCU/MACA 适配证据库：架构/版本对比/HIP 移植/DTK 兼容/历史优化决策与实验结论 | 回答 DCU 适配项目问题前先查证据 |
| `perf/dcu-external-kernel-validation` | 外部 kernel 优化端到端验证（外部 GPU kernel 优化 → DCU 移植 → 正确性/性能验证） | 外部 kernel 优化拿到 DCU 上验证 |
| `perf/dcu-external-opt-scout` | GitHub GPU/DCU 优化项目筛选（为 DCU 测试选型） | 找可试的 GPU/DCU 优化项目 |

> 注（2026-08-19）：`generate-profile` / `llm-pipeline-analysis` / `llm-serving-capacity-planner` / `humanize-gen-plan` / `humanize-rlcr` 5 个新 skill 来自 skills-dcu-autoopt 包（性能优化闭环：gen-plan → generate-profile → profiler/pipeline 分析 → simulation/capacity → auto-benchmark → rlcr 迭代）。autoopt 包中另有 `llm-torch-profiler-analysis` / `llm-serving-auto-benchmark` / `model-compute-simulation` 3 个同名 skill，但本地已有 BBuf 版（内容更丰富、脚本更全），故保留 BBuf 版、未采用 autoopt 子集版。

> 注（2026-10-12 KDA 对照）：Nvidia `kernel-design-agents`（现规范名 `NVlabs/kda`）的可迁移价值已吸收——主 workflow（Task Contract/最小循环/晋升规则）→ `dcu-kernel-dev` §12；ncu-report-skill 六维+14 模式（profile 怎么读/先修哪个）→ `dcu-perf-profiling` §D；Humanize 闭环 harness → `perf/humanize-gen-plan` + `perf/humanize-rlcr`；候选台账 `candidates.jsonl` + 四档决策标签（ACCEPTED/REJECTED/INCONCLUSIVE/DIAGNOSTIC-ONLY）+ 预注册同窗 A/B 噪声门禁 → `dcu-op-optimization` 输出物约定。KernelWiki 内容为 Blackwell/Hopper/tcgen05 平台特定，**不迁**，只借鉴其结构（已有 `dcu-kernelwiki`）。

### 5. 环境域（硬件/驱动/故障排查）
| skill | 定位 | 触发场景 |
|---|---|---|
| `env/dcu-env` | DCU 服务器环境探测 | 拓扑/CPU/NUMA/驱动预调 |
| `env/dcu-driver` | 海光 DCU 驱动安装/排障 | 装驱动、验证加载 |
| `troubleshooting/dcu-ontrack-experience` | DCU 故障排查速查 | 按报错关键词反查根因 |
| `perf/dcu-hipprof` | hipprof 基线建立（环境） | 建立 trace 基线 |
| `perf/dcu-hipprof` | hipprof 稳定性（环境） | 泄漏/寄存器压力 |
| `env/dcu-hy-smi` | hy-smi 巡检（环境） | 温度/功耗/显存/拓扑 |
| `env/dcu-dtk` | DTK 安装布局（环境） | 装 DTK / /opt/dtk |
| `env/dcu-dtk-layout` | DTK 目录结构（环境） | 找工具/库路径 |
| `env/dcu-das-hub-api` | DCU 包源（DAS Hub）JSON API：查版本/跨仓库枚举/下 wheel + torch ABI 不匹配诊断 | 查 DCU 源某包版本、找更新构建、下 wheel |
| `env/dcu-image-hub` | DCU 镜像仓库查询：sourcefind servicelist API（列镜像/Keyword 搜/ListTag 列 tag）+ Harbor 匿名 token 直查 + 全量快照与刷新脚本 | 找 DCU 适配镜像、查某框架全部版本 tag、拿 docker pull 命令 |
| `env/dcu-modelzoo` | ModelZoo 模型适配仓库查询（GitLab API v4：列/搜仓库 + model.properties 元数据 + README 提 DTK 基线/推荐镜像/卡数） | 查某模型 DCU 适配仓库、适配基线（DTK/框架/卡型）、旧项目上新设备需重新适配的判断 |
| `troubleshooting/debug-cuda-crash` | CUDA/HIP kernel 崩溃诊断（非法访存/设备断言/VMFault/NaN·Inf/OOM/自定义算子破坏） | SGLang/vLLM 算子崩溃、DCU/HCU 排障 |
| `troubleshooting/dcu-cudagraph-baked-scalars` | CUDA/HIP graph 烧死 Python 标量排查（静默读错、accept 从第 1 轮起为 0、不崩不报错） | graph 化路径结果错但无异常、投机解码 draft graph 失效 |
| `troubleshooting/debug-distributed-hang` | 分布式推理卡死诊断（区分集合通信死锁/慢 rank 尾/崩溃/资源耗尽） | 多卡 TP/PP/DP/EP 冻结、RCCL 阻塞 |
| `troubleshooting/cross-engine-acceptance-gate` | 移植/换引擎后的精度验收门禁（bf16 跨引擎逐 bit 一致不可作硬门禁，判别实验定性后分级放行） | 换引擎/换后端后的精度验收标准 |
| `env/hy-smi-tool` | hy-smi 深度巡检（利用率/显存/功耗/温度/拓扑/进程/RAS/固件） | 部署/压测/剖析前硬件体检（**未随本包分发**，包内替代 `env/dcu-hy-smi`） |
| `env/dcu-opensource-platform` | 官方开源平台与运维组件（集群监控 hcu-dcgm+hcu-exporter/Prometheus + K8s 组件 device-plugin v3/DRA driver/label-node/volcano + 容器对接 hcu-container-toolkit） | 集群级监控、K8s 资源调度（DRA/四模式）、官方开源组件版本（补本地监控+K8s 双缺口） |
| `troubleshooting/dcu-engine-precision-localization` | 自定义 C++ 推理引擎 DCU 精度定位到算子（引擎能跑通但输出垃圾时用） | 引擎输出垃圾/精度异常定位 |
| `inference/dcu-ninfer-port` | NInfer to DCU(gfx936) 移植（SIMT-first + 4-layer quant） | NInfer 移植 DCU |
| `dcu-offline-container-deps` | 离线 DCU 容器缺 Python 包时的取包/传包路径（包根平铺） | 离线补包 |
| `dcu-offline-wheel-install` | 无外网容器纯 Python 包：探测、wheel 传送与离线 pip 安装坑（包根平铺） | 离线装包 |

> 注：`dcu-hipprof-*` 五个子 skill 与 `dcu-vllm-source-*` 五个分版本均已合并收敛为父/总 skill（见上文路由表）。`debug-cuda-crash` / `debug-distributed-hang` / `hy-smi-tool` 来自 skills-dcu-autoopt（2026-08-19 整合）。

### 6. 训练域（训练/微调/RL 强化学习）
| skill | 定位 | 触发场景 |
|---|---|---|
| `training/dcu-train-cookbook` | 训练/微调/RL 配方速查 + 训练框架版本锚点：train-cookbook-das（Megatron/mAP-cv/Physical AI 微调）、verl-das/slime-das/AReaL-das（RL GRPO 矩阵）、Megatron-LM-das（hcu-megatron core_v0.18.2）、flash-train、DiffSynth-Studio-das、TurboPhysAI、hcu-train-simulator（训练性能估算） | 在 DCU 上训练/微调/RL 一个模型；问训练框架版本与支持矩阵；训练容量/并行规划（hcu-train-simulator） |

> 注：本域为 **2026-10-01 新建**（此前本地训练域仅 `dcu-training-frameworks-source` 讲包结构）。框架包结构与适配落点仍看 `inference/dcu-training-frameworks-source`；本域偏「配方/版本/RL」。

### 7. 工程与发布域（SGLang 工程规范 / CI / 重构 / 发布）
| skill | 定位 | 触发场景 |
|---|---|---|
| `engineering/ci-workflow-guide` | 修改/诊断上游 SGLang CI 编排（阶段/门禁/分片/runner matrix/快速失败） | 改 SGLang CI workflow |
| `engineering/sglang-bisect-ci-regression` | CI 失败归一化 + 区分代码回归与环境漂移 + bisect 有效区间 | 可复现 CI 失败定位 |
| `engineering/sglang-cherrypick` | 验证/发起/监控已授权的 SGLang release branch cherry-pick | 发布分支回合（需 release manager 授权） |
| `engineering/clean-startup-log` | 清理 SGLang 启动日志噪声，保留兼容/正确性/性能证据 | 去重/去预期弃用/去 rank 重复打印 |
| `engineering/env-var-conventions` | SGLang 环境变量约定（定义/读取/子进程传播/默认/别名/弃用） | 增改 `SGLANG_*` / `SGL_*` 变量 |
| `engineering/large-class-init-style` | Scheduler/TokenizerManager/ModelRunner 构造拆为可覆盖生命周期步骤 | 改这三类初始化逻辑 |
| `engineering/scripted-runtime-notes` | 设计保持真实 scheduler 生命周期语义的脚本化运行时测试/harness | 加 probe/控制原语/hook/测试 |
| `engineering/mechanical-refactor-verify` | 用可复现转换 + 严格语义边界验证文件拆分/函数移动/模块提取 | 机械重构 commit/PR 校验 |
| `engineering/model-pr-history-knowledge` | 建立可追溯的模型/运行时决策历史（接受/拒绝/未定/回滚实验） | 代码评审/回归分析/恢复历史依据 |
| `engineering/cookbook-add-model` | 配置驱动 Mintlify cookbook 新增模型页（config+基准+MDX+导航） | SGLang docs_new 布局下加模型文档 |
| `engineering/cookbook-migrate-model` | 旧版 cookbook 页迁移到配置驱动结构，保留已验证命令与数据 | 迁移 legacy cookbook 页 |
| `engineering/cookbook-review-pr` | 审查 cookbook PR：结构/来源/事实/基准边界/MDX 有效性/披露安全 | review cookbook PR |

> 注：本域 12 个 skill 全部来自 skills-dcu-autoopt（2026-08-19 整合），面向 SGLang 上游工程、发布流程与 cookbook 文档维护，与 DCU 硬件无关但服务于 DCU 适配的 SGLang 代码维护。

### 8. 算子卡域（`ops` 插件，2026-10-27 新增；skill 不带 `dcu-` 前缀）

> 与上文三域正交：**找方法**去性能调优域，**找源码**去算子开发域，**认算子 / 查历史结论 / 跑基准 / 判是否通过**来这里。入口 `Plugins/ops/SKILL.md`。

| skill | 定位 | 触发场景 |
|---|---|---|
| `op-sparse-mla` | DSA 稀疏 MLA 前向：q/kv/indices 契约、四变体矩阵（prod-vllm / tilelang / flash_mla / fp32 参考）、`-1` 哨兵与布局能力 | 稀疏注意力 kernel 选型、topk 性能问题 |
| `op-moe-w8a8` | FP8 MoE（routed GEMM1/GEMM2 + silu-mul + moe_sum）：M 分档、融合收益判据 | MoE 算子选型、融合改动判定 |
| `op-gemm-fp8` | FP8 GEMM（batched / 分组）：tilelang / hipBLASLt 两条路、gfx938 限制 | FP8 GEMM 选型与寻优 |
| `op-contract` | 契约怎么写：字段规范、容差分级、不变量清单、性能地板依据 | 新建/补齐算子卡 |
| `op-bench` | 怎么量：口径三问、六条铁律、MAD 显著性、跨卡 ratio | 微基准口径设计 |
| `op-profiling` | 实测卡在哪：PMC 计数器→`bound`/`facts`/`ratios` 三级证据；TA_* 跨 SE 求和、L1 无命中率等口径陷阱 | 「卡在算力还是内存」「能不能写『到顶』」「跨变体找区分项」 |
| `op-gate` | 怎么判：四闸四档（ACCEPTED / REJECTED / INCONCLUSIVE / DIAGNOSTIC-ONLY）、回退留痕 | 判断改动能否进生产 |
| `op-optimize` | 怎么持续推进：候选队列→单实验→机器判定→落盘记账，拒绝重跑已证伪项 | 优化做了一半要接着干 / 单模型跑完一轮 |
| `op-stack-design` | 先选型再排流水线：十一库画像 + 十条流水线定律，编译前否掉结构性不可行的设计 | 不知道用哪个库 / 单算子突破换成整库选型 |
| `op-roofline` | 先算再写：物理常数→推导规则→算子模型，输出**绑定约束**与地板；写不出预测不许写代码 | 「下一个改动往哪切」「这个设计会不会溢出/超 LDS」「为什么实测和预期差很远」 |
| `op-hip-compile` | 手写 HIP 编译调优：用 aicc 而非 hipcc、镜像 tilelang 的 `-mllvm` 集、从 .so 读 VGPR 算占用率 | 手写 HIP 慢但数值对 / 编译参数怎么对齐 |

> 各 skill 的分工边界以 `Plugins/ops/SKILL.md` §5.1「唯一权威表」为准，本表只作索引。

## 二、dcu-coding-skill 分类路由（面向代码检索/迁移/CI）

| skill | 定位 | 触发场景 |
|---|---|---|
| `dcu-dispatcher` | DCU Coding Skill 统一入口（意图分类→路由分发→migrate-and-test 工作流；跨域多步任务链编排已由 §0 `dcu-orchestrator` 承接） | 编码类意图分发 |
| `dcu-hip-reference` | HIP runtime/基础 API 查询（包内 `operators/dcu-coding-skill/skills/`，2026-10-19 收编，顶层 junction 同目标） | hip API 速查 |
| `dcu-triton-kernel` | Triton 在 HIP/DTK 语义检索 | Triton kernel 检索 |
| `dcu-code-migrator` | HIP/DTK 知识分片关键词检索（包内 `operators/dcu-coding-skill/skills/`，2026-10-19 收编，含 kb_shard 离线分片） | 代码迁移关键词 |
| `dcu-ci-testcase` | CI 测试用例管理 | 用例创建/验证/组织 |

> 注：`dcu-triton-kernel` / `dcu-ci-testcase` / `dcu-dispatcher` / `dcu-code-migrator` / `dcu-hip-reference` 在仓库内 `operators/dcu-coding-skill/skills/` 下（dispatcher 2026-10-17 收编入库，包根补 `marketplace.json`）；`dcu-hip-reference` / `dcu-code-migrator` 2026-10-19 收编，clone 本仓库即可用。跨域多步任务链（部署验证/故障排查/算子优化/推理优化 4 条）的编排执行层见 §0 `dcu-orchestrator`。

## 三、使用须知

- 遇 DCU 任务：单步明确（查配方/查版本/直接用某 skill 方法）**直接点名叶子 skill**，不经本表；跨 ≥2 域、多步不确定或拿不准时**先查本路由表**定位，再进具体 skill 读方法。
- **官方开源上游（HYGON-AI GitHub org）**：35 个仓库索引/版本锚点见 `../_dcu-toolkit-shared/references/hygon-upstream/repos-index.md`；刷新命令 `python3 ../_dcu-toolkit-shared/scripts/hygon-upstream-sync.py --check|--full`。涉及部署/适配/训练/监控任务、且任务面涉及这些上游时，先跑 `--check` 确认版本没变。
- 本 skill 不含方法本身；方法是各 skill 的正文。
- 本 skill 与 `dcu-inference-infra`（推理域路由器）互补：infra 更细，本表更全。
- 若某 skill 内容被上下文压缩/截断丢失，先完整重读该 skill 再继续。
- 同一会话内**已加载过的 skill 不重复重读**（多步链各节点常复用同一批 skill，重复加载纯耗上下文）；仅当内容被截断或跨会话后才重读。

## 四、常见任务链路（2026-10 会话审计归纳：7 条高频链）

> 多步/模糊任务先在此选链、按序加载链内 skill；每链给出入口→顺序→分支判定→退出条件。链上 skill 均在上文七域表中（标 † 者为本机独有 skill，git 仓库暂未收录）。单步明确任务不必走链，直接点名具体 skill。需要门槛审批（proceed: ask）与 evidence 留痕的编排执行层见 §0 `dcu-orchestrator`。

| 链路 | 典型触发 | 入口 → 顺序 | 分支判定 / 退出条件 |
|---|---|---|---|
| ① 部署验证 | 把某模型跑上 DCU 并确认可用 / 要吞吐数字 | `dcu-ssh` → `env/dcu-docker`（挂 hyhal）→ `inference/dcu-model-deploy-catalog`（查现成配方）→ `inference/dcu-vllm-deploy` →（涉新软件包时 `env/dcu-software-adaptation-guide`）→ `inference/dcu-model-deploy-verify` → `perf/dcu-llm-benchmark` | catalog 已有该模型配方 → 跳过 adaptation-guide；只要"跑通" → 到 model-deploy-verify 即退出，不进 benchmark |
| ② 性能优化闭环 | 吞吐/延迟不达标、算子慢、kernel 耗时异常 | `perf/dcu-llm-benchmark`（先定基线口径）→ `perf/dcu-perf-profiling` → `dcu-kernel-layer-microbench`（kernel 级细定位）→ `perf/dcu-op-optimization` →（找现成优化：`dcu-external-opt-scout` / `operators/dcu-triton-kernels` / `operators/dcu-kernelwiki`） | 无 baseline 不动手；收益未过噪声门禁 → 回滚并记录，不硬凑数字 |
| ③ 崩溃/故障定位 | 服务崩、VMFault、hang、输出乱码/精度异常 | `dcu-crash-log-triage` †（取证）→ `troubleshooting/dcu-vmfault-coredump`（faulting kernel + 越界值）或 `troubleshooting/dcu-ontrack-experience`（报错关键词反查）→ `troubleshooting/dcu-vllm-precision`（精度类）/ `dcu-sglang-eagle-sampling` †（EAGLE 环路） | coredump 缺失 → 先 `HSA_ENABLE_COREDUMP=1` 复现再取证；定位到具体算子 → 转 ② 链优化 |
| ④ 环境搭建 | 新节点/新容器初始化、多机测试前 | `dcu-ssh` → `env/dcu-docker` → `env/dcu-env` → `env/dcu-env-packages` → `env/dcu-driver` → `env/dcu-dtk` → `env/dcu-dtk-layout` → `env/dcu-hy-smi` → `env/dcu-topology` → `env/dcu-cpu-affinity` → `env/dcu-rccl-test`（多机） | 单卡巡检 → hy-smi 后即退出；多机 RCCL 带宽不达标 → 加 `env/dcu-roce-network` 查物理网络层 |
| ⑤ 源码移植 | 改 vLLM/SGLang/Triton 等源码做 DCU 适配 | 对应 `*-source` 索引（`inference/dcu-vllm-source` / `dcu-sglang-source` / `operators/dcu-triton-source` 等）→ `env/dcu-adaptation-overview`（版本兼容矩阵）→ `inference/dcu-vllm-op-replace` / `dcu-vllm-custom-op`（改动落点） | 只查结构/适配点不动源码 → source 索引即退出；改完必回 ① 链验证 |
| ⑥ 新模型可移植性评估 | 某模型能否即插即用上 DCU | `inference/dcu-model-portability` → `inference/dcu-libtorch`（C++ 部署时）→ `inference/dcu-migraphx-custom-op` / `dcu-migraphx-driver` / `perf/dcu-migraphx-perf`（ONNX 路径）→ `troubleshooting/dcu-ontrack-experience`（踩坑反查） | 模型已在 model-deploy-catalog → 直接走 ① 链查配方，跳过评估 |
| ⑦ 调研→复现→基线 | 外部项目/文章/新模型要在 DCU 上复现并拿基线数字（2026-10 实测 3 例：RoCE 调研、sglang 文章复现、新模型复现） | `web-tools-guide` † / `anysearch` † / `github-repo-deep-dive` †（通用调研工具，未随包分发）→ `dcu-modelzoo` / `dcu-das-hub-api` / `env/dcu-image-hub`（查适配仓库/包/镜像）→ `inference/dcu-model-deploy-catalog`（有现成配方则直查）→ `inference/dcu-vllm-deploy` 或 `inference/dcu-model-engine-runbook` †（复现起服务）→ `perf/dcu-llm-benchmark`（基线口径） | 调研发现无 DCU 适配 → 转 ⑥ 链评估；只要"跑通"不要数字 → 到复现即退出；基线后要优化 → 转 ② 链；编排执行走 `dcu-orchestrator` 的 `research-baseline` workflow |

> 链路来源：2026-07~10 真机会话 352 例 DCU 任务序列的调用栈聚类，非拍脑袋设计；链的执行编排（依赖/重试/审批/evidence）用 `dcu-orchestrator` 对应 workflow（① model-deploy、② infer-optimize / op-optimize、③ fault-triage、⑦ research-baseline；④ 环境搭建 / ⑤ 源码移植 / ⑥ 可移植性评估无内置 workflow——纯查询单步直达，需要验收留痕、占卡审批或跨会话续跑才升级 workflow）。**何时升级 workflow**：要出验收数字、要写设备状态、要 evidence 留痕跨会话续跑 → 升级；否则按链手工串。

## 五、改名/合并 alias 对照（旧名调用 miss 时查此表）

历史会话（尤其 2026-08 及更早）仍按旧名调用会 miss；本表为机器可读映射，合并细则见各目标 skill 说明。

| 旧名（已不存在，调用会 miss） | 现名 / 去向 | 说明 |
|---|---|---|
| `dcu-master-router` | `dcu-skill-router` | 旧总入口更名 |
| `dcu-runtime-patch` | `inference/dcu-vllm-runtime-patch` | 2026-08-07（v2.23.0）定名为 vLLM 运行时 patch 专用 |
| `dcu-vllm-source-0.15` / `-0.18` / `-0.21.0` / `-0.23.0` / `-0.26` | `inference/dcu-vllm-source` | 五个分版本合并为单 skill（按版本查其内章节） |
| `dcu-hipprof-baseline-trace`（及 `dcu-hipprof-*` 其余子域） | `perf/dcu-hipprof` | 五个子 skill 并入父 skill 方法论总纲 |
| `dcu-model-offline-delivery` | `inference/dcu-model-deploy-verify` + `env/dcu-docker` | 已下架；离线交付（docker save 成品 tar + README）场景拆入两者 |
| `dcu/env/dcu-image-hub` 等带斜杠路径式调用 | `env/dcu-image-hub` | skill 调用只认 name（不带域前缀），域前缀仅本表导航用 |
