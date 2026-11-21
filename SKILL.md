---
name: dcu-toolkit-open
description: "海光 DCU 测试调优技能集（开源版）总入口。覆盖基础环境、性能剖析、推理框架部署适配、算子开发、故障排查、工程实践，按 8 个插件组织，共 144 个 skill（139 顶层 + 5 嵌套子 skill，含薄编排层 dcu-orchestrator）。本文件是包级入口：路由表 + 安装说明。Claude Code / OpenCode 插件包结构（Plugins/<domain>/skills/）。"
version: 1.6.0
author: DCU Toolkit Contributors
license: MIT
---

# DCU Toolkit Skills（开源版）— 包级总入口

海光 DCU（Hygon DCU）测试调优技能集，按 Claude Code / OpenCode 插件包结构组织：

```
.claude-plugin/marketplace.json      ← marketplace 索引（8 个插件）
Plugins/
├── dcu-core/          核心路由与通用能力（skill-router、hicache、融合 kernel、微基准、训练手册、薄编排 orchestrator）
├── dcu-env/           基础环境与连接（SSH/环境检测/DTK/驱动/容器/hy-smi/NUMA/RCCL/RoCE/镜像重建）
├── dcu-perf/          性能剖析与优化（profiling 方法论/算子优化闭环/推理优化/benchmark）
├── dcu-inference/     推理框架部署与适配（vLLM 部署/精度/自定义算子/trace、MIGraphX、LMCache）
├── dcu-operators/     算子开发（CUDA-to-HIP 移植、Triton kernels、kernel wiki、MoE 融合）
├── dcu-troubleshooting/ 故障排查（XID/SXID、VM fault coredump、分布式 hang、训练精度）
├── dcu-engineering/   工程与发布实践（CI bisect、cherry-pick、代码规范、cookbook）
└── ops/               算子卡域（一算子一目录：契约 contract.yaml / 变体 variants.yaml / 档位 shapes / 基准 bench / 判定 tests / 台账 ledger）
data/ references/ scripts/ docs/   共享数据层（跨 skill 引用）
```

> 共享层说明：`data/`（issue-kb 问题知识库、product-matrix 产品矩阵、templates 卡片/工单模板、perf-db-schema.sql 性能库结构）、`docs/`（STANDARDS.md）、`references/`、`scripts/` 为跨 skill 复用的领域数据/规范与工具；归入各 skill 的目录/工件处理不了时放在这里，供相关 skill 按需引用。
> 沉淀闭环：`dcu-orchestrator` 的**算子优化工作流**（profile→optimize→validate→distill、调研复现 research-collect→env-reproduce(ask)→baseline-run）收尾的 `op-distill` 门控节点会把本次结果沉淀到**用户知识库** `~/.config/dcu-toolkit/kb/op-opt/`（台账 + 证据 JSON，数字只来自 run 产物，缺失字段标 MISSING）；`issue/` 故障记录与 `solution/` 部署方案卡片**暂无自动沉淀节点**，链路收尾时按 `data/kb/README.md` 的目录与写入约定**人工补写**（模板见 `data/kb/templates/`）。skill 树与共享层是厂商内容，重装/升级以包内版本为准，**不往里面写沉淀**（包内 `data/issue-kb/`、`data/product-matrix/` 仅作只读模板）。
> 安装态路径：安装时共享层同步到 `<skills根>/_dcu-toolkit-shared/`，skill 内引用包根共享资源的路径在安装态解析为 `../_dcu-toolkit-shared/`。

## 工具介绍

海光 DCU（Hygon DCU）测试调优技能集，按 Claude Code / OpenCode 插件包结构组织，覆盖 DCU 全生命周期：
基础环境与连接、性能剖析与调优、推理框架部署与适配、算子开发、故障排查、训练与工程实践。

- **规模**：144 个 skill（139 顶层 + 5 嵌套子 skill），8 个插件域（dcu-core / dcu-env / dcu-perf / dcu-inference / dcu-operators / dcu-troubleshooting / dcu-engineering / ops）
- **组织**：每个 skill = `SKILL.md`（方法论与流程）+ `scripts/`（可执行工具）+ `references/`（领域知识），自包含、按需加载
- **路由**：包级路由表（下节）→ `dcu-skill-router` 全量索引 → 具体 skill；多步任务走 `dcu-orchestrator` 内置工作流
- **共享层**：`data/ references/ scripts/ docs/` 为跨 skill 复用的领域数据与工具（见上文说明）

## 路由表（按用户意图分发）

| 用户意图关键词 | 分发目标 | 插件 |
|---|---|---|
| skill 路由 / 全量能力索引 | dcu-skill-router | dcu-core |
| 多任务编排 / 工作流契约 / 证据验收（五条内置工作流：推理优化 env-check→baseline→quantize、模型部署 precheck→launch→verify、故障排查 collect→locate→fix、算子优化 profile→optimize→validate→distill、调研复现 research-collect→env-reproduce(ask)→baseline-run） | dcu-orchestrator | dcu-core |
| KV Cache 卸载 / LMCache / 长上下文降 TTFT | dcu-hicache、dcu-lmcache | dcu-core / dcu-inference |
| 自定义融合 kernel / kernel 层微基准 / 训练手册 | dcu-custom-fusion-kernel、dcu-kernel-layer-microbench、dcu-train-cookbook | dcu-core |
| SSH 连接 / 远程 / 登录设备 | dcu-ssh | dcu-env |
| 环境检测 / 设备信息 / 适配包安装 | dcu-env、dcu-env-packages | dcu-env |
| 驱动安装升级 / hy-smi 监控 / DTK 安装激活与目录结构 | dcu-driver、dcu-hy-smi、dcu-dtk、dcu-dtk-layout | dcu-env |
| 拓扑 / NUMA 绑核 / CPU 亲和 | dcu-topology、dcu-cpu-affinity | dcu-env |
| 集合通信 RCCL 通测 / 多机 RoCEv2 网络 | dcu-rccl-test、dcu-roce-network | dcu-env |
| Docker 容器操作 / 镜像反推 Dockerfile 重建 / 镜像→物理机复刻 | dcu-docker、dcu-image-reverse-rebuild、dcu-image-to-native-env | dcu-env |
| Kubernetes 部署 / DCU Device-Plugin / vDCU 切分 | dcu-kubernetes | dcu-env |
| 性能分析全栈总纲 / 工具选型路由 | dcu-perf-analysis | dcu-perf |
| 性能量化 / rocprof / hipprof 采集 | dcu-perf-profiling、dcu-hipprof、dcu-framework-profilers | dcu-perf |
| 算子慢 / kernel 优化 / 加速比验证 | dcu-op-optimization | dcu-perf |
| 通用模型推理优化（torch.compile/NHWC/TF32/MIOpen/rocBLAS） | dcu-inference-optimization | dcu-perf |
| 真实优化案例（LSTM/UNet/Qwen/SGLang 等） | dcu-perf-cases | dcu-perf |
| LLM 端到端压测 / vllm bench / sglang bench | dcu-llm-benchmark | dcu-perf |
| vLLM trace 按 kernel 类别汇总 | dcu-vllm-trace-analysis | dcu-perf |
| torch.profiler 层级追踪 / 计算建模 | llm-torch-profiler-analysis、model-compute-simulation、llm-pipeline-analysis | dcu-perf |
| SGLang 工程规范 / CI bisect / cherry-pick / PR 优化史 | sglang-* 系列、model-pr-* 系列 | dcu-perf / dcu-engineering |
| vLLM 部署 / 启动参数 / 精度定位 / runtime patch / 自定义算子 | dcu-vllm-deploy、dcu-vllm-precision、dcu-vllm-runtime-patch、dcu-vllm-custom-op、dcu-vllm-op-replace | dcu-inference |
| 模型部署验证 / 部署配方目录 / 可行性评估 | dcu-model-deploy-verify、dcu-model-deploy-catalog、dcu-model-portability | dcu-inference |
| MIGraphX 自定义算子 / driver 扩展 / ORT 推理 profile | dcu-migraphx-*、dcu-ort-perf、dcu-ort-profile | dcu-inference |
| LibTorch C++ 推理 / CosyVoice TTS 优化 | dcu-libtorch、dcu-cosyvoice-opt | dcu-inference |
| CUDA→HIP 移植 / 指令映射 | dcu-cuda-hip-port、dcu-cuda-to-hip-port | dcu-operators |
| HIP kernel 开发规范 / kernel 优化知识库 / Triton 算子 | dcu-kernel-dev、dcu-kernelwiki、dcu-triton-kernels | dcu-operators |
| MoE 融合算子（AITER/LightOp）/ DeepGEMM 架构 | dcu-aiter-moe、dcu-lightop-moe、dcu-lightop-fusion、dcu-deepgemm-arch | dcu-operators |
| XID/SXID 错误码 / VM fault coredump 取证 | dcu-vmfault-coredump | dcu-troubleshooting |
| CUDA crash / 分布式 hang / 训练精度 NaN / vLLM 精度 | debug-cuda-crash、debug-distributed-hang、dcu-train-precision、dcu-vllm-precision、dcu-engine-precision-localization | dcu-troubleshooting |
| CI 工作流 / 环境变量规范 / 代码风格 / 模型接入 cookbook | ci-workflow-guide、env-var-conventions、cookbook-* 等 | dcu-engineering |
| 点名某个算子 / 建算子卡 / 查某算子历史结论 / 判断 kernel 改动是否算通过 / 推进一轮优化 / **先算出预测再写代码** / 手写 HIP 慢但数值对 | op-sparse-mla、op-moe-w8a8、op-gemm-fp8；方法见 op-contract、op-bench、op-profiling、op-gate、op-optimize、op-roofline、op-hip-compile | ops |

## 安装与使用

### Claude Code（marketplace 方式）

```bash
claude plugins marketplace add <repo-url>
claude plugins install dcu-perf --marketplace dcu-toolkit-marketplace
# 或安装全部
claude plugins install dcu-core dcu-env dcu-perf dcu-inference dcu-operators dcu-troubleshooting dcu-engineering ops --marketplace dcu-toolkit-marketplace
```

### 脚本方式（Claude Code / OpenCode 通用）

```bash
bash install.sh [level] [tool] [install_path]
# 例：bash install.sh project claude        # 项目级装到 .claude/skills/
#     bash install.sh global opencode       # 全局装到 ~/.config/opencode/skills/
# 多框架：bash install.sh                   # 默认 --detect，自动装到全部检测到的 agent
#         bash install.sh --tool codex      # 只装到指定 agent
#         bash install.sh --list            # 只列出检测到的 agent
```

> 共享层：安装时包根 `data/ references/ scripts/ docs/` 会同步到 `<skills根>/_dcu-toolkit-shared/`。
> skill 内引用包根共享资源（`references/hygon-upstream/`、`scripts/hygon-upstream-sync.py`、`references/dcu-models.md` 等）
> 的路径在安装态解析为 `../_dcu-toolkit-shared/`（开发态仍在包根）。

### 使用

安装完成后，Agent 侧入口：

1. **单任务**：按本文件「路由表（按用户意图分发）」定位目标 skill，读取其 `SKILL.md` 按流程执行；
2. **全量索引**：`dcu-skill-router`（dcu-core 插件）按七域列出全部 skill 的触发场景；
3. **多任务编排**：`dcu-orchestrator` 五条内置工作流（推理优化 / 模型部署 / 故障排查 / 算子优化 / 调研复现基线），三段式 probe 采集 → normal 分析验证 → gate 写状态（需审批）。

## 安全说明

- 包内零密码、零密钥、零内网地址：真实凭证在 `~/.config/dcu-toolkit/credentials.json`（用户侧自建，chmod 600）；示例中的 `<DCU_NODE_IP>` `<DCU_CODE_HOST>` 等为占位符
- 发布流程含自动脱敏检查（仓库根 `sanitize.py` + CI grep 断言零残留）
- 远程密码认证用 `sshpass -f <文件>` 或 `sshpass -e`，不要用 `sshpass -p`
