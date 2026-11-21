---
name: dcu-inference-infra
description: DCU推理infra子域路由器，分发到各skill
version: 1.0.1
author: DCU Engineer
license: MIT
---

# DCU 推理 Infra 总入口（dcu-inference-infra）

本 skill 是 **DCU 推理 infra 子域的路由器 + 组合剧本层**，从 `dcu-toolkit-skills` 包根（硬件/测试/调优工具）下钻到"推理框架适配与优化"场景。按用户意图分发，并按剧本组合调用子 skill。

## 路由表（问题 → skill）

| 用户意图关键词 | 分发目标 | 说明 |
|---|---|---|
| PyTorch 源码编译 / 低版本适配 / LibTorch C++ 提取 / 算子兼容性验证 / DTK 上编 torch | `dcu-pytorch-source` | DTK 26.04 编低版本 PyTorch + 196 项算子验证；含高版本 wheel 差异 |
| DCU 适配软件包安装 / sourcefind 私有 PyPI / 按 DTK 拉 +das 轮 / torch 绑定 / 环境缺包排查 | `dcu-env-packages` | 包清单 + 索引路径 + dtk2604 torch 绑定矩阵 + 排障 |
| vLLM 源码结构 / 改哪 / 后端注册 / attention backend / 算子注册 / 构建编译 | `dcu-vllm-source` | 框架源码地图 |
| 算子慢 / 想优化 kernel / profile 后怎么改 / 加速比怎么验证 | `dcu-op-optimization` | 优化闭环方法学 |
| 怎么量化性能 / rocprof 怎么用 / 瓶颈类型判定 / 指标约定 | `dcu-perf-profiling` | 剖析方法论 |
| 部署 vLLM / 启动参数 / 容器怎么起 / 已知坑 | `dcu-vllm-deploy` | 部署经验速查 |
| Triton Server + ORT 集成调优 / config.pbtxt / 多实例 / 动态批处理 / 七步 SOP / 反模式 | `dcu-ort-perf` §7 | 调度层×算子层正交调优（YOLOv7 BW100 案例见 `dcu-perf-cases` 案例 16） |
| vLLM 0.15.x 源码 / 深度改造版适配点（与 0.18 拆分） | `dcu-vllm-source` | 已实测 |
| 自定义算子替换 vLLM 算子 / HIP kernel→注册→重定向→重编译→验证 | `dcu-vllm-op-replace` | ✅ |
| vLLM 推理精度异常 / 输出乱码·NULL·token 异常 / 与 HF 对不齐 / 定位具体算子（逐层 cos 对比→层内钻取 attn/mlp/router→绕过验证→出报告） | `dcu-vllm-precision` | 推理精度定位方法论（含 vLLM 源码调用链抽取） |
| 多卡通信差 / RCCL 带宽 | `dcu-rccl-test` | 集合通信测试 |
| AITER 算子库源码 / 海光 das 适配 / 四后端机制 / 算子族索引 | `dcu-aiter-source` | ✅ 已实测 |
| AITER MoE 融合算子 / 四后端调度 / shuffle 踩坑 / 调优表 | `dcu-aiter-moe` | ✅ |
| `dcu-lightop-source` | LightOp 自研融合算子库 / 海光 DTK-gfx 适配 / 与 AITER 选型关系 | ✅ |
| `dcu-lightop-moe` | LightOp MoE 融合算子深度索引 / 多量化路径 / permute 机制 / vLLM 兼容层 | ✅ |
| SGLang/vLLM/TRT-LLM 服务已跑出 trace.json，想分析 kernel/空泡/融合（torch.profiler 系） | `llm-torch-profiler-analysis` | 通用 AI-Infra skill（BBuf HG-AI-Infra-SKILLS，互补） |
| 同模型跨 SGLang/vLLM/TRT-LLM 比吞吐/延迟、自动搜最优部署命令 | `llm-serving-auto-benchmark` | 同上，依赖 dcu-inference-cookbook（需自备，zip 未打包） |
| 要某模型的架构图/结构图链接（DeepSeek/GLM/Qwen/Kimi 等） | `model-architecture-diagram` | 同上 |
| 问某模型算子级算力/FLOPs/MFU/并行占比 | `model-compute-simulation` | 同上 |
| 优化开工前立预注册实验计划（固定工作负载/门禁/证据契约/停止条件） | `humanize-gen-plan` | autoopt 性能闭环（2026-08-19 整合） |
| 采集可复现的 SGLang/vLLM 服务 profile（graph-off/on、prefill/decode、rank 级 trace） | `generate-profile` | autoopt 性能闭环 |
| 服务 trace 延迟拆解到 prefill/decode/MoE/attn/通信/rank 长尾 | `llm-pipeline-analysis` | autoopt 性能闭环 |
| 显存容量规划 / OOM·VMFault 诊断 / mem-fraction 调优 / 并发上限 | `llm-serving-capacity-planner` | autoopt 性能闭环 |
| 多轮可恢复优化闭环（baseline→profile→patch→A/B→接受/回滚） | `humanize-rlcr` | autoopt 性能闭环 |
| SGLang 线上问题（延迟回归/崩溃/队列增长）转 replay 调试 | `sglang-prod-incident-triage` | 同上 |
| 让某模型 SGLang 追上/超过最优框架的自主优化循环 | `sglang-sota-humanize-loop` | 同上（依赖未打包的 Humanize/KernelPilot 运行时） |
| 分析/总结 vLLM 官方 PR 号（Mermaid 图+风险） | `vllm-pr-summary` | 同上 |
| 改模型专属 serving 路径前查开源社区 PR 证据 | `model-pr-optimization-history` | 同上（按模型族知识库） |

## 子 skill 清单（推理 infra 子域，含通用互补 skill 共 23 个）

| 子 skill | 职责 | 状态 |
|---|---|---|
| `dcu-vllm-source` | vLLM 各版本源码结构与 DCU 适配点索引（合并 0.15/0.18/0.21/0.23/0.26；0.26 段待源码核验） | ✅ |
| `dcu-vllm-op-replace` | 自定义算子替换 vLLM 原生算子（HIP kernel→注册→重定向→重编译→验证） | ✅ |
| `dcu-op-optimization` | 算子优化标准闭环（profile→定位→改→验证） | ✅ |
| `dcu-perf-profiling` | 性能剖析方法论（rocprof/rocTracer/指标约定） | ✅ |
| `dcu-vllm-deploy` | vLLM 部署调优经验速查 | ✅ |
| `dcu-aiter-source` | AITER 算子库源码 / 海光 das 适配 / 四后端机制 / 算子族索引 | ✅ 已实测 |
| `dcu-aiter-moe` | AITER MoE 融合算子深度索引 / 四后端调度 / shuffle 踩坑 / 调优表 | ✅ |
| `dcu-lightop-source` | LightOp 自研融合算子库 / 海光 DTK-gfx 适配 / 与 AITER 选型关系 | ✅ |
| `dcu-lightop-moe` | LightOp MoE 融合算子深度索引 / 多量化路径 / permute 机制 / vLLM 兼容层 | ✅ |
| `dcu-deepgemm-source` | DeepGEMM 自研 GEMM 库 / grouped GEMM 路径 / 与 AITER·LightOp GEMM 选型 | ✅ |
| `dcu-flashattn-source` | flash-attn 海光 DCU 适配源码索引 / 三层适配结构 / FA3 不可移植结论 / 版本谱系 | ✅ |
| `dcu-pytorch-source` | 低版本 PyTorch 源码编译适配（DTK26.04）+ LibTorch 提取 + 196 项算子验证；高版本预编译 wheel 适配差异 | ✅ 已填充 |
| `dcu-env-packages` | DCU 适配软件包清单 / sourcefind 私有 PyPI / dtk2604 torch 绑定矩阵 / 缺包排障 | ✅ 已填充 |
| `dcu-vllm-precision` | vLLM 推理精度异常定位（算子级：逐层 cos 对比→层内钻取 attn/mlp/router→绕过验证→出报告，含 vLLM 源码调用链抽取） | ✅ 新建 |
| `llm-torch-profiler-analysis` | 统一 LLM torch.profiler 分析（SGLang/vLLM/TRT-LLM） | 通用互补（已整合） |
| `llm-serving-auto-benchmark` | 跨框架自动 benchmark 搜索 | 通用互补（已整合，依赖 dcu-inference-cookbook） |
| `model-architecture-diagram` | 公开原始模型结构图检索 | 通用互补（已整合） |
| `model-compute-simulation` | 算子级算力/FLOPs/MFU 估算 | 通用互补（已整合） |
| `sglang-prod-incident-triage` | SGLang 线上问题 replay-first 调试 | 通用互补（已整合） |
| `sglang-sota-humanize-loop` | SGLang SOTA 自主优化 Humanize 循环 | 通用互补（已整合，依赖 Humanize/KernelPilot 运行时） |
| `vllm-pr-summary` | vLLM 官方 PR 抓取+结构化分析 | 通用互补（已整合） |
| `model-pr-optimization-history` | 开源社区模型优化 PR 历史库 | 通用互补（已整合） |
| `humanize-gen-plan` | 优化需求转预注册实验计划（工作负载/门禁/证据契约/停止条件） | autoopt 性能闭环（2026-08-19 整合） |
| `generate-profile` | 采集可复现服务 profile（graph-off/on、prefill/decode、rank 级） | autoopt 性能闭环 |
| `llm-pipeline-analysis` | 服务 trace 延迟拆解（prefill/decode/MoE/attn/通信/rank 长尾） | autoopt 性能闭环 |
| `llm-serving-capacity-planner` | 显存容量规划 / OOM·VMFault 诊断 / mem-fraction 调优 | autoopt 性能闭环 |
| `humanize-rlcr` | 可恢复证据驱动优化闭环（baseline→profile→patch→A/B→接受/回滚） | autoopt 性能闭环 |

> 实建中：`dcu-sglang-source`（SGLang 源码）、`dcu-kernel-dev`（HIP kernel 开发规范）均已建成。
> 性能优化闭环（autoopt 2026-08-19 整合）：`humanize-gen-plan` → `generate-profile` → `llm-torch-profiler-analysis`/`llm-pipeline-analysis` → `model-compute-simulation`/`llm-serving-capacity-planner` → 候选实现 → `llm-serving-auto-benchmark` → `humanize-rlcr`。

## 版本化约定（硬性，用户决策）
**vLLM 各版本源码适配索引已合并为单一 `dcu-vllm-source`**（覆盖 0.15/0.18/0.21.0/0.23.0/0.26 五版，各版本分区小节），**不**再为每个大版本单独建 `dcu-vllm-source-<版本>` skill。理由（2026-08 用户决策，CHANGELOG 2.22.0「合并 vLLM 五版」）：
- 五版差异集中在 attention backend 目录（0.18 在 `vllm/v1/attention/backends/`，旧版在 `vllm/attention/`）、platform 注册点等少量维度，合并成一张「版本→适配点」对照索引比 5 个独立 skill 更易横向对比、避免互相污染。
- 合并后以 `dcu-vllm-source` 内的「版本分区 + 差异对照」覆盖多版本；新增/旧版本不再各自建独立 skill。
- 历史分别记录五个版本的分版本内容（0.15/0.18/0.21.0/0.23.0/0.26）已收敛为合并版内的「各版本特有适配要点」分区小节，不再单独建 `dcu-vllm-source-<版本>` skill（变更史见 CHANGELOG 2.22.0）。

**新版本接入流程（标准动作）**：
1. 解压发布包到 `~/dcu-src/vllm-<版本>/`，与上游同版本并排。
2. 跑 `dcu-vllm-source/scripts/diff-hygon-release.sh <上游> <海光发布版>` 定位真实差异（海光发布版往往 Python 层零差异、仅 CMakeLists.txt 裁 gfx）。
3. 在 `dcu-vllm-source` 合并版内**新增一个版本分区小节**：先写「版本变更要点（来自发布说明/素材）」，源码级路径等真实包到位后填；路径复用上一版本的探查命令。
4. 包根 `dcu/SKILL.md`：related_skills + 路由表 + 版本号升位；`dcu-skill-router/SKILL.md` 六域路由表 + 计数（⚠ 4.0.0 起根 SKILL.md 不再维护「子 skill 清单」表格，计数源迁到 `dcu-skill-router`），各方一致。

## 组合剧本（标准工作流）

**剧本 1「vLLM 部署慢」**
`dcu-vllm-deploy`（看已知坑/启动参数）→ `dcu-perf-profiling`（量化定位瓶颈 kernel）→ `dcu-op-optimization`（若需改算子）→ `dcu-vllm-source`（定位源码落点）

**剧本 2「新模型上线」**
`dcu-vllm-source`（确认适配点/后端）→ `dcu-vllm-deploy`（起服务）→ `dcu-perf-profiling`（压测）

**剧本 3「算子性能追查」**
`dcu-perf-profiling`（量化）→ `dcu-op-optimization`（改+验证闭环）→ `dcu-vllm-source`（若目标是 vLLM 内部算子）

**剧本 4「报错/故障」**
`dcu-ontrack-experience`（关键词反查）→ 按结论回灌到上述剧本对应步骤

**剧本 5「DTK 上编低版本 PyTorch / LibTorch 提取」**
`dcu-pytorch-source`（阶段一编译 + 阶段二 LibTorch 提取 + 阶段三 196 项算子验证）→ 若要做自定义算子再接 `dcu-vllm-op-replace` 思路

## 通用纪律（子类共享）
- 不存性能数据/基线/日志：数据用 `dcu-op-optimization` 模板留痕。
- 容器必挂 `/opt/hyhal:/opt/hyhal:ro`；gfx 用物理机 `rocminfo`；指定卡用 `HIP_VISIBLE_DEVICES`。
- 海光发布版的"真实改动"已用 `dcu-vllm-source/scripts/diff-hygon-release.sh` 验证方法闭合（0.18 实证：Python 层零差异、仅 CMakeLists.txt 裁 gfx）。新版本到手先跑该脚本，不要臆测适配点；sourcefind 克隆用 PAT 而非密码。
- 高危操作谨慎，优先测试机验证。

## 使用须知
- 经验/方法仅供参考，不保证修复。先核对设备型号+驱动/DTK版本+报错原文。
- 本入口只做路由与编排，具体手段在子 skill；子 skill 缺失时告知用户"规划中"。
