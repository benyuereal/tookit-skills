---
name: dcu-perf
description: "DCU 性能剖析与调优技能集。覆盖海光 DCU 性能剖析与优化技能集：profiling 方法论、算子优化闭环、推理优化、压测与真实案例库。"
version: 1.5.1
author: DCU Toolkit Contributors
license: MIT
---

# DCU 性能剖析与调优 — 域级技能包

## 工具介绍

海光 DCU 性能剖析与优化技能集：profiling 方法论、算子优化闭环、推理优化、压测与真实案例库。本目录为 Kits Platform Skill Tool `dcu-perf` 的发布单元，包含 31 个子 skill：

| skill | 说明 |
|---|---|
| dcu-external-kernel-validation | Vet external GPU kernel optimizations end-to-end on DCU. |
| dcu-external-opt-scout | Vet GitHub GPU/DCU optimization projects for DCU testing. |
| dcu-framework-profilers | 六大框架在DCU上Profiler实战：PT/TF/ORT/MIGraphX/vLLM/Megatron |
| dcu-gdn-prefill-kernel-opt | GDN prefill kernel 优化。优化 DCU 上 GDN prefill TTFT 时加载。 |
| dcu-hipprof | hipprof性能分析总纲：Trace/PMC/RCCL/MPI/泄漏/CoreDump（子域已并入父，方法论总纲） |
| dcu-inference-optimization | DCU通用模型推理优化方法论 |
| dcu-llm-benchmark | DCU大模型推理压测 + W8A8 INT8 首字慢归因、tune-cache 补桶、冷编译 warmup |
| dcu-migraphx-perf | 海光DCU上MIGraphX推理引擎性能优化最佳实践（测量/批处理/序列化/精度/融合/TensorCore/多框架/精度调试/环境配置） |
| dcu-op-optimization | DCU算子优化闭环：profile→定位→改kernel→验证正确性→验证性能 |
| dcu-ort-perf | 海光DCU上ONNX Runtime推理引擎性能优化最佳实践（perf_test测量/批处理/FP16/EP选择/图优化/多线程/io_bi… |
| dcu-ort-profile | ORT在DCU上的Profile机制解析+MOE显存雪崩HIP流修复 |
| dcu-perf-analysis | DCU性能分析全栈总纲：五层体系与工具选型路由入口 |
| dcu-perf-cases | DCU真实性能优化案例库：跨平台对标+根因定位+量化对比 |
| dcu-perf-profiling | DCU性能剖析方法论：rocprof/rocTracer命令模板与瓶颈判定 |
| dcu-vllm-trace-analysis | 解析 vLLM V1 torch.profiler trace（.pt.trace.json[.gz]），定位 prefill/decode… |
| dcu-vllm-wiki | Navigate project evidence for vLLM and SGLang adaptation on DCU, HCU, … |
| generate-profile | Capture reproducible SGLang or vLLM serving profiles on CUDA, HIP, DCU… |
| humanize-gen-plan | Convert an LLM inference optimization request into a pre-registered ex… |
| humanize-rlcr | Run a resumable evidence-driven LLM inference optimization loop from a… |
| llm-pipeline-analysis | Decompose LLM serving traces into mutually exclusive latency groups, e… |
| llm-serving-auto-benchmark | Framework-independent LLM serving benchmark skill for comparing SGLang… |
| llm-serving-capacity-planner | Plan and audit LLM serving memory capacity from startup logs, model ge… |
| llm-torch-profiler-analysis | 面向 `sglang`、`vllm` 和 `TensorRT-LLM` 的统一 LLM torch-profiler 分析 skill。用于… |
| model-architecture-diagram | Return public original model architecture diagrams for user-specified … |
| model-compute-simulation | Build an operator-level compute template for an LLM and estimate FLOPs… |
| model-pr-optimization-history | Use when an SGLang, vLLM, or TensorRT-LLM serving/model optimization t… |
| sglang-humanize-review | 用 SGLang 人类 review 语料（至 2026-07-27）做 PR 审查 |
| sglang-model-day0-support | SGLang 新模型 Day-0 接入：支持范围/架构缺口/PR DAG/验证矩阵 |
| sglang-prod-incident-triage | Replay-first debug flow for SGLang serving problems. Use when a live o… |
| torch-profiler-layer-track | 给 Torch Profiler trace 加 L0/L1 层导航标记与紧凑 GPU 轨道 |
| vllm-pr-summary | Fetch and analyze a Pull Request from the vllm-project/vllm GitHub rep… |

## 安装与使用

- **单 skill 使用**：Agent 按上表选定 skill 后，读取 `skills/<skill>/SKILL.md` 并按其流程执行（子 skill 自包含，含 scripts/ 与 references/）。
- **统一路由**：跨域任务优先读仓库根 `SKILL.md`（包级路由表）或 `dcu-core/skills/dcu-skill-router`（全量索引）。
- **安装**：见仓库根 `README.md` / `install.sh`（支持 Claude Code / OpenCode / Codex 等多框架，项目级或全局）。
- **版本**：与包根 SKILL.md 一致（1.5.1）；升级以包内版本为准。
