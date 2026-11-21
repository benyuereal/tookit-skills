---
name: dcu-operators
description: "DCU 算子开发技能集。覆盖海光 DCU 算子开发技能集：CUDA→HIP 移植、HIP/Triton 算子开发、kernel 优化知识库与 MoE 融合算子。"
version: 1.5.1
author: DCU Toolkit Contributors
license: MIT
---

# DCU 算子开发 — 域级技能包

## 工具介绍

海光 DCU 算子开发技能集：CUDA→HIP 移植、HIP/Triton 算子开发、kernel 优化知识库与 MoE 融合算子。本目录为 Kits Platform Skill Tool `dcu-operators` 的发布单元，包含 20 个子 skill：

| skill | 说明 |
|---|---|
| dcu-aiter-moe | AITER MoE融合算子四后端调度与适配索引 |
| dcu-aiter-source | AITER算子库DCU源码结构与适配点索引 |
| dcu-coding-skill | DCU Coding Skill 整包统一入口：dcu-dispatcher 意图分发 + 五个子 skill（dcu-hip-refere… |
| dcu-cuda-hip-port | Port CUDA code to DCU/HIP on DTK. Use for CUDA→HIP porting. |
| dcu-cuda-to-hip-port | Port a standalone CUDA C++ project to Hygon DCU/HIP. |
| dcu-deepep-source | DeepEP DCU 适配源码结构——MoE 专家并行 dispatch/combine（Buffer + low_latency 路由策略… |
| dcu-deepgemm-arch | DeepGEMM三路径架构与MODE调优机制：masked/contiguous/ll选型与layout陷阱 |
| dcu-deepgemm-source | DeepGEMM通用GEMM库DCU源码结构与适配 |
| dcu-hytlass-source | hytlass（CUTLASS 3.5.0 的 HCU 移植）源码与用法索引：两套编程模型 + 示例清单 + 何时用它搭自己的 GEMM |
| dcu-flashattn-source | flash-attn 海光 DCU 适配源码索引：三层适配结构 + FA3 不可移植结论 + 版本谱系 |
| dcu-hipdsa-source | hip-dsa / hip-dsa-kernels / hipdnn DCU 适配——cudnn.DSA（DeepSeek 稀疏注意力）HI… |
| dcu-kernel-dev | DCU算子开发规范：HIP kernel开发与TileLang DSL写算子两路径 |
| dcu-kernelwiki | 海光DCU kernel优化知识库，agent可查询的可复用方法与事实 |
| dcu-lightop-fusion | Use when writing or understanding LightOp fusion kernels. |
| dcu-lightop-moe | LightOp MoE 融合算子多量化路径与适配索引 |
| dcu-lightop-source | LightOp海光融合算子库源码结构与适配 |
| dcu-mhc-kernels-source | MHC/线性注意力 kernel 包 DCU 适配——causal-conv1d + fast-hadamard-transform + m… |
| dcu-tilelang-source | TileLang DCU 适配源码结构——TVM 系 DSL 编译器（contrib/hcu.py 海光 HIP 后处理 + TILELAN… |
| dcu-triton-kernels | Use when writing/optimizing Triton kernels on DCU (Hygon BW1000), or m… |
| dcu-triton-source | Triton DCU/HCU 源码结构——独立 hcu backend（DTK 特化：libgalaxyhip/aillvm/gfx936-… |

## 安装与使用

- **单 skill 使用**：Agent 按上表选定 skill 后，读取 `skills/<skill>/SKILL.md` 并按其流程执行（子 skill 自包含，含 scripts/ 与 references/）。
- **统一路由**：跨域任务优先读仓库根 `SKILL.md`（包级路由表）或 `dcu-core/skills/dcu-skill-router`（全量索引）。
- **安装**：见仓库根 `README.md` / `install.sh`（支持 Claude Code / OpenCode / Codex 等多框架，项目级或全局）。
- **版本**：与包根 SKILL.md 一致（1.5.1）；升级以包内版本为准。
