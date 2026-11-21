---
name: dcu-troubleshooting
description: "DCU 故障排查技能集。覆盖海光 DCU 故障排查技能集：XID/SXID 错误码、VM fault coredump、CUDA crash、分布式 hang 与训练精度。"
version: 1.6.0
author: DCU Toolkit Contributors
license: MIT
---

# DCU 故障排查 — 域级技能包

## 工具介绍

海光 DCU 故障排查技能集：XID/SXID 错误码、VM fault coredump、CUDA crash、分布式 hang 与训练精度。本目录为 Kits Platform Skill Tool `dcu-troubleshooting` 的发布单元，包含 9 个子 skill：

| skill | 说明 |
|---|---|
| cross-engine-acceptance-gate | 移植/换引擎后的精度验收门禁。bf16 跨引擎逐 bit 一致不可作硬门禁，用判别实验定性后分级放行。 |
| dcu-cudagraph-baked-scalars | 排查 CUDA/HIP graph 下"每 batch 变化的 Python 标量被烧死在 capture"导致的静默读错（accept 从… |
| dcu-engine-precision-localization | 自定义C++推理引擎DCU精度定位到算子。引擎能跑通但输出垃圾时用。 |
| dcu-ontrack-experience | DCU故障排查速查：按报错关键词反查根因与规避手段 |
| dcu-train-precision | DCU训练精度排查：NV对齐、逐层定位异常层、NaN修复方法论 |
| dcu-vllm-precision | 定位 vLLM 推理精度问题到具体算子。输入模型路径，逐步用 hook 抓中间态、对比 transformers，锁定误差首次出现的算子。适… |
| dcu-vmfault-coredump | Locate the faulting kernel and the offending memory value from a Hygon… |
| debug-cuda-crash | Debug CUDA or HIP kernel crashes in SGLang and vLLM with pre-call logg… |
| debug-distributed-hang | Debug stalls and hangs in distributed SGLang inference across TP, PP, … |

## 安装与使用

- **单 skill 使用**：Agent 按上表选定 skill 后，读取 `skills/<skill>/SKILL.md` 并按其流程执行（子 skill 自包含，含 scripts/ 与 references/）。
- **统一路由**：跨域任务优先读仓库根 `SKILL.md`（包级路由表）或 `dcu-core/skills/dcu-skill-router`（全量索引）。
- **安装**：见仓库根 `README.md` / `install.sh`（支持 Claude Code / OpenCode / Codex 等多框架，项目级或全局）。
- **版本**：与包根 SKILL.md 一致（1.6.0）；升级以包内版本为准。
