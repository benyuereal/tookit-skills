---
name: dcu-coding-skill
version: 1.2.3
description: DCU Coding Skill 整包统一入口：dcu-dispatcher 意图分发 + 五个子 skill（dcu-hip-reference HIP API 查询、dcu-code-migrator CUDA→HIP 迁移知识分片、dcu-ci-testcase 测试用例、dcu-triton-kernel kernel 开发，skills/ 下平铺），common/ 提供 KB 分片检索与 SPI 路由基础设施。
category: entry
standalone: true
author: DCU Team
license: MIT
---

# DCU Coding Skill

这是整包入口 skill。查询与任务会先进入 dispatcher，再按意图分发到对应子 skill：

- `dcu-hip-reference`（HIP API 查询）
- `dcu-code-migrator`（CUDA -> HIP 迁移）
- `dcu-ci-testcase`（测试用例生成）
- `dcu-triton-kernel`（Triton kernel 开发）

若只需单一能力，请使用子 skill 独立安装包。

## DCU-first 策略（产品化）

为提升实际开发场景命中率，整包默认采用 `DCU-first` 而非 `DCU-only`：

- **默认优先**：用户请求命中 DCU/CUDA/HIP/ROCm 领域时，优先走 `dcu-dispatcher`。
- **可退出**：用户明确表示“不要走 skill / 直接回答”时，触发全绕过并返回 `__self__`（不进入任何子 skill）。
- **自动降级**：检索链路异常时，按 `hybrid -> keyword -> lookup_api` 逐级降级，不阻塞回答。
- **证据回写**：在评测/报告任务中，输出 route + 检索命中 + API 命中摘要。

## 路由与执行约束

- **MUST**：DCU 相关请求先经 `dcu-dispatcher` 路由，再进入目标子 skill。
- **MUST**：`code-migrate` 请求至少执行一次 `dcu-code-migrator` 的检索或 API 查询。
- **SHOULD**：简单问答走轻量路径（`lookup_api`/`search`），复杂迁移再走完整工作流。
- **MAY**：当用户明确要求速度优先时，跳过多步骤编排，但需保留关键风险提示。

## 路由到包外 toolkit skills

- 通用 HIP kernel 开发规范（包外）→ `dcu-kernel-dev`
- Triton 原生 DSL 优化 PyTorch 算子（包外）→ `dcu-triton-kernels`
- 性能剖析 → `dcu-hipprof`；算子优化闭环 → `dcu-op-optimization`

## 边界与说明

- 本包是编码/测试类入口，不直接承载具体优化方法。
- 子 skill frontmatter 为外部包原生 keywords/intents 风格（历史遗留），功能不受影响。
- 五个子 skill 与 common/ 基础设施已全部在本仓库内（2026-10-19 自 89 live 包 v1.2.1 反向收编），无需自备。

## 推荐 DCU 领域关键词

用于触发 DCU-first 的关键词（中英文混合）：

- **平台与工具链**：`dcu` `dtk` `rocm` `hip` `hipcc` `rocminfo` `rocm-smi` `gfx90a` `wavefront`
- **运行时与编译**：`cuda` `nvcc` `cuda runtime` `hip runtime` `offload-arch` `hipify`
- **核心 API**：`cudaMalloc` `cudaMemcpy` `cudaStream` `cudaEvent` `hipMalloc` `hipMemcpy`
- **基础库迁移**：`cublas` `cufft` `curand` `cudnn` `nccl` `hipblas` `hipfft` `hiprand` `miopen` `rccl`
- **常见开发意图词**：`迁移` `适配` `移植` `编不过` `报错` `构建失败` `性能优化` `回归` `兼容层`

## 包内结构与收编说明

- 五个子 skill 全部在包内 `skills/` 下（dcu-dispatcher / dcu-hip-reference / dcu-code-migrator / dcu-ci-testcase / dcu-triton-kernel）；`common/` 为 KB 分片检索与 SPI 路由共享层，`config.yaml` 为组件配置（凭证留空，token 走环境变量）。
- 2026-10-19 自 89 live 包（v1.2.1）反向收编入库；dispatcher 文档取仓库 1.1.0 演进版，ci-testcase / triton-kernel SKILL.md 取仓库收编修正版。
- 知识分片 `kb_shard*.sqlite` 带 CHECKSUM 与 manifest，属分发内容；重建用 `scripts/rebuild_embeddings.py`。
