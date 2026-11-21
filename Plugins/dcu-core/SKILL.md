---
name: dcu-core
description: "DCU 核心路由与编排技能集。覆盖海光 DCU 核心技能集：skill 统一路由、多任务工作流编排、KV Cache 卸载、融合 kernel 与训练手册。"
version: 1.6.0
author: DCU Toolkit Contributors
license: MIT
---

# DCU 核心路由与编排 — 域级技能包

## 工具介绍

海光 DCU 核心技能集：skill 统一路由、多任务工作流编排、KV Cache 卸载、融合 kernel 与训练手册。本目录为 Kits Platform Skill Tool `dcu-core` 的发布单元，包含 9 个子 skill：

| skill | 说明 |
|---|---|
| dcu-custom-fusion-kernel | Write a DCU HIP fusion kernel for unsupported model configs. |
| dcu-hicache | SGLang HiCache 三层缓存（L1 GPU/L2 host/L3 磁盘）在 DCU 上的部署、补丁与验证；含 DeepSeek-V… |
| dcu-kernel-layer-microbench | DCU 上 LLM 推理 kernel 层微基准方法论：profiler 定位不到 decode 时间去向时，用微基准量化 kernel 开… |
| dcu-offline-container-deps | Use when an offline DCU container lacks a Python package. |
| dcu-offline-wheel-install | 无外网DCU容器缺少量纯Python包时用：探测、wheel传送与离线pip安装坑。 |
| dcu-orchestrator | DCU 多任务薄编排层（公开版）。依据 workflow YAML 与 task 契约，做多任务编排：读 workflow → 算就绪节点 … |
| dcu-skill-on-device-verification | 在真机/容器实跑DCU技能自带脚本，对照SKILL.md基线判断生效与准确。用户让扫描或去设备验证技能时加载。 |
| dcu-skill-router | DCU 系列 skills 路由与链路手册：按部署/推理/算子开发/性能调优/环境/训练/工程发布七域分组，索引全部 dcu-* skill… |
| dcu-train-cookbook | DCU/HCU 训练域 skill：训练/微调/RL 强化学习配方速查 + 训练框架版本锚点。覆盖 train-cookbook-das（M… |

## 安装与使用

- **单 skill 使用**：Agent 按上表选定 skill 后，读取 `skills/<skill>/SKILL.md` 并按其流程执行（子 skill 自包含，含 scripts/ 与 references/）。
- **统一路由**：跨域任务优先读仓库根 `SKILL.md`（包级路由表）或 `dcu-core/skills/dcu-skill-router`（全量索引）。
- **安装**：见仓库根 `README.md` / `install.sh`（支持 Claude Code / OpenCode / Codex 等多框架，项目级或全局）。
- **版本**：与包根 SKILL.md 一致（1.6.0）；升级以包内版本为准。
