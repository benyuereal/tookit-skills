---
name: dcu-dispatcher
version: 1.1.0
description: DCU Coding Skill 统一入口，纯路由层：意图分类 → 路由分发 → 多 skill 工作流编排（migrate-and-test）；跨域多步任务链（部署/故障/算子/推理/调研 5 条工作流 + 任务契约）见 dcu-orchestrator；不持有知识分片
author: DCU Team
license: MIT
kind: composite
---

# DCU Dispatcher

DCU 编程 AI 助手的**统一入口**。负责理解用户意图并路由到对应的专业 skill，支持跨 skill 工作流编排。

## 支持的意图

| 意图 | 目标 Skill | 典型问题 |
|------|-----------|---------|
| `api-lookup` | dcu-hip-reference | "hipMalloc 怎么用" |
| `code-migrate` | dcu-code-migrator | "把这段 CUDA 迁移到 HIP" |
| `ci-test` | dcu-ci-testcase | "生成单元测试" |
| `triton` | dcu-triton-kernel | "写一个 Triton matmul" |
| `unknown` | dcu-hip-reference | 其他 DCU 相关问题 |

## 多 skill 工作流

- `migrate-and-test`：迁移代码后自动生成测试用例
- 跨域多步任务链（部署验证/故障排查/算子优化/推理优化，probe→normal→gate(ask) 三段式 + evidence 验收）已由 `dcu-orchestrator` 承接（包根平铺 skill，4 条 workflow + 13 个 task 契约 + 渲染/验收/沉淀脚本）；本目录 `workflows/` 保留 migrate-and-test 旧式单链（steps/on_error schema，与 orchestrator 的 nodes/depends_on/proceed schema 不同，勿混用）。

## 路由策略补充

- 默认采用 `DCU-first`：命中 DCU 领域关键词时优先路由到 DCU 子 skill。
- 用户可显式绕过：命中“不要走skill/直接回答”等短语时，触发全绕过并返回 `__self__`（不进入子 skill）。

## 安装说明

整包安装时自动激活（包根 `marketplace.json` 为 5-skills 清单：本 dispatcher + hip-reference + code-migrator + triton-kernel + ci-testcase；hip-reference/code-migrator 仅存在于安装包，git 仓库不含）。经 junction 路径加载时 dispatcher.py 的 marketplace 解析指向安装包根；DSH 注册模式只消费 SKILL.md，不受影响。单独安装单个 core skill 时不需要此 dispatcher。

## References

- 工作流配置: [workflows/README.md](workflows/README.md)
- 跨域任务链编排: `dcu-orchestrator`（包根平铺；workflow YAML + task 契约 + render_workflow.py/verify_evidence.py/dcu_distill.py）
