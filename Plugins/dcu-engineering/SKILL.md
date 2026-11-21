---
name: dcu-engineering
description: "DCU 工程实践技能集。覆盖海光 DCU 工程实践技能集：CI bisect、cherry-pick、代码规范、模型接入 cookbook 与 PR 优化史。"
version: 1.5.1
author: DCU Toolkit Contributors
license: MIT
---

# DCU 工程实践 — 域级技能包

## 工具介绍

海光 DCU 工程实践技能集：CI bisect、cherry-pick、代码规范、模型接入 cookbook 与 PR 优化史。本目录为 Kits Platform Skill Tool `dcu-engineering` 的发布单元，包含 12 个子 skill：

| skill | 说明 |
|---|---|
| ci-workflow-guide | Modify and diagnose upstream SGLang CI orchestration, including stages… |
| clean-startup-log | Clean noisy SGLang startup logs without suppressing actionable compati… |
| cookbook-add-model | Add a new model to the config-driven SGLang Mintlify cookbook with a p… |
| cookbook-migrate-model | Migrate an existing legacy SGLang cookbook page to the config-driven M… |
| cookbook-review-pr | Review an SGLang Mintlify cookbook PR for config-driven structure, pro… |
| env-var-conventions | Apply SGLang environment-variable conventions and lifecycle checks for… |
| large-class-init-style | Structure SGLang Scheduler, TokenizerManager, and ModelRunner construc… |
| mechanical-refactor-verify | Verify file splits, function moves, renames, and module extractions wi… |
| model-pr-history-knowledge | Build and query an evidence-backed history of model and runtime decisi… |
| scripted-runtime-notes | Design and review SGLang scripted-runtime tests and harness APIs while… |
| sglang-bisect-ci-regression | Investigate repeatable SGLang CI failures by normalizing the signature… |
| sglang-cherrypick | Validate, dispatch, and monitor SGLang bot cherry-picks of merged PRs … |

## 安装与使用

- **单 skill 使用**：Agent 按上表选定 skill 后，读取 `skills/<skill>/SKILL.md` 并按其流程执行（子 skill 自包含，含 scripts/ 与 references/）。
- **统一路由**：跨域任务优先读仓库根 `SKILL.md`（包级路由表）或 `dcu-core/skills/dcu-skill-router`（全量索引）。
- **安装**：见仓库根 `README.md` / `install.sh`（支持 Claude Code / OpenCode / Codex 等多框架，项目级或全局）。
- **版本**：与包根 SKILL.md 一致（1.5.1）；升级以包内版本为准。
