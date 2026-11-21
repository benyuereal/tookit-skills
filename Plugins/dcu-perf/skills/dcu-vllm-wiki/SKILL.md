---
name: dcu-vllm-wiki
description: Navigate project evidence for vLLM and SGLang adaptation on DCU, HCU, and MACA accelerators. Use for architecture, version comparison, HIP porting, DTK/DAS compatibility, Docker images, API mapping, historical optimization decisions, accepted/rejected experiments, or when searching all dcu-adapt worktrees before answering a project-specific question.
version: 1.0.1
author: DCU Engineer
license: MIT
---

# DCU vLLM Project Wiki

> ⚠ **内部仓库依赖**：本 skill 的知识库位于内部 dcu-adapt 仓库的 `docs/wiki/`（含 `.qoder/plans/` 等内部开发产物），该仓库及其 worktree **不随本包分发**。无内部仓库的用户本 skill 不可用；条目仅为内部团队保留。

A project knowledge base lives at `docs/wiki/` in the dcu-adapt repository. Resolve the repository
with `git rev-parse --show-toplevel`; do not depend on a user-specific absolute path. Read the
specific source rather than answering from memory.

Start points: `docs/wiki/README.md` (navigation) and `docs/wiki/SUMMARY.md` (full table of contents).

For inference optimization, read
[`references/inference-optimization-history.md`](references/inference-optimization-history.md)
before proposing a benchmark, profile, kernel replacement or acceptance decision.

To inventory every registered worktree and its latest reports:

```bash
SKILL_DIR=/path/to/dcu-vllm-wiki
bash "$SKILL_DIR/scripts/index_worktree_history.sh" /path/to/dcu-adapt
```

## Index — pick the file that matches the question

### Overview & architecture
- `architecture-overview.md` — vLLM core architecture and design
- `code-structure/key-components.md` — code structure, core components

### vLLM variants & porting
- `vllm-community/introduction.md` — community vLLM codebase
- `vllm-dcu/introduction.md` — DCU-ported vLLM codebase
- `dcu-port/modifications.md` — DCU port modifications/special config
- `dcu-port/hcu-plugin.md` — HCU platform plugin architecture
- `dcu-port/hip-kernel-optimization-report.md` — HIP kernel optimization report
- `comparison/community-vs-dcu.md` — community vs DCU differences
- `comparison/version-diff.md` — per-version differences

### Build, install, tooling
- `usage-guide/build-and-install.md` — build/deploy (community + DCU)
- `usage-guide/toolkit.md` — vllm_toolkit performance-test tool
- `vllm-0.20.2-migration-guide.md` — migration guide to 0.20.2

### Docker images (`docker-images/`)
- `overview.md`, `quick-reference.md` — image index + cheat sheet
- `vllm-deployment.md`, `sglang-deployment.md`, `pytorch-deployment.md` — per-stack deploy
- `compatibility-matrix.md` — image ↔ component ↔ hardware compatibility
- `performance-tuning.md` — memory/parallel/quant/batch tuning
- `other-images.md`

### DCU-Code / OpenCompass evaluation toolkit (`dcu-code/`)
- `overview.md`, `architecture.md`, `documentation.md`, `tools-reference.md`,
  `source-packages.md`, `usage-scenarios.md`, `troubleshooting.md`, `development.md`

### DCU software sources — DTK & DAS (`dcu-sources/`)
- `overview.md`, `quick-reference.md` — DAS+DTK source index + cheat sheet
- `dtk-index.md`, `das-index.md` — DTK / DAS site indexes
- `version-mapping.md` — DAS ↔ DTK ↔ component version map
- `package-catalog.md`, `das-release-notes.md` — package versions, release notes
- `dtk-api-mapping.md` — DTK ↔ CUDA ↔ HIP API mapping
- `dtk-docs/index.md`, `das-docs/index.md` — archived official DTK/DAS docs
- `ai-components/` — `overview.md`, `vllm-archive.md`, `sglang-archive.md`,
  `flashattn-archive.md`, `distributed-archive.md`, `download-manifest.md`

### Beginner / planning
- `beginner-friendly-tutorials.md`, `harness-development-plan.md`

## How to use

1. Resolve the main repository and enumerate worktrees when the question concerns optimization history.
2. Search `docs/wiki/`, `.qoder/plans/`, `docs/reports/`, commit messages and dirty worktree
   artifacts; state which branch/worktree supplied each conclusion.
3. Prefer accepted/rejected reports and raw manifests over roadmap estimates.
4. Distinguish historical facts from current environment state. Re-probe remote hardware before
   acting on versions, paths, services or GPU health.
5. Use hardware-operation skills for live SSH, container and GPU work. This skill covers evidence
   retrieval and interpretation, not production mutation.
