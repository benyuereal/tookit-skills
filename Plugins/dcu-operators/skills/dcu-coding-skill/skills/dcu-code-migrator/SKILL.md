---
name: dcu-code-migrator
version: 1.0.1
description: CUDA 到 HIP/DTK 的代码迁移助手，基于内置 HIP 知识分片做 API 映射、迁移报告生成与残留 CUDA 符号检测
author: DCU Team
- migrate
- migration
- cuda
- hip
- dtk
- cuda-to-hip
- port
- rocm
- 迁移
- 适配
- 移植
- api-mapping
- type: code-migrate
  description: 把 CUDA 代码迁移到 HIP / DTK / DCU
  examples:
  - "把这段 CUDA 代码迁移到 HIP"
  - "cublasSgemm 在 DTK 下等价写法"
  - "cudaMemcpy 迁移后要注意什么"
  - "帮我把 cuFFT 换成 hipFFT"
  - "检查这份代码里还有没有 CUDA 符号"
- dcu-hip-reference
- dcu-ci-testcase
- action: migrate_cuda_to_hip
  cost: medium
  description: 基于知识分片生成 CUDA→HIP 迁移代码
- action: get_migration_report
  cost: medium
  description: 汇总迁移覆盖度、已处理/未处理 API 与风险点
- action: check_remaining_cuda
  cost: cheap
  description: 扫描残留的 CUDA 符号 / 头文件 / API
- action: lookup_api
  cost: cheap
  description: 直接查 HIP/DTK API 签名与使用说明
- action: search_docs
  cost: medium
  description: 在 HIP/DTK 知识分片中做关键词/语义检索
license: MIT
kind: tool
---
---

# DCU Code Migrator

CUDA → HIP/DTK 代码迁移的专项 sub-skill。主要能力：CUDA API 映射到 HIP、迁移报告生成、残留 CUDA 符号检测；同时内置 HIP 知识分片支持 API 查询。

## 支持的意图

本 skill 只承接 `code-migrate` 意图，由 `dcu-dispatcher` 路由过来。如果用户问的是：

- 纯 API 查询 → 应由 `dcu-hip-reference` 处理
- 测试用例生成 → 应由 `dcu-ci-testcase` 处理
- Triton kernel → 应由 `dcu-triton-kernel` 处理

## Quick Start

> User: "把这段 cublasSgemm 迁移到 DTK"
> → `migrate_cuda_to_hip(code)` → 返回 hipblasSgemm 版本 + 注意事项
>
> User: "我这份代码还有没有 CUDA 符号没迁完"
> → `check_remaining_cuda(code)` → 列出 `cudaMalloc` / `cudaMemcpy` / `<cuda_runtime.h>` 等未处理项
>
> User: "输出迁移报告"
> → `get_migration_report(code)` → 返回 API 覆盖率、已迁移列表、未迁移列表、风险提示

## Core Functions


| 函数                           | 描述                                |
| ---------------------------- | --------------------------------- |
| `migrate_cuda_to_hip(code)`  | 基于知识分片做 CUDA→HIP API 映射生成         |
| `get_migration_report(code)` | 汇总迁移状态、API 映射表、风险项                |
| `check_remaining_cuda(code)` | 扫描 cuda 符号、CUDA 头文件、不支持特性         |
| `lookup_api(symbol)`         | 精确查询 HIP/DTK API 签名（sub-skill 内置） |
| `search_docs(query, top_k)`  | 关键词/语义检索知识分片（sub-skill 内置）        |


## Knowledge base shards

脚本按下面的优先级自动选分片，不需要手动指定：

1. **Primary shard** — `kb_shard/shard.sqlite`（本 sub-skill 自带）
2. **External shard** — `kb_shard_ext/shard.sqlite`（由大框架聚合层写入，可选）
3. **Legacy DB** — `references/kb.sqlite`（仅迁移期兜底，不建议依赖）

`search.py` / `lookup_api.py` 的返回里会带 `source_layer`（`native`/`ext`/`legacy`）和 `searched_layers` 字段，便于排查命中层。

## Routing Quick Reference

- `references/routing-index.json` — 域到文档的预计算路由
- `references/api-summary.json` — API 符号族
- `references/source-manifest.json` — 文档目录
- `references/config.json` — 分片状态（敏感字段由 dispatcher/install 脚本清洗）

## Hard Rules

- NEVER invent HIP/DTK APIs — 只使用知识分片里存在的符号
- ALWAYS cite the source document when providing API details
- 对 `code-migrate` 意图：给出 CUDA 原文和 HIP/DTK 等价写法的并列对照
- 生成迁移代码时给出**最小可编译项目**结构（CMakeLists.txt + 源文件），除非用户只要片段
- NEVER 用 Glob/Read/ls/dir 去探测 `kb_shard/shard.sqlite` 是否存在。脚本内部已做分层解析和存在性检查，直接执行；如三层都缺，脚本会在 JSON 里用 `"error"` 字段明确报出
- ALWAYS 先 `cd` 到本 sub-skill 根目录再执行 `python scripts/search.py ...`，不要用绝对路径或 stdin 传入脚本内容，否则 `__file`__ 解析不到分片
- NEVER 手动创建或修改 `kb_shard_ext/` 的任何文件。它由大框架拥有，`rebuild_embeddings.py` 也不会触碰它
