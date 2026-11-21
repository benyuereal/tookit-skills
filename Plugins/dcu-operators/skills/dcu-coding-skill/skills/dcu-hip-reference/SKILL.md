---
name: dcu-hip-reference
version: 1.0.1
description: HIP runtime / 基础 API 查询与示例，承接 api-lookup 意图，可作为其他 core skill 的共同知识基础
author: DCU Team
license: MIT
kind: tool
---

# DCU HIP Reference Skill

HIP runtime / 基础 API 的**权威查询助手**。从内置知识分片检索 HIP API 签名、使用示例、常见错误处理等。

## Quick Start

> User: "hipMalloc 怎么用"
> → `lookup_api("hipMalloc")` → 返回签名 + 参数说明 + 示例代码

> User: "HIP stream 相关 API 有哪些"
> → `search_docs("HIP stream API")` → 返回 hipStreamCreate / hipStreamSynchronize 等文档摘要

> User: "hipMemcpyAsync 和同步版本的区别"
> → `search_docs(...)` → 返回对比说明

## Core Functions

| 函数 | 用途 |
|------|------|
| `lookup_api(api_name)` | 精确查询 API 签名与参数（top-1） |
| `search_docs(query, domains, top_k)` | 语义 + 关键词混合检索文档 |

## References

- HIP API 参考: [references/hip-api-reference.md](references/hip-api-reference.md)
- HIP runtime 指南: [references/hip-runtime-guide.md](references/hip-runtime-guide.md)
- 最佳实践: [references/best-practices.md](references/best-practices.md)
