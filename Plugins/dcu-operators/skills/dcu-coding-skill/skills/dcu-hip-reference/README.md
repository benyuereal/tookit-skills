# dcu-hip-reference 使用说明

`dcu-hip-reference` 是 HIP runtime 与基础 API 的查询型子 skill，适合做 API 解释、参数说明和示例检索。

## 适用场景

- 想查某个 HIP API 的签名、参数和返回值。
- 想快速确认 HIP runtime 的基础概念或常见用法。
- 需要示例代码或最佳实践参考。

## 核心能力

- `lookup_api`：精确查询 API 级信息。
- `search_docs`：按语义或关键词检索 HIP 相关文档。
- 面向 `api-lookup` 意图，作为其他 core skill 的通用知识基础。

## 典型问题

- “`hipMalloc` 怎么用？”
- “`hipMemcpyAsync` 参数分别是什么？”
- “HIP stream 创建、同步、销毁流程是什么？”
- “`hipDeviceProperties` 有哪些字段需要关注？”

## 输出特点

- 优先给出可执行、可落地的 API 使用方式。
- 包含函数签名、参数说明、返回值和注意事项。
- 需要时补充最小示例代码片段。
