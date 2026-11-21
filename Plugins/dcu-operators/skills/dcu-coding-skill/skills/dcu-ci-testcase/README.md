# dcu-ci-testcase 使用说明

`dcu-ci-testcase` 是面向 CI 的测试用例子 skill，负责测试用例生成、结构化校验与测试套件组织。

## 适用场景

- 需要为 HIP/Triton 代码生成单元测试或集成测试。
- 需要把零散测试组织为可执行的测试套件。
- 需要先做测试结构和断言质量校验，再接入 CI。

## 核心能力

- `create_testcase`：根据代码与目标测试类型生成测试用例。
- `validate_testcase`：验证测试结构、关键断言和基本规范。
- `create_suite`：将多个测试组装成 suite 并给出运行建议。
- `run_testcase`：提供测试执行计划（便于在 CI 中落地）。

## 典型问题

- “给这段 HIP 代码生成 `pytest` 单元测试。”
- “创建一个覆盖主流程的 integration test。”
- “把这些测试组织成 suite 并输出执行命令。”
- “帮我检查这个 gtest 用例是否关键断言齐全。”

## 输出特点

- 输出结构化、可复用的测试资产。
- 默认关注可执行性、断言有效性与 CI 集成友好性。
- 支持 unit/integration 两类常见场景。
