# dcu-dispatcher 使用说明

`dcu-dispatcher` 是 DCU Coding Skill 的统一入口路由层，负责把用户问题分发到正确的子 skill。

## 适用场景

- 用户问题涉及 HIP API 查询、CUDA 到 HIP 迁移、CI 测试用例生成或 Triton kernel 开发。
- 需要多 skill 串联工作流（例如先迁移代码再生成测试）。
- 需要按意图自动分发，而不是手工指定子 skill。

## 核心能力

- 意图分类：识别 `api-lookup`、`code-migrate`、`ci-test`、`triton`。
- 路由分发：将请求转发到对应 core skill。
- 工作流编排：支持跨 skill 的组合流程（如 `migrate-and-test`）。
- 路由策略控制：支持 DCU-first 与用户显式 bypass（例如“不要走 skill”）。

## 路由目标一览

- `api-lookup` -> `dcu-hip-reference`
- `code-migrate` -> `dcu-code-migrator`
- `ci-test` -> `dcu-ci-testcase`
- `triton` -> `dcu-triton-kernel`

## 使用示例

- “hipMalloc 怎么用” -> 路由到 `dcu-hip-reference`
- “把这段 CUDA 迁移到 HIP” -> 路由到 `dcu-code-migrator`
- “帮我生成这个 HIP kernel 的单元测试” -> 路由到 `dcu-ci-testcase`
- “写一个 Triton matmul kernel” -> 路由到 `dcu-triton-kernel`

## 安装说明

- 该 skill 通常在整包安装时生效。
- 单独安装某个 core skill 时，通常不需要单独带上 dispatcher。
