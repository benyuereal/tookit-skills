# dcu-triton-kernel 使用说明

`dcu-triton-kernel` 是 Triton 算子开发子 skill，覆盖 kernel 生成、性能优化与 HIP -> Triton 转换。

## 适用场景

- 需要从零生成 Triton kernel（如 matmul、reduce、elementwise）。
- 需要对现有 Triton kernel 做性能优化。
- 需要把 HIP kernel 逻辑迁移到 Triton 实现。

## 核心能力

- `generate_triton_kernel`：根据需求生成 Triton kernel 模板与实现。
- `optimize_kernel`：围绕并行配置、访存和调度进行优化建议。
- `hip_to_triton`：将 HIP 实现映射到 Triton 风格代码。
- 提供模板化知识，便于快速复用常见算子结构。

## 典型问题

- “帮我写一个 Triton matmul kernel。”
- “这个 Triton kernel 太慢了，怎么优化？”
- “把这段 HIP kernel 改成 Triton 版本。”
- “给一个带 autotune 思路的 Triton 实现模板。”

## 输出特点

- 优先给出可运行的 kernel 代码骨架。
- 在代码层面说明优化点与参数调优方向。
- 必要时给出从功能正确到性能调优的分步建议。
