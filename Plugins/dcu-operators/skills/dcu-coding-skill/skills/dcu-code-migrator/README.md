# dcu-code-migrator 使用说明

`dcu-code-migrator` 是 CUDA 到 HIP/DTK 的迁移子 skill，面向代码转换、迁移检查与迁移报告生成。

## 适用场景

- 需要把 CUDA 代码改写为 HIP/DTK。
- 需要核对某个 CUDA API 的 HIP 等价写法。
- 需要扫描代码中残留的 CUDA 符号或头文件。
- 需要输出迁移覆盖率与风险点报告。

## 核心能力

- `migrate_cuda_to_hip`：执行 CUDA -> HIP API 映射与代码改写。
- `check_remaining_cuda`：检测未迁移完成的 CUDA 痕迹。
- `get_migration_report`：汇总迁移结果、覆盖度和风险项。
- 内置 `lookup_api` 与 `search_docs` 用于迁移过程中的 API 校验。

## 典型问题

- “把这段 CUDA kernel 迁移到 HIP。”
- “`cublasSgemm` 在 DTK 下对应什么实现？”
- “检查这份工程里还有哪些 CUDA 调用没替换。”
- “给我一份迁移报告，包含已迁移和未迁移项。”

## 输出特点

- 给出 CUDA 原写法和 HIP/DTK 等价写法对照。
- 标注潜在兼容性差异与迁移风险。
- 尽量提供可编译的最小代码结构建议。
