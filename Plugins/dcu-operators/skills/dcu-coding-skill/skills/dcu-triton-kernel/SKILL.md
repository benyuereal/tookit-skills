---
name: dcu-triton-kernel
version: 1.0.1
license: MIT
description: Triton kernel 开发助手，支持 CUDA/Triton 到 DCU HIP/TileLang 的 kernel 迁移与开发
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




- dcu-hip-reference
- dcu-ci-testcase

---

# DCU 代码迁移助手

CUDA → HIP/DTK 代码迁移的专项子技能。主要能力：CUDA API 映射到 HIP、迁移报告生成、残留 CUDA 符号检测；同时内置 HIP 知识分片支持 API 查询。

## 支持的意图

本技能承接两类意图，由 `dcu-dispatcher` 路由过来：

- **Triton kernel 开发/优化**：用 Triton/TileLang 写或优化 DCU kernel（本技能主职责）
- **CUDA → HIP/DTK 代码迁移**：CUDA API 映射到 HIP、迁移报告生成、残留 CUDA 符号检测（内置 HIP 知识分片支持 API 查询）

如果用户问的是（应转其他 skill）：

- 纯 HIP API 查询 → `dcu-hip-reference`
- CI 测试用例生成 → `dcu-ci-testcase`
- 深海 DCU 专属算子库源码（AITER/LightOp/DeepGEMM）→ 各 `dcu-*` 源码 skill

## 快速开始

> 用户：「把这段 cublasSgemm 迁移到 DTK」
> → `migrate_cuda_to_hip(code)` → 返回 hipblasSgemm 版本 + 注意事项

> 用户：「我这份代码还有没有 CUDA 符号没迁完」
> → `check_remaining_cuda(code)` → 列出 `cudaMalloc` / `cudaMemcpy` / `<cuda_runtime.h>` 等未处理项

> 用户：「输出迁移报告」
> → `get_migration_report(code)` → 返回 API 覆盖率、已迁移列表、未迁移列表、风险提示
## 核心函数

| 函数 | 描述 |
|------|------|
| `migrate_cuda_to_hip(code)` | 基于知识分片做 CUDA→HIP API 映射生成 |
| `get_migration_report(code)` | 汇总迁移状态、API 映射表、风险项 |
| `check_remaining_cuda(code)` | 扫描 cuda 符号、CUDA 头文件、不支持特性 |
| `lookup_api(symbol)` | 精确查询 HIP/DTK API 签名 |
| `search_docs(query, top_k)` | 关键词/语义检索知识分片 |

## Triton kernel 开发规范（借鉴自 sglang-diffusion-triton-kernel）

写 Triton kernel 的标准动作（DCU 上对应 TileLang/hip 后端）：
1. 指针全部传原始 tensor，让 Triton/TileLang 内部取 `.data_ptr()`；**不要假设张量连续**，每 stride 单独传参。
2. `tl.constexpr` 用于块尺寸与布尔分支；`mask=mask, other=0.0` 兜越界读。
3. 精度敏感时在 `tl.float32` 计算再 cast 回输出 dtype；用 `tl.fma(a,b,c)` 做融合乘加避免舍入。
4. `@triton.autotune`：`key=["C"]`（hidden 维度变化时重调）；配置升序，warps×32 ≤ 1024。
5. `torch.compile` 区域里包 `torch.library.wrap_triton`；skip 只在 Python 图边界用的工具 kernel。
6. Naming：kernel 函数 `_<op>_kernel`（下划线前缀，私有）；launcher 同名公开。
适用边界：DCU 上 Triton 后端选 ROCm 时同样适用；TileLang（见 `dcu-kernel-dev`）写法同源但 target 须 `hip`。

注意：此处参考的是 SGLang 的 Triton 写法，DCU 的等价实现对应 `dcu-kernel-dev` 的 TileLang 路径。

## 向量化 + warp 规约通用内存优化（提炼自 sglang-diffusion-cuda-kernel）

- `AlignedVector<T, N>`：128-bit 向量 load/store（8×bf16/fp16、4×fp32），打满访存带宽；**DCU 上等价于 HIP float4/int4 访存**。
- Warp 级规约：`warp::reduce_sum` / `warp::reduce_max` 替代 raw `__shfl_xor_sync`；**DCU warp_size=64**（见 `dcu-kernel-dev` §3.4，跳 32 会越界）。
- 共享内存 bank conflict：加 padding（`[32][33]` 等）规避。
- Occupancy 上限：grid 数 = min(SM×blocks_per_SM, div_ceil(n, block))；DCU 上用 `dcu_detect_arch.sh` 现查 gfx 定 arch。


## 知识分片

脚本按下面的优先级自动选分片，不需要手动指定：

1. **Primary shard** — `kb_shard/shard.sqlite`（本子技能自带）
2. **External shard** — `kb_shard_ext/shard.sqlite`（由大框架聚合层写入，可选）
3. **Legacy DB** — `references/kb.sqlite`（仅迁移期兜底，**可选**：文件不存在时脚本自动跳过该层，不报错，不建议依赖）

`search.py` / `lookup_api.py` 的返回里会带 `source_layer`（`native`/`ext`/`legacy`）和 `searched_layers` 字段，便于排查命中层。

## 路由速查

- `references/routing-index.json` — 域到文档的预计算路由
- `references/api-summary.json` — API 符号族
- `references/source-manifest.json` — 文档目录
- `references/config.json` — 分片状态（敏感字段由 dispatcher/install 脚本清洗）

## 硬性规则

- NEVER 臆造 HIP/DTK API — 只使用知识分片里存在的符号
- ALWAYS 引用源文档提供 API 详情
- 对 `code-migrate` 意图：给出 CUDA 原文和 HIP/DTK 等价写法的并列对照
- 生成迁移代码时给出**最小可编译项目**结构（CMakeLists.txt + 源文件），除非用户只要片段
- NEVER 用 Glob/Read/ls/dir 去探测 `kb_shard/shard.sqlite` 是否存在。脚本内部已做分层解析和存在性检查，直接执行；如三层都缺，脚本会在 JSON 里用 `"error"` 字段明确报出
- ALWAYS 先 `cd` 到本子技能根目录再执行 `python scripts/search.py ...`，不要用绝对路径或 stdin 传入脚本内容，否则 `__file__` 解析不到分片
- NEVER 手动创建或修改 `kb_shard_ext/` 的任何文件。它由大框架拥有，`rebuild_embeddings.py` 也不会触碰它