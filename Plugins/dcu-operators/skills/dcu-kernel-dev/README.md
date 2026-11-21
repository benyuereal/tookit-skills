# dcu-kernel-dev — DCU HIP / TileLang 算子开发规范

DCU（海光）算子开发方法论，覆盖 **手写 HIP kernel** 与 **TileLang DSL** 两条路径，并提供 DCU 专属的优化轴、工具速查、精度陷阱与编译硬约束。是整个 DCU 算子库（vLLM/AITER/LightOp/DeepGEMM）和「算子替换/优化」类 skill 的底层写 kernel 能力。

## 解决什么问题 / 何时用

- 要在 DCU 上写/优化一个 HIP kernel（如给 vLLM 自定义算子、替换原生算子）——先看本 skill 的 HIP 调优速查表与示例代码
- 要用 TileLang/DSL 写 kernel——先看本 skill 的「何时用 TileLang vs 手写 HIP」判断和 TileLang DCU 已知问题
- 遇到 kernel 慢/精度不对/编不过——本 skill 的瓶颈→方法映射、RMSNorm 精度陷阱、编译硬约束章节直接对症
- 做算子级性能优化闭环（profile→定位→改→验证）——配合 `dcu-op-optimization` / `dcu-perf-profiling` 使用

## 目录

```
dcu-kernel-dev/
├── SKILL.md              # 正文：16 章完整开发规范
├── README.md             # 本文件（人向概览）
├── scripts/
│   ├── dcu_build_kernel.sh      # 编译辅助
│   ├── dcu_detect_arch.sh       # 设备架构探测（不硬编码 gfx）
│   ├── hip_gemm_tiled.cpp       # HIP GEMM Tiling 示例
│   ├── tl_gemma_rmsnorm_naive.py # TileLang RMSNorm 示例
│   ├── tl_paged_mqa_logits.py   # TileLang paged MQA logits 示例
│   ├── tl_rmsnorm_benchmark.sh  # RMSNorm 基准
│   ├── tl_rmsnorm_precision_test.py # RMSNorm 精度测试
│   └── (XDZS2025 已外置，未随包分发，见 SKILL §13.4)
└── references/
    └── launch.json              # VS Code 调试配置
```

## SKILL.md 内容速览（16 章）

1. **何时用 TileLang（路径 B）vs 手写 HIP（路径 A）**：两条路径选型判断
2. **TileLang 安装与环境**：清华源基础依赖、两条安装源、DTK/镜像
3. **TileLang 架构与写法核心（DCU 必读）**：后端选择、动态维度、`alloc_local` vs `alloc_shared`（精度命门）、DCU `warp_size=64`、`k_pack`、dynamo 墙
4. **TileLang 在 DCU 的专属已知问题**（实测汇总）
5. **编译安装与 lower-pass 调试实战**（GLM-5.1 K100 案例）
6. **参考实现（scripts/）**
7. **交叉引用**
8. **RMSNorm TileLang 优化方法论**：三级优化 + 一级精度陷阱 + bit-identical 门禁
9. **tilelang 各 example 在 DCU 的通过性矩阵**
10. **结构化 kernel 开发 playbook**（源自 tensormux/kernel-skills）
11. **profile 驱动优化循环协议**（源自 auto-kernel-research）
12. **Agent 驱动 kernel 优化 flow**：Task Contract 模板 / 最小循环 / 证据留痕
13. **DCU 编译硬约束与设备架构探测**（不硬编码 gfx，`dcu_detect_arch.sh`）
14. **HIP kernel 调优速查表与示例代码**：性能瓶颈表、向量化加载、LDS 树形规约、GEMM Tiling
15. **DCU 专属优化轴与工具速查**：DCU≠NVIDIA 事实、优化轴优先级、瓶颈→方法映射、工具速查（profiling/ISA）、ISA 自检清单
16. **跨架构微优化速查**（手法可迁移 DCU）

> 涉及最少的可直接执行示例见 `scripts/`；设备架构探测与编译脚本在此，避免写死 gfx 号。

## 使用

- **写 kernel 前**：读 §3（TileLang 核心）/ §14（HIP 速查表）
- **优化一个 kernel**：走 §11 profile 驱动闭环 + §15 瓶颈映射
- **精度异常**：重点看 §8.5 精度陷阱、§8.5b bit-identical 门禁

## 与相关 skill 的分工

- 实际落地到框架：`dcu-vllm-op-replace`（替换算子）、`dcu-vllm-custom-op`（从零造算子）
- 优化闭环方法论：`dcu-op-optimization`
- profiling/瓶颈判定：`dcu-perf-profiling` / `dcu-hipprof`
- 知识库查单个算子怎么写：`dcu-kernelwiki`
