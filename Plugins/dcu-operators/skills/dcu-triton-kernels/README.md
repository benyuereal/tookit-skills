# dcu-triton-kernels — Triton 原生 DSL 优化 PyTorch 算子

在 DCU（海光 BW1000）上用 Triton 原生 DSL 优化 PyTorch 算子的完整流程与十题实战索引：Benchmark 流程、
五大优化模式、DCU 陷阱。可跨 BW1000 / C500 / Ascend910B 迁移参考。

## 核心内容
- Benchmark 标准流程
- 五大优化模式（tile/访存/向量化/融合/调度）
- DCU 专属陷阱 + 十题实战（GroupedTopk、FusedMoE、FlexAttention、SPLADESparsePooler 等）

## 目录
```
dcu-triton-kernels/
├── SKILL.md                     Triton 优化方法论
└── references/
    ├── evaluator/auto_bench.py  自动评测
    └── tasks/                 十题实战（0x_*.md）
```

## 快速用法
写/优化 Triton kernel、做跨卡迁移时读 `SKILL.md` + 对应实战题。
