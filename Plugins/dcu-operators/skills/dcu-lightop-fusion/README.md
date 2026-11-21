# dcu-lightop-fusion — LightOp 算子融合方法论

写或理解海光 LightOp 融合 kernel 的方法论：融合模式、手法、checklist。基于 LightOp csrc 中 30+ 个融合
kernel 提炼（8 个代表 kernel 完整读过）。

## 核心内容
- **融合模式**：常见融合 kernel 的划分与数据流
- **手法/checklist**：写新融合 kernel 的步骤与核对项
- 与 `dcu-lightop-source`（源码库索引）/ `dcu-lightop-moe`（MoE 专题）互补

## 目录
```
dcu-lightop-fusion/
└── SKILL.md   融合方法论 + 模式/手法/checklist
```

## 快速用法
写新融合 kernel / 理解融合实现时读 `SKILL.md`；找具体算子源码去 `dcu-lightop-source`。
