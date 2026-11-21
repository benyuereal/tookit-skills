# dcu-deepgemm-arch — DeepGEMM 三路径架构与 MODE 调优

详解海光 DeepGEMM 的三种 GEMM 路径（masked/contiguous/ll）与 MODE 调优机制：怎么按场景选路径、
layout 陷阱、怎么加调优表。与 `dcu-deepgemm-source`（源码地图）互补——本文回答「三路径怎么选、MODE 怎么调」。

## 核心内容
- **三路径选型**：masked / contiguous / ll 各自适用场景
- **MODE 调优**：调优表怎么组织、触发条件
- **layout 陷阱**：跨路径常见的布局相关的坑

## 目录
```
dcu-deepgemm-arch/
└── SKILL.md   架构剖析 + 选型/调优指南
```

## 快速用法
选 GEMM 路径 / 遇 layout 陷阱 / 加调优表时读 `SKILL.md`；要看源码位置去 `dcu-deepgemm-source`。
