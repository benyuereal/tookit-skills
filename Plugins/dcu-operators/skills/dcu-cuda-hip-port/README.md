# dcu-cuda-hip-port — CUDA → DCU/HIP 移植（DTK 环境）

把 CUDA 代码移植到海光 DCU/HIP 的起步 skill。面向 DTK 工具链，提供 hip_shim.h 兼容头与探测配方，
用于理解 CUDA→HIP 的 API/语义差异并把代码跑上 DCU。

## 核心内容
- **hip_shim.h**：常用 CUDA API 到 HIP 的兼容映射头（模板），减少手改工作量
- **探针配方**（references/probe-recipe.md）：怎么在 DCU 上确认架构/gfx 号与运行约束

## 目录
```
dcu-cuda-hip-port/
├── SKILL.md                 移植流程 + API 映射要点
├── templates/hip_shim.h    HIP 兼容头模板
└── references/probe-recipe.md  DCU 运行环境探测配方
```

## 快速用法
1. 读 `SKILL.md` 了解 CUDA→HIP 移植步骤
2. 需要兼容头时取 `templates/hip_shim.h`
3. 移植前先看 `references/probe-recipe.md` 确认目标卡环境

> 更完整的独立 CUDA C++ 项目移植（含 MFMA 可用性/符号 stub/精度调试）见 `dcu-cuda-to-hip-port`。
