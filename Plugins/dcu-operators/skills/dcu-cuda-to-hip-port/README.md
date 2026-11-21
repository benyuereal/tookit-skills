# dcu-cuda-to-hip-port — 独立 CUDA C++ 项目移植到海光 DCU/HIP

把一个独立 CUDA C++ 工程（如 NInfer）完整移植到海光 DCU/HIP 的方法与配套模板。比 dcu-cuda-hip-port
更偏「整项目落地」：覆盖 MFMA 指令可用性、缺失符号 stub、端到端精度调试。

## 核心内容
- **MFMA 可用性**（references/mfma-availability.md）：哪些 warp 级 MFMA 形状/布局可用、陷阱
- **链接 stub**（references/link-stub-recipe.md）：缺失符号怎么 stub
- **端到端精度调试**（references/e2e-accuracy-debug.md）：移植后输出不对怎么定位
- **templates/bf16_gemm_simt.cuh**：SIMT-first bf16 GEMM 参考实现

## 目录
```
dcu-cuda-to-hip-port/
├── SKILL.md                      整项目移植主流程
├── templates/bf16_gemm_simt.cuh  SIMT bf16 GEMM 参考
└── references/
    ├── ninfer-triage.md          NInfer 移植排查
    ├── mfma-availability.md     MFMA 指令可用性
    ├── link-stub-recipe.md      缺失符号 stub
    └── e2e-accuracy-debug.md   精度调试
```

## 快速用法
1. `SKILL.md` 给整项目移植步骤
2. 遇 MFMA/符号/精度问题分别查对应 references
