---
name: dcu-cuda-hip-port
description: "Port CUDA code to DCU/HIP on DTK. Use for CUDA→HIP porting."
version: 1.1.0
author: DCU Engineer
license: MIT
---

# DCU CUDA→HIP 移植

把外部 CUDA 代码库（推理引擎、kernel 库、.cu 源码）移植到海光 DCU（DTK/HIP）。
本 skill 沉淀 2026-08 NInfer→DCU 移植中**验证过**的构建方案、API 差异、hipify 策略和 kernel 对拍方法。

## When to Use
- 把一个 CUDA 项目（CMake + .cu）搬到 DCU 上编译运行
- hipify 转换 CUDA→HIP
- 移植后验证 kernel 数值正确性（对拍 CPU 参考）
- 判断哪些 CUDA 特性在 DCU 上可用 / 需重写 / 需砍

Don't use for: 从零写 HIP kernel（用 dcu-kernel-dev）；纯部署/测试（用 dcu-llm-benchmark 等）。

## 0. 先定 gfx 目标（决定哪些路径能留）
DCU 型号 → gfx 架构（`hipcc --offload-arch` 用）：

| 型号 | gfx | 备注 |
|---|---|---|
| Z100L | gfx906 | 无 FP8 |
| Z100 | gfx908 | 无 FP8 |
| BW100 | gfx936 | CDNA3 级，有 FP8 matrix core (e4m3/e5m2) |
| BW150 / BW1000 | gfx936 | 同上 |

先 `rocminfo`（**宿主机**，带 `ROCM_PATH`/`LD_LIBRARY_PATH`）确认 gfx，再决定：
- **NVFP4 / mxf4nvf4 / TMA / PDL = Blackwell 独占** → DCU 无对应，整条路径砍
- **FP8 mma** = 仅 gfx90a / gfx936 有
- **通用 `mma.sync` (bf16/f16/s8) + `ldmatrix` + `cp.async`** = 需重写为 CDNA wmma / `__pipeline`

## 1. 构建方案：绕开 CMake 的 HIP 语言，直接用 hipcc
⚠️ DTK 26.04 的 CMake HIP 语言支持是坏的：`project(... LANGUAGES C CXX HIP)` 报
`CMake can not determine linker language for target`。试 `CMAKE_HIP_COMPILER=hipcc`/`clang++`、
`CMAKE_HIP_LINKER_LANGUAGE=CXX`、`CMAKE_PREFIX_PATH` 都无效。

**FIX：不用 CMake 的 HIP 语言，直接用 hipcc 构建**——hipcc 能同时编译 .cpp + .cu 并链接：
```bash
/opt/dtk/bin/hipcc -O2 -std=c++20 --offload-arch=gfx936 main.cpp kernel.cu -o app
```
- hipcc 默认 offload 目标是 gfx928/906/926（**不是你的卡**），必须显式 `--offload-arch=gfxXXX`。
- 大项目用构建脚本（bash/Makefile）调 hipcc，别硬修 CMake。
- 验证：最小 kernel（hipMalloc + `__global__` + hipDeviceSynchronize）编译+运行 OK 再铺开。

## 2. hipify 策略（大代码库）
- hipify-perl 单文件 10–60s，几百文件串行要几小时——太慢。
- 策略：**sed 批量做 90% 机械替换**（cuda→hip API、`__nv_bfloat16`→`__hip_bfloat16`、
  `cuda_runtime.h`→`hip/hip_runtime.h`），hipify-perl 只兜底边角（`__shfl`/`__ballot` 等 intrinsic）。
- hipify-perl **幂等**（同文件跑两次结果一致），但**并发跑同一棵树会互相拖慢**——别同时起多个。
- 先 git baseline（`git init` + commit），hipify 前留好回滚点。
- 位置：`/opt/dtk/hip/bin/hipify/hipify-perl`。

## 3. DTK API 差异（CUDA→HIP 不是 1:1）
- **warp shuffle 无 mask**：DTK 提供 `__shfl`/`__shfl_xor`/`__shfl_down`/`__shfl_up`（无 mask 版），
  **没有** `__shfl_*_sync`。AMD 上 mask 被忽略（整 warp 参与）。
  → 用 shim 头把 CUDA 的 masked 形式映射到 DTK 的 mask-less 形式。见 `templates/hip_shim.h`。
  ⚠️ shim 里**不要**用 `#if defined(__HIP_DEVICE_COMPILE__)` 守卫——hipcc 的 host pass 也会
  解析 kernel 体，守卫会导致 host pass 报 `use of undeclared identifier '__shfl_sync'`。
  `__device__` 声明在两个 pass 都可见、函数体只在 device pass 编译，所以**不加守卫**即可。
- `__nv_bfloat16` → `__hip_bfloat16`（hipify-perl 自动转，上千处这种量级没问题）
- `cudaStream_t`→`hipStream_t`、`cudaMalloc`→`hipMalloc` 等 runtime API：hipify-perl 自动转
- `cp.async` PTX 内联汇编 → `__pipeline_memcpy_async`（HIP 有）
- `mma.sync` PTX → CDNA wmma / gfx90x inline asm（需重写，见 dcu-kernel-dev）
- `ex2.approx.f32` / `cvt.rn.bf16x2.f32` 等 PTX：DTK 有对应或需换

## 4. kernel 对拍方法（验证移植正确性）
移植一个 kernel 后，别只信"编译通过"，要数值对拍：
1. 从原代码抽该 kernel + **最小依赖闭包**（warp_sum/head_map/常量等），做成自包含 .cu
2. 写一个**直白的 CPU 参考实现**（朴素循环，同样的数学，不优化）
3. driver：随机数据（固定 seed）→ DCU 跑 kernel → 拷回 → CPU 参考 → 比 max abs error
4. 用**真实模型维度**（不是玩具尺寸），scale 用真实值
5. 容差：fp32 递归 kernel 累积误差 ~1e-6 量级，容差设 5e-3（能抓 O(1) 真 bug，避免假 FAIL）
6. 探针 PASS 才推进下一个 kernel。完整 worked example 见 `references/probe-recipe.md`

## 5. 移植范围裁剪（闭集引擎）
对"只支持固定几个模型"的闭集引擎，先锁定目标模型，砍掉用不到的路径：
- 读 target 注册（registry.cpp）确认目标模型走哪个 target 类
- 读 config.h 确认模型结构（层数/头数/是否 MoE/是否混合注意力）
- 按目标模型的权重档（如 groupwise-int）确认实际用到的算子集合
- 砍掉：其他模型 target、用不到的量化档（NVFP4/FP8）、MoE（若目标稠密）、vision（若纯文本）
- 砍得越狠，要移植的 kernel 越少

## Pitfalls
1. **别硬修 CMake HIP**——DTK 的 CMake HIP 语言检测是坏的，直接 hipcc 构建（见 §1）
2. **hipcc 必须显式 `--offload-arch`**——默认目标是 gfx928/906/926，不是你的卡
3. **warp shuffle shim 别加 `__HIP_DEVICE_COMPILE__` 守卫**——host pass 会报未声明
4. **hipify-perl 别并发跑同一棵树**——幂等但互相拖慢；大库用 sed 批量 + hipify-perl 兜底
5. **别只信编译通过**——kernel 必须数值对拍（见 §4）
6. **先定 gfx 再定路径**——NVFP4/TMA/PDL 是 Blackwell 独占，DCU 砍；FP8 仅 gfx90a/936
7. **rocminfo 查 gfx 要在宿主机**（带 ROCM_PATH/LD_LIBRARY_PATH），容器内不可靠

## 验证清单
- [ ] gfx 目标已确认（rocminfo 宿主机）
- [ ] hipcc `--offload-arch=gfxXXX` 最小 kernel 编译+运行 OK
- [ ] hipify 完成（cuda 符号残留≈0，`__nv_bfloat16` 已转）
- [ ] warp shuffle shim 就位（无 `__HIP_DEVICE_COMPILE__` 守卫）
- [ ] 核心 kernel 探针 PASS（对拍 CPU 参考，max err < 容差）
- [ ] 移植范围已裁剪（砍掉用不到的 target/量化档/MoE/vision）
## 官方文档中心指针（2026-10-14 实测）

- 《DCU 编程实战（2026 版）》版本树 `a2e964e8-a67d-11f0-909a-b6dd5499a79e`（25 叶：概述/软硬件架构/编程方法/应用移植与优化/AI 应用/附录）；配套示例代码在 z-file Cat7「DCU编程实战 代码」SampleChapter3-6.zip（download.sourcefind.cn:65024）。
- DCC 编译器深水区（编译选项/-mllvm/GCVM/LingShu 后端）见 `dcu-triton-source/references/dcc-compiler-docs.md`。
- 取全文：`GET /api-doc/document/TreeDocument?VersionUUID=<树uuid>` 列目录、`DetailDocument?UUID=<叶uuid>`（匿名，2026-10-14 实测）；总索引见包内 `../_dcu-toolkit-shared/references/sourcefind/doc-center-index.md`。
