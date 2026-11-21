---
title: DCU K100-AI Matrix Core 指令集与 wavefront 数据布局
scope: [gfx928, K100-AI]
source: "github.com/shibingzhang/XDZS2025 (Readme §SGEMM, dtk25.04) + 内部文档 dcu-kernel-dev"
verified_on: "XDZS2025 于 K100-AI/dtk25.04 实测；DCU 工程师复用于 dcu-kernel-dev"
status: verified
---

# Matrix Core 指令集与 wavefront 数据布局（DCU K100-AI / gfx928）

## 1. 可用 Matrix Core 指令（按数据类型）

来源：XDZS2025 通过研读 AMD RDNA 资料 + 在 K100-AI 上反编译/实测确定。
DCU 官方文档对指令尺寸与矩阵布局说明有限，下列为实测可用的 intrinsic：

| 数据类型 | intrinsic | 计算形状 (M,N)=(M,K)x(K,N) |
|---|---|---|
| FP16 | `__builtin_amdgcn_mmac_f32_16x16x16f16` | (16,16) = (16,16) x (16,16) |
| FP32 | `__builtin_amdgcn_mmac_f32_16x16x4f32`  | (16,16) = (16,4)  x (4,16)  |
| FP32 | `__builtin_amdgcn_mmac_f32_16x16x8f32`  | (16,16) = (16,8)  x (8,16)  |
| INT8 | `__builtin_amdgcn_mmac_i32_16x16x32i8`  | (16,16) = (16,32) x (32,16) |

> SGEMM 实战选用 `__builtin_amdgcn_mmac_f32_16x16x8f32`。

## 2. wavefront 数据布局（关键，反编译得）

DCU wavefront = 64 线程。以 SGEMM 的 FP32 MC 指令为例，**每个 wavefront 的 64 个线程**
需按固定布局向寄存器提供 A/B/C 分块：

- A：每线程 2 个 FP32 元素
- B：每线程 2 个 FP32 元素
- C：每线程 4 个 FP32 元素（累加器，寄存器持有）

线程 0..63 各自持有的元素位置见 XDZS2025 `assets/Fig1.svg`（wavefront 内布局图）。
**布局是反编译实验确定的，DCU 官方未公开**——移植其他 GEMM 形状时必须沿用该布局约定，
否则 MC 指令结果错乱。

## 3. 线程块 / 寄存器 / LDS 资源（K100-AI）

- 每 CU：64 KB LDS，256 KB 寄存器（4 个 SIMD 共享 LDS；寄存器每 SIMD 最多 64 KB）。
- 每 thread block：512 线程 = 8 wavefront。每线程最多 128 个 32-bit 寄存器。
- 例：BM=256, BN=128, BK=16 的 tile：
  - LDS 占用 = (256+128)×16×4 = 24 KB
  - 每线程寄存器 = 80（4×2 + 4×2 + 4×4×4）

## 4. 适配其他 DCU 架构的注意

- 上述指令/布局是 **K100-AI (gfx928)** 实测。换卡（如 gfx906 系列）前：
  1. 用 `dcu_detect_arch.sh` 确认 gfx；
  2. MC 指令集可能不同代，**先小 kernel 验证该 intrinsic 是否存在**，不要假设可用；
  3. LDS/寄存器上限以物理机 `rocminfo` 为准（容器里不可靠）。
