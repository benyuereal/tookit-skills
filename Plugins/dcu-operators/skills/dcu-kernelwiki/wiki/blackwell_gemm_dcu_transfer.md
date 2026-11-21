---
title: Blackwell GEMM 8步优化在 DCU 的可迁移性实测
scope: [gfx936, BW100, BW150, BW1000]
source: "微信《手写Blackwell GEMM:164到1401 TFLOP/s》(AI圈的9527, mp.weixin.qq.com/s/ks5ZWHA5qw6qrRavoKrgBg) + <device-B> 真机实测"
verified_on: "2026-10-05 设备<device-B> BW100 gfx936 dtk26.04 卡6"
status: verified
---

# Blackwell GEMM 8步优化在 DCU 的可迁移性实测

> 来源：微信文章《手写Blackwell GEMM:164到1401 TFLOP/s》（AI圈的9527），
> 在 B200 上用 CuTe DSL 手写 GEMM，8 步从 164 爬到 1401 TFLOP/s（cuBLAS 97%）。
> 本文记录其中**可迁移手法**在 DCU gfx936（<device-B> BW100）上的真机实测结论。
> 硬件专属步骤（TMA multicast / TMEM 双缓冲 / persistent kernel / 2-CTA / warp 特化 / K-loop）
> 依赖 Blackwell 的 TMA/TMEM/tcgen05，DCU 无对应物，**不可测、不可搬**。

## 结论速览

| 文章手法 | 文章收益(B200) | DCU gfx936 实测 | 可迁移? |
|---|---|---|---|
| 128B XOR swizzle（治 LDS bank 冲突） | 4.1x | 微基准 ~1.2x 净正；scalar GEMM 净负(~0.9x) | 机制可迁移，净收益取决于访问模式 |
| 多 stage 双缓冲（load/compute overlap） | 1.9x | scalar GEMM 净负(~0.9x) | 仅对 latency-bound kernel 有效 |
| TMA multicast / TMEM 双缓冲 / persistent / 2-CTA / warp 特化 / K-loop | — | DCU 无 TMA/TMEM/tcgen05 | 不可搬 |

## 一、128B XOR swizzle：机制真实，净收益取决于访问模式

### 手法
每 128B 行切 8 个 16B 块，块下标 XOR 行下标（`block ^ (row & 7)`），
把"多行写同一列全撞同组 bank"打散到不同 bank。治 LDS 写 bank 冲突。
（与 `lds_bank_conflict.md` 的 padding 法并列的另一种 bank 冲突规避手段。）

### DCU 实测（<device-B> gfx936）
- **隔离微基准**（global→LDS 写，float4 向量化，512 线程/块）：
  plain 7.5us → swizzle 6.0us = **~1.24x 净正**（3 次稳定）。
  → bank 冲突消除机制在 DCU 上**真实存在**（plain 8 路冲突，swizzle 无冲突）。
- **真实 scalar GEMM**（4096³，tile 32³，scalar LDS 读写）：
  swizzle 反而**慢 ~11%（0.895x）**。
  → 每次 LDS 读都要软件重算 XOR 地址，ALU 开销 > bank 冲突收益，净负。

### 关键判据（DCU 专属）
- 文章 4.1x 靠 **TMA 硬件 swizzle（零 ALU 成本）+ MMA 向量化读**。
- DCU **无 TMA**，swizzle 是纯软件地址变换：
  - **向量化访问**（float4/矩阵核读）下：地址变换摊薄，bank 冲突收益 > ALU 成本 → **净正**。
  - **scalar 访问**下：每次读都重算 XOR，ALU 成本反超 → **净负**。
- **结论**：DCU 上用 swizzle 前，先确认 LDS 访问是向量化的；scalar 场景优先用 padding 法（零地址开销）。

## 二、多 stage 双缓冲：只对 latency-bound kernel 有效

### 手法
num_stages 个 LDS buffer 环形复用，load tile k+1 与 compute tile k 重叠，隐藏 global load 延迟。

### DCU 实测（<device-B> gfx936）
- **scalar GEMM**（LDS-read/FMA-bound，global load 延迟已被掩盖）：
  2-stage 比 1-stage **慢 ~9%（0.912x）**。
  → 双缓冲只加寄存器压力，无延迟可隐藏，净负。
- 文章 1.9x 是在 **TMA-latency-bound** 场景（load path 占 92% 运行时）。

### 关键判据（DCU 专属）
- 双缓冲/多 stage **只在 kernel 是 global-load-latency-bound 时有效**。
- 先 profile 确认瓶颈：若 LDS/FMA/compute-bound，双缓冲净负（寄存器压力）。
- 呼应 `optimization_loop_protocol.md` 的"先量再动手"与"瓶颈会搬家"。

## 三、可迁移的是方法论，不是手法

文章 8 步里 3 步白干（warp 特化 / TMA multicast / 流水 epilogue）、1 步干了几乎全部（swizzle），
且哪步有用取决于跑 prefill 还是 decode。真正能抄到 DCU 的是：
1. **先量再动手**：epilogue 只占 12% 就不值得流水。
2. **null 结果也是结果**：白干的步骤逼出"瓶颈在 load path 不在 issue"的结论。
3. **瓶颈会搬家**：stage 数第 3 步最优到第 7 步就反了，sweep 只在当时 regime 有效。
4. **按形状分账**：prefill/decode 收益完全不同，别混着报。

## 移植注意
- 本文 GEMM 是 naive scalar 实现（~4 TFLOPS），"swizzle 净负 / 双缓冲净负"是针对该 scalar GEMM 的结论。
  向量化 + 矩阵核（v_mmac）的优化 GEMM 里，swizzle（向量化读 + 低成本地址变换）行为可能不同，未测。
- 性能数字为结论性量级（非基线表），逐条数据见本地留痕 / dcu-perf-cases。
