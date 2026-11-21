---
title: GEMM 缩放因子 SFA 的 bank conflict 分析与修复（Colfax RTX Pro 6000 SM120 案例）
scope: [gfx936, gfx928, gfx906, RDNA/CDNA 通用]
source: "Colfax Research - 'Optimizing an NVFP4 Blockscaled GEMM on RTX PRO 6000 Blackwell GPU (SM120)', 2026-08-09, https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/"
verified_on: "Colfax Research 原文（RTX Pro 6000 SM120，BW150/BW100 同属 gfx936 代，方法论通用）；DCU 方法论对应 dcu-kernel-dev §14"
status: verified
---

# GEMM 缩放因子 SFA 的 bank conflict 分析与修复

> 来源：Colfax Research RTX Pro 6000 SM120 NVFP4 块缩放 GEMM 优化实录（七轮迭代版本 5）。
> 本案例方法论可迁移至 DCU（gfx936 / gfx928 等），尤其在 GEMM / Attention kernel 出现 SMEM 访问瓶颈时。

## 冲突的根因：SFA SMEM 布局导致行 r 与 r+8 落在同一 bank

### 原始 SFA 布局

```
SFA 布局：((32, 4), REST_M) 与 ((16, 4), 1, REST_K)
```

- M 子布局 `(32, 4):(16, 4)` 把 128×128 CTA tile 的 m 坐标分解为 `m = m0 + 32·m1`（`m0 = m mod 32`，`m1 = ⌊m/32⌋ mod 4`）。
- K 子布局 `(16, 4):(0, 1)` 反映缩放因子组织，stride 0 把一个缩放因子广播到 NVFP4 元素的 16 个元素。

由此推出的 **byte offset** 公式：
```
byte_offset = 16·m0 + 4·m1 + b
bank_index  = ⌊(byte_offset / 4)⌋ (mod 32)
```

### 冲突量化

考虑 SFA 第 0–15 行的元素（`m1 = 0`），每个 bank index 都是 4 的倍数。观察可知：
- **row r 与 row r+8 的缩放因子地址落在同一 bank**。
- 16 个地址分布在 8 个 bank 上，每个 bank 有 **两个地址访问**，需要 2 个 wavefront。

Nsight 指标信号：`L1 Wavefronts Shared Excessive = 8.39M`（对 8k 问题形状 M=N=8192）。

### 修复方法：修改 stride 布局

布局层面的修复很直接——修改 stride 从 `(32, 4):(4, 128)` 改为 `(4, 128)`（即 rank-1 布局 128:4）：

```
byte_offset_new = 4·m0 + 128·m1 + b = 4(m0 + 32·m1) + b
```

于是**每个行有独立的 bank index**，SMEM 到寄存器的访问在单个 wavefront 内完成。

## 通用方法论：bank conflict 的量化排查流程

Colfax 案例提供了一个**可直接套用到 DCU 的排查流程**：

```
1. 怀疑 SMEM 瓶颈 → 查 hipprof/rocprof 的 LDS stall 指标
   → DCU 上对应 "LDS bank conflict" 相关 PMC 事件

2. 量化冲突规模
   → 计算目标 SMEM 布局的 byte_offset / bank_index 公式
   → 找"多地址落同一 bank"的行号规律（如 r 与 r+8 同 bank）
   → 用 Nsight 等工具验证理论值（如 8.39M wavefront 冲突）

3. 布局层面修复
   → 调整 stride，打破冲突模式（每行独立 bank）
   → 验证 cosize 不变（双射性质保持，数据不变）

4. 重新测 L1 Wavefronts Shared Excessive（或 DCU 上对应指标）
   → 确认冲突消除
```

> **关键洞察**：bank conflict 是**隐形瓶颈**——纯布局问题靠直觉难以发现，只有把它量化成 Nsight/`hipprof` 指标才能拆解。
> 这与 dcu-kernel-dev §8.5「RMSNorm 舍入悬崖」（`max_diff < 0.03` 判据）同构：都需要量化手段发现、不靠代码审查。

## DCU 适配注意事项

| 方面 | Colfax SM120（NVIDIA） | DCU gfx936/gfx928 |
|------|----------------------|-------------------|
| SMEM bank 数 | 32 bank（每 bank 4 字节） | **相同**（DCU LDS 64KB 同为 32 bank × 4B） |
| bank conflict 信号 | Nsight `L1 Wavefronts Shared Excessive` | hipprof/rocprof 对应指标（查 `dcu-hipprof` 工具链） |
| 排查工具 | Nsight Compute | `hipprof --pmc` + ISA 反查（`dccobjdump`） |
| 修复手段 | 改 stride 布局 | 同理：改 LDS tile 的 layout/stride，打破同 bank 访问模式 |

## 与既有 DCU 知识库的交叉

- 本案例是 `lds_bank_conflict.md`（padding 规避法）的**量化版补充**：padding 法解决"64 线程访问同一 bank"；本案例补充"如何数学推导 + 量化发现冲突"。
- 冲突消除后 SMEM→寄存器搬运从 2 wavefront 降到 1 wavefront，对带宽受限的 GEMM/Attention kernel 收益明显（如 vLLM flash attention 的 `LDS 64KB` 限制下）。
- 案例属于 `dcu-op-optimization` 第 2 步（定位瓶颈类型：Memory-bound → SMEM bank conflict）的具体实例化。