---
title: LDS bank conflict 与 padding 规避（DCU）
scope: [gfx928, K100-AI, gfx906]
source: "github.com/shibingzhang/XDZS2025 (Readme §SGEMM.2 LDS bank conflict) + dcu-kernel-dev"
verified_on: "XDZS2025 实测 K100-AI；DCU 工程师沉淀 dcu-kernel-dev"
status: verified
---

# LDS bank conflict 与 padding 规避（DCU）

## 现象

GEMM kernel 中 tile_A 从 global 载入 shared memory 后，64 线程的 wavefront 做读事务时，
因数据布局导致**严重的 LDS bank conflict**（同一事务内多冲突），显存访问效率骤降、延迟上升。

## 规避方法（padding）

对 tile_A 的 SMEM 在 BK 维 pad 2 个元素：分配 `BM × (BK+2)` 但只用 `BM × BK`，
每行末尾 2 个 float 作 dummy padding。

- 效果：padding 后 64 线程可在 2 笔 32-float 事务内读 64 个 float，
  **达成 100% shared memory 带宽利用率**。
- 对照见 XDZS2025 `assets/Fig3.svg`（(a) 冲突 vs (b) padding 后）。

## 另一种规避法：128B XOR swizzle（块置换）

padding 之外，还可对每 128B 行做 8×16B 块下标 XOR 行下标（`block ^ (row & 7)`）打散 bank。
**DCU 专属 caveat**：swizzle 是纯软件地址变换（DCU 无 TMA 硬件 swizzle），
**向量化访问下净正、scalar 访问下净负**（软件 XOR 重算地址的 ALU 开销反超 bank 冲突收益）。
scalar 场景优先用上面的 padding 法（零地址开销）。详见 `blackwell_gemm_dcu_transfer.md` §一（<device-B> gfx936 实测）。

## 通用判据

- LDS 按 32 bank（典型）；连续 32 个 4-byte 字各落不同 bank。
- 任何"按行连续 stride = bank 数整数倍"的访问都会冲突 → 靠 pad 奇数/偏移打破。
- DCU 上验证 bank conflict：用 `hipprof`/`rocprof` 看 LDS 带宽与 stall 指标，
  而非肉眼猜（呼应 dcu-kernel-dev 的 profile 纪律）。

## 量化发现法：SFA/缩放因子的 bank conflict 数学推导（溯源 Colfax SM120 案例）

> 来源：Colfax Research RTX Pro 6000 NVFP4 GEMM 优化（七轮迭代版本 5），
> https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/
> 方法论通用，DCU gfx936/gfx928 同为 32 bank LDS。

### 典型冲突模式：SFA 行 r 与 r+8 同 bank

以 NVFP4 的 Scale Factor A（SFA）为例，原始布局：

```
SFA 布局：((32, 4), REST_M) 与 ((16, 4), 1, REST_K)
byte_offset = 16·m0 + 4·m1 + b
bank_index  = ⌊(byte_offset / 4)⌋ (mod 32)
```

其中 `m0 = m mod 32`，`m1 = ⌊m/32⌋ mod 4`。

**冲突成因**：考虑 SFA 第 0–15 行（m1=0），`bank_index = m0`，同一 bank 内两个行地址相等：
- **row r 与 row r+8** 的缩放因子地址落在同一 bank。
- 16 个地址分布在 8 个 bank 上，每 bank 有 2 个地址 → 需要 **2 个 wavefront** 完成。

Nsight 指标信号：`L1 Wavefronts Shared Excessive`（8.39M，对 8k 形状）。

### 修复方法：改 stride 打破冲突

```
原 stride：(32, 4):(4, 128)  →  byte_offset = 16·m0 + 4·m1 + b（冲突）
新 stride：(4, 128)           →  byte_offset = 4·m0 + 128·m1 + b = 4(m0+32·m1)+b（每行独立 bank）
```

修改后 SFA 原子的 cosize 不变（相同定义域上双射），只是字节在原子内部重排。

### DCU 适配通式

任何 DCU kernel 的 SMEM bank conflict 排查，用同一通式：

```
1. 写出该 SMEM 布局的 byte_offset 公式
   → byte_offset = f(stride, coord_dim_0, coord_dim_1, ...)

2. 计算 bank_index = (byte_offset / 4) mod 32

3. 观察冲突模式：
   → 哪些行/列组合落在同一 bank？
   → 是否存在 row_r 与 row_{r+N}（N=8 的倍数）同 bank 的规律？

4. 修改 stride/padding：
   → 目标：同一 bank 内最多 1 个访问
   → 量化验证：bank conflict 降为 0 或 1

5. hipprof/rocprof 对应指标验证理论值（DCU 上查 LDS stall PMC 事件）
```

> 附：完整的 Colfax 七轮迭代系统化方法论（v1→v7：swizzling / epilogue / warp 专用化 / bank conflict / 12路 MMA / autotune）
> 已沉淀为 `dcu-op-optimization` 案例库 案例七，作为 GEMM 类算子"profile → 量化 → 迭代"的完整实例。

## 移植注意

- padding 量随 tile 维与 dtype 变；**不要照搬 XDZS2025 的 +2**，按你的 BK 与元素宽度重算。
- 其他架构（gfx906 等）bank 数/宽度可能不同，以物理机为准。
